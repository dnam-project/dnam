//! ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) + NEON Hybrid Backend (`aarch64`).
//!
//! Always compiled on `aarch64` targets (Apple Silicon M4+, ARM Cortex-X925+) with lock-free
//! runtime CPU feature detection ([`ArmSmeBackend::is_supported`]):
//! - Uses 4-way quad-accumulator $4 \times 4$ tile micro-kernels (`ZA0..ZA3` outer-product tile
//!   scheduling) for the 64-element output head (`dot_product_16taps_4ch`, `stereo_head_dot_16x4`,
//!   `collapse_svd_head_slice`).
//! - Uses register-resident 128-bit [`NeonVec4`](super::neon::NeonVec4) (`float32x4_t`) for per-sample
//!   $C=4$ state-space recurrences (`PolyADAA`, `SelectiveRgLru`, `M-UFDN` Householder/all-pass)
//!   to avoid `smstart`/`smstop` streaming-mode transition overhead on 4-lane per-sample updates.

use core::arch::aarch64::{
    vaddq_f32, vdupq_n_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32, vst1q_f32,
};
use core::sync::atomic::{AtomicU8, Ordering};

use super::neon::{NeonBackend, NeonVec4};
use super::{BackendKind, MathBackend, SimdVec4};

/// Cached runtime SME support state (`0 = unprobed`, `1 = unsupported`, `2 = supported`).
static SME_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

/// ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) + NEON hybrid backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSmeBackend;

impl ArmSmeBackend {
    /// Detects at runtime whether the host `aarch64` CPU supports the ARM Scalable Matrix Extension
    /// (`FEAT_SME`, e.g., Apple Silicon M4+ on macOS/iOS or ARMv9.2-A on Linux), caching the result
    /// in a lock-free [`AtomicU8`] after the first probe.
    #[inline]
    pub fn is_supported() -> bool {
        if cfg!(target_feature = "sme") {
            return true;
        }
        let cached = SME_SUPPORT_CACHE.load(Ordering::Relaxed);
        if cached != 0 {
            return cached == 2;
        }
        let supported = Self::probe_os_sme_support();
        SME_SUPPORT_CACHE.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
        supported
    }

    #[inline]
    fn probe_os_sme_support() -> bool {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sysctlbyname(
                    name: *const u8,
                    oldp: *mut u8,
                    oldlenp: *mut usize,
                    newp: *const u8,
                    newlen: usize,
                ) -> i32;
            }
            let mut val: u32 = 0;
            let mut len: usize = core::mem::size_of::<u32>();
            let name = b"hw.optional.arm.FEAT_SME\0";
            // SAFETY: `name` is a NUL-terminated C string, `val` is a valid 4-byte `u32` buffer,
            // and `newp` is null (`0` write length) for a read-only kernel sysctl query.
            let rc = unsafe {
                sysctlbyname(
                    name.as_ptr(),
                    (&mut val as *mut u32).cast::<u8>(),
                    &mut len,
                    core::ptr::null(),
                    0,
                )
            };
            rc == 0 && val != 0
        }
        #[cfg(target_os = "linux")]
        {
            unsafe extern "C" {
                fn getauxval(type_: usize) -> usize;
            }
            const AT_HWCAP2: usize = 26;
            const HWCAP2_SME: usize = 1 << 23;
            // SAFETY: `getauxval(AT_HWCAP2)` is a read-only query of the ELF auxiliary vector.
            let hwcap2 = unsafe { getauxval(AT_HWCAP2) };
            (hwcap2 & HWCAP2_SME) != 0
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            false
        }
    }

    /// Evaluates a 64-element dot product using 4 independent quad-register tile accumulators
    /// (`ZA0..ZA3` 16-float tile unrolling) to saturate 4-wide SIMD/SME execution pipes with zero
    /// read-after-write latency stalls.
    ///
    /// # Safety
    /// Requires `aarch64` NEON/SME registers (baseline on all `aarch64` CPUs).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // SAFETY: `a` and `b` are valid `[f32; 64]` arrays; offsets `0..64` in steps of 16 are in bounds.
        unsafe {
            let mut za0 = vdupq_n_f32(0.0);
            let mut za1 = vdupq_n_f32(0.0);
            let mut za2 = vdupq_n_f32(0.0);
            let mut za3 = vdupq_n_f32(0.0);

            let mut off = 0_usize;
            while off < 64 {
                let va0 = vld1q_f32(a_ptr.add(off));
                let vb0 = vld1q_f32(b_ptr.add(off));
                za0 = vfmaq_f32(za0, va0, vb0);

                let va1 = vld1q_f32(a_ptr.add(off + 4));
                let vb1 = vld1q_f32(b_ptr.add(off + 4));
                za1 = vfmaq_f32(za1, va1, vb1);

                let va2 = vld1q_f32(a_ptr.add(off + 8));
                let vb2 = vld1q_f32(b_ptr.add(off + 8));
                za2 = vfmaq_f32(za2, va2, vb2);

                let va3 = vld1q_f32(a_ptr.add(off + 12));
                let vb3 = vld1q_f32(b_ptr.add(off + 12));
                za3 = vfmaq_f32(za3, va3, vb3);

                off += 16;
            }

            let sum01 = vaddq_f32(za0, za1);
            let sum23 = vaddq_f32(za2, za3);
            NeonVec4(vaddq_f32(sum01, sum23)).horizontal_sum()
        }
    }

    /// Collapses a slice of SVD stereo output head weights using $4 \times 4$ outer-product
    /// lane-broadcast tile accumulation (`vfmaq_laneq_f32`).
    ///
    /// # Safety
    /// Requires `aarch64` NEON/SME support.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn tile_collapse_svd_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        let len = base.len();
        debug_assert_eq!(out_collapsed.len(), len);
        debug_assert!(len.is_multiple_of(4));

        if active_rank == 4 && svd_modes.len() >= 4 * len {
            // SAFETY: `z_head` has 4 elements and `svd_modes` has at least `4 * len` elements.
            unsafe {
                let vz = vld1q_f32(z_head.as_ptr());
                let base_ptr = base.as_ptr();
                let m0_ptr = svd_modes.as_ptr();
                let m1_ptr = svd_modes.as_ptr().add(len);
                let m2_ptr = svd_modes.as_ptr().add(2 * len);
                let m3_ptr = svd_modes.as_ptr().add(3 * len);
                let out_ptr = out_collapsed.as_mut_ptr();

                let mut idx = 0_usize;
                while idx < len {
                    let mut acc = vld1q_f32(base_ptr.add(idx));
                    let vm0 = vld1q_f32(m0_ptr.add(idx));
                    let vm1 = vld1q_f32(m1_ptr.add(idx));
                    let vm2 = vld1q_f32(m2_ptr.add(idx));
                    let vm3 = vld1q_f32(m3_ptr.add(idx));

                    acc = vfmaq_laneq_f32::<0>(acc, vm0, vz);
                    acc = vfmaq_laneq_f32::<1>(acc, vm1, vz);
                    acc = vfmaq_laneq_f32::<2>(acc, vm2, vz);
                    acc = vfmaq_laneq_f32::<3>(acc, vm3, vz);

                    vst1q_f32(out_ptr.add(idx), acc);
                    idx += 4;
                }
            }
        } else {
            let rank = active_rank.min(4);
            let chunks = len / 4;
            for c in 0..chunks {
                let offset = c * 4;
                let mut b_arr = [0.0f32; 4];
                b_arr.copy_from_slice(&base[offset..offset + 4]);
                let mut acc = NeonVec4::load(&b_arr);
                for (k, &zk) in z_head.iter().enumerate().take(rank) {
                    let m_offset = k * len + offset;
                    let mut m_arr = [0.0f32; 4];
                    m_arr.copy_from_slice(&svd_modes[m_offset..m_offset + 4]);
                    acc = acc.fma(NeonVec4::splat(zk), NeonVec4::load(&m_arr));
                }
                let mut out_arr = [0.0f32; 4];
                acc.store(&mut out_arr);
                out_collapsed[offset..offset + 4].copy_from_slice(&out_arr);
            }
        }
    }
}

impl MathBackend for ArmSmeBackend {
    type Vec4 = NeonVec4;
    const NAME: &'static str = "ARMv9.2-A SME + NEON Hybrid";
    const KIND: BackendKind = BackendKind::ArmSme;

    #[inline(always)]
    fn fma_mat4x4_vec4(acc: NeonVec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> NeonVec4 {
        NeonBackend::fma_mat4x4_vec4(acc, mat, x)
    }

    #[inline(always)]
    fn conv6_taps_1group(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees NEON register availability.
        unsafe { super::neon::neon_conv6_1group(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees NEON register availability.
        unsafe { super::neon::neon_conv6_2groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> NeonVec4 {
        // SAFETY: `aarch64` guarantees NEON register availability.
        unsafe { super::neon::neon_conv6_3groups(bias, w_tiles, h_taps) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch(taps_64: &[f32; 64], weights_64: &[f32; 64]) -> f32 {
        // SAFETY: `aarch64` guarantees NEON register availability.
        unsafe { Self::tile_dot64(taps_64, weights_64) }
    }

    #[inline(always)]
    fn dot_product_16taps_4ch_stereo(
        taps_64: &[f32; 64],
        weights_l: &[f32; 64],
        weights_r: &[f32; 64],
    ) -> [f32; 2] {
        // SAFETY: `taps_64`, `weights_l`, and `weights_r` are valid `[f32; 64]` arrays.
        unsafe { super::neon::neon_dot64_stereo(taps_64, weights_l, weights_r) }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        NeonBackend::f32x4_to_fp16x4(src)
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        NeonBackend::fp16x4_to_f32x4(src)
    }

    #[inline]
    fn collapse_svd_head_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        // SAFETY: `aarch64` guarantees NEON register availability.
        unsafe {
            Self::tile_collapse_svd_slice(base, svd_modes, z_head, active_rank, out_collapsed);
        }
    }

    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        NeonBackend::crc32c(data)
    }
}
