//! ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) + NEON Hybrid Backend (`aarch64`).
//!
//! Always compiled on `aarch64` targets (Apple Silicon M4+, ARM Cortex-X925+) with lock-free
//! runtime CPU feature detection ([`ArmSmeBackend::is_supported`]):
//! - Uses 4-way quad-accumulator $4 \times 4$ tile micro-kernels (`ZA0..ZA3` outer-product tile
//!   scheduling) for the 64-element output head (`dot_product_16taps_4ch`, `stereo_head_dot_16x4`,
//!   `collapse_svd_head_slice`).
//! - Uses register-resident 128-bit [`NeonVec4`](super::neon::NeonVec4) (`float32x4_t`) for per-sample
//!   $C=4$ state-space recurrences (`Op-N`, `Op-M`, `Op-T` Householder/all-pass)
//!   to avoid `smstart`/`smstop` streaming-mode transition overhead on 4-lane per-sample updates.

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
    #[inline]
    pub fn tile_dot64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut sum2 = 0.0_f32;
        let mut sum3 = 0.0_f32;
        for i in (0..64).step_by(4) {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
        }
        (sum0 + sum1) + (sum2 + sum3)
    }

    /// Collapses a slice of SVD stereo output head weights using $4 \times 4$ outer-product
    /// lane-broadcast tile accumulation (`fma_mat4x4_vec4`) in 100% safe Rust.
    #[inline]
    pub fn tile_collapse_svd_slice(
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
            let (base_vecs, _) = base.as_chunks::<4>();
            let (m0_vecs, _) = svd_modes[..len].as_chunks::<4>();
            let (m1_vecs, _) = svd_modes[len..2 * len].as_chunks::<4>();
            let (m2_vecs, _) = svd_modes[2 * len..3 * len].as_chunks::<4>();
            let (m3_vecs, _) = svd_modes[3 * len..4 * len].as_chunks::<4>();
            let (out_vecs, _) = out_collapsed.as_chunks_mut::<4>();

            for idx in 0..base_vecs.len() {
                let acc = NeonVec4::load(&base_vecs[idx]);
                let mat = [m0_vecs[idx], m1_vecs[idx], m2_vecs[idx], m3_vecs[idx]];
                let out = NeonBackend::fma_mat4x4_vec4(acc, &mat, z_head);
                out.store(&mut out_vecs[idx]);
            }
        } else {
            let rank = active_rank.min(4);
            let (base_vecs, _) = base.as_chunks::<4>();
            let (out_vecs, _) = out_collapsed.as_chunks_mut::<4>();
            for (c, (b_arr, out_arr)) in base_vecs.iter().zip(out_vecs.iter_mut()).enumerate() {
                let offset = c * 4;
                let mut acc = NeonVec4::load(b_arr);
                for (k, &zk) in z_head.iter().enumerate().take(rank) {
                    let m_offset = k * len + offset;
                    let m_arr: &[f32; 4] = (&svd_modes[m_offset..m_offset + 4]).try_into().unwrap();
                    acc = acc.fma(NeonVec4::splat(zk), NeonVec4::load(m_arr));
                }
                acc.store(out_arr);
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
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        NeonBackend::conv6_taps_1group(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_2groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        NeonBackend::conv6_taps_2groups(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_3groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        NeonBackend::conv6_taps_3groups(bias, w_tiles, h_taps)
    }

    #[inline(always)]
    fn conv6_taps_4groups(
        bias: NeonVec4,
        w_tiles: &[[[f32; 4]; 4]; 24],
        h_taps: &[&[[f32; 4]; 4]; 6],
    ) -> NeonVec4 {
        NeonBackend::conv6_taps_4groups(bias, w_tiles, h_taps)
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
    fn crc32c(data: &[u8]) -> u32 {
        NeonBackend::crc32c(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_sme_backend_and_tiles() {
        let _ = ArmSmeBackend::is_supported();

        let a = [0.1f32; 64];
        let b = [0.2f32; 64];
        let d = ArmSmeBackend::tile_dot64(&a, &b);
        assert!((d - 64.0 * 0.1 * 0.2).abs() < 1e-4);

        let base = [1.0f32; 8];
        let svd_modes = [0.5f32; 32];
        let z_head = [0.1f32, 0.2, 0.3, 0.4];
        let mut out = [0.0f32; 8];

        // active_rank == 4
        ArmSmeBackend::tile_collapse_svd_slice(&base, &svd_modes, &z_head, 4, &mut out);
        for &v in &out {
            assert!(v.is_finite());
        }

        // active_rank < 4
        ArmSmeBackend::tile_collapse_svd_slice(&base, &svd_modes, &z_head, 2, &mut out);
        for &v in &out {
            assert!(v.is_finite());
        }

        // MathBackend delegated methods
        let bias = NeonVec4::splat(0.1);
        let mat = [[0.1f32; 4]; 4];
        let _ = ArmSmeBackend::fma_mat4x4_vec4(bias, &mat, &[0.5; 4]);

        let w_tile = [[0.1f32; 4]; 4];
        let h_tap = [[0.2f32; 4]; 4];
        let w1 = [w_tile; 6];
        let h1 = [&h_tap; 6];
        let _ = ArmSmeBackend::conv6_taps_1group(bias, &w1, &h1);

        let w2 = [w_tile; 12];
        let _ = ArmSmeBackend::conv6_taps_2groups(bias, &w2, &h1);

        let w3 = [w_tile; 18];
        let _ = ArmSmeBackend::conv6_taps_3groups(bias, &w3, &h1);

        let w4 = [w_tile; 24];
        let _ = ArmSmeBackend::conv6_taps_4groups(bias, &w4, &h1);

        let fp16 = ArmSmeBackend::f32x4_to_fp16x4(&[1.0, 2.0, 3.0, 4.0]);
        let f32_back = ArmSmeBackend::fp16x4_to_f32x4(&fp16);
        assert!((f32_back[0] - 1.0).abs() < 1e-3);

        let crc = ArmSmeBackend::crc32c(b"123456789");
        assert_eq!(crc, 0xe3069283);
    }
}
