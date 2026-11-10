//! ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` / `SME2`) hardware math backend (`ArmSmeBackend`).
//!
//! Falls back at runtime to [`ArmSve2Backend`] when `FEAT_SVE2` is present, and to [`NeonBackend`]
//! when `FEAT_SVE2` is absent (such as on Apple M4 processors, which implement `FEAT_SME` + `NEON`
//! without `FEAT_SVE2`).

use super::neon::{NeonBackend, NeonVec4, aarch64_crc32c_update};
use super::sve2::ArmSve2Backend;
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend};

/// ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` / `SME2`) hardware backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSmeBackend;

impl ArmSmeBackend {
    /// Detects at runtime whether the host `aarch64` CPU supports `FEAT_SME` (cached in an `AtomicU8`).
    #[inline]
    pub fn is_supported() -> bool {
        use core::sync::atomic::{AtomicU8, Ordering};
        static CACHE: AtomicU8 = AtomicU8::new(0);
        let c = CACHE.load(Ordering::Relaxed);
        if c != 0 {
            return c == 2;
        }
        let ok = Self::detect_uncached();
        CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
    }

    #[cold]
    fn detect_uncached() -> bool {
        #[cfg(all(target_os = "macos", any(test, feature = "std")))]
        {
            type Void = core::ffi::c_void;
            let (mut val, mut size) = (0_i32, core::mem::size_of::<i32>());
            unsafe extern "C" {
                fn sysctlbyname(
                    n: *const u8,
                    o: *mut Void,
                    ol: *mut usize,
                    np: *const Void,
                    nl: usize,
                ) -> i32;
            }
            // SAFETY: Null-terminated constant sysctl name and valid stack `val`/`size` pointers.
            let rc = unsafe {
                sysctlbyname(
                    c"hw.optional.arm.FEAT_SME".as_ptr().cast::<u8>(),
                    (&raw mut val).cast::<Void>(),
                    &raw mut size,
                    core::ptr::null(),
                    0,
                )
            };
            return rc == 0 && val == 1;
        }
        #[cfg(all(target_os = "linux", any(test, feature = "std")))]
        {
            unsafe extern "C" {
                fn getauxval(t: u64) -> u64;
            }
            // SAFETY: `getauxval(AT_HWCAP2 = 26)` is a read-only Linux libc query; bit 23 is `HWCAP2_SME`.
            return unsafe { (getauxval(26) & (1_u64 << 23)) != 0 };
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            let pfr1: u64;
            // SAFETY: `ID_AA64PFR1_EL1` (`SME` bits [27:24]) is readable at EL1.
            unsafe {
                core::arch::asm!(
                    "mrs {p}, id_aa64pfr1_el1",
                    p = out(reg) pfr1,
                    options(nomem, nostack, preserves_flags),
                );
            }
            return ((pfr1 >> 24) & 0xF) >= 1;
        }
        #[allow(unreachable_code)]
        false
    }
}

impl Crc32cBackend for ArmSmeBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSmeBackend {
    type Vec4 = NeonVec4;
    const NAME: &'static str = "ARMv9.2-A SME (Scalable Matrix Extension + SVE2/NEON)";
    const KIND: BackendKind = BackendKind::ArmSme;

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::f32x4_to_fp16x4(src)
        } else {
            NeonBackend::f32x4_to_fp16x4(src)
        }
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::fp16x4_to_f32x4(src)
        } else {
            NeonBackend::fp16x4_to_f32x4(src)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::rsqrt_nr2_f32(v0)
        } else {
            NeonBackend::rsqrt_nr2_f32(v0)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::rsqrt_nr2_pair_f32(v0, v1)
        } else {
            NeonBackend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}
