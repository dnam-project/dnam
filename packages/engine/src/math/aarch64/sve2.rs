//! Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) hardware math backend
//! (`Sve2Vec4` & `ArmSve2Backend`).
//!
//! Uses `z0..z31` and `p0` (`ptrue p0.s, vl4`) SVE2 instructions directly when `FEAT_SVE2`
//! is detected at runtime via [`std::arch::is_aarch64_feature_detected!`], falling back to
//! [`ScalarVec4`] / [`ScalarBackend`] when `FEAT_SVE2` is absent (zero `NeonBackend` dependency).

use core::arch::aarch64::float32x4_t;

use super::neon::aarch64_crc32c_update;
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, ScalarBackend, ScalarVec4, SimdVec4};

/// 128-bit 4-lane `f32` vector backed by the low 128 bits (`vl4`) of an SVE2 `z0..z31` register.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sve2Vec4(pub [f32; 4]);

#[inline(always)]
fn to_zreg(v: [f32; 4]) -> float32x4_t {
    zerocopy::transmute!(v)
}

#[inline(always)]
fn from_zreg(v: float32x4_t) -> [f32; 4] {
    zerocopy::transmute!(v)
}

impl SimdVec4 for Sve2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        if ArmSve2Backend::is_supported() {
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\ndup z0.s, z0.s[0]",
                    in("s0") val,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([val; 4])
        }
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        if ArmSve2Backend::is_supported() {
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nzip1 z2.s, z0.s, z1.s\ndup z0.d, z2.d[0]",
                    in("s0") a,
                    in("s1") b,
                    out("q2") _,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([a, b, a, b])
        }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfadd z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).add(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfsub z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).sub(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfmul z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).mul(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmla z0.s, p0/m, z1.s, z2.s",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarVec4(self.0).fma(ScalarVec4(a.0), ScalarVec4(b.0)).0)
        }
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vv) = (to_zreg(self.0), to_zreg(a.0), to_zreg(v.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                match LANE {
                    0 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[0]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    1 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[1]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    2 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[2]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    _ => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[3]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                }
            }
            Self(from_zreg(io))
        } else {
            Self(
                ScalarVec4(self.0)
                    .fma_lane::<LANE>(ScalarVec4(a.0), ScalarVec4(v.0))
                    .0,
            )
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmls z0.s, p0/m, z1.s, z2.s",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarVec4(self.0).fms(ScalarVec4(a.0), ScalarVec4(b.0)).0)
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmaxnm z0.s, p0/m, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).max(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfminnm z0.s, p0/m, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).min(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        if ArmSve2Backend::is_supported() {
            let inp = to_zreg(self.0);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfabs z0.s, p0/m, z0.s",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).abs().0)
        }
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        if ArmSve2Backend::is_supported() {
            let inp = to_zreg(self.0);
            let mask: float32x4_t = zerocopy::transmute!([0x8000_0000_u32, 0, 0x8000_0000_u32, 0]);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nrevw z0.d, p0/m, z0.d\neor z0.d, z0.d, z1.d",
                    in("q0") inp,
                    in("q1") mask,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([-self.0[1], self.0[0], -self.0[3], self.0[2]])
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        if ArmSve2Backend::is_supported() {
            let inp = to_zreg(self.0);
            let out: f32;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfaddv s0, p0, z0.s",
                    in("q0") inp,
                    lateout("s0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            out
        } else {
            ScalarVec4(self.0).horizontal_sum()
        }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let (out_l, out_r): (f32, f32);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfaddv s0, p0, z0.s\nfaddv s1, p0, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("s0") out_l,
                    lateout("s1") out_r,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            (out_l, out_r)
        } else {
            (
                ScalarVec4(self.0).horizontal_sum(),
                ScalarVec4(rhs.0).horizontal_sum(),
            )
        }
    }
}

/// ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) hardware backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSve2Backend;

impl ArmSve2Backend {
    /// Detects at runtime whether the host `aarch64` CPU supports `FEAT_SVE2`.
    #[inline]
    pub fn is_supported() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_aarch64_feature_detected!("sve2")
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            use core::sync::atomic::{AtomicU8, Ordering};
            static CACHE: AtomicU8 = AtomicU8::new(0);
            let c = CACHE.load(Ordering::Relaxed);
            if c != 0 {
                return c == 2;
            }
            let (pfr0, zfr0): (u64, u64);
            // SAFETY: `ID_AA64PFR0_EL1` (`SVE` bits [35:32]) and `ID_AA64ZFR0_EL1` (`S3_0_C0_C4_4`,
            // `SVEver` bits [3:0]) are architectural feature registers readable at EL1 (and emulated at EL0 on Linux).
            unsafe {
                core::arch::asm!(
                    "mrs {p}, id_aa64pfr0_el1",
                    p = out(reg) pfr0,
                    options(nomem, nostack, preserves_flags),
                );
                if ((pfr0 >> 32) & 0xF) >= 1 {
                    core::arch::asm!(
                        "mrs {z}, S3_0_C0_C4_4",
                        z = out(reg) zfr0,
                        options(nomem, nostack, preserves_flags),
                    );
                } else {
                    zfr0 = 0;
                }
            }
            let ok = (zfr0 & 0xF) >= 1;
            CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` in SVE2 `z0..z4` registers
/// when `FEAT_SVE2` is supported at runtime.
#[inline(always)]
pub(crate) fn sve2_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> Option<(f32, f32, f32, f32)> {
    if !ArmSve2Backend::is_supported() {
        return None;
    }
    let max_v = to_zreg([max_l, max_r, 1.0, 1.0]);
    let exp_mask: float32x4_t = zerocopy::transmute!([0x7f80_0000_u32; 4]);
    let cap_253: float32x4_t = zerocopy::transmute!([253_u32 << 23; 4]);
    let two54: float32x4_t = zerocopy::transmute!([254_u32 << 23; 4]);
    let (su_v, sd_v): (float32x4_t, float32x4_t);
    // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
    unsafe {
        core::arch::asm!(
            ".arch armv9-a+sve2\nptrue p0.s, vl4\nand z4.d, z0.d, z1.d\numin z4.s, p0/m, z4.s, z2.s\nsub z5.s, z3.s, z4.s",
            in("q0") max_v,
            in("q1") exp_mask,
            in("q2") cap_253,
            in("q3") two54,
            out("q4") su_v,
            out("q5") sd_v,
            out("p0") _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    let (sd, su) = (from_zreg(sd_v), from_zreg(su_v));
    Some((sd[0], su[0], sd[1], su[1]))
}

impl Crc32cBackend for ArmSve2Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSve2Backend {
    type Vec4 = Sve2Vec4;
    const NAME: &'static str = "ARMv9-A SVE2 (Scalable Vector Extension 2)";
    const KIND: BackendKind = BackendKind::ArmSve2;

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if Self::is_supported() {
            let clamped =
                Sve2Vec4::load(src).clamp(Sve2Vec4::splat(-65504.0), Sve2Vec4::splat(65504.0));
            let inp = to_zreg(clamped.0);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfcvt z0.h, p0/m, z0.s\nuzp1 z0.h, z0.h, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let words: [u16; 8] = zerocopy::transmute!(out);
            [words[0], words[1], words[2], words[3]]
        } else {
            ScalarBackend::f32x4_to_fp16x4(src)
        }
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if Self::is_supported() {
            let raw_in: [u16; 8] = [src[0], src[1], src[2], src[3], 0, 0, 0, 0];
            let inp: float32x4_t = zerocopy::transmute!(raw_in);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nuunpklo z0.s, z0.h\nfcvt z0.s, p0/m, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            from_zreg(out)
        } else {
            ScalarBackend::fp16x4_to_f32x4(src)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        Self::rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if Self::is_supported() {
            let v = to_zreg([v0, v1, 1.0, 1.0]);
            let magic: float32x4_t = zerocopy::transmute!([0x5f37_59df_u32; 4]);
            let half = to_zreg([0.5_f32; 4]);
            let three_halves = to_zreg([1.5_f32; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\n",
                    "fmul z4.s, z0.s, z2.s\n",
                    "lsr z5.s, z0.s, #1\n",
                    "sub z5.s, z1.s, z5.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    in("q0") v,
                    in("q1") magic,
                    in("q2") half,
                    in("q3") three_halves,
                    out("q4") _,
                    out("q5") out,
                    out("q6") _,
                    out("q7") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let arr = from_zreg(out);
            (arr[0], arr[1])
        } else {
            ScalarBackend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}
