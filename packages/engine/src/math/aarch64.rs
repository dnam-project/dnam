//! Unified `aarch64` Hardware Math Library (`ArmSmeBackend`, `ArmSve2Backend`, `NeonBackend`)
//! with runtime CPU feature detection.
//!
//! Submodules:
//! - [`neon`]: Baseline 128-bit NEON (`float32x4_t`, `vfmaq_laneq_f32`, hardware `FCVTN`/`FCVTL`, `__crc32cd`).
//! - [`sve2`]: Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`, `z0..z31` + `p0` `vl4`) without NEON fallback.
//! - [`sme`]: ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime fallback to [`ArmSve2Backend`] and [`NeonBackend`].

pub mod neon;
pub mod sme;
pub mod sve2;

pub use neon::{NeonBackend, NeonVec4};
pub use sme::ArmSmeBackend;
pub use sve2::{ArmSve2Backend, Sve2Vec4};

use super::BackendKind;

/// Detects the highest-capability `aarch64` backend supported by the current CPU at runtime:
/// `ArmSme` (`FEAT_SME`) -> `ArmSve2` (`FEAT_SVE2`) -> `ArmNeon` (baseline `aarch64`).
#[inline]
pub fn detect_aarch64_backend() -> BackendKind {
    if ArmSmeBackend::is_supported() {
        BackendKind::ArmSme
    } else if ArmSve2Backend::is_supported() {
        BackendKind::ArmSve2
    } else {
        BackendKind::ArmNeon
    }
}
