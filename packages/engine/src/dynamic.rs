//! Block-level runtime-dispatched DNAM engine (`DynamicDnamEngine<'a>`).

use crate::engine::DnamEngine;
use crate::math::{BackendKind, ScalarBackend};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider};
use crate::schema::{DnamError, DnamProtectionProfile};
use crate::types::{DnamTelemetrySnapshot, Tier};

/// Runtime-dispatched wrapper around [`DnamEngine`] supporting block-level backend selection
/// ([`BackendKind`]) with **zero per-sample branching**.
#[derive(Debug)]
pub enum DynamicDnamEngine<'a> {
    /// Portable `#![no_std]` IEEE-754 scalar backend.
    Scalar(DnamEngine<'a, ScalarBackend>),
    /// ARM64 NEON `float32x4_t` backend.
    #[cfg(target_arch = "aarch64")]
    Neon(DnamEngine<'a, crate::math::neon::NeonBackend>),
    /// ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) + NEON hybrid backend.
    #[cfg(target_arch = "aarch64")]
    ArmSme(DnamEngine<'a, crate::math::arm_sme::ArmSmeBackend>),
    /// x86_64 AVX2+FMA / SSE2 backend.
    #[cfg(target_arch = "x86_64")]
    X86Simd(DnamEngine<'a, crate::math::x86_simd::X86SimdBackend>),
    /// WebAssembly SIMD128 `v128` backend.
    #[cfg(target_arch = "wasm32")]
    WasmSimd128(DnamEngine<'a, crate::math::wasm_simd128::WasmSimd128Backend>),
}

impl<'a> DynamicDnamEngine<'a> {
    /// Creates a [`DynamicDnamEngine`] using runtime CPU feature detection
    /// ([`crate::math::detect_best_backend`]: `ArmSme` if `FEAT_SME` is supported on `aarch64`,
    /// otherwise `Neon`, `X86Simd`, `WasmSimd128`, or `Scalar`).
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        Self::from_bytes_with_backend(
            bytes,
            tier,
            sample_rate_hz,
            crate::math::detect_best_backend(),
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using runtime CPU feature detection.
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using a custom [`CryptoProvider`] and runtime CPU feature detection.
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            crate::math::detect_best_backend(),
            crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`].
    pub fn from_bytes_owned_with_backend(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            backend,
            &crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`] and [`CryptoProvider`].
    pub fn from_bytes_owned_with_backend_and_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Ok(DynamicDnamEngine::ArmSme(
                DnamEngine::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, crypto)?,
            )),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => Ok(DynamicDnamEngine::Neon(
                DnamEngine::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, crypto)?,
            )),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => {
                Ok(DynamicDnamEngine::X86Simd(
                    DnamEngine::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, crypto)?,
                ))
            }
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => Ok(DynamicDnamEngine::WasmSimd128(
                DnamEngine::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, crypto)?,
            )),
            _ => Ok(DynamicDnamEngine::Scalar(
                DnamEngine::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, crypto)?,
            )),
        }
    }

    /// Creates a [`DynamicDnamEngine`] with an explicit [`BackendKind`] (falling back to `Scalar`
    /// if the requested ISA is not compiled for the current target).
    pub fn from_bytes_with_backend(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<Self, DnamError> {
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Ok(Self::ArmSme(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => Ok(Self::Neon(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => Ok(
                Self::X86Simd(DnamEngine::from_bytes(bytes, tier, sample_rate_hz)?),
            ),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => Ok(Self::WasmSimd128(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
            _ => Ok(Self::Scalar(DnamEngine::from_bytes(
                bytes,
                tier,
                sample_rate_hz,
            )?)),
        }
    }

    /// Returns the active [`BackendKind`].
    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Scalar(_) => BackendKind::Scalar,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => BackendKind::Neon,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(_) => BackendKind::ArmSme,
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(_) => BackendKind::X86Simd,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(_) => BackendKind::WasmSimd128,
        }
    }

    /// Returns the container's resolved [`DnamProtectionProfile`].
    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        match self {
            Self::Scalar(e) => e.protection_profile(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.protection_profile(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.protection_profile(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.protection_profile(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.protection_profile(),
        }
    }

    /// Switches the active [`Tier`].
    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        match self {
            Self::Scalar(e) => e.set_tier(tier),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_tier(tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_tier(tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_tier(tier),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_tier(tier),
        }
    }

    /// Rescales discrete-time parameters for a new host sample rate.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        match self {
            Self::Scalar(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_sample_rate(sample_rate_hz),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_sample_rate(sample_rate_hz),
        }
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        match self {
            Self::Scalar(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_control_value(ctrl_idx, value),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_control_value(ctrl_idx, value),
        }
    }

    /// Sets a user control value and snaps immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        match self {
            Self::Scalar(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_control_immediate(ctrl_idx, value),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_control_immediate(ctrl_idx, value),
        }
    }

    /// Resets all states to their analytical DC resting equilibrium.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        match self {
            Self::Scalar(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.reset_to_dc_equilibrium(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.reset_to_dc_equilibrium(),
        }
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
    #[inline]
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        match self {
            Self::Scalar(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.process_block(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.process_block(left_in, right_in, left_out, right_out),
        }
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    #[inline]
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        match self {
            Self::Scalar(e) => e.process_mono_to_stereo(mono_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.process_mono_to_stereo(mono_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.process_mono_to_stereo(mono_in, left_out, right_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.process_mono_to_stereo(mono_in, left_out, right_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.process_mono_to_stereo(mono_in, left_out, right_out),
        }
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    #[inline]
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        match self {
            Self::Scalar(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.process_mono(mono_in, mono_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.process_mono(mono_in, mono_out),
        }
    }

    /// Hot-swaps a new `.dnam` model in-place into an owned engine without requiring `bytes`
    /// to outlive `'a`.
    #[inline]
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        match self {
            Self::Scalar(e) => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz),
        }
    }

    /// Returns a lock-free [`DnamTelemetrySnapshot`].
    #[inline]
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        match self {
            Self::Scalar(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.telemetry_snapshot(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.telemetry_snapshot(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;

    #[test]
    fn test_dynamic_dnam_engine_all_variants() {
        let mut header = DnamJsonHeader::default_reference(false);
        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let kinds = [
            BackendKind::Scalar,
            BackendKind::Neon,
            BackendKind::ArmSme,
            BackendKind::X86Simd,
            BackendKind::WasmSimd128,
        ];
        for kind in kinds {
            let mut dyn_eng =
                DynamicDnamEngine::from_bytes_with_backend(&container, Tier::Ultra, 48_000, kind)
                    .unwrap();
            let _ = dyn_eng.backend_kind();
            dyn_eng.set_tier(Tier::Lite);
            dyn_eng.set_sample_rate(44_100).unwrap();
            dyn_eng.set_control_value(0, 5.0);
            dyn_eng.set_control_immediate(0, 5.0);
            dyn_eng.reset_to_dc_equilibrium();

            let l_in = [0.1f32; 32];
            let r_in = [-0.1f32; 32];
            let mut l_out = [0.0f32; 32];
            let mut r_out = [0.0f32; 32];
            dyn_eng.process_block(&l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.process_mono_to_stereo(&l_in, &mut l_out, &mut r_out);
            dyn_eng.process_mono(&l_in, &mut l_out);
            dyn_eng
                .reload_model_in_place_owned(&container, Tier::Lite, 48_000)
                .unwrap();
            let snap = dyn_eng.telemetry_snapshot();
            assert_eq!(snap.active_tier, Tier::Lite as u8);

            let _owned_kind = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Lite,
                48_000,
                kind,
            )
            .unwrap();
        }

        let _owned = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        let _auto = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    }
}
