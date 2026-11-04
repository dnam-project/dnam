//! Method dispatch and tests for [`DynamicDnamEngine`].

use super::DynamicDnamEngine;
use crate::schema::{DnamError, DnamProtectionProfile};
use crate::types::{DnamTelemetrySnapshot, Tier};

impl<'a> DynamicDnamEngine<'a> {
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

    /// Returns the requested [`Tier`].
    #[inline]
    pub const fn tier(&self) -> Tier {
        match self {
            Self::Scalar(e) => e.tier(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.tier(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.tier(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.tier(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.tier(),
        }
    }

    /// Returns the highest [`Tier`] present in the loaded `.dnam` container.
    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        match self {
            Self::Scalar(e) => e.max_available_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.max_available_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.max_available_tier(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.max_available_tier(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.max_available_tier(),
        }
    }

    /// Returns the plugin/host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        match self {
            Self::Scalar(e) => e.max_supported_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.max_supported_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.max_supported_tier(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.max_supported_tier(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.max_supported_tier(),
        }
    }

    /// Sets the plugin/host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        match self {
            Self::Scalar(e) => e.set_max_supported_tier(max_tier),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_max_supported_tier(max_tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_max_supported_tier(max_tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_max_supported_tier(max_tier),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_max_supported_tier(max_tier),
        }
    }

    /// Returns the effective [`Tier`] (`min(max_available_tier, max_supported_tier)` when
    /// `offline_bounce && maximize_tier_on_bounce`, otherwise `self.tier()`).
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        match self {
            Self::Scalar(e) => e.effective_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.effective_tier(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.effective_tier(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.effective_tier(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.effective_tier(),
        }
    }

    /// Returns `true` if Offline Bounce mode ($4\times$ linear-phase polyphase oversampling) is enabled.
    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        match self {
            Self::Scalar(e) => e.offline_bounce(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.offline_bounce(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.offline_bounce(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.offline_bounce(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.offline_bounce(),
        }
    }

    /// Enables or disables Offline Bounce mode ($4\times$ linear-phase polyphase oversampling).
    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        match self {
            Self::Scalar(e) => e.set_offline_bounce(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_offline_bounce(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_offline_bounce(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_offline_bounce(enabled),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_offline_bounce(enabled),
        }
    }

    /// Returns `true` if `maximize_tier_on_bounce` is enabled.
    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        match self {
            Self::Scalar(e) => e.maximize_tier_on_bounce(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.maximize_tier_on_bounce(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.maximize_tier_on_bounce(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.maximize_tier_on_bounce(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.maximize_tier_on_bounce(),
        }
    }

    /// Enables or disables automatic tier promotion during Offline Bounce.
    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        match self {
            Self::Scalar(e) => e.set_maximize_tier_on_bounce(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_maximize_tier_on_bounce(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_maximize_tier_on_bounce(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_maximize_tier_on_bounce(enabled),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_maximize_tier_on_bounce(enabled),
        }
    }

    /// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        match self {
            Self::Scalar(e) => e.latency_samples(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.latency_samples(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.latency_samples(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.latency_samples(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.latency_samples(),
        }
    }

    /// Renders a complete stereo buffer in Offline Bounce mode, automatically stripping the 12-sample PDC lead-in.
    #[inline]
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        match self {
            Self::Scalar(e) => e.render_offline_bounce(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.render_offline_bounce(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.render_offline_bounce(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.render_offline_bounce(left_in, right_in, left_out, right_out),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.render_offline_bounce(left_in, right_in, left_out, right_out),
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

    /// Engages or disengages Unitary Spectral Freeze (`alpha = 1.0` vs `0.0`).
    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        match self {
            Self::Scalar(e) => e.set_freeze(freeze),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_freeze(freeze),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_freeze(freeze),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_freeze(freeze),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_freeze(freeze),
        }
    }

    /// Sets the continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        match self {
            Self::Scalar(e) => e.set_freeze_amount(alpha),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.set_freeze_amount(alpha),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.set_freeze_amount(alpha),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.set_freeze_amount(alpha),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.set_freeze_amount(alpha),
        }
    }

    /// Returns `true` if Unitary Spectral Freeze is currently engaged ($\alpha_{\text{freeze}} > 0.5$).
    #[inline]
    pub fn is_frozen(&self) -> bool {
        match self {
            Self::Scalar(e) => e.is_frozen(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.is_frozen(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.is_frozen(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.is_frozen(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.is_frozen(),
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
    use crate::math::BackendKind;
    use crate::schema::DnamJsonHeader;

    #[test]
    fn test_dynamic_dnam_engine_all_variants() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        // Verify unmodeled freeze (`supports_freeze == false`) is ignored
        let no_freeze_container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut no_freeze_eng = DynamicDnamEngine::from_bytes_with_backend(
            &no_freeze_container,
            Tier::Lite,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        no_freeze_eng.set_freeze(true);
        assert!(!no_freeze_eng.is_frozen());

        header.io_config.capabilities.supports_freeze = true;
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
            dyn_eng.set_freeze(true);
            assert!(dyn_eng.is_frozen());
            dyn_eng.set_freeze_amount(0.0);
            assert!(!dyn_eng.is_frozen());
            let _ = dyn_eng.protection_profile();
            let _ = dyn_eng.tier();
            let _ = dyn_eng.max_available_tier();
            let _ = dyn_eng.max_supported_tier();
            dyn_eng.set_max_supported_tier(Tier::Full);
            dyn_eng.set_offline_bounce(true);
            assert!(dyn_eng.offline_bounce());
            assert_eq!(dyn_eng.latency_samples(), 12);
            dyn_eng.set_offline_bounce(false);
            assert_eq!(dyn_eng.latency_samples(), 0);
            dyn_eng.set_maximize_tier_on_bounce(true);
            assert!(dyn_eng.maximize_tier_on_bounce());
            assert_eq!(dyn_eng.effective_tier(), Tier::Lite);

            let l_in = [0.1f32; 32];
            let r_in = [-0.1f32; 32];
            let mut l_out = [0.0f32; 32];
            let mut r_out = [0.0f32; 32];
            dyn_eng.render_offline_bounce(&l_in, &r_in, &mut l_out, &mut r_out);
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
