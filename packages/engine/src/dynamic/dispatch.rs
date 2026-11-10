//! Method dispatch and tests for [`DynamicDnamEngine`].

use super::DynamicDnamEngine;
use crate::container::{DnamPreamble, ZohRescaledParams};
use crate::control::ControlClampDescriptor;
use crate::provenance::VerificationStatus;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision};
use crate::types::{DnamTelemetrySnapshot, Tier};

macro_rules! dispatch_engine {
    ($self:expr, $e:ident => $body:expr) => {
        match $self {
            DynamicDnamEngine::Scalar($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::Neon($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::ArmSve2($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::ArmSme($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Sse2($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Sse42($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Avx2Fma($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Simd($e) => $body,
            #[cfg(target_arch = "wasm32")]
            DynamicDnamEngine::WasmSimd128($e) => $body,
        }
    };
}

impl<'a> DynamicDnamEngine<'a> {
    /// Returns the container's resolved [`DnamProtectionProfile`].
    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        dispatch_engine!(self, e => e.protection_profile())
    }

    /// Returns the container's cryptographic [`VerificationStatus`].
    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        dispatch_engine!(self, e => e.verification_status())
    }

    /// Returns the loaded container [`DnamPreamble`].
    #[inline]
    pub const fn preamble(&self) -> &DnamPreamble {
        dispatch_engine!(self, e => e.preamble())
    }

    /// Returns the parsed [`DnamJsonHeader`] if loaded.
    #[inline]
    pub const fn header(&self) -> Option<&DnamJsonHeader> {
        dispatch_engine!(self, e => e.header())
    }

    /// Returns the active host sample rate in Hz.
    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        dispatch_engine!(self, e => e.sample_rate_hz())
    }

    /// Returns the current [`ZohRescaledParams`] for the active host sample rate.
    #[inline]
    pub const fn zoh(&self) -> &ZohRescaledParams {
        dispatch_engine!(self, e => e.zoh())
    }

    /// Switches the active [`Tier`].
    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        dispatch_engine!(self, e => e.set_tier(tier))
    }

    /// Returns the requested [`Tier`].
    #[inline]
    pub const fn tier(&self) -> Tier {
        dispatch_engine!(self, e => e.tier())
    }

    /// Returns the highest [`Tier`] present in the loaded `.dnam` container.
    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.max_available_tier())
    }

    /// Returns the plugin/host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.max_supported_tier())
    }

    /// Sets the plugin/host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        dispatch_engine!(self, e => e.set_max_supported_tier(max_tier))
    }

    /// Returns the effective [`Tier`] (`min(max_available_tier, max_supported_tier)` when
    /// `offline_bounce && maximize_tier_on_bounce`, otherwise `self.tier()`).
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.effective_tier())
    }

    /// Returns `true` if Offline Bounce mode ($4\times$ linear-phase polyphase oversampling) is enabled.
    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        dispatch_engine!(self, e => e.offline_bounce())
    }

    /// Enables or disables Offline Bounce mode ($4\times$ linear-phase polyphase oversampling).
    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_offline_bounce(enabled))
    }

    /// Returns `true` if `maximize_tier_on_bounce` is enabled.
    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        dispatch_engine!(self, e => e.maximize_tier_on_bounce())
    }

    /// Enables or disables automatic tier promotion during Offline Bounce.
    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_maximize_tier_on_bounce(enabled))
    }

    /// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        dispatch_engine!(self, e => e.latency_samples())
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
        dispatch_engine!(self, e => e.render_offline_bounce(left_in, right_in, left_out, right_out))
    }

    /// Rescales discrete-time parameters for a new host sample rate.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.set_sample_rate(sample_rate_hz))
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        dispatch_engine!(self, e => e.set_control_value(ctrl_idx, value))
    }

    /// Sets a user control value and snaps immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        dispatch_engine!(self, e => e.set_control_immediate(ctrl_idx, value))
    }

    /// Returns the number of user controls $P \in [0, 255]$.
    #[inline]
    pub const fn num_controls(&self) -> usize {
        dispatch_engine!(self, e => e.num_controls())
    }

    /// Returns the slice of raw user-requested physical control values (`len == num_controls`).
    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        dispatch_engine!(self, e => e.raw_control_values())
    }

    /// Returns the slice of clamped physical control values (`len == num_controls`).
    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        dispatch_engine!(self, e => e.clamped_control_values())
    }

    /// Returns the compiled 64-byte [`ControlClampDescriptor`] slice (`len == num_controls`).
    #[inline]
    pub fn control_descriptors(&self) -> &[ControlClampDescriptor] {
        dispatch_engine!(self, e => e.control_descriptors())
    }

    /// Snaps all controls immediately back to their schema defaults and refreshes static state.
    #[inline]
    pub fn reset_controls_to_defaults(&mut self) {
        dispatch_engine!(self, e => e.reset_controls_to_defaults())
    }

    /// Resets all states to their analytical DC resting equilibrium.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        dispatch_engine!(self, e => e.reset_to_dc_equilibrium())
    }

    /// Returns whether this model supports captured hardware freeze (`supports_freeze == true` or
    /// a `FreezeLatch`-calibrated control in `controls_schema`, `ref[dsp.stage4.captured-freeze]`).
    #[inline]
    pub fn supports_freeze(&self) -> bool {
        dispatch_engine!(self, e => e.supports_freeze())
    }

    /// Engages or disengages Unitary Spectral Freeze (`alpha = 1.0` vs `0.0`).
    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        dispatch_engine!(self, e => e.set_freeze(freeze))
    }

    /// Sets the continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        dispatch_engine!(self, e => e.set_freeze_amount(alpha))
    }

    /// Returns `true` if Unitary Spectral Freeze is currently engaged ($\alpha_{\text{freeze}} > 0.5$).
    #[inline]
    pub fn is_frozen(&self) -> bool {
        dispatch_engine!(self, e => e.is_frozen())
    }

    /// Sets the bypass state (`bypassed`) and spillover trails mode (`trails`).
    #[inline]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        dispatch_engine!(self, e => e.set_bypass(bypassed, trails))
    }

    /// Returns `(bypassed, trails_enabled)`.
    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        dispatch_engine!(self, e => e.bypass_state())
    }

    /// Sets the output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$.
    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        dispatch_engine!(self, e => e.set_stereo_width(width))
    }

    /// Returns the output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$.
    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        dispatch_engine!(self, e => e.stereo_width())
    }

    /// Configures host dry/wet blend, output gain trim (dB), and automatic unity loudness leveling.
    #[inline]
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        dispatch_engine!(self, e => e.configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity_leveling))
    }

    /// Returns the active host tempo in BPM.
    #[inline]
    pub const fn current_bpm(&self) -> f32 {
        dispatch_engine!(self, e => e.current_bpm())
    }

    /// Sets the active host tempo in BPM without modifying control values.
    #[inline]
    pub fn set_bpm(&mut self, bpm: f32) {
        dispatch_engine!(self, e => e.set_bpm(bpm))
    }

    /// Accumulates additional clamping bits into `active_clamp_mask`.
    #[inline]
    pub(crate) fn or_active_clamp_mask(&mut self, mask: u16) {
        dispatch_engine!(self, e => e.or_active_clamp_mask(mask))
    }

    /// Returns `(r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr)` for ZOH pole rescaling.
    #[inline]
    pub(crate) fn training_zoh_poles(&self) -> (f32, f32, f32, u32) {
        dispatch_engine!(self, e => e.training_zoh_poles())
    }

    /// Returns the cumulative number of processed audio frames tracked for tap-tempo intervals.
    #[inline]
    pub(crate) const fn tempo_total_frames(&self) -> u64 {
        dispatch_engine!(self, e => e.tempo_state.total_frames)
    }

    /// Synchronizes tempo-calibrated controls to `bpm` and optional `subdivision_override`.
    #[inline]
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        dispatch_engine!(self, e => e.sync_tempo(bpm, subdivision_override, fold_octave))
    }

    /// Registers a tap-tempo event at the current audio frame counter, updates BPM, and syncs controls.
    #[inline]
    pub fn register_tap_tempo(
        &mut self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        dispatch_engine!(self, e => e.register_tap_tempo(subdivision_override, fold_octave))
    }

    /// Applies a named preset atomically (`ref[dsp.ctrl.switch.conditional-batch]`).
    #[inline]
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.apply_preset(preset_id))
    }

    /// Applies a preset by index without any heap allocations (`ref[dsp.ctrl.op-m.spsc-ingress]`).
    #[inline]
    pub fn apply_preset_by_index(&mut self, preset_index: usize) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.apply_preset_by_index(preset_index))
    }

    /// Real-time preset application by index returning `bool` (zero `DnamError` heap allocation or drop glue).
    #[inline]
    pub(crate) fn apply_preset_by_index_rt(&mut self, preset_index: usize) -> bool {
        dispatch_engine!(self, e => e.apply_preset_by_index_rt(preset_index))
    }

    /// Processes a stereo audio block in-place (`left_io`, `right_io`).
    #[inline]
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        dispatch_engine!(self, e => e.process_block_in_place(left_io, right_io))
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
        dispatch_engine!(self, e => e.process_block(left_in, right_in, left_out, right_out))
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    #[inline]
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_mono_to_stereo(mono_in, left_out, right_out))
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    #[inline]
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        dispatch_engine!(self, e => e.process_mono(mono_in, mono_out))
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
        dispatch_engine!(self, e => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz))
    }

    /// Returns a lock-free [`DnamTelemetrySnapshot`].
    #[inline]
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        dispatch_engine!(self, e => e.telemetry_snapshot())
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
            BackendKind::ArmSve2,
            BackendKind::ArmSme,
            BackendKind::X86Sse2,
            BackendKind::X86Sse42,
            BackendKind::X86Avx2Fma,
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
            assert!(dyn_eng.supports_freeze());
            assert_eq!(dyn_eng.sample_rate_hz(), 44_100);
            assert_eq!(dyn_eng.zoh().sample_rate_host_hz, 44_100);
            let _ = dyn_eng.verification_status();
            let _ = dyn_eng.preamble();
            assert!(dyn_eng.header().is_some());
            assert_eq!(dyn_eng.num_controls(), 0);
            assert_eq!(dyn_eng.raw_control_values().len(), dyn_eng.num_controls());
            assert_eq!(
                dyn_eng.clamped_control_values().len(),
                dyn_eng.num_controls()
            );
            assert_eq!(dyn_eng.control_descriptors().len(), dyn_eng.num_controls());
            dyn_eng.reset_controls_to_defaults();
            dyn_eng.set_bypass(true, true);
            assert_eq!(dyn_eng.bypass_state(), (true, true));
            dyn_eng.set_bypass(false, false);
            assert_eq!(dyn_eng.bypass_state(), (false, false));
            dyn_eng.set_stereo_width(1.5);
            assert!((dyn_eng.stereo_width() - 1.5).abs() < 1e-6);
            dyn_eng.configure_host_mix_and_leveling(0.8, -1.5, true);
            let _ = dyn_eng.sync_tempo(128.0, Some(TempoSubdivision::Eighth), false);
            assert!((dyn_eng.current_bpm() - 128.0).abs() < 1e-6);
            let (tap_bpm, _) = dyn_eng.register_tap_tempo(None, false);
            assert!(tap_bpm >= 20.0);
            assert!(dyn_eng.apply_preset("nonexistent").is_err());

            let l_in = [0.1f32; 32];
            let r_in = [-0.1f32; 32];
            let mut l_out = [0.0f32; 32];
            let mut r_out = [0.0f32; 32];
            dyn_eng.render_offline_bounce(&l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.process_block(&l_in, &r_in, &mut l_out, &mut r_out);
            let mut l_ip = l_in;
            let mut r_ip = r_in;
            dyn_eng.process_block_in_place(&mut l_ip, &mut r_ip);
            dyn_eng.process_mono_to_stereo(&l_in, &mut l_out, &mut r_out);
            dyn_eng.process_mono(&l_in, &mut l_out);
            dyn_eng.reset_to_dc_equilibrium();
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

    proptest::proptest! {
        #[test]
        fn prop_dynamic_engine_in_place_and_forwarding_parity(
            width in 0.0f32..=2.0f32,
            bpm in 20.0f32..=400.0f32,
            bypassed in proptest::bool::ANY,
            trails in proptest::bool::ANY,
            blend_wet in 0.0f32..=1.0f32,
            gain_db in -12.0f32..=12.0f32,
            in_sample in -1.0f32..=1.0f32,
        ) {
            let mut header = DnamJsonHeader::default_reference();
            let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
            let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

            let mut eng_oop = DynamicDnamEngine::from_bytes_with_backend(
                &container,
                Tier::Lite,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();
            let mut eng_ip = DynamicDnamEngine::from_bytes_with_backend(
                &container,
                Tier::Lite,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();

            for eng in [&mut eng_oop, &mut eng_ip] {
                eng.set_stereo_width(width);
                eng.set_bypass(bypassed, trails);
                eng.configure_host_mix_and_leveling(blend_wet, gain_db, true);
                eng.sync_tempo(bpm, None, false);
            }

            let l_in = [in_sample; 16];
            let r_in = [-in_sample * 0.5; 16];
            let mut l_out = [0.0f32; 16];
            let mut r_out = [0.0f32; 16];
            eng_oop.process_block(&l_in, &r_in, &mut l_out, &mut r_out);

            let mut l_ip = l_in;
            let mut r_ip = r_in;
            eng_ip.process_block_in_place(&mut l_ip, &mut r_ip);

            for i in 0..16 {
                proptest::prop_assert_eq!(l_out[i].to_bits(), l_ip[i].to_bits());
                proptest::prop_assert_eq!(r_out[i].to_bits(), r_ip[i].to_bits());
            }
        }
    }
}
