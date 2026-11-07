//! Synchronous producer-side validation and state shadow (`AcceptedStateShadow`)
//! (`ref[dsp.ctrl.op-m.spsc-ingress]`).

use alloc::vec::Vec;

use crate::container::ZohRescaledParams;
use crate::control::{ControlClampDescriptor, TELEMETRY_CONTROL_SLOTS};
use crate::dynamic::DynamicDnamEngine;
use crate::engine::{TempoSyncState, apply_calibrated_control_role};
use crate::host::telemetry::RawTelemetryRecord;
use crate::math::BackendKind;
use crate::provenance::VerificationStatus;
use crate::schema::{DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::ring::{FallbackSnapshot, MAX_CONTROLS};

/// Synchronous producer-side validation and state shadow (`AcceptedStateShadow`)
/// (`ref[dsp.ctrl.op-m.spsc-ingress]`).
///
/// Mirrors the exact clamping, preset resolution, tempo-sync, and host configuration rules of
/// [`DynamicDnamEngine`] so synchronous queries (`raw_control_values`, `clamped_control_values`,
/// `active_tier`, `effective_tier`, `bypass_state`, `is_frozen`, `stereo_width`, `current_bpm`)
/// immediately reflect newly accepted commands before the next audio block drains the SPSC ring.
#[derive(Clone, Debug)]
pub struct AcceptedStateShadow {
    pub(crate) num_controls: usize,
    pub(crate) descriptors: Vec<ControlClampDescriptor>,
    pub(crate) raw_values: Vec<f32>,
    pub(crate) clamped_values: Vec<f32>,
    pub(crate) active_switch_options: Vec<u8>,
    pub(crate) immediate_bits: [u64; 4],
    pub(crate) active_clamp_mask: u16,
    pub(crate) tempo_clamp_mask: u16,
    pub(crate) header: Option<DnamJsonHeader>,
    pub(crate) verification_status: VerificationStatus,
    pub(crate) protection_profile: DnamProtectionProfile,
    pub(crate) backend_kind: BackendKind,
    pub(crate) stereo_out: bool,
    pub(crate) tail_time_ms: f32,
    pub(crate) sample_rate_train_hz: u32,
    pub(crate) sample_rate_hz: u32,
    pub(crate) training_zoh_poles: (f32, f32, f32, u32),
    pub(crate) zoh: ZohRescaledParams,
    pub(crate) tier: Tier,
    pub(crate) max_available_tier: Tier,
    pub(crate) max_supported_tier: Tier,
    pub(crate) offline_bounce: bool,
    pub(crate) maximize_tier_on_bounce: bool,
    pub(crate) supports_freeze: bool,
    pub(crate) is_frozen: bool,
    pub(crate) bypassed: bool,
    pub(crate) trails_enabled: bool,
    pub(crate) stereo_width: f32,
    pub(crate) host_mix: Option<(f32, f32, bool)>,
    pub(crate) tempo_state: TempoSyncState,
}

impl AcceptedStateShadow {
    /// Captures the initial [`AcceptedStateShadow`] from a [`DynamicDnamEngine`].
    pub fn from_engine(engine: &DynamicDnamEngine<'_>) -> Self {
        let num_controls = engine.num_controls();
        let descriptors = engine.control_descriptors().to_vec();
        let raw_values = engine.raw_control_values().to_vec();
        let clamped_values = engine.clamped_control_values().to_vec();
        let mut active_switch_options = alloc::vec![0_u8; num_controls];
        for (p, desc) in descriptors.iter().enumerate() {
            if desc.ctrl_kind == 1 {
                let c = clamped_values.get(p).copied().unwrap_or(desc.default_val);
                active_switch_options[p] = (c.max(0.0) + 0.5) as u8;
            }
        }
        let snap = engine.telemetry_snapshot();
        let (bypassed, trails_enabled) = engine.bypass_state();
        let mut tempo_state = TempoSyncState::new();
        tempo_state.sample_rate_hz = engine.sample_rate_hz();
        tempo_state.set_bpm(engine.current_bpm());
        tempo_state.total_frames = engine.tempo_total_frames();

        Self {
            num_controls,
            descriptors,
            raw_values,
            clamped_values,
            active_switch_options,
            immediate_bits: [0; 4],
            active_clamp_mask: snap.active_clamp_mask,
            tempo_clamp_mask: 0,
            header: engine.header().cloned(),
            verification_status: engine.verification_status().clone(),
            protection_profile: engine.protection_profile(),
            backend_kind: engine.backend_kind(),
            stereo_out: engine.preamble().is_stereo_out(),
            tail_time_ms: engine.preamble().tail_time_ms,
            sample_rate_train_hz: engine.preamble().sample_rate_train_hz,
            sample_rate_hz: engine.sample_rate_hz(),
            training_zoh_poles: engine.training_zoh_poles(),
            zoh: *engine.zoh(),
            tier: engine.tier(),
            max_available_tier: engine.max_available_tier(),
            max_supported_tier: engine.max_supported_tier(),
            offline_bounce: engine.offline_bounce(),
            maximize_tier_on_bounce: engine.maximize_tier_on_bounce(),
            supports_freeze: engine.supports_freeze(),
            is_frozen: engine.is_frozen(),
            bypassed,
            trails_enabled,
            stereo_width: engine.stereo_width(),
            host_mix: None,
            tempo_state,
        }
    }

    pub(crate) fn recompute_clamped_values(&mut self) {
        let clamp_mask = ControlClampDescriptor::clamp_all_controls(
            &self.descriptors,
            self.num_controls,
            &self.raw_values,
            &mut self.clamped_values,
            &mut self.active_switch_options,
        );
        self.active_clamp_mask = clamp_mask | self.tempo_clamp_mask;
    }

    pub(crate) fn refresh_calibrated_control_effects(
        &mut self,
        ctrl_idx: usize,
        physical_value: f32,
    ) {
        apply_calibrated_control_role(
            self.header.as_ref(),
            ctrl_idx,
            physical_value,
            &mut self.stereo_width,
            &mut self.is_frozen,
            self.supports_freeze,
        );
    }

    /// Returns the number of controls $P$.
    #[inline]
    pub const fn num_controls(&self) -> usize {
        self.num_controls
    }

    /// Returns the slice of compiled [`ControlClampDescriptor`]s.
    #[inline]
    pub fn control_descriptors(&self) -> &[ControlClampDescriptor] {
        &self.descriptors
    }

    /// Returns the raw user-requested physical control values.
    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        &self.raw_values
    }

    /// Returns the clamped physical control values.
    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        &self.clamped_values
    }

    /// Returns the clamped physical value for control `ctrl_idx`.
    #[inline]
    pub fn clamped_control_value(&self, ctrl_idx: usize) -> Option<f32> {
        self.clamped_values.get(ctrl_idx).copied()
    }

    /// Returns the active clamp bitmask (`u16`).
    #[inline]
    pub const fn active_clamp_mask(&self) -> u16 {
        self.active_clamp_mask
    }

    /// Returns the requested compute [`Tier`].
    #[inline]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    /// Returns the highest [`Tier`] available in the container.
    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        self.max_available_tier
    }

    /// Returns the host ceiling [`Tier`] (`max_supported_tier`).
    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        self.max_supported_tier
    }

    /// Returns the effective [`Tier`] accounting for `offline_bounce && maximize_tier_on_bounce`.
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        if self.offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
        }
    }

    /// Returns whether $4\times$ oversampled Offline Bounce mode is enabled.
    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.offline_bounce
    }

    /// Returns whether `maximize_tier_on_bounce` is enabled.
    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        self.maximize_tier_on_bounce
    }

    /// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        if self.offline_bounce { 12 } else { 0 }
    }

    /// Returns the active host sample rate in Hz.
    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Returns the active [`ZohRescaledParams`].
    #[inline]
    pub const fn zoh(&self) -> &ZohRescaledParams {
        &self.zoh
    }

    /// Returns whether this model supports Unitary Spectral Freeze.
    #[inline]
    pub const fn supports_freeze(&self) -> bool {
        self.supports_freeze
    }

    /// Returns whether Unitary Spectral Freeze is currently engaged.
    #[inline]
    pub const fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Returns `(bypassed, trails_enabled)`.
    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        (self.bypassed, self.trails_enabled)
    }

    /// Returns the stereo width in `[0.0, 2.0]`.
    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        self.stereo_width
    }

    /// Returns the active host tempo in BPM.
    #[inline]
    pub const fn current_bpm(&self) -> f32 {
        self.tempo_state.current_bpm
    }

    /// Returns the active [`BackendKind`].
    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    /// Returns the container's [`VerificationStatus`].
    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        &self.verification_status
    }

    /// Returns the container's [`DnamProtectionProfile`].
    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        self.protection_profile
    }

    /// Returns the leading 16 clamped control values for [`crate::types::DnamTelemetrySnapshot`].
    #[inline]
    pub fn telemetry_clamped_controls(&self) -> [f32; TELEMETRY_CONTROL_SLOTS] {
        let mut out = [0.0_f32; TELEMETRY_CONTROL_SLOTS];
        let n = self.clamped_values.len().min(TELEMETRY_CONTROL_SLOTS);
        out[..n].copy_from_slice(&self.clamped_values[..n]);
        out
    }

    /// Overlays the accepted control-plane state onto `base` (preserving real-time DSP fields
    /// `z_head`, `env_states`, `lfo_harmonics`, `max_effective_loop_gain`, `overflow_or_saturation_flags`).
    #[inline]
    pub fn overlay_telemetry(
        &self,
        mut base: RawTelemetryRecord,
        causal_epoch: u64,
    ) -> RawTelemetryRecord {
        base.snapshot.active_tier = self.effective_tier() as u8;
        base.snapshot.stereo_out = u8::from(self.stereo_out);
        base.snapshot.bypassed = u8::from(self.bypassed);
        base.snapshot.is_frozen = u8::from(self.is_frozen);
        base.snapshot.active_clamp_mask = self.active_clamp_mask;
        base.snapshot.control_decimation_host = self.zoh.control_decimation_host as u16;
        base.snapshot.tail_time_ms = self.tail_time_ms;
        base.snapshot.clamped_controls = self.telemetry_clamped_controls();
        base.zoh = self.zoh;
        base.stereo_width = self.stereo_width;
        base.current_bpm = self.current_bpm();
        base.latency_samples = self.latency_samples();
        base.num_controls = self.num_controls.min(u8::MAX as usize) as u8;
        base.backend_kind = self.backend_kind;
        base.sample_rate_hz = self.sample_rate_hz;
        base.requested_tier = self.tier;
        base.max_available_tier = self.max_available_tier;
        base.max_supported_tier = self.max_supported_tier;
        base.offline_bounce = self.offline_bounce;
        base.maximize_tier_on_bounce = self.maximize_tier_on_bounce;
        base.supports_freeze = self.supports_freeze;
        base.trails_enabled = self.trails_enabled;
        base.causal_epoch = causal_epoch;
        base
    }

    #[inline]
    pub(crate) fn is_control_immediate(&self, ctrl_idx: usize) -> bool {
        if ctrl_idx < 256 {
            (self.immediate_bits[ctrl_idx / 64] & (1_u64 << (ctrl_idx % 64))) != 0
        } else {
            false
        }
    }

    pub(crate) fn apply_control_update(
        &mut self,
        ctrl_idx: usize,
        physical_value: f32,
        immediate: bool,
    ) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.raw_values[ctrl_idx] = physical_value;
        if ctrl_idx < 256 {
            let word_idx = ctrl_idx / 64;
            let bit = 1_u64 << (ctrl_idx % 64);
            if immediate {
                self.immediate_bits[word_idx] |= bit;
            } else {
                self.immediate_bits[word_idx] &= !bit;
            }
        }
        self.tempo_clamp_mask = 0;
        self.recompute_clamped_values();
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
    }

    pub(crate) fn reset_single_control(&mut self, ctrl_idx: usize) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        let def_val = self.descriptors[ctrl_idx].default_val;
        self.apply_control_update(ctrl_idx, def_val, true);
    }

    pub(crate) fn reset_all_controls(&mut self) {
        self.tempo_clamp_mask = 0;
        for p in 0..self.num_controls {
            let def_val = self.descriptors[p].default_val;
            self.raw_values[p] = def_val;
            self.clamped_values[p] = def_val;
            if self.descriptors[p].ctrl_kind == 1 {
                self.active_switch_options[p] = (def_val.max(0.0) + 0.5) as u8;
            }
        }
        self.immediate_bits = [u64::MAX; 4];
        self.recompute_clamped_values();
        for p in 0..self.num_controls {
            let val = self.clamped_values[p];
            self.refresh_calibrated_control_effects(p, val);
        }
    }

    pub(crate) fn reset_engine(&mut self) {
        self.tempo_state.reset();
        let n = self.num_controls;
        for idx in 0..n {
            let raw_val = self.raw_values[idx];
            self.refresh_calibrated_control_effects(idx, raw_val);
        }
    }

    #[inline]
    pub(crate) fn set_freeze(&mut self, freeze: bool) {
        self.is_frozen = freeze && self.supports_freeze;
    }

    #[inline]
    pub(crate) fn set_freeze_amount(&mut self, alpha: f32) {
        self.is_frozen = (alpha > 0.5) && self.supports_freeze;
    }

    pub(crate) fn to_fallback_snapshot(
        &self,
        epoch: u32,
        reset_engine_requested: bool,
    ) -> FallbackSnapshot {
        let mut raw_arr = [0.0_f32; MAX_CONTROLS];
        let copy_n = self.raw_values.len().min(MAX_CONTROLS);
        raw_arr[..copy_n].copy_from_slice(&self.raw_values[..copy_n]);
        FallbackSnapshot {
            epoch,
            num_controls: self.num_controls as u16,
            reset_engine_requested,
            raw_values: raw_arr,
            immediate_mask: self.immediate_bits[0] as u32,
            tier: self.tier,
            max_supported_tier: self.max_supported_tier,
            offline_bounce: self.offline_bounce,
            maximize_tier_on_bounce: self.maximize_tier_on_bounce,
            bypassed: self.bypassed,
            trails_enabled: self.trails_enabled,
            is_frozen: self.is_frozen,
            stereo_width: self.stereo_width,
            host_mix: self.host_mix,
            current_bpm: self.tempo_state.current_bpm,
            tempo_clamp_mask: self.tempo_clamp_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    };
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use proptest::prelude::*;

    fn build_shadow_test_engine() -> DynamicDnamEngine<'static> {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        let mut cond_max = BTreeMap::new();
        cond_max.insert(String::from("mode=Mono"), 6.0);
        header.controls_schema = vec![
            ControlSchemaDef {
                id: String::from("gain"),
                name: String::from("Gain"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([1.0, 9.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: cond_max,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("mode"),
                name: String::from("Mode"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Stereo"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![
                    String::from("Mono"),
                    String::from("Stereo"),
                    String::from("Wide"),
                ]),
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoMode,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("freeze_sw"),
                name: String::from("Freeze"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Off"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![String::from("Off"), String::from("On")]),
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::FreezeLatch,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        let bytes = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        DynamicDnamEngine::from_bytes_owned_with_backend(
            &bytes,
            Tier::Lite,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap()
    }

    #[test]
    fn test_accepted_state_shadow_getters_roles_and_telemetry_overlay() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        // verify[arch.telemetry.seqlock-publication]
        let engine = build_shadow_test_engine();
        let mut shadow = AcceptedStateShadow::from_engine(&engine);

        assert_eq!(shadow.num_controls(), 3);
        assert_eq!(shadow.control_descriptors().len(), 3);
        assert_eq!(shadow.raw_control_values().len(), 3);
        assert_eq!(shadow.clamped_control_values().len(), 3);
        assert_eq!(shadow.tier(), Tier::Lite);
        assert_eq!(shadow.max_available_tier(), Tier::Ultra);
        assert_eq!(shadow.max_supported_tier(), Tier::Ultra);
        assert_eq!(shadow.effective_tier(), Tier::Lite);
        assert!(!shadow.offline_bounce());
        assert!(shadow.maximize_tier_on_bounce());
        assert_eq!(shadow.latency_samples(), 0);
        assert_eq!(shadow.sample_rate_hz(), 48_000);
        let _ = shadow.zoh();
        assert!(shadow.supports_freeze());
        assert!(!shadow.is_frozen());
        assert_eq!(shadow.bypass_state(), (false, false));
        assert!((shadow.stereo_width() - 1.0).abs() < 1e-6);
        assert!((shadow.current_bpm() - 120.0).abs() < 1e-6);
        assert_eq!(shadow.backend_kind(), BackendKind::Scalar);
        let _ = shadow.verification_status();
        let _ = shadow.protection_profile();

        // Out-of-bounds updates and resets are safe no-ops
        shadow.apply_control_update(99, 1.0, false);
        shadow.reset_single_control(99);
        assert_eq!(shadow.clamped_control_value(99), None);

        // Conditional clamp when mode == Mono (0) and gain == 8.0 -> clamps to 6.0
        shadow.apply_control_update(1, 0.0, true);
        assert!((shadow.stereo_width() - 0.0).abs() < 1e-6);
        shadow.apply_control_update(0, 8.0, false);
        assert!((shadow.clamped_control_value(0).unwrap() - 6.0).abs() < 1e-6);
        assert_ne!(shadow.active_clamp_mask() & 1, 0);

        // FreezeLatch switch role
        shadow.apply_control_update(2, 1.0, true);
        assert!(shadow.is_frozen());
        shadow.set_freeze_amount(0.2);
        assert!(!shadow.is_frozen());
        shadow.set_freeze(true);
        assert!(shadow.is_frozen());

        // Bounce effective tier & latency
        shadow.offline_bounce = true;
        shadow.maximize_tier_on_bounce = true;
        assert_eq!(shadow.effective_tier(), Tier::Ultra);
        assert_eq!(shadow.latency_samples(), 12);

        // Overlay telemetry & fallback snapshot
        let base_rec = RawTelemetryRecord::from_engine(&engine, 1);
        let overlaid = shadow.overlay_telemetry(base_rec, 99);
        assert_eq!(overlaid.causal_epoch, 99);
        assert_eq!(overlaid.snapshot.active_tier, Tier::Ultra as u8);
        assert_eq!(overlaid.latency_samples, 12);

        let fb = shadow.to_fallback_snapshot(77, true);
        assert_eq!(fb.epoch, 77);
        assert!(fb.reset_engine_requested);

        // Reset single and all controls and engine
        shadow.reset_single_control(0);
        assert!((shadow.clamped_control_value(0).unwrap() - 5.0).abs() < 1e-6);
        shadow.reset_all_controls();
        shadow.reset_engine();
        assert!((shadow.stereo_width() - 1.0).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_shadow_clamping_matches_engine(
            gain in -5.0f32..15.0f32,
            mode_idx in 0u32..3,
            freeze_idx in 0u32..2,
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            let mut engine = build_shadow_test_engine();
            let mut shadow = AcceptedStateShadow::from_engine(&engine);

            engine.set_control_immediate(1, mode_idx as f32);
            shadow.apply_control_update(1, mode_idx as f32, true);

            engine.set_control_value(0, gain);
            shadow.apply_control_update(0, gain, false);

            engine.set_control_immediate(2, freeze_idx as f32);
            shadow.apply_control_update(2, freeze_idx as f32, true);

            prop_assert_eq!(engine.clamped_control_values(), shadow.clamped_control_values());
            prop_assert_eq!(engine.is_frozen(), shadow.is_frozen());
            prop_assert!((engine.stereo_width() - shadow.stereo_width()).abs() < 1e-6);
        }
    }
}
