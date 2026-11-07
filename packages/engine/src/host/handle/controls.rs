//! Non-blocking SPSC control, preset, tempo, bypass, freeze, and stereo-width delegation
//! for [`DnamHostHandle`] (`ref[dsp.ctrl.op-m.spsc-ingress]`).

use alloc::vec::Vec;

use crate::control::ControlClampDescriptor;
use crate::schema::{DnamError, TempoSubdivision};

use super::DnamHostHandle;

impl DnamHostHandle {
    /// Returns the number of user controls $P \in [0, 255]$ wait-free without acquiring any mutex.
    #[inline]
    pub fn num_controls(&self) -> usize {
        self.telemetry_record().num_controls as usize
    }

    /// Returns a copy of the raw user-requested physical control values (`len == num_controls`).
    #[inline]
    pub fn raw_control_values(&self) -> Vec<f32> {
        self.producer
            .lock()
            .producer
            .shadow()
            .raw_control_values()
            .to_vec()
    }

    /// Returns a copy of the clamped physical control values (`len == num_controls`).
    #[inline]
    pub fn clamped_control_values(&self) -> Vec<f32> {
        self.producer
            .lock()
            .producer
            .shadow()
            .clamped_control_values()
            .to_vec()
    }

    /// Returns the clamped physical value for control `ctrl_idx`, or `None` if out of bounds.
    #[inline]
    pub fn clamped_control_value(&self, ctrl_idx: usize) -> Option<f32> {
        self.producer
            .lock()
            .producer
            .shadow()
            .clamped_control_value(ctrl_idx)
    }

    /// Returns a copy of the compiled [`ControlClampDescriptor`]s (`len == num_controls`).
    #[inline]
    pub fn control_descriptors(&self) -> Vec<ControlClampDescriptor> {
        self.producer
            .lock()
            .producer
            .shadow()
            .control_descriptors()
            .to_vec()
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ interpolation via Lane 1 without
    /// blocking on an active audio block.
    pub fn set_control_value(&self, ctrl_idx: usize, physical_value: f32) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_control_value(ctrl_idx, physical_value);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Fallible variant of [`Self::set_control_value`] that validates bounds and finiteness.
    pub fn try_set_control_value(
        &self,
        ctrl_idx: usize,
        physical_value: f32,
    ) -> Result<f32, DnamError> {
        let clamped = {
            let mut prod = self.producer.lock();
            let c = prod
                .producer
                .try_set_control_value(ctrl_idx, physical_value)?;
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
            c
        };
        self.try_drain_if_audio_idle();
        Ok(clamped)
    }

    /// Sets a user control value and snaps immediately via Lane 1 without blocking on an active audio block.
    pub fn set_control_immediate(&self, ctrl_idx: usize, physical_value: f32) {
        {
            let mut prod = self.producer.lock();
            prod.producer
                .set_control_immediate(ctrl_idx, physical_value);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Fallible variant of [`Self::set_control_immediate`] that validates bounds and finiteness.
    pub fn try_set_control_immediate(
        &self,
        ctrl_idx: usize,
        physical_value: f32,
    ) -> Result<f32, DnamError> {
        let clamped = {
            let mut prod = self.producer.lock();
            let c = prod
                .producer
                .try_set_control_immediate(ctrl_idx, physical_value)?;
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
            c
        };
        self.try_drain_if_audio_idle();
        Ok(clamped)
    }

    /// Applies a batch of immediate control updates (`ref[dsp.ctrl.switch.conditional-batch]`),
    /// ordering switch controls before continuous range controls, without blocking on an active audio block.
    pub fn set_controls_immediate_batch(&self, updates: &[(usize, f32)]) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_controls_immediate_batch(updates);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Resets a single control `ctrl_idx` immediately back to its schema default.
    pub fn reset_control(&self, ctrl_idx: usize) -> Result<f32, DnamError> {
        let clamped = {
            let mut prod = self.producer.lock();
            let c = prod.producer.reset_control(ctrl_idx)?;
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
            c
        };
        self.try_drain_if_audio_idle();
        Ok(clamped)
    }

    /// Snaps all controls immediately back to their schema defaults without blocking on an active audio block.
    pub fn reset_controls_to_defaults(&self) {
        {
            let mut prod = self.producer.lock();
            prod.producer.reset_controls_to_defaults();
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Resets all states to analytical DC resting equilibrium without blocking on an active audio block.
    pub fn reset_to_dc_equilibrium(&self) {
        {
            let mut prod = self.producer.lock();
            prod.producer.reset_to_dc_equilibrium();
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns whether this model supports captured hardware freeze wait-free.
    #[inline]
    pub fn supports_freeze(&self) -> bool {
        self.telemetry_record().supports_freeze
    }

    /// Returns `true` if Unitary Spectral Freeze is currently engaged wait-free.
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.telemetry_record().snapshot.is_frozen != 0
    }

    /// Engages or disengages Unitary Spectral Freeze without blocking on an active audio block.
    pub fn set_freeze(&self, freeze: bool) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_freeze(freeze);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Sets continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$
    /// without blocking on an active audio block.
    pub fn set_freeze_amount(&self, alpha: f32) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_freeze_amount(alpha);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns `(bypassed, trails_enabled)` wait-free.
    #[inline]
    pub fn bypass_state(&self) -> (bool, bool) {
        let rec = self.telemetry_record();
        (rec.snapshot.bypassed != 0, rec.trails_enabled)
    }

    /// Sets the bypass state (`bypassed`) and spillover trails mode (`trails`) without blocking on
    /// an active audio block.
    pub fn set_bypass(&self, bypassed: bool, trails: bool) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_bypass(bypassed, trails);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns the output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$ wait-free.
    #[inline]
    pub fn stereo_width(&self) -> f32 {
        self.telemetry_record().stereo_width
    }

    /// Sets the output Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$ without
    /// blocking on an active audio block.
    pub fn set_stereo_width(&self, width: f32) {
        {
            let mut prod = self.producer.lock();
            prod.producer.set_stereo_width(width);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Configures host dry/wet blend, output gain trim (dB), and automatic unity loudness leveling
    /// without blocking on an active audio block.
    pub fn configure_host_mix_and_leveling(
        &self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        {
            let mut prod = self.producer.lock();
            prod.producer.configure_host_mix_and_leveling(
                blend_wet,
                output_gain_db,
                apply_unity_leveling,
            );
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
    }

    /// Returns the active host tempo in BPM wait-free.
    #[inline]
    pub fn current_bpm(&self) -> f32 {
        self.telemetry_record().current_bpm
    }

    /// Synchronizes tempo-calibrated controls to `bpm` and optional `subdivision_override` without
    /// blocking on an active audio block.
    pub fn sync_tempo(
        &self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        let count = {
            let mut prod = self.producer.lock();
            let c = prod
                .producer
                .sync_tempo(bpm, subdivision_override, fold_octave);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
            c
        };
        self.try_drain_if_audio_idle();
        count
    }

    /// Registers a tap-tempo event, updates BPM, and syncs controls without blocking on an active audio block.
    pub fn register_tap_tempo(
        &self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        let res = {
            let mut prod = self.producer.lock();
            let r = prod
                .producer
                .register_tap_tempo(subdivision_override, fold_octave);
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
            r
        };
        self.try_drain_if_audio_idle();
        res
    }

    /// Applies a named preset atomically (`ref[dsp.ctrl.switch.conditional-batch]`) without
    /// blocking on an active audio block.
    pub fn apply_preset(&self, preset_id: &str) -> Result<(), DnamError> {
        {
            let mut prod = self.producer.lock();
            prod.producer.apply_preset(preset_id)?;
            prod.publish_shadow_telemetry(&self.audio_telemetry_reader);
        }
        self.try_drain_if_audio_idle();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
        DnamJsonHeader, PresetDef,
    };
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use proptest::prelude::*;

    fn build_controls_test_container() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
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
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("channel"),
                name: String::from("Channel"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Clean"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![String::from("Clean"), String::from("Crunch")]),
                active_options: Some(vec![String::from("Clean"), String::from("Crunch")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("delay_ms"),
                name: String::from("Delay"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([20.0, 1000.0]),
                safe_clamped_range: Some([50.0, 800.0]),
                default: serde_json::json!(250.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: Some([20.0, 510.0, 1000.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: Some(TempoSubdivision::Quarter),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let mut p1 = BTreeMap::new();
        p1.insert(String::from("gain"), serde_json::json!(8.0));
        p1.insert(String::from("channel"), serde_json::json!("Crunch"));
        header.presets = vec![PresetDef {
            id: String::from("crunch"),
            name: String::from("Crunch"),
            description: None,
            controls: p1,
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_host_controls_presets_tempo_freeze_bypass_and_mix() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_controls_test_container();
        let host = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();

        assert_eq!(host.num_controls(), 3);
        assert_eq!(host.control_descriptors().len(), 3);
        assert!(host.supports_freeze());

        assert!((host.try_set_control_value(0, 7.5).unwrap() - 7.5).abs() < 1e-6);
        assert!(host.try_set_control_value(99, 1.0).is_err());
        assert!(host.try_set_control_value(0, f32::NAN).is_err());

        assert!((host.try_set_control_immediate(0, 6.25).unwrap() - 6.25).abs() < 1e-6);
        assert!(host.try_set_control_immediate(99, 1.0).is_err());
        assert!(host.try_set_control_immediate(0, f32::INFINITY).is_err());

        host.set_control_immediate(0, 9.5);
        assert!((host.raw_control_values()[0] - 9.5).abs() < 1e-6);
        assert!((host.clamped_control_values()[0] - 9.0).abs() < 1e-6);
        assert!((host.clamped_control_value(0).unwrap() - 9.0).abs() < 1e-6);
        assert_eq!(host.clamped_control_value(99), None);

        host.set_controls_immediate_batch(&[(0, 4.0), (1, 1.0)]);
        assert!((host.clamped_control_value(0).unwrap() - 4.0).abs() < 1e-6);
        assert!((host.clamped_control_value(1).unwrap() - 1.0).abs() < 1e-6);

        assert!((host.reset_control(0).unwrap() - 5.0).abs() < 1e-6);
        assert!(host.reset_control(99).is_err());
        host.set_control_value(0, 8.0);
        host.reset_controls_to_defaults();
        assert!((host.clamped_control_value(0).unwrap() - 5.0).abs() < 1e-6);
        host.reset_to_dc_equilibrium();

        host.set_freeze(true);
        assert!(host.is_frozen());
        host.set_freeze_amount(0.2);
        assert!(!host.is_frozen());
        host.set_freeze_amount(0.8);
        assert!(host.is_frozen());

        host.set_bypass(true, true);
        assert_eq!(host.bypass_state(), (true, true));
        host.set_stereo_width(1.4);
        assert!((host.stereo_width() - 1.4).abs() < 1e-6);
        host.configure_host_mix_and_leveling(0.8, -1.0, true);

        assert_eq!(
            host.sync_tempo(120.0, Some(TempoSubdivision::Quarter), true),
            1
        );
        assert!((host.current_bpm() - 120.0).abs() < 1e-5);
        let (_, tap_cnt) = host.register_tap_tempo(Some(TempoSubdivision::Eighth), true);
        assert_eq!(tap_cnt, 1);

        host.apply_preset("crunch").unwrap();
        assert!(host.apply_preset("unknown").is_err());
        assert!((host.clamped_control_value(0).unwrap() - 8.0).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_host_controls_clamp_and_update_wait_free_telemetry(
            raw_gain in -10.0f32..20.0f32,
            width in -1.0f32..3.0f32,
            bypassed in any::<bool>(),
            trails in any::<bool>(),
            frozen in any::<bool>(),
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            // verify[arch.telemetry.seqlock-publication]
            let container = build_controls_test_container();
            let host = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();

            host.set_control_value(0, raw_gain);
            host.set_stereo_width(width);
            host.set_bypass(bypassed, trails);
            host.set_freeze(frozen);

            let expected_gain = raw_gain.clamp(1.0, 9.0);
            let expected_width = width.clamp(0.0, 2.0);
            prop_assert!((host.clamped_control_value(0).unwrap() - expected_gain).abs() < 1e-6);
            prop_assert!((host.stereo_width() - expected_width).abs() < 1e-6);
            prop_assert_eq!(host.bypass_state(), (bypassed, trails));
            prop_assert_eq!(host.is_frozen(), frozen);
        }
    }
}
