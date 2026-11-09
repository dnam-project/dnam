//! Host Tempo & Musical Subdivision Synchronization (`TempoSyncState`).

use crate::schema::{DnamJsonHeader, TempoSubdivision};

/// Host tempo synchronization and tap-tempo tracking state.
#[derive(Clone, Debug, PartialEq)]
pub struct TempoSyncState {
    /// Active host tempo in Beats Per Minute (BPM).
    pub current_bpm: f32,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Tap tempo circular buffer of frame intervals.
    tap_intervals: [u32; 4],
    tap_idx: usize,
    last_tap_frame: Option<u64>,
    pub(crate) total_frames: u64,
}

impl Default for TempoSyncState {
    fn default() -> Self {
        Self::new()
    }
}

impl TempoSyncState {
    /// Creates a default [`TempoSyncState`] at 120.0 BPM and 48,000 Hz.
    pub const fn new() -> Self {
        Self {
            current_bpm: 120.0,
            sample_rate_hz: 48_000,
            tap_intervals: [0; 4],
            tap_idx: 0,
            last_tap_frame: None,
            total_frames: 0,
        }
    }

    /// Resets tap-tempo history.
    pub fn reset(&mut self) {
        self.tap_intervals = [0; 4];
        self.tap_idx = 0;
        self.last_tap_frame = None;
        self.total_frames = 0;
    }

    /// Sets the host BPM.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.current_bpm = bpm.clamp(20.0, 400.0);
    }

    /// Advances frame counter by `num_frames`.
    #[inline]
    pub fn advance_frames(&mut self, num_frames: usize) {
        self.total_frames = self.total_frames.saturating_add(num_frames as u64);
    }

    /// Registers a tap event and updates `current_bpm` from the interval moving average.
    pub fn register_tap(&mut self) -> f32 {
        let now = self.total_frames;
        if let Some(last) = self.last_tap_frame
            && now > last
        {
            let interval = (now - last).min(u32::MAX as u64) as u32;
            let min_interval = self.sample_rate_hz / 7; // Max ~420 BPM
            let max_interval = self.sample_rate_hz.saturating_mul(3); // Min ~20 BPM
            if interval > max_interval {
                // Pause > 3s starts a fresh tap-tempo sequence without stale interval history
                self.tap_intervals = [0; 4];
                self.tap_idx = 0;
            } else if interval >= min_interval {
                self.tap_intervals[self.tap_idx] = interval;
                self.tap_idx = (self.tap_idx + 1) % 4;

                let mut sum = 0u32;
                let mut count = 0u32;
                for &intv in &self.tap_intervals {
                    if intv > 0 {
                        sum += intv;
                        count += 1;
                    }
                }
                if count > 0 {
                    let avg_interval = (sum as f32) / (count as f32);
                    let bpm = (60.0 * (self.sample_rate_hz as f32)) / avg_interval;
                    self.current_bpm = bpm.clamp(20.0, 400.0);
                }
            }
        }
        self.last_tap_frame = Some(now);
        self.current_bpm
    }

    /// Synchronizes all calibrated controls in `header` matching `(bpm, subdivision_override)`.
    pub fn resolve_sync_updates(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> alloc::vec::Vec<(usize, f32)> {
        let mut buf = [(0usize, 0.0f32); 512];
        let n = Self::resolve_sync_updates_into(
            header,
            bpm,
            subdivision_override,
            fold_octave,
            &mut buf,
        );
        buf[..n].to_vec()
    }

    /// Zero-allocation helper that writes `(ctrl_idx, physical_value)` tempo updates into `out`.
    pub fn resolve_sync_updates_into(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
        out: &mut [(usize, f32)],
    ) -> usize {
        let mut clamp_mask = 0u16;
        Self::resolve_sync_updates_with_clamp_mask_into(
            header,
            bpm,
            subdivision_override,
            fold_octave,
            out,
            &mut clamp_mask,
        )
    }

    /// Zero-allocation helper that writes `(ctrl_idx, physical_value)` tempo updates into `out`
    /// and sets bits in `clamp_mask` for any control whose target was clamped to its reachable bound
    /// (`ref[dsp.ctrl.range.f32-rank-bounds]`, `ref[dsp.tempo.physical-inversion]`).
    pub fn resolve_sync_updates_with_clamp_mask_into(
        header: &DnamJsonHeader,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
        out: &mut [(usize, f32)],
        clamp_mask: &mut u16,
    ) -> usize {
        let mut count = 0;
        for (ctrl_idx, ctrl_def) in header.controls_schema.iter().enumerate() {
            let Some(ref cal) = ctrl_def.calibration else {
                continue;
            };
            let Some(res) = cal.resolve_tempo_sync(bpm, subdivision_override, fold_octave) else {
                continue;
            };
            if res.was_clamped && ctrl_idx < 16 {
                *clamp_mask |= 1u16 << ctrl_idx;
            }
            if let Some((sw_id, opt_label)) = res.switch_selection
                && let Some((sw_idx, sw_def)) = header
                    .controls_schema
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.id == sw_id)
                && let Some(ref opts) = sw_def.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == &opt_label)
                && count < out.len()
            {
                out[count] = (sw_idx, opt_idx as f32);
                count += 1;
            }
            if count < out.len() {
                let decl = ctrl_def.declared_range.unwrap_or([0.0, 10.0]);
                let phys = decl[0] + 0.5 * (res.normalized_u + 1.0) * (decl[1] - decl[0]);
                out[count] = (ctrl_idx, phys);
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_sync_tap_and_bpm() {
        let mut tempo = TempoSyncState::default();
        tempo.set_bpm(130.0);
        assert_eq!(tempo.current_bpm, 130.0);
        // BPM clamping [20.0, 400.0]
        tempo.set_bpm(5.0);
        assert_eq!(tempo.current_bpm, 20.0);
        tempo.set_bpm(999.0);
        assert_eq!(tempo.current_bpm, 400.0);

        tempo.reset();
        tempo.sample_rate_hz = 48_000;
        // Tap at 120 BPM: interval = 48,000 * 60 / 120 = 24,000 frames (4 taps wrap circular buffer)
        tempo.register_tap();
        for _ in 0..5 {
            tempo.advance_frames(24_000);
            let bpm = tempo.register_tap();
            assert!((bpm - 120.0).abs() < 1e-2);
        }

        // Out-of-bounds tap interval (e.g. 10 frames, > 420 BPM) is ignored and preserves BPM
        tempo.advance_frames(10);
        let bpm_after_glitch = tempo.register_tap();
        assert!((bpm_after_glitch - 120.0).abs() < 1e-2);

        // Pause > 3s (> 144,000 frames) clears stale tap_intervals so new sequence locks immediately to 90 BPM
        tempo.advance_frames(200_000);
        tempo.register_tap();
        tempo.advance_frames(32_000); // 90 BPM at 48 kHz = 32,000 frames
        let bpm_new_seq = tempo.register_tap();
        assert!((bpm_new_seq - 90.0).abs() < 1e-2);

        tempo.reset();
        assert_eq!(tempo.total_frames, 0);
        assert_eq!(tempo.last_tap_frame, None);
    }

    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::engine::DnamEngine;
    use crate::math::ScalarBackend;
    use crate::provenance::VerificationStatus;
    use crate::schema::{
        BlendMode, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
        ControlUiHints, PresetDef,
    };
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::string::String;

    #[test]
    fn test_dnam_engine_presets_and_reload_errors() {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = alloc::vec![
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
                declared_options: Some(alloc::vec![String::from("Clean"), String::from("Lead")]),
                active_options: Some(alloc::vec![String::from("Clean"), String::from("Lead")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        header.presets = alloc::vec![PresetDef {
            id: String::from("lead_boost"),
            name: String::from("Lead Boost"),
            description: None,
            controls: BTreeMap::from([
                (String::from("gain"), serde_json::json!(8.0)),
                (String::from("channel"), serde_json::json!("Lead")),
            ]),
            user_metadata: BTreeMap::new(),
        }];

        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
        assert_eq!(engine.tier(), Tier::Ultra);
        assert_eq!(engine.sample_rate_hz(), 48_000);
        assert_eq!(engine.zoh().sample_rate_host_hz, 48_000);
        assert!(engine.header().is_some());
        assert_eq!(engine.preamble().num_controls, 2);
        assert_eq!(engine.verification_status(), &VerificationStatus::Unsigned);

        assert!(!engine.live_analog_drift_enabled());
        engine.set_live_analog_drift(true);
        assert!(engine.live_analog_drift_enabled());
        engine.set_live_analog_drift(false);

        engine.apply_preset("lead_boost").unwrap();
        assert!(engine.apply_preset("missing").is_err());

        let mut cap_engine = DnamEngine::<ScalarBackend>::with_max_capacity();
        assert!(cap_engine.apply_preset("lead_boost").is_err());
        cap_engine
            .reload_model_in_place(&container, Tier::Full, 48_000)
            .unwrap();
        assert_eq!(cap_engine.tier(), Tier::Full);

        assert!(
            cap_engine
                .reload_model_in_place(&container[..4096], Tier::Lite, 48_000)
                .is_err()
        );
        let mut bad_crc = container.clone();
        bad_crc[60] ^= 0xFF;
        assert!(
            cap_engine
                .reload_model_in_place(&bad_crc, Tier::Lite, 48_000)
                .is_err()
        );
        let mut enc_flag = container.clone();
        enc_flag[6] |= (crate::container::FLAG_ENCRYPTED_ARENA & 0xFF) as u8;
        enc_flag[7] |= ((crate::container::FLAG_ENCRYPTED_ARENA >> 8) & 0xFF) as u8;
        assert!(
            cap_engine
                .reload_model_in_place(&enc_flag, Tier::Lite, 48_000)
                .is_err()
        );

        let mut owned_eng =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(owned_eng.tier(), Tier::Lite);
        assert!(
            owned_eng
                .reload_model_in_place_owned(&enc_flag, Tier::Lite, 48_000)
                .is_err()
        );

        // Exercise set_max_supported_tier clamping active tier
        engine.set_max_supported_tier(Tier::Lite);
        assert_eq!(engine.tier(), Tier::Lite);

        // Exercise configure_host_mix_and_leveling & sync_tempo on empty engine (header == None)
        let mut empty_eng = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        empty_eng.configure_host_mix_and_leveling(0.5, 0.0, false);
        assert_eq!(empty_eng.sync_tempo(120.0, None, false), 0);
    }

    #[test]
    fn test_dnam_engine_leveling_blend_trails_tempo_and_stereo_mode() {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        header.io_config.capabilities.wet_polarity_inverted = true;
        header.io_config.capabilities.supports_trails = true;
        header.io_config.capabilities.tempo_sync_capable = true;
        header.io_config.leveling.auto_unity_enabled = true;
        header.io_config.leveling.calibrated_trim_db = 6.0206; // ~2.0x linear gain

        let mut cond_anchors = BTreeMap::new();
        cond_anchors.insert(String::from("range=0.01-0.1s"), [10.0, 55.0, 100.0]);
        cond_anchors.insert(String::from("range=0.1-1s"), [100.0, 550.0, 1000.0]);
        cond_anchors.insert(String::from("range=1-2s"), [1000.0, 1500.0, 2000.0]);

        header.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("time"),
                name: String::from("Time"),
                kind: ControlKind::Range,
                unit: Some(String::from("ms")),
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: cond_anchors,
                    tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("range"),
                name: String::from("Range"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("0.1-1s"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(alloc::vec![
                    String::from("0.01-0.1s"),
                    String::from("0.1-1s"),
                    String::from("1-2s"),
                ]),
                active_options: Some(alloc::vec![
                    String::from("0.01-0.1s"),
                    String::from("0.1-1s"),
                    String::from("1-2s"),
                ]),
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
                declared_options: Some(alloc::vec![String::from("Mono"), String::from("Stereo")]),
                active_options: Some(alloc::vec![String::from("Mono"), String::from("Stereo")]),
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
        ];

        let arena = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        engine.configure_host_mix_and_leveling(0.5, 0.0, true);
        assert!((engine.mix_dry_gain - 0.5).abs() < 1e-4);
        assert!((engine.mix_wet_gain - (-1.0)).abs() < 1e-3);
        engine.configure_host_mix_and_leveling(0.5, 0.0, false);

        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);
        engine.set_control_immediate(2, 0.0);
        assert!(engine.stereo_width().abs() < 1e-6);
        // Process block with stereo_width == 0.0 (stereo width folding active)
        let mut l_fold = [0.0_f32; 4];
        let mut r_fold = [0.0_f32; 4];
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_fold,
            &mut r_fold,
        );
        engine.reset_to_dc_equilibrium();
        engine.set_control_immediate(2, 1.0);
        assert!((engine.stereo_width() - 1.0).abs() < 1e-6);

        let updated = engine.sync_tempo(120.0, Some(TempoSubdivision::HalfDotted), false);
        assert_eq!(updated, 2);
        let snap = engine.telemetry_snapshot();
        assert!((snap.clamped_controls[0] - 5.0).abs() < 1e-3);
        assert!((snap.clamped_controls[1] - 2.0).abs() < 1e-3);

        engine.set_bypass(true, false);
        assert_eq!(engine.bypass_state(), (true, false));
        let mut l_out = [0.0_f32; 4];
        let mut r_out = [0.0_f32; 4];
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_out,
            &mut r_out,
        );
        assert_eq!(l_out, [0.25, -0.5, 0.75, -0.25]);
        assert_eq!(r_out, [-0.25, 0.5, -0.75, 0.25]);

        engine.set_bypass(true, true);
        assert_eq!(engine.bypass_state(), (true, true));
        engine.process_block(
            &[0.25, -0.5, 0.75, -0.25],
            &[-0.25, 0.5, -0.75, 0.25],
            &mut l_out,
            &mut r_out,
        );
        assert_eq!(l_out, [0.25, -0.5, 0.75, -0.25]);
    }
}
