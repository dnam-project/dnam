//! Parametric `range` and `switch` control schemas, physical calibration (`ControlCalibration`),
//! 12-subdivision tempo sync (`TempoSubdivision`), UI interaction hints (`ControlUiHints`),
//! power-skew gesture mapping, and built-in preset definitions (`PresetDef`).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::error::cold_sqrt_f64;

pub use super::tempo_subdiv::{TempoSubdivision, TempoSyncResolution};

/// Physical calibration role for a parametric control (`ControlCalibration::role`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlCalibrationRole {
    /// Delay time control calibrated in milliseconds (`lobatto_physical_anchors` / `conditional_physical_anchors`).
    DelayTimeMs,
    /// LFO modulation rate control calibrated in Hz.
    LfoRateHz,
    /// Delay/reverb regeneration feedback control.
    Feedback,
    /// Modeled dry/wet mix control.
    Mix,
    /// Filter or split crossover frequency calibrated in Hz.
    CrossoverHz,
    /// Discrete stereo mode selector (`"Mono"` vs `"Stereo"` on Reverb/Chorus/Delay blocks).
    StereoMode,
    /// Continuous stereo width spread control (`0.0 = Mono` to `1.0 = Full Stereo`).
    StereoWidth,
    /// Spectral freeze latch / sustain hold switch or continuous freeze amount (`0.0 = Normal` to `1.0 = Infinite Freeze`).
    FreezeLatch,
    /// Granular window length control calibrated in milliseconds (`10..500 ms`, supports MIDI tempo sync).
    GrainSizeMs,
    /// Granular / pitch-shifter / shimmer interval control calibrated in semitones (`-24..+24 st`).
    GrainPitchSemitones,
    /// Granular random delay jitter / spray depth control (`0..1` normalized or milliseconds).
    GrainSpray,
}

/// Physical calibration and MIDI tempo-sync metadata for a single parametric control.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlCalibration {
    /// Physical semantic role of the control.
    pub role: ControlCalibrationRole,
    /// 3-point Lobatto physical anchor values $[v_{\min}, v_{\text{mid}}, v_{\max}]$ at $u \in \{-1, 0, +1\}$.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lobatto_physical_anchors: Option<[f32; 3]>,
    /// Switch-conditioned 3-point Lobatto physical anchors keyed by `"switch_id=option_id"`,
    /// e.g., `{"range=0.01-0.1s": [10.0, 55.0, 100.0], "range=0.1-1s": [100.0, 550.0, 1000.0], "range=1-2s": [1000.0, 1500.0, 2000.0]}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub conditional_physical_anchors: BTreeMap<String, [f32; 3]>,
    /// Default musical subdivision when tempo sync is enabled on this control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_sync_default_subdivision: Option<TempoSubdivision>,
}

impl ControlCalibration {
    /// Inverts a 3-point Lobatto quadratic $v(u) = v_{\text{mid}} + c_1 u + c_2 u^2$ ($u \in [-1, 1]$)
    /// anchored at `anchors = [v_min, v_mid, v_max]`, returning `(normalized_u, was_clamped)`.
    ///
    /// When `fold_octave` is `true` and `target_val > 0.0`, out-of-range targets are halved or doubled
    /// by powers of 2 until they fall within `[min(v_min, v_max), max(v_min, v_max)]` (up to 4 octaves).
    pub fn invert_lobatto_anchors(
        anchors: &[f32; 3],
        target_val: f32,
        fold_octave: bool,
    ) -> (f32, bool) {
        let v_min = anchors[0] as f64;
        let v_mid = anchors[1] as f64;
        let v_max = anchors[2] as f64;
        let lo = v_min.min(v_max);
        let hi = v_min.max(v_max);
        if (hi - lo) <= 1e-9 {
            return (0.0, true);
        }

        let mut t = target_val as f64;
        if fold_octave && t > 1e-6 && lo > 1e-6 {
            for _ in 0..4 {
                if t > hi {
                    t *= 0.5;
                } else if t < lo {
                    t *= 2.0;
                } else {
                    break;
                }
            }
        }

        let ascending = v_max >= v_min;
        if t <= lo {
            let clamped = (t - lo).abs() > 1e-6;
            return (if ascending { -1.0 } else { 1.0 }, clamped);
        }
        if t >= hi {
            let clamped = (t - hi).abs() > 1e-6;
            return (if ascending { 1.0 } else { -1.0 }, clamped);
        }

        let c1 = 0.5 * (v_max - v_min);
        let c2 = 0.5 * (v_max + v_min) - v_mid;
        if c2.abs() <= 1e-7 * c1.abs().max(1.0) {
            if c1.abs() <= 1e-12 {
                return (0.0, false);
            }
            let u = ((t - v_mid) / c1).clamp(-1.0, 1.0) as f32;
            return (u, false);
        }

        // Solve c2 * u^2 + c1 * u + (v_mid - t) = 0; select the root with same sign as c1
        let disc = (c1 * c1 - 4.0 * c2 * (v_mid - t)).max(0.0);
        let root_disc = cold_sqrt_f64(disc);
        let r1 = (-c1 + root_disc) / (2.0 * c2);
        let r2 = (-c1 - root_disc) / (2.0 * c2);
        let u = if (-1.0 - 1e-4..=1.0 + 1e-4).contains(&r1)
            && !(-1.0 - 1e-4..=1.0 + 1e-4).contains(&r2)
        {
            r1
        } else if (-1.0 - 1e-4..=1.0 + 1e-4).contains(&r2)
            && !(-1.0 - 1e-4..=1.0 + 1e-4).contains(&r1)
        {
            r2
        } else if c1 >= 0.0 {
            r1
        } else {
            r2
        };
        (u.clamp(-1.0, 1.0) as f32, false)
    }

    /// Evaluates the 3-point Lobatto quadratic $v(u) = v_{\text{mid}} + c_1 u + c_2 u^2$ for $u \in [-1, 1]$
    /// when `lobatto_physical_anchors` is present.
    #[inline]
    pub fn evaluate_lobatto_anchors(&self, u: f32) -> Option<f32> {
        let [v_min, v_mid, v_max] = self.lobatto_physical_anchors?;
        let u_clamped = u.clamp(-1.0, 1.0);
        let c1 = 0.5 * (v_max - v_min);
        let c2 = 0.5 * (v_max + v_min) - v_mid;
        Some(v_mid + c1 * u_clamped + c2 * u_clamped * u_clamped)
    }

    /// Maps `target_val` to normalized $u \in [-1, 1]$ using `lobatto_physical_anchors`.
    pub fn physical_value_to_normalized_u(
        &self,
        target_val: f32,
        fold_octave: bool,
    ) -> Option<(f32, bool)> {
        let anchors = self.lobatto_physical_anchors.as_ref()?;
        Some(Self::invert_lobatto_anchors(
            anchors,
            target_val,
            fold_octave,
        ))
    }

    /// Resolves `target_val` across `conditional_physical_anchors` (if non-empty) or
    /// `lobatto_physical_anchors`, returning the matching switch throw and normalized $u \in [-1, 1]$.
    pub fn resolve_conditional_target(
        &self,
        target_val: f32,
        fold_octave: bool,
    ) -> Option<TempoSyncResolution> {
        if !self.conditional_physical_anchors.is_empty() {
            // Find global [global_lo, global_hi] across all switch throws for optional octave folding
            let mut global_lo = f32::INFINITY;
            let mut global_hi = f32::NEG_INFINITY;
            for anchors in self.conditional_physical_anchors.values() {
                let lo = anchors[0].min(anchors[2]);
                let hi = anchors[0].max(anchors[2]);
                if lo < global_lo {
                    global_lo = lo;
                }
                if hi > global_hi {
                    global_hi = hi;
                }
            }

            let mut eff_target = target_val;
            if fold_octave && eff_target > 1e-6 && global_lo > 1e-6 && global_hi > global_lo {
                for _ in 0..4 {
                    if eff_target > global_hi {
                        eff_target *= 0.5;
                    } else if eff_target < global_lo {
                        eff_target *= 2.0;
                    } else {
                        break;
                    }
                }
            }

            let mut best_key: Option<&str> = None;
            let mut best_anchors: Option<&[f32; 3]> = None;
            let mut best_dist = f32::INFINITY;

            for (key, anchors) in &self.conditional_physical_anchors {
                let lo = anchors[0].min(anchors[2]);
                let hi = anchors[0].max(anchors[2]);
                let dist = if eff_target < lo {
                    lo - eff_target
                } else if eff_target > hi {
                    eff_target - hi
                } else {
                    0.0
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_key = Some(key.as_str());
                    best_anchors = Some(anchors);
                    if dist == 0.0 {
                        break;
                    }
                }
            }

            let key = best_key?;
            let anchors = best_anchors?;
            let (u, was_clamped) = Self::invert_lobatto_anchors(anchors, eff_target, false);
            let switch_sel = key
                .split_once('=')
                .map(|(sw_id, opt_id)| (sw_id.to_string(), opt_id.to_string()));
            return Some(TempoSyncResolution {
                switch_selection: switch_sel,
                normalized_u: u,
                was_clamped,
            });
        }

        let (u, was_clamped) = self.physical_value_to_normalized_u(target_val, fold_octave)?;
        Some(TempoSyncResolution {
            switch_selection: None,
            normalized_u: u,
            was_clamped,
        })
    }

    /// Resolves a MIDI/Host `(bpm, subdivision)` into a [`TempoSyncResolution`] when `role` is
    /// `DelayTimeMs`, `GrainSizeMs`, or `LfoRateHz`.
    pub fn resolve_tempo_sync(
        &self,
        bpm: f32,
        subdivision: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> Option<TempoSyncResolution> {
        let sub = subdivision
            .or(self.tempo_sync_default_subdivision)
            .unwrap_or(TempoSubdivision::Quarter);
        let target_val = match self.role {
            ControlCalibrationRole::DelayTimeMs | ControlCalibrationRole::GrainSizeMs => {
                sub.delay_ms_at_bpm(bpm)
            }
            ControlCalibrationRole::LfoRateHz => sub.lfo_hz_at_bpm(bpm),
            _ => return None,
        };
        self.resolve_conditional_target(target_val, fold_octave)
    }

    /// Maps a `StereoMode` switch option label (`"Mono"` vs `"Stereo"`) to stereo width $\alpha \in [0.0, 1.0]$.
    #[inline]
    pub fn stereo_width_for_option(&self, option_label: &str) -> Option<f32> {
        if self.role != ControlCalibrationRole::StereoMode {
            return None;
        }
        let lower = option_label.trim();
        if lower.eq_ignore_ascii_case("mono") || lower == "0" {
            Some(0.0)
        } else if lower.eq_ignore_ascii_case("stereo") || lower == "1" {
            Some(1.0)
        } else {
            None
        }
    }

    /// Maps a `FreezeLatch` switch option label (`"Off"` vs `"On"` / `"Freeze"` / `"Hold"`) to
    /// freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    #[inline]
    pub fn freeze_amount_for_option(&self, option_label: &str) -> Option<f32> {
        if self.role != ControlCalibrationRole::FreezeLatch {
            return None;
        }
        let lower = option_label.trim();
        if lower.eq_ignore_ascii_case("off")
            || lower.eq_ignore_ascii_case("false")
            || lower.eq_ignore_ascii_case("normal")
            || lower.eq_ignore_ascii_case("bypass")
            || lower == "0"
        {
            Some(0.0)
        } else if lower.eq_ignore_ascii_case("on")
            || lower.eq_ignore_ascii_case("true")
            || lower.eq_ignore_ascii_case("freeze")
            || lower.eq_ignore_ascii_case("hold")
            || lower.eq_ignore_ascii_case("latch")
            || lower == "1"
        {
            Some(1.0)
        } else {
            None
        }
    }
}

/// Parametric control kind (`"range"` vs `"switch"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ControlKind {
    /// Continuous knob / slider mapped via Node-Anchored Lobatto polynomials.
    Range,
    /// Discrete selector / toggle switch.
    Switch,
}

/// Schema definition for a single parametric control (`range` or `switch`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlSchemaDef {
    /// Machine-readable control identifier (e.g., `"gain"`, `"feedback"`, `"boost"`).
    pub id: String,
    /// Human-readable display label (e.g., `"Drive"`, `"Repeats"`).
    pub name: String,
    /// Control kind (`"range"` or `"switch"`).
    #[serde(rename = "type")]
    pub kind: ControlKind,
    /// Optional physical unit string (e.g., `"dB"`, `"%"`, `"Hz"`).
    #[serde(default)]
    pub unit: Option<String>,
    /// `range` only: Node-Anchored Lobatto polynomial degree $K_p \in \{0, 1, 2, 3\}$.
    #[serde(default)]
    pub lobatto_degree: Option<u8>,
    /// `range` only: Normalized Newton/Lobatto anchor nodes $[u_{p,1}, u_{p,2}, u_{p,3}]$ in $[-1, +1]$.
    #[serde(default)]
    pub anchor_nodes_norm: Option<Vec<f32>>,
    /// `range` only: Full physical dial range $[x_{\text{decl\_min}}, x_{\text{decl\_max}}]$.
    #[serde(default)]
    pub declared_range: Option<[f32; 2]>,
    /// `range` only: Phase-1 `[CLAMP-RANGE]` safe observed interval $[x_{\text{safe\_min}}, x_{\text{safe\_max}}]$.
    #[serde(default)]
    pub safe_clamped_range: Option<[f32; 2]>,
    /// Default value (a JSON number for `range` controls or a JSON string for `switch` controls).
    pub default: serde_json::Value,
    /// `range` only: Switch-conditioned minimum bounds (`"switch_id=option_id" -> min_val`).
    #[serde(default)]
    pub conditional_min: BTreeMap<String, f32>,
    /// `range` only: Switch-conditioned maximum bounds (`"switch_id=option_id" -> max_val`).
    #[serde(default)]
    pub conditional_max: BTreeMap<String, f32>,
    /// `switch` only: All declared switch option IDs (`1..=16`).
    #[serde(default)]
    pub declared_options: Option<Vec<String>>,
    /// `switch` only: Observed/active switch option IDs enabled by `[CLAMP-SWITCH]`.
    #[serde(default)]
    pub active_options: Option<Vec<String>>,
    /// Optional human-readable clamp badge warning (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, etc.).
    #[serde(default)]
    pub clamp_warning: Option<String>,
    /// Optional physical calibration and MIDI tempo-sync / stereo-mode metadata (covered by `immutable_header_sha256`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<ControlCalibration>,
    /// Intrinsic visual and interaction hints for plugin/web/hardware UI rendering.
    #[serde(default)]
    pub ui_hints: ControlUiHints,
    /// Open dictionary for control-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
}

pub use super::ui_hints::ControlUiHints;

/// Built-in preset snapshot (`presets[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresetDef {
    /// Machine-readable preset ID (e.g., `"warm_slapback"`).
    pub id: String,
    /// Human-readable preset display name.
    pub name: String,
    /// Optional preset description.
    #[serde(default)]
    pub description: Option<String>,
    /// Control value assignments (`"control_id" -> number | string`).
    pub controls: BTreeMap<String, serde_json::Value>,
    /// Open dictionary for preset-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_ui_hints_skew_and_mapping_edge_cases() {
        let hints = ControlUiHints {
            midpoint: Some(2.5),
            ..Default::default()
        };
        assert_eq!(hints.skew_exponent(5.0, 5.0), 1.0);
        assert_eq!(hints.normalized_to_physical(0.0, 0.0, 10.0), 0.0);
        assert_eq!(hints.normalized_to_physical(1.0, 0.0, 10.0), 10.0);
        let mid_phys = hints.normalized_to_physical(0.5, 0.0, 10.0);
        assert!((mid_phys - 2.5).abs() < 1e-3);
        assert_eq!(hints.physical_to_normalized(0.0, 5.0, 5.0), 0.0);
        assert_eq!(hints.physical_to_normalized(0.0, 0.0, 10.0), 0.0);
        assert_eq!(hints.physical_to_normalized(10.0, 0.0, 10.0), 1.0);
        let mid_norm = hints.physical_to_normalized(2.5, 0.0, 10.0);
        assert!((mid_norm - 0.5).abs() < 1e-3);

        let linear_hints = ControlUiHints {
            midpoint: Some(5.0),
            ..Default::default()
        };
        assert_eq!(linear_hints.skew_exponent(0.0, 10.0), 1.0);
    }

    #[test]
    fn test_tempo_subdivisions_and_anagram_conditional_calibration() {
        assert_eq!(TempoSubdivision::ALL.len(), 12);
        for sub in TempoSubdivision::ALL {
            assert_eq!(TempoSubdivision::from_str_label(sub.as_str()), Some(sub));
        }
        assert!((TempoSubdivision::HalfDotted.delay_ms_at_bpm(120.0) - 1500.0).abs() < 1e-3);
        assert!((TempoSubdivision::Quarter.delay_ms_at_bpm(120.0) - 500.0).abs() < 1e-3);
        assert!((TempoSubdivision::EighthDotted.delay_ms_at_bpm(120.0) - 375.0).abs() < 1e-3);
        assert!((TempoSubdivision::SixteenthTriplet.delay_ms_at_bpm(120.0) - 83.3333).abs() < 1e-2);
        assert!((TempoSubdivision::Quarter.lfo_hz_at_bpm(120.0) - 2.0).abs() < 1e-4);

        let mut cond = BTreeMap::new();
        cond.insert("range=0.01-0.1s".to_string(), [10.0, 55.0, 100.0]);
        cond.insert("range=0.1-1s".to_string(), [100.0, 550.0, 1000.0]);
        cond.insert("range=1-2s".to_string(), [1000.0, 1500.0, 2000.0]);

        let cal = ControlCalibration {
            role: ControlCalibrationRole::DelayTimeMs,
            lobatto_physical_anchors: None,
            conditional_physical_anchors: cond,
            tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
        };

        // 120 BPM + 1/16T = 83.333 ms -> range=0.01-0.1s
        let r_short = cal
            .resolve_tempo_sync(120.0, Some(TempoSubdivision::SixteenthTriplet), false)
            .unwrap();
        assert_eq!(
            r_short.switch_selection,
            Some(("range".to_string(), "0.01-0.1s".to_string()))
        );
        assert!(!r_short.was_clamped);
        assert!((r_short.normalized_u - 0.6296).abs() < 1e-3);

        // 120 BPM + default (1/8.) = 375 ms -> range=0.1-1s
        let r_mid = cal.resolve_tempo_sync(120.0, None, false).unwrap();
        assert_eq!(
            r_mid.switch_selection,
            Some(("range".to_string(), "0.1-1s".to_string()))
        );
        assert!(!r_mid.was_clamped);
        assert!((r_mid.normalized_u - (-0.3889)).abs() < 1e-3);

        // 120 BPM + 1/2. = 1500 ms -> range=1-2s at u=0.0
        let r_long = cal
            .resolve_tempo_sync(120.0, Some(TempoSubdivision::HalfDotted), false)
            .unwrap();
        assert_eq!(
            r_long.switch_selection,
            Some(("range".to_string(), "1-2s".to_string()))
        );
        assert!(!r_long.was_clamped);
        assert!(r_long.normalized_u.abs() < 1e-5);

        // 60 BPM + 1/2. = 3000 ms (> 2000 ms): clamped when fold_octave=false, folded to 1500 ms when fold_octave=true
        let r_clamped = cal
            .resolve_tempo_sync(60.0, Some(TempoSubdivision::HalfDotted), false)
            .unwrap();
        assert!(r_clamped.was_clamped);
        assert!((r_clamped.normalized_u - 1.0).abs() < 1e-5);

        let r_folded = cal
            .resolve_tempo_sync(60.0, Some(TempoSubdivision::HalfDotted), true)
            .unwrap();
        assert!(!r_folded.was_clamped);
        assert_eq!(
            r_folded.switch_selection,
            Some(("range".to_string(), "1-2s".to_string()))
        );
        assert!(r_folded.normalized_u.abs() < 1e-5);

        // StereoMode switch mapping
        let stereo_cal = ControlCalibration {
            role: ControlCalibrationRole::StereoMode,
            lobatto_physical_anchors: None,
            conditional_physical_anchors: BTreeMap::new(),
            tempo_sync_default_subdivision: None,
        };
        assert_eq!(stereo_cal.stereo_width_for_option("Mono"), Some(0.0));
        assert_eq!(stereo_cal.stereo_width_for_option("Stereo"), Some(1.0));
        assert_eq!(stereo_cal.stereo_width_for_option("Invalid"), None);

        // FreezeLatch switch mapping
        let freeze_cal = ControlCalibration {
            role: ControlCalibrationRole::FreezeLatch,
            lobatto_physical_anchors: None,
            conditional_physical_anchors: BTreeMap::new(),
            tempo_sync_default_subdivision: None,
        };
        for opt in ["Off", "false", "NORMAL", "bypass", "0"] {
            assert_eq!(freeze_cal.freeze_amount_for_option(opt), Some(0.0));
        }
        for opt in ["On", "true", "FREEZE", "hold", "latch", "1"] {
            assert_eq!(freeze_cal.freeze_amount_for_option(opt), Some(1.0));
        }
        assert_eq!(freeze_cal.freeze_amount_for_option("unknown"), None);
        assert_eq!(stereo_cal.freeze_amount_for_option("on"), None);
        assert_eq!(freeze_cal.stereo_width_for_option("mono"), None);

        // evaluate_lobatto_anchors & invert
        let anchors = [10.0, 50.0, 100.0];
        let cal_anchors = ControlCalibration {
            role: ControlCalibrationRole::DelayTimeMs,
            lobatto_physical_anchors: Some(anchors),
            conditional_physical_anchors: BTreeMap::new(),
            tempo_sync_default_subdivision: None,
        };
        assert_eq!(cal_anchors.evaluate_lobatto_anchors(0.0), Some(50.0));
        let (u0, _) = cal_anchors
            .physical_value_to_normalized_u(50.0, false)
            .unwrap();
        assert!((u0 - 0.0).abs() < 1e-4);

        // LfoRateHz tempo sync
        let lfo_cal = ControlCalibration {
            role: ControlCalibrationRole::LfoRateHz,
            lobatto_physical_anchors: Some([0.1, 2.0, 10.0]),
            conditional_physical_anchors: BTreeMap::new(),
            tempo_sync_default_subdivision: Some(TempoSubdivision::Quarter),
        };
        let lfo_res = lfo_cal.resolve_tempo_sync(120.0, None, false).unwrap();
        assert!(lfo_res.normalized_u.is_finite());

        // Incompatible role returns None
        assert!(stereo_cal.resolve_tempo_sync(120.0, None, false).is_none());

        // Edge cases for invert_lobatto_anchors and resolve_conditional_target
        assert_eq!(
            ControlCalibration::invert_lobatto_anchors(&[10.0, 10.0, 10.0], 10.0, false),
            (0.0, true)
        );
        // Octave folding upward (target < lo) and downward (target > hi)
        let (u_fold_up, clamped_up) =
            ControlCalibration::invert_lobatto_anchors(&[100.0, 150.0, 200.0], 37.5, true);
        assert!(!clamped_up);
        assert!(u_fold_up.abs() < 1e-4);
        let (u_fold_dn, clamped_dn) =
            ControlCalibration::invert_lobatto_anchors(&[100.0, 150.0, 200.0], 600.0, true);
        assert!(!clamped_dn);
        assert!(u_fold_dn.abs() < 1e-4);

        // Descending anchors (v_max < v_min)
        let desc_anchors = [200.0, 140.0, 100.0];
        assert_eq!(
            ControlCalibration::invert_lobatto_anchors(&desc_anchors, 50.0, false),
            (1.0, true)
        );
        assert_eq!(
            ControlCalibration::invert_lobatto_anchors(&desc_anchors, 250.0, false),
            (-1.0, true)
        );
        let (u_desc_mid, clamped_desc) =
            ControlCalibration::invert_lobatto_anchors(&desc_anchors, 140.0, false);
        assert!(!clamped_desc);
        assert!(u_desc_mid.abs() < 1e-4);

        // Upward octave folding in resolve_conditional_target (eff_target < global_lo)
        let r_fold_up = cal.resolve_conditional_target(5.0, true).unwrap();
        assert!(!r_fold_up.was_clamped);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_ui_skew_roundtrip(
            midpoint in 1.0_f32..9.0_f32,
            t_norm in 0.01_f32..0.99_f32,
        ) {
            let hints = ControlUiHints {
                midpoint: Some(midpoint),
                ..Default::default()
            };
            let phys = hints.normalized_to_physical(t_norm, 0.0, 10.0);
            let t_back = hints.physical_to_normalized(phys, 0.0, 10.0);
            prop_assert!((t_back - t_norm).abs() < 1e-4);
        }

        #[test]
        fn prop_lobatto_anchors_evaluate_invert_roundtrip(
            v_min in 10.0_f32..100.0_f32,
            span in 50.0_f32..500.0_f32,
            alpha_mid in 0.3_f32..0.7_f32,
            u in -0.95_f32..0.95_f32,
        ) {
            let v_max = v_min + span;
            let v_mid = v_min + alpha_mid * span;
            let cal = ControlCalibration {
                role: ControlCalibrationRole::DelayTimeMs,
                lobatto_physical_anchors: Some([v_min, v_mid, v_max]),
                conditional_physical_anchors: BTreeMap::new(),
                tempo_sync_default_subdivision: None,
            };
            let phys = cal.evaluate_lobatto_anchors(u).unwrap();
            let (u_back, was_clamped) = cal.physical_value_to_normalized_u(phys, false).unwrap();
            prop_assert!(!was_clamped);
            prop_assert!((u_back - u).abs() < 1e-3);
        }
    }
}
