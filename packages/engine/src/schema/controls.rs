//! Parametric `range` and `switch` control schemas, physical calibration (`ControlCalibration`),
//! 12-subdivision tempo sync (`TempoSubdivision`), UI interaction hints (`ControlUiHints`),
//! power-skew gesture mapping, and built-in preset definitions (`PresetDef`).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::error::{cold_ln_f64, cold_pow_f64, cold_sqrt_f64};

/// Canonical tempo subdivisions (`1/2` through `1/16` with straight, dotted `.`, and triplet `T` variants).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TempoSubdivision {
    /// Dotted half note (`"1/2."`, $3.0$ quarter-note beats).
    #[serde(rename = "1/2.")]
    HalfDotted,
    /// Half note (`"1/2"`, $2.0$ quarter-note beats).
    #[serde(rename = "1/2")]
    Half,
    /// Half-note triplet (`"1/2T"`, $4/3 \approx 1.333333$ quarter-note beats).
    #[serde(rename = "1/2T")]
    HalfTriplet,
    /// Dotted quarter note (`"1/4."`, $1.5$ quarter-note beats).
    #[serde(rename = "1/4.")]
    QuarterDotted,
    /// Quarter note (`"1/4"`, $1.0$ quarter-note beat).
    #[serde(rename = "1/4")]
    Quarter,
    /// Quarter-note triplet (`"1/4T"`, $2/3 \approx 0.666667$ quarter-note beats).
    #[serde(rename = "1/4T")]
    QuarterTriplet,
    /// Dotted eighth note (`"1/8."`, $0.75$ quarter-note beats).
    #[serde(rename = "1/8.")]
    EighthDotted,
    /// Eighth note (`"1/8"`, $0.5$ quarter-note beats).
    #[serde(rename = "1/8")]
    Eighth,
    /// Eighth-note triplet (`"1/8T"`, $1/3 \approx 0.333333$ quarter-note beats).
    #[serde(rename = "1/8T")]
    EighthTriplet,
    /// Dotted sixteenth note (`"1/16."`, $0.375$ quarter-note beats).
    #[serde(rename = "1/16.")]
    SixteenthDotted,
    /// Sixteenth note (`"1/16"`, $0.25$ quarter-note beats).
    #[serde(rename = "1/16")]
    Sixteenth,
    /// Sixteenth-note triplet (`"1/16T"`, $1/6 \approx 0.166667$ quarter-note beats).
    #[serde(rename = "1/16T")]
    SixteenthTriplet,
}

impl TempoSubdivision {
    /// All 12 canonical tempo subdivisions ordered from longest (`"1/2."`) to shortest (`"1/16T"`).
    pub const ALL: [Self; 12] = [
        Self::HalfDotted,
        Self::Half,
        Self::HalfTriplet,
        Self::QuarterDotted,
        Self::Quarter,
        Self::QuarterTriplet,
        Self::EighthDotted,
        Self::Eighth,
        Self::EighthTriplet,
        Self::SixteenthDotted,
        Self::Sixteenth,
        Self::SixteenthTriplet,
    ];

    /// Returns the canonical string label (e.g., `"1/8."`, `"1/4T"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HalfDotted => "1/2.",
            Self::Half => "1/2",
            Self::HalfTriplet => "1/2T",
            Self::QuarterDotted => "1/4.",
            Self::Quarter => "1/4",
            Self::QuarterTriplet => "1/4T",
            Self::EighthDotted => "1/8.",
            Self::Eighth => "1/8",
            Self::EighthTriplet => "1/8T",
            Self::SixteenthDotted => "1/16.",
            Self::Sixteenth => "1/16",
            Self::SixteenthTriplet => "1/16T",
        }
    }

    /// Parses a [`TempoSubdivision`] from its string notation (`"1/2."`..`"1/16T"`).
    #[inline]
    pub fn from_str_label(s: &str) -> Option<Self> {
        match s.trim() {
            "1/2." => Some(Self::HalfDotted),
            "1/2" => Some(Self::Half),
            "1/2T" => Some(Self::HalfTriplet),
            "1/4." => Some(Self::QuarterDotted),
            "1/4" => Some(Self::Quarter),
            "1/4T" => Some(Self::QuarterTriplet),
            "1/8." => Some(Self::EighthDotted),
            "1/8" => Some(Self::Eighth),
            "1/8T" => Some(Self::EighthTriplet),
            "1/16." => Some(Self::SixteenthDotted),
            "1/16" => Some(Self::Sixteenth),
            "1/16T" => Some(Self::SixteenthTriplet),
            _ => None,
        }
    }

    /// Quarter-note beat multiplier $\beta_{\text{sub}}$.
    #[inline]
    pub const fn beat_multiplier(self) -> f32 {
        match self {
            Self::HalfDotted => 3.0,
            Self::Half => 2.0,
            Self::HalfTriplet => 4.0 / 3.0,
            Self::QuarterDotted => 1.5,
            Self::Quarter => 1.0,
            Self::QuarterTriplet => 2.0 / 3.0,
            Self::EighthDotted => 0.75,
            Self::Eighth => 0.5,
            Self::EighthTriplet => 1.0 / 3.0,
            Self::SixteenthDotted => 0.375,
            Self::Sixteenth => 0.25,
            Self::SixteenthTriplet => 1.0 / 6.0,
        }
    }

    /// Target delay time in milliseconds at `bpm` ($\tau_{\text{ms}} = \frac{60\,000}{\text{BPM}} \cdot \beta_{\text{sub}}$).
    #[inline]
    pub fn delay_ms_at_bpm(self, bpm: f32) -> f32 {
        let safe_bpm = bpm.clamp(20.0, 400.0);
        (60_000.0 / safe_bpm) * self.beat_multiplier()
    }

    /// Target modulation rate in Hz at `bpm` ($f_{\text{Hz}} = \frac{\text{BPM}}{60 \cdot \beta_{\text{sub}}}$).
    #[inline]
    pub fn lfo_hz_at_bpm(self, bpm: f32) -> f32 {
        let safe_bpm = bpm.clamp(20.0, 400.0);
        safe_bpm / (60.0 * self.beat_multiplier())
    }
}

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
}

/// Result of resolving a physical target or MIDI tempo sync against a [`ControlCalibration`].
#[derive(Debug, Clone, PartialEq)]
pub struct TempoSyncResolution {
    /// Optional `(switch_id, option_label)` when resolved through `conditional_physical_anchors`
    /// (e.g., `Some(("range", "0.1-1s"))` on the Darkglass Anagram Analog Delay).
    pub switch_selection: Option<(String, String)>,
    /// Resolved normalized Lobatto coordinate $u \in [-1.0, +1.0]$.
    pub normalized_u: f32,
    /// `true` if the target value fell outside the calibrated physical anchor range and was clamped.
    pub was_clamped: bool,
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
    /// `DelayTimeMs` or `LfoRateHz`.
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
            ControlCalibrationRole::DelayTimeMs => sub.delay_ms_at_bpm(bpm),
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

/// Intrinsic visual and interaction hints for a parametric control (`ui_hints`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ControlUiHints {
    /// Logical rack/pedal section name (e.g., `"Preamp"`, `"Tape Echo"`).
    #[serde(default)]
    pub section: Option<String>,
    /// Ordering index within `section`.
    #[serde(default)]
    pub order: i32,
    /// Widget archetype (`"knob"`, `"slider"`, `"trim"`, `"toggle"`, `"segmented"`, `"rotary_select"`, `"footswitch"`).
    #[serde(default)]
    pub widget: Option<String>,
    /// Relative visual size (`"sm"`, `"md"`, `"lg"`).
    #[serde(default)]
    pub size: Option<String>,
    /// Visual fill polarity (`"unipolar"` or `"bipolar"`).
    #[serde(default)]
    pub polarity: Option<String>,
    /// Physical value at $50\%$ ($12\text{ o'clock}$, $t = 0.5$) UI gesture position.
    #[serde(default)]
    pub midpoint: Option<f32>,
    /// Physical click/detent snap positions.
    #[serde(default)]
    pub detents: Vec<f32>,
    /// Optional UI step increment.
    #[serde(default)]
    pub step: Option<f32>,
    /// Numeric display precision in decimal places.
    #[serde(default)]
    pub decimals: Option<u8>,
    /// Landmark scale labels (`{"0.0": "Clean", "10.0": "Saturated"}`).
    #[serde(default)]
    pub scale_labels: BTreeMap<String, String>,
    /// Switch option display labels (`{"off": "Bypass", "on": "Vintage +6 dB"}`).
    #[serde(default)]
    pub option_labels: BTreeMap<String, String>,
    /// Optional CSS hex accent color (e.g., `"#F59E0B"`).
    #[serde(default)]
    pub color_hex: Option<String>,
}

impl ControlUiHints {
    /// Computes the closed-form power-skew exponent $\gamma_{\text{skew}}$ for `[x_min, x_max]`
    /// anchored at `midpoint`:
    /// $$\gamma_{\text{skew}} = \frac{\ln(0.5)}{\ln\!\left(\frac{x_{\text{mid}} - x_{\min}}{x_{\max} - x_{\min}}\right)}$$
    #[inline]
    pub fn skew_exponent(&self, x_min: f32, x_max: f32) -> f64 {
        let span = (x_max - x_min) as f64;
        if span <= 1e-9 {
            return 1.0;
        }
        if let Some(mid) = self.midpoint {
            let ratio = ((mid - x_min) as f64) / span;
            if ratio > 1e-4 && ratio < (1.0 - 1e-4) && (ratio - 0.5).abs() > 1e-6 {
                return cold_ln_f64(0.5) / cold_ln_f64(ratio);
            }
        }
        1.0
    }

    /// Maps a normalized host/UI gesture coordinate $t \in [0.0, 1.0]$ to physical parameter value $x \in [x_{\min}, x_{\max}]$:
    /// $$x(t) = x_{\min} + (x_{\max} - x_{\min})\, t^{1 / \gamma_{\text{skew}}}$$
    #[inline]
    pub fn normalized_to_physical(&self, t: f32, x_min: f32, x_max: f32) -> f32 {
        let t_clamped = t.clamp(0.0, 1.0) as f64;
        if t_clamped <= 0.0 {
            return x_min;
        }
        if t_clamped >= 1.0 {
            return x_max;
        }
        let gamma = self.skew_exponent(x_min, x_max);
        let curved = cold_pow_f64(t_clamped, 1.0 / gamma);
        x_min + ((x_max - x_min) as f64 * curved) as f32
    }

    /// Maps a physical parameter value $x \in [x_{\min}, x_{\max}]$ to normalized host/UI gesture coordinate $t \in [0.0, 1.0]$:
    /// $$t(x) = \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right)^{\gamma_{\text{skew}}}$$
    #[inline]
    pub fn physical_to_normalized(&self, x: f32, x_min: f32, x_max: f32) -> f32 {
        let span = (x_max - x_min) as f64;
        if span <= 1e-9 {
            return 0.0;
        }
        let ratio = (((x - x_min) as f64) / span).clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return 0.0;
        }
        if ratio >= 1.0 {
            return 1.0;
        }
        let gamma = self.skew_exponent(x_min, x_max);
        cold_pow_f64(ratio, gamma) as f32
    }
}

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
    }
}
