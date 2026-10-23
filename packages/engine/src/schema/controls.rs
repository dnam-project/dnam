//! Parametric `range` and `switch` control schemas, UI interaction hints (`ControlUiHints`),
//! power-skew gesture mapping, and built-in preset definitions (`PresetDef`).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::error::{cold_ln_f64, cold_pow_f64};

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
