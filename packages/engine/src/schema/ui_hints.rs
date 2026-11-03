//! UI interaction hints and power-skew gesture mapping.
//!
//! Canonical Reference: `docs/dsp/parametric_control_bus.md`.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use super::error::{cold_ln_f64, cold_pow_f64};

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
