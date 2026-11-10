//! Canonical 12-subdivision tempo sync and quarter-beat resolution.
//!
//! Canonical Reference: `docs/dsp/parametric_control_bus.md` and `docs/overview/tora_architecture.md`.

use alloc::string::String;
use serde::{Deserialize, Serialize};

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

    /// Target delay time in milliseconds at `bpm` ($\tau_{\text{ms}} = \frac{60\,000}{\text{BPM}} \cdot \beta_{\text{sub}}$, `0 FDIV`).
    #[inline]
    pub fn delay_ms_at_bpm(self, bpm: f32) -> f32 {
        let safe_bpm = bpm.clamp(20.0, 400.0);
        let inv_bpm = crate::math::newton6_reciprocal_f64(safe_bpm as f64);
        ((60_000.0 * inv_bpm) as f32) * self.beat_multiplier()
    }

    /// Target modulation rate in Hz at `bpm` ($f_{\text{Hz}} = \frac{\text{BPM}}{60 \cdot \beta_{\text{sub}}}$, `0 FDIV`).
    #[inline]
    pub fn lfo_hz_at_bpm(self, bpm: f32) -> f32 {
        let safe_bpm = bpm.clamp(20.0, 400.0);
        let denom = 60.0 * (self.beat_multiplier() as f64);
        ((safe_bpm as f64) * crate::math::newton6_reciprocal_f64(denom)) as f32
    }
}

/// Result of resolving a physical target or MIDI tempo sync against a [`super::controls::ControlCalibration`].
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_subdivisions_all_variants_roundtrip() {
        for sub in TempoSubdivision::ALL {
            let label = sub.as_str();
            let parsed = TempoSubdivision::from_str_label(label);
            assert_eq!(parsed, Some(sub));

            let mult = sub.beat_multiplier();
            assert!(mult > 0.0);

            let delay = sub.delay_ms_at_bpm(120.0);
            assert!(delay > 0.0);

            let lfo = sub.lfo_hz_at_bpm(120.0);
            assert!(lfo > 0.0);
        }

        assert_eq!(TempoSubdivision::from_str_label("invalid"), None);
    }
}
