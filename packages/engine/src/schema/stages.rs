//! TORA Stage and subsystem configurations (`schema/stages.rs`).
//!
//! Canonical Reference: `docs/dsp/parametric_control_bus.md` and `docs/overview/tora_architecture.md`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

pub const fn default_envelopes() -> u8 {
    4
}
pub const fn default_lfos() -> u8 {
    2
}
pub const fn default_lfo_harmonics() -> u8 {
    4
}
pub const fn default_fm_rank() -> u8 {
    4
}
pub const fn default_modulation_bus_dim() -> u8 {
    16
}
pub const fn default_trust_radius() -> f32 {
    2.5
}

/// Control-Bus configuration (`control_bus`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlBusConfig {
    /// Number of envelope follower bands (`4`: Low, Mid, High, Aux Key).
    #[serde(default = "default_envelopes")]
    pub envelopes: u8,
    /// Number of LFO oscillators (`2`).
    #[serde(default = "default_lfos")]
    pub lfo_oscillators: u8,
    /// Number of Chebyshev harmonics per LFO (`4`).
    #[serde(default = "default_lfo_harmonics")]
    pub lfo_harmonics: u8,
    /// Factorization Machine (FM) interaction rank ($R = 4$).
    #[serde(default = "default_fm_rank")]
    pub fm_interaction_rank: u8,
    /// Dimension of the unified modulation bus ($D = 16$).
    #[serde(default = "default_modulation_bus_dim")]
    pub modulation_bus_dim: u8,
    /// Trust-region soft-clip radius $M_h$ (`CL-15`, default `2.5`).
    #[serde(default = "default_trust_radius")]
    pub head_trust_region_radius: f32,
}

impl Default for ControlBusConfig {
    fn default() -> Self {
        Self {
            envelopes: 4,
            lfo_oscillators: 2,
            lfo_harmonics: 4,
            fm_interaction_rank: 4,
            modulation_bus_dim: 16,
            head_trust_region_radius: 2.5,
        }
    }
}

pub const fn default_router_ports() -> u8 {
    4
}
pub const fn default_sub_crossover_hz() -> f32 {
    200.0
}

/// Stage 0 8-Port Matrix Router configuration (`stage0_router`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterConfig {
    /// Number of matrix input ports (`4`).
    #[serde(default = "default_router_ports")]
    pub input_ports: u8,
    /// Number of matrix output ports (`4`).
    #[serde(default = "default_router_ports")]
    pub output_ports: u8,
    /// Clean sub crossover frequency in Hz (`200.0`).
    #[serde(default = "default_sub_crossover_hz")]
    pub sub_crossover_hz: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            input_ports: 4,
            output_ports: 4,
            sub_crossover_hz: 200.0,
        }
    }
}

pub const fn default_true() -> bool {
    true
}
pub const fn default_gate_threshold_db() -> f32 {
    -60.0
}

/// Stage I Pre-Dynamics and Tone Shaping configuration (`stage1_pre_fx`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Stage1PreFxConfig {
    /// 4-bit topology selector (`0..=3`).
    #[serde(default)]
    pub topology_selector: u8,
    /// Whether noise gate is active (`true`).
    #[serde(default = "default_true")]
    pub noise_gate_enabled: bool,
    /// Noise gate threshold in dBFS (`-60.0`).
    #[serde(default = "default_gate_threshold_db")]
    pub gate_threshold_db: f32,
}

impl Default for Stage1PreFxConfig {
    fn default() -> Self {
        Self {
            topology_selector: 0,
            noise_gate_enabled: true,
            gate_threshold_db: -60.0,
        }
    }
}

pub const fn default_cascade_stages() -> u8 {
    3
}
pub fn default_adaa_mode() -> String {
    "1x_bspline_algebraic".to_string()
}
pub const fn default_ac_pole() -> f32 {
    0.9993
}

/// Stage II Commuted Cascade configuration (`stage2_cascade`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Stage2CascadeConfig {
    /// Number of cascaded non-linear stages (`3`).
    #[serde(default = "default_cascade_stages")]
    pub num_stages: u8,
    /// Anti-aliasing mode (`"1x_bspline_algebraic"`).
    #[serde(default = "default_adaa_mode")]
    pub adaa_mode: String,
    /// $5\text{ Hz}$ AC-coupling pole at $f_{s,\text{train}}$ (`0.9993`).
    #[serde(default = "default_ac_pole")]
    pub ac_coupling_pole: f32,
}

impl Default for Stage2CascadeConfig {
    fn default() -> Self {
        Self {
            num_stages: 3,
            adaa_mode: default_adaa_mode(),
            ac_coupling_pole: 0.9993,
        }
    }
}

pub const fn default_fir_taps() -> u8 {
    16
}
pub const fn default_kautz_modals() -> u8 {
    8
}

/// Stage III Cabinet Early FIR + Orthonormal Kautz Modal Bank configuration (`stage3_cabinet_kautz`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Stage3CabinetKautzConfig {
    /// Number of direct early FIR taps (`16`).
    #[serde(default = "default_fir_taps")]
    pub fir_taps: u8,
    /// Number of Kautz modal sections (`4, 8, 16, 32`).
    #[serde(default = "default_kautz_modals")]
    pub kautz_modals: u8,
}

impl Default for Stage3CabinetKautzConfig {
    fn default() -> Self {
        Self {
            fir_taps: 16,
            kautz_modals: 8,
        }
    }
}

pub fn default_delay_storage() -> String {
    "fp16".to_string()
}
pub fn default_interp() -> String {
    "catmull_rom_cubic".to_string()
}
pub const fn default_tzf_tap() -> f32 {
    64.0
}
pub const fn default_switch_xfade() -> u16 {
    128
}
pub const fn default_allpass_delays() -> [u16; 16] {
    [
        19, 31, 47, 73, 103, 139, 179, 223, 29, 53, 83, 113, 151, 191, 233, 251,
    ]
}

pub fn deserialize_allpass_delays<'de, D>(deserializer: D) -> Result<[u16; 16], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Vec::<u16>::deserialize(deserializer)?;
    let mut out = default_allpass_delays();
    if v.len() == 8 || v.len() == 16 {
        for (i, &val) in v.iter().enumerate() {
            out[i] = val;
        }
        Ok(out)
    } else {
        Err(serde::de::Error::custom(
            "allpass_delays_samples must contain 8 or 16 u16 elements",
        ))
    }
}

pub const fn default_dc_pole() -> f32 {
    0.995
}
pub const fn default_loop_ceiling() -> f32 {
    0.985
}
/// Stage IV Spatial Tank configuration (`stage4_spatial_tank`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Stage4SpatialTankConfig {
    /// Delay buffer storage format (`"fp16"`).
    #[serde(default = "default_delay_storage")]
    pub delay_storage: String,
    /// Fractional delay interpolation kernel (`"catmull_rom_cubic"`).
    #[serde(default = "default_interp")]
    pub interp: String,
    /// Through-Zero Flanging (TZF) dry reference tap in samples (`64.0` at $48\text{ kHz}$).
    #[serde(default = "default_tzf_tap")]
    pub tzf_reference_tap_samples: f32,
    /// Raised-cosine switch crossfade window in samples (`128` at $48\text{ kHz}$).
    #[serde(default = "default_switch_xfade")]
    pub switch_xfade_samples: u16,
    /// Coprime Schroeder all-pass delay lengths in samples (`16` lines across `Lite`/`Full`/`Pro`/`Ultra`).
    #[serde(
        default = "default_allpass_delays",
        deserialize_with = "deserialize_allpass_delays"
    )]
    pub allpass_delays_samples: [u16; 16],
    /// In-loop $5\text{ Hz}$ DC blocker pole (`0.995` at $48\text{ kHz}$).
    #[serde(default = "default_dc_pole")]
    pub dc_blocker_pole: f32,
    /// Global loop ceiling (`0.985`).
    #[serde(default = "default_loop_ceiling")]
    pub loop_ceiling: f32,
}

impl Default for Stage4SpatialTankConfig {
    fn default() -> Self {
        Self {
            delay_storage: default_delay_storage(),
            interp: default_interp(),
            tzf_reference_tap_samples: 64.0,
            switch_xfade_samples: 128,
            allpass_delays_samples: default_allpass_delays(),
            dc_blocker_pole: 0.995,
            loop_ceiling: 0.985,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_configs_serde_and_defaults() {
        let bus = ControlBusConfig::default();
        assert_eq!(bus.envelopes, 4);
        assert_eq!(bus.modulation_bus_dim, 16);

        let router = RouterConfig::default();
        assert_eq!(router.input_ports, 4);
        assert_eq!(router.sub_crossover_hz, 200.0);

        let stage1 = Stage1PreFxConfig::default();
        assert!(stage1.noise_gate_enabled);
        assert_eq!(stage1.gate_threshold_db, -60.0);

        let stage2 = Stage2CascadeConfig::default();
        assert_eq!(stage2.num_stages, 3);
        assert_eq!(stage2.ac_coupling_pole, 0.9993);

        let stage3 = Stage3CabinetKautzConfig::default();
        assert_eq!(stage3.fir_taps, 16);
        assert_eq!(stage3.kautz_modals, 8);

        let stage4 = Stage4SpatialTankConfig::default();
        assert_eq!(stage4.delay_storage, "fp16");
        assert_eq!(stage4.tzf_reference_tap_samples, 64.0);

        // Test allpass delays deserializer
        let json_8 = serde_json::json!([10, 20, 30, 40, 50, 60, 70, 80]);
        let parsed_8: [u16; 16] = deserialize_allpass_delays(json_8).unwrap();
        assert_eq!(parsed_8[0], 10);
        assert_eq!(parsed_8[7], 80);

        let json_invalid = serde_json::json!([1, 2, 3]);
        assert!(deserialize_allpass_delays(json_invalid).is_err());
    }
}
