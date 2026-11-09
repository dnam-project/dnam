//! Real-time $3\text{ kHz}$ Parametric Control Bus (`ControlBus<B>`).
//!
//! Submodules:
//! - [`tick`]: $3\text{ kHz}$ tick execution, static basis evaluation, tank modulation, and micro-ramping.
//! - [`cp1_and_uint`]: Inline `CP1` contributor decoding and `U_int` ANOVA-FM interaction matrix evaluation.

pub mod cp1_and_uint;
pub mod tick;

pub use cp1_and_uint::*;
pub use tick::*;

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::math::{MathBackend, flush_recursive_state_f32};

use super::descriptor::ControlClampDescriptor;
use super::plane::{MAX_EXACT_CONTRIBUTORS, eval_tier_stage_dispatch};
use super::record::{
    ControlChannelRecord, ControlLatentProjection, Cp1ContributorSlot, LatentAccumulator64B,
};
use super::{ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, TELEMETRY_CONTROL_SLOTS};

/// Real-time $3\text{ kHz}$ Parametric Control Bus state (`#[repr(C, align(64))]`).
///
/// Implements Theorem 3 (`docs/dsp/parametric_control_bus.md` §2.5): supports $P \in [0, 255]$
/// user controls via the unified 128-byte [`ControlChannelRecord`] table, `CP1\0` contributor slots,
/// and the fixed-width exact-integer / 64-byte running sufficient-statistic accumulators ($\mathcal{O}(1)$ per moved knob).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct ControlBus<B: MathBackend> {
    /// O(1) 64-byte running latent sufficient-statistic accumulator $\mathcal{S}_{\text{lat}} = (\mathbf{z}_{\text{main}}, \mathbf{s}_{vx}, \mathbf{s}_{v2x2})$ (`Lite`/`Full`/`Pro`, $D_h=4, R=4`).
    pub latent_acc: LatentAccumulator64B,
    /// Ultra-tier ($D_h=8, R=8$) FM linear sum accumulator for factors `4..8`.
    pub s_vx_ultra: [f32; 4],
    /// Ultra-tier ($D_h=8, R=8$) FM quadratic sum accumulator for factors `4..8`.
    pub s_v2x2_ultra: [f32; 4],
    /// Compiled 64-byte binary clamp descriptors (`CL 0` of `ControlChannelRecord[0..P]`).
    pub descriptors: Vec<ControlClampDescriptor>,
    /// Compiled 64-byte binary latent projections (`CL 1` of `ControlChannelRecord[0..P]`).
    pub projections: Vec<ControlLatentProjection>,
    /// Pre-allocated `CP1` contributor slots (`0..num_contributors`, capacity `MAX_EXACT_CONTRIBUTORS = 255`).
    pub contributors: Vec<Cp1ContributorSlot>,
    /// Active number of `CP1` contributor slots in `contributors`.
    pub num_contributors: usize,
    /// Number of active controls $P \in [0, 255]$.
    pub num_controls: usize,
    /// Raw user-requested physical control values.
    pub raw_values: Vec<f32>,
    /// Clamped physical control values after `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`.
    pub clamped_values: Vec<f32>,
    /// Target normalized coordinates $u_{p,\text{target}} \in [-1, +1]$ (`range`) or option index (`switch`).
    pub target_u_norm: Vec<f32>,
    /// Current anti-zipper smoothed normalized coordinates $u_p[k] \in [-1, +1]$.
    pub current_u_norm: Vec<f32>,
    /// Last evaluated normalized coordinates $u_p^{\text{prev}}$ for incremental $\mathcal{O}(1)$ `LatentAccumulator64B::update_delta`.
    pub prev_eval_u_norm: Vec<f32>,
    /// Last evaluated scalar basis $\psi_p^{\text{prev}}$ for incremental $\mathcal{O}(1)$ `LatentAccumulator64B::update_delta`.
    pub prev_eval_psi: Vec<f32>,
    /// Active integer option index per `switch` control ($0 \dots 15$).
    pub active_switch_options: Vec<u8>,
    /// Bitmask of the leading 16 controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// 1-pole anti-zipper smoothing coefficient $\alpha_{\text{smooth}} \in [0, 0.999]$ at $3\text{ kHz}$.
    pub anti_zipper_pole: f32,
    /// Whether any control is currently slewing or dirty (`false` = static `0 MACs/sample` cache active).
    pub is_dirty: bool,
    /// Current base SVD control coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$ (`Lite`/`Full`/`Pro` modes `1..4`) after trust-region clip (`ref[dsp.ctrl.op-n.trust-clip]`).
    pub z_head: [f32; 4],
    /// Current `Ultra`-tier (`Tier::Ultra`, modes `5..8`, $D_h = 8$) collapsed SVD control coordinates $\mathbf{z}_{\text{head,ultra}} \in \mathbb{R}^4$.
    pub z_head_ultra: [f32; 4],
    /// Current Ultra-tier stage control delta $\Delta\mathbf{z}_{\text{stage,ultra}} \in \mathbb{R}^4$.
    pub z_stage_ultra: [f32; 4],
    /// Gauge-invariant leading 4-D control basis coordinates $\mathbf{x}_{\text{basis},0..4} \in \mathbb{R}^4$ for stage/tank dispatch.
    pub ctrl_basis_4: [f32; 4],
    /// Current 3-band envelope follower states $\mathbf{e}[k] \in \mathbb{R}^3$ (`Low`, `Mid`, `High`) + pad.
    pub env_states: [f32; 4],
    /// Envelope 1-pole smoothing poles $R_{\text{env},b}$ at host control rate.
    pub env_poles: [f32; 4],
    /// LFO 0 and LFO 1 unit-phasor states $(\cos\phi_j, \sin\phi_j)$.
    pub lfo_phasors: [[f32; 2]; 2],
    /// LFO 0 and LFO 1 per-tick rotor increments $(\cos\Delta\phi_j, \sin\Delta\phi_j)$.
    pub lfo_rotors: [[f32; 2]; 2],
    /// LFO 0 and LFO 1 exact `Q0.192` phase accumulators (`3 x u64`, `ref[dsp.op-m.rotor.q0-192-lfo]`).
    pub lfo_phases_q192: [crate::math::Q0_192Phase; 2],
    /// LFO 0 and LFO 1 per-tick angle bit patterns for `Q0.192` phase accumulation (`ref[dsp.op-m.rotor.q0-192-lfo]`).
    pub lfo_angle_bits: [u32; 2],
    /// Exact-integer (`9-word S` + `18-word D`) FM interaction accumulators for all 8 factors (`0..4` base + `4..8` Ultra, `ref[dsp.ctrl.accum.exact-int]`).
    pub exact_fm_acc: [super::plane::ExactIntFactorAccumulator; 8],
    /// Current 4-harmonic Chebyshev LFO waves $\mathbf{l}[k] \in [-1, +1]^8$ ($T_1..T_4$ for LFO 0 and LFO 1).
    pub lfo_harmonics: [f32; 8],
    /// Stage modulation offsets ($\Delta\theta_{\text{stage}} \in \mathbb{R}^{16}$).
    pub stage_targets: [f32; 16],
    /// Spatial Tank per-line delay modulation offsets in samples ($\Delta D_m[k] \in \mathbb{R}^8$).
    pub tank_delay_mod_samples: [f32; 8],
    /// Spatial Tank per-line quiet-state feedback gain modulation offsets ($\Delta g_{\text{lin},m}[k] \in \mathbb{R}^8$).
    pub tank_gain_targets: [f32; 8],
    /// Spatial Tank per-line time-warp / pitch-shift rate modulation offsets ($\Delta\nu_m[k] \in \mathbb{R}^8$).
    pub tank_warp_mod: [f32; 8],
    /// Spatial Tank per-group grain window length scale offsets ($\Delta L_m / L_{0,m} \in \mathbb{R}^4$).
    pub tank_grain_len_scale: [f32; 4],
    /// Spatial Tank global state modulation offsets (`[delta_freeze_alpha, delta_window_mix, delta_feedback_mix, delta_spray_samples]`).
    pub tank_state_mod: [f32; 4],
    /// Whether Spatial Tank modulation outputs were updated on the most recent $3\text{ kHz}$ tick.
    pub tank_mod_dirty: bool,
    /// Trust-region radius $M_h$ (`ref[dsp.ctrl.op-n.trust-clip]`, default `10.0`).
    pub trust_radius_mh: f32,
    /// Pre-computed reciprocal square $M_h^{-2}$ for division-free $3\text{ kHz}$ trust-region evaluation (`0 FDIV`).
    pub inv_trust_radius_mh2: f32,
    /// Live-only Ornstein-Uhlenbeck (OU) analog drift standard deviation $\sigma_{\text{drift}} \in [0.0, 0.05]$ (`Arena 0 [7,496..7,500]`).
    pub sigma_drift: f32,
    /// Training-rate $3\text{ kHz}$ OU analog drift pole $\rho_{\text{drift,train}} \in [0.0, 0.9999]$ (`Arena 0 [7,500..7,504]`).
    pub rho_drift_train: f32,
    /// Host-rate rescaled OU analog drift pole $\rho_{\text{drift,host}} = \rho_{\text{drift,train}}^{\rho_{\text{ctrl}}}$.
    pub rho_drift_host: f32,
    /// Calibration dither SNR scale (`Arena 0 [7,516..7,520]`, clamped to $[0.0, 0.0056234133]$ $\le -45\text{ dB}$).
    pub dither_snr_scale: f32,
    /// Opt-in live-only analog drift switch (`false` by default for deterministic regression & training).
    pub live_analog_drift_enabled: bool,
    /// Whether the host engine is currently rendering an offline bounce (suppresses live-only OU drift for 0-ULP determinism).
    pub offline_bounce: bool,
    /// Branchless `xorshift64*` PRNG state for live-only OU analog drift (`0 FSQRT/FDIV/libm`).
    pub drift_prng_state: u64,
    /// 4-channel OU analog drift state $\boldsymbol{\eta}_{\text{drift}}[k] \in \mathbb{R}^4$.
    pub ou_drift_state: [f32; 4],
    _backend: PhantomData<B>,
}

impl<B: MathBackend> ControlBus<B> {
    pub const DEFAULT_DRIFT_PRNG_SEED: u64 = 0x853c_49e6_748f_ea9b;

    /// Creates a new [`ControlBus`] initialized at the default preset (`z_head = [0.0; 4]`, `is_dirty = false`).
    pub fn new() -> Self {
        Self {
            latent_acc: LatentAccumulator64B::new(),
            s_vx_ultra: [0.0; 4],
            s_v2x2_ultra: [0.0; 4],
            descriptors: alloc::vec![ControlClampDescriptor::empty(); TELEMETRY_CONTROL_SLOTS],
            projections: alloc::vec![ControlLatentProjection::new(); TELEMETRY_CONTROL_SLOTS],
            contributors: alloc::vec![Cp1ContributorSlot::zero(); MAX_EXACT_CONTRIBUTORS],
            num_contributors: 0,
            num_controls: 0,
            raw_values: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            clamped_values: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            target_u_norm: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            current_u_norm: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            prev_eval_u_norm: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            prev_eval_psi: alloc::vec![0.0; TELEMETRY_CONTROL_SLOTS],
            active_switch_options: alloc::vec![0; TELEMETRY_CONTROL_SLOTS],
            active_clamp_mask: 0,
            anti_zipper_pole: DEFAULT_ANTI_ZIPPER_POLE,
            is_dirty: false,
            z_head: [0.0; 4],
            z_head_ultra: [0.0; 4],
            z_stage_ultra: [0.0; 4],
            ctrl_basis_4: [0.0; 4],
            env_states: [0.0; 4],
            env_poles: [0.985, 0.960, 0.910, 0.0],
            lfo_phasors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_rotors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_phases_q192: [crate::math::Q0_192Phase::ZERO; 2],
            lfo_angle_bits: [0; 2],
            exact_fm_acc: core::array::from_fn(|_| {
                super::plane::ExactIntFactorAccumulator::new(MAX_EXACT_CONTRIBUTORS)
            }),
            lfo_harmonics: [0.0; 8],
            stage_targets: [0.0; 16],
            tank_delay_mod_samples: [0.0; 8],
            tank_gain_targets: [0.0; 8],
            tank_warp_mod: [0.0; 8],
            tank_grain_len_scale: [0.0; 4],
            tank_state_mod: [0.0; 4],
            tank_mod_dirty: false,
            trust_radius_mh: 10.0,
            inv_trust_radius_mh2: 1.0 / (10.0 * 10.0),
            sigma_drift: 0.0,
            rho_drift_train: 0.999,
            rho_drift_host: 0.999,
            dither_snr_scale: 0.0,
            live_analog_drift_enabled: false,
            offline_bounce: false,
            drift_prng_state: Self::DEFAULT_DRIFT_PRNG_SEED,
            ou_drift_state: [0.0; 4],
            _backend: PhantomData,
        }
    }

    /// Ensures internal control state vectors have capacity for at least `num_controls` controls ($P \in [0, 255]$).
    pub fn ensure_control_capacity(&mut self, num_controls: usize) {
        let cap = num_controls.max(TELEMETRY_CONTROL_SLOTS);
        if self.descriptors.len() < cap {
            self.descriptors
                .resize(cap, ControlClampDescriptor::empty());
        }
        if self.projections.len() < cap {
            self.projections.resize(cap, ControlLatentProjection::new());
        }
        if self.contributors.len() < MAX_EXACT_CONTRIBUTORS {
            self.contributors
                .resize(MAX_EXACT_CONTRIBUTORS, Cp1ContributorSlot::zero());
        }
        if self.raw_values.len() < cap {
            self.raw_values.resize(cap, 0.0);
        }
        if self.clamped_values.len() < cap {
            self.clamped_values.resize(cap, 0.0);
        }
        if self.target_u_norm.len() < cap {
            self.target_u_norm.resize(cap, 0.0);
        }
        if self.current_u_norm.len() < cap {
            self.current_u_norm.resize(cap, 0.0);
        }
        if self.prev_eval_u_norm.len() < cap {
            self.prev_eval_u_norm.resize(cap, 0.0);
        }
        if self.prev_eval_psi.len() < cap {
            self.prev_eval_psi.resize(cap, 0.0);
        }
        if self.active_switch_options.len() < cap {
            self.active_switch_options.resize(cap, 0);
        }
        if self.exact_fm_acc[0].count < cap {
            self.exact_fm_acc = core::array::from_fn(|_| {
                super::plane::ExactIntFactorAccumulator::new(MAX_EXACT_CONTRIBUTORS)
            });
        }
    }

    /// Loads `CP1\0` contributor slots into the pre-allocated `contributors` buffer without heap allocation.
    pub fn load_cp1_contributors(&mut self, slots: &[Cp1ContributorSlot]) {
        let n = slots.len().min(self.contributors.len());
        self.contributors[..n].copy_from_slice(&slots[..n]);
        self.num_contributors = n;
    }

    /// Loads the unified 128-byte [`ControlChannelRecord`] table (`Region 2B`, $P \in [0, 255]$).
    pub fn load_control_records(&mut self, records: &[ControlChannelRecord]) {
        self.ensure_control_capacity(records.len());
        self.num_controls = records.len();
        for (p, rec) in records.iter().enumerate() {
            let is_clamp_populated = rec.clamp.ctrl_kind <= 1
                && (rec.clamp.safe_max_base > rec.clamp.safe_min_base
                    || rec.clamp.scale_norm != 0.0
                    || rec.clamp.active_options_mask != 0);
            if is_clamp_populated {
                self.descriptors[p] = rec.clamp;
            }
            self.projections[p] = rec.projection;
        }
    }

    /// Loads `num_controls` 128-byte [`ControlChannelRecord`] entries directly from a contiguous
    /// Region 2B byte slice (`128 * num_controls` bytes) without allocating a temporary `Vec`.
    pub fn load_control_records_from_bytes(&mut self, bytes: &[u8], num_controls: usize) {
        self.ensure_control_capacity(num_controls);
        self.num_controls = num_controls;
        for p in 0..num_controls {
            let off = p * 128;
            if off + 128 <= bytes.len() {
                let mut rec_bytes = [0u8; 128];
                rec_bytes.copy_from_slice(&bytes[off..off + 128]);
                let rec = ControlChannelRecord::from_bytes(&rec_bytes);
                let is_clamp_populated = rec.clamp.ctrl_kind <= 1
                    && (rec.clamp.safe_max_base > rec.clamp.safe_min_base
                        || rec.clamp.scale_norm != 0.0
                        || rec.clamp.active_options_mask != 0);
                if is_clamp_populated {
                    self.descriptors[p] = rec.clamp;
                }
                self.projections[p] = rec.projection;
            }
        }
    }

    /// Copies the leading [`TELEMETRY_CONTROL_SLOTS`] (`16`) clamped control values into a fixed C-ABI array.
    #[inline]
    pub fn telemetry_clamped_controls(&self) -> [f32; TELEMETRY_CONTROL_SLOTS] {
        let mut out = [0.0_f32; TELEMETRY_CONTROL_SLOTS];
        let n = self.clamped_values.len().min(TELEMETRY_CONTROL_SLOTS);
        out[..n].copy_from_slice(&self.clamped_values[..n]);
        out
    }

    /// Loads the embedded [`ControlClampDescriptor`] table (`CL 16..23`, `bytes[8512..9024]`),
    /// inline `CP1` contributor slots (`Arena 0 [7,680..8,512]` and `Arena 3 [61,312..63,424]`), and
    /// global LFO/Envelope/Trust-Region anchors from `Arena 0..3` and snaps all controls to their `default_val`.
    pub fn load_from_arena0(&mut self, arena0_bytes: &[u8], num_controls: usize, rho_ctrl: f64) {
        self.ensure_control_capacity(num_controls);
        self.num_controls = num_controls;
        let table_base = 8_512_usize;
        if arena0_bytes.len() >= table_base + 512 {
            for p in 0..8.min(self.num_controls) {
                let off = table_base + p * 64;
                let mut desc_bytes = [0u8; 64];
                desc_bytes.copy_from_slice(&arena0_bytes[off..off + 64]);
                let desc = ControlClampDescriptor::from_bytes(&desc_bytes);
                let is_populated = desc.ctrl_kind <= 1
                    && (desc.safe_max_base > desc.safe_min_base
                        || desc.scale_norm != 0.0
                        || desc.active_options_mask != 0);
                if is_populated {
                    self.descriptors[p] = desc;
                }
            }
        }

        // Decode inline CP1 contributor slots if Region 4 CP1 stream was not already loaded
        if self.num_contributors == 0
            && let Ok(inline_count) =
                decode_cp1_inline_contributors(arena0_bytes, &mut self.contributors)
            && inline_count > 0
        {
            self.num_contributors = inline_count;
        }

        self.update_sample_rate_params(arena0_bytes, rho_ctrl);
        self.reset_to_defaults();
    }

    /// Updates sample-rate-dependent LFO phasor increments and envelope follower poles from `Arena 0`
    /// (`CL 00..03` at `7,488..7,744` and `CL 48` at `10,560..10,624`) without resetting user control values.
    pub fn update_sample_rate_params(&mut self, arena0_bytes: &[u8], rho_ctrl: f64) {
        self.anti_zipper_pole =
            crate::schema::cold_pow_f64(DEFAULT_ANTI_ZIPPER_POLE as f64, rho_ctrl) as f32;
        let global_base = 7_488_usize;
        if arena0_bytes.len() >= global_base + 64 {
            for j in 0..2 {
                let dphi_off = global_base + j * 4;
                let dphi_train = f32::from_le_bytes([
                    arena0_bytes[dphi_off],
                    arena0_bytes[dphi_off + 1],
                    arena0_bytes[dphi_off + 2],
                    arena0_bytes[dphi_off + 3],
                ]);
                let dphi_host = ((dphi_train as f64) * rho_ctrl)
                    .clamp(0.0, core::f64::consts::PI * 0.25)
                    as f32;
                let (c_inc, s_inc) = B::fast_unit_phasor(dphi_host);
                self.lfo_rotors[j] = [c_inc, s_inc];
                self.lfo_angle_bits[j] = dphi_host.to_bits();
            }
            let sigma_raw = f32::from_le_bytes([
                arena0_bytes[global_base + 8],
                arena0_bytes[global_base + 9],
                arena0_bytes[global_base + 10],
                arena0_bytes[global_base + 11],
            ]);
            self.sigma_drift = if sigma_raw.is_finite() && sigma_raw > 0.0 {
                sigma_raw.clamp(0.0, 0.05)
            } else {
                0.0
            };
            let rho_drift_raw = f32::from_le_bytes([
                arena0_bytes[global_base + 12],
                arena0_bytes[global_base + 13],
                arena0_bytes[global_base + 14],
                arena0_bytes[global_base + 15],
            ]);
            self.rho_drift_train = if rho_drift_raw.is_finite() && rho_drift_raw > 0.0 {
                rho_drift_raw.clamp(0.0, 0.9999)
            } else {
                0.999
            };
            self.rho_drift_host =
                crate::schema::cold_pow_f64(self.rho_drift_train as f64, rho_ctrl) as f32;
            for b in 0..3 {
                let r_off = global_base + (4 + b) * 4;
                let r_train = f32::from_le_bytes([
                    arena0_bytes[r_off],
                    arena0_bytes[r_off + 1],
                    arena0_bytes[r_off + 2],
                    arena0_bytes[r_off + 3],
                ]);
                if r_train > 0.0 && r_train < 1.0 {
                    self.env_poles[b] =
                        crate::schema::cold_pow_f64(r_train as f64, rho_ctrl) as f32;
                }
            }
            let dither_raw = f32::from_le_bytes([
                arena0_bytes[global_base + 28],
                arena0_bytes[global_base + 29],
                arena0_bytes[global_base + 30],
                arena0_bytes[global_base + 31],
            ]);
            self.dither_snr_scale = if dither_raw.is_finite() && dither_raw > 0.0 {
                dither_raw.clamp(0.0, 0.005_623_413_3)
            } else {
                0.0
            };
            let mh_off = global_base + 32;
            let mh_from_table = if arena0_bytes.len() >= 10_564 {
                f32::from_le_bytes([
                    arena0_bytes[10_560],
                    arena0_bytes[10_561],
                    arena0_bytes[10_562],
                    arena0_bytes[10_563],
                ])
            } else {
                0.0
            };
            let mh_val = if (0.5..=10.0).contains(&mh_from_table) {
                mh_from_table
            } else {
                f32::from_le_bytes([
                    arena0_bytes[mh_off],
                    arena0_bytes[mh_off + 1],
                    arena0_bytes[mh_off + 2],
                    arena0_bytes[mh_off + 3],
                ])
            };
            if (0.5..=10.0).contains(&mh_val) {
                self.trust_radius_mh = mh_val;
                self.inv_trust_radius_mh2 = 1.0 / (mh_val * mh_val);
            }
        }
    }

    /// Resets dynamic LFO phasor and envelope follower states to zero-input equilibrium
    /// without altering user control values.
    #[inline]
    pub fn reset_dynamic_state(&mut self) {
        self.lfo_phasors = [[1.0, 0.0], [1.0, 0.0]];
        self.lfo_phases_q192 = [crate::math::Q0_192Phase::ZERO; 2];
        self.lfo_harmonics = [0.0; 8];
        self.env_states = [0.0; 4];
        self.ou_drift_state = [0.0; 4];
        self.drift_prng_state = Self::DEFAULT_DRIFT_PRNG_SEED;
    }

    /// Snaps all controls immediately to their default values (`z_head = 0`, `is_dirty = false`).
    pub fn reset_to_defaults(&mut self) {
        self.ensure_control_capacity(self.num_controls);
        self.latent_acc.reset();
        self.s_vx_ultra = [0.0; 4];
        self.s_v2x2_ultra = [0.0; 4];
        for acc in &mut self.exact_fm_acc {
            acc.reset();
        }
        for p in 0..self.num_controls {
            let def_val = self.descriptors[p].default_val;
            self.raw_values[p] = def_val;
            if self.descriptors[p].ctrl_kind == 1 {
                self.active_switch_options[p] = (def_val.max(0.0) + 0.5) as u8;
            }
        }
        self.recompute_targets(true);
        for p in 0..self.num_controls {
            self.prev_eval_u_norm[p] = self.current_u_norm[p];
            let lob = self.descriptors[p].eval_lobatto_basis(self.current_u_norm[p]);
            self.prev_eval_psi[p] = lob[0] + lob[1] + lob[2];
        }
        self.z_head = [0.0; 4];
        self.z_head_ultra = [0.0; 4];
        self.ctrl_basis_4 = [0.0; 4];
        self.stage_targets = [0.0; 16];
        self.tank_delay_mod_samples = [0.0; 8];
        self.tank_gain_targets = [0.0; 8];
        self.tank_warp_mod = [0.0; 8];
        self.tank_grain_len_scale = [0.0; 4];
        self.tank_state_mod = [0.0; 4];
        self.tank_mod_dirty = false;
        self.lfo_phasors = [[1.0, 0.0], [1.0, 0.0]];
        self.lfo_phases_q192 = [crate::math::Q0_192Phase::ZERO; 2];
        self.env_states = [0.0; 4];
        self.ou_drift_state = [0.0; 4];
        self.drift_prng_state = Self::DEFAULT_DRIFT_PRNG_SEED;
        self.is_dirty = false;
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation for continuous
    /// `range` controls.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.ensure_control_capacity(self.num_controls);
        self.raw_values[ctrl_idx] = physical_value;
        self.recompute_targets(false);
    }

    /// Sets a user control value and snaps `current_u_norm` immediately without anti-zipper lag.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.ensure_control_capacity(self.num_controls);
        self.raw_values[ctrl_idx] = physical_value;
        self.recompute_targets(true);
    }

    pub(super) fn recompute_targets(&mut self, snap_immediately: bool) {
        let mut clamp_mask: u16 = 0;
        for p in 0..self.num_controls {
            if self.descriptors[p].ctrl_kind == 1 {
                let (clamped, u_norm, was_clamped) = self.descriptors[p]
                    .clamp_and_normalize(self.raw_values[p], &self.active_switch_options);
                self.clamped_values[p] = clamped;
                self.active_switch_options[p] = (clamped + 0.5) as u8;
                if (self.current_u_norm[p] - u_norm).abs() > ANTI_ZIPPER_SNAP_EPS {
                    self.is_dirty = true;
                }
                self.target_u_norm[p] = u_norm;
                self.current_u_norm[p] = flush_recursive_state_f32(u_norm);
                if was_clamped && p < 16 {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }

        for p in 0..self.num_controls {
            if self.descriptors[p].ctrl_kind == 0 {
                let (clamped, u_norm, was_clamped) = self.descriptors[p]
                    .clamp_and_normalize(self.raw_values[p], &self.active_switch_options);
                self.clamped_values[p] = clamped;
                self.target_u_norm[p] = u_norm;
                if snap_immediately {
                    self.current_u_norm[p] = flush_recursive_state_f32(u_norm);
                    self.is_dirty = true;
                } else if (self.current_u_norm[p] - u_norm).abs() > ANTI_ZIPPER_SNAP_EPS {
                    self.is_dirty = true;
                }
                if was_clamped && p < 16 {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }
        self.active_clamp_mask = clamp_mask;
    }
}

impl<B: MathBackend> Default for ControlBus<B> {
    fn default() -> Self {
        Self::new()
    }
}
