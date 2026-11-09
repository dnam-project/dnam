//! Real-time $3\text{ kHz}$ Parametric Control Bus (`ControlBus<B>`).

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

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
    _backend: PhantomData<B>,
}

impl<B: MathBackend> ControlBus<B> {
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

    fn recompute_targets(&mut self, snap_immediately: bool) {
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

    /// Executes one $3\text{ kHz}$ Control-Bus tick (`0 FDIV`, `0 FSQRT`, `0` heap allocations).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline]
    pub fn tick_3khz(&mut self, audio_abs_energy: f32, arena_bytes: &[u8]) -> bool {
        self.tick_3khz_with_detectors(audio_abs_energy, None, arena_bytes)
    }

    /// Executes one $3\text{ kHz}$ Control-Bus tick with optional 4-band stereo-linked ballistic
    /// detector states `[e_low, e_mid, e_high, e_aux]` from `Stage 0` (`r[dsp.ctrl.modulators.dynamic]`).
    #[inline]
    pub fn tick_3khz_with_detectors(
        &mut self,
        audio_abs_energy: f32,
        stage0_detectors: Option<[f32; 4]>,
        arena_bytes: &[u8],
    ) -> bool {
        if let Some(det) = stage0_detectors {
            for b in 0..4 {
                self.env_states[b] = flush_recursive_state_f32(det[b]);
            }
        } else {
            for b in 0..3 {
                let pole = self.env_poles[b];
                self.env_states[b] = flush_recursive_state_f32(
                    pole * self.env_states[b] + (1.0 - pole) * audio_abs_energy,
                );
            }
            // Causal attack onset detector (fast high-band envelope minus slow low-band envelope)
            self.env_states[3] = (self.env_states[2] - self.env_states[0]).max(0.0);
        }
        for j in 0..2 {
            let angle_bits = self.lfo_angle_bits[j];
            if (angle_bits & 0x7fff_ffff) == 0 {
                let c_hold = self.lfo_phasors[j][0];
                let two_x = 2.0 * c_hold;
                let t2 = two_x * c_hold - 1.0;
                let t3 = two_x * t2 - c_hold;
                let t4 = two_x * t3 - t2;
                let base = j * 4;
                self.lfo_harmonics[base] = c_hold;
                self.lfo_harmonics[base + 1] = t2;
                self.lfo_harmonics[base + 2] = t3;
                self.lfo_harmonics[base + 3] = t4;
                continue;
            }
            let (c_next, s_next) = if self.lfo_phases_q192[j]
                .step_angle_bits(angle_bits)
                .is_some()
            {
                self.lfo_phases_q192[j].to_phasor()
            } else {
                let c_prev = self.lfo_phasors[j][0];
                let s_prev = self.lfo_phasors[j][1];
                let c_inc = self.lfo_rotors[j][0];
                let s_inc = self.lfo_rotors[j][1];

                let c_hat = c_prev * c_inc - s_prev * s_inc;
                let s_hat = s_prev * c_inc + c_prev * s_inc;
                let r2 = c_hat * c_hat + s_hat * s_hat;
                let y1 = 1.5 - 0.5 * r2;
                (c_hat * y1, s_hat * y1)
            };
            self.lfo_phasors[j] = [
                flush_recursive_state_f32(c_next),
                flush_recursive_state_f32(s_next),
            ];

            let t1 = c_next;
            let two_x = 2.0 * t1;
            let t2 = two_x * t1 - 1.0;
            let t3 = two_x * t2 - t1;
            let t4 = two_x * t3 - t2;
            let base = j * 4;
            self.lfo_harmonics[base] = t1;
            self.lfo_harmonics[base + 1] = t2;
            self.lfo_harmonics[base + 2] = t3;
            self.lfo_harmonics[base + 3] = t4;
        }

        if !self.is_dirty {
            self.evaluate_tank_modulation(arena_bytes);
            return false;
        }

        let mut still_slewing = false;
        let pole = self.anti_zipper_pole;
        let one_minus_pole = 1.0 - pole;
        for p in 0..self.num_controls {
            if self.descriptors[p].ctrl_kind == 0 {
                let target = self.target_u_norm[p];
                let curr = self.current_u_norm[p];
                let diff = target - curr;
                if diff.abs() > ANTI_ZIPPER_SNAP_EPS {
                    let next = pole * curr + one_minus_pole * target;
                    if (target - next).abs() <= ANTI_ZIPPER_SNAP_EPS {
                        self.current_u_norm[p] = flush_recursive_state_f32(target);
                    } else {
                        self.current_u_norm[p] = flush_recursive_state_f32(next);
                        still_slewing = true;
                    }
                } else {
                    self.current_u_norm[p] = flush_recursive_state_f32(target);
                }
            }
        }

        self.evaluate_static_control_basis(arena_bytes);

        self.is_dirty = still_slewing;
        true
    }

    /// Snaps `current_u_norm` to `target_u_norm`, evaluates `z_head`, `z_head_ultra`, and
    /// `evaluate_tank_modulation` at the current control positions and LFO/envelope states
    /// **without** advancing LFO phasors or envelope followers, and clears `is_dirty`.
    #[inline]
    pub fn refresh_static_state(&mut self, arena_bytes: &[u8]) {
        self.ensure_control_capacity(self.num_controls);
        assert_eq!(self.current_u_norm.len(), self.target_u_norm.len());
        for (current, &target) in self.current_u_norm.iter_mut().zip(&self.target_u_norm) {
            *current = flush_recursive_state_f32(target);
        }
        self.evaluate_static_control_basis(arena_bytes);
        self.is_dirty = false;
    }

    #[inline]
    fn evaluate_static_control_basis(&mut self, arena_bytes: &[u8]) {
        if self.num_contributors > 0 {
            let mut z_main_base = [0.0_f32; 8];
            let mut z_main_ultra = [0.0_f32; 8];
            let mut s_emb = [0.0_f32; 3];
            for c in 0..self.num_contributors.min(self.contributors.len()) {
                let slot = &self.contributors[c];
                let p = slot.control_idx as usize;
                let effective_kind = if slot.ctrl_kind != 0 {
                    slot.ctrl_kind
                } else if p < self.num_controls {
                    self.descriptors[p].ctrl_kind
                } else {
                    0
                };
                let (lob, scalar) = match effective_kind {
                    0 if p < self.num_controls => {
                        let u = self.current_u_norm[p];
                        (self.descriptors[p].eval_lobatto_basis(u), u)
                    }
                    1 if p < self.num_controls => {
                        let active_opt = self.active_switch_options[p];
                        let def_opt = (self.descriptors[p].default_val.max(0.0) + 0.5) as u8;
                        let s = if slot.option_idx == 0xFF {
                            if active_opt != def_opt { 1.0 } else { 0.0 }
                        } else if active_opt != def_opt && active_opt == slot.option_idx {
                            1.0
                        } else {
                            0.0
                        };
                        ([s, 0.0, 0.0], s)
                    }
                    2 => {
                        let q = slot.partner_control_id as usize;
                        if p < self.num_controls && q < self.num_controls {
                            let u_p = self.current_u_norm[p];
                            let u_q = self.current_u_norm[q];
                            let lob_p = self.descriptors[p].eval_lobatto_basis(u_p);
                            let lob_q = self.descriptors[q].eval_lobatto_basis(u_q);
                            let prod = lob_p[0] * lob_q[0];
                            ([prod, lob_p[1] * lob_q[0], lob_p[0] * lob_q[1]], prod)
                        } else {
                            ([0.0; 3], 0.0)
                        }
                    }
                    _ => ([0.0; 3], 0.0),
                };
                for k in 0..3 {
                    let lk = lob[k];
                    if lk.abs() > 1e-12 {
                        for d in 0..8 {
                            z_main_base[d] += slot.main_base[k][d] * lk;
                            if slot.has_ultra {
                                z_main_ultra[d] += slot.main_ultra[k][d] * lk;
                            }
                        }
                    }
                }
                let has_emb = slot.emb.iter().any(|&w| w.abs() > 1e-12);
                let z_c = if has_emb {
                    slot.emb[0] * lob[0] + slot.emb[1] * lob[1] + slot.emb[2] * lob[2]
                } else {
                    scalar
                };
                for e in 0..3 {
                    s_emb[e] += slot.emb[e] * scalar;
                }
                if c < self.exact_fm_acc[0].count {
                    for f in 0..4 {
                        let _ = self.exact_fm_acc[f]
                            .update_contributor(c, (slot.v_base[f] * z_c).to_bits());
                        let v_u = if slot.has_ultra {
                            slot.v_ultra[f] * z_c
                        } else {
                            0.0
                        };
                        let _ = self.exact_fm_acc[4 + f].update_contributor(c, v_u.to_bits());
                    }
                }
            }
            let mut z_fm_base = [0.0_f32; 4];
            let mut z_fm_ultra = [0.0_f32; 4];
            for f in 0..4 {
                if let Some(bits) = self.exact_fm_acc[f].twice_pairs.to_interaction_f32_bits() {
                    z_fm_base[f] = f32::from_bits(bits);
                }
                if let Some(bits) = self.exact_fm_acc[4 + f]
                    .twice_pairs
                    .to_interaction_f32_bits()
                {
                    z_fm_ultra[f] = f32::from_bits(bits);
                }
            }
            let (uint_base, has_uint_base) = eval_uint_base_4x8(arena_bytes, z_fm_base, s_emb);
            let mut z_base_8 = [0.0_f32; 8];
            for d in 0..8 {
                z_base_8[d] = z_main_base[d] + uint_base[d];
            }
            if !has_uint_base {
                for f in 0..4 {
                    z_base_8[f] += z_fm_base[f];
                    z_base_8[4 + f] += z_fm_base[f];
                }
            }

            if arena_bytes.len() >= 64_000 {
                let (delta_base_8, uint_ultra_8, has_uint_ultra) =
                    eval_uint_ultra_delta_3x4x8(arena_bytes, z_fm_base, z_fm_ultra, s_emb);
                for d in 0..8 {
                    z_base_8[d] += delta_base_8[d];
                }
                let mut z_ultra_8 = [0.0_f32; 8];
                for d in 0..8 {
                    z_ultra_8[d] = z_main_ultra[d] + uint_ultra_8[d];
                }
                if !has_uint_ultra {
                    for f in 0..4 {
                        z_ultra_8[f] += z_fm_ultra[f];
                        z_ultra_8[4 + f] += z_fm_ultra[f];
                    }
                }
                let mut z_head_u4 = [z_ultra_8[0], z_ultra_8[1], z_ultra_8[2], z_ultra_8[3]];
                let mut z_stage_u4 = [z_ultra_8[4], z_ultra_8[5], z_ultra_8[6], z_ultra_8[7]];
                trust_clip_outward_4d(&mut z_head_u4, self.trust_radius_mh);
                trust_clip_outward_4d(&mut z_stage_u4, self.trust_radius_mh);
                self.z_head_ultra = z_head_u4;
                self.z_stage_ultra = z_stage_u4;
            } else {
                self.z_head_ultra = [0.0; 4];
                self.z_stage_ultra = [0.0; 4];
            }
            let mut z_head_4 = [z_base_8[0], z_base_8[1], z_base_8[2], z_base_8[3]];
            let mut z_stage_4 = [z_base_8[4], z_base_8[5], z_base_8[6], z_base_8[7]];
            trust_clip_outward_4d(&mut z_head_4, self.trust_radius_mh);
            trust_clip_outward_4d(&mut z_stage_4, self.trust_radius_mh);
            self.z_head = z_head_4;
            self.ctrl_basis_4 = z_stage_4;
            self.evaluate_tank_modulation(arena_bytes);
            return;
        }

        // Apply Theorem 3 O(1) Incremental Single-Knob Delta Update Law over any moved controls p in 0..P
        for p in 0..self.num_controls {
            let u_new = self.current_u_norm[p];
            let u_old = self.prev_eval_u_norm[p];
            let psi_new = if self.descriptors[p].ctrl_kind == 0 {
                let lob = self.descriptors[p].eval_lobatto_basis(u_new);
                lob[0] + lob[1] + lob[2]
            } else {
                let active_opt = self.active_switch_options[p];
                let def_opt = (self.descriptors[p].default_val.max(0.0) + 0.5) as u8;
                if active_opt != def_opt { 1.0 } else { 0.0 }
            };
            let psi_old = self.prev_eval_psi[p];
            let delta_psi = psi_new - psi_old;
            if (u_new - u_old).abs() > 1e-12 || delta_psi.abs() > 1e-12 {
                let proj = &self.projections[p];
                self.latent_acc.update_delta(proj, u_old, u_new, delta_psi);
                LatentAccumulator64B::update_delta_ultra(
                    proj,
                    u_old,
                    u_new,
                    &mut self.s_vx_ultra,
                    &mut self.s_v2x2_ultra,
                );
                if p < self.exact_fm_acc[0].count {
                    for f in 0..4 {
                        let _ = self.exact_fm_acc[f]
                            .update_contributor(p, (proj.v_factors[f] * u_new).to_bits());
                        let _ = self.exact_fm_acc[4 + f]
                            .update_contributor(p, (proj.v_factors[4 + f] * u_new).to_bits());
                    }
                }
                self.prev_eval_u_norm[p] = u_new;
                self.prev_eval_psi[p] = psi_new;
            }
        }
        let mut z_acc_8 = self.latent_acc.evaluate_z_ctrl();
        for f in 0..4 {
            if let Some(bits) = self.exact_fm_acc[f].twice_pairs.to_interaction_f32_bits() {
                z_acc_8[4 + f] = f32::from_bits(bits);
            }
        }
        let mut z_acc_ultra_4 =
            LatentAccumulator64B::evaluate_z_ultra_fm(&self.s_vx_ultra, &self.s_v2x2_ultra);
        for f in 0..4 {
            if let Some(bits) = self.exact_fm_acc[4 + f]
                .twice_pairs
                .to_interaction_f32_bits()
            {
                z_acc_ultra_4[f] = f32::from_bits(bits);
            }
        }

        let mut z_head_4 = [z_acc_8[0], z_acc_8[1], z_acc_8[2], z_acc_8[3]];
        let mut z_stage_4 = [z_acc_8[4], z_acc_8[5], z_acc_8[6], z_acc_8[7]];
        if z_stage_4.iter().all(|&v| v.abs() <= 1e-12) && self.num_controls > 0 {
            let lob = self.descriptors[0].eval_lobatto_basis(self.current_u_norm[0]);
            z_stage_4 = [lob[0], lob[1], lob[2], 0.0];
        }
        trust_clip_outward_4d(&mut z_head_4, self.trust_radius_mh);
        trust_clip_outward_4d(&mut z_stage_4, self.trust_radius_mh);
        self.z_head = z_head_4;
        self.ctrl_basis_4 = z_stage_4;

        if arena_bytes.len() >= 64_000 {
            trust_clip_outward_4d(&mut z_acc_ultra_4, self.trust_radius_mh);
            self.z_head_ultra = z_acc_ultra_4;
            self.z_stage_ultra = [0.0; 4];
        } else {
            self.z_head_ultra = [0.0; 4];
            self.z_stage_ultra = [0.0; 4];
        }

        self.evaluate_tank_modulation(arena_bytes);
    }

    /// Evaluates the $3\text{ kHz}$ `8 × 32` / `16 × 16` Control/LFO/Envelope $\to$ Stage & Spatial Tank
    /// state-space projection matrix `W_ctrl` (`CL 32..47`, `bytes 9536..10560` in `Arena 0`, plus
    /// `16 × 4` Ultra delta at `bytes 60672..60928` in `Arena 3`, `0 FDIV`, `0 FSQRT`).
    #[inline]
    fn evaluate_tank_modulation(&mut self, arena_bytes: &[u8]) {
        const W_CTRL_TANK_BASE: usize = 9_536;
        const W_CTRL_ULTRA_BASE: usize = 60_672;
        if arena_bytes.len() < W_CTRL_TANK_BASE + 1024 {
            self.tank_mod_dirty = false;
            return;
        }

        let lfo0 =
            self.lfo_harmonics[0] + 0.35 * self.lfo_harmonics[1] + 0.15 * self.lfo_harmonics[2];
        let lfo1 =
            self.lfo_harmonics[4] + 0.35 * self.lfo_harmonics[5] + 0.15 * self.lfo_harmonics[6];
        let env_sustain = 0.5 * (self.env_states[0] + self.env_states[1]);
        let env_onset = self.env_states[3];
        let q8 = [
            self.ctrl_basis_4[0],
            self.ctrl_basis_4[1],
            self.ctrl_basis_4[2],
            self.ctrl_basis_4[3],
            lfo0,
            lfo1,
            env_sustain,
            env_onset,
        ];

        let mut acc = [B::Vec4::zero(); 8];
        for (row, &q_val) in q8.iter().enumerate() {
            if q_val.abs() > 1e-12 {
                let q_vec = B::Vec4::splat(q_val);
                let row_off = W_CTRL_TANK_BASE + row * 128;
                for (c, acc_c) in acc.iter_mut().enumerate() {
                    let w_vec = read_vec4_le::<B>(arena_bytes, row_off + c * 16);
                    *acc_c = acc_c.fma(q_vec, w_vec);
                }
            }
        }

        let d0 = acc[0].to_array();
        let d1 = acc[1].to_array();
        self.tank_delay_mod_samples = [d0[0], d0[1], d0[2], d0[3], d1[0], d1[1], d1[2], d1[3]];

        let g0 = acc[2].to_array();
        let g1 = acc[3].to_array();
        self.tank_gain_targets = [g0[0], g0[1], g0[2], g0[3], g1[0], g1[1], g1[2], g1[3]];

        let w0 = acc[4].to_array();
        let w1 = acc[5].to_array();
        self.tank_warp_mod = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];

        self.tank_grain_len_scale = acc[6].to_array();
        self.tank_state_mod = acc[7].to_array();

        // Also evaluate 16x16 + 16x4 stage dispatch into stage_targets (`r[dsp.ctrl.modulators.dynamic]`)
        let z_ctrl_8 = [
            self.z_head[0],
            self.z_head[1],
            self.z_head[2],
            self.z_head[3],
            self.ctrl_basis_4[0],
            self.ctrl_basis_4[1],
            self.ctrl_basis_4[2],
            self.ctrl_basis_4[3],
        ];
        let m_dyn_8 = [
            self.lfo_phasors[0][0],
            self.lfo_phasors[0][1],
            self.lfo_harmonics[1],
            self.lfo_harmonics[5],
            self.env_states[0],
            self.env_states[1],
            self.env_states[2],
            self.env_states[3],
        ];
        let q16 = Self::assemble_q16(&z_ctrl_8, &m_dyn_8);
        let mut w_base = [[0.0f32; 16]; 16];
        for (r, row_arr) in w_base.iter_mut().enumerate() {
            let row_off = W_CTRL_TANK_BASE + r * 64;
            for c in 0..4 {
                let v = read_vec4_le::<B>(arena_bytes, row_off + c * 16).to_array();
                row_arr[c * 4..c * 4 + 4].copy_from_slice(&v);
            }
        }
        if arena_bytes.len() >= W_CTRL_ULTRA_BASE + 256 {
            let mut w_ultra = [[0.0f32; 4]; 16];
            for (r, row_arr) in w_ultra.iter_mut().enumerate() {
                *row_arr = read_vec4_le::<B>(arena_bytes, W_CTRL_ULTRA_BASE + r * 16).to_array();
            }
            self.stage_targets =
                eval_tier_stage_dispatch(&w_base, Some(&w_ultra), &q16, &q16, &self.z_stage_ultra);
        } else {
            self.stage_targets =
                eval_tier_stage_dispatch(&w_base, None, &q16, &q16, &self.z_stage_ultra);
        }

        self.tank_mod_dirty = true;
    }

    /// Assembles the unified 16-D Modulation Bus q_16[k] = [z_ctrl^(8), m_dyn^(8)]^T.
    #[inline(always)]
    pub fn assemble_q16(z_ctrl: &[f32; 8], m_dyn: &[f32; 8]) -> [f32; 16] {
        let mut q = [0.0; 16];
        q[0..8].copy_from_slice(z_ctrl);
        q[8..16].copy_from_slice(m_dyn);
        q
    }

    /// Evaluates division-free linear micro-ramping deltas with precomputed step alpha_ramp (`ref[dsp.ctrl.dispatch.micro-ramp]`):
    /// delta_theta = (theta_next - theta_curr) * alpha_ramp (0 FDIV).
    #[inline(always)]
    pub fn compute_micro_ramp_deltas(
        theta_curr: &[f32; 16],
        theta_next: &[f32; 16],
        alpha_ramp: f32,
    ) -> [f32; 16] {
        let mut deltas = [0.0; 16];
        for i in 0..16 {
            deltas[i] = (theta_next[i] - theta_curr[i]) * alpha_ramp;
        }
        deltas
    }

    /// Convenience 16-sample micro-ramping delta evaluation (`alpha_ramp = 0.0625f32`).
    #[inline(always)]
    pub fn compute_micro_ramp_deltas_16(
        theta_curr: &[f32; 16],
        theta_next: &[f32; 16],
    ) -> [f32; 16] {
        Self::compute_micro_ramp_deltas(theta_curr, theta_next, 0.0625)
    }

    /// Audio-rate sample-by-sample parameter update via 4-lane SIMD addition.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_micro_ramp(theta_audio: &mut [f32; 16], deltas: &[f32; 16]) {
        let t_chunks: &mut [[f32; 4]; 4] = zerocopy::transmute_mut!(theta_audio);
        let d_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(deltas);
        for c in 0..4 {
            B::Vec4::load(&t_chunks[c])
                .add(B::Vec4::load(&d_chunks[c]))
                .store(&mut t_chunks[c]);
        }
    }
}

/// Applies the outward-checked 4-D trust-region clip (`ref[dsp.ctrl.op-n.trust-clip]`) with
/// fallback to `trust_region_softclip_4d`.
#[inline]
pub fn trust_clip_outward_4d(vec: &mut [f32; 4], radius_mh: f32) {
    let in_bits = [
        vec[0].to_bits(),
        vec[1].to_bits(),
        vec[2].to_bits(),
        vec[3].to_bits(),
    ];
    if let Some((out_bits, _, _)) = crate::math::trust_clip_outward_checked_bits::<4>(
        &in_bits,
        radius_mh.to_bits(),
        (1.0e-12_f32).to_bits(),
    ) {
        *vec = [
            f32::from_bits(out_bits[0]),
            f32::from_bits(out_bits[1]),
            f32::from_bits(out_bits[2]),
            f32::from_bits(out_bits[3]),
        ];
    } else {
        crate::operators::op_n::trust_region_softclip_4d(vec, radius_mh);
    }
}

/// Evaluates `U_int^base` (`4 × 8 = 128 B` at `Arena 0 [7,552..7,680]`) on `z_fm_base` (`r[dsp.ctrl.accum.anova-fm]`).
#[inline]
pub fn eval_uint_base_4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], bool) {
    const UINT_BASE_OFF: usize = 7_552;
    if arena_bytes.len() < UINT_BASE_OFF + 128 {
        return ([0.0; 8], false);
    }
    let has_nonzero_uint = arena_bytes[UINT_BASE_OFF..UINT_BASE_OFF + 128]
        .iter()
        .any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], false);
    }
    let inputs = if z_fm_base.iter().any(|&v| v.abs() > 1e-12) {
        z_fm_base
    } else {
        [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            s_emb[0] * s_emb[0],
        ]
    };
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in inputs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            let row_off = UINT_BASE_OFF + r * 32;
            for (d, out_d) in out.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += p_val * w;
            }
        }
    }
    (out, true)
}

/// Evaluates `U_int^base` (`3 × 8` at `Arena 0 [7,552..7,648]`) on the 3 pairwise products of `s_emb`.
#[inline]
pub fn eval_uint_base_3x8(arena_bytes: &[u8], s_emb: [f32; 3]) -> [f32; 8] {
    eval_uint_base_4x8(arena_bytes, [0.0; 4], s_emb).0
}

/// Evaluates `3 × U_int,Delta` (`3 × 4 × 8 = 384 B` at `Arena 3 [60,928..61,312]`, `r[dsp.ctrl.accum.anova-fm]`):
/// - Block 0 (`60,928..61,056`): maps `z_fm_ultra` into `\Delta z_raw[0..8]`
/// - Block 1 (`61,056..61,184`): maps `z_fm_base` into `z_raw[8..16]`
/// - Block 2 (`61,184..61,312`): maps `z_fm_ultra` into `z_raw[8..16]`
#[inline]
pub fn eval_uint_ultra_delta_3x4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    z_fm_ultra: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], [f32; 8], bool) {
    const UINT_ULTRA_OFF: usize = 60_928;
    if arena_bytes.len() < UINT_ULTRA_OFF + 384 {
        return ([0.0; 8], [0.0; 8], false);
    }
    let has_nonzero_uint = arena_bytes[UINT_ULTRA_OFF..UINT_ULTRA_OFF + 384]
        .iter()
        .any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], [0.0; 8], false);
    }
    let has_fm =
        z_fm_base.iter().any(|&v| v.abs() > 1e-12) || z_fm_ultra.iter().any(|&v| v.abs() > 1e-12);
    let (in_b0, in_b1, in_b2) = if has_fm {
        (z_fm_ultra, z_fm_base, z_fm_ultra)
    } else {
        let p = [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            0.0,
        ];
        ([0.0; 4], p, [0.0; 4])
    };
    let mut delta_base_8 = [0.0_f32; 8];
    let mut ultra_new_8 = [0.0_f32; 8];
    for (r, &val) in in_b0.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + r * 32;
            for (d, out_d) in delta_base_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    for (r, &val) in in_b1.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + 128 + r * 32;
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    for (r, &val) in in_b2.iter().enumerate() {
        if val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + 256 + r * 32;
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += val * w;
            }
        }
    }
    (delta_base_8, ultra_new_8, true)
}

/// Evaluates `U_int,Delta` (`6 × 8` at `Arena 3 [60,928..61,120]`) on 6 pairwise products.
#[inline]
pub fn eval_uint_ultra_delta_6x8(
    arena_bytes: &[u8],
    s_emb: [f32; 3],
    s_emb_ultra: [f32; 3],
) -> [f32; 8] {
    const UINT_ULTRA_OFF: usize = 60_928;
    if arena_bytes.len() < UINT_ULTRA_OFF + 192 {
        return [0.0; 8];
    }
    let pairs = [
        s_emb[0] * s_emb[1],
        s_emb[0] * s_emb[2],
        s_emb[1] * s_emb[2],
        s_emb_ultra[0] * s_emb[0],
        s_emb_ultra[1] * s_emb[1],
        s_emb_ultra[2] * s_emb[2],
    ];
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in pairs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            let row_off = UINT_ULTRA_OFF + r * 32;
            for (d, out_d) in out.iter_mut().enumerate() {
                let off = row_off + d * 4;
                let w = f32::from_le_bytes([
                    arena_bytes[off],
                    arena_bytes[off + 1],
                    arena_bytes[off + 2],
                    arena_bytes[off + 3],
                ]);
                *out_d += p_val * w;
            }
        }
    }
    out
}

/// Decodes inline `CP1` contributors from `Arena 0 [7,680..8,512]` (`832 B` base) and
/// `Arena 3 [61,312..63,424]` (`2,112 B` Ultra) using `cp1_lanes_dir` at `Arena 0 [7,524..7,552]`.
/// Returns `Ok(0)` when `num_inline_contributors == 0`.
pub fn decode_cp1_inline_contributors(
    arena_bytes: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, crate::schema::DnamError> {
    const DIR_OFF: usize = 7_524;
    const BASE_OFF: usize = 7_680;
    const BASE_END: usize = 8_512;
    const ULTRA_OFF: usize = 61_312;
    const ULTRA_END: usize = 63_424;

    if arena_bytes.len() < BASE_END {
        return Ok(0);
    }
    let count = u16::from_le_bytes([arena_bytes[DIR_OFF], arena_bytes[DIR_OFF + 1]]) as usize;
    if count == 0 {
        return Ok(0);
    }
    if count > 26 || count > out_slots.len() {
        return Err(crate::schema::DnamError::SchemaValidationError(
            alloc::string::String::from("inline CP1 contributor count exceeds 26"),
        ));
    }
    let use_3b_dir = count <= 8
        && arena_bytes[DIR_OFF + 2] == 0
        && arena_bytes[DIR_OFF + 3] == 0
        && ((1..=3).contains(&(arena_bytes[DIR_OFF + 5] & 0x0F))
            || (1..=3).contains(&(arena_bytes[DIR_OFF + 6] & 0x0F)));
    let has_ultra = arena_bytes.len() >= ULTRA_END;
    let mut b_cursor = BASE_OFF;
    let mut u_cursor = ULTRA_OFF;
    for (c, slot) in out_slots.iter_mut().take(count).enumerate() {
        let (control_idx, ctrl_kind, option_idx, partner_control_id, lanes) = if use_3b_dir {
            let entry_off = DIR_OFF + 4 + c * 3;
            let c_idx = arena_bytes[entry_off] as u16;
            let b1 = arena_bytes[entry_off + 1];
            let b2 = arena_bytes[entry_off + 2];
            let (packed, aux) = if (1..=3).contains(&(b1 & 0x0F)) {
                (b1, b2)
            } else {
                (b2, b1)
            };
            let kind = (packed >> 4) & 0x0F;
            let l = packed & 0x0F;
            let opt = if kind == 1 { aux } else { 0xFF };
            let partner = if kind == 2 { aux as u16 } else { 0 };
            (c_idx, kind, opt, partner, l)
        } else {
            let packed = arena_bytes[DIR_OFF + 2 + c];
            let kind = (packed >> 4) & 0x0F;
            let l = packed & 0x0F;
            let opt = if kind == 1 { 1 } else { 0xFF };
            let partner = if kind == 2 { (c + 1) as u16 } else { 0 };
            (c as u16, kind, opt, partner, l)
        };
        if !(1..=3).contains(&lanes) {
            return Err(crate::schema::DnamError::SchemaValidationError(
                alloc::string::String::from("invalid inline CP1 lane count"),
            ));
        }
        let l_cnt = lanes as usize;
        let b_bytes = 4 * (9 * l_cnt + 4);
        if b_cursor + b_bytes > BASE_END {
            return Err(crate::schema::DnamError::SchemaValidationError(
                alloc::string::String::from("inline CP1 base payload exceeds 832 B"),
            ));
        }
        let mut main_base = [[0.0f32; 8]; 3];
        for row in main_base.iter_mut().take(l_cnt) {
            for val in row.iter_mut() {
                *val = f32::from_le_bytes([
                    arena_bytes[b_cursor],
                    arena_bytes[b_cursor + 1],
                    arena_bytes[b_cursor + 2],
                    arena_bytes[b_cursor + 3],
                ]);
                b_cursor += 4;
            }
        }
        let mut emb = [0.0f32; 3];
        for val in emb.iter_mut().take(l_cnt) {
            *val = f32::from_le_bytes([
                arena_bytes[b_cursor],
                arena_bytes[b_cursor + 1],
                arena_bytes[b_cursor + 2],
                arena_bytes[b_cursor + 3],
            ]);
            b_cursor += 4;
        }
        let mut v_base = [0.0f32; 4];
        for val in &mut v_base {
            *val = f32::from_le_bytes([
                arena_bytes[b_cursor],
                arena_bytes[b_cursor + 1],
                arena_bytes[b_cursor + 2],
                arena_bytes[b_cursor + 3],
            ]);
            b_cursor += 4;
        }
        let mut main_ultra = [[0.0f32; 8]; 3];
        let mut v_ultra = [0.0f32; 4];
        if has_ultra {
            let u_bytes = (32 * l_cnt) + 16;
            if u_cursor + u_bytes <= ULTRA_END {
                for row in main_ultra.iter_mut().take(l_cnt) {
                    for val in row.iter_mut() {
                        *val = f32::from_le_bytes([
                            arena_bytes[u_cursor],
                            arena_bytes[u_cursor + 1],
                            arena_bytes[u_cursor + 2],
                            arena_bytes[u_cursor + 3],
                        ]);
                        u_cursor += 4;
                    }
                }
                for val in &mut v_ultra {
                    *val = f32::from_le_bytes([
                        arena_bytes[u_cursor],
                        arena_bytes[u_cursor + 1],
                        arena_bytes[u_cursor + 2],
                        arena_bytes[u_cursor + 3],
                    ]);
                    u_cursor += 4;
                }
            }
        }
        *slot = Cp1ContributorSlot {
            lanes,
            ctrl_kind,
            control_idx,
            option_idx,
            partner_control_id,
            main_base,
            emb,
            v_base,
            main_ultra,
            v_ultra,
            has_ultra,
        };
    }
    Ok(count)
}

impl<B: MathBackend> Default for ControlBus<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn read_vec4_le<B: MathBackend>(bytes: &[u8], off: usize) -> B::Vec4 {
    #[cfg(target_endian = "little")]
    {
        if let Ok(chunk) = <&[u8; 16]>::try_from(&bytes[off..off + 16]) {
            let arr: [f32; 4] = zerocopy::transmute!(*chunk);
            return B::Vec4::load(&arr);
        }
    }
    let f0 = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    let f1 = f32::from_le_bytes([
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ]);
    let f2 = f32::from_le_bytes([
        bytes[off + 8],
        bytes[off + 9],
        bytes[off + 10],
        bytes[off + 11],
    ]);
    let f3 = f32::from_le_bytes([
        bytes[off + 12],
        bytes[off + 13],
        bytes[off + 14],
        bytes[off + 15],
    ]);
    B::Vec4::from_array([f0, f1, f2, f3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ARENA_OFFSET_LITE_END;
    use crate::control::ConditionalClampRule;
    use crate::math::ScalarBackend;
    use alloc::boxed::Box;
    use alloc::vec;
    use proptest::prelude::*;

    #[test]
    fn test_control_bus_slewing_and_trust_region_softclip() {
        let _audio = crate::FtzDazGuard::new();
        let mut arena0 = vec![0u8; 64_000];
        arena0[7488 + 16..7488 + 20].copy_from_slice(&0.95_f32.to_le_bytes());
        arena0[10560..10564].copy_from_slice(&0.5_f32.to_le_bytes());
        // Populate cp1_lanes_dir at 7,524 with 2 inline CP1 contributors (ctrl 0: 3-lane Range, ctrl 1: 1-lane Switch)
        arena0[7_524..7_526].copy_from_slice(&2_u16.to_le_bytes());
        arena0[7_526] = 3;
        arena0[7_527] = (1 << 4) | 1;
        for off in (7_680..8_512).step_by(4) {
            arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        }
        // Populate inline Ultra CP1 payload (61,312..63,424) with large weights to exercise outward_checked_trust_clip
        for off in (61_312..63_424).step_by(4) {
            arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        }
        let mut cb = Box::new(ControlBus::<ScalarBackend>::default());
        cb.load_from_arena0(&arena0, 2, 1.0);
        cb.descriptors[0] = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 2,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0],
            _reserved: 0,
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.descriptors[1] = ControlClampDescriptor {
            ctrl_kind: 1,
            lobatto_degree: 1,
            active_options_mask: 0b0010,
            default_val: 0.0,
            safe_min_base: 0.0,
            safe_max_base: 1.0,
            scale_norm: 1.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 3],
            _reserved: 0,
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.num_controls = 2;
        cb.set_control_value(0, 8.0);
        cb.set_control_value(99, 10.0);
        cb.set_control_immediate(99, 10.0);
        assert!(cb.tick_3khz(0.5, &arena0));

        cb.anti_zipper_pole = 0.0;
        cb.set_control_value(0, 10.0);
        cb.set_control_immediate(1, 0.0);
        assert_eq!(cb.clamped_values[1], 1.0);
        assert!(cb.tick_3khz(0.5, &arena0));
        assert!(!cb.tick_3khz(0.5, &arena0));

        // Also test canonical R17-E2 3-byte cp1_lanes_dir encoding: (control_id, packed_kind_and_lanes, option_or_partner)
        let mut arena_3b = vec![0u8; 64_000];
        arena_3b[7_524] = 3;
        // Slot 0: control 0, Range (kind 0), 2 lanes
        arena_3b[7_528] = 0;
        arena_3b[7_529] = 2;
        arena_3b[7_530] = 0;
        // Slot 1: control 1, Switch (kind 1), 1 lane, option 1
        arena_3b[7_531] = 1;
        arena_3b[7_532] = (1 << 4) | 1;
        arena_3b[7_533] = 1;
        // Slot 2: control 0 x control 2 Cross-term (kind 2), 1 lane, partner 2
        arena_3b[7_534] = 0;
        arena_3b[7_535] = (2 << 4) | 1;
        arena_3b[7_536] = 2;
        for off in (7_552..7_680).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.15_f32.to_le_bytes());
        }
        for off in (7_680..8_100).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.25_f32.to_le_bytes());
        }
        for off in (60_928..61_600).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.10_f32.to_le_bytes());
        }
        cb.ensure_control_capacity(3);
        cb.descriptors[2] = cb.descriptors[0];
        cb.num_contributors = 0;
        cb.load_from_arena0(&arena_3b, 3, 1.0);
        assert_eq!(cb.num_contributors, 3);
        cb.set_control_immediate(0, 8.0);
        cb.set_control_immediate(1, 1.0);
        cb.set_control_immediate(2, 8.0);
        cb.refresh_static_state(&arena_3b);
        assert!(cb.z_head[0].abs() > 0.0);
        assert!(cb.z_head_ultra[0].abs() > 0.0);

        // Short arena slice exercises early return in evaluate_tank_modulation
        let short_arena = [0u8; 1024];
        cb.refresh_static_state(&short_arena);
        assert!(!cb.tank_mod_dirty);

        let q16 = ControlBus::<ScalarBackend>::assemble_q16(&[1.0; 8], &[2.0; 8]);
        assert_eq!(&q16[0..8], &[1.0; 8]);
        assert_eq!(&q16[8..16], &[2.0; 8]);
    }

    #[test]
    fn test_micro_ramping_deltas_and_stepping() {
        let _audio = crate::FtzDazGuard::new();
        let curr = [1.0_f32; 16];
        let next = [2.0_f32; 16];
        let deltas_16 = ControlBus::<ScalarBackend>::compute_micro_ramp_deltas_16(&curr, &next);
        assert!((deltas_16[0] - 0.0625).abs() < 1e-6);

        let mut audio = curr;
        for _ in 0..16 {
            ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio, &deltas_16);
        }
        for i in 0..16 {
            assert!((audio[i] - 2.0).abs() < 1e-5);
        }

        // Test with custom alpha_ramp for non-native sample rates (e.g. 96 kHz, K_ctrl = 32)
        let alpha_32 = 1.0 / 32.0;
        let deltas_32 =
            ControlBus::<ScalarBackend>::compute_micro_ramp_deltas(&curr, &next, alpha_32);
        assert!((deltas_32[0] - 0.03125).abs() < 1e-6);
        let mut audio32 = curr;
        for _ in 0..32 {
            ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio32, &deltas_32);
        }
        for i in 0..16 {
            assert!((audio32[i] - 2.0).abs() < 1e-5);
        }
    }

    proptest! {
        #[test]
        fn prop_control_bus_lfo_unit_circle_and_micro_ramping(
            c_step in -0.99_f32..0.99_f32,
            t_curr in -10.0_f32..10.0_f32,
            t_next in -10.0_f32..10.0_f32,
            steps in 1_usize..64_usize,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let s_step = (1.0 - c_step * c_step).max(0.0).sqrt();
            let mut arena0 = vec![0u8; ARENA_OFFSET_LITE_END as usize];
            arena0[7488..7492].copy_from_slice(&c_step.to_le_bytes());
            arena0[7492..7496].copy_from_slice(&s_step.to_le_bytes());
            arena0[7496..7500].copy_from_slice(&c_step.to_le_bytes());
            arena0[7500..7504].copy_from_slice(&s_step.to_le_bytes());

            let mut cb = Box::new(ControlBus::<ScalarBackend>::default());
            cb.load_from_arena0(&arena0, 0, 1.0);
            for _ in 0..32 {
                cb.tick_3khz(0.25, &arena0);
                let r0_sq = cb.lfo_phasors[0][0] * cb.lfo_phasors[0][0]
                    + cb.lfo_phasors[0][1] * cb.lfo_phasors[0][1];
                let r1_sq = cb.lfo_phasors[1][0] * cb.lfo_phasors[1][0]
                    + cb.lfo_phasors[1][1] * cb.lfo_phasors[1][1];
                prop_assert!((r0_sq - 1.0).abs() < 1e-4, "LFO 0 drifted off unit circle: {}", r0_sq);
                prop_assert!((r1_sq - 1.0).abs() < 1e-4, "LFO 1 drifted off unit circle: {}", r1_sq);
            }

            // Micro-ramping exact reconstruction over K steps
            let curr = [t_curr; 16];
            let next = [t_next; 16];
            let alpha = 1.0 / (steps as f32);
            let deltas = ControlBus::<ScalarBackend>::compute_micro_ramp_deltas(&curr, &next, alpha);
            let mut audio = curr;
            for _ in 0..steps {
                ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio, &deltas);
            }
            prop_assert!((audio[0] - t_next).abs() < 1e-3);
        }
    }
}

#[cfg(test)]
mod ckpt_and_control_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical TORA Control Bus, 3-Phase Clamping, and `.dnam-ckpt` Verification (`Tests 8A, 8B, 8D, 8E, 8G`).
    //
    // Canonical Reference: `docs/specs/dsp/parametric_control_bus.md`, `docs/specs/containers/dnam_binary_format.md`,
    // and `docs/specs/training/control_space_audit.md`.
    //
    // Verifies:
    // - `Test 8A & 8B` (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): 3-node Lobatto physical delay interpolation accuracy and
    //   128-sample Hann-windowed dual-pointer switch crossfade vs. single-pointer phase jump.
    // - `Test 8D & 8E` (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): 128-byte `ControlChannelRecord` table at
    //   `tensor_arena_offset - 128 * P`, `update_header_in_place` preservation of `ControlChannelRecord`
    //   bytes, 64-byte $\mathcal{O}(1)$ `LatentAccumulator64B` delta updates across $P = 32$ controls,
    //   and non-stop Phase-1/2/3 clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`).
    // - `Test 8G`: `LAPACK uplo='U'` Nested Principal Submatrix Prefix Theorem across all 4 tiers
    //   (`Lite = 64`, `Full = 128`, `Pro = 192`, `Ultra = 256`) and `.dnam-ckpt` `Arena 4` / `Arena 5`
    //   (`DNCA` / `INPC` / `STMC`) container verification (`verify[container.ckpt.uplo-u]`).

    use std::collections::BTreeMap;
    use std::f64::consts::TAU;

    use crate::FtzDazGuard;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, ARENA4_STEM_BYTES_ULTRA, DNCA_MAGIC, DnamContainerView,
        build_dnam_container, crc32c_compute, pack_uplo_u_f64, unpack_uplo_u_f64,
        update_header_in_place, uplo_u_packed_len, verify_ckpt_arena5_calib_stats,
        verify_ckpt_stem_arena4_crc32c,
    };
    use crate::control::{
        ControlBus, ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection,
        LatentAccumulator64B,
    };
    use crate::math::ScalarBackend;
    use crate::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader, StemProvenance,
        StemRegistryEntry, TensorSliceEntry,
    };
    use proptest::prelude::*;

    /// Test 8A & 8B (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): Verifies 3-node Lobatto physical delay interpolation accuracy
    /// at intermediate knob positions (`u = 0.5`) and verifies that `OpTDelayLine`'s 128-sample
    /// Hann-windowed dual-pointer switch crossfade eliminates single-pointer discontinuity jumps.
    #[test]
    fn test_8a_8b_lobatto_physical_delay_interpolation_and_dual_pointer_switch_crossfade() {
        let tau_at = |u: f64| -> f64 {
            let psi1 = u;
            let psi2 = u * (u + 1.0);
            200.0 + 100.0 * psi1 + 50.0 * psi2
        };
        let exact_mid = 200.0 + 150.0 * 0.5 + 50.0 * 0.25;
        let interp_mid = tau_at(0.5);
        assert!((interp_mid - exact_mid).abs() < 1e-12);

        let _audio = FtzDazGuard::new();
        let mut dl_xfade = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let mut dl_jump = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl_xfade.current_delay = 40.0;
        dl_xfade.target_delay = 40.0;
        dl_jump.current_delay = 40.0;
        dl_jump.target_delay = 40.0;

        let mut prev_x = 0.0_f32;
        let mut prev_j = 0.0_f32;
        for n in 0..315 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            prev_x = dl_xfade.step(s);
            prev_j = dl_jump.step(s);
        }

        dl_xfade.set_target_delay(100.0, true);
        dl_jump.current_delay = 100.0;
        dl_jump.target_delay = 100.0;

        let mut max_diff_xfade = 0.0_f32;
        let mut max_diff_jump = 0.0_f32;

        for n in 315..450 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            let yx = dl_xfade.step(s);
            let yj = dl_jump.step(s);
            max_diff_xfade = max_diff_xfade.max((yx - prev_x).abs());
            max_diff_jump = max_diff_jump.max((yj - prev_j).abs());
            prev_x = yx;
            prev_j = yj;
        }

        assert!(max_diff_jump > 10.0 * max_diff_xfade);
    }

    /// Test 8D & 8E (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): Verifies 128-byte `ControlChannelRecord` framing,
    /// `control_table_offset()` preservation across `update_header_in_place`, $\mathcal{O}(1)$
    /// `LatentAccumulator64B` delta updates across $P = 32$ controls, and non-stop Phase-1/2/3 clamping.
    #[test]
    fn test_8d_8e_control_channel_record_o1_latent_and_3phase_clamping() {
        assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
        assert_eq!(core::mem::size_of::<ControlLatentProjection>(), 64);
        assert_eq!(core::mem::size_of::<ControlChannelRecord>(), 128);
        assert_eq!(core::mem::size_of::<LatentAccumulator64B>(), 64);

        let mut projections = [ControlLatentProjection::new(); 32];
        let mut u_vals = [0.0_f32; 32];
        let mut psi_vals = [0.0_f32; 32];
        let mut acc = LatentAccumulator64B::new();

        for p in 0..32 {
            let pf = (p + 1) as f32;
            projections[p].w_main = core::array::from_fn(|d| 0.02 * pf * ((d + 1) as f32));
            projections[p].v_factors = core::array::from_fn(|f| 0.05 * ((p + f + 1) as f32));
            u_vals[p] = -0.8 + 0.05 * (p as f32);
            psi_vals[p] = 0.5 * u_vals[p];
            acc.update_delta(&projections[p], 0.0, u_vals[p], psi_vals[p]);
        }

        let new_u7 = 0.75_f32;
        let new_psi7 = 0.5 * new_u7;
        acc.update_delta(&projections[7], u_vals[7], new_u7, new_psi7 - psi_vals[7]);
        u_vals[7] = new_u7;
        psi_vals[7] = new_psi7;

        let z_o1 = acc.evaluate_z_ctrl();
        for d in 0..8 {
            let mut expected = 0.0_f32;
            for p in 0..32 {
                expected += projections[p].w_main[d] * psi_vals[p];
            }
            if d < 4 {
                for p in 0..32 {
                    for q in (p + 1)..32 {
                        expected += projections[p].v_factors[d]
                            * projections[q].v_factors[d]
                            * u_vals[p]
                            * u_vals[q];
                    }
                }
            }
            assert!((z_o1[d] - expected).abs() < 1e-4);
        }

        let mut header = DnamJsonHeader::default_reference();
        let mut cond_max = BTreeMap::new();
        cond_max.insert("boost=on".to_string(), 9.37_f32);
        header.controls_schema = vec![
            ControlSchemaDef {
                id: "gain".to_string(),
                name: "Gain".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([2.0, 8.0]),
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
                id: "feedback".to_string(),
                name: "Feedback".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
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
                id: "boost".to_string(),
                name: "Boost".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("off"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![
                    "off".to_string(),
                    "on".to_string(),
                    "ultra".to_string(),
                ]),
                active_options: Some(vec!["off".to_string(), "on".to_string()]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];

        let descriptors = header.compile_clamp_descriptors().unwrap();
        let mut cb = ControlBus::<ScalarBackend>::new();
        for (i, d) in descriptors.iter().enumerate() {
            cb.descriptors[i] = *d;
        }
        cb.num_controls = 3;

        cb.set_control_immediate(0, 9.5);
        assert!((cb.clamped_values[0] - 8.0).abs() < 1e-6);
        cb.set_control_immediate(0, 0.5);
        assert!((cb.clamped_values[0] - 2.0).abs() < 1e-6);

        cb.set_control_immediate(2, 2.0);
        assert!(cb.active_switch_options[2] <= 1);

        cb.set_control_immediate(2, 1.0);
        cb.set_control_immediate(1, 10.0);
        assert!((cb.clamped_values[1] - 9.37).abs() < 1e-5);
        assert_ne!(cb.active_clamp_mask & (1 << 1), 0);

        // Verify ControlChannelRecord table at tensor_arena_offset - 128 * P survives update_header_in_place
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let crypto = DefaultCryptoProvider::default();
        let view_before = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        let table_off = view_before.control_table_offset();
        assert_eq!(
            table_off,
            (view_before.preamble.tensor_arena_offset as usize) - 3 * 128
        );
        let table_snapshot = container[table_off..table_off + 3 * 128].to_vec();

        header
            .user_metadata
            .insert("tag".to_string(), serde_json::json!("preserved"));
        update_header_in_place(&mut container, &header).unwrap();
        assert_eq!(&container[table_off..table_off + 3 * 128], &table_snapshot);
    }

    /// Test 8G: Verifies the `LAPACK uplo='U'` Nested Principal Submatrix Prefix Theorem across all
    /// 4 tiers (`Lite = 64`, `Full = 128`, `Pro = 192`, `Ultra = 256`) and verifies `.dnam-ckpt`
    /// `Arena 4` stem `CRC32C` and `Arena 5` (`DNCA` / `INPC` / `STMC`) binary container framing.
    #[test]
    fn test_8g_dnam_ckpt_uplo_u_prefix_theorem_and_arena4_arena5_validation() {
        assert_eq!(uplo_u_packed_len(64) * 8, 16_640);
        assert_eq!(uplo_u_packed_len(128) * 8, 66_048);
        assert_eq!(uplo_u_packed_len(192) * 8, 148_224);
        assert_eq!(uplo_u_packed_len(256) * 8, 263_168);

        const N_ULTRA: usize = 256;
        let mut dense_ultra = vec![0.0_f64; N_ULTRA * N_ULTRA];
        for i in 0..N_ULTRA {
            for j in i..N_ULTRA {
                let val = ((i * 31 + j * 17 + 5) as f64) * 0.001;
                dense_ultra[i * N_ULTRA + j] = val;
                dense_ultra[j * N_ULTRA + i] = val;
            }
        }

        let mut packed_ultra = vec![0.0_f64; uplo_u_packed_len(N_ULTRA)];
        pack_uplo_u_f64(&dense_ultra, N_ULTRA, &mut packed_ultra);

        for &n_tier in &[64_usize, 128, 192, 256] {
            let prefix_len = uplo_u_packed_len(n_tier);
            let mut unpacked = vec![0.0_f64; n_tier * n_tier];
            unpack_uplo_u_f64(&packed_ultra[..prefix_len], n_tier, &mut unpacked);
            for r in 0..n_tier {
                for c in 0..n_tier {
                    assert_eq!(unpacked[r * n_tier + c], dense_ultra[r * N_ULTRA + c]);
                }
            }
        }

        let stem_bytes = vec![0x42_u8; ARENA4_STEM_BYTES_ULTRA];
        let stem_crc = crc32c_compute(&stem_bytes);
        assert!(
            verify_ckpt_stem_arena4_crc32c(&stem_bytes, 0, ARENA4_STEM_BYTES_ULTRA, stem_crc)
                .is_ok()
        );
        assert!(
            verify_ckpt_stem_arena4_crc32c(&stem_bytes, 0, ARENA4_STEM_BYTES_ULTRA, stem_crc ^ 1)
                .is_err()
        );

        let a5_len = (64 + 128 + 192) as u32;
        let mut a5 = vec![0u8; a5_len as usize];
        a5[0..4].copy_from_slice(&DNCA_MAGIC);
        a5[4..6].copy_from_slice(&1_u16.to_le_bytes());
        a5[8..10].copy_from_slice(&1_u16.to_le_bytes());
        a5[10..12].copy_from_slice(&1_u16.to_le_bytes());
        a5[12..16].copy_from_slice(&a5_len.to_le_bytes());

        let inpc_pos = 64_usize;
        a5[inpc_pos..inpc_pos + 4].copy_from_slice(b"INPC");
        a5[inpc_pos + 56..inpc_pos + 60].copy_from_slice(&128_u32.to_le_bytes());
        let inpc_crc = crc32c_compute(&a5[inpc_pos + 64..inpc_pos + 128]);
        a5[inpc_pos + 60..inpc_pos + 64].copy_from_slice(&inpc_crc.to_le_bytes());

        let stmc_pos = 192_usize;
        a5[stmc_pos..stmc_pos + 4].copy_from_slice(b"STMC");
        a5[stmc_pos + 116..stmc_pos + 120].copy_from_slice(&192_u32.to_le_bytes());
        let stmc_crc = crc32c_compute(&a5[stmc_pos + 128..stmc_pos + 192]);
        a5[stmc_pos + 120..stmc_pos + 124].copy_from_slice(&stmc_crc.to_le_bytes());

        let body_crc = crc32c_compute(&a5[64..]);
        a5[16..20].copy_from_slice(&body_crc.to_le_bytes());
        assert_eq!(verify_ckpt_arena5_calib_stats(&a5, 0).unwrap(), 384);

        // Also build a complete .dnam-ckpt container with Arena 4 + Arena 5 and load via DnamContainerView
        let mut ckpt_hdr = DnamJsonHeader::default_reference();
        ckpt_hdr.container_kind = "dnam-ckpt".to_string();
        ckpt_hdr.stems_registry = Some(vec![StemRegistryEntry {
            stem_id: "stem_0".to_string(),
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256:
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
                target_pcm_sha256:
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
                num_frames: 48_000,
            },
            stem_arena4_crc32c: stem_crc,
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            g_audio_uplo_u_f64: TensorSliceEntry {
                arena: 4,
                dtype: "f64".to_string(),
                packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
                feature_dim: Some(256),
                packed_len: Some(32_896),
                shape: Some(vec![32_896]),
                byte_offset: 64_000,
                byte_length: 263_168,
            },
            c_audio_tier_f64: TensorSliceEntry {
                arena: 4,
                dtype: "f64".to_string(),
                packing: None,
                feature_dim: None,
                packed_len: None,
                shape: Some(vec![2, 256]),
                byte_offset: 64_000 + 263_168,
                byte_length: 4_096,
            },
        }]);
        let mut full_ckpt_arena = vec![0u8; 64_000];
        full_ckpt_arena.extend_from_slice(&stem_bytes);
        full_ckpt_arena.extend_from_slice(&a5);
        let ckpt_container =
            build_dnam_container(&mut ckpt_hdr, &full_ckpt_arena, &full_ckpt_arena, false).unwrap();
        let crypto = DefaultCryptoProvider::default();
        let ckpt_view = DnamContainerView::from_bytes(&ckpt_container, &crypto).unwrap();
        assert_eq!(ckpt_view.header.container_kind, "dnam-ckpt");
    }

    #[test]
    fn test_theorem_3_unlimited_controls_31_band_eq_and_64_rack() {
        use crate::container::compute_header_crc32c;
        use crate::control::ControlLatentProjection;
        use crate::{DnamEngine, Tier};

        for &num_ctrls in &[31_usize, 64_usize] {
            let mut header = DnamJsonHeader::default_reference();
            header.controls_schema = (0..num_ctrls)
                .map(|p| ControlSchemaDef {
                    id: format!("band_{p}"),
                    name: format!("Band {p}"),
                    kind: ControlKind::Range,
                    unit: Some("dB".to_string()),
                    lobatto_degree: Some(2),
                    anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                    declared_range: Some([-12.0, 12.0]),
                    safe_clamped_range: Some([-12.0, 12.0]),
                    default: serde_json::json!(0.0),
                    conditional_min: BTreeMap::new(),
                    conditional_max: BTreeMap::new(),
                    declared_options: None,
                    active_options: None,
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                })
                .collect();

            let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
            let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

            let crypto = DefaultCryptoProvider::default();
            let view0 = DnamContainerView::from_bytes(&container, &crypto).unwrap();
            assert_eq!(view0.preamble.num_controls as usize, num_ctrls);
            assert_eq!(view0.control_records.len(), num_ctrls);

            // Populate CL 1 (ControlLatentProjection) of the last control record (p = num_ctrls - 1)
            let ctrl_off = view0.control_table_offset();
            let arena_off = view0.preamble.tensor_arena_offset as usize;
            let last_p = num_ctrls - 1;
            let proj = ControlLatentProjection {
                w_main: [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.05, 0.15],
                v_factors: [0.4, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
            };
            let rec_off = ctrl_off + last_p * 128;
            container[rec_off + 64..rec_off + 128].copy_from_slice(&proj.to_bytes());
            let new_hdr_crc = compute_header_crc32c(&container, arena_off);
            container[0x3C..0x40].copy_from_slice(&new_hdr_crc.to_le_bytes());

            let mut engine =
                DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
            assert_eq!(engine.preamble().num_controls as usize, num_ctrls);
            assert_eq!(engine.telemetry_snapshot().z_head, [0.0; 4]);

            // Move the last control (index 30 or 63, well beyond the old 16-control wall!)
            engine.set_control_immediate(last_p, 12.0);
            let z_after = engine.telemetry_snapshot().z_head;
            assert!(
                z_after[0].abs() > 0.1,
                "Control #{last_p} failed to update z_head via O(1) LatentAccumulator64B: {:?}",
                z_after
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property test verifying that `ControlChannelRecord` roundtrips bit-identically through
        /// 128-byte serialization and that `uplo='U'` packing/unpacking is lossless.
        #[test]
        fn prop_control_record_and_uplo_u_roundtrip(
            u0 in -1.0_f32..1.0_f32,
            w0 in -2.0_f32..2.0_f32,
            v0 in -1.0_f32..1.0_f32,
            diag in 0.1_f64..10.0_f64,
            off_diag in -2.0_f64..2.0_f64,
        ) {
            let mut rec = ControlChannelRecord::new();
            rec.clamp.anchor_nodes_norm[1] = u0;
            rec.projection.w_main[0] = w0;
            rec.projection.v_factors[0] = v0;
            let bytes = rec.to_bytes();
            let decoded = ControlChannelRecord::from_bytes(&bytes);
            prop_assert_eq!(decoded, rec);

            let dense = [diag, off_diag, off_diag, diag * 1.5];
            let mut packed = [0.0_f64; 3];
            pack_uplo_u_f64(&dense, 2, &mut packed);
            let mut unpacked = [0.0_f64; 4];
            unpack_uplo_u_f64(&packed, 2, &mut unpacked);
            prop_assert_eq!(unpacked, dense);
        }
    }

    #[test]
    fn test_update_header_in_place_recompiles_control_clamp_records() {
        use crate::container::{DnamContainerView, update_header_in_place};
        use crate::provenance::DefaultCryptoProvider;

        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = vec![ControlSchemaDef {
            id: "drive".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
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
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        // Update safe_clamped_range via update_header_in_place and verify ControlChannelRecord[0].clamp updates
        header.controls_schema[0].safe_clamped_range = Some([2.5, 7.5]);
        update_header_in_place(&mut container, &header).unwrap();

        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view.control_records.len(), 1);
        assert!((view.control_records[0].clamp.safe_min_base - 2.5).abs() < 1e-5);
        assert!((view.control_records[0].clamp.safe_max_base - 7.5).abs() < 1e-5);
    }
}
