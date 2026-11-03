//! Real-time $3\text{ kHz}$ Parametric Control Bus (`ControlBus<B>`).

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

use super::descriptor::ControlClampDescriptor;
use super::record::{ControlChannelRecord, ControlLatentProjection, LatentAccumulator64B};
use super::{ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, TELEMETRY_CONTROL_SLOTS};

/// Real-time $3\text{ kHz}$ Parametric Control Bus state (`#[repr(C, align(64))]`).
///
/// Implements Theorem 3 (`docs/dsp/parametric_control_bus.md` §2.5): supports $P \in [0, 255]$
/// user controls via the unified 128-byte [`ControlChannelRecord`] table and the fixed 64-byte
/// running sufficient-statistic accumulator [`LatentAccumulator64B`] ($\mathcal{O}(1)$ per moved knob).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct ControlBus<B: MathBackend> {
    /// O(1) 64-byte running latent sufficient-statistic accumulator $\mathcal{S}_{\text{lat}} = (\mathbf{z}_{\text{main}}, \mathbf{s}_{vx}, \mathbf{s}_{v2x2})$ (`Lite`/`Full`/`Pro`, $D_h=4, R=4$).
    pub latent_acc: LatentAccumulator64B,
    /// Ultra-tier ($D_h=8, R=8$) FM linear sum accumulator for factors `4..8`.
    pub s_vx_ultra: [f32; 4],
    /// Ultra-tier ($D_h=8, R=8$) FM quadratic sum accumulator for factors `4..8`.
    pub s_v2x2_ultra: [f32; 4],
    /// Compiled 64-byte binary clamp descriptors (`CL 0` of `ControlChannelRecord[0..P]`).
    pub descriptors: Vec<ControlClampDescriptor>,
    /// Compiled 64-byte binary latent projections (`CL 1` of `ControlChannelRecord[0..P]`).
    pub projections: Vec<ControlLatentProjection>,
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
    /// Current base SVD control coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$ (`Lite`/`Full`/`Pro` modes `1..4`) after trust-region soft-clip (`CL-15`).
    pub z_head: [f32; 4],
    /// Current `Ultra`-tier (`Tier::Ultra`, modes `5..8`, $D_h = 8$) collapsed SVD control coordinates $\mathbf{z}_{\text{head,ultra}} \in \mathbb{R}^4$.
    pub z_head_ultra: [f32; 4],
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
    /// Trust-region radius $M_h$ (`CL-15`, default `2.5`).
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
            ctrl_basis_4: [0.0; 4],
            env_states: [0.0; 4],
            env_poles: [0.985, 0.960, 0.910, 0.0],
            lfo_phasors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_rotors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_harmonics: [0.0; 8],
            stage_targets: [0.0; 16],
            tank_delay_mod_samples: [0.0; 8],
            tank_gain_targets: [0.0; 8],
            tank_warp_mod: [0.0; 8],
            tank_grain_len_scale: [0.0; 4],
            tank_state_mod: [0.0; 4],
            tank_mod_dirty: false,
            trust_radius_mh: 2.5,
            inv_trust_radius_mh2: 1.0 / (2.5 * 2.5),
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

    /// Loads the embedded [`ControlClampDescriptor`] table (`CL 16..23`, `bytes[8512..9024]`) and
    /// global LFO/Envelope/Trust-Region anchors from `Arena 0` (`bytes[7488..10752]`) and snaps
    /// all controls to their `default_val`.
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
            let rho_head_cl48 = if arena0_bytes.len() >= 10_564 {
                f32::from_le_bytes([
                    arena0_bytes[10_560],
                    arena0_bytes[10_561],
                    arena0_bytes[10_562],
                    arena0_bytes[10_563],
                ])
            } else {
                0.0
            };
            let mh_off = global_base + 32;
            let mh_val = if (0.5..=10.0).contains(&rho_head_cl48) {
                rho_head_cl48
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
        self.lfo_harmonics = [0.0; 8];
        self.env_states = [0.0; 4];
    }

    /// Snaps all controls immediately to their default values (`z_head = 0`, `is_dirty = false`).
    pub fn reset_to_defaults(&mut self) {
        self.ensure_control_capacity(self.num_controls);
        self.latent_acc.reset();
        self.s_vx_ultra = [0.0; 4];
        self.s_v2x2_ultra = [0.0; 4];
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
    #[inline]
    pub fn tick_3khz(&mut self, audio_abs_energy: f32, arena_bytes: &[u8]) -> bool {
        for b in 0..3 {
            let pole = self.env_poles[b];
            self.env_states[b] = flush_recursive_state_f32(
                pole * self.env_states[b] + (1.0 - pole) * audio_abs_energy,
            );
        }
        // Causal attack onset detector (fast high-band envelope minus slow low-band envelope)
        self.env_states[3] = (self.env_states[2] - self.env_states[0]).max(0.0);
        for j in 0..2 {
            let c_prev = self.lfo_phasors[j][0];
            let s_prev = self.lfo_phasors[j][1];
            let c_inc = self.lfo_rotors[j][0];
            let s_inc = self.lfo_rotors[j][1];

            let c_hat = c_prev * c_inc - s_prev * s_inc;
            let s_hat = s_prev * c_inc + c_prev * s_inc;
            let r2 = c_hat * c_hat + s_hat * s_hat;
            let y1 = 1.5 - 0.5 * r2;
            let c_next = c_hat * y1;
            let s_next = s_hat * y1;
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
        // 1. Apply Theorem 3 O(1) Incremental Single-Knob Delta Update Law over any moved controls p in 0..P
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
                self.prev_eval_u_norm[p] = u_new;
                self.prev_eval_psi[p] = psi_new;
            }
        }
        let z_acc_8 = self.latent_acc.evaluate_z_ctrl();
        let z_acc_ultra_4 =
            LatentAccumulator64B::evaluate_z_ultra_fm(&self.s_vx_ultra, &self.s_v2x2_ultra);

        let mut basis_16 = [0.0_f32; 16];
        let mut b_idx = 0_usize;
        for p in 0..self.num_controls {
            if b_idx >= 16 {
                break;
            }
            if self.descriptors[p].ctrl_kind == 0 {
                let lob = self.descriptors[p].eval_lobatto_basis(self.current_u_norm[p]);
                let deg = (self.descriptors[p].lobatto_degree as usize).min(3);
                for k in 0..deg {
                    if b_idx < 16 {
                        basis_16[b_idx] = lob[k];
                        b_idx += 1;
                    }
                }
            } else if self.descriptors[p].ctrl_kind == 1 {
                let active_opt = self.active_switch_options[p] as usize;
                let num_non_def = self.descriptors[p].lobatto_degree as usize;
                for opt_k in 1..=num_non_def {
                    if b_idx < 16 {
                        basis_16[b_idx] = if active_opt == opt_k { 1.0 } else { 0.0 };
                        b_idx += 1;
                    }
                }
            }
        }

        self.ctrl_basis_4 = [
            basis_16[0] + z_acc_8[4],
            basis_16[1] + z_acc_8[5],
            basis_16[2] + z_acc_8[6],
            basis_16[3] + z_acc_8[7],
        ];
        let p_svd_base = 8_256_usize;
        let fm_base = 7_744_usize;
        let mut z_raw = B::Vec4::from_array([z_acc_8[0], z_acc_8[1], z_acc_8[2], z_acc_8[3]]);

        if arena_bytes.len() >= p_svd_base + 256 {
            let mut sum_vx = B::Vec4::zero();
            let mut sum_v2x2 = B::Vec4::zero();

            for row in 0..12 {
                let x_val = basis_16[row];
                if x_val.abs() > 1e-12 {
                    let x_vec = B::Vec4::splat(x_val);
                    let v_off = fm_base + row * 16;
                    let v_row = read_vec4_le::<B>(arena_bytes, v_off);
                    let vx = v_row.mul(x_vec);
                    sum_vx = sum_vx.add(vx);
                    sum_v2x2 = sum_v2x2.fma(vx, vx);
                }
            }
            let fm_arr = sum_vx
                .mul(sum_vx)
                .sub(sum_v2x2)
                .mul(B::Vec4::splat(0.5))
                .to_array();
            for m in 0..4 {
                if 12 + m >= b_idx {
                    basis_16[12 + m] = fm_arr[m];
                }
            }

            for row in 0..16 {
                let x_val = basis_16[row];
                if x_val.abs() > 1e-12 {
                    let x_vec = B::Vec4::splat(x_val);
                    let p_off = p_svd_base + row * 16;
                    let p_row = read_vec4_le::<B>(arena_bytes, p_off);
                    z_raw = z_raw.fma(x_vec, p_row);
                }
            }
        }

        let z_norm_sq = (z_raw.mul(z_raw).horizontal_sum() * self.inv_trust_radius_mh2).min(1.0);
        let radial_scale = if z_norm_sq <= 0.25 {
            1.0
        } else {
            let t = (z_norm_sq - 0.25) * (4.0 / 3.0);
            1.0 - (1.0 / 3.0) * t * t * (3.0 - 2.0 * t)
        };
        self.z_head = z_raw.mul(B::Vec4::splat(radial_scale)).to_array();

        // Evaluate `Arena 3` (`Tier::Ultra`) Rank-8 FM (`V_int_ultra` at 62,912..63,168)
        // and Nested SVD Prefix modes 5..8 (`P_svd_ultra` at 63,168..63,424) when present.
        const FM_ULTRA_BASE: usize = 62_912;
        const P_SVD_ULTRA_BASE: usize = 63_168;
        if arena_bytes.len() >= P_SVD_ULTRA_BASE + 256 {
            let mut sum_vx_u = B::Vec4::zero();
            let mut sum_v2x2_u = B::Vec4::zero();
            for row in 0..12 {
                let x_val = basis_16[row];
                if x_val.abs() > 1e-12 {
                    let x_vec = B::Vec4::splat(x_val);
                    let v_off = FM_ULTRA_BASE + row * 16;
                    let v_row = read_vec4_le::<B>(arena_bytes, v_off);
                    let vx = v_row.mul(x_vec);
                    sum_vx_u = sum_vx_u.add(vx);
                    sum_v2x2_u = sum_v2x2_u.fma(vx, vx);
                }
            }
            let fm_u_arr = sum_vx_u
                .mul(sum_vx_u)
                .sub(sum_v2x2_u)
                .mul(B::Vec4::splat(0.5))
                .to_array();
            let mut basis_ultra_16 = basis_16;
            for m in 0..4 {
                if 12 + m >= b_idx {
                    basis_ultra_16[12 + m] = fm_u_arr[m];
                }
            }
            let mut z_raw_u = B::Vec4::from_array(z_acc_ultra_4);
            for (row, &x_val) in basis_ultra_16.iter().enumerate() {
                if x_val.abs() > 1e-12 {
                    let x_vec = B::Vec4::splat(x_val);
                    let p_off = P_SVD_ULTRA_BASE + row * 16;
                    let p_row = read_vec4_le::<B>(arena_bytes, p_off);
                    z_raw_u = z_raw_u.fma(x_vec, p_row);
                }
            }
            let z_u_norm_sq =
                (z_raw_u.mul(z_raw_u).horizontal_sum() * self.inv_trust_radius_mh2).min(1.0);
            let radial_scale_u = if z_u_norm_sq <= 0.25 {
                1.0
            } else {
                let t = (z_u_norm_sq - 0.25) * (4.0 / 3.0);
                1.0 - (1.0 / 3.0) * t * t * (3.0 - 2.0 * t)
            };
            self.z_head_ultra = z_raw_u.mul(B::Vec4::splat(radial_scale_u)).to_array();
        } else {
            self.z_head_ultra = [0.0; 4];
        }

        self.evaluate_tank_modulation(arena_bytes);
    }

    /// Evaluates the $3\text{ kHz}$ `8 × 32` / `16 × 16` Control/LFO/Envelope $\to$ Stage & Spatial Tank
    /// state-space projection matrix `W_ctrl` (`CL 32..47`, `bytes 9536..10560` in `Arena 0`, `0 FDIV`, `0 FSQRT`).
    #[inline]
    fn evaluate_tank_modulation(&mut self, arena_bytes: &[u8]) {
        const W_CTRL_TANK_BASE: usize = 9_536;
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

    /// Evaluates division-free linear micro-ramping deltas with precomputed step alpha_ramp (CL-16):
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

    /// Audio-rate sample-by-sample parameter update via single FMA.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_micro_ramp(theta_audio: &mut [f32; 16], deltas: &[f32; 16]) {
        for i in 0..16 {
            theta_audio[i] += deltas[i];
        }
    }
}

impl<B: MathBackend> Default for ControlBus<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn read_vec4_le<B: MathBackend>(bytes: &[u8], off: usize) -> B::Vec4 {
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
        for off in (7744..8512).step_by(4) {
            arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        }
        // Populate P_SVD_ULTRA_BASE (63,168..63,424) with large weights to exercise z_u_norm_sq > 0.25
        for off in (63_168..63_424).step_by(4) {
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
