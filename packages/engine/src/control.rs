//! $3\text{ kHz}$ Parametric Control Bus, `ControlClampDescriptor` & Anti-Zipper Smoother (`control.rs`).
//!
//! Implements Subsystem 3 from [`docs/parametric_control_bus.md`](../../../docs/parametric_control_bus.md)
//! and [`docs/clamp_and_audit_registry.md`](../../../docs/clamp_and_audit_registry.md):
//! - **64-Byte [`ControlClampDescriptor`]:** Zero-`FDIV`, zero-string-parsing input clamping (`CL-12`),
//!   `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and switch-conditioned `[CLAMP-STABILITY]` / `[CLAMP-DYNAMIC]` rules.
//! - **1-Pole Anti-Zipper Smoother & Static Dirty-Bit Cache:** Smooths continuous `range` knob movements
//!   at $3\text{ kHz}$ ($\tau \approx 5\text{ ms}$) and snaps to `u_target` once $|u_p[k] - u_{p,\text{target}}| < 10^{-6}$,
//!   re-engaging the `0 MACs/sample` static collapsed head cache.
//! - **Node-Anchored Newton/Lobatto Polynomials (`CL-13`):** Anchored at $u_{p,1} = u_{p,\text{default}}$ so
//!   $\tilde{\boldsymbol{\psi}}_p(u_{p,\text{default}}) \equiv \mathbf{0}$.
//! - **Rank-4 Factorization Machine (FM) & SVD Principal Subspace (`CL-14`, `CL-15`, `CL-18`):**
//!   Evaluates arbitrary $N$-way knob+switch interactions in $\mathcal{O}(R \cdot P)$ flops with cubic
//!   trust-region soft-clipping ($M_h = 2.5$).
//! - **3-Band Envelope Follower & 4-Harmonic Chebyshev LFO Bank (`CL-16`):** Division-free 4-FMA +
//!   Newton rotor + $T_1..T_4$ Chebyshev recurrence (`0 sinf/cosf/fsqrt`).

use core::marker::PhantomData;
use core::mem::{align_of, size_of};

use crate::math::{MathBackend, SimdVec4};

/// Maximum number of parametric user controls (`range` + `switch`) in the inline `Arena 0` clamp table.
pub const MAX_CONTROLS: usize = 16;

/// SVD principal control subspace rank ($D_h = 4$).
pub const SVD_HEAD_RANK: usize = 4;

/// Factorization Machine interaction rank ($R = 4$).
pub const FM_INTERACTION_RANK: usize = 4;

/// Anti-zipper convergence threshold: once $|u_p - u_{p,\text{target}}| < 10^{-6}$, the smoother snaps
/// to `u_target` and re-engages the static `0 MACs/sample` head cache.
pub const ANTI_ZIPPER_SNAP_EPS: f32 = 1e-6;

/// Default $3\text{ kHz}$ 1-pole anti-zipper smoothing pole ($\tau \approx 5\text{ ms} \implies \alpha \approx 0.9355$).
pub const DEFAULT_ANTI_ZIPPER_POLE: f32 = 0.9355;

/// Binary switch-conditioned clamp rule (`8 B`, `#[repr(C)]`) inside [`ControlClampDescriptor`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalClampRule {
    /// Index of the conditioning `switch` control ($0 \dots 15$).
    pub switch_ctrl_idx: u8,
    /// Active option index of the `switch` control ($0 \dots 15$) that triggers this rule.
    pub switch_opt_idx: u8,
    /// Rule operation: `0 = none`, `1 = max_ceil` ($x \le \text{bound}$), `2 = min_floor` ($x \ge \text{bound}$).
    pub rule_op: u8,
    /// Reserved alignment byte (`0`).
    pub _pad: u8,
    /// Conditional physical bound value (`[CLAMP-STABILITY]` or `[CLAMP-DYNAMIC]`).
    pub bound_val: f32,
}

const _: () = assert!(size_of::<ConditionalClampRule>() == 8);

impl ConditionalClampRule {
    /// Returns an inactive (`rule_op = 0`) conditional clamp rule.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            switch_ctrl_idx: 0,
            switch_opt_idx: 0,
            rule_op: 0,
            _pad: 0,
            bound_val: 0.0,
        }
    }
}

/// Fixed 64-byte binary control normalization and clamp descriptor (`#[repr(C, align(64))]`, `<2BH9f4B1f4B1f4B1f`).
///
/// Stored at the start of `arena0_lite.control_bus_and_clamp_table` (`7040..8064` = `16 × 64 B`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlClampDescriptor {
    /// Control kind: `0 = range`, `1 = switch`, `0xFF = unused slot`.
    pub ctrl_kind: u8,
    /// For `range`: Lobatto polynomial degree $K_p \in \{0, 1, 2, 3\}$.
    /// For `switch`: `num_declared_options - 1`.
    pub lobatto_degree: u8,
    /// Bitmask of observed/active switch options (`[CLAMP-SWITCH]`; bit $i = 1$ if option $i$ is active).
    pub active_options_mask: u16,
    /// Default physical value (`range`) or default option index (`switch`).
    pub default_val: f32,
    /// Base safe minimum physical value (`[CLAMP-RANGE]`).
    pub safe_min_base: f32,
    /// Base safe maximum physical value (`[CLAMP-RANGE]`).
    pub safe_max_base: f32,
    /// Pre-inverted normalization scale $\frac{2}{u_{\text{decl\_max}} - u_{\text{decl\_min}}}$ (`0 FDIV` at $3\text{ kHz}$).
    pub scale_norm: f32,
    /// Normalization offset $-1 - u_{\text{decl\_min}} \cdot \text{scale\_norm}$.
    pub offset_norm: f32,
    /// Normalized Newton/Lobatto anchor nodes $[u_{p,1}, u_{p,2}, u_{p,3}, 0.0] \in [-1, +1]^4$.
    pub anchor_nodes_norm: [f32; 4],
    /// Up to 3 switch-coupled conditional clamp rules (`[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`).
    pub cond_rules: [ConditionalClampRule; 3],
}

const _: () = assert!(size_of::<ControlClampDescriptor>() == 64);
const _: () = assert!(align_of::<ControlClampDescriptor>() == 64);

impl ControlClampDescriptor {
    /// Returns an unused 64-byte [`ControlClampDescriptor`] slot (`ctrl_kind = 0xFF`).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            ctrl_kind: 0xFF,
            lobatto_degree: 0,
            active_options_mask: 0,
            default_val: 0.0,
            safe_min_base: 0.0,
            safe_max_base: 0.0,
            scale_norm: 0.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 4],
            cond_rules: [ConditionalClampRule::empty(); 3],
        }
    }

    /// Serializes this 64-byte descriptor into little-endian bytes (`<2BH9f4B1f4B1f4B1f`).
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0] = self.ctrl_kind;
        out[1] = self.lobatto_degree;
        out[2..4].copy_from_slice(&self.active_options_mask.to_le_bytes());
        out[4..8].copy_from_slice(&self.default_val.to_le_bytes());
        out[8..12].copy_from_slice(&self.safe_min_base.to_le_bytes());
        out[12..16].copy_from_slice(&self.safe_max_base.to_le_bytes());
        out[16..20].copy_from_slice(&self.scale_norm.to_le_bytes());
        out[20..24].copy_from_slice(&self.offset_norm.to_le_bytes());
        for i in 0..4 {
            let base = 24 + i * 4;
            out[base..base + 4].copy_from_slice(&self.anchor_nodes_norm[i].to_le_bytes());
        }
        for r in 0..3 {
            let base = 40 + r * 8;
            let rule = &self.cond_rules[r];
            out[base] = rule.switch_ctrl_idx;
            out[base + 1] = rule.switch_opt_idx;
            out[base + 2] = rule.rule_op;
            out[base + 3] = rule._pad;
            out[base + 4..base + 8].copy_from_slice(&rule.bound_val.to_le_bytes());
        }
        out
    }

    /// Parses a 64-byte [`ControlClampDescriptor`] from little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let ctrl_kind = bytes[0];
        let lobatto_degree = bytes[1];
        let active_options_mask = u16::from_le_bytes([bytes[2], bytes[3]]);
        let default_val = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let safe_min_base = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let safe_max_base = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let scale_norm = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let offset_norm = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let mut anchor_nodes_norm = [0.0_f32; 4];
        for i in 0..4 {
            let b = 24 + i * 4;
            anchor_nodes_norm[i] =
                f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
        }
        let mut cond_rules = [ConditionalClampRule::empty(); 3];
        for r in 0..3 {
            let b = 40 + r * 8;
            cond_rules[r] = ConditionalClampRule {
                switch_ctrl_idx: bytes[b],
                switch_opt_idx: bytes[b + 1],
                rule_op: bytes[b + 2],
                _pad: bytes[b + 3],
                bound_val: f32::from_le_bytes([
                    bytes[b + 4],
                    bytes[b + 5],
                    bytes[b + 6],
                    bytes[b + 7],
                ]),
            };
        }
        Self {
            ctrl_kind,
            lobatto_degree,
            active_options_mask,
            default_val,
            safe_min_base,
            safe_max_base,
            scale_norm,
            offset_norm,
            anchor_nodes_norm,
            cond_rules,
        }
    }

    /// Evaluates division-free clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`,
    /// `[CLAMP-DYNAMIC]`) and normalization (`CL-12`) in `< 5 ns` with `0 FDIV`.
    ///
    /// Returns `(clamped_physical_val, normalized_u_in_minus1_to_1, was_clamped)`.
    #[inline(always)]
    pub fn clamp_and_normalize(
        &self,
        raw_physical_val: f32,
        active_switch_options: &[u8; MAX_CONTROLS],
    ) -> (f32, f32, bool) {
        if self.ctrl_kind == 1 {
            // Switch control: check if requested option bit is enabled in active_options_mask
            let req_opt = (raw_physical_val.max(0.0) + 0.5) as u8;
            let bit = 1_u16.checked_shl(req_opt as u32).unwrap_or(0);
            if (self.active_options_mask & bit) != 0 {
                return (req_opt as f32, req_opt as f32, false);
            }
            // Fallback to first enabled option (or default_val)
            let def_opt = (self.default_val.max(0.0) + 0.5) as u8;
            let def_bit = 1_u16.checked_shl(def_opt as u32).unwrap_or(0);
            let safe_opt = if (self.active_options_mask & def_bit) != 0 {
                def_opt
            } else {
                self.active_options_mask.trailing_zeros().min(15) as u8
            };
            return (safe_opt as f32, safe_opt as f32, true);
        }

        // Range control: start with [safe_min_base, safe_max_base]
        let mut eff_min = self.safe_min_base;
        let mut eff_max = self.safe_max_base;

        for rule in &self.cond_rules {
            if rule.rule_op != 0 {
                let sw_idx = (rule.switch_ctrl_idx as usize) & (MAX_CONTROLS - 1);
                if active_switch_options[sw_idx] == rule.switch_opt_idx {
                    if rule.rule_op == 1 {
                        // max_ceil
                        eff_max = eff_max.min(rule.bound_val);
                    } else if rule.rule_op == 2 {
                        // min_floor
                        eff_min = eff_min.max(rule.bound_val);
                    }
                }
            }
        }
        if eff_max < eff_min {
            eff_max = eff_min;
        }

        let clamped = raw_physical_val.clamp(eff_min, eff_max);
        let was_clamped = (clamped - raw_physical_val).abs() > 1e-6;
        // u_p = fma(clamped, scale_norm, offset_norm), clamped to [-1.0, 1.0] (CL-12)
        let u_norm = (clamped * self.scale_norm + self.offset_norm).clamp(-1.0, 1.0);
        (clamped, u_norm, was_clamped)
    }

    /// Evaluates the Node-Anchored Newton/Lobatto polynomial basis (`CL-13`) at $u_p \in [-1, +1]$:
    /// - $\tilde{\psi}_{p,1}(u_p) = u_p - u_{p,1}$
    /// - $\tilde{\psi}_{p,2}(u_p) = (u_p - u_{p,1})(u_p - u_{p,2})$
    /// - $\tilde{\psi}_{p,3}(u_p) = (u_p - u_{p,1})(u_p - u_{p,2})(u_p - u_{p,3})$
    ///
    /// Because $u_{p,1} = u_{p,\text{default}}$, at the default preset $\tilde{\boldsymbol{\psi}}_p \equiv \mathbf{0}$.
    #[inline(always)]
    pub fn eval_lobatto_basis(&self, u_p: f32) -> [f32; 3] {
        if self.ctrl_kind != 0 || self.lobatto_degree == 0 {
            return [0.0; 3];
        }
        let d1 = u_p - self.anchor_nodes_norm[0];
        let psi1 = d1;
        let psi2 = if self.lobatto_degree >= 2 {
            d1 * (u_p - self.anchor_nodes_norm[1])
        } else {
            0.0
        };
        let psi3 = if self.lobatto_degree >= 3 {
            psi2 * (u_p - self.anchor_nodes_norm[2])
        } else {
            0.0
        };
        [psi1, psi2, psi3]
    }
}

/// Real-time $3\text{ kHz}$ Parametric Control Bus state (`#[repr(C, align(64))]`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct ControlBus<B: MathBackend> {
    /// Compiled 64-byte binary clamp descriptors (`16 × 64 B = 1,024 B`).
    pub descriptors: [ControlClampDescriptor; MAX_CONTROLS],
    /// Number of active controls $P \in [0, 16]$.
    pub num_controls: usize,
    /// Raw user-requested physical control values.
    pub raw_values: [f32; MAX_CONTROLS],
    /// Clamped physical control values after `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`.
    pub clamped_values: [f32; MAX_CONTROLS],
    /// Target normalized coordinates $u_{p,\text{target}} \in [-1, +1]$ (`range`) or option index (`switch`).
    pub target_u_norm: [f32; MAX_CONTROLS],
    /// Current anti-zipper smoothed normalized coordinates $u_p[k] \in [-1, +1]$.
    pub current_u_norm: [f32; MAX_CONTROLS],
    /// Active integer option index per `switch` control ($0 \dots 15$).
    pub active_switch_options: [u8; MAX_CONTROLS],
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// 1-pole anti-zipper smoothing coefficient $\alpha_{\text{smooth}} \in [0, 0.999]$ at $3\text{ kHz}$.
    pub anti_zipper_pole: f32,
    /// Whether any control is currently slewing or dirty (`false` = static `0 MACs/sample` cache active).
    pub is_dirty: bool,
    /// Current collapsed SVD control coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$ after trust-region soft-clip (`CL-15`).
    pub z_head: [f32; SVD_HEAD_RANK],
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
    /// Core 1 per-channel modulation offsets ($\Delta\theta_{\text{core1}} \in \mathbb{R}^{12}$).
    pub core1_mod_offsets: [f32; 12],
    /// Core 2 per-line delay modulation offsets in samples ($\Delta D_m[k] \in \mathbb{R}^8$).
    pub core2_delay_mod_samples: [f32; 8],
    /// Core 2 per-line quiet-state feedback gain targets ($g_{\text{lin},m}[k] \in [0, 0.9990]^8$).
    pub core2_gain_targets: [f32; 8],
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
            descriptors: [ControlClampDescriptor::empty(); MAX_CONTROLS],
            num_controls: 0,
            raw_values: [0.0; MAX_CONTROLS],
            clamped_values: [0.0; MAX_CONTROLS],
            target_u_norm: [0.0; MAX_CONTROLS],
            current_u_norm: [0.0; MAX_CONTROLS],
            active_switch_options: [0; MAX_CONTROLS],
            active_clamp_mask: 0,
            anti_zipper_pole: DEFAULT_ANTI_ZIPPER_POLE,
            is_dirty: false,
            z_head: [0.0; SVD_HEAD_RANK],
            env_states: [0.0; 4],
            env_poles: [0.985, 0.960, 0.910, 0.0],
            lfo_phasors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_rotors: [[1.0, 0.0], [1.0, 0.0]],
            lfo_harmonics: [0.0; 8],
            core1_mod_offsets: [0.0; 12],
            core2_delay_mod_samples: [0.0; 8],
            core2_gain_targets: [0.0; 8],
            trust_radius_mh: 2.5,
            inv_trust_radius_mh2: 1.0 / (2.5 * 2.5),
            _backend: PhantomData,
        }
    }

    /// Loads the 16-slot [`ControlClampDescriptor`] table and global LFO/Envelope anchors from
    /// `Arena 0` (`bytes[7040..10752]`) and snaps all controls to their `default_val`.
    pub fn load_from_arena0(&mut self, arena0_bytes: &[u8], num_controls: usize, rho_ctrl: f64) {
        self.num_controls = num_controls.min(MAX_CONTROLS);
        let table_base = 7040_usize;
        if arena0_bytes.len() >= table_base + 1024 {
            for p in 0..MAX_CONTROLS {
                let off = table_base + p * 64;
                let mut desc_bytes = [0u8; 64];
                desc_bytes.copy_from_slice(&arena0_bytes[off..off + 64]);
                let desc = ControlClampDescriptor::from_bytes(&desc_bytes);
                if p < self.num_controls && desc.ctrl_kind <= 1 {
                    self.descriptors[p] = desc;
                }
            }
        }

        // Load global LFO & Envelope anchors from 8064..8128 (if non-zero)
        let global_base = 8064_usize;
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
            let mh_off = global_base + 32;
            let mh_val = f32::from_le_bytes([
                arena0_bytes[mh_off],
                arena0_bytes[mh_off + 1],
                arena0_bytes[mh_off + 2],
                arena0_bytes[mh_off + 3],
            ]);
            if (0.5..=10.0).contains(&mh_val) {
                self.trust_radius_mh = mh_val;
                self.inv_trust_radius_mh2 = 1.0 / (mh_val * mh_val);
            }
        }

        self.reset_to_defaults();
    }

    /// Snaps all controls immediately to their default values (`z_head = 0`, `is_dirty = false`).
    pub fn reset_to_defaults(&mut self) {
        for p in 0..self.num_controls {
            let def_val = self.descriptors[p].default_val;
            self.raw_values[p] = def_val;
            if self.descriptors[p].ctrl_kind == 1 {
                self.active_switch_options[p] = (def_val.max(0.0) + 0.5) as u8;
            }
        }
        self.recompute_targets(true);
        self.z_head = [0.0; SVD_HEAD_RANK];
        self.core1_mod_offsets = [0.0; 12];
        self.core2_delay_mod_samples = [0.0; 8];
        self.lfo_phasors = [[1.0, 0.0], [1.0, 0.0]];
        self.env_states = [0.0; 4];
        self.is_dirty = false;
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation for continuous
    /// `range` controls (preventing zipper noise during rapid DAW automation or knob sweeps).
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.raw_values[ctrl_idx] = physical_value;
        self.recompute_targets(false);
    }

    /// Sets a user control value and snaps `current_u_norm` immediately without anti-zipper lag
    /// (used for preset loading or offline bounce initialization).
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.raw_values[ctrl_idx] = physical_value;
        self.recompute_targets(true);
    }

    /// Recomputes switch masks, conditional clamps, and normalized targets across all $P$ controls.
    fn recompute_targets(&mut self, snap_immediately: bool) {
        // Pass 1: update switch controls first so conditional range rules see the latest switch state
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
                self.current_u_norm[p] = u_norm;
                if was_clamped {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }

        // Pass 2: update range controls with active_switch_options applied
        for p in 0..self.num_controls {
            if self.descriptors[p].ctrl_kind == 0 {
                let (clamped, u_norm, was_clamped) = self.descriptors[p]
                    .clamp_and_normalize(self.raw_values[p], &self.active_switch_options);
                self.clamped_values[p] = clamped;
                self.target_u_norm[p] = u_norm;
                if snap_immediately {
                    self.current_u_norm[p] = u_norm;
                    self.is_dirty = true;
                } else if (self.current_u_norm[p] - u_norm).abs() > ANTI_ZIPPER_SNAP_EPS {
                    self.is_dirty = true;
                }
                if was_clamped {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }
        self.active_clamp_mask = clamp_mask;
    }

    /// Executes one $3\text{ kHz}$ Control-Bus tick (`0 FDIV`, `0 FSQRT`, `0` heap allocations):
    /// 1. Updates the 3-band envelope follower and 4-harmonic Chebyshev LFO bank (if `lfo_env_active`).
    /// 2. Advances the 1-pole anti-zipper smoother on continuous `range` controls and snaps once
    ///    $|u_p[k] - u_{p,\text{target}}| < 10^{-6}$.
    /// 3. When `is_dirty` is true, evaluates the Node-Anchored Lobatto basis (`CL-13`), Rank-4 FM
    ///    interaction (`CL-14`), and SVD principal subspace projection with cubic trust-region
    ///    soft-clipping (`CL-15`), returning `true` if the collapsed stereo output head $\mathbf{W}_{\text{head}}$
    ///    needs to be refreshed.
    #[inline]
    pub fn tick_3khz(
        &mut self,
        audio_abs_energy: f32,
        lfo_env_active: bool,
        arena_bytes: &[u8],
    ) -> bool {
        // 1. Advance 3-band envelope follower & 2-oscillator 4-harmonic Chebyshev LFO bank
        if lfo_env_active {
            for b in 0..3 {
                let pole = self.env_poles[b];
                self.env_states[b] = pole * self.env_states[b] + (1.0 - pole) * audio_abs_energy;
            }
            for j in 0..2 {
                let c_prev = self.lfo_phasors[j][0];
                let s_prev = self.lfo_phasors[j][1];
                let c_inc = self.lfo_rotors[j][0];
                let s_inc = self.lfo_rotors[j][1];

                // 4-FMA + 1-Step Newton-Raphson rotor renormalization (CL-03 / CL-16, 0 FSQRT/FDIV)
                let c_hat = c_prev * c_inc - s_prev * s_inc;
                let s_hat = s_prev * c_inc + c_prev * s_inc;
                let r2 = c_hat * c_hat + s_hat * s_hat;
                let y1 = 1.5 - 0.5 * r2;
                let c_next = c_hat * y1;
                let s_next = s_hat * y1;
                self.lfo_phasors[j] = [c_next, s_next];

                // 4-Harmonic Chebyshev recurrence T_1..T_4(cos phi_j) (CL-16):
                // T_1(x) = x
                // T_2(x) = 2x^2 - 1
                // T_3(x) = 2x T_2(x) - T_1(x)
                // T_4(x) = 2x T_3(x) - T_2(x)
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
        }

        // 2. Advance 1-pole anti-zipper smoother on continuous range controls
        if !self.is_dirty {
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
                        self.current_u_norm[p] = target;
                    } else {
                        self.current_u_norm[p] = next;
                        still_slewing = true;
                    }
                } else {
                    self.current_u_norm[p] = target;
                }
            }
        }

        // 3. Evaluate Node-Anchored Lobatto basis (16 features) + Rank-4 FM + SVD projection P_svd
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
                // One-hot non-default switch indicators (option 0 == default -> 0.0)
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

        // Project 16-element basis_16 through Rank-4 FM (at 8128..8384) and P_svd (16 x 4 floats at 8960..9216)
        let p_svd_base = 8960_usize;
        let fm_base = 8128_usize;
        let mut z_raw = B::Vec4::zero();

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
            // Rank-4 FM second-order interaction: 0.5 * ((sum v_i x_i)^2 - sum (v_i x_i)^2)
            // Placed in basis_16[12..16] so the 16x4 P_svd matrix projects both Lobatto and FM features
            // into the gauge-invariant SVD subspace!
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

        // Isotropic Euclidean trust-region soft-clip on z_head (CL-15, 0 FSQRT, 0 FDIV):
        // Uses ||z_raw||_2^2 which is 100% invariant under any orthogonal gauge rotation Q_buyer in SO(4)!
        let z_norm_sq = (z_raw.mul(z_raw).horizontal_sum() * self.inv_trust_radius_mh2).min(1.0);
        // Smooth radial cubic compression: unity gain inside trust region, soft-clamped at boundary
        let radial_scale = if z_norm_sq <= 0.25 {
            1.0
        } else {
            let t = (z_norm_sq - 0.25) * (4.0 / 3.0);
            1.0 - (1.0 / 3.0) * t * t * (3.0 - 2.0 * t)
        };
        self.z_head = z_raw.mul(B::Vec4::splat(radial_scale)).to_array();

        self.is_dirty = still_slewing;
        true
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

// ============================================================================
// Standalone `f32` / `f64` Reference DSP Kernels (for `packages/python` & Tests)
// ============================================================================

/// Reference `f64` Node-Anchored Newton/Lobatto basis evaluation (`CL-13`):
/// $$\psi_1(u) = u - u_1, \quad \psi_2(u) = (u - u_1)(u - u_2), \quad \psi_3(u) = (u - u_1)(u - u_2)(u - u_3)$$
#[inline]
pub fn eval_newton_lobatto_basis_f64(u: f64, anchors: [f64; 3], degree: usize) -> [f64; 3] {
    if degree == 0 {
        return [0.0; 3];
    }
    let d1 = u - anchors[0];
    let p1 = d1;
    let p2 = if degree >= 2 {
        d1 * (u - anchors[1])
    } else {
        0.0
    };
    let p3 = if degree >= 3 {
        p2 * (u - anchors[2])
    } else {
        0.0
    };
    [p1, p2, p3]
}

/// Reference `f64` Rank-4 Factorization Machine interaction evaluation (`CL-14`):
/// $$\Delta\theta_{\text{FM}}(\mathbf{x}) = \frac{1}{2} \sum_{r=1}^R \left[\left(\sum_{i=1}^D v_{i,r} x_i\right)^2 - \sum_{i=1}^D v_{i,r}^2 x_i^2\right]$$
pub fn eval_rank4_fm_interaction_f64(features: &[f64], v_factors: &[[f64; 4]]) -> [f64; 4] {
    let mut sum_vx = [0.0_f64; 4];
    let mut sum_v2x2 = [0.0_f64; 4];
    let n = features.len().min(v_factors.len());
    for i in 0..n {
        let x = features[i];
        for r in 0..4 {
            let vx = v_factors[i][r] * x;
            sum_vx[r] += vx;
            sum_v2x2[r] += vx * vx;
        }
    }
    let mut out = [0.0_f64; 4];
    for r in 0..4 {
        out[r] = 0.5 * (sum_vx[r] * sum_vx[r] - sum_v2x2[r]);
    }
    out
}

/// Reference `f64` 4-Harmonic Chebyshev LFO polynomial evaluation ($T_1..T_4(\cos\phi)$, `CL-16`).
#[inline]
pub fn eval_chebyshev_lfo_4harm_f64(cos_phi: f64) -> [f64; 4] {
    let t1 = cos_phi;
    let two_x = 2.0 * t1;
    let t2 = two_x * t1 - 1.0;
    let t3 = two_x * t2 - t1;
    let t4 = two_x * t3 - t2;
    [t1, t2, t3, t4]
}

/// Reference `f64` cubic trust-region soft-clipper (`CL-15`):
/// $$z_{\text{safe}} = M_h \cdot S_3(z / M_h), \qquad S_3(v) = \operatorname{clamp}(v, -1, 1)\big(1.5 - 0.5\operatorname{clamp}(v, -1, 1)^2\big)$$
#[inline]
pub fn eval_trust_region_softclip_f64(z: f64, mh: f64) -> f64 {
    let v = (z / mh).clamp(-1.0, 1.0);
    mh * v * (1.5 - 0.5 * v * v)
}
