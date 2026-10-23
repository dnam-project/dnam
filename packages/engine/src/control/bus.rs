//! Real-time $3\text{ kHz}$ Parametric Control Bus (`ControlBus<B>`).

use core::marker::PhantomData;

use crate::math::{MathBackend, SimdVec4};

use super::descriptor::ControlClampDescriptor;
use super::{ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, MAX_CONTROLS, SVD_HEAD_RANK};

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
    /// `range` controls.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
        self.raw_values[ctrl_idx] = physical_value;
        self.recompute_targets(false);
    }

    /// Sets a user control value and snaps `current_u_norm` immediately without anti-zipper lag.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.num_controls {
            return;
        }
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
                self.current_u_norm[p] = u_norm;
                if was_clamped {
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

    /// Executes one $3\text{ kHz}$ Control-Bus tick (`0 FDIV`, `0 FSQRT`, `0` heap allocations).
    #[inline]
    pub fn tick_3khz(
        &mut self,
        audio_abs_energy: f32,
        lfo_env_active: bool,
        arena_bytes: &[u8],
    ) -> bool {
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

                let c_hat = c_prev * c_inc - s_prev * s_inc;
                let s_hat = s_prev * c_inc + c_prev * s_inc;
                let r2 = c_hat * c_hat + s_hat * s_hat;
                let y1 = 1.5 - 0.5 * r2;
                let c_next = c_hat * y1;
                let s_next = s_hat * y1;
                self.lfo_phasors[j] = [c_next, s_next];

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ARENA_OFFSET_LITE_END;
    use crate::control::ConditionalClampRule;
    use crate::math::ScalarBackend;
    use alloc::boxed::Box;
    use alloc::vec;

    #[test]
    fn test_control_bus_slewing_and_trust_region_softclip() {
        let mut arena0 = vec![0u8; ARENA_OFFSET_LITE_END as usize];
        arena0[8064 + 16..8064 + 20].copy_from_slice(&0.95_f32.to_le_bytes());
        arena0[8064 + 32..8064 + 36].copy_from_slice(&0.5_f32.to_le_bytes());
        for off in (8128..9216).step_by(4) {
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
            anchor_nodes_norm: [0.0, -1.0, 1.0, 0.0],
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
            anchor_nodes_norm: [0.0; 4],
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.num_controls = 2;
        cb.set_control_value(0, 8.0);
        cb.set_control_value(99, 10.0);
        cb.set_control_immediate(99, 10.0);
        assert!(cb.tick_3khz(0.5, true, &arena0));

        cb.anti_zipper_pole = 0.0;
        cb.set_control_value(0, 10.0);
        cb.set_control_immediate(1, 0.0);
        assert_eq!(cb.clamped_values[1], 1.0);
        assert!(cb.tick_3khz(0.5, true, &arena0));
        assert!(!cb.tick_3khz(0.5, false, &arena0));
    }
}
