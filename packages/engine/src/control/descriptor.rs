//! 64-byte binary control normalization & clamp descriptor (`ControlClampDescriptor`) and
//! switch-conditioned clamp rules (`ConditionalClampRule`).

use core::mem::{align_of, size_of};

use super::MAX_CONTROLS;

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
            let req_opt = (raw_physical_val.max(0.0) + 0.5) as u8;
            let bit = 1_u16.checked_shl(req_opt as u32).unwrap_or(0);
            if (self.active_options_mask & bit) != 0 {
                return (req_opt as f32, req_opt as f32, false);
            }
            let def_opt = (self.default_val.max(0.0) + 0.5) as u8;
            let def_bit = 1_u16.checked_shl(def_opt as u32).unwrap_or(0);
            let safe_opt = if (self.active_options_mask & def_bit) != 0 {
                def_opt
            } else {
                self.active_options_mask.trailing_zeros().min(15) as u8
            };
            return (safe_opt as f32, safe_opt as f32, true);
        }

        let mut eff_min = self.safe_min_base;
        let mut eff_max = self.safe_max_base;

        for rule in &self.cond_rules {
            if rule.rule_op != 0 {
                let sw_idx = (rule.switch_ctrl_idx as usize) & (MAX_CONTROLS - 1);
                if active_switch_options[sw_idx] == rule.switch_opt_idx {
                    if rule.rule_op == 1 {
                        eff_max = eff_max.min(rule.bound_val);
                    } else if rule.rule_op == 2 {
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
        let u_norm = (clamped * self.scale_norm + self.offset_norm).clamp(-1.0, 1.0);
        (clamped, u_norm, was_clamped)
    }

    /// Evaluates the Node-Anchored Newton/Lobatto polynomial basis (`CL-13`) at $u_p \in [-1, +1]$.
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_descriptor_serialization_and_conditional_rules() {
        let mut d_cond = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 2,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0, 0.0],
            cond_rules: [
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 1,
                    _pad: 0,
                    bound_val: 2.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 2,
                    _pad: 0,
                    bound_val: 5.0,
                },
                ConditionalClampRule::empty(),
            ],
        };
        let bytes = d_cond.to_bytes();
        let parsed = ControlClampDescriptor::from_bytes(&bytes);
        assert_eq!(parsed, d_cond);

        let mut sw_opts = [0u8; 16];
        sw_opts[1] = 1;
        let (clamped_v, _, was_cl) = d_cond.clamp_and_normalize(8.0, &sw_opts);
        assert_eq!(clamped_v, 5.0);
        assert!(was_cl);

        d_cond.cond_rules[0].bound_val = 7.0;
        let (clamped_v2, _, _) = d_cond.clamp_and_normalize(1.0, &sw_opts);
        assert_eq!(clamped_v2, 5.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_control_clamping_and_default_lobatto_zero(
            raw_val in -100.0_f32..100.0_f32,
        ) {
            let desc = ControlClampDescriptor {
                ctrl_kind: 0,
                lobatto_degree: 3,
                active_options_mask: 1,
                default_val: 5.0,
                safe_min_base: 2.0,
                safe_max_base: 8.0,
                scale_norm: 0.2,
                offset_norm: -1.0,
                anchor_nodes_norm: [0.0, -1.0, 1.0, 0.0],
                cond_rules: [ConditionalClampRule::empty(); 3],
            };
            let sw = [0u8; 16];
            let (clamped, u_norm, _) = desc.clamp_and_normalize(raw_val, &sw);
            prop_assert!((2.0..=8.0).contains(&clamped));
            prop_assert!((-1.0..=1.0).contains(&u_norm));

            let (_, u_def, _) = desc.clamp_and_normalize(5.0, &sw);
            let basis_def = desc.eval_lobatto_basis(u_def);
            prop_assert_eq!(basis_def, [0.0, 0.0, 0.0]);
        }
    }
}
