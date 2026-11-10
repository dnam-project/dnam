#![forbid(unsafe_code)]
//! Canonical Stage Decomposition, Step-0 Variance Gating & 3-Phase Control-Space Audit Helpers.

/// Stability and clamping bounds enforced across Sweep 1 and Sweep 2 (`ref[train.audit.phase1.clamp-stability]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditBounds {
    /// Maximum discrete-time pole radius $r_c \le 0.9995$.
    pub max_pole_radius: f64,
    /// Maximum Stage IV linear feedback gain $g_{\mathrm{lin}} \le 0.9990$.
    pub max_tank_feedback_gain: f64,
    /// Maximum Schroeder all-pass coefficient $|a_{\mathrm{diff}}| \le 0.75$.
    pub max_allpass_coeff: f64,
    /// Minimum causal loop delay $\tau_m \ge 4.0$ samples.
    pub min_causal_delay_samples: f64,
}

impl Default for AuditBounds {
    fn default() -> Self {
        Self {
            max_pole_radius: 0.9995,
            max_tank_feedback_gain: 0.9990,
            max_allpass_coeff: 0.75,
            min_causal_delay_samples: 4.0,
        }
    }
}

impl AuditBounds {
    /// Clamps a candidate modal pole radius, tank feedback gain, all-pass coefficient, and loop delay
    /// into the certified stability envelope (`ref[train.audit.phase1.clamp-stability]`).
    #[must_use]
    pub fn clamp_parameters(
        &self,
        pole_radius: f64,
        tank_gain: f64,
        allpass_coeff: f64,
        delay_samples: f64,
    ) -> (f64, f64, f64, f64) {
        let max_r = if self.max_pole_radius.is_finite() {
            self.max_pole_radius.clamp(0.0, 0.9995)
        } else {
            0.9995
        };
        let max_g = if self.max_tank_feedback_gain.is_finite() {
            self.max_tank_feedback_gain.clamp(0.0, 0.9990)
        } else {
            0.9990
        };
        let max_a = if self.max_allpass_coeff.is_finite() {
            self.max_allpass_coeff.abs().min(0.75)
        } else {
            0.75
        };
        let min_d = if self.min_causal_delay_samples.is_finite() {
            self.min_causal_delay_samples.max(4.0)
        } else {
            4.0
        };
        let r = if pole_radius.is_finite() {
            pole_radius.clamp(0.0, max_r)
        } else {
            0.0
        };
        let g = if tank_gain.is_finite() {
            tank_gain.clamp(0.0, max_g)
        } else {
            0.0
        };
        let a = if allpass_coeff.is_finite() {
            allpass_coeff.clamp(-max_a, max_a)
        } else {
            0.0
        };
        let d = if delay_samples.is_finite() {
            delay_samples.max(min_d)
        } else {
            min_d
        };
        (r, g, a, d)
    }
}

/// Evaluates the Step-0 variance gate for control parameter `p` (`ref[dsp.ctrl.step0.variance-gating]`).
/// Returns `true` when the control is active across stems and `false` when it must be locked to $K_p = 0$.
#[must_use]
pub fn is_control_variance_active(observed_values: &[f64], min_spread: f64) -> bool {
    if observed_values.len() < 2 || !min_spread.is_finite() {
        return false;
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in observed_values {
        if !v.is_finite() {
            return false;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let spread = 0.5 * max_v - 0.5 * min_v;
    spread > 0.5 * min_spread.max(1.0e-9)
}

/// Smooth periodic cosine surrogate for Family-5 bitcrusher quantization step size $\Delta$
/// during gradient/coordinate refinement (`ref[dsp.op-n.bitcrusher.surrogate]`).
#[must_use]
pub fn bitcrusher_smooth_surrogate(x: f64, step_delta: f64, temp_tau: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let delta = if step_delta.is_finite() {
        step_delta.max(1.0e-6)
    } else {
        1.0e-6
    };
    let tau = if temp_tau.is_finite() {
        temp_tau.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q_hard = (x / delta).round() * delta;
    let q_smooth =
        x - (delta / (2.0 * std::f64::consts::PI)) * (2.0 * std::f64::consts::PI * x / delta).sin();
    (1.0 - tau) * q_hard + tau * q_smooth
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_variance_gating_edge_cases() {
        assert!(!is_control_variance_active(&[], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5, 0.5], 1.0e-4));
        assert!(is_control_variance_active(&[0.0, 1.0], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::INFINITY], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::NAN], 1.0e-4));
        let neg_bounds = AuditBounds {
            max_pole_radius: -1.0,
            max_tank_feedback_gain: -1.0,
            max_allpass_coeff: -0.5,
            min_causal_delay_samples: -2.0,
        };
        assert_eq!(
            neg_bounds.clamp_parameters(f64::NAN, f64::INFINITY, f64::NAN, f64::NAN),
            (0.0, 0.0, 0.0, 4.0)
        );
        let nan_bounds = AuditBounds {
            max_pole_radius: f64::NAN,
            max_tank_feedback_gain: f64::NAN,
            max_allpass_coeff: f64::NAN,
            min_causal_delay_samples: f64::NAN,
        };
        assert_eq!(
            nan_bounds.clamp_parameters(2.0, 2.0, 2.0, 1.0),
            (0.9995, 0.9990, 0.75, 4.0)
        );
        assert_eq!(bitcrusher_smooth_surrogate(f64::NAN, 0.1, 0.5), 0.0);
        assert!(bitcrusher_smooth_surrogate(0.5, f64::NAN, f64::NAN).is_finite());
    }

    proptest! {
        #[test]
        fn prop_audit_bounds_always_enforce_invariants(
            r in -2.0_f64..2.0_f64,
            g in -2.0_f64..2.0_f64,
            a in -2.0_f64..2.0_f64,
            d in -10.0_f64..100.0_f64,
        ) {
            let bounds = AuditBounds::default();
            let (rc, gc, ac, dc) = bounds.clamp_parameters(r, g, a, d);
            prop_assert!((0.0..=0.9995).contains(&rc));
            prop_assert!((0.0..=0.9990).contains(&gc));
            prop_assert!((-0.75..=0.75).contains(&ac));
            prop_assert!(dc >= 4.0);
        }

        #[test]
        fn prop_bitcrusher_surrogate_bounded_error(
            x in -10.0_f64..10.0_f64,
            delta in 0.01_f64..1.0_f64,
            tau in 0.0_f64..=1.0_f64,
        ) {
            let y = bitcrusher_smooth_surrogate(x, delta, tau);
            prop_assert!((y - x).abs() <= delta);
        }
    }
}
