#![forbid(unsafe_code)]
//! Non-Stop 3-Phase Control-Space Audit Helpers (`dnam_trainer::audit`:
//! `ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//! `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//! `ref[train.audit.report.prescription]`).

use std::collections::BTreeSet;

/// Phase-1 `[CLAMP-RANGE]`: intersects `declared_range` with `observed_range` and locks `K_p = 0`
/// if `distinct_observed_count <= 1` (`ref[train.audit.phase1.clamp-range]`).
#[inline]
#[must_use]
pub fn audit_phase1_clamp_range(
    declared: (f32, f32),
    observed: (f32, f32),
    distinct_observed_count: usize,
) -> ((f32, f32), u8, bool) {
    let d_lo = if declared.0.is_finite() {
        declared.0
    } else {
        0.0
    };
    let d_hi = if declared.1.is_finite() {
        declared.1.max(d_lo)
    } else {
        d_lo
    };
    let o_lo = if observed.0.is_finite() {
        observed.0
    } else {
        d_lo
    };
    let o_hi = if observed.1.is_finite() {
        observed.1.max(o_lo)
    } else {
        o_lo
    };
    let safe_min = d_lo.clamp(d_lo, d_hi).max(o_lo).min(d_hi);
    let safe_max = d_hi.min(o_hi).max(safe_min);
    let degree = if distinct_observed_count <= 1 || safe_max <= safe_min {
        0u8
    } else {
        (distinct_observed_count - 1).min(3) as u8
    };
    let was_clamped = !declared.0.is_finite()
        || !declared.1.is_finite()
        || !observed.0.is_finite()
        || !observed.1.is_finite()
        || declared.1 < declared.0
        || observed.1 < observed.0
        || safe_min != declared.0
        || safe_max != declared.1;
    ((safe_min, safe_max), degree, was_clamped)
}

/// Phase-1 `[CLAMP-SWITCH]`: restricts `declared_options` to options present in `observed_options`
/// (`ref[train.audit.phase1.clamp-switch]`).
#[must_use]
pub fn audit_phase1_clamp_switch(
    declared_options: &[&str],
    observed_options: &BTreeSet<&str>,
) -> (u16, Vec<String>) {
    let mut mask = 0u16;
    let mut disabled = Vec::new();
    for (idx, &opt) in declared_options.iter().enumerate().take(16) {
        if observed_options.contains(opt) {
            mask |= 1u16 << idx;
        } else {
            disabled.push(opt.to_string());
        }
    }
    (mask, disabled)
}

/// Phase-1 `[CLAMP-STABILITY]` & Phase-3 `[CLAMP-DYNAMIC]` worst-case excursion verifier
/// (`r_worst <= 0.9995`, `g_lin <= 0.9990`, `|a_diff| <= 0.75`, `tau_min >= 4.0`,
/// `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
/// `ref[train.audit.phase3.numeric-envelope]`).
#[inline]
#[must_use]
pub fn verify_dynamic_stability_excursion(
    r_high: f32,
    r_mod_l1: f32,
    tau_0: f32,
    tau_mod_l1: f32,
    g_lin: f32,
    a_diff: f32,
) -> (f32, f32, f32, f32, bool) {
    let finite = r_high.is_finite()
        && r_mod_l1.is_finite()
        && tau_0.is_finite()
        && tau_mod_l1.is_finite()
        && g_lin.is_finite()
        && a_diff.is_finite();
    let r_worst = r_high + r_mod_l1.abs();
    let tau_min = tau_0 - tau_mod_l1.abs();
    let ok = finite
        && r_high >= 0.0
        && (0.0..=0.9995).contains(&r_worst)
        && tau_min.is_finite()
        && tau_min >= 4.0
        && (0.0..=0.9990).contains(&g_lin)
        && a_diff.abs() <= 0.75;
    let r_c = if r_worst.is_finite() {
        r_worst.clamp(0.0, 0.9995)
    } else {
        0.0
    };
    let tau_c = if tau_min.is_finite() {
        tau_min.max(4.0)
    } else {
        4.0
    };
    let g_c = if g_lin.is_finite() {
        g_lin.clamp(0.0, 0.9990)
    } else {
        0.0
    };
    let a_c = if a_diff.is_finite() {
        a_diff.clamp(-0.75, 0.75)
    } else {
        0.0
    };
    (r_c, tau_c, g_c, a_c, ok)
}

/// Phase-3 `[CLAMP-HEAD-GAIN]`: clamps rendered head spectral gain to `<= +6.0 dB` (`<= 1.9952623`
/// linear amplitude ratio, `ref[train.audit.phase3.clamp-head-gain]`).
#[inline]
#[must_use]
pub fn clamp_head_spectral_gain_linear(peak_gain_linear: f32) -> (f32, bool) {
    const MAX_GAIN_6DB_LIN: f32 = 1.995_262_3; // 10^(6/20)
    if !peak_gain_linear.is_finite() || peak_gain_linear < 0.0 {
        (0.0, true)
    } else if peak_gain_linear > MAX_GAIN_6DB_LIN {
        (MAX_GAIN_6DB_LIN / peak_gain_linear, true)
    } else {
        (1.0, false)
    }
}

/// Formats a consolidated post-training audit & stem prescription banner (`ref[train.audit.report.prescription]`).
#[must_use]
pub fn format_audit_prescription_report(model_name: &str, clamp_lines: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "DNAM TRAINING COMPLETE — EXPORTED: {model_name}.dnam | {model_name}.dnam-ckpt\n"
    ));
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "Active Control Clamps Applied ({}):\n",
        clamp_lines.len()
    ));
    for (idx, line) in clamp_lines.iter().enumerate() {
        out.push_str(&format!("  {}. {line}\n", idx + 1));
    }
    out.push_str(
        "================================================================================\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[train.audit.phase1.clamp-range]
    /// verify[train.audit.phase1.clamp-switch]
    /// verify[train.audit.phase1.clamp-stability]
    /// verify[train.audit.phase3.clamp-dynamic]
    /// verify[train.audit.phase3.clamp-head-gain]
    /// verify[train.audit.phase3.numeric-envelope]
    /// verify[train.audit.report.prescription]
    #[test]
    fn test_three_phase_control_space_audit_and_prescription_report() {
        let ((s_min, s_max), deg0, clamped) = audit_phase1_clamp_range((0.0, 10.0), (2.0, 8.0), 1);
        assert_eq!((s_min, s_max), (2.0, 8.0));
        assert_eq!(deg0, 0);
        assert!(clamped);

        let (_, deg3, _) = audit_phase1_clamp_range((0.0, 10.0), (0.0, 10.0), 5);
        assert_eq!(deg3, 3);

        let obs_sw = BTreeSet::from(["off", "on"]);
        let (mask, disabled) = audit_phase1_clamp_switch(&["off", "on", "ultra"], &obs_sw);
        assert_eq!(mask, 0b011);
        assert_eq!(disabled, vec!["ultra".to_string()]);

        let (r_c, tau_c, g_c, a_c, ok) =
            verify_dynamic_stability_excursion(0.9990, 0.0020, 3.5, 1.0, 1.02, 0.85);
        assert!(!ok);
        assert_eq!(r_c, 0.9995);
        assert_eq!(tau_c, 4.0);
        assert_eq!(g_c, 0.9990);
        assert_eq!(a_c, 0.75);

        let (scale_ok, clamped_ok) = clamp_head_spectral_gain_linear(1.5);
        assert_eq!((scale_ok, clamped_ok), (1.0, false));
        let (scale_hi, clamped_hi) = clamp_head_spectral_gain_linear(4.0);
        assert!(clamped_hi && scale_hi < 0.5);
        assert_eq!(clamp_head_spectral_gain_linear(f32::NAN), (0.0, true));
        assert_eq!(clamp_head_spectral_gain_linear(-1.0), (0.0, true));
        let (r_n, tau_n, g_n, a_n, ok_n) =
            verify_dynamic_stability_excursion(-10.0, 0.0, f32::INFINITY, 0.0, -1.0, f32::NAN);
        assert!(!ok_n && r_n == 0.0 && tau_n == 4.0 && g_n == 0.0 && a_n == 0.0);
        let (r_nan, _, g_nan, _, ok_nan) =
            verify_dynamic_stability_excursion(f32::NAN, 0.0, 10.0, 0.0, f32::NAN, 0.0);
        assert!(!ok_nan && r_nan == 0.0 && g_nan == 0.0);
        let ((c_lo, c_hi), _, c_nan) = audit_phase1_clamp_range((0.0, 10.0), (f32::NAN, 8.0), 2);
        assert!(c_nan && c_lo.is_finite() && c_hi.is_finite());
        let ((d_lo, d_hi), _, d_nan) =
            audit_phase1_clamp_range((f32::NAN, f32::NAN), (0.0, f32::NAN), 2);
        assert!(d_nan && d_lo == 0.0 && d_hi == 0.0);

        let ((z_lo, z_hi), z_deg, z_clamped) =
            audit_phase1_clamp_range((0.0, 10.0), (-5.0, -2.0), 3);
        assert_eq!((z_lo, z_hi, z_deg, z_clamped), (0.0, 0.0, 0, true));
        let ((inv_lo, inv_hi), inv_deg, inv_clamped) =
            audit_phase1_clamp_range((10.0, 0.0), (10.0, 10.0), 1);
        assert_eq!(
            (inv_lo, inv_hi, inv_deg, inv_clamped),
            (10.0, 10.0, 0, true)
        );
        let (_, narrow_deg, narrow_clamped) =
            audit_phase1_clamp_range((0.0, 1.0e-6), (0.2e-6, 0.8e-6), 2);
        assert_eq!((narrow_deg, narrow_clamped), (1, true));

        let report = format_audit_prescription_report(
            "model",
            &["[CLAMP-RANGE] 'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]"],
        );
        assert!(report.contains("DNAM TRAINING COMPLETE"));
        assert!(report.contains("Active Control Clamps Applied (1):"));
    }
}
