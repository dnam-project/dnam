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
    let safe_min = declared.0.max(observed.0);
    let safe_max = declared.1.min(observed.1).max(safe_min);
    let degree = if distinct_observed_count <= 1 {
        0u8
    } else {
        (distinct_observed_count - 1).min(3) as u8
    };
    let was_clamped = (safe_min - declared.0).abs() > 1e-6 || (safe_max - declared.1).abs() > 1e-6;
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
    let r_worst = r_high + r_mod_l1.abs();
    let tau_min = tau_0 - tau_mod_l1.abs();
    let ok = r_worst <= 0.9995 && tau_min >= 4.0 && g_lin <= 0.9990 && a_diff.abs() <= 0.75;
    (
        r_worst.min(0.9995),
        tau_min.max(4.0),
        g_lin.min(0.9990),
        a_diff.clamp(-0.75, 0.75),
        ok,
    )
}

/// Phase-3 `[CLAMP-HEAD-GAIN]`: clamps rendered head spectral gain to `<= +6.0 dB` (`<= 1.9952623`
/// linear amplitude ratio, `ref[train.audit.phase3.clamp-head-gain]`).
#[inline]
#[must_use]
pub fn clamp_head_spectral_gain_linear(peak_gain_linear: f32) -> (f32, bool) {
    const MAX_GAIN_6DB_LIN: f32 = 1.995_262_3; // 10^(6/20)
    if peak_gain_linear > MAX_GAIN_6DB_LIN {
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

        let report = format_audit_prescription_report(
            "model",
            &["[CLAMP-RANGE] 'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]"],
        );
        assert!(report.contains("DNAM TRAINING COMPLETE"));
        assert!(report.contains("Active Control Clamps Applied (1):"));
    }
}
