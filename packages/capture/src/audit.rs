#![forbid(unsafe_code)]
//! Non-Stop 3-Phase Control-Space Audit Helpers (`dnam_capture::audit`:
//! `ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//! `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase2.corner-barrier]`,
//! `ref[train.audit.phase2.clamp-rail]`, `ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//! `ref[train.audit.report.prescription]`).

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::FtzDazGuard;
use dnam_engine::provenance::{STAGE3_KAUTZ_BASES, STAGE3_KAUTZ_BLOCKS_PER_TIER};
use dnam_engine::schema::ControlKind;
use dnam_engine::stages::Stage2CommutedCascade;

use crate::proj::capsules::StmcTrajectoryWindow;
use crate::proj::schema::{AuditReport, DnamProjManifest, StemRegistryEntry};

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
        "DNAM TRAINING COMPLETE — EXPORTED: {model_name}.dnam | {model_name}.dnam-proj\n"
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

/// Recomputes Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` safe control domains
/// and updates `manifest.audit_report` across all stems with `enabled == true`.
pub fn recompute_vault_phase1_audit_and_clamps(manifest: &mut DnamProjManifest) {
    recompute_vault_phase1_audit_and_clamps_with_windows(manifest, &BTreeMap::new());
}

/// Recomputes Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` safe control domains
/// across all enabled stems in `manifest`, incorporating both `controls_meta` and positive-weight
/// [`StmcTrajectoryWindow`] observations from `enabled_windows_by_stem`.
pub fn recompute_vault_phase1_audit_and_clamps_with_windows(
    manifest: &mut DnamProjManifest,
    enabled_windows_by_stem: &BTreeMap<u32, Vec<StmcTrajectoryWindow>>,
) {
    let enabled_stems: Vec<&StemRegistryEntry> = manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled)
        .collect();

    let switch_ids: Vec<String> = manifest
        .base
        .controls_schema
        .iter()
        .filter(|c| c.kind == ControlKind::Switch)
        .map(|c| c.id.clone())
        .collect();

    let mut phase1_tags = BTreeSet::new();
    let mut prescriptions = Vec::new();

    for (ctrl_idx, ctrl) in manifest.base.controls_schema.iter_mut().enumerate() {
        match ctrl.kind {
            ControlKind::Range => {
                let decl = ctrl.declared_range.unwrap_or([0.0, 1.0]);
                let default_f32 = ctrl.default.as_f64().unwrap_or(f64::from(decl[0])) as f32;
                let mut vals: Vec<f32> = enabled_stems
                    .iter()
                    .filter_map(|s| s.controls_meta.get(&ctrl.id).and_then(|v| v.as_f64()))
                    .map(|v| v as f32)
                    .filter(|v| v.is_finite())
                    .collect();
                for s in &enabled_stems {
                    if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
                        for w in wins {
                            if usize::from(w.control_id) == ctrl_idx
                                && w.ctrl_kind == 0
                                && w.window_weight > 0.0
                                && w.u_phys.is_finite()
                            {
                                vals.push(w.u_phys);
                            }
                        }
                    }
                }

                let ((safe_lo, safe_hi), deg, clamped) = if vals.is_empty() {
                    let c = default_f32.clamp(decl[0], decl[1]);
                    ((c, c), 0u8, true)
                } else {
                    vals.sort_by(|a, b| a.total_cmp(b));
                    let mut distinct = 1usize;
                    for pair in vals.windows(2) {
                        if (pair[1] - pair[0]).abs() > 1.0e-4 {
                            distinct += 1;
                        }
                    }
                    let obs = (vals[0], *vals.last().unwrap());
                    audit_phase1_clamp_range((decl[0], decl[1]), obs, distinct)
                };

                ctrl.safe_clamped_range = Some([safe_lo, safe_hi]);
                ctrl.lobatto_degree = Some(deg);

                ctrl.conditional_min.clear();
                ctrl.conditional_max.clear();
                for sw_id in &switch_ids {
                    let mut by_opt: BTreeMap<String, (f32, f32)> = BTreeMap::new();
                    for s in &enabled_stems {
                        let Some(opt_str) = s.controls_meta.get(sw_id).and_then(|v| v.as_str())
                        else {
                            continue;
                        };
                        let mut stem_vals = Vec::new();
                        if let Some(v_num) = s.controls_meta.get(&ctrl.id).and_then(|v| v.as_f64())
                            && v_num.is_finite()
                        {
                            stem_vals.push((v_num as f32).clamp(safe_lo, safe_hi));
                        }
                        if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
                            for w in wins {
                                if usize::from(w.control_id) == ctrl_idx
                                    && w.ctrl_kind == 0
                                    && w.window_weight > 0.0
                                    && w.u_phys.is_finite()
                                {
                                    stem_vals.push(w.u_phys.clamp(safe_lo, safe_hi));
                                }
                            }
                        }
                        if stem_vals.is_empty() {
                            continue;
                        }
                        let key = format!("{sw_id}={opt_str}");
                        for vf in stem_vals {
                            by_opt
                                .entry(key.clone())
                                .and_modify(|(lo, hi)| {
                                    *lo = lo.min(vf);
                                    *hi = hi.max(vf);
                                })
                                .or_insert((vf, vf));
                        }
                    }
                    for (cond_key, (c_lo, c_hi)) in by_opt {
                        if c_lo > safe_lo + 1.0e-4 {
                            ctrl.conditional_min.insert(cond_key.clone(), c_lo);
                            phase1_tags.insert("CLAMP-CORNER".to_string());
                        }
                        if c_hi < safe_hi - 1.0e-4 {
                            ctrl.conditional_max.insert(cond_key, c_hi);
                            phase1_tags.insert("CLAMP-CORNER".to_string());
                        }
                    }
                }

                if clamped {
                    phase1_tags.insert("CLAMP-RANGE".to_string());
                    ctrl.clamp_warning = Some(format!(
                        "[CLAMP-RANGE] '{}': clamped [{:.3}, {:.3}] -> [{:.3}, {:.3}]",
                        ctrl.id, decl[0], decl[1], safe_lo, safe_hi
                    ));
                    prescriptions.push(format!(
                        "Capture stems at '{}={:.3}' and '{}={:.3}' to unlock full range",
                        ctrl.id, decl[0], ctrl.id, decl[1]
                    ));
                } else {
                    ctrl.clamp_warning = None;
                }
            }
            ControlKind::Switch => {
                let decl_opts = ctrl.declared_options.clone().unwrap_or_default();
                let decl_refs: Vec<&str> = decl_opts.iter().map(String::as_str).collect();
                let mut obs_strings: BTreeSet<String> = enabled_stems
                    .iter()
                    .filter_map(|s| s.controls_meta.get(&ctrl.id).and_then(|v| v.as_str()))
                    .map(ToString::to_string)
                    .collect();
                for s in &enabled_stems {
                    if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
                        for w in wins {
                            if usize::from(w.control_id) == ctrl_idx
                                && w.ctrl_kind == 1
                                && w.window_weight > 0.0
                                && let Some(opt_name) = decl_opts.get(usize::from(w.active_opt))
                            {
                                obs_strings.insert(opt_name.clone());
                            }
                        }
                    }
                }
                let obs_refs: BTreeSet<&str> = obs_strings.iter().map(String::as_str).collect();
                let (_mask, disabled) = audit_phase1_clamp_switch(&decl_refs, &obs_refs);
                let mut active: Vec<String> = decl_opts
                    .iter()
                    .filter(|o| obs_strings.contains(*o))
                    .cloned()
                    .collect();
                if active.is_empty() && !decl_opts.is_empty() {
                    let fallback = ctrl
                        .default
                        .as_str()
                        .filter(|d| decl_opts.iter().any(|o| o == d))
                        .unwrap_or(decl_opts[0].as_str())
                        .to_string();
                    active.push(fallback);
                }
                ctrl.active_options = Some(active);
                if !disabled.is_empty() {
                    phase1_tags.insert("CLAMP-SWITCH".to_string());
                    ctrl.clamp_warning = Some(format!(
                        "[CLAMP-SWITCH] '{}': disabled unobserved options {:?}",
                        ctrl.id, disabled
                    ));
                    for opt in disabled {
                        prescriptions.push(format!(
                            "Capture a stem with '{}={opt}' to unlock switch option",
                            ctrl.id
                        ));
                    }
                } else {
                    ctrl.clamp_warning = None;
                }
            }
        }
    }

    let prev = manifest.audit_report.take().unwrap_or_default();
    manifest.audit_report = Some(AuditReport {
        phase1_clamps: phase1_tags.into_iter().collect(),
        phase2_rail_warnings: prev.phase2_rail_warnings,
        phase3_clamps: prev.phase3_clamps,
        prescriptions,
    });
}

#[inline]
fn read_arena_f32(arena: &[u8], off: usize) -> f32 {
    if off + 4 <= arena.len() {
        f32::from_le_bytes([arena[off], arena[off + 1], arena[off + 2], arena[off + 3]])
    } else {
        0.0
    }
}

#[inline]
fn write_arena_f32(arena: &mut [u8], off: usize, val: f32) {
    if off + 4 <= arena.len() {
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Phase-2 Stop-Gradient Corner Barrier Loss $\mathcal{L}_{\mathrm{barrier}}$ evaluated across
/// physical stability bounds ($r_c \le 0.9995$, $g_{\mathrm{lin}} \le 0.9990$,
/// $|a_{\mathrm{diff}}| \le 0.75$, $\tau_m \ge 4.0$; `ref[train.audit.phase2.corner-barrier]`).
#[must_use]
pub fn compute_phase2_corner_barrier_loss(
    r_worst: f64,
    g_lin: f64,
    a_diff: f64,
    tau_min: f64,
) -> f64 {
    if !r_worst.is_finite() || !g_lin.is_finite() || !a_diff.is_finite() || !tau_min.is_finite() {
        return 1.0e6;
    }
    if !(0.0..=0.9995).contains(&r_worst)
        || !(0.0..=0.9990).contains(&g_lin)
        || a_diff.abs() > 0.75
        || tau_min < 4.0
    {
        let over_r = (r_worst - 0.9995).max(0.0) + (-r_worst).max(0.0);
        let over_g = (g_lin - 0.9990).max(0.0) + (-g_lin).max(0.0);
        let over_a = (a_diff.abs() - 0.75).max(0.0);
        let under_tau = (4.0 - tau_min).max(0.0);
        return 1.0e4
            + 1.0e5
                * (over_r * over_r + over_g * over_g + over_a * over_a + under_tau * under_tau);
    }
    let margin_r = (0.9995 - r_worst).max(1.0e-7);
    let margin_g = (0.9990 - g_lin).max(1.0e-7);
    let margin_a = (0.75 - a_diff.abs()).max(1.0e-7);
    let margin_tau = (tau_min - 4.0).max(1.0e-7);
    let barrier = -1.0e-4
        * ((margin_r / 0.01).min(1.0).ln()
            + (margin_g / 0.01).min(1.0).ln()
            + (margin_a / 0.05).min(1.0).ln()
            + (margin_tau / 1.0).min(1.0).ln());
    barrier.max(0.0)
}

/// Phase-2 `[CLAMP-RAIL]` `2,048`-Sample `PolyADAA` Rail-Lockup Audit (`ref[train.audit.phase2.clamp-rail]`).
///
/// Steps `2,048` samples of a multi-tone probe through [`Stage2CommutedCascade`] at `corner_drive`
/// and `corner_sag`. Returns `(rail_occupancy_ratio, safe_drive_scale, rail_locked)` where
/// `rail_locked == true` when normalized drive occupancy exceeds `0.95` (`95%`).
#[must_use]
pub fn audit_phase2_polyadaa_rail_lockup(
    arena_bytes: &[u8],
    corner_drive: f32,
    corner_sag: f32,
) -> (f32, f32, bool) {
    const BURST_SAMPLES: usize = 2_048;
    const RAIL_OCCUPANCY_LIMIT: f32 = 0.95;

    if !corner_drive.is_finite() || corner_drive <= 0.0 || !corner_sag.is_finite() {
        return (1.0, 0.25, true);
    }
    let _ftz = FtzDazGuard::new();
    let mut cascade = Stage2CommutedCascade::default();
    cascade.load_from_arena(arena_bytes);
    cascade.init_dc_equilibrium(0.0, cascade.v1a_beta, cascade.v1a_delta);

    let eff_drive = corner_drive.clamp(0.1, 64.0);
    let eff_sag = corner_sag.clamp(0.0, 0.8);
    let mut rail_hits = 0usize;
    let v1a_beta = cascade.v1a_beta.max(0.1);
    let v1b_beta = cascade.v1b_beta.max(0.1);

    for n in 0..BURST_SAMPLES {
        let t = n as f32;
        let x = 0.85 * ((0.065_449_85 * t).sin() + 0.35 * (0.219_911_49 * t).cos());
        let _ = cascade.step_frame(x, -x, eff_drive, eff_sag);
        let u1_norm = (x * eff_drive * v1a_beta).abs();
        let u2_norm = (cascade.prev_v1a_l * v1b_beta).abs();
        if u1_norm >= 1.0 || u2_norm >= 1.0 {
            rail_hits += 1;
        }
    }

    let occupancy = (rail_hits as f32) / (BURST_SAMPLES as f32);
    if occupancy > RAIL_OCCUPANCY_LIMIT {
        let safe_scale = (0.85 / (eff_drive * v1a_beta).max(1.0)).clamp(0.05, 1.0);
        (occupancy, safe_scale, true)
    } else {
        (occupancy, 1.0, false)
    }
}

/// Runs the complete Phase-1/2/3 stability, corner-barrier, `2,048`-sample `PolyADAA` rail-lockup,
/// dynamic excursion, and `+6.0 dB` head-gain audit in-place over `arena_64k`, updating
/// `manifest.audit_report` (`ref[train.audit.phase1.clamp-stability]`,
/// `ref[train.audit.phase2.corner-barrier]`, `ref[train.audit.phase2.clamp-rail]`,
/// `ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.clamp-head-gain]`).
pub fn run_full_3phase_arena_audit(manifest: &mut DnamProjManifest, arena_64k: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena_64k, true);
}

/// Runs Phase-3 `[CLAMP-HEAD-GAIN]` (`Modes 1..8`) and `[NUMERIC-ENVELOPE]` without mutating an
/// already-frozen `StarCenter` physical backbone (`Path 1` fast-path or `preserve_backbone()`).
pub fn run_head_only_3phase_arena_audit(manifest: &mut DnamProjManifest, arena_64k: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena_64k, false);
}

fn run_3phase_arena_audit_with_mode(
    manifest: &mut DnamProjManifest,
    arena_64k: &mut [u8],
    clamp_backbone: bool,
) {
    const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];
    let mut phase1_set: BTreeSet<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase1_clamps.iter().cloned().collect())
        .unwrap_or_default();
    let mut phase2_warns: Vec<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase2_rail_warnings.clone())
        .unwrap_or_default();
    let mut phase3_set: BTreeSet<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase3_clamps.iter().cloned().collect())
        .unwrap_or_default();
    let prescriptions = manifest
        .audit_report
        .as_ref()
        .map(|r| r.prescriptions.clone())
        .unwrap_or_default();

    if clamp_backbone {
        phase2_warns.clear();
        // 1. Phase-1 [CLAMP-STABILITY] & Phase-3 [CLAMP-DYNAMIC] on Stage I Op-M, Stage III Kautz & Stage IV tank
        for &s1_base in &[256usize, 10_752] {
            if s1_base + 32 <= arena_64k.len() {
                for lane in 0..4 {
                    let r_low = read_arena_f32(arena_64k, s1_base + lane * 4);
                    let dr = read_arena_f32(arena_64k, s1_base + 16 + lane * 4);
                    if r_low == 0.0 && dr == 0.0 {
                        continue;
                    }
                    let r_high = r_low + dr.max(0.0);
                    let (r_c, _, _, _, ok) =
                        verify_dynamic_stability_excursion(r_high, 0.0, 16.0, 0.0, 0.85, 0.5);
                    if !ok || !r_low.is_finite() || !dr.is_finite() || r_low < 0.0 {
                        let clamped_low = if r_low.is_finite() {
                            r_low.clamp(0.0, r_c)
                        } else {
                            0.0
                        };
                        let clamped_dr = if dr.is_finite() {
                            dr.clamp(0.0, (r_c - clamped_low).max(0.0))
                        } else {
                            0.0
                        };
                        write_arena_f32(arena_64k, s1_base + lane * 4, clamped_low);
                        write_arena_f32(arena_64k, s1_base + 16 + lane * 4, clamped_dr);
                        phase1_set.insert("CLAMP-STABILITY".to_string());
                        phase3_set.insert("CLAMP-DYNAMIC".to_string());
                    }
                }
            }
        }

        for (tier, (&base, &blocks)) in STAGE3_KAUTZ_BASES
            .iter()
            .zip(&STAGE3_KAUTZ_BLOCKS_PER_TIER)
            .enumerate()
        {
            for blk in 0..blocks {
                let b_off = base + blk * 128;
                if b_off + 128 > arena_64k.len() {
                    continue;
                }
                for lane in 0..4 {
                    let r_low = read_arena_f32(arena_64k, b_off + lane * 4);
                    let dr = read_arena_f32(arena_64k, b_off + (4 + lane) * 4);
                    if r_low == 0.0 && dr == 0.0 {
                        continue;
                    }
                    let r_high = r_low + dr.max(0.0);
                    let (r_c, _, _, _, ok) =
                        verify_dynamic_stability_excursion(r_high, 0.0, 16.0, 0.0, 0.85, 0.5);
                    if !ok {
                        let clamped_low = r_low.clamp(0.0, r_c);
                        let clamped_dr = dr.clamp(0.0, (r_c - clamped_low).max(0.0));
                        write_arena_f32(arena_64k, b_off + lane * 4, clamped_low);
                        write_arena_f32(arena_64k, b_off + (4 + lane) * 4, clamped_dr);
                        phase1_set.insert("CLAMP-STABILITY".to_string());
                        phase3_set.insert("CLAMP-DYNAMIC".to_string());
                        let _ = tier;
                    }
                }
            }
        }

        if arena_64k.len() >= 4_716 {
            for l in 0..4 {
                let d = read_arena_f32(arena_64k, 4_672 + l * 4);
                if d != 0.0 && (!d.is_finite() || !(4.0..=255.0).contains(&d)) {
                    write_arena_f32(arena_64k, 4_672 + l * 4, d.clamp(4.0, 255.0));
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
                let a_diff = read_arena_f32(arena_64k, 4_700 + l * 4);
                if a_diff != 0.0 && (!a_diff.is_finite() || a_diff.abs() > 0.75) {
                    let c_ad = if a_diff.is_finite() {
                        a_diff.clamp(-0.75, 0.75)
                    } else {
                        0.0
                    };
                    write_arena_f32(arena_64k, 4_700 + l * 4, c_ad);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
            let g_loop = read_arena_f32(arena_64k, 4_688);
            if g_loop != 0.0 {
                let (_, _, g_c, _, ok) =
                    verify_dynamic_stability_excursion(0.95, 0.0, 64.0, 0.0, g_loop, 0.5);
                if !ok {
                    write_arena_f32(arena_64k, 4_688, g_c);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                    phase3_set.insert("CLAMP-DYNAMIC".to_string());
                }
            }
        }
        for &(d_base, count, max_d) in &[
            (17_792usize, 4usize, 1_024.0_f32),
            (34_496, 4, 1_024.0),
            (48_896, 8, 2_048.0),
        ] {
            if d_base + count * 4 <= arena_64k.len() {
                for k in 0..count {
                    let d = read_arena_f32(arena_64k, d_base + k * 4);
                    if d != 0.0 && (!d.is_finite() || !(4.0..=max_d).contains(&d)) {
                        let cd = if d.is_finite() {
                            d.clamp(4.0, max_d)
                        } else {
                            4.0
                        };
                        write_arena_f32(arena_64k, d_base + k * 4, cd);
                        phase1_set.insert("CLAMP-STABILITY".to_string());
                    }
                }
            }
        }

        // 2. Phase-2 Corner Barrier + 2,048-sample PolyADAA Rail-Lockup Audit
        let g_now = read_arena_f32(arena_64k, 4_688).clamp(0.0, 0.9990);
        let tau_now = read_arena_f32(arena_64k, 4_672).max(4.0);
        let barrier =
            compute_phase2_corner_barrier_loss(0.9990, f64::from(g_now), 0.5, f64::from(tau_now));
        if barrier >= 1.0e4 {
            phase1_set.insert("CLAMP-STABILITY".to_string());
        }
        let (occ, safe_scale, locked) = audit_phase2_polyadaa_rail_lockup(arena_64k, 3.0, 0.25);
        if locked && arena_64k.len() >= 4_136 {
            let cur_b = read_arena_f32(arena_64k, 4_112).max(0.1);
            let cur_br = read_arena_f32(arena_64k, 4_116);
            write_arena_f32(arena_64k, 4_112, (cur_b * safe_scale).clamp(0.1, 4.0));
            if cur_br > 0.0 {
                write_arena_f32(arena_64k, 4_116, (cur_br * safe_scale).clamp(0.1, 4.0));
            }
            phase2_warns.push(format!(
                "[CLAMP-RAIL] PolyADAA corner rail occupancy {:.1}% > 95.0%; scaled V1A beta by {:.3}",
                occ * 100.0,
                safe_scale
            ));
        }
    }

    // 3. Phase-3 [CLAMP-HEAD-GAIN] on dynamic control modes 1..=8 (<= +6.0 dB excursion per mode)
    let arena_len = arena_64k.len();
    let mut clamp_mode_slice = |m_off: usize| {
        if m_off + 512 > arena_len {
            return;
        }
        let mut l2_sq_l = 0.0_f32;
        let mut l2_sq_r = 0.0_f32;
        let mut any_non_finite = false;
        for i in 0..64 {
            let wl = read_arena_f32(arena_64k, m_off + i * 4);
            let wr = read_arena_f32(arena_64k, m_off + 256 + i * 4);
            if !wl.is_finite() || !wr.is_finite() {
                any_non_finite = true;
            } else {
                l2_sq_l += wl * wl;
                l2_sq_r += wr * wr;
            }
        }
        let rms_gain = (l2_sq_l.max(l2_sq_r) / 16.0).sqrt();
        let (scale, was_clamped) = clamp_head_spectral_gain_linear(rms_gain);
        if was_clamped || any_non_finite {
            for i in 0..128 {
                let off = m_off + i * 4;
                let v = read_arena_f32(arena_64k, off);
                write_arena_f32(arena_64k, off, if v.is_finite() { v * scale } else { 0.0 });
            }
            phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
        }
    };
    for &h_base in &HEAD_BASES {
        for mode in 1..5 {
            clamp_mode_slice(h_base + mode * 512);
        }
    }
    if arena_len >= 49_408 + 8_192 {
        for t in 0..4 {
            for mode in 0..4 {
                clamp_mode_slice(49_408 + t * 2_048 + mode * 512);
            }
        }
    }
    phase3_set.insert("NUMERIC-ENVELOPE".to_string());

    manifest.audit_report = Some(AuditReport {
        phase1_clamps: phase1_set.into_iter().collect(),
        phase2_rail_warnings: phase2_warns,
        phase3_clamps: phase3_set.into_iter().collect(),
        prescriptions,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[train.audit.phase1.clamp-range]
    /// verify[train.audit.phase1.clamp-switch]
    /// verify[train.audit.phase1.clamp-stability]
    /// verify[train.audit.phase2.corner-barrier]
    /// verify[train.audit.phase2.clamp-rail]
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

        let b_safe = compute_phase2_corner_barrier_loss(0.95, 0.85, 0.4, 32.0);
        let b_edge = compute_phase2_corner_barrier_loss(0.9994, 0.9989, 0.749, 4.01);
        let b_viol = compute_phase2_corner_barrier_loss(1.01, 1.02, 0.9, 2.0);
        let b_nan = compute_phase2_corner_barrier_loss(f64::NAN, 0.85, 0.4, 32.0);
        assert!((0.0..1.0).contains(&b_safe));
        assert!(b_edge > b_safe && b_edge < 100.0);
        assert!(b_viol > 1.0e4);
        assert!(b_nan >= 1.0e6);

        let mut arena = vec![0u8; 64_000];
        let (occ_lo, scale_lo, locked_lo) = audit_phase2_polyadaa_rail_lockup(&arena, 0.5, 0.2);
        assert!(!locked_lo && scale_lo == 1.0 && occ_lo < 0.95);
        write_arena_f32(&mut arena, 4_112, 8.0);
        let (occ_hi, scale_hi, locked_hi) = audit_phase2_polyadaa_rail_lockup(&arena, 12.0, 0.2);
        assert!(locked_hi && occ_hi > 0.95 && scale_hi < 1.0);
        let (_, _, locked_nan) = audit_phase2_polyadaa_rail_lockup(&arena, f32::NAN, 0.2);
        assert!(locked_nan);

        // Trigger full 3-phase arena audit clamping on out-of-bounds poles, tank, and head modes
        write_arena_f32(&mut arena, 4_416, 1.02);
        write_arena_f32(&mut arena, 4_672, 2.0);
        write_arena_f32(&mut arena, 4_688, 1.05);
        write_arena_f32(&mut arena, 4_928 + 512, 25.0);
        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);
        assert!(read_arena_f32(&arena, 4_416) <= 0.9995);
        assert!(read_arena_f32(&arena, 4_672) >= 4.0);
        assert!(read_arena_f32(&arena, 4_688) <= 0.9990);
        assert!(read_arena_f32(&arena, 4_928 + 512) < 25.0);
        let ar = manifest.audit_report.expect("audit report");
        assert!(ar.phase1_clamps.iter().any(|t| t == "CLAMP-STABILITY"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
    }
}
