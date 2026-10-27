//! Star + Lobatto Axis + Switch + Corner + `.dnam-ckpt` Prescription capture schedule planner.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dnam_engine::container::DnamContainerView;
use dnam_engine::provenance::DefaultCryptoProvider;
use serde::{Deserialize, Serialize};

use crate::project::manifest::{ControlRangeSpec, ModelManifest};

/// Priority tier of a planned capture stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureTier {
    /// Center reference stem (`u = 0.5` across all ranges, default switches).
    StarCenter,
    /// Lobatto axis node (`u in {0.0, 1.0}`) along a single continuous knob.
    StarAxis,
    /// Discrete switch throw anchor.
    SwitchState,
    /// Multi-knob high/low-gain corner anchor.
    CornerAnchor,
    /// Prescribed stem coordinate from a `.dnam-ckpt` Phase 1..3 audit report.
    Prescribed,
}

/// Single step in a model's capture plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturePlanStep {
    /// Deterministic stem identifier.
    pub stem_id: String,
    /// Capture priority tier.
    pub tier: CaptureTier,
    /// Normalized `[0.0, 1.0]` continuous knob coordinates.
    pub normalized_ranges: BTreeMap<String, f32>,
    /// Zero-based discrete switch indices.
    pub switches: BTreeMap<String, usize>,
    /// Human-readable hardware dial instructions.
    pub prompt: String,
    /// Whether this stem has already been captured in `model.stems`.
    pub already_captured: bool,
}

/// Complete capture plan for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturePlan {
    /// Model identifier.
    pub model_id: String,
    /// Ordered capture steps.
    pub steps: Vec<CapturePlanStep>,
}

/// Prescribed capture coordinate `(normalized_ranges, switches, reason)` extracted from a `.dnam-ckpt` audit report.
pub type PrescriptionCoord = (BTreeMap<String, f32>, BTreeMap<String, usize>, String);

/// Extracts prescribed normalized range coordinates from an existing `.dnam-ckpt` container's
/// `audit_report` if present on disk.
pub fn extract_checkpoint_prescriptions(
    ckpt_path: &Path,
    model: &ModelManifest,
) -> Vec<PrescriptionCoord> {
    let Ok(bytes) = fs::read(ckpt_path) else {
        return Vec::new();
    };
    let crypto = DefaultCryptoProvider::default();
    let Ok(view) = DnamContainerView::from_bytes(&bytes, &crypto) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ctrl in &view.header.controls_schema {
        if let Some(warning) = &ctrl.clamp_warning
            && let Some(r) = model.ranges.iter().find(|r| r.id == ctrl.id)
        {
            let mut map = BTreeMap::new();
            for other in &model.ranges {
                let span = (other.max - other.min).max(1e-6);
                map.insert(
                    other.id.clone(),
                    ((other.default_value - other.min) / span).clamp(0.0, 1.0),
                );
            }
            map.insert(r.id.clone(), 1.0);
            let sw_map: BTreeMap<String, usize> = model
                .switches
                .iter()
                .map(|s| (s.id.clone(), s.default_index))
                .collect();
            out.push((map, sw_map, warning.clone()));
        }
    }
    if let Some(audit) = &view.header.audit_report {
        for text_p in &audit.prescriptions {
            let mut map: BTreeMap<String, f32> = model
                .ranges
                .iter()
                .map(|r| {
                    let span = (r.max - r.min).max(1e-6);
                    (
                        r.id.clone(),
                        ((r.default_value - r.min) / span).clamp(0.0, 1.0),
                    )
                })
                .collect();
            let mut sw_map: BTreeMap<String, usize> = model
                .switches
                .iter()
                .map(|s| (s.id.clone(), s.default_index))
                .collect();
            let mut matched_any = false;
            for token in text_p.replace([',', '(', ')'], " ").split_whitespace() {
                if let Some((k, v_raw)) = token.split_once('=') {
                    let key = k.trim();
                    let val_s = v_raw.trim().trim_matches(['\'', '"']);
                    if let Some(r) = model.ranges.iter().find(|r| r.id == key)
                        && let Ok(phys) = val_s.parse::<f32>()
                    {
                        let span = (r.max - r.min).max(1e-6);
                        map.insert(r.id.clone(), ((phys - r.min) / span).clamp(0.0, 1.0));
                        matched_any = true;
                    } else if let Some(s) = model.switches.iter().find(|s| s.id == key)
                        && let Some(opt) = s.options.iter().find(|o| o.label == val_s)
                    {
                        sw_map.insert(s.id.clone(), opt.index);
                        matched_any = true;
                    }
                }
            }
            if matched_any {
                out.push((map, sw_map, text_p.clone()));
            }
        }
    }
    out
}

/// Builds the deterministic Star + Switch + Corner + `.dnam-ckpt` Prescription capture schedule.
pub fn build_capture_plan_with_prescriptions(
    model: &ModelManifest,
    include_corners: bool,
    prescriptions: &[PrescriptionCoord],
) -> CapturePlan {
    let mut steps = Vec::new();
    let default_ranges: BTreeMap<String, f32> = model
        .ranges
        .iter()
        .map(|r| {
            let span = (r.max - r.min).max(1e-6);
            let norm = ((r.default_value - r.min) / span).clamp(0.0, 1.0);
            (r.id.clone(), norm)
        })
        .collect();
    let default_switches: BTreeMap<String, usize> = model
        .switches
        .iter()
        .map(|s| (s.id.clone(), s.default_index))
        .collect();

    // 1. Star center
    steps.push(make_step(
        model,
        "stem_00_star_center".to_string(),
        CaptureTier::StarCenter,
        default_ranges.clone(),
        default_switches.clone(),
        None,
    ));

    // 2. Lobatto axis sweeps (grouped: all `min=0.0` cuts first, then all `max=1.0` boosts,
    //    ending with primary gain `max=1.0` so it remains dialed in for Step 3 corners)
    let corner_pair = select_corner_range_pair(&model.ranges);
    let primary_id = corner_pair.map(|(p, _)| p.id.as_str());
    for (tag, norm_val) in [("min", 0.0_f32), ("max", 1.0_f32)] {
        let mut ordered_ranges: Vec<&ControlRangeSpec> = model.ranges.iter().collect();
        if tag == "max"
            && let Some(pid) = primary_id
        {
            ordered_ranges.sort_by_key(|r| usize::from(r.id == pid));
        }
        for range in ordered_ranges {
            if (norm_val - default_ranges.get(&range.id).copied().unwrap_or(0.5)).abs() < 1e-4 {
                continue;
            }
            let mut coords = default_ranges.clone();
            coords.insert(range.id.clone(), norm_val);
            let stem_id = format!("stem_axis_{}_{}", range.id, tag);
            steps.push(make_step(
                model,
                stem_id,
                CaptureTier::StarAxis,
                coords,
                default_switches.clone(),
                None,
            ));
        }
    }

    // 3. Multi-knob high-corner anchors on default switch state (immediately after primary
    //    gain `max=1.0`, before switch throws)
    if include_corners && let Some((primary, secondary_opt)) = corner_pair {
        if let Some(secondary) = secondary_opt {
            let mut corner_coords = default_ranges.clone();
            corner_coords.insert(primary.id.clone(), 1.0);
            corner_coords.insert(secondary.id.clone(), 1.0);
            steps.push(make_step(
                model,
                "stem_corner_high_pair".to_string(),
                CaptureTier::CornerAnchor,
                corner_coords,
                default_switches.clone(),
                None,
            ));

            // When the primary control is a dual-circuit morph control (`duality`, `mod`, `morph`)
            // where both `0.0` (Circuit A) and `1.0` (Circuit B) are active nonlinear engines,
            // schedule the remaining 3 corners of `(primary, secondary)` in 1-knob-at-a-time
            // Gray-code order immediately after `(1.0, 1.0)`.
            if is_dual_engine_morph_control(primary) {
                for &(p_tag, p_val, s_tag, s_val) in &[
                    ("max", 1.0_f32, "min", 0.0_f32),
                    ("min", 0.0_f32, "min", 0.0_f32),
                    ("min", 0.0_f32, "max", 1.0_f32),
                ] {
                    let mut morph_coords = default_ranges.clone();
                    morph_coords.insert(primary.id.clone(), p_val);
                    morph_coords.insert(secondary.id.clone(), s_val);
                    let morph_id = format!(
                        "stem_corner_{}_{}_{}_{}",
                        primary.id, p_tag, secondary.id, s_tag
                    );
                    steps.push(make_step(
                        model,
                        morph_id,
                        CaptureTier::CornerAnchor,
                        morph_coords,
                        default_switches.clone(),
                        None,
                    ));
                }
            }
        }

        // 3a. Cascaded gain/drive saturation corners (e.g. `Gain=max` × `Drive=max`)
        let secondary_id = secondary_opt.map(|s| s.id.as_str());
        for sec_gain in &model.ranges {
            if sec_gain.id != primary.id
                && Some(sec_gain.id.as_str()) != secondary_id
                && is_gain_like_control(sec_gain)
            {
                let mut gd_coords = default_ranges.clone();
                gd_coords.insert(primary.id.clone(), 1.0);
                gd_coords.insert(sec_gain.id.clone(), 1.0);
                let gd_id = format!("stem_corner_{}_max_{}_max", primary.id, sec_gain.id);
                steps.push(make_step(
                    model,
                    gd_id,
                    CaptureTier::CornerAnchor,
                    gd_coords,
                    default_switches.clone(),
                    None,
                ));
            }
        }
    }

    // 3b. Parametric EQ (band × freq) 2x2 interaction corners in Gray-code order:
    //     (min, min) -> (min, max) -> (max, max) -> (max, min) so only 1 knob changes per step.
    if include_corners {
        const GRAY_2X2: &[(&str, f32, &str, f32)] = &[
            ("min", 0.0, "min", 0.0),
            ("min", 0.0, "max", 1.0),
            ("max", 1.0, "max", 1.0),
            ("max", 1.0, "min", 0.0),
        ];
        for (band_ctrl, freq_ctrl) in select_parametric_freq_pairs(&model.ranges) {
            let band_default = default_ranges.get(&band_ctrl.id).copied().unwrap_or(0.5);
            let freq_default = default_ranges.get(&freq_ctrl.id).copied().unwrap_or(0.5);
            for &(b_tag, b_val, f_tag, f_val) in GRAY_2X2 {
                if (b_val - band_default).abs() < 1e-4 || (f_val - freq_default).abs() < 1e-4 {
                    continue;
                }
                let mut eq_corner_coords = default_ranges.clone();
                eq_corner_coords.insert(band_ctrl.id.clone(), b_val);
                eq_corner_coords.insert(freq_ctrl.id.clone(), f_val);
                let eq_corner_id = format!(
                    "stem_corner_{}_{}_{}_{}",
                    band_ctrl.id, b_tag, freq_ctrl.id, f_tag
                );
                steps.push(make_step(
                    model,
                    eq_corner_id,
                    CaptureTier::CornerAnchor,
                    eq_corner_coords,
                    default_switches.clone(),
                    None,
                ));
            }
        }
    }

    // 4. Switch throws + Switch x Primary-Gain min/max interaction corners
    //    Grouped by Primary-Gain level (default -> min -> max) so `Gain` and `Output Clean Level`
    //    only change twice across all switches rather than thrashing on every switch option.
    for sw in &model.switches {
        for opt in &sw.options {
            if opt.index == sw.default_index {
                continue;
            }
            let mut sw_map = default_switches.clone();
            sw_map.insert(sw.id.clone(), opt.index);
            let stem_id = format!("stem_switch_{}_{}", sw.id, opt.index);
            steps.push(make_step(
                model,
                stem_id,
                CaptureTier::SwitchState,
                default_ranges.clone(),
                sw_map,
                None,
            ));
        }
    }
    if include_corners && let Some((primary, _)) = corner_pair {
        let primary_default = default_ranges.get(&primary.id).copied().unwrap_or(0.5);
        for (tag, norm_val) in [("min", 0.0_f32), ("max", 1.0_f32)] {
            if (norm_val - primary_default).abs() < 1e-4 {
                continue;
            }
            for sw in &model.switches {
                for opt in &sw.options {
                    if opt.index == sw.default_index {
                        continue;
                    }
                    let mut sw_map = default_switches.clone();
                    sw_map.insert(sw.id.clone(), opt.index);
                    let mut sw_corner_coords = default_ranges.clone();
                    sw_corner_coords.insert(primary.id.clone(), norm_val);
                    let sw_corner_id = format!(
                        "stem_corner_switch_{}_{}_{}_{}",
                        sw.id, opt.index, primary.id, tag
                    );
                    steps.push(make_step(
                        model,
                        sw_corner_id,
                        CaptureTier::CornerAnchor,
                        sw_corner_coords,
                        sw_map,
                        None,
                    ));
                }
            }
        }
    }

    // 5. `.dnam-ckpt` Audit Report prescribed stems
    for (idx, (ranges_p, switches_p, reason)) in prescriptions.iter().enumerate() {
        let stem_id = format!("stem_prescribed_{idx:02}");
        steps.push(make_step(
            model,
            stem_id,
            CaptureTier::Prescribed,
            ranges_p.clone(),
            switches_p.clone(),
            Some(reason.as_str()),
        ));
    }

    CapturePlan {
        model_id: model.model_id.clone(),
        steps,
    }
}

/// Builds the deterministic Star + Switch + Corner capture schedule for `model`.
pub fn build_capture_plan(model: &ModelManifest, include_corners: bool) -> CapturePlan {
    build_capture_plan_with_prescriptions(model, include_corners, &[])
}

/// Single control delta row (`Control | Previous | Target Setting | Action`) for interactive capture prompts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlDelta {
    /// Control identifier (`id`).
    pub control_id: String,
    /// Human-readable control label (`label`).
    pub label: String,
    /// Previous dial/switch setting (or `None` on the first stem).
    pub previous_value: Option<String>,
    /// Target dial/switch setting for the current stem.
    pub target_value: String,
    /// Whether the user needs to turn or flip this control (`true` = `TURN / FLIP`, `false` = `Unchanged`).
    pub changed: bool,
}

/// Computes per-control `(previous_value, target_value, changed)` deltas between the previous
/// stem and the next planned stem for interactive hardware capture prompts.
pub fn compute_control_deltas(
    model: &ModelManifest,
    prev_ranges: Option<&BTreeMap<String, f32>>,
    prev_switches: Option<&BTreeMap<String, usize>>,
    next_ranges: &BTreeMap<String, f32>,
    next_switches: &BTreeMap<String, usize>,
) -> Vec<ControlDelta> {
    let mut deltas = Vec::new();
    for r in &model.ranges {
        let next_u = next_ranges.get(&r.id).copied().unwrap_or(0.5);
        let next_dial = r.min + next_u * (r.max - r.min);
        let target_value = format!("{next_dial:.1}");
        let previous_value = prev_ranges
            .and_then(|pr| pr.get(&r.id).copied())
            .map(|prev_u| {
                let prev_dial = r.min + prev_u * (r.max - r.min);
                format!("{prev_dial:.1}")
            });
        let changed = previous_value.as_deref() != Some(target_value.as_str());
        deltas.push(ControlDelta {
            control_id: r.id.clone(),
            label: r.label.clone(),
            previous_value,
            target_value,
            changed,
        });
    }
    for s in &model.switches {
        let next_idx = next_switches.get(&s.id).copied().unwrap_or(s.default_index);
        let target_value = s
            .options
            .iter()
            .find(|o| o.index == next_idx)
            .map(|o| o.label.clone())
            .unwrap_or_else(|| next_idx.to_string());
        let previous_value = prev_switches
            .and_then(|ps| ps.get(&s.id).copied())
            .map(|prev_idx| {
                s.options
                    .iter()
                    .find(|o| o.index == prev_idx)
                    .map(|o| o.label.clone())
                    .unwrap_or_else(|| prev_idx.to_string())
            });
        let changed = previous_value.as_deref() != Some(target_value.as_str());
        deltas.push(ControlDelta {
            control_id: s.id.clone(),
            label: s.label.clone(),
            previous_value,
            target_value,
            changed,
        });
    }
    deltas
}

fn is_gain_like_control(r: &ControlRangeSpec) -> bool {
    let id = r.id.to_ascii_lowercase();
    let label = r.label.to_ascii_lowercase();
    const KEYWORDS: &[&str] = &[
        "gain",
        "drive",
        "dist",
        "fuzz",
        "duality",
        "morph",
        "overdrive",
        "preamp",
        "saturat",
        "boost",
        "comp",
        "sustain",
    ];
    KEYWORDS
        .iter()
        .any(|kw| id.contains(kw) || label.contains(kw))
}

fn is_dual_engine_morph_control(r: &ControlRangeSpec) -> bool {
    let id = r.id.to_ascii_lowercase();
    let label = r.label.to_ascii_lowercase();
    id == "duality"
        || id == "mod"
        || id == "morph"
        || label.contains("duality")
        || label.contains("morph")
}

fn is_tone_body_control(r: &ControlRangeSpec) -> bool {
    let id = r.id.to_ascii_lowercase();
    let label = r.label.to_ascii_lowercase();
    const KEYWORDS: &[&str] = &[
        "drive", "gain", "bass", "low", "body", "depth", "reson", "tone", "filter", "freq",
    ];
    KEYWORDS
        .iter()
        .any(|kw| id.contains(kw) || label.contains(kw))
}

fn select_corner_range_pair(
    ranges: &[ControlRangeSpec],
) -> Option<(&ControlRangeSpec, Option<&ControlRangeSpec>)> {
    let primary = ranges
        .iter()
        .find(|r| is_gain_like_control(r))
        .or_else(|| ranges.first())?;
    let secondary = ranges
        .iter()
        .filter(|r| r.id != primary.id)
        .find(|r| is_tone_body_control(r))
        .or_else(|| ranges.iter().find(|r| r.id != primary.id));
    Some((primary, secondary))
}

/// Identifies parametric EQ `(band_level, band_freq)` control pairs (such as `mid` + `mid_freq`,
/// `low_mid` + `low_mid_freq`, `bass` + `bass_freq`, etc.) so 2x2 interaction corner stems can be
/// scheduled.
fn select_parametric_freq_pairs(
    ranges: &[ControlRangeSpec],
) -> Vec<(&ControlRangeSpec, &ControlRangeSpec)> {
    let mut pairs = Vec::new();
    for freq_ctrl in ranges {
        let freq_id = freq_ctrl.id.to_ascii_lowercase();
        let prefix = freq_id
            .strip_suffix("_freq")
            .or_else(|| freq_id.strip_suffix("_frequency"))
            .or_else(|| freq_id.strip_suffix("_hz"));
        if let Some(band_id) = prefix
            && let Some(band_ctrl) = ranges
                .iter()
                .find(|r| r.id.eq_ignore_ascii_case(band_id) && r.id != freq_ctrl.id)
        {
            pairs.push((band_ctrl, freq_ctrl));
        }
    }
    pairs
}

fn make_step(
    model: &ModelManifest,
    stem_id: String,
    tier: CaptureTier,
    normalized_ranges: BTreeMap<String, f32>,
    switches: BTreeMap<String, usize>,
    custom_reason: Option<&str>,
) -> CapturePlanStep {
    let mut parts = Vec::new();
    for r in &model.ranges {
        let u = normalized_ranges.get(&r.id).copied().unwrap_or(0.5);
        let dial = r.min + u * (r.max - r.min);
        parts.push(format!("{}={:.1}", r.label, dial));
    }
    for s in &model.switches {
        let idx = switches.get(&s.id).copied().unwrap_or(s.default_index);
        let opt_label = s
            .options
            .iter()
            .find(|o| o.index == idx)
            .map(|o| o.label.as_str())
            .unwrap_or("Default");
        parts.push(format!("{}={}", s.label, opt_label));
    }
    let prompt = if let Some(reason) = custom_reason {
        format!("Set {} ({reason})", parts.join(", "))
    } else if parts.is_empty() {
        "Capture unity reference stem".to_string()
    } else {
        format!("Set {}", parts.join(", "))
    };
    let already_captured = model.stems.iter().any(|s| s.stem_id == stem_id);

    CapturePlanStep {
        stem_id,
        tier,
        normalized_ranges,
        switches,
        prompt,
        already_captured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::manifest::{
        CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    };
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::schema::{AuditReport, ControlSchemaDef, DnamJsonHeader};
    use proptest::prelude::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_build_capture_plan_normalized_bounds_and_tiers(
            n_ranges in 0_usize..=5,
            n_switches in 0_usize..=3,
            include_corners in proptest::bool::ANY,
            default_fracs in proptest::collection::vec(0.0_f32..=1.0_f32, 5),
        ) {
            let ranges: Vec<ControlRangeSpec> = (0..n_ranges)
                .map(|i| {
                    let def = default_fracs[i] * 10.0;
                    ControlRangeSpec {
                        id: format!("r_{i}"),
                        label: format!("R{i}"),
                        min: 0.0,
                        max: 10.0,
                        default_value: def,
                        unit: None,
                    }
                })
                .collect();
            let switches: Vec<ControlSwitchSpec> = (0..n_switches)
                .map(|i| ControlSwitchSpec {
                    id: format!("sw_{i}"),
                    label: format!("SW{i}"),
                    options: vec![
                        ControlSwitchOptionSpec { index: 0, label: "a".to_string() },
                        ControlSwitchOptionSpec { index: 1, label: "b".to_string() },
                    ],
                    default_index: 0,
                })
                .collect();
            let model = ModelManifest {
                model_id: "prop_m".to_string(),
                title: "Prop".to_string(),
                tier: "Full".to_string(),
                description: String::new(),
                topology: "auto".to_string(),
                output_mode: "stereo".to_string(),
                capture: CaptureSettings::default(),
                ranges,
                switches,
                presets: Vec::new(),
                stems: Vec::new(),
            };

            let plan = build_capture_plan(&model, include_corners);
            prop_assert!(!plan.steps.is_empty());
            prop_assert_eq!(plan.steps[0].tier, CaptureTier::StarCenter);
            for step in &plan.steps {
                prop_assert_eq!(step.normalized_ranges.len(), n_ranges);
                prop_assert_eq!(step.switches.len(), n_switches);
                for &u in step.normalized_ranges.values() {
                    prop_assert!((0.0..=1.0).contains(&u));
                }
                for &idx in step.switches.values() {
                    prop_assert!(idx < 2);
                }
            }
            if include_corners && n_ranges >= 2 {
                prop_assert!(plan.steps.iter().any(|s| s.tier == CaptureTier::CornerAnchor));
            }
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_planner_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn sample_model() -> ModelManifest {
        ModelManifest {
            model_id: "amp".to_string(),
            title: "Amp".to_string(),
            tier: "Full".to_string(),
            description: String::new(),
            topology: "auto".to_string(),
            output_mode: "stereo".to_string(),
            capture: CaptureSettings::default(),
            ranges: vec![
                ControlRangeSpec {
                    id: "gain".to_string(),
                    label: "Gain".to_string(),
                    min: 0.0,
                    max: 10.0,
                    default_value: 5.0,
                    unit: None,
                },
                ControlRangeSpec {
                    id: "tone".to_string(),
                    label: "Tone".to_string(),
                    min: 0.0,
                    max: 10.0,
                    default_value: 10.0,
                    unit: None,
                },
            ],
            switches: vec![ControlSwitchSpec {
                id: "bright".to_string(),
                label: "Bright".to_string(),
                options: vec![
                    ControlSwitchOptionSpec {
                        index: 0,
                        label: "off".to_string(),
                    },
                    ControlSwitchOptionSpec {
                        index: 1,
                        label: "on".to_string(),
                    },
                ],
                default_index: 0,
            }],
            presets: Vec::new(),
            stems: Vec::new(),
        }
    }

    #[test]
    fn planner_builds_star_switch_corner_and_deltas() {
        let empty_model = ModelManifest {
            model_id: "unity".to_string(),
            title: "Unity".to_string(),
            tier: "Full".to_string(),
            description: String::new(),
            topology: "auto".to_string(),
            output_mode: "stereo".to_string(),
            capture: CaptureSettings::default(),
            ranges: Vec::new(),
            switches: Vec::new(),
            presets: Vec::new(),
            stems: Vec::new(),
        };
        let unity_plan = build_capture_plan(&empty_model, false);
        assert_eq!(unity_plan.steps.len(), 1);
        assert_eq!(unity_plan.steps[0].prompt, "Capture unity reference stem");

        let model = sample_model();
        let plan = build_capture_plan(&model, true);
        assert!(plan.steps.iter().any(|s| s.tier == CaptureTier::StarCenter));
        assert!(plan.steps.iter().any(|s| s.tier == CaptureTier::StarAxis));
        assert!(
            plan.steps
                .iter()
                .any(|s| s.tier == CaptureTier::SwitchState)
        );
        assert!(
            plan.steps
                .iter()
                .any(|s| s.tier == CaptureTier::CornerAnchor)
        );

        let d0 = compute_control_deltas(
            &model,
            None,
            None,
            &plan.steps[0].normalized_ranges,
            &plan.steps[0].switches,
        );
        assert_eq!(d0.len(), 3);
        assert!(d0.iter().all(|d| d.changed && d.previous_value.is_none()));

        let d1 = compute_control_deltas(
            &model,
            Some(&plan.steps[0].normalized_ranges),
            Some(&plan.steps[0].switches),
            &plan.steps[1].normalized_ranges,
            &plan.steps[1].switches,
        );
        assert!(d1.iter().any(|d| d.changed));
        assert!(d1.iter().any(|d| !d.changed));
    }

    #[test]
    fn extract_checkpoint_prescriptions_parses_clamp_warning_and_audit_report() {
        let dir = temp_dir("presc");
        let model = sample_model();
        let missing = dir.join("missing.dnam-ckpt");
        assert!(extract_checkpoint_prescriptions(&missing, &model).is_empty());

        let bad = dir.join("bad.dnam-ckpt");
        fs::write(&bad, b"not a container").expect("write bad");
        assert!(extract_checkpoint_prescriptions(&bad, &model).is_empty());

        let mut header = DnamJsonHeader::default_reference(false);
        let c1: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": "gain",
            "name": "Gain",
            "type": "range",
            "declared_range": [0.0, 10.0],
            "safe_clamped_range": [0.0, 5.0],
            "default": 5.0,
            "clamp_warning": "Capture gain=10.0"
        }))
        .expect("c1");
        header.controls_schema = vec![c1];
        header.audit_report = Some(AuditReport {
            phase1_clamps: vec!["CLAMP-RANGE".to_string()],
            phase2_rail_warnings: Vec::new(),
            phase3_clamps: Vec::new(),
            prescriptions: vec!["Capture stem at (gain=8.0, bright='on')".to_string()],
        });

        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let bytes =
            build_dnam_container(&mut header, &arena, &arena, false).expect("build container");
        let ckpt_path = dir.join("test.dnam-ckpt");
        fs::write(&ckpt_path, &bytes).expect("write ckpt");

        let presc = extract_checkpoint_prescriptions(&ckpt_path, &model);
        assert_eq!(presc.len(), 2);
        assert!((presc[0].0["gain"] - 1.0).abs() < 1e-5);
        assert!((presc[1].0["gain"] - 0.8).abs() < 1e-5);
        assert_eq!(presc[1].1["bright"], 1);

        let plan_with_presc = build_capture_plan_with_prescriptions(&model, false, &presc);
        assert!(
            plan_with_presc
                .steps
                .iter()
                .any(|s| s.tier == CaptureTier::Prescribed)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn planner_schedules_parametric_eq_2x2_corners() {
        let mut model = sample_model();
        model.ranges.push(ControlRangeSpec {
            id: "mid".to_string(),
            label: "Mid".to_string(),
            min: 0.0,
            max: 10.0,
            default_value: 5.0,
            unit: None,
        });
        model.ranges.push(ControlRangeSpec {
            id: "mid_freq".to_string(),
            label: "Mid Freq".to_string(),
            min: 57.0,
            max: 4000.0,
            default_value: 250.0,
            unit: Some("Hz".to_string()),
        });
        let plan = build_capture_plan(&model, true);
        for expected_id in [
            "stem_corner_mid_min_mid_freq_min",
            "stem_corner_mid_min_mid_freq_max",
            "stem_corner_mid_max_mid_freq_min",
            "stem_corner_mid_max_mid_freq_max",
        ] {
            assert!(
                plan.steps.iter().any(|s| s.stem_id == expected_id),
                "missing parametric EQ corner stem {expected_id}"
            );
        }
    }

    #[test]
    fn planner_schedules_dual_engine_morph_2x2_corners() {
        let model = ModelManifest {
            model_id: "duality_fuzz".to_string(),
            title: "Duality Fuzz".to_string(),
            tier: "Full".to_string(),
            description: String::new(),
            output_mode: "stereo".to_string(),
            topology: "auto".to_string(),
            capture: CaptureSettings::default(),
            ranges: vec![
                ControlRangeSpec {
                    id: "duality".to_string(),
                    label: "Duality".to_string(),
                    min: 0.0,
                    max: 10.0,
                    default_value: 5.0,
                    unit: None,
                },
                ControlRangeSpec {
                    id: "filter".to_string(),
                    label: "Filter".to_string(),
                    min: 0.0,
                    max: 10.0,
                    default_value: 5.0,
                    unit: None,
                },
            ],
            switches: Vec::new(),
            presets: Vec::new(),
            stems: Vec::new(),
        };
        let plan = build_capture_plan(&model, true);
        assert_eq!(plan.steps.len(), 9);
        for expected_id in [
            "stem_corner_high_pair",
            "stem_corner_duality_max_filter_min",
            "stem_corner_duality_min_filter_min",
            "stem_corner_duality_min_filter_max",
        ] {
            assert!(
                plan.steps.iter().any(|s| s.stem_id == expected_id),
                "missing dual-engine morph corner stem {expected_id}"
            );
        }
    }
}
