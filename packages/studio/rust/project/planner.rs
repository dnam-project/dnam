//! Star + Lobatto Axis + Switch + Corner + `.dnam-ckpt` Prescription capture schedule planner.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dnam_engine::container::DnamContainerView;
use dnam_engine::provenance::DefaultCryptoProvider;
use serde::{Deserialize, Serialize};

use crate::project::manifest::ModelManifest;

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

    // 2. Lobatto axis sweeps (min=0.0, max=1.0) per range control
    for range in &model.ranges {
        for (tag, norm_val) in [("min", 0.0_f32), ("max", 1.0_f32)] {
            if (norm_val - default_ranges.get(&range.id).copied().unwrap_or(0.5)).abs() < 0.05 {
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

    // 3. Switch throws
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

    // 4. Optional high-corner anchor when >= 2 continuous controls exist
    if include_corners && model.ranges.len() >= 2 {
        let mut corner_coords = default_ranges;
        for r in model.ranges.iter().take(2) {
            corner_coords.insert(r.id.clone(), 1.0);
        }
        steps.push(make_step(
            model,
            "stem_corner_high_pair".to_string(),
            CaptureTier::CornerAnchor,
            corner_coords,
            default_switches,
            None,
        ));
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
