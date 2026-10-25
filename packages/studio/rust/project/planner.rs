//! Star + Lobatto Axis + Switch + Corner capture schedule planner.

use std::collections::BTreeMap;

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
    /// Multi-knob high-gain corner anchor.
    CornerAnchor,
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

/// Builds the deterministic Star + Switch + Corner capture schedule for `model`.
pub fn build_capture_plan(model: &ModelManifest, include_corners: bool) -> CapturePlan {
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
        ));
    }

    CapturePlan {
        model_id: model.model_id.clone(),
        steps,
    }
}

fn make_step(
    model: &ModelManifest,
    stem_id: String,
    tier: CaptureTier,
    normalized_ranges: BTreeMap<String, f32>,
    switches: BTreeMap<String, usize>,
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
    let prompt = if parts.is_empty() {
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
