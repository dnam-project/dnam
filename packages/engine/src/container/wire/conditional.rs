//! Structured Conditional Control Rules & Midpoint-Anchored UI Power-Skew Hints
//! (`ref[container.dnam.conditional-rules]`, `ref[container.dnam.metadata-ui-hints]`).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::Value;

use crate::schema::{DnamError, SchemaOptionExt, cold_ln_f64, cold_pow_f64, reject_if};

/// Schema definition of a discrete switch control for selector validation (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchControlSpec {
    pub control_id: String,
    pub options: Vec<String>,
    pub active_options: Vec<String>,
}

/// Validated conditional bound rule (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredConditionalBound {
    pub when: Vec<(String, String)>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Validated conditional calibration curve rule (`ref[container.dnam.conditional-rules]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredConditionalCurve {
    pub when: Vec<(String, String)>,
    pub anchors: [f64; 3],
}

fn parse_structured_selector(
    when_val: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<(String, String)>, DnamError> {
    let arr = when_val.as_array().or_schema_err("invalid conjunction")?;
    reject_if(arr.is_empty(), "invalid conjunction")?;
    let mut out = Vec::with_capacity(arr.len());
    let mut prev_idx: Option<usize> = None;
    for item in arr {
        let obj = item.as_object().or_schema_err("invalid selector object")?;
        reject_if(
            obj.len() != 2 || !obj.contains_key("control_id") || !obj.contains_key("option_id"),
            "invalid selector object",
        )?;
        let cid = obj.get("control_id").and_then(Value::as_str).unwrap_or("");
        let oid = obj.get("option_id").and_then(Value::as_str).unwrap_or("");
        reject_if(
            cid.is_empty() || oid.is_empty(),
            "invalid control or option identity",
        )?;
        let sw_idx = switches
            .iter()
            .position(|s| s.control_id == cid)
            .or_schema_err("unknown switch")?;
        if let Some(prev) = prev_idx
            && sw_idx <= prev
        {
            return Err(DnamError::SchemaValidationError(
                "duplicate or unordered switch".to_string(),
            ));
        }
        reject_if(
            !switches[sw_idx].options.iter().any(|o| o == oid),
            "unknown option",
        )?;
        out.push((cid.to_string(), oid.to_string()));
        prev_idx = Some(sw_idx);
    }
    Ok(out)
}

/// Parses and validates a `conditional_bounds` array (`ref[container.dnam.conditional-rules]`).
pub fn parse_structured_conditional_bounds(
    value: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<StructuredConditionalBound>, DnamError> {
    let arr = value.as_array().or_schema_err("invalid bounds array")?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item.as_object().or_schema_err("invalid bound object")?;
        for k in obj.keys() {
            reject_if(
                !matches!(k.as_str(), "when" | "min" | "max"),
                "invalid bound object",
            )?;
        }
        let when_val = obj.get("when").or_schema_err("invalid bound object")?;
        let has_min = obj.contains_key("min");
        let has_max = obj.contains_key("max");
        reject_if(!has_min && !has_max, "invalid bound object")?;
        let when = parse_structured_selector(when_val, switches)?;
        reject_if(!seen.insert(when.clone()), "duplicate conjunction")?;
        let min = if let Some(m) = obj.get("min") {
            let v = m.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            Some(v)
        } else {
            None
        };
        let max = if let Some(m) = obj.get("max") {
            let v = m.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            Some(v)
        } else {
            None
        };
        if let (Some(lo), Some(hi)) = (min, max)
            && lo > hi
        {
            return Err(DnamError::SchemaValidationError(
                "reversed bound".to_string(),
            ));
        }
        out.push(StructuredConditionalBound { when, min, max });
    }
    Ok(out)
}

/// Parses and validates a `conditional_physical_anchors` array (`ref[container.dnam.conditional-rules]`).
pub fn parse_structured_conditional_curves(
    value: &Value,
    switches: &[SwitchControlSpec],
) -> Result<Vec<StructuredConditionalCurve>, DnamError> {
    let arr = value
        .as_array()
        .or_schema_err("invalid calibration array")?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item
            .as_object()
            .or_schema_err("invalid calibration object")?;
        reject_if(
            obj.len() != 2 || !obj.contains_key("when") || !obj.contains_key("anchors"),
            "invalid calibration object",
        )?;
        let when = parse_structured_selector(obj.get("when").unwrap(), switches)?;
        reject_if(!seen.insert(when.clone()), "duplicate conjunction")?;
        let anchors_arr = obj
            .get("anchors")
            .and_then(Value::as_array)
            .or_schema_err("invalid anchors")?;
        reject_if(anchors_arr.len() != 3, "invalid anchors")?;
        let mut anchors = [0.0f64; 3];
        for (i, a) in anchors_arr.iter().enumerate() {
            let v = a.as_f64().or_schema_err("invalid number")?;
            reject_if(!v.is_finite(), "invalid number")?;
            anchors[i] = v;
        }
        out.push(StructuredConditionalCurve { when, anchors });
    }
    Ok(out)
}

fn evaluate_selectors(
    selectors: &[&[(String, String)]],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
) -> Result<Vec<bool>, DnamError> {
    for when in selectors {
        for (cid, _) in *when {
            let eff_opt = effective
                .get(cid)
                .or_schema_err("missing effective switch state")?;
            let sw = switches
                .iter()
                .find(|s| &s.control_id == cid)
                .or_schema_err("unknown switch")?;
            reject_if(
                !sw.active_options.iter().any(|o| o == eff_opt),
                "inactive effective option",
            )?;
        }
    }
    Ok(selectors
        .iter()
        .map(|when| {
            when.iter()
                .all(|(cid, oid)| effective.get(cid).is_some_and(|v| v == oid))
        })
        .collect())
}

/// Evaluates the intersected safe interval `(lower, upper)` across all matching `conditional_bounds`
/// (`ref[container.dnam.conditional-rules]`).
pub fn resolve_conditional_safe_interval(
    base: (f64, f64),
    rules: &[StructuredConditionalBound],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
) -> Result<(f64, f64), DnamError> {
    let selectors: Vec<&[(String, String)]> = rules.iter().map(|r| r.when.as_slice()).collect();
    let matched = evaluate_selectors(&selectors, switches, effective)?;
    let (mut lower, mut upper) = base;
    for (rule, &sel) in rules.iter().zip(matched.iter()) {
        if sel {
            if let Some(lo) = rule.min {
                lower = lower.max(lo);
            }
            if let Some(hi) = rule.max {
                upper = upper.min(hi);
            }
        }
    }
    reject_if(lower > upper, "empty safe intersection")?;
    Ok((lower, upper))
}

/// Selects the active calibration anchors with greatest selector specificity (`ref[container.dnam.conditional-rules]`),
/// returning an error on a specificity tie.
pub fn select_conditional_calibration_curve(
    rules: &[StructuredConditionalCurve],
    switches: &[SwitchControlSpec],
    effective: &BTreeMap<String, String>,
    default_anchors: Option<[f64; 3]>,
) -> Result<[f64; 3], DnamError> {
    let selectors: Vec<&[(String, String)]> = rules.iter().map(|r| r.when.as_slice()).collect();
    let matched = evaluate_selectors(&selectors, switches, effective)?;
    let applicable: Vec<&StructuredConditionalCurve> = rules
        .iter()
        .zip(matched.iter())
        .filter_map(|(r, &m)| m.then_some(r))
        .collect();
    if applicable.is_empty() {
        return default_anchors.ok_or_else(|| {
            DnamError::SchemaValidationError("calibration unavailable".to_string())
        });
    }
    let max_spec = applicable.iter().map(|r| r.when.len()).max().unwrap();
    let best: Vec<&&StructuredConditionalCurve> = applicable
        .iter()
        .filter(|r| r.when.len() == max_spec)
        .collect();
    reject_if(best.len() != 1, "ambiguous calibration")?;
    Ok(best[0].anchors)
}

/// Evaluates the midpoint-anchored UI power-skew exponent $\gamma_{\mathrm{skew}}$, forward map $x(t)$,
/// and inverse map $t(x)$ (`ref[container.dnam.metadata-ui-hints]`).
pub fn eval_ui_midpoint_power_skew(
    x_min: f64,
    x_mid: f64,
    x_max: f64,
    t_norm_0_to_1: f64,
) -> Option<(f64, f64, f64)> {
    if !(x_min < x_mid && x_mid < x_max && (0.0..=1.0).contains(&t_norm_0_to_1)) {
        return None;
    }
    let ratio = (x_mid - x_min) / (x_max - x_min);
    let gamma = cold_ln_f64(0.5) / cold_ln_f64(ratio);
    let x_val = if t_norm_0_to_1 <= 0.0 {
        x_min
    } else if t_norm_0_to_1 >= 1.0 {
        x_max
    } else {
        x_min + (x_max - x_min) * cold_pow_f64(t_norm_0_to_1, 1.0 / gamma)
    };
    let t_roundtrip = if x_val <= x_min {
        0.0
    } else if x_val >= x_max {
        1.0
    } else {
        cold_pow_f64((x_val - x_min) / (x_max - x_min), gamma)
    };
    Some((gamma, x_val, t_roundtrip))
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::container::{
        build_v2_open_dnam_container, strip_v2_open_dnam_container, validate_v2_open_dnam_layout,
    };
    use alloc::vec;
    use proptest::prelude::*;
    use serde_json::json;

    /// verify[container.dnam.conditional-rules]
    /// verify[container.dnam.metadata-ui-hints]
    /// Verifies structured switch conjunctions (`when: [{"control_id", "option_id"}]`), safe-interval
    /// intersection, specificity-based calibration curve selection, and midpoint power-skew UI hints.
    #[test]
    fn test_structured_conditional_rules_and_ui_power_skew() {
        let switches = vec![
            SwitchControlSpec {
                control_id: "mode=variant".to_string(),
                options: vec!["x".to_string(), "off".to_string()],
                active_options: vec!["x".to_string(), "off".to_string()],
            },
            SwitchControlSpec {
                control_id: "mode".to_string(),
                options: vec![
                    "variant=x".to_string(),
                    "off".to_string(),
                    "masked".to_string(),
                ],
                active_options: vec!["variant=x".to_string(), "off".to_string()],
            },
        ];

        let bounds_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "max": 8.0 },
            { "when": [{"control_id": "mode", "option_id": "variant=x"}], "min": 2.0 },
            {
                "when": [
                    {"control_id": "mode=variant", "option_id": "x"},
                    {"control_id": "mode", "option_id": "variant=x"}
                ],
                "min": 4.0,
                "max": 6.0
            }
        ]);
        let rules = parse_structured_conditional_bounds(&bounds_json, &switches).unwrap();
        let eff = BTreeMap::from([
            ("mode=variant".to_string(), "x".to_string()),
            ("mode".to_string(), "variant=x".to_string()),
        ]);
        assert_eq!(
            resolve_conditional_safe_interval((0.0, 10.0), &rules, &switches, &eff).unwrap(),
            (4.0, 6.0)
        );

        // Specificity selection and ambiguity detection on curves
        let curves_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "anchors": [1.0, 2.0, 3.0] },
            {
                "when": [
                    {"control_id": "mode=variant", "option_id": "x"},
                    {"control_id": "mode", "option_id": "variant=x"}
                ],
                "anchors": [10.0, 20.0, 30.0]
            }
        ]);
        let curves = parse_structured_conditional_curves(&curves_json, &switches).unwrap();
        assert_eq!(
            select_conditional_calibration_curve(&curves, &switches, &eff, None).unwrap(),
            [10.0, 20.0, 30.0]
        );

        let tie_json = json!([
            { "when": [{"control_id": "mode=variant", "option_id": "x"}], "anchors": [1.0, 2.0, 3.0] },
            { "when": [{"control_id": "mode", "option_id": "variant=x"}], "anchors": [4.0, 5.0, 6.0] }
        ]);
        let tie_curves = parse_structured_conditional_curves(&tie_json, &switches).unwrap();
        assert!(select_conditional_calibration_curve(&tie_curves, &switches, &eff, None).is_err());

        // Midpoint-anchored UI power-skew law: x(0.5) == x_mid
        let (gamma, x_at_half, t_back) =
            eval_ui_midpoint_power_skew(20.0, 200.0, 2020.0, 0.5).unwrap();
        assert!(gamma > 0.0);
        assert!((x_at_half - 200.0).abs() < 1e-9);
        assert!((t_back - 0.5).abs() < 1e-12);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[container.dnam.open-layout]
        /// verify[container.dnam.mutable-frame]
        /// verify[container.dnam.metadata-ui-hints]
        #[test]
        fn prop_v2_container_stripping_dmu1_and_ui_power_skew_bijection(
            cls in prop::array::uniform4(1usize..=4usize),
            target_tier in 0u8..=3u8,
            x_min in -100.0_f64..0.0_f64,
            span in 10.0_f64..500.0_f64,
            mid_frac in 0.1_f64..0.9_f64,
            t_norm in 0.0_f64..=1.0_f64,
        ) {
            let s0 = vec![0x11u8; cls[0] * 64];
            let s1 = vec![0x22u8; cls[1] * 64];
            let s2 = vec![0x33u8; cls[2] * 64];
            let s3 = vec![0x44u8; cls[3] * 64];
            let container = build_v2_open_dnam_container(
                b"{\"schema_version\":\"2.0\"}",
                &[0u8; 64],
                b"DAU1-TEST",
                &[&s0, &s1, &s2, &s3],
                None,
                0,
            ).unwrap();
            let stripped = strip_v2_open_dnam_container(&container, target_tier).unwrap();
            let layout = validate_v2_open_dnam_layout(&stripped).unwrap();
            prop_assert_eq!(layout.retained_tier, target_tier);

            let x_max = x_min + span;
            let x_mid = x_min + mid_frac * span;
            let (_, phys, back) = eval_ui_midpoint_power_skew(x_min, x_mid, x_max, t_norm).unwrap();
            prop_assert!(phys >= x_min && phys <= x_max);
            prop_assert!((back - t_norm).abs() < 1e-11);
        }
    }
}
