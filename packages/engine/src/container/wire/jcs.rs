//! Strict JSON Parsing & RFC 8785 JCS Header Projection (`ref[container.dnam.jcs-header-schema]`,
//! `ref[container.dnam.security-crossref]`, `ref[sec.jcs.immutable-header-digest]`).

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::Value;

use crate::schema::{DnamError, SchemaOptionExt, reject_if};

/// Returns `true` if `s` contains no Unicode noncharacters (`0xFDD0..=0xFDEF`, `..0xFFFE/0xFFFF`).
pub fn is_valid_jcs_unicode_text(s: &str) -> bool {
    for ch in s.chars() {
        let code = ch as u32;
        if (0xFDD0..=0xFDEF).contains(&code) || (code & 0xFFFF) >= 0xFFFE {
            return false;
        }
    }
    true
}

fn validate_json_value_unicode_and_finite(v: &Value) -> Result<(), DnamError> {
    match v {
        Value::Null | Value::Bool(_) => Ok(()),
        Value::Number(n) => {
            reject_if(
                n.as_f64().is_none_or(|f| !f.is_finite()),
                "non-finite number",
            )?;
            Ok(())
        }
        Value::String(s) => {
            reject_if(!is_valid_jcs_unicode_text(s), "invalid Unicode")?;
            Ok(())
        }
        Value::Array(arr) => {
            for item in arr {
                validate_json_value_unicode_and_finite(item)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, val) in map {
                reject_if(!is_valid_jcs_unicode_text(k), "invalid Unicode")?;
                validate_json_value_unicode_and_finite(val)?;
            }
            Ok(())
        }
    }
}

/// Checks raw JSON UTF-8 bytes for duplicate object keys (including `\uXXXX` escapes), BOM, and invalid
/// Unicode, and parses a top-level JSON object (`ref[container.dnam.jcs-header-schema]`).
pub fn parse_strict_json_object_bytes(data: &[u8]) -> Result<Value, DnamError> {
    reject_if(
        data.starts_with(&[0xEF, 0xBB, 0xBF]),
        "UTF-8 BOM is forbidden",
    )?;
    let text = core::str::from_utf8(data)
        .map_err(|_| DnamError::SchemaValidationError("invalid UTF-8 JSON".to_string()))?;
    check_no_duplicate_json_keys(text)?;
    let val: Value = serde_json::from_str(text)
        .map_err(|e| DnamError::SchemaValidationError(alloc::format!("invalid JSON: {e}")))?;
    validate_json_value_unicode_and_finite(&val)?;
    reject_if(!val.is_object(), "header must be an object")?;
    Ok(val)
}

fn check_no_duplicate_json_keys(text: &str) -> Result<(), DnamError> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut stack: Vec<(bool, BTreeSet<String>, bool)> = Vec::new(); // (is_object, seen_keys, expect_key)

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'{' => {
                stack.push((true, BTreeSet::new(), true));
                i += 1;
            }
            b'}' => {
                stack.pop();
                if let Some((true, _, expect_key)) = stack.last_mut() {
                    *expect_key = false;
                }
                i += 1;
            }
            b'[' => {
                stack.push((false, BTreeSet::new(), false));
                i += 1;
            }
            b']' => {
                stack.pop();
                if let Some((true, _, expect_key)) = stack.last_mut() {
                    *expect_key = false;
                }
                i += 1;
            }
            b',' => {
                if let Some((is_obj, _, expect_key)) = stack.last_mut()
                    && *is_obj
                {
                    *expect_key = true;
                }
                i += 1;
            }
            b':' => {
                if let Some((is_obj, _, expect_key)) = stack.last_mut()
                    && *is_obj
                {
                    *expect_key = false;
                }
                i += 1;
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                    } else if bytes[i] == b'"' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
                let raw_str = &text[start..i];
                if let Some((true, seen, expect_key)) = stack.last_mut()
                    && *expect_key
                {
                    let parsed_key: String = serde_json::from_str(raw_str).map_err(|_| {
                        DnamError::SchemaValidationError("invalid Unicode".to_string())
                    })?;
                    reject_if(!seen.insert(parsed_key), "duplicate object member")?;
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    Ok(())
}

/// Projects out the designated mutable members (`presets`, `user_metadata`, `metadata.signatures`,
/// `controls_schema[*].user_metadata`, `stems_registry[*].user_metadata`, and
/// `slimmable_tiers.{lite,full,pro,ultra}.prefix_crc32c`) from a parsed JSON header object
/// (`ref[container.dnam.jcs-header-schema]`, `ref[sec.jcs.immutable-header-digest]`).
pub fn project_immutable_header_value(header: &Value) -> Result<Value, DnamError> {
    let mut result = header.clone();
    let root = result
        .as_object_mut()
        .or_schema_err("header must be an object")?;
    root.remove("presets");
    root.remove("user_metadata");

    if let Some(meta) = root.get_mut("metadata") {
        let meta_obj = meta
            .as_object_mut()
            .or_schema_err("metadata must be an object")?;
        meta_obj.remove("signatures");
    }
    if let Some(ctrls) = root.get_mut("controls_schema") {
        let arr = ctrls
            .as_array_mut()
            .or_schema_err("controls_schema must be an array")?;
        for ctrl in arr {
            let c_obj = ctrl
                .as_object_mut()
                .or_schema_err("control must be an object")?;
            c_obj.remove("user_metadata");
        }
    }
    if let Some(stems) = root.get_mut("stems_registry")
        && !stems.is_null()
    {
        let arr = stems
            .as_array_mut()
            .or_schema_err("stems_registry must be an array")?;
        for stem in arr {
            let s_obj = stem
                .as_object_mut()
                .or_schema_err("stem must be an object")?;
            s_obj.remove("user_metadata");
        }
    }
    if let Some(tiers) = root.get_mut("slimmable_tiers") {
        let t_obj = tiers
            .as_object_mut()
            .or_schema_err("slimmable_tiers must be an object")?;
        for name in ["lite", "full", "pro", "ultra"] {
            if let Some(tier_entry) = t_obj.get_mut(name) {
                let te_obj = tier_entry
                    .as_object_mut()
                    .or_schema_err("tier entry must be an object")?;
                te_obj.remove("prefix_crc32c");
            }
        }
    }
    Ok(result)
}

/// Serializes a JSON `Value` using RFC 8785 JCS UTF-16 code-unit key ordering (`ref[container.dnam.jcs-header-schema]`,
/// `ref[sec.jcs.immutable-header-digest]`). When `reject_floats` is `true`, enforces the integer/string
/// JCS subset (integers within `±(2^53 - 1)`).
pub fn serialize_jcs_value(value: &Value, reject_floats: bool) -> Result<String, DnamError> {
    const MAX_SAFE_INT: i64 = (1i64 << 53) - 1;
    validate_json_value_unicode_and_finite(value)?;
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(true) => Ok("true".to_string()),
        Value::Bool(false) => Ok("false".to_string()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                reject_if(i.abs() > MAX_SAFE_INT, "integer outside reference subset")?;
                Ok(i.to_string())
            } else if let Some(u) = n.as_u64() {
                reject_if(
                    u > (MAX_SAFE_INT as u64),
                    "integer outside reference subset",
                )?;
                Ok(u.to_string())
            } else if reject_floats {
                Err(DnamError::SchemaValidationError(
                    "float outside reference subset".to_string(),
                ))
            } else {
                let f = n.as_f64().or_schema_err("non-finite number")?;
                if f.fract() == 0.0 && f.abs() <= (MAX_SAFE_INT as f64) {
                    Ok((f as i64).to_string())
                } else {
                    serde_json::to_string(&f).map_err(|e| {
                        DnamError::SchemaValidationError(alloc::format!("JCS float error: {e}"))
                    })
                }
            }
        }
        Value::String(s) => serde_json::to_string(s)
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("JCS string error: {e}"))),
        Value::Array(arr) => {
            let mut parts = Vec::with_capacity(arr.len());
            for item in arr {
                parts.push(serialize_jcs_value(item, reject_floats)?);
            }
            Ok(alloc::format!("[{}]", parts.join(",")))
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.encode_utf16().cmp(b.encode_utf16()));
            let mut parts = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let ks = serde_json::to_string(k).unwrap();
                let vs = serialize_jcs_value(v, reject_floats)?;
                parts.push(alloc::format!("{ks}:{vs}"));
            }
            Ok(alloc::format!("{{{}}}", parts.join(",")))
        }
    }
}

/// Validates that `wire_bytes` are already canonical projected JCS bytes without mutable members
/// (`ref[container.dnam.jcs-header-schema]`).
pub fn validate_canonical_jcs_header_bytes(
    wire_bytes: &[u8],
    reject_floats: bool,
) -> Result<Vec<u8>, DnamError> {
    let parsed = parse_strict_json_object_bytes(wire_bytes)?;
    let projected = project_immutable_header_value(&parsed)?;
    let canonical = serialize_jcs_value(&projected, reject_floats)?.into_bytes();
    reject_if(
        wire_bytes != canonical.as_slice(),
        "noncanonical or mutable header",
    )?;
    Ok(canonical)
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use serde_json::json;

    /// verify[container.dnam.jcs-header-schema]
    /// verify[container.dnam.security-crossref]
    /// Verifies RFC 8785 JCS UTF-16 code-unit key ordering, strict duplicate-key & noncharacter rejection,
    /// and exact mutable-member projection.
    #[test]
    fn test_jcs_immutable_header_projection_and_utf16_ordering() {
        let fixture = br#"{
      "version":"future-format",
      "metadata":{"name":"amp","signatures":{"sig":"old"}},
      "user_metadata":{"stars":1},
      "presets":[{"name":"old"}],
      "controls_schema":[{"id":"gain","user_metadata":{"tag":"a"},"calibration":{"user_metadata":"signed"}}],
      "stems_registry":[{"stem_id":"a","user_metadata":{"tag":"b"}}],
      "slimmable_tiers":{"lite":{"channels":4,"prefix_crc32c":12},"future":{"prefix_crc32c":13}}
    }"#;
        let orig_val = parse_strict_json_object_bytes(fixture).unwrap();
        let proj_orig = project_immutable_header_value(&orig_val).unwrap();
        let jcs_orig = serialize_jcs_value(&proj_orig, true).unwrap();

        // Mutating only designated mutable paths produces the exact same projected JCS string
        let modified = br#"{
      "version":"future-format",
      "metadata":{"name":"amp","signatures":{"sig":"new"}},
      "user_metadata":{"stars":5},
      "presets":[{"name":"new"}],
      "controls_schema":[{"id":"gain","user_metadata":{"tag":"b"},"calibration":{"user_metadata":"signed"}}],
      "stems_registry":[{"stem_id":"a","user_metadata":{"tag":"c"}}],
      "slimmable_tiers":{"lite":{"channels":4,"prefix_crc32c":99},"future":{"prefix_crc32c":13}}
    }"#;
        let mod_val = parse_strict_json_object_bytes(modified).unwrap();
        let proj_mod = project_immutable_header_value(&mod_val).unwrap();
        assert_eq!(serialize_jcs_value(&proj_mod, true).unwrap(), jcs_orig);

        // Canonical wire validation succeeds on canonical bytes and rejects non-canonical or unprojected bytes
        assert_eq!(
            validate_canonical_jcs_header_bytes(jcs_orig.as_bytes(), true).unwrap(),
            jcs_orig.as_bytes()
        );
        assert!(validate_canonical_jcs_header_bytes(fixture, true).is_err());

        // UTF-16 code-unit order puts supplementary U+10000 (0xD800 0xDC00) before U+E000 (0xE000)
        let utf16_obj = json!({ "\u{e000}": 2, "\u{10000}": 1 });
        assert_eq!(
            serialize_jcs_value(&utf16_obj, true).unwrap(),
            "{\"\u{10000}\":1,\"\u{e000}\":2}"
        );

        // Duplicate keys (including escaped \u0061 == 'a'), surrogates, and noncharacters are rejected
        for bad in [
            br#"{"a":1,"\u0061":2}"#.as_slice(),
            br#"{"name":"\ud800"}"#.as_slice(),
            br#"{"name":"\ufdd0"}"#.as_slice(),
            br#"{"name":"\uffff"}"#.as_slice(),
            b"[]".as_slice(),
        ] {
            assert!(parse_strict_json_object_bytes(bad).is_err());
        }
    }
}

#[cfg(test)]
mod jcs_reference_tests {
    use alloc::format;
    use serde_json::json;

    #[test]
    fn utf8_string_order_is_not_jcs_utf16_key_order() {
        let bmp = "\u{e000}";
        let supplementary = "\u{10000}";
        assert!(bmp < supplementary);
        assert!(supplementary.encode_utf16().cmp(bmp.encode_utf16()).is_lt());
        assert_eq!(
            serde_json::to_string(&json!({bmp: 1, supplementary: 2})).unwrap(),
            format!("{{\"{bmp}\":1,\"{supplementary}\":2}}")
        );
    }
}
