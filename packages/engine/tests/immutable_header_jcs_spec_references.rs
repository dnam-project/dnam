//! Characterizes legacy digest gaps; these are not production conformance tests.

use dnam_engine::provenance::compute_immutable_header_sha256;
use dnam_engine::schema::{ControlSchemaDef, DnamJsonHeader};
use serde_json::{Value, json};

fn header_with_default(default: Value) -> DnamJsonHeader {
    let control: ControlSchemaDef = serde_json::from_value(json!({
        "id": "gain",
        "name": "Gain",
        "type": "range",
        "declared_range": [0.0, 10.0],
        "safe_clamped_range": [0.0, 10.0],
        "default": default
    }))
    .unwrap();
    let mut header = DnamJsonHeader::default_reference();
    header.controls_schema.push(control);
    header
}

#[test]
fn legacy_digest_distinguishes_jcs_equivalent_numeric_values() {
    let integer = header_with_default(json!(1));
    let floating = header_with_default(json!(1.0));
    assert_eq!(integer.controls_schema[0].default.as_f64(), Some(1.0));
    assert_eq!(floating.controls_schema[0].default.as_f64(), Some(1.0));
    assert_ne!(
        compute_immutable_header_sha256(&integer).unwrap(),
        compute_immutable_header_sha256(&floating).unwrap()
    );
}

#[test]
fn legacy_typed_header_discards_unknown_fields_before_hashing() {
    let reference = DnamJsonHeader::default_reference();
    let mut wire = serde_json::to_value(&reference).unwrap();
    wire.as_object_mut()
        .unwrap()
        .insert("future_critical_policy".into(), json!("must understand"));
    let parsed: DnamJsonHeader = serde_json::from_value(wire).unwrap();
    assert_eq!(
        compute_immutable_header_sha256(&reference).unwrap(),
        compute_immutable_header_sha256(&parsed).unwrap()
    );
}

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
