//! Example 4: Container Tier Slimming, In-Place 4 KB Header Editing & `.dnam-ckpt` `uplo='U'` Packing
//! (`examples/container_slimming_and_ckpt.rs`).
//!
//! Demonstrates:
//! 1. Stripping an `Ultra` (`64,000 B` arena) `.dnam` container down to `Pro` (`36,224 B`),
//!    `Full` (`22,336 B`), and `Lite` (`10,752 B`) via [`strip_container_to_tier`] without
//!    modifying the immutable JSON header.
//! 2. Editing mutable `presets` and `user_metadata` in place within the 4 KB page-aligned
//!    header slack via [`update_header_in_place`].
//! 3. Round-tripping the v2.0 `DMU1` mutable suffix trailer (`build_dmu1_mutable_frame` /
//!    `parse_and_validate_dmu1_frame`, `ref[container.dnam.mutable-frame]`).
//!
//! Run with:
//!   `cargo run -p dnam-engine --example container_slimming_and_ckpt`

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dmu1_mutable_frame, build_dnam_container,
    parse_and_validate_dmu1_frame, strip_container_to_tier, update_header_in_place,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{DnamJsonHeader, PresetDef};

fn main() {
    let mut header = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    for (i, byte) in arena[128..62_000].iter_mut().enumerate() {
        *byte = ((i * 13 + 7) & 0x3F) as u8;
    }

    // 1. Build an Ultra (64,000 B arena) .dnam container and strip to Pro, Full, and Lite
    let ultra_container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let crypto = DefaultCryptoProvider::default();

    for tier in [Tier::Ultra, Tier::Pro, Tier::Full, Tier::Lite] {
        let stripped = strip_container_to_tier(&ultra_container, tier).unwrap();
        let view = DnamContainerView::from_bytes(&stripped, &crypto).unwrap();
        println!(
            "Stripped Tier {:<5} -> Total File: {:>6} B | Arena Prefix: {:>6} B | Highest Tier: {:?}",
            tier.as_str(),
            stripped.len(),
            view.arena_storage.as_bytes().len(),
            view.preamble.highest_tier(),
        );
    }

    // 2. Edit user_metadata and presets in place within the 4 KB page-aligned header slack
    let mut editable_container = ultra_container;
    header
        .user_metadata
        .insert("favorite".to_string(), serde_json::json!(true));
    header.presets.push(PresetDef {
        id: "studio_clean".to_string(),
        name: "Studio Clean".to_string(),
        description: Some("Added in-place without touching tensor arena".to_string()),
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    });
    update_header_in_place(&mut editable_container, &header)
        .expect("in-place 4 KB header update succeeded");

    let updated_view = DnamContainerView::from_bytes(&editable_container, &crypto).unwrap();
    println!(
        "In-place header edit verified | Presets count: {} | user_metadata['favorite'] = {}",
        updated_view.header.presets.len(),
        updated_view.header.user_metadata.get("favorite").unwrap(),
    );

    // 3. Round-trip a 64B-aligned DMU1 mutable trailer frame
    let dmu1_json = br#"{"presets":[],"user_metadata":{"tag":"warm"}}"#;
    let dmu1_frame = build_dmu1_mutable_frame(dmu1_json, 64).expect("valid DMU1 frame");
    let parsed_dmu1 = parse_and_validate_dmu1_frame(&dmu1_frame).expect("verified DMU1 CRC32C");
    println!(
        "DMU1 frame length: {} B (64B aligned) | Parsed tag: {}",
        dmu1_frame.len(),
        parsed_dmu1["user_metadata"]["tag"],
    );
}
