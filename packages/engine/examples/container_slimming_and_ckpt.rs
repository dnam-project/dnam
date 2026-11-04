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
//! 4. Extracting a `Lite` (`64 x 64`) leading principal Gram submatrix directly from the
//!    contiguous prefix of an `Ultra` (`256 x 256`) `LAPACK uplo='U'` packed buffer via
//!    [`pack_uplo_u_f64`] and [`unpack_uplo_u_f64`] (`ref[container.ckpt.uplo-u]`).
//!
//! Run with:
//!   `cargo run -p dnam-engine --example container_slimming_and_ckpt`

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dmu1_mutable_frame, build_dnam_container,
    pack_uplo_u_f64, parse_and_validate_dmu1_frame, strip_container_to_tier, unpack_uplo_u_f64,
    update_header_in_place, uplo_u_packed_len,
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

    // 4. Verify Nested Principal Submatrix Prefix Theorem on LAPACK uplo='U' packed Gram matrix
    let n_ultra = 16_usize;
    let n_lite = 4_usize;
    let mut dense_ultra = vec![0.0_f64; n_ultra * n_ultra];
    for r in 0..n_ultra {
        for c in 0..n_ultra {
            dense_ultra[r * n_ultra + c] = ((r + 1) * (c + 1)) as f64;
        }
    }
    let mut packed_ultra = vec![0.0_f64; uplo_u_packed_len(n_ultra)];
    pack_uplo_u_f64(&dense_ultra, n_ultra, &mut packed_ultra);

    // Extract the leading 4x4 submatrix using ONLY the first `uplo_u_packed_len(4) == 10` elements
    let lite_prefix_len = uplo_u_packed_len(n_lite);
    let mut unpacked_lite = vec![0.0_f64; n_lite * n_lite];
    unpack_uplo_u_f64(&packed_ultra[..lite_prefix_len], n_lite, &mut unpacked_lite);
    for r in 0..n_lite {
        for c in 0..n_lite {
            assert_eq!(unpacked_lite[r * n_lite + c], dense_ultra[r * n_ultra + c]);
        }
    }
    println!(
        "uplo='U' Prefix Theorem verified: extracted {}x{} Gram submatrix from first {} packed f64s",
        n_lite, n_lite, lite_prefix_len,
    );
}
