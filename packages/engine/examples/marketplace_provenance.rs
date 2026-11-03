//! Example 3: Digital Marketplace Provenance, Isometric Gauge Watermarking & Tardos Tracing (`examples/marketplace_provenance.rs`).
//!
//! Demonstrates:
//! 1. Extracting the 224-D Gauge-Invariant Structural Hash (`extract_structural_hash_224d`).
//! 2. Embedding a buyer-specific Exact Isometric Gauge Watermark (`apply_isometric_gauge_watermark`)
//!    that preserves the 224-D structural hash (`cosine_similarity == 1.000000`).
//! 3. Embedding and scoring the 7,680-weight Spread-Spectrum Tardos Traitor-Tracing Code
//!    (`generate_tardos_code_7680`, `apply_tardos_watermark_7680`, `score_tardos_watermark_7680`).
//!
//! Run with:
//!   `cargo run -p dnam-engine --example marketplace_provenance`

use dnam_engine::container::ARENA_OFFSET_ULTRA_END;
use dnam_engine::provenance::{
    apply_isometric_gauge_watermark, apply_tardos_watermark_7680, extract_structural_hash_224d,
    generate_tardos_code_7680, score_tardos_watermark_7680, structural_cosine_similarity,
};

fn main() {
    // 1. Construct a 64,000-byte `Arena 0..3` weight buffer with non-zero Op-M Kautz modal poles & residues
    let mut master_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    for (i, chunk) in master_arena[128..63_000]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .enumerate()
    {
        let val = 0.05 + 0.02 * ((i as f32) * 0.17).sin();
        *chunk = val.to_le_bytes();
    }

    // 2. Extract the 224-D gauge-invariant structural hash from the master model
    let hash_master = extract_structural_hash_224d(&master_arena).expect("valid master arena");

    // 3. Embed an Exact Isometric Gauge Watermark (64-channel reciprocal scaling + SO(4) head rotation)
    let mut buyer1_arena = master_arena.clone();
    let mut kappa = [1.0_f32; 64];
    for (c, k) in kappa.iter_mut().enumerate() {
        *k = if c % 2 == 0 { 1.03 } else { 0.97 };
    }
    let theta = 0.25_f32;
    let (ct, st) = (theta.cos(), theta.sin());
    let q_rot = [
        [ct, -st, 0.0, 0.0],
        [st, ct, 0.0, 0.0],
        [0.0, 0.0, ct, -st],
        [0.0, 0.0, st, ct],
    ];
    apply_isometric_gauge_watermark(&mut buyer1_arena, &kappa, &q_rot)
        .expect("isometric gauge watermark applied");

    // 4. Verify the 224-D structural hash is invariant under the isometric gauge transformation
    let hash_buyer1 = extract_structural_hash_224d(&buyer1_arena).expect("valid buyer1 arena");
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer1);
    println!(
        "224-D Structural Hash Cosine Similarity (Master vs. Gauge-Rotated Buyer #1): {:.8}",
        cos_sim
    );

    // 5. Embed & detect 7,680-chip Coupled Tardos Traitor-Tracing Code for Buyer #1
    let secret_seed = [0x42_u8; 32];
    let code_buyer1 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1001");
    let code_buyer2 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1002");
    let code_buyer3 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1003");

    apply_tardos_watermark_7680(&mut buyer1_arena, &code_buyer1, 1.5e-3)
        .expect("Tardos watermark applied");

    for (buyer_id, code) in [
        ("buyer_1001 (actual)", &code_buyer1),
        ("buyer_1002 (innocent)", &code_buyer2),
        ("buyer_1003 (innocent)", &code_buyer3),
    ] {
        let score = score_tardos_watermark_7680(&master_arena, &buyer1_arena, code)
            .expect("Tardos correlation computed");
        println!(
            "Candidate {:<22} -> Tardos correlation score: {:+.4}",
            buyer_id, score
        );
    }
}
