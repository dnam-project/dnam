//! Shared cryptographic provenance, RFC 8785 JCS SHA-256, `Ed25519` signing messages,
//! `ChaCha20` 64B cache-line arena encryption, Dual-Mechanism Buyer Watermarking (`Test 8H`),
//! and 160-D structural hash functions.

use dnam_engine::provenance::{
    RGLRU_TOTAL_CHANNELS, STRUCTURAL_HASH_DIM, TARDOS_CONV_WEIGHTS,
    apply_isometric_gauge_watermark, apply_tardos_watermark_4608, build_creator_signing_message,
    build_marketplace_receipt_signing_message, chacha20_xor_arena_blocks,
    compute_immutable_header_sha256, decode_hex_array, encode_hex, extract_structural_hash_160d,
    generate_tardos_code_4608, score_tardos_watermark_4608, sha256_digest,
    structural_cosine_similarity,
};
use dnam_engine::schema::{DnamJsonHeader, MarketplaceReceipt};

use crate::format_tagged_dnam_error;

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest over `header_json`.
pub fn immutable_header_sha256_core(header_json: &str) -> Result<String, String> {
    let header: DnamJsonHeader =
        serde_json::from_str(header_json).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?;
    let digest =
        compute_immutable_header_sha256(&header).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(encode_hex(&digest))
}

/// Computes the lowercase hex SHA-256 digest of `data`.
pub fn sha256_hex_core(data: &[u8]) -> String {
    let digest = sha256_digest(data);
    encode_hex(&digest)
}

/// Builds the canonical `Ed25519` creator signing message for `(immutable_header_sha256_hex, master_arena_sha256_hex)`.
pub fn build_creator_signing_message_core(
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
) -> Result<Vec<u8>, String> {
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(build_creator_signing_message(&hdr_sha, &arena_sha).to_vec())
}

/// Builds the canonical `Ed25519` marketplace receipt signing message.
pub fn build_marketplace_signing_message_core(
    immutable_header_sha256_hex: &str,
    master_arena_sha256_hex: &str,
    receipt_json: &str,
) -> Result<Vec<u8>, String> {
    let hdr_sha = decode_hex_array::<32>(immutable_header_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let arena_sha = decode_hex_array::<32>(master_arena_sha256_hex)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let receipt: MarketplaceReceipt =
        serde_json::from_str(receipt_json).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?;
    build_marketplace_receipt_signing_message(&hdr_sha, &arena_sha, &receipt)
        .map_err(|e| format_tagged_dnam_error(&e))
}

/// Applies RFC 8439 `ChaCha20` 64-byte cache-line XOR encryption/decryption over `arena_bytes`.
pub fn chacha20_xor_arena_core(
    arena_bytes: &[u8],
    key_32: &[u8],
    nonce_12: &[u8],
    start_block_counter: u32,
) -> Result<Vec<u8>, String> {
    let key: &[u8; 32] = key_32.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] key32 must be 32 bytes, got {}",
            key_32.len()
        )
    })?;
    let nonce: &[u8; 12] = nonce_12.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] nonce12 must be 12 bytes, got {}",
            nonce_12.len()
        )
    })?;
    let mut buf = arena_bytes.to_vec();
    chacha20_xor_arena_blocks(&mut buf, key, nonce, start_block_counter)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

/// Applies Mechanism A (48-channel RG-LRU reciprocal scaling + SO(4) SVD head rotation) to `arena_bytes`.
pub fn apply_isometric_gauge_watermark_core(
    arena_bytes: &[u8],
    kappa_48: &[f32],
    q_buyer_flat_16: &[f32],
) -> Result<Vec<u8>, String> {
    if kappa_48.len() != RGLRU_TOTAL_CHANNELS {
        return Err(format!(
            "[DNAM-VALIDATION] kappa48 must have {RGLRU_TOTAL_CHANNELS} elements, got {}",
            kappa_48.len()
        ));
    }
    if q_buyer_flat_16.len() != 16 {
        return Err(format!(
            "[DNAM-VALIDATION] qBuyer4x4 must have 16 elements, got {}",
            q_buyer_flat_16.len()
        ));
    }
    let mut kappa_arr = [1.0_f32; RGLRU_TOTAL_CHANNELS];
    kappa_arr.copy_from_slice(kappa_48);
    let mut q_arr = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        q_arr[r].copy_from_slice(&q_buyer_flat_16[r * 4..(r + 1) * 4]);
    }

    let mut buf = arena_bytes.to_vec();
    apply_isometric_gauge_watermark(&mut buf, &kappa_arr, &q_arr)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

/// Generates a 4,608-chip Tardos binary spread-spectrum watermark code `{-1.0, +1.0}^4608`.
pub fn generate_tardos_code_core(
    marketplace_secret_32: &[u8],
    model_id: &str,
    license_id: &str,
) -> Result<Vec<f32>, String> {
    let secret: &[u8; 32] = marketplace_secret_32.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] marketplaceSecret must be 32 bytes, got {}",
            marketplace_secret_32.len()
        )
    })?;
    let code = generate_tardos_code_4608(secret, model_id, license_id);
    Ok(code.to_vec())
}

/// Embeds Mechanism B (4,608-weight Tardos spread-spectrum watermark) into `arena_bytes`.
pub fn apply_tardos_watermark_core(
    arena_bytes: &[u8],
    code_4608: &[f32],
    epsilon: f32,
) -> Result<Vec<u8>, String> {
    if code_4608.len() != TARDOS_CONV_WEIGHTS {
        return Err(format!(
            "[DNAM-VALIDATION] Tardos code must have {TARDOS_CONV_WEIGHTS} elements, got {}",
            code_4608.len()
        ));
    }
    let mut code_arr = [1.0_f32; TARDOS_CONV_WEIGHTS];
    code_arr.copy_from_slice(code_4608);

    let mut buf = arena_bytes.to_vec();
    apply_tardos_watermark_4608(&mut buf, &code_arr, epsilon)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

/// Computes the Tardos matched-filter Z-score (`sigma`) of `suspect_arena` against `master_arena` and `code_4608`.
pub fn score_tardos_watermark_core(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code_4608: &[f32],
) -> Result<f64, String> {
    if code_4608.len() != TARDOS_CONV_WEIGHTS {
        return Err(format!(
            "[DNAM-VALIDATION] Tardos code must have {TARDOS_CONV_WEIGHTS} elements, got {}",
            code_4608.len()
        ));
    }
    let mut code_arr = [1.0_f32; TARDOS_CONV_WEIGHTS];
    code_arr.copy_from_slice(code_4608);

    score_tardos_watermark_4608(master_arena, suspect_arena, &code_arr)
        .map_err(|e| format_tagged_dnam_error(&e))
}

/// Extracts the 160-Dimensional Gauge-Invariant Structural Hash vector `f_struct in R^160` from `arena_bytes`.
pub fn extract_structural_hash_core(arena_bytes: &[u8]) -> Result<Vec<f64>, String> {
    let vec160 =
        extract_structural_hash_160d(arena_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(vec160.to_vec())
}

/// Computes the cosine similarity `cos(a, b) in [-1, +1]` between two 160-D structural hash vectors.
pub fn structural_cosine_similarity_core(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != STRUCTURAL_HASH_DIM || b.len() != STRUCTURAL_HASH_DIM {
        return Err(format!(
            "[DNAM-VALIDATION] Structural hash vectors must have length {STRUCTURAL_HASH_DIM}"
        ));
    }
    let mut a_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    let mut b_arr = [0.0_f64; STRUCTURAL_HASH_DIM];
    a_arr.copy_from_slice(a);
    b_arr.copy_from_slice(b);
    Ok(structural_cosine_similarity(&a_arr, &b_arr))
}
