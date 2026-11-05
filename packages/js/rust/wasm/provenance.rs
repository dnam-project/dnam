//! WASM v2 provenance, signing, sealing, `DAU1`/`DME1`, preview containment,
//! and buyer watermarking exports (`ref[sdk.ts.provenance.v2-security]`).

use crate::core::provenance::{
    apply_isometric_gauge_watermark_core, apply_tardos_watermark_core,
    build_creator_signing_message_core, build_dme1_header_core,
    build_marketplace_signing_message_core, build_oem_cert_v2_signing_message_core,
    chacha20_xor_arena_core, compute_preview_window_core, decode_dau1_envelope_core,
    derive_watermark_seed_hex_core, ed25519_pubkey_hex_core, ed25519_sign_hex_core,
    encode_dau1_envelope_core, encode_hpke_slots_core, encode_recipient_key_ids_core,
    encode_record_directory_core, enforce_distribution_policy_core, evaluate_anticlone_policy_core,
    extract_stage3_fingerprint_core, extract_structural_hash_core, generate_tardos_code_core,
    hmac_sha256_hex_core, immutable_header_sha256_core, parse_dme1_plaintext_core,
    parse_hpke_slots_core, parse_recipient_key_ids_core, parse_record_directory_core,
    score_tardos_watermark_core, seal_container_core, sha256_hex_core, sign_creator_header_core,
    sign_marketplace_receipt_core, sign_oem_device_cert_core, structural_cosine_similarity_core,
    unseal_container_core, validate_preview_switch_options_core, validate_royalty_splits_core,
    verify_container_provenance_core, verify_oem_device_cert_core, wrap_device_cek_hex_core,
    x25519_pubkey_hex_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{set_error, set_result, slice_from_raw, slice_from_raw_mut, str_from_raw};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_container_provenance(
    hdr_ptr: *const u8,
    hdr_len: usize,
    arena_ptr: *const u8,
    arena_len: usize,
    has_sig: u8,
    dev_id_ptr: *const u8,
    dev_id_len: usize,
    dev_sec_ptr: *const u8,
    dev_sec_len: usize,
    mkt_pk_ptr: *const u8,
    mkt_pk_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slices in WASM linear memory.
    let hdr = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = slice_from_raw(arena_ptr, arena_len);
    let dev_id = if dev_id_len == 0 || dev_id_ptr.is_null() {
        None
    } else {
        match str_from_raw(dev_id_ptr, dev_id_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    let dev_sec = if dev_sec_len == 0 || dev_sec_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(dev_sec_ptr, dev_sec_len))
    };
    let mkt_pk = if mkt_pk_len == 0 || mkt_pk_ptr.is_null() {
        None
    } else {
        match str_from_raw(mkt_pk_ptr, mkt_pk_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    match verify_container_provenance_core(hdr, arena, has_sig != 0, dev_id, dev_sec, mkt_pk) {
        Ok(info) => match serde_json::to_vec(&info) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_enforce_distribution_policy(
    hdr_ptr: *const u8,
    hdr_len: usize,
    prof_ptr: *const u8,
    prof_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slices in WASM linear memory.
    let hdr = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let prof = match str_from_raw(prof_ptr, prof_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match enforce_distribution_policy_core(hdr, prof) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_evaluate_anticlone_policy(
    fingerprint_matched: u8,
    buyer_watermark_identified: u8,
    independent_device_source_evidence: u8,
) -> i32 {
    let s = evaluate_anticlone_policy_core(
        fingerprint_matched != 0,
        buyer_watermark_identified != 0,
        independent_device_source_evidence != 0,
    );
    set_result(s.as_bytes().to_vec())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_immutable_header_sha256(hdr_ptr: *const u8, hdr_len: usize) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let hdr = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match immutable_header_sha256_core(hdr) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sha256_hex(data_ptr: *const u8, data_len: usize) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let data = slice_from_raw(data_ptr, data_len);
    set_result(sha256_hex_core(data).into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_hmac_sha256_hex(
    key_ptr: *const u8,
    key_len: usize,
    msg_ptr: *const u8,
    msg_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slices in WASM linear memory.
    let key = slice_from_raw(key_ptr, key_len);
    let msg = slice_from_raw(msg_ptr, msg_len);
    match hmac_sha256_hex_core(key, msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_pubkey_hex(seed_ptr: *const u8, seed_len: usize) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let seed = slice_from_raw(seed_ptr, seed_len);
    match ed25519_pubkey_hex_core(seed) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_sign_hex(
    seed_ptr: *const u8,
    seed_len: usize,
    msg_ptr: *const u8,
    msg_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slices in WASM linear memory.
    let seed = slice_from_raw(seed_ptr, seed_len);
    let msg = slice_from_raw(msg_ptr, msg_len);
    match ed25519_sign_hex_core(seed, msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_x25519_pubkey_hex(sec_ptr: *const u8, sec_len: usize) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let sec = slice_from_raw(sec_ptr, sec_len);
    match x25519_pubkey_hex_core(sec) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_creator_signing_message(
    hdr_sha_ptr: *const u8,
    hdr_sha_len: usize,
    arena_sha_ptr: *const u8,
    arena_sha_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slices in WASM linear memory.
    let hdr_sha = match str_from_raw(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match str_from_raw(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_creator_signing_message_core(hdr_sha, arena_sha) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_marketplace_signing_message(
    hdr_sha_ptr: *const u8,
    hdr_sha_len: usize,
    arena_sha_ptr: *const u8,
    arena_sha_len: usize,
    rcpt_ptr: *const u8,
    rcpt_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slices in WASM linear memory.
    let hdr_sha = match str_from_raw(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match str_from_raw(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let rcpt = match str_from_raw(rcpt_ptr, rcpt_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_marketplace_signing_message_core(hdr_sha, arena_sha, rcpt) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_creator_header(
    hdr_ptr: *const u8,
    hdr_len: usize,
    arena_ptr: *const u8,
    arena_len: usize,
    seed_ptr: *const u8,
    seed_len: usize,
    signer_ptr: *const u8,
    signer_len: usize,
    include_tier_prefixes: u8,
) -> i32 {
    let hdr = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = slice_from_raw(arena_ptr, arena_len);
    let seed = slice_from_raw(seed_ptr, seed_len);
    let signer = match str_from_raw(signer_ptr, signer_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_creator_header_core(hdr, arena, seed, signer, include_tier_prefixes != 0) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_marketplace_receipt(
    hdr_ptr: *const u8,
    hdr_len: usize,
    arena_ptr: *const u8,
    arena_len: usize,
    seed_ptr: *const u8,
    seed_len: usize,
    spec_ptr: *const u8,
    spec_len: usize,
) -> i32 {
    let hdr = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = slice_from_raw(arena_ptr, arena_len);
    let seed = slice_from_raw(seed_ptr, seed_len);
    let spec = match str_from_raw(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_marketplace_receipt_core(hdr, arena, seed, spec) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_oem_device_cert(
    seed_ptr: *const u8,
    seed_len: usize,
    vendor_ptr: *const u8,
    vendor_len: usize,
    model_ptr: *const u8,
    model_len: usize,
    dev_id_ptr: *const u8,
    dev_id_len: usize,
    pk_ptr: *const u8,
    pk_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let seed = slice_from_raw(seed_ptr, seed_len);
    let vendor = match str_from_raw(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match str_from_raw(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = match str_from_raw(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk = slice_from_raw(pk_ptr, pk_len);
    match sign_oem_device_cert_core(seed, vendor, model, dev_id, pk) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_oem_device_cert(
    cert_ptr: *const u8,
    cert_len: usize,
    pk_hex_ptr: *const u8,
    pk_hex_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slices in WASM linear memory.
    let cert = match str_from_raw(cert_ptr, cert_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk_hex = match str_from_raw(pk_hex_ptr, pk_hex_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match verify_oem_device_cert_core(cert, pk_hex) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_build_oem_cert_v2_signing_message(
    vendor_ptr: *const u8,
    vendor_len: usize,
    model_ptr: *const u8,
    model_len: usize,
    dev_id_ptr: *const u8,
    dev_id_len: usize,
    pk_ptr: *const u8,
    pk_len: usize,
    iss_pk_ptr: *const u8,
    iss_pk_len: usize,
    assurance: u8,
    pol_ptr: *const u8,
    pol_len: usize,
    ev_ptr: *const u8,
    ev_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let vendor = match str_from_raw(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match str_from_raw(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = slice_from_raw(dev_id_ptr, dev_id_len);
    let pk = slice_from_raw(pk_ptr, pk_len);
    let iss_pk = slice_from_raw(iss_pk_ptr, iss_pk_len);
    let pol = if pol_len == 0 || pol_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(pol_ptr, pol_len))
    };
    let ev = if ev_len == 0 || ev_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(ev_ptr, ev_len))
    };
    match build_oem_cert_v2_signing_message_core(
        vendor, model, dev_id, pk, iss_pk, assurance, pol, ev,
    ) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_chacha20_xor_arena(
    arena_ptr: *const u8,
    arena_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    nonce_ptr: *const u8,
    nonce_len: usize,
    start_block_counter: u32,
) -> i32 {
    // SAFETY: Caller passes valid byte slices in WASM linear memory.
    let arena = slice_from_raw(arena_ptr, arena_len);
    let key = slice_from_raw(key_ptr, key_len);
    let nonce = slice_from_raw(nonce_ptr, nonce_len);
    match chacha20_xor_arena_core(arena, key, nonce, start_block_counter) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_wrap_device_cek_hex(
    eph_ptr: *const u8,
    eph_len: usize,
    dev_pk_ptr: *const u8,
    dev_pk_len: usize,
    dev_id_ptr: *const u8,
    dev_id_len: usize,
    cek_ptr: *const u8,
    cek_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let eph = slice_from_raw(eph_ptr, eph_len);
    let dev_pk = slice_from_raw(dev_pk_ptr, dev_pk_len);
    let dev_id = match str_from_raw(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let cek = slice_from_raw(cek_ptr, cek_len);
    match wrap_device_cek_hex_core(eph, dev_pk, dev_id, cek) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_seal_container(
    inner_ptr: *const u8,
    inner_len: usize,
    certs_ptr: *const u8,
    certs_len: usize,
    oem_pks_ptr: *const u8,
    oem_pks_len: usize,
    mkt_seed_ptr: *const u8,
    mkt_seed_len: usize,
    eph_sec_ptr: *const u8,
    eph_sec_len: usize,
    cek_ptr: *const u8,
    cek_len: usize,
    nonce_ptr: *const u8,
    nonce_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let inner = slice_from_raw(inner_ptr, inner_len);
    let certs = match str_from_raw(certs_ptr, certs_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let oem_pks = match str_from_raw(oem_pks_ptr, oem_pks_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let mkt_seed = slice_from_raw(mkt_seed_ptr, mkt_seed_len);
    let eph_sec = slice_from_raw(eph_sec_ptr, eph_sec_len);
    let cek = slice_from_raw(cek_ptr, cek_len);
    let nonce = slice_from_raw(nonce_ptr, nonce_len);
    match seal_container_core(inner, certs, oem_pks, mkt_seed, eph_sec, cek, nonce) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unseal_container(
    sealed_ptr: *const u8,
    sealed_len: usize,
    dev_id_ptr: *const u8,
    dev_id_len: usize,
    dev_sec_ptr: *const u8,
    dev_sec_len: usize,
    mkt_pk_ptr: *const u8,
    mkt_pk_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let sealed = slice_from_raw(sealed_ptr, sealed_len);
    let dev_id = if dev_id_len == 0 || dev_id_ptr.is_null() {
        None
    } else {
        match str_from_raw(dev_id_ptr, dev_id_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    let dev_sec = if dev_sec_len == 0 || dev_sec_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(dev_sec_ptr, dev_sec_len))
    };
    let mkt_pk = if mkt_pk_len == 0 || mkt_pk_ptr.is_null() {
        None
    } else {
        match str_from_raw(mkt_pk_ptr, mkt_pk_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    match unseal_container_core(sealed, dev_id, dev_sec, mkt_pk) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_dau1_envelope(spec_ptr: *const u8, spec_len: usize) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let spec = match str_from_raw(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match encode_dau1_envelope_core(spec) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_decode_dau1_envelope(wire_ptr: *const u8, wire_len: usize) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let wire = slice_from_raw(wire_ptr, wire_len);
    match decode_dau1_envelope_core(wire) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_dme1_header(
    profile: u8,
    recipient_count: u32,
    ciphertext_length: u64,
) -> i32 {
    match build_dme1_header_core(profile, recipient_count, ciphertext_length) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_dme1_plaintext(
    pt_ptr: *const u8,
    pt_len: usize,
    digests_json_ptr: *const u8,
    digests_json_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte/UTF-8 slices in WASM linear memory.
    let pt = slice_from_raw(pt_ptr, pt_len);
    let digests_str = match str_from_raw(digests_json_ptr, digests_json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(digests_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match parse_dme1_plaintext_core(pt, &digests) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_recipient_key_ids(json_ptr: *const u8, json_len: usize) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let json_str = match str_from_raw(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_recipient_key_ids_core(&digests) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_recipient_key_ids(
    data_ptr: *const u8,
    data_len: usize,
    max_recipients: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let data = slice_from_raw(data_ptr, data_len);
    match parse_recipient_key_ids_core(data, max_recipients) {
        Ok(list) => match serde_json::to_vec(&list) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_hpke_slots(json_ptr: *const u8, json_len: usize) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let json_str = match str_from_raw(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let slots: Vec<(String, String)> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_hpke_slots_core(&slots) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_hpke_slots(
    data_ptr: *const u8,
    data_len: usize,
    expected_count: usize,
    max_recipients: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let data = slice_from_raw(data_ptr, data_len);
    match parse_hpke_slots_core(data, expected_count, max_recipients) {
        Ok(slots) => match serde_json::to_vec(&slots) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_record_directory(
    maximum_tier: u8,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let json_str = match str_from_raw(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let records: Vec<(u64, u64)> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match encode_record_directory_core(maximum_tier, &records) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_parse_record_directory(
    data_ptr: *const u8,
    data_len: usize,
    maximum_tier: u8,
) -> i32 {
    // SAFETY: Caller passes valid byte slice in WASM linear memory.
    let data = slice_from_raw(data_ptr, data_len);
    match parse_record_directory_core(data, maximum_tier) {
        Ok(records) => match serde_json::to_vec(&records) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_compute_preview_window(
    anchor: f64,
    delta: f64,
    safe_min: f64,
    safe_max: f64,
    out2_ptr: *mut f64,
) -> i32 {
    if out2_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null out2_ptr".to_string());
    }
    match compute_preview_window_core(anchor, delta, (safe_min, safe_max)) {
        Ok(res) => {
            // SAFETY: `out2_ptr` points to 2 f64 elements in WASM linear memory.
            let out = slice_from_raw_mut(out2_ptr, 2);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_preview_switch_options(
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let json_str = match str_from_raw(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Spec {
        preset: String,
        alternative: String,
        observed: Vec<String>,
        active: Vec<String>,
        safe: Vec<String>,
    }
    let spec: Spec = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match validate_preview_switch_options_core(
        &spec.preset,
        &spec.alternative,
        &spec.observed,
        &spec.active,
        &spec.safe,
    ) {
        Ok((p, a)) => match serde_json::to_vec(&[p, a]) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_royalty_splits(
    tier_ptr: *const u8,
    tier_len: usize,
    splits_ptr: *const u8,
    splits_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid UTF-8 slices in WASM linear memory.
    let tier = match str_from_raw(tier_ptr, tier_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let splits = match str_from_raw(splits_ptr, splits_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match validate_royalty_splits_core(tier, splits) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_isometric_gauge_watermark(
    arena_ptr: *const u8,
    arena_len: usize,
    kappa_ptr: *const f32,
    kappa_len: usize,
    q_ptr: *const f32,
    q_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let arena = slice_from_raw(arena_ptr, arena_len);
    let kappa = slice_from_raw(kappa_ptr, kappa_len);
    let q = slice_from_raw(q_ptr, q_len);
    match apply_isometric_gauge_watermark_core(arena, kappa, q) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_generate_tardos_code(
    sec_ptr: *const u8,
    sec_len: usize,
    model_ptr: *const u8,
    model_len: usize,
    lic_ptr: *const u8,
    lic_len: usize,
    out_ptr: *mut f32,
    out_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let sec = slice_from_raw(sec_ptr, sec_len);
    let model = match str_from_raw(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let lic = match str_from_raw(lic_ptr, lic_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let out = slice_from_raw_mut(out_ptr, out_len);
    match generate_tardos_code_core(sec, model, lic) {
        Ok(code) => {
            if out.len() < code.len() {
                return set_error(validation_err("out buffer too short for Tardos code"));
            }
            out[..code.len()].copy_from_slice(&code);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_tardos_watermark(
    arena_ptr: *const u8,
    arena_len: usize,
    code_ptr: *const f32,
    code_len: usize,
    epsilon: f32,
) -> i32 {
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let arena = slice_from_raw(arena_ptr, arena_len);
    let code = slice_from_raw(code_ptr, code_len);
    match apply_tardos_watermark_core(arena, code, epsilon) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_score_tardos_watermark(
    master_ptr: *const u8,
    master_len: usize,
    suspect_ptr: *const u8,
    suspect_len: usize,
    code_ptr: *const f32,
    code_len: usize,
    out_score_ptr: *mut f64,
) -> i32 {
    if out_score_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null out_score_ptr".to_string());
    }
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let master = slice_from_raw(master_ptr, master_len);
    let suspect = slice_from_raw(suspect_ptr, suspect_len);
    let code = slice_from_raw(code_ptr, code_len);
    match score_tardos_watermark_core(master, suspect, code) {
        Ok(score) => {
            // SAFETY: `out_score_ptr` is non-null and points to a valid `f64` in WASM linear memory.
            unsafe { *out_score_ptr = score };
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_structural_hash(
    arena_ptr: *const u8,
    arena_len: usize,
    out_ptr: *mut f64,
    out_cap: usize,
) -> i32 {
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let arena = slice_from_raw(arena_ptr, arena_len);
    let out = slice_from_raw_mut(out_ptr, out_cap);
    match extract_structural_hash_core(arena) {
        Ok(vec) => {
            if out.len() < vec.len() {
                return set_error(validation_err("out buffer too short for structural hash"));
            }
            out[..vec.len()].copy_from_slice(&vec);
            vec.len() as i32
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_structural_cosine_similarity(
    a_ptr: *const f64,
    a_len: usize,
    b_ptr: *const f64,
    b_len: usize,
    out_sim_ptr: *mut f64,
) -> i32 {
    if out_sim_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null out_sim_ptr".to_string());
    }
    // SAFETY: Caller passes valid f64 slices in WASM linear memory.
    let a = slice_from_raw(a_ptr, a_len);
    let b = slice_from_raw(b_ptr, b_len);
    match structural_cosine_similarity_core(a, b) {
        Ok(sim) => {
            // SAFETY: `out_sim_ptr` is non-null and points to a valid `f64` in WASM linear memory.
            unsafe { *out_sim_ptr = sim };
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_derive_watermark_seed_hex(
    key_ptr: *const u8,
    key_len: usize,
    mkt_pk_ptr: *const u8,
    mkt_pk_len: usize,
    wm_kid_ptr: *const u8,
    wm_kid_len: usize,
    man_sha_ptr: *const u8,
    man_sha_len: usize,
    model_id_ptr: *const u8,
    model_id_len: usize,
    prof_sha_ptr: *const u8,
    prof_sha_len: usize,
    owning_tier: u8,
    lic_ptr: *const u8,
    lic_len: usize,
    buyer_ptr: *const u8,
    buyer_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid byte slices in WASM linear memory.
    let key = slice_from_raw(key_ptr, key_len);
    let mkt_pk = slice_from_raw(mkt_pk_ptr, mkt_pk_len);
    let wm_kid = slice_from_raw(wm_kid_ptr, wm_kid_len);
    let man_sha = slice_from_raw(man_sha_ptr, man_sha_len);
    let model_id = slice_from_raw(model_id_ptr, model_id_len);
    let prof_sha = slice_from_raw(prof_sha_ptr, prof_sha_len);
    let lic = if lic_len == 0 || lic_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(lic_ptr, lic_len))
    };
    let buyer = if buyer_len == 0 || buyer_ptr.is_null() {
        None
    } else {
        Some(slice_from_raw(buyer_ptr, buyer_len))
    };
    match derive_watermark_seed_hex_core(
        key,
        mkt_pk,
        wm_kid,
        man_sha,
        model_id,
        prof_sha,
        owning_tier,
        lic,
        buyer,
    ) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_stage3_fingerprint(
    fir_ptr: *const u8,
    fir_len: usize,
    kautz_ptr: *const u8,
    kautz_len: usize,
    added_modes: usize,
    out_ptr: *mut f64,
    out_cap: usize,
) -> i32 {
    // SAFETY: Caller passes valid slices in WASM linear memory.
    let fir = slice_from_raw(fir_ptr, fir_len);
    let kautz = slice_from_raw(kautz_ptr, kautz_len);
    let out = slice_from_raw_mut(out_ptr, out_cap);
    match extract_stage3_fingerprint_core(fir, kautz, added_modes) {
        Ok(vec) => {
            if out.len() < vec.len() {
                return set_error(validation_err(
                    "out buffer too short for Stage III fingerprint",
                ));
            }
            out[..vec.len()].copy_from_slice(&vec);
            vec.len() as i32
        }
        Err(msg) => set_error(msg),
    }
}
