//! WASM v2 provenance, signing, sealing, `DAU1`/`DME1`, preview containment,
//! and buyer watermarking exports (`ref[sdk.ts.provenance.v2-security]`, `ref[sdk.ts.runtime.zero-unsafe]`).

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
use crate::wasm::abi::{
    read_bytes, read_f32_vec, read_f64_vec, read_opt_bytes, read_opt_str, read_str, set_error,
    set_result, with_f32_mut, with_f64_mut, write_f64, write_f64_slice,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_container_provenance(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    has_sig: u8,
    dev_id_ptr: usize,
    dev_id_len: usize,
    dev_sec_ptr: usize,
    dev_sec_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let dev_id = match read_opt_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_sec = read_opt_bytes(dev_sec_ptr, dev_sec_len);
    let mkt_pk = match read_opt_str(mkt_pk_ptr, mkt_pk_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match verify_container_provenance_core(
        &hdr,
        &arena,
        has_sig != 0,
        dev_id.as_deref(),
        dev_sec.as_deref(),
        mkt_pk.as_deref(),
    ) {
        Ok(info) => match serde_json::to_vec(&info) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_enforce_distribution_policy(
    hdr_ptr: usize,
    hdr_len: usize,
    prof_ptr: usize,
    prof_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let prof = match read_str(prof_ptr, prof_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match enforce_distribution_policy_core(&hdr, &prof) {
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
pub extern "C" fn dnam_wasm_immutable_header_sha256(hdr_ptr: usize, hdr_len: usize) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match immutable_header_sha256_core(&hdr) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sha256_hex(data_ptr: usize, data_len: usize) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    set_result(sha256_hex_core(&data).into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_hmac_sha256_hex(
    key_ptr: usize,
    key_len: usize,
    msg_ptr: usize,
    msg_len: usize,
) -> i32 {
    let key = read_bytes(key_ptr, key_len);
    let msg = read_bytes(msg_ptr, msg_len);
    match hmac_sha256_hex_core(&key, &msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_pubkey_hex(seed_ptr: usize, seed_len: usize) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    match ed25519_pubkey_hex_core(&seed) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_ed25519_sign_hex(
    seed_ptr: usize,
    seed_len: usize,
    msg_ptr: usize,
    msg_len: usize,
) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    let msg = read_bytes(msg_ptr, msg_len);
    match ed25519_sign_hex_core(&seed, &msg) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_x25519_pubkey_hex(sec_ptr: usize, sec_len: usize) -> i32 {
    let sec = read_bytes(sec_ptr, sec_len);
    match x25519_pubkey_hex_core(&sec) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_creator_signing_message(
    hdr_sha_ptr: usize,
    hdr_sha_len: usize,
    arena_sha_ptr: usize,
    arena_sha_len: usize,
) -> i32 {
    let hdr_sha = match read_str(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match read_str(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_creator_signing_message_core(&hdr_sha, &arena_sha) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_marketplace_signing_message(
    hdr_sha_ptr: usize,
    hdr_sha_len: usize,
    arena_sha_ptr: usize,
    arena_sha_len: usize,
    rcpt_ptr: usize,
    rcpt_len: usize,
) -> i32 {
    let hdr_sha = match read_str(hdr_sha_ptr, hdr_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena_sha = match read_str(arena_sha_ptr, arena_sha_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let rcpt = match read_str(rcpt_ptr, rcpt_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match build_marketplace_signing_message_core(&hdr_sha, &arena_sha, &rcpt) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_creator_header(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    seed_ptr: usize,
    seed_len: usize,
    signer_ptr: usize,
    signer_len: usize,
    include_tier_prefixes: u8,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let seed = read_bytes(seed_ptr, seed_len);
    let signer = match read_str(signer_ptr, signer_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_creator_header_core(&hdr, &arena, &seed, &signer, include_tier_prefixes != 0) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_marketplace_receipt(
    hdr_ptr: usize,
    hdr_len: usize,
    arena_ptr: usize,
    arena_len: usize,
    seed_ptr: usize,
    seed_len: usize,
    spec_ptr: usize,
    spec_len: usize,
) -> i32 {
    let hdr = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let arena = read_bytes(arena_ptr, arena_len);
    let seed = read_bytes(seed_ptr, seed_len);
    let spec = match read_str(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match sign_marketplace_receipt_core(&hdr, &arena, &seed, &spec) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sign_oem_device_cert(
    seed_ptr: usize,
    seed_len: usize,
    vendor_ptr: usize,
    vendor_len: usize,
    model_ptr: usize,
    model_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    pk_ptr: usize,
    pk_len: usize,
) -> i32 {
    let seed = read_bytes(seed_ptr, seed_len);
    let vendor = match read_str(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = match read_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk = read_bytes(pk_ptr, pk_len);
    match sign_oem_device_cert_core(&seed, &vendor, &model, &dev_id, &pk) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_verify_oem_device_cert(
    cert_ptr: usize,
    cert_len: usize,
    pk_hex_ptr: usize,
    pk_hex_len: usize,
) -> i32 {
    let cert = match read_str(cert_ptr, cert_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let pk_hex = match read_str(pk_hex_ptr, pk_hex_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match verify_oem_device_cert_core(&cert, &pk_hex) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_oem_cert_v2_signing_message(
    vendor_ptr: usize,
    vendor_len: usize,
    model_ptr: usize,
    model_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    pk_ptr: usize,
    pk_len: usize,
    iss_pk_ptr: usize,
    iss_pk_len: usize,
    assurance: u8,
    pol_ptr: usize,
    pol_len: usize,
    ev_ptr: usize,
    ev_len: usize,
) -> i32 {
    let vendor = match read_str(vendor_ptr, vendor_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_id = read_bytes(dev_id_ptr, dev_id_len);
    let pk = read_bytes(pk_ptr, pk_len);
    let iss_pk = read_bytes(iss_pk_ptr, iss_pk_len);
    let pol = read_opt_bytes(pol_ptr, pol_len);
    let ev = read_opt_bytes(ev_ptr, ev_len);
    match build_oem_cert_v2_signing_message_core(
        &vendor,
        &model,
        &dev_id,
        &pk,
        &iss_pk,
        assurance,
        pol.as_deref(),
        ev.as_deref(),
    ) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_chacha20_xor_arena(
    arena_ptr: usize,
    arena_len: usize,
    key_ptr: usize,
    key_len: usize,
    nonce_ptr: usize,
    nonce_len: usize,
    start_block_counter: u32,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let key = read_bytes(key_ptr, key_len);
    let nonce = read_bytes(nonce_ptr, nonce_len);
    match chacha20_xor_arena_core(&arena, &key, &nonce, start_block_counter) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_wrap_device_cek_hex(
    eph_ptr: usize,
    eph_len: usize,
    dev_pk_ptr: usize,
    dev_pk_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    cek_ptr: usize,
    cek_len: usize,
) -> i32 {
    let eph = read_bytes(eph_ptr, eph_len);
    let dev_pk = read_bytes(dev_pk_ptr, dev_pk_len);
    let dev_id = match read_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let cek = read_bytes(cek_ptr, cek_len);
    match wrap_device_cek_hex_core(&eph, &dev_pk, &dev_id, &cek) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_seal_container(
    inner_ptr: usize,
    inner_len: usize,
    certs_ptr: usize,
    certs_len: usize,
    oem_pks_ptr: usize,
    oem_pks_len: usize,
    mkt_seed_ptr: usize,
    mkt_seed_len: usize,
    eph_sec_ptr: usize,
    eph_sec_len: usize,
    cek_ptr: usize,
    cek_len: usize,
    nonce_ptr: usize,
    nonce_len: usize,
) -> i32 {
    let inner = read_bytes(inner_ptr, inner_len);
    let certs = match read_str(certs_ptr, certs_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let oem_pks = match read_str(oem_pks_ptr, oem_pks_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let mkt_seed = read_bytes(mkt_seed_ptr, mkt_seed_len);
    let eph_sec = read_bytes(eph_sec_ptr, eph_sec_len);
    let cek = read_bytes(cek_ptr, cek_len);
    let nonce = read_bytes(nonce_ptr, nonce_len);
    match seal_container_core(&inner, &certs, &oem_pks, &mkt_seed, &eph_sec, &cek, &nonce) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unseal_container(
    sealed_ptr: usize,
    sealed_len: usize,
    dev_id_ptr: usize,
    dev_id_len: usize,
    dev_sec_ptr: usize,
    dev_sec_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
) -> i32 {
    let sealed = read_bytes(sealed_ptr, sealed_len);
    let dev_id = match read_opt_str(dev_id_ptr, dev_id_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let dev_sec = read_opt_bytes(dev_sec_ptr, dev_sec_len);
    let mkt_pk = match read_opt_str(mkt_pk_ptr, mkt_pk_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match unseal_container_core(
        &sealed,
        dev_id.as_deref(),
        dev_sec.as_deref(),
        mkt_pk.as_deref(),
    ) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_dau1_envelope(spec_ptr: usize, spec_len: usize) -> i32 {
    let spec = match read_str(spec_ptr, spec_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match encode_dau1_envelope_core(&spec) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_decode_dau1_envelope(wire_ptr: usize, wire_len: usize) -> i32 {
    let wire = read_bytes(wire_ptr, wire_len);
    match decode_dau1_envelope_core(&wire) {
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
    pt_ptr: usize,
    pt_len: usize,
    digests_json_ptr: usize,
    digests_json_len: usize,
) -> i32 {
    let pt = read_bytes(pt_ptr, pt_len);
    let digests_str = match read_str(digests_json_ptr, digests_json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(&digests_str) {
        Ok(v) => v,
        Err(e) => return set_error(validation_err(e.to_string())),
    };
    match parse_dme1_plaintext_core(&pt, &digests) {
        Ok(json) => set_result(json.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_recipient_key_ids(json_ptr: usize, json_len: usize) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let digests: Vec<String> = match serde_json::from_str(&json_str) {
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
    data_ptr: usize,
    data_len: usize,
    max_recipients: usize,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_recipient_key_ids_core(&data, max_recipients) {
        Ok(list) => match serde_json::to_vec(&list) {
            Ok(vec) => set_result(vec),
            Err(e) => set_error(validation_err(e.to_string())),
        },
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_encode_hpke_slots(json_ptr: usize, json_len: usize) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let slots: Vec<(String, String)> = match serde_json::from_str(&json_str) {
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
    data_ptr: usize,
    data_len: usize,
    expected_count: usize,
    max_recipients: usize,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_hpke_slots_core(&data, expected_count, max_recipients) {
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
    json_ptr: usize,
    json_len: usize,
) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let records: Vec<(u64, u64)> = match serde_json::from_str(&json_str) {
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
    data_ptr: usize,
    data_len: usize,
    maximum_tier: u8,
) -> i32 {
    let data = read_bytes(data_ptr, data_len);
    match parse_record_directory_core(&data, maximum_tier) {
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
    out2_ptr: usize,
) -> i32 {
    if out2_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out2_ptr".to_string());
    }
    match compute_preview_window_core(anchor, delta, (safe_min, safe_max)) {
        Ok(res) => {
            if !write_f64_slice(out2_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out2_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_preview_switch_options(
    json_ptr: usize,
    json_len: usize,
) -> i32 {
    let json_str = match read_str(json_ptr, json_len) {
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
    let spec: Spec = match serde_json::from_str(&json_str) {
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
    tier_ptr: usize,
    tier_len: usize,
    splits_ptr: usize,
    splits_len: usize,
) -> i32 {
    let tier = match read_str(tier_ptr, tier_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let splits = match read_str(splits_ptr, splits_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match validate_royalty_splits_core(&tier, &splits) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_isometric_gauge_watermark(
    arena_ptr: usize,
    arena_len: usize,
    kappa_ptr: usize,
    kappa_len: usize,
    q_ptr: usize,
    q_len: usize,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let kappa = read_f32_vec(kappa_ptr, kappa_len);
    let q = read_f32_vec(q_ptr, q_len);
    match apply_isometric_gauge_watermark_core(&arena, &kappa, &q) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_generate_tardos_code(
    sec_ptr: usize,
    sec_len: usize,
    model_ptr: usize,
    model_len: usize,
    lic_ptr: usize,
    lic_len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let sec = read_bytes(sec_ptr, sec_len);
    let model = match read_str(model_ptr, model_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let lic = match read_str(lic_ptr, lic_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match generate_tardos_code_core(&sec, &model, &lic) {
        Ok(code) => {
            let Some(ok) = with_f32_mut(out_ptr, out_len, |out| {
                if out.len() < code.len() {
                    return false;
                }
                out[..code.len()].copy_from_slice(&code);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                0
            } else {
                set_error(validation_err("out buffer too short for Tardos code"))
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_tardos_watermark(
    arena_ptr: usize,
    arena_len: usize,
    code_ptr: usize,
    code_len: usize,
    epsilon: f32,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    let code = read_f32_vec(code_ptr, code_len);
    match apply_tardos_watermark_core(&arena, &code, epsilon) {
        Ok(bytes) => set_result(bytes),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_score_tardos_watermark(
    master_ptr: usize,
    master_len: usize,
    suspect_ptr: usize,
    suspect_len: usize,
    code_ptr: usize,
    code_len: usize,
    out_score_ptr: usize,
) -> i32 {
    if out_score_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out_score_ptr".to_string());
    }
    let master = read_bytes(master_ptr, master_len);
    let suspect = read_bytes(suspect_ptr, suspect_len);
    let code = read_f32_vec(code_ptr, code_len);
    match score_tardos_watermark_core(&master, &suspect, &code) {
        Ok(score) => {
            if !write_f64(out_score_ptr, score) {
                return set_error("[DNAM-VALIDATION] invalid out_score_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_structural_hash(
    arena_ptr: usize,
    arena_len: usize,
    out_ptr: usize,
    out_cap: usize,
) -> i32 {
    let arena = read_bytes(arena_ptr, arena_len);
    match extract_structural_hash_core(&arena) {
        Ok(vec) => {
            let Some(ok) = with_f64_mut(out_ptr, out_cap, |out| {
                if out.len() < vec.len() {
                    return false;
                }
                out[..vec.len()].copy_from_slice(&vec);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                vec.len() as i32
            } else {
                set_error(validation_err("out buffer too short for structural hash"))
            }
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_structural_cosine_similarity(
    a_ptr: usize,
    a_len: usize,
    b_ptr: usize,
    b_len: usize,
    out_sim_ptr: usize,
) -> i32 {
    if out_sim_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out_sim_ptr".to_string());
    }
    let a = read_f64_vec(a_ptr, a_len);
    let b = read_f64_vec(b_ptr, b_len);
    match structural_cosine_similarity_core(&a, &b) {
        Ok(sim) => {
            if !write_f64(out_sim_ptr, sim) {
                return set_error("[DNAM-VALIDATION] invalid out_sim_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_derive_watermark_seed_hex(
    key_ptr: usize,
    key_len: usize,
    mkt_pk_ptr: usize,
    mkt_pk_len: usize,
    wm_kid_ptr: usize,
    wm_kid_len: usize,
    man_sha_ptr: usize,
    man_sha_len: usize,
    model_id_ptr: usize,
    model_id_len: usize,
    prof_sha_ptr: usize,
    prof_sha_len: usize,
    owning_tier: u8,
    lic_ptr: usize,
    lic_len: usize,
    buyer_ptr: usize,
    buyer_len: usize,
) -> i32 {
    let key = read_bytes(key_ptr, key_len);
    let mkt_pk = read_bytes(mkt_pk_ptr, mkt_pk_len);
    let wm_kid = read_bytes(wm_kid_ptr, wm_kid_len);
    let man_sha = read_bytes(man_sha_ptr, man_sha_len);
    let model_id = read_bytes(model_id_ptr, model_id_len);
    let prof_sha = read_bytes(prof_sha_ptr, prof_sha_len);
    let lic = read_opt_bytes(lic_ptr, lic_len);
    let buyer = read_opt_bytes(buyer_ptr, buyer_len);
    match derive_watermark_seed_hex_core(
        &key,
        &mkt_pk,
        &wm_kid,
        &man_sha,
        &model_id,
        &prof_sha,
        owning_tier,
        lic.as_deref(),
        buyer.as_deref(),
    ) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_stage3_fingerprint(
    fir_ptr: usize,
    fir_len: usize,
    kautz_ptr: usize,
    kautz_len: usize,
    added_modes: usize,
    out_ptr: usize,
    out_cap: usize,
) -> i32 {
    let fir = read_bytes(fir_ptr, fir_len);
    let kautz = read_bytes(kautz_ptr, kautz_len);
    match extract_stage3_fingerprint_core(&fir, &kautz, added_modes) {
        Ok(vec) => {
            let Some(ok) = with_f64_mut(out_ptr, out_cap, |out| {
                if out.len() < vec.len() {
                    return false;
                }
                out[..vec.len()].copy_from_slice(&vec);
                true
            }) else {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            };
            if ok {
                vec.len() as i32
            } else {
                set_error(validation_err(
                    "out buffer too short for Stage III fingerprint",
                ))
            }
        }
        Err(msg) => set_error(msg),
    }
}
