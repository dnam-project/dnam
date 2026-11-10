#![forbid(unsafe_code)]
//! `.dnam-ckpt` v2.0 Bundle Framing, `DCA1` Signature, and Zero-Copy Runtime Export
//! (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`,
//! `ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`,
//! `ref[container.ckpt.uplo-u]`).

use dnam_engine::Tier;
use dnam_engine::container::{
    DEFAULT_MAX_MUTABLE_CAPACITY, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, V2_HEADER_OFFSET,
    V2_PREAMBLE_SIZE, V2ContainerLayout, align64_u64, append_dmue_mutable_suffix,
    build_dmu1_mutable_frame, crc32c_update, encode_domain_field, parse_and_validate_dmu1_frame,
    parse_dcrc_record, validate_mutable_json_schema, validate_v2_open_dnam_layout,
};
use dnam_engine::provenance::{
    CryptoProvider, Dau1Envelope, DefaultCryptoProvider, build_buyer_receipt_v1,
    build_creator_manifest_v1, build_plaintext_shared_v1, compute_creator_signed_manifest_sha256,
    compute_license_policy_sha256, compute_plaintext_prefixes_v1, sha256_digest,
};
use dnam_engine::schema::{
    DistributionPolicy, DnamError, DnamProtectionProfile, SchemaOptionExt, reject_if,
};

pub use super::view::{
    DnamCkptView, build_dnam_ckpt_container, build_dnam_ckpt_container_with_cp1,
    is_valid_arena4_stem_slice_len, resolve_ckpt_stems_end,
};

/// Computes the per-tier SHA-256 checkpoint prefix commitments under `"DNCK-CHECKPOINT-PREFIX-V1"`
/// (`ref[container.ckpt.dca1-signature]`).
#[must_use]
pub fn compute_dnck_bundle_prefixes(
    preamble: &[u8],
    directory: &[u8],
    header: &[u8],
    shared: &[u8],
    bundles: &[&[u8]],
) -> Vec<[u8; 32]> {
    let header_sha = sha256_digest(header);
    let mut base = Vec::new();
    base.extend_from_slice(&encode_domain_field("domain", b"DNCK-CHECKPOINT-PREFIX-V1"));
    base.extend_from_slice(&encode_domain_field(
        "checkpoint_header_sha256",
        &header_sha,
    ));
    base.extend_from_slice(&encode_domain_field("preamble", preamble));
    base.extend_from_slice(&encode_domain_field("directory", directory));
    base.extend_from_slice(&encode_domain_field("shared_checkpoint_binary", shared));

    let labels = ["bundle_lite", "bundle_full", "bundle_pro", "bundle_ultra"];
    let mut out = Vec::with_capacity(bundles.len());
    for idx in 0..bundles.len() {
        let mut msg = base.clone();
        for t in 0..=idx {
            msg.extend_from_slice(&encode_domain_field(labels[t], bundles[t]));
        }
        out.push(sha256_digest(&msg));
    }
    out
}

/// Builds the `"DNCK-CHECKPOINT-MANIFEST-V1"` signing message for `DCA1` (`ref[container.ckpt.dca1-signature]`).
#[must_use]
pub fn build_dnck_writer_manifest_message(
    writer_pubkey: &[u8; 32],
    checkpoint_header_sha256: &[u8; 32],
    issued_max_tier: u8,
    prefixes: &[[u8; 32]],
) -> Vec<u8> {
    let mut prefix_list = Vec::with_capacity(1 + 33 * prefixes.len());
    prefix_list.push(prefixes.len() as u8);
    for (t, d) in prefixes.iter().enumerate() {
        prefix_list.push(t as u8);
        prefix_list.extend_from_slice(d);
    }
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field(
        "domain",
        b"DNCK-CHECKPOINT-MANIFEST-V1",
    ));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_writer_pubkey_ed25519",
        writer_pubkey,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_header_sha256",
        checkpoint_header_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field("issued_max_tier", &[issued_max_tier]));
    msg.extend_from_slice(&encode_domain_field(
        "checkpoint_prefix_sha256",
        &prefix_list,
    ));
    msg
}

/// Validates a v2.0 `.dnam-ckpt` (`b"DNCK"`) container, its `DCA1` envelope, and its retained
/// tier bundle prefix commitment (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`).
pub fn validate_v2_dnck_checkpoint(data: &[u8]) -> Result<(u8, Vec<usize>, usize), DnamError> {
    reject_if(data.len() < 208, "short checkpoint")?;
    let magic = &data[0..4];
    let major = u16::from_le_bytes([data[4], data[5]]);
    let minor = u16::from_le_bytes([data[6], data[7]]);
    let profile = data[8];
    let auth_kind = data[9];
    let maximum = data[10];
    let reserved = data[11];
    let rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    reject_if(
        magic != b"DNCK"
            || (major, minor, profile, auth_kind, reserved, rate) != (2, 0, 0, 0, 0, 48_000)
            || maximum > 3
            || data[16..64].iter().any(|&b| b != 0),
        "invalid preamble",
    )?;

    let read_u64 = |idx: usize| -> u64 {
        let off = V2_PREAMBLE_SIZE + idx * 8;
        u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
    };
    let header_offset = read_u64(0);
    let header_length = read_u64(1);
    let shared_offset = read_u64(2);
    let shared_length = read_u64(3);
    let auth_offset = read_u64(4);
    let auth_length = read_u64(5);
    let res0 = read_u64(6);
    let res1 = read_u64(7);
    let bundle_start = read_u64(8);
    let ends = [read_u64(9), read_u64(10), read_u64(11), read_u64(12)];
    let count = (maximum as usize) + 1;
    let expected_auth_len = (106 + 33 * count) as u64;

    let header_end = header_offset
        .checked_add(header_length)
        .or_schema_err("invalid directory")?;
    let shared_end = shared_offset
        .checked_add(shared_length)
        .or_schema_err("invalid directory")?;
    let auth_end = auth_offset
        .checked_add(auth_length)
        .or_schema_err("invalid directory")?;

    reject_if(
        res0 != 0
            || res1 != 0
            || data[V2_PREAMBLE_SIZE + 13 * 8..V2_HEADER_OFFSET]
                .iter()
                .any(|&b| b != 0)
            || header_offset != 192
            || header_length == 0
            || Some(shared_offset) != align64_u64(header_end)
            || Some(auth_offset) != align64_u64(shared_end)
            || auth_length != expected_auth_len
            || Some(bundle_start) != align64_u64(auth_end)
            || ends[count..].iter().any(|&e| e != 0),
        "invalid directory",
    )?;

    let mut prev = bundle_start;
    let mut issued_ends = Vec::with_capacity(count);
    for &end in &ends[..count] {
        reject_if(end <= prev || !end.is_multiple_of(64), "invalid directory")?;
        let end_usize = usize::try_from(end)
            .ok()
            .or_schema_err("invalid directory")?;
        issued_ends.push(end_usize);
        prev = end;
    }

    let footer = &data[data.len() - DMUE_FOOTER_SIZE..];
    let frame_length_u64 = u64::from_le_bytes(footer[4..12].try_into().unwrap());
    let frame_length = usize::try_from(frame_length_u64)
        .ok()
        .or_schema_err("invalid footer")?;
    let footer_res = u32::from_le_bytes(footer[12..16].try_into().unwrap());
    reject_if(
        &footer[0..4] != b"DMUE"
            || footer_res != 0
            || !(DMU1_HEADER_SIZE + 2..=DEFAULT_MAX_MUTABLE_CAPACITY).contains(&frame_length)
            || frame_length > data.len() - DMUE_FOOTER_SIZE,
        "invalid footer",
    )?;
    let boundary = data.len() - DMUE_FOOTER_SIZE - frame_length;
    let tier = issued_ends
        .iter()
        .position(|&e| e == boundary)
        .or_schema_err("invalid retained boundary")?;

    for (s, e) in [
        (header_end as usize, shared_offset as usize),
        (shared_end as usize, auth_offset as usize),
        (auth_end as usize, bundle_start as usize),
    ] {
        reject_if(
            data[s..e].iter().any(|&b| b != 0),
            "nonzero alignment padding",
        )?;
    }

    let dmu_doc = parse_and_validate_dmu1_frame(&data[boundary..data.len() - DMUE_FOOTER_SIZE])?;
    validate_mutable_json_schema(&dmu_doc)?;

    let a_off = auth_offset as usize;
    let a_len = auth_length as usize;
    let envelope = &data[a_off..a_off + a_len];
    reject_if(
        &envelope[0..8] != b"DCA1\0\0\0\0"
            || envelope[40] != maximum
            || envelope[41] != (count as u8)
            || (0..count).any(|idx| envelope[42 + 33 * idx] != (idx as u8)),
        "invalid envelope",
    )?;

    let mut bundles = Vec::with_capacity(tier + 1);
    let mut b_start = bundle_start as usize;
    for &b_end in &issued_ends[..=tier] {
        bundles.push(&data[b_start..b_end]);
        b_start = b_end;
    }
    let h_off = header_offset as usize;
    let h_len = header_length as usize;
    let s_off = shared_offset as usize;
    let s_len = shared_length as usize;
    let computed = compute_dnck_bundle_prefixes(
        &data[..64],
        &data[64..192],
        &data[h_off..h_off + h_len],
        &data[s_off..s_off + s_len],
        &bundles,
    );
    for (idx, calc_pfx) in computed.iter().enumerate() {
        let expected_prefix = &envelope[43 + 33 * idx..75 + 33 * idx];
        reject_if(
            calc_pfx.as_slice() != expected_prefix,
            "checkpoint prefix mismatch",
        )?;
    }

    Ok((tier as u8, issued_ends, boundary))
}

type Ed25519VerifyFn<'a> = &'a dyn Fn(&[u8; 32], &[u8], &[u8; 64]) -> Result<(), DnamError>;

/// Verifies the `DCA1` `Ed25519` `checkpoint_writer_signature` on a v2.0 `.dnam-ckpt` (`b"DNCK"`)
/// container over `"DNCK-CHECKPOINT-MANIFEST-V1"` (`ref[container.ckpt.dca1-signature]`), returning
/// the verified 32-byte `checkpoint_writer_pubkey_ed25519`.
pub fn verify_v2_dnck_dca1_signature<C: CryptoProvider>(
    data: &[u8],
    crypto: &C,
) -> Result<[u8; 32], DnamError> {
    verify_v2_dnck_dca1_signature_inner(data, &|pk, msg, sig| crypto.ed25519_verify(pk, msg, sig))
}

fn verify_v2_dnck_dca1_signature_inner(
    data: &[u8],
    verify_ed25519: Ed25519VerifyFn<'_>,
) -> Result<[u8; 32], DnamError> {
    validate_v2_dnck_checkpoint(data)?;
    let maximum = data[10];
    let count = (maximum as usize) + 1;
    let h_off = u64::from_le_bytes(data[64..72].try_into().unwrap()) as usize;
    let h_len = u64::from_le_bytes(data[72..80].try_into().unwrap()) as usize;
    let a_off = u64::from_le_bytes(data[64 + 32..64 + 40].try_into().unwrap()) as usize;
    let a_len = u64::from_le_bytes(data[64 + 40..64 + 48].try_into().unwrap()) as usize;
    let envelope = &data[a_off..a_off + a_len];

    let writer_pubkey: [u8; 32] = envelope[8..40].try_into().unwrap();
    let writer_sig: [u8; 64] = envelope[a_len - 64..a_len].try_into().unwrap();
    let mut prefixes = Vec::with_capacity(count);
    for idx in 0..count {
        let pfx: [u8; 32] = envelope[43 + 33 * idx..75 + 33 * idx].try_into().unwrap();
        prefixes.push(pfx);
    }
    let header_sha = sha256_digest(&data[h_off..h_off + h_len]);
    let manifest_msg =
        build_dnck_writer_manifest_message(&writer_pubkey, &header_sha, maximum, &prefixes);
    verify_ed25519(&writer_pubkey, &manifest_msg, &writer_sig)?;
    Ok(writer_pubkey)
}

/// Builds a canonical v2.0 `.dnam-ckpt` (`b"DNCK"`) container with a `DCA1` envelope and embedded
/// `.dnam` runtime template (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`,
/// `ref[container.ckpt.export-template]`).
pub fn build_v2_dnck_checkpoint(
    header_bytes: &[u8],
    runtime_dnam_container: &[u8],
    extra_shared_ckpt_bytes: &[u8],
    tier_bundles: &[&[u8]],
    writer_pubkey: &[u8; 32],
    writer_signature: &[u8; 64],
    mutable_frame_bytes: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    let runtime_layout = validate_v2_open_dnam_layout(runtime_dnam_container)?;
    let count = tier_bundles.len();
    reject_if(
        header_bytes.is_empty()
            || count == 0
            || count > 4
            || runtime_layout.maximum_tier != runtime_layout.retained_tier
            || (count - 1) != (runtime_layout.retained_tier as usize)
            || !extra_shared_ckpt_bytes.len().is_multiple_of(64),
        "invalid v2 checkpoint inputs",
    )?;

    let mut prev_runtime = runtime_layout.tier_start;
    for (idx, bundle) in tier_bundles.iter().enumerate() {
        let runtime_len = runtime_layout.tier_ends[idx].saturating_sub(prev_runtime);
        reject_if(
            bundle.len() < runtime_len || !bundle.len().is_multiple_of(64),
            "invalid checkpoint tier bundle length or alignment",
        )?;
        prev_runtime = runtime_layout.tier_ends[idx];
    }

    let mut shared = Vec::with_capacity(runtime_layout.tier_start + extra_shared_ckpt_bytes.len());
    shared.extend_from_slice(&runtime_dnam_container[..runtime_layout.tier_start]);
    shared.extend_from_slice(extra_shared_ckpt_bytes);

    let maximum_tier = (count - 1) as u8;
    let shared_offset = align64_u64(192 + header_bytes.len() as u64).unwrap();
    let auth_offset = align64_u64(shared_offset + shared.len() as u64).unwrap();
    let auth_length = (106 + 33 * count) as u64;
    let bundle_start = align64_u64(auth_offset + auth_length).unwrap();

    let mut ends = [0u64; 4];
    let mut cur_end = bundle_start;
    for (idx, bundle) in tier_bundles.iter().enumerate() {
        cur_end = cur_end.checked_add(bundle.len() as u64).unwrap();
        ends[idx] = cur_end;
    }

    let mut preamble = [0u8; 64];
    preamble[0..4].copy_from_slice(b"DNCK");
    preamble[4..6].copy_from_slice(&2u16.to_le_bytes());
    preamble[6..8].copy_from_slice(&0u16.to_le_bytes());
    preamble[8] = 0;
    preamble[9] = 0;
    preamble[10] = maximum_tier;
    preamble[11] = 0;
    preamble[12..16].copy_from_slice(&48_000u32.to_le_bytes());

    let mut directory = [0u8; 128];
    let dir_fields: [u64; 13] = [
        192,
        header_bytes.len() as u64,
        shared_offset,
        shared.len() as u64,
        auth_offset,
        auth_length,
        0,
        0,
        bundle_start,
        ends[0],
        ends[1],
        ends[2],
        ends[3],
    ];
    for (idx, val) in dir_fields.iter().enumerate() {
        directory[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
    }

    let digests =
        compute_dnck_bundle_prefixes(&preamble, &directory, header_bytes, &shared, tier_bundles);

    let mut envelope = Vec::with_capacity(auth_length as usize);
    envelope.extend_from_slice(b"DCA1\0\0\0\0");
    envelope.extend_from_slice(writer_pubkey);
    envelope.push(maximum_tier);
    envelope.push(count as u8);
    for (t, d) in digests.iter().enumerate() {
        envelope.push(t as u8);
        envelope.extend_from_slice(d);
    }
    envelope.extend_from_slice(writer_signature);

    let mut ckpt = vec![0u8; ends[count - 1] as usize];
    ckpt[..64].copy_from_slice(&preamble);
    ckpt[64..192].copy_from_slice(&directory);
    ckpt[192..192 + header_bytes.len()].copy_from_slice(header_bytes);
    let s_off = shared_offset as usize;
    ckpt[s_off..s_off + shared.len()].copy_from_slice(&shared);
    let a_off = auth_offset as usize;
    ckpt[a_off..a_off + envelope.len()].copy_from_slice(&envelope);
    let mut b_off = bundle_start as usize;
    for b in tier_bundles {
        ckpt[b_off..b_off + b.len()].copy_from_slice(b);
        b_off += b.len();
    }

    let default_dmu1;
    let m_bytes = match mutable_frame_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            default_dmu1 = build_dmu1_mutable_frame(b"{}", 48)?;
            &default_dmu1
        }
    };
    append_dmue_mutable_suffix(&mut ckpt, m_bytes);
    Ok(ckpt)
}

fn enforce_open_distribution_policy_json(json_slice: &[u8]) -> Result<(), DnamError> {
    let val: serde_json::Value = serde_json::from_slice(json_slice)
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    if let Some(dp_val) = val
        .get("metadata")
        .and_then(|m| m.get("distribution_policy"))
        && !dp_val.is_null()
    {
        let policy: DistributionPolicy = serde_json::from_value(dp_val.clone())
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        if !policy.is_profile_allowed(DnamProtectionProfile::Open)
            || !policy.allowed_oem_vendors.is_empty()
        {
            return Err(DnamError::CryptoVerificationFailed(
                "container protection profile is not permitted by creator distribution_policy",
            ));
        }
    }
    Ok(())
}

fn verify_exported_v2_dau1_if_present(
    bytes: &[u8],
    layout: &V2ContainerLayout,
    verify_ed25519: Ed25519VerifyFn<'_>,
    trusted_marketplace_pubkey: Option<&[u8; 32]>,
) -> Result<(), DnamError> {
    let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    if layout.auth_kind == 0 && !auth_slice.starts_with(b"DAU1") && auth_slice.len() < 139 {
        if let Some(dcrc) = parse_dcrc_record(auth_slice)? {
            let calc_prefix = crc32c_update(0, &bytes[..layout.auth_extent.0]);
            if calc_prefix != dcrc.crc_prefix {
                return Err(DnamError::Crc32cMismatch {
                    region: "header",
                    expected: dcrc.crc_prefix,
                    computed: calc_prefix,
                });
            }
            let ret_idx = usize::from(layout.retained_tier);
            let expected_arena_crc = dcrc.crc_tiers[ret_idx];
            let calc_arena_crc =
                crc32c_update(0, &bytes[layout.tier_start..layout.tier_ends[ret_idx]]);
            if calc_arena_crc != expected_arena_crc {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena",
                    expected: expected_arena_crc,
                    computed: calc_arena_crc,
                });
            }
        }
        return Ok(());
    }
    let env = Dau1Envelope::from_bytes(auth_slice)?;
    if env.auth_kind != layout.auth_kind {
        return Err(DnamError::CryptoVerificationFailed(
            "DAU1 auth_kind mismatch with preamble",
        ));
    }
    let json_slice =
        &bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let shared_slice =
        &bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
    let shared_framing = build_plaintext_shared_v1(&bytes[0..64], &bytes[64..192], shared_slice);
    let hdr_sha = sha256_digest(json_slice);
    let mut tier_slices_vec = Vec::with_capacity((layout.retained_tier as usize) + 1);
    let mut prev_t = layout.tier_start;
    for t_idx in 0..=(layout.retained_tier as usize) {
        let end_t = layout.tier_ends[t_idx];
        tier_slices_vec.push(&bytes[prev_t..end_t]);
        prev_t = end_t;
    }
    if env.auth_kind == 0 {
        if env.creator_max_tier != layout.maximum_tier {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 creator_max_tier mismatch with preamble",
            ));
        }
        let prefixes = compute_plaintext_prefixes_v1(
            b"DNAM-MASTER-PREFIX-V1",
            &hdr_sha,
            &shared_framing,
            &tier_slices_vec,
        )?;
        for (t_idx, pfx) in prefixes.iter().enumerate() {
            if env.master_prefixes.get(t_idx) != Some(pfx) {
                return Err(DnamError::CryptoVerificationFailed(
                    "DAU1 master tier prefix digest mismatch",
                ));
            }
        }
    }
    let creator_manifest =
        build_creator_manifest_v1(&env.creator_pubkey, &hdr_sha, &env.master_prefixes)?;
    verify_ed25519(
        &env.creator_pubkey,
        &creator_manifest,
        &env.creator_signature,
    )?;
    if env.auth_kind == 1 {
        let m_pub = env
            .marketplace_pubkey
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 marketplace_pubkey",
            ))?;
        if let Some(pinned_mp) = trusted_marketplace_pubkey
            && &m_pub != pinned_mp
        {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 marketplace_pubkey mismatch against pinned trusted marketplace key",
            ));
        }
        let b_max = env
            .buyer_max_tier
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 buyer_max_tier",
            ))?;
        if b_max != layout.maximum_tier || b_max > env.creator_max_tier {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 buyer_max_tier mismatch with preamble or creator_max_tier",
            ));
        }
        let m_sig = env
            .marketplace_signature
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 marketplace_signature",
            ))?;
        let buyer_prefixes = compute_plaintext_prefixes_v1(
            b"DNAM-BUYER-PREFIX-V1",
            &hdr_sha,
            &shared_framing,
            &tier_slices_vec,
        )?;
        for (t_idx, bpfx) in buyer_prefixes.iter().enumerate() {
            if env.buyer_prefixes.get(t_idx) != Some(bpfx) {
                return Err(DnamError::CryptoVerificationFailed(
                    "DAU1 buyer tier prefix digest mismatch",
                ));
            }
        }
        let creator_signed_sha =
            compute_creator_signed_manifest_sha256(&creator_manifest, &env.creator_signature);
        let policy_sha = compute_license_policy_sha256(&env.policy_bytes);
        let buyer_receipt = build_buyer_receipt_v1(
            &m_pub,
            &creator_signed_sha,
            &policy_sha,
            layout.protection_profile,
            &env.buyer_prefixes,
            None,
            &[],
        )?;
        verify_ed25519(&m_pub, &buyer_receipt, &m_sig)?;
    }
    Ok(())
}

/// Exports the retained tier of a validated canonical `v2.0` `.dnam-ckpt` (`DNCK`) container into
/// a standalone `.dnam` runtime container using `crypto` to verify `DCA1` and `DAU1` signatures.
pub fn export_runtime_dnam_from_ckpt_with_crypto<C: CryptoProvider>(
    data: &[u8],
    crypto: &C,
) -> Result<Vec<u8>, DnamError> {
    export_v2_dnck_inner(
        data,
        &|pk, msg, sig| crypto.ed25519_verify(pk, msg, sig),
        crypto.marketplace_pubkey(),
    )
}

fn export_v2_dnck_inner(
    data: &[u8],
    verify_ed25519: Ed25519VerifyFn<'_>,
    trusted_marketplace_pubkey: Option<&[u8; 32]>,
) -> Result<Vec<u8>, DnamError> {
    let (tier, ends, boundary) = validate_v2_dnck_checkpoint(data)?;
    verify_v2_dnck_dca1_signature_inner(data, verify_ed25519)?;
    let ckpt_h_off = u64::from_le_bytes(data[64..72].try_into().unwrap()) as usize;
    let ckpt_h_len = u64::from_le_bytes(data[72..80].try_into().unwrap()) as usize;
    enforce_open_distribution_policy_json(&data[ckpt_h_off..ckpt_h_off + ckpt_h_len])?;
    let shared_offset = u64::from_le_bytes(data[64 + 16..64 + 24].try_into().unwrap()) as usize;
    let shared_length = u64::from_le_bytes(data[64 + 24..64 + 32].try_into().unwrap()) as usize;
    let bundle_start = u64::from_le_bytes(data[64 + 64..64 + 72].try_into().unwrap()) as usize;
    reject_if(shared_length < V2_HEADER_OFFSET, "short runtime template")?;
    let template = &data[shared_offset..shared_offset + shared_length];
    let runtime_start = usize::try_from(u64::from_le_bytes(
        template[64 + 64..64 + 72].try_into().unwrap(),
    ))
    .ok()
    .or_schema_err("invalid runtime template")?;
    let mut runtime_ends = [0usize; 4];
    for (i, slot) in runtime_ends.iter_mut().enumerate() {
        let off = 64 + 72 + i * 8;
        *slot = usize::try_from(u64::from_le_bytes(
            template[off..off + 8].try_into().unwrap(),
        ))
        .ok()
        .or_schema_err("invalid runtime template")?;
    }
    reject_if(runtime_start > template.len(), "invalid runtime template")?;

    let mut output = Vec::new();
    output.extend_from_slice(&template[..runtime_start]);
    let mut prev_runtime = runtime_start;
    let mut ckpt_start = bundle_start;
    for idx in 0..=(tier as usize) {
        let runtime_len = runtime_ends[idx]
            .checked_sub(prev_runtime)
            .or_schema_err("invalid runtime slice")?;
        let ckpt_len = ends[idx]
            .checked_sub(ckpt_start)
            .or_schema_err("invalid runtime slice")?;
        reject_if(
            runtime_len == 0 || !runtime_len.is_multiple_of(64) || runtime_len > ckpt_len,
            "invalid runtime slice",
        )?;
        output.extend_from_slice(&data[ckpt_start..ckpt_start + runtime_len]);
        prev_runtime = runtime_ends[idx];
        ckpt_start = ends[idx];
    }
    output.extend_from_slice(&data[boundary..]);
    let exported_layout = validate_v2_open_dnam_layout(&output)?;
    let max_ok = if exported_layout.auth_kind == 1 {
        exported_layout.maximum_tier <= data[10]
    } else {
        exported_layout.maximum_tier == data[10]
    };
    reject_if(
        exported_layout.retained_tier != tier || !max_ok,
        "runtime tier mismatch",
    )?;
    enforce_open_distribution_policy_json(
        &output[exported_layout.header_extent.0
            ..exported_layout.header_extent.0 + exported_layout.header_extent.1],
    )?;
    verify_exported_v2_dau1_if_present(
        &output,
        &exported_layout,
        verify_ed25519,
        trusted_marketplace_pubkey,
    )?;
    Ok(output)
}

/// Exports the retained tier of a validated `.dnam-ckpt` (`DNCK`) container into a standalone
/// `.dnam` runtime container (`ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`).
pub fn export_runtime_dnam_from_ckpt(data: &[u8]) -> Result<Vec<u8>, DnamError> {
    export_runtime_dnam_from_ckpt_with_crypto(data, &DefaultCryptoProvider::default())
}

/// Strips a `.dnam-ckpt` (`DNCK`) checkpoint container down to `max_tier`, preserving `DCA1`
/// prefix commitments and `DMUE` mutable suffix without re-signing on v2 checkpoints.
pub fn strip_dnck_checkpoint_to_tier(data: &[u8], max_tier: Tier) -> Result<Vec<u8>, DnamError> {
    strip_dnck_checkpoint_to_tier_with_crypto(data, max_tier, &DefaultCryptoProvider::default())
}

/// Strips a canonical `v2.0` `.dnam-ckpt` (`DNCK`) checkpoint container down to `max_tier` using
/// `crypto` to verify `DCA1` and `DAU1` (including pinned marketplace public keys when configured).
pub fn strip_dnck_checkpoint_to_tier_with_crypto<C: CryptoProvider>(
    data: &[u8],
    max_tier: Tier,
    crypto: &C,
) -> Result<Vec<u8>, DnamError> {
    let (ret_tier, ends, boundary) = validate_v2_dnck_checkpoint(data)?;
    export_v2_dnck_inner(
        data,
        &|pk, msg, sig| crypto.ed25519_verify(pk, msg, sig),
        crypto.marketplace_pubkey(),
    )?;
    let target = usize::from(ret_tier.min(max_tier as u8));
    let mut stripped = data[..ends[target]].to_vec();
    stripped.extend_from_slice(&data[boundary..]);
    Ok(stripped)
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::ckpt::{
        ARENA4_STEM_BYTES_FULL, ARENA4_STEM_BYTES_LITE, ARENA4_STEM_BYTES_PRO,
        ARENA4_STEM_BYTES_ULTRA,
    };
    use dnam_engine::container::build_v2_open_dnam_container;

    /// verify[container.ckpt.bundle-framing]
    /// verify[container.ckpt.dca1-signature]
    /// verify[container.ckpt.export-template]
    /// verify[container.ckpt.declarative-no-pickle]
    /// verify[container.ckpt.uplo-u]
    #[test]
    fn test_v2_dnck_checkpoint_bundle_framing_dca1_export_and_uplo_u() {
        let header = br#"{"version":1}"#;
        let bundles_owned: Vec<Vec<u8>> = (0..4u8)
            .map(|t| [[0x41 + t; 64], [0x61 + t; 64], [0x31 + t; 64]].concat())
            .collect();
        let runtime_slices: Vec<&[u8]> = bundles_owned.iter().map(|b| &b[..64]).collect();
        let runtime = build_v2_open_dnam_container(
            header,
            &[b'S'; 64],
            &[b'E'; 16],
            &runtime_slices,
            None,
            0,
        )
        .unwrap();
        let runtime_layout = validate_v2_open_dnam_layout(&runtime).unwrap();
        let mut shared = runtime[..runtime_layout.tier_start].to_vec();
        shared.extend_from_slice(&[b'C'; 64]);

        let count = 4usize;
        let shared_offset = align64_u64(192 + header.len() as u64).unwrap();
        let auth_offset = align64_u64(shared_offset + shared.len() as u64).unwrap();
        let auth_length = (106 + 33 * count) as u64;
        let bundle_start = align64_u64(auth_offset + auth_length).unwrap();
        let ends: [u64; 4] = core::array::from_fn(|t| bundle_start + 192 * ((t as u64) + 1));

        let mut preamble = [0u8; 64];
        preamble[0..4].copy_from_slice(b"DNCK");
        preamble[4..6].copy_from_slice(&2u16.to_le_bytes());
        preamble[10] = 3;
        preamble[12..16].copy_from_slice(&48_000u32.to_le_bytes());

        let mut directory = [0u8; 128];
        for (i, v) in [
            192,
            header.len() as u64,
            shared_offset,
            shared.len() as u64,
            auth_offset,
            auth_length,
            0,
            0,
            bundle_start,
            ends[0],
            ends[1],
            ends[2],
            ends[3],
        ]
        .into_iter()
        .enumerate()
        {
            directory[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        let bundle_refs: Vec<&[u8]> = bundles_owned.iter().map(Vec::as_slice).collect();
        let digests =
            compute_dnck_bundle_prefixes(&preamble, &directory, header, &shared, &bundle_refs);
        let expected_hex = [
            "71979cd5235f0d914d99f79adcd3a3184e382b8019ba86e39891fa4ad9531675",
            "3d9055890214c04d532c434a058c93f6d09a94e8f428aad9b1299232df3a7e68",
            "46d550042a1a5b45e2f359b7d732b7ce5bb83055eb330ecd4a35a21309bf4013",
            "d44970efb24b4f5a79ffb4e9b987c6527e79c61de5784032f9628b0e24b636c0",
        ];
        for (d, exp_h) in digests.iter().zip(expected_hex) {
            assert_eq!(
                d.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                exp_h
            );
        }

        let seed = [0x4B_u8; 32];
        let writer_pk =
            dnam_engine::provenance::RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
        let writer_msg =
            build_dnck_writer_manifest_message(&writer_pk, &sha256_digest(header), 3, &digests);
        assert!(writer_msg.starts_with(&encode_domain_field(
            "domain",
            b"DNCK-CHECKPOINT-MANIFEST-V1"
        )));
        let writer_sig =
            dnam_engine::provenance::RustCryptoProvider::ed25519_sign(&seed, &writer_msg);

        let ckpt_frame =
            build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"amp"}}"#, 47).unwrap();
        let ckpt = build_v2_dnck_checkpoint(
            header,
            &runtime,
            &[b'C'; 64],
            &bundle_refs,
            &writer_pk,
            &writer_sig,
            Some(&ckpt_frame),
        )
        .unwrap();
        let suffix = ckpt[ends[3] as usize..].to_vec();

        for tier in 0..4usize {
            let stripped_ckpt =
                strip_dnck_checkpoint_to_tier(&ckpt, Tier::from_u8(tier as u8)).unwrap();
            let (ret_tier, _, boundary) = validate_v2_dnck_checkpoint(&stripped_ckpt).unwrap();
            assert_eq!(ret_tier, tier as u8);
            assert_eq!(boundary, ends[tier] as usize);

            let exported_dnam = export_runtime_dnam_from_ckpt(&stripped_ckpt).unwrap();
            let mut expected_dnam = runtime[..runtime_layout.tier_ends[tier]].to_vec();
            expected_dnam.extend_from_slice(&suffix);
            assert_eq!(exported_dnam, expected_dnam);
        }
        let mut forged_ckpt = ckpt.clone();
        let sig_byte_idx = (auth_offset + auth_length) as usize - 1;
        forged_ckpt[sig_byte_idx] ^= 0xFF;
        assert!(export_runtime_dnam_from_ckpt(&forged_ckpt).is_err());
        assert!(strip_dnck_checkpoint_to_tier(&forged_ckpt, Tier::Lite).is_err());
        assert_eq!(ARENA4_STEM_BYTES_LITE, 17_664);
        assert_eq!(ARENA4_STEM_BYTES_FULL, 68_096);
        assert_eq!(ARENA4_STEM_BYTES_PRO, 151_296);
        assert_eq!(ARENA4_STEM_BYTES_ULTRA, 267_264);
        assert!(is_valid_arena4_stem_slice_len(
            3 * 267_264,
            3,
            ARENA4_STEM_BYTES_ULTRA
        ));
    }
}
