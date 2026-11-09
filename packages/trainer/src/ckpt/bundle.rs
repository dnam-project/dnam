#![forbid(unsafe_code)]
//! `.dnam-ckpt` v2.0 Bundle Framing, `DCA1` Signature, and Zero-Copy Runtime Export
//! (`ref[container.ckpt.bundle-framing]`, `ref[container.ckpt.dca1-signature]`,
//! `ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`,
//! `ref[container.ckpt.uplo-u]`).

use dnam_engine::Tier;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DMUE_FOOTER_SIZE, DNAM_MAGIC, V2_HEADER_OFFSET, V2_PREAMBLE_SIZE,
    align64_u64, append_dmue_mutable_suffix, build_dmu1_mutable_frame, build_dnam_container,
    encode_domain_field, parse_and_validate_dmu1_frame, validate_v2_open_dnam_layout,
};
use dnam_engine::provenance::{OpenArenaCryptoProvider, sha256_digest};
use dnam_engine::schema::{DnamError, SchemaOptionExt, reject_if};

pub use super::view::{
    DnamCkptView, build_dnam_ckpt_container, is_valid_arena4_stem_slice_len, resolve_ckpt_stems_end,
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

    reject_if(
        res0 != 0
            || res1 != 0
            || data[V2_PREAMBLE_SIZE + 13 * 8..V2_HEADER_OFFSET]
                .iter()
                .any(|&b| b != 0)
            || header_offset != 192
            || header_length == 0
            || Some(shared_offset) != align64_u64(header_offset + header_length)
            || Some(auth_offset) != align64_u64(shared_offset + shared_length)
            || auth_length != expected_auth_len
            || Some(bundle_start) != align64_u64(auth_offset + auth_length)
            || ends[count..].iter().any(|&e| e != 0),
        "invalid directory",
    )?;

    let mut prev = bundle_start;
    let mut issued_ends = Vec::with_capacity(count);
    for &end in &ends[..count] {
        reject_if(end <= prev || !end.is_multiple_of(64), "invalid directory")?;
        issued_ends.push(end as usize);
        prev = end;
    }

    for (s, e) in [
        (
            (header_offset + header_length) as usize,
            shared_offset as usize,
        ),
        (
            (shared_offset + shared_length) as usize,
            auth_offset as usize,
        ),
        ((auth_offset + auth_length) as usize, bundle_start as usize),
    ] {
        reject_if(
            data[s..e].iter().any(|&b| b != 0),
            "nonzero alignment padding",
        )?;
    }

    let footer = &data[data.len() - DMUE_FOOTER_SIZE..];
    let frame_length = u64::from_le_bytes(footer[4..12].try_into().unwrap()) as usize;
    let footer_res = u32::from_le_bytes(footer[12..16].try_into().unwrap());
    reject_if(
        &footer[0..4] != b"DMUE" || footer_res != 0 || frame_length > data.len() - DMUE_FOOTER_SIZE,
        "invalid footer",
    )?;
    let boundary = data.len() - DMUE_FOOTER_SIZE - frame_length;
    let tier = issued_ends
        .iter()
        .position(|&e| e == boundary)
        .or_schema_err("invalid retained boundary")?;

    parse_and_validate_dmu1_frame(&data[boundary..data.len() - DMUE_FOOTER_SIZE])?;

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
    let expected_prefix = &envelope[43 + 33 * tier..75 + 33 * tier];
    reject_if(
        computed[tier].as_slice() != expected_prefix,
        "checkpoint prefix mismatch",
    )?;

    Ok((tier as u8, issued_ends, boundary))
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

/// Exports the retained tier of a validated `.dnam-ckpt` (`DNCK`) container into a standalone
/// `.dnam` runtime container (`ref[container.ckpt.export-template]`, `ref[container.ckpt.declarative-no-pickle]`).
pub fn export_runtime_dnam_from_ckpt(data: &[u8]) -> Result<Vec<u8>, DnamError> {
    if data.len() >= 64 && &data[0..4] == b"DNCK" && data[16..64].iter().any(|&b| b != 0) {
        let view = DnamCkptView::from_bytes(data, &OpenArenaCryptoProvider)?;
        let mut header = view.header.base.clone();
        header.version = "2.0.0".to_string();
        header.container_kind = "dnam".to_string();
        header.tensor_directory.retain(|_, entry| entry.arena < 4);
        header.metadata.signatures = None;

        let full_arena = view.arena_storage.as_bytes();
        let runtime_end = (view.preamble.arena_offset_ultra_end() as usize)
            .max(ARENA_OFFSET_ULTRA_END as usize)
            .min(full_arena.len());
        let runtime_arena = &full_arena[..runtime_end];
        return build_dnam_container(&mut header, runtime_arena, runtime_arena, false);
    }

    let (tier, ends, boundary) = validate_v2_dnck_checkpoint(data)?;
    let shared_offset = u64::from_le_bytes(data[64 + 16..64 + 24].try_into().unwrap()) as usize;
    let shared_length = u64::from_le_bytes(data[64 + 24..64 + 32].try_into().unwrap()) as usize;
    let bundle_start = u64::from_le_bytes(data[64 + 64..64 + 72].try_into().unwrap()) as usize;
    reject_if(shared_length < V2_HEADER_OFFSET, "short runtime template")?;
    let template = &data[shared_offset..shared_offset + shared_length];
    let runtime_start = u64::from_le_bytes(template[64 + 64..64 + 72].try_into().unwrap()) as usize;
    let runtime_ends: [usize; 4] = core::array::from_fn(|i| {
        let off = 64 + 72 + i * 8;
        u64::from_le_bytes(template[off..off + 8].try_into().unwrap()) as usize
    });
    reject_if(runtime_start > template.len(), "invalid runtime template")?;

    let mut output = Vec::new();
    output.extend_from_slice(&template[..runtime_start]);
    let mut prev_runtime = runtime_start;
    let mut ckpt_start = bundle_start;
    for idx in 0..=(tier as usize) {
        let runtime_len = runtime_ends[idx].saturating_sub(prev_runtime);
        let ckpt_len = ends[idx].saturating_sub(ckpt_start);
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
    reject_if(
        exported_layout.retained_tier != tier,
        "runtime tier mismatch",
    )?;
    let _ = Tier::from_u8(tier);
    let _ = DNAM_MAGIC;
    Ok(output)
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::ckpt::{
        ARENA4_STEM_BYTES_FULL, ARENA4_STEM_BYTES_LITE, ARENA4_STEM_BYTES_PRO,
        ARENA4_STEM_BYTES_ULTRA, pack_uplo_u_f64, unpack_uplo_u_f64, uplo_u_packed_len,
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
            .map(|tier| {
                let mut b = Vec::with_capacity(192);
                b.extend_from_slice(&[0x41 + tier; 64]);
                b.extend_from_slice(&[0x61 + tier; 64]);
                b.extend_from_slice(&[0x31 + tier; 64]);
                b
            })
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
        let dir_fields: [u64; 13] = [
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
        ];
        for (idx, val) in dir_fields.iter().enumerate() {
            directory[idx * 8..(idx + 1) * 8].copy_from_slice(&val.to_le_bytes());
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
        for (d, exp_h) in digests.iter().zip(expected_hex.iter()) {
            let hex_str: String = d.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex_str, *exp_h);
        }

        let mut envelope = Vec::with_capacity(auth_length as usize);
        envelope.extend_from_slice(b"DCA1\0\0\0\0");
        envelope.extend_from_slice(&[b'K'; 32]);
        envelope.push(3);
        envelope.push(4);
        for (t, d) in digests.iter().enumerate() {
            envelope.push(t as u8);
            envelope.extend_from_slice(d);
        }
        envelope.extend_from_slice(&[b'X'; 64]);

        let writer_msg = build_dnck_writer_manifest_message(&[b'K'; 32], &[0x11; 32], 3, &digests);
        assert!(writer_msg.starts_with(&encode_domain_field(
            "domain",
            b"DNCK-CHECKPOINT-MANIFEST-V1"
        )));

        let mut ckpt = vec![0u8; ends[3] as usize];
        ckpt[..64].copy_from_slice(&preamble);
        ckpt[64..192].copy_from_slice(&directory);
        ckpt[192..192 + header.len()].copy_from_slice(header);
        let s_off = shared_offset as usize;
        ckpt[s_off..s_off + shared.len()].copy_from_slice(&shared);
        let a_off = auth_offset as usize;
        ckpt[a_off..a_off + envelope.len()].copy_from_slice(&envelope);
        let mut b_off = bundle_start as usize;
        for b in &bundles_owned {
            ckpt[b_off..b_off + b.len()].copy_from_slice(b);
            b_off += b.len();
        }
        let ckpt_frame =
            build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"amp"}}"#, 47).unwrap();
        append_dmue_mutable_suffix(&mut ckpt, &ckpt_frame);
        let suffix = ckpt[ends[3] as usize..].to_vec();

        for tier in 0..4usize {
            let mut stripped_ckpt = ckpt[..ends[tier] as usize].to_vec();
            stripped_ckpt.extend_from_slice(&suffix);
            let (ret_tier, _, boundary) = validate_v2_dnck_checkpoint(&stripped_ckpt).unwrap();
            assert_eq!(ret_tier, tier as u8);
            assert_eq!(boundary, ends[tier] as usize);

            let exported_dnam = export_runtime_dnam_from_ckpt(&stripped_ckpt).unwrap();
            let mut expected_dnam = runtime[..runtime_layout.tier_ends[tier]].to_vec();
            expected_dnam.extend_from_slice(&suffix);
            assert_eq!(exported_dnam, expected_dnam);
        }

        assert_eq!(ARENA4_STEM_BYTES_LITE, 17_664);
        assert_eq!(ARENA4_STEM_BYTES_FULL, 68_096);
        assert_eq!(ARENA4_STEM_BYTES_PRO, 151_296);
        assert_eq!(ARENA4_STEM_BYTES_ULTRA, 267_264);
        assert!(is_valid_arena4_stem_slice_len(
            ARENA4_STEM_BYTES_ULTRA,
            3,
            ARENA4_STEM_BYTES_ULTRA
        ));
        assert!(is_valid_arena4_stem_slice_len(
            3 * ARENA4_STEM_BYTES_ULTRA,
            3,
            ARENA4_STEM_BYTES_ULTRA
        ));
        assert!(!is_valid_arena4_stem_slice_len(
            2 * ARENA4_STEM_BYTES_ULTRA,
            3,
            ARENA4_STEM_BYTES_ULTRA
        ));

        let n_ultra = 256usize;
        let mut dense = vec![0.0_f64; n_ultra * n_ultra];
        for i in 0..n_ultra {
            for j in i..n_ultra {
                let val = ((i + 1) * 1000 + (j + 1)) as f64;
                dense[i * n_ultra + j] = val;
                dense[j * n_ultra + i] = val;
            }
        }
        let mut packed = vec![0.0_f64; uplo_u_packed_len(n_ultra)];
        pack_uplo_u_f64(&dense, n_ultra, &mut packed);

        for n_tier in [64usize, 128, 192, 256] {
            let mut unpacked = vec![0.0_f64; n_tier * n_tier];
            unpack_uplo_u_f64(&packed[..uplo_u_packed_len(n_tier)], n_tier, &mut unpacked);
            for i in 0..n_tier {
                for j in 0..n_tier {
                    assert_eq!(unpacked[i * n_tier + j], dense[i * n_ultra + j]);
                }
            }
        }

        // Export runtime .dnam from v1 page-aligned DNCK container
        let mut ckpt_hdr = crate::ckpt::DnamCkptHeader::default_reference();
        let arena_ultra = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let v1_ckpt = crate::ckpt::build_dnam_ckpt_container(
            &mut ckpt_hdr,
            &arena_ultra,
            &arena_ultra,
            false,
        )
        .unwrap();
        let exported_v1 = export_runtime_dnam_from_ckpt(&v1_ckpt).unwrap();
        assert_eq!(&exported_v1[0..4], b"DNAM");

        // Error branches in build_v2_dnck_checkpoint
        assert!(
            build_v2_dnck_checkpoint(
                b"",
                &runtime,
                &[0u8; 64],
                &bundle_refs,
                &[0u8; 32],
                &[0u8; 64],
                None
            )
            .is_err()
        );
        assert!(
            build_v2_dnck_checkpoint(
                header,
                &runtime,
                &[0u8; 63],
                &bundle_refs,
                &[0u8; 32],
                &[0u8; 64],
                None
            )
            .is_err()
        );
    }
}
