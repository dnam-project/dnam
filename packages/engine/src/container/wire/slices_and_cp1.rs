//! Model-Declared Delay Histories, Stage III Separate 64B-Aligned Slices, and `CP1` Packed Control
//! Projections (`ref[container.dnam.delay-histories]`, `ref[container.dnam.load-validation]`,
//! `ref[container.dnam.stage3-slices]`, `ref[container.dnam.cp1-profile]`).

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::Value;

use crate::schema::{DnamError, SchemaOptionExt, reject_if};

/// Magic bytes for a packed `CP1\0` control projection contributor stream in `shared_extent`.
pub const CP1_MAGIC: [u8; 4] = *b"CP1\0";

/// Validated `delay_histories` declaration (`ref[container.dnam.delay-histories]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredDelayHistory {
    pub id: String,
    pub stage: String,
    pub kind: String,
    pub introduced_tier: u8,
    pub sample_format: String,
    /// Per-tier reference sample counts at `48 kHz` (`0` for tiers below `introduced_tier`).
    pub reference_samples_by_tier: [u64; 4],
}

/// Parses a canonical decimal string (`[1-9][0-9]*`) as a non-zero `u64` (`ref[container.dnam.delay-histories]`).
pub fn parse_canonical_reference_samples(text: &str) -> Result<u64, DnamError> {
    let bytes = text.as_bytes();
    reject_if(
        bytes.is_empty() || !bytes[0].is_ascii_digit() || bytes[0] == b'0',
        "noncanonical unsigned count",
    )?;
    reject_if(
        bytes.iter().any(|b| !b.is_ascii_digit()),
        "noncanonical unsigned count",
    )?;
    text.parse::<u64>()
        .map_err(|_| DnamError::SchemaValidationError("count exceeds u64".to_string()))
}

fn tier_name_to_index(name: &str) -> Option<u8> {
    match name {
        "lite" => Some(0),
        "full" => Some(1),
        "pro" => Some(2),
        "ultra" => Some(3),
        _ => None,
    }
}

/// Validates and parses the `delay_histories` array for a container with `max_tier` (`0..=3`,
/// `ref[container.dnam.delay-histories]`, `ref[dsp.op-t.ring.declaration]`).
pub fn validate_delay_histories(
    entries: &[Value],
    max_tier: u8,
) -> Result<Vec<DeclaredDelayHistory>, DnamError> {
    reject_if(max_tier > 3, "invalid max_tier")?;
    let tier_names = ["lite", "full", "pro", "ultra"];
    let mut seen_ids = BTreeSet::new();
    let mut out = Vec::with_capacity(entries.len());

    for item in entries {
        let obj = item
            .as_object()
            .or_schema_err("delay history must be an object")?;
        reject_if(obj.len() != 6, "invalid delay history fields")?;
        let id = obj.get("id").and_then(Value::as_str).unwrap_or("");
        reject_if(
            id.is_empty() || !seen_ids.insert(id.to_string()),
            "duplicate or empty delay history id",
        )?;
        let stage = obj.get("stage").and_then(Value::as_str).unwrap_or("");
        reject_if(
            !matches!(stage, "stage1_pre" | "stage4_tank"),
            "invalid delay history stage",
        )?;
        let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
        reject_if(
            !matches!(kind, "fractional" | "allpass"),
            "invalid delay history kind",
        )?;
        let intro_str = obj
            .get("introduced_tier")
            .and_then(Value::as_str)
            .unwrap_or("");
        let introduced_tier =
            tier_name_to_index(intro_str).or_schema_err("invalid introduced_tier")?;
        reject_if(
            introduced_tier > max_tier,
            "introduced_tier exceeds max_tier",
        )?;
        let sample_format = obj
            .get("sample_format")
            .and_then(Value::as_str)
            .unwrap_or("");
        reject_if(sample_format != "fp16", "invalid sample_format")?;
        let ref_map = obj
            .get("reference_samples")
            .and_then(Value::as_object)
            .or_schema_err("invalid reference_samples")?;

        let expected_tier_count = (max_tier - introduced_tier + 1) as usize;
        reject_if(
            ref_map.len() != expected_tier_count,
            "mismatched reference_samples tier keys",
        )?;
        let mut ref_by_tier = [0u64; 4];
        for t in (introduced_tier as usize)..=(max_tier as usize) {
            let s = ref_map
                .get(tier_names[t])
                .and_then(Value::as_str)
                .or_schema_err("missing tier sample count")?;
            let count = parse_canonical_reference_samples(s)?;
            reject_if(
                kind == "fractional" && count < 4,
                "fractional delay reference_samples must be >= 4",
            )?;
            ref_by_tier[t] = count;
        }
        out.push(DeclaredDelayHistory {
            id: id.to_string(),
            stage: stage.to_string(),
            kind: kind.to_string(),
            introduced_tier,
            sample_format: sample_format.to_string(),
            reference_samples_by_tier: ref_by_tier,
        });
    }
    Ok(out)
}

/// Validates that a Stage III early FIR slice (`128 B` per tier) and Kautz modal slice
/// (`128 B * {1, 1, 2, 4}` blocks per tier) are 64-byte aligned, non-overlapping, within their
/// tier bounds, and contain only finite `f32` values (`ref[container.dnam.stage3-slices]`,
/// `ref[container.dnam.load-validation]`).
pub fn validate_stage3_tier_slices(
    tier_index: u8,
    tier_bytes: &[u8],
    early_fir_offset_in_tier: usize,
    early_fir_len: usize,
    kautz_offset_in_tier: usize,
    kautz_len: usize,
) -> Result<(), DnamError> {
    let expected_kautz_blocks = match tier_index {
        0 => 1usize,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => {
            return Err(DnamError::SchemaValidationError(
                "invalid tier_index".to_string(),
            ));
        }
    };
    let expected_kautz_len = expected_kautz_blocks * 128;
    reject_if(
        early_fir_len != 128 || kautz_len != expected_kautz_len,
        "invalid Stage III slice byte length",
    )?;
    reject_if(
        !early_fir_offset_in_tier.is_multiple_of(64) || !kautz_offset_in_tier.is_multiple_of(64),
        "misaligned Stage III slice offset",
    )?;
    let fir_end = early_fir_offset_in_tier + early_fir_len;
    let kautz_end = kautz_offset_in_tier + kautz_len;
    reject_if(
        fir_end > tier_bytes.len() || kautz_end > tier_bytes.len(),
        "Stage III slice outside tier",
    )?;
    reject_if(
        early_fir_offset_in_tier < kautz_end && kautz_offset_in_tier < fir_end,
        "overlapping Stage III slices",
    )?;
    for slice in [
        &tier_bytes[early_fir_offset_in_tier..fir_end],
        &tier_bytes[kautz_offset_in_tier..kautz_end],
    ] {
        for &chunk in slice.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(chunk);
            reject_if(!v.is_finite(), "non-finite Stage III coefficient")?;
        }
    }
    Ok(())
}

/// Returns the byte length of a single `CP1` contributor record with `lanes` (`1..=3`):
/// `4 * (9 * lanes + 4)` in `Arena 0` (`ultra == false`) or `4 * (8 * lanes + 4)` in `Arena 3`
/// (`ultra == true`, `ref[container.dnam.cp1-profile]`).
#[inline]
pub const fn cp1_contributor_record_bytes(lanes: usize, ultra_delta: bool) -> Option<usize> {
    if lanes < 1 || lanes > 3 {
        None
    } else if ultra_delta {
        Some(4 * (8 * lanes + 4))
    } else {
        Some(4 * (9 * lanes + 4))
    }
}

/// Decoded `CP1` contributor record (`ref[container.dnam.cp1-profile]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Cp1DecodedRecord {
    pub lanes: usize,
    pub base_main: Vec<f32>,
    pub embedding: Vec<f32>,
    pub base_factors: [f32; 4],
    pub extra_main: Vec<f32>,
    pub extra_factors: Option<[f32; 4]>,
}

/// Decodes a stream of packed `CP1` contributor records from `base_bytes` (`Arena 0`) and optional
/// `ultra_delta_bytes` (`Arena 3`) for the given sequence of contributor lane counts (`ref[container.dnam.cp1-profile]`).
pub fn decode_cp1_contributor_stream(
    base_bytes: &[u8],
    ultra_delta_bytes: Option<&[u8]>,
    contributor_lanes: &[usize],
) -> Result<Vec<Cp1DecodedRecord>, DnamError> {
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for &lanes in contributor_lanes {
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
    }
    reject_if(
        base_bytes.len() != expected_base,
        "incorrect base tensor length",
    )?;
    if let Some(ub) = ultra_delta_bytes
        && ub.len() != expected_ultra
    {
        return Err(DnamError::SchemaValidationError(
            "incorrect delta tensor length".to_string(),
        ));
    }

    let read_f32_vec = |bytes: &[u8]| -> Result<Vec<f32>, DnamError> {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for &chunk in bytes.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(chunk);
            reject_if(!v.is_finite(), "nonfinite coefficient")?;
            out.push(v);
        }
        Ok(out)
    };

    let mut records = Vec::with_capacity(contributor_lanes.len());
    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for &lanes in contributor_lanes {
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_vals = read_f32_vec(&base_bytes[b_off..b_off + b_len])?;
        b_off += b_len;
        let main_end = 8 * lanes;
        let base_main = b_vals[..main_end].to_vec();
        let embedding = b_vals[main_end..main_end + lanes].to_vec();
        let base_factors = [
            b_vals[main_end + lanes],
            b_vals[main_end + lanes + 1],
            b_vals[main_end + lanes + 2],
            b_vals[main_end + lanes + 3],
        ];
        let (extra_main, extra_factors) = if let Some(ub) = ultra_delta_bytes {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_vals = read_f32_vec(&ub[u_off..u_off + u_len])?;
            u_off += u_len;
            (
                u_vals[..main_end].to_vec(),
                Some([
                    u_vals[main_end],
                    u_vals[main_end + 1],
                    u_vals[main_end + 2],
                    u_vals[main_end + 3],
                ]),
            )
        } else {
            (Vec::new(), None)
        };
        records.push(Cp1DecodedRecord {
            lanes,
            base_main,
            embedding,
            base_factors,
            extra_main,
            extra_factors,
        });
    }
    Ok(records)
}

/// Decodes a stream of packed `CP1` contributor records directly into a caller-supplied pre-allocated
/// [`crate::control::Cp1ContributorSlot`] slice without heap allocation (`ref[container.dnam.cp1-profile]`).
pub fn decode_cp1_contributor_stream_into_slice(
    base_bytes: &[u8],
    ultra_delta_bytes: Option<&[u8]>,
    contributor_lanes: &[usize],
    out_slots: &mut [crate::control::Cp1ContributorSlot],
) -> Result<usize, DnamError> {
    reject_if(
        contributor_lanes.len() > out_slots.len(),
        "contributor count exceeds slot capacity",
    )?;
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for &lanes in contributor_lanes {
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
    }
    reject_if(
        base_bytes.len() < expected_base,
        "incorrect base tensor length",
    )?;
    if let Some(ub) = ultra_delta_bytes
        && ub.len() < expected_ultra
    {
        return Err(DnamError::SchemaValidationError(
            "incorrect delta tensor length".to_string(),
        ));
    }

    let read_f32_at = |bytes: &[u8], idx: usize| -> Result<f32, DnamError> {
        let off = idx * 4;
        let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        reject_if(!v.is_finite(), "nonfinite coefficient")?;
        Ok(v)
    };

    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for (slot_idx, &lanes) in contributor_lanes.iter().enumerate() {
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_slice = &base_bytes[b_off..b_off + b_len];
        b_off += b_len;

        let mut slot = crate::control::Cp1ContributorSlot::zero();
        slot.lanes = lanes as u8;
        slot.control_idx = slot_idx as u16;
        slot.option_idx = 0xFF;

        for k in 0..lanes {
            for r in 0..8 {
                slot.main_base[k][r] = read_f32_at(b_slice, k * 8 + r)?;
            }
            slot.emb[k] = read_f32_at(b_slice, 8 * lanes + k)?;
        }
        let v_start = 9 * lanes;
        for f in 0..4 {
            slot.v_base[f] = read_f32_at(b_slice, v_start + f)?;
        }

        if let Some(ub) = ultra_delta_bytes {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_slice = &ub[u_off..u_off + u_len];
            u_off += u_len;
            for k in 0..lanes {
                for r in 0..8 {
                    slot.main_ultra[k][r] = read_f32_at(u_slice, k * 8 + r)?;
                }
            }
            let uv_start = 8 * lanes;
            for f in 0..4 {
                slot.v_ultra[f] = read_f32_at(u_slice, uv_start + f)?;
            }
            slot.has_ultra = true;
        }
        out_slots[slot_idx] = slot;
    }
    Ok(contributor_lanes.len())
}

/// Decodes a self-describing `CP1\0` contributor stream from `shared_extent` into `out_slots`
/// without heap allocation (`ref[container.dnam.cp1-profile]`).
///
/// Framing: `b"CP1\0"` (`4 B`), `count: u16le` (`2 B`), `flags: u8` (`1 B`, bit 0 = `has_ultra`),
/// `reserved: u8` (`1 B`), followed by `count * 4 B` contributor directory entries
/// `(control_idx: u16le, lanes: u8, option_idx: u8)`, followed by packed `base_bytes` and optional `ultra_bytes`.
pub fn decode_cp1_shared_stream_into_slice(
    cp1_slice: &[u8],
    out_slots: &mut [crate::control::Cp1ContributorSlot],
) -> Result<usize, DnamError> {
    if cp1_slice.len() < 8 || cp1_slice[0..4] != CP1_MAGIC {
        return Ok(0);
    }
    let count = u16::from_le_bytes([cp1_slice[4], cp1_slice[5]]) as usize;
    if count == 0 {
        return Ok(0);
    }
    let has_ultra = (cp1_slice[6] & 1) != 0;
    let dir_end = 8 + count * 4;
    reject_if(
        count > out_slots.len() || cp1_slice.len() < dir_end,
        "invalid CP1 shared stream header",
    )?;
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for i in 0..count {
        let raw_kl = cp1_slice[8 + i * 4 + 2];
        let lanes = (raw_kl & 0x03) as usize;
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        if has_ultra {
            expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
        }
    }
    reject_if(
        cp1_slice.len() < dir_end + expected_base + expected_ultra,
        "truncated CP1 shared stream payload",
    )?;
    let base_bytes = &cp1_slice[dir_end..dir_end + expected_base];
    let ultra_bytes = &cp1_slice[dir_end + expected_base..dir_end + expected_base + expected_ultra];
    let read_f32_at = |bytes: &[u8], idx: usize| -> Result<f32, DnamError> {
        let off = idx * 4;
        let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        reject_if(!v.is_finite(), "nonfinite coefficient")?;
        Ok(v)
    };
    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for (slot_idx, out_slot) in out_slots.iter_mut().take(count).enumerate() {
        let d_off = 8 + slot_idx * 4;
        let control_idx = u16::from_le_bytes([cp1_slice[d_off], cp1_slice[d_off + 1]]);
        let raw_kl = cp1_slice[d_off + 2];
        let lanes = (raw_kl & 0x03) as usize;
        let ctrl_kind = (raw_kl >> 2) & 0x03;
        let aux = cp1_slice[d_off + 3];
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_slice = &base_bytes[b_off..b_off + b_len];
        b_off += b_len;

        let mut slot = crate::control::Cp1ContributorSlot::zero();
        slot.lanes = lanes as u8;
        slot.ctrl_kind = ctrl_kind;
        slot.control_idx = control_idx;
        if ctrl_kind == 2 {
            slot.partner_control_id = aux as u16;
            slot.option_idx = 0xFF;
        } else {
            slot.option_idx = aux;
            slot.partner_control_id = 0;
        }
        for k in 0..lanes {
            for r in 0..8 {
                slot.main_base[k][r] = read_f32_at(b_slice, k * 8 + r)?;
            }
            slot.emb[k] = read_f32_at(b_slice, 8 * lanes + k)?;
        }
        let v_start = 9 * lanes;
        for f in 0..4 {
            slot.v_base[f] = read_f32_at(b_slice, v_start + f)?;
        }
        if has_ultra {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_slice = &ultra_bytes[u_off..u_off + u_len];
            u_off += u_len;
            for k in 0..lanes {
                for r in 0..8 {
                    slot.main_ultra[k][r] = read_f32_at(u_slice, k * 8 + r)?;
                }
            }
            let uv_start = 8 * lanes;
            for f in 0..4 {
                slot.v_ultra[f] = read_f32_at(u_slice, uv_start + f)?;
            }
            slot.has_ultra = true;
        }
        *out_slot = slot;
    }
    Ok(count)
}

/// Encodes a slice of [`crate::control::Cp1ContributorSlot`] entries into a self-describing
/// `CP1\0` stream suitable for Region 4 `shared_extent` (`ref[container.dnam.cp1-profile]`).
pub fn encode_cp1_shared_stream(
    slots: &[crate::control::Cp1ContributorSlot],
) -> Result<Vec<u8>, DnamError> {
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    reject_if(
        slots.len() > u16::MAX as usize,
        "CP1 contributor count exceeds u16::MAX",
    )?;
    let has_ultra = slots.iter().any(|s| s.has_ultra);
    let mut base_len = 0usize;
    let mut ultra_len = 0usize;
    for slot in slots {
        let lanes = slot.lanes as usize;
        base_len += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        if has_ultra {
            ultra_len += cp1_contributor_record_bytes(lanes, true).unwrap();
        }
    }
    let mut out = Vec::with_capacity(8 + slots.len() * 4 + base_len + ultra_len);
    out.extend_from_slice(&CP1_MAGIC);
    out.extend_from_slice(&(slots.len() as u16).to_le_bytes());
    out.push(u8::from(has_ultra));
    out.push(0);
    for slot in slots {
        let raw_kl = ((slot.ctrl_kind & 0x03) << 2) | (slot.lanes & 0x03);
        let aux = if slot.ctrl_kind == 2 {
            (slot.partner_control_id & 0xFF) as u8
        } else {
            slot.option_idx
        };
        out.extend_from_slice(&slot.control_idx.to_le_bytes());
        out.push(raw_kl);
        out.push(aux);
    }
    let push_f32 = |buf: &mut Vec<u8>, val: f32| -> Result<(), DnamError> {
        reject_if(!val.is_finite(), "nonfinite coefficient")?;
        buf.extend_from_slice(&val.to_le_bytes());
        Ok(())
    };
    for slot in slots {
        let lanes = slot.lanes as usize;
        for k in 0..lanes {
            for r in 0..8 {
                push_f32(&mut out, slot.main_base[k][r])?;
            }
        }
        for k in 0..lanes {
            push_f32(&mut out, slot.emb[k])?;
        }
        for f in 0..4 {
            push_f32(&mut out, slot.v_base[f])?;
        }
    }
    if has_ultra {
        for slot in slots {
            let lanes = slot.lanes as usize;
            for k in 0..lanes {
                for r in 0..8 {
                    push_f32(&mut out, slot.main_ultra[k][r])?;
                }
            }
            for f in 0..4 {
                push_f32(&mut out, slot.v_ultra[f])?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use alloc::vec;
    use serde_json::json;

    /// verify[container.dnam.delay-histories]
    /// Verifies canonical decimal parsing (`u64` without leading zeros) and `delay_histories` validation.
    #[test]
    fn test_model_declared_delay_histories_validation() {
        assert_eq!(
            parse_canonical_reference_samples("432000").unwrap(),
            432_000
        );
        assert_eq!(
            parse_canonical_reference_samples("18446744073709551615").unwrap(),
            u64::MAX
        );
        for bad in ["", "0", "04", "+4", "4.0", "18446744073709551616"] {
            assert!(parse_canonical_reference_samples(bad).is_err());
        }

        let valid_entries = vec![
            json!({
                "id": "pre.left",
                "stage": "stage1_pre",
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": {
                    "lite": "480",
                    "full": "480",
                    "pro": "960",
                    "ultra": "960"
                }
            }),
            json!({
                "id": "tank.ap.0",
                "stage": "stage4_tank",
                "kind": "allpass",
                "introduced_tier": "pro",
                "sample_format": "fp16",
                "reference_samples": {
                    "pro": "149",
                    "ultra": "149"
                }
            }),
        ];
        let parsed = validate_delay_histories(&valid_entries, 3).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].reference_samples_by_tier, [480, 480, 960, 960]);
        assert_eq!(parsed[1].reference_samples_by_tier, [0, 0, 149, 149]);

        // Fractional delay < 4 is rejected
        let too_short = vec![json!({
            "id": "pre.left",
            "stage": "stage1_pre",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "fp16",
            "reference_samples": { "lite": "3" }
        })];
        assert!(validate_delay_histories(&too_short, 0).is_err());
    }

    /// verify[container.dnam.load-validation]
    /// verify[container.dnam.stage3-slices]
    /// verify[container.dnam.cp1-profile]
    /// Verifies Stage III separate 64B-aligned slices and `CP1` packed control projection records.
    #[test]
    fn test_stage3_slices_and_cp1_packed_projection_validation() {
        // 1. Stage III slices across Lite (1 block), Full (1 block), Pro (2 blocks), Ultra (4 blocks)
        for (tier, blocks) in [(0u8, 1usize), (1, 1), (2, 2), (3, 4)] {
            let kautz_len = blocks * 128;
            let arena_bytes = vec![0u8; 128 + kautz_len + 64];
            validate_stage3_tier_slices(tier, &arena_bytes, 0, 128, 128, kautz_len).unwrap();
            // Overlapping slices rejected
            assert!(
                validate_stage3_tier_slices(tier, &arena_bytes, 0, 128, 64, kautz_len).is_err()
            );
            // Unaligned offset rejected
            assert!(
                validate_stage3_tier_slices(tier, &arena_bytes, 32, 128, 192, kautz_len).is_err()
            );
        }

        // 2. CP1 contributor sizes: 4*(9k+4) B in Arena 0, 4*(8k+4) B in Arena 3
        assert_eq!(cp1_contributor_record_bytes(1, false), Some(52));
        assert_eq!(cp1_contributor_record_bytes(2, false), Some(88));
        assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
        assert_eq!(cp1_contributor_record_bytes(1, true), Some(48));
        assert_eq!(cp1_contributor_record_bytes(2, true), Some(80));
        assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));
        assert_eq!(cp1_contributor_record_bytes(0, false), None);
        assert_eq!(cp1_contributor_record_bytes(4, false), None);

        let lanes = [3usize, 1, 2];
        let base_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, false).unwrap())
            .sum();
        let ultra_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, true).unwrap())
            .sum();
        let base_bytes = vec![0u8; base_total];
        let ultra_bytes = vec![0u8; ultra_total];
        let decoded =
            decode_cp1_contributor_stream(&base_bytes, Some(&ultra_bytes), &lanes).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].lanes, 3);
        assert_eq!(decoded[0].base_main.len(), 24);
        assert_eq!(decoded[0].embedding.len(), 3);
        assert!(decoded[0].extra_factors.is_some());
        let decoded_no_ultra = decode_cp1_contributor_stream(&base_bytes, None, &lanes).unwrap();
        assert!(decoded_no_ultra[0].extra_factors.is_none());

        let mut slots = [crate::control::Cp1ContributorSlot::zero(); 4];
        let n_slots = decode_cp1_contributor_stream_into_slice(
            &base_bytes,
            Some(&ultra_bytes),
            &lanes,
            &mut slots,
        )
        .unwrap();
        assert_eq!(n_slots, 3);
        assert!(slots[0].has_ultra);

        let mut shared_stream = Vec::new();
        shared_stream.extend_from_slice(&CP1_MAGIC);
        shared_stream.extend_from_slice(&2_u16.to_le_bytes());
        shared_stream.push(1); // has_ultra = true
        shared_stream.push(0);
        // Entry 0: control_idx = 0, lanes = 3, ctrl_kind = 0, aux = 0xFF
        shared_stream.extend_from_slice(&0_u16.to_le_bytes());
        shared_stream.push(3);
        shared_stream.push(0xFF);
        // Entry 1: control_idx = 1, lanes = 1, ctrl_kind = 2 (dual-axis), aux = 2
        shared_stream.extend_from_slice(&1_u16.to_le_bytes());
        shared_stream.push((2 << 2) | 1);
        shared_stream.push(2);
        shared_stream.resize(8 + 8 + (124 + 52) + (112 + 48), 0);
        let n_shared = decode_cp1_shared_stream_into_slice(&shared_stream, &mut slots).unwrap();
        assert_eq!(n_shared, 2);
        assert_eq!(slots[1].partner_control_id, 2);
        assert_eq!(
            decode_cp1_shared_stream_into_slice(&[], &mut slots).unwrap(),
            0
        );

        assert!(encode_cp1_shared_stream(&[]).unwrap().is_empty());
        let mut s0 = crate::control::Cp1ContributorSlot::zero();
        s0.lanes = 2;
        s0.ctrl_kind = 0;
        s0.control_idx = 5;
        s0.option_idx = 0xFF;
        s0.main_base[0][0] = 0.75;
        s0.emb[1] = -0.25;
        s0.v_base[0] = 0.5;
        s0.main_ultra[1][3] = 0.125;
        s0.v_ultra[2] = -0.5;
        s0.has_ultra = true;
        let mut s1 = crate::control::Cp1ContributorSlot::zero();
        s1.lanes = 1;
        s1.ctrl_kind = 2;
        s1.control_idx = 6;
        s1.partner_control_id = 7;
        s1.has_ultra = true;
        let encoded = encode_cp1_shared_stream(&[s0, s1]).unwrap();
        let mut rt_slots = [crate::control::Cp1ContributorSlot::zero(); 4];
        let rt_n = decode_cp1_shared_stream_into_slice(&encoded, &mut rt_slots).unwrap();
        assert_eq!(rt_n, 2);
        assert_eq!(rt_slots[0].control_idx, 5);
        assert_eq!(rt_slots[0].main_base[0][0], 0.75);
        assert_eq!(rt_slots[0].emb[1], -0.25);
        assert_eq!(rt_slots[0].main_ultra[1][3], 0.125);
        assert_eq!(rt_slots[1].ctrl_kind, 2);
        assert_eq!(rt_slots[1].partner_control_id, 7);

        let mut bad_slot = s0;
        bad_slot.v_base[0] = f32::NAN;
        assert!(encode_cp1_shared_stream(&[bad_slot]).is_err());
    }
}
