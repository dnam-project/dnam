//! 12/16/24/32-bit PCM & 32-bit IEEE-754 float WAV reader/writer and `<f4` SHA-256 digest.

use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{StudioError, StudioResult};

/// Decoded interleaved floating-point PCM WAV container.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count (`1` = mono, `2` = stereo).
    pub channels: u16,
    /// Interleaved `[-1.0, 1.0]` floating-point samples.
    pub samples: Vec<f32>,
}

/// Computes the canonical IEEE-754 little-endian `<f4` SHA-256 hex digest of `samples`.
pub fn compute_pcm_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &sample in samples {
        let canonical = if sample == 0.0 || sample.is_nan() {
            0.0_f32
        } else {
            sample
        };
        hasher.update(canonical.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex
}

fn write_wav_riff_header(
    sample_count: usize,
    sample_rate_hz: u32,
    channels: u16,
    bytes_per_sample: u16,
    bits_per_sample: u16,
    audio_format: u16,
) -> StudioResult<(Vec<u8>, u32)> {
    if channels == 0 || sample_rate_hz == 0 {
        return Err(StudioError::Audio(
            "channels and sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if !sample_count.is_multiple_of(usize::from(channels)) {
        return Err(StudioError::Audio(
            "WAV sample count must be a multiple of channels".to_string(),
        ));
    }
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| StudioError::Audio("WAV block_align overflow".to_string()))?;
    let byte_rate = sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| StudioError::Audio("WAV byte_rate overflow".to_string()))?;
    let raw_data_len = sample_count
        .checked_mul(usize::from(bytes_per_sample))
        .ok_or_else(|| StudioError::Audio("WAV payload size overflow".to_string()))?;
    let data_bytes_len = u32::try_from(raw_data_len)
        .map_err(|_| StudioError::Audio("WAV payload exceeds 4 GiB limit".to_string()))?;
    let pad = data_bytes_len & 1;
    let riff_size = 36_u32
        .checked_add(data_bytes_len)
        .and_then(|v| v.checked_add(pad))
        .ok_or_else(|| StudioError::Audio("WAV RIFF size exceeds 4 GiB limit".to_string()))?;
    let cap = raw_data_len
        .checked_add(44 + usize::from(pad == 1))
        .ok_or_else(|| StudioError::Audio("WAV buffer capacity overflow".to_string()))?;

    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16_u32.to_le_bytes());
    buf.extend_from_slice(&audio_format.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate_hz.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes_len.to_le_bytes());
    Ok((buf, pad))
}

fn finish_wav_write(path: &Path, mut buf: Vec<u8>, pad: u32) -> StudioResult<()> {
    if pad == 1 {
        buf.push(0_u8);
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, buf)?;
    Ok(())
}

/// Writes `samples` (`[-1.0, 1.0]`) to a 12-bit signed PCM `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`,
/// `wBitsPerSample = 16`, `wValidBitsPerSample = 12`, MSB-aligned) RIFF/WAVE file at `path`.
pub fn write_wav_12bit(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    write_wav_12bit_inner(path.as_ref(), samples, sample_rate_hz, channels)
}

fn write_wav_12bit_inner(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    if channels == 0 || sample_rate_hz == 0 {
        return Err(StudioError::Audio(
            "channels and sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(StudioError::Audio(
            "WAV sample count must be a multiple of channels".to_string(),
        ));
    }
    let block_align = channels
        .checked_mul(2)
        .ok_or_else(|| StudioError::Audio("WAV block_align overflow".to_string()))?;
    let byte_rate = sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| StudioError::Audio("WAV byte_rate overflow".to_string()))?;
    let raw_data_len = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| StudioError::Audio("WAV payload size overflow".to_string()))?;
    let data_bytes_len = u32::try_from(raw_data_len)
        .map_err(|_| StudioError::Audio("WAV payload exceeds 4 GiB limit".to_string()))?;
    let pad = data_bytes_len & 1;
    let riff_size = 60_u32
        .checked_add(data_bytes_len)
        .and_then(|v| v.checked_add(pad))
        .ok_or_else(|| StudioError::Audio("WAV RIFF size exceeds 4 GiB limit".to_string()))?;
    let cap = raw_data_len
        .checked_add(68 + usize::from(pad == 1))
        .ok_or_else(|| StudioError::Audio("WAV buffer capacity overflow".to_string()))?;

    let channel_mask: u32 = match channels {
        1 => 0x0000_0004,
        2 => 0x0000_0003,
        _ => 0,
    };

    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&40_u32.to_le_bytes());
    buf.extend_from_slice(&0xFFFE_u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate_hz.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&16_u16.to_le_bytes());
    buf.extend_from_slice(&22_u16.to_le_bytes());
    buf.extend_from_slice(&12_u16.to_le_bytes());
    buf.extend_from_slice(&channel_mask.to_le_bytes());
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&KSDATAFORMAT_SUBTYPE_TAIL);
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes_len.to_le_bytes());

    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 2_047.0 / 2_048.0)
        } else {
            0.0_f64
        };
        let q12 = (clamped * 2_048.0)
            .round_ties_even()
            .clamp(-2_048.0, 2_047.0) as i16;
        let msb_aligned = q12 << 4;
        buf.extend_from_slice(&msb_aligned.to_le_bytes());
    }
    finish_wav_write(path, buf, pad)
}

/// Writes `samples` (`[-1.0, 1.0]`) to a 16-bit signed PCM RIFF/WAVE file at `path`.
pub fn write_wav_16bit(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    write_wav_16bit_inner(path.as_ref(), samples, sample_rate_hz, channels)
}

fn write_wav_16bit_inner(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    let (mut buf, pad) = write_wav_riff_header(samples.len(), sample_rate_hz, channels, 2, 16, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 32_767.0 / 32_768.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16;
        buf.extend_from_slice(&q.to_le_bytes());
    }
    finish_wav_write(path, buf, pad)
}

/// Writes `samples` (`[-1.0, 1.0]`) to a 24-bit signed PCM RIFF/WAVE file at `path`.
pub fn write_wav_24bit(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    write_wav_24bit_inner(path.as_ref(), samples, sample_rate_hz, channels)
}

fn write_wav_24bit_inner(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    let (mut buf, pad) = write_wav_riff_header(samples.len(), sample_rate_hz, channels, 3, 24, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 8_388_607.0 / 8_388_608.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 8_388_608.0)
            .round()
            .clamp(-8_388_608.0, 8_388_607.0) as i32;
        let le = q.to_le_bytes();
        buf.push(le[0]);
        buf.push(le[1]);
        buf.push(le[2]);
    }
    finish_wav_write(path, buf, pad)
}

/// Writes `samples` (`[-1.0, 1.0]`) to a 32-bit signed PCM (`WAVE_FORMAT_PCM`, `format_tag = 1`)
/// RIFF/WAVE file at `path`.
pub fn write_wav_32bit_int(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    write_wav_32bit_int_inner(path.as_ref(), samples, sample_rate_hz, channels)
}

fn write_wav_32bit_int_inner(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    let (mut buf, pad) = write_wav_riff_header(samples.len(), sample_rate_hz, channels, 4, 32, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 2_147_483_647.0 / 2_147_483_648.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 2_147_483_648.0)
            .round()
            .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
        buf.extend_from_slice(&q.to_le_bytes());
    }
    finish_wav_write(path, buf, pad)
}

/// Writes `samples` to an unclipped 32-bit IEEE-754 float (`WAVE_FORMAT_IEEE_FLOAT`, `format_tag = 3`)
/// RIFF/WAVE file at `path`, preserving full 32-bit float headroom beyond `[-1.0, 1.0]`.
pub fn write_wav_32bit_float(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    write_wav_32bit_float_inner(path.as_ref(), samples, sample_rate_hz, channels)
}

fn write_wav_32bit_float_inner(
    path: &Path,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    let (mut buf, pad) = write_wav_riff_header(samples.len(), sample_rate_hz, channels, 4, 32, 3)?;
    for &s in samples {
        let safe_s = if s.is_finite() { s } else { 0.0_f32 };
        buf.extend_from_slice(&safe_s.to_le_bytes());
    }
    finish_wav_write(path, buf, pad)
}

const KSDATAFORMAT_SUBTYPE_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

fn parse_wav_fmt_chunk(body: &[u8]) -> StudioResult<(u32, u16, u16, u16, bool)> {
    if body.len() < 16 || body.len() == 17 {
        return Err(StudioError::Audio("Truncated WAV fmt chunk".to_string()));
    }
    if body.len() >= 18 {
        let cb_size = usize::from(u16::from_le_bytes([body[16], body[17]]));
        if 18_usize.checked_add(cb_size).is_none_or(|n| n > body.len()) {
            return Err(StudioError::Audio(
                "Truncated WAV fmt extension bytes (18 + cbSize > chunk length)".to_string(),
            ));
        }
    }
    let audio_format = u16::from_le_bytes([body[0], body[1]]);
    let channels = u16::from_le_bytes([body[2], body[3]]);
    let sample_rate_hz = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    let avg_bytes_per_sec = u32::from_le_bytes([body[8], body[9], body[10], body[11]]);
    let block_align = u16::from_le_bytes([body[12], body[13]]);
    let bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
    if channels == 0 || sample_rate_hz == 0 {
        return Err(StudioError::Audio(
            "WAV fmt chunk requires channels >= 1 and sample_rate_hz >= 1".to_string(),
        ));
    }
    let (effective_bits, container_bytes, is_float) = match audio_format {
        1 => match bits_per_sample {
            12 | 16 => (bits_per_sample, 2_u16, false),
            24 => (24, 3_u16, false),
            32 => (32, 4_u16, false),
            _ => {
                return Err(StudioError::Audio(format!(
                    "Unsupported WAV PCM bit depth ({bits_per_sample}-bit)"
                )));
            }
        },
        3 if bits_per_sample == 32 => (32, 4_u16, true),
        0xFFFE => {
            if body.len() < 40
                || u16::from_le_bytes([body[16], body[17]]) < 22
                || body[26..40] != KSDATAFORMAT_SUBTYPE_TAIL
            {
                return Err(StudioError::Audio(
                    "Unsupported or truncated WAVE_FORMAT_EXTENSIBLE SubFormat".to_string(),
                ));
            }
            let valid_bits = u16::from_le_bytes([body[18], body[19]]);
            if valid_bits > bits_per_sample {
                return Err(StudioError::Audio(format!(
                    "WAVE_FORMAT_EXTENSIBLE wValidBitsPerSample ({valid_bits}) exceeds wBitsPerSample ({bits_per_sample})"
                )));
            }
            let eff = if valid_bits == 0 {
                bits_per_sample
            } else {
                valid_bits
            };
            match u16::from_le_bytes([body[24], body[25]]) {
                1 if matches!(bits_per_sample, 16 | 24 | 32)
                    && matches!(eff, 12 | 16 | 24 | 32) =>
                {
                    (eff, bits_per_sample / 8, false)
                }
                3 if bits_per_sample == 32 && eff == 32 => (32, 4_u16, true),
                sub => {
                    return Err(StudioError::Audio(format!(
                        "Unsupported WAVE_FORMAT_EXTENSIBLE SubFormat tag {sub} ({bits_per_sample}-bit, {eff} valid)"
                    )));
                }
            }
        }
        other => {
            return Err(StudioError::Audio(format!(
                "Unsupported WAV format tag {other} ({bits_per_sample}-bit)"
            )));
        }
    };
    let expected_align = channels
        .checked_mul(container_bytes)
        .ok_or_else(|| StudioError::Audio("WAV block_align overflow".to_string()))?;
    if block_align != expected_align {
        return Err(StudioError::Audio(format!(
            "Invalid WAV block_align {block_align} (expected {expected_align})"
        )));
    }
    let expected_byte_rate = sample_rate_hz
        .checked_mul(u32::from(expected_align))
        .ok_or_else(|| StudioError::Audio("WAV byte_rate overflow".to_string()))?;
    if avg_bytes_per_sec != expected_byte_rate {
        return Err(StudioError::Audio(format!(
            "Invalid WAV byte_rate {avg_bytes_per_sec} (expected {expected_byte_rate})"
        )));
    }
    Ok((
        sample_rate_hz,
        channels,
        container_bytes,
        effective_bits,
        is_float,
    ))
}

/// Reads a 12-bit, 16-bit, 24-bit, or 32-bit (signed PCM or IEEE-754 float) RIFF/WAVE file from `path`.
pub fn read_wav(path: impl AsRef<Path>) -> StudioResult<WavData> {
    read_wav_inner(path.as_ref())
}

fn read_wav_inner(path: &Path) -> StudioResult<WavData> {
    let bytes = fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(StudioError::Audio("Invalid RIFF/WAVE header".to_string()));
    }
    let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let riff_end = 8_usize
        .checked_add(riff_size)
        .filter(|&end| riff_size >= 4 && end <= bytes.len())
        .ok_or_else(|| StudioError::Audio("Truncated or invalid WAV RIFF size".to_string()))?;

    let mut fmt_info: Option<(u32, u16, u16, u16, bool)> = None;
    let mut data_samples: Option<Vec<f32>> = None;
    let mut offset = 12_usize;
    while offset < riff_end {
        if offset.checked_add(8).is_none_or(|h| h > riff_end) {
            return Err(StudioError::Audio(
                "Truncated WAV RIFF chunk header".to_string(),
            ));
        }
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;
        let end = offset
            .checked_add(chunk_len)
            .filter(|&e| e <= riff_end)
            .ok_or_else(|| StudioError::Audio("Truncated WAV RIFF chunk".to_string()))?;
        let body = &bytes[offset..end];

        if chunk_id == b"fmt " {
            if fmt_info.is_some() {
                return Err(StudioError::Audio(
                    "Duplicate fmt chunk in WAV file".to_string(),
                ));
            }
            fmt_info = Some(parse_wav_fmt_chunk(body)?);
        } else if chunk_id == b"data" {
            if data_samples.is_some() {
                return Err(StudioError::Audio(
                    "Duplicate data chunk in WAV file".to_string(),
                ));
            }
            let (_, channels, container_bytes, effective_bits, is_float) =
                fmt_info.ok_or_else(|| {
                    StudioError::Audio("WAV data chunk precedes fmt chunk".to_string())
                })?;
            let samples =
                decode_pcm_payload_with_format(body, container_bytes, effective_bits, is_float)?;
            if !samples.len().is_multiple_of(usize::from(channels)) {
                return Err(StudioError::Audio(
                    "WAV data chunk contains an incomplete multi-channel frame".to_string(),
                ));
            }
            data_samples = Some(samples);
        }

        let padded_end = end
            .checked_add(chunk_len & 1)
            .filter(|&pe| pe <= riff_end)
            .ok_or_else(|| {
                StudioError::Audio("Missing RIFF odd-chunk alignment pad byte".to_string())
            })?;
        offset = padded_end;
    }

    let (sample_rate_hz, channels, _, _, _) =
        fmt_info.ok_or_else(|| StudioError::Audio("Missing fmt chunk in WAV file".to_string()))?;
    let samples = data_samples
        .ok_or_else(|| StudioError::Audio("Missing data chunk in WAV file".to_string()))?;
    Ok(WavData {
        sample_rate_hz,
        channels,
        samples,
    })
}

#[cfg(test)]
fn decode_pcm_payload(payload: &[u8], bits_per_sample: u16) -> StudioResult<Vec<f32>> {
    let container_bytes = match bits_per_sample {
        12 | 16 => 2,
        24 => 3,
        32 => 4,
        other => {
            return Err(StudioError::Audio(format!(
                "Unsupported WAV bit depth: {other}"
            )));
        }
    };
    decode_pcm_payload_with_format(payload, container_bytes, bits_per_sample, false)
}

fn decode_pcm_payload_with_format(
    payload: &[u8],
    container_bytes: u16,
    effective_bits: u16,
    is_float: bool,
) -> StudioResult<Vec<f32>> {
    if is_float {
        if container_bytes != 4 || effective_bits != 32 {
            return Err(StudioError::Audio(format!(
                "Unsupported WAV float format ({container_bytes}B container, {effective_bits}-bit)"
            )));
        }
        let (chunks, rem) = payload.as_chunks::<4>();
        if !rem.is_empty() {
            return Err(StudioError::Audio(
                "WAV data chunk length is not a multiple of 4 bytes (32-bit float)".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(chunks.len());
        for &chunk in chunks {
            let v = f32::from_le_bytes(chunk);
            if !v.is_finite() {
                return Err(StudioError::Audio(
                    "WAV 32-bit float payload contains non-finite (NaN or Inf) sample".to_string(),
                ));
            }
            out.push(v);
        }
        return Ok(out);
    }

    if !matches!(container_bytes, 2..=4)
        || !matches!(effective_bits, 12 | 16 | 24 | 32)
        || effective_bits > container_bytes * 8
    {
        return Err(StudioError::Audio(format!(
            "Unsupported WAV PCM format ({container_bytes}B container, {effective_bits}-bit)"
        )));
    }
    let width = usize::from(container_bytes);
    if !payload.len().is_multiple_of(width) {
        return Err(StudioError::Audio(format!(
            "WAV data chunk length is not a multiple of {width} bytes ({effective_bits}-bit PCM)"
        )));
    }
    let mask: i32 = if effective_bits < 32 {
        !((1_i32 << (32 - effective_bits)) - 1)
    } else {
        -1_i32
    };
    let mut out = Vec::with_capacity(payload.len() / width);
    for chunk in payload.chunks_exact(width) {
        let v32 = match width {
            2 => i32::from(i16::from_le_bytes([chunk[0], chunk[1]])) << 16,
            3 => i32::from_le_bytes([0, chunk[0], chunk[1], chunk[2]]),
            _ => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        };
        let masked = v32 & mask;
        out.push((f64::from(masked) / 2_147_483_648.0) as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_wav_24bit_roundtrip_and_canonical_sha256(
            samples in proptest::collection::vec(-0.999_f32..0.999_f32, 4..128),
            flip_idx in 0_usize..128,
        ) {
            // 1. Canonical -0.0 / +0.0 / NaN invariance in compute_pcm_sha256
            let mut pos_zero = samples.clone();
            let mut neg_zero = samples.clone();
            let mut nan_zero = samples.clone();
            let idx = flip_idx % samples.len();
            pos_zero[idx] = 0.0;
            neg_zero[idx] = -0.0;
            nan_zero[idx] = f32::NAN;
            prop_assert_eq!(compute_pcm_sha256(&pos_zero), compute_pcm_sha256(&neg_zero));
            prop_assert_eq!(compute_pcm_sha256(&pos_zero), compute_pcm_sha256(&nan_zero));

            // 2. 24-bit WAV encode/decode quantization error <= 2 LSB (2 / 8_388_608)
            let scaled: Vec<u8> = samples
                .iter()
                .flat_map(|&s| {
                    let q = (s.clamp(-1.0, 0.999_999_9) * 8_388_607.0).round() as i32;
                    let b = q.to_le_bytes();
                    [b[0], b[1], b[2]]
                })
                .collect();
            let decoded = decode_pcm_payload(&scaled, 24).expect("decode 24-bit payload");
            prop_assert_eq!(decoded.len(), samples.len());
            for (&orig, &dec) in samples.iter().zip(decoded.iter()) {
                prop_assert!((orig - dec).abs() <= 2.5e-7, "orig={orig}, dec={dec}");
            }
        }
    }

    #[test]
    fn test_wav_16_24_32bit_io_and_error_branches() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_wav_unit_{nanos}"));
        let _ = fs::create_dir_all(&dir);

        // Canonical -0.0 and NaN handling in compute_pcm_sha256
        let h1 = compute_pcm_sha256(&[0.0, 0.5]);
        let h2 = compute_pcm_sha256(&[-0.0, 0.5]);
        let h3 = compute_pcm_sha256(&[f32::NAN, 0.5]);
        assert_eq!(h1, h2);
        assert_eq!(h1, h3);

        // 0 channels error
        assert!(write_wav_24bit(dir.join("bad.wav"), &[0.0], 48_000, 0).is_err());

        // 24-bit write & read (including +1.0 full-scale without -1.0 wraparound)
        let p24 = dir.join("m24.wav");
        let sig = vec![0.0, 0.25, -0.5, 1.0, -1.0];
        write_wav_24bit(&p24, &sig, 48_000, 1).expect("write 24-bit");
        let r24 = read_wav(&p24).expect("read 24-bit");
        assert_eq!(r24.sample_rate_hz, 48_000);
        assert_eq!(r24.channels, 1);
        assert_eq!(r24.samples.len(), 5);
        assert!(
            (r24.samples[3] - (8_388_607.0 / 8_388_608.0)).abs() <= 2e-7,
            "+1.0 in 24-bit PCM must saturate to +8388607/8388608, got {}",
            r24.samples[3]
        );
        assert_eq!(r24.samples[4], -1.0);

        // 32-bit signed integer PCM write & read
        assert!(write_wav_32bit_int(dir.join("badi32.wav"), &[0.0], 48_000, 0).is_err());
        assert!(write_wav_32bit_int(dir.join("badi32s.wav"), &[0.1, 0.2, 0.3], 48_000, 2).is_err());
        let pi32 = dir.join("mi32.wav");
        write_wav_32bit_int(&pi32, &[-1.0, 0.0, 0.5, 1.0, f32::NAN], 96_000, 1)
            .expect("write 32-bit int");
        let ri32 = read_wav(&pi32).expect("read 32-bit int");
        assert_eq!(ri32.sample_rate_hz, 96_000);
        assert_eq!(ri32.channels, 1);
        assert_eq!(ri32.samples[0], -1.0);
        assert_eq!(ri32.samples[1], 0.0);
        assert!((ri32.samples[2] - 0.5).abs() <= 1e-7);
        assert!((ri32.samples[3] - 1.0).abs() <= 1e-6);
        assert_eq!(ri32.samples[4], 0.0);

        // 32-bit IEEE-754 float write & read (unclipped beyond [-1.0, 1.0])
        assert!(write_wav_32bit_float(dir.join("bad32.wav"), &[0.0], 48_000, 0).is_err());
        let pf32 = dir.join("mf32.wav");
        let sig_f32 = vec![-3.5_f32, -1.25, 0.0, 1.5, 4.25];
        write_wav_32bit_float(&pf32, &sig_f32, 48_000, 1).expect("write 32-bit float");
        let rf32 = read_wav(&pf32).expect("read 32-bit float");
        assert_eq!(rf32.sample_rate_hz, 48_000);
        assert_eq!(rf32.channels, 1);
        assert_eq!(rf32.samples, sig_f32);

        // 16-bit decode
        let decoded_16 = decode_pcm_payload(&0x4000_i16.to_le_bytes(), 16).expect("decode 16");
        assert!((decoded_16[0] - 0.5).abs() < 1e-4);

        // 32-bit decode
        let decoded_32 = decode_pcm_payload(&0x4000_0000_i32.to_le_bytes(), 32).expect("decode 32");
        assert!((decoded_32[0] - 0.5).abs() < 1e-5);

        // Unsupported bit depth
        assert!(decode_pcm_payload(&[0, 1, 2, 3], 8).is_err());

        // Invalid RIFF header
        let bad_hdr = dir.join("bad_hdr.wav");
        fs::write(&bad_hdr, b"NOT_A_WAV").expect("write bad hdr");
        assert!(read_wav(&bad_hdr).is_err());

        // Valid RIFF/WAVE header + valid fmt but missing data chunk
        let mut no_data = Vec::new();
        no_data.extend_from_slice(b"RIFF");
        no_data.extend_from_slice(&36_u32.to_le_bytes());
        no_data.extend_from_slice(b"WAVE");
        no_data.extend_from_slice(b"fmt ");
        no_data.extend_from_slice(&16_u32.to_le_bytes());
        no_data.extend_from_slice(&1_u16.to_le_bytes());
        no_data.extend_from_slice(&1_u16.to_le_bytes());
        no_data.extend_from_slice(&48_000_u32.to_le_bytes());
        no_data.extend_from_slice(&96_000_u32.to_le_bytes());
        no_data.extend_from_slice(&2_u16.to_le_bytes());
        no_data.extend_from_slice(&16_u16.to_le_bytes());
        no_data.extend_from_slice(b"JUNK");
        no_data.extend_from_slice(&0_u32.to_le_bytes());
        let no_data_path = dir.join("no_data.wav");
        fs::write(&no_data_path, &no_data).expect("write no_data");
        assert!(read_wav(&no_data_path).is_err());

        // Leading odd-length JUNK/LIST chunks before WAVE_FORMAT_EXTENSIBLE fmt + data
        let mut ext_wav = Vec::new();
        ext_wav.extend_from_slice(b"RIFF");
        ext_wav.extend_from_slice(&94_u32.to_le_bytes());
        ext_wav.extend_from_slice(b"WAVE");
        // Odd-length (5 bytes + 1 RIFF pad byte) JUNK chunk BEFORE fmt!
        ext_wav.extend_from_slice(b"JUNK");
        ext_wav.extend_from_slice(&5_u32.to_le_bytes());
        ext_wav.extend_from_slice(&[1, 2, 3, 4, 5, 0]);
        // 40-byte WAVE_FORMAT_EXTENSIBLE fmt chunk (IEEE float 32-bit, 96 kHz mono)
        ext_wav.extend_from_slice(b"fmt ");
        ext_wav.extend_from_slice(&40_u32.to_le_bytes());
        ext_wav.extend_from_slice(&0xFFFE_u16.to_le_bytes());
        ext_wav.extend_from_slice(&1_u16.to_le_bytes());
        ext_wav.extend_from_slice(&96_000_u32.to_le_bytes());
        ext_wav.extend_from_slice(&384_000_u32.to_le_bytes());
        ext_wav.extend_from_slice(&4_u16.to_le_bytes());
        ext_wav.extend_from_slice(&32_u16.to_le_bytes());
        ext_wav.extend_from_slice(&22_u16.to_le_bytes());
        ext_wav.extend_from_slice(&32_u16.to_le_bytes());
        ext_wav.extend_from_slice(&4_u32.to_le_bytes());
        ext_wav.extend_from_slice(&3_u16.to_le_bytes());
        ext_wav.extend_from_slice(&KSDATAFORMAT_SUBTYPE_TAIL);
        // Odd-length LIST chunk between fmt and data
        ext_wav.extend_from_slice(b"LIST");
        ext_wav.extend_from_slice(&3_u32.to_le_bytes());
        ext_wav.extend_from_slice(&[b'x', b'y', b'z', 0]);
        // data chunk with 2 float samples
        ext_wav.extend_from_slice(b"data");
        ext_wav.extend_from_slice(&8_u32.to_le_bytes());
        ext_wav.extend_from_slice(&0.25_f32.to_le_bytes());
        ext_wav.extend_from_slice(&(-0.75_f32).to_le_bytes());
        let ext_path = dir.join("ext_junk.wav");
        fs::write(&ext_path, &ext_wav).expect("write ext_wav");
        let parsed_ext = read_wav(&ext_path).expect("read extensible wav after odd JUNK");
        assert_eq!(parsed_ext.sample_rate_hz, 96_000);
        assert_eq!(parsed_ext.channels, 1);
        assert_eq!(parsed_ext.samples, vec![0.25_f32, -0.75_f32]);

        // Reject unsupported format tag (e.g. A-law = 6), duplicate fmt chunk, and data-before-fmt
        let mut alaw = no_data.clone();
        alaw[4..8].copy_from_slice(&46_u32.to_le_bytes());
        alaw[20..22].copy_from_slice(&6_u16.to_le_bytes());
        alaw.extend_from_slice(b"data");
        alaw.extend_from_slice(&2_u32.to_le_bytes());
        alaw.extend_from_slice(&[0, 0]);
        let alaw_path = dir.join("alaw.wav");
        fs::write(&alaw_path, &alaw).expect("write alaw");
        assert!(read_wav(&alaw_path).is_err());

        let mut dup_fmt = no_data[..36].to_vec();
        dup_fmt.extend_from_slice(&no_data[12..36]);
        dup_fmt[4..8].copy_from_slice(&52_u32.to_le_bytes());
        let dup_fmt_path = dir.join("dup_fmt.wav");
        fs::write(&dup_fmt_path, &dup_fmt).expect("write dup_fmt");
        assert!(read_wav(&dup_fmt_path).is_err());

        let mut data_first = Vec::new();
        data_first.extend_from_slice(b"RIFF");
        data_first.extend_from_slice(&12_u32.to_le_bytes());
        data_first.extend_from_slice(b"WAVE");
        data_first.extend_from_slice(b"data");
        data_first.extend_from_slice(&0_u32.to_le_bytes());
        let data_first_path = dir.join("data_first.wav");
        fs::write(&data_first_path, &data_first).expect("write data_first");
        assert!(read_wav(&data_first_path).is_err());

        // parse_wav_fmt_chunk error & extensible PCM branches + 12/16/24/32f writers + word pad + truncation checks
        assert!(parse_wav_fmt_chunk(&[0u8; 8]).is_err());
        assert!(parse_wav_fmt_chunk(&[0u8; 16]).is_err());
        let mut pcm20 = [0u8; 16];
        pcm20[0..2].copy_from_slice(&1_u16.to_le_bytes());
        pcm20[2..4].copy_from_slice(&1_u16.to_le_bytes());
        pcm20[4..8].copy_from_slice(&48_000_u32.to_le_bytes());
        pcm20[12..14].copy_from_slice(&3_u16.to_le_bytes());
        pcm20[14..16].copy_from_slice(&20_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&pcm20).is_err());
        let mut bad_cb17 = no_data[20..36].to_vec();
        bad_cb17.push(0);
        assert!(parse_wav_fmt_chunk(&bad_cb17).is_err());
        let mut ext_fmt = ext_wav[34..74].to_vec();
        ext_fmt[24..26].copy_from_slice(&1_u16.to_le_bytes());
        ext_fmt[8..12].copy_from_slice(&288_000_u32.to_le_bytes());
        ext_fmt[12..14].copy_from_slice(&3_u16.to_le_bytes());
        ext_fmt[14..16].copy_from_slice(&24_u16.to_le_bytes());
        ext_fmt[18..20].copy_from_slice(&24_u16.to_le_bytes());
        assert_eq!(
            parse_wav_fmt_chunk(&ext_fmt).unwrap(),
            (96_000, 1, 3, 24, false)
        );
        // Extensible 12-bit valid inside 16-bit container
        ext_fmt[8..12].copy_from_slice(&192_000_u32.to_le_bytes());
        ext_fmt[12..14].copy_from_slice(&2_u16.to_le_bytes());
        ext_fmt[14..16].copy_from_slice(&16_u16.to_le_bytes());
        ext_fmt[18..20].copy_from_slice(&12_u16.to_le_bytes());
        assert_eq!(
            parse_wav_fmt_chunk(&ext_fmt).unwrap(),
            (96_000, 1, 2, 12, false)
        );
        // Extensible S24_32LE (24 valid bits MSB-aligned inside 32-bit container)
        ext_fmt[8..12].copy_from_slice(&384_000_u32.to_le_bytes());
        ext_fmt[12..14].copy_from_slice(&4_u16.to_le_bytes());
        ext_fmt[14..16].copy_from_slice(&32_u16.to_le_bytes());
        ext_fmt[18..20].copy_from_slice(&24_u16.to_le_bytes());
        assert_eq!(
            parse_wav_fmt_chunk(&ext_fmt).unwrap(),
            (96_000, 1, 4, 24, false)
        );
        let s24_32_raw = [0xFF_u8, 0x00, 0x00, 0x40]; // +0.5 with 0xFF low padding byte masked off
        let s24_32_dec =
            decode_pcm_payload_with_format(&s24_32_raw, 4, 24, false).expect("decode S24_32LE");
        assert_eq!(s24_32_dec, vec![0.5_f32]);
        // Reject extensible wBitsPerSample == 12 (container must be 16), bad cbSize < 22, and bad byte_rate
        ext_fmt[14..16].copy_from_slice(&12_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        ext_fmt[14..16].copy_from_slice(&16_u16.to_le_bytes());
        ext_fmt[16..18].copy_from_slice(&18_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        ext_fmt[16..18].copy_from_slice(&22_u16.to_le_bytes());
        ext_fmt[8..12].copy_from_slice(&123_456_u32.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        ext_fmt[8..12].copy_from_slice(&192_000_u32.to_le_bytes());
        // Reject wValidBitsPerSample > wBitsPerSample and invalid block_align
        ext_fmt[18..20].copy_from_slice(&24_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        ext_fmt[18..20].copy_from_slice(&16_u16.to_le_bytes());
        ext_fmt[12..14].copy_from_slice(&5_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        ext_fmt[12..14].copy_from_slice(&2_u16.to_le_bytes());
        ext_fmt[24..26].copy_from_slice(&99_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_fmt).is_err());
        assert!(parse_wav_fmt_chunk(&ext_fmt[..20]).is_err());

        // 12-bit WAVE_FORMAT_EXTENSIBLE writer & roundtrip
        let p12 = dir.join("m12.wav");
        assert!(write_wav_12bit(&p12, &[0.0], 48_000, 0).is_err());
        assert!(write_wav_12bit(&p12, &[0.1, 0.2, 0.3], 48_000, 2).is_err());
        write_wav_12bit(&p12, &[-1.0, 0.0, 0.5, 1.0, f32::NAN, -0.25], 24_000, 2)
            .expect("write 12-bit extensible wav");
        let r12 = read_wav(&p12).expect("read 12-bit extensible wav");
        assert_eq!((r12.sample_rate_hz, r12.channels), (24_000, 2));
        assert_eq!(
            r12.samples,
            vec![-1.0, 0.0, 0.5, 2_047.0 / 2_048.0, 0.0, -0.25]
        );

        // 12-bit PCM decode (masks low 4 padding bits; -2048 -> -1.0, +2047 -> 2047/2048)
        let mut p12_raw = Vec::new();
        p12_raw.extend_from_slice(&(-32_768_i16).to_le_bytes());
        p12_raw.extend_from_slice(&0x7FFF_i16.to_le_bytes());
        let d12 = decode_pcm_payload(&p12_raw, 12).expect("decode 12-bit");
        assert_eq!(d12[0], -1.0);
        assert_eq!(d12[1], 2047.0 / 2048.0);
        assert!(decode_pcm_payload(&[0], 12).is_err());
        assert!(decode_pcm_payload(&[0], 16).is_err());
        assert!(decode_pcm_payload(&[0, 1], 24).is_err());
        assert!(decode_pcm_payload(&[0, 1, 2], 32).is_err());
        assert!(decode_pcm_payload(&[0, 1], 20).is_err());
        assert!(decode_pcm_payload_with_format(&[0, 1, 2], 4, 32, true).is_err());
        assert!(decode_pcm_payload_with_format(&[0, 1], 2, 16, true).is_err());
        assert!(decode_pcm_payload_with_format(&[0, 1], 2, 24, false).is_err());
        assert!(decode_pcm_payload_with_format(&f32::NAN.to_le_bytes(), 4, 32, true).is_err());
        assert!(decode_pcm_payload_with_format(&f32::INFINITY.to_le_bytes(), 4, 32, true).is_err());

        // 16-bit writer + odd-sample 24-bit RIFF pad byte + multi-channel frame completeness
        let p16 = dir.join("m16.wav");
        assert!(write_wav_16bit(&p16, &[0.0], 48_000, 0).is_err());
        assert!(write_wav_16bit(&p16, &[0.1, 0.2, 0.3], 48_000, 2).is_err());
        assert!(write_wav_24bit(&p24, &[0.1, 0.2, 0.3], 48_000, 2).is_err());
        assert!(write_wav_32bit_float(&pf32, &[0.1, 0.2, 0.3], 48_000, 2).is_err());
        write_wav_16bit(&p16, &[-1.0, 0.0, 0.5, f32::NAN], 44_100, 2).expect("write 16-bit");
        let r16 = read_wav(&p16).expect("read 16-bit");
        assert_eq!((r16.sample_rate_hz, r16.channels), (44_100, 2));
        assert_eq!(r16.samples, vec![-1.0, 0.0, 0.5, 0.0]);
        let raw_24_bytes = fs::read(&p24).expect("read raw 24-bit");
        assert_eq!(
            raw_24_bytes.len() % 2,
            0,
            "odd 24-bit WAV must have RIFF pad byte"
        );
        assert_eq!(
            u32::from_le_bytes(raw_24_bytes[4..8].try_into().unwrap()) as usize + 8,
            raw_24_bytes.len()
        );

        let p_nan = dir.join("nan24.wav");
        write_wav_24bit(&p_nan, &[f32::NAN, f32::INFINITY], 48_000, 2).expect("write nan24");
        assert_eq!(
            read_wav(&p_nan).expect("read nan24").samples,
            vec![0.0, 0.0]
        );
        write_wav_32bit_float(&p_nan, &[f32::NAN], 48_000, 1).expect("write nan32");
        assert_eq!(read_wav(&p_nan).expect("read nan32").samples, vec![0.0]);

        // Truncated fmt chunk, truncated data chunk, & incomplete multi-channel frame in data chunk
        let mut trunc_fmt = no_data[..24].to_vec();
        trunc_fmt[16..20].copy_from_slice(&40_u32.to_le_bytes());
        let trunc_fmt_path = dir.join("trunc_fmt.wav");
        fs::write(&trunc_fmt_path, &trunc_fmt).expect("write trunc_fmt");
        assert!(read_wav(&trunc_fmt_path).is_err());

        let mut trunc_data = no_data[..36].to_vec();
        trunc_data.extend_from_slice(b"data");
        trunc_data.extend_from_slice(&100_u32.to_le_bytes());
        trunc_data.extend_from_slice(&[0, 0, 0, 0]);
        let trunc_data_path = dir.join("trunc_data.wav");
        fs::write(&trunc_data_path, &trunc_data).expect("write trunc_data");
        assert!(read_wav(&trunc_data_path).is_err());

        let mut odd_stereo = no_data[..36].to_vec();
        odd_stereo[4..8].copy_from_slice(&38_u32.to_le_bytes());
        odd_stereo[22..24].copy_from_slice(&2_u16.to_le_bytes()); // 2 channels
        odd_stereo[28..32].copy_from_slice(&192_000_u32.to_le_bytes()); // byte_rate = 48_000 * 4
        odd_stereo[32..34].copy_from_slice(&4_u16.to_le_bytes()); // block_align = 4
        odd_stereo.extend_from_slice(b"data");
        odd_stereo.extend_from_slice(&2_u32.to_le_bytes()); // only 1 sample (2 bytes) for 2 channels!
        odd_stereo.extend_from_slice(&[0, 0]);
        let odd_stereo_path = dir.join("odd_stereo.wav");
        fs::write(&odd_stereo_path, &odd_stereo).expect("write odd_stereo");
        assert!(read_wav(&odd_stereo_path).is_err());

        // Reject oversized RIFF ckSize (8 + riff_size > bytes.len()), riff_size < 4, body.len() == 17, and 18 + cbSize > body.len()
        let mut bad_riff = ext_wav.clone();
        bad_riff[4..8].copy_from_slice(&100_u32.to_le_bytes());
        let bad_riff_path = dir.join("bad_riff.wav");
        fs::write(&bad_riff_path, &bad_riff).expect("write bad_riff");
        assert!(read_wav(&bad_riff_path).is_err());
        bad_riff[4..8].copy_from_slice(&2_u32.to_le_bytes());
        fs::write(&bad_riff_path, &bad_riff).expect("write tiny_riff");
        assert!(read_wav(&bad_riff_path).is_err());
        assert!(parse_wav_fmt_chunk(&[0u8; 17]).is_err());
        let mut cb_over = ext_wav[34..74].to_vec();
        cb_over[16..18].copy_from_slice(&64_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&cb_over).is_err());

        // Truncated unknown RIFF chunk (e.g. JUNK claiming 50 bytes with only 4 present)
        let mut trunc_junk = no_data[..36].to_vec();
        trunc_junk.extend_from_slice(b"JUNK");
        trunc_junk.extend_from_slice(&50_u32.to_le_bytes());
        trunc_junk.extend_from_slice(&[1, 2, 3, 4]);
        let trunc_junk_path = dir.join("trunc_junk.wav");
        fs::write(&trunc_junk_path, &trunc_junk).expect("write trunc_junk");
        assert!(read_wav(&trunc_junk_path).is_err());

        // Odd-byte 24-bit data chunk missing its RIFF word-alignment pad byte + duplicate data chunk
        let mut missing_pad = raw_24_bytes[..raw_24_bytes.len() - 1].to_vec();
        let unpadded_riff = (missing_pad.len() - 8) as u32;
        missing_pad[4..8].copy_from_slice(&unpadded_riff.to_le_bytes());
        let missing_pad_path = dir.join("missing_pad.wav");
        fs::write(&missing_pad_path, &missing_pad).expect("write missing_pad");
        assert!(read_wav(&missing_pad_path).is_err());

        let mut dup_data = fs::read(&p16).expect("read valid 16-bit");
        dup_data.extend_from_slice(b"data");
        dup_data.extend_from_slice(&4_u32.to_le_bytes());
        dup_data.extend_from_slice(&[0, 0, 0, 0]);
        let dup_riff = (dup_data.len() - 8) as u32;
        dup_data[4..8].copy_from_slice(&dup_riff.to_le_bytes());
        let dup_data_path = dir.join("dup_data.wav");
        fs::write(&dup_data_path, &dup_data).expect("write dup_data");
        assert!(read_wav(&dup_data_path).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
