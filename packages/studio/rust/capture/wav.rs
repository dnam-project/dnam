//! 16/24/32-bit PCM WAV reader, 24-bit PCM WAV writer, and IEEE-754 `<f4` SHA-256 digest.

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

/// Writes `samples` (`[-1.0, 1.0]`) to a 24-bit signed PCM RIFF/WAVE file at `path`.
pub fn write_wav_24bit(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> StudioResult<()> {
    if channels == 0 {
        return Err(StudioError::Audio("channels must be >= 1".to_string()));
    }
    let bytes_per_sample: u16 = 3;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate_hz * u32::from(block_align);
    let data_bytes_len = u32::try_from(samples.len() * 3)
        .map_err(|_| StudioError::Audio("WAV payload exceeds 4 GiB limit".to_string()))?;
    let riff_size = 36 + data_bytes_len;

    let mut buf = Vec::with_capacity(44 + samples.len() * 3);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16_u32.to_le_bytes());
    buf.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate_hz.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&24_u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes_len.to_le_bytes());

    let scale = 8_388_607.0_f32;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let q = (clamped * scale).round() as i32;
        let le = q.to_le_bytes();
        buf.push(le[0]);
        buf.push(le[1]);
        buf.push(le[2]);
    }

    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, buf)?;
    Ok(())
}

/// Reads a 16-bit, 24-bit, or 32-bit PCM RIFF/WAVE file from `path`.
pub fn read_wav(path: impl AsRef<Path>) -> StudioResult<WavData> {
    let bytes = fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(StudioError::Audio("Invalid RIFF/WAVE header".to_string()));
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate_hz = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits_per_sample = u16::from_le_bytes([bytes[34], bytes[35]]);

    let mut offset = 12_usize;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;
        if chunk_id == b"data" {
            let end = (offset + chunk_len).min(bytes.len());
            let payload = &bytes[offset..end];
            let samples = decode_pcm_payload(payload, bits_per_sample)?;
            return Ok(WavData {
                sample_rate_hz,
                channels,
                samples,
            });
        }
        offset = offset.saturating_add(chunk_len);
    }

    Err(StudioError::Audio(
        "Missing data chunk in WAV file".to_string(),
    ))
}

fn decode_pcm_payload(payload: &[u8], bits_per_sample: u16) -> StudioResult<Vec<f32>> {
    match bits_per_sample {
        16 => {
            let mut out = Vec::with_capacity(payload.len() / 2);
            for &chunk in payload.as_chunks::<2>().0 {
                let v = i16::from_le_bytes(chunk);
                out.push(f32::from(v) / 32_768.0);
            }
            Ok(out)
        }
        24 => {
            let mut out = Vec::with_capacity(payload.len() / 3);
            for &chunk in payload.as_chunks::<3>().0 {
                let sign_ext = if (chunk[2] & 0x80) != 0 { 0xFF } else { 0x00 };
                let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], sign_ext]);
                out.push((v as f32) / 8_388_608.0);
            }
            Ok(out)
        }
        32 => {
            let mut out = Vec::with_capacity(payload.len() / 4);
            for &chunk in payload.as_chunks::<4>().0 {
                let v = i32::from_le_bytes(chunk);
                out.push((v as f32) / 2_147_483_648.0);
            }
            Ok(out)
        }
        other => Err(StudioError::Audio(format!(
            "Unsupported WAV bit depth: {other}"
        ))),
    }
}
