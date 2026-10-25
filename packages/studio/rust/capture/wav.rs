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

        // 24-bit write & read
        let p24 = dir.join("m24.wav");
        let sig = vec![0.0, 0.25, -0.5, 0.75, -1.0];
        write_wav_24bit(&p24, &sig, 48_000, 1).expect("write 24-bit");
        let r24 = read_wav(&p24).expect("read 24-bit");
        assert_eq!(r24.sample_rate_hz, 48_000);
        assert_eq!(r24.channels, 1);
        assert_eq!(r24.samples.len(), 5);

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

        // Valid RIFF/WAVE header but missing data chunk
        let mut no_data = vec![0_u8; 44];
        no_data[0..4].copy_from_slice(b"RIFF");
        no_data[8..12].copy_from_slice(b"WAVE");
        no_data[12..16].copy_from_slice(b"fmt ");
        no_data[16..20].copy_from_slice(&16_u32.to_le_bytes());
        no_data[36..40].copy_from_slice(b"JUNK");
        no_data[40..44].copy_from_slice(&0_u32.to_le_bytes());
        let no_data_path = dir.join("no_data.wav");
        fs::write(&no_data_path, &no_data).expect("write no_data");
        assert!(read_wav(&no_data_path).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
