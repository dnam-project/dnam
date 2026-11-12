#![forbid(unsafe_code)]
//! Binary Builders & Verifiers for `.dnam-proj` (`DNPJ`) Payload Capsules (`proj::capsules`):
//! `CALB` (`64 B`), `INPC` (`24,640 B` / `48,640 B`), `STMC` (`28,800 B` / `52,800 B`),
//! and `STAT` (`uplo='U'` `StarCenter` sufficient statistics).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md`
//! (`ref[container.proj.inpc-capsule]`, `ref[container.proj.stmc-capsule]`, `ref[container.proj.stat-uplo-u]`).

use crate::solver::uplo_u_index;
use dnam_engine::container::crc32c_compute;
use dnam_engine::schema::DnamError;

use super::{
    INPC_MAGIC, INPC_MONO_BYTES, INPC_STEREO_BYTES, PROJ_STAT_BYTES_FULL, PROJ_STAT_BYTES_LITE,
    PROJ_STAT_BYTES_PRO, PROJ_STAT_BYTES_ULTRA, STMC_FLAG_ENABLED, STMC_FLAG_KNOWN_MASK,
    STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO, STMC_MAGIC, STMC_MAX_TRAJECTORY_WINDOWS,
    STMC_MONO_BYTES, STMC_STEREO_BYTES,
};

/// Fixed `64 B` `CALB` loopback calibration chunk (`r[container.proj.calb]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalbChunk64B {
    /// Device sample rate in Hz (`8000..=384000`).
    pub sample_rate_device_hz: u32,
    /// Bit-depth code (`0 = Int12`, `1 = Int16`, `2 = Int24`, `3 = Int32`, `4 = Float32`).
    pub bit_depth_code: u8,
    /// Polarity inversion flag (`0` or `1`).
    pub polarity_inverted: bool,
    /// Sub-sample loopback latency in samples at `48 kHz`.
    pub loopback_lag_samples: f64,
    /// Signed linear loopback gain.
    pub signed_gain: f64,
    /// Calibrated input reference level in dBFS.
    pub input_reference_dbfs: f32,
    /// Measured quiescent hardware noise floor in dBFS.
    pub noise_floor_dbfs: f32,
}

impl CalbChunk64B {
    /// Serializes this `CALB` record into an exact `64 B` little-endian array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(b"CALB");
        out[4..6].copy_from_slice(&2_u16.to_le_bytes());
        out[6] = self.bit_depth_code;
        out[7] = u8::from(self.polarity_inverted);
        out[8..12].copy_from_slice(&self.sample_rate_device_hz.to_le_bytes());
        out[16..24].copy_from_slice(&self.loopback_lag_samples.to_le_bytes());
        out[24..32].copy_from_slice(&self.signed_gain.to_le_bytes());
        out[32..36].copy_from_slice(&self.input_reference_dbfs.to_le_bytes());
        out[36..40].copy_from_slice(&self.noise_floor_dbfs.to_le_bytes());
        let body_crc = crc32c_compute(&out[0..60]);
        out[60..64].copy_from_slice(&body_crc.to_le_bytes());
        out
    }

    /// Parses and validates a `64 B` `CALB` chunk.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        if bytes.len() != 64 {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        }
        if bytes[0..4] != *b"CALB" {
            return Err(DnamError::InvalidMagic([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]));
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        if ver != 2 {
            return Err(DnamError::UnsupportedVersion {
                major: ver,
                minor: 0,
            });
        }
        let bit_depth_code = bytes[6];
        let pol_byte = bytes[7];
        if bit_depth_code > 4 || pol_byte > 1 || bytes[12..16] != [0; 4] || bytes[40..60] != [0; 20]
        {
            return Err(DnamError::SchemaValidationError(
                "invalid CALB flags or non-zero reserved bytes".to_string(),
            ));
        }
        let expected_crc = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);
        let calc_crc = crc32c_compute(&bytes[0..60]);
        if calc_crc != expected_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "CALB",
                expected: expected_crc,
                computed: calc_crc,
            });
        }
        let sample_rate_device_hz = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if !(8_000..=384_000).contains(&sample_rate_device_hz) {
            return Err(DnamError::InvalidSampleRate(sample_rate_device_hz));
        }
        let loopback_lag_samples = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let signed_gain = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let input_reference_dbfs = f32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let noise_floor_dbfs = f32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        if !loopback_lag_samples.is_finite()
            || !signed_gain.is_finite()
            || !input_reference_dbfs.is_finite()
            || !noise_floor_dbfs.is_finite()
        {
            return Err(DnamError::SchemaValidationError(
                "non-finite float in CALB chunk".to_string(),
            ));
        }
        Ok(Self {
            sample_rate_device_hz,
            bit_depth_code,
            polarity_inverted: pol_byte != 0,
            loopback_lag_samples,
            signed_gain,
            input_reference_dbfs,
            noise_floor_dbfs,
        })
    }
}

fn verify_finite_fp16_stream(fp16_bytes: &[u8], label: &'static str) -> Result<(), DnamError> {
    for pair in fp16_bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        if (bits & 0x7C00) == 0x7C00 {
            return Err(DnamError::SchemaValidationError(format!(
                "non-finite IEEE-754 binary16 sample ({bits:#06x}) in {label}"
            )));
        }
    }
    Ok(())
}

/// Validated zero-copy view over a `24,640 B` mono or `48,640 B` stereo `INPC` dry stimulus capsule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InpcCapsuleView<'a> {
    /// Channel count (`1` = mono, `2` = stereo).
    pub channels: u16,
    /// Canonical sample rate (`48,000 Hz`).
    pub sample_rate_hz: u32,
    /// Coreset frame count (`12,000`).
    pub num_frames: u32,
    /// `256 B` deterministic stimulus seed $\boldsymbol{\Xi}$.
    pub seed_256: &'a [u8],
    /// `320 B` `AdaptiveStimulusController` frame schedule / Stage-II basis summary.
    pub schedule_320: &'a [u8],
    /// `24,000 B` (mono) or `48,000 B` (stereo) RNE `fp16` dry coreset payload.
    pub dry_coreset_fp16: &'a [u8],
}

/// Builds a canonical `24,640 B` (mono) or `48,640 B` (stereo) `INPC` dry stimulus capsule (`r[container.proj.inpc]`).
pub fn build_proj_inpc_capsule(
    channels: u16,
    seed_256: &[u8; 256],
    schedule_320: &[u8; 320],
    dry_coreset_fp16: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let expected_coreset = match channels {
        1 => 24_000usize,
        2 => 48_000usize,
        _ => {
            return Err(DnamError::SchemaValidationError(format!(
                "INPC channels must be 1 or 2, got {channels}"
            )));
        }
    };
    if dry_coreset_fp16.len() != expected_coreset {
        return Err(DnamError::SchemaValidationError(format!(
            "INPC dry_coreset_fp16 length {} != expected {expected_coreset}",
            dry_coreset_fp16.len()
        )));
    }
    verify_finite_fp16_stream(dry_coreset_fp16, "INPC")?;

    let rec_len = 640 + expected_coreset;
    let mut out = vec![0u8; rec_len];
    out[0..4].copy_from_slice(&INPC_MAGIC);
    out[4..6].copy_from_slice(&2_u16.to_le_bytes());
    out[6..8].copy_from_slice(&channels.to_le_bytes());
    out[8..12].copy_from_slice(&48_000_u32.to_le_bytes());
    out[12..16].copy_from_slice(&12_000_u32.to_le_bytes());
    out[56..60].copy_from_slice(&(rec_len as u32).to_le_bytes());
    out[64..320].copy_from_slice(seed_256);
    out[320..640].copy_from_slice(schedule_320);
    out[640..rec_len].copy_from_slice(dry_coreset_fp16);
    let crc = crc32c_compute(&out[64..rec_len]);
    out[60..64].copy_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Validates a `24,640 B` mono or `48,640 B` stereo `INPC` dry stimulus capsule (`r[container.proj.inpc]`).
pub fn verify_proj_inpc_capsule(bytes: &[u8]) -> Result<InpcCapsuleView<'_>, DnamError> {
    if bytes.len() < 64 {
        return Err(DnamError::BufferTooSmall {
            expected: 64,
            actual: bytes.len(),
        });
    }
    if bytes[0..4] != INPC_MAGIC {
        return Err(DnamError::InvalidMagic([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]));
    }
    let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
    if ver != 2 {
        return Err(DnamError::UnsupportedVersion {
            major: ver,
            minor: 0,
        });
    }
    let channels = u16::from_le_bytes([bytes[6], bytes[7]]);
    let sample_rate_hz = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let num_frames = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    if bytes[16..56].iter().any(|&b| b != 0) {
        return Err(DnamError::SchemaValidationError(
            "non-zero reserved bytes in INPC[16..56]".to_string(),
        ));
    }
    let expected_len = match channels {
        1 => INPC_MONO_BYTES,
        2 => INPC_STEREO_BYTES,
        _ => {
            return Err(DnamError::SchemaValidationError(format!(
                "invalid INPC channels {channels}"
            )));
        }
    };
    let rec_len = u32::from_le_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]) as usize;
    if sample_rate_hz != 48_000 || num_frames != 12_000 || rec_len != expected_len {
        return Err(DnamError::SchemaValidationError(
            "invalid INPC sample_rate_hz, num_frames, or rec_len".to_string(),
        ));
    }
    if bytes.len() != expected_len {
        return Err(DnamError::BufferTooSmall {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    let expected_crc = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]);
    let calc_crc = crc32c_compute(&bytes[64..rec_len]);
    if calc_crc != expected_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "INPC",
            expected: expected_crc,
            computed: calc_crc,
        });
    }
    let dry_coreset_fp16 = &bytes[640..rec_len];
    verify_finite_fp16_stream(dry_coreset_fp16, "INPC")?;
    Ok(InpcCapsuleView {
        channels,
        sample_rate_hz,
        num_frames,
        seed_256: &bytes[64..320],
        schedule_320: &bytes[320..640],
        dry_coreset_fp16,
    })
}

/// Single `16 B` entry in the `576 B` (`R <= 36`) Windowed Control Trajectory Table of `STMC`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StmcTrajectoryWindow {
    /// Control index `p` in `manifest.base.controls_schema`.
    pub control_id: u16,
    /// Control kind (`0` = Range, `1` = Switch).
    pub ctrl_kind: u8,
    /// Active discrete switch option index (`0` for Range controls).
    pub active_opt: u8,
    /// Physical control value $u_{p,r}$ in this `96 ms` window.
    pub u_phys: f32,
    /// Leverage / SNR quadrature weight $w_r \ge 0$ for this window.
    pub window_weight: f32,
    /// Measured EMR velocity metric in this window.
    pub emr_velocity: f32,
}

impl StmcTrajectoryWindow {
    /// Encodes this window into `16 B` little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..2].copy_from_slice(&self.control_id.to_le_bytes());
        out[2] = self.ctrl_kind;
        out[3] = self.active_opt;
        out[4..8].copy_from_slice(&self.u_phys.to_le_bytes());
        out[8..12].copy_from_slice(&self.window_weight.to_le_bytes());
        out[12..16].copy_from_slice(&self.emr_velocity.to_le_bytes());
        out
    }

    /// Decodes and validates a `16 B` trajectory window.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        let control_id = u16::from_le_bytes([bytes[0], bytes[1]]);
        let ctrl_kind = bytes[2];
        let active_opt = bytes[3];
        let u_phys = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let window_weight = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let emr_velocity = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if ctrl_kind > 1
            || !u_phys.is_finite()
            || !window_weight.is_finite()
            || window_weight < 0.0
            || !emr_velocity.is_finite()
        {
            return Err(DnamError::SchemaValidationError(
                "invalid StmcTrajectoryWindow ctrl_kind or non-finite float".to_string(),
            ));
        }
        Ok(Self {
            control_id,
            ctrl_kind,
            active_opt,
            u_phys,
            window_weight,
            emr_velocity,
        })
    }
}

/// Validated view over a `28,800 B` (mono) or `52,800 B` (stereo) `STMC` `StemCaptureRecord` capsule.
#[derive(Debug, Clone, PartialEq)]
pub struct StmcCapsuleView<'a> {
    /// Numeric stem index (`0..=65535`).
    pub stem_index: u16,
    /// `STMC_FLAG_*` bitmask (Bit 0 = `enabled`).
    pub flags: u16,
    /// Analog/digital input pad gain applied during capture.
    pub pad_gain_f32: f32,
    /// Sub-sample alignment lag applied to this stem.
    pub fractional_lag_samples_f32: f32,
    /// Decoded trajectory windows (`1..=36`).
    pub windows: Vec<StmcTrajectoryWindow>,
    /// `4,096 B` spectral, sideband & ballistic summary slice (`704..4_800`).
    pub spectral_summary_4096: &'a [u8],
    /// `24,000 B` (mono) or `48,000 B` (stereo) RNE `fp16` wet coreset slice (`4_800..rec_len`).
    pub wet_coreset_fp16: &'a [u8],
}

impl StmcCapsuleView<'_> {
    /// Returns `true` if `STMC_FLAG_ENABLED` (`Bit 0`) is set.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        (self.flags & STMC_FLAG_ENABLED) != 0
    }

    /// Returns `true` if `STMC_FLAG_STAR_CENTER` (`Bit 1`) is set.
    #[must_use]
    pub const fn is_star_center(&self) -> bool {
        (self.flags & STMC_FLAG_STAR_CENTER) != 0
    }
}

/// Builds a canonical `28,800 B` (mono) or `52,800 B` (stereo) `STMC` `StemCaptureRecord` capsule (`r[container.proj.stmc]`).
#[allow(clippy::too_many_arguments)]
pub fn build_proj_stmc_capsule(
    stem_index: u16,
    flags: u16,
    pad_gain_f32: f32,
    fractional_lag_samples_f32: f32,
    band_psd_summary_88: &[u8; 88],
    n_comb_bins: u32,
    windows: &[StmcTrajectoryWindow],
    spectral_summary_4096: &[u8; 4_096],
    wet_coreset_fp16: &[u8],
) -> Result<Vec<u8>, DnamError> {
    if (flags & !STMC_FLAG_KNOWN_MASK) != 0 {
        return Err(DnamError::SchemaValidationError(format!(
            "unknown STMC flags {flags:#06x}"
        )));
    }
    if windows.is_empty() || windows.len() > STMC_MAX_TRAJECTORY_WINDOWS {
        return Err(DnamError::SchemaValidationError(format!(
            "STMC windows count {} must be in 1..={STMC_MAX_TRAJECTORY_WINDOWS}",
            windows.len()
        )));
    }
    if !pad_gain_f32.is_finite() || !fractional_lag_samples_f32.is_finite() {
        return Err(DnamError::SchemaValidationError(
            "non-finite pad_gain_f32 or fractional_lag_samples_f32 in STMC".to_string(),
        ));
    }
    let is_stereo = (flags & STMC_FLAG_STEREO) != 0;
    let expected_coreset = if is_stereo { 48_000 } else { 24_000 };
    if wet_coreset_fp16.len() != expected_coreset {
        return Err(DnamError::SchemaValidationError(format!(
            "STMC wet_coreset_fp16 length {} != expected {expected_coreset}",
            wet_coreset_fp16.len()
        )));
    }
    verify_finite_fp16_stream(wet_coreset_fp16, "STMC")?;

    let rec_len = 4_800 + expected_coreset;
    let mut out = vec![0u8; rec_len];
    out[0..4].copy_from_slice(&STMC_MAGIC);
    out[4..6].copy_from_slice(&2_u16.to_le_bytes());
    out[6..8].copy_from_slice(&stem_index.to_le_bytes());
    out[8..10].copy_from_slice(&flags.to_le_bytes());
    out[10..12].copy_from_slice(&(windows.len() as u16).to_le_bytes());
    out[12..16].copy_from_slice(&pad_gain_f32.to_le_bytes());
    out[16..20].copy_from_slice(&fractional_lag_samples_f32.to_le_bytes());
    out[20..108].copy_from_slice(band_psd_summary_88);
    out[108..112].copy_from_slice(&12_000_u32.to_le_bytes());
    out[112..116].copy_from_slice(&n_comb_bins.to_le_bytes());
    out[116..120].copy_from_slice(&(rec_len as u32).to_le_bytes());
    for (idx, w) in windows.iter().enumerate() {
        let _ = StmcTrajectoryWindow::from_bytes(&w.to_bytes())?;
        let off = 128 + idx * 16;
        out[off..off + 16].copy_from_slice(&w.to_bytes());
    }
    out[704..4_800].copy_from_slice(spectral_summary_4096);
    out[4_800..rec_len].copy_from_slice(wet_coreset_fp16);
    let body_crc = crc32c_compute(&out[128..rec_len]);
    out[120..124].copy_from_slice(&body_crc.to_le_bytes());
    Ok(out)
}

/// Toggles `STMC_FLAG_ENABLED` (`Bit 0`) in-place inside a validated `STMC` capsule byte buffer.
pub fn set_stmc_capsule_enabled_flag(
    stmc_bytes: &mut [u8],
    enabled: bool,
) -> Result<(), DnamError> {
    let view = verify_proj_stmc_capsule(stmc_bytes)?;
    let new_flags = if enabled {
        view.flags | STMC_FLAG_ENABLED
    } else {
        view.flags & !STMC_FLAG_ENABLED
    };
    stmc_bytes[8..10].copy_from_slice(&new_flags.to_le_bytes());
    Ok(())
}

/// Toggles `STMC_FLAG_STAR_CENTER` (`Bit 1`) in-place inside a validated `STMC` capsule byte buffer.
pub fn set_stmc_capsule_star_center_flag(
    stmc_bytes: &mut [u8],
    is_star_center: bool,
) -> Result<(), DnamError> {
    let view = verify_proj_stmc_capsule(stmc_bytes)?;
    let new_flags = if is_star_center {
        view.flags | STMC_FLAG_STAR_CENTER
    } else {
        view.flags & !STMC_FLAG_STAR_CENTER
    };
    stmc_bytes[8..10].copy_from_slice(&new_flags.to_le_bytes());
    Ok(())
}

/// Validates a `28,800 B` mono or `52,800 B` stereo `STMC` `StemCaptureRecord` capsule (`r[container.proj.stmc]`).
pub fn verify_proj_stmc_capsule(bytes: &[u8]) -> Result<StmcCapsuleView<'_>, DnamError> {
    if bytes.len() < 128 {
        return Err(DnamError::BufferTooSmall {
            expected: 128,
            actual: bytes.len(),
        });
    }
    if bytes[0..4] != STMC_MAGIC {
        return Err(DnamError::InvalidMagic([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]));
    }
    let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
    if ver != 2 {
        return Err(DnamError::UnsupportedVersion {
            major: ver,
            minor: 0,
        });
    }
    let stem_index = u16::from_le_bytes([bytes[6], bytes[7]]);
    let flags = u16::from_le_bytes([bytes[8], bytes[9]]);
    if (flags & !STMC_FLAG_KNOWN_MASK) != 0 {
        return Err(DnamError::SchemaValidationError(format!(
            "unknown STMC flags {flags:#06x}"
        )));
    }
    let n_windows = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
    if n_windows == 0 || n_windows > STMC_MAX_TRAJECTORY_WINDOWS {
        return Err(DnamError::SchemaValidationError(format!(
            "invalid STMC n_trajectory_windows {n_windows}"
        )));
    }
    let pad_gain_f32 = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let fractional_lag_samples_f32 =
        f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    if !pad_gain_f32.is_finite() || !fractional_lag_samples_f32.is_finite() {
        return Err(DnamError::SchemaValidationError(
            "non-finite pad_gain_f32 or fractional_lag_samples_f32 in STMC".to_string(),
        ));
    }
    let n_coreset_frames = u32::from_le_bytes([bytes[108], bytes[109], bytes[110], bytes[111]]);
    let rec_len = u32::from_le_bytes([bytes[116], bytes[117], bytes[118], bytes[119]]) as usize;
    let expected_crc = u32::from_le_bytes([bytes[120], bytes[121], bytes[122], bytes[123]]);
    if bytes[124..128] != [0; 4] {
        return Err(DnamError::SchemaValidationError(
            "non-zero reserved bytes in STMC[124..128]".to_string(),
        ));
    }
    let expected_len = if (flags & STMC_FLAG_STEREO) != 0 {
        STMC_STEREO_BYTES
    } else {
        STMC_MONO_BYTES
    };
    if n_coreset_frames != 12_000 || rec_len != expected_len {
        return Err(DnamError::SchemaValidationError(
            "invalid STMC n_coreset_frames or rec_len".to_string(),
        ));
    }
    if bytes.len() != expected_len {
        return Err(DnamError::BufferTooSmall {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    let calc_crc = crc32c_compute(&bytes[128..rec_len]);
    if calc_crc != expected_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "STMC",
            expected: expected_crc,
            computed: calc_crc,
        });
    }
    let mut windows = Vec::with_capacity(n_windows);
    for idx in 0..n_windows {
        let off = 128 + idx * 16;
        windows.push(StmcTrajectoryWindow::from_bytes(&bytes[off..off + 16])?);
    }
    let tail_start = 128 + n_windows * 16;
    if bytes[tail_start..704].iter().any(|&b| b != 0) {
        return Err(DnamError::SchemaValidationError(
            "non-zero unused trajectory window slot in STMC[..704]".to_string(),
        ));
    }
    let wet_coreset_fp16 = &bytes[4_800..rec_len];
    verify_finite_fp16_stream(wet_coreset_fp16, "STMC")?;
    Ok(StmcCapsuleView {
        stem_index,
        flags,
        pad_gain_f32,
        fractional_lag_samples_f32,
        windows,
        spectral_summary_4096: &bytes[704..4_800],
        wet_coreset_fp16,
    })
}

/// Validates a `STAT` `StarCenter` `uplo='U'` sufficient-statistics chunk (`r[container.proj.stat-uplo-u]`),
/// returning the retained tier dimension (`64`, `128`, `192`, or `256`).
pub fn verify_proj_stat_uplo_u(bytes: &[u8]) -> Result<usize, DnamError> {
    let dim = match bytes.len() {
        PROJ_STAT_BYTES_LITE => 64,
        PROJ_STAT_BYTES_FULL => 128,
        PROJ_STAT_BYTES_PRO => 192,
        PROJ_STAT_BYTES_ULTRA => 256,
        other => {
            return Err(DnamError::SchemaValidationError(format!(
                "invalid STAT byte length {other}: expected {PROJ_STAT_BYTES_LITE}, {PROJ_STAT_BYTES_FULL}, {PROJ_STAT_BYTES_PRO}, or {PROJ_STAT_BYTES_ULTRA}"
            )));
        }
    };
    for elem in bytes.as_chunks::<8>().0 {
        let val = f64::from_le_bytes(*elem);
        if !val.is_finite() {
            return Err(DnamError::SchemaValidationError(
                "non-finite f64 element in STAT uplo='U' chunk".to_string(),
            ));
        }
    }
    for j in 0..dim {
        let off = uplo_u_index(j, j) * 8;
        let g_jj = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        if g_jj < 0.0 {
            return Err(DnamError::SchemaValidationError(format!(
                "negative Gram diagonal G[{j},{j}] = {g_jj} in STAT uplo='U' chunk"
            )));
        }
    }
    Ok(dim)
}
