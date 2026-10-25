//! Pluggable hardware and software re-amp capture backends (`software`, `usb`, `anagram`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::capture::alignment::estimate_subsample_lag_and_snr;
use crate::capture::excitation::generate_calibration_probe;
use crate::error::StudioResult;

/// Audio device descriptor returned by `list_backend_devices`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioDeviceDescriptor {
    /// Backend identifier (`"software"`, `"usb"`, `"anagram"`).
    pub backend: String,
    /// Device identifier.
    pub device_id: String,
    /// Human-readable device name.
    pub name: String,
    /// Supported sample rate in Hz.
    pub sample_rate_hz: u32,
}

/// Result of a 2-step hardware/software interface calibration run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Backend identifier.
    pub backend: String,
    /// Device identifier.
    pub device_id: String,
    /// Measured round-trip latency in samples.
    pub latency_samples: i32,
    /// Measured input noise floor in dBFS.
    pub noise_floor_dbfs: f32,
    /// Measured calibration SNR in dB.
    pub snr_db: f32,
}

/// Pluggable re-amp capture backend trait.
pub trait CaptureBackend {
    /// Backend identifier (`"software"`, `"usb"`, `"anagram"`).
    fn backend_name(&self) -> &'static str;

    /// Performs round-trip latency and noise floor calibration.
    fn calibrate(&self, device_id: &str, sample_rate_hz: u32) -> StudioResult<CalibrationResult>;

    /// Executes a synchronous dry-to-wet re-amp capture pass.
    fn capture_stem(
        &self,
        dry_samples: &[f32],
        normalized_ranges: &BTreeMap<String, f32>,
        switches: &BTreeMap<String, usize>,
    ) -> StudioResult<Vec<f32>>;
}

/// Lists all supported re-amp capture backends.
pub fn list_capture_backends() -> Vec<&'static str> {
    vec!["software", "usb", "anagram"]
}

/// Lists available devices for `backend`.
pub fn list_backend_devices(backend: &str) -> Vec<AudioDeviceDescriptor> {
    match backend {
        "anagram" => vec![AudioDeviceDescriptor {
            backend: "anagram".to_string(),
            device_id: "anagram_rk3582_usb".to_string(),
            name: "Darkglass Anagram RK3582 Re-Amp Interface".to_string(),
            sample_rate_hz: 48_000,
        }],
        "usb" => vec![AudioDeviceDescriptor {
            backend: "usb".to_string(),
            device_id: "usb_class_compliant_0".to_string(),
            name: "USB Audio Class 2.0 24-Bit Re-Amp Interface".to_string(),
            sample_rate_hz: 48_000,
        }],
        _ => vec![AudioDeviceDescriptor {
            backend: "software".to_string(),
            device_id: "loopback_default".to_string(),
            name: "DNAM Deterministic Software Re-Amp Loopback".to_string(),
            sample_rate_hz: 48_000,
        }],
    }
}

/// Deterministic software nonlinear loopback backend for headless CLI and CI verification.
#[derive(Debug, Clone, Copy)]
pub struct SoftwareLoopbackBackend {
    /// Simulated round-trip latency in samples.
    pub simulated_latency_samples: usize,
}

impl Default for SoftwareLoopbackBackend {
    fn default() -> Self {
        Self {
            simulated_latency_samples: 6,
        }
    }
}

impl CaptureBackend for SoftwareLoopbackBackend {
    fn backend_name(&self) -> &'static str {
        "software"
    }

    fn calibrate(&self, device_id: &str, sample_rate_hz: u32) -> StudioResult<CalibrationResult> {
        let dry = generate_calibration_probe(sample_rate_hz, 2_048);
        let wet = self.capture_stem(&dry, &BTreeMap::new(), &BTreeMap::new())?;
        let report = estimate_subsample_lag_and_snr(&dry, &wet, 32)?;
        Ok(CalibrationResult {
            backend: "software".to_string(),
            device_id: device_id.to_string(),
            latency_samples: report.coarse_lag_samples,
            noise_floor_dbfs: report.noise_floor_dbfs,
            snr_db: report.snr_db,
        })
    }

    fn capture_stem(
        &self,
        dry_samples: &[f32],
        normalized_ranges: &BTreeMap<String, f32>,
        switches: &BTreeMap<String, usize>,
    ) -> StudioResult<Vec<f32>> {
        let avg_range = if normalized_ranges.is_empty() {
            0.5_f32
        } else {
            normalized_ranges.values().copied().sum::<f32>() / (normalized_ranges.len() as f32)
        };
        let sw_boost = switches
            .values()
            .next()
            .copied()
            .map(|idx| 1.0 + 0.15 * (idx as f32))
            .unwrap_or(1.0);
        let drive = (1.2 + 2.4 * avg_range) * sw_boost;
        let mut wet = vec![0.0_f32; dry_samples.len()];
        let lag = self.simulated_latency_samples;
        for i in lag..dry_samples.len() {
            let x = dry_samples[i - lag] * drive;
            wet[i] = x / (1.0 + x.abs());
        }
        Ok(wet)
    }
}
