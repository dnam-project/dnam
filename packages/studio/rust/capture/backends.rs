//! Pluggable hardware and software re-amp capture backends (`software`, `usb`, `anagram`)
//! powered by `cpal` full-duplex audio I/O with 2-Step Reference Unity & Engaged Hot-Corner
//! probe calibration.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

use crate::capture::alignment::{
    analyze_hot_corner_nonlinearity, apply_subsample_shift_fft, estimate_subsample_lag_and_snr,
    measure_silent_window_noise_floor,
};
use crate::capture::excitation::{
    TargetExcitationProfile, analyze_probe_for_excitation, generate_calibration_probe,
};
use crate::error::{StudioError, StudioResult};

#[repr(align(64))]
#[derive(Debug, Default)]
struct CachePaddedAtomicUsize(AtomicUsize);

/// Wait-free, 100% `#![forbid(unsafe_code)]` Single-Producer Single-Consumer (SPSC) ring buffer
/// for real-time `f32` audio callbacks (`cpal`).
///
/// Uses power-of-two capacity masking and 64-byte cache-line-isolated `head` and `tail` cursors
/// to eliminate false sharing between the real-time audio thread and worker thread.
#[derive(Debug)]
pub struct AtomicF32RingBuffer {
    slots: Box<[AtomicU32]>,
    mask: usize,
    write_idx: CachePaddedAtomicUsize,
    read_idx: CachePaddedAtomicUsize,
}

impl AtomicF32RingBuffer {
    /// Creates a new [`AtomicF32RingBuffer`] rounded up to the next power-of-two capacity (`>= 2`).
    pub fn with_capacity(min_capacity: usize) -> Self {
        let cap = min_capacity.max(2).next_power_of_two();
        let slots: Vec<AtomicU32> = (0..cap).map(|_| AtomicU32::new(0)).collect();
        Self {
            slots: slots.into_boxed_slice(),
            mask: cap - 1,
            write_idx: CachePaddedAtomicUsize(AtomicUsize::new(0)),
            read_idx: CachePaddedAtomicUsize(AtomicUsize::new(0)),
        }
    }

    /// Returns the total usable capacity of the ring buffer in samples.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Returns the number of samples currently available to read.
    #[inline]
    pub fn available_read(&self) -> usize {
        let w = self.write_idx.0.load(Ordering::Acquire);
        let r = self.read_idx.0.load(Ordering::Relaxed);
        w.wrapping_sub(r)
    }

    /// Returns the number of free sample slots currently available to write.
    #[inline]
    pub fn available_write(&self) -> usize {
        self.capacity().saturating_sub(self.available_read())
    }

    /// Pushes as many samples from `src` as fit without blocking; returns the number of samples written.
    #[inline]
    pub fn push_slice(&self, src: &[f32]) -> usize {
        let w = self.write_idx.0.load(Ordering::Relaxed);
        let r = self.read_idx.0.load(Ordering::Acquire);
        let free = self.capacity().saturating_sub(w.wrapping_sub(r));
        let n = src.len().min(free);
        for (i, &sample) in src.iter().take(n).enumerate() {
            self.slots[(w.wrapping_add(i)) & self.mask].store(sample.to_bits(), Ordering::Relaxed);
        }
        self.write_idx.0.store(w.wrapping_add(n), Ordering::Release);
        n
    }

    /// Pops up to `dst.len()` samples without blocking; returns the number of samples read.
    #[inline]
    pub fn pop_slice(&self, dst: &mut [f32]) -> usize {
        let r = self.read_idx.0.load(Ordering::Relaxed);
        let w = self.write_idx.0.load(Ordering::Acquire);
        let avail = w.wrapping_sub(r);
        let n = dst.len().min(avail);
        for (i, out_s) in dst.iter_mut().take(n).enumerate() {
            let bits = self.slots[(r.wrapping_add(i)) & self.mask].load(Ordering::Relaxed);
            *out_s = f32::from_bits(bits);
        }
        self.read_idx.0.store(r.wrapping_add(n), Ordering::Release);
        n
    }
}

/// Audio device descriptor returned by [`list_backend_devices`].
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
    /// Maximum hardware input channels reported by `cpal`.
    #[serde(default)]
    pub max_input_channels: u16,
    /// Maximum hardware output channels reported by `cpal`.
    #[serde(default)]
    pub max_output_channels: u16,
}

/// Result of a 2-step Reference Unity + Engaged Hot-Corner calibration run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Backend identifier.
    pub backend: String,
    /// Device identifier.
    pub device_id: String,
    /// Automatically probed and locked output send gain / pad in dB.
    #[serde(default)]
    pub send_gain_db: f32,
    /// Measured round-trip latency in samples (locked from Step-1 Bypass).
    pub latency_samples: i32,
    /// Measured input noise floor in dBFS.
    pub noise_floor_dbfs: f32,
    /// Measured calibration SNR in dB.
    pub snr_db: f32,
    /// Step-1 unity reference level in dBFS.
    #[serde(default)]
    pub unity_ref_dbfs: f32,
    /// Step-2 engaged hot-corner return peak in dBFS.
    #[serde(default)]
    pub hot_corner_peak_dbfs: f32,
    /// Step-2 engaged hot-corner clipped sample fraction (`[0.0, 1.0]`).
    #[serde(default)]
    pub hot_corner_clipped_fraction: f32,
    /// Probed target device excitation profile used to synthesize `models/<model>/stems/input.wav`.
    #[serde(default)]
    pub excitation_profile: TargetExcitationProfile,
}

/// Intermediate metrics measured at the end of Calibration Step 1 (Bypass / Dry-Through Unity).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step1CalibrationMetrics {
    /// Active backend name (`software`, `usb`, `anagram`).
    pub backend: String,
    /// Target device identifier.
    pub device_id: String,
    /// Locked output send gain in dB.
    pub send_gain_db: f32,
    /// Coarse round-trip hardware latency in samples.
    pub latency_samples: i32,
    /// Measured bypass return RMS relative to dry excitation RMS in dBFS.
    pub unity_ref_dbfs: f32,
    /// Measured silent-window noise floor in dBFS.
    pub noise_floor_dbfs: f32,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,
}

/// Pluggable re-amp capture backend trait.
pub trait CaptureBackend {
    /// Backend identifier (`"software"`, `"usb"`, `"anagram"`).
    fn backend_name(&self) -> &'static str;

    /// Returns human-readable hardware routing instructions for this backend.
    fn setup_instructions(&self, sample_rate_hz: u32) -> String;

    /// Returns the configured output send gain in dB.
    fn send_gain_db(&self) -> f32 {
        0.0
    }

    /// Updates the output send gain in dB.
    fn set_send_gain_db(&mut self, _gain_db: f32) {}

    /// Performs 2-Step Reference Unity & Hot-Corner latency/noise/excitation calibration
    /// with interactive prompt hooks before Step 1 and Step 2.
    fn calibrate_with_prompts(
        &self,
        device_id: &str,
        sample_rate_hz: u32,
        step1_prompt: &mut dyn FnMut(&str) -> StudioResult<()>,
        step2_prompt: &mut dyn FnMut(&Step1CalibrationMetrics) -> StudioResult<()>,
    ) -> StudioResult<CalibrationResult>;

    /// Performs 2-Step Reference Unity & Hot-Corner latency/noise/excitation calibration.
    fn calibrate(&self, device_id: &str, sample_rate_hz: u32) -> StudioResult<CalibrationResult> {
        self.calibrate_with_prompts(device_id, sample_rate_hz, &mut |_| Ok(()), &mut |_| Ok(()))
    }

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

fn cpal_device_name(device: &cpal::Device, index: usize) -> String {
    device
        .description()
        .map(|desc| desc.name().to_string())
        .unwrap_or_else(|_| format!("Audio Device {index}"))
}

/// Queries the native OS audio host via `cpal` (`CoreAudio` / `ALSA` / `WASAPI`).
pub fn query_cpal_devices(backend: &str) -> Vec<AudioDeviceDescriptor> {
    let host = cpal::default_host();
    let Ok(devices) = host.devices() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (idx, dev) in devices.enumerate() {
        let name = cpal_device_name(&dev, idx);
        let in_cfg = dev.default_input_config().ok();
        let out_cfg = dev.default_output_config().ok();
        let max_in = in_cfg.as_ref().map(|c| c.channels()).unwrap_or(0);
        let max_out = out_cfg.as_ref().map(|c| c.channels()).unwrap_or(0);
        let sr = in_cfg
            .as_ref()
            .map(cpal::SupportedStreamConfig::sample_rate)
            .or_else(|| {
                out_cfg
                    .as_ref()
                    .map(cpal::SupportedStreamConfig::sample_rate)
            })
            .unwrap_or(48_000);
        let lower = name.to_ascii_lowercase();
        let is_anagram = lower.contains("anagram") || lower.contains("darkglass");
        if backend == "anagram" && !is_anagram {
            continue;
        }
        let resolved_backend = if backend == "all" || backend.is_empty() {
            if is_anagram { "anagram" } else { "usb" }
        } else {
            backend
        };
        out.push(AudioDeviceDescriptor {
            backend: resolved_backend.to_string(),
            device_id: format!("cpal_{idx}"),
            name,
            sample_rate_hz: sr,
            max_input_channels: max_in,
            max_output_channels: max_out,
        });
    }
    out
}

/// Lists available devices for `backend` (`"all"`, `"anagram"`, `"usb"`, or `"software"`).
pub fn list_backend_devices(backend: &str) -> Vec<AudioDeviceDescriptor> {
    match backend {
        "all" | "" => {
            let mut devs = query_cpal_devices("all");
            if devs.is_empty() {
                devs.push(AudioDeviceDescriptor {
                    backend: "usb".to_string(),
                    device_id: "usb_class_compliant_0".to_string(),
                    name: "USB Audio Class 2.0 24-Bit Re-Amp Interface".to_string(),
                    sample_rate_hz: 48_000,
                    max_input_channels: 2,
                    max_output_channels: 2,
                });
            }
            devs
        }
        "anagram" => {
            let mut devs = query_cpal_devices("anagram");
            if devs.is_empty() {
                devs.push(AudioDeviceDescriptor {
                    backend: "anagram".to_string(),
                    device_id: "anagram_rk3582_usb".to_string(),
                    name: "Darkglass Anagram RK3582 Re-Amp Interface (USB Out 9 -> In 1/2)"
                        .to_string(),
                    sample_rate_hz: 48_000,
                    max_input_channels: 3,
                    max_output_channels: 9,
                });
            }
            devs
        }
        "usb" => {
            let mut devs = query_cpal_devices("usb");
            if devs.is_empty() {
                devs.push(AudioDeviceDescriptor {
                    backend: "usb".to_string(),
                    device_id: "usb_class_compliant_0".to_string(),
                    name: "USB Audio Class 2.0 24-Bit Re-Amp Interface".to_string(),
                    sample_rate_hz: 48_000,
                    max_input_channels: 2,
                    max_output_channels: 2,
                });
            }
            devs
        }
        _ => vec![AudioDeviceDescriptor {
            backend: "software".to_string(),
            device_id: "loopback_default".to_string(),
            name: "DNAM Deterministic Software Re-Amp Loopback".to_string(),
            sample_rate_hz: 48_000,
            max_input_channels: 2,
            max_output_channels: 2,
        }],
    }
}

fn is_virtual_fallback_device(device_id: &str) -> bool {
    let trimmed = device_id.trim();
    trimmed.is_empty()
        || trimmed == "loopback_default"
        || trimmed == "usb_class_compliant_0"
        || trimmed == "anagram_rk3582_usb"
}

fn resolve_cpal_device(device_query: Option<&str>, want_input: bool) -> StudioResult<cpal::Device> {
    let host = cpal::default_host();
    if let Some(query) = device_query {
        let q = query.trim();
        if !q.is_empty() && !is_virtual_fallback_device(q) {
            if let Ok(devices) = host.devices() {
                for (idx, dev) in devices.enumerate() {
                    let id = format!("cpal_{idx}");
                    let name = cpal_device_name(&dev, idx);
                    if id.eq_ignore_ascii_case(q)
                        || idx.to_string() == q
                        || name.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
                    {
                        return Ok(dev);
                    }
                }
            }
            return Err(StudioError::Audio(format!(
                "Requested cpal audio device '{q}' not found"
            )));
        }
    }
    let dev = if want_input {
        host.default_input_device()
    } else {
        host.default_output_device()
    };
    dev.ok_or_else(|| {
        StudioError::Audio("No default hardware audio device available via cpal".to_string())
    })
}

/// Executes full-duplex hardware playback and capture via `cpal`.
pub fn cpal_play_and_record(
    device_query: Option<&str>,
    sample_rate_hz: u32,
    send_channel: u16,
    return_channels: &[u16],
    send_gain_db: f32,
    dry_samples: &[f32],
) -> StudioResult<Vec<f32>> {
    if dry_samples.is_empty() {
        return Ok(Vec::new());
    }
    let out_dev = resolve_cpal_device(device_query, false)?;
    let in_dev = resolve_cpal_device(device_query, true)?;

    let out_default = out_dev
        .default_output_config()
        .map_err(|e| StudioError::Audio(format!("Failed to query cpal output config: {e}")))?;
    let in_default = in_dev
        .default_input_config()
        .map_err(|e| StudioError::Audio(format!("Failed to query cpal input config: {e}")))?;

    let out_channels = out_default.channels().max(1);
    let in_channels = in_default.channels().max(1);
    let send_ch_idx = (send_channel.max(1) as usize - 1).min(out_channels as usize - 1);
    let ret_ch_idx = return_channels
        .first()
        .copied()
        .map(|c| (c.max(1) as usize - 1).min(in_channels as usize - 1))
        .unwrap_or(0);

    let out_config = cpal::StreamConfig {
        channels: out_channels,
        sample_rate: sample_rate_hz,
        buffer_size: cpal::BufferSize::Default,
    };
    let in_config = cpal::StreamConfig {
        channels: in_channels,
        sample_rate: sample_rate_hz,
        buffer_size: cpal::BufferSize::Default,
    };

    let gain_lin = 10.0_f32.powf(send_gain_db / 20.0);
    let target_frames = dry_samples.len();
    let scaled_dry: Arc<[f32]> = dry_samples
        .iter()
        .map(|s| *s * gain_lin)
        .collect::<Vec<_>>()
        .into();
    let play_cursor = Arc::new(AtomicUsize::new(0));
    let recorded_slots: Arc<[AtomicU32]> = (0..target_frames)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>()
        .into();
    let rec_cursor = Arc::new(AtomicUsize::new(0));
    let streams_armed = Arc::new(AtomicBool::new(false));
    let out_started = Arc::new(AtomicBool::new(false));

    let play_dry = Arc::clone(&scaled_dry);
    let play_pos = Arc::clone(&play_cursor);
    let out_armed = Arc::clone(&streams_armed);
    let out_began = Arc::clone(&out_started);
    let out_ch_usize = out_channels as usize;
    let out_stream = out_dev
        .build_output_stream(
            out_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if !out_armed.load(Ordering::Acquire) {
                    data.fill(0.0);
                    return;
                }
                out_began.store(true, Ordering::Release);
                let mut pos = play_pos.load(Ordering::Relaxed);
                for frame in data.chunks_mut(out_ch_usize) {
                    frame.fill(0.0);
                    if pos < play_dry.len() {
                        frame[send_ch_idx] = play_dry[pos];
                        pos += 1;
                    }
                }
                play_pos.store(pos, Ordering::Release);
            },
            move |err| {
                let _ = err;
            },
            None,
        )
        .map_err(|e| StudioError::Audio(format!("Failed to build cpal output stream: {e}")))?;

    let rec_buf = Arc::clone(&recorded_slots);
    let rec_pos = Arc::clone(&rec_cursor);
    let in_began = Arc::clone(&out_started);
    let in_ch_usize = in_channels as usize;
    let in_stream = in_dev
        .build_input_stream(
            in_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !in_began.load(Ordering::Acquire) {
                    return;
                }
                let mut pos = rec_pos.load(Ordering::Relaxed);
                if pos < target_frames {
                    for frame in data.chunks(in_ch_usize) {
                        if pos >= target_frames {
                            break;
                        }
                        let sample = frame.get(ret_ch_idx).copied().unwrap_or(0.0);
                        rec_buf[pos].store(sample.to_bits(), Ordering::Relaxed);
                        pos += 1;
                    }
                    rec_pos.store(pos, Ordering::Release);
                }
            },
            move |err| {
                let _ = err;
            },
            None,
        )
        .map_err(|e| StudioError::Audio(format!("Failed to build cpal input stream: {e}")))?;

    in_stream
        .play()
        .map_err(|e| StudioError::Audio(format!("Failed to start cpal input stream: {e}")))?;
    out_stream
        .play()
        .map_err(|e| StudioError::Audio(format!("Failed to start cpal output stream: {e}")))?;
    streams_armed.store(true, Ordering::Release);

    let expected_dur =
        Duration::from_secs_f64((target_frames as f64) / (sample_rate_hz.max(1) as f64));
    let timeout = expected_dur + Duration::from_secs(3);
    let start = Instant::now();
    while start.elapsed() < timeout {
        if rec_cursor.load(Ordering::Acquire) >= target_frames {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    drop(out_stream);
    drop(in_stream);

    let out: Vec<f32> = recorded_slots
        .iter()
        .map(|slot| f32::from_bits(slot.load(Ordering::Relaxed)))
        .collect();
    Ok(out)
}

fn run_two_step_calibration_with_prompts<B: CaptureBackend + ?Sized>(
    backend: &B,
    device_id: &str,
    sample_rate_hz: u32,
    step1_prompt: &mut dyn FnMut(&str) -> StudioResult<()>,
    step2_prompt: &mut dyn FnMut(&Step1CalibrationMetrics) -> StudioResult<()>,
) -> StudioResult<CalibrationResult> {
    let is_hw = backend.backend_name() != "software" && !is_virtual_fallback_device(device_id);
    let probe_len = if is_hw {
        ((sample_rate_hz as usize) * 4).max(2_048)
    } else {
        2_048
    };
    let base_dry = generate_calibration_probe(sample_rate_hz, probe_len);
    step1_prompt(&backend.setup_instructions(sample_rate_hz))?;

    // Step 1: Reference Unity / Bypass probe with automatic gain/pad attenuation if return clips
    let mut unity_ranges = BTreeMap::new();
    unity_ranges.insert("drive".to_string(), 0.0_f32);
    let base_send_gain_db = backend.send_gain_db();
    let mut locked_send_gain_db = base_send_gain_db;
    let mut scaled_dry = base_dry.clone();
    let mut wet_step1 = Vec::new();

    for attempt in 0..5 {
        let pad_db = locked_send_gain_db - base_send_gain_db;
        let pad_lin = 10.0_f32.powf(pad_db / 20.0);
        scaled_dry = base_dry.iter().map(|&s| s * pad_lin).collect();
        wet_step1 = backend.capture_stem(&scaled_dry, &unity_ranges, &BTreeMap::new())?;
        let bypass_clipped = wet_step1.iter().any(|&s| s.abs() >= 0.994);
        if !bypass_clipped || attempt == 4 {
            break;
        }
        locked_send_gain_db -= 6.0;
    }

    let max_lag = (scaled_dry.len() / 4).max(32);
    let report = estimate_subsample_lag_and_snr(&scaled_dry, &wet_step1, max_lag)?;
    let noise_1 = measure_silent_window_noise_floor(&wet_step1, sample_rate_hz, None);

    let dry_view = ArrayView1::from(&scaled_dry);
    let wet1_view = ArrayView1::from(&wet_step1);
    let dry_rms = (dry_view.dot(&dry_view) / (scaled_dry.len().max(1) as f32))
        .sqrt()
        .max(1e-9);
    let wet1_rms = (wet1_view.dot(&wet1_view) / (wet_step1.len().max(1) as f32))
        .sqrt()
        .max(1e-9);
    let unity_ref_dbfs = 20.0 * (wet1_rms / dry_rms).log10();

    let step1_metrics = Step1CalibrationMetrics {
        backend: backend.backend_name().to_string(),
        device_id: device_id.to_string(),
        send_gain_db: locked_send_gain_db,
        latency_samples: report.coarse_lag_samples,
        unity_ref_dbfs,
        noise_floor_dbfs: noise_1,
        snr_db: report.snr_db,
    };
    step2_prompt(&step1_metrics)?;

    // Step 2: Engaged Hot-Corner probe (max drive) at locked send_gain_db
    let mut hot_ranges = BTreeMap::new();
    hot_ranges.insert("drive".to_string(), 1.0_f32);
    let wet_step2 = backend.capture_stem(&scaled_dry, &hot_ranges, &BTreeMap::new())?;
    let aligned_wet2 = apply_subsample_shift_fft(&wet_step2, report.subsample_lag_samples);
    let hot_analysis = analyze_hot_corner_nonlinearity(&scaled_dry, &aligned_wet2, unity_ref_dbfs);
    let profile = analyze_probe_for_excitation(&scaled_dry, &aligned_wet2, sample_rate_hz);

    Ok(CalibrationResult {
        backend: backend.backend_name().to_string(),
        device_id: device_id.to_string(),
        send_gain_db: locked_send_gain_db,
        latency_samples: report.coarse_lag_samples,
        noise_floor_dbfs: noise_1,
        snr_db: report.snr_db,
        unity_ref_dbfs,
        hot_corner_peak_dbfs: hot_analysis.measured_return_peak_dbfs,
        hot_corner_clipped_fraction: hot_analysis.clipped_sample_fraction,
        excitation_profile: profile,
    })
}

/// Deterministic software nonlinear loopback backend for headless CLI and CI verification.
#[derive(Debug, Clone, Copy)]
pub struct SoftwareLoopbackBackend {
    /// Simulated round-trip latency in samples.
    pub simulated_latency_samples: usize,
    /// Output send gain in dB (`0.0` default).
    pub send_gain_db: f32,
}

impl Default for SoftwareLoopbackBackend {
    fn default() -> Self {
        Self {
            simulated_latency_samples: 6,
            send_gain_db: 0.0,
        }
    }
}

impl CaptureBackend for SoftwareLoopbackBackend {
    fn backend_name(&self) -> &'static str {
        "software"
    }

    fn setup_instructions(&self, sample_rate_hz: u32) -> String {
        format!(
            "[Software DSP / DnamEngine Loopback]\n  In-memory deterministic loopback @ {sample_rate_hz} Hz (Latency = {} samples, Unity Send = {:+.1} dB).",
            self.simulated_latency_samples, self.send_gain_db
        )
    }

    fn send_gain_db(&self) -> f32 {
        self.send_gain_db
    }

    fn set_send_gain_db(&mut self, gain_db: f32) {
        self.send_gain_db = gain_db;
    }

    fn calibrate_with_prompts(
        &self,
        device_id: &str,
        sample_rate_hz: u32,
        step1_prompt: &mut dyn FnMut(&str) -> StudioResult<()>,
        step2_prompt: &mut dyn FnMut(&Step1CalibrationMetrics) -> StudioResult<()>,
    ) -> StudioResult<CalibrationResult> {
        run_two_step_calibration_with_prompts(
            self,
            device_id,
            sample_rate_hz,
            step1_prompt,
            step2_prompt,
        )
    }

    fn capture_stem(
        &self,
        dry_samples: &[f32],
        normalized_ranges: &BTreeMap<String, f32>,
        switches: &BTreeMap<String, usize>,
    ) -> StudioResult<Vec<f32>> {
        let send_lin = 10.0_f32.powf(self.send_gain_db / 20.0);
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
        let drive = (1.2 + 2.4 * avg_range) * sw_boost * send_lin;
        let mut wet = vec![0.0_f32; dry_samples.len()];
        let lag = self.simulated_latency_samples;
        for i in lag..dry_samples.len() {
            let x = dry_samples[i - lag] * drive;
            wet[i] = x / (1.0 + x.abs());
        }
        Ok(wet)
    }
}

/// Multi-channel USB/Thunderbolt hardware audio interface capture backend (`cpal`).
#[derive(Debug, Clone)]
pub struct UsbAudioInterfaceBackend {
    /// Optional `cpal` device ID or substring filter.
    pub device_id: Option<String>,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// 1-indexed hardware output channel for the dry reamp send.
    pub send_channel: u16,
    /// 1-indexed hardware input channel(s) for the wet return.
    pub return_channels: Vec<u16>,
    /// Output send gain in dB.
    pub send_gain_db: f32,
}

impl Default for UsbAudioInterfaceBackend {
    fn default() -> Self {
        Self {
            device_id: None,
            sample_rate_hz: 48_000,
            send_channel: 1,
            return_channels: vec![1, 2],
            send_gain_db: 0.0,
        }
    }
}

impl CaptureBackend for UsbAudioInterfaceBackend {
    fn backend_name(&self) -> &'static str {
        "usb"
    }

    fn setup_instructions(&self, sample_rate_hz: u32) -> String {
        let ret_str = self
            .return_channels
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "[USB / Thunderbolt Audio Interface]\n  1. Connect Audio Interface Output {} -> Reamp / Device In.\n  2. Connect Device Output(s) -> Audio Interface Input(s) [{ret_str}].\n  3. Lock Sample Rate to {sample_rate_hz} Hz (Send Gain = {:+.1} dB).",
            self.send_channel, self.send_gain_db
        )
    }

    fn send_gain_db(&self) -> f32 {
        self.send_gain_db
    }

    fn set_send_gain_db(&mut self, gain_db: f32) {
        self.send_gain_db = gain_db;
    }

    fn calibrate_with_prompts(
        &self,
        device_id: &str,
        sample_rate_hz: u32,
        step1_prompt: &mut dyn FnMut(&str) -> StudioResult<()>,
        step2_prompt: &mut dyn FnMut(&Step1CalibrationMetrics) -> StudioResult<()>,
    ) -> StudioResult<CalibrationResult> {
        let mut clone = self.clone();
        clone.device_id = Some(device_id.to_string());
        clone.sample_rate_hz = sample_rate_hz;
        run_two_step_calibration_with_prompts(
            &clone,
            device_id,
            sample_rate_hz,
            step1_prompt,
            step2_prompt,
        )
    }

    fn capture_stem(
        &self,
        dry_samples: &[f32],
        normalized_ranges: &BTreeMap<String, f32>,
        switches: &BTreeMap<String, usize>,
    ) -> StudioResult<Vec<f32>> {
        if let Some(dev) = self.device_id.as_deref()
            && is_virtual_fallback_device(dev)
        {
            return SoftwareLoopbackBackend {
                send_gain_db: self.send_gain_db,
                ..SoftwareLoopbackBackend::default()
            }
            .capture_stem(dry_samples, normalized_ranges, switches);
        }
        cpal_play_and_record(
            self.device_id.as_deref(),
            self.sample_rate_hz,
            self.send_channel,
            &self.return_channels,
            self.send_gain_db,
            dry_samples,
        )
    }
}

/// Pre-configured Darkglass Anagram 48 kHz 24-bit USB-C Digital Reamp Capture Backend (`cpal`).
///
/// Routes USB Output 9 (`send_channel = 9`) -> Input 1 (`[1]`) or True-Stereo Inputs 1 & 2
/// (`[1, 2]`) at Unity `send_gain_db = 0.0 dB`.
#[derive(Debug, Clone)]
pub struct DarkglassAnagramBackend {
    /// Optional `cpal` device ID or substring filter (defaults to auto-discovering `"Anagram"`).
    pub device_id: Option<String>,
    /// Sample rate in Hz (48 kHz on Darkglass Anagram).
    pub sample_rate_hz: u32,
    /// 1-indexed USB output channel (`9` = USB Out 9 Dry DI Reamp Send).
    pub send_channel: u16,
    /// 1-indexed USB input channel(s) (`[1, 2]` = USB In 1/2 Processed Stereo Return).
    pub return_channels: Vec<u16>,
    /// Output send gain in dB (`0.0` dB Unity).
    pub send_gain_db: f32,
    /// Optional Anagram preset label.
    pub anagram_preset: Option<String>,
}

impl Default for DarkglassAnagramBackend {
    fn default() -> Self {
        Self {
            device_id: None,
            sample_rate_hz: 48_000,
            send_channel: 9,
            return_channels: vec![1, 2],
            send_gain_db: 0.0,
            anagram_preset: None,
        }
    }
}

impl CaptureBackend for DarkglassAnagramBackend {
    fn backend_name(&self) -> &'static str {
        "anagram"
    }

    fn setup_instructions(&self, _sample_rate_hz: u32) -> String {
        let ret_str = if self.return_channels.len() >= 2 {
            "Inputs 1 & 2 (True Stereo)"
        } else {
            "Input 1 (Mono)"
        };
        let preset_note = self
            .anagram_preset
            .as_deref()
            .map(|p| format!(" | Preset: {p}"))
            .unwrap_or_default();
        format!(
            "[Darkglass Anagram (USB-C Digital Reamp){preset_note}]\n  1. Connect Darkglass Anagram via USB-C (48 kHz / 24-bit).\n  2. Route USB Output {} (Reamp Send) -> Anagram Block Chain Input.\n  3. Capture Return on USB {ret_str} at Unity Send Gain ({:+.1} dB).\n  4. Step-1 measures Bypass Unity (0.0 dBFS); Step-2 preserves engaged gain.",
            self.send_channel, self.send_gain_db
        )
    }

    fn send_gain_db(&self) -> f32 {
        self.send_gain_db
    }

    fn set_send_gain_db(&mut self, gain_db: f32) {
        self.send_gain_db = gain_db;
    }

    fn calibrate_with_prompts(
        &self,
        device_id: &str,
        sample_rate_hz: u32,
        step1_prompt: &mut dyn FnMut(&str) -> StudioResult<()>,
        step2_prompt: &mut dyn FnMut(&Step1CalibrationMetrics) -> StudioResult<()>,
    ) -> StudioResult<CalibrationResult> {
        let mut clone = self.clone();
        clone.device_id = Some(device_id.to_string());
        clone.sample_rate_hz = sample_rate_hz;
        run_two_step_calibration_with_prompts(
            &clone,
            device_id,
            sample_rate_hz,
            step1_prompt,
            step2_prompt,
        )
    }

    fn capture_stem(
        &self,
        dry_samples: &[f32],
        normalized_ranges: &BTreeMap<String, f32>,
        switches: &BTreeMap<String, usize>,
    ) -> StudioResult<Vec<f32>> {
        let query = self.device_id.as_deref().unwrap_or("anagram");
        if is_virtual_fallback_device(query) {
            return SoftwareLoopbackBackend {
                send_gain_db: self.send_gain_db,
                ..SoftwareLoopbackBackend::default()
            }
            .capture_stem(dry_samples, normalized_ranges, switches);
        }
        cpal_play_and_record(
            Some(query),
            self.sample_rate_hz,
            self.send_channel,
            &self.return_channels,
            self.send_gain_db,
            dry_samples,
        )
    }
}

/// Instantiates the registered [`CaptureBackend`] implementation for `backend` with a custom `send_gain_db`.
pub fn create_capture_backend_with_gain(
    backend: &str,
    device_id: Option<&str>,
    sample_rate_hz: u32,
    send_gain_db: f32,
) -> Box<dyn CaptureBackend> {
    match backend {
        "anagram" => Box::new(DarkglassAnagramBackend {
            device_id: device_id.map(str::to_string),
            sample_rate_hz,
            send_gain_db,
            ..DarkglassAnagramBackend::default()
        }),
        "usb" => Box::new(UsbAudioInterfaceBackend {
            device_id: device_id.map(str::to_string),
            sample_rate_hz,
            send_gain_db,
            ..UsbAudioInterfaceBackend::default()
        }),
        _ => Box::new(SoftwareLoopbackBackend {
            send_gain_db,
            ..SoftwareLoopbackBackend::default()
        }),
    }
}

/// Instantiates the registered [`CaptureBackend`] implementation for `backend`.
pub fn create_capture_backend(
    backend: &str,
    device_id: Option<&str>,
    sample_rate_hz: u32,
) -> Box<dyn CaptureBackend> {
    create_capture_backend_with_gain(backend, device_id, sample_rate_hz, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_and_backends_virtual_loopback() {
        let rb = AtomicF32RingBuffer::with_capacity(8);
        assert_eq!(rb.capacity(), 8);
        assert_eq!(rb.available_write(), 8);
        assert_eq!(rb.push_slice(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(rb.available_read(), 3);
        assert_eq!(rb.available_write(), 5);

        let all_devs = list_backend_devices("all");
        assert!(!all_devs.is_empty());
        let sw_devs = list_backend_devices("software");
        assert_eq!(sw_devs[0].device_id, "loopback_default");

        let mut usb = UsbAudioInterfaceBackend {
            device_id: Some("usb_class_compliant_0".to_string()),
            ..UsbAudioInterfaceBackend::default()
        };
        assert_eq!(usb.backend_name(), "usb");
        usb.set_send_gain_db(-3.0);
        assert_eq!(usb.send_gain_db(), -3.0);
        let cal_usb = usb
            .calibrate("usb_class_compliant_0", 48_000)
            .expect("calibrate usb virtual");
        assert_eq!(cal_usb.backend, "usb");

        let mut anagram = DarkglassAnagramBackend {
            device_id: Some("anagram_rk3582_usb".to_string()),
            return_channels: vec![1],
            anagram_preset: Some("B7K Lead".to_string()),
            ..DarkglassAnagramBackend::default()
        };
        assert_eq!(anagram.backend_name(), "anagram");
        assert!(
            anagram
                .setup_instructions(48_000)
                .contains("Input 1 (Mono)")
        );
        assert!(
            anagram
                .setup_instructions(48_000)
                .contains("Preset: B7K Lead")
        );
        anagram.set_send_gain_db(0.0);
        assert_eq!(anagram.send_gain_db(), 0.0);
        let cal_anagram = anagram
            .calibrate("anagram_rk3582_usb", 48_000)
            .expect("calibrate anagram virtual");
        assert_eq!(cal_anagram.backend, "anagram");

        let mut sw = SoftwareLoopbackBackend::default();
        sw.set_send_gain_db(-1.5);
        assert_eq!(sw.send_gain_db(), -1.5);
        assert!(sw.setup_instructions(48_000).contains("-1.5 dB"));

        let mut dyn_usb = create_capture_backend("usb", Some("usb_class_compliant_0"), 48_000);
        assert_eq!(dyn_usb.backend_name(), "usb");
        dyn_usb.set_send_gain_db(0.0);

        // Empty dry slice returns empty Vec immediately in cpal_play_and_record
        let empty_rec = cpal_play_and_record(None, 48_000, 1, &[1], 0.0, &[]).expect("empty cpal");
        assert!(empty_rec.is_empty());

        assert!(!list_backend_devices("usb").is_empty());
        assert!(!list_backend_devices("anagram").is_empty());
        assert!(resolve_cpal_device(Some("nonexistent_device_xyz_999"), false).is_err());
        let _ = resolve_cpal_device(Some("0"), false);
        let _ = cpal_play_and_record(None, 48_000, 1, &[1], 0.0, &[0.05; 256]);

        let usb_hw = UsbAudioInterfaceBackend {
            device_id: None,
            ..UsbAudioInterfaceBackend::default()
        };
        let _ = usb_hw.capture_stem(&[0.0; 64], &BTreeMap::new(), &BTreeMap::new());

        let anagram_hw = DarkglassAnagramBackend {
            device_id: None,
            ..DarkglassAnagramBackend::default()
        };
        let _ = anagram_hw.capture_stem(&[0.0; 64], &BTreeMap::new(), &BTreeMap::new());
    }
}
