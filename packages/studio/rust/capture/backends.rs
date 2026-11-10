//! Lock-free SPSC ring buffer and `cpal` hardware/software audio device enumeration.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};

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

fn default_usb_device() -> AudioDeviceDescriptor {
    AudioDeviceDescriptor {
        backend: "usb".to_string(),
        device_id: "usb_class_compliant_0".to_string(),
        name: "USB Audio Class 2.0 24-Bit Re-Amp Interface".to_string(),
        sample_rate_hz: 48_000,
        max_input_channels: 2,
        max_output_channels: 2,
    }
}

/// Lists available devices for `backend` (`"all"`, `"anagram"`, `"usb"`, or `"software"`).
pub fn list_backend_devices(backend: &str) -> Vec<AudioDeviceDescriptor> {
    match backend {
        "all" | "" | "usb" => {
            let query_key = if backend == "usb" { "usb" } else { "all" };
            let mut devs = query_cpal_devices(query_key);
            if devs.is_empty() {
                devs.push(default_usb_device());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_and_device_lists() {
        let rb = AtomicF32RingBuffer::with_capacity(3);
        assert_eq!(rb.capacity(), 4);
        assert_eq!(rb.available_read(), 0);
        assert_eq!(rb.available_write(), 4);

        let written = rb.push_slice(&[0.25, -0.5, 0.75, 1.0, 2.0]);
        assert_eq!(written, 4);
        assert_eq!(rb.available_read(), 4);
        assert_eq!(rb.available_write(), 0);

        let mut dst = [0.0_f32; 4];
        assert_eq!(rb.pop_slice(&mut dst), 4);
        assert_eq!(dst, [0.25, -0.5, 0.75, 1.0]);
        assert_eq!(rb.available_read(), 0);
        assert_eq!(rb.available_write(), 4);

        assert_eq!(list_capture_backends(), vec!["software", "usb", "anagram"]);
        assert!(!list_backend_devices("all").is_empty());
        assert!(!list_backend_devices("anagram").is_empty());
        assert!(!list_backend_devices("usb").is_empty());
        assert_eq!(default_usb_device().backend, "usb");
        assert_eq!(list_backend_devices("software")[0].backend, "software");
    }
}
