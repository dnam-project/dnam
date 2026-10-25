//! Subsystem 2: 24-Bit WAV I/O, Excitation Signal Synthesis, Sub-Sample Alignment & Re-Amp Backends.

pub mod alignment;
pub mod backends;
pub mod excitation;
pub mod wav;

pub use alignment::{AlignmentReport, estimate_subsample_lag_and_snr};
pub use backends::{
    AudioDeviceDescriptor, CalibrationResult, CaptureBackend, SoftwareLoopbackBackend,
    list_backend_devices, list_capture_backends,
};
pub use excitation::{generate_calibration_probe, generate_excitation_signal};
pub use wav::{WavData, compute_pcm_sha256, read_wav, write_wav_24bit};
