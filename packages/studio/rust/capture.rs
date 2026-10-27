//! Subsystem 2: 24-Bit WAV I/O, Excitation Signal Synthesis, Sub-Sample Alignment & Re-Amp Backends.

pub mod alignment;
pub mod backends;
pub mod excitation;
pub mod wav;

pub use alignment::{
    AlignmentReport, HotCornerAnalysis, PostAlignmentVerification, align_hardware_latency,
    analyze_hot_corner_nonlinearity, apply_subsample_shift_fft, estimate_subsample_lag_and_snr,
    measure_silent_window_noise_floor, verify_and_polish_post_alignment,
};
pub use backends::{
    AtomicF32RingBuffer, AudioDeviceDescriptor, CalibrationResult, CaptureBackend,
    DarkglassAnagramBackend, SoftwareLoopbackBackend, Step1CalibrationMetrics,
    UsbAudioInterfaceBackend, cpal_play_and_record, create_capture_backend,
    create_capture_backend_with_gain, list_backend_devices, list_capture_backends,
    query_cpal_devices,
};
pub use excitation::{
    ALIGNMENT_HEADER_TOTAL_SEC, ALIGNMENT_PULSE_DURATION_SEC, ALIGNMENT_SETTLE_SILENCE_SEC,
    DEFAULT_HARDWARE_EXCITATION_SEC, DEFAULT_HARDWARE_PROBE_SEC, TRAILING_NOISE_WINDOW_SEC,
    TargetExcitationProfile, analyze_probe_for_excitation, generate_calibration_probe,
    generate_excitation_signal, generate_excitation_signal_with_profile, synth_alignment_pulses,
};
pub use wav::{WavData, compute_pcm_sha256, read_wav, write_wav_24bit, write_wav_32bit_float};
