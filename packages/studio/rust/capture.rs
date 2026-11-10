//! Subsystem 2: 12/16/24/32-Bit PCM & 32-Bit Float WAV I/O and Hardware Audio Device Enumeration.

pub mod backends;
pub mod wav;

pub use backends::{
    AtomicF32RingBuffer, AudioDeviceDescriptor, list_backend_devices, list_capture_backends,
    query_cpal_devices,
};
pub use wav::{
    WavData, compute_pcm_sha256, read_wav, write_wav_12bit, write_wav_16bit, write_wav_24bit,
    write_wav_32bit_float, write_wav_32bit_int,
};
