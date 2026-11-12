//! Native OS Audio Device Enumeration (`cpal`) and Lock-Free SPSC Ring Buffer.

pub mod backends;

pub use backends::{
    AtomicF32RingBuffer, AudioDeviceDescriptor, list_backend_devices, list_capture_backends,
    query_cpal_devices,
};
