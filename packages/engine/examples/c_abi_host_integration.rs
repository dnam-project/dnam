//! Example 6: C-ABI Host Plugin & Embedded Firmware FFI Lifecycle (`examples/c_abi_host_integration.rs`).
//!
//! Demonstrates driving [`dnam_engine`] exclusively through its zero-allocation `extern "C"`
//! FFI surface (`dnam_engine_*`), as used by C/C++ VST3/AU/CLAP wrappers, iOS/Android native
//! audio threads, and WebAssembly `AudioWorkletProcessor` hosts.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example c_abi_host_integration`

use dnam_engine::DnamTelemetrySnapshot;
use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
use dnam_engine::ffi::{
    dnam_engine_create_with_max_tier, dnam_engine_destroy, dnam_engine_get_latency_samples,
    dnam_engine_get_telemetry, dnam_engine_process_block, dnam_engine_process_mono,
    dnam_engine_reset, dnam_engine_set_control_value, dnam_engine_set_offline_bounce,
    dnam_engine_set_sample_rate, dnam_engine_set_tier,
};
use dnam_engine::schema::DnamJsonHeader;

fn main() {
    let mut header = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    for (i, b) in arena[128..36_000].iter_mut().enumerate() {
        *b = ((i * 5 + 1) & 0x3F) as u8;
    }
    let container_bytes = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

    // SAFETY: All pointers passed to the C-ABI functions below are valid, non-null stack/vec
    // slices matching `num_frames`, and `handle` is destroyed exactly once at the end of scope.
    unsafe {
        // 1. Create an engine instance capped at Pro tier (`max_supported_tier = 2`) at 48 kHz
        let handle = dnam_engine_create_with_max_tier(
            container_bytes.as_ptr(),
            container_bytes.len(),
            1, // Initial tier = Full (1)
            2, // Max supported tier = Pro (2)
            48_000,
        );
        assert!(!handle.is_null(), "C-ABI engine creation succeeded");

        // 2. Automate control 0 and process a 128-frame stereo callback + 128-frame mono callback
        dnam_engine_set_control_value(handle, 0, 0.75);

        let left_in = [0.20_f32; 128];
        let right_in = [-0.20_f32; 128];
        let mut left_out = [0.0_f32; 128];
        let mut right_out = [0.0_f32; 128];
        dnam_engine_process_block(
            handle,
            left_in.as_ptr(),
            right_in.as_ptr(),
            left_out.as_mut_ptr(),
            right_out.as_mut_ptr(),
            128,
        );

        let mut mono_out = [0.0_f32; 128];
        dnam_engine_process_mono(handle, left_in.as_ptr(), mono_out.as_mut_ptr(), 128);

        // 3. Switch host sample rate to 96 kHz and enable 4x Offline Bounce (12-sample PDC latency)
        assert_eq!(dnam_engine_set_sample_rate(handle, 96_000), 0);
        dnam_engine_set_tier(handle, 3); // Request Ultra (3); capped to Pro (2) by max_supported_tier
        dnam_engine_set_offline_bounce(handle, true);
        let pdc_samples = dnam_engine_get_latency_samples(handle);

        // 4. Read lock-free 60 fps telemetry struct over C-ABI
        let mut telemetry = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::zeroed().assume_init();
        dnam_engine_get_telemetry(handle, &mut telemetry);
        println!(
            "C-ABI Host Telemetry | Effective Tier: {} (capped at Pro=2) | PDC Latency: {} samples | Out L[127]: {:.6}",
            telemetry.active_tier, pdc_samples, left_out[127],
        );

        // 5. Reset to DC equilibrium and destroy handle
        dnam_engine_reset(handle);
        dnam_engine_destroy(handle);
    }
}
