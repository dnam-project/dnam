//! C-ABI FFI Integration Suite (`tests/ffi_integration.rs`).
//!
//! Exercises the `extern "C"` C-ABI surface exported by `dnam_engine::ffi`.

use dnam_engine::DnamTelemetrySnapshot;
use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
use dnam_engine::ffi::{
    dnam_engine_create_from_bytes, dnam_engine_destroy, dnam_engine_get_telemetry,
    dnam_engine_process_block, dnam_engine_process_mono, dnam_engine_reset,
    dnam_engine_set_control_immediate, dnam_engine_set_control_value, dnam_engine_set_sample_rate,
    dnam_engine_set_tier,
};
use dnam_engine::schema::DnamJsonHeader;

#[test]
fn test_c_abi_ffi_lifecycle_and_null_pointer_safety() {
    let mut header = DnamJsonHeader::default_reference(false);
    let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

    // SAFETY: Exercises all FFI functions with explicit null pointers and valid stack-allocated
    // container/audio buffers matching `num_frames`.
    unsafe {
        assert!(dnam_engine_create_from_bytes(core::ptr::null(), 100, 0, 48_000).is_null());
        assert!(dnam_engine_create_from_bytes(container.as_ptr(), 32, 0, 48_000).is_null());
        assert!(dnam_engine_create_from_bytes([0u8; 128].as_ptr(), 128, 0, 48_000).is_null());

        dnam_engine_destroy(core::ptr::null_mut());
        dnam_engine_set_tier(core::ptr::null_mut(), 1);
        assert_eq!(
            dnam_engine_set_sample_rate(core::ptr::null_mut(), 48_000),
            -1
        );
        dnam_engine_set_control_value(core::ptr::null_mut(), 0, 0.5);
        dnam_engine_set_control_immediate(core::ptr::null_mut(), 0, 0.5);
        dnam_engine_reset(core::ptr::null_mut());
        dnam_engine_process_block(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            64,
        );
        dnam_engine_process_mono(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null_mut(),
            64,
        );
        dnam_engine_get_telemetry(core::ptr::null(), core::ptr::null_mut());

        let handle = dnam_engine_create_from_bytes(container.as_ptr(), container.len(), 2, 48_000);
        assert!(!handle.is_null());

        dnam_engine_set_tier(handle, 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 48_000), 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 8_000), -1);
        dnam_engine_set_control_value(handle, 0, 6.0);
        dnam_engine_set_control_immediate(handle, 0, 6.0);
        dnam_engine_reset(handle);

        let l_in = [0.25_f32; 32];
        let r_in = [-0.25_f32; 32];
        let mut l_out = [0.0_f32; 32];
        let mut r_out = [0.0_f32; 32];
        dnam_engine_process_block(
            handle,
            l_in.as_ptr(),
            r_in.as_ptr(),
            l_out.as_mut_ptr(),
            r_out.as_mut_ptr(),
            32,
        );
        dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 32);

        let mut snap = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::uninit();
        dnam_engine_get_telemetry(handle, snap.as_mut_ptr());
        let snap = snap.assume_init();
        assert_eq!(snap.active_tier, 0);

        dnam_engine_destroy(handle);
    }
}
