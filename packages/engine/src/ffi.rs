//! `extern "C"` C-ABI & WASM Linear-Memory FFI Exports (`ffi.rs`).
//!
//! Exposes a zero-allocation C-ABI surface over [`DynamicDnamEngine`](crate::DynamicDnamEngine)
//! and [`DnamTelemetrySnapshot`](crate::DnamTelemetrySnapshot) for C/C++ audio hosts,
//! `packages/js` WASM `AudioWorkletProcessor`, and embedded firmware.

use alloc::boxed::Box;
use core::slice;

use crate::{DnamTelemetrySnapshot, DynamicDnamEngine, Tier};

/// Opaque C-ABI handle owning a `'static` [`DynamicDnamEngine`].
#[derive(Debug)]
pub struct DnamEngineHandle {
    engine: DynamicDnamEngine<'static>,
}

/// Creates a new [`DnamEngineHandle`] from a `.dnam` byte buffer, copying the
/// 64-byte aligned tensor arena into owned memory so the caller can free `data_ptr` immediately.
///
/// Returns `core::ptr::null_mut()` if `data_ptr` is null or container validation fails.
///
/// # Safety
/// `data_ptr` must point to a valid readable byte buffer of at least `data_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_create_from_bytes(
    data_ptr: *const u8,
    data_len: usize,
    tier: u8,
    sample_rate_hz: u32,
) -> *mut DnamEngineHandle {
    if data_ptr.is_null() || data_len < 64 {
        return core::ptr::null_mut();
    }
    // SAFETY: Caller guarantees `data_ptr` is valid for `data_len` bytes.
    let bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let tier_enum = Tier::from_u8(tier);
    match DynamicDnamEngine::from_bytes_owned(bytes, tier_enum, sample_rate_hz) {
        Ok(engine) => Box::into_raw(Box::new(DnamEngineHandle { engine })),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Creates a new [`DnamEngineHandle`] capped at `max_supported_tier` (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
///
/// # Safety
/// `data_ptr` must point to a valid readable byte buffer of at least `data_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_create_with_max_tier(
    data_ptr: *const u8,
    data_len: usize,
    tier: u8,
    max_supported_tier: u8,
    sample_rate_hz: u32,
) -> *mut DnamEngineHandle {
    if data_ptr.is_null() || data_len < 64 {
        return core::ptr::null_mut();
    }
    // SAFETY: Caller guarantees `data_ptr` is valid for `data_len` bytes.
    let bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let tier_enum = Tier::from_u8(tier);
    let max_tier_enum = Tier::from_u8(max_supported_tier);
    match DynamicDnamEngine::from_bytes_owned_with_max_tier(
        bytes,
        tier_enum,
        max_tier_enum,
        sample_rate_hz,
    ) {
        Ok(engine) => Box::into_raw(Box::new(DnamEngineHandle { engine })),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Destroys a [`DnamEngineHandle`] previously created by [`dnam_engine_create_from_bytes`].
///
/// # Safety
/// `handle` must be either null or a valid pointer returned by [`dnam_engine_create_from_bytes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_destroy(handle: *mut DnamEngineHandle) {
    if !handle.is_null() {
        // SAFETY: `handle` was allocated via `Box::into_raw` in `dnam_engine_create_from_bytes`.
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Switches the active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`) with `0.00 dB` LUFS continuity.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_tier(handle: *mut DnamEngineHandle, tier: u8) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_tier(Tier::from_u8(tier));
    }
}

/// Sets the plugin/host ceiling tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_max_supported_tier(
    handle: *mut DnamEngineHandle,
    max_tier: u8,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_max_supported_tier(Tier::from_u8(max_tier));
    }
}

/// Enables or disables Offline Bounce mode ($4\times$ linear-phase polyphase oversampling, `12`-sample PDC).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_offline_bounce(
    handle: *mut DnamEngineHandle,
    enabled: bool,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_offline_bounce(enabled);
    }
}

/// Enables or disables automatic tier promotion (`min(max_available_tier, max_supported_tier)`) during Offline Bounce.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_maximize_tier_on_bounce(
    handle: *mut DnamEngineHandle,
    enabled: bool,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_maximize_tier_on_bounce(enabled);
    }
}

/// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_get_latency_samples(handle: *const DnamEngineHandle) -> usize {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_ref() } {
        h.engine.latency_samples()
    } else {
        0
    }
}

/// Rescales discrete-time parameters for a new host sample rate $f_{s,\text{host}} \in [44100, 96000]\text{ Hz}$.
/// Returns `0` on success or `-1` on invalid sample rate.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_sample_rate(
    handle: *mut DnamEngineHandle,
    sample_rate_hz: u32,
) -> i32 {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() }
        && h.engine.set_sample_rate(sample_rate_hz).is_ok()
    {
        return 0;
    }
    -1
}

/// Updates a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_control_value(
    handle: *mut DnamEngineHandle,
    ctrl_idx: usize,
    value: f32,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_control_value(ctrl_idx, value);
    }
}

/// Updates a user control value and snaps immediately without anti-zipper smoothing.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_control_immediate(
    handle: *mut DnamEngineHandle,
    ctrl_idx: usize,
    value: f32,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_control_immediate(ctrl_idx, value);
    }
}

/// Resets all TORA stage and operator states to their exact analytical DC resting equilibrium.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_reset(handle: *mut DnamEngineHandle) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.reset_to_dc_equilibrium();
    }
}

/// Processes a stereo audio block of `num_frames` samples with zero heap allocations.
///
/// # Safety
/// `handle` must be a valid pointer to [`DnamEngineHandle`], and `left_in`, `right_in`,
/// `left_out`, `right_out` must each point to valid `f32` buffers of at least `num_frames` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_process_block(
    handle: *mut DnamEngineHandle,
    left_in: *const f32,
    right_in: *const f32,
    left_out: *mut f32,
    right_out: *mut f32,
    num_frames: usize,
) {
    if handle.is_null()
        || left_in.is_null()
        || right_in.is_null()
        || left_out.is_null()
        || right_out.is_null()
        || num_frames == 0
    {
        return;
    }
    // SAFETY: Caller guarantees non-null pointers valid for `num_frames` elements.
    unsafe {
        let h = &mut *handle;
        let l_in = slice::from_raw_parts(left_in, num_frames);
        let r_in = slice::from_raw_parts(right_in, num_frames);
        let l_out = slice::from_raw_parts_mut(left_out, num_frames);
        let r_out = slice::from_raw_parts_mut(right_out, num_frames);
        h.engine.process_block(l_in, r_in, l_out, r_out);
    }
}

/// Processes a mono audio block (`Mono -> Mono`, saving 50% of output head MACs) with zero heap allocations.
///
/// # Safety
/// `handle` must be a valid pointer to [`DnamEngineHandle`], and `mono_in`, `mono_out`
/// must each point to valid `f32` buffers of at least `num_frames` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_process_mono(
    handle: *mut DnamEngineHandle,
    mono_in: *const f32,
    mono_out: *mut f32,
    num_frames: usize,
) {
    if handle.is_null() || mono_in.is_null() || mono_out.is_null() || num_frames == 0 {
        return;
    }
    // SAFETY: Caller guarantees non-null pointers valid for `num_frames` elements.
    unsafe {
        let h = &mut *handle;
        let m_in = slice::from_raw_parts(mono_in, num_frames);
        let m_out = slice::from_raw_parts_mut(mono_out, num_frames);
        h.engine.process_mono(m_in, m_out);
    }
}

/// Copies the latest 60 fps [`DnamTelemetrySnapshot`] into `out_snapshot`.
///
/// # Safety
/// `handle` and `out_snapshot` must be valid non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_get_telemetry(
    handle: *const DnamEngineHandle,
    out_snapshot: *mut DnamTelemetrySnapshot,
) {
    // SAFETY: Caller guarantees `handle` and `out_snapshot` are either null or valid pointers.
    if let (Some(h), Some(out)) = unsafe { (handle.as_ref(), out_snapshot.as_mut()) } {
        *out = h.engine.telemetry_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_all_ffi_functions_and_null_safety() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        // SAFETY: Unit test exercises FFI C-ABI interface with both null and valid test pointers.
        unsafe {
            // Null pointer safety
            assert!(dnam_engine_create_from_bytes(core::ptr::null(), 0, 0, 48000).is_null());
            assert!(dnam_engine_create_with_max_tier(core::ptr::null(), 0, 0, 0, 48000).is_null());
            dnam_engine_destroy(core::ptr::null_mut());
            dnam_engine_set_tier(core::ptr::null_mut(), 0);
            dnam_engine_set_max_supported_tier(core::ptr::null_mut(), 0);
            dnam_engine_set_offline_bounce(core::ptr::null_mut(), true);
            dnam_engine_set_maximize_tier_on_bounce(core::ptr::null_mut(), true);
            assert_eq!(dnam_engine_get_latency_samples(core::ptr::null()), 0);
            assert_eq!(
                dnam_engine_set_sample_rate(core::ptr::null_mut(), 48000),
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
                0,
            );
            dnam_engine_process_mono(
                core::ptr::null_mut(),
                core::ptr::null(),
                core::ptr::null_mut(),
                0,
            );
            dnam_engine_get_telemetry(core::ptr::null(), core::ptr::null_mut());

            // Valid handle with create_with_max_tier
            let handle =
                dnam_engine_create_with_max_tier(container.as_ptr(), container.len(), 1, 2, 48000);
            assert!(!handle.is_null());

            dnam_engine_set_max_supported_tier(handle, 1);
            dnam_engine_set_offline_bounce(handle, true);
            assert_eq!(dnam_engine_get_latency_samples(handle), 12);
            dnam_engine_set_offline_bounce(handle, false);
            assert_eq!(dnam_engine_get_latency_samples(handle), 0);
            dnam_engine_set_maximize_tier_on_bounce(handle, true);

            let l_in = [0.1f32; 16];
            let r_in = [-0.1f32; 16];
            let mut l_out = [0.0f32; 16];
            let mut r_out = [0.0f32; 16];
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 16);

            let mut snap = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::uninit();
            dnam_engine_get_telemetry(handle, snap.as_mut_ptr());

            // Edge cases: null pointers in process_block and process_mono
            dnam_engine_process_block(
                handle,
                core::ptr::null(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                core::ptr::null(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                core::ptr::null_mut(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                core::ptr::null_mut(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                0,
            );

            dnam_engine_process_mono(handle, core::ptr::null(), l_out.as_mut_ptr(), 16);
            dnam_engine_process_mono(handle, l_in.as_ptr(), core::ptr::null_mut(), 16);
            dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 0);

            // Valid setters on active handle
            dnam_engine_set_tier(handle, 2);
            dnam_engine_set_control_value(handle, 0, 0.7);
            dnam_engine_set_control_immediate(handle, 0, 0.7);
            dnam_engine_reset(handle);
            assert_eq!(dnam_engine_set_sample_rate(handle, 48000), 0);
            assert_eq!(dnam_engine_set_sample_rate(handle, 0), -1);

            // Valid container via dnam_engine_create_from_bytes + Debug formatting
            let handle2 =
                dnam_engine_create_from_bytes(container.as_ptr(), container.len(), 0, 48000);
            assert!(!handle2.is_null());
            let dbg_str = alloc::format!("{:?}", *handle2);
            assert!(dbg_str.contains("DnamEngineHandle"));
            dnam_engine_destroy(handle2);

            // Invalid container parsing returns null
            let invalid_bytes = [0u8; 128];
            assert!(
                dnam_engine_create_from_bytes(
                    invalid_bytes.as_ptr(),
                    invalid_bytes.len(),
                    0,
                    48000
                )
                .is_null()
            );
            assert!(
                dnam_engine_create_with_max_tier(
                    invalid_bytes.as_ptr(),
                    invalid_bytes.len(),
                    0,
                    0,
                    48000
                )
                .is_null()
            );

            dnam_engine_destroy(handle);
        }
    }
}
