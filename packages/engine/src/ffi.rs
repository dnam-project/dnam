//! `extern "C"` C-ABI & WASM Linear-Memory FFI Exports (`ffi.rs`).
//!
//! Exposes a zero-allocation C-ABI surface over [`DynamicDnamEngine`](crate::DynamicDnamEngine)
//! and [`DnamTelemetrySnapshot`](crate::DnamTelemetrySnapshot) for C/C++ audio hosts,
//! `packages/nodejs` WASM `AudioWorkletProcessor`, and embedded firmware.

use alloc::boxed::Box;
use core::slice;

use crate::{DnamTelemetrySnapshot, DynamicDnamEngine, Tier};

/// Opaque C-ABI handle owning a `'static` [`DynamicDnamEngine`].
#[derive(Debug)]
pub struct DnamEngineHandle {
    engine: DynamicDnamEngine<'static>,
}

/// Creates a new [`DnamEngineHandle`] from a `.dnam` or `.dnam-ckpt` byte buffer, copying the
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

/// Switches the active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Ultra`) with `0.00 dB` LUFS continuity.
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

/// Resets all Core 1 and Core 2 states to their exact analytical DC resting equilibrium.
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
