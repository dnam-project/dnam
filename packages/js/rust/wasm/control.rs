//! WASM `ControlSpaceCore`, ZOH rescaling, and tempo calibration inversion exports
//! (`ref[sdk.ts.parameters.audio-param-semantics]`).

use crate::core::control::{
    ControlSpaceCore, compute_zoh_core, control_stride_for_rate_core, control_tick_ratio_core,
    eval_3node_calibration_core, eval_subdivision_phase_core, invert_tempo_calibration_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{set_error, slice_from_raw, slice_from_raw_mut, str_from_raw};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_create(
    hdr_ptr: *const u8,
    hdr_len: usize,
) -> *mut ControlSpaceCore {
    // SAFETY: Caller passes valid UTF-8 slice in WASM linear memory.
    let hdr_str = match str_from_raw(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => {
            set_error(msg);
            return core::ptr::null_mut();
        }
    };
    match ControlSpaceCore::new(hdr_str) {
        Ok(space) => Box::into_raw(Box::new(space)),
        Err(msg) => {
            set_error(msg);
            core::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_free(ptr: *mut ControlSpaceCore) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `Box::into_raw` in `dnam_wasm_control_space_create`.
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_num_controls(ptr: *const ControlSpaceCore) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `ControlSpaceCore` pointer.
    unsafe { (*ptr).num_controls() as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_clamp_into(
    ptr: *const ControlSpaceCore,
    raw_ptr: *const f32,
    clamped_ptr: *mut f32,
    norm_ptr: *mut f32,
    len: usize,
) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: Caller passes valid non-overlapping f32 slices in WASM linear memory.
    let space = unsafe { &*ptr };
    let raw = slice_from_raw(raw_ptr, len);
    let clamped = slice_from_raw_mut(clamped_ptr, len);
    let norm = slice_from_raw_mut(norm_ptr, len);
    space.clamp_into(raw, clamped, norm)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_to_physical(
    ptr: *const ControlSpaceCore,
    ctrl_idx: usize,
    normalized_u: f32,
    out_ptr: *mut f32,
) -> i32 {
    if ptr.is_null() || out_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let space = unsafe { &*ptr };
    match space.to_physical(ctrl_idx, normalized_u) {
        Ok(v) => {
            // SAFETY: `out_ptr` is non-null and points to a valid `f32` in WASM linear memory.
            unsafe { *out_ptr = v };
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_to_normalized(
    ptr: *const ControlSpaceCore,
    ctrl_idx: usize,
    physical_val: f32,
    out_ptr: *mut f32,
) -> i32 {
    if ptr.is_null() || out_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let space = unsafe { &*ptr };
    match space.to_normalized(ctrl_idx, physical_val) {
        Ok(v) => {
            // SAFETY: `out_ptr` is non-null and points to a valid `f32` in WASM linear memory.
            unsafe { *out_ptr = v };
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_compute_zoh(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
    out9_ptr: *mut f64,
) -> i32 {
    if out9_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null out9_ptr".to_string());
    }
    match compute_zoh_core(
        sample_rate_train_hz,
        sample_rate_host_hz,
        r_ac_train,
        r_dc_train,
        tau_ref_train,
        switch_xfade_train,
    ) {
        Ok(arr) => {
            // SAFETY: `out9_ptr` points to 9 f64 slots in WASM linear memory.
            let out = slice_from_raw_mut(out9_ptr, 9);
            out.copy_from_slice(&arr);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_stride_for_rate(sample_rate_hz: u32) -> u32 {
    control_stride_for_rate_core(sample_rate_hz)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_tick_ratio(sample_rate_hz: u32) -> f64 {
    control_tick_ratio_core(sample_rate_hz)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_subdivision_phase(
    beat_mod12: u8,
    within_beat_q32: u32,
    sub_ptr: *const u8,
    sub_len: usize,
    offset_q32: u32,
    out3_ptr: *mut f64,
) -> i32 {
    if out3_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null out3_ptr".to_string());
    }
    // SAFETY: Caller passes valid UTF-8 slice and 3-f64 output buffer in WASM linear memory.
    let sub = match str_from_raw(sub_ptr, sub_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match eval_subdivision_phase_core(beat_mod12, within_beat_q32, sub, offset_q32) {
        Ok((q32, cos_phi, sin_phi)) => {
            let out = slice_from_raw_mut(out3_ptr, 3);
            out[0] = f64::from(q32);
            out[1] = f64::from(cos_phi);
            out[2] = f64::from(sin_phi);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_3node_calibration(
    anchors_ptr: *const f64,
    u: f64,
    out_ptr: *mut f64,
) -> i32 {
    if anchors_ptr.is_null() || out_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: Caller passes 3 f64 anchors and 1 f64 output in WASM linear memory.
    let anc_slice = slice_from_raw(anchors_ptr, 3);
    let Ok(anc) = <[f64; 3]>::try_from(anc_slice) else {
        return set_error(validation_err("anchors must have 3 elements"));
    };
    // SAFETY: `out_ptr` is non-null and points to a valid `f64` in WASM linear memory.
    unsafe { *out_ptr = eval_3node_calibration_core(anc, u) };
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_invert_tempo_calibration(
    declared_min: f32,
    declared_max: f32,
    safe_min: f32,
    safe_max: f32,
    anchors_ptr: *const f64,
    target: f64,
    flush_subnormals: u8,
    out5_ptr: *mut f64,
) -> i32 {
    if anchors_ptr.is_null() || out5_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: Caller passes 3 f64 anchors and 5 f64 outputs in WASM linear memory.
    let anc_slice = slice_from_raw(anchors_ptr, 3);
    let Ok(anc) = <[f64; 3]>::try_from(anc_slice) else {
        return set_error(validation_err("anchors must have 3 elements"));
    };
    match invert_tempo_calibration_core(
        (declared_min, declared_max),
        (safe_min, safe_max),
        anc,
        target,
        flush_subnormals != 0,
    ) {
        Ok(res) => {
            let out = slice_from_raw_mut(out5_ptr, 5);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}
