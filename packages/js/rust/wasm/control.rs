//! WASM `ControlSpaceCore`, ZOH rescaling, and tempo calibration inversion exports
//! (`ref[sdk.ts.parameters.audio-param-semantics]`, `ref[sdk.ts.runtime.zero-unsafe]`).

use crate::core::control::{
    ControlSpaceCore, compute_zoh_core, control_stride_for_rate_core, control_tick_ratio_core,
    eval_3node_calibration_core, eval_subdivision_phase_core, invert_tempo_calibration_core,
};
use crate::error::validation_err;
use crate::wasm::abi::{
    insert_control_space, read_f64_vec, read_str, remove_control_space, set_error,
    with_control_space, with_f32_1_in_2_out, write_f32, write_f64, write_f64_slice,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_create(hdr_ptr: usize, hdr_len: usize) -> u32 {
    let hdr_str = match read_str(hdr_ptr, hdr_len) {
        Ok(s) => s,
        Err(msg) => {
            set_error(msg);
            return 0;
        }
    };
    match ControlSpaceCore::new(&hdr_str) {
        Ok(space) => insert_control_space(space),
        Err(msg) => {
            set_error(msg);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_free(handle: u32) {
    remove_control_space(handle);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_num_controls(handle: u32) -> u32 {
    with_control_space(handle, |space| space.num_controls() as u32).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_clamp_into(
    handle: u32,
    raw_ptr: usize,
    clamped_ptr: usize,
    norm_ptr: usize,
    len: usize,
) -> u32 {
    with_control_space(handle, |space| {
        with_f32_1_in_2_out(raw_ptr, clamped_ptr, norm_ptr, len, |raw, clamped, norm| {
            space.clamp_into(raw, clamped, norm)
        })
        .unwrap_or(0)
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_to_physical(
    handle: u32,
    ctrl_idx: usize,
    normalized_u: f32,
    out_ptr: usize,
) -> i32 {
    if out_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let Some(res) = with_control_space(handle, |space| space.to_physical(ctrl_idx, normalized_u))
    else {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    };
    match res {
        Ok(v) => {
            if !write_f32(out_ptr, v) {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_to_normalized(
    handle: u32,
    ctrl_idx: usize,
    physical_val: f32,
    out_ptr: usize,
) -> i32 {
    if out_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let Some(res) = with_control_space(handle, |space| space.to_normalized(ctrl_idx, physical_val))
    else {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    };
    match res {
        Ok(v) => {
            if !write_f32(out_ptr, v) {
                return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
            }
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
    out9_ptr: usize,
) -> i32 {
    if out9_ptr == 0 {
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
            if !write_f64_slice(out9_ptr, &arr) {
                return set_error("[DNAM-VALIDATION] invalid out9_ptr".to_string());
            }
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
    sub_ptr: usize,
    sub_len: usize,
    offset_q32: u32,
    out3_ptr: usize,
) -> i32 {
    if out3_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null out3_ptr".to_string());
    }
    let sub = match read_str(sub_ptr, sub_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match eval_subdivision_phase_core(beat_mod12, within_beat_q32, &sub, offset_q32) {
        Ok((q32, cos_phi, sin_phi)) => {
            let arr = [f64::from(q32), f64::from(cos_phi), f64::from(sin_phi)];
            if !write_f64_slice(out3_ptr, &arr) {
                return set_error("[DNAM-VALIDATION] invalid out3_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_3node_calibration(
    anchors_ptr: usize,
    u: f64,
    out_ptr: usize,
) -> i32 {
    if anchors_ptr == 0 || out_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let anc_vec = read_f64_vec(anchors_ptr, 3);
    let Ok(anc) = <[f64; 3]>::try_from(anc_vec.as_slice()) else {
        return set_error(validation_err("anchors must have 3 elements"));
    };
    if !write_f64(out_ptr, eval_3node_calibration_core(anc, u)) {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_invert_tempo_calibration(
    declared_min: f32,
    declared_max: f32,
    safe_min: f32,
    safe_max: f32,
    anchors_ptr: usize,
    target: f64,
    flush_subnormals: u8,
    out5_ptr: usize,
) -> i32 {
    if anchors_ptr == 0 || out5_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let anc_vec = read_f64_vec(anchors_ptr, 3);
    let Ok(anc) = <[f64; 3]>::try_from(anc_vec.as_slice()) else {
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
            if !write_f64_slice(out5_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out5_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}
