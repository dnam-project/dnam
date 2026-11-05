//! WASM `EngineCore` lifecycle and zero-copy real-time audio processing exports
//! (`ref[sdk.ts.engine.zero-copy-and-async]`, `ref[sdk.ts.web-audio.realtime-bridge]`).

use crate::core::engine::{
    EngineCore, TELEMETRY_F64_LEN, backend_kind_code, backend_kind_from_code,
};
use crate::error::validation_err;
use crate::wasm::abi::{set_error, slice_from_raw, slice_from_raw_mut, str_from_raw};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_create(
    container_ptr: *const u8,
    container_len: usize,
    tier: u8,
    sample_rate_hz: u32,
    backend_code: u8,
    max_supported_tier: i32,
    offline_bounce: u8,
    maximize_tier_on_bounce: u8,
) -> *mut EngineCore {
    let backend = match backend_kind_from_code(backend_code) {
        Ok(b) => b,
        Err(msg) => {
            set_error(msg);
            return core::ptr::null_mut();
        }
    };
    // SAFETY: Caller passes a valid container byte slice in WASM linear memory.
    let bytes = slice_from_raw(container_ptr, container_len);
    let max_tier = if max_supported_tier >= 0 {
        Some(max_supported_tier as u8)
    } else {
        None
    };
    match EngineCore::new(
        bytes,
        tier,
        sample_rate_hz,
        backend,
        max_tier,
        offline_bounce != 0,
        maximize_tier_on_bounce != 0,
    ) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(msg) => {
            set_error(msg);
            core::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_free(ptr: *mut EngineCore) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated via `Box::into_raw` in `dnam_wasm_engine_create`.
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reload(
    ptr: *mut EngineCore,
    container_ptr: *const u8,
    container_len: usize,
    tier: i32,
    sample_rate_hz: u32,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is valid and `container_ptr` points to `container_len` bytes.
    let engine = unsafe { &mut *ptr };
    let bytes = slice_from_raw(container_ptr, container_len);
    let tier_opt = if tier >= 0 { Some(tier as u8) } else { None };
    let sr_opt = if sample_rate_hz > 0 {
        Some(sample_rate_hz)
    } else {
        None
    };
    match engine.reload(bytes, tier_opt, sr_opt) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_backend_code(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    let engine = unsafe { &*ptr };
    backend_kind_code(engine.backend_kind())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_tier(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).tier() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_tier(ptr: *mut EngineCore, tier: u8) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    match unsafe { (*ptr).set_tier(tier) } {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_max_supported_tier(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 3;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).max_supported_tier() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_max_supported_tier(ptr: *mut EngineCore, tier: u8) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    match unsafe { (*ptr).set_max_supported_tier(tier) } {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_max_available_tier(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).max_available_tier() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_effective_tier(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).effective_tier() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_sample_rate(ptr: *const EngineCore) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).sample_rate_hz() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_sample_rate(
    ptr: *mut EngineCore,
    sample_rate_hz: u32,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    match unsafe { (*ptr).set_sample_rate(sample_rate_hz) } {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_latency_samples(ptr: *const EngineCore) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).latency_samples() as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_offline_bounce(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    u8::from(unsafe { (*ptr).offline_bounce() })
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_offline_bounce(ptr: *mut EngineCore, enabled: u8) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_offline_bounce(enabled != 0) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_maximize_tier_on_bounce(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    u8::from(unsafe { (*ptr).maximize_tier_on_bounce() })
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_maximize_tier_on_bounce(ptr: *mut EngineCore, enabled: u8) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_maximize_tier_on_bounce(enabled != 0) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_num_controls(ptr: *const EngineCore) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).num_controls() as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_raw_controls_into(
    ptr: *const EngineCore,
    out_ptr: *mut f32,
    len: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let engine = unsafe { &*ptr };
    let out = slice_from_raw_mut(out_ptr, len);
    let src = engine.raw_control_values();
    let n = src.len().min(out.len());
    out[..n].copy_from_slice(&src[..n]);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_clamped_controls_into(
    ptr: *const EngineCore,
    out_ptr: *mut f32,
    len: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let engine = unsafe { &*ptr };
    let out = slice_from_raw_mut(out_ptr, len);
    let src = engine.clamped_control_values();
    let n = src.len().min(out.len());
    out[..n].copy_from_slice(&src[..n]);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_clamped_control_value(
    ptr: *const EngineCore,
    ctrl_idx: usize,
    out_ptr: *mut f32,
) -> i32 {
    if ptr.is_null() || out_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let engine = unsafe { &*ptr };
    match engine.clamped_control_value(ctrl_idx) {
        Ok(v) => {
            // SAFETY: `out_ptr` is non-null and points to a valid `f32` in WASM linear memory.
            unsafe { *out_ptr = v };
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_control_value(
    ptr: *mut EngineCore,
    ctrl_idx: usize,
    value: f32,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    match unsafe { (*ptr).set_control_value(ctrl_idx, value) } {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_control_immediate(
    ptr: *mut EngineCore,
    ctrl_idx: usize,
    value: f32,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    match unsafe { (*ptr).set_control_immediate(ctrl_idx, value) } {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_controls_immediate_batch(
    ptr: *mut EngineCore,
    indices_ptr: *const u32,
    values_ptr: *const f32,
    len: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr`, `indices_ptr`, and `values_ptr` are valid in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let indices = slice_from_raw(indices_ptr, len);
    let values = slice_from_raw(values_ptr, len);
    let pairs: Vec<(usize, f32)> = indices
        .iter()
        .copied()
        .zip(values.iter().copied())
        .map(|(i, v)| (i as usize, v))
        .collect();
    match engine.set_controls_immediate_batch(&pairs) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reset_controls_to_defaults(ptr: *mut EngineCore) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).reset_controls_to_defaults() };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reset(ptr: *mut EngineCore) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).reset_to_dc_equilibrium() };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_supports_freeze(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    u8::from(unsafe { (*ptr).supports_freeze() })
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_is_frozen(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    u8::from(unsafe { (*ptr).is_frozen() })
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_freeze(ptr: *mut EngineCore, freeze: u8) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_freeze(freeze != 0) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_freeze_amount(ptr: *mut EngineCore, alpha: f32) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_freeze_amount(alpha) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_bypass_state(ptr: *const EngineCore) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    let (b, t) = unsafe { (*ptr).bypass_state() };
    u8::from(b) | (u8::from(t) << 1)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_bypass(ptr: *mut EngineCore, bypassed: u8, trails: u8) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_bypass(bypassed != 0, trails != 0) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_stereo_width(ptr: *const EngineCore) -> f32 {
    if ptr.is_null() {
        return 1.0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).stereo_width() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_stereo_width(ptr: *mut EngineCore, width: f32) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe { (*ptr).set_stereo_width(width) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_configure_host_mix(
    ptr: *mut EngineCore,
    blend_wet: f32,
    output_gain_db: f32,
    apply_unity_leveling: u8,
) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a valid `EngineCore` pointer.
        unsafe {
            (*ptr).configure_host_mix_and_leveling(
                blend_wet,
                output_gain_db,
                apply_unity_leveling != 0,
            )
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_current_bpm(ptr: *const EngineCore) -> f32 {
    if ptr.is_null() {
        return 120.0;
    }
    // SAFETY: `ptr` is a valid `EngineCore` pointer.
    unsafe { (*ptr).current_bpm() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_sync_tempo(
    ptr: *mut EngineCore,
    bpm: f32,
    sub_ptr: *const u8,
    sub_len: usize,
    fold_octave: u8,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` and `sub_ptr` are valid in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let sub_opt = if sub_len == 0 || sub_ptr.is_null() {
        None
    } else {
        match str_from_raw(sub_ptr, sub_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    match engine.sync_tempo(bpm, sub_opt, fold_octave != 0) {
        Ok(c) => c as i32,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_register_tap_tempo(
    ptr: *mut EngineCore,
    sub_ptr: *const u8,
    sub_len: usize,
    fold_octave: u8,
    out_bpm_ptr: *mut f32,
) -> i32 {
    if ptr.is_null() || out_bpm_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    // SAFETY: `ptr`, `sub_ptr`, and `out_bpm_ptr` are valid in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let sub_opt = if sub_len == 0 || sub_ptr.is_null() {
        None
    } else {
        match str_from_raw(sub_ptr, sub_len) {
            Ok(s) => Some(s),
            Err(msg) => return set_error(msg),
        }
    };
    match engine.register_tap_tempo(sub_opt, fold_octave != 0) {
        Ok((bpm, count)) => {
            // SAFETY: `out_bpm_ptr` is non-null and points to a valid `f32` in WASM linear memory.
            unsafe { *out_bpm_ptr = bpm };
            count as i32
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_apply_preset(
    ptr: *mut EngineCore,
    preset_ptr: *const u8,
    preset_len: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr` and `preset_ptr` are valid in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let preset_id = match str_from_raw(preset_ptr, preset_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    match engine.apply_preset(preset_id) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_stereo_in_place(
    ptr: *mut EngineCore,
    left_io_ptr: *mut f32,
    right_io_ptr: *mut f32,
    num_samples: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: `ptr`, `left_io_ptr`, and `right_io_ptr` are valid non-overlapping slices in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let left_io = slice_from_raw_mut(left_io_ptr, num_samples);
    let right_io = slice_from_raw_mut(right_io_ptr, num_samples);
    match engine.process_stereo_in_place(left_io, right_io) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_stereo(
    ptr: *mut EngineCore,
    left_in_ptr: *const f32,
    right_in_ptr: *const f32,
    left_out_ptr: *mut f32,
    right_out_ptr: *mut f32,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: Input and output slices are valid and non-overlapping in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let left_in = slice_from_raw(left_in_ptr, num_samples);
    let right_in = slice_from_raw(right_in_ptr, num_samples);
    let left_out = slice_from_raw_mut(left_out_ptr, num_samples);
    let right_out = slice_from_raw_mut(right_out_ptr, num_samples);
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    match engine.process_stereo(left_in, right_in, left_out, right_out, bs) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono_to_stereo(
    ptr: *mut EngineCore,
    mono_in_ptr: *const f32,
    left_out_ptr: *mut f32,
    right_out_ptr: *mut f32,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: Input and output slices are valid and non-overlapping in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let mono_in = slice_from_raw(mono_in_ptr, num_samples);
    let left_out = slice_from_raw_mut(left_out_ptr, num_samples);
    let right_out = slice_from_raw_mut(right_out_ptr, num_samples);
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    match engine.process_mono_to_stereo(mono_in, left_out, right_out, bs) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono(
    ptr: *mut EngineCore,
    mono_in_ptr: *const f32,
    mono_out_ptr: *mut f32,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: Input and output slices are valid and non-overlapping in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let mono_in = slice_from_raw(mono_in_ptr, num_samples);
    let mono_out = slice_from_raw_mut(mono_out_ptr, num_samples);
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    match engine.process_mono(mono_in, mono_out, bs) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_render_offline_bounce(
    ptr: *mut EngineCore,
    left_in_ptr: *const f32,
    right_in_ptr: *const f32,
    left_out_ptr: *mut f32,
    right_out_ptr: *mut f32,
    num_samples: usize,
) -> i32 {
    if ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    }
    // SAFETY: Input and output slices are valid and non-overlapping in WASM linear memory.
    let engine = unsafe { &mut *ptr };
    let left_in = slice_from_raw(left_in_ptr, num_samples);
    let right_in = slice_from_raw(right_in_ptr, num_samples);
    let left_out = slice_from_raw_mut(left_out_ptr, num_samples);
    let right_out = slice_from_raw_mut(right_out_ptr, num_samples);
    match engine.render_offline_bounce(left_in, right_in, left_out, right_out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_telemetry_into(
    ptr: *const EngineCore,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    if ptr.is_null() || out_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    if out_len < TELEMETRY_F64_LEN {
        return set_error(validation_err(format!(
            "telemetry_into requires {TELEMETRY_F64_LEN} f64 slots, got {out_len}"
        )));
    }
    // SAFETY: `ptr` and `out_ptr` are valid in WASM linear memory.
    let engine = unsafe { &*ptr };
    let out_slice = slice_from_raw_mut(out_ptr, TELEMETRY_F64_LEN);
    let Ok(out_arr) = <&mut [f64; TELEMETRY_F64_LEN]>::try_from(out_slice) else {
        return set_error(validation_err("telemetry_into length mismatch"));
    };
    engine.telemetry_into_f64(out_arr);
    0
}
