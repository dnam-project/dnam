//! WebAssembly (`wasm32-unknown-unknown` + `simd128`) C-ABI exports wrapping the shared `dnam-js` core.

use std::alloc::{Layout, alloc, dealloc};
use std::cell::RefCell;
use std::slice;

use crate::container::{
    ControlSpaceCore, build_container_core, build_reference_container_core, compute_zoh_core,
    crc32c_core, pack_uplo_u_core, unpack_uplo_u_core, update_header_in_place_core,
    validate_container_core,
};
use crate::dsp::{
    cubic_lagrange_weights_core, eval_chebyshev_lfo_core, eval_chebyshev_lfo_slice_core,
    eval_lobatto_basis_core, eval_lobatto_basis_slice_core, eval_rank4_fm_core,
    eval_trust_region_softclip_core, eval_trust_region_softclip_slice_core,
    givens_householder_mix_8ch_core, mufdn_coercive_step_core, mufdn_slice_core,
    poly_adaa_algebraic_core, poly_adaa_scan_core, poly_adaa_step_core, selective_rglru_scan_core,
    selective_rglru_step_core, unit_rotor_core, unit_rotor_slice_core,
};
use crate::engine::{EngineCore, backend_kind_code, backend_kind_from_code};
use crate::provenance::{
    apply_isometric_gauge_watermark_core, apply_tardos_watermark_core,
    build_creator_signing_message_core, build_marketplace_signing_message_core,
    chacha20_xor_arena_core, extract_structural_hash_core, generate_tardos_code_core,
    immutable_header_sha256_core, score_tardos_watermark_core, sha256_hex_core,
    structural_cosine_similarity_core,
};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static LAST_RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LAST_ARENA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(msg: String) -> i32 {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
    -1
}

fn set_result(bytes: Vec<u8>) -> i32 {
    LAST_RESULT.with(|r| {
        *r.borrow_mut() = bytes;
    });
    0
}

unsafe fn slice_from_raw<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

unsafe fn slice_from_raw_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 || ptr.is_null() {
        &mut []
    } else {
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

unsafe fn str_from_raw<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    let bytes = unsafe { slice_from_raw(ptr, len) };
    core::str::from_utf8(bytes).map_err(|e| format!("[DNAM-VALIDATION] Invalid UTF-8: {e}"))
}

// ============================================================================
// 1. Linear Memory Allocator & Thread-Local Result / Error Buffers
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_alloc(size: usize, align: usize) -> *mut u8 {
    let safe_align = align.max(8).next_power_of_two();
    let safe_size = size.max(1);
    let Ok(layout) = Layout::from_size_align(safe_size, safe_align) else {
        return core::ptr::null_mut();
    };
    unsafe { alloc(layout) }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    let safe_align = align.max(8).next_power_of_two();
    let safe_size = size.max(1);
    if let Ok(layout) = Layout::from_size_align(safe_size, safe_align) {
        unsafe { dealloc(ptr, layout) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_ptr() -> *const u8 {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_len() -> usize {
    LAST_ERROR.with(|e| e.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_ptr() -> *const u8 {
    LAST_RESULT.with(|r| r.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_len() -> usize {
    LAST_RESULT.with(|r| r.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_ptr() -> *const u8 {
    LAST_ARENA.with(|a| a.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_len() -> usize {
    LAST_ARENA.with(|a| a.borrow().len())
}

// ============================================================================
// 2. EngineCore Lifecycle & Real-Time Processing
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_create(
    container_ptr: *const u8,
    container_len: usize,
    tier: u8,
    sample_rate_hz: u32,
    backend_code: u8,
) -> *mut EngineCore {
    let backend = match backend_kind_from_code(backend_code) {
        Ok(b) => b,
        Err(e) => {
            set_error(e);
            return core::ptr::null_mut();
        }
    };
    let bytes = unsafe { slice_from_raw(container_ptr, container_len) };
    match EngineCore::new(bytes, tier, sample_rate_hz, backend) {
        Ok(core) => Box::into_raw(Box::new(core)),
        Err(e) => {
            set_error(e);
            core::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_destroy(handle: *mut EngineCore) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reload(
    handle: *mut EngineCore,
    container_ptr: *const u8,
    container_len: usize,
    tier: i32,
    sample_rate_hz: u32,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    let bytes = unsafe { slice_from_raw(container_ptr, container_len) };
    let opt_tier = if tier >= 0 { Some(tier as u8) } else { None };
    let opt_sr = if sample_rate_hz > 0 {
        Some(sample_rate_hz)
    } else {
        None
    };
    match engine.reload(bytes, opt_tier, opt_sr) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_backend_code(handle: *const EngineCore) -> u8 {
    if handle.is_null() {
        return 0;
    }
    let engine = unsafe { &*handle };
    backend_kind_code(engine.backend_kind())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_active_tier(handle: *const EngineCore) -> u8 {
    if handle.is_null() {
        return 0;
    }
    let engine = unsafe { &*handle };
    engine.active_tier()
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_tier(handle: *mut EngineCore, tier: u8) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    match engine.set_tier(tier) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_sample_rate(handle: *const EngineCore) -> u32 {
    if handle.is_null() {
        return 48_000;
    }
    let engine = unsafe { &*handle };
    engine.sample_rate_hz()
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_sample_rate(
    handle: *mut EngineCore,
    sample_rate_hz: u32,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    match engine.set_sample_rate(sample_rate_hz) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_control_value(
    handle: *mut EngineCore,
    ctrl_idx: u32,
    value: f32,
) {
    if !handle.is_null() {
        let engine = unsafe { &mut *handle };
        engine.set_control_value(ctrl_idx as usize, value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_control_immediate(
    handle: *mut EngineCore,
    ctrl_idx: u32,
    value: f32,
) {
    if !handle.is_null() {
        let engine = unsafe { &mut *handle };
        engine.set_control_immediate(ctrl_idx as usize, value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reset(handle: *mut EngineCore) {
    if !handle.is_null() {
        let engine = unsafe { &mut *handle };
        engine.reset();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_stereo(
    handle: *mut EngineCore,
    left_in_ptr: *const f32,
    right_in_ptr: *const f32,
    left_out_ptr: *mut f32,
    right_out_ptr: *mut f32,
    num_samples: usize,
    buffer_size: usize,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    let left_in = unsafe { slice_from_raw(left_in_ptr, num_samples) };
    let right_in = unsafe { slice_from_raw(right_in_ptr, num_samples) };
    let left_out = unsafe { slice_from_raw_mut(left_out_ptr, num_samples) };
    let right_out = unsafe { slice_from_raw_mut(right_out_ptr, num_samples) };
    let opt_chunk = if buffer_size > 0 {
        Some(buffer_size)
    } else {
        None
    };
    match engine.process_stereo(left_in, right_in, left_out, right_out, opt_chunk) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono_to_stereo(
    handle: *mut EngineCore,
    mono_in_ptr: *const f32,
    left_out_ptr: *mut f32,
    right_out_ptr: *mut f32,
    num_samples: usize,
    buffer_size: usize,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    let mono_in = unsafe { slice_from_raw(mono_in_ptr, num_samples) };
    let left_out = unsafe { slice_from_raw_mut(left_out_ptr, num_samples) };
    let right_out = unsafe { slice_from_raw_mut(right_out_ptr, num_samples) };
    let opt_chunk = if buffer_size > 0 {
        Some(buffer_size)
    } else {
        None
    };
    match engine.process_mono_to_stereo(mono_in, left_out, right_out, opt_chunk) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono(
    handle: *mut EngineCore,
    mono_in_ptr: *const f32,
    mono_out_ptr: *mut f32,
    num_samples: usize,
    buffer_size: usize,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine handle".to_string());
    }
    let engine = unsafe { &mut *handle };
    let mono_in = unsafe { slice_from_raw(mono_in_ptr, num_samples) };
    let mono_out = unsafe { slice_from_raw_mut(mono_out_ptr, num_samples) };
    let opt_chunk = if buffer_size > 0 {
        Some(buffer_size)
    } else {
        None
    };
    match engine.process_mono(mono_in, mono_out, opt_chunk) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

/// Writes 50 `f64` telemetry values into `out_f64_ptr`.
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_telemetry(
    handle: *const EngineCore,
    out_f64_ptr: *mut f64,
) -> i32 {
    if handle.is_null() || out_f64_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] Null engine or output pointer".to_string());
    }
    let engine = unsafe { &*handle };
    let out = unsafe { slice_from_raw_mut(out_f64_ptr, 50) };
    let Ok(arr) = out.try_into() else {
        return set_error("[DNAM-VALIDATION] Telemetry output slice length < 50".to_string());
    };
    engine.telemetry_into_f64(arr);
    0
}

// ============================================================================
// 3. ControlSpaceCore, Container, ZOH, Gram & CRC32C
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_create(
    header_json_ptr: *const u8,
    header_json_len: usize,
) -> *mut ControlSpaceCore {
    let header_json = match unsafe { str_from_raw(header_json_ptr, header_json_len) } {
        Ok(s) => s,
        Err(e) => {
            set_error(e);
            return core::ptr::null_mut();
        }
    };
    match ControlSpaceCore::new(header_json) {
        Ok(cs) => Box::into_raw(Box::new(cs)),
        Err(e) => {
            set_error(e);
            core::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_destroy(handle: *mut ControlSpaceCore) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_control_space_clamp(
    handle: *const ControlSpaceCore,
    raw_ptr: *const f32,
    num_controls: usize,
    clamped_out_ptr: *mut f32,
    norm_out_ptr: *mut f32,
    was_clamped_out_ptr: *mut u8,
) -> i32 {
    if handle.is_null() {
        return set_error("[DNAM-VALIDATION] Null ControlSpace handle".to_string());
    }
    let cs = unsafe { &*handle };
    let n = cs.num_controls().min(num_controls);
    let raw = unsafe { slice_from_raw(raw_ptr, num_controls) };
    let clamped_out = unsafe { slice_from_raw_mut(clamped_out_ptr, n) };
    let norm_out = unsafe { slice_from_raw_mut(norm_out_ptr, n) };
    let was_clamped_out = unsafe { slice_from_raw_mut(was_clamped_out_ptr, n) };
    let mask = cs.clamp_into(raw, clamped_out, norm_out);
    for (i, slot) in was_clamped_out.iter_mut().enumerate().take(n) {
        *slot = u8::from((mask & (1_u32 << i)) != 0);
    }
    n as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_validate_container(
    container_ptr: *const u8,
    container_len: usize,
    out_f64_ptr: *mut f64,
) -> i32 {
    if out_f64_ptr.is_null() {
        return set_error("[DNAM-VALIDATION] Null output pointer".to_string());
    }
    let bytes = unsafe { slice_from_raw(container_ptr, container_len) };
    let info = match validate_container_core(bytes) {
        Ok(i) => i,
        Err(e) => return set_error(e),
    };
    let p = info.preamble;
    let out = unsafe { slice_from_raw_mut(out_f64_ptr, 25) };
    out[0] = if info.magic_str == "DNCK" { 1.0 } else { 0.0 };
    out[1] = f64::from(p.version_major);
    out[2] = f64::from(p.version_minor);
    out[3] = f64::from(p.flags);
    out[4] = f64::from(p.num_controls);
    out[5] = f64::from(p.ring_buffer_pow2_log2);
    out[6] = f64::from(p.sample_rate_train_hz);
    out[7] = f64::from(p.json_header_byte_len);
    out[8] = f64::from(p.tensor_arena_offset);
    out[9] = f64::from(p.tensor_arena_byte_len);
    out[10] = f64::from(p.arena_offset_lite_end);
    out[11] = f64::from(p.arena_offset_full_end);
    out[12] = f64::from(p.arena_offset_ultra_end);
    out[13] = f64::from(p.arena_crc32c_lite);
    out[14] = f64::from(p.arena_crc32c_full);
    out[15] = f64::from(p.arena_crc32c_ultra);
    out[16] = f64::from(p.num_ckpt_stems);
    out[17] = f64::from(p.control_decimation);
    out[18] = f64::from(p.head_svd_rank);
    out[19] = f64::from(p.tail_time_ms);
    out[20] = f64::from(p.header_crc32c);
    out[21] = if p.is_core2_active() { 1.0 } else { 0.0 };
    out[22] = if p.is_lfo_env_active() { 1.0 } else { 0.0 };
    out[23] = if p.is_stereo_out() { 1.0 } else { 0.0 };
    out[24] = match info.verification_status.as_str() {
        "CreatorVerified" => 1.0,
        "MarketplaceVerified" => 2.0,
        _ => 0.0,
    };
    LAST_ARENA.with(|a| {
        *a.borrow_mut() = info.arena_bytes;
    });
    set_result(info.header_json.into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_container(
    header_json_ptr: *const u8,
    header_json_len: usize,
    arena_ptr: *const u8,
    arena_len: usize,
    crc_src_ptr: *const u8,
    crc_src_len: usize,
    encrypted_arena: u8,
) -> i32 {
    let header_json = match unsafe { str_from_raw(header_json_ptr, header_json_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    let arena = unsafe { slice_from_raw(arena_ptr, arena_len) };
    let crc_src = if !crc_src_ptr.is_null() && crc_src_len > 0 {
        Some(unsafe { slice_from_raw(crc_src_ptr, crc_src_len) })
    } else {
        None
    };
    match build_container_core(header_json, arena, crc_src, encrypted_arena != 0) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_reference_container(core2_active: u8, is_checkpoint: u8) -> i32 {
    match build_reference_container_core(core2_active != 0, is_checkpoint != 0) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_update_header_in_place(
    container_ptr: *const u8,
    container_len: usize,
    header_json_ptr: *const u8,
    header_json_len: usize,
) -> i32 {
    let container = unsafe { slice_from_raw(container_ptr, container_len) };
    let header_json = match unsafe { str_from_raw(header_json_ptr, header_json_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    match update_header_in_place_core(container, header_json) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_compute_zoh(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    control_decimation_train: u8,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
    out_f64_ptr: *mut f64,
) -> i32 {
    let z = match compute_zoh_core(
        sample_rate_train_hz,
        sample_rate_host_hz,
        control_decimation_train,
        r_ac_train,
        r_dc_train,
        tau_ref_train,
        switch_xfade_train,
    ) {
        Ok(z) => z,
        Err(e) => return set_error(e),
    };
    let out = unsafe { slice_from_raw_mut(out_f64_ptr, 8) };
    out[0] = f64::from(z.sample_rate_host_hz);
    out[1] = z.rho_sr;
    out[2] = z.control_decimation_host as f64;
    out[3] = z.rho_ctrl;
    out[4] = f64::from(z.r_ac_host);
    out[5] = f64::from(z.r_dc_host);
    out[6] = f64::from(z.tau_ref_samples_host);
    out[7] = f64::from(z.switch_xfade_samples_host);
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_pack_uplo_u(
    dense_ptr: *const f64,
    dense_len: usize,
    n_dim: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    let dense = unsafe { slice_from_raw(dense_ptr, dense_len) };
    match pack_uplo_u_core(dense, n_dim) {
        Ok(packed) => {
            if out_len < packed.len() {
                return set_error("[DNAM-VALIDATION] Output buffer too small".to_string());
            }
            let out = unsafe { slice_from_raw_mut(out_ptr, packed.len()) };
            out.copy_from_slice(&packed);
            packed.len() as i32
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unpack_uplo_u(
    packed_ptr: *const f64,
    packed_len: usize,
    n_tier: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    let packed = unsafe { slice_from_raw(packed_ptr, packed_len) };
    match unpack_uplo_u_core(packed, n_tier) {
        Ok(unpacked) => {
            if out_len < unpacked.len() {
                return set_error("[DNAM-VALIDATION] Output buffer too small".to_string());
            }
            let out = unsafe { slice_from_raw_mut(out_ptr, unpacked.len()) };
            out.copy_from_slice(&unpacked);
            unpacked.len() as i32
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_crc32c(data_ptr: *const u8, data_len: usize) -> u32 {
    let data = unsafe { slice_from_raw(data_ptr, data_len) };
    crc32c_core(data)
}

// ============================================================================
// 4. Reference DSP Kernels (`f64`)
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_poly_adaa_algebraic(
    u_prev: f64,
    u_curr: f64,
    beta: f64,
    delta: f64,
) -> f64 {
    poly_adaa_algebraic_core(u_prev, u_curr, beta, delta)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_poly_adaa_step(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    out_4_ptr: *mut f64,
) {
    let res = poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc);
    let out = unsafe { slice_from_raw_mut(out_4_ptr, 4) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_poly_adaa_scan(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    reset: u8,
    state_out_3_ptr: *mut f64,
) -> i32 {
    let inp = unsafe { slice_from_raw(in_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_ptr, len) };
    match poly_adaa_scan_core(
        inp,
        out,
        b_conv,
        alpha,
        beta,
        delta,
        r_ac,
        u_prev,
        a_prev,
        m_dc,
        reset != 0,
    ) {
        Ok(state) => {
            let st_out = unsafe { slice_from_raw_mut(state_out_3_ptr, 3) };
            st_out.copy_from_slice(&state);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_selective_rglru_step(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: f64,
    z_im: f64,
    out_3_ptr: *mut f64,
) {
    let res = selective_rglru_step_core(
        x,
        r_low,
        dr_span,
        w_sel,
        b_gamma_norm,
        c_real,
        c_imag,
        u_rot_cos,
        u_rot_sin,
        z_re,
        z_im,
    );
    let out = unsafe { slice_from_raw_mut(out_3_ptr, 3) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_selective_rglru_scan(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: f64,
    z_im: f64,
    reset: u8,
    state_out_2_ptr: *mut f64,
) -> i32 {
    let inp = unsafe { slice_from_raw(in_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_ptr, len) };
    match selective_rglru_scan_core(
        inp,
        out,
        r_low,
        dr_span,
        w_sel,
        b_gamma_norm,
        c_real,
        c_imag,
        u_rot_cos,
        u_rot_sin,
        z_re,
        z_im,
        reset != 0,
    ) {
        Ok(state) => {
            let st_out = unsafe { slice_from_raw_mut(state_out_2_ptr, 2) };
            st_out.copy_from_slice(&state);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor(phi: f64, out_2_ptr: *mut f64) {
    let res = unit_rotor_core(phi as f32);
    let out = unsafe { slice_from_raw_mut(out_2_ptr, 2) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor_slice(
    phi_ptr: *const f64,
    len: usize,
    out_2n_ptr: *mut f64,
) -> i32 {
    let phi = unsafe { slice_from_raw(phi_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_2n_ptr, 2 * len) };
    match unit_rotor_slice_core(phi, out) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_cubic_lagrange_weights(mu: f64, out_4_ptr: *mut f64) {
    let res = cubic_lagrange_weights_core(mu);
    let out = unsafe { slice_from_raw_mut(out_4_ptr, 4) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_mix_8ch(
    x_ptr: *const f64,
    x_len: usize,
    cos_psi: f64,
    sin_psi: f64,
    out_8_ptr: *mut f64,
) -> i32 {
    let x = unsafe { slice_from_raw(x_ptr, x_len) };
    match givens_householder_mix_8ch_core(x, cos_psi, sin_psi) {
        Ok(res) => {
            let out = unsafe { slice_from_raw_mut(out_8_ptr, 8) };
            out.copy_from_slice(&res);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_mufdn_coercive_step(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: f64,
    dc_y_prev: f64,
    out_3_ptr: *mut f64,
) {
    let res = mufdn_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev);
    let out = unsafe { slice_from_raw_mut(out_3_ptr, 3) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_mufdn_slice(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    delay_samples: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> i32 {
    let inp = unsafe { slice_from_raw(in_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_ptr, len) };
    match mufdn_slice_core(inp, out, delay_samples, g_lin, sigma_knee, v_ceil, r_dc) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis(
    u: f64,
    anchors_ptr: *const f64,
    anchors_len: usize,
    degree: usize,
    out_3_ptr: *mut f64,
) -> i32 {
    let anchors = unsafe { slice_from_raw(anchors_ptr, anchors_len) };
    match eval_lobatto_basis_core(u, anchors, degree) {
        Ok(res) => {
            let out = unsafe { slice_from_raw_mut(out_3_ptr, 3) };
            out.copy_from_slice(&res);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis_slice(
    u_ptr: *const f64,
    len: usize,
    out_3n_ptr: *mut f64,
    anchors_ptr: *const f64,
    anchors_len: usize,
    degree: usize,
) -> i32 {
    let u = unsafe { slice_from_raw(u_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_3n_ptr, 3 * len) };
    let anchors = unsafe { slice_from_raw(anchors_ptr, anchors_len) };
    match eval_lobatto_basis_slice_core(u, out, anchors, degree) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_rank4_fm(
    features_ptr: *const f64,
    features_len: usize,
    v_factors_ptr: *const f64,
    v_factors_len: usize,
    out_4_ptr: *mut f64,
) -> i32 {
    let features = unsafe { slice_from_raw(features_ptr, features_len) };
    let v_factors = unsafe { slice_from_raw(v_factors_ptr, v_factors_len) };
    match eval_rank4_fm_core(features, v_factors) {
        Ok(res) => {
            let out = unsafe { slice_from_raw_mut(out_4_ptr, 4) };
            out.copy_from_slice(&res);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo(cos_phi: f64, out_4_ptr: *mut f64) {
    let res = eval_chebyshev_lfo_core(cos_phi);
    let out = unsafe { slice_from_raw_mut(out_4_ptr, 4) };
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo_slice(
    cos_phi_ptr: *const f64,
    len: usize,
    out_4n_ptr: *mut f64,
) -> i32 {
    let cos_phi = unsafe { slice_from_raw(cos_phi_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_4n_ptr, 4 * len) };
    match eval_chebyshev_lfo_slice_core(cos_phi, out) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_trust_region_softclip(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_core(z, mh)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_trust_region_softclip_slice(
    z_ptr: *const f64,
    len: usize,
    out_ptr: *mut f64,
    mh: f64,
) -> i32 {
    let z = unsafe { slice_from_raw(z_ptr, len) };
    let out = unsafe { slice_from_raw_mut(out_ptr, len) };
    match eval_trust_region_softclip_slice_core(z, out, mh) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

// ============================================================================
// 5. Cryptographic Provenance & Watermarking
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_immutable_header_sha256(
    header_json_ptr: *const u8,
    header_json_len: usize,
) -> i32 {
    let header_json = match unsafe { str_from_raw(header_json_ptr, header_json_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    match immutable_header_sha256_core(header_json) {
        Ok(hex) => set_result(hex.into_bytes()),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_sha256_hex(data_ptr: *const u8, data_len: usize) -> i32 {
    let data = unsafe { slice_from_raw(data_ptr, data_len) };
    let hex = sha256_hex_core(data);
    set_result(hex.into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_creator_signing_message(
    hdr_sha_ptr: *const u8,
    hdr_sha_len: usize,
    arena_sha_ptr: *const u8,
    arena_sha_len: usize,
) -> i32 {
    let hdr_sha = match unsafe { str_from_raw(hdr_sha_ptr, hdr_sha_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    let arena_sha = match unsafe { str_from_raw(arena_sha_ptr, arena_sha_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    match build_creator_signing_message_core(hdr_sha, arena_sha) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_build_marketplace_signing_message(
    hdr_sha_ptr: *const u8,
    hdr_sha_len: usize,
    arena_sha_ptr: *const u8,
    arena_sha_len: usize,
    receipt_json_ptr: *const u8,
    receipt_json_len: usize,
) -> i32 {
    let hdr_sha = match unsafe { str_from_raw(hdr_sha_ptr, hdr_sha_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    let arena_sha = match unsafe { str_from_raw(arena_sha_ptr, arena_sha_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    let receipt_json = match unsafe { str_from_raw(receipt_json_ptr, receipt_json_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    match build_marketplace_signing_message_core(hdr_sha, arena_sha, receipt_json) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_chacha20_xor_arena(
    arena_ptr: *const u8,
    arena_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    nonce_ptr: *const u8,
    nonce_len: usize,
    start_block_counter: u32,
) -> i32 {
    let arena = unsafe { slice_from_raw(arena_ptr, arena_len) };
    let key = unsafe { slice_from_raw(key_ptr, key_len) };
    let nonce = unsafe { slice_from_raw(nonce_ptr, nonce_len) };
    match chacha20_xor_arena_core(arena, key, nonce, start_block_counter) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_isometric_gauge_watermark(
    arena_ptr: *const u8,
    arena_len: usize,
    kappa_ptr: *const f32,
    kappa_len: usize,
    q_ptr: *const f32,
    q_len: usize,
) -> i32 {
    let arena = unsafe { slice_from_raw(arena_ptr, arena_len) };
    let kappa = unsafe { slice_from_raw(kappa_ptr, kappa_len) };
    let q = unsafe { slice_from_raw(q_ptr, q_len) };
    match apply_isometric_gauge_watermark_core(arena, kappa, q) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_generate_tardos_code(
    secret_ptr: *const u8,
    secret_len: usize,
    model_id_ptr: *const u8,
    model_id_len: usize,
    license_id_ptr: *const u8,
    license_id_len: usize,
    out_4608_ptr: *mut f32,
) -> i32 {
    let secret = unsafe { slice_from_raw(secret_ptr, secret_len) };
    let model_id = match unsafe { str_from_raw(model_id_ptr, model_id_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    let license_id = match unsafe { str_from_raw(license_id_ptr, license_id_len) } {
        Ok(s) => s,
        Err(e) => return set_error(e),
    };
    match generate_tardos_code_core(secret, model_id, license_id) {
        Ok(code) => {
            let out = unsafe { slice_from_raw_mut(out_4608_ptr, code.len()) };
            out.copy_from_slice(&code);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_apply_tardos_watermark(
    arena_ptr: *const u8,
    arena_len: usize,
    code_ptr: *const f32,
    code_len: usize,
    epsilon: f32,
) -> i32 {
    let arena = unsafe { slice_from_raw(arena_ptr, arena_len) };
    let code = unsafe { slice_from_raw(code_ptr, code_len) };
    match apply_tardos_watermark_core(arena, code, epsilon) {
        Ok(bytes) => set_result(bytes),
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_score_tardos_watermark(
    master_ptr: *const u8,
    master_len: usize,
    suspect_ptr: *const u8,
    suspect_len: usize,
    code_ptr: *const f32,
    code_len: usize,
    out_score_ptr: *mut f64,
) -> i32 {
    let master = unsafe { slice_from_raw(master_ptr, master_len) };
    let suspect = unsafe { slice_from_raw(suspect_ptr, suspect_len) };
    let code = unsafe { slice_from_raw(code_ptr, code_len) };
    match score_tardos_watermark_core(master, suspect, code) {
        Ok(score) => {
            unsafe {
                *out_score_ptr = score;
            }
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_extract_structural_hash(
    arena_ptr: *const u8,
    arena_len: usize,
    out_160_ptr: *mut f64,
) -> i32 {
    let arena = unsafe { slice_from_raw(arena_ptr, arena_len) };
    match extract_structural_hash_core(arena) {
        Ok(vec160) => {
            let out = unsafe { slice_from_raw_mut(out_160_ptr, vec160.len()) };
            out.copy_from_slice(&vec160);
            0
        }
        Err(e) => set_error(e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_structural_cosine_similarity(
    a_ptr: *const f64,
    a_len: usize,
    b_ptr: *const f64,
    b_len: usize,
    out_cos_ptr: *mut f64,
) -> i32 {
    let a = unsafe { slice_from_raw(a_ptr, a_len) };
    let b = unsafe { slice_from_raw(b_ptr, b_len) };
    match structural_cosine_similarity_core(a, b) {
        Ok(cos) => {
            unsafe {
                *out_cos_ptr = cos;
            }
            0
        }
        Err(e) => set_error(e),
    }
}
