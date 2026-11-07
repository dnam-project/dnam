//! WASM `EngineCore` lifecycle and zero-copy real-time audio processing exports
//! (`ref[sdk.ts.engine.zero-copy-and-async]`, `ref[sdk.ts.runtime.zero-unsafe]`,
//! `ref[sdk.ts.web-audio.realtime-bridge]`).

use crate::core::engine::{
    EngineCore, TELEMETRY_F64_LEN, backend_kind_code, backend_kind_from_code,
};
use crate::error::validation_err;
use crate::wasm::abi::{
    insert_engine, read_bytes, read_f32_vec, read_opt_str, read_str, read_u32_vec, remove_engine,
    set_error, with_engine, with_engine_mut, with_f32_1_in_1_out, with_f32_1_in_2_out,
    with_f32_2_in_2_out, with_f32_2_mut, with_f32_mut, with_f64_mut, write_f32,
};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_create(
    container_ptr: usize,
    container_len: usize,
    tier: u8,
    sample_rate_hz: u32,
    backend_code: u8,
    max_supported_tier: i32,
    offline_bounce: u8,
    maximize_tier_on_bounce: u8,
) -> u32 {
    let backend = match backend_kind_from_code(backend_code) {
        Ok(b) => b,
        Err(msg) => {
            set_error(msg);
            return 0;
        }
    };
    let bytes = read_bytes(container_ptr, container_len);
    let max_tier = if max_supported_tier >= 0 {
        Some(max_supported_tier as u8)
    } else {
        None
    };
    match EngineCore::new(
        &bytes,
        tier,
        sample_rate_hz,
        backend,
        max_tier,
        offline_bounce != 0,
        maximize_tier_on_bounce != 0,
    ) {
        Ok(engine) => insert_engine(engine),
        Err(msg) => {
            set_error(msg);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_free(handle: u32) {
    remove_engine(handle);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reload(
    handle: u32,
    container_ptr: usize,
    container_len: usize,
    tier: i32,
    sample_rate_hz: u32,
) -> i32 {
    let bytes = read_bytes(container_ptr, container_len);
    let tier_opt = if tier >= 0 { Some(tier as u8) } else { None };
    let sr_opt = if sample_rate_hz > 0 {
        Some(sample_rate_hz)
    } else {
        None
    };
    let Some(res) = with_engine_mut(handle, |engine| engine.reload(&bytes, tier_opt, sr_opt))
    else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_backend_code(handle: u32) -> u8 {
    with_engine(handle, |engine| backend_kind_code(engine.backend_kind())).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_tier(handle: u32) -> u8 {
    with_engine(handle, EngineCore::tier).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_tier(handle: u32, tier: u8) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| engine.set_tier(tier)) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_max_supported_tier(handle: u32) -> u8 {
    with_engine(handle, EngineCore::max_supported_tier).unwrap_or(3)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_max_supported_tier(handle: u32, tier: u8) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| engine.set_max_supported_tier(tier)) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_max_available_tier(handle: u32) -> u8 {
    with_engine(handle, EngineCore::max_available_tier).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_effective_tier(handle: u32) -> u8 {
    with_engine(handle, EngineCore::effective_tier).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_sample_rate(handle: u32) -> u32 {
    with_engine(handle, EngineCore::sample_rate_hz).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_sample_rate(handle: u32, sample_rate_hz: u32) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| engine.set_sample_rate(sample_rate_hz)) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_latency_samples(handle: u32) -> u32 {
    with_engine(handle, |engine| engine.latency_samples() as u32).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_offline_bounce(handle: u32) -> u8 {
    with_engine(handle, |engine| u8::from(engine.offline_bounce())).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_offline_bounce(handle: u32, enabled: u8) {
    let _ = with_engine_mut(handle, |engine| engine.set_offline_bounce(enabled != 0));
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_maximize_tier_on_bounce(handle: u32) -> u8 {
    with_engine(handle, |engine| u8::from(engine.maximize_tier_on_bounce())).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_maximize_tier_on_bounce(handle: u32, enabled: u8) {
    let _ = with_engine_mut(handle, |engine| {
        engine.set_maximize_tier_on_bounce(enabled != 0);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_num_controls(handle: u32) -> u32 {
    with_engine(handle, |engine| engine.num_controls() as u32).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_raw_controls_into(
    handle: u32,
    out_ptr: usize,
    len: usize,
) -> i32 {
    let Some(ok) = with_engine(handle, |engine| {
        let src = engine.raw_control_values();
        with_f32_mut(out_ptr, len, |out| {
            let n = src.len().min(out.len());
            out[..n].copy_from_slice(&src[..n]);
        })
        .is_some()
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    if ok {
        0
    } else {
        set_error("[DNAM-VALIDATION] invalid out_ptr".to_string())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_clamped_controls_into(
    handle: u32,
    out_ptr: usize,
    len: usize,
) -> i32 {
    let Some(ok) = with_engine(handle, |engine| {
        let src = engine.clamped_control_values();
        with_f32_mut(out_ptr, len, |out| {
            let n = src.len().min(out.len());
            out[..n].copy_from_slice(&src[..n]);
        })
        .is_some()
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    if ok {
        0
    } else {
        set_error("[DNAM-VALIDATION] invalid out_ptr".to_string())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_clamped_control_value(
    handle: u32,
    ctrl_idx: usize,
    out_ptr: usize,
) -> i32 {
    if out_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let Some(res) = with_engine(handle, |engine| engine.clamped_control_value(ctrl_idx)) else {
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
pub extern "C" fn dnam_wasm_engine_set_control_value(
    handle: u32,
    ctrl_idx: usize,
    value: f32,
) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| engine.set_control_value(ctrl_idx, value))
    else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_control_immediate(
    handle: u32,
    ctrl_idx: usize,
    value: f32,
) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| {
        engine.set_control_immediate(ctrl_idx, value)
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_controls_immediate_batch(
    handle: u32,
    indices_ptr: usize,
    values_ptr: usize,
    len: usize,
) -> i32 {
    let indices = read_u32_vec(indices_ptr, len);
    let values = read_f32_vec(values_ptr, len);
    let pairs: Vec<(usize, f32)> = indices
        .into_iter()
        .zip(values)
        .map(|(i, v)| (i as usize, v))
        .collect();
    let Some(res) = with_engine_mut(handle, |engine| engine.set_controls_immediate_batch(&pairs))
    else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reset_controls_to_defaults(handle: u32) {
    let _ = with_engine(handle, EngineCore::reset_controls_to_defaults);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_reset(handle: u32) {
    let _ = with_engine(handle, EngineCore::reset_to_dc_equilibrium);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_supports_freeze(handle: u32) -> u8 {
    with_engine(handle, |engine| u8::from(engine.supports_freeze())).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_is_frozen(handle: u32) -> u8 {
    with_engine(handle, |engine| u8::from(engine.is_frozen())).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_freeze(handle: u32, freeze: u8) {
    let _ = with_engine_mut(handle, |engine| engine.set_freeze(freeze != 0));
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_freeze_amount(handle: u32, alpha: f32) {
    let _ = with_engine_mut(handle, |engine| engine.set_freeze_amount(alpha));
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_bypass_state(handle: u32) -> u8 {
    with_engine(handle, |engine| {
        let (b, t) = engine.bypass_state();
        u8::from(b) | (u8::from(t) << 1)
    })
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_bypass(handle: u32, bypassed: u8, trails: u8) {
    let _ = with_engine_mut(handle, |engine| {
        engine.set_bypass(bypassed != 0, trails != 0);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_stereo_width(handle: u32) -> f32 {
    with_engine(handle, EngineCore::stereo_width).unwrap_or(1.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_set_stereo_width(handle: u32, width: f32) {
    let _ = with_engine_mut(handle, |engine| engine.set_stereo_width(width));
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_configure_host_mix(
    handle: u32,
    blend_wet: f32,
    output_gain_db: f32,
    apply_unity_leveling: u8,
) {
    let _ = with_engine_mut(handle, |engine| {
        engine.configure_host_mix_and_leveling(
            blend_wet,
            output_gain_db,
            apply_unity_leveling != 0,
        );
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_current_bpm(handle: u32) -> f32 {
    with_engine(handle, EngineCore::current_bpm).unwrap_or(120.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_sync_tempo(
    handle: u32,
    bpm: f32,
    sub_ptr: usize,
    sub_len: usize,
    fold_octave: u8,
) -> i32 {
    let sub_opt = match read_opt_str(sub_ptr, sub_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let Some(res) = with_engine_mut(handle, |engine| {
        engine.sync_tempo(bpm, sub_opt.as_deref(), fold_octave != 0)
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(c) => c as i32,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_register_tap_tempo(
    handle: u32,
    sub_ptr: usize,
    sub_len: usize,
    fold_octave: u8,
    out_bpm_ptr: usize,
) -> i32 {
    if out_bpm_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    let sub_opt = match read_opt_str(sub_ptr, sub_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let Some(res) = with_engine_mut(handle, |engine| {
        engine.register_tap_tempo(sub_opt.as_deref(), fold_octave != 0)
    }) else {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    };
    match res {
        Ok((bpm, count)) => {
            if !write_f32(out_bpm_ptr, bpm) {
                return set_error("[DNAM-VALIDATION] invalid out_bpm_ptr".to_string());
            }
            count as i32
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_apply_preset(
    handle: u32,
    preset_ptr: usize,
    preset_len: usize,
) -> i32 {
    let preset_id = match read_str(preset_ptr, preset_len) {
        Ok(s) => s,
        Err(msg) => return set_error(msg),
    };
    let Some(res) = with_engine_mut(handle, |engine| engine.apply_preset(&preset_id)) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_stereo_in_place(
    handle: u32,
    left_io_ptr: usize,
    right_io_ptr: usize,
    num_samples: usize,
) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| {
        with_f32_2_mut(
            left_io_ptr,
            right_io_ptr,
            num_samples,
            |left_io, right_io| engine.process_stereo_in_place(left_io, right_io),
        )
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    let Some(res) = res else {
        return set_error("[DNAM-VALIDATION] invalid audio buffer pointers".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_stereo(
    handle: u32,
    left_in_ptr: usize,
    right_in_ptr: usize,
    left_out_ptr: usize,
    right_out_ptr: usize,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    let Some(res) = with_engine_mut(handle, |engine| {
        with_f32_2_in_2_out(
            left_in_ptr,
            right_in_ptr,
            left_out_ptr,
            right_out_ptr,
            num_samples,
            |left_in, right_in, left_out, right_out| {
                engine.process_stereo(left_in, right_in, left_out, right_out, bs)
            },
        )
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    let Some(res) = res else {
        return set_error("[DNAM-VALIDATION] invalid audio buffer pointers".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono_to_stereo(
    handle: u32,
    mono_in_ptr: usize,
    left_out_ptr: usize,
    right_out_ptr: usize,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    let Some(res) = with_engine_mut(handle, |engine| {
        with_f32_1_in_2_out(
            mono_in_ptr,
            left_out_ptr,
            right_out_ptr,
            num_samples,
            |mono_in, left_out, right_out| {
                engine.process_mono_to_stereo(mono_in, left_out, right_out, bs)
            },
        )
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    let Some(res) = res else {
        return set_error("[DNAM-VALIDATION] invalid audio buffer pointers".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_process_mono(
    handle: u32,
    mono_in_ptr: usize,
    mono_out_ptr: usize,
    num_samples: usize,
    block_size: usize,
) -> i32 {
    let bs = if block_size > 0 {
        Some(block_size)
    } else {
        None
    };
    let Some(res) = with_engine_mut(handle, |engine| {
        with_f32_1_in_1_out(
            mono_in_ptr,
            mono_out_ptr,
            num_samples,
            |mono_in, mono_out| engine.process_mono(mono_in, mono_out, bs),
        )
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    let Some(res) = res else {
        return set_error("[DNAM-VALIDATION] invalid audio buffer pointers".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_render_offline_bounce(
    handle: u32,
    left_in_ptr: usize,
    right_in_ptr: usize,
    left_out_ptr: usize,
    right_out_ptr: usize,
    num_samples: usize,
) -> i32 {
    let Some(res) = with_engine_mut(handle, |engine| {
        with_f32_2_in_2_out(
            left_in_ptr,
            right_in_ptr,
            left_out_ptr,
            right_out_ptr,
            num_samples,
            |left_in, right_in, left_out, right_out| {
                engine.render_offline_bounce(left_in, right_in, left_out, right_out)
            },
        )
    }) else {
        return set_error("[DNAM-VALIDATION] null EngineCore pointer".to_string());
    };
    let Some(res) = res else {
        return set_error("[DNAM-VALIDATION] invalid audio buffer pointers".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_engine_telemetry_into(
    handle: u32,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    if out_ptr == 0 {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    }
    if out_len < TELEMETRY_F64_LEN {
        return set_error(validation_err(format!(
            "telemetry_into requires {TELEMETRY_F64_LEN} f64 slots, got {out_len}"
        )));
    }
    let Some(ok) = with_engine(handle, |engine| {
        with_f64_mut(out_ptr, TELEMETRY_F64_LEN, |out_slice| {
            let Ok(out_arr) = <&mut [f64; TELEMETRY_F64_LEN]>::try_from(out_slice) else {
                return false;
            };
            engine.telemetry_into_f64(out_arr);
            true
        })
        .unwrap_or(false)
    }) else {
        return set_error("[DNAM-VALIDATION] null pointer".to_string());
    };
    if ok {
        0
    } else {
        set_error(validation_err("telemetry_into length mismatch"))
    }
}
