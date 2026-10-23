//! Safe Node-API v8 (`napi-rs` v3) bindings (`#[deny(unsafe_code)]`) for host targets.
#![allow(missing_debug_implementations)]

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

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
use crate::engine::{EngineCore, backend_kind_str, parse_backend_kind};
use crate::provenance::{
    apply_isometric_gauge_watermark_core, apply_tardos_watermark_core,
    build_creator_signing_message_core, build_marketplace_signing_message_core,
    chacha20_xor_arena_core, extract_structural_hash_core, generate_tardos_code_core,
    immutable_header_sha256_core, score_tardos_watermark_core, sha256_hex_core,
    structural_cosine_similarity_core,
};

fn to_napi_err(msg: String) -> Error {
    Error::from_reason(msg)
}

#[allow(unsafe_code)]
#[inline]
fn f32_slice_mut(arr: &mut Float32Array) -> &mut [f32] {
    // SAFETY: Synchronous N-API call on the JS thread; the output TypedArray is mutated in-place.
    unsafe { arr.as_mut() }
}

#[allow(unsafe_code)]
#[inline]
fn f64_slice_mut(arr: &mut Float64Array) -> &mut [f64] {
    // SAFETY: Synchronous N-API call on the JS thread; the output TypedArray is mutated in-place.
    unsafe { arr.as_mut() }
}

/// Telemetry & ZOH snapshot returned to JavaScript.
#[napi(object)]
pub struct NativeTelemetryData {
    pub active_tier: u8,
    pub backend: String,
    pub sample_rate_hz: u32,
    pub core2_active: bool,
    pub lfo_env_active: bool,
    pub stereo_out: bool,
    pub active_clamp_mask: u16,
    pub control_decimation_host: u16,
    pub max_adaa_drive: f64,
    pub max_pole_radius: f64,
    pub max_effective_loop_gain: f64,
    pub tail_time_ms: f64,
    pub z_head: Vec<f64>,
    pub env_states: Vec<f64>,
    pub lfo_harmonics: Vec<f64>,
    pub clamped_controls: Vec<f64>,
    pub rho_sr: f64,
    pub rho_ctrl: f64,
    pub r_ac_host: f64,
    pub r_dc_host: f64,
    pub tau_ref_samples_host: f64,
    pub switch_xfade_samples_host: u32,
}

/// Background worker task for `DnamEngine.processAsync(...)` executing on the libuv worker pool.
pub struct ProcessAsyncTask {
    engine: Arc<Mutex<EngineCore>>,
    left_in: Vec<f32>,
    right_in: Option<Vec<f32>>,
    buffer_size: Option<usize>,
}

#[napi]
impl Task for ProcessAsyncTask {
    type Output = (Vec<f32>, Vec<f32>);
    type JsValue = Vec<Float32Array>;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut guard = self
            .engine
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        let n = self.left_in.len();
        let mut left_out = vec![0.0_f32; n];
        let mut right_out = vec![0.0_f32; n];
        if let Some(ref right_in) = self.right_in {
            guard
                .process_stereo(
                    &self.left_in,
                    right_in,
                    &mut left_out,
                    &mut right_out,
                    self.buffer_size,
                )
                .map_err(to_napi_err)?;
        } else {
            guard
                .process_mono_to_stereo(
                    &self.left_in,
                    &mut left_out,
                    &mut right_out,
                    self.buffer_size,
                )
                .map_err(to_napi_err)?;
        }
        Ok((left_out, right_out))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(vec![
            Float32Array::new(output.0),
            Float32Array::new(output.1),
        ])
    }
}

/// Native N-API wrapper around [`EngineCore`].
#[napi]
pub struct NativeDnamEngine {
    inner: Arc<Mutex<EngineCore>>,
}

#[napi]
impl NativeDnamEngine {
    #[napi(constructor)]
    pub fn new(
        container_bytes: &[u8],
        tier: u8,
        sample_rate_hz: u32,
        backend: Option<String>,
    ) -> Result<Self> {
        let backend_str = backend.as_deref().unwrap_or("auto");
        let backend_kind = parse_backend_kind(backend_str).map_err(to_napi_err)?;
        let core = EngineCore::new(container_bytes, tier, sample_rate_hz, backend_kind)
            .map_err(to_napi_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(core)),
        })
    }

    #[napi]
    pub fn reload(
        &self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard
            .reload(container_bytes, tier, sample_rate_hz)
            .map_err(to_napi_err)
    }

    #[napi(getter)]
    pub fn backend(&self) -> Result<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        Ok(backend_kind_str(guard.backend_kind()).to_string())
    }

    #[napi(getter)]
    pub fn active_tier(&self) -> Result<u8> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        Ok(guard.active_tier())
    }

    #[napi(setter)]
    pub fn set_active_tier(&self, tier: u8) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard.set_tier(tier).map_err(to_napi_err)
    }

    #[napi(getter)]
    pub fn sample_rate_hz(&self) -> Result<u32> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        Ok(guard.sample_rate_hz())
    }

    #[napi(setter)]
    pub fn set_sample_rate_hz(&self, sample_rate_hz: u32) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard.set_sample_rate(sample_rate_hz).map_err(to_napi_err)
    }

    #[napi]
    pub fn set_control_value(&self, ctrl_idx: u32, value: f64) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard.set_control_value(ctrl_idx as usize, value as f32);
        Ok(())
    }

    #[napi]
    pub fn set_control_immediate(&self, ctrl_idx: u32, value: f64) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard.set_control_immediate(ctrl_idx as usize, value as f32);
        Ok(())
    }

    #[napi]
    pub fn reset(&self) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard.reset();
        Ok(())
    }

    #[napi]
    pub fn process_stereo(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        mut left_out: Float32Array,
        mut right_out: Float32Array,
        buffer_size: Option<u32>,
    ) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard
            .process_stereo(
                left_in,
                right_in,
                f32_slice_mut(&mut left_out),
                f32_slice_mut(&mut right_out),
                buffer_size.map(|b| b as usize),
            )
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn process_mono_to_stereo(
        &self,
        mono_in: &[f32],
        mut left_out: Float32Array,
        mut right_out: Float32Array,
        buffer_size: Option<u32>,
    ) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard
            .process_mono_to_stereo(
                mono_in,
                f32_slice_mut(&mut left_out),
                f32_slice_mut(&mut right_out),
                buffer_size.map(|b| b as usize),
            )
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn process_mono(
        &self,
        mono_in: &[f32],
        mut mono_out: Float32Array,
        buffer_size: Option<u32>,
    ) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        guard
            .process_mono(
                mono_in,
                f32_slice_mut(&mut mono_out),
                buffer_size.map(|b| b as usize),
            )
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn process_async(
        &self,
        left_in: &[f32],
        right_in: Option<&[f32]>,
        buffer_size: Option<u32>,
    ) -> AsyncTask<ProcessAsyncTask> {
        AsyncTask::new(ProcessAsyncTask {
            engine: Arc::clone(&self.inner),
            left_in: left_in.to_vec(),
            right_in: right_in.map(<[f32]>::to_vec),
            buffer_size: buffer_size.map(|b| b as usize),
        })
    }

    #[napi]
    pub fn telemetry(&self) -> Result<NativeTelemetryData> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        let t = guard.telemetry().map_err(to_napi_err)?;
        Ok(NativeTelemetryData {
            active_tier: t.active_tier,
            backend: t.backend.to_string(),
            sample_rate_hz: t.sample_rate_hz,
            core2_active: t.core2_active,
            lfo_env_active: t.lfo_env_active,
            stereo_out: t.stereo_out,
            active_clamp_mask: t.active_clamp_mask,
            control_decimation_host: t.control_decimation_host,
            max_adaa_drive: f64::from(t.max_adaa_drive),
            max_pole_radius: f64::from(t.max_pole_radius),
            max_effective_loop_gain: f64::from(t.max_effective_loop_gain),
            tail_time_ms: f64::from(t.tail_time_ms),
            z_head: t.z_head.iter().map(|&v| f64::from(v)).collect(),
            env_states: t.env_states.iter().map(|&v| f64::from(v)).collect(),
            lfo_harmonics: t.lfo_harmonics.iter().map(|&v| f64::from(v)).collect(),
            clamped_controls: t.clamped_controls.iter().map(|&v| f64::from(v)).collect(),
            rho_sr: t.rho_sr,
            rho_ctrl: t.rho_ctrl,
            r_ac_host: f64::from(t.r_ac_host),
            r_dc_host: f64::from(t.r_dc_host),
            tau_ref_samples_host: f64::from(t.tau_ref_samples_host),
            switch_xfade_samples_host: t.switch_xfade_samples_host,
        })
    }

    #[napi]
    pub fn telemetry_into(&self, mut out: Float64Array) -> Result<()> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| to_napi_err("[DNAM-VALIDATION] Engine mutex poisoned".to_string()))?;
        let slice = f64_slice_mut(&mut out);
        let arr: &mut [f64; 50] = slice
            .get_mut(..50)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| {
                to_napi_err(
                    "[DNAM-VALIDATION] telemetry_into requires >= 50 f64 elements".to_string(),
                )
            })?;
        guard.telemetry_into_f64(arr);
        Ok(())
    }
}

/// Result of `NativeControlSpace::clamp`.
#[napi(object)]
pub struct NativeClampResult {
    pub clamped_numeric: Float32Array,
    pub normalized_u: Float32Array,
    pub was_clamped: Vec<bool>,
}

/// Pre-compiled 3-Phase control clamping space (`< 100 ns` evaluation).
#[napi]
pub struct NativeControlSpace {
    inner: ControlSpaceCore,
}

#[napi]
impl NativeControlSpace {
    #[napi(constructor)]
    pub fn new(header_json: String) -> Result<Self> {
        let inner = ControlSpaceCore::new(&header_json).map_err(to_napi_err)?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn clamp(&self, raw_numeric_values: &[f32]) -> NativeClampResult {
        let (clamped, norm, was_clamped) = self.inner.clamp(raw_numeric_values);
        NativeClampResult {
            clamped_numeric: Float32Array::new(clamped),
            normalized_u: Float32Array::new(norm),
            was_clamped,
        }
    }

    #[napi]
    pub fn clamp_into(
        &self,
        raw_numeric_values: &[f32],
        mut clamped_out: Float32Array,
        mut norm_out: Float32Array,
    ) -> u32 {
        self.inner.clamp_into(
            raw_numeric_values,
            f32_slice_mut(&mut clamped_out),
            f32_slice_mut(&mut norm_out),
        )
    }
}

/// Validated container metadata returned by `nativeValidateContainer`.
#[napi(object)]
pub struct NativeValidatedContainer {
    pub magic: String,
    pub version_major: u16,
    pub version_minor: u16,
    pub flags_bits: u16,
    pub num_controls: u8,
    pub ring_buffer_pow2_log2: u8,
    pub sample_rate_train_hz: u32,
    pub json_header_byte_len: u32,
    pub tensor_arena_offset: u32,
    pub tensor_arena_byte_len: u32,
    pub arena_offset_lite_end: u32,
    pub arena_offset_full_end: u32,
    pub arena_offset_ultra_end: u32,
    pub arena_crc32c_lite: u32,
    pub arena_crc32c_full: u32,
    pub arena_crc32c_ultra: u32,
    pub num_ckpt_stems: u16,
    pub control_decimation: u8,
    pub head_svd_rank: u8,
    pub tail_time_ms: f64,
    pub header_crc32c: u32,
    pub core2_active: bool,
    pub lfo_env_active: bool,
    pub stereo_out: bool,
    pub verification_status: String,
    pub header_json: String,
    pub arena_bytes: Uint8Array,
}

#[napi]
pub fn native_validate_container(container_bytes: &[u8]) -> Result<NativeValidatedContainer> {
    let info = validate_container_core(container_bytes).map_err(to_napi_err)?;
    let p = info.preamble;
    Ok(NativeValidatedContainer {
        magic: info.magic_str,
        version_major: p.version_major,
        version_minor: p.version_minor,
        flags_bits: p.flags,
        num_controls: p.num_controls,
        ring_buffer_pow2_log2: p.ring_buffer_pow2_log2,
        sample_rate_train_hz: p.sample_rate_train_hz,
        json_header_byte_len: p.json_header_byte_len,
        tensor_arena_offset: p.tensor_arena_offset,
        tensor_arena_byte_len: p.tensor_arena_byte_len,
        arena_offset_lite_end: p.arena_offset_lite_end,
        arena_offset_full_end: p.arena_offset_full_end,
        arena_offset_ultra_end: p.arena_offset_ultra_end,
        arena_crc32c_lite: p.arena_crc32c_lite,
        arena_crc32c_full: p.arena_crc32c_full,
        arena_crc32c_ultra: p.arena_crc32c_ultra,
        num_ckpt_stems: p.num_ckpt_stems,
        control_decimation: p.control_decimation,
        head_svd_rank: p.head_svd_rank,
        tail_time_ms: f64::from(p.tail_time_ms),
        header_crc32c: p.header_crc32c,
        core2_active: p.is_core2_active(),
        lfo_env_active: p.is_lfo_env_active(),
        stereo_out: p.is_stereo_out(),
        verification_status: info.verification_status,
        header_json: info.header_json,
        arena_bytes: Uint8Array::new(info.arena_bytes),
    })
}

#[napi]
pub fn native_build_container(
    header_json: String,
    arena_bytes: &[u8],
    plaintext_arena_crc_source: Option<&[u8]>,
    encrypted_arena: bool,
) -> Result<Uint8Array> {
    let bytes = build_container_core(
        &header_json,
        arena_bytes,
        plaintext_arena_crc_source,
        encrypted_arena,
    )
    .map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_build_reference_container(
    core2_active: bool,
    is_checkpoint: bool,
) -> Result<Uint8Array> {
    let bytes = build_reference_container_core(core2_active, is_checkpoint).map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_update_header_in_place(
    container_bytes: &[u8],
    updated_header_json: String,
) -> Result<Uint8Array> {
    let bytes =
        update_header_in_place_core(container_bytes, &updated_header_json).map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi(object)]
pub struct NativeZohParams {
    pub sample_rate_host_hz: u32,
    pub rho_sr: f64,
    pub control_decimation_host: u16,
    pub rho_ctrl: f64,
    pub r_ac_host: f64,
    pub r_dc_host: f64,
    pub tau_ref_samples_host: f64,
    pub switch_xfade_samples_host: u32,
}

#[napi]
pub fn native_compute_zoh(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    control_decimation_train: u8,
    r_ac_train: f64,
    r_dc_train: f64,
    tau_ref_train: f64,
    switch_xfade_train: u32,
) -> Result<NativeZohParams> {
    let z = compute_zoh_core(
        sample_rate_train_hz,
        sample_rate_host_hz,
        control_decimation_train,
        r_ac_train as f32,
        r_dc_train as f32,
        tau_ref_train as f32,
        switch_xfade_train,
    )
    .map_err(to_napi_err)?;
    Ok(NativeZohParams {
        sample_rate_host_hz: z.sample_rate_host_hz,
        rho_sr: z.rho_sr,
        control_decimation_host: z.control_decimation_host as u16,
        rho_ctrl: z.rho_ctrl,
        r_ac_host: f64::from(z.r_ac_host),
        r_dc_host: f64::from(z.r_dc_host),
        tau_ref_samples_host: f64::from(z.tau_ref_samples_host),
        switch_xfade_samples_host: z.switch_xfade_samples_host,
    })
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_compute_zoh_into(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    control_decimation_train: u8,
    r_ac_train: f64,
    r_dc_train: f64,
    tau_ref_train: f64,
    switch_xfade_train: u32,
    mut out: Float64Array,
) -> Result<()> {
    let z = compute_zoh_core(
        sample_rate_train_hz,
        sample_rate_host_hz,
        control_decimation_train,
        r_ac_train as f32,
        r_dc_train as f32,
        tau_ref_train as f32,
        switch_xfade_train,
    )
    .map_err(to_napi_err)?;
    let s = f64_slice_mut(&mut out);
    if s.len() >= 8 {
        s[0] = f64::from(z.sample_rate_host_hz);
        s[1] = z.rho_sr;
        s[2] = f64::from(u32::try_from(z.control_decimation_host).unwrap_or(16));
        s[3] = z.rho_ctrl;
        s[4] = f64::from(z.r_ac_host);
        s[5] = f64::from(z.r_dc_host);
        s[6] = f64::from(z.tau_ref_samples_host);
        s[7] = f64::from(z.switch_xfade_samples_host);
    }
    Ok(())
}

#[napi]
pub fn native_pack_uplo_u(dense_flat: &[f64], n_dim: u32) -> Result<Float64Array> {
    let out = pack_uplo_u_core(dense_flat, n_dim as usize).map_err(to_napi_err)?;
    Ok(Float64Array::new(out))
}

#[napi]
pub fn native_unpack_uplo_u(packed_u: &[f64], n_tier: u32) -> Result<Float64Array> {
    let out = unpack_uplo_u_core(packed_u, n_tier as usize).map_err(to_napi_err)?;
    Ok(Float64Array::new(out))
}

#[napi]
pub fn native_crc32c(data: &[u8]) -> u32 {
    crc32c_core(data)
}

#[napi]
pub fn native_poly_adaa_algebraic(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    poly_adaa_algebraic_core(u_prev, u_curr, beta, delta)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_step(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
) -> Vec<f64> {
    poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc).to_vec()
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_step_into(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    mut out: Float64Array,
) {
    let res = poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc);
    let s = f64_slice_mut(&mut out);
    if s.len() >= 4 {
        s[..4].copy_from_slice(&res);
    }
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_scan(
    inp: &[f64],
    mut out: Float64Array,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    reset: bool,
) -> Result<Vec<f64>> {
    let next_state = poly_adaa_scan_core(
        inp,
        f64_slice_mut(&mut out),
        b_conv,
        alpha,
        beta,
        delta,
        r_ac,
        u_prev,
        a_prev,
        m_dc,
        reset,
    )
    .map_err(to_napi_err)?;
    Ok(next_state.to_vec())
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_selective_rglru_step(
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
) -> Vec<f64> {
    selective_rglru_step_core(
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
    )
    .to_vec()
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_selective_rglru_step_into(
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
    mut out: Float64Array,
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
    let s = f64_slice_mut(&mut out);
    if s.len() >= 3 {
        s[..3].copy_from_slice(&res);
    }
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_selective_rglru_scan(
    inp: &[f64],
    mut out: Float64Array,
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
    reset: bool,
) -> Result<Vec<f64>> {
    let next_state = selective_rglru_scan_core(
        inp,
        f64_slice_mut(&mut out),
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
        reset,
    )
    .map_err(to_napi_err)?;
    Ok(next_state.to_vec())
}

#[napi]
pub fn native_unit_rotor(phi: f64) -> Vec<f64> {
    unit_rotor_core(phi as f32).to_vec()
}

#[napi]
pub fn native_unit_rotor_into(phi: f64, mut out: Float64Array) {
    let res = unit_rotor_core(phi as f32);
    let s = f64_slice_mut(&mut out);
    if s.len() >= 2 {
        s[..2].copy_from_slice(&res);
    }
}

#[napi]
pub fn native_unit_rotor_slice(phi: &[f64], mut out: Float64Array) -> Result<()> {
    unit_rotor_slice_core(phi, f64_slice_mut(&mut out)).map_err(to_napi_err)
}

#[napi]
pub fn native_cubic_lagrange_weights(mu: f64) -> Vec<f64> {
    cubic_lagrange_weights_core(mu).to_vec()
}

#[napi]
pub fn native_cubic_lagrange_weights_into(mu: f64, mut out: Float64Array) {
    let res = cubic_lagrange_weights_core(mu);
    let s = f64_slice_mut(&mut out);
    if s.len() >= 4 {
        s[..4].copy_from_slice(&res);
    }
}

#[napi]
pub fn native_givens_householder_mix_8ch(
    x: &[f64],
    cos_psi: f64,
    sin_psi: f64,
) -> Result<Float64Array> {
    let out = givens_householder_mix_8ch_core(x, cos_psi, sin_psi).map_err(to_napi_err)?;
    Ok(Float64Array::new(out.to_vec()))
}

#[napi]
pub fn native_mufdn_coercive_step(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: f64,
    dc_y_prev: f64,
) -> Vec<f64> {
    mufdn_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev).to_vec()
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_mufdn_coercive_step_into(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: f64,
    dc_y_prev: f64,
    mut out: Float64Array,
) {
    let res = mufdn_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev);
    let s = f64_slice_mut(&mut out);
    if s.len() >= 3 {
        s[..3].copy_from_slice(&res);
    }
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn native_mufdn_slice(
    inp: &[f64],
    mut out: Float64Array,
    delay_samples: u32,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> Result<()> {
    mufdn_slice_core(
        inp,
        f64_slice_mut(&mut out),
        delay_samples as usize,
        g_lin,
        sigma_knee,
        v_ceil,
        r_dc,
    )
    .map_err(to_napi_err)
}

#[napi]
pub fn native_eval_lobatto_basis(u: f64, anchors: &[f64], degree: u32) -> Result<Vec<f64>> {
    let out = eval_lobatto_basis_core(u, anchors, degree as usize).map_err(to_napi_err)?;
    Ok(out.to_vec())
}

#[napi]
pub fn native_eval_lobatto_basis_into(
    u: f64,
    anchors: &[f64],
    degree: u32,
    mut out: Float64Array,
) -> Result<()> {
    let res = eval_lobatto_basis_core(u, anchors, degree as usize).map_err(to_napi_err)?;
    let s = f64_slice_mut(&mut out);
    if s.len() >= 3 {
        s[..3].copy_from_slice(&res);
    }
    Ok(())
}

#[napi]
pub fn native_eval_lobatto_basis_slice(
    u: &[f64],
    mut out: Float64Array,
    anchors: &[f64],
    degree: u32,
) -> Result<()> {
    eval_lobatto_basis_slice_core(u, f64_slice_mut(&mut out), anchors, degree as usize)
        .map_err(to_napi_err)
}

#[napi]
pub fn native_eval_rank4_fm(features: &[f64], v_factors_flat: &[f64]) -> Result<Vec<f64>> {
    let out = eval_rank4_fm_core(features, v_factors_flat).map_err(to_napi_err)?;
    Ok(out.to_vec())
}

#[napi]
pub fn native_eval_chebyshev_lfo(cos_phi: f64) -> Vec<f64> {
    eval_chebyshev_lfo_core(cos_phi).to_vec()
}

#[napi]
pub fn native_eval_chebyshev_lfo_into(cos_phi: f64, mut out: Float64Array) {
    let res = eval_chebyshev_lfo_core(cos_phi);
    let s = f64_slice_mut(&mut out);
    if s.len() >= 4 {
        s[..4].copy_from_slice(&res);
    }
}

#[napi]
pub fn native_eval_chebyshev_lfo_slice(cos_phi: &[f64], mut out: Float64Array) -> Result<()> {
    eval_chebyshev_lfo_slice_core(cos_phi, f64_slice_mut(&mut out)).map_err(to_napi_err)
}

#[napi]
pub fn native_eval_trust_region_softclip(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_core(z, mh)
}

#[napi]
pub fn native_eval_trust_region_softclip_slice(
    z: &[f64],
    mut out: Float64Array,
    mh: f64,
) -> Result<()> {
    eval_trust_region_softclip_slice_core(z, f64_slice_mut(&mut out), mh).map_err(to_napi_err)
}

#[napi]
pub fn native_immutable_header_sha256(header_json: String) -> Result<String> {
    immutable_header_sha256_core(&header_json).map_err(to_napi_err)
}

#[napi]
pub fn native_sha256_hex(data: &[u8]) -> String {
    sha256_hex_core(data)
}

#[napi]
pub fn native_build_creator_signing_message(
    immutable_header_sha256_hex: String,
    master_arena_sha256_hex: String,
) -> Result<Uint8Array> {
    let bytes =
        build_creator_signing_message_core(&immutable_header_sha256_hex, &master_arena_sha256_hex)
            .map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_build_marketplace_signing_message(
    immutable_header_sha256_hex: String,
    master_arena_sha256_hex: String,
    receipt_json: String,
) -> Result<Uint8Array> {
    let bytes = build_marketplace_signing_message_core(
        &immutable_header_sha256_hex,
        &master_arena_sha256_hex,
        &receipt_json,
    )
    .map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_chacha20_xor_arena(
    arena_bytes: &[u8],
    key_32: &[u8],
    nonce_12: &[u8],
    start_block_counter: u32,
) -> Result<Uint8Array> {
    let bytes = chacha20_xor_arena_core(arena_bytes, key_32, nonce_12, start_block_counter)
        .map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_apply_isometric_gauge_watermark(
    arena_bytes: &[u8],
    kappa_48: &[f32],
    q_buyer_flat_16: &[f32],
) -> Result<Uint8Array> {
    let bytes = apply_isometric_gauge_watermark_core(arena_bytes, kappa_48, q_buyer_flat_16)
        .map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_generate_tardos_code(
    marketplace_secret_32: &[u8],
    model_id: String,
    license_id: String,
) -> Result<Float32Array> {
    let code = generate_tardos_code_core(marketplace_secret_32, &model_id, &license_id)
        .map_err(to_napi_err)?;
    Ok(Float32Array::new(code))
}

#[napi]
pub fn native_apply_tardos_watermark(
    arena_bytes: &[u8],
    code_4608: &[f32],
    epsilon: f64,
) -> Result<Uint8Array> {
    let bytes =
        apply_tardos_watermark_core(arena_bytes, code_4608, epsilon as f32).map_err(to_napi_err)?;
    Ok(Uint8Array::new(bytes))
}

#[napi]
pub fn native_score_tardos_watermark(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code_4608: &[f32],
) -> Result<f64> {
    score_tardos_watermark_core(master_arena, suspect_arena, code_4608).map_err(to_napi_err)
}

#[napi]
pub fn native_extract_structural_hash(arena_bytes: &[u8]) -> Result<Float64Array> {
    let vec160 = extract_structural_hash_core(arena_bytes).map_err(to_napi_err)?;
    Ok(Float64Array::new(vec160))
}

#[napi]
pub fn native_structural_cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
    structural_cosine_similarity_core(a, b).map_err(to_napi_err)
}
