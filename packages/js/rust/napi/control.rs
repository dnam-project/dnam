//! Node-API v10 `NativeControlSpace`, ZOH rescaling, and tempo calibration inversion bindings
//! (`ref[sdk.ts.parameters.audio-param-semantics]`).

use napi::bindgen_prelude::{Float32Array, Float64Array, Result};
use napi_derive::napi;

use crate::core::control::{
    ControlSpaceCore, compute_zoh_core, control_stride_for_rate_core, control_tick_ratio_core,
    eval_3node_calibration_core, eval_subdivision_phase_core, invert_tempo_calibration_core,
};
use crate::error::validation_err;
use crate::napi::{f32_slice_mut, napi_err};

#[napi]
#[derive(Debug, Clone)]
pub struct NativeControlSpace {
    inner: ControlSpaceCore,
}

#[napi]
impl NativeControlSpace {
    #[napi(constructor)]
    pub fn new(header_json: String) -> Result<Self> {
        let inner = ControlSpaceCore::new(&header_json).map_err(napi_err)?;
        Ok(Self { inner })
    }

    #[napi(getter)]
    pub fn num_controls(&self) -> u32 {
        self.inner.num_controls() as u32
    }

    #[napi]
    pub fn clamp_into(
        &self,
        raw_numeric_values: Float32Array,
        mut clamped_out: Float32Array,
        mut norm_out: Float32Array,
    ) -> u32 {
        let c_out = f32_slice_mut(&mut clamped_out);
        let n_out = f32_slice_mut(&mut norm_out);
        self.inner.clamp_into(&raw_numeric_values, c_out, n_out)
    }

    #[napi]
    pub fn to_physical(&self, ctrl_idx: u32, normalized_u: f64) -> Result<f64> {
        self.inner
            .to_physical(ctrl_idx as usize, normalized_u as f32)
            .map(f64::from)
            .map_err(napi_err)
    }

    #[napi]
    pub fn to_normalized(&self, ctrl_idx: u32, physical_val: f64) -> Result<f64> {
        self.inner
            .to_normalized(ctrl_idx as usize, physical_val as f32)
            .map(f64::from)
            .map_err(napi_err)
    }
}

#[napi]
pub fn compute_zoh_native(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    r_ac_train: Option<f64>,
    r_dc_train: Option<f64>,
    tau_ref_train: Option<f64>,
    switch_xfade_train: Option<u32>,
) -> Result<Float64Array> {
    let arr = compute_zoh_core(
        sample_rate_train_hz,
        sample_rate_host_hz,
        r_ac_train.unwrap_or(0.9993) as f32,
        r_dc_train.unwrap_or(0.995) as f32,
        tau_ref_train.unwrap_or(64.0) as f32,
        switch_xfade_train.unwrap_or(128),
    )
    .map_err(napi_err)?;
    Ok(Float64Array::new(arr.to_vec()))
}

#[napi]
pub fn control_stride_for_rate_native(sample_rate_hz: u32) -> u32 {
    control_stride_for_rate_core(sample_rate_hz)
}

#[napi]
pub fn control_tick_ratio_native(sample_rate_hz: u32) -> f64 {
    control_tick_ratio_core(sample_rate_hz)
}

#[napi]
pub fn eval_subdivision_phase_native(
    beat_mod12: u8,
    within_beat_q32: u32,
    subdivision: String,
    offset_q32: Option<u32>,
) -> Result<Vec<f64>> {
    let (q32, cos_phi, sin_phi) = eval_subdivision_phase_core(
        beat_mod12,
        within_beat_q32,
        &subdivision,
        offset_q32.unwrap_or(0),
    )
    .map_err(napi_err)?;
    Ok(vec![f64::from(q32), f64::from(cos_phi), f64::from(sin_phi)])
}

#[napi]
pub fn eval_3node_calibration_native(anchors: Float64Array, u: f64) -> Result<f64> {
    let anc: &[f64; 3] = (&anchors[..]).try_into().map_err(|_| {
        napi_err(validation_err(format!(
            "anchors must have 3 elements, got {}",
            anchors.len()
        )))
    })?;
    Ok(eval_3node_calibration_core(*anc, u))
}

#[napi]
pub fn invert_tempo_calibration_native(
    declared_min: f64,
    declared_max: f64,
    safe_min: f64,
    safe_max: f64,
    anchors: Float64Array,
    target: f64,
    flush_subnormals: Option<bool>,
) -> Result<Float64Array> {
    let anc: &[f64; 3] = (&anchors[..]).try_into().map_err(|_| {
        napi_err(validation_err(format!(
            "anchors must have 3 elements, got {}",
            anchors.len()
        )))
    })?;
    let res = invert_tempo_calibration_core(
        (declared_min as f32, declared_max as f32),
        (safe_min as f32, safe_max as f32),
        *anc,
        target,
        flush_subnormals.unwrap_or(true),
    )
    .map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}
