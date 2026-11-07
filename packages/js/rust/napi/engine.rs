//! Node-API v10 `NativeEngine` & `StereoProcessTask` (`ref[sdk.ts.engine.zero-copy-and-async]`,
//! `ref[dsp.ctrl.op-m.spsc-ingress]`, `ref[arch.telemetry.seqlock-publication]`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::{AsyncTask, Env, Float32Array, Float64Array, Result, Task, Uint8Array};
use napi_derive::napi;

use crate::core::engine::{EngineCore, TELEMETRY_F64_LEN, backend_kind_str, parse_backend_kind};
use crate::error::{busy_err, validation_err};
use crate::napi::{f32_slice_mut, f64_slice_mut, napi_err};

#[derive(Debug)]
pub struct BusyResetGuard(Arc<AtomicBool>);

impl Drop for BusyResetGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub struct StereoProcessTask {
    core: EngineCore,
    busy_guard: BusyResetGuard,
    left_in: Vec<f32>,
    right_in: Vec<f32>,
    block_size: Option<usize>,
}

impl std::fmt::Debug for StereoProcessTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StereoProcessTask")
            .field("frames", &self.left_in.len())
            .field("block_size", &self.block_size)
            .finish()
    }
}

#[napi]
impl Task for StereoProcessTask {
    type Output = (Vec<f32>, Vec<f32>);
    type JsValue = Vec<Float32Array>;

    fn compute(&mut self) -> Result<Self::Output> {
        let n = self.left_in.len();
        let mut left_out = vec![0.0_f32; n];
        let mut right_out = vec![0.0_f32; n];
        self.core
            .process_stereo(
                &self.left_in,
                &self.right_in,
                &mut left_out,
                &mut right_out,
                self.block_size,
            )
            .map_err(napi_err)?;
        Ok((left_out, right_out))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        let _ = &self.busy_guard;
        Ok(vec![
            Float32Array::new(output.0),
            Float32Array::new(output.1),
        ])
    }
}

#[napi]
pub struct NativeEngine {
    core: EngineCore,
    busy: Arc<AtomicBool>,
}

impl std::fmt::Debug for NativeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEngine").finish()
    }
}

impl NativeEngine {
    #[inline]
    fn ensure_audio_idle(&self) -> Result<&EngineCore> {
        if self.busy.load(Ordering::Acquire) {
            return Err(napi_err(busy_err(
                "Engine is busy executing an in-flight processAsync task",
            )));
        }
        Ok(&self.core)
    }
}

#[napi]
impl NativeEngine {
    #[napi(constructor)]
    pub fn new(
        container_bytes: Uint8Array,
        tier: u8,
        sample_rate_hz: u32,
        backend: Option<String>,
        max_supported_tier: Option<u8>,
        offline_bounce: Option<bool>,
        maximize_tier_on_bounce: Option<bool>,
    ) -> Result<Self> {
        let backend_enum =
            parse_backend_kind(backend.as_deref().unwrap_or("auto")).map_err(napi_err)?;
        let core = EngineCore::new(
            &container_bytes,
            tier,
            sample_rate_hz,
            backend_enum,
            max_supported_tier,
            offline_bounce.unwrap_or(false),
            maximize_tier_on_bounce.unwrap_or(true),
        )
        .map_err(napi_err)?;
        Ok(Self {
            core,
            busy: Arc::new(AtomicBool::new(false)),
        })
    }

    #[napi]
    pub fn reload(
        &self,
        container_bytes: Uint8Array,
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> Result<()> {
        self.core
            .reload(&container_bytes, tier, sample_rate_hz)
            .map_err(napi_err)
    }

    #[napi(getter)]
    pub fn backend(&self) -> Result<String> {
        Ok(backend_kind_str(self.core.backend_kind()).to_string())
    }

    #[napi(getter)]
    pub fn tier(&self) -> Result<u8> {
        Ok(self.core.tier())
    }

    #[napi]
    pub fn set_tier(&self, tier: u8) -> Result<()> {
        self.core.set_tier(tier).map_err(napi_err)
    }

    #[napi(getter)]
    pub fn max_supported_tier(&self) -> Result<u8> {
        Ok(self.core.max_supported_tier())
    }

    #[napi]
    pub fn set_max_supported_tier(&self, tier: u8) -> Result<()> {
        self.core.set_max_supported_tier(tier).map_err(napi_err)
    }

    #[napi(getter)]
    pub fn max_available_tier(&self) -> Result<u8> {
        Ok(self.core.max_available_tier())
    }

    #[napi(getter)]
    pub fn effective_tier(&self) -> Result<u8> {
        Ok(self.core.effective_tier())
    }

    #[napi(getter)]
    pub fn sample_rate(&self) -> Result<u32> {
        Ok(self.core.sample_rate_hz())
    }

    #[napi]
    pub fn set_sample_rate(&self, sample_rate_hz: u32) -> Result<()> {
        self.ensure_audio_idle()?
            .set_sample_rate(sample_rate_hz)
            .map_err(napi_err)
    }

    #[napi(getter)]
    pub fn latency_samples(&self) -> Result<u32> {
        Ok(self.core.latency_samples() as u32)
    }

    #[napi(getter)]
    pub fn offline_bounce(&self) -> Result<bool> {
        Ok(self.core.offline_bounce())
    }

    #[napi]
    pub fn set_offline_bounce(&self, enabled: bool) -> Result<()> {
        self.core.set_offline_bounce(enabled);
        Ok(())
    }

    #[napi(getter)]
    pub fn maximize_tier_on_bounce(&self) -> Result<bool> {
        Ok(self.core.maximize_tier_on_bounce())
    }

    #[napi]
    pub fn set_maximize_tier_on_bounce(&self, enabled: bool) -> Result<()> {
        self.core.set_maximize_tier_on_bounce(enabled);
        Ok(())
    }

    #[napi(getter)]
    pub fn num_controls(&self) -> Result<u32> {
        Ok(self.core.num_controls() as u32)
    }

    #[napi]
    pub fn raw_control_values(&self) -> Result<Float32Array> {
        Ok(Float32Array::new(self.core.raw_control_values()))
    }

    #[napi]
    pub fn clamped_control_values(&self) -> Result<Float32Array> {
        Ok(Float32Array::new(self.core.clamped_control_values()))
    }

    #[napi]
    pub fn clamped_control_value(&self, ctrl_idx: u32) -> Result<f64> {
        self.core
            .clamped_control_value(ctrl_idx as usize)
            .map(f64::from)
            .map_err(napi_err)
    }

    #[napi]
    pub fn set_control_value(&self, ctrl_idx: u32, value: f64) -> Result<()> {
        self.core
            .set_control_value(ctrl_idx as usize, value as f32)
            .map_err(napi_err)
    }

    #[napi]
    pub fn set_control_immediate(&self, ctrl_idx: u32, value: f64) -> Result<()> {
        self.core
            .set_control_immediate(ctrl_idx as usize, value as f32)
            .map_err(napi_err)
    }

    #[napi]
    pub fn set_controls_immediate_batch(&self, indices: Vec<u32>, values: Vec<f64>) -> Result<()> {
        if indices.len() != values.len() {
            return Err(napi_err(validation_err(
                "indices and values must have identical length",
            )));
        }
        let pairs: Vec<(usize, f32)> = indices
            .into_iter()
            .zip(values)
            .map(|(i, v)| (i as usize, v as f32))
            .collect();
        self.core
            .set_controls_immediate_batch(&pairs)
            .map_err(napi_err)
    }

    #[napi]
    pub fn reset_controls_to_defaults(&self) -> Result<()> {
        self.core.reset_controls_to_defaults();
        Ok(())
    }

    #[napi]
    pub fn reset(&self) -> Result<()> {
        self.core.reset_to_dc_equilibrium();
        Ok(())
    }

    #[napi(getter)]
    pub fn supports_freeze(&self) -> Result<bool> {
        Ok(self.core.supports_freeze())
    }

    #[napi(getter)]
    pub fn is_frozen(&self) -> Result<bool> {
        Ok(self.core.is_frozen())
    }

    #[napi]
    pub fn set_freeze(&self, freeze: bool) -> Result<()> {
        self.core.set_freeze(freeze);
        Ok(())
    }

    #[napi]
    pub fn set_freeze_amount(&self, alpha: f64) -> Result<()> {
        self.core.set_freeze_amount(alpha as f32);
        Ok(())
    }

    #[napi]
    pub fn bypass_state(&self) -> Result<Vec<bool>> {
        let (b, t) = self.core.bypass_state();
        Ok(vec![b, t])
    }

    #[napi]
    pub fn set_bypass(&self, bypassed: bool, trails: bool) -> Result<()> {
        self.core.set_bypass(bypassed, trails);
        Ok(())
    }

    #[napi(getter)]
    pub fn stereo_width(&self) -> Result<f64> {
        Ok(f64::from(self.core.stereo_width()))
    }

    #[napi]
    pub fn set_stereo_width(&self, width: f64) -> Result<()> {
        self.core.set_stereo_width(width as f32);
        Ok(())
    }

    #[napi]
    pub fn configure_host_mix(
        &self,
        blend_wet: f64,
        output_gain_db: f64,
        apply_unity_leveling: bool,
    ) -> Result<()> {
        self.core.configure_host_mix_and_leveling(
            blend_wet as f32,
            output_gain_db as f32,
            apply_unity_leveling,
        );
        Ok(())
    }

    #[napi(getter)]
    pub fn current_bpm(&self) -> Result<f64> {
        Ok(f64::from(self.core.current_bpm()))
    }

    #[napi]
    pub fn sync_tempo(
        &self,
        bpm: f64,
        subdivision: Option<String>,
        fold_octave: Option<bool>,
    ) -> Result<u32> {
        self.core
            .sync_tempo(
                bpm as f32,
                subdivision.as_deref(),
                fold_octave.unwrap_or(false),
            )
            .map(|c| c as u32)
            .map_err(napi_err)
    }

    #[napi]
    pub fn register_tap_tempo(
        &self,
        subdivision: Option<String>,
        fold_octave: Option<bool>,
    ) -> Result<Vec<f64>> {
        let (bpm, count) = self
            .core
            .register_tap_tempo(subdivision.as_deref(), fold_octave.unwrap_or(false))
            .map_err(napi_err)?;
        Ok(vec![f64::from(bpm), count as f64])
    }

    #[napi]
    pub fn apply_preset(&self, preset_id: String) -> Result<()> {
        self.core.apply_preset(&preset_id).map_err(napi_err)
    }

    #[napi]
    pub fn process_stereo_in_place(
        &self,
        mut left_io: Float32Array,
        mut right_io: Float32Array,
    ) -> Result<()> {
        let l = f32_slice_mut(&mut left_io);
        let r = f32_slice_mut(&mut right_io);
        self.ensure_audio_idle()?
            .process_stereo_in_place(l, r)
            .map_err(napi_err)
    }

    #[napi]
    pub fn process_stereo(
        &self,
        left_in: Float32Array,
        right_in: Float32Array,
        mut left_out: Float32Array,
        mut right_out: Float32Array,
        block_size: Option<u32>,
    ) -> Result<()> {
        let bs = block_size.map(|b| b as usize);
        let lo = f32_slice_mut(&mut left_out);
        let ro = f32_slice_mut(&mut right_out);
        self.ensure_audio_idle()?
            .process_stereo(&left_in, &right_in, lo, ro, bs)
            .map_err(napi_err)
    }

    #[napi]
    pub fn process_mono_to_stereo(
        &self,
        mono_in: Float32Array,
        mut left_out: Float32Array,
        mut right_out: Float32Array,
        block_size: Option<u32>,
    ) -> Result<()> {
        let bs = block_size.map(|b| b as usize);
        let lo = f32_slice_mut(&mut left_out);
        let ro = f32_slice_mut(&mut right_out);
        self.ensure_audio_idle()?
            .process_mono_to_stereo(&mono_in, lo, ro, bs)
            .map_err(napi_err)
    }

    #[napi]
    pub fn process_mono(
        &self,
        mono_in: Float32Array,
        mut mono_out: Float32Array,
        block_size: Option<u32>,
    ) -> Result<()> {
        let bs = block_size.map(|b| b as usize);
        let mo = f32_slice_mut(&mut mono_out);
        self.ensure_audio_idle()?
            .process_mono(&mono_in, mo, bs)
            .map_err(napi_err)
    }

    #[napi]
    pub fn render_offline_bounce(
        &self,
        left_in: Float32Array,
        right_in: Float32Array,
        mut left_out: Float32Array,
        mut right_out: Float32Array,
    ) -> Result<()> {
        let lo = f32_slice_mut(&mut left_out);
        let ro = f32_slice_mut(&mut right_out);
        self.ensure_audio_idle()?
            .render_offline_bounce(&left_in, &right_in, lo, ro)
            .map_err(napi_err)
    }

    #[napi]
    pub fn process_stereo_async(
        &self,
        left_in: Float32Array,
        right_in: Float32Array,
        block_size: Option<u32>,
    ) -> Result<AsyncTask<StereoProcessTask>> {
        if left_in.len() != right_in.len() {
            return Err(napi_err(validation_err(
                "leftIn and rightIn must have identical length",
            )));
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(napi_err(busy_err(
                "Engine is busy executing an in-flight processAsync task",
            )));
        }
        Ok(AsyncTask::new(StereoProcessTask {
            core: self.core.clone(),
            busy_guard: BusyResetGuard(Arc::clone(&self.busy)),
            left_in: left_in.to_vec(),
            right_in: right_in.to_vec(),
            block_size: block_size.map(|b| b as usize),
        }))
    }

    #[napi]
    pub fn telemetry_into(&self, mut out: Float64Array) -> Result<()> {
        let slice = f64_slice_mut(&mut out);
        let out_arr: &mut [f64; TELEMETRY_F64_LEN] = slice.try_into().map_err(|_| {
            napi_err(validation_err(format!(
                "telemetry_into requires Float64Array of length {TELEMETRY_F64_LEN}"
            )))
        })?;
        self.core.telemetry_into_f64(out_arr);
        Ok(())
    }
}
