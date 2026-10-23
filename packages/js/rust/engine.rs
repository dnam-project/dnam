//! Shared `EngineCore` wrapping [`DynamicDnamEngine<'static>`] for both N-API and WASM targets.

use dnam_engine::container::{DnamPreamble, ZohRescaledParams};
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::{DynamicDnamEngine, Tier};

use crate::format_tagged_dnam_error;

/// Parses a backend string (`"auto"`, `"scalar"`, `"arm_neon"`, `"arm_sme"`, `"x86_simd"`, `"wasm_simd128"`)
/// into a [`BackendKind`].
pub fn parse_backend_kind(backend: &str) -> Result<BackendKind, String> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" | "native" => Ok(detect_best_backend()),
        "scalar" => Ok(BackendKind::Scalar),
        "neon" | "arm_neon" | "arm64_neon" => Ok(BackendKind::Neon),
        "arm_sme" | "armsme" | "sme" => Ok(BackendKind::ArmSme),
        "x86_simd" | "x86_avx2_fma" | "avx2" | "x86_sse42" => Ok(BackendKind::X86Simd),
        "wasm_simd128" | "wasm" => Ok(BackendKind::WasmSimd128),
        other => Err(format!("[DNAM-VALIDATION] Unknown backend kind: '{other}'")),
    }
}

/// Converts a [`BackendKind`] into its canonical string identifier.
pub const fn backend_kind_str(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon | BackendKind::ArmNeon => "arm_neon",
        BackendKind::ArmSme => "arm_sme",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => "x86_simd",
        BackendKind::WasmSimd128 => "wasm_simd128",
    }
}

/// Converts a [`BackendKind`] into a compact numeric code (`0..=4`) for C-ABI / WASM transport.
pub const fn backend_kind_code(kind: BackendKind) -> u8 {
    match kind {
        BackendKind::Scalar => 0,
        BackendKind::Neon | BackendKind::ArmNeon => 1,
        BackendKind::ArmSme => 2,
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => 3,
        BackendKind::WasmSimd128 => 4,
    }
}

/// Decodes a numeric backend code (`0..=4`, or `255` for `"auto"`) into a [`BackendKind`].
pub fn backend_kind_from_code(code: u8) -> Result<BackendKind, String> {
    match code {
        0 => Ok(BackendKind::Scalar),
        1 => Ok(BackendKind::Neon),
        2 => Ok(BackendKind::ArmSme),
        3 => Ok(BackendKind::X86Simd),
        4 => Ok(BackendKind::WasmSimd128),
        255 => Ok(detect_best_backend()),
        other => Err(format!("[DNAM-VALIDATION] Unknown backend code: {other}")),
    }
}

/// Combined telemetry & ZOH snapshot returned to JavaScript / TypeScript.
#[derive(Debug, Clone)]
pub struct TelemetryData {
    pub active_tier: u8,
    pub backend: &'static str,
    pub backend_code: u8,
    pub sample_rate_hz: u32,
    pub core2_active: bool,
    pub lfo_env_active: bool,
    pub stereo_out: bool,
    pub active_clamp_mask: u16,
    pub control_decimation_host: u16,
    pub max_adaa_drive: f32,
    pub max_pole_radius: f32,
    pub max_effective_loop_gain: f32,
    pub tail_time_ms: f32,
    pub z_head: [f32; 4],
    pub env_states: [f32; 4],
    pub lfo_harmonics: [f32; 8],
    pub clamped_controls: [f32; 16],
    pub rho_sr: f64,
    pub rho_ctrl: f64,
    pub r_ac_host: f32,
    pub r_dc_host: f32,
    pub tau_ref_samples_host: f32,
    pub switch_xfade_samples_host: u32,
}

/// Shared Rust engine state wrapping [`DynamicDnamEngine<'static>`].
#[derive(Debug)]
pub struct EngineCore {
    inner: DynamicDnamEngine<'static>,
    sample_rate_hz: u32,
    sample_rate_train_hz: u32,
    control_decimation_train: u8,
    zoh: ZohRescaledParams,
}

impl EngineCore {
    /// Creates a new [`EngineCore`] from `.dnam` or `.dnam-ckpt` bytes.
    pub fn new(
        container_bytes: &[u8],
        tier: u8,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<Self, String> {
        if !(0..=2).contains(&tier) {
            return Err(format!("[DNAM-VALIDATION] Invalid tier: {tier}"));
        }
        let preamble =
            DnamPreamble::from_bytes(container_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
        let sample_rate_train_hz = preamble.sample_rate_train_hz;
        let control_decimation_train = preamble.control_decimation.max(8);

        let tier_enum = Tier::from_u8(tier);
        let inner = DynamicDnamEngine::from_bytes_owned_with_backend(
            container_bytes,
            tier_enum,
            sample_rate_hz,
            backend,
        )
        .map_err(|e| format_tagged_dnam_error(&e))?;

        let zoh = ZohRescaledParams::compute(
            sample_rate_train_hz,
            sample_rate_hz,
            control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(|e| format_tagged_dnam_error(&e))?;

        Ok(Self {
            inner,
            sample_rate_hz,
            sample_rate_train_hz,
            control_decimation_train,
            zoh,
        })
    }

    /// Hot-swaps a new `.dnam` or `.dnam-ckpt` container in-place.
    pub fn reload(
        &mut self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> Result<(), String> {
        if let Some(t) = tier
            && !(0..=2).contains(&t)
        {
            return Err(format!("[DNAM-VALIDATION] Invalid tier: {t}"));
        }
        let preamble =
            DnamPreamble::from_bytes(container_bytes).map_err(|e| format_tagged_dnam_error(&e))?;
        let target_tier = tier.map_or_else(
            || Tier::from_u8(self.inner.telemetry_snapshot().active_tier),
            Tier::from_u8,
        );
        let target_sr = sample_rate_hz.unwrap_or(self.sample_rate_hz);

        self.inner
            .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
            .map_err(|e| format_tagged_dnam_error(&e))?;

        let sample_rate_train_hz = preamble.sample_rate_train_hz;
        let control_decimation_train = preamble.control_decimation.max(8);
        self.zoh = ZohRescaledParams::compute(
            sample_rate_train_hz,
            target_sr,
            control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(|e| format_tagged_dnam_error(&e))?;
        self.sample_rate_hz = target_sr;
        self.sample_rate_train_hz = sample_rate_train_hz;
        self.control_decimation_train = control_decimation_train;
        Ok(())
    }

    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        self.inner.backend_kind()
    }

    #[inline]
    pub fn active_tier(&self) -> u8 {
        self.inner.telemetry_snapshot().active_tier
    }

    #[inline]
    pub fn set_tier(&mut self, tier: u8) -> Result<(), String> {
        if !(0..=2).contains(&tier) {
            return Err(format!("[DNAM-VALIDATION] Invalid tier: {tier}"));
        }
        self.inner.set_tier(Tier::from_u8(tier));
        Ok(())
    }

    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), String> {
        self.inner
            .set_sample_rate(sample_rate_hz)
            .map_err(|e| format_tagged_dnam_error(&e))?;
        self.zoh = ZohRescaledParams::compute(
            self.sample_rate_train_hz,
            sample_rate_hz,
            self.control_decimation_train,
            0.9993,
            0.995,
            64.0,
            128,
        )
        .map_err(|e| format_tagged_dnam_error(&e))?;
        self.sample_rate_hz = sample_rate_hz;
        Ok(())
    }

    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        self.inner.set_control_value(ctrl_idx, value);
    }

    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        self.inner.set_control_immediate(ctrl_idx, value);
    }

    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset_to_dc_equilibrium();
    }

    /// Processes planar stereo slices in chunks of `buffer_size`.
    pub fn process_stereo(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        buffer_size: Option<usize>,
    ) -> Result<(), String> {
        let n = left_in.len();
        if right_in.len() != n || left_out.len() != n || right_out.len() != n {
            return Err(
                "[DNAM-VALIDATION] leftIn, rightIn, leftOut, and rightOut must have identical length"
                    .to_string(),
            );
        }
        if n == 0 {
            return Ok(());
        }
        let chunk = buffer_size.unwrap_or(n).max(1);
        if chunk >= n {
            self.inner
                .process_block(left_in, right_in, left_out, right_out);
        } else {
            for (((li, ri), lo), ro) in left_in
                .chunks(chunk)
                .zip(right_in.chunks(chunk))
                .zip(left_out.chunks_mut(chunk))
                .zip(right_out.chunks_mut(chunk))
            {
                self.inner.process_block(li, ri, lo, ro);
            }
        }
        Ok(())
    }

    /// Processes a mono slice into true-stereo `left_out` and `right_out` slices.
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        buffer_size: Option<usize>,
    ) -> Result<(), String> {
        let n = mono_in.len();
        if left_out.len() != n || right_out.len() != n {
            return Err(
                "[DNAM-VALIDATION] monoIn, leftOut, and rightOut must have identical length"
                    .to_string(),
            );
        }
        if n == 0 {
            return Ok(());
        }
        let chunk = buffer_size.unwrap_or(n).max(1);
        if chunk >= n {
            self.inner
                .process_mono_to_stereo(mono_in, left_out, right_out);
        } else {
            for ((mi, lo), ro) in mono_in
                .chunks(chunk)
                .zip(left_out.chunks_mut(chunk))
                .zip(right_out.chunks_mut(chunk))
            {
                self.inner.process_mono_to_stereo(mi, lo, ro);
            }
        }
        Ok(())
    }

    /// Processes a mono slice into a 1-channel `mono_out` slice.
    pub fn process_mono(
        &mut self,
        mono_in: &[f32],
        mono_out: &mut [f32],
        buffer_size: Option<usize>,
    ) -> Result<(), String> {
        let n = mono_in.len();
        if mono_out.len() != n {
            return Err(
                "[DNAM-VALIDATION] monoIn and monoOut must have identical length".to_string(),
            );
        }
        if n == 0 {
            return Ok(());
        }
        let chunk = buffer_size.unwrap_or(n).max(1);
        if chunk >= n {
            self.inner.process_mono(mono_in, mono_out);
        } else {
            for (mi, mo) in mono_in.chunks(chunk).zip(mono_out.chunks_mut(chunk)) {
                self.inner.process_mono(mi, mo);
            }
        }
        Ok(())
    }

    /// Extracts a lock-free [`TelemetryData`] snapshot combined with cached [`ZohRescaledParams`].
    #[inline]
    pub fn telemetry(&self) -> Result<TelemetryData, String> {
        let snap = self.inner.telemetry_snapshot();
        let zoh = &self.zoh;
        let kind = self.inner.backend_kind();
        Ok(TelemetryData {
            active_tier: snap.active_tier,
            backend: backend_kind_str(kind),
            backend_code: backend_kind_code(kind),
            sample_rate_hz: self.sample_rate_hz,
            core2_active: snap.core2_active != 0,
            lfo_env_active: snap.lfo_env_active != 0,
            stereo_out: snap.stereo_out != 0,
            active_clamp_mask: snap.active_clamp_mask,
            control_decimation_host: snap.control_decimation_host,
            max_adaa_drive: snap.max_adaa_drive,
            max_pole_radius: snap.max_pole_radius,
            max_effective_loop_gain: snap.max_effective_loop_gain,
            tail_time_ms: snap.tail_time_ms,
            z_head: snap.z_head,
            env_states: snap.env_states,
            lfo_harmonics: snap.lfo_harmonics,
            clamped_controls: snap.clamped_controls,
            rho_sr: zoh.rho_sr,
            rho_ctrl: zoh.rho_ctrl,
            r_ac_host: zoh.r_ac_host,
            r_dc_host: zoh.r_dc_host,
            tau_ref_samples_host: zoh.tau_ref_samples_host,
            switch_xfade_samples_host: zoh.switch_xfade_samples_host,
        })
    }

    /// Writes 50 `f64` telemetry & ZOH values directly into `out` with zero heap allocations.
    #[inline]
    pub fn telemetry_into_f64(&self, out: &mut [f64; 50]) {
        let snap = self.inner.telemetry_snapshot();
        let zoh = &self.zoh;
        let kind = self.inner.backend_kind();
        out[0] = f64::from(snap.active_tier);
        out[1] = f64::from(backend_kind_code(kind));
        out[2] = f64::from(self.sample_rate_hz);
        out[3] = f64::from(snap.core2_active);
        out[4] = f64::from(snap.lfo_env_active);
        out[5] = f64::from(snap.stereo_out);
        out[6] = f64::from(snap.active_clamp_mask);
        out[7] = f64::from(snap.control_decimation_host);
        out[8] = f64::from(snap.max_adaa_drive);
        out[9] = f64::from(snap.max_pole_radius);
        out[10] = f64::from(snap.max_effective_loop_gain);
        out[11] = f64::from(snap.tail_time_ms);
        for i in 0..4 {
            out[12 + i] = f64::from(snap.z_head[i]);
            out[16 + i] = f64::from(snap.env_states[i]);
        }
        for i in 0..8 {
            out[20 + i] = f64::from(snap.lfo_harmonics[i]);
        }
        for i in 0..16 {
            out[28 + i] = f64::from(snap.clamped_controls[i]);
        }
        out[44] = zoh.rho_sr;
        out[45] = zoh.rho_ctrl;
        out[46] = f64::from(zoh.r_ac_host);
        out[47] = f64::from(zoh.r_dc_host);
        out[48] = f64::from(zoh.tau_ref_samples_host);
        out[49] = f64::from(zoh.switch_xfade_samples_host);
    }
}
