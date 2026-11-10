//! Target-agnostic `EngineCore` wrapping [`DnamHostHandle`] with zero duplicated DSP,
//! ZOH, SPSC control ingress, or Seqlock telemetry logic (`ref[sdk.ts.runtime.dual-target-drivers]`,
//! `ref[sdk.ts.engine.zero-copy-and-async]`, `ref[dsp.ctrl.op-m.spsc-ingress]`,
//! `ref[arch.telemetry.seqlock-publication]`).

use dnam_engine::host::DnamHostHandle;
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::schema::TempoSubdivision;
use dnam_engine::{DynamicDnamEngine, Tier};

use crate::error::{control_err, format_tagged_dnam_error, validation_err};

/// Parses a backend string (`"auto"`, `"scalar"`, `"neon"`, `"arm-sve2"`, `"arm-sme"`, `"x86-simd"`, `"wasm-simd128"`)
/// into a [`BackendKind`].
pub fn parse_backend_kind(backend: &str) -> Result<BackendKind, String> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" | "native" => Ok(detect_best_backend()),
        "scalar" => Ok(BackendKind::Scalar),
        "neon" | "arm_neon" | "arm-neon" | "arm64_neon" => Ok(BackendKind::Neon),
        "arm_sve2" | "arm-sve2" | "armsve2" | "sve2" => Ok(BackendKind::ArmSve2),
        "arm_sme" | "arm-sme" | "armsme" | "sme" => Ok(BackendKind::ArmSme),
        "x86_sse2" | "x86-sse2" | "sse2" => Ok(BackendKind::X86Sse2),
        "x86_sse42" | "x86-sse42" | "sse42" | "sse4.2" => Ok(BackendKind::X86Sse42),
        "x86_avx2_fma" | "x86-avx2-fma" | "avx2" => Ok(BackendKind::X86Avx2Fma),
        "x86_simd" | "x86-simd" => Ok(BackendKind::X86Simd),
        "wasm_simd128" | "wasm-simd128" | "wasm" => Ok(BackendKind::WasmSimd128),
        other => Err(validation_err(format!("Unknown backend kind: '{other}'"))),
    }
}

/// Converts a [`BackendKind`] into its canonical kebab-case TypeScript union label.
pub const fn backend_kind_str(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon | BackendKind::ArmNeon | BackendKind::ArmSve2 => "neon",
        BackendKind::ArmSme => "arm-sme",
        BackendKind::X86Simd
        | BackendKind::X86Avx2Fma
        | BackendKind::X86Sse42
        | BackendKind::X86Sse2 => "x86-simd",
        BackendKind::WasmSimd128 => "wasm-simd128",
    }
}

/// Converts a [`BackendKind`] into a compact numeric code (`0..=4`) for C-ABI / WASM transport.
pub const fn backend_kind_code(kind: BackendKind) -> u8 {
    match kind {
        BackendKind::Scalar => 0,
        BackendKind::Neon | BackendKind::ArmNeon | BackendKind::ArmSve2 => 1,
        BackendKind::ArmSme => 2,
        BackendKind::X86Simd
        | BackendKind::X86Avx2Fma
        | BackendKind::X86Sse42
        | BackendKind::X86Sse2 => 3,
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
        other => Err(validation_err(format!("Unknown backend code: {other}"))),
    }
}

pub fn parse_subdivision_opt(
    subdivision: Option<&str>,
) -> Result<Option<TempoSubdivision>, String> {
    match subdivision {
        None | Some("") => Ok(None),
        Some(s) => TempoSubdivision::from_str_label(s)
            .map(Some)
            .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{s}'"))),
    }
}

/// Number of `f64` elements written by [`EngineCore::telemetry_into_f64`].
pub const TELEMETRY_F64_LEN: usize = 56;

/// Target-agnostic Rust engine wrapper delegating 100% of DSP, ZOH rescaling, control clamping,
/// preset loading, tempo synchronization, lock-free SPSC control ingress, and wait-free Seqlock
/// telemetry to [`DnamHostHandle`].
#[derive(Debug, Clone)]
pub struct EngineCore {
    pub(crate) host: DnamHostHandle,
}

impl EngineCore {
    pub fn new(
        container_bytes: &[u8],
        tier: u8,
        sample_rate_hz: u32,
        backend: BackendKind,
        max_supported_tier: Option<u8>,
        offline_bounce: bool,
        maximize_tier_on_bounce: bool,
    ) -> Result<Self, String> {
        if tier > 3 {
            return Err(validation_err(format!("Invalid tier: {tier}")));
        }
        if let Some(max_t) = max_supported_tier
            && max_t > 3
        {
            return Err(validation_err(format!("Invalid maxSupportedTier: {max_t}")));
        }
        let mut inner = DynamicDnamEngine::from_bytes_owned_with_backend(
            container_bytes,
            Tier::from_u8(tier),
            sample_rate_hz,
            backend,
        )
        .map_err(|e| format_tagged_dnam_error(&e))?;
        if let Some(max_t) = max_supported_tier {
            inner.set_max_supported_tier(Tier::from_u8(max_t));
        }
        inner.set_maximize_tier_on_bounce(maximize_tier_on_bounce);
        if offline_bounce {
            inner.set_offline_bounce(true);
        }
        Ok(Self {
            host: DnamHostHandle::new(inner),
        })
    }

    pub fn reload(
        &self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> Result<(), String> {
        if let Some(t) = tier
            && t > 3
        {
            return Err(validation_err(format!("Invalid tier: {t}")));
        }
        let target_tier = tier.map_or_else(|| self.host.tier(), Tier::from_u8);
        let target_sr = sample_rate_hz.unwrap_or_else(|| self.host.sample_rate_hz());
        self.host
            .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
            .map_err(|e| format_tagged_dnam_error(&e))
    }

    #[inline]
    pub fn backend_kind(&self) -> BackendKind {
        self.host.backend_kind()
    }

    #[inline]
    pub fn tier(&self) -> u8 {
        self.host.tier() as u8
    }

    #[inline]
    pub fn set_tier(&self, tier: u8) -> Result<(), String> {
        if tier > 3 {
            return Err(validation_err(format!("Invalid tier: {tier}")));
        }
        self.host.set_tier(Tier::from_u8(tier));
        Ok(())
    }

    #[inline]
    pub fn max_supported_tier(&self) -> u8 {
        self.host.max_supported_tier() as u8
    }

    #[inline]
    pub fn set_max_supported_tier(&self, tier: u8) -> Result<(), String> {
        if tier > 3 {
            return Err(validation_err(format!("Invalid tier: {tier}")));
        }
        self.host.set_max_supported_tier(Tier::from_u8(tier));
        Ok(())
    }

    #[inline]
    pub fn max_available_tier(&self) -> u8 {
        self.host.max_available_tier() as u8
    }

    #[inline]
    pub fn effective_tier(&self) -> u8 {
        self.host.effective_tier() as u8
    }

    #[inline]
    pub fn sample_rate_hz(&self) -> u32 {
        self.host.sample_rate_hz()
    }

    #[inline]
    pub fn set_sample_rate(&self, sample_rate_hz: u32) -> Result<(), String> {
        self.host
            .set_sample_rate(sample_rate_hz)
            .map_err(|e| format_tagged_dnam_error(&e))
    }

    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.host.latency_samples()
    }

    #[inline]
    pub fn offline_bounce(&self) -> bool {
        self.host.offline_bounce()
    }

    #[inline]
    pub fn set_offline_bounce(&self, enabled: bool) {
        self.host.set_offline_bounce(enabled);
    }

    #[inline]
    pub fn maximize_tier_on_bounce(&self) -> bool {
        self.host.maximize_tier_on_bounce()
    }

    #[inline]
    pub fn set_maximize_tier_on_bounce(&self, enabled: bool) {
        self.host.set_maximize_tier_on_bounce(enabled);
    }

    #[inline]
    pub fn num_controls(&self) -> usize {
        self.host.num_controls()
    }

    #[inline]
    pub fn raw_control_values(&self) -> Vec<f32> {
        self.host.raw_control_values()
    }

    #[inline]
    pub fn clamped_control_values(&self) -> Vec<f32> {
        self.host.clamped_control_values()
    }

    pub fn clamped_control_value(&self, ctrl_idx: usize) -> Result<f32, String> {
        self.host.clamped_control_value(ctrl_idx).ok_or_else(|| {
            control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={})",
                self.host.num_controls()
            ))
        })
    }

    pub fn set_control_value(&self, ctrl_idx: usize, value: f32) -> Result<(), String> {
        let num_controls = self.host.num_controls();
        if ctrl_idx >= num_controls {
            return Err(control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={num_controls})"
            )));
        }
        self.host.set_control_value(ctrl_idx, value);
        Ok(())
    }

    pub fn set_control_immediate(&self, ctrl_idx: usize, value: f32) -> Result<(), String> {
        let num_controls = self.host.num_controls();
        if ctrl_idx >= num_controls {
            return Err(control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={num_controls})"
            )));
        }
        self.host.set_control_immediate(ctrl_idx, value);
        Ok(())
    }

    pub fn set_controls_immediate_batch(&self, updates: &[(usize, f32)]) -> Result<(), String> {
        let num_controls = self.host.num_controls();
        for &(ctrl_idx, _) in updates {
            if ctrl_idx >= num_controls {
                return Err(control_err(format!(
                    "parameter index {ctrl_idx} out of bounds (num_controls={num_controls})"
                )));
            }
        }
        if !updates.is_empty() {
            self.host.set_controls_immediate_batch(updates);
        }
        Ok(())
    }

    #[inline]
    pub fn reset_controls_to_defaults(&self) {
        self.host.reset_controls_to_defaults();
    }

    #[inline]
    pub fn reset_to_dc_equilibrium(&self) {
        self.host.reset_to_dc_equilibrium();
    }

    #[inline]
    pub fn supports_freeze(&self) -> bool {
        self.host.supports_freeze()
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.host.is_frozen()
    }

    #[inline]
    pub fn set_freeze(&self, freeze: bool) {
        self.host.set_freeze(freeze);
    }

    #[inline]
    pub fn set_freeze_amount(&self, alpha: f32) {
        self.host.set_freeze_amount(alpha);
    }

    #[inline]
    pub fn bypass_state(&self) -> (bool, bool) {
        self.host.bypass_state()
    }

    #[inline]
    pub fn set_bypass(&self, bypassed: bool, trails: bool) {
        self.host.set_bypass(bypassed, trails);
    }

    #[inline]
    pub fn stereo_width(&self) -> f32 {
        self.host.stereo_width()
    }

    #[inline]
    pub fn set_stereo_width(&self, width: f32) {
        self.host.set_stereo_width(width);
    }

    #[inline]
    pub fn configure_host_mix_and_leveling(
        &self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        self.host
            .configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity_leveling);
    }

    #[inline]
    pub fn current_bpm(&self) -> f32 {
        self.host.current_bpm()
    }

    pub fn sync_tempo(
        &self,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> Result<usize, String> {
        let sub = parse_subdivision_opt(subdivision)?;
        Ok(self.host.sync_tempo(bpm, sub, fold_octave))
    }

    pub fn register_tap_tempo(
        &self,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> Result<(f32, usize), String> {
        let sub = parse_subdivision_opt(subdivision)?;
        Ok(self.host.register_tap_tempo(sub, fold_octave))
    }

    pub fn apply_preset(&self, preset_id: &str) -> Result<(), String> {
        self.host
            .apply_preset(preset_id)
            .map_err(|e| control_err(format!("{e}")))
    }

    #[inline]
    pub fn process_stereo_in_place(
        &self,
        left_io: &mut [f32],
        right_io: &mut [f32],
    ) -> Result<(), String> {
        if left_io.len() != right_io.len() {
            return Err(validation_err(
                "left and right buffers must have identical length",
            ));
        }
        self.host.process_block_in_place(left_io, right_io);
        Ok(())
    }

    pub fn process_stereo(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        block_size: Option<usize>,
    ) -> Result<(), String> {
        let n = left_in.len();
        if right_in.len() != n || left_out.len() != n || right_out.len() != n {
            return Err(validation_err(
                "leftIn, rightIn, leftOut, and rightOut must have identical length",
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.host
            .process_block_chunked(left_in, right_in, left_out, right_out, block_size);
        Ok(())
    }

    pub fn process_mono_to_stereo(
        &self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        block_size: Option<usize>,
    ) -> Result<(), String> {
        let n = mono_in.len();
        if left_out.len() != n || right_out.len() != n {
            return Err(validation_err(
                "monoIn, leftOut, and rightOut must have identical length",
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.host
            .process_mono_to_stereo_chunked(mono_in, left_out, right_out, block_size);
        Ok(())
    }

    pub fn process_mono(
        &self,
        mono_in: &[f32],
        mono_out: &mut [f32],
        block_size: Option<usize>,
    ) -> Result<(), String> {
        let n = mono_in.len();
        if mono_out.len() != n {
            return Err(validation_err(
                "monoIn and monoOut must have identical length",
            ));
        }
        if n == 0 {
            return Ok(());
        }
        self.host
            .process_mono_chunked(mono_in, mono_out, block_size);
        Ok(())
    }

    pub fn render_offline_bounce(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) -> Result<(), String> {
        let n = left_in.len();
        if right_in.len() != n || left_out.len() != n || right_out.len() != n {
            return Err(validation_err(
                "leftIn, rightIn, leftOut, and rightOut must have identical length",
            ));
        }
        self.host
            .render_offline_bounce(left_in, right_in, left_out, right_out);
        Ok(())
    }

    /// Writes `56` `f64` telemetry & ZOH values directly into `out` wait-free with zero heap allocations.
    #[inline]
    pub fn telemetry_into_f64(&self, out: &mut [f64; TELEMETRY_F64_LEN]) {
        let rec = self.host.telemetry_record();
        let snap = rec.snapshot;
        let zoh = rec.zoh;
        out[0] = f64::from(snap.active_tier);
        out[1] = f64::from(backend_kind_code(rec.backend_kind));
        out[2] = f64::from(rec.sample_rate_hz);
        out[3] = f64::from(snap.stereo_out);
        out[4] = f64::from(snap.bypassed);
        out[5] = f64::from(u8::from(rec.trails_enabled));
        out[6] = f64::from(snap.is_frozen);
        out[7] = f64::from(u8::from(rec.supports_freeze));
        out[8] = f64::from(rec.stereo_width);
        out[9] = f64::from(rec.current_bpm);
        out[10] = f64::from(snap.active_clamp_mask);
        out[11] = f64::from(snap.control_decimation_host);
        out[12] = f64::from(snap.max_op_n_drive);
        out[13] = f64::from(snap.max_pole_radius);
        out[14] = f64::from(snap.max_effective_loop_gain);
        out[15] = f64::from(snap.tail_time_ms);
        for i in 0..4 {
            out[16 + i] = f64::from(snap.z_head[i]);
            out[20 + i] = f64::from(snap.env_states[i]);
        }
        for i in 0..8 {
            out[24 + i] = f64::from(snap.lfo_harmonics[i]);
        }
        for i in 0..16 {
            out[32 + i] = f64::from(snap.clamped_controls[i]);
        }
        out[48] = f64::from(snap.overflow_or_saturation_flags);
        out[49] = zoh.rho_sr;
        out[50] = zoh.rho_ctrl;
        out[51] = f64::from(zoh.alpha_ramp_host);
        out[52] = f64::from(zoh.r_ac_host);
        out[53] = f64::from(zoh.r_dc_host);
        out[54] = f64::from(zoh.tau_ref_samples_host);
        out[55] = f64::from(zoh.switch_xfade_samples_host);
    }
}
