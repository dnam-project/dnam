//! Target-agnostic `EngineCore` wrapping [`DynamicDnamEngine<'static>`] with zero duplicated DSP
//! or ZOH logic (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.engine.zero-copy-and-async]`).

use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::schema::TempoSubdivision;
use dnam_engine::{DynamicDnamEngine, Tier};

use crate::error::{control_err, format_tagged_dnam_error, validation_err};

/// Parses a backend string (`"auto"`, `"scalar"`, `"neon"`, `"arm-sme"`, `"x86-simd"`, `"wasm-simd128"`)
/// into a [`BackendKind`].
pub fn parse_backend_kind(backend: &str) -> Result<BackendKind, String> {
    match backend.to_ascii_lowercase().as_str() {
        "auto" | "native" => Ok(detect_best_backend()),
        "scalar" => Ok(BackendKind::Scalar),
        "neon" | "arm_neon" | "arm-neon" | "arm64_neon" => Ok(BackendKind::Neon),
        "arm_sme" | "arm-sme" | "armsme" | "sme" => Ok(BackendKind::ArmSme),
        "x86_simd" | "x86-simd" | "x86_avx2_fma" | "avx2" | "x86_sse42" => Ok(BackendKind::X86Simd),
        "wasm_simd128" | "wasm-simd128" | "wasm" => Ok(BackendKind::WasmSimd128),
        other => Err(validation_err(format!("Unknown backend kind: '{other}'"))),
    }
}

/// Converts a [`BackendKind`] into its canonical kebab-case TypeScript union label.
pub const fn backend_kind_str(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon | BackendKind::ArmNeon => "neon",
        BackendKind::ArmSme => "arm-sme",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => "x86-simd",
        BackendKind::WasmSimd128 => "wasm-simd128",
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
/// preset loading, and tempo synchronization to [`DynamicDnamEngine<'static>`].
#[derive(Debug)]
pub struct EngineCore {
    pub(crate) inner: DynamicDnamEngine<'static>,
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
        Ok(Self { inner })
    }

    pub fn reload(
        &mut self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> Result<(), String> {
        if let Some(t) = tier
            && t > 3
        {
            return Err(validation_err(format!("Invalid tier: {t}")));
        }
        let target_tier = tier.map_or_else(|| self.inner.tier(), Tier::from_u8);
        let target_sr = sample_rate_hz.unwrap_or_else(|| self.inner.sample_rate_hz());
        self.inner
            .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
            .map_err(|e| format_tagged_dnam_error(&e))
    }

    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        self.inner.backend_kind()
    }

    #[inline]
    pub const fn tier(&self) -> u8 {
        self.inner.tier() as u8
    }

    #[inline]
    pub fn set_tier(&mut self, tier: u8) -> Result<(), String> {
        if tier > 3 {
            return Err(validation_err(format!("Invalid tier: {tier}")));
        }
        self.inner.set_tier(Tier::from_u8(tier));
        Ok(())
    }

    #[inline]
    pub const fn max_supported_tier(&self) -> u8 {
        self.inner.max_supported_tier() as u8
    }

    #[inline]
    pub fn set_max_supported_tier(&mut self, tier: u8) -> Result<(), String> {
        if tier > 3 {
            return Err(validation_err(format!("Invalid tier: {tier}")));
        }
        self.inner.set_max_supported_tier(Tier::from_u8(tier));
        Ok(())
    }

    #[inline]
    pub const fn max_available_tier(&self) -> u8 {
        self.inner.max_available_tier() as u8
    }

    #[inline]
    pub fn effective_tier(&self) -> u8 {
        self.inner.effective_tier() as u8
    }

    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz()
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), String> {
        self.inner
            .set_sample_rate(sample_rate_hz)
            .map_err(|e| format_tagged_dnam_error(&e))
    }

    #[inline]
    pub const fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.inner.offline_bounce()
    }

    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        self.inner.set_offline_bounce(enabled);
    }

    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        self.inner.maximize_tier_on_bounce()
    }

    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        self.inner.set_maximize_tier_on_bounce(enabled);
    }

    #[inline]
    pub const fn num_controls(&self) -> usize {
        self.inner.num_controls()
    }

    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        self.inner.raw_control_values()
    }

    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        self.inner.clamped_control_values()
    }

    pub fn clamped_control_value(&self, ctrl_idx: usize) -> Result<f32, String> {
        self.inner
            .clamped_control_values()
            .get(ctrl_idx)
            .copied()
            .ok_or_else(|| {
                control_err(format!(
                    "parameter index {ctrl_idx} out of bounds (num_controls={})",
                    self.inner.num_controls()
                ))
            })
    }

    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) -> Result<(), String> {
        if ctrl_idx >= self.inner.num_controls() {
            return Err(control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={})",
                self.inner.num_controls()
            )));
        }
        self.inner.set_control_value(ctrl_idx, value);
        Ok(())
    }

    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) -> Result<(), String> {
        if ctrl_idx >= self.inner.num_controls() {
            return Err(control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={})",
                self.inner.num_controls()
            )));
        }
        self.inner.set_control_immediate(ctrl_idx, value);
        Ok(())
    }

    pub fn set_controls_immediate_batch(&mut self, updates: &[(usize, f32)]) -> Result<(), String> {
        let num_controls = self.inner.num_controls();
        for &(ctrl_idx, _) in updates {
            if ctrl_idx >= num_controls {
                return Err(control_err(format!(
                    "parameter index {ctrl_idx} out of bounds (num_controls={num_controls})"
                )));
            }
        }
        let Some((&(last_idx, last_val), init)) = updates.split_last() else {
            return Ok(());
        };
        for &(idx, val) in init {
            self.inner.set_control_value(idx, val);
        }
        self.inner.set_control_immediate(last_idx, last_val);
        Ok(())
    }

    #[inline]
    pub fn reset_controls_to_defaults(&mut self) {
        self.inner.reset_controls_to_defaults();
    }

    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.inner.reset_to_dc_equilibrium();
    }

    #[inline]
    pub fn supports_freeze(&self) -> bool {
        self.inner.supports_freeze()
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.inner.is_frozen()
    }

    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        self.inner.set_freeze(freeze);
    }

    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        self.inner.set_freeze_amount(alpha);
    }

    #[inline]
    pub fn bypass_state(&self) -> (bool, bool) {
        self.inner.bypass_state()
    }

    #[inline]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        self.inner.set_bypass(bypassed, trails);
    }

    #[inline]
    pub fn stereo_width(&self) -> f32 {
        self.inner.stereo_width()
    }

    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        self.inner.set_stereo_width(width);
    }

    #[inline]
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        self.inner
            .configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity_leveling);
    }

    #[inline]
    pub fn current_bpm(&self) -> f32 {
        self.inner.current_bpm()
    }

    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> Result<usize, String> {
        let sub = parse_subdivision_opt(subdivision)?;
        Ok(self.inner.sync_tempo(bpm, sub, fold_octave))
    }

    pub fn register_tap_tempo(
        &mut self,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> Result<(f32, usize), String> {
        let sub = parse_subdivision_opt(subdivision)?;
        Ok(self.inner.register_tap_tempo(sub, fold_octave))
    }

    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), String> {
        self.inner
            .apply_preset(preset_id)
            .map_err(|e| control_err(format!("{e}")))
    }

    #[inline]
    pub fn process_stereo_in_place(
        &mut self,
        left_io: &mut [f32],
        right_io: &mut [f32],
    ) -> Result<(), String> {
        if left_io.len() != right_io.len() {
            return Err(validation_err(
                "left and right buffers must have identical length",
            ));
        }
        self.inner.process_block_in_place(left_io, right_io);
        Ok(())
    }

    pub fn process_stereo(
        &mut self,
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
        let chunk = block_size.unwrap_or(n).max(1);
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

    pub fn process_mono_to_stereo(
        &mut self,
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
        let chunk = block_size.unwrap_or(n).max(1);
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

    pub fn process_mono(
        &mut self,
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
        let chunk = block_size.unwrap_or(n).max(1);
        if chunk >= n {
            self.inner.process_mono(mono_in, mono_out);
        } else {
            for (mi, mo) in mono_in.chunks(chunk).zip(mono_out.chunks_mut(chunk)) {
                self.inner.process_mono(mi, mo);
            }
        }
        Ok(())
    }

    pub fn render_offline_bounce(
        &mut self,
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
        self.inner
            .render_offline_bounce(left_in, right_in, left_out, right_out);
        Ok(())
    }

    /// Writes `56` `f64` telemetry & ZOH values directly into `out` with zero heap allocations.
    #[inline]
    pub fn telemetry_into_f64(&self, out: &mut [f64; TELEMETRY_F64_LEN]) {
        let snap = self.inner.telemetry_snapshot();
        let zoh = self.inner.zoh();
        let (bypassed, trails_enabled) = self.inner.bypass_state();
        let kind = self.inner.backend_kind();
        out[0] = f64::from(snap.active_tier);
        out[1] = f64::from(backend_kind_code(kind));
        out[2] = f64::from(self.inner.sample_rate_hz());
        out[3] = f64::from(snap.stereo_out);
        out[4] = f64::from(u8::from(bypassed));
        out[5] = f64::from(u8::from(trails_enabled));
        out[6] = f64::from(snap.is_frozen);
        out[7] = f64::from(u8::from(self.inner.supports_freeze()));
        out[8] = f64::from(self.inner.stereo_width());
        out[9] = f64::from(self.inner.current_bpm());
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
