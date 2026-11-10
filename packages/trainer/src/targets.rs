#![forbid(unsafe_code)]
//! Sans-IO [`DistillationTarget`] trait and multi-port excitation/response descriptors
//! (`dnam_trainer::targets`, `ref[train.capture.excitation.universal-30s]`).

use std::collections::BTreeMap;

use dnam_engine::engine::EngineBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamAudioIo, DnamEngine, DynamicDnamEngine};

/// Single causal excitation frame (e.g. `96 ms` = `4,608` samples at `48 kHz`) for closed-loop
/// `AdaptiveStimulusController` target probing (`ref[train.capture.excitation.universal-30s]`).
#[derive(Clone, Debug, PartialEq)]
pub struct StimulusFrame {
    /// Zero-based sequential frame index within the active stem.
    pub frame_index: usize,
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Primary input channel `in_1` (`Inst In L` / `Mono`).
    pub in_1: Vec<f32>,
    /// Optional stereo right input channel `in_2` (`Inst In R`).
    pub in_2: Option<Vec<f32>>,
    /// Optional auxiliary/return left input channel `in_3` (`FX Loop Return L` / `Sidechain Key L`).
    pub in_3: Option<Vec<f32>>,
    /// Optional auxiliary/return right input channel `in_4` (`FX Loop Return R` / `Sidechain Key R`).
    pub in_4: Option<Vec<f32>>,
    /// Physical or normalized control values `[0..P]` at the start of this frame.
    pub control_values: Vec<f32>,
    /// Optional per-control sample-accurate sweep trajectories (`len == P`, each `len == in_1.len()`).
    pub control_trajectories: Option<Vec<Vec<f32>>>,
}

/// Complete excitation stem (`Mode A` continuous sweep or `Mode B` stationary stop) for
/// hardware capture or white-box/grey-box target distillation.
#[derive(Clone, Debug, PartialEq)]
pub struct StemStimulus {
    /// Unique stem identifier (e.g. `"star_center"`, `"sweep_ctrl_0"`, `"hadamard_diag_0"`).
    pub stem_id: String,
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Primary input channel `in_1` (`Inst In L` / `Mono`).
    pub in_1: Vec<f32>,
    /// Optional stereo right input channel `in_2` (`Inst In R`).
    pub in_2: Option<Vec<f32>>,
    /// Optional auxiliary/return left input channel `in_3` (`FX Loop Return L` / `Sidechain Key L`).
    pub in_3: Option<Vec<f32>>,
    /// Optional auxiliary/return right input channel `in_4` (`FX Loop Return R` / `Sidechain Key R`).
    pub in_4: Option<Vec<f32>>,
    /// Stationary control values `[0..P]` for this stem.
    pub control_values: Vec<f32>,
    /// Optional per-control continuous sweep trajectories (`len == P`, each `len == in_1.len()`).
    pub control_trajectories: Option<Vec<Vec<f32>>>,
    /// Optional `[n_earliest, n_latest)` sample window for timed `freeze_latch` transitions.
    pub freeze_transition_window: Option<[usize; 2]>,
}

/// Batch of [`StemStimulus`] descriptors for Sans-IO 1-or-2-pass distillation.
#[derive(Clone, Debug, PartialEq)]
pub struct StemStimulusBatch {
    /// Sample rate in Hz (`48_000` canonical).
    pub sample_rate_hz: u32,
    /// Ordered stems in this batch.
    pub stems: Vec<StemStimulus>,
}

/// Rendered multi-port audio response (plus optional white-box/grey-box internal circuit nodes)
/// produced by a [`DistillationTarget`].
#[derive(Clone, Debug, PartialEq)]
pub struct StemResponse {
    /// Stem or frame identifier matching the request.
    pub stem_id: String,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Primary output channel `out_1` (`Main Out L` / `Mono`).
    pub out_1: Vec<f32>,
    /// Optional stereo right output channel `out_2` (`Main Out R`).
    pub out_2: Option<Vec<f32>>,
    /// Optional auxiliary output channel `out_3` (`FX Loop Send L` or `WDW Dry Center`).
    pub out_3: Option<Vec<f32>>,
    /// Optional auxiliary output channel `out_4` (`FX Loop Send R`).
    pub out_4: Option<Vec<f32>>,
    /// Optional tapped internal circuit or stage node waveforms (`grey-box` / `white-box` distillation).
    pub internal_nodes: BTreeMap<String, Vec<f32>>,
}

/// Converter / hardware signal-path sample bit depth for multi-bit-depth device capture and
/// noise-qualified certification (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CaptureBitDepth {
    /// Vintage `12-bit` signed uniform PCM (`±2,048` steps, $\sigma_q \approx -77.0\text{ dBFS}$).
    Int12,
    /// Standard `16-bit` signed uniform PCM (`±32,768` steps, $\sigma_q \approx -101.1\text{ dBFS}$).
    Int16,
    /// Studio `24-bit` signed uniform PCM (`±8,388,608` steps, $\sigma_q \approx -149.3\text{ dBFS}$).
    Int24,
    /// High-resolution `32-bit` signed uniform PCM (`±2,147,483,648` steps, $\sigma_q \approx -197.4\text{ dBFS}$).
    Int32,
    /// Unquantized `32-bit` IEEE-754 floating-point (`f32`).
    #[default]
    Float32,
}

impl CaptureBitDepth {
    /// All five canonical converter bit-depth formats in ascending resolution order.
    pub const ALL: [Self; 5] = [
        Self::Int12,
        Self::Int16,
        Self::Int24,
        Self::Int32,
        Self::Float32,
    ];

    /// Nominal word length in bits (`12`, `16`, `24`, or `32`).
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Int12 => 12,
            Self::Int16 => 16,
            Self::Int24 => 24,
            Self::Int32 | Self::Float32 => 32,
        }
    }

    /// Returns `true` for fixed-point integer PCM formats (`Int12`, `Int16`, `Int24`, `Int32`).
    #[inline]
    #[must_use]
    pub const fn is_integer_pcm(self) -> bool {
        !matches!(self, Self::Float32)
    }

    /// Returns `true` for IEEE-754 floating-point representation (`Float32`).
    #[inline]
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float32)
    }

    /// Maps `(bits, is_float)` (`12`/`16`/`24`/`32` signed PCM or `32` IEEE-754 float) to [`CaptureBitDepth`].
    #[inline]
    #[must_use]
    pub const fn from_format(bits: u8, is_float: bool) -> Option<Self> {
        match (bits, is_float) {
            (12, false) => Some(Self::Int12),
            (16, false) => Some(Self::Int16),
            (24, false) => Some(Self::Int24),
            (32, false) => Some(Self::Int32),
            (32, true) => Some(Self::Float32),
            _ => None,
        }
    }

    /// Maps fixed-point signed integer PCM bit count (`12`, `16`, `24`, `32`) to [`CaptureBitDepth`].
    #[inline]
    #[must_use]
    pub const fn from_pcm_bits(bits: u8) -> Option<Self> {
        Self::from_format(bits, false)
    }

    /// Uniform PCM quantization step size $\Delta = 2^{-(B-1)}$ (`0.0` for `Float32`).
    #[inline]
    #[must_use]
    pub const fn lsb_step(self) -> f64 {
        match self {
            Self::Int12 => 1.0 / 2048.0,
            Self::Int16 => 1.0 / 32_768.0,
            Self::Int24 => 1.0 / 8_388_608.0,
            Self::Int32 => 1.0 / 2_147_483_648.0,
            Self::Float32 => 0.0,
        }
    }

    /// Worst-case deterministic mid-tread rounding error magnitude $\Delta / 2$ (`0.0` for `Float32`).
    #[inline]
    #[must_use]
    pub const fn peak_quantization_error(self) -> f64 {
        0.5 * self.lsb_step()
    }

    /// Exact `f64` positive two's-complement full-scale rail (`1.0 - Δ` for integer PCM, `1.0e4` headroom for `Float32`).
    #[inline]
    #[must_use]
    pub const fn positive_full_scale_f64(self) -> f64 {
        match self {
            Self::Int12 => 2047.0 / 2048.0,
            Self::Int16 => 32_767.0 / 32_768.0,
            Self::Int24 => 8_388_607.0 / 8_388_608.0,
            Self::Int32 => 2_147_483_647.0 / 2_147_483_648.0,
            Self::Float32 => 1.0e4,
        }
    }

    /// Positive two's-complement full-scale rail in `f32` (`< 1.0` for `Int12`/`Int16`/`Int24`,
    /// `1.0` for `Int32` due to `binary32` rounding of `1 - 2^-31`, `1.0e4` headroom for `Float32`).
    #[inline]
    #[must_use]
    pub const fn positive_full_scale(self) -> f32 {
        self.positive_full_scale_f64() as f32
    }

    /// Theoretical dithered uniform quantization error RMS $\sigma_q = \Delta / \sqrt{12}$ (`0.0` for `Float32`).
    #[inline]
    #[must_use]
    pub fn quantization_noise_rms(self) -> f64 {
        const INV_SQRT_12: f64 = 0.288_675_134_594_812_87;
        self.lsb_step() * INV_SQRT_12
    }

    /// Alias for [`Self::quantization_noise_rms`].
    #[inline]
    #[must_use]
    pub fn quantization_rms(self) -> f64 {
        self.quantization_noise_rms()
    }

    /// Quantizes and saturates sample `x` to this converter representation (`0.0 -> +0.0` exact).
    #[inline]
    #[must_use]
    pub fn quantize_sample(self, x: f32) -> f32 {
        let s = if x.is_finite() {
            x.clamp(-1.0e4, 1.0e4)
        } else {
            0.0
        };
        let scale = match self {
            Self::Int12 => 2048.0_f64,
            Self::Int16 => 32_768.0_f64,
            Self::Int24 => 8_388_608.0_f64,
            Self::Int32 => 2_147_483_648.0_f64,
            Self::Float32 => return if s == 0.0 { 0.0 } else { s },
        };
        let q = (f64::from(s) * scale).round().clamp(-scale, scale - 1.0);
        if q == 0.0 { 0.0 } else { (q / scale) as f32 }
    }

    /// Synthesizes a deterministic uniform `[-Δ/2, +Δ/2)` converter quantization floor trace of length
    /// `num_samples` (`0.0` trace for `Float32`) for selection-matched `bESR` / `MR-IIR` / `LTAS` / `SegESR`
    /// noise-floor calibration (`ref[qual.besr.finite-timeline]`, `ref[qual.besr.noise-qualified]`).
    #[must_use]
    pub fn synthesize_quantization_floor_trace(self, num_samples: usize, seed: u64) -> Vec<f64> {
        let step = self.lsb_step();
        if step == 0.0 {
            return vec![0.0_f64; num_samples];
        }
        (0..num_samples)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5;
                u * step
            })
            .collect()
    }

    /// Synthesizes a wideband Rademacher $\pm \sigma_{\text{eff}}$ floor trace combining deterministic
    /// undithered mid-tread peak error $\Delta / 2$ with `analog_floor_rms` in quadrature
    /// ($\sigma_{\text{eff}} = \sqrt{(\Delta / 2)^2 + \sigma_{\text{analog}}^2}$), guaranteeing exact
    /// frame-invariant RMS across short `20 ms` `SegESR` windows (`ref[qual.besr.noise-qualified]`).
    #[must_use]
    pub fn synthesize_undithered_floor_trace(
        self,
        num_samples: usize,
        analog_floor_rms: f64,
        seed: u64,
    ) -> Vec<f64> {
        let q_peak = self.peak_quantization_error();
        let a_rms = if analog_floor_rms.is_finite() && analog_floor_rms > 0.0 {
            analog_floor_rms
        } else {
            0.0
        };
        let eff_rms = q_peak.hypot(a_rms);
        if eff_rms == 0.0 {
            return vec![0.0_f64; num_samples];
        }
        (0..num_samples)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add(0xD6E8_FEB8_6659_FD93);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                if (z & 1) == 0 { eff_rms } else { -eff_rms }
            })
            .collect()
    }
}

/// Coherence-safe target trait implemented by [`DnamEngine`], [`DynamicDnamEngine`],
/// white-box circuit/DSP runners (`FaustRunner`, `SpiceRunner`), and black-box hardware/fixture adapters.
pub trait DistillationTarget {
    /// Resets all internal recursive filter, delay, and dynamic control states to resting DC equilibrium.
    fn reset_state(&mut self);

    /// Resets state to DC equilibrium and renders a complete [`StemStimulus`].
    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError>;

    /// Renders a single causal [`StimulusFrame`] **without** resetting state, enabling closed-loop
    /// frame-by-frame adaptive feedback excitation (`AdaptiveStimulusController::next_frame`).
    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        let stem = StemStimulus {
            stem_id: format!("frame_{}", frame.frame_index),
            sample_rate_hz: frame.sample_rate_hz,
            in_1: frame.in_1.clone(),
            in_2: frame.in_2.clone(),
            in_3: frame.in_3.clone(),
            in_4: frame.in_4.clone(),
            control_values: frame.control_values.clone(),
            control_trajectories: frame.control_trajectories.clone(),
            freeze_transition_window: None,
        };
        self.render_stem(&stem)
    }
}

struct EngineSliceSpec<'a> {
    stem_id: String,
    sample_rate_hz: u32,
    in_1: &'a [f32],
    in_2: Option<&'a Vec<f32>>,
    in_3: Option<&'a Vec<f32>>,
    in_4: Option<&'a Vec<f32>>,
    control_values: &'a [f32],
    control_trajectories: Option<&'a Vec<Vec<f32>>>,
    freeze_window: Option<[usize; 2]>,
    snap_initial_controls: bool,
}

fn validate_engine_stimulus_layout(spec: &EngineSliceSpec<'_>) -> Result<(), DnamError> {
    let n = spec.in_1.len();
    for (label, ch) in [
        ("in_2", spec.in_2),
        ("in_3", spec.in_3),
        ("in_4", spec.in_4),
    ] {
        if let Some(buf) = ch
            && buf.len() != n
        {
            return Err(DnamError::SchemaValidationError(format!(
                "{label} length ({}) must equal in_1 length ({n})",
                buf.len()
            )));
        }
    }
    if let Some(trajs) = spec.control_trajectories {
        if trajs.len() != spec.control_values.len() {
            return Err(DnamError::SchemaValidationError(format!(
                "control_trajectories count ({}) must equal control_values count ({})",
                trajs.len(),
                spec.control_values.len()
            )));
        }
        for (p, tr) in trajs.iter().enumerate() {
            if tr.len() != n {
                return Err(DnamError::SchemaValidationError(format!(
                    "control_trajectories[{p}] length ({}) must equal in_1 length ({n})",
                    tr.len()
                )));
            }
        }
    }
    if let Some([fs, fe]) = spec.freeze_window
        && (fs > fe || fe > n)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "invalid freeze_transition_window [{fs}, {fe}] for stem length {n}"
        )));
    }
    Ok(())
}

fn render_engine_slice<B: EngineBackend>(
    engine: &mut DnamEngine<'_, B>,
    spec: EngineSliceSpec<'_>,
) -> Result<StemResponse, DnamError> {
    let n = spec.in_1.len();
    validate_engine_stimulus_layout(&spec)?;
    if spec.sample_rate_hz != engine.sample_rate_hz() {
        engine.set_sample_rate(spec.sample_rate_hz)?;
    }

    if spec.snap_initial_controls {
        engine.set_freeze(false);
        engine.control_bus.reset_to_defaults();
        for idx in 0..engine.num_controls() {
            let init_v = spec
                .control_trajectories
                .and_then(|tr| tr.get(idx))
                .and_then(|v| v.first().copied())
                .or_else(|| spec.control_values.get(idx).copied());
            if let Some(v) = init_v {
                engine.set_control_immediate(idx, v);
            }
        }
        engine.reset_to_dc_equilibrium();
    } else if spec.control_trajectories.is_none() {
        for (idx, &val) in spec.control_values.iter().enumerate() {
            if idx < engine.num_controls() {
                engine.set_control_value(idx, val);
            }
        }
    }

    let mut out_1 = vec![0.0_f32; n];
    let mut out_2 = vec![0.0_f32; n];
    let has_aux =
        spec.in_3.is_some() || spec.in_4.is_some() || engine.stage_pipeline.stage0.wdw_mode;
    let mut out_3 = if has_aux {
        Some(vec![0.0_f32; n])
    } else {
        None
    };
    let mut out_4 = if spec.in_3.is_some() || spec.in_4.is_some() {
        Some(vec![0.0_f32; n])
    } else {
        None
    };

    let active_fw = spec.freeze_window.filter(|&[fs, fe]| fs < fe);
    if spec.control_trajectories.is_some() || active_fw.is_some() {
        for i in 0..n {
            if let Some([fs, fe]) = active_fw {
                engine.set_freeze(i >= fs && i < fe);
            }
            if let Some(trajs) = spec.control_trajectories {
                for (p, tr) in trajs.iter().enumerate() {
                    if p < engine.num_controls() {
                        engine.set_control_value(p, tr[i]);
                    }
                }
            }
            engine.process_block_io(DnamAudioIo {
                in_1: &spec.in_1[i..i + 1],
                in_2: spec.in_2.map(|v| &v[i..i + 1]),
                in_3: spec.in_3.map(|v| &v[i..i + 1]),
                in_4: spec.in_4.map(|v| &v[i..i + 1]),
                out_1: &mut out_1[i..i + 1],
                out_2: Some(&mut out_2[i..i + 1]),
                out_3: out_3.as_mut().map(|v| &mut v[i..i + 1]),
                out_4: out_4.as_mut().map(|v| &mut v[i..i + 1]),
            });
        }
        if active_fw.is_some() {
            engine.set_freeze(false);
        }
    } else {
        engine.process_block_io(DnamAudioIo {
            in_1: spec.in_1,
            in_2: spec.in_2.map(Vec::as_slice),
            in_3: spec.in_3.map(Vec::as_slice),
            in_4: spec.in_4.map(Vec::as_slice),
            out_1: &mut out_1,
            out_2: Some(&mut out_2),
            out_3: out_3.as_deref_mut(),
            out_4: out_4.as_deref_mut(),
        });
    }

    Ok(StemResponse {
        stem_id: spec.stem_id,
        sample_rate_hz: spec.sample_rate_hz,
        out_1,
        out_2: Some(out_2),
        out_3,
        out_4,
        internal_nodes: BTreeMap::new(),
    })
}

impl<'a, B: EngineBackend> DistillationTarget for DnamEngine<'a, B> {
    fn reset_state(&mut self) {
        self.control_bus.reset_to_defaults();
        self.reset_to_dc_equilibrium();
        self.set_freeze(false);
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        let spec = EngineSliceSpec {
            stem_id: stimulus.stem_id.clone(),
            sample_rate_hz: stimulus.sample_rate_hz,
            in_1: &stimulus.in_1,
            in_2: stimulus.in_2.as_ref(),
            in_3: stimulus.in_3.as_ref(),
            in_4: stimulus.in_4.as_ref(),
            control_values: &stimulus.control_values,
            control_trajectories: stimulus.control_trajectories.as_ref(),
            freeze_window: stimulus.freeze_transition_window,
            snap_initial_controls: true,
        };
        render_engine_slice(self, spec)
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        render_engine_slice(
            self,
            EngineSliceSpec {
                stem_id: format!("frame_{}", frame.frame_index),
                sample_rate_hz: frame.sample_rate_hz,
                in_1: &frame.in_1,
                in_2: frame.in_2.as_ref(),
                in_3: frame.in_3.as_ref(),
                in_4: frame.in_4.as_ref(),
                control_values: &frame.control_values,
                control_trajectories: frame.control_trajectories.as_ref(),
                freeze_window: None,
                snap_initial_controls: frame.frame_index == 0,
            },
        )
    }
}

impl<'a> DistillationTarget for DynamicDnamEngine<'a> {
    fn reset_state(&mut self) {
        match self {
            Self::Scalar(e) => e.reset_state(),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.reset_state(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(e) => e.reset_state(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.reset_state(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(e) => e.reset_state(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(e) => e.reset_state(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(e) => e.reset_state(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.reset_state(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.reset_state(),
        }
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        match self {
            Self::Scalar(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.render_stem(stimulus),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.render_stem(stimulus),
        }
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        match self {
            Self::Scalar(e) => e.render_frame(frame),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(e) => e.render_frame(frame),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(e) => e.render_frame(frame),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(e) => e.render_frame(frame),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(e) => e.render_frame(frame),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(e) => e.render_frame(frame),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(e) => e.render_frame(frame),
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(e) => e.render_frame(frame),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(e) => e.render_frame(frame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::Tier;
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::math::{BackendKind, ScalarBackend};
    use dnam_engine::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};

    struct DummyTarget;
    impl DistillationTarget for DummyTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
            Ok(StemResponse {
                stem_id: stimulus.stem_id.clone(),
                sample_rate_hz: stimulus.sample_rate_hz,
                out_1: stimulus.in_1.clone(),
                out_2: stimulus.in_2.clone(),
                out_3: stimulus.in_3.clone(),
                out_4: stimulus.in_4.clone(),
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_distillation_target_trait_implementations() {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = vec![ControlSchemaDef {
            id: String::from("gain"),
            name: String::from("Gain"),
            kind: ControlKind::Range,
            unit: Some(String::from("dB")),
            lobatto_degree: Some(2),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([1.0, 9.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        header.io_config.capabilities.supports_freeze = true;
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut dummy = DummyTarget;
        dummy.reset_state();
        let frame = StimulusFrame {
            frame_index: 7,
            sample_rate_hz: 96_000,
            in_1: vec![0.1, -0.1, 0.2, -0.2],
            in_2: Some(vec![0.05; 4]),
            in_3: Some(vec![0.02; 4]),
            in_4: Some(vec![-0.02; 4]),
            control_values: vec![4.0],
            control_trajectories: Some(vec![vec![3.0, 4.0, 5.0, 6.0]]),
        };
        let dummy_resp = dummy.render_frame(&frame).unwrap();
        assert_eq!(dummy_resp.stem_id, "frame_7");

        let mut eng =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng.reset_state();
        let stem = StemStimulus {
            stem_id: String::from("stem_0"),
            sample_rate_hz: 96_000,
            in_1: vec![0.1; 8],
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![5.0],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        let batch = StemStimulusBatch {
            sample_rate_hz: 96_000,
            stems: vec![stem.clone()],
        };
        assert_eq!(batch.stems.len(), 1);
        let stem_resp = eng.render_stem(&stem).unwrap();
        assert_eq!(stem_resp.stem_id, "stem_0");
        assert_eq!(stem_resp.out_1.len(), 8);
        assert!(stem_resp.out_3.is_none());

        let frame_resp = eng.render_frame(&frame).unwrap();
        assert_eq!(frame_resp.stem_id, "frame_7");
        assert!(frame_resp.out_3.is_some());
        assert!(frame_resp.out_4.is_some());

        for backend in [BackendKind::Scalar, BackendKind::Neon, BackendKind::ArmSme] {
            let mut dyn_eng = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Lite,
                48_000,
                backend,
            )
            .unwrap();
            dyn_eng.reset_state();
            let r1 = dyn_eng.render_stem(&stem).unwrap();
            assert_eq!(r1.out_1.len(), 8);
            let r2 = dyn_eng.render_frame(&frame).unwrap();
            assert_eq!(r2.out_1.len(), 4);
        }

        // CaptureBitDepth exhaustive format, quantization RMS, peak error, and floor trace tests
        assert_eq!(CaptureBitDepth::ALL.len(), 5);
        for bd in CaptureBitDepth::ALL {
            assert_eq!(
                CaptureBitDepth::from_format(bd.bits(), bd.is_float()),
                Some(bd)
            );
            if bd.is_integer_pcm() {
                assert_eq!(CaptureBitDepth::from_pcm_bits(bd.bits()), Some(bd));
                assert!(bd.peak_quantization_error() > bd.quantization_noise_rms());
            } else {
                assert_eq!(bd.peak_quantization_error(), 0.0);
                assert_eq!(bd.quantization_noise_rms(), 0.0);
            }
        }
        assert_eq!(CaptureBitDepth::from_pcm_bits(8), None);
        assert_eq!(CaptureBitDepth::from_format(24, true), None);
        assert_eq!(
            CaptureBitDepth::Float32.synthesize_quantization_floor_trace(16, 1),
            vec![0.0; 16]
        );
        assert_eq!(
            CaptureBitDepth::Float32.synthesize_undithered_floor_trace(16, 0.0, 1),
            vec![0.0; 16]
        );
        let tr32 = CaptureBitDepth::Int32.synthesize_quantization_floor_trace(64, 7);
        assert!(
            tr32.iter()
                .all(|&v| v.abs() <= CaptureBitDepth::Int32.peak_quantization_error())
        );
        let undith = CaptureBitDepth::Int16.synthesize_undithered_floor_trace(64, 1.0e-5, 9);
        assert!(undith.iter().all(|&v| v != 0.0));

        // CaptureBitDepth quantize_sample, positive_full_scale, & canonical +0.0 parity
        for bd in CaptureBitDepth::ALL {
            assert_eq!(bd.quantization_rms(), bd.quantization_noise_rms());
            assert_eq!(bd.quantize_sample(-0.0).to_bits(), 0);
            assert_eq!(bd.quantize_sample(f32::NAN).to_bits(), 0);
            if bd.is_integer_pcm() {
                assert!(bd.positive_full_scale_f64() < 1.0);
                assert_eq!(bd.quantize_sample(2.0), bd.positive_full_scale());
                assert_eq!(bd.quantize_sample(-2.0), -1.0);
            } else {
                assert_eq!(bd.positive_full_scale(), 1.0e4);
                assert_eq!(bd.quantize_sample(2.5), 2.5);
            }
        }

        // Validation guards, freeze_transition_window, and streaming vs batch trajectory parity
        let mut bad_in2 = stem.clone();
        bad_in2.in_2 = Some(vec![0.0; 3]);
        assert!(eng.render_stem(&bad_in2).is_err());
        let mut bad_traj_cnt = stem.clone();
        bad_traj_cnt.control_trajectories = Some(vec![]);
        assert!(eng.render_stem(&bad_traj_cnt).is_err());
        let mut bad_traj_len = stem.clone();
        bad_traj_len.control_trajectories = Some(vec![vec![5.0; 3]]);
        assert!(eng.render_stem(&bad_traj_len).is_err());
        let mut bad_fw = stem.clone();
        bad_fw.freeze_transition_window = Some([5, 3]);
        assert!(eng.render_stem(&bad_fw).is_err());

        let mut fw_stem = stem.clone();
        fw_stem.freeze_transition_window = Some([2, 6]);
        assert_eq!(eng.render_stem(&fw_stem).unwrap().out_1.len(), 8);

        let full_traj = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut traj_stem = stem.clone();
        traj_stem.control_values = vec![9.0];
        traj_stem.control_trajectories = Some(vec![full_traj.clone()]);
        let batch_out = eng.render_stem(&traj_stem).unwrap();
        eng.reset_state();
        let f0 = StimulusFrame {
            frame_index: 0,
            sample_rate_hz: 96_000,
            in_1: stem.in_1[..4].to_vec(),
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![9.0],
            control_trajectories: Some(vec![full_traj[..4].to_vec()]),
        };
        let f1 = StimulusFrame {
            frame_index: 1,
            sample_rate_hz: 96_000,
            in_1: stem.in_1[4..].to_vec(),
            in_2: None,
            in_3: None,
            in_4: None,
            control_values: vec![9.0],
            control_trajectories: Some(vec![full_traj[4..].to_vec()]),
        };
        let mut stream_out = eng.render_frame(&f0).unwrap().out_1;
        stream_out.extend(eng.render_frame(&f1).unwrap().out_1);
        assert_eq!(stream_out, batch_out.out_1);

        // DynamicDnamEngine::reset_state clears freeze and restores control_bus defaults;
        // render_stem snaps controls before equilibrating with zero cross-stem parameter bleed
        eng.set_freeze(true);
        eng.set_control_immediate(0, 9.0);
        assert!(eng.is_frozen());
        let mut dyn_eng = DynamicDnamEngine::Scalar(eng);
        DistillationTarget::reset_state(&mut dyn_eng);
        assert!(matches!(
            dyn_eng,
            DynamicDnamEngine::Scalar(ref e)
                if !e.is_frozen() && (e.control_bus.raw_values[0] - 5.0).abs() < 1.0e-6
        ));

        let mut fresh =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 96_000).unwrap();
        if let DynamicDnamEngine::Scalar(ref mut reused) = dyn_eng {
            let mut hot_stem = stem.clone();
            hot_stem.control_values = vec![9.0];
            let _ = reused.render_stem(&hot_stem).unwrap();
            let mut default_stem = stem.clone();
            default_stem.control_values = vec![];
            let out_reused = reused.render_stem(&default_stem).unwrap();
            let out_fresh = fresh.render_stem(&default_stem).unwrap();
            assert_eq!(out_reused.out_1, out_fresh.out_1);
            assert_eq!(reused.control_bus.raw_values, fresh.control_bus.raw_values);
        }
    }
}
