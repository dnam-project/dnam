//! `dnam-fixtures` — Pure-Rust Zero-Dependency Black-Box Ground-Truth Audio Signal-Chain
//! Models (`Chain-01..12`), Device/Control Metadata (`sample_rate_hz`, `bit_depth`, controls, switches),
//! and Composable Hardware Impairment Decorators.

#![forbid(unsafe_code)]

pub mod adversaries;
pub mod chains;
#[cfg(test)]
pub(crate) mod test_utils;

pub use adversaries as impairments;

/// Canonical default fixture device sample rate (`48,000 Hz`).
pub const DEFAULT_FIXTURE_SAMPLE_RATE_HZ: u32 = 48_000;

/// Minimum supported fixture device sample rate (`8,000 Hz`).
pub const MIN_FIXTURE_SAMPLE_RATE_HZ: u32 = 8_000;

/// Maximum supported fixture device sample rate (`192,000 Hz`).
pub const MAX_FIXTURE_SAMPLE_RATE_HZ: u32 = 192_000;

/// Clamps an arbitrary requested device sample rate into `[MIN_FIXTURE_SAMPLE_RATE_HZ, MAX_FIXTURE_SAMPLE_RATE_HZ]`.
#[inline]
#[must_use]
pub const fn clamp_sample_rate_hz(sample_rate_hz: u32) -> u32 {
    if sample_rate_hz < MIN_FIXTURE_SAMPLE_RATE_HZ {
        MIN_FIXTURE_SAMPLE_RATE_HZ
    } else if sample_rate_hz > MAX_FIXTURE_SAMPLE_RATE_HZ {
        MAX_FIXTURE_SAMPLE_RATE_HZ
    } else {
        sample_rate_hz
    }
}

/// Sanitizes an incoming audio sample so `NaN`, `±Inf`, or IEEE-754 subnormals (`|x| < f32::MIN_POSITIVE`)
/// never poison recursive IIR/ODE states, while preserving wide headroom (`±10_000.0`) for extreme-envelope tests.
#[inline]
#[must_use]
pub(crate) fn sanitize_sample(x: f32) -> f32 {
    if x.is_finite() {
        let clamped = x.clamp(-1.0e4, 1.0e4);
        if clamped.abs() < f32::MIN_POSITIVE {
            0.0
        } else {
            clamped
        }
    } else {
        0.0
    }
}

/// Sanitizes an `f64` internal audio sample to `[-10_000.0, 10_000.0]`, mapping non-finite values,
/// subnormals (`|x| < f64::MIN_POSITIVE`), and `-0.0` to canonical `+0.0`.
#[inline]
#[must_use]
pub(crate) fn sanitize_sample_f64(x: f64) -> f64 {
    if x.is_finite() {
        let clamped = x.clamp(-1.0e4, 1.0e4);
        if clamped.abs() < f64::MIN_POSITIVE {
            0.0
        } else {
            clamped
        }
    } else {
        0.0
    }
}

/// Sanitizes a normalized continuous control coordinate `u in [-1.0, 1.0]`, substituting `default_norm`
/// if `u` is `NaN` or infinite, and flushing subnormals (`|u| < f32::MIN_POSITIVE`) to `+0.0`.
#[inline]
#[must_use]
pub(crate) fn sanitize_norm(u: f32, default_norm: f32) -> f32 {
    let fallback = if default_norm.is_finite() {
        let c = default_norm.clamp(-1.0, 1.0);
        if c.abs() < f32::MIN_POSITIVE { 0.0 } else { c }
    } else {
        0.0
    };
    if u.is_finite() {
        let c = u.clamp(-1.0, 1.0);
        if c.abs() < f32::MIN_POSITIVE { 0.0 } else { c }
    } else {
        fallback
    }
}

/// Converter / signal-path sample bit depth exposed by a [`BlackBoxModel`] device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FixtureBitDepth {
    /// Vintage `12-bit` signed uniform PCM (`R-2R` / early digital delay & sampler units, `±2,048` steps).
    Int12,
    /// Standard `16-bit` signed uniform PCM (`CD` / early digital rack & USB pedal interfaces, `±32,768` steps).
    Int16,
    /// Studio `24-bit` signed uniform PCM (`±8,388,608` steps).
    Int24,
    /// High-resolution `32-bit` signed uniform PCM (`±2,147,483,648` steps).
    Int32,
    /// Unquantized `32-bit` IEEE-754 floating-point (`f32`, pure analog or floating-point DSP).
    #[default]
    Float32,
}

impl FixtureBitDepth {
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

    /// Resolves `(bits, is_float)` (`12`/`16`/`24`/`32` PCM or `32` IEEE-754 float) to [`FixtureBitDepth`].
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

    /// Resolves a signed integer PCM word length (`12`, `16`, `24`, or `32`) to [`FixtureBitDepth`].
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

    /// Theoretical uniform quantization noise RMS $\sigma_q = \Delta / \sqrt{12}$ (`0.0` for `Float32`).
    #[inline]
    #[must_use]
    pub fn quantization_noise_rms(self) -> f64 {
        const INV_SQRT_12: f64 = 0.288_675_134_594_812_87;
        self.lsb_step() * INV_SQRT_12
    }

    /// Convenience alias for [`Self::quantization_noise_rms`].
    #[inline]
    #[must_use]
    pub fn quantization_rms(self) -> f64 {
        self.quantization_noise_rms()
    }

    /// Quantizes `x` directly in `f64` to its exact two's-complement integer PCM code in
    /// `[-2^{B-1}, 2^{B-1} - 1]` without `f32` intermediate double-rounding, or returns `None`
    /// for unquantized [`Self::Float32`].
    #[inline]
    #[must_use]
    pub fn quantize_pcm_i32(self, x: f64) -> Option<i32> {
        let s = sanitize_sample_f64(x);
        let scale = match self {
            Self::Int12 => 2048.0_f64,
            Self::Int16 => 32_768.0_f64,
            Self::Int24 => 8_388_608.0_f64,
            Self::Int32 => 2_147_483_648.0_f64,
            Self::Float32 => return None,
        };
        let q = (s * scale).round_ties_even().clamp(-scale, scale - 1.0);
        Some(q as i32)
    }

    /// Quantizes and saturates `x` directly in `f64` (`0.0 -> 0.0` exact, zero `f32` pre-cast
    /// double-rounding on `Int24` or `Int32`).
    #[inline]
    #[must_use]
    pub fn quantize_sample_f64(self, x: f64) -> f64 {
        let s = sanitize_sample_f64(x);
        let scale = match self {
            Self::Int12 => 2048.0_f64,
            Self::Int16 => 32_768.0_f64,
            Self::Int24 => 8_388_608.0_f64,
            Self::Int32 => 2_147_483_648.0_f64,
            Self::Float32 => return s,
        };
        let q = (s * scale).round_ties_even().clamp(-scale, scale - 1.0);
        if q == 0.0 { 0.0 } else { q / scale }
    }

    /// Quantizes an `f64` internal sample directly at converter resolution before narrowing to `f32`,
    /// eliminating `f64 -> f32` pre-cast double-rounding bias and flushing `f32` subnormals to `+0.0`.
    #[inline]
    #[must_use]
    pub fn quantize_f64_to_f32(self, x: f64) -> f32 {
        let y = self.quantize_sample_f64(x) as f32;
        if y.abs() < f32::MIN_POSITIVE { 0.0 } else { y }
    }

    /// Quantizes and saturates sample `x` to this device's converter representation (`0.0 -> 0.0` exact).
    #[inline]
    #[must_use]
    pub fn quantize_sample(self, x: f32) -> f32 {
        let s = sanitize_sample(x);
        if self == Self::Float32 {
            s
        } else {
            self.quantize_f64_to_f32(f64::from(s))
        }
    }
}

/// Metadata descriptor for a continuous control parameter (`u_p in [-1.0, 1.0]`) on a fixture device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixtureControlSpec {
    /// Canonical control identifier (e.g. `"drive"`, `"tone"`, `"level"`).
    pub id: &'static str,
    /// Human-readable control label (e.g. `"Drive"`, `"Tone"`, `"Level"`).
    pub name: &'static str,
    /// Physical unit label (e.g. `"%"`, `"dB"`, `"Hz"`, `"ms"`, `""`).
    pub unit: &'static str,
    /// Default normalized position in `[-1.0, 1.0]`.
    pub default_norm: f32,
    /// Minimum physical value corresponding to `u = -1.0`.
    pub min_physical: f32,
    /// Maximum physical value corresponding to `u = +1.0`.
    pub max_physical: f32,
}

impl FixtureControlSpec {
    /// Maps a normalized control coordinate `norm in [-1.0, 1.0]` to `[min_physical, max_physical]`.
    #[inline]
    #[must_use]
    pub fn norm_to_physical(&self, norm: f32) -> f32 {
        let u = sanitize_norm(norm, self.default_norm);
        let t = 0.5 * (u + 1.0);
        self.min_physical + t * (self.max_physical - self.min_physical)
    }

    /// Maps a physical control value in `[min_physical, max_physical]` to normalized `[-1.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn physical_to_norm(&self, physical: f32) -> f32 {
        let span = self.max_physical - self.min_physical;
        if !physical.is_finite() || !span.is_finite() || span.abs() <= 1.0e-12 {
            return sanitize_norm(self.default_norm, 0.0);
        }
        let t = ((physical - self.min_physical) / span).clamp(0.0, 1.0);
        2.0 * t - 1.0
    }
}

/// Metadata descriptor for a discrete multi-throw switch on a fixture device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixtureSwitchSpec {
    /// Canonical switch identifier (e.g. `"bright"`, `"mode"`, `"freeze"`).
    pub id: &'static str,
    /// Human-readable switch label (e.g. `"Bright"`, `"Filter Mode"`, `"Freeze Latch"`).
    pub name: &'static str,
    /// Ordered discrete position labels (e.g. `&["LP", "BP", "HP"]`).
    pub options: &'static [&'static str],
    /// Default zero-based switch index (`0..options.len()`).
    pub default_index: u8,
}

/// Device-intrinsic metadata (`sample_rate_hz`, `bit_depth`, controls, switches, category, and tail decay)
/// exposed by a [`BlackBoxModel`] physical signal-chain or hardware fault fixture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixtureDeviceMetadata {
    /// Canonical fixture identifier (e.g. `"Chain-01-Symmetric-OD"`).
    pub id: &'static str,
    /// Human-readable device name.
    pub name: &'static str,
    /// Device archetype category (e.g. `"overdrive"`, `"fuzz"`, `"tube_amp"`, `"cabinet"`, `"full_rig"`).
    pub category: &'static str,
    /// Active operating sample rate in Hz (e.g. `44_100`, `48_000`, `88_200`, `96_000`).
    pub sample_rate_hz: u32,
    /// Active converter / signal-path bit depth (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`).
    pub bit_depth: FixtureBitDepth,
    /// Expected impulse/reverb tail decay duration in milliseconds.
    pub expected_tail_ms: f32,
    /// Whether the device supports host tempo synchronization (`BPM` / rhythmic subdivisions).
    pub supports_tempo_sync: bool,
    /// Continuous control descriptors (`len == num_controls()`).
    pub controls: &'static [FixtureControlSpec],
    /// Discrete switch descriptors (`len == num_switches()`).
    pub switches: &'static [FixtureSwitchSpec],
}

impl FixtureDeviceMetadata {
    /// Returns a vector of default normalized continuous control values (`[-1.0, 1.0]`).
    #[must_use]
    pub fn default_controls(&self) -> Vec<f32> {
        self.controls.iter().map(|c| c.default_norm).collect()
    }

    /// Returns a vector of default discrete switch indices (`u8`).
    #[must_use]
    pub fn default_switches(&self) -> Vec<u8> {
        self.switches.iter().map(|s| s.default_index).collect()
    }
}

/// Clean black-box audio device interface for ground-truth physical signal chains and
/// composable hardware impairment decorators, accompanied by device-intrinsic metadata
/// for sample rate, bit depth, controls, switches, and ports.
pub trait BlackBoxModel {
    /// Resets all internal delay lines, filter states, and envelopes to DC equilibrium
    /// while preserving configured device `sample_rate_hz` and `bit_depth`.
    fn reset(&mut self);

    /// Returns device-intrinsic metadata (`sample_rate_hz`, `bit_depth`, controls, switches, category, tail).
    fn metadata(&self) -> FixtureDeviceMetadata;

    /// Active operating sample rate of this device in Hz (`mirrors metadata().sample_rate_hz`).
    fn sample_rate_hz(&self) -> u32 {
        self.metadata().sample_rate_hz
    }

    /// Active converter / signal-path bit depth of this device (`mirrors metadata().bit_depth`).
    fn bit_depth(&self) -> FixtureBitDepth {
        self.metadata().bit_depth
    }

    /// Reconfigures the device's operating sample rate (`8_000..=192_000 Hz`) and resets internal DSP state.
    fn set_sample_rate_hz(&mut self, _sample_rate_hz: u32) {
        self.reset();
    }

    /// Reconfigures the device's operating bit depth and resets internal DSP state.
    fn set_bit_depth(&mut self, _bit_depth: FixtureBitDepth) {
        self.reset();
    }

    /// Number of input audio channels (`1` for mono, `2` for stereo or main + sidechain).
    fn num_inputs(&self) -> usize;

    /// Number of output audio channels (`1` for mono, `2` for stereo).
    fn num_outputs(&self) -> usize;

    /// Number of continuous control parameters (`u_p in [-1.0, 1.0]`).
    fn num_controls(&self) -> usize {
        self.metadata().controls.len()
    }

    /// Number of discrete multi-throw switches.
    fn num_switches(&self) -> usize {
        self.metadata().switches.len()
    }

    /// Processes a single multi-channel sample frame at [`Self::sample_rate_hz`].
    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    );

    /// Processes a block of non-interleaved input/output channel slices at [`Self::sample_rate_hz`].
    fn process_block(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
        controls: &[f32],
        switches: &[u8],
    ) {
        let n_in = self.num_inputs().min(inputs.len()).min(4);
        let model_out = self.num_outputs().min(4);
        let copy_out = model_out.min(outputs.len());
        let out_frames = if copy_out == 0 {
            outputs.iter().map(|ch| ch.len()).min().unwrap_or(0)
        } else {
            outputs
                .iter()
                .take(copy_out)
                .map(|ch| ch.len())
                .min()
                .unwrap_or(0)
        };
        let n_frames = if self.num_inputs() == 0 {
            out_frames
        } else if n_in == 0 {
            0
        } else {
            inputs
                .iter()
                .take(n_in)
                .map(|ch| ch.len())
                .min()
                .unwrap_or(0)
                .min(out_frames)
        };
        for extra_ch in outputs.iter_mut().skip(copy_out) {
            let fill_len = n_frames.min(extra_ch.len());
            extra_ch[..fill_len].fill(0.0);
        }
        let mut in_frame = [0.0_f32; 4];
        let mut out_frame = [0.0_f32; 4];

        for f in 0..n_frames {
            for (c, slot) in in_frame.iter_mut().enumerate().take(n_in) {
                *slot = inputs
                    .get(c)
                    .and_then(|ch| ch.get(f))
                    .copied()
                    .map(sanitize_sample)
                    .unwrap_or(0.0);
            }
            out_frame[..model_out].fill(0.0);
            self.process_frame(
                &in_frame[..n_in],
                &mut out_frame[..model_out],
                controls,
                switches,
            );
            for (c, &val) in out_frame.iter().enumerate().take(copy_out) {
                if let Some(ch) = outputs.get_mut(c)
                    && let Some(dst) = ch.get_mut(f)
                {
                    *dst = val;
                }
            }
        }
    }
}

pub use BlackBoxModel as AudioFixture;

pub use adversaries::{
    AdcClipAndPadDecorator, CapsuleClipMetrics, ConverterLoopbackDecorator, DigitalQuirksDecorator,
    GermaniumTempDriftDecorator, HardwareNoiseGateDecorator, HumanHandTrajectoryDecorator,
    IdleSquealDecorator, MainsHumAndNoiseFloorDecorator, MicrophoneCapsuleClipDecorator,
    PolarityAndDcOffsetDecorator, PotTaperAndSwitchDecorator,
    ScratchyPotAndSteppedAttenuatorDecorator, SpringChirpDecorator, TaperCurveKind,
    TubeThermalHysteresisDecorator,
};
pub use chains::{
    CANONICAL_SUBDIVISION_RATIOS, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz,
    Chain03ParallelCleanBlend, Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom,
    Chain06EnvelopeResonantSvf, Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer,
    Chain09La2aOpticalCompressor, Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron,
    Chain12FullEndToEndStereoRig, Chain12RigMode,
};

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroInputGeneratorFixture;

    const ZERO_GEN_META: FixtureDeviceMetadata = FixtureDeviceMetadata {
        id: "Zero-Input-Generator",
        name: "Zero Input Test Generator",
        category: "generator",
        sample_rate_hz: 44_100,
        bit_depth: FixtureBitDepth::Int16,
        expected_tail_ms: 0.0,
        supports_tempo_sync: false,
        controls: &[FixtureControlSpec {
            id: "level",
            name: "Level",
            unit: "dB",
            default_norm: 0.0,
            min_physical: -20.0,
            max_physical: 20.0,
        }],
        switches: &[],
    };

    impl BlackBoxModel for ZeroInputGeneratorFixture {
        fn reset(&mut self) {}
        fn metadata(&self) -> FixtureDeviceMetadata {
            ZERO_GEN_META
        }
        fn num_inputs(&self) -> usize {
            0
        }
        fn num_outputs(&self) -> usize {
            1
        }
        fn process_frame(
            &mut self,
            inputs: &[f32],
            outputs: &mut [f32],
            controls: &[f32],
            _switches: &[u8],
        ) {
            assert!(inputs.is_empty());
            if let Some(dst) = outputs.first_mut() {
                let raw = sanitize_norm(controls.first().copied().unwrap_or(0.5), 0.5);
                *dst = self.bit_depth().quantize_sample(raw);
            }
        }
    }

    #[test]
    fn test_bit_depth_sample_rate_control_spec_and_zero_input_process_block() {
        assert_eq!(clamp_sample_rate_hz(100), MIN_FIXTURE_SAMPLE_RATE_HZ);
        assert_eq!(clamp_sample_rate_hz(44_100), 44_100);
        assert_eq!(clamp_sample_rate_hz(999_999), MAX_FIXTURE_SAMPLE_RATE_HZ);

        assert_eq!(FixtureBitDepth::ALL.len(), 5);
        assert_eq!(FixtureBitDepth::Int12.bits(), 12);
        assert_eq!(FixtureBitDepth::Int16.bits(), 16);
        assert_eq!(FixtureBitDepth::Int24.bits(), 24);
        assert_eq!(FixtureBitDepth::Int32.bits(), 32);
        assert_eq!(FixtureBitDepth::Float32.bits(), 32);
        assert!(FixtureBitDepth::Int16.is_integer_pcm());
        assert!(FixtureBitDepth::Int32.is_integer_pcm());
        assert!(!FixtureBitDepth::Int32.is_float());
        assert!(!FixtureBitDepth::Float32.is_integer_pcm());
        assert!(FixtureBitDepth::Float32.is_float());
        for bd in FixtureBitDepth::ALL {
            assert_eq!(
                FixtureBitDepth::from_format(bd.bits(), bd.is_float()),
                Some(bd)
            );
            assert_eq!(bd.quantize_sample(0.0), 0.0);
            assert_eq!(bd.quantize_sample(f32::NAN), 0.0);
            assert_eq!(bd.quantize_sample(f32::INFINITY), 0.0);
            assert_eq!(bd.quantize_sample(f32::NEG_INFINITY), 0.0);
            assert_eq!(bd.quantize_sample(0.25), 0.25);
            if bd.is_integer_pcm() {
                assert_eq!(bd.quantize_sample(-1.0), -1.0);
                assert_eq!(bd.quantize_sample(1.0), bd.positive_full_scale());
                assert_eq!(bd.quantize_sample(2.0), bd.positive_full_scale());
                assert_eq!(bd.quantize_sample(-2.0), -1.0);
            } else {
                assert_eq!(bd.quantize_sample(-1.0), -1.0);
                assert_eq!(bd.quantize_sample(1.0), 1.0);
            }
        }
        assert_eq!(
            FixtureBitDepth::from_pcm_bits(12),
            Some(FixtureBitDepth::Int12)
        );
        assert_eq!(
            FixtureBitDepth::from_pcm_bits(16),
            Some(FixtureBitDepth::Int16)
        );
        assert_eq!(
            FixtureBitDepth::from_pcm_bits(24),
            Some(FixtureBitDepth::Int24)
        );
        assert_eq!(
            FixtureBitDepth::from_pcm_bits(32),
            Some(FixtureBitDepth::Int32)
        );
        assert_eq!(
            FixtureBitDepth::from_format(32, true),
            Some(FixtureBitDepth::Float32)
        );
        assert_eq!(FixtureBitDepth::from_pcm_bits(8), None);
        assert_eq!(FixtureBitDepth::from_format(8, false), None);
        assert_eq!(FixtureBitDepth::from_format(24, true), None);
        assert!(
            FixtureBitDepth::Int12.quantization_noise_rms()
                > FixtureBitDepth::Int16.quantization_noise_rms()
        );
        assert!(
            FixtureBitDepth::Int16.quantization_noise_rms()
                > FixtureBitDepth::Int24.quantization_noise_rms()
        );
        assert!(
            FixtureBitDepth::Int24.quantization_noise_rms()
                > FixtureBitDepth::Int32.quantization_noise_rms()
        );
        assert!(FixtureBitDepth::Int32.quantization_noise_rms() > 0.0);
        assert_eq!(FixtureBitDepth::Float32.quantization_noise_rms(), 0.0);
        assert_eq!(
            FixtureBitDepth::Int12.quantize_sample(1.0 / 2048.0),
            1.0 / 2048.0
        );
        assert!(FixtureBitDepth::Int12.positive_full_scale() < 1.0);
        assert!(FixtureBitDepth::Int16.positive_full_scale() < 1.0);
        assert!(FixtureBitDepth::Int24.positive_full_scale() < 1.0);
        assert!(FixtureBitDepth::Int32.positive_full_scale_f64() < 1.0);
        assert_eq!(FixtureBitDepth::Int32.positive_full_scale(), 1.0);
        assert!(FixtureBitDepth::Float32.positive_full_scale() > 1.0);
        assert_eq!(
            FixtureBitDepth::Int16.quantize_sample(2.0),
            FixtureBitDepth::Int16.positive_full_scale()
        );
        assert_eq!(FixtureBitDepth::Int32.quantize_sample(2.0), 1.0);
        assert_eq!(FixtureBitDepth::Int32.quantize_sample(-2.0), -1.0);
        assert_eq!(FixtureBitDepth::Int32.quantize_sample(1.0e-11), 0.0);
        for bd in FixtureBitDepth::ALL {
            assert_eq!(bd.quantization_rms(), bd.quantization_noise_rms());
            assert_eq!(bd.quantize_sample(-0.0_f32).to_bits(), 0);
            if bd.is_integer_pcm() {
                assert_eq!(bd.quantize_sample(-1.0e-11_f32).to_bits(), 0);
            }
        }
        assert_eq!(sanitize_sample(-0.0_f32).to_bits(), 0);
        assert_eq!(
            FixtureBitDepth::Int32.quantize_sample(3.5e-10),
            (1.0 / 2_147_483_648.0) as f32
        );
        assert_eq!(FixtureBitDepth::Int16.quantize_sample(-2.0), -1.0);
        assert_eq!(
            FixtureBitDepth::Float32.quantize_sample(0.123_456),
            0.123_456
        );

        let spec = &ZERO_GEN_META.controls[0];
        assert!((spec.norm_to_physical(-1.0) - (-20.0)).abs() < 1.0e-6);
        assert!((spec.norm_to_physical(0.0) - 0.0).abs() < 1.0e-6);
        assert!((spec.norm_to_physical(1.0) - 20.0).abs() < 1.0e-6);
        assert!((spec.norm_to_physical(f32::NAN) - 0.0).abs() < 1.0e-6);
        assert!((spec.physical_to_norm(0.0) - 0.0).abs() < 1.0e-6);
        assert!((spec.physical_to_norm(20.0) - 1.0).abs() < 1.0e-6);
        assert_eq!(spec.physical_to_norm(f32::NAN), 0.0);
        assert_eq!(sanitize_sample(f32::NAN), 0.0);
        assert_eq!(sanitize_sample(f32::INFINITY), 0.0);
        assert_eq!(sanitize_sample_f64(f64::NAN), 0.0);
        assert_eq!(sanitize_sample_f64(f64::INFINITY), 0.0);
        assert_eq!(sanitize_sample_f64(-0.0_f64).to_bits(), 0);
        assert_eq!(sanitize_norm(f32::NAN, f32::NAN), 0.0);

        // Exact f64 quantization: no f64 -> f32 pre-cast double-rounding on Int24 / Int32
        let mid_adj_24 = (4_194_304.0 + 0.26) / 8_388_608.0;
        assert_eq!(
            FixtureBitDepth::Int24.quantize_pcm_i32(mid_adj_24),
            Some(4_194_304)
        );
        assert_eq!(
            FixtureBitDepth::Int24.quantize_f64_to_f32(mid_adj_24),
            (4_194_304.0 / 8_388_608.0) as f32
        );
        assert_eq!(FixtureBitDepth::Int32.quantize_pcm_i32(2.0), Some(i32::MAX));
        assert_eq!(
            FixtureBitDepth::Int32.quantize_pcm_i32(-2.0),
            Some(i32::MIN)
        );
        assert_eq!(FixtureBitDepth::Float32.quantize_pcm_i32(0.5), None);
        assert_eq!(FixtureBitDepth::Float32.quantize_sample_f64(0.5), 0.5);

        let degen_spec = FixtureControlSpec {
            id: "fixed",
            name: "Fixed",
            unit: "",
            default_norm: 0.0,
            min_physical: 5.0,
            max_physical: 5.0,
        };
        assert_eq!(degen_spec.physical_to_norm(5.0), 0.0);

        let mut generator = ZeroInputGeneratorFixture;
        generator.reset();
        generator.set_sample_rate_hz(48_000);
        generator.set_bit_depth(FixtureBitDepth::Int24);
        assert_eq!(generator.sample_rate_hz(), 44_100);
        assert_eq!(generator.bit_depth(), FixtureBitDepth::Int16);
        assert_eq!(generator.num_controls(), 1);
        assert_eq!(generator.num_switches(), 0);
        assert_eq!(generator.metadata().default_controls(), vec![0.0]);
        assert!(generator.metadata().default_switches().is_empty());

        let mut out = vec![0.0_f32; 8];
        generator.process_block(&[], &mut [&mut out], &[0.75], &[]);
        assert_eq!(out, vec![0.75_f32; 8]);

        // Zero-input and mono input isolation on input-inspecting decorators
        let mut pol_gen = PolarityAndDcOffsetDecorator::new(ZeroInputGeneratorFixture, false, 0.0);
        let mut pol_out = [0.0_f32; 1];
        pol_gen.process_frame(&[0.5], &mut pol_out, &[0.75], &[]);
        assert_eq!(pol_out[0], 0.75);
        let mut gate_gen = HardwareNoiseGateDecorator::new(ZeroInputGeneratorFixture);
        gate_gen.process_frame(&[0.5], &mut pol_out, &[0.75], &[]);
        assert_eq!(pol_out[0], 0.75);
        let mut squeal_mono = IdleSquealDecorator::new(Chain01SymmetricOd::new());
        let mut sq_f0 = [0.0_f32; 1];
        let mut sq_f1 = [0.0_f32; 1];
        squeal_mono.process_frame(&[0.0, 0.5], &mut sq_f0, &[0.95, 0.0, 0.0], &[]);
        squeal_mono.process_frame(&[0.0, 0.5], &mut sq_f1, &[0.95, 0.0, 0.0], &[]);
        assert!(sq_f1[0].abs() > 1.0e-4);

        // 1 -> 2 shared-supply model preserves channel-0 output when process_block receives 1 output slice
        let mut sag_stereo = TubeThermalHysteresisDecorator::new(Chain05PureLtiCabAndRoom::new());
        let dry: Vec<f32> = (0..64).map(|i| 0.7 * ((i as f32) * 0.25).sin()).collect();
        let mut out_l_full = vec![0.0_f32; 64];
        let mut out_r_full = vec![0.0_f32; 64];
        sag_stereo.process_block(
            &[&dry],
            &mut [&mut out_l_full, &mut out_r_full],
            &[0.5],
            &[],
        );
        sag_stereo.reset();
        let mut out_l_single = vec![0.0_f32; 64];
        sag_stereo.process_block(&[&dry], &mut [&mut out_l_single], &[0.5], &[]);
        assert_eq!(out_l_single, out_l_full);

        // DigitalQuirksDecorator new() vs set_bit_depth(d.bit_depth()) 0-ULP parity across non-zero tail
        let mut dq = DigitalQuirksDecorator::new(Chain01SymmetricOd::new());
        let dry128: Vec<f32> = (0..128).map(|i| 0.5 * ((i as f32) * 0.3).sin()).collect();
        let mut y_fresh = vec![0.0_f32; 128];
        dq.process_block(&[&dry128], &mut [&mut y_fresh], &[0.8, 0.2, 0.5], &[]);
        let bd = dq.bit_depth();
        dq.set_bit_depth(bd);
        let mut y_reapplied = vec![0.0_f32; 128];
        dq.process_block(&[&dry128], &mut [&mut y_reapplied], &[0.8, 0.2, 0.5], &[]);
        assert_eq!(y_fresh, y_reapplied);
    }
}
