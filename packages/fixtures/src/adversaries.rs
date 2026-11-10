//! Composable Black-Box Hardware & Interface Impairment Decorators (`adversaries` / `impairments`).
//!
//! Organized into domain submodules:
//! - [`interface_and_control`]: ADC clipping/pad, converter loopback (latency, HPF, clock drift, USB drop, converter rate/bit-depth), human hand trajectory, pot tapers & switches, polarity/DC offset & numeric fault sanitization (`Cases #1, #2, #4, #7, #11`)
//! - [`thermal_and_noise`]: Tube sag & thermal hysteresis, germanium drift, mains hum & uncorrelated wideband floor, hysteretic hardware noise gate (`Cases #5, #14, #15, #16`)
//! - [`mechanical_and_capsule`]: Spring dispersion chirp, digital quirks (`12`-bit `R-2R`, `28 kHz` ZOH, `1.8 ms` latency), scratchy pot & `12`-step attenuator, envelope-quenched idle squeal, and sub-`0 dBFS` mic capsule clip + LF `X_max` stiffening (`Cases #18, #20, #22, #25, #26`)

#![forbid(unsafe_code)]

pub mod interface_and_control;
pub mod mechanical_and_capsule;
pub mod thermal_and_noise;

pub use interface_and_control::*;
pub use mechanical_and_capsule::*;
pub use thermal_and_noise::*;

pub(crate) fn resolve_controls_with_defaults<M: crate::BlackBoxModel>(
    model: &M,
    controls: &[f32],
) -> Vec<f32> {
    let meta = model.metadata();
    let n_c = meta.controls.len().max(controls.len());
    let mut out = Vec::with_capacity(n_c);
    for p in 0..n_c {
        let default_norm = meta.controls.get(p).map_or(0.0, |c| c.default_norm);
        let raw = controls.get(p).copied().unwrap_or(default_norm);
        out.push(crate::sanitize_norm(raw, default_norm));
    }
    out
}

#[inline]
pub(crate) fn init_inner_float32<M: crate::BlackBoxModel>(
    mut inner: M,
) -> (M, crate::FixtureBitDepth) {
    let bd = inner.bit_depth();
    inner.set_bit_depth(crate::FixtureBitDepth::Float32);
    (inner, bd)
}

#[inline]
pub(crate) fn metadata_with_bit_depth<M: crate::BlackBoxModel>(
    inner: &M,
    bit_depth: crate::FixtureBitDepth,
) -> crate::FixtureDeviceMetadata {
    let mut meta = inner.metadata();
    meta.bit_depth = bit_depth;
    meta
}

#[inline]
pub(crate) fn step_inner_frame<M: crate::BlackBoxModel>(
    inner: &mut M,
    inputs: &[f32],
    controls: &[f32],
    switches: &[u8],
    out_buf: &mut [f32; 4],
) -> usize {
    let n_in = inputs.len().min(inner.num_inputs()).min(4);
    let mut in_buf = [0.0_f32; 4];
    for (dst, &src) in in_buf.iter_mut().zip(inputs.iter()).take(n_in) {
        *dst = crate::sanitize_sample(src);
    }
    let model_out = inner.num_outputs().min(4);
    out_buf[..model_out].fill(0.0);
    inner.process_frame(
        &in_buf[..n_in],
        &mut out_buf[..model_out],
        controls,
        switches,
    );
    model_out
}

#[inline]
pub(crate) fn write_outputs(outputs: &mut [f32], out_buf: &[f32; 4], model_out: usize) {
    let copy_out = outputs.len().min(model_out);
    for (dst, &src) in outputs[..copy_out].iter_mut().zip(&out_buf[..copy_out]) {
        *dst = if src.abs() < f32::MIN_POSITIVE {
            0.0
        } else {
            src
        };
    }
    if outputs.len() > copy_out {
        outputs[copy_out..].fill(0.0);
    }
}

macro_rules! impl_output_decorator_common {
    ($bd_field:ident) => {
        fn metadata(&self) -> crate::FixtureDeviceMetadata {
            crate::adversaries::metadata_with_bit_depth(&self.inner, self.$bd_field)
        }

        fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
            self.inner.set_sample_rate_hz(sample_rate_hz);
            self.reset();
        }

        fn set_bit_depth(&mut self, bit_depth: crate::FixtureBitDepth) {
            self.$bd_field = bit_depth;
            self.inner.set_bit_depth(crate::FixtureBitDepth::Float32);
            self.reset();
        }

        fn num_inputs(&self) -> usize {
            self.inner.num_inputs()
        }

        fn num_outputs(&self) -> usize {
            self.inner.num_outputs()
        }
    };
}

pub(crate) use impl_output_decorator_common;
