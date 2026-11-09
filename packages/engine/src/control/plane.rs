//! Canonical v2.0 3 kHz Self-Similar Control Plane, Exact-Integer Sufficient-Statistic
//! Accumulator (`9-word S` + `18-word D`), SPSC Per-Producer Ingress, Host Curve & Timestamp
//! Arbitration, `Q64`/`Q32` Tempo Synchronization, and Physical `f32` Calibration Inversion
//! (`src/control/plane.rs`).
//!
//! Implements:
//! - `ref[dsp.ctrl.op-m.ballistic-lfo]`, `ref[dsp.ctrl.op-m.rate-retime]`,
//!   `ref[dsp.ctrl.op-m.timing]`, `ref[dsp.ctrl.op-m.spsc-ingress]`
//! - `ref[dsp.ctrl.op-n.engine]`, `ref[dsp.ctrl.switch.option-contributors]`,
//!   `ref[dsp.ctrl.switch.conditional-batch]`, `ref[dsp.ctrl.accum.cp1-scaling]`,
//!   `ref[dsp.ctrl.accum.anova-fm]`, `ref[dsp.ctrl.accum.exact-int]`,
//!   `ref[dsp.op-n.ctrl.incremental-update]`, `ref[dsp.op-n.ctrl.fm-interaction]`
//! - `ref[dsp.ctrl.modulators.dynamic]`, `ref[dsp.ctrl.dispatch.matrix]`,
//!   `ref[dsp.ctrl.dispatch.micro-ramp]`, `ref[dsp.ctrl.ledger.memory-compute]`
//! - `ref[dsp.tempo.subdivisions]`, `ref[dsp.tempo.q64-q32-phase]`,
//!   `ref[dsp.tempo.event-tracker]`, `ref[dsp.tempo.physical-inversion]`,
//!   `ref[dsp.tempo.coprime-fdn]`, `ref[dsp.tempo.transport-jump]`
//! - `ref[dsp.zoh.ctrl-stride]`
//!
//! Submodules:
//! - [`stride_and_ingress`]: Control stride, window retiming, stage dispatch matrix, bounded SPSC ingress, and host curve arbitration.
//! - [`exact_int`]: `9-word S` + `18-word D` exact-integer sufficient-statistic accumulator (`ExactIntFactorAccumulator`).
//! - [`tempo`]: `Q64`/`Q32` tempo synchronization, clock pulse tracker, and physical `f32` calibration inversion.

pub mod exact_int;
#[cfg(test)]
mod exact_int_verify;
pub mod stride_and_ingress;
pub mod tempo;

pub use exact_int::*;
pub use stride_and_ingress::*;
pub use tempo::*;
