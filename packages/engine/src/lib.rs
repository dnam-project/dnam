//! `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time Stereo Neural Audio Modeling & Differentiable DSP Core.
//!
//! Unifies the three subsystems of **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")**:
//! 1. **Core 1 ([`core1`]):** 12-layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + Radial-Polar
//!    `SelectiveRgLru` stack with 3-state DC resting equilibrium (`0.000000` Sample-0 transient).
//! 2. **Core 2 ([`core2`]):** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network
//!    (`M-UFDN`) with dynamic 64B-aligned `fp16` delay buffers (`0 KB` when bypassed), 64-sample TZF
//!    reference tap, 4-tap Cubic Lagrange interpolation, 128-sample switch crossfades, Givens+Householder
//!    unitary mixing, and Coercive-Saturating Lyapunov loop (`S'(0) == 1.0000`).
//! 3. **Control Bus ([`control`]):** $3\text{ kHz}$ Envelopes/LFOs + Node-Anchored Lobatto HDMR & Rank-4
//!    Factorization Machine with 64-byte [`ControlClampDescriptor`](control::ControlClampDescriptor)
//!    enforcement (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`),
//!    1-pole anti-zipper smoothing, and static `0 MACs/sample` collapsed head caching.
//!
//! ## Core Real-Time Invariants
//! - **100% `#![no_std]` + `alloc` Foundation:** Zero OS calls, zero locks, and **zero heap allocations**
//!   inside [`DnamEngine::process_block`], [`DnamEngine::process_block_in_place`],
//!   [`DnamEngine::process_mono_to_stereo`], [`DnamEngine::process_mono`], and [`DnamEngine::reload_model_in_place`].
//! - **Zero `FSQRT`, `FDIV`, or `libm` Calls** at $48\text{ kHz}$ and $3\text{ kHz}$.
//! - **Automatic Hardware Denormal Suppression ([`FtzDazGuard`]):** Sets `FPCR.FZ` (`aarch64`) or
//!   `MXCSR` `FTZ | DAZ` (`x86_64`) on block entry and restores on exit.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

#[forbid(unsafe_code)]
pub mod container;
#[forbid(unsafe_code)]
pub mod control;
#[forbid(unsafe_code)]
pub mod core1;
#[forbid(unsafe_code)]
pub mod core2;
#[forbid(unsafe_code)]
pub mod dynamic;
#[forbid(unsafe_code)]
pub mod engine;
pub mod ffi;
pub mod math;
#[forbid(unsafe_code)]
pub mod memory;
#[forbid(unsafe_code)]
pub mod provenance;
#[forbid(unsafe_code)]
pub mod schema;
pub mod types;

pub use dynamic::DynamicDnamEngine;
pub use engine::DnamEngine;
pub use types::{DnamTelemetrySnapshot, FtzDazGuard, Tier};
