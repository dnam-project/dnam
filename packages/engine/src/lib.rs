//! `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time Stereo Neural Audio Modeling & Differentiable DSP Core.
//!
//! Unifies the Tri-Operator Recurrent Architecture (**TORA**) of **Dynamic Neural Audio Modeler (DNAM)**:
//! 1. **Operators ([`operators`]):** Op-T (Transport / Elastic Buffer), Op-M (Symplectic Modal / Kautz Bank),
//!    and Op-N (Multi-Drive PolyADAA Commuted Cascade / Waveshaping).
//! 2. **Stages ([`stages`]):** Stage 0 8-Port Matrix Router, Stages I–IV Physical Topology, and
//!    256-D Mid/Side Kronecker Output Head ([`StagePipeline`]).
//! 3. **Control Bus ([`control`]):** $3\text{ kHz}$ Envelopes/LFOs + Node-Anchored Lobatto HDMR & Rank-4
//!    Factorization Machine with 128-byte CCR table, division-free 16-sample micro-ramping,
//!    and static `0 MACs/sample` collapsed head caching.
//!
//! ## Core Real-Time Invariants
//! - **100% `#![no_std]` + `alloc` Foundation:** Zero OS calls, zero locks, and **zero heap allocations**
//!   inside [`DnamEngine::process_block`], [`DnamEngine::process_block_in_place`],
//!   [`DnamEngine::process_mono_to_stereo`], [`DnamEngine::process_mono`], and [`DnamEngine::reload_model_in_place`].
//! - **Zero `FSQRT`, `FDIV`, or `libm` Calls** in $48\text{ kHz}$ audio and $3\text{ kHz}$ control loops.
//! - **Total Lite State RAM:** Strictly audited at 8,656 B ([`TOTAL_LITE_STATE_RAM_BYTES`]).
//! - **Block Floating-Point Environment ([`FtzDazGuard`]):** Selects round-to-nearest-even and
//!   non-trapping exceptions and `FPCR.FZ` with conditional `FIZ` (`aarch64`) or `MXCSR`
//!   `FTZ | DAZ` (`x86_64`) on block entry, then restores the caller's control register. On x86-64,
//!   sticky flags are cleared on entry and restored on exit; on AArch64, the separate `FPSR` status
//!   is not saved. Portable targets flush recursive writebacks. Direct per-sample and control-tick
//!   callers hold a guard once per batch.

#![no_std]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

#[forbid(unsafe_code)]
pub mod container;
#[forbid(unsafe_code)]
pub mod control;
#[forbid(unsafe_code)]
pub mod dynamic;
#[forbid(unsafe_code)]
pub mod engine;
pub mod ffi;
#[forbid(unsafe_code)]
pub mod host;
pub mod math;
#[forbid(unsafe_code)]
pub mod memory;
#[forbid(unsafe_code)]
pub mod operators;
#[forbid(unsafe_code)]
pub mod provenance;
#[forbid(unsafe_code)]
pub mod schema;
#[forbid(unsafe_code)]
pub mod stages;
#[cfg(test)]
#[forbid(unsafe_code)]
pub(crate) mod test_support;
pub mod types;

pub use container::strip_container_to_tier;
pub use dynamic::DynamicDnamEngine;
pub use engine::{DnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES};
pub use stages::{StagePipeline, TOTAL_LITE_STATE_RAM_BYTES};
pub use types::{DnamTelemetrySnapshot, FtzDazGuard, Tier};
