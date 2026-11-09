//! Op-M: Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_m_modal_kautz.md` (`ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.stage3.sweep-contraction]`, `REQ-1.2`, `REQ-2.5`).
//!
//! Submodules:
//! - [`kautz`]: Symplectic modal recurrence, contractive rotor, and orthonormal Kautz modal bank (`Stage III`).
//! - [`hilbert`]: 12-allpass `H1` Hilbert lattice, 15-rate anchor table, and history-consistent rate remap (`Stage I`).
//! - [`tracker`]: `T1` sub-harmonic tracker, 160-bit midpoint `5 Hz` DC blocker, and exponent-split normalization (`Stage I`).

pub mod hilbert;
pub mod kautz;
pub mod tracker;

pub use hilbert::*;
pub use kautz::*;
pub use tracker::*;
