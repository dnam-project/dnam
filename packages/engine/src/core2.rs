//! Core 2: Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`) (`core2.rs`).
//!
//! Implements Subsystem 2 from [`docs/core2_modulated_ufdn.md`](../../../docs/core2_modulated_ufdn.md):
//! - **Dynamic 64-Byte Aligned `fp16` Delay Buffers ([`crate::memory::DelayScratchStorage`]):** Allocates `0 KB` when
//!   `CORE2_ACTIVE == 0` (`ring_buffer_pow2_log2 == 0`), or $2^{\text{ring\_log2}}$ `fp16` samples per line
//!   when active, with cold-init volatile page pre-faulting (`0` page faults on the real-time thread).
//! - **64-Sample TZF Reference Delay Tap (`CL-06`):** Enables causal Through-Zero Flanging (`-54.99 dB` null).
//! - **4-Tap Cubic Lagrange Fractional Delay Read (`CL-06`) & 128-Sample Switch Crossfade (`CL-07`):**
//!   Division-free polynomial weights with zero-transient click-free delay switching.
//! - **Coprime Schroeder All-Pass Diffuser & Givens+Householder Orthogonal Mixing (`CL-08`):**
//!   Preserves exact unitary energy ($\|\mathbf{U}(\psi)\|_2 = 1.00000000$).
//! - **Gain-Normalized $5\text{ Hz}$ DC Blocker (`CL-09`) & Coercive-Saturating Lyapunov Loop (`CL-10`, `CL-11`):**
//!   Guarantees $S'(0) \equiv 1.0000$ and bounded self-oscillation even when pushed into Dub Echo regeneration.

mod reference;
mod ufdn;

pub use reference::{
    cubic_lagrange_weights_f64, givens_householder_mix_8ch_f64, mufdn_coercive_step_f64,
    mufdn_slice_f64,
};
pub use ufdn::{ModulatedUfdn, MufdnGroupWeights};

/// Maximum number of FDN delay lines (`4` in `Lite`, `8` in `Full`/`Ultra`).
pub const CORE2_MAX_LINES: usize = 8;

/// Length of the per-line Schroeder all-pass ring buffer (`256` samples covers max coprime delay `223`).
pub const CORE2_ALLPASS_RING_LEN: usize = 256;
pub(crate) const CORE2_ALLPASS_RING_MASK: usize = CORE2_ALLPASS_RING_LEN - 1;

/// Length of the stereo Through-Zero Flanging (TZF) dry reference ring buffer (`256` samples).
pub const CORE2_TZF_RING_LEN: usize = 256;
pub(crate) const CORE2_TZF_RING_MASK: usize = CORE2_TZF_RING_LEN - 1;

/// Canonical coprime Schroeder all-pass delay lengths in samples at $48\text{ kHz}$.
pub const CORE2_DEFAULT_ALLPASS_DELAYS: [u16; CORE2_MAX_LINES] =
    [19, 31, 47, 73, 103, 139, 179, 223];
