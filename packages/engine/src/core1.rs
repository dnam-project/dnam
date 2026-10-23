//! Core 1: 12-Layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + `SelectiveRgLru` Stack (`core1.rs`).
//!
//! Implements Subsystem 1 from [`docs/core1_hybrid_ssm_and_polyadaa.md`](../../../docs/core1_hybrid_ssm_and_polyadaa.md)
//! and [`docs/dnam_container_schema.md` §2](../../../docs/dnam_container_schema.md#2-five-hardware-loading-optimizations--50-ns-zero-allocation-load):
//! - **4 Execution-Order Interleaved Blocks (`b = 0..3`):** Each block executes two causal dilated
//!   convolutions ($K=6$, dilations $d \in [1, 3, 7, 19]$) with division-free algebraic `PolyADAA`
//!   (`CL-01`, `CL-02`) followed by a Radial-Polar `SelectiveRgLru` complex resonator layer (`CL-03`..`CL-05`).
//! - **Exact 3-State `PolyADAA` DC Resting Equilibrium:** Initializes $(u_c[-1], \mathcal{A}_c[-1], m_{\text{dc},c}[-1])$
//!   to their analytical fixed point on `reset_to_dc_equilibrium()`, guaranteeing `0.000000` Sample-0 transient.
//! - **16-Tap Collapsed SVD Output Head:** Cached in L1D memory when controls are static (`0` extra MACs/sample)
//!   and evaluated via `B::dot_product_16taps_4ch` (`vDSP_dotpr` on `AppleAmxBackend`).

mod reference;
mod stack;
mod weights;

pub use reference::{
    poly_adaa_algebraic_f64, poly_adaa_slice_f64, poly_adaa_step_f64, selective_rglru_slice_f64,
    selective_rglru_step_f64,
};
pub use stack::Core1Stack;
pub use weights::{PolyAdaaLayerWeights, RgLruLayerWeights};

/// Number of interleaved Core 1 blocks (`4`).
pub const CORE1_NUM_BLOCKS: usize = 4;

/// Number of dilated conv + `PolyADAA` layers (`8`).
pub const CORE1_NUM_CONV_LAYERS: usize = 8;

/// Causal dilated convolution kernel width ($K = 6$).
pub const CORE1_CONV_KERNEL: usize = 6;

/// Power-of-two ring buffer length per dilated conv layer (`128` samples covers max delay $(6-1)\times 19 = 95 < 128$).
pub const CORE1_CONV_HIST_LEN: usize = 128;
pub(crate) const CORE1_CONV_HIST_MASK: usize = CORE1_CONV_HIST_LEN - 1;

/// Unfolded output head tap count ($K_{\text{head}} = 16$).
pub const CORE1_HEAD_TAPS: usize = 16;
pub(crate) const CORE1_HEAD_TAP_MASK: usize = CORE1_HEAD_TAPS - 1;

/// Canonical block dilation schedule `[1, 3, 7, 19]`.
pub const CORE1_DILATIONS: [usize; CORE1_NUM_BLOCKS] = [1, 3, 7, 19];
