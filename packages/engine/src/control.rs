//! $3\text{ kHz}$ Parametric Control Bus, `ControlClampDescriptor` & Anti-Zipper Smoother (`control`).
//!
//! Implements Subsystem 3 from [`docs/parametric_control_bus.md`](../../../docs/parametric_control_bus.md)
//! and [`docs/clamp_and_audit_registry.md`](../../../docs/clamp_and_audit_registry.md).

pub mod bus;
pub mod descriptor;
pub mod reference;

pub use bus::ControlBus;
pub use descriptor::{ConditionalClampRule, ControlClampDescriptor};
pub use reference::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};

/// Maximum number of parametric user controls (`range` + `switch`) in the inline `Arena 0` clamp table.
pub const MAX_CONTROLS: usize = 16;

/// SVD principal control subspace rank ($D_h = 4$).
pub const SVD_HEAD_RANK: usize = 4;

/// Factorization Machine interaction rank ($R = 4$).
pub const FM_INTERACTION_RANK: usize = 4;

/// Anti-zipper convergence threshold: once $|u_p - u_{p,\text{target}}| < 10^{-6}$, the smoother snaps
/// to `u_target` and re-engages the static `0 MACs/sample` head cache.
pub const ANTI_ZIPPER_SNAP_EPS: f32 = 1e-6;

/// Default $3\text{ kHz}$ 1-pole anti-zipper smoothing pole ($\tau \approx 5\text{ ms} \implies \alpha \approx 0.9355$).
pub const DEFAULT_ANTI_ZIPPER_POLE: f32 = 0.9355;
