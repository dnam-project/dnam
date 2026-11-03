//! $3\text{ kHz}$ Parametric Control Bus, `ControlChannelRecord` & Anti-Zipper Smoother (`control`).
//!
//! Implements Subsystem 3 from `docs/dsp/parametric_control_bus.md`
//! and `docs/training/clamp_and_audit_registry.md`.

pub mod bus;
pub mod descriptor;
pub mod record;
pub mod reference;

pub use bus::ControlBus;
pub use descriptor::{ConditionalClampRule, ControlClampDescriptor};
pub use record::{ControlChannelRecord, ControlLatentProjection, LatentAccumulator64B};
pub use reference::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};

/// Number of leading control slots captured in the fixed C-ABI [`crate::DnamTelemetrySnapshot`] preview array.
/// Note: TORA supports an arbitrary number of user controls $P \in [0, 255]$ via the unified 128-byte
/// [`ControlChannelRecord`] table and $\mathcal{O}(1)$ [`LatentAccumulator64B`] (`Theorem 3`), while
/// SVD head rank ($D_h \in \{4, 8\}$) and FM interaction rank ($R \in \{4, 8\}$) scale dynamically with
/// [`crate::Tier::svd_control_modes`] and [`crate::Tier::fm_interaction_rank`].
pub const TELEMETRY_CONTROL_SLOTS: usize = 16;

/// Anti-zipper convergence threshold: once $|u_p - u_{p,\text{target}}| < 10^{-6}$, the smoother snaps
/// to `u_target` and re-engages the static `0 MACs/sample` head cache.
pub const ANTI_ZIPPER_SNAP_EPS: f32 = 1e-6;

/// Default $3\text{ kHz}$ 1-pole anti-zipper smoothing pole ($\tau \approx 5\text{ ms} \implies \alpha \approx 0.9355$).
pub const DEFAULT_ANTI_ZIPPER_POLE: f32 = 0.9355;
