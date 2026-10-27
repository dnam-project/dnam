# `src/control` — $3\text{ kHz}$ Parametric Control Bus & Non-Stop 3-Phase Clamping

Implements the $3\text{ kHz}$ Control Bus (`#[forbid(unsafe_code)]`): 64-byte binary [`ControlClampDescriptor`](descriptor.rs) cache lines, anti-zipper slewing, Node-Anchored Lobatto HDMR + Rank-4 Factorization Machine, autonomous Chebyshev LFOs, envelope followers, and trust-region soft-clipping (`CL-12..16`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`descriptor.rs`](descriptor.rs) | 64-byte `ControlClampDescriptor` (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`) + `proptest` | [`parametric_control_bus.md`](../../../../docs/parametric_control_bus.md) |
| [`bus.rs`](bus.rs) | `ControlBus<B>`, `tick_3khz`, dirty-flag bypass (`0` MACs @ 48 kHz when static) | [`parametric_control_bus.md`](../../../../docs/parametric_control_bus.md) |
| [`reference.rs`](reference.rs) | Reference `f64` Lobatto, Rank-4 FM, Chebyshev LFO, and trust-region kernels | [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `u = 0` preset anchoring invariants.
