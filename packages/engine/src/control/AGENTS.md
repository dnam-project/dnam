# `src/control` — $3\text{ kHz}$ Parametric Control Bus & Non-Stop 3-Phase Clamping

Implements the $3\text{ kHz}$ Control Bus (`#[forbid(unsafe_code)]`): 64-byte binary [`ControlClampDescriptor`](descriptor.rs) cache lines, anti-zipper slewing, Node-Anchored Lobatto HDMR + Rank-4 Factorization Machine, autonomous Chebyshev LFOs, envelope followers, and trust-region soft-clipping.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`descriptor.rs`](descriptor.rs) | 64-byte `ControlClampDescriptor` (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`) + `proptest` | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`bus.rs`](bus.rs) | `ControlBus<B>`, `tick_3khz`, and dirty-flag handling | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`reference.rs`](reference.rs) | Reference `f64` Lobatto, Rank-4 FM, Chebyshev LFO, and trust-region kernels | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for control-basis and verification guidance.
