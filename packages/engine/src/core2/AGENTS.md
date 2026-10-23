# `src/core2` — Unified Stereo In-Loop Diffusive Modulated Unitary FDN (`M-UFDN`)

Implements Core 2 (`#[forbid(unsafe_code)]`): early stereo reflections + $M$-line (`Lite` $M=4$, `Full`/`Ultra` $M=8$) `fp16` delay network with stateless 4-tap Cubic Lagrange interpolation, $1\text{-pole}$ loop damping, $1\text{-pole}$ allpass diffusion, $O(M)$ Givens + Householder mixing, $5\text{ Hz}$ DC blocker, and Coercive-Saturating Lyapunov loop (`CL-06..11`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`ufdn.rs`](ufdn.rs) | `MufdnGroupWeights`, `ModulatedUfdn<'a, B>`, dual-tap `switch` crossfader | [`core2_modulated_ufdn.md`](../../../../docs/core2_modulated_ufdn.md) |
| [`reference.rs`](reference.rs) | Reference `f64` Cubic Lagrange, Givens+Householder, and Lyapunov loop + `Tests 3..5` & `proptest` | [`verification_and_guardrails.md`](../../../../docs/verification_and_guardrails.md) |

## 2. Invariants

- Enforce all stability clamps (`CL-06..11`) from [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md).
- Store delay lines as 64-byte aligned [`AlignedFp16Block`](../memory.rs) blocks converted via `B::f32x4_to_fp16x4` / `B::fp16x4_to_f32x4`.
