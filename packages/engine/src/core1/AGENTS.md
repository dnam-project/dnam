# `src/core1` — 12-Layer Hybrid Dilated-Conv (`PolyADAA`) + Selective RG-LRU

Implements Core 1 (`#[forbid(unsafe_code)]`): 4 macro-blocks $\times$ (2 Dilated Causal Convolutions with $1\times$ B-Spline algebraic `PolyADAA` + 1 Radial-Polar `SelectiveRgLru`) + 16-tap SVD stereo output head (`CL-01..05`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`weights.rs`](weights.rs) | `PolyAdaaLayerWeights` & `RgLruLayerWeights` SoA views | [`core1_hybrid_ssm_and_polyadaa.md`](../../../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| [`stack.rs`](stack.rs) | `Core1Stack<B>`, const-specialized `process_sample_const::<GROUPS>`, 3-state DC equilibrium | [`core1_hybrid_ssm_and_polyadaa.md`](../../../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| [`reference.rs`](reference.rs) | Reference `f64` `PolyADAA` & `SelectiveRgLru` kernels + `Test 1` & `Test 2` unit tests | [`verification_and_guardrails.md`](../../../../docs/verification_and_guardrails.md) |

## 2. Invariants

- Keep `process_sample_const::<GROUPS>` (`GROUPS = 1, 2, 3`) register-resident with zero stack spills and zero `FDIV`/`FSQRT`.
- Reinterpret `Arena 0..2` `4x4` weight tiles exclusively via safe `zerocopy::FromBytes::ref_from_bytes`.
