# `packages/trainer/src/solver/` — Analytical `VarPro`, `uplo='U'` Cholesky & Audit Guide

**`dnam_trainer::solver`** ([`../solver.rs`](../solver.rs) and [`stages.rs`](stages.rs)) implements the analytical Variable-Projection (`VarPro`), `uplo='U'` column-major Gram accumulation, regularized Cholesky linear solver, algebraic Bézout 3-band crossover weights, Noise-Subtracted `B-ESR` evaluation, Step-0 variance gating, and Non-Stop 3-Phase Control-Space Audit bounds.

## 1. Module Map

| File | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs) | `tier_feature_dimension`, `uplo_u_packed_len`, `uplo_u_index`, `accumulate_uplo_u_gram_and_cross`, `solve_uplo_u_cholesky_2ch`, `bezout_three_band_weights`, and `compute_noise_subtracted_besr`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md), [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`checkpoint_and_cache_format.md`](../../../../docs/specs/containers/checkpoint_and_cache_format.md) |
| [`stages.rs`](stages.rs) | `AuditBounds` (`clamp_parameters`), `is_control_variance_active` (Step-0 variance gating), and `bitcrusher_smooth_surrogate` (`Family 5` cosine surrogate). | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md), [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md), [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |

## 2. Numerical & Stability Invariants

- **`f64` Sufficient Statistics (`ref[train.gram.accuracy-contract]`):** All `uplo='U'` Gram matrices $\mathbf{G}$ and two-channel cross-correlations $\mathbf{C}$ accumulate and solve in IEEE-754 `f64`.
- **Unconditional Stability (`ref[train.audit.phase1.clamp-stability]`):** Pole radii $r_c \le 0.9995$, Stage IV feedback gain $g_{\mathrm{lin}} \le 0.9990$, Schroeder all-pass $|a_{\mathrm{diff}}| \le 0.75$, and causal delay $\tau_m \ge 4.0$ samples.
