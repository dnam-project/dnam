---
id: ADR-0008
title: "Per-Stem uplo='U' Kronecker Statistics & 192-Bit Wide-Significand Gram Contraction"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/training/four_stage_pipeline.md
  - docs/specs/training/iterative_resume_and_caching.md
  - docs/specs/containers/checkpoint_and_cache_format.md
requirements:
  - REQ-3.1
  - REQ-3.2
  - REQ-3.3
  - REQ-4.2
---

# ADR-0008: Per-Stem `uplo='U'` Kronecker Statistics & 192-Bit Wide-Significand Gram Contraction

This record justifies the per-stem sufficient-statistic layout and wide-significand Gram accumulation in [`../specs/training/four_stage_pipeline.md`](../specs/training/four_stage_pipeline.md), [`../specs/training/iterative_resume_and_caching.md`](../specs/training/iterative_resume_and_caching.md), and [`../specs/containers/checkpoint_and_cache_format.md`](../specs/containers/checkpoint_and_cache_format.md) fulfilling [`REQ-3.1`–`REQ-3.3`, `REQ-4.2`](../requirements/system_goals_and_requirements.md).

## 1. Context: Why Pre-Summed Kronecker Matrices Fail on Stem Additions

Storing only a pre-collapsed sum $\mathbf{G}_{\text{old}} = \sum_{s=1}^S \mathbf{G}_{\text{audio}}^{(s)} \otimes (\tilde{\boldsymbol{\psi}}_s \tilde{\boldsymbol{\psi}}_s^\top)$ in a training checkpoint fails whenever a newly added stem increases a control's polynomial degree $K_p$, unlocks a switch option (`[CLAMP-SWITCH]`), or updates the low-rank control projection (`CP1`), because the old control factors $\tilde{\boldsymbol{\psi}}_s$ are irrecoverably entangled inside $\mathbf{G}_{\text{old}}$.

## 2. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Per-Stem `uplo='U'` `Arena 4` Statistics (`ref[train.resume.kronecker-reassembly]`, `ref[container.ckpt.uplo-u]`)** | Persist per-stem, per-tier upper-triangular Gram matrices `G_{s,T}` and cross-correlations `C_{s,T}` in tier-bundled `Arena 4` slices (`StandaloneFull` or `PrefixDelta`), reassembling Kronecker products with updated control factors at solve time. | Pre-summed Kronecker normal matrices require re-running audio forward passes over all unchanged stems whenever control basis nodes or `CP1` projections change. |
| **Exponent-Aware & 192-Bit Wide-Significand Gram Contraction (`ref[train.gram.accuracy-contract]`, `ref[train.gram.wide192]`)** | Accumulate weighted `binary32` products via radix-256 digit decomposition or 192-bit wide-significand (`8 × 24-bit` limbs in `int32` + `16-bit` exponent) streaming before host `float64` rounding (`\|ĝ - g\| ≤ γ_N S`). | Naive `float32` normal-equation accumulation overflows above `2^128` or loses small cross-terms (`[1, 2^-25]`) due to subblock-maximum underflow. |
