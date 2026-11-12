---
id: ADR-0008
title: "Per-Stem uplo='U' Kronecker Statistics & 192-Bit Wide-Significand Gram Contraction"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/training/four_stage_pipeline.md
  - docs/specs/training/iterative_resume_and_caching.md
  - docs/specs/containers/project_container_format.md
requirements:
  - REQ-3.1
  - REQ-3.2
  - REQ-3.3
  - REQ-4.2
---

# ADR-0008: Per-Stem `uplo='U'` Kronecker Statistics & 192-Bit Wide-Significand Gram Contraction

This record justifies the per-stem sufficient-statistic layout and wide-significand Gram accumulation in [`../specs/training/four_stage_pipeline.md`](../specs/training/four_stage_pipeline.md), [`../specs/training/iterative_resume_and_caching.md`](../specs/training/iterative_resume_and_caching.md), and [`../specs/containers/project_container_format.md`](../specs/containers/project_container_format.md) fulfilling [`REQ-3.1`–`REQ-3.3`, `REQ-4.2`](../requirements/system_goals_and_requirements.md).

## 1. Context: Why Pre-Summed Kronecker Matrices Fail on Stem Additions

Storing only a pre-collapsed sum $\mathbf{G}_{\text{old}} = \sum_{s=1}^S \mathbf{G}_{\text{audio}}^{(s)} \otimes (\tilde{\boldsymbol{\psi}}_s \tilde{\boldsymbol{\psi}}_s^\top)$ in a project container fails whenever a newly added stem increases a control's polynomial degree $K_p$, unlocks a switch option (`[CLAMP-SWITCH]`), or updates the low-rank control projection (`CP1`), because the old control factors $\tilde{\boldsymbol{\psi}}_s$ are irrecoverably entangled inside $\mathbf{G}_{\text{old}}$.

## 2. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Tier-Ordered `256-D` `uplo='U'` `STAT` Statistics & Static Kautz Contraction `P_c` (`ref[train.resume.kronecker-reassembly]`, `ref[container.proj.preamble-and-directory]`, `ref[container.proj.stat-uplo-u]`)** | Persist `StarCenter` and per-stem tier-ordered upper-triangular Gram matrices `G_{s,T}` (`263,168 B` at `D = 256`) and cross-correlations `C_{s,T}` (`4,096 B`) in `.dnam-proj` (`STAT`), contracting static unit Stage III Kautz readout vectors `(c_re, c_im)` via `G̃_s = P_c^T G_s P_c` and reassembling Kronecker products with updated `CP1` control factors at solve time. | Pre-summed Kronecker normal matrices or dense `128 × 128` legacy checkpoints require re-running audio forward passes over all unchanged stems whenever control basis nodes or `CP1` projections change. |
| **Pre-Baked `MM-Cholesky` + `LTAS` KKT Cross-Correlation `c_{s,eff}^{MM-KKT}` (`ref[train.resume.two-tier-execution]`, [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md))** | Store `c_{s,eff}^{MM-KKT} = (G_s + Λ) w_s^{MM-KKT}` inside `STAT` so any subsequent Fast-Path incremental vault re-solve or live hardware null-test auto-patch `(G_s + Λ)^{-1} c_{s,eff}^{MM-KKT}` reproduces the exact selective `1/12`-octave `MM-Cholesky` + 33-band `LTAS` KKT solution in `< 5 ms`. | Storing raw unprojected time-domain cross-correlations `c_s^{(0)}` loses phase-relaxed high-frequency and 33-band `LTAS` KKT corrections upon vault re-solve. |
| **Stereo-Aware `24,000 B` / `48,000 B` `fp16` `INPC` & `STMC` Coreset Capsules (`ref[container.proj.inpc-capsule]`, `ref[container.proj.stmc-capsule]`)** | Persist a `250 ms` (`12,000`-frame at `48 kHz`) high-leverage `fp16` dry/wet coreset (`24,000 B` for `C = 1` mono or `48,000 B` for `C = 2` stereo, padded to `64 B` alignment inside `INPC` `24,640 B`/`48,640 B` and `STMC` `28,800 B`/`52,800 B` capsules in `.dnam-proj`) enabling `< 25 ms` full re-pooling without source WAV files. | Retaining full-length `f32` `256-D` feature trajectories (`> 32–150 MB` per stem) violates the `< 64 MiB` browser worker memory ceiling (`REQ-4.3`) and bloats `.dnam-proj` files. |
| **Exponent-Aware & 192-Bit Wide-Significand Gram Contraction (`ref[train.gram.accuracy-contract]`, `ref[train.gram.wide192]`)** | Accumulate weighted `binary32` products via radix-256 digit decomposition or 192-bit wide-significand (`8 × 24-bit` limbs in `int32` + `16-bit` exponent) streaming before host `float64` rounding (`\|ĝ - g\| ≤ γ_N S`). | Naive `float32` normal-equation accumulation overflows above `2^128` or loses small cross-terms (`[1, 2^-25]`) due to subblock-maximum underflow. |
