# `packages/capture/src/solver/` — 5-Stage `VarPro`, `Wide192Accum` & `uplo='U'` Guide

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Spec |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs) | `SolverOptions`, `uplo_u_*`, `solve_uplo_u_cholesky_2ch`, and `compute_noise_subtracted_besr`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`wide_gram.rs`](wide_gram.rs) | `Wide192Accum` (`> 192-bit` mantissa) and tiled `256 x 256` `uplo='U'` Gram contraction. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`bezout_gram.rs`](bezout_gram.rs) | 4-tier 3-band Binomial/Bézout (`LF`, `MF`, `HF`) `Wide192Accum` contraction and `bESR` quadratic evaluation. | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout_schur.rs`](bezout_schur.rs) | Multi-tier nested SPD normal equations (`solve_multitier_besr_4tier`) and `MM-Cholesky` polish. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`stage_a.rs`](stage_a.rs) | Stage A (`< 5 ms`) anchor lifting (`Stage 0..IV`, 64-tap FIR, 32-pole Kautz bank, `VP-GN`). | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage_a_id.rs`](stage_a_id.rs) | Steiglitz–McBride rational pole ID, linear/nonlinear ARX split, and Stage IV delay anchors. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`stage_b_c.rs`](stage_b_c.rs) | Stage B (`256-D` `StarCenter` harvest) and Stage C (4-tier Schur, `LTAS` KKT, `STAT`). | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`stage_d.rs`](stage_d.rs) | Stage D (`< 5 ms`) arbitrary-`P` `CP1`, Kronecker SVD Modes `1..8`, Walsh-Hadamard FM, and audit. | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`stage_e.rs`](stage_e.rs) | Stage E (`< 0.2 ms`) `[STAB-ZERO-DC]`, full-coreset 4-tier `LUFS` trim, watermark export, and phase ID. | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`stages.rs`](stages.rs) | `AuditBounds`, `is_control_variance_active`, and `bitcrusher_smooth_surrogate`. | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
