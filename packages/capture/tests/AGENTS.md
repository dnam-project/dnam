# `packages/capture/tests/` — Suite Guide

Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md). Keep files `#![forbid(unsafe_code)]`, `< 900` lines, and cite `// verify[<id>]` only.

| Test File | Scope |
| :--- | :--- |
| [`capture_spec_verification.rs`](capture_spec_verification.rs) | `arch.isomorphism.*`, `qual.*`, `container.proj.*`, `dsp.*`, `train.*`. |
| [`capture_multirate_bitdepth_fixtures.rs`](capture_multirate_bitdepth_fixtures.rs) | `32..192 kHz`, `Int12..Float32`, `Chain-01..12` fixtures. |
| [`capture_stimulus_and_safeguards.rs`](capture_stimulus_and_safeguards.rs) | `AdaptiveStimulusController`, `Laws 0.A–0.F`, `Laws 1–3`. |
| [`capture_dual_mode_and_align.rs`](capture_dual_mode_and_align.rs) | `CALB`, 4-regime align, null homing, Mode 1/2 API, `1.0s` Null-Test. |
| [`capture_step2_2_adversarial_edges.rs`](capture_step2_2_adversarial_edges.rs) | `EnvelopeSvf`, `Law 4D` coreset, `Mode B`, `FreezeLatch`, `8..384 kHz`. |
| [`capture_proj_security.rs`](capture_proj_security.rs) | `DNPJ` `HTTP Range`, capsules, `[STAB-ZERO-DC]`, `compact_to_capsules`, vault. |
| [`capture_proj_dau1_security.rs`](capture_proj_dau1_security.rs) | `AUTH`, `DAU1`/`DMU1`, `DNWB` workers. |
| [`capture_align_and_capsules_edges.rs`](capture_align_and_capsules_edges.rs) | WAV 5-bit-depth edges, `DnamProjManifest` JCS, capsule bounds. |
| [`capture_proj_bundle_and_vault_edges.rs`](capture_proj_bundle_and_vault_edges.rs) | Directory CRC/SHA tamper guards, `STMC` bijection, stereo vault. |
| [`capture_proj_adversarial_regressions.rs`](capture_proj_adversarial_regressions.rs) | `P > 8` Mode A `R = 36`, `> 64 KB` Range, `STAT` PSD, `DMU1` growth. |
| [`solver_oracles.rs`](solver_oracles.rs) | `Oracle-01..03`: `256-D` Schur, `0 ULP` strip/gauge, `P = 6/16` `CP1` + FM, vault. |
| [`solver_chains_and_watermarks.rs`](solver_chains_and_watermarks.rs) | `Chain-01..12` `VarPro` + `Case #8` 3-layer watermark, `N = 5` collusion, phase ID. |
| [`solver_adversaries_and_topologies.rs`](solver_adversaries_and_topologies.rs) | 14 `adversaries`, `Case #9` `0.05 LU`, `Case #11` `+-1e30`/`[STAB-ZERO-DC]`, `Case #19`. |
| [`solver_step2_3_regressions.rs`](solver_step2_3_regressions.rs) | `Wide192` edges, sample-0 sync/trims, `Path 1` stride, `P = 68` `CP1`, audit, `DNWB`. |
| [`solver_step2_3_correctness_audit.rs`](solver_step2_3_correctness_audit.rs) | Narrow-span/mantissa/subnormal `Wide192`, `round-to-odd`, clamp sync, `Mode 0` guards. |
