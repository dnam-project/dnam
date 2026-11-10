# Contributing to `dnam-trainer` (`packages/trainer/`)

This guide covers engineering, numerical accuracy, safety, testing, and coverage standards for **`dnam-trainer`** ([`README.md`](README.md)), the pure-Rust Analytical Variable-Projection (`VarPro`) Two-Sweep system-identification trainer for **DNAM**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md), [`src/AGENTS.md`](src/AGENTS.md), [`src/metrics/AGENTS.md`](src/metrics/AGENTS.md), [`src/whitebox/AGENTS.md`](src/whitebox/AGENTS.md), [`src/ckpt/AGENTS.md`](src/ckpt/AGENTS.md), [`src/solver/AGENTS.md`](src/solver/AGENTS.md), and [`tests/AGENTS.md`](tests/AGENTS.md):

- **`src/lib.rs`:** Crate root (`#![forbid(unsafe_code)]`), `TrainerCli` (`clap`), `DNAM-GRAM-INPUT-v1` SHA-256 fingerprint (`compute_gram_input_fingerprint`), and IEEE-754 RNE `fp16` coreset quantization (`f32_to_fp16_bits`, `fp16_bits_to_f32`).
- **`src/bin/dnam_train.rs`:** `dnam-train` CLI binary entrypoint (`check` and `dnwb-serve`).
- **`src/targets.rs`, `src/resampler.rs` & `src/audit.rs`:** `DistillationTarget` (`CaptureBitDepth` across `Int12`/`Int16`/`Int24`/`Int32`/`Float32`, `StemStimulus`, `StemResponse`, `StimulusFrame`), zero-phase Kaiser-sinc `8..192 kHz <-> 48 kHz` audio/trajectory resampling (`resample_arbitrary_rate_to_48k`, `resample_48k_to_arbitrary_rate`), and Non-Stop 3-Phase Control-Space Audit helpers.
- **`src/metrics.rs` & `src/metrics/{companion,certification}.rs`:** 4-metric psychoacoustic certification suite (`bESR`, 5-scale 32-probe `MR-IIR`, 33-band Welch `LTAS`, `SegESR`), `certify_4metric_with_bit_depth`, and 4-tier non-regression verification (`verify_monotone_4tier_summaries`).
- **`src/whitebox.rs` & `src/whitebox/{runners,faust,ipc,session}.rs`:** `64 B` `DnwbHeader64B` pipe IPC framing, `FaustRunner`, `SpiceRunner` (`PCHIP` $4f_s$ + `309`-tap Kaiser-sinc $4\times \to 1\times$ decimation), persistent stateful `DnwbWorkerSession`, and non-deadlocking concurrent `DNWB` subprocess execution (`execute_dnwb_subprocess_request`).
- **`src/ckpt.rs` & `src/ckpt/{arena,schema,view,bundle}.rs`:** `.dnam-ckpt` (`DNCK`) container view/builder, `DCA1`, `Arena 4` (`uplo='U'`), `Arena 5` (`DNCA`), and `.dnam` runtime export.
- **`src/solver.rs` & `src/solver/stages.rs`:** Column-major `uplo='U'` Gram accumulation, regularized `f64` Cholesky solver, degree-7 Bézout 3-band crossover weights, sum-denominator Noise-Subtracted `bESR`, Step-0 variance gating, and Non-Stop 3-Phase Control-Space Audit bounds.
- **`tests/{trainer_spec_verification,trainer_multirate_bitdepth_fixtures,trainer_ckpt_security,trainer_ckpt_dau1_security}.rs`:** Integration and `proptest!` verification suites covering specification rules (`verify[...]`), multi-rate (`32`..`192 kHz` native and canonical `48 kHz`) / 5-bit-depth (`Int12`/`Int16`/`Int24`/`Int32`/`Float32`) fixtures, `.dnam-ckpt` / `DCA1` / `DAU1` / `DMU1` / `DNCA` tamper hardening, and live stateful/large-payload `dnam-train dnwb-serve` subprocess IPC.
- **`benches/bench_trainer.rs`:** Criterion throughput benchmarks for `uplo='U'` Gram accumulation and Cholesky solves.

---

## 2. Core Invariants

1. **100% Safe Rust (`#![forbid(unsafe_code)]`) & File Hygiene:**
   - `dnam-trainer` enforces `#![forbid(unsafe_code)]` across all library, binary, test, and benchmark targets (`ref[repo.code.type-safety-and-imports]`), zero `mod.rs` files, and `< 900` lines per `.rs` file.
2. **Exact Engine-Trainer Isomorphism (`ref[arch.isomorphism.two-sweep]`, `ref[arch.isomorphism.audit-wiring]`):**
   - Never duplicate DSP operator math (`Op-T`, `Op-M`, `Op-N`), `3 kHz` control-bus recurrences, or `.dnam` binary framing; depend directly on [`dnam-engine`](../engine/README.md).
3. **`f64` Sufficient Statistics & `uplo='U'` Packing (`ref[container.ckpt.uplo-u]`, `ref[train.gram.accuracy-contract]`):**
   - Accumulate and factor all Gram matrices $\mathbf{G}$ and cross-correlations $\mathbf{C}$ in IEEE-754 `f64` column-major upper-triangular order (`uplo='U'`).
4. **Non-Stop 3-Phase Control-Space Audit (`ref[train.audit.phase1.clamp-stability]`):**
   - Never block unattended training on interactive prompts; clamp unobserved control ranges, switches, and stability bounds automatically.

---

## 3. Quality Gates & Verification Commands

Run all checks from the workspace root via `moon`:

```bash
moon check trainer
moon run trainer:coverage
moon run trainer:bench
```
