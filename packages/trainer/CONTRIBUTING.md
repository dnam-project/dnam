# Contributing to `dnam-trainer` (`packages/trainer/`)

This guide covers engineering, numerical accuracy, safety, testing, and coverage standards for **`dnam-trainer`** ([`README.md`](README.md)), the pure-Rust Analytical Variable-Projection (`VarPro`) Two-Sweep system-identification trainer for **DNAM**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md), [`src/AGENTS.md`](src/AGENTS.md), [`src/solver/AGENTS.md`](src/solver/AGENTS.md), and [`tests/AGENTS.md`](tests/AGENTS.md):

- **`src/lib.rs`:** Crate root (`#![forbid(unsafe_code)]`), `TrainerCli` (`clap`), `DNAM-GRAM-INPUT-v1` SHA-256 fingerprint (`compute_gram_input_fingerprint`), and `fp16` coreset quantization (`f32_to_fp16_bits`, `fp16_bits_to_f32`).
- **`src/bin/dnam_train.rs`:** `dnam-train` CLI binary entrypoint.
- **`src/solver.rs` & `src/solver/stages.rs`:** Column-major `uplo='U'` Gram accumulation, regularized `f64` Cholesky solver, algebraic Bézout 3-band crossover weights, Noise-Subtracted `B-ESR`, Step-0 variance gating, and Non-Stop 3-Phase Control-Space Audit bounds.
- **`tests/trainer_spec_verification.rs`:** Integration and `proptest!` verification suite with `verify[...]` citations covering training, quality, and `.dnam-ckpt`/`DNCA` specification rules.
- **`benches/bench_trainer.rs`:** Criterion throughput benchmarks for `uplo='U'` Gram accumulation and Cholesky solves.

---

## 2. Core Invariants

1. **100% Safe Rust (`#![forbid(unsafe_code)]`):**
   - `dnam-trainer` enforces `#![forbid(unsafe_code)]` across all library, binary, test, and benchmark targets (`ref[repo.code.type-safety-and-imports]`).
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
