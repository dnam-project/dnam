# `dnam-trainer` — Pure-Rust Analytical `VarPro` Two-Sweep System-ID Trainer

**`dnam-trainer`** (`packages/trainer/`) is the 100% safe-Rust (`#![forbid(unsafe_code)]`) Analytical Variable-Projection (`VarPro`) Two-Sweep Block-Coordinate Cholesky system-identification trainer for **Dynamic Neural Audio Modeler (DNAM)**.

By linking [`dnam-engine`](../engine/README.md) directly in the same Cargo workspace, `dnam-trainer` executes the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`), `3 kHz` Parametric Control Bus, `uplo='U'` Gram matrix layout, and `.dnam` / `.dnam-ckpt` binary containers during system identification (`ref[arch.isomorphism.two-sweep]`, [`ADR-0017`](../../docs/adr/0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md)).

---

## Key Capabilities

- **Two-Sweep Block-Coordinate Cholesky & Analytical `VarPro` Solver ([`src/solver.rs`](src/solver.rs)):**
  - Column-major upper-triangular (`LAPACK uplo='U'`) `f64` Gram accumulation and regularized 2-channel Cholesky solver across all 4 slimmable tiers (`Lite` `64-D`, `Full` `128-D`, `Pro` `192-D`, `Ultra` `256-D`).
  - Exact algebraic Bézout 3-band crossover weighting and Noise-Subtracted `B-ESR` evaluation ([`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md)).
- **Non-Stop 3-Phase Control-Space Audit & Step-0 Variance Gating ([`src/solver/stages.rs`](src/solver/stages.rs)):**
  - Enforces discrete pole radius ($r_c \le 0.9995$), Stage IV feedback gain ($g_{\mathrm{lin}} \le 0.9990$), Schroeder all-pass ($|a_{\mathrm{diff}}| \le 0.75$), and causal delay ($\tau_m \ge 4.0$ samples) bounds ([`control_space_audit.md`](../../docs/specs/training/control_space_audit.md)).
- **`.dnam-ckpt` Gram Fingerprinting & `DNCA` `fp16` Coreset Storage ([`src/lib.rs`](src/lib.rs)):**
  - Streaming `DNAM-GRAM-INPUT-v1` SHA-256 verification and `64,000`-sample (`256 × 250`) `fp16` coreset quantization (`32,768,000 B` per stem, `64 B`-aligned; [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md)).

---

## Package Structure

```text
packages/trainer/
├── Cargo.toml                  # dnam-trainer crate manifest & dnam-train binary target
├── moon.yml                    # Rust quality, test, coverage (>= 95%), bench & CLI tasks
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)]), CLI runner, Gram hash & fp16
│   ├── solver.rs               # uplo='U' Gram accumulation, Cholesky solver & B-ESR metrics
│   ├── solver/
│   │   └── stages.rs           # AuditBounds, Step-0 variance gating & Family-5 surrogate
│   └── bin/
│       └── dnam_train.rs       # dnam-train CLI binary entrypoint
├── tests/
│   └── trainer_spec_verification.rs # Integration & proptest verify[...] specification suite
└── benches/
    └── bench_trainer.rs        # Criterion uplo='U' Cholesky solver throughput benchmark
```

---

## Quickstart

```bash
# Run all trainer quality gates (format-check, clippy, typecheck, build, test)
moon check trainer

# Enforce >= 95% line coverage
moon run trainer:coverage

# Run the dnam-train CLI
moon run trainer:cli -- check --max-tier ultra
```
