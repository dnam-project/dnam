# `dnam-trainer` — Pure-Rust Analytical `VarPro` Two-Sweep System-ID Trainer

**`dnam-trainer`** (`packages/trainer/`) is the 100% safe-Rust (`#![forbid(unsafe_code)]`) Analytical Variable-Projection (`VarPro`) Two-Sweep Block-Coordinate Cholesky system-identification trainer for **Dynamic Neural Audio Modeler (DNAM)**.

By linking [`dnam-engine`](../engine/README.md) directly in the same Cargo workspace, `dnam-trainer` executes the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`), `3 kHz` Parametric Control Bus, `uplo='U'` Gram matrix layout, and `.dnam` / `.dnam-ckpt` binary containers during system identification (`ref[arch.isomorphism.two-sweep]`, [`ADR-0017`](../../docs/adr/0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md)).

---

## Key Capabilities

- **Two-Sweep Block-Coordinate Cholesky & Analytical `VarPro` Solver ([`src/solver.rs`](src/solver.rs)):**
  - Column-major upper-triangular (`LAPACK uplo='U'`) `f64` Gram accumulation and regularized 2-channel Cholesky solver across all 4 slimmable tiers (`Lite` `64-D`, `Full` `128-D`, `Pro` `192-D`, `Ultra` `256-D`).
  - Exact degree-7 Binomial/Bézout 3-band crossover weighting and sum-denominator Noise-Subtracted `bESR` evaluation ([`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md)).
- **Non-Stop 3-Phase Control-Space Audit & Step-0 Variance Gating ([`src/solver/stages.rs`](src/solver/stages.rs)):**
  - Enforces discrete pole radius ($r_c \le 0.9995$), Stage IV feedback gain ($g_{\mathrm{lin}} \le 0.9990$), Schroeder all-pass ($|a_{\mathrm{diff}}| \le 0.75$), and causal delay ($\tau_m \ge 4.0$ samples) bounds ([`control_space_audit.md`](../../docs/specs/training/control_space_audit.md)).
- **4-Metric Psychoacoustic Certification, Zero-Phase Multi-Rate Resampler & `DNWB` White-Box Distillation ([`src/metrics.rs`](src/metrics.rs), [`src/resampler.rs`](src/resampler.rs), [`src/whitebox.rs`](src/whitebox.rs)):**
  - Evaluates noise-qualified 3-band `bESR`, 5-scale $\times$ 32-probe resonator bank `MR-IIR` ($M \in \lbrace 256, 512, 1024, 2048, 4096 \rbrace$), 33-band `1/3`-octave Welch `LTAS`, and duration-weighted `SegESR`, plus zero-phase Kaiser-sinc multi-rate (`8`..`192 kHz` $\leftrightarrow$ `48 kHz`) resampling (`resample_arbitrary_rate_to_48k`, `resample_48k_to_arbitrary_rate`), 5-format `CaptureBitDepth` (`Int12`/`Int16`/`Int24`/`Int32`/`Float32`) floor calibration, and GPL-isolated `DNWB` (`FaustRunner`, `SpiceRunner`) pipe execution ([`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)).
- **`.dnam-ckpt` Gram Fingerprinting & `DNCA` `fp16` Coreset Storage ([`src/lib.rs`](src/lib.rs), [`src/ckpt.rs`](src/ckpt.rs)):**
  - Streaming `DNAM-GRAM-INPUT-v1` SHA-256 verification and IEEE-754 RNE `12,000`-frame (`250 ms` at `48 kHz`) `fp16` coreset quantization (`24,000 B` mono / `48,000 B` stereo per stem, `64 B`-aligned; [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md)).

---

## Package Structure

```text
packages/trainer/
├── Cargo.toml                  # dnam-trainer crate manifest & dnam-train binary target
├── moon.yml                    # Rust quality, test, coverage (>= 95%), bench & CLI tasks
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)]), CLI runner, Gram hash & fp16
│   ├── targets.rs              # DistillationTarget, CaptureBitDepth (Int12..Int32, Float32) & stems
│   ├── resampler.rs            # Zero-phase Kaiser-sinc 8..192 kHz <-> 48 kHz audio & trajectory resampler
│   ├── audit.rs                # 3-Phase Control-Space Audit range/switch/stability clamps
│   ├── metrics.rs              # Degree-7 Bezout 3-band bESR, noise qualification & interval audit
│   ├── metrics/
│   │   ├── companion.rs        # 5-scale 32-probe MR-IIR, 33-band 1/3-octave LTAS & SegESR
│   │   └── certification.rs    # PerceptualCertificationSummary, bit-depth floor & 4-tier non-regression
│   ├── whitebox.rs             # 64B DnwbHeader64B binary request/response wire framing
│   ├── whitebox/
│   │   ├── runners.rs          # SpiceRunner (adaptive ODE + PCHIP 4x + 309-tap Kaiser-sinc decimation)
│   │   ├── faust.rs            # FaustRunner recursive faust -json UI parser & DSP runner
│   │   ├── ipc.rs              # Concurrent non-deadlocking DNWB stdin/stdout subprocess transport
│   │   └── session.rs          # Persistent stateful DnwbWorkerSession & multi-packet stream server
│   ├── ckpt.rs                 # .dnam-ckpt (DNCK), Arena 4 (uplo='U') & Arena 5 (DNCA) root
│   ├── ckpt/
│   │   ├── arena.rs            # uplo='U' packing/verification & DNCA (INPC/STMC)
│   │   ├── schema.rs           # DnamCkptHeader, StemRegistryEntry & AuditReport
│   │   ├── view.rs             # DnamCkptView & 64B-aligned container builder
│   │   └── bundle.rs           # DCA1 signatures & zero-copy .dnam runtime export
│   ├── solver.rs               # uplo='U' Gram accumulation, Cholesky solver & bESR helpers
│   ├── solver/
│   │   └── stages.rs           # AuditBounds, Step-0 variance gating & Family-5 surrogate
│   └── bin/
│       └── dnam_train.rs       # dnam-train CLI binary entrypoint (check, dnwb-serve)
├── tests/
│   ├── trainer_spec_verification.rs           # Integration & proptest verify[...] specification suite
│   ├── trainer_multirate_bitdepth_fixtures.rs # Multi-rate, 5-bit-depth & live large-payload DNWB suite
│   └── trainer_ckpt_security.rs               # .dnam-ckpt / DCA1 / DAU1 / DNCA & DNWB hardening suite
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
