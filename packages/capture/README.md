# `dnam-capture` — Pure-IO Capture, `.dnam-proj` (`DNPJ`) Container & `VarPro` Engine

**`dnam-capture`** (`packages/capture/`) is the 100% safe-Rust (`#![forbid(unsafe_code)]`, `wasm32`-safe) pure IO-buffer capture, single-file `.dnam-proj` (`DNPJ`, `v2.0`) project container, and `< 25 ms` Analytical Variable-Projection (`VarPro`) system-identification crate for **Dynamic Neural Audio Modeler (DNAM)**.

By linking [`dnam-engine`](../engine/README.md) directly (`engine <- capture` `[dependencies]`, `fixtures <- capture` `[dev-dependencies]`), `dnam-capture` executes the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`), `256-D` `step_frame_with_feature_bus`, `3 kHz` Parametric Control Bus, and `.dnam` binary framing across native hosts, Node.js `N-API`, Browser `WebWorker`s, and Cloudflare Workers (`ref[arch.isomorphism.two-sweep]`, [`ADR-0017`](../../docs/adr/0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), [`ADR-0020`](../../docs/adr/0020-modular-monorepo-and-dnam-proj-container.md)).

---

## Key Capabilities

- **Single-File Cloud-Ready `.dnam-proj` (`DNPJ`, `v2.0`) Container & Two-Path Capture Vault ([`src/proj.rs`](src/proj.rs)):**
  - Front-loaded `64 B` Preamble + `64 B` Chunk Directory (`CALB`, `SPCM`, `INPC`, `STMC`, `STAT`, `WGHT`, `AUTH` with `crc32c` + `sha256`) + `64 B`-aligned RFC 8785 JCS `DnamProjManifest` inspectable via a single `HTTP Range: bytes=0-65535` read (`parse_proj_manifest_range`, [`project_container_format.md`](../../docs/specs/containers/project_container_format.md)).
  - Compositional Capture Vault (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`), `< 1 ms` raw `SPCM` stripping (`compact_to_capsules`), and `< 0.2 ms` runtime `.dnam` emission (`export_runtime_dnam`).
- **Closed-Loop `AdaptiveStimulusController` & Unified Dual-Mode API ([`src/stimulus.rs`](src/stimulus.rs)):**
  - `< 1.5 ms` per `96 ms` cyclic frame `A1..A6` synthesis, `Laws 0.A–0.F` hardware safeguards (including `Law 0.F` sub-`0 dBFS` microphone capsule / external preamp clip protection & `recapture_clipped_stem`), `Laws 1–3` + `Theorem 4.5–4.8` $D$-optimal active learning & Two-Episode Pump-and-Probe Bloom, Mode-1 streaming (`train_adaptive_target` + `1.0s` `run_live_null_test_and_autopatch`), and Mode-2 sans-IO batch (`generate_stimulus_batch`, `generate_refinement_stimulus_batch`, `solve_from_responses`).
- **5-Bit-Depth In-Memory WAV Codec & 4-Regime Sub-Sample Alignment ([`src/align.rs`](src/align.rs), [`src/align/regimes.rs`](src/align/regimes.rs)):**
  - Buffer-in/buffer-out `Int12`/`Int16`/`Int24`/`Int32`/`Float32` WAV PCM encoding/decoding, `<f4` SHA-256 provenance hashing, Phase-0 `0.5s` `CALB` deconvolution, and `<= 0.02`-sample 4-regime Kaiser-sinc alignment (`CleanLinear`, `PolarityInverted`, `DispersiveAllPass`, `NonlinearEnvelopeFallback`, [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md)).
- **Analytical `VarPro` & `uplo='U'` Cholesky Solver ([`src/solver.rs`](src/solver.rs)):**
  - Column-major upper-triangular (`LAPACK uplo='U'`) `f64` Gram accumulation and regularized 2-channel Cholesky solver across all 4 slimmable tiers (`Lite` `64-D`, `Full` `128-D`, `Pro` `192-D`, `Ultra` `256-D`, [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md)).
- **4-Metric Psychoacoustic Certification, Resampler & `DNWB` White-Box Distillation ([`src/metrics.rs`](src/metrics.rs), [`src/resampler.rs`](src/resampler.rs), [`src/whitebox.rs`](src/whitebox.rs)):**
  - Evaluates noise-qualified 3-band `bESR`, 5-scale $\times$ 32-probe resonator bank `MR-IIR`, 33-band `1/3`-octave Welch `LTAS`, and `SegESR`, plus zero-phase Kaiser-sinc `8..192 kHz` $\leftrightarrow$ `48 kHz` resampling and `64 B` `DNWB` (`FaustRunner`, `SpiceRunner`) pipe execution ([`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)).

---

## Package Structure

```text
packages/capture/
├── Cargo.toml                  # dnam-capture pure library manifest (zero CLI / OS audio)
├── moon.yml                    # Rust quality, test, coverage (>= 95%) & bench tasks
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)]), Gram hash & fp16 helpers
│   ├── align.rs                # 5-bit-depth WAV PCM codec, <f4 SHA-256 & sub-sample alignment
│   ├── align/
│   │   └── regimes.rs          # CALB deconvolution, 4-regime Kaiser-sinc align & null/thermal/pad gates
│   ├── stimulus.rs             # AdaptiveStimulusController (<1.5ms/96ms frame) & capsule sealing
│   ├── stimulus/
│   │   ├── fft.rs              # Radix-2 f64 FFT/IFFT, disjoint-comb scheduler & Schroeder phases
│   │   ├── waveforms.rs        # A1..A6 primitives, Theorem 4.7-4.8 Bloom & fp16 coreset/PSD encoders
│   │   ├── safeguards.rs       # Laws 0.A-0.F hardware safeguards & CaptureWarning
│   │   ├── laws.rs             # Laws 1-3, Theorem 4.5-4.6 sweeps/topologies & 4-Gate Cramér-Rao stop
│   │   ├── pipeline.rs         # Mode 1 train_adaptive_target & 1.0s Live Null-Test / <5ms Auto-Patch
│   │   └── batch.rs            # Mode 2 sans-IO batch API & 400ms mic-pad recapture_clipped_stem
│   ├── targets.rs              # DistillationTarget, CaptureBitDepth & stem batch types
│   ├── resampler.rs            # Zero-phase Kaiser-sinc 8..192 kHz <-> 48 kHz resampler
│   ├── audit.rs                # 3-Phase Control-Space Audit range/switch/corner/stability clamps
│   ├── proj.rs                 # .dnam-proj (DNPJ v2.0) constants, uplo='U' & fp16 helpers
│   ├── proj/
│   │   ├── schema.rs           # DnamProjManifest, StemRegistryEntry (enabled) & JCS digest
│   │   ├── directory.rs        # 64B Preamble, 64B Chunk Directory, Range parser & AUTH
│   │   ├── capsules.rs         # CALB (64B + IR), INPC (24.6/48.6 KB), STMC (28.8/52.8 KB) & STAT
│   │   ├── bundle.rs           # DnamProjView, builder, compact_to_capsules & export_runtime_dnam
│   │   └── vault.rs            # Two-Path Capture Vault (add/remove/replace/set_capture_enabled)
│   ├── metrics.rs              # Degree-7 Bezout 3-band bESR, noise qualification & interval audit
│   ├── metrics/
│   │   ├── companion.rs        # 5-scale 32-probe MR-IIR, 33-band 1/3-octave LTAS & SegESR
│   │   └── certification.rs    # PerceptualCertificationSummary & 4-tier non-regression
│   ├── whitebox.rs             # 64B DnwbHeader64B binary request/response wire framing
│   ├── whitebox/
│   │   ├── runners.rs          # SpiceRunner (adaptive ODE + PCHIP 4x + 309-tap Kaiser-sinc)
│   │   ├── faust.rs            # FaustRunner recursive faust -json UI parser & DSP runner
│   │   ├── ipc.rs              # Concurrent non-deadlocking DNWB stdin/stdout subprocess transport
│   │   └── session.rs          # Persistent stateful DnwbWorkerSession & stream server
│   ├── solver.rs               # uplo='U' Gram accumulation, Cholesky solver & bESR helpers
│   └── solver/
│       └── stages.rs           # AuditBounds, Step-0 variance gating & Family-5 surrogate
├── tests/
│   ├── capture_spec_verification.rs           # Integration & proptest verify[...] suite
│   ├── capture_multirate_bitdepth_fixtures.rs # Multi-rate & 5-bit-depth Chain-01..12 suite
│   ├── capture_stimulus_and_safeguards.rs     # AdaptiveStimulusController, Laws 0.A-0.F, Laws 1-3 & Bloom suite
│   ├── capture_dual_mode_and_align.rs         # CALB, 4-regime align, Theorem 4.5-4.6 & Mode 1/2 Null-Test suite
│   ├── capture_proj_security.rs               # .dnam-proj Range, capsules, zero-DC & vault suite
│   ├── capture_proj_dau1_security.rs          # AUTH, DAU1/DMU1 & live DNWB subprocess suite
│   ├── capture_align_and_capsules_edges.rs    # WAV codec, schema JCS & capsule edge suite
│   ├── capture_proj_bundle_and_vault_edges.rs # Directory, bijection & stereo vault edge suite
│   └── capture_proj_adversarial_regressions.rs# P>8 Mode A, >64KB Range, STAT PSD & DMU1 growth suite
└── benches/
    └── bench_capture.rs        # Criterion Cholesky, compact_to_capsules & export_runtime_dnam
```

---

## Quickstart

```bash
# Run all capture quality gates (format-check, clippy, typecheck, build, test)
moon check capture

# Enforce >= 95% line coverage
moon run capture:coverage
```
