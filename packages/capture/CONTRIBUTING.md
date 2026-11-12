# Contributing to `dnam-capture` (`packages/capture/`)

This guide covers engineering, numerical accuracy, safety, testing, and coverage standards for **`dnam-capture`** ([`README.md`](README.md)), the pure-IO capture, `.dnam-proj` (`DNPJ`) container, and Analytical Variable-Projection (`VarPro`) system-identification crate for **DNAM**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md), [`src/AGENTS.md`](src/AGENTS.md), [`src/align/AGENTS.md`](src/align/AGENTS.md), [`src/stimulus/AGENTS.md`](src/stimulus/AGENTS.md), [`src/proj/AGENTS.md`](src/proj/AGENTS.md), [`src/metrics/AGENTS.md`](src/metrics/AGENTS.md), [`src/whitebox/AGENTS.md`](src/whitebox/AGENTS.md), [`src/solver/AGENTS.md`](src/solver/AGENTS.md), and [`tests/AGENTS.md`](tests/AGENTS.md):

- **`src/lib.rs`:** Crate root (`#![forbid(unsafe_code)]`, pure IO buffers, zero CLI/OS audio), `DNAM-GRAM-INPUT-v2` SHA-256 fingerprint (`compute_gram_input_fingerprint`), and IEEE-754 RNE `fp16` coreset quantization (`f32_to_fp16_bits`, `fp16_bits_to_f32`).
- **`src/align.rs`, `src/align/regimes.rs`, `src/targets.rs`, `src/resampler.rs` & `src/audit.rs`:** 5-bit-depth (`Int12`/`Int16`/`Int24`/`Int32`/`Float32`) in-memory WAV PCM codec, `<f4` SHA-256 provenance digest, Phase-0 `CALB` loopback deconvolution, 4-regime Kaiser-sinc sub-sample alignment (`<= 0.02` samples), `< 0.05 dB` acoustic null homing, `<= 0.15 dB` thermal recovery gate, `400 ms` mic-pad probe ratio, `DistillationTarget`, zero-phase Kaiser-sinc `8..192 kHz <-> 48 kHz` resampling, and Non-Stop 3-Phase Control-Space Audit helpers.
- **`src/stimulus.rs` & `src/stimulus/{fft,waveforms,safeguards,laws,pipeline,batch}.rs`:** `< 1.5 ms` closed-loop `AdaptiveStimulusController`, `A1..A6` excitation primitives, `Laws 0.A–0.F` hardware safeguards (`Law 0.F` sub-`0 dBFS` mic-capsule/preamp clip & `recapture_clipped_stem`), `Laws 1–3` & `Theorem 4.5–4.8` active learning, Mode-1 streaming (`train_adaptive_target` + `1.0s` `run_live_null_test_and_autopatch`), and Mode-2 sans-IO batch (`generate_stimulus_batch`, `generate_refinement_stimulus_batch`, `solve_from_responses`).
- **`src/proj.rs` & `src/proj/{schema,directory,capsules,bundle,vault}.rs`:** Single-file cloud-ready `.dnam-proj` (`DNPJ`, `v2.0`-only) container (`parse_proj_manifest_range`, `DnamProjView`, `build_dnam_proj_container`, `compact_to_capsules`, `< 0.2 ms` `export_runtime_dnam`), `CALB`/`SPCM`/`INPC`/`STMC`/`STAT`/`WGHT`/`AUTH` chunk framing, `uplo='U'` packing, and Two-Path Capture Vault (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`).
- **`src/metrics.rs` & `src/metrics/{companion,certification}.rs`:** 4-metric psychoacoustic certification suite (`bESR`, 5-scale 32-probe `MR-IIR`, 33-band Welch `LTAS`, `SegESR`) and 4-tier non-regression verification.
- **`src/whitebox.rs` & `src/whitebox/{runners,faust,ipc,session}.rs`:** `64 B` `DnwbHeader64B` pipe IPC framing, `FaustRunner`, `SpiceRunner` (`PCHIP` $4f_s$ + `309`-tap Kaiser-sinc $4\times \to 1\times$ decimation), and persistent `DnwbWorkerSession`.
- **`src/solver.rs` & `src/solver/stages.rs`:** Column-major `uplo='U'` Gram accumulation, regularized `f64` Cholesky solver, degree-7 Bézout 3-band crossover weights, sum-denominator Noise-Subtracted `bESR`, Step-0 variance gating, and Non-Stop 3-Phase Control-Space Audit bounds.
- **`tests/*.rs`:** Integration, adversarial boundary, and `proptest!` verification suites (`verify[...]`, see [`tests/AGENTS.md`](tests/AGENTS.md)).
- **`benches/bench_capture.rs`:** Criterion throughput benchmarks for `uplo='U'` Cholesky solves, `compact_to_capsules`, and `export_runtime_dnam`.

---

## 2. Core Invariants

1. **100% Safe Rust (`#![forbid(unsafe_code)]`), Pure IO Buffers & File Hygiene:**
   - `dnam-capture` enforces `#![forbid(unsafe_code)]` across all library, test, and benchmark targets (`ref[repo.code.type-safety-and-imports]`), zero `mod.rs` files, zero `clap` or `cpal` dependencies, and `< 900` lines per `.rs` file.
2. **Exact Engine-Capture Isomorphism (`ref[arch.isomorphism.two-sweep]`, `ref[arch.isomorphism.audit-wiring]`):**
   - Never duplicate DSP operator math (`Op-T`, `Op-M`, `Op-N`), `3 kHz` control-bus recurrences, or `.dnam` binary framing; depend directly on [`dnam-engine`](../engine/README.md).
3. **`f64` Sufficient Statistics & `uplo='U'` Packing (`ref[container.proj.stat-uplo-u]`, `ref[train.gram.accuracy-contract]`):**
   - Accumulate and factor all Gram matrices $\mathbf{G}$ and cross-correlations $\mathbf{C}$ in IEEE-754 `f64` column-major upper-triangular order (`uplo='U'`).
4. **Non-Stop 3-Phase Control-Space Audit (`ref[train.audit.phase1.clamp-stability]`):**
   - Never block unattended capture or vault editing on interactive prompts; clamp unobserved control ranges, switches, and stability bounds automatically.

---

## 3. Quality Gates & Verification Commands

Run all checks from the workspace root via `moon`:

```bash
moon check capture
moon run capture:coverage
moon run capture:bench
```
