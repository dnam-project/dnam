---
id: ADR-0018
title: "Pure-Rust Black-Box Signal-Chain Fixtures Crate (dnam-fixtures), Dual-Mode White-Box Distillation & Python Retirement"
status: accepted
date: "2026-03-30"
supersedes:
  - ADR-0012
  - ADR-0014
superseded_by: null
specs:
  - ../specs/architecture/tora_architecture.md
  - ../specs/architecture/psychoacoustic_quality_and_tiers.md
  - ../specs/training/four_stage_pipeline.md
  - ../specs/sdks/typescript_sdk.md
  - ../specs/architecture/project_and_documentation_quality.md
requirements:
  - REQ-1.5
  - REQ-1.6
  - REQ-1.8
  - REQ-6.1
  - REQ-6.2
---

# ADR-0018: Pure-Rust Black-Box Signal-Chain Fixtures Crate (`dnam-fixtures`), Dual-Mode White-Box Distillation & Python Retirement

## 1. Context & Problem Statement

Following [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), all DSP execution, 5-Stage `VarPro` training, desktop Studio, and WebAudio/Node.js SDKs run natively in Rust (`dnam-engine`, `dnam-capture`, `dnam-studio`) and TypeScript (`@dnam-project/dnam`). Retaining the legacy `PyO3` Python SDK (`packages/python`, `ADR-0012`, `ADR-0014`) imposed packaging, ABI, and security surface overhead without serving any active runtime or training consumer (`REQ-1.8`, `REQ-6.1`, `REQ-6.2`). Meanwhile, verifying end-to-end solver recovery across all physical effect classes, sample rates, bit depths, and pathological edge cases (`REQ-1.5`, `REQ-1.6`, `REQ-1.8`) requires both a deterministic pure-Rust black-box ground-truth fixture suite (`packages/fixtures`) and a unified programmatic white-box/grey-box distillation API (`packages/capture`) for external circuit simulators, DSP languages, and plugins.

---

## 2. Considered Candidates & Failure Modes

| Candidate | Workspace Package Graph | Ground-Truth & External Distillation | Validation Matrix Coverage | Failure Mode |
| :--- | :---: | :--- | :--- | :--- |
| **A. Retain `PyO3` Python SDK + Static Pre-Rendered WAV Blobs** | 7 packages (`Rust` + `Node.js` + `Python`/`uv`/`maturin`) | Static pre-rendered WAVs only; cannot probe adaptive excitation | Ad-hoc unit tests | Cannot test closed-loop `AdaptiveStimulusController` (`ADR-0016`) or dynamic knob trajectories; maintains dead `PyO3` FFI surface |
| **B. Acyclic Pure-Rust + TypeScript Workspace (`fixtures <- capture` `[dev-dependencies]` & `engine <- capture` `[dependencies]`, `dnam-engine -> dnam-capture -> dnam-studio`) + `dnam_capture::DistillationTarget` Dual-Mode API (Selected)** | **Pure-Rust + TypeScript (`engine`, `fixtures`, `capture`, `studio`, `js`, `ui` + `tests`; see [`ADR-0020`](0020-modular-monorepo-and-dnam-proj-container.md))** | **Zero-dependency `BlackBoxModel` + `FixtureDeviceMetadata` in `dnam_fixtures` (`fixtures <- capture` `[dev-dependencies]`) + `DistillationTarget` (`render_stem` + `render_frame`), `DNWB` `64 B` pipes (`FaustRunner`, `SpiceRunner`), and `MR-IIR` metrics in `dnam_capture`** | **4-Tier matrix: `3` Oracles (`Oracle-01..03`) + `12` `4x`-oversampled Archetypes (`Chain-01..12`) + `14` Black-Box Fault Decorators (`Cases #1..#26`) + `DNWB` White-Box Runners** | Zero identified failure modes; keeps `dnam-fixtures` 100% engine/capture-agnostic and `dnam-engine` / `dnam-studio` 100% fixture-free; supersedes `ADR-0012` and `ADR-0014` |

---

## 3. Decision & Architectural Consequences

We adopt **Candidate B**, superseding [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md) and [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md):

1. **Retirement of `packages/python` (`PyO3`) & Zero-Dependency `dnam-fixtures` Leaf Crate:**
   - DNAM standardizes on a 100% Rust + TypeScript acyclic Cargo + `pnpm` workspace: **`dnam-fixtures`** has **zero runtime `[dependencies]`** (`dependsOn: []`, exposing `BlackBoxModel` audio I/O plus device-intrinsic `FixtureDeviceMetadata` — `sample_rate_hz`, `FixtureBitDepth` across `Int12`/`Int16`/`Int24`/`Int32`/`Float32`, `FixtureControlSpec`, `FixtureSwitchSpec`, `expected_tail_ms`, `supports_tempo_sync` — while knowing nothing about `Tier`, `DistillationTarget`, `internal_nodes`, `DNWB`, `Oracle-01..03`, or certification metrics) and is consumed **exclusively** via **`fixtures <- capture`** (`[dev-dependencies]`, `scope: "development"` in `packages/capture/moon.yml`) alongside **`engine <- capture`** (`[dependencies]` in `packages/capture/Cargo.toml`), while `dnam-engine` (`dependsOn: []`), `dnam-studio`, and `dnam-js` have zero dependency on `dnam-fixtures` alongside `dnam-ui` (`packages/ui`) and Node/TypeScript `tests/`. All repository guardrails execute natively in TypeScript (`vitest` in `tests/test_guardrails.test.ts`), zero `.py` files remain in `packages/` or `tests/`, and zero `PyO3` / `maturin` / Python build toolchains or external `realfft` / `rustfft` / `ndarray` crates remain in `Cargo.toml` or `mise.toml`.
2. **`DistillationTarget`, `CaptureBitDepth`, `resampler`, `whitebox`, and `metrics` (`MR-IIR`) in `dnam_capture` (`REQ-1.8`):**
   - **`dnam_capture::targets`** (re-exported at crate root) defines **`DistillationTarget`** (`StemStimulus`, `StemStimulusBatch`, `StemResponse`, `StimulusFrame`, `CaptureBitDepth` across `Int12`/`Int16`/`Int24`/`Int32`/`Float32`, implemented directly on `DnamEngine` and `DynamicDnamEngine`) supporting both batch stem rendering (`render_stem`) and frame-by-frame closed-loop stepping (`render_frame` for `AdaptiveStimulusController`), paired with **`dnam_capture::resampler`** (`resample_arbitrary_rate_to_48k`, `resample_48k_to_arbitrary_rate`) so `dnam-engine` stays strictly focused on real-time `.dnam` execution and `dnam-fixtures` stays strictly focused on pure black-box ground-truth audio simulation (`BlackBoxModel`).
   - **`dnam_capture::metrics`** implements the 4-metric perceptual certification suite (`ref[qual.companion.mr-iir]`, `ref[qual.tiers.benchmark-matrix]`): degree-7 Bézout 3-band sum-denominator `bESR` with 3-condition noise-limited qualification and 5-row rectangular joint-interval bounds, 5-scale 32-probe unity-peak bandpass IIR resonator bank **`MR-IIR`** across $M \in \lbrace 256, 512, 1024, 2048, 4096 \rbrace$ with $r_M = \exp(-\pi / M)$ and hop $H = M / 4$, 33-band Welch FFT `LTAS`, and duration-weighted linear `SegESR`.
   - `dnam-capture` exposes the **Unified Dual-Mode API**:
     - **Mode 1 (Streaming Frame-by-Frame Closed-Loop):** `AdaptiveStimulusController::next_frame` driving `DistillationTarget::render_frame` in `< 1.5 ms` per `96 ms` cyclic frame (`train_adaptive_target`, plus automatic `1.0s` Live Hardware-vs-Model Null-Test & `< 5 ms` Auto-Patch).
     - **Mode 2 (1-or-2-Pass Batch Sans-IO):** `generate_stimulus_batch` (Pass 1 baseline schedule), `generate_refinement_stimulus_batch` (Pass 2 targeted `0.5s–1.0s` refinement stems when `Laws 1–3` flag gate opening, `> 18 dB` pre-attenuation, high-$Q$ pole zoom, or Gram conditioning), and `solve_from_responses`.
   - External white-box/grey-box circuit and DSP simulators execute in **`dnam_capture::whitebox`** via **GPL-isolated out-of-process subprocess runners** over `64 B`-header `DNWB` binary stdin/stdout pipes (`run_dnwb_stdio_server`): **`FaustRunner`** (recursive `faust -json` UI parser + multi-rate stereo DSP graphs) and **`SpiceRunner`** (`ngspice` CLI netlists with variable-timestep `PCHIP` interpolation to $4f_s$ + causal `309`-tap Kaiser-sinc anti-aliased decimation to device $f_s$).
3. **4-Tier Pure-Rust Signal-Chain Validation Matrix (`Oracle-01..03`, `Chain-01..12`, `Cases #1..#26`, `DNWB` White-Box Runners):**
   - **Tier 1 (`3` Structural Isomorphism Oracles in `packages/capture`, `Oracle-01..03`):** Exact `TORA` parameter recovery (`0 ULP` / `bESR <= -60 dB`) covering `Oracle-01-256D-Schur-And-Gauge`, `Oracle-02-CP1-ArbitraryP-And-Hadamard-FM` ($P = 6$ and $P = 16$), and `Oracle-03-Zero-WAV-Project-Vault-Resume`.
   - **Tier 2 (`12` `4x`-Oversampled Black-Box Musical Signal-Chain Archetypes in `packages/fixtures`, `Chain-01..12`):** Independent `4x`-oversampled ($4f_s$) pure `f64` black-box physical ODE/SVF/Kautz/FDN signal chains implementing `BlackBoxModel` (`with_format(sample_rate_hz, bit_depth)`) and spanning tube heads, solid-state drives, wavefolders, optical/FET compressors, bass-reflex cabinets, envelope wahs, phasers/rotary, BBD/tape echoes, shimmer/freeze tanks, monophonic octave dividers, and full `4CM` / `WDW` / `2->1` aux-key stereo rigs.
   - **Tier 3 (`Cases #1..#26` across `packages/fixtures` & `packages/capture`):** Fourteen composable pure black-box hardware/interface fault decorators (`AdcClipAndPadDecorator`, `ConverterLoopbackDecorator`, `HumanHandTrajectoryDecorator`, `TubeThermalHysteresisDecorator`, `PotTaperAndSwitchDecorator`, `PolarityAndDcOffsetDecorator`, `GermaniumTempDriftDecorator`, `MainsHumAndNoiseFloorDecorator`, `HardwareNoiseGateDecorator`, `SpringChirpDecorator`, `DigitalQuirksDecorator`, `ScratchyPotAndSteppedAttenuatorDecorator`, `IdleSquealDecorator`, and `MicrophoneCapsuleClipDecorator` for `Case #26` Sub-`0 dBFS` Microphone Capsule / Preamp Clipping & LF Excursion `X_max`) wrapping any `M: BlackBoxModel` and forwarding its `FixtureDeviceMetadata` in `packages/fixtures`, paired with `Ed25519` + 3-Layer Tardos `N = 5` collusion, multi-rate ZOH, and topology verification harnesses in `packages/capture`.
   - **Tier 4 (`DNWB` Out-of-Process White-Box Runners & Multi-Rate / Multi-Bit-Depth Certification in `packages/capture`):** `FaustRunner`, `SpiceRunner`, live large-payload `DNWB` subprocess pipe IPC, and multi-rate (`32 / 44.1 / 48 / 88.2 / 96 / 192 kHz` native and canonical `48 kHz`) + 5-bit-depth (`Int12 / Int16 / Int24 / Int32 / Float32`) 4-metric certification.

---

## 4. Verification & References

- **Supersedes:** [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md), [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md)
- **Canonical Specifications:** [`../specs/architecture/tora_architecture.md`](../specs/architecture/tora_architecture.md), [`../specs/architecture/psychoacoustic_quality_and_tiers.md`](../specs/architecture/psychoacoustic_quality_and_tiers.md), [`../specs/training/four_stage_pipeline.md`](../specs/training/four_stage_pipeline.md), [`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md), [`../specs/architecture/project_and_documentation_quality.md`](../specs/architecture/project_and_documentation_quality.md)
- **Requirements:** [`../requirements/system_goals_and_requirements.md`](../requirements/system_goals_and_requirements.md) (`REQ-1.5`, `REQ-1.6`, `REQ-1.8`, `REQ-6.1`, `REQ-6.2`)
