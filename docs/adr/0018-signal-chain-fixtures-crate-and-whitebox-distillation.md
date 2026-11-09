---
id: ADR-0018
title: "Pure-Rust White-Box Signal-Chain Fixtures Crate, Dual-Mode Distillation API & Retirement of Legacy Python SDK"
status: accepted
date: "2026-03-30"
supersedes:
  - ADR-0012
  - ADR-0014
superseded_by: null
specs:
  - ../specs/architecture/tora_architecture.md
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

# ADR-0018: Pure-Rust White-Box Signal-Chain Fixtures Crate, Dual-Mode Distillation API & Retirement of Legacy Python SDK

## 1. Context & Problem Statement

Following [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), all DSP execution, 5-Stage `VarPro` training, CLI tooling, desktop Studio, and WebAudio/Node.js SDKs run natively in Rust (`dnam-engine`, `dnam-trainer`, `dnam-studio`) and TypeScript (`@dnam-project/dnam`). Retaining the legacy `PyO3` Python SDK (`packages/python`, `ADR-0012`, `ADR-0014`) imposed packaging, ABI, and security surface overhead without serving any active runtime or training consumer (`REQ-1.8`, `REQ-6.1`, `REQ-6.2`). Meanwhile, verifying end-to-end solver recovery across all physical effect classes and pathological edge cases (`REQ-1.5`, `REQ-1.6`, `REQ-1.8`) requires both a deterministic pure-Rust white-box fixture suite and a unified programmatic distillation API for external circuit simulators, DSP languages, and plugins.

---

## 2. Considered Candidates & Failure Modes

| Candidate | Workspace Package Graph | Ground-Truth & External Distillation | Validation Matrix Coverage | Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **A. Retain `PyO3` Python SDK + Static Pre-Rendered WAV Blobs** | 7 packages (`Rust` + `Node.js` + `Python`/`uv`/`maturin`) | Static pre-rendered WAVs only; cannot probe adaptive excitation | Ad-hoc unit tests | Cannot test closed-loop `AdaptiveStimulusController` (`ADR-0016`) or dynamic knob trajectories; maintains dead `PyO3` FFI surface |
| **B. 6-Package Acyclic Pure-Rust + TypeScript Workspace (`dnam-engine -> dnam-trainer -> (dev-dep) dnam-fixtures -> dnam-studio`) + `dnam_trainer::DistillationTarget` Dual-Mode API + 3-Tier Validation Matrix (Selected)** | **6 packages (`engine`, `fixtures`, `trainer`, `studio`, `js`, `ui` + `tests`)** | **`#![forbid(unsafe_code)]` `DistillationTarget` (`render_stem` + `render_frame`) in `dnam_trainer` + `DNWB` `64 B` pipes (`FaustRunner`, `SpiceRunner`)** | **`3` Oracles (`Oracle-01..03`) + `12` `4x`-oversampled Archetypes (`Chain-01..12`) + `26` Adversaries (`Cases #1..#26`)** | Zero identified failure modes; keeps `dnam-engine` 100% training-free; supersedes `ADR-0012` and `ADR-0014` |

---

## 3. Decision & Architectural Consequences

We adopt **Candidate B**, superseding [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md) and [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md):

1. **Complete Retirement of Python / `PyO3` (`packages/python`) & Acyclic 6-Package Dependency Graph:**
   - DNAM standardizes on a 100% Rust + TypeScript 6-package acyclic Cargo + `pnpm` workspace: **`dnam-engine -> dnam-trainer -> (dev-dep) dnam-fixtures` and `dnam-trainer -> dnam-studio`**, alongside `dnam-js` (`packages/js`), `dnam-ui` (`packages/ui`), and Node/TypeScript `tests/`. Specifically, **`dnam-engine` is 100% training-free**, **`dnam-fixtures` depends on `dnam-engine` and `dnam-trainer`**, and **`dnam-trainer` depends on `dnam-fixtures` as a `dev-dependency`**. All repository guardrails execute natively in TypeScript (`vitest` in `tests/test_guardrails.test.ts`), and zero `.py` files or Python/`uv` toolchains remain in the repository.
2. **`DistillationTarget` in `dnam_trainer::targets`, Unified Dual-Mode API & GPL-Isolated External Runners (`REQ-1.8`):**
   - **`dnam_trainer::targets`** (re-exported as `dnam_trainer::DistillationTarget`) defines the **`DistillationTarget`** trait (`StemStimulus`, `StemStimulusBatch`, `StemResponse`, `StimulusFrame`, `ToraWhiteBoxTarget`) supporting both batch stem rendering (`render_stem`) and frame-by-frame closed-loop stepping (`render_frame` for `AdaptiveStimulusController`), keeping `dnam-engine` strictly focused on real-time `.dnam` execution while allowing `dnam-fixtures` and `dnam-trainer` to implement and consume `DistillationTarget` cleanly.
   - `dnam-trainer` exposes the **Unified Dual-Mode API**:
     - **Mode 1 (Streaming Frame-by-Frame Closed-Loop):** `AdaptiveStimulusController::next_frame` driving `DistillationTarget::render_frame` in `< 1.5 ms` per `96 ms` cyclic frame (`train_adaptive_target`, plus automatic `1.0s` Live Hardware-vs-Model Null-Test & `< 5 ms` Auto-Patch).
     - **Mode 2 (1-or-2-Pass Batch Sans-IO):** `generate_stimulus_batch` (Pass 1 baseline schedule), `generate_refinement_stimulus_batch` (Pass 2 targeted `0.5s–1.0s` refinement stems when `Laws 1–3` flag gate opening, `> 18 dB` pre-attenuation, high-$Q$ pole zoom, or Gram conditioning), and `solve_from_responses`.
   - External white-box/grey-box circuit and DSP simulators execute via **GPL-isolated out-of-process subprocess runners** over `64 B`-header `DNWB` binary stdin/stdout pipes: **`FaustRunner`** (`faust` CLI DSP graphs) and **`SpiceRunner`** (`ngspice` CLI netlists with variable-timestep `PCHIP` interpolation + Kaiser-sinc anti-aliased decimation to `48 kHz`), paired with deterministic analytical Rust `f64` reference twins in CI.
3. **3-Tier Pure-Rust Signal-Chain Validation Matrix (`Oracle-01..03`, `Chain-01..12`, `Cases #1..#26`):**
   - **Tier 1 (`3` Structural Isomorphism Oracles, `Oracle-01..03`):** Exact `TORA` parameter recovery (`0 ULP` / `B-ESR <= -60 dB`) covering `Oracle-01-256D-Schur-And-Gauge`, `Oracle-02-CP1-ArbitraryP-And-Hadamard-FM` ($P = 6$ and $P = 16$), and `Oracle-03-Zero-WAV-Checkpoint-Resume`.
   - **Tier 2 (`12` `4x`-Oversampled Physical Musical Signal-Chain Archetypes, `Chain-01..12`):** Independent `4x`-oversampled (`192 kHz`) physical ODE/WDF/FDN signal chains spanning tube heads, solid-state drives, wavefolders, optical/FET compressors, bass-reflex cabinets, envelope wahs, phasers/rotary, BBD/tape echoes, shimmer/freeze tanks, monophonic octave dividers, and full `4CM` / `WDW` / `2->1` aux-key stereo rigs (`B-ESR <= -35..-42 dB`).
   - **Tier 3 (`26` Composable Hardware Adversaries, `Cases #1..#26`):** Composable `HardwareAdversary<T>` fault decorators covering hysteresis memory jumps, hard gating, autonomous LFOs, long tank tails, mains hum, idle squeal, USB clock slips, sub-harmonic divider chatter, dispersive all-pass chains, uncentered `Switch` defaults, $P = 20$ mega-rigs, and **`Case #26` (`MicrophoneCapsuleClipDecorator`: Sub-`0 dBFS` Microphone Capsule / Preamp Clipping & LF Excursion `X_max`)**.

---

## 4. Verification & References

- **Supersedes:** [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md), [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md)
- **Canonical Specifications:** [`../specs/architecture/tora_architecture.md`](../specs/architecture/tora_architecture.md), [`../specs/training/four_stage_pipeline.md`](../specs/training/four_stage_pipeline.md), [`../specs/sdks/typescript_sdk.md`](../specs/sdks/typescript_sdk.md), [`../specs/architecture/project_and_documentation_quality.md`](../specs/architecture/project_and_documentation_quality.md)
- **Requirements:** [`../requirements/system_goals_and_requirements.md`](../requirements/system_goals_and_requirements.md) (`REQ-1.5`, `REQ-1.6`, `REQ-1.8`, `REQ-6.1`, `REQ-6.2`)

