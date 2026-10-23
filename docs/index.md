# Dynamic Neural Audio Modeler (DNAM A3) — Documentation & Agent Implementation Roadmap

Welcome to the canonical Single-Source-of-Truth (SSOT) documentation for **Dynamic Neural Audio Modeler — Architecture 3 ("Tri-Physics" Architecture)**.

This repository is structured as a symmetric multi-language monorepo with two top-level directories (`docs/` and `packages/`):
1. **Canonical Specification Documentation (`docs/`):** Pure Single-Source-of-Truth (SSOT) mathematical, DSP, training, control-clamping, and container schema Markdown specifications.
2. **Rust SIMD Engine Core (`packages/engine/`):** Zero-allocation, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full), `FTZ`/`DAZ`-enabled pure Rust core (`dnam-engine`) targeting ARM64 NEON (**Rockchip RK3582 / Darkglass Anagram**), desktop x86_64 AVX2/SSE4.2, and WebAssembly SIMD128 with zero `FSQRT` or `FDIV` instructions at $48\text{ kHz}$ and $3\text{ kHz}$.
3. **Packaged Python Engine Library (`packages/python/`):** Standalone `PyO3` / `maturin` Python wheel wrapping `packages/engine` with zero-copy NumPy audio inference, `.dnam` v3 validation, 3-Phase control clamping, and Reference DSP bindings.
4. **Packaged Node.js / WebAudio Engine Library (`packages/nodejs/`):** Standalone TypeScript / npm package (`@dnam/nodejs`) wrapping `packages/engine` via WebAssembly SIMD128 & N-API bindings + WebAudio `DnamAudioWorkletProcessor`.
5. **PyTorch Differentiable Trainer (`packages/trainer/`):** PyTorch 4-stage deterministic trainer, `.dnam-ckpt` factorized Kronecker iterative stem expansion, Non-Stop 3-Phase Control-Space Clamping Audit, `.dnam` v3 Pydantic schema, and the 8-suite empirical verification suite (imports `packages/python` as Reference DSP).
6. **Rust Real-Time Audio Plugin (`packages/plugin/`):** Real-time VST3/CLAP/Standalone plugin wrapping `packages/engine`, dynamically binding `.dnam` v3 `controls_schema` (`range` knobs and `switch` selectors with safe clamped bounds) and ZOH sample-rate rescaling ($44.1\text{--}96\text{ kHz}$).
7. **Interactive Web Documentation & React WASM Studio (`packages/web/`):** Astro + Starlight + React Islands web application that mounts `docs/*.md` as a comprehensive **Specification Reference** section enriched with interactive React/Canvas DSP visualizers alongside a live in-browser **React WASM Modeler Plugin** (`/playground`, powered by `packages/nodejs`).

---

## 1. Canonical Specification Sitemap

| Document | Scope & Contents |
| :--- | :--- |
| **[`architecture_3_design.md`](architecture_3_design.md)** | Complete original master design specification artifact used during R&D iteration. |
| **[`../scripts/verify_a3_architecture.py`](../scripts/verify_a3_architecture.py)** | Executable 8-suite mathematical, numerical, optimization, and PyTorch empirical verification script. |
| **[`architecture_overview.md`](architecture_overview.md)** | Executive summary of the **18 empirically verified breakthroughs**, end-to-end "Tri-Physics" Mermaid signal-flow diagram, and Rockchip RK3582 / desktop benchmarks. |
| **[`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)** | **Core 1A:** 1× Passive B-Spline Division-Free Algebraic `PolyADAA` (exact `(m_n, d_n)` polynomial identity, proof why post-ADAA high-pass extrapolation fails, linear path separation).<br/>**Core 1B:** Log-Octave, H₂-Normalized Radial-Polar Selective Complex RG-LRU, 3 kHz 4-FMA + Algebraic Newton Unit-Circle Rotor, and `O(log₂ L)` parallel associative scan. |
| **[`core2_modulated_ufdn.md`](core2_modulated_ufdn.md)** | **Core 2:** Unified Stereo In-Loop Diffusive `M-UFDN`, stateless 4-tap Cubic Lagrange reader, Dual-Mode runtime delay transitions (`range` slew vs. `switch` 128-sample raised-cosine dual-pointer crossfade), Schroeder all-pass diffusers, `O(M)` Givens + Householder stereo matrix `U(ψ, v)`, passband-normalized 5 Hz DC blocker, Unity-Slope Coercive-Saturating Lyapunov Loop Law, and 3 kHz Stereo Tremolo / Rotary gate. |
| **[`parametric_control_bus.md`](parametric_control_bus.md)** | **3 kHz Control Bus:** Autonomous 3-band envelope follower & 4-harmonic Chebyshev LFO bank (with Blind Phase-Locked Teacher CNN), Fixed Declared-Range coordinate system, Cardinality-Bounded Node-Anchored Newton/Lobatto basis (`K_p = min(U_p-1, 3)`), Centered ANOVA + Rank-4 Factorization Machine, Step-0 Order-Statistic Optimal-Transport matching, and Step-0 Physical Variance Gating `G_p`. |
| **[`training_and_iterative_resume.md`](training_and_iterative_resume.md)** | **4-Stage Deterministic Training Pipeline:** Step-0 Order-Statistic System-ID, Step-0 SVD-Orthonormal Kronecker `float64` Wiener-Hopf Warm-Start (`λ_init = 10⁻³ τ̄`), `Adam(weight_decay=0.0)` + 20-step warmup + Envelope-Guided Surrogate Delay Gradient + State-Carry TBPTT, Step-N `float64` Cholesky Export Polish (`λ_polish = 10⁻⁶ τ̄`), and **Single-Checkpoint (`.dnam-ckpt`) Two-Tier Iterative Stem Expansion** via per-stem audio sufficient statistics `{G_audio^(s), C_audio^(s)}` (130 KB/stem). |
| **[`clamp_and_audit_registry.md`](clamp_and_audit_registry.md)** | **Non-Stop 3-Phase Control-Space Audit:** Phase 1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), Phase 2 (Stop-Gradient Corner Barrier `L_barrier` & `[CLAMP-RAIL]`), Phase 3 (`[CLAMP-DYNAMIC]` & `[CLAMP-HEAD-GAIN]`), Consolidated Stem Prescription Report, and the canonical **`CL-01`..`CL-18` Clamp & Limit Registry**. |
| **[`dnam_v3_container_schema.md`](dnam_v3_container_schema.md)** | **Container & Checkpoint Schemas:** `.dnam` v3 JSON/binary layout, `.dnam-ckpt` iterative checkpoint layout, Continuous-Time ZOH sample-rate rescaling (44.1–96 kHz), and slimmable `Lite` (`C=4, M=4`) / `Full` (`C=8, M=8`) channel mask rules. |
| **[`monorepo_implementation_guide.md`](monorepo_implementation_guide.md)** | **Deliverable Engineering Blueprints:** Detailed architecture, directory layouts, API contracts, and build configurations for the Web Docs site, Python trainer/DSP package, Native SIMD DSP engine, Audio Plugin, and Node.js WASM/WebAudio package. |
| **[`verification_and_guardrails.md`](verification_and_guardrails.md)** | **Empirical Verification & Guardrail Suite:** Complete reference implementation of all **8 empirical verification test suites** (`Test 1` through `Test 8G`) and automated documentation/clamp guardrail tests. |

---

## 2. Recommended Agent Execution Sequence

When developing this monorepo from the specifications in `docs/`, execute in the following 6 phases:

1. **Phase 1 — Pure Rust SIMD Engine Core (`Cargo.toml`, `packages/engine/`):**
   - Implement the allocation-free, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full) pure Rust engine (`dnam-engine`) with `FTZ`/`DAZ` hardware guards, `PolyADAA`, `SelectiveRgLru`, `ModulatedUfdn`, `ControlBus`, and `.dnam` v3 ZOH loader per [`monorepo_implementation_guide.md`](monorepo_implementation_guide.md).
2. **Phase 2 — Packaged Python & Node.js Engine Libraries (`packages/python/`, `packages/nodejs/`):**
   - Build `packages/python` (`PyO3` + `maturin` wheel) exposing zero-copy NumPy inference and Reference DSP primitives from `packages/engine`.
   - Build `packages/nodejs` (`@dnam/nodejs` via WebAssembly SIMD128 / N-API) exposing `.dnam` v3 validation, 3-Phase clamping, and `DnamAudioWorkletProcessor`.
3. **Phase 3 — PyTorch Differentiable Trainer & 8-Suite Verification (`pyproject.toml`, `packages/trainer/`):**
   - Implement `dnam_trainer` (`control`, `trainer`, `schema`, and differentiable PyTorch `autograd` graph) under `packages/trainer/`, importing `packages/python` as the canonical Reference DSP.
   - Port the 8 verification suites and guardrail tests from [`verification_and_guardrails.md`](verification_and_guardrails.md) into `packages/trainer/tests/` and verify `uv run pytest`, `uv run ruff check`, and `uv run pyrefly check`.
4. **Phase 4 — Real-Time Audio Plugin (`packages/plugin/`):**
   - Wrap `packages/engine` in a real-time VST3/CLAP/Standalone plugin processor with lock-free atomic parameter slots mapped to `.dnam` v3 `controls_schema` (`range` & `switch`) and runtime safe-domain clamping.
5. **Phase 5 — Web Platform, Specification Visualizers & React WASM Studio (`packages/web/`):**
   - Configure Astro + Starlight + `@astrojs/react` (`packages/web/astro.config.mjs`) to mount `docs/*.md` directly as the **Specification Reference** section with KaTeX math and Mermaid diagrams.
   - Build interactive React/Canvas visualizers (`PolyAdaaVisualizer`, `RgLruPoleVisualizer`, `ModulatedUfdnVisualizer`, `ControlBusVisualizer`, `KroneckerResumeVisualizer`, `ClampAuditVisualizer`, `ZohRescalerVisualizer`) embedded across the specification pages, plus the live React + WebAssembly (`AudioWorklet` SIMD128) Modeler Plugin (`DnamPluginRack` powered by `packages/nodejs`).
6. **Phase 6 — End-to-End Cross-Language Parity & Guardrail Certification:**
   - Run all test suites across Rust (`cargo test --workspace`), Python (`uv run pytest`), Node.js (`npm --workspace=packages/nodejs test`), and web platform build (`npm --workspace=packages/web run build`).
