# Neural Audio Modeler (NAM A3) — Documentation & Agent Implementation Roadmap

Welcome to the canonical Single-Source-of-Truth (SSOT) documentation for **Neural Audio Modeler — Architecture 3 ("Tri-Physics" Architecture)**.

This repository is structured as a multi-language monorepo containing:
1. **Project & Web-Based Documentation (`docs/`):** Complete mathematical, DSP, training, control-clamping, and container schema documentation, plus a web-based documentation site.
2. **Python Trainer & Reference DSP (`src/nam_a3/`, `tests/`):** PyTorch 4-stage deterministic trainer, `.nam-ckpt` factorized Kronecker iterative stem expansion, Non-Stop 3-Phase Control-Space Clamping Audit, `.nam` v3 Pydantic schema, reference DSP, and the 8-suite empirical verification suite.
3. **Rust or C++ Native SIMD DSP Engine (`crates/nam-dsp/` or `native/dsp/`):** Zero-allocation, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full), `FTZ`/`DAZ`-enabled native DSP core targeting ARM64 NEON (**Rockchip RK3582 / Darkglass Anagram**) and desktop x86_64 AVX2/SSE4.2 with zero `FSQRT` or `FDIV` instructions at $48\text{ kHz}$ and $3\text{ kHz}$.
4. **Rust or C++ Real-Time Audio Plugin (`crates/nam-plugin/` or `plugin/`):** Real-time VST3/CLAP/AU plugin wrapping the native DSP engine, dynamically binding `.nam` v3 `controls_schema` (`range` knobs and `switch` selectors with safe clamped bounds) and ZOH sample-rate rescaling ($44.1\text{--}96\text{ kHz}$).
5. **Node.js WASM / Native Package (`packages/nodejs/`):** TypeScript SDK, WebAssembly SIMD128 / N-API native bindings, and WebAudio `AudioWorkletProcessor`.

---

## 1. Canonical Specification Sitemap

| Document | Scope & Contents |
| :--- | :--- |
| **[`architecture_3_design.md`](architecture_3_design.md)** | Complete original master design specification artifact used during R&D iteration. |
| **[`../scripts/verify_a3_architecture.py`](../scripts/verify_a3_architecture.py)** | Executable 8-suite mathematical, numerical, optimization, and PyTorch empirical verification script. |
| **[`architecture_overview.md`](architecture_overview.md)** | Executive summary of the **18 empirically verified breakthroughs**, end-to-end "Tri-Physics" Mermaid signal-flow diagram, and Rockchip RK3582 / desktop benchmarks. |
| **[`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)** | **Core 1A:** $1\times$ Passive B-Spline Division-Free Algebraic `PolyADAA` (exact $(m_n, d_n)$ polynomial identity, proof why post-ADAA high-pass extrapolation fails, linear path separation).<br/>**Core 1B:** Log-Octave, $\mathcal{H}_2$-Normalized Radial-Polar Selective Complex RG-LRU, $3\text{ kHz}$ 4-FMA + Algebraic Newton Unit-Circle Rotor, and $O(\log_2 L)$ parallel associative scan. |
| **[`core2_modulated_ufdn.md`](core2_modulated_ufdn.md)** | **Core 2:** Unified Stereo In-Loop Diffusive `M-UFDN`, stateless 4-tap Cubic Lagrange reader, Dual-Mode runtime delay transitions (`range` slew vs. `switch` 128-sample raised-cosine dual-pointer crossfade), Schroeder all-pass diffusers, $O(M)$ Givens + Householder stereo matrix $\mathbf{U}(\psi, \mathbf{v})$, passband-normalized $5\text{ Hz}$ DC blocker, Unity-Slope Coercive-Saturating Lyapunov Loop Law, and $3\text{ kHz}$ Stereo Tremolo / Rotary gate. |
| **[`parametric_control_bus.md`](parametric_control_bus.md)** | **$3\text{ kHz}$ Control Bus:** Autonomous 3-band envelope follower & 4-harmonic Chebyshev LFO bank (with Blind Phase-Locked Teacher CNN), Fixed Declared-Range coordinate system, Cardinality-Bounded Node-Anchored Newton/Lobatto basis ($K_p = \min(U_p-1, 3)$), Centered ANOVA + Rank-4 Factorization Machine, Step-0 Order-Statistic Optimal-Transport matching, and Step-0 Physical Variance Gating $\mathbf{G}_p$. |
| **[`training_and_iterative_resume.md`](training_and_iterative_resume.md)** | **4-Stage Deterministic Training Pipeline:** Step-0 Order-Statistic System-ID, Step-0 SVD-Orthonormal Kronecker `float64` Wiener-Hopf Warm-Start ($\lambda_{\text{init}} = 10^{-3}\bar{\tau}$), `Adam(weight_decay=0.0)` + 20-step warmup + Envelope-Guided Surrogate Delay Gradient + State-Carry TBPTT, Step-$N$ `float64` Cholesky Export Polish ($\lambda_{\text{polish}} = 10^{-6}\bar{\tau}$), and **Single-Checkpoint (`.nam-ckpt`) Two-Tier Iterative Stem Expansion** via per-stem audio sufficient statistics $\{\mathbf{G}_{\text{audio}}^{(s)}, \mathbf{C}_{\text{audio}}^{(s)}\}$. |
| **[`clamp_and_audit_registry.md`](clamp_and_audit_registry.md)** | **Non-Stop 3-Phase Control-Space Audit:** Phase 1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), Phase 2 (Stop-Gradient Corner Barrier $\mathcal{L}_{\text{barrier}}$ & `[CLAMP-RAIL]`), Phase 3 (`[CLAMP-DYNAMIC]` & `[CLAMP-HEAD-GAIN]`), Consolidated Stem Prescription Report, and the canonical **`CL-01`..`CL-18` Clamp & Limit Registry**. |
| **[`nam_v3_container_schema.md`](nam_v3_container_schema.md)** | **Container & Checkpoint Schemas:** `.nam` v3 JSON/binary layout, `.nam-ckpt` iterative checkpoint layout, Continuous-Time ZOH sample-rate rescaling ($44.1\text{--}96\text{ kHz}$), and slimmable `Lite` ($C=4, M=4$) / `Full` ($C=8, M=8$) channel mask rules. |
| **[`monorepo_implementation_guide.md`](monorepo_implementation_guide.md)** | **Deliverable Engineering Blueprints:** Detailed architecture, directory layouts, API contracts, and build configurations for the Web Docs site, Python trainer/DSP package, Native SIMD DSP engine, Audio Plugin, and Node.js WASM/WebAudio package. |
| **[`verification_and_guardrails.md`](verification_and_guardrails.md)** | **Empirical Verification & Guardrail Suite:** Complete reference implementation of all **8 empirical verification test suites** (`Test 1` through `Test 8G`) and automated documentation/clamp guardrail tests. |

---

## 2. Recommended Agent Execution Sequence

When developing this monorepo from the specifications in `docs/`, execute in the following 6 phases:

1. **Phase 1 — Python Reference DSP, Trainer, Schema & 8-Suite Verification (`pyproject.toml`, `src/nam_a3/`, `tests/`):**
   - Scaffold `pyproject.toml` with `uv` targeting Python 3.14 (`py314`), `ruff`, `pyrefly` (`preset = "strict"`), and `pytest`.
   - Implement `nam_a3.dsp`, `nam_a3.control`, `nam_a3.trainer`, and `nam_a3.schema` along with scoped `AGENTS.md` files (`< 2,500` B each).
   - Port the 8 verification suites and guardrail tests from [`verification_and_guardrails.md`](verification_and_guardrails.md) into `tests/` and verify `uv run pytest`, `uv run ruff check`, and `uv run pyrefly check`.
2. **Phase 2 — Native SIMD DSP Engine (`crates/nam-dsp/`):**
   - Implement the allocation-free, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full) native DSP engine with `FTZ`/`DAZ` hardware guards, `PolyADAA`, `SelectiveRgLru`, `ModulatedUfdn`, `ControlBus`, and `.nam` v3 ZOH loader per [`monorepo_implementation_guide.md`](monorepo_implementation_guide.md).
   - Verify with native unit tests confirming parity with the Python reference DSP and zero `FSQRT`/`FDIV` instructions in hot loops.
3. **Phase 3 — Real-Time Audio Plugin (`crates/nam-plugin/`):**
   - Wrap `nam-dsp` in a real-time VST3/CLAP/Standalone plugin processor with lock-free atomic parameter slots mapped to `.nam` v3 `controls_schema` (`range` & `switch`) and runtime safe-domain clamping.
4. **Phase 4 — Node.js WASM / Native & WebAudio Package (`packages/nodejs/`):**
   - Expose `nam-dsp` to Node.js and WebAudio (`AudioWorkletProcessor`) via WebAssembly SIMD128 / N-API bindings and a strict TypeScript SDK.
5. **Phase 5 — Web-Based Documentation Site (`docs/`):**
   - Configure VitePress (`docs/.vitepress/config.mts`) and interactive HTML/Canvas visualizers so `docs/*.md` builds into a searchable, interactive web documentation site.
6. **Phase 6 — End-to-End Cross-Language Parity & Guardrail Certification:**
   - Run all test suites across Python (`uv run pytest`), Rust/C++ (`cargo test --workspace`), Node.js (`node --test`), and web docs build (`npm run docs:build`).
