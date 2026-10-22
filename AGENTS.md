# Dynamic Neural Audio Modeler (DNAM A3) — Agent Development & Discovery Guide

**Dynamic Neural Audio Modeler (DNAM Architecture 3 / "Tri-Physics" Architecture)** is a real-time stereo neural audio modeling, differentiable DSP, and parametric system-identification monorepo unifying:
1. **Core 1:** 12-layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + Radial-Polar Split Selective Complex RG-LRU stack.
2. **Core 2:** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`).
3. **Control Bus:** $3\text{ kHz}$ Autonomous Envelopes/LFOs + Node-Anchored Newton/Lobatto HDMR & Rank-4 Factorization Machine for user-definable `range` and `switch` controls, backed by **Non-Stop 3-Phase Control-Space Clamping** and **Single-Checkpoint (`.dnam-ckpt`) Iterative Stem Expansion**.

## 1. Monorepo Target Deliverables & Tooling Rules

- **Web & Markdown Documentation (`docs/`):** Single-Source-of-Truth (SSOT) Markdown specifications + web-based documentation site (VitePress) with interactive visualizers.
- **Python Trainer & Reference DSP (`src/dnam/`, `tests/`):** Manage with **`uv`** (`uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, `uv run pyrefly check`). Target Python 3.14 (`py314`) with strict typing (`preset = "strict"`). Never use `# type: ignore`.
- **Native SIMD DSP Engine (`crates/dnam-dsp/`):** Zero-allocation real-time audio path, 16-byte SIMD aligned (`C=4` Lite / `C=8` Full), hardware `FTZ`/`DAZ` enabled, and **zero `FSQRT`, `FDIV`, or `libm` calls** at $48\text{ kHz}$ and $3\text{ kHz}$.
- **Real-Time Audio Plugin (`crates/dnam-plugin/`):** VST3/CLAP/AU plugin wrapping the native DSP engine with lock-free parameter binding for dynamic `.dnam` v3 `controls_schema` (`range` & `switch`) and ZOH sample-rate rescaling ($44.1\text{--}96\text{ kHz}$).
- **Node.js WASM / Native Package (`packages/nodejs/`):** TypeScript SDK + WebAssembly SIMD128 / N-API bindings + WebAudio `AudioWorkletProcessor`.
- **Git & CLA:** Sign off commits via `git commit -s` per [`CLA.md`](CLA.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero `FSQRT` / `FDIV` in Real-Time Loops:** Use division-free algebraic `PolyADAA` ($m_n, d_n$ factorization) with $1\times$ passive B-spline weights $(0.875, 0.125)$ and the $3\text{ kHz}$ 4-FMA + Newton unit-circle rotor ($y_1 = 1.5 - 0.5(\hat{c}^2 + \hat{s}^2)$).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce all bounds in [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) ($r_c \le 0.9995$, $g_{\text{lin},m} \le 0.9990$, $|a_{\text{diff},m}| \le 0.75$, $\tau_m \ge 4.0$, passband-normalized DC blocker $\alpha_{\text{dc}} = \frac{1+R_{\text{dc}}}{2}$, and unity-slope cubic loop saturator $S'(0) \equiv 1.0000$).
3. **Non-Stop 3-Phase Control Clamping:** Training never pauses for user input. Automatically clamp unobserved ranges (`[CLAMP-RANGE]`), switches (`[CLAMP-SWITCH]`), and unstable corners (`[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`).
4. **Factorized Kronecker `.dnam-ckpt` Resume:** Store per-stem audio sufficient statistics $\{\mathbf{G}_{\text{audio}}^{(s)}, \mathbf{C}_{\text{audio}}^{(s)}\}$ ($130\text{ KB/stem}$) so new stems, range expansions, and SVD rotations assemble in $<1\text{ ms}$.

## 3. Progressive Disclosure & Documentation Discovery Map

| Subsystem / Domain | Canonical Documentation (SSOT) |
| :--- | :--- |
| **Master Index & Roadmap** | [`docs/index.md`](docs/index.md), [`docs/monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md) |
| **Master Spec & Overview** | [`docs/architecture_3_design.md`](docs/architecture_3_design.md), [`docs/architecture_overview.md`](docs/architecture_overview.md) |
| **Core 1: `PolyADAA` & RG-LRU** | [`docs/core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md) |
| **Core 2: Diffusive `M-UFDN`** | [`docs/core2_modulated_ufdn.md`](docs/core2_modulated_ufdn.md) |
| **Parametric Control Bus** | [`docs/parametric_control_bus.md`](docs/parametric_control_bus.md) |
| **4-Stage Trainer & `.dnam-ckpt`** | [`docs/training_and_iterative_resume.md`](docs/training_and_iterative_resume.md) |
| **3-Phase Audit & Clamps (`CL-XX`)** | [`docs/clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) |
| **`.dnam` v3 Schema & ZOH** | [`docs/dnam_v3_container_schema.md`](docs/dnam_v3_container_schema.md) |
| **Verification & Guardrails** | [`scripts/verify_a3_architecture.py`](scripts/verify_a3_architecture.py), [`docs/verification_and_guardrails.md`](docs/verification_and_guardrails.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Byte Budgets:** Keep root `AGENTS.md` `< 5,000` B and each scoped `AGENTS.md` `< 2,500` B.
- **Zero Duplication:** Keep equations, clamp tables, and schemas in `docs/` and link to them.
