# Monorepo Implementation Guide (Web Docs, Python, Native DSP, Plugin & Node.js WASM)

This document defines the concrete architecture, directory structure, build configurations, and hardware/runtime requirements for all five deliverables in the **Dynamic Neural Audio Modeler (DNAM Architecture 3)** monorepo.

---

## 1. Target Monorepo Directory Structure

```text
dynamic-neural-audio-modeler/
├── AGENTS.md                        # Root progressive-disclosure router (< 5,000 B)
├── README.md                        # Project overview, benchmarks & quickstart
├── CONTRIBUTING.md                  # Engineering, numerical & documentation rules
├── CLA.md                           # Contributor License Agreement
├── pyproject.toml                   # uv Python project (py314, ruff, pyrefly strict, pytest)
├── Cargo.toml                       # Rust workspace (crates/dnam-dsp, crates/dnam-plugin)
├── package.json                     # Node.js workspace (packages/nodejs, VitePress docs)
├── docs/                            # Single-Source-of-Truth Markdown + VitePress Web Docs
│   ├── .vitepress/
│   │   └── config.mts               # Web documentation site configuration & sidebar
│   ├── public/
│   │   └── explorer.html            # Interactive HTML/Canvas Tri-Physics & Control Explorer
│   ├── index.md
│   ├── architecture_overview.md
│   ├── core1_hybrid_ssm_and_polyadaa.md
│   ├── core2_modulated_ufdn.md
│   ├── parametric_control_bus.md
│   ├── training_and_iterative_resume.md
│   ├── clamp_and_audit_registry.md
│   ├── dnam_v3_container_schema.md
│   ├── monorepo_implementation_guide.md
│   └── verification_and_guardrails.md
├── src/
│   └── dnam/                        # Python Trainer & Reference DSP Package
│       ├── __init__.py
│       ├── dsp/
│       │   ├── AGENTS.md            # Scoped router (< 2,500 B)
│       │   ├── __init__.py
│       │   ├── poly_adaa.py         # Core 1A: 1x B-Spline Division-Free PolyADAA (CL-01, CL-02)
│       │   ├── selective_rglru.py   # Core 1B: 3 kHz Rotor, H2 Norm & Radial-Polar RG-LRU (CL-03..05)
│       │   └── modulated_ufdn.py    # Core 2: Lagrange, Switch XFade, U(psi,v) & Loop Law (CL-06..11)
│       ├── control/
│       │   ├── AGENTS.md            # Scoped router (< 2,500 B)
│       │   ├── __init__.py
│       │   ├── basis.py             # Declared-range clamp, Newton/Lobatto basis & Variance Gate (CL-12..14)
│       │   └── audit.py             # Non-Stop 3-Phase Control-Space Audit (CL-15..17)
│       ├── trainer/
│       │   ├── AGENTS.md            # Scoped router (< 2,500 B)
│       │   ├── __init__.py
│       │   ├── surrogate_delay.py   # Envelope-Guided Multi-Scale Surrogate Delay Gradient
│       │   └── kronecker_head.py    # SVD Projector, Dual-Ridge float64 Solve & .dnam-ckpt Resume (CL-18)
│       └── schema/
│           ├── AGENTS.md            # Scoped router (< 2,500 B)
│           ├── __init__.py
│           └── models.py            # Pydantic v2 .dnam v3 & .dnam-ckpt schemas + ZOH rescaling
├── tests/
│   ├── AGENTS.md                    # Scoped router (< 2,500 B)
│   ├── test_architecture_3_verification.py  # All 8 empirical verification suites
│   └── test_guardrails.py           # AGENTS.md byte budgets, link checker & CL-01..18 verifier
├── crates/
│   ├── dnam-dsp/                    # Rust Zero-Allocation SIMD Native DSP Engine
│   │   ├── AGENTS.md                # Scoped router (< 2,500 B)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs               # DnamEngine, Tier (Lite C=4 / Full C=8), FtzDazGuard
│   │       ├── math.rs              # Branchless PolyADAA, Algebraic Rotor & 4-Tap Cubic Lagrange
│   │       ├── core1.rs             # 12-Layer Hybrid Dilated-Conv + Selective Complex RG-LRU
│   │       ├── core2.rs             # Stereo Diffusive M-UFDN, Switch XFade & Lyapunov Loop
│   │       └── control.rs           # 3 kHz Decimated ControlBus & Safe-Domain Clamping
│   └── dnam-plugin/                 # Rust Real-Time Audio Plugin (VST3 / CLAP / Standalone)
│       ├── AGENTS.md                # Scoped router (< 2,500 B)
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs               # Lock-free atomic parameter binding & host ZOH adaptation
└── packages/
    └── nodejs/                      # Node.js WASM / N-API & WebAudio AudioWorklet Package
        ├── AGENTS.md                # Scoped router (< 2,500 B)
        ├── package.json
        ├── tsconfig.json
        ├── src/
        │   └── index.ts             # TypeScript SDK, Safe Control Clamper & AudioWorkletProcessor
        └── test/
            └── dnam.test.mjs        # Node.js test runner suite
```

---

## 2. Deliverable 1: Web-Based Documentation (`docs/`)

- **Dual-Purpose Design:** Every Markdown file in `docs/*.md` serves simultaneously as the canonical Single Source of Truth for AI agents and as a page in the VitePress web documentation site (`docs/.vitepress/config.mts`).
- **Math & Diagrams:** Enable KaTeX math rendering (`markdown: { math: true }`) and Mermaid flowcharts in VitePress so all equations in [`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md), [`core2_modulated_ufdn.md`](core2_modulated_ufdn.md), and [`parametric_control_bus.md`](parametric_control_bus.md) render natively in the browser.
- **Interactive Explorer (`docs/public/explorer.html`):** Provide a self-contained HTML5/Canvas interactive visual explorer illustrating:
  1. `PolyADAA` division-free cubic/even waveshaping and $1\times$ B-spline aliasing suppression vs. `LeakyReLU`.
  2. Radial-Polar Split Selective Complex RG-LRU pole trajectory vs. Cartesian chord dip.
  3. `M-UFDN` unity-slope coercive-saturating Lyapunov loop law and 3-Phase Control-Space Clamping.

---

## 3. Deliverable 2: Python Trainer & Reference DSP (`src/dnam/`, `tests/`)

- **Environment & Dependencies:** Managed via `uv` (`pyproject.toml`) targeting `requires-python = ">=3.14"`. Core dependencies: `numpy`, `torch`, `pydantic`, `polars`, `altair`, `numba`, `pedalboard`. Dev dependencies: `pytest`, `hypothesis`, `ruff`, `pyrefly`.
- **Strict Typing & Lint:** Must pass `uv run ruff check`, `uv run ruff format --check`, and `uv run pyrefly check` (`preset = "strict"`).
- **Clamp Registry Alignment:** Every `CL-01`..`CL-18` row in [`clamp_and_audit_registry.md`](clamp_and_audit_registry.md) maps to its designated module under `src/dnam/` and is programmatically verified by `tests/test_guardrails.py`.

---

## 4. Deliverable 3: Rust Native SIMD DSP Engine (`crates/dnam-dsp/`)

### 4.1 Why Unified Rust (`dnam-dsp`) Is Recommended Across Native, Plugin & WASM
A single zero-allocation Rust crate (`crates/dnam-dsp`) compiles directly to:
1. **ARM64 NEON (`aarch64-unknown-linux-gnu` / `aarch64-apple-darwin`)** for Rockchip RK3582 (Darkglass Anagram) and Apple Silicon.
2. **x86_64 AVX2/FMA (`x86_64-unknown-linux-gnu` / `x86_64-pc-windows-msvc`)** for desktop DAWs.
3. **WebAssembly SIMD128 (`wasm32-unknown-unknown`)** for browser `AudioWorklet` and Node.js WASM runtimes—eliminating any C++/JS port divergence.
*(A C++20 header/library wrapper exposing `extern "C"` ABI bindings from `dnam-dsp` can also be included for C++ hosts).*

### 4.2 Hardware Microarchitecture & Real-Time Audio Rules
1. **16-Byte SIMD Alignment (`#[repr(align(16))]`):**
   - `Lite` tier uses `C = 4, M = 4` (1 `float32x4_t` register per layer).
   - `Full` tier uses `C = 8, M = 8` (2 `float32x4_t` registers per layer).
2. **Hardware Denormal Immunity (`FtzDazGuard`):**
   Decaying IIR/RG-LRU poles ($r_c \le 0.9995$) and FDN delay tails ($g_{\text{lin},m} \le 0.9990$) enter the IEEE-754 subnormal range ($< 1.18 \times 10^{-38}$) during silence, which can trigger $100\times$ microcode traps if `FTZ`/`DAZ` is disabled. `FtzDazGuard::new()` sets:
   - **ARM64 (`aarch64`):** Bit 24 (`FZ = 1`) of `FPCR` (`mrs x0, fpcr; orr x0, x0, #(1 << 24); msr fpcr, x0`).
   - **x86_64:** `FTZ` (`bit 15`) and `DAZ` (`bit 6`) in `MXCSR`.
   - **Software Fallback (WASM / portable):** Adds $+10^{-30}\text{ f32}$ DC bias injection or branchless flush before state write-back.
3. **Zero Audio-Thread Allocations:**
   All delay ring buffers (`fp16` storage converted via single-cycle `FCVTL`/`FCVTN` on ARM64 NEON, or pre-allocated `f32` ring buffers with power-of-two bitmask wrap `(write_idx - offset) & mask`), all-pass buffers, and control caches are pre-allocated during `DnamEngine::new(config, host_sample_rate)`. `process_block(&mut self, left_in, right_in, left_out, right_out)` performs **zero heap allocations, zero locks, and zero system calls**.

---

## 5. Deliverable 4: Real-Time Audio Plugin (`crates/dnam-plugin/`)

- **Plugin Topologies Supported:** True-Stereo (`2 -> 2`), Mono-to-Stereo (`1 -> 2`, where `x_mid = L_in, x_side = 0`), and Dual-Mono (`2 -> 2` with Givens angle $\psi = 0$).
- **Dynamic `controls_schema` Parameter Binding:**
  - Pre-allocates lock-free `AtomicU32` (for `f32` `range` bits) and `AtomicUsize` (for `switch` option indices) slots.
  - When a `.dnam` v3 file is loaded, maps each entry in `controls_schema` to a host parameter with automatic enforcement of `safe_clamped_range`, `active_options`, and `conditional_max` ceilings.
  - Displays any non-null `clamp_warning` badge in the plugin UI so the user knows when a knob or switch is safety-clamped by the model's training coverage.
- **Host Sample-Rate Adaptation (`prepare_to_play` / `initialize`):**
  Calls `engine.set_sample_rate(host_sample_rate_hz)` to apply the exact Continuous-Time ZOH rescaling formulas from [`dnam_v3_container_schema.md`](dnam_v3_container_schema.md).

---

## 6. Deliverable 5: Node.js WASM / Native & WebAudio Package (`packages/nodejs/`)

- **Package Exports (`packages/nodejs/src/index.ts`):**
  1. `validateDnamV3Model(json: unknown): DnamV3Model`: Validates `.dnam` v3 containers and returns typed control metadata and active clamp warnings.
  2. `clampControlValues(model: DnamV3Model, rawControls: Record<string, number | string>)`: Applies `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-STABILITY]` (`conditional_max`) rules identically to the Python and Rust engines and returns normalized declared-range coordinates $u_p \in [-1, +1]$.
  3. `DnamProcessor`: Synchronous stereo block processor for Node.js CLI/server audio rendering and WebAudio `AudioWorkletProcessor` integration.
  4. `DnamAudioWorkletProcessor`: Ready-to-register WebAudio `AudioWorkletProcessor` class using pre-allocated `Float32Array` ring buffers (`128`-sample render quantum = 8 Control-Bus ticks per quantum at $48\text{ kHz}$).
