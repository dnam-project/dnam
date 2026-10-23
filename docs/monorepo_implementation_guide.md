# Monorepo Implementation Guide (Web Docs, Python, Native DSP, Plugin & Node.js WASM)

This document defines the concrete architecture, directory structure, build configurations, and hardware/runtime requirements for all deliverables in the **Dynamic Neural Audio Modeler (DNAM)** monorepo.

---

## 1. Target Monorepo Directory Structure

```text
dynamic-neural-audio-modeler/
├── AGENTS.md                        # Root progressive-disclosure router (< 5,000 B)
├── README.md                        # Project overview, benchmarks & quickstart
├── CONTRIBUTING.md                  # Engineering, numerical & documentation rules
├── CLA.md                           # Contributor License Agreement
├── LICENSE                          # DNAM Source-Available & Branding License 1.0.0
├── pyproject.toml                   # uv workspace root -> ["packages/python", "packages/trainer"]
├── Cargo.toml                       # Cargo workspace root -> ["packages/engine", "packages/python", "packages/plugin"]
├── package.json                     # JS/TS workspace root -> ["packages/nodejs", "packages/web"]
├── docs/                            # 1. Pure Single-Source-of-Truth (SSOT) Markdown Specifications
│   ├── index.md
│   ├── architecture_design.md
│   ├── architecture_overview.md
│   ├── core1_hybrid_ssm_and_polyadaa.md
│   ├── core2_modulated_ufdn.md
│   ├── parametric_control_bus.md
│   ├── training_and_iterative_resume.md
│   ├── clamp_and_audit_registry.md
│   ├── dnam_container_schema.md
│   ├── monorepo_implementation_guide.md
│   └── verification_and_guardrails.md
├── scripts/
│   └── verify_dnam_architecture.py  # Executable 8-suite empirical verification reference
└── packages/                        # 2. All 6 Monorepo Packages (3-Layer Architecture)
    ├── engine/                      # Layer 1: Pure Rust Zero-Allocation SIMD Engine Core (dnam-engine)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs               # DnamEngine, Tier (Lite C=4 / Full C=8), FtzDazGuard
    │       ├── math.rs              # Branchless PolyADAA, Algebraic Rotor & 4-Tap Cubic Lagrange (CL-01..03, CL-06)
    │       ├── core1.rs             # 12-Layer Hybrid Dilated-Conv + Selective Complex RG-LRU (CL-04, CL-05)
    │       ├── core2.rs             # Stereo Diffusive M-UFDN, Switch XFade & Lyapunov Loop (CL-07..11)
    │       └── control.rs           # 3 kHz Decimated ControlBus & Safe-Domain Clamping (CL-12..14)
    ├── python/                      # Layer 2A: Packaged Python Library for Rust Engine (PyO3 / maturin wheel)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── pyproject.toml           # maturin build backend (lightweight, depends only on numpy)
    │   ├── Cargo.toml               # PyO3 zero-copy NumPy bindings to packages/engine
    │   ├── src/
    │   │   ├── lib.rs               # PyO3 module exposing DnamEngine & Reference DSP primitives
    │   │   └── dnam_engine/
    │   │       └── __init__.py      # Typed Python API & Reference DSP wrapper
    │   └── tests/
    │       └── test_engine_bindings.py
    ├── nodejs/                      # Layer 2B: Packaged Node.js & WebAudio Library for Rust Engine (@dnam/nodejs)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── package.json
    │   ├── tsconfig.json
    │   ├── src/
    │   │   └── index.ts             # WASM SIMD128 / N-API SDK, Control Clamper & DnamAudioWorkletProcessor
    │   └── test/
    │       └── dnam.test.mjs        # Node.js test runner suite
    ├── trainer/                     # Layer 3A: PyTorch 4-Stage Differentiable Trainer & .dnam-ckpt Solver
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── pyproject.toml           # Imports packages/python (dnam-engine) as Reference DSP
    │   ├── src/
    │   │   └── dnam_trainer/
    │   │       ├── __init__.py
    │   │       ├── dsp/             # Differentiable PyTorch autograd & Reference DSP bridge (CL-01..11)
    │   │       │   ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │       │   ├── __init__.py
    │   │       │   ├── poly_adaa.py
    │   │       │   ├── selective_rglru.py
    │   │       │   └── modulated_ufdn.py
    │   │       ├── control/         # Lobatto HDMR-FM basis & Non-Stop 3-Phase Audit (CL-12..17)
    │   │       │   ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │       │   ├── __init__.py
    │   │       │   ├── basis.py
    │   │       │   └── audit.py
    │   │       ├── trainer/         # Surrogate Delay Gradient & Kronecker .dnam-ckpt Solver (CL-18)
    │   │       │   ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │       │   ├── __init__.py
    │   │       │   ├── surrogate_delay.py
    │   │       │   └── kronecker_head.py
    │   │       └── schema/          # Pydantic v2 .dnam / .dnam-ckpt schemas & ZOH rescaling
    │   │           ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │           ├── __init__.py
    │   │           └── models.py
    │   └── tests/
    │       ├── AGENTS.md            # Scoped router (< 2,500 B)
    │       ├── test_dnam_verification.py  # All 8 empirical verification suites
    │       └── test_guardrails.py   # AGENTS.md byte budgets, link checker & CL-01..18 verifier
    ├── plugin/                      # Layer 3B: Rust Real-Time Audio Plugin (VST3 / CLAP / Standalone)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs               # Lock-free atomic parameter binding & host ZOH adaptation
    └── web/                         # Layer 3C: Astro + Starlight Docs Site & Interactive React WASM Studio
        ├── AGENTS.md                # Scoped router (< 2,500 B)
        ├── package.json
        ├── astro.config.mjs         # Starlight + @astrojs/react + KaTeX + Mermaid + Vite WASM
        └── src/
            ├── content.config.ts    # Mounts ../../docs/*.md directly into Specification section
            ├── pages/
            │   └── playground.astro # Full-screen Interactive React WASM Modeler Studio
            └── components/
                ├── plugin/          # Live React + WebAssembly (AudioWorklet SIMD128) Modeler Plugin
                │   ├── DnamPluginRack.tsx       # Dynamic range knobs, switch toggles & clamp badges
                │   ├── AudioWorkletEngine.ts    # Real-time WebAudio graph (DI loops / live input -> WASM)
                │   └── TriPhysicsTelemetry.tsx  # Live 60 fps spectrum, waveform & pole-zero monitor
                └── visualizers/     # Interactive Specification Visualizers (React + HTML5 Canvas)
                    ├── PolyAdaaVisualizer.tsx         # Core 1A: 1x B-Spline PolyADAA vs. Naive/Unfiltered ADAA
                    ├── RgLruPoleVisualizer.tsx        # Core 1B: Radial-Polar vs. Cartesian chord dip & 3 kHz Rotor
                    ├── ModulatedUfdnVisualizer.tsx    # Core 2: Switch 128-sample XFade, U(psi,v) & Lyapunov Loop Law
                    ├── ControlBusVisualizer.tsx       # Control Bus: Lobatto HDMR-FM & Order-Statistic LFO Transport
                    ├── KroneckerResumeVisualizer.tsx  # Trainer: Surrogate Delay basin & .dnam-ckpt rank-1 expansion
                    ├── ClampAuditVisualizer.tsx       # Safety: Interactive 3-Phase Clamping polygon & CL-01..18 explorer
                    └── ZohRescalerVisualizer.tsx      # Container: Continuous-Time ZOH 44.1-96 kHz invariant rescaling
```

---

## 2. Deliverable 1: Web Platform, Specification Section & Interactive React WASM Plugin (`packages/web/`, `docs/`)

- **Zero-Clutter SSOT Specification Mounting:** All canonical Markdown specifications live in `docs/*.md` with zero framework clutter. `packages/web/src/content.config.ts` uses Astro 5's Content Layer (`glob({ pattern: "**/*.md", base: "../../docs" })`) to mount `docs/*.md` directly as the **Specification Reference** section of the Starlight documentation website with full KaTeX (`remark-math` + `rehype-katex`) and Mermaid diagram support.
- **Interactive Specification Visualizers (`packages/web/src/components/visualizers/`):** Each specification section pairs its mathematical derivations with interactive React + HTML5 Canvas visualizers (`client:visible` islands):
  1. **`PolyAdaaVisualizer` ([`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)):** Interactive drive, frequency, and asymmetry sliders comparing Naive cubic waveshaping, unfiltered ADAA1, post-ADAA high-pass extrapolation ($+9.54\text{ dB}$ Nyquist spike), and **$1\times$ Passive B-Spline $(0.875, 0.125)$ Division-Free `PolyADAA`** ($38.87\text{ dB}$ aliasing suppression).
  2. **`RgLruPoleVisualizer` ([`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)):** Unit-disk ($|z| \le 1$) complex plane comparing **Radial-Polar Split** interpolation along $|z| = r_c \le 0.9995$ against **Cartesian linear interpolation** ($-20.7\text{ dB}$ mid-sweep chord dip), plus the $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor.
  3. **`ModulatedUfdnVisualizer` ([`core2_modulated_ufdn.md`](core2_modulated_ufdn.md)):** Demonstrates `range` delay slewing vs. `switch` **128-sample raised-cosine dual-pointer crossfading** (zero Doppler pitch-zipper), $O(M)$ Givens + Householder unitary mixing $\mathbf{U}(\psi, \mathbf{v})$, and the **Unity-Slope Coercive-Saturating Lyapunov Loop Law** ($S'(0) \equiv 1.0000$).
  4. **`ControlBusVisualizer` ([`parametric_control_bus.md`](parametric_control_bus.md)):** Interactive node count ($U_p = 2..6$) showing **Cardinality-Bounded Node-Anchored Newton/Lobatto basis** ($K_p = \min(U_p-1, 3)$), Centered ANOVA + Rank-4 Factorization Machine surfaces, and **Order-Statistic Optimal-Transport** LFO trajectory matching.
  5. **`KroneckerResumeVisualizer` ([`training_and_iterative_resume.md`](training_and_iterative_resume.md)):** Visualizes the **Envelope-Guided Surrogate Delay Gradient** convex loss basin and interactive `.dnam-ckpt` rank-1 stem addition ($\{\mathbf{G}_{\text{audio}}^{(s)}, \mathbf{C}_{\text{audio}}^{(s)}\}$, $0.00\text{ dB}$ old-stem drift).
  6. **`ClampAuditVisualizer` ([`clamp_and_audit_registry.md`](clamp_and_audit_registry.md)):** Interactive 2D/switch control-space coverage map demonstrating Phase 1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), Phase 2 (`[CLAMP-RAIL]`), and Phase 3 (`[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`), plus a filterable `CL-01`..`CL-18` registry inspector.
  7. **`ZohRescalerVisualizer` ([`dnam_container_schema.md`](dnam_container_schema.md)):** Interactive host sample-rate selector ($44.1\text{--}96\text{ kHz}$) verifying invariant Hz frequency response under Continuous-Time ZOH rescaling.
- **Interactive React WASM Modeler Plugin (`packages/web/src/components/plugin/DnamPluginRack.tsx`):**
  - Mounted via `<DnamPluginRack client:only="react" />` and powered by the WASM SIMD128 build of `packages/engine` and `packages/nodejs` (`@dnam/nodejs`).
  - Dynamically renders rotary knobs (`range`) and discrete selectors (`switch`) from any loaded `.dnam` model's `controls_schema`, enforcing `safe_clamped_range`, `active_options`, and `conditional_max` bounds with real-time `clamp_warning` badges, built-in DI guitar/synth test stems, live WebAudio microphone/instrument input, and $60\text{ fps}$ DSP telemetry.

---

## 3. Deliverable 2: Pure Rust SIMD Engine Core (`packages/engine/`)

### 3.1 Why Unified Rust (`packages/engine`) Powers Python, Node.js, Plugin & WASM
A single zero-allocation Rust crate (`packages/engine`, `dnam-engine`) compiles directly to:
1. **ARM64 NEON (`aarch64-unknown-linux-gnu` / `aarch64-apple-darwin`)** for Rockchip RK3582 (Darkglass Anagram) and Apple Silicon.
2. **x86_64 AVX2/FMA (`x86_64-unknown-linux-gnu` / `x86_64-pc-windows-msvc`)** for desktop DAWs and Python/Node.js native wheels.
3. **WebAssembly SIMD128 (`wasm32-unknown-unknown`)** for browser `AudioWorklet` and Node.js WASM runtimes—eliminating any C++/Python/JS DSP port divergence.

### 3.2 Hardware Microarchitecture & Real-Time Audio Rules
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

## 4. Deliverable 3: Packaged Python & Node.js Engine Libraries (`packages/python/`, `packages/nodejs/`)

### 4.1 Packaged Python Engine Library (`packages/python/`)
- **Build & Packaging:** Built via **`PyO3` + `maturin`** (`packages/python/pyproject.toml` and `packages/python/Cargo.toml`) into a standalone, lightweight Python wheel (`dnam-engine`) that depends only on `numpy` (no PyTorch dependency required for inference users).
- **Zero-Copy NumPy & Reference DSP API:**
  1. `DnamEngine`: Zero-copy NumPy stereo block processor (`process_block(left_in, right_in)`), `.dnam` loader, Continuous-Time ZOH sample-rate adapter, and 3-Phase control clamper.
  2. Reference DSP Primitives (`poly_adaa`, `selective_rglru`, `modulated_ufdn`, `control_bus`): Direct Python bindings to the exact Rust `packages/engine` kernels, imported by `packages/trainer` as the canonical Reference DSP and certification oracle.

### 4.2 Packaged Node.js & WebAudio Engine Library (`packages/nodejs/`)
- **Package Exports (`packages/nodejs/src/index.ts`, `@dnam/nodejs`):**
  1. `validateDnamModel(json: unknown): DnamModel`: Validates `.dnam` containers and returns typed control metadata and active clamp warnings.
  2. `clampControlValues(model: DnamModel, rawControls: Record<string, number | string>)`: Applies `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-STABILITY]` (`conditional_max`) rules identically to the Rust and Python bindings and returns normalized declared-range coordinates $u_p \in [-1, +1]$.
  3. `DnamProcessor`: Synchronous stereo block processor backed by `packages/engine` (WASM SIMD128 / N-API) for Node.js CLI/server audio rendering and WebAudio `AudioWorkletProcessor` integration.
  4. `DnamAudioWorkletProcessor`: Ready-to-register WebAudio `AudioWorkletProcessor` class using pre-allocated `Float32Array` ring buffers (`128`-sample render quantum = 8 Control-Bus ticks per quantum at $48\text{ kHz}$).

---

## 5. Deliverable 4: PyTorch Differentiable Trainer (`packages/trainer/`)

- **Environment & Dependencies:** Managed via `uv` (`pyproject.toml` / `packages/trainer/pyproject.toml`) targeting `requires-python = ">=3.14"`. Imports `packages/python` (`dnam-engine`) as a workspace dependency for Reference DSP evaluation and Phase 1/3 export certification. Core dependencies: `dnam-engine`, `numpy`, `torch`, `pydantic`, `polars`, `altair`. Dev dependencies: `pytest`, `hypothesis`, `ruff`, `pyrefly`.
- **Strict Typing & Lint:** Must pass `uv run ruff check`, `uv run ruff format --check`, and `uv run pyrefly check` (`preset = "strict"`).
- **Clamp Registry Alignment:** Every `CL-01`..`CL-18` row in [`clamp_and_audit_registry.md`](clamp_and_audit_registry.md) maps to its designated module under `packages/trainer/src/dnam_trainer/` (and `packages/engine/src/`) and is programmatically verified by `packages/trainer/tests/test_guardrails.py`.

---

## 6. Deliverable 5: Real-Time Audio Plugin (`packages/plugin/`)

- **Plugin Topologies Supported:** True-Stereo (`2 -> 2`), Mono-to-Stereo (`1 -> 2`, where `x_mid = L_in, x_side = 0`), and Dual-Mono (`2 -> 2` with Givens angle $\psi = 0$).
- **Dynamic `controls_schema` Parameter Binding:**
  - Pre-allocates lock-free `AtomicU32` (for `f32` `range` bits) and `AtomicUsize` (for `switch` option indices) slots.
  - When a `.dnam` file is loaded, maps each entry in `controls_schema` to a host parameter with automatic enforcement of `safe_clamped_range`, `active_options`, and `conditional_max` ceilings.
  - Displays any non-null `clamp_warning` badge in the plugin UI so the user knows when a knob or switch is safety-clamped by the model's training coverage.
- **Host Sample-Rate Adaptation (`prepare_to_play` / `initialize`):**
  Calls `engine.set_sample_rate(host_sample_rate_hz)` to apply the exact Continuous-Time ZOH rescaling formulas from [`dnam_container_schema.md`](dnam_container_schema.md).
