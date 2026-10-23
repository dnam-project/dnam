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
│   ├── digital_marketplace_and_provenance.md
│   ├── monorepo_implementation_guide.md
│   └── verification_and_guardrails.md
├── scripts/
│   └── verify_dnam_architecture.py  # Executable 8-suite empirical verification reference
└── packages/                        # 2. All 6 Monorepo Packages (3-Layer Architecture)
    ├── engine/                      # Layer 1: Pure Rust #![no_std] + alloc, Zero-Audio-Alloc SIMD Engine Core (dnam-engine)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── Cargo.toml
    │   ├── src/
    │   │   ├── lib.rs               # DnamEngine<B>, DynamicDnamEngine, Tier (Lite/Full/Ultra), FtzDazGuard
    │   │   ├── memory.rs            # CacheLine64, 64B AlignedBuffer (page pre-faulting), ArenaStorage & Scratch
    │   │   ├── schema.rs            # #![no_std] serde + serde_json .dnam/.dnam-ckpt header, validator & skew
    │   │   ├── provenance.rs        # CryptoProvider (Zeroizing CEK), RustCryptoProvider, ChaCha20 & watermarks
    │   │   ├── container.rs         # DnamPreamble, Arena 0..3 (uplo='U'), 4 KB in-place editor & ZOH rescaler
    │   │   ├── math.rs              # SimdVec4 & MathBackend traits, branchless PolyADAA, Rotor & Lagrange (CL-01..03, CL-06)
    │   │   ├── math/
    │   │   │   ├── scalar.rs        # Portable 16B-aligned ScalarBackend & reference oracle
    │   │   │   ├── neon.rs          # ARM64 NEON (float32x4_t, vfmaq_f32, hardware __crc32cd)
    │   │   │   ├── arm_sme.rs       # ARMv9.2-A Scalable Matrix Extension (ARM SME / SME2) + NEON Hybrid
    │   │   │   ├── x86_simd.rs      # x86-64 SSE4.2 / AVX2 + FMA
    │   │   │   └── wasm_simd128.rs  # WebAssembly SIMD128 (v128)
    │   │   ├── core1.rs             # 12-Layer Hybrid Dilated-Conv + Selective Complex RG-LRU (CL-04, CL-05)
    │   │   ├── core2.rs             # Stereo Diffusive M-UFDN, Switch XFade & Lyapunov Loop (CL-07..11)
    │   │   ├── control.rs           # 3 kHz Decimated ControlBus, Anti-Zipper Smoother & Clamping (CL-12..14)
    │   │   └── ffi.rs               # Zero-allocation extern "C" ABI for WASM & C/C++ embedded audio hosts
    │   ├── tests/
    │   │   ├── empirical_verification.rs  # Black-box empirical verification suites (Tests 1-5 & 8)
    │   │   ├── backend_parity.rs    # Differential parity across Scalar, NEON, ARM SME, x86 & WASM
    │   │   └── realtime_safety.rs   # Counting GlobalAllocator (0-alloc proof) + asm FSQRT/FDIV audit
    │   └── benches/
    │       ├── engine_bench.rs      # DNAM audio budget & backend speedup telemetry table
    │       └── criterion_bench.rs   # Criterion statistical micro- and macro-benchmarks
    ├── python/                      # Layer 2A: Packaged Python Library for Rust Engine (PyO3 / maturin wheel)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── README.md                # Package overview & quickstart
    │   ├── pyproject.toml           # maturin build backend (python-source = "src", depends on numpy & pydantic)
    │   ├── Cargo.toml               # PyO3 (abi3-py314) zero-copy NumPy FFI bindings -> rust/lib.rs
    │   ├── rust/                    # Modular Zero-Copy PyO3 FFI Layer (dnam_engine._native)
    │   │   ├── lib.rs               # #[pymodule] _native entrypoint & typed exception mapping
    │   │   ├── engine.rs            # NativeDnamEngine zero-copy NumPy block processor, backend dispatch & GIL release
    │   │   ├── container.rs         # Container validation, builder, in-place editor, clamping, ZOH & uplo='U' FFI
    │   │   ├── dsp.rs               # Reference DSP FFI kernels (Core 1, Core 2 & Control Bus)
    │   │   └── provenance.rs        # JCS SHA-256, Ed25519 messages, ChaCha20 arena, Tardos/Gauge watermark & 160-D hash FFI
    │   ├── src/                     # Idiomatic Python 3.14 Interface (dnam_engine)
    │   │   └── dnam_engine/
    │   │       ├── __init__.py      # Curated top-level exports & __all__
    │   │       ├── _native.pyi      # PEP 561 strict type stubs for dnam_engine._native
    │   │       ├── py.typed         # PEP 561 marker
    │   │       ├── exceptions.py    # Typed exception hierarchy (DnamError, DnamContainerError, DnamControlError, DnamValidationError)
    │   │       ├── container.py     # First-class DnamContainer (.dnam/.dnam-ckpt object model), GramMatrix & crc32c
    │   │       ├── control.py       # ControlRack, ControlSpace, callable LobattoBasis, Rank4Fm & LFO/softclip kernels
    │   │       ├── dsp.py           # Callable stateful PolyAdaa, SelectiveRgLru, ModulatedUfdn & algebraic kernels
    │   │       ├── provenance.py    # JCS digest, Ed25519 signing helpers, ChaCha20 arena, watermarking & 160-D structural hash
    │   │       ├── engine.py        # Callable DnamEngine (pedalboard-compatible), hot-reload, @property accessors & context manager
    │   │       └── schema/          # Strongly typed Pydantic v2 schemas, settings, DSP params & telemetry models
    │   │           ├── __init__.py
    │   │           ├── base.py      # StrictModel, OpenMetadataModel & core enums (Tier, BackendKind, ControlKind)
    │   │           ├── controls.py  # ControlUiHints, ControlSchemaDef, PresetDef, ClampedControlsResult, LobattoParams
    │   │           ├── header.py    # DnamJsonHeader, ModelMetadata, IoConfig, SlimmableTiersConfig, Core1/Core2/ControlBusConfig
    │   │           ├── signatures.py # CreatorSignature, MarketplaceReceipt, AuthorizedDevice, DatasetProvenance, WatermarkParams
    │   │           └── runtime.py   # DnamEngineSettings, DnamTelemetrySnapshot, ZohRescaledParams, PolyAdaa/RgLru/MufdnParams
    │   └── tests/                   # Modular Unit, Property-Based (Hypothesis) & Symbolic (CrossHair) Test Suite
    │       ├── conftest.py          # Shared fixtures & Hypothesis profiles (dev, ci, debug, smt)
    │       ├── strategies.py        # Reusable Hypothesis, hypothesis.extra.numpy & hypothesis-jsonschema strategies
    │       ├── test_schema.py       # Pydantic v2 models, JSON-Schema fuzzing & CrossHair symbolic tests
    │       ├── test_container.py    # DnamContainer object model, 4 KB in-place edit, ZOH & GramMatrix property tests
    │       ├── test_control.py      # ControlSpace, ControlRack, LobattoBasis, Rank4Fm & 3-Phase clamping property tests
    │       ├── test_dsp.py          # PolyAdaa, SelectiveRgLru & ModulatedUfdn unit & Hypothesis property tests
    │       ├── test_provenance.py   # JCS SHA-256, ChaCha20, Dual-Mechanism buyer watermarking & 160-D hash property tests
    │       └── test_engine.py       # Callable DnamEngine, SIMD vs. Scalar parity, hot-reload, ControlRack & property tests
    ├── nodejs/                      # Layer 2B: Packaged Node.js & WebAudio Library for Rust Engine (@dnam/nodejs)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── README.md                # Package overview & quickstart
    │   ├── Cargo.toml               # Native N-API cdylib linking packages/engine (cargo-zigbuild ready)
    │   ├── build.rs                 # Target-aware dynamic N-API symbol resolution flags
    │   ├── package.json
    │   ├── tsconfig.json
    │   ├── scripts/
    │   │   └── build_wasm.mjs       # Builds Native N-API (.node via cargo/zigbuild) + WASM SIMD128 (.wasm)
    │   ├── src/
    │   │   ├── lib.rs               # Thin zero-copy Rust N-API v8 bindings
    │   │   ├── index.ts             # Public TypeScript SDK barrel exports
    │   │   ├── types.ts             # Domain interfaces & telemetry types
    │   │   ├── model.ts             # validateDnamModel, clampControlValues & computeZohRescaledParams
    │   │   ├── processor.ts         # Idiomatic DnamProcessor interface (auto: native N-API | WASM SIMD128)
    │   │   ├── worklet.ts           # Ready-to-register WebAudio DnamAudioWorkletProcessor
    │   │   └── backends/
    │   │       ├── native.ts        # Low-level Node-API driver
    │   │       └── wasm.ts          # Low-level synchronous WebAssembly SIMD128 driver
    │   └── test/
    │       └── dnam.test.mjs        # Node.js test runner + fast-check property tests & >=95% coverage
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
1. **16-Byte SIMD Alignment (`#[repr(align(16))]`) & Rockchip RK3582 Dual-A76 Core Isolation:**
   - On the Darkglass Anagram (**Rockchip RK3582**), KosmOS isolates the **2× ARM Cortex-A76 performance cores** ($2.4\text{ GHz}$, dual 128-bit NEON pipes, $64\text{ KB}$ L1D + $512\text{ KB}$ L2 per core) for real-time `JACK2` / `mod-host` audio DSP, while the **4× Cortex-A55 cores** handle Linux system tasks and the **LVGL** touchscreen UI. Because a series signal chain executes sequentially on a single Cortex-A76 core, all active blocks share that core's $64\text{ KB}$ L1D cache across buffers and time-share its 32 architectural NEON registers (`v0`–`v31`) per block call:
   - **`Lite` tier (`C = 4, M = 4`, $\sim 1,245\text{ MACs/sample}$):** Uses 1 `float32x4_t` NEON register per vector ($\sim 14/32$ registers active, $3.7\text{ KB}$ Core 1 state).
   - **`Full` tier (`C = 8, M = 8`, $\sim 4,625\text{ MACs/sample}$):** Uses 2 `float32x4_t` NEON registers per vector ($\sim 26/32$ registers active, $7.5\text{ KB}$ Core 1 state), keeping an entire layer's state and coefficients 100% resident in `v0`–`v31` in a single pass with zero stack spills.
   - **`Ultra` tier (`C = 12, M = 8`, $\sim 9,950\text{ MACs/sample}$):** Uses 3 `float32x4_t` NEON registers per vector ($11.2\text{ KB}$ Core 1 state), executed via a **2-pass `8 + 4` register-tiled loop** (`channels 0..7` followed by `channels 8..11`) on 32-register ARM64 NEON to guarantee zero stack spilling while delivering $50\%$ more channels than NAM A2 Full at $1.22\times$ higher speed.
2. **Hardware Denormal Immunity (`FtzDazGuard`):**
   Decaying IIR/RG-LRU poles ($r_c \le 0.9995$) and FDN delay tails ($g_{\text{lin},m} \le 0.9990$) enter the IEEE-754 subnormal range ($< 1.18 \times 10^{-38}$) during silence, which can trigger $100\times$ microcode traps if `FTZ`/`DAZ` is disabled. `FtzDazGuard::new()` sets:
   - **ARM64 (`aarch64`):** Bit 24 (`FZ = 1`) of `FPCR` (`mrs x0, fpcr; orr x0, x0, #(1 << 24); msr fpcr, x0`).
   - **x86_64:** `FTZ` (`bit 15`) and `DAZ` (`bit 6`) in `MXCSR`.
   - **Software Fallback (WASM / portable):** Adds $+10^{-30}\text{ f32}$ DC bias injection or branchless flush before state write-back.
3. **Zero Audio-Thread Allocations:**
   All delay ring buffers (`fp16` storage converted via single-cycle `FCVTL`/`FCVTN` on ARM64 NEON, or pre-allocated `f32` ring buffers with power-of-two bitmask wrap `(write_idx - offset) & mask`), all-pass buffers, and control caches are pre-allocated during `DnamEngine::new(config, host_sample_rate)`. `process_block(&mut self, left_in, right_in, left_out, right_out)` performs **zero heap allocations, zero locks, and zero system calls**.

---

## 4. Deliverable 3: Packaged Python & Node.js Engine Libraries (`packages/python/`, `packages/nodejs/`)

Both Layer 2 packages wrap a **thin, zero-copy Rust FFI binding core** inside an **idiomatic, strictly-typed target-language interface**, support cross-compilation via **`zig` (`maturin --zig` / `cargo-zigbuild`)**, and enforce a full target-language code-quality suite (formatter, linter, strict type checker, unit + property-based tests, and $\ge 95\%$ test coverage in the target language without duplicating Rust-side unit/coverage tests on the FFI glue crates).

### 4.1 Packaged Python Engine Library (`packages/python/` — Phase 2A)
- **Build, Packaging & Cross-Compilation:** Built via **`PyO3` (`abi3-py314`) + `maturin`** (`packages/python/pyproject.toml` with `python-source = "src"` and `packages/python/Cargo.toml` with `path = "rust/lib.rs"`) into a standalone, lightweight Python wheel (`dnam-engine`) that depends only on `numpy` and `pydantic` at runtime (no PyTorch dependency required for inference users) and cross-compiles via `uv run maturin build --release --zig --target <triple>`.
- **Two-Layer Architecture (Modular PyO3 Core in `rust/` + Idiomatic Python 3.14 Interface in `src/dnam_engine/`):**
  1. **Modular Zero-Copy FFI Core (`rust/{lib,engine,container,dsp,provenance}.rs` $\to$ `dnam_engine._native` + `_native.pyi`):** Marshals contiguous `float32`/`float64` NumPy slices (`PyReadonlyArray1`, `PyReadwriteArray1`) with zero copies and releases the Python GIL (`py.allow_threads`) during SIMD block execution.
  2. **Idiomatic Python 3.14 Interface (`src/dnam_engine/`):**
     - `exceptions.py`: Typed exception hierarchy (`DnamError`, `DnamValidationError`, `DnamContainerError`, `DnamControlError`, `DnamSampleRateError`).
     - `schema/` (`base.py`, `controls.py`, `header.py`, `signatures.py`, `runtime.py`): Strongly typed `IntEnum`/`StrEnum` (`Tier`, `BackendKind`, `ControlKind`, `VerificationStatus`) and **Pydantic v2 (`BaseModel`, `ConfigDict(frozen=True, extra="forbid")`)** schemas, settings, and DSP parameter models (`DnamEngineSettings`, `DnamJsonHeader`, `ControlSchemaDef`, `ControlUiHints`, `PresetDef`, `DnamTelemetrySnapshot`, `ZohRescaledParams`, `ClampedControlValue`, `ClampedControlsResult`, `DnamPreambleInfo`, `PolyAdaaParams`, `SelectiveRgLruParams`, `MufdnParams`, `LobattoParams`, `WatermarkParams`), serving as the canonical Python schema SSOT shared with `packages/trainer`.
     - `container.py` (`DnamContainer`, `GramMatrix`, `crc32c`): First-class immutable/copy-on-write `.dnam`/`.dnam-ckpt` container model (`DnamContainer.from_file`, `from_bytes`, `reference`, `build`, `.header`, `.preamble`, `.controls`, `.presets`, `.arena`, `.with_header`, `.clamp_controls`, `.zoh_params`, `.immutable_header_sha256`, `.structural_hash`, `.with_watermark`, `.save`, `__bytes__`) and `Arena 3` `uplo='U'` Kronecker matrix packer (`GramMatrix.pack`, `GramMatrix.unpack`).
     - `control.py` (`ControlRack`, `ControlSpace`, `LobattoBasis`, `Rank4Fm`, and Control Bus helpers): Live `MutableMapping[str, float | str]` proxy (`engine.controls["gain"] = 7.5`, `engine.controls.snap(gain=7.5)`, `engine.controls.apply_preset("warm_slapback")`, `engine.controls.clamped`), standalone `ControlSpace` 3-Phase clamper, callable `LobattoBasis(LobattoParams)` (with `.from_schema(ctrl)`) and `Rank4Fm(v_factors)` classes, and `chebyshev_lfo(cos_phi)` / `trust_region_softclip(z, *, mh)`.
     - `dsp.py`: Callable stateful Reference DSP processor classes (`PolyAdaa(PolyAdaaParams)`, `SelectiveRgLru(SelectiveRgLruParams)`, `ModulatedUfdn(MufdnParams)` supporting `__call__(x, *, reset=False, out=None)`, `.step(x_n)`, `.reset()`, `.state`, `.params`) plus stateless algebraic helpers (`PolyAdaa.algebraic`, `unit_rotor`, `lagrange_weights`, `givens_householder_mix`) imported by `packages/trainer` as the certification oracle.
     - `provenance.py`: RFC 8785 JCS `immutable_header_sha256`, `Ed25519` creator/marketplace signing messages, RFC 8439 `ChaCha20` 64-byte cache-line arena encryption, Dual-Mechanism Buyer Watermarking (`Test 8H`: 48-ch RG-LRU gauge + 4,608-weight Tardos code), and 160-D structural hash extraction and cosine similarity.
     - `engine.py` (`DnamEngine`): Callable stereo/mono processor with `pedalboard`-compatible duck-typing (`y = engine(x, sample_rate=None, *, buffer_size=None, reset=False, mono_downmix=False, out=None)` and `engine.process(...)`, defaulting mono `(N,)` / `(1, N)` input to true-stereo `(2, N)` output via `process_mono_to_stereo`, plus `process_planar`), explicit `backend="auto" | BackendKind` selection, zero-allocation hot-reloading (`engine.reload(...)`), Pythonic `@property` accessors (`engine.tier`, `engine.sample_rate`, `engine.backend`, `engine.telemetry`, `engine.container`, `engine.controls`), and context-manager support.
- **Modular Test & Code-Quality Suite (`tests/{conftest,strategies,test_schema,test_container,test_control,test_dsp,test_provenance,test_engine}.py`):** Enforces `uv run ruff format --check`, `uv run ruff check`, `uv run pyrefly check` (`preset = "strict"`, zero `# type: ignore`), `uv run pytest` with `hypothesis[numpy,crosshair]` & `hypothesis-jsonschema` property-based/symbolic tests, and `pytest-cov` (`--cov=dnam_engine --cov-fail-under=95`).

### 4.2 Packaged Node.js & WebAudio Engine Library (`packages/nodejs/` — Phase 2B)
- **Dual Native N-API + WebAssembly SIMD128 Architecture & Cross-Compilation:**
  1. **Native N-API Addon (`src/lib.rs`, `build.rs` $\to$ `dist/dnam_nodejs.<platform>-<arch>.node`):** Zero-C-dependency Rust `cdylib` using Node-API v8 to operate **zero-copy** on V8 `Float32Array`/`Uint8Array` buffers with native hardware SIMD (`ARM SME`, `ARM64 NEON`, `x86 AVX2+FMA`), hardware `FTZ`/`DAZ`, and `__crc32cd`. Cross-compiles cleanly via `cargo zigbuild --package dnam-nodejs --target <triple> --release`.
  2. **WebAssembly SIMD128 Module (`dist/dnam_engine.wasm`):** Compiled from `packages/engine` (`wasm32-unknown-unknown` with `-C target-feature=+simd128`) and bundled for synchronous instantiation inside browser `AudioWorkletGlobalScope` and portable JS runtimes.
- **Idiomatic TypeScript / WebAudio Interface (`src/{types,model,processor,worklet,index}.ts`):**
  1. `validateDnamModel(input: Uint8Array | ArrayBuffer | unknown): DnamModel`: Validates binary `.dnam`/`.dnam-ckpt` containers or JSON headers and attaches `ui_hints` power-skew mappers (`normalizedToPhysical`, `physicalToNormalized`).
  2. `clampControlValues(model: DnamModel, rawControls: Record<string, number | string>): ClampedControlsResult`: Applies `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-STABILITY]` (`conditional_max`/`conditional_min`) rules identically to Rust and Python.
  3. `DnamProcessor`: Idiomatic stereo/mono block processor (`[Symbol.dispose]()` + `FinalizationRegistry`, control updates by ID or index, preset loading) supporting `backend: "auto" | "native" | "wasm"`.
  4. `DnamAudioWorkletProcessor`: Ready-to-register WebAudio `AudioWorkletProcessor` class (`128`-sample render quantum = 8 Control-Bus ticks at $48\text{ kHz}$).
- **Node.js / TypeScript Code-Quality Suite:** Enforces `@biomejs/biome` (`format:check`, `lint`), `tsc --noEmit` (`strict: true`), `node --test` + `fast-check` property-based tests (including differential parity between `"native"` N-API and `"wasm"` SIMD128), and $\ge 95\%$ test coverage.

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
