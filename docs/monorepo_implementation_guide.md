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
├── Cargo.toml                       # Cargo workspace root -> ["packages/engine", "packages/python", "packages/js", "packages/studio"]
├── pnpm-workspace.yaml              # pnpm workspace root -> ["packages/js", "packages/ui", "packages/studio", "packages/web"]
├── package.json                     # JS/TS workspace root -> ["packages/js", "packages/ui", "packages/studio", "packages/web"]
├── biome.json                       # Strict agentic @biomejs/biome configuration
├── .moon/                           # moonrepo polyglot DAG & content-addressed cache configuration
│   └── workspace.yml
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
    ├── js/                          # Layer 2B: Packaged Node.js & WebAudio Library for Rust Engine (@dnam-project/engine)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── CONTRIBUTING.md          # Package contributing & verification guide
    │   ├── README.md                # Package overview & quickstart
    │   ├── Cargo.toml               # Unified Dual-Target Rust crate (dnam-js: N-API v8 host + wasm32 SIMD128)
    │   ├── build.rs                 # Conditional napi-build setup for non-wasm32 targets
    │   ├── package.json             # @dnam-project/engine multi-entry exports (., ./browser, ./node, ./worklet, ./wasm)
    │   ├── tsconfig.json            # Strict TypeScript 6.0 configuration
    │   ├── vite.config.ts           # Vite 8 library mode bundler + Vitest 5 / coverage-v8 config
    │   ├── scripts/
    │   │   └── build_wasm.mjs       # Builds Native N-API (.node via cargo/zigbuild) + WASM SIMD128 (.wasm)
    │   ├── rust/                    # Shared Dual-Target Rust Core + Target Glue
    │   │   ├── lib.rs               # Conditional module router (napi_bindings vs wasm_ffi)
    │   │   ├── engine.rs            # Shared EngineCore (DynamicDnamEngine, backend selection, planar/interleaved)
    │   │   ├── container.rs         # Shared container validation, builder, 4 KB editor, ZOH, clamping & uplo='U'
    │   │   ├── dsp.rs               # Shared Reference DSP stateful processors & algebraic kernels
    │   │   ├── provenance.rs        # Shared JCS SHA-256, Ed25519 messages, ChaCha20, Tardos/Gauge & 160-D hash
    │   │   ├── napi_bindings.rs     # #[cfg(not(target_arch = "wasm32"))] safe napi-rs v3 bindings + AsyncTask
    │   │   └── wasm_ffi.rs          # #[cfg(target_arch = "wasm32")] linear-memory C-ABI exports + arena allocator
    │   ├── src/                     # Idiomatic TypeScript & WebAudio SDK
    │   │   ├── index.ts             # Universal barrel export (prefers Native N-API on Node/Bun, falls back to WASM)
    │   │   ├── browser.ts           # Zero-node-imports browser / bundler entry point
    │   │   ├── node.ts              # Explicit Node.js N-API entry point
    │   │   ├── errors.ts            # Typed error hierarchy (DnamError, DnamValidationError, DnamContainerError, etc.)
    │   │   ├── container.ts         # First-class DnamContainer, GramMatrix (uplo='U') & crc32c
    │   │   ├── control.ts           # ControlRack, ControlSpace, LobattoBasis, Rank4Fm & Chebyshev/softclip helpers
    │   │   ├── dsp.ts               # Stateful PolyAdaa, SelectiveRgLru, ModulatedUfdn & algebraic primitives
    │   │   ├── provenance.ts        # JCS digest, Ed25519 messages, ChaCha20 arena, buyer watermarks & 160-D hash
    │   │   ├── engine.ts            # DnamEngine (process, processPlanar, processMonoToStereo, processAsync, reload)
    │   │   ├── schema/              # Runtime validators, frozen models & power-skew UI mappers
    │   │   ├── drivers/             # Unified DnamDriver contract (wasm.ts & native.ts)
    │   │   └── worklet/             # Lock-free SharedArrayBuffer Seqlock/Ring, DnamAudioWorkletProcessor & DnamWorkletNode
    │   └── test/                    # Modular Vitest + fast-check Unit, Property, Parity & Benchmark Suite (>= 95% cov)
    ├── ui/                          # Layer 2C: Unified Design System & Audio UI Kit (@dnam-project/ui — React 19 + Tailwind v4 + shadcn/ui Base UI + lucide-react)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── CONTRIBUTING.md          # Tailwind v4, shadcn/ui (Base UI / @base-ui/react), lucide-react, DnamAudioHost & zero-node:* rules
    │   ├── README.md                # Component & visualizer catalog for Tauri Studio & Astro/Starlight Web
    │   ├── components.json          # Canonical shadcn/ui config ("style": "base-nova", "ui": "src/components/ui", "utils": "src/lib/utils")
    │   ├── package.json             # Exports: ., ./tokens, ./host, ./ui, ./controls, ./rack, ./workflows, ./visualizers, ./styles.css
    │   ├── tsconfig.json            # Strict TS ("jsx": "react-jsx", ES2024, strict, exactOptionalPropertyTypes)
    │   ├── vite.config.ts           # Vite 8 + @tailwindcss/vite library bundler + Vitest v5 (>= 95% coverage)
    │   ├── src/
    │   │   ├── index.ts             # Curated barrel export
    │   │   ├── styles.css           # Tailwind CSS v4 (`@import "tailwindcss"; @theme { ... }`) + Starlight & shadcn/ui CSS vars
    │   │   ├── tokens.ts            # Obsidian surfaces, Core1/Core2/ControlBus accents, Lite/Full/Ultra, CL-01..20 & CSS var generator
    │   │   ├── lib/
    │   │   │   └── utils.ts         # Canonical shadcn/ui `cn(...inputs)` (`clsx` + `tailwind-merge`) & unit/numeric helpers
    │   │   ├── host/                # Pluggable Audio & Hardware SIMD Acceleration Host (`DnamAudioHost`, `<DnamAudioHostProvider>`, `WebAudioWorkletHost`)
    │   │   └── components/
    │   │       ├── ui/              # Canonical `shadcn/ui` (Base UI `@base-ui/react` + CVA + Tailwind v4) Primitives (`button`, `badge`, `card`, `tabs`, `progress`)
    │   │       ├── controls/        # Hardware Audio Controls (`geometry.ts`, `RotaryKnob.tsx`, `DiscreteSwitch.tsx`, `StatusBadges.tsx`)
    │   │       ├── rack/            # Shared `DnamPluginRack.tsx` & `TriPhysicsTelemetry.tsx`
    │   │       ├── workflows/       # Shared `CapturePlanPanel.tsx` & `TrainingTimelinePanel.tsx`
    │   │       └── visualizers/     # All 7 Interactive Specification & Studio Visualizers (`PolyAdaaVisualizer` .. `ZohRescalerVisualizer`)
    │   └── test/                    # 1:1 Mirrored Vitest + `fast-check` Suite (>= 95% coverage)
    ├── studio/                      # Layer 3A: Unified Rust/Node `dnam` CLI & Tauri v2 Desktop Studio (`dnam-studio` / `@dnam-project/studio`)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── CONTRIBUTING.md          # Dual Rust/TS layout, Tauri v2 IPC, Native Audio Host & Python trainer bridge guide
    │   ├── README.md                # `dnam` CLI & Desktop Studio documentation
    │   ├── Cargo.toml               # `dnam-studio` crate (`[lib] path = "rust/lib.rs"`, `[[bin]] name = "dnam" path = "rust/main.rs"`)
    │   ├── package.json             # `@dnam-project/studio` (`"bin": { "dnam": "./bin/dnam.mjs" }`, React 19 + `@tailwindcss/vite` + `@dnam-project/ui`)
    │   ├── tauri.conf.json          # Tauri v2 app configuration, window spec, CSP policy & `"isolation"` IPC pattern
    │   ├── capabilities/            # Tauri v2 capability permissions (`default.json` for Tier 1 & Tier 2 official plugins)
    │   ├── isolation/               # Sandboxed Tauri v2 IPC Isolation iframe (`index.html` with `__TAURI_ISOLATION_HOOK__`)
    │   ├── rust/                    # Modular Safe Rust 2024 Core (`#![forbid(unsafe_code)]`, zero `mod.rs` files)
    │   │   ├── lib.rs               # Crate root exporting `error`, `project`, `capture`, `runner`, `trainer`, `ipc`, `cli`
    │   │   ├── main.rs              # Dual-mode `dnam` entrypoint (Tauri GUI on `dnam studio` / bundle launch; CLI help in terminal)
    │   │   ├── error.rs             # Typed `StudioError` & `StudioResult<T>` error hierarchy
    │   │   ├── project.rs           # Single Owner of `dnam.toml`, `models/<id>/model.toml` & Capture Planner (`project/{manifest,workspace,planner}.rs`)
    │   │   ├── capture.rs           # WAV I/O, Excitation, Alignment & Reamp Backends (`capture/{wav,excitation,alignment,backends}.rs`)
    │   │   ├── runner.rs            # Native SIMD Container Inspector, 24-Bit WAV Renderer & Kernel Host (`dnam-engine`)
    │   │   ├── trainer.rs           # `uv` Auto-Bootstrap Venv Manager & JSON-Job/JSONL Subprocess Bridge (`trainer/{env_manager,bridge}.rs`)
    │   │   ├── ipc.rs               # Tauri v2 Command & Native Hardware Acceleration Host Handler
    │   │   └── cli.rs               # `clap` v4 Subcommand Router (`cli/{workspace_cmd,capture_cmd,train_cmd,cache_cmd,format}.rs`)
    │   ├── tests/                   # Rust Integration Tests (`project_and_capture.rs`, `cli_and_trainer_bridge.rs`)
    │   ├── src/                     # TypeScript SDK & Tauri v2 React 19 Desktop Studio (`binary.ts`, `ipc.ts`, `host.ts`, `ui/StudioApp.tsx`)
    │   └── test/                    # 1:1 Mirrored Vitest Suite (>= 95% coverage)
    ├── trainer/                     # Layer 3B: PyTorch 4-Stage Differentiable Trainer & Headless `dnam-train` Worker (`dnam_trainer`)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── CONTRIBUTING.md          # Package contributing & float64/hardware guide
    │   ├── README.md                # Python Trainer API & `dnam-train` headless worker documentation
    │   ├── pyproject.toml           # Exposes `dnam-train = "dnam_trainer.cli:main"` and imports `packages/python` (`dnam-engine`)
    │   ├── src/
    │   │   └── dnam_trainer/
    │   │       ├── __init__.py
    │   │       ├── schema.py        # Strict Pydantic v2 training & wire-protocol schemas (`TrainingJobRequest`, `ResumeJobRequest`, `StemAudioRef`)
    │   │       ├── events.py        # UI-agnostic Capture/Training Event & Observer protocols
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
    │   │       ├── training/        # Hardware runtime, Cascade solvers, Kronecker .dnam-ckpt Solver, `wav_io.py` & TrainingBackend (CL-18)
    │   │       │   ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │       │   ├── __init__.py
    │   │       │   ├── accelerator.py
    │   │       │   ├── hardware.py
    │   │       │   ├── numerics.py
    │   │       │   ├── windowing.py
    │   │       │   ├── calib_cache.py
    │   │       │   ├── metrics.py
    │   │       │   ├── selective_cascade.py
    │   │       │   ├── cascade_primitives.py
    │   │       │   ├── cascade_orthogonal_fft.py
    │   │       │   ├── cascade_selective_stages.py
    │   │       │   ├── surrogate_delay.py
    │   │       │   ├── kronecker_head.py
    │   │       │   ├── arena.py
    │   │       │   ├── telemetry.py
    │   │       │   ├── wav_io.py
    │   │       │   ├── pipeline.py
    │   │       │   └── backends.py
    │   │       └── cli/             # Streamlined Headless `dnam-train` Worker CLI (callable from `packages/studio`)
    │   │           ├── AGENTS.md    # Scoped router (< 2,500 B)
    │   │           ├── __init__.py
    │   │           ├── worker.py
    │   │           └── observers.py
    │   └── tests/                   # 1:1 mirrored unit, property-based, guardrail & verification test suite
    │       ├── AGENTS.md            # Scoped router (< 2,500 B)
    │       ├── conftest.py
    │       ├── strategies.py
    │       ├── test_guardrails.py   # AGENTS.md byte budgets, link checker & CL-01..18 verifier
    │       ├── test_schema_and_events.py
    │       ├── dsp/
    │       ├── control/
    │       ├── training/
    │       ├── cli/
    │       │   └── test_train_worker_cli.py
    │       ├── integration/
    │       │   └── test_dnam_verification.py  # All 8 empirical verification suites
    │       └── test_benchmarks.py
    ├── plugin/                      # Layer 3B: Rust Real-Time Audio Plugin (VST3 / CLAP / Standalone)
    │   ├── AGENTS.md                # Scoped router (< 2,500 B)
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs               # Lock-free atomic parameter binding & host ZOH adaptation
    └── web/                         # Layer 3C: Astro + Starlight Docs, React WASM Studio & Edge Marketplace
        ├── AGENTS.md                # Scoped router (< 2,500 B)
        ├── package.json
        ├── wrangler.jsonc           # Cloudflare Workers, R2 (Public CDN + Private Master), Queues & KV bindings
        ├── astro.config.mjs         # @astrojs/cloudflare + Starlight + @astrojs/react + Tailwind v4 + KaTeX + Mermaid
        └── src/
            ├── content.config.ts    # Mounts ../../docs/*.md directly into Specification section
            ├── middleware.ts        # Better Auth + KV session verify & per-route COOP/COEP SharedArrayBuffer headers
            ├── actions/
            │   └── index.ts         # Zod-validated Astro Actions (ratings, comments, profile edits, device auth)
            ├── db/
            │   ├── index.ts         # @tidbcloud/serverless + drizzle-orm/tidb-serverless client
            │   └── schema.ts        # TiDB Cloud Starter MySQL 8.0 + TiFlash VECTOR(160) Drizzle schema
            ├── stores/
            │   └── audio.ts         # nanostores cross-island persistent audition player state
            ├── pages/
            │   ├── playground.astro # Full-screen Interactive React WASM Modeler Studio
            │   ├── explore.astro    # SSR faceted catalog search, category pills & inline audio/WASM preview
            │   ├── models/
            │   │   └── [slug].astro # SSR model detail, live DnamPluginRack, ratings & server:defer comments
            │   ├── u/
            │   │   └── [handle].astro # SSR creator profile, verified badge & published model catalog
            │   └── api/
            │       └── marketplace/ # 3-Stage Ingestion scanner & < 0.2 ms buyer-watermarked R2 download endpoints
            └── components/
                ├── ui/              # shadcn/ui (Base UI / @base-ui/react) + Tailwind CSS v4 + Lucide primitives
                ├── marketplace/     # Faceted search explorer, model cards, comment threads & persistent player bar
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

## 2. Deliverable 1: Web Platform, Specification Section, Interactive React WASM Plugin & Edge Marketplace (`packages/web/`, `docs/`)

- **Zero-Clutter SSOT Specification Mounting:** All canonical Markdown specifications live in `docs/*.md` with zero framework clutter. `packages/web/src/content.config.ts` uses Astro 5's Content Layer (`glob({ pattern: "**/*.md", base: "../../docs" })`) to mount `docs/*.md` directly as the **Specification Reference** section of the Starlight documentation website with full KaTeX (`remark-math` + `rehype-katex`) and Mermaid diagram support.
- **Interactive Specification Visualizers (`packages/web/src/components/visualizers/`):** Each specification section pairs its mathematical derivations with interactive React + HTML5 Canvas visualizers (`client:visible` islands):
  1. **`PolyAdaaVisualizer` ([`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)):** Interactive drive, frequency, and asymmetry sliders comparing Naive cubic waveshaping, unfiltered ADAA1, post-ADAA high-pass extrapolation ($+9.54\text{ dB}$ Nyquist spike), and **$1\times$ Passive B-Spline $(0.875, 0.125)$ Division-Free `PolyADAA`** ($38.87\text{ dB}$ aliasing suppression).
  2. **`RgLruPoleVisualizer` ([`core1_hybrid_ssm_and_polyadaa.md`](core1_hybrid_ssm_and_polyadaa.md)):** Unit-disk ($|z| \le 1$) complex plane comparing **Radial-Polar Split** interpolation along $|z| = r_c \le 0.9995$ against **Cartesian linear interpolation** ($-20.7\text{ dB}$ mid-sweep chord dip), plus the $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor.
  3. **`ModulatedUfdnVisualizer` ([`core2_modulated_ufdn.md`](core2_modulated_ufdn.md)):** Demonstrates `range` delay slewing vs. `switch` **128-sample raised-cosine dual-pointer crossfading** (zero Doppler pitch-zipper), $O(M)$ Givens + Householder unitary mixing $\mathbf{U}(\psi, \mathbf{v})$, and the **Unity-Slope Coercive-Saturating Lyapunov Loop Law** ($S'(0) \equiv 1.0000$).
  4. **`ControlBusVisualizer` ([`parametric_control_bus.md`](parametric_control_bus.md)):** Interactive node count ($U_p = 2..6$) showing **Cardinality-Bounded Node-Anchored Newton/Lobatto basis** ($K_p = \min(U_p-1, 3)$), Centered ANOVA + Rank-4 Factorization Machine surfaces, and **Order-Statistic Optimal-Transport** LFO trajectory matching.
  5. **`KroneckerResumeVisualizer` ([`training_and_iterative_resume.md`](training_and_iterative_resume.md)):** Visualizes the **Envelope-Guided Surrogate Delay Gradient** convex loss basin and interactive `.dnam-ckpt` rank-1 stem addition ($\lbrace \mathbf{G}_{\text{audio}}^{(s)}, \mathbf{C}_{\text{audio}}^{(s)} \rbrace$, $0.00\text{ dB}$ old-stem drift).
  6. **`ClampAuditVisualizer` ([`clamp_and_audit_registry.md`](clamp_and_audit_registry.md)):** Interactive 2D/switch control-space coverage map demonstrating Phase 1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), Phase 2 (`[CLAMP-RAIL]`), and Phase 3 (`[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`), plus a filterable `CL-01`..`CL-18` registry inspector.
  7. **`ZohRescalerVisualizer` ([`dnam_container_schema.md`](dnam_container_schema.md)):** Interactive host sample-rate selector ($44.1\text{–}96\text{ kHz}$) verifying invariant Hz frequency response under Continuous-Time ZOH rescaling.
- **Interactive React WASM Modeler Plugin (`packages/web/src/components/plugin/DnamPluginRack.tsx`):**
  - Mounted via `<DnamPluginRack client:only="react" />` and powered by the WASM SIMD128 build of `packages/engine` and `packages/js` (`@dnam-project/engine`).
  - Dynamically renders rotary knobs (`range`) and discrete selectors (`switch`) from any loaded `.dnam` model's `controls_schema`, enforcing `safe_clamped_range`, `active_options`, and `conditional_max` bounds with real-time `clamp_warning` badges, built-in DI guitar/synth test stems, live WebAudio microphone/instrument input, and $60\text{ fps}$ DSP telemetry.
- **Digital Marketplace, UI Design System & `TiDB Cloud Starter` + Cloudflare R2 Backend ([`digital_marketplace_and_provenance.md` §8](digital_marketplace_and_provenance.md#8-cloud-edge-infrastructure-tidb-cloud-starter-schema--oem-sealing-service)):**
  - **UI & Persistent Auditioning:** Styled with **Tailwind CSS v4** (`@tailwindcss/vite` + `@astrojs/starlight-tailwind`), **`shadcn/ui`** (Base UI / `@base-ui/react` primitives), **`lucide-react`**, and **`nanostores`** (`@nanostores/react`) paired with Astro's `<ClientRouter />` (`transition:persist="dnam-audio-engine"`) so audio/WASM preview playback continues uninterrupted across `/explore`, `/models/[slug]`, and `/u/[handle]` navigations.
  - **Edge Compute, Auth & Database:** Deployed on **Cloudflare Workers** (`@astrojs/cloudflare`) with per-route `COOP`/`COEP` header isolation (`src/middleware.ts`), **`Better Auth`** + Cloudflare KV session caching, **Cloudflare R2** Two-Bucket storage (Public CDN bucket for `.webp`/`.opus`/`Lite` previews at `$0` egress + Private bucket for `Ultra` masters), **Cloudflare Queues** write-behind counter batching, and **`TiDB Cloud Starter`** (`@tidbcloud/serverless` + `drizzle-orm/tidb-serverless` with auto-sharded Rust `TiKV` row storage and `TiFlash` `VECTOR(160)` `VEC_COSINE_DISTANCE` clone detection).

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
   - **`Ultra` tier (`C = 12, M = 8`, $\sim 9,950\text{ MACs/sample}$):** Uses 3 `float32x4_t` NEON registers per vector ($11.2\text{ KB}$ Core 1 state), executed via a **2-pass `8 + 4` register-tiled loop** (`channels 0..7` followed by `channels 8..11`) on 32-register ARM64 NEON to guarantee zero stack spilling while delivering 50% more channels than NAM A2 Full at $1.22\times$ higher speed.
2. **Hardware Denormal Immunity (`FtzDazGuard`):**
   Decaying IIR/RG-LRU poles ($r_c \le 0.9995$) and FDN delay tails ($g_{\text{lin},m} \le 0.9990$) enter the IEEE-754 subnormal range ($< 1.18 \times 10^{-38}$) during silence, which can trigger $100\times$ microcode traps if `FTZ`/`DAZ` is disabled. `FtzDazGuard::new()` sets:
   - **ARM64 (`aarch64`):** Bit 24 (`FZ = 1`) of `FPCR` (`mrs x0, fpcr; orr x0, x0, #(1 << 24); msr fpcr, x0`).
   - **x86_64:** `FTZ` (`bit 15`) and `DAZ` (`bit 6`) in `MXCSR`.
   - **Software Fallback (WASM / portable):** Adds $+10^{-30}\text{ f32}$ DC bias injection or branchless flush before state write-back.
3. **Zero Audio-Thread Allocations:**
   All delay ring buffers (`fp16` storage converted via single-cycle `FCVTL`/`FCVTN` on ARM64 NEON, or pre-allocated `f32` ring buffers with power-of-two bitmask wrap `(write_idx - offset) & mask`), all-pass buffers, and control caches are pre-allocated during `DnamEngine::new(config, host_sample_rate)`. `process_block(&mut self, left_in, right_in, left_out, right_out)` performs **zero heap allocations, zero locks, and zero system calls**.

---

## 4. Deliverable 3: Packaged Python & Node.js / WebAudio Engine Libraries (`packages/python/`, `packages/js/`)

Both Layer 2 packages wrap a **thin, zero-copy Rust FFI binding core** inside an **idiomatic, strictly-typed target-language interface**, support cross-compilation via **`zig` (`maturin --zig` / `cargo-zigbuild`)**, and enforce a full target-language code-quality suite (formatter, linter, strict type checker, unit + property-based tests, and ≥ 95% test coverage in the target language without duplicating Rust-side unit/coverage tests on the FFI glue crates).

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

### 4.2 Packaged Node.js & WebAudio Engine Library (`packages/js/` — Phase 2B)
- **Unified Dual-Target Rust Core (`rust/{lib,engine,container,dsp,provenance,napi_bindings,wasm_ffi}.rs`):**
  1. **Native N-API v8 Addon (`dist/dnam_js.<platform>-<arch>.node`):** Safe Rust (`#[deny(unsafe_code)]`) `cdylib` using `napi-rs` v3 (`Float32Array`, `Float64Array`, `Uint8Array` zero-copy slices + libuv `AsyncTask` background rendering) with native hardware SIMD (`ARM SME`, `ARM64 NEON`, `x86 AVX2+FMA`), hardware `FTZ`/`DAZ`, and `__crc32cd`. Cross-compiles cleanly via `cargo zigbuild --package dnam-js --target <triple> --release`.
  2. **WebAssembly SIMD128 Module (`dist/dnam_engine.wasm`):** Compiled from the exact same shared Rust core in `packages/js/rust/` (`wasm32-unknown-unknown` with `-C target-feature=+simd128`) for synchronous instantiation inside browser `AudioWorkletGlobalScope` and portable JS runtimes.
- **Idiomatic TypeScript & WebAudio SDK (`src/` $\to$ `@dnam-project/engine`):**
  1. **Multi-Entry Exports:** `@dnam-project/engine` (universal auto-detect), `@dnam-project/engine/browser` (pure WASM, zero `node:*` imports), `@dnam-project/engine/node` (explicit native N-API), `@dnam-project/engine/worklet` (`AudioWorkletGlobalScope`), and `@dnam-project/engine/wasm`.
  2. **Symmetric Domain Model (`DnamContainer`, `GramMatrix`, `ControlSpace`, `ControlRack`, `LobattoBasis`, `Rank4Fm`, `PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `DnamEngine`):** Mirrors `packages/python` with `Float32Array`/`Float64Array` zero-copy semantics, `processAsync` libuv worker pool rendering, `[Symbol.dispose]()` + `FinalizationRegistry` lifecycle safety, and 5-Layer provenance/watermarking APIs.
  3. **Dual-Mode WebAudio `AudioWorklet` Architecture (`DnamAudioWorkletProcessor`, `DnamWorkletNode`):** Zero-allocation $128\text{-sample}$ render quantum (`8` Control-Bus ticks at $48\text{ kHz}$) with automatic `SharedArrayBuffer` lock-free atomic parameter slots (`Atomics.store`/`load`) and `Seqlock` $60\text{ fps}$ telemetry (`SharedTelemetryView`), plus automatic `MessagePort` fallback when `crossOriginIsolated === false`.
- **Node.js / TypeScript Code-Quality Suite:** Enforces `@biomejs/biome` (`pnpm format:check`, `pnpm lint`), `tsc --noEmit` (`strict: true`), `vitest` + `fast-check` property-based tests (including differential parity between `"native"` N-API and `"wasm"` SIMD128), `vitest bench --run` statistical benchmarks, and `@vitest/coverage-v8` (≥ 95% test coverage).

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
