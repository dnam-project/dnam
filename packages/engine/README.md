# `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time SIMD Audio Engine

**`dnam-engine`** is the zero-audio-allocation, 16B/64B SIMD-aligned real-time inference, container, and provenance core for **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")**.

It is engineered for embedded ARMv8.2-A/ARMv9.2-A pedalboards (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins (`packages/plugin`), zero-copy **Python/PyTorch** bindings (`packages/python`), and **WebAssembly SIMD128 (`AudioWorklet`) / Node.js N-API** runtimes (`packages/js`, `packages/web`).

---

## Key Capabilities & Architectural Guarantees

- **Unconditional `#![no_std]` + `alloc` & Zero Real-Time Allocations:**
  - Compiles without `std` across all targets using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`) and `zerocopy` (`0.8`).
  - Verified by a custom `#[global_allocator]` integration test ([`tests/realtime_safety.rs`](tests/realtime_safety.rs)) proving **`0` heap allocations and `0` deallocations** across all audio block topologies, parameter automation, preset switching, tier switching, and in-place model hot-swapping.
- **Zero `FSQRT`, Zero `FDIV`, and Zero `libm` Calls at $48\text{ kHz}$ & $3\text{ kHz}$:**
  - **Core 1 (`src/core1/`):** 12-layer Hybrid Dilated-Conv (`1x B-Spline` division-free algebraic `PolyADAA`, `CL-01..02`) + Radial-Polar Selective Complex RG-LRU (`CL-03..05`) with 4-FMA + 1-Step Newton unit-circle rotor projection ($y_1 = 1.5 - 0.5(\hat{c}^2 + \hat{s}^2)$).
  - **Core 2 (`src/core2/`):** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`) with stateless 4-tap Cubic Lagrange fractional delay (`CL-06`), `fp16` delay storage, $O(M)$ Givens + Householder unitary mixing, and Coercive-Saturating Lyapunov loop (`CL-09..11`).
  - **Control Bus (`src/control/`):** $3\text{ kHz}$ Node-Anchored Lobatto HDMR + Rank-4 Factorization Machine + Order-Statistic Transport with Non-Stop 3-Phase Control Clamping (`CL-12..16`) and **0 extra MACs/sample at $48\text{ kHz}$ when controls are static**.
- **Multi-Platform Hardware SIMD (`SimdVec4` & `MathBackend`):**
  - [`ArmSmeBackend`](src/math/arm_sme.rs): ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime `sysctlbyname` detection.
  - [`NeonBackend`](src/math/neon.rs): ARM64 128-bit NEON (`vfmaq_laneq_f32` 4-accumulator outer-product tiles, hardware `FCVTL`/`FCVTN` `fp16`, hardware `__crc32cd`).
  - [`X86SimdBackend`](src/math/x86_simd.rs): x86_64 AVX2 + FMA + F16C + SSE4.2 hardware `CRC32C`.
  - [`WasmSimd128Backend`](src/math/wasm_simd128.rs): WebAssembly 128-bit SIMD (`v128`).
  - [`ScalarBackend`](src/math/scalar.rs): Portable 100% safe `#![no_std]` IEEE-754 reference backend.
- **Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Rescaling ($44.1\text{--}96\text{ kHz}$):**
  - Automatically rescales discrete-time poles, delay lengths, and control-bus decimation strides for any host sample rate in $[44.1\text{ kHz}, 96\text{ kHz}]$ ([`src/container/zoh.rs`](src/container/zoh.rs)).
- **5-Layer Digital Marketplace Provenance & Traitor Tracing (`src/provenance/`):**
  - Preset-safe RFC 8785 JCS `immutable_header_sha256`, 2-certificate `Ed25519` signature verification, `X25519-HKDF-ChaCha20` 64-byte block-aligned arena encryption, zero-distortion Isometric Gauge Watermarking (`0.00 dB` audio error), 4,608-chip Coupled Tardos Traitor-Tracing codes, and 160-D Gauge-Invariant Structural Hashing.

---

## Real-Time Performance & WCET Telemetry ($48\text{ kHz}$ Stereo, `128`-Sample Blocks)

Measured via `cargo bench -p dnam-engine --bench realtime_wcet`:

| Tier & Subsystem Profile | Backend | Latency / Frame | $\times$ Real-Time | Single-Core CPU |Modulated `P99.9` Block | `Max` / $2,666.67\text{ }\mu\text{s}$ Budget |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Lite` ($C=4$, Core 1 + Core 2)** | `ARM64 NEON` | **`354.76 ns`** | **`58.7x`** | **`1.70%`** | `112.12 µs` | **`5.12%`** |
| **`Full` ($C=8$, Core 1 Only)** | `ARM64 NEON` | **`394.79 ns`** | **`52.8x`** | **`1.90%`** | — | — |
| **`Full` ($C=8$, Core 1 + Core 2)** | `ARM64 NEON` | **`461.76 ns`** | **`45.1x`** | **`2.22%`** | `158.21 µs` | **`5.96%`** |
| **`Full` ($C=8$, ARM SME + Core 2)** | `ARM SME` | **`466.36 ns`** | **`44.7x`** | **`2.24%`** | — | — |
| **`Ultra` ($C=12$, Core 1 + Core 2)** | `ARM64 NEON` | **`555.07 ns`** | **`37.5x`** | **`2.66%`** | `172.75 µs` | **`7.24%`** |
| **`Full` ($C=8$, Scalar Baseline)** | `Scalar` | **`880.86 ns`** | **`23.7x`** | **`4.23%`** | — | — |

---

## Package Structure

```text
packages/engine/
├── src/
│   ├── lib.rs              # Crate root (#![no_std] + alloc) & public API re-exports
│   ├── types.rs            # Tier (Lite/Full/Ultra), FtzDazGuard, DnamTelemetrySnapshot
│   ├── engine.rs           # Generic DnamEngine<'a, B> (4 I/O topologies, presets, hot-swap)
│   ├── dynamic.rs          # Runtime-dispatched DynamicDnamEngine<'a>
│   ├── memory.rs           # 64B CacheLine64, AlignedBuffer, ArenaStorage, DelayScratchStorage
│   ├── ffi.rs              # Zero-allocation extern "C" ABI (dnam_engine_*)
│   ├── math/               # SimdVec4, MathBackend, division-free kernels, NEON/SME/AVX2/WASM
│   ├── core1/              # 12-Layer Hybrid Dilated-Conv (PolyADAA) + SelectiveRgLru stack
│   ├── core2/              # Stereo In-Loop Diffusive ModulatedUfdn & Coercive Lyapunov loop
│   ├── control/            # 3 kHz ControlBus, ControlClampDescriptor, Lobatto/Chebyshev/FM
│   ├── container/          # DnamPreamble, DnamContainerView, ZOH rescaling, Arena 3 uplo='U'
│   ├── provenance/         # CryptoProvider, Ed25519/JCS, ChaCha20 64B, Gauge & Tardos watermarks
│   └── schema/             # #![no_std] serde/serde_json .dnam/.dnam-ckpt header & UI skew
├── tests/                  # Integration tests (backend parity, 0-alloc proof, marketplace, FFI)
├── benches/                # Criterion benchmarks (engine_benchmarks) & WCET audit (realtime_wcet)
└── examples/               # Executable API usage examples
```

---

## Quickstart: Running Examples, Tests & Benchmarks

### 1. Run Examples (`examples/`)

```bash
# Real-time stereo/mono block processing, presets, knob automation & telemetry
cargo run -p dnam-engine --example realtime_audio_block

# Embedded pedalboard zero-copy SRAM delay scratch & zero-allocation model hot-swap
cargo run -p dnam-engine --example embedded_zero_alloc

# Digital marketplace 160-D structural hash, isometric gauge watermark & Tardos tracing
cargo run -p dnam-engine --example marketplace_provenance
```

### 2. Run Unit, Property (`proptest`) & Integration Tests

```bash
# Full test suite (47 co-located unit/property tests + 4 integration suites)
cargo test -p dnam-engine

# Pure #![no_std] + alloc build without optional RustCrypto dependencies
cargo test -p dnam-engine --lib --no-default-features

# Line, region & function coverage via cargo-llvm-cov (>97% coverage)
cargo llvm-cov -p dnam-engine
```

### 3. Run Criterion & Real-Time WCET Benchmarks (`benches/`)

```bash
# Criterion statistical benchmark suite (container, ZOH, tiers, topologies, buffer sizes, subsystems)
cargo bench -p dnam-engine --bench engine_benchmarks

# Real-Time WCET & Tail-Latency Jitter (P50 / P99 / P99.9 / Max vs. 2,666.67 us deadline)
cargo bench -p dnam-engine --bench realtime_wcet
```

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
