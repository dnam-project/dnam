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
  - **Core 2 (`src/core2/`):** Mode-Free Windowed Time-Warped Modulated Unitary Feedback Delay Network (`M-UFDN` / `WT-UFDN`) with stateless 4-tap Cubic Lagrange fractional delay (`CL-06`), Self-Activating Dual-Tap Complementary Hermite-Hann Grain Reader ($H(\phi_A) + H(\phi_B) \equiv 1$), 4-Tier (`Lite`/`Full`/`Pro`/`Ultra`, $M = 4, 8, 8, 16$) Pre-Head Kronecker Routing, Unitary Spectral Freeze (`set_freeze`, `set_grain_params`), `fp16` delay storage, $O(M)$ Givens + Householder unitary mixing, and Coercive-Saturating Lyapunov loop (`CL-09..11`).
  - **Control Bus (`src/control/`):** $3\text{ kHz}$ Node-Anchored Lobatto HDMR + Rank-4/Rank-8 Factorization Machine + $D_h = 4 / 8$ SVD Head Projector + $8 \times 32$ `W_ctrl_core2` state-space projection + Order-Statistic Transport with Non-Stop 3-Phase Control Clamping (`CL-12..18`), `max_supported_tier` plugin ceiling (`CL-19`), $4\times$ linear-phase ZOH offline bounce oversampling (`CL-20`), and **0 extra MACs/sample at $48\text{ kHz}$ when controls are static**.
- **Multi-Platform Hardware SIMD (`SimdVec4` & `MathBackend`):**
  - [`ArmSmeBackend`](src/math/arm_sme.rs): ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime `sysctlbyname` detection.
  - [`NeonBackend`](src/math/neon.rs): ARM64 128-bit NEON (`vfmaq_laneq_f32` 4-accumulator outer-product tiles, hardware `FCVTL`/`FCVTN` `fp16`, hardware `__crc32cd`).
  - [`X86SimdBackend`](src/math/x86_simd.rs): x86_64 AVX2 + FMA + F16C + SSE4.2 hardware `CRC32C`.
  - [`WasmSimd128Backend`](src/math/wasm_simd128.rs): WebAssembly 128-bit SIMD (`v128`).
  - [`ScalarBackend`](src/math/scalar.rs): Portable 100% safe `#![no_std]` IEEE-754 reference backend.
- **Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Rescaling ($44.1\text{–}96\text{ kHz}$) & $4\times$ Offline Bounce (`CL-20`):**
  - Automatically rescales discrete-time poles, delay lengths, and control-bus decimation strides for any host sample rate in $[44.1\text{ kHz}, 96\text{ kHz}]$ ([`src/container/zoh.rs`](src/container/zoh.rs)) and supports $4\times$ ($176.4\text{–}384\text{ kHz}$) linear-phase polyphase ZOH oversampling (`set_offline_bounce`, `set_maximize_tier_on_bounce`).
- **5-Layer Digital Marketplace Provenance, Prefix-Composable Signatures & Traitor Tracing (`src/provenance/`):**
  - Preset-safe & tier-stripping-safe (`strip_container_to_tier`) RFC 8785 JCS `immutable_header_sha256` with 4-tier `tier_arena_sha256`, 2-certificate `Ed25519` signature verification, `X25519-HKDF-ChaCha20` 64-byte block-aligned arena encryption, zero-distortion Isometric Gauge Watermarking (`0.00 dB` audio error), 7,680-chip (`1,920/tier`) Coupled Tardos Traitor-Tracing codes, and Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hashing.

---

## Real-Time Performance & WCET Telemetry ($48\text{ kHz}$ Stereo, `128`-Sample Blocks)

Measured via `cargo bench -p dnam-engine --bench realtime_wcet`:

| Tier & Subsystem Profile | Backend | Latency / Frame | `×` Real-Time | Single-Core CPU | Modulated `P99.9` Block | `Max` / `2,666.67 µs` Budget |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Lite` (`C=4`, Core 1 Only)** | `ARM64 NEON` | **`321.28 ns`** | **`64.8x`** | **`1.54%`** | — | — |
| **`Lite` (`C=4`, Core 1 + Core 2)** | `ARM64 NEON` | **`391.75 ns`** | **`53.2x`** | **`1.88%`** | `163.58 µs` | **`6.18%`** |
| **`Full` (`C=8`, Core 1 Only)** | `ARM64 NEON` | **`417.40 ns`** | **`49.9x`** | **`2.00%`** | — | — |
| **`Full` (`C=8`, Core 1 + Core 2)** | `ARM64 NEON` | **`598.34 ns`** | **`34.8x`** | **`2.87%`** | `149.46 µs` | **`6.40%`** |
| **`Pro` (`C=12`, Core 1 Only)** | `ARM64 NEON` | **`494.91 ns`** | **`42.1x`** | **`2.38%`** | — | — |
| **`Pro` (`C=12`, Core 1 + Core 2)** | `ARM64 NEON` | **`667.26 ns`** | **`31.2x`** | **`3.20%`** | `158.38 µs` | **`6.71%`** |
| **`Ultra` (`C=16`, Core 1 Only)** | `ARM64 NEON` | **`669.48 ns`** | **`31.1x`** | **`3.21%`** | — | — |
| **`Ultra` (`C=16`, Core 1 + Core 2)** | `ARM64 NEON` | **`1,040.27 ns`** | **`20.0x`** | **`4.99%`** | `242.88 µs` | **`9.39%`** |
| **`Ultra` (`C=16`, 4× Offline Bounce)** | `ARM64 NEON` | **`4,290.58 ns`** | **`4.9x`** | **`20.59%`** | — | — |
| **`Full` (`C=8`, ARM SME + Core 2)** | `ARM SME` | **`631.62 ns`** | **`33.0x`** | **`3.03%`** | — | — |
| **`Full` (`C=8`, Scalar Baseline)** | `Scalar` | **`1,106.88 ns`** | **`18.8x`** | **`5.31%`** | — | — |

---

## Package Structure

```text
packages/engine/
├── src/
│   ├── lib.rs              # Crate root (#![no_std] + alloc) & public API re-exports
│   ├── types.rs            # Tier (Lite/Full/Pro/Ultra), FtzDazGuard, DnamTelemetrySnapshot
│   ├── engine.rs           # Generic DnamEngine<'a, B> (4 I/O topologies, bounce, presets, hot-swap)
│   ├── dynamic.rs          # Runtime-dispatched DynamicDnamEngine<'a>
│   ├── memory.rs           # 64B CacheLine64, AlignedBuffer, ArenaStorage, DelayScratchStorage
│   ├── ffi.rs              # Zero-allocation extern "C" ABI (dnam_engine_*)
│   ├── math/               # SimdVec4, MathBackend, division-free kernels, NEON/SME/AVX2/WASM
│   ├── core1/              # 12-Layer Hybrid Dilated-Conv (PolyADAA) + SelectiveRgLru stack
│   ├── core2/              # Stereo In-Loop Diffusive ModulatedUfdn & Coercive Lyapunov loop
│   ├── control/            # 3 kHz ControlBus, ControlClampDescriptor, Lobatto/Chebyshev/FM
│   ├── container/          # DnamPreamble, DnamContainerView, ZOH rescaling, Arena 4 uplo='U'
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

# Digital marketplace 224-D structural hash, isometric gauge watermark & Tardos tracing
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
