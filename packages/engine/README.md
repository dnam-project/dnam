# `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time SIMD Audio Engine

**`dnam-engine`** is the zero-audio-allocation, 16B/64B SIMD-aligned real-time inference, container, and provenance core for **Dynamic Neural Audio Modeler (DNAM)**.

It is engineered for embedded ARMv8.2-A/ARMv9.2-A pedalboards (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins (`packages/plugin`), zero-copy **Python/PyTorch** bindings (`packages/python`), and **WebAssembly SIMD128 (`AudioWorklet`) / Node.js N-API** runtimes (`packages/js`, `packages/web`).

---

## Architecture Overview

- **`#![no_std]` + `alloc` & Zero Real-Time Allocations:**
  - Compiles without `std` across all targets using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`) and `zerocopy` (`0.8`).
  - Verified by the custom `#[global_allocator]` integration test ([`tests/realtime_safety.rs`](tests/realtime_safety.rs)).
- **Zero `FSQRT`, `FDIV`, and `libm` Calls at $48\text{ kHz}$ & $3\text{ kHz}$:**
  - **Operators (`src/operators/`):**
    - **`Op-T` (Transport & Elastic Buffer):** Catmull-Rom cubic fractional delay, branchless wrap, Schroeder diffusers, and dual-tap complementary Hermite-Hann grain reader ($H(\phi_A) + H(\phi_B) \equiv 1$).
    - **`Op-M` (Modal Bank & Kautz):** 4-lane SIMD symplectic resonator, contractive rotor, `H1` Hilbert pair, `T1` sub-octave tracker, and shared-pole Kautz modal bank.
    - **`Op-N` (Commuted Cascade & PolyADAA):** Algebraic `1× B-Spline PolyADAA`, resting DC equilibrium, `C²` wavefolder, and fixed-hertz ZOH bitcrusher.
  - **Stages & Matrix Router (`src/stages/`):**
    - **Stage 0 Router:** 8-Port Matrix Router with return-to-send isolation, `200 Hz` TPT LR4 crossover, parallel sub-sample alignment, WDW routing, and voltage-normalized Mid/Side encoding.
    - **Stage I Pre-FX Bus:** Pre-dynamics and tone shaping (`Op-T` + `Op-M` + `Op-N` with 4-bit topology selector).
    - **Stage II Amp Cascade:** 3-stage commuted nonlinear cascade (`V1A` Preamp $\to$ `V1B` Driver $\to$ Power Amp/Sag) with `[Direct, LPF_80, LPF_400, HPF_1200]` basis.
    - **Stage III Cabinet Modals:** Separate 64B-aligned L/R early FIR (`16/32/48/64` taps/ch) and shared Kautz modal (`4/8/16/32` pairs) slices.
    - **Stage IV Spatial Tank:** Partitioned `fp16` delay/all-pass rings, Givens/Householder mixing, blended feedback knee, input/readout `5 Hz` DC conditioning, and grouped bounded-history feedback guard.
    - **Output Head:** Unified 256-D feature bus and canonical Mid/Side Kronecker output head.
  - **Control Bus (`src/control/`):** $3\text{ kHz}$ node-anchored Lobatto basis, `CP1` tier-local projections (`8-D` Rank-4 in `Lite..Pro`, `16-D` Rank-8 in `Ultra`), exact-integer sufficient statistics, outward-checked trust-region clipping, and `20–400 BPM` tempo synchronization ([`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md)).
- **Multi-Platform Hardware SIMD (`SimdVec4` & `MathBackend`):**
  - [`ArmSmeBackend`](src/math/arm_sme.rs): ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime `sysctlbyname` detection.
  - [`NeonBackend`](src/math/neon.rs): ARM64 128-bit NEON (`vfmaq_laneq_f32` 4-accumulator outer-product tiles, hardware `FCVTL`/`FCVTN` `fp16`, hardware `__crc32cd`).
  - [`X86SimdBackend`](src/math/x86_simd.rs): x86_64 AVX2 + FMA + F16C + SSE4.2 hardware `CRC32C`.
  - [`WasmSimd128Backend`](src/math/wasm_simd128.rs): WebAssembly 128-bit SIMD (`v128`).
  - [`ScalarBackend`](src/math/scalar.rs): Portable 100% safe `#![no_std]` IEEE-754 reference backend.
- **Sample-Rate Conversion & Offline Bounce:**
  - [`src/container/zoh.rs`](src/container/zoh.rs): Continuous-Time ZOH parameter rescaling (`44.1–96 kHz`) and `4×` polyphase offline bounce ([`router_and_stages_topology.md` §9](../../docs/specs/dsp/router_and_stages_topology.md#9-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--4-bounce)).
- **Marketplace Provenance & Watermarking (`src/provenance/`):**
  - RFC 8785 JCS header canonicalization, `Ed25519` prefix commitments (`DAU1`/`DAI1`/`DSA1`), RFC 9180 HPKE recipient slots, and dual-mechanism buyer watermarking ([`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../docs/specs/security/watermarking_and_marketplace.md)).

---

## Floating-Point Scope

Block-processing APIs establish and restore the audio floating-point environment automatically. Direct per-sample engine, operator, stage, or control-tick calls require one [`FtzDazGuard`](src/types.rs) around the entire batch. The low-level `step_frame_tora` does not provide the block APIs' preview accounting, tempo advancement, or offline oversampling. See the [audio-callback policy](../../docs/specs/architecture/tora_architecture.md#floating-point-environment-at-the-audio-callback).

---

## Previously Reported Performance Figures ($48\text{ kHz}$ Stereo, `128`-Sample Blocks)

The following figures were previously reported for `realtime_wcet`, but the
revision, hardware details, and raw results are not recorded here. They are
not evidence that the reconciled design meets its budgets. Measure the actual
build and target using `moon run engine:bench` and the
[hardware budget requirements](../../docs/specs/architecture/tora_architecture.md#22-hardware--dsp-benchmark-summary).

| Tier & Subsystem Profile | Backend | Latency / Frame | `×` Real-Time | Single-Core CPU | Modulated `P99.9` Block | `Max` / `2,666.67 µs` Budget |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Lite` (`64-D` Bus, Stages I–III)** | `ARM64 NEON` | **`321.28 ns`** | **`64.8x`** | **`1.54%`** | — | — |
| **`Lite` (`64-D` Bus, Stages I–IV Full Rig)** | `ARM64 NEON` | **`391.75 ns`** | **`53.2x`** | **`1.88%`** | `163.58 µs` | **`6.18%`** |
| **`Full` (`128-D` Bus, Stages I–III)** | `ARM64 NEON` | **`417.40 ns`** | **`49.9x`** | **`2.00%`** | — | — |
| **`Full` (`128-D` Bus, Stages I–IV Full Rig)** | `ARM64 NEON` | **`598.34 ns`** | **`34.8x`** | **`2.87%`** | `149.46 µs` | **`6.40%`** |
| **`Pro` (`192-D` Bus, Stages I–III)** | `ARM64 NEON` | **`494.91 ns`** | **`42.1x`** | **`2.38%`** | — | — |
| **`Pro` (`192-D` Bus, Stages I–IV Full Rig)** | `ARM64 NEON` | **`667.26 ns`** | **`31.2x`** | **`3.20%`** | `158.38 µs` | **`6.71%`** |
| **`Ultra` (`256-D` Bus, Stages I–III)** | `ARM64 NEON` | **`669.48 ns`** | **`31.1x`** | **`3.21%`** | — | — |
| **`Ultra` (`256-D` Bus, Stages I–IV Full Rig)** | `ARM64 NEON` | **`1,040.27 ns`** | **`20.0x`** | **`4.99%`** | `242.88 µs` | **`9.39%`** |
| **`Ultra` (`256-D` Bus, 4× Offline Bounce)** | `ARM64 NEON` | **`4,290.58 ns`** | **`4.9x`** | **`20.59%`** | — | — |
| **`Full` (`128-D` Bus, ARM SME Full Rig)** | `ARM SME` | **`631.62 ns`** | **`33.0x`** | **`3.03%`** | — | — |
| **`Full` (`128-D` Bus, Scalar Baseline)** | `Scalar` | **`1,106.88 ns`** | **`18.8x`** | **`5.31%`** | — | — |

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
│   ├── operators/          # Op-T (Transport), Op-M (Modal Kautz), Op-N (PolyADAA Cascade)
│   ├── stages/             # Stage 0 Router, Stages I–IV Topology, 256-D Mid/Side Head
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

### Tests, Coverage & Benchmarks

```bash
moon check engine
moon run engine:test
moon run engine:test-portable
moon run engine:coverage
moon run engine:bench
```

API demonstrations live in [`examples/`](examples/). The coverage gate is a
minimum, not a bug-free guarantee; see the [engine test guidance](CONTRIBUTING.md#3-testing-coverage--95--benchmarking-commands).

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
