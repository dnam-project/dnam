# `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time SIMD Audio Engine

**`dnam-engine`** is the zero-audio-allocation, 16B/64B SIMD-aligned real-time inference, container, and provenance core for **Dynamic Neural Audio Modeler (DNAM)**.

It is engineered for embedded ARMv8.2-A/ARMv9.2-A pedalboards (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins (`packages/plugin`), zero-copy **Python/PyTorch** bindings (`packages/python`), and **WebAssembly SIMD128 (`AudioWorklet`) / Node.js N-API** runtimes (`packages/js`, `packages/web`).

---

## Architecture Overview

- **`#![no_std]` + `alloc` & Zero Real-Time Allocations:**
  - Compiles without `std` across all targets using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`) and `zerocopy` (`0.8`).
  - Verified by the custom `#[global_allocator]` integration test ([`tests/realtime_safety.rs`](tests/realtime_safety.rs)) and [`spec_scope.toml`](spec_scope.toml) (`207` engine-scoped `r[...]` rules verified via `verify[...]`).
- **Zero `FSQRT`, `FDIV`, and `libm` Calls at $48\text{ kHz}$ & $3\text{ kHz}$:**
  - **Operators (`src/operators/`):**
    - **`Op-T` (Transport & Elastic Buffer):** Catmull-Rom cubic fractional delay with ZOH-rescaled dual-mode slew/crossfade ($L_{\text{xfade}} = \max(16, \mathrm{round}(128/\rho_{\text{sr}}))$), branchless wrap, partitioned Schroeder diffusers, and dual-tap complementary Hermite-Hann grain reader ($H(\phi_A) + H(\phi_B) \equiv 1$).
    - **`Op-M` (Modal Bank & Kautz):** 4-lane SIMD symplectic resonator, quadrant-reduced degree-13/14 Horner contractive rotor with adaptive 1–2 ULP guard, `Q0.192` LFO phase accumulator, 12-allpass (`6` I + `6` Q) `H1` Hilbert SSB lattice, `5 Hz` `T1` DC-blocked sub-harmonic tracker, and Stage III 16-sample delayed Kautz modal bank.
    - **`Op-N` (Commuted Cascade & PolyADAA):** 6-step rail-crossing exact cubic spline `MultiDrivePolyAdaaChannel` (`1× B-Spline PolyADAA` + `5 Hz` AC coupling), analytical DC resting equilibrium, `C²` wavefolder, and `Q64.32` zero-order-hold bitcrusher with TPDF dither.
  - **Stages & Matrix Router (`src/stages/`):**
    - **Stage 0 Router:** 8-Port Matrix Router with return-to-send isolation, `200 Hz` TPT LR4 crossover, Kaiser-sinc sub-sample alignment, WDW routing, and voltage-normalized Mid/Side encoding.
    - **Stage I Pre-FX Bus:** Pre-dynamics and tone shaping (`Op-T` + `Op-M` + `Op-N` with 4-bit topology selector).
    - **Stage II Amp Cascade:** 3-stage commuted nonlinear cascade (`V1A` Preamp $\to$ `V1B` Driver $\to$ Power Amp/Sag) with `[Direct, LPF_80, LPF_400, HPF_1200]` basis.
    - **Stage III Cabinet Modals:** Separate 64B-aligned L/R early FIR (`16/32/48/64` taps/ch) and shared Kautz modal (`4/8/16/32` pairs) slices with 16-sample delayed modal onset.
    - **Stage IV Spatial Tank:** Partitioned `fp16` delay/all-pass rings, Givens/Householder mixing, 6-step Newton reciprocal blended feedback knee, outer-clamped loop saturator, input/readout `5 Hz` DC conditioning (bypassing loop damping during captured hardware freeze while maintaining DC blocking on wet readout), and grouped bounded-history feedback guard.
    - **Output Head:** Unified 256-D feature bus (`CANONICAL-256D-FEATURE-BUS-MAP`) and Mid/Side Kronecker output head.
  - **Control Bus & Runtime UX (`src/control/`, [`ADR-0011`](../../docs/adr/0011-end-to-end-stage-wiring-and-arena-slice-topology.md)):**
    - $3\text{ kHz}$ node-anchored Lobatto basis, `CP1` tier-local projections (`8-D` Rank-4 in `Lite..Pro`, `16-D` Rank-8 in `Ultra`), 9-word/18-word `ExactIntFactorAccumulator`, outward-checked trust-region clipping, and `Q64`/`Q32` `20–400 BPM` tempo synchronization with 32-step binary-search physical calibration inversion ([`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md)).
    - Atomic preset batch evaluation with two-path output crossfading on disjoint conditional switch transitions (`apply_preset`), `SpilloverTrails` vs. `HardBypass` ramping (`set_bypass`), `supports_freeze` hardware freeze gating, out-of-range tempo-sync rank clamping (`active_clamp_mask`), and `OUTSIDE_CERTIFIED_INPUT` (`bit 4`) telemetry reporting above $\pm 100.0$ (`+40 dBFS`) without unmodeled soft-limiting.
- **Multi-Platform Hardware SIMD (`SimdVec4` & `MathBackend`):**
  - [`ArmSmeBackend`](src/math/arm_sme.rs): ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime `sysctlbyname` detection.
  - [`NeonBackend`](src/math/neon.rs): ARM64 128-bit NEON (`vfmaq_laneq_f32` 4-accumulator outer-product tiles, hardware `FCVTL`/`FCVTN` `fp16`, hardware `__crc32cd`).
  - [`X86SimdBackend`](src/math/x86_simd.rs): x86_64 AVX2 + FMA + F16C + SSE4.2 hardware `CRC32C`.
  - [`WasmSimd128Backend`](src/math/wasm_simd128.rs): WebAssembly 128-bit SIMD (`v128`).
  - [`ScalarBackend`](src/math/scalar.rs): Portable 100% safe `#![no_std]` IEEE-754 reference backend.
- **Sample-Rate Conversion & Offline Bounce:**
  - [`src/container/zoh.rs`](src/container/zoh.rs): Continuous-Time ZOH parameter rescaling (`44.1–192 kHz`), Stage III first-tap/6-point Lagrange FIR resampling, and `4×` polyphase offline bounce ([`router_and_stages_topology.md` §9](../../docs/specs/dsp/router_and_stages_topology.md#9-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--4-bounce)).
- **Marketplace Provenance & Watermarking (`src/provenance/`):**
  - RFC 8785 JCS header canonicalization, `Ed25519` prefix commitments (`DAU1`/`DAI1`/`DSA1`), `OemDeviceCertificate` V2, RFC 9180 HPKE recipient slots (`DNSL`/`DNPL`/`DME1`), `HMAC-SHA256` watermark seed derivation, Stage III gauge-invariant modal fingerprinting, dual-mechanism buyer watermarking, and `.dnam-prev` (`DNPR`) preview container rejection ([`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../docs/specs/security/watermarking_and_marketplace.md)).

---

## Floating-Point Scope

Block-processing APIs establish and restore the audio floating-point environment automatically. Direct per-sample engine, operator, stage, or control-tick calls require one [`FtzDazGuard`](src/types.rs) around the entire batch. The low-level `step_frame_tora` steps a single stereo frame and does not perform block-level tempo advancement or `4×` offline bounce oversampling. See the [audio-callback policy](../../docs/specs/architecture/tora_architecture.md#floating-point-environment-at-the-audio-callback).

---

## Measured Real-Time & WCET Benchmark Summary ($48\text{ kHz}$ Stereo, `128`-Sample Blocks)

Measured via `cargo bench -p dnam-engine --bench realtime_wcet` (`2,666.67 µs` per `128`-sample block budget at $48\text{ kHz}$, `33.34 µs` `#![no_std]` container load + `CRC32C` verification; see the [hardware budget requirements](../../docs/specs/architecture/tora_architecture.md#22-hardware--dsp-benchmark-summary)):

| Tier & Subsystem Profile | Backend | Latency / Frame | `×` Real-Time | Single-Core CPU | Modulated `P50` Block | Modulated `P99.9` Block | `Max` / `2,666.67 µs` Budget |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Lite` (`C = 4`, `64-D` Bus)** | `ARM64 NEON` | **`161.12 ns`** | **`129.3x`** | **`0.77%`** | `27.54 µs` | `73.88 µs` | **`3.12%`** |
| **`Full` (`C = 8`, `128-D` Bus)** | `ARM64 NEON` | **`176.39 ns`** | **`118.1x`** | **`0.85%`** | `34.67 µs` | `83.29 µs` | **`3.33%`** |
| **`Pro` (`C = 12`, `192-D` Bus)** | `ARM64 NEON` | **`180.56 ns`** | **`115.4x`** | **`0.87%`** | `41.33 µs` | `100.25 µs` | **`3.99%`** |
| **`Ultra` (`C = 16`, `256-D` Bus)** | `ARM64 NEON` | **`199.28 ns`** | **`104.5x`** | **`0.96%`** | `64.12 µs` | `114.17 µs` | **`5.31%`** |
| **`Ultra` (`C = 16`, `4×` Offline Bounce)** | `ARM64 NEON` | **`780.00 ns`** | **`26.7x`** | **`3.74%`** | — | — | — |
| **`Full` (`C = 8`, ARM SME)** | `ARM SME` | **`175.52 ns`** | **`118.7x`** | **`0.84%`** | — | — | — |
| **`Full` (`C = 8`, Scalar Baseline)** | `Scalar` | **`192.12 ns`** | **`108.4x`** | **`0.92%`** | — | — | — |

---

## Package Structure

```text
packages/engine/
├── spec_scope.toml         # Manifest mapping all 289 r[...] spec rules (207 engine-scoped)
├── src/
│   ├── lib.rs              # Crate root (#![no_std] + alloc) & public API re-exports
│   ├── types.rs            # Tier (Lite/Full/Pro/Ultra), FtzDazGuard, DnamTelemetrySnapshot
│   ├── engine.rs           # Generic DnamEngine<'a, B> (4 I/O topologies, bounce, presets, hot-swap)
│   ├── engine/             # Block processing (process.rs) & runtime transitions (runtime.rs)
│   ├── dynamic.rs          # Runtime-dispatched DynamicDnamEngine<'a>
│   ├── dynamic/            # Backend dispatch & parity tests (dispatch.rs)
│   ├── memory.rs           # 64B CacheLine64, AlignedBuffer, ArenaStorage, DelayScratchStorage
│   ├── ffi.rs              # Zero-allocation extern "C" ABI (dnam_engine_*)
│   ├── test_support.rs     # #[cfg(test)] exact rational (Rat), SplitMix64 & ULP test harness
│   ├── math.rs & math/     # SimdVec4, MathBackend, spec_math.rs, NEON/SME/AVX2/WASM
│   ├── operators.rs & operators/ # Op-T (op_t.rs), Op-M (op_m.rs), Op-N (op_n.rs), reference.rs
│   ├── stages.rs & stages/ # Stage 0–IV topology.rs, stage1..stage4, head.rs, weights.rs
│   ├── control.rs & control/ # 3 kHz bus.rs, plane.rs (ExactIntFactorAccumulator, Q64/Q32 tempo)
│   ├── container.rs & container/ # wire.rs, preamble.rs, view.rs, zoh.rs, ckpt.rs, strip.rs
│   ├── provenance.rs & provenance/ # security.rs (DAU1/HPKE/DME1), seal.rs, watermark.rs
│   └── schema.rs & schema/ # #![no_std] serde/serde_json .dnam/.dnam-ckpt header & UI skew
├── tests/                  # Cross-cutting integration tests (backend parity, 0-alloc, FFI, seal)
├── benches/                # Subsystem Criterion suites (pipeline, operators, stages_and_control,
│                           # container_and_provenance) & WCET jitter audit (realtime_wcet)
└── examples/               # 6 runnable API examples (realtime_audio_block, embedded_zero_alloc,
                            # marketplace_provenance, container_slimming_and_ckpt,
                            # custom_operator_chain, c_abi_host_integration)
```

---

## Quickstart: Running Examples, Tests & Benchmarks

### Tests, Coverage & Benchmarks

```bash
# Quality gates & unit/integration/portable tests
moon check engine
moon run engine:test
moon run engine:test-portable
moon run engine:coverage

# Benchmarks (all suites or individual targets)
moon run engine:bench
cargo bench -p dnam-engine --bench realtime_wcet
cargo bench -p dnam-engine --bench pipeline
cargo bench -p dnam-engine --bench operators
cargo bench -p dnam-engine --bench stages_and_control
cargo bench -p dnam-engine --bench container_and_provenance

# Profiling with hotpath-rs (--profile profiling)
HOTPATH_LIMIT=0 HOTPATH_CRITERION_ITERATIONS=500 \
  cargo bench -p dnam-engine --profile profiling --bench pipeline --features hotpath-alloc \
  -- --test --exact "process_block_128/ARM64 NEON/ultra"
cargo bench -p dnam-engine --profile profiling --bench realtime_wcet --features hotpath-alloc

# Runnable Examples
cargo run -p dnam-engine --example realtime_audio_block
cargo run -p dnam-engine --example embedded_zero_alloc
cargo run -p dnam-engine --example marketplace_provenance
cargo run -p dnam-engine --example container_slimming_and_ckpt
cargo run -p dnam-engine --example custom_operator_chain
cargo run -p dnam-engine --example c_abi_host_integration
```

API demonstrations live in [`examples/`](examples/). The coverage gate is a
minimum, not a bug-free guarantee; see the [engine test guidance](CONTRIBUTING.md#3-testing-coverage--95--benchmarking-commands).

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
