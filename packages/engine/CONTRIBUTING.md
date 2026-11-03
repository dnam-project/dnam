# Contributing to `dnam-engine` (`packages/engine/`)

This guide covers engineering, safety, testing, coverage, and benchmarking standards for **`dnam-engine`** ([`README.md`](README.md)), the `#![no_std]` + `alloc` Rust SIMD real-time audio engine at the heart of **DNAM**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md) and the submodule `AGENTS.md` files for progressive-disclosure routing:

- **`src/engine/` ([`src/engine/AGENTS.md`](src/engine/AGENTS.md)):** Top-level generic engine dispatcher, dual-rate ($48\text{ kHz}$ / $3\text{ kHz}$) lifecycle, and 4-tier slimmable SIMD hierarchy.
- **`src/operators/` ([`src/operators/AGENTS.md`](src/operators/AGENTS.md)):** Op-T (Transport/Elastic Buffer), Op-M (Modal Bank/Kautz), and Op-N (Commuted Cascade/PolyADAA).
- **`src/stages/` ([`src/stages/AGENTS.md`](src/stages/AGENTS.md)):** Stage 0 8-Port Matrix Router and Stages I–IV canonical rig topology.
- **`src/control/` ([`src/control/AGENTS.md`](src/control/AGENTS.md)):** $3\text{ kHz}$ envelopes, 4-FMA + Newton LFO rotors, Node-Anchored Lobatto HDMR, Rank-4 FM, and 3-Phase control clamping.
- **`src/container/` ([`src/container/AGENTS.md`](src/container/AGENTS.md)):** 64-byte `.dnam` / `.dnam-ckpt` preamble, zero-copy `DnamContainerView`, Continuous-Time ZOH sample-rate rescaler ($44.1\text{–}96\text{ kHz}$), and `uplo='U'` Gram matrix packer.
- **`src/provenance/` ([`src/provenance/AGENTS.md`](src/provenance/AGENTS.md)):** Signature, encryption, watermark, and fingerprint prototypes; see the canonical security specification for the unfinished authentication contracts.
- **`src/schema/` ([`src/schema/AGENTS.md`](src/schema/AGENTS.md)):** `#![no_std]` `serde` JSON header types, UI skew curves, and `DnamError`.

### Test, Benchmark & Example Placement

- **Unit & Property Tests (`#[cfg(test)] mod tests`):** Co-locate unit tests and `proptest!` suites inside the submodule they test (e.g., `src/operators/reference.rs`, `src/math/backend.rs`).
- **Integration Tests (`tests/`):** Keep `tests/` strictly for cross-cutting crate integration tests:
  - [`tests/backend_parity.rs`](tests/backend_parity.rs): Backend and topology parity checks for the exercised configurations.
  - [`tests/realtime_safety.rs`](tests/realtime_safety.rs): `TrackingAllocator` checks for the exercised real-time paths; these do not prove every path allocation-free.
  - [`tests/tora_operators_verification.rs`](tests/tora_operators_verification.rs): Empirical operator verification (`Tests 1A–1C, 2A–2C, 3`) for Op-N (`PolyADAA`), Op-M (Kautz/Symplectic), and Op-T (Catmull-Rom).
  - [`tests/tora_stages_and_tank_verification.rs`](tests/tora_stages_and_tank_verification.rs): Empirical verification (`Tests 4, 5`) for Stage 0 Router, Stage IV Spatial Tank Lyapunov bounds, and 256-D Mid/Side Kronecker Head (`CL-18`).
  - [`tests/tora_control_and_ckpt_verification.rs`](tests/tora_control_and_ckpt_verification.rs): Empirical verification (`Tests 8A, 8B, 8D, 8E, 8G`) for 3-Phase control clamping, `LatentAccumulator64B`, and `.dnam-ckpt` `uplo='U'` prefix packing.
  - [`tests/tora_pipeline_and_zoh_verification.rs`](tests/tora_pipeline_and_zoh_verification.rs): End-to-end TORA pipeline (`Tests 8H, 8I`), ZOH multi-rate rescaling, `4x` offline bounce (`CL-20`), and calibration roles.
  - [`tests/marketplace_provenance_and_watermark.rs`](tests/marketplace_provenance_and_watermark.rs) & [`tests/marketplace_seal_and_policy.rs`](tests/marketplace_seal_and_policy.rs): End-to-end cryptographic provenance, `.dnam-seal` hardware enclave envelopes, and Dual-Mechanism buyer watermarking.
  - [`tests/ffi_integration.rs`](tests/ffi_integration.rs): C-ABI lifecycle and null-pointer safety.
- **Benchmarks (`benches/`):** Place throughput and WCET latency benchmarks in [`benches/engine_benchmarks.rs`](benches/engine_benchmarks.rs) and [`benches/realtime_wcet.rs`](benches/realtime_wcet.rs).
- **Examples (`examples/`):** Place runnable API demonstrations in [`examples/realtime_audio_block.rs`](examples/realtime_audio_block.rs), [`examples/embedded_zero_alloc.rs`](examples/embedded_zero_alloc.rs), and [`examples/marketplace_provenance.rs`](examples/marketplace_provenance.rs).

---

## 2. Safety, `#![forbid(unsafe_code)]` & Real-Time Audio Invariants

1. **Strict `unsafe` Isolation:**
   - `src/lib.rs` and all algorithmic submodules (`operators`, `stages`, `control`, `container`, `provenance`, `schema`, `engine`, `dynamic`) enforce `#[forbid(unsafe_code)]`.
   - Use disjoint struct field borrows (`step_frame_tora`) rather than raw pointer casts in `engine/`.
   - `unsafe` is permitted **only** in hardware SIMD intrinsic modules (`src/math/neon.rs`, `src/math/x86_simd.rs`, `src/math/wasm_simd128.rs`, `src/math/arm_sme.rs`), hardware `FTZ`/`DAZ` CSR guards (`src/types.rs`), 64-byte aligned allocation (`src/memory.rs`), and C-ABI pointer entrypoints (`src/ffi.rs`). Every `unsafe` block must carry an explicit `// SAFETY:` justification.
2. **Zero Audio-Thread Heap Allocations:**
   - `process_block`, `process_block_in_place`, `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_sample_rate`, `set_control_value`, `reset_to_dc_equilibrium`, `telemetry_snapshot`, `from_bytes_with_scratch`, and `reload_model_in_place` must never allocate or deallocate heap memory.
3. **Zero `FSQRT`, `FDIV`, or `libm` in $48\text{ kHz}$ / $3\text{ kHz}$ Paths:**
   - Never use division, square-root, or transcendental functions in real-time audio or control-bus loops; use the division-free algebraic formulations (`CL-01`–`CL-20`).
4. **Submodule Implementation Invariants:**
   - **`src/math` & `src/operators`:** Use `zerocopy::transmute!` for `[f32; 4]` $\leftrightarrow$ SIMD register conversions and `B: MathBackend` (`B::Vec4`) for SIMD vectorization.
   - **`src/stages`:** Account for all per-instance state and verify tier-local allocation and alignment against the [state audit](../../docs/overview/tora_architecture.md#23-state-ram-audit-lite-budget). Existing arithmetic constants are not measured budgets.
   - **`src/control`:** Follow the [node-anchored control basis](../../docs/dsp/parametric_control_bus.md#23-cardinality-bounded-node-anchored-newtonlobatto-basis-cl-13). A normalized default is zero only when the physical default is the declared midpoint.
   - **`src/container` & `src/schema`:** Validate checked bounds, alignment, tier-prefix integrity, and authenticated shared data against the [container format](../../docs/containers/dnam_binary_format.md); do not assume fixed tier lengths or treat `CRC32C` as authentication. Keep `serde` / `serde_json` configured with `default-features = false, features = ["alloc"]`.
   - **`src/provenance`:** Apply the exact versioned [immutable-header projection](../../docs/security/provenance_and_enclave_drm.md#21-preset-safe-canonical-header-digest-rfc-8785-jcs); do not recursively exclude same-named fields. Mutable edits are constrained by the authenticated region and its capacity.

---

## 3. Testing, Coverage (≥ 95%) & Benchmarking Commands

Run all commands from the monorepo root via `moon`:

```bash
# 1. Run all engine quality gates (format-check, lint, typecheck, build, test)
moon check engine

# 2. Run individual engine tasks
moon run engine:format-check
moon run engine:lint
moon run engine:test

# 3. Check the >= 95% instrumented-line coverage floor
moon run engine:coverage

# 4. Run real-time throughput benchmarks
moon run engine:bench
```

Coverage is a floor, not a bug-free guarantee. Independently challenge the
specification and implementation with boundary, differential, long-run,
backend, integration, and target-hardware tests; inspect untested branches
and platform-specific paths separately.

---

## 4. Pre-Commit Checklist for `packages/engine/`

- [ ] `moon check engine` passes all formatting, Clippy (`-D warnings`), build, and unit/`proptest`/integration tests (`tests/realtime_safety.rs` and `tests/backend_parity.rs`).
- [ ] `moon run engine:coverage` confirms ≥ 95% coverage, including all newly added or modified functions and tier branches.
- [ ] If any public API or FFI signature changed, `packages/python/rust/` bindings and scoped `AGENTS.md` files (`< 2,500 B`) are updated accordingly.
