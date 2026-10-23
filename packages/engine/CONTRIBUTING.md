# Contributing to `dnam-engine` (`packages/engine/`)

This guide covers engineering, safety, testing, coverage, and benchmarking standards for **`dnam-engine`** ([`README.md`](README.md)), the `#![no_std]` + `alloc` Rust SIMD real-time audio engine at the heart of **DNAM ("Tri-Physics")**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md) and the submodule `AGENTS.md` files for progressive-disclosure routing:

- **`src/lib.rs`, `src/types.rs`, `src/engine.rs`, `src/dynamic.rs`, `src/memory.rs`, `src/ffi.rs`:** Top-level generic `DnamEngine<'a, B>`, runtime SIMD dispatcher (`DynamicDnamEngine<'a>`), 64-byte aligned arena/delay allocators, `FtzDazGuard`, and C-ABI (`extern "C"`).
- **`src/math/` ([`src/math/AGENTS.md`](src/math/AGENTS.md)):** Hardware SIMD backends (`arm_neon`, `arm_sme`, `x86_simd`, `wasm_simd128`, `scalar`), `f64` reference DSP kernels, `fp16` (`IEEE 754 binary16`), cold-path init math, and hardware `CRC32C`.
- **`src/core1/` ([`src/core1/AGENTS.md`](src/core1/AGENTS.md)):** 12-layer `1x B-Spline PolyADAA` + Radial-Polar Selective Complex RG-LRU stack and 3 kHz Rank-4 SVD output head collapse.
- **`src/core2/` ([`src/core2/AGENTS.md`](src/core2/AGENTS.md)):** Modulated Unitary Feedback Delay Network (`M-UFDN`), Givens/Householder reflectors, cubic Lagrange delay interpolation, and coercive-saturating Lyapunov loop.
- **`src/control/` ([`src/control/AGENTS.md`](src/control/AGENTS.md)):** $3\text{ kHz}$ envelopes, 4-FMA + Newton LFO rotors, Node-Anchored Lobatto HDMR, Rank-4 FM, and 3-Phase control clamping.
- **`src/container/` ([`src/container/AGENTS.md`](src/container/AGENTS.md)):** 64-byte `.dnam` / `.dnam-ckpt` preamble, zero-copy `DnamContainerView`, Continuous-Time ZOH sample-rate rescaler ($44.1\text{--}96\text{ kHz}$), and `uplo='U'` Gram matrix packer.
- **`src/provenance/` ([`src/provenance/AGENTS.md`](src/provenance/AGENTS.md)):** RFC 8785 JCS SHA-256, `Ed25519` creator/marketplace verification, 64B `ChaCha20` arena encryption, and Dual-Mechanism Buyer Watermarking (`Test 8H`).
- **`src/schema/` ([`src/schema/AGENTS.md`](src/schema/AGENTS.md)):** `#![no_std]` `serde` JSON header types, UI skew curves, and `DnamError`.

### Test, Benchmark & Example Placement

- **Unit & Property Tests (`#[cfg(test)] mod tests`):** Co-locate unit tests and `proptest!` suites inside the submodule they test (e.g., `src/core1/reference.rs`, `src/math/backend.rs`).
- **Integration Tests (`tests/`):** Keep `tests/` strictly for cross-cutting crate integration tests:
  - [`tests/backend_parity.rs`](tests/backend_parity.rs): Bit-accurate parity across all SIMD/scalar backends and all 4 I/O topologies across `Lite`, `Full`, and `Ultra`.
  - [`tests/realtime_safety.rs`](tests/realtime_safety.rs): Global `TrackingAllocator` proof that all real-time paths execute with `0` heap allocations.
  - [`tests/marketplace_integration.rs`](tests/marketplace_integration.rs): End-to-end cryptographic provenance, enclave decryption, and buyer watermarking.
  - [`tests/ffi_integration.rs`](tests/ffi_integration.rs): C-ABI lifecycle and null-pointer safety.
- **Benchmarks (`benches/`):** Place throughput and WCET latency benchmarks in [`benches/engine_benchmarks.rs`](benches/engine_benchmarks.rs) and [`benches/realtime_wcet.rs`](benches/realtime_wcet.rs).
- **Examples (`examples/`):** Place runnable API demonstrations in [`examples/realtime_audio_block.rs`](examples/realtime_audio_block.rs), [`examples/embedded_zero_alloc.rs`](examples/embedded_zero_alloc.rs), and [`examples/marketplace_provenance.rs`](examples/marketplace_provenance.rs).

---

## 2. Safety, `#![forbid(unsafe_code)]` & Real-Time Audio Invariants

1. **Strict `unsafe` Isolation:**
   - `src/lib.rs` and all algorithmic submodules (`core1`, `core2`, `control`, `container`, `provenance`, `schema`, `engine`, `dynamic`) enforce `#[forbid(unsafe_code)]`.
   - Use disjoint struct field borrows (such as `DnamEngine::step_frame_const`) rather than raw pointer casts in `engine.rs`.
   - `unsafe` is permitted **only** in hardware SIMD intrinsic modules (`src/math/neon.rs`, `src/math/x86_simd.rs`, `src/math/wasm_simd128.rs`, `src/math/arm_sme.rs`), hardware `FTZ`/`DAZ` CSR guards (`src/types.rs`), 64-byte aligned allocation (`src/memory.rs`), and C-ABI pointer entrypoints (`src/ffi.rs`). Every `unsafe` block must carry an explicit `// SAFETY:` justification.
2. **Zero Audio-Thread Heap Allocations:**
   - `process_block`, `process_block_in_place`, `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_sample_rate`, `set_control_value`, `reset_to_dc_equilibrium`, `telemetry_snapshot`, `from_bytes_with_scratch`, and `reload_model_in_place` must never allocate or deallocate heap memory.
3. **Zero `FSQRT`, `FDIV`, or `libm` in $48\text{ kHz}$ / $3\text{ kHz}$ Paths:**
   - Keep `arena_tiles` resolution and `finalize_block_telemetry` horizontal reductions outside the per-sample loop, and specialize block loops via compile-time tier monomorphization (`step_frame_const::<GROUPS, FDN_GROUPS>`).

---

## 3. Testing, Coverage ($\ge 95\%$) & Benchmarking Commands

Run all commands from the monorepo root:

```bash
# 1. Format and lint with zero warnings
cargo fmt --all -- --check
cargo clippy --package dnam-engine --all-targets -- -D warnings

# 2. Run all unit, property-based (proptest), and integration tests
cargo test --package dnam-engine

# 3. Verify >= 95% line, region, and function coverage
cargo llvm-cov --package dnam-engine --show-missing-lines

# 4. Run real-time throughput benchmarks
cargo bench --package dnam-engine

# 5. Run the standalone examples
cargo run --package dnam-engine --example realtime_audio_block --release
```

---

## 4. Pre-Commit Checklist for `packages/engine/`

- [ ] `cargo fmt --all -- --check` and `cargo clippy --package dnam-engine --all-targets -- -D warnings` pass with `0` warnings.
- [ ] `cargo test --package dnam-engine` passes all unit, `proptest`, and integration tests (including `tests/realtime_safety.rs` and `tests/backend_parity.rs`).
- [ ] `cargo llvm-cov --package dnam-engine` confirms $\ge 95\%$ coverage, including all newly added or modified functions and tier branches.
- [ ] If any public API or FFI signature changed, `packages/python/rust/` bindings and scoped `AGENTS.md` files (`< 2,500 B`) are updated accordingly.
