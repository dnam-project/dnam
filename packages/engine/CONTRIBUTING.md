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
- **`src/container/` ([`src/container/AGENTS.md`](src/container/AGENTS.md)):** 64-byte `.dnam` preamble, zero-copy `DnamContainerView`, `DMU1`/`DMUE` mutable suffix framing, and Continuous-Time ZOH sample-rate rescaler ($44.1\text{–}96\text{ kHz}$). (`.dnam-ckpt` and `uplo='U'` Gram packing live in `packages/trainer/src/ckpt/`.)
- **`src/provenance/` ([`src/provenance/AGENTS.md`](src/provenance/AGENTS.md)):** `DAU1` authentication envelopes, RFC 8785 JCS header digests, `Ed25519` prefix commitments, `OemDeviceCertificate` attestation, RFC 9180 HPKE & `DME1` encryption, Dual-Mechanism buyer watermarking, and `.dnam-prev` containment.
- **`src/schema/` ([`src/schema/AGENTS.md`](src/schema/AGENTS.md)):** `#![no_std]` `serde` JSON header types, UI skew curves, and `DnamError`.

### Test, Benchmark & Example Placement

- **Unit & Property Tests (`#[cfg(test)] mod tests` / `mod spec_tests`):** Co-locate all module unit tests, specification verification suites (`verify[...]`), and `proptest!` suites inside the submodule they test (`src/math/spec_math.rs`, `src/container/wire.rs`, `src/provenance/security.rs`, `src/control/plane.rs`, `src/control/bus.rs`, `src/operators/{op_t,op_m,op_n,reference}.rs`, `src/stages/{topology,stage4_tank}.rs`).
  - **Use `proptest!` Wherever Possible:** Every mathematical kernel, fixed-point accumulator, DSP operator (`Op-T`, `Op-M`, `Op-N`), stage recurrence, wire parser/serializer, and cryptographic framing helper must include `proptest!` property tests sweeping its full input domain (coupled with deterministic boundary/vector tests for exact IEEE-754 bit patterns). Keep `tests/` strictly for cross-cutting multi-subsystem integration tests (which should also use `proptest!` for end-to-end block/rate/tier sweeps).
- **Integration Tests (`tests/`):** Keep `tests/` strictly for cross-cutting crate integration tests:
  - [`tests/backend_parity.rs`](tests/backend_parity.rs): Backend and topology parity checks (`0.000000 ULP`) across `ScalarBackend`, `NeonBackend`, `ArmSve2Backend`, `ArmSmeBackend`, `X86Sse2Backend`, `X86Sse42Backend`, `X86Avx2FmaBackend`, and `WasmSimd128Backend`.
  - [`tests/realtime_safety.rs`](tests/realtime_safety.rs): `TrackingAllocator` checks for zero-allocation real-time audio callbacks.
  - [`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs): Automated release-binary `objdump -dr` disassembly and transitive call-graph audit verifying `0` `FDIV`, `FSQRT`, `IDIV`/`UDIV`/`SDIV`, `libm`/`fmodf`, heap allocator, or uncached OS syscall instructions (`ref[repo.code.realtime-disassembly-audit]`).
  - [`tests/tora_pipeline_and_zoh_verification.rs`](tests/tora_pipeline_and_zoh_verification.rs): End-to-end TORA pipeline (`Tests 8H, 8I`), ZOH multi-rate rescaling, `4x` offline bounce, calibration roles, and `arch.*` system architecture integration.
  - [`tests/marketplace_provenance_and_watermark.rs`](tests/marketplace_provenance_and_watermark.rs) & [`tests/marketplace_seal_and_policy.rs`](tests/marketplace_seal_and_policy.rs): End-to-end cryptographic provenance, `.dnam-seal` hardware enclave envelopes, and Dual-Mechanism buyer watermarking.
  - [`tests/ffi_integration.rs`](tests/ffi_integration.rs): C-ABI lifecycle and null-pointer safety.
- **Benchmarks (`benches/`):** Organize Criterion throughput and WCET latency benchmarks by subsystem across [`benches/pipeline.rs`](benches/pipeline.rs), [`benches/operators.rs`](benches/operators.rs), [`benches/stages_and_control.rs`](benches/stages_and_control.rs), [`benches/container_and_provenance.rs`](benches/container_and_provenance.rs), and [`benches/realtime_wcet.rs`](benches/realtime_wcet.rs).
- **Examples (`examples/`):** Place runnable API demonstrations in [`examples/realtime_audio_block.rs`](examples/realtime_audio_block.rs), [`examples/embedded_zero_alloc.rs`](examples/embedded_zero_alloc.rs), [`examples/marketplace_provenance.rs`](examples/marketplace_provenance.rs), [`examples/container_slimming_and_ckpt.rs`](examples/container_slimming_and_ckpt.rs), [`examples/custom_operator_chain.rs`](examples/custom_operator_chain.rs), and [`examples/c_abi_host_integration.rs`](examples/c_abi_host_integration.rs).

---

## 2. Safety, `#![forbid(unsafe_code)]` & Real-Time Audio Invariants

1. **Strict `unsafe` Isolation & Safe-Rust-First Requirement (`ref[repo.code.type-safety-and-imports]`):**
   - `src/lib.rs` enforces `#![deny(unsafe_code)]` and all algorithmic and concurrency submodules (`operators`, `stages`, `control`, `container`, `provenance`, `schema`, `engine`, `dynamic`, `host`) enforce `#[forbid(unsafe_code)]` (zero `#[allow(unsafe_code)]`).
   - **Never use `#[allow(unsafe_code)]` out of convenience:** Always design lock-free rings, seqlocks, hot-swap handles, and DSP state borrowing using safe Rust primitives first (`core::sync::atomic::AtomicU64` word arrays, atomic slot indices with `try_lock()`, disjoint struct field borrows in `step_frame_tora`, and `zerocopy::transmute!`).
   - `#[allow(unsafe_code)]` is permitted **only** when **(a)** it is strictly unavoidable at a hardware/OS intrinsic or C-ABI boundary (`src/math/{aarch64,x86}.rs` and `src/math/aarch64/{neon,sve2,sme}.rs` + `src/math/x86/{sse2,sse42,avx2}.rs`, `src/types.rs` `FTZ`/`DAZ` CSR guards, `src/memory.rs` 64-byte aligned allocation, and `src/ffi.rs` C-ABI entrypoints; `src/math/wasm.rs` is `#[forbid(unsafe_code)]`), or **(b)** Criterion benchmarks prove the safe Rust checker version incurs a high real-time performance penalty **and** the `unsafe` block is guaranteed sound with an explicit `// SAFETY:` proof.
2. **Zero Audio-Thread Heap Allocations:**
   - `process_block`, `process_block_in_place`, `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_sample_rate`, `set_control_value`, `reset_to_dc_equilibrium`, `telemetry_snapshot`, `from_bytes_with_scratch`, and `reload_model_in_place` must never allocate or deallocate heap memory.
3. **Zero `FSQRT`, `FDIV`, `IDIV`/`UDIV`/`SDIV`, `libm`, or Nondeterministic Calls in $48\text{ kHz}$ / $3\text{ kHz}$ Paths (`ref[arch.simd.zero-stall-lowering]`):**
   - Never use floating-point division, square-root, integer division/modulo (`/` or `%` on runtime integers, `.is_multiple_of`), `f32::rem_euclid` (which lowers to `bl _fmodf`), transcendental `libm` calls, unbounded `while` loops, or uncached OS syscalls in real-time audio or control-bus loops; use `floor_f32((theta + PI) * INV_TAU)`, incremental `(block_idx, in_block_offset)` counters, cached reciprocals, `newton6_reciprocal_f64`, and `sqrt_f64`.
4. **Submodule & SIMD Implementation Invariants (`ref[arch.simd.two-level-traits]`, `ref[repo.code.explicit-simd-and-coverage]`):**
   - **`src/math` & Domain Backend Traits:** Keep `SimdVec4` and `MathBackend` (`Crc32cBackend`) in `src/math/` and define higher-level operator/stage traits (`OpMBackend`, `OpNBackend`, `OpTBackend`, `Stage2Backend`, `Stage3Backend`, `FeatureBusBackend`, `StageBackend`, `EngineBackend`) inside their respective domain modules (`engine -> stages -> operators -> math`). Write explicit trait implementations per ISA (`aarch64/{neon,sve2,sme}`, `x86/{sse2,sse42,avx2}`, `wasm`, `scalar`) with zero `macro_rules!` SIMD boilerplate and zero `#[cfg_attr(coverage, coverage(off))]`. Use `zerocopy::transmute!` for `[f32; 4]` $\leftrightarrow$ SIMD register conversions and unfused `rsqrt_nr2_pair_f32` for `0.000000 ULP` cross-backend parity.
   - **`src/stages` & `src/engine`:** Preserve in-place `feature_bus_scratch` packing, load-time `cached_role_b_slices` folding, `3 kHz` dirty-checked `collapsed_head_wl/wr` pre-collapse, and Stage III double-mirrored reverse-order FIR histories (`fir_rev_history`). Account for all per-instance state and verify tier-local allocation and alignment against the [state audit](../../docs/specs/architecture/tora_architecture.md#23-state-ram-audit-lite-budget). Existing arithmetic constants are not measured budgets.
   - **`src/control`:** Follow the [node-anchored control basis](../../docs/specs/dsp/parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis). A normalized default is zero only when the physical default is the declared midpoint.
   - **`src/container` & `src/schema`:** Validate checked bounds, alignment, tier-prefix integrity, and authenticated shared data against the [container format](../../docs/specs/containers/dnam_binary_format.md); do not assume fixed tier lengths or treat `CRC32C` as authentication. Keep `serde` / `serde_json` configured with `default-features = false, features = ["alloc"]`.
   - **`src/provenance`:** Apply the exact versioned [immutable-header projection](../../docs/specs/security/provenance_and_enclave_drm.md#21-preset-safe-canonical-header-digest-rfc-8785-jcs); do not recursively exclude same-named fields. Mutable edits are constrained by the authenticated region and its capacity.

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

# 5. Profile Criterion benchmarks with hotpath-rs (--profile profiling)
HOTPATH_LIMIT=0 HOTPATH_CRITERION_ITERATIONS=500 \
  cargo bench -p dnam-engine --profile profiling --bench pipeline --features hotpath-alloc \
  -- --test --exact "process_block_128/ARM64 NEON/ultra"
```

Coverage is a floor, not a bug-free guarantee. Independently challenge the
specification and implementation with boundary, differential, long-run,
backend, integration, and target-hardware tests; inspect untested branches
and platform-specific paths separately.

---

## 4. Pre-Commit Checklist for `packages/engine/`

- [ ] Every new or modified module includes `proptest!` property-based tests (alongside exact boundary/vector tests) co-located in its source file (`#[cfg(test)]`), with `verify[...]` citations for all covered `r[...]` specification rules.
- [ ] `moon check engine` passes all formatting, Clippy (`-D warnings`), build, and unit/`proptest`/integration tests (`tests/realtime_safety.rs` and `tests/backend_parity.rs`).
- [ ] `moon run engine:coverage` confirms ≥ 95% coverage, including all newly added or modified functions and tier branches.
- [ ] If any public API or FFI signature changed, `packages/js/rust/` bindings and scoped `AGENTS.md` files (`< 2,500 B`) are updated accordingly.
