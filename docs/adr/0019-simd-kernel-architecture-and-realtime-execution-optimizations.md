---
id: ADR-0019
title: "Two-Level SIMD Trait Architecture, Single-Library Multi-ISA Dispatch & Real-Time Loop Optimizations"
status: accepted
date: "2026-11-10"
deciders:
  - DNAM Architecture Team
supersedes: []
superseded_by: null
specs:
  - "../specs/architecture/tora_architecture.md"
  - "../specs/architecture/project_and_documentation_quality.md"
  - "../specs/dsp/op_m_modal_kautz.md"
  - "../specs/dsp/op_n_commuted_cascade.md"
  - "../specs/dsp/op_t_transport.md"
  - "../specs/dsp/router_and_stages_topology.md"
  - "../specs/dsp/parametric_control_bus.md"
  - "../specs/sdks/typescript_sdk.md"
requirements:
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
---

# ADR-0019: Two-Level SIMD Trait Architecture, Single-Library Multi-ISA Dispatch & Real-Time Loop Optimizations

## Context and Problem Statement

To meet `REQ-6.1` (`≥ 10× Lite`, `≥ 6× Full`, and `≥ 2× Pro` simultaneous `48 kHz` instances on a single `2.4 GHz` ARM Cortex-A76 core), `REQ-6.2` (zero pipeline-stalling `FDIV`, `FSQRT`, `IDIV`/`UDIV`/`SDIV`, `libm`/`fmodf` calls, or unbounded loops), and `REQ-6.3` (zero real-time allocations, zero blocking locks or uncached OS syscalls, and deterministic subnormal immunity) while maintaining `0.000000 ULP` bit parity across `ARM64` (`NEON`, `SVE2`, `SME`), `x86_64` (`SSE2`, `SSE4.2`, `AVX2+FMA`), `wasm32` (`SIMD128`), and `ScalarBackend`, how should `dnam-engine` structure its SIMD trait hierarchy, multi-ISA binary dispatch, stereo operator vectorization, feature-bus/head collapse, and CI disassembly verification so real-time DSP code is implemented optimally the first time?

## Decision Drivers

- **Strict One-Way Dependency DAG (`engine -> stages -> operators -> math`):** Low-level register abstractions in `math/` must never depend on higher-level operator or stage structs, yet wide-ISA backends (`AVX2`, `ARM SME`) must be able to override multi-vector stage/head kernels (`ref[arch.simd.two-level-traits]`).
- **Single Binary Per Architecture & Zero-Syscall Dispatch:** Shipping separate `.so`/`.dylib`/`.dll` builds per sub-ISA complicates distribution, while calling `sysctlbyname`/`getauxval`/`mrs` on the audio path violates `REQ-6.3` (`ref[arch.simd.single-library-dispatch]`).
- **Cross-Backend `0.000000 ULP` Bit Parity:** Hardware `FMA` vs. unfused multiply-add in sensitive Newton-Raphson iterations (`rsqrt_nr2_pair_f32`) or divergent control-plane math can cause ULP drift across architectures (`ref[arch.simd.single-library-dispatch]`).
- **True-Stereo & Multi-Lane Arithmetic Density:** Naive per-channel scalar loops waste `50%`–`75%` of 4-lane/8-lane SIMD registers, suffer per-tap modulo index overhead in FIR and `fp16` delay buffers, and recompute static control/head projections on every `48 kHz` sample.
- **Compiler Lowering Pitfalls (`bl _fmodf`, `udiv`, `fdiv`):** High-level Rust constructs such as `f32::rem_euclid`, `.is_multiple_of(self.block_size)`, and runtime `1.0 / (len - 1)` silently lower to external `libm` calls or variable-latency division instructions unless prohibited by design and caught by automated release-binary disassembly (`ref[arch.simd.zero-stall-lowering]`, `ref[repo.code.realtime-disassembly-audit]`).

## Considered Options

1. **Option 1 — Monolithic `MathBackend` Trait, Separate Per-ISA Binaries & Per-Channel Scalar Loops:** Put all operator/stage methods on a single trait in `math/`, compile separate binaries per CPU feature level, evaluate Left and Right channels sequentially, and collapse the 256-D Kronecker head per audio sample.
2. **Option 2 — Macro-Generated SIMD Boilerplate & Source-Only Regex Linting:** Generate SIMD backends with `macro_rules!`, rely on compiler auto-vectorization for stereo/FIR/head loops, and check only source text for `/` or `sqrt`.
3. **Option 3 (Chosen) — Two-Level Domain-Scoped SIMD Trait Hierarchy, Single-Library `#[target_feature]` + `AtomicU8` Dispatch, Stereo/Pre-Collapsed Kernels & Automated Release Disassembly Audit.**

## Decision Outcome

Chosen option: **Option 3**, organized across five normative architectural pillars (`ref[arch.simd.two-level-traits]`, `ref[arch.simd.single-library-dispatch]`, `ref[arch.simd.zero-stall-lowering]`, `ref[repo.code.realtime-disassembly-audit]`, `ref[repo.code.explicit-simd-and-coverage]`):

### Pillar 1 — Two-Level SIMD Trait Hierarchy & Domain-Scoped Backend Traits (`ref[arch.simd.two-level-traits]`)

| Hierarchy Level | Trait & Module Location | Architectural Role & Key Primitives |
| :--- | :--- | :--- |
| **Level 1 (Register & Core Math)** | `SimdVec4`, `MathBackend` & `Crc32cBackend` (`math/backend.rs`) | Portable 4-lane `f32` register (`fma_lane::<const LANE: usize>`, `horizontal_sum_pair`, `splat_pair`, `pair_swap_neg`, `flush_subnormal`, paired with `flush_recursive_state_vec4`) plus `MathBackend` (`fma_mat4x4_vec4`, `f32x4_to_fp16x4`, `fp16x4_to_f32x4`, `rsqrt_nr2_f32`, `rsqrt_nr2_pair_f32`, `fast_unit_phasor`, and hardware `CRC32C`). |
| **Level 2 (Domain-Scoped Traits)** | `OpMBackend`, `OpNBackend`, `OpTBackend` (`operators/`), `Stage2Backend`, `Stage3Backend`, `StageBackend` (`stages/`), `FeatureBusBackend`, `EngineBackend` (`engine/`) | Defined inside their respective domain modules (`engine -> stages -> operators -> math`), providing default `SimdVec4` implementations that wide-ISA backends (`X86Avx2FmaBackend`, `ArmSmeBackend`) override with multi-vector intrinsics (`tile_dot64`, `tile_dot64_stereo`, `dot16_stereo`, `conv6_taps_{1..4}groups`, `tracker_exp_scales_f32`). |

- **Hardware `FtzDazGuard` Elision (`ref[arch.fp.callback-env]`):** Because `FtzDazGuard` sets `FPCR.FZ` on `aarch64` and `MXCSR.FTZ`/`DAZ` on `x86_64` once at block entry, `flush_recursive_state_f32` and `flush_recursive_state_vec4` compile to zero-cost no-ops on `aarch64`/`x86_64` while preserving branchless exponent-mask flushing on `wasm32` and portable targets.

### Pillar 2 — Single-Library Per-Architecture SIMD Dispatch & `0.000000 ULP` Parity (`ref[arch.simd.single-library-dispatch]`)

| Target Architecture | Compiled Backend Modules (`packages/engine/src/math/`) | Runtime Detection & Caching (`AtomicU8`) |
| :--- | :--- | :--- |
| **`aarch64`** | `aarch64/neon.rs` (`NeonBackend`), `aarch64/sve2.rs` (`ArmSve2Backend`), `aarch64/sme.rs` (`ArmSmeBackend`) | Function-level `#[target_feature(enable = "sve2")]` / `sme` + `AtomicU8`-cached `hw.optional.arm.FEAT_SME` / `getauxval(AT_HWCAP2)` + `ID_AA64PFR1_EL1` fallback. |
| **`x86_64`** | `x86/sse2.rs` (`X86Sse2Backend`), `x86/sse42.rs` (`X86Sse42Backend`), `x86/avx2.rs` (`X86Avx2FmaBackend`) | Function-level `#[target_feature(enable = "avx2,fma,f16c")]` / `sse4.2` + `AtomicU8`-cached `cpuid` (`AVX2 + FMA + F16C + OSXSAVE + XGETBV`). |
| **`wasm32` & Fallback** | `wasm.rs` (`WasmSimd128Backend`), `scalar.rs` (`ScalarBackend`) | Compile-time `target_feature = "simd128"` with zero-copy shared `WebAssembly.Memory` (`--import-memory`, `-zstack-size=786432`, `[profile.release]` / `[profile.profiling]` `opt-level = 3`, `lto = "thin"`, `codegen-units = 16`, `panic = "abort"`, `wasm-opt -O1`, `ref[sdk.ts.engine.zero-copy-and-async]`). |

- **Unfused `rsqrt_nr2_pair_f32` for Cross-ISA `0.000000 ULP` Parity:** All SIMD backends evaluate the 2-step Newton-Raphson `rsqrt` refinement $\rho_{j+1} = \rho_j(1.5 - (0.5 r^2 \rho_j)\rho_j)$ using separate IEEE-754 multiply and subtract instructions (`vmulq_f32`/`vsubq_f32`, `_mm_mul_ps`/`_mm_sub_ps`, `f32x4_mul`/`f32x4_sub`) rather than fused `vfmsq_f32`/`vfnmadd`, eliminating single-ULP `FMA` rounding drift against `ScalarBackend`.

### Pillar 3 — Stereo Vectorization & Operator/Stage Algorithmic Optimizations

1. **`Op-M` & Stage I/III (`ref[dsp.op-m.recurrence.stereo-simd]`, `ref[dsp.op-m.rotor.angle-wrap-and-retune]`, `ref[dsp.op-m.tracker.stereo-batching]`):**
   - `HilbertH1Lattice6Sec::step_stereo` packs `[I_L, Q_L, I_R, Q_R]` into one `B::Vec4` register across all 6 all-pass sections (`4×` reduction in lattice loops).
   - `T1SubHarmonicTracker::step_analytic_stereo` batches L/R envelope and half-angle `rsqrt` calls via `B::rsqrt_nr2_pair_f32`.
   - `SymplecticModal4Lane::step_4lane_stereo_simd_with_readout` (Stage I) and `KautzModalBank<4>::step_4lane_stereo_simd_unscaled` (Stage III) step L/R complex modal states together in 4-pole `B::Vec4` blocks.
   - `Stage1PreDynamics::retune_op_m_pre_3khz` and `StagePipeline::retune_kautz_formant_3khz` cache `(host_op_m_cos, host_op_m_sin)` and `(kautz_host_cos, kautz_host_sin)`, skip clean ticks (`|Δ| <= 1e-9`), and rotate 4 poles in `B::Vec4` using `algebraic_unit_rotor_op_m` angle-addition + 1st-order Taylor renormalization (`scale = min(1.0, 1.5 - 0.5 * norm_sq)`).
2. **`Op-N` & Stage II (`ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`, `ref[dsp.op-n.stage2.conv6-simd]`):**
   - Highest-drive-lane (`lane 3 = 16.0x` in `V1A`, `lane 2 = 4.0x` in `V1B`) interior check skips the 6-step `f64` Newton rail-crossing branch for all 4 lanes when the highest drive is within `[-1, 1]`.
   - `step_stereo_pair` interleaves L/R `V1A` registers, while `step_stereo_lanes_1_2_lr` packs only the active `[l_1x, l_4x, r_1x, r_4x]` lanes (`lanes 1..3` of `left` and `right`) in `V1B` (`50%` fewer `V1B` SIMD operations while preserving both channels' canonical 4-drive layout).
   - `Stage2Backend::conv6_taps_{1..4}groups` (`conv6_taps_{1..4}groups_vec4`) uses 4 independent vector accumulators (`fma_lane::<0..3>`), paired with stereo `step_interstage_filter_stereo` and `step_branch_shaper_stereo`.
3. **Stage III Double-Mirrored Reverse-Order FIR History (`ref[dsp.stage3.mirrored-fir-simd]`):**
   - `Stage3CabinetModals` maintains `fir_rev_history: [f32; 128]` per channel (`stage3` and `stage3_r`), writing each sample at `rev` and `rev + 64` for `rev = (64 - (self.fir_write_ptr & 63)) & 63`, guaranteeing contiguous 16-tap slices evaluated via `Stage3Backend::dot16_stereo` + `SimdVec4::horizontal_sum_pair` with zero per-tap index wrapping.
4. **`Op-T` & Stage IV (`ref[dsp.op-t.reader.contiguous-fp16x4]`, `ref[dsp.stage4.simd-tank-loop]`):**
   - `read_delay_stereo` uses contiguous 4-lane `MathBackend::fp16x4_to_f32x4` hardware loads whenever `p_0 >= 2 && p_0 + 1 < Self::HALF_CAPACITY`.
   - `Stage4SpatialTank::step_stereo_with_readouts` vectorizes the 4-line tank DC blockers (`dc_x_prev`, `dc_y_prev`), 4-lane 3-step Newton `coercive_saturate_4ch`, cached-angle Givens `pair_swap_neg` (`|target_psi - current_psi| <= 1e-7`), and 4-channel Householder reflection in `B::Vec4`.

### Pillar 4 — Zero-Copy Feature Bus, Two-Stage Head Pre-Collapse & `3 kHz` Control Bus (`ref[dsp.head.precollapse-and-zerocopy]`, `ref[dsp.ctrl.op-n.static-and-interior-fastpath]`, `ref[dsp.ctrl.dispatch.sparse-and-active-ramps]`)

1. **In-Place 256-D Feature Bus (`feature_bus_scratch`):** Stages I–IV write directly into `feature_bus_scratch: [f32; 256]` (`Stage2CommutedCascade::step_frame_into_feature_bus`, `KautzModalBank<4>::step_4lane_stereo_simd_unscaled`, `Stage4SpatialTank::read_multi_tier_taps_into`) with zero intermediate buffer copies.
2. **Two-Stage Kronecker Head Pre-Collapse:** Load-time folding of `191` Role B blocks (`DnamEngine::refresh_cached_role_b_slices`) into `cached_role_b_slices: [[B::Vec4; 32]; 4]` + `3 kHz` dirty-checked SVD head collapse (`DnamEngine::refresh_collapsed_head_weights`) into `collapsed_head_wl/wr: [[B::Vec4; 16]; 4]` reduces `48 kHz` `eval_tier_feature_bus_head_dot` to a single 4-way unrolled `B::Vec4` FMA loop + `SimdVec4::horizontal_sum_pair` over `collapsed_head_wl/wr` (alongside `FeatureBusBackend::tile_dot64_stereo` for `[f32; 64]` slice tiles).
3. **`3 kHz` Control Plane Fast Paths:** `ControlBus<B>` caches converted `Q0.192` LFO phase increments (`lfo_cached_angle_bits`, `lfo_cached_q192_inc`), skips `evaluate_static_control_basis` when `!is_dirty`, exits `trust_clip_outward_4d` immediately when `norm_sq + 1e-9 < r^2`, skips sparse `q_8` columns (`|q_val| <= 1e-12`) and evaluates `W_ctrl` stage rows `3..12` (`2..13` when `live_analog_drift_enabled && !offline_bounce && sigma_drift > 0.0`) in `evaluate_tank_modulation`, and steps `theta_stage[0..12]` via `3 × B::Vec4` in `compute_micro_ramp_deltas_12` / `step_micro_ramp_12`.
4. **L1i-Resident Frame Kernel & Cross-CGU Monomorphization Deduplication:** Marking `step_frame_tora`, `step_frame_4x_tora`, `step_frame_4port_tora`, `step_frame_4x_4port_tora`, `step_control_tick_3khz`, and `refresh_collapsed_head_weights` `#[inline(never)]` collapses 20 force-inlined copies per backend down to a single L1 instruction-cache-resident frame kernel (improving both native ARM64 NEON and WASM SIMD128 runtime throughput while cutting LLVM register-allocation time by `> 6×`), and routing `DnamJsonHeader` / `DnamCkptHeader` `serde_json` serialization through non-generic `#[inline(never)]` `from_json_slice` / `to_json_vec` helpers eliminates `1.5 MB` of duplicated cross-CGU `serde` monomorphizations under `codegen-units = 16`.

### Pillar 5 — Zero Pipeline Stalls, Zero Nondeterministic Calls & Automated Disassembly Gate (`ref[arch.simd.zero-stall-lowering]`, `ref[repo.code.realtime-disassembly-audit]`, `ref[repo.code.explicit-simd-and-coverage]`)

1. **Division-Free & `libm`-Free Lowering:** `floor_f32((theta + PI) * INV_TAU)` replaces `rem_euclid` (`bl _fmodf`) and `while` loops; incremental `(block_idx, in_block_offset)` counters replace `pos / block_size` (`udiv`); cached `inv_xfade_denom` and `newton6_reciprocal_f64` / `sqrt_f64` eliminate all `fdiv` and `fsqrt` instructions across `packages/engine`.
2. **Automated CI Disassembly & Call-Graph Audit (`realtime_disassembly_audit.rs`):** Disassembles release `libdnam_engine.rlib` in `moon run engine:test` to verify `0` `fdiv`/`fsqrt`/`udiv`/`sdiv`/`div`/`idiv`/`libm`/allocator/syscall instructions across all functions in the real-time transitive call-graph closure and zero `libm` relocations crate-wide.
3. **Explicit SIMD Implementations, Zero `coverage(off)` & Cross-ISA Emulation:** Prohibits `macro_rules!` SIMD wrappers and `#[cfg_attr(coverage, coverage(off))]`, and verifies `ArmSve2Backend` under QEMU (`engine:test-sve2`) and `X86Sse2Backend`/`X86Sse42Backend`/`X86Avx2FmaBackend` under Rosetta 2 (`engine:test-x86`).

### Consequences

- **Positive:** Achieves `161.59 ns` (`Lite`) to `257.41 ns` (`Ultra`) per stereo frame on desktop ARM64 NEON (`0.78%–1.24%` of the `20,833 ns` `48 kHz` budget; `20.68 µs` `Lite` to `32.95 µs` `Ultra` per `128`-frame block in `benches/pipeline.rs`, `21.73 µs` `Lite` to `34.32 µs` `Ultra` in Node.js N-API, and `31.55 µs` `Lite` to `49.05 µs` `Ultra` in WASM SIMD128 `js:profile`), `~22 s` total WASM build + Binaryen optimization (`2.2 MB` `dnam.wasm`), `0.000000 ULP` cross-backend parity across 8 backends, `> 95%` line coverage without `coverage(off)`, and `0` forbidden instructions verified automatically in CI.
- **Negative:** Requires maintaining explicit trait implementations per ISA family (`neon`, `sve2`, `sme`, `sse2`, `sse42`, `avx2`, `wasm`, `scalar`) and using `newton6_reciprocal_f64` / `floor_f32` instead of built-in `/` or `rem_euclid`.
