# `src/math` — Hardware SIMD & Division-Free Algebraic Primitives

Implements the two-level hardware math abstraction ([`SimdVec4`](backend.rs) & [`MathBackend`](backend.rs)) and division-free DSP kernels (`CL-01..03`, `CL-06`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`backend.rs`](backend.rs) | `SimdVec4` (4-lane `f32`) & `MathBackend` traits + cross-backend `proptest` | [`core1_hybrid_ssm_and_polyadaa.md`](../../../../docs/core1_hybrid_ssm_and_polyadaa.md) |
| [`kernels.rs`](kernels.rs) | Division-free algebraic `PolyADAA`, 4-FMA + Newton rotor, Cubic Lagrange | [`clamp_and_audit_registry.md`](../../../../docs/clamp_and_audit_registry.md) |
| [`init_math.rs`](init_math.rs) | `#![no_std]` cold-init `f64` math (`exp_f64`, `ln_f64`, `powf_f64`, `sqrt_f64`) | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`fp16.rs`](fp16.rs), [`crc32c.rs`](crc32c.rs) | IEEE-754 `f32` $\leftrightarrow$ `fp16` (`u16`) & Castagnoli `CRC32C` (`0x82F63B78`) | [`dnam_container_schema.md`](../../../../docs/dnam_container_schema.md) |
| [`scalar.rs`](scalar.rs), [`wasm_simd128.rs`](wasm_simd128.rs) | `#[forbid(unsafe_code)]` `ScalarBackend` & WebAssembly `v128` SIMD128 backend | [`monorepo_implementation_guide.md`](../../../../docs/monorepo_implementation_guide.md) |
| [`neon.rs`](neon.rs), [`arm_sme.rs`](arm_sme.rs), [`x86_simd.rs`](x86_simd.rs) | ARM64 NEON, ARMv9.2-A `FEAT_SME`, and `x86_64` AVX2+FMA+F16C+SSE4.2 backends | [`monorepo_implementation_guide.md`](../../../../docs/monorepo_implementation_guide.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `zerocopy::transmute!` and SIMD safety rules.
