# `src/math` — Hardware SIMD & Division-Free Algebraic Primitives

Implements the two-level hardware math abstraction ([`SimdVec4`](backend.rs) & [`MathBackend`](backend.rs)) and division-free DSP kernels.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`backend.rs`](backend.rs) | `SimdVec4` (4-lane `f32`) & `MathBackend` traits + cross-backend `proptest` | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`kernels.rs`](kernels.rs), [`spec_math.rs`](spec_math.rs) | Division-free `PolyADAA`, quadrant-reduced Horner rotor, `Q0.192` phase, pitch/log2 LUT, outward trust clip, `f32` rank bounds | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md), [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`init_math.rs`](init_math.rs) | `#![no_std]` cold-init `f64` math (`exp_f64`, `ln_f64`, `powf_f64`, `sqrt_f64`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`fp16.rs`](fp16.rs), [`crc32c.rs`](crc32c.rs) | IEEE-754 `f32` ↔ `fp16` (`u16`) & Castagnoli `CRC32C` (`0x82F63B78`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`scalar.rs`](scalar.rs), [`wasm.rs`](wasm.rs) | `#[forbid(unsafe_code)]` `ScalarBackend` & WebAssembly `v128` SIMD128 backend (`WasmSimd128Backend`) | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`aarch64.rs`](aarch64.rs), [`x86.rs`](x86.rs) | Unified `aarch64` (`ArmSmeBackend`, `ArmSve2Backend`, `NeonBackend`) and `x86_64` (`X86Avx2FmaBackend`, `X86Sse42Backend`, `X86Sse2Backend`) hardware libraries | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `zerocopy::transmute!` and SIMD safety rules.
