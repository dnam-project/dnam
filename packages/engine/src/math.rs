//! Multi-platform SIMD & hardware math abstraction (`SimdVec4` & `MathBackend`) plus
//! division-free algebraic DSP primitives (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-t.reader.catmull-rom]`).
//!
//! All real-time audio ($48\text{ kHz}$) and control-bus ($3\text{ kHz}$) kernels execute with
//! **zero `FSQRT`, zero `FDIV`, and zero `libm` transcendental calls**.

#[forbid(unsafe_code)]
pub mod backend;
#[forbid(unsafe_code)]
pub mod crc32c;
#[forbid(unsafe_code)]
pub mod default_ops;
#[forbid(unsafe_code)]
pub mod fp16;
#[forbid(unsafe_code)]
pub mod init_math;
#[forbid(unsafe_code)]
pub mod kernels;
#[forbid(unsafe_code)]
pub mod scalar;
#[forbid(unsafe_code)]
pub mod spec_math;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod arm_sme;

#[cfg(target_arch = "x86_64")]
pub mod x86_simd;

#[cfg(target_arch = "wasm32")]
#[forbid(unsafe_code)]
pub mod wasm_simd128;

pub use backend::{MathBackend, SimdVec4, flush_subnormal_f32_bits};
pub(crate) use backend::{flush_recursive_state_f32, flush_recursive_state_vec4};
pub use crc32c::{crc32c, crc32c_slice};
pub use fp16::{
    f32_to_fp16, f32_to_fp16 as f32_to_fp16_bits, fp16_to_f32, fp16_to_f32 as fp16_bits_to_f32,
};
pub use init_math::{
    cos_pi_unit_f64, exp_f64, floor_f32, floor_f64, ln_f64, powf_f64, round_f32, round_f64,
    sqrt_f64,
};
pub use kernels::{
    algebraic_unit_rotor, algebraic_unit_rotor_f64, clamp_pre_activation, cubic_lagrange_weights,
    cubic_lagrange_weights_f64, eval_poly_adaa_algebraic, eval_poly_adaa_algebraic_f64,
    eval_poly_adaa_antiderivative_f64, eval_poly_adaa_waveshaper, eval_poly_adaa_waveshaper_f64,
    stabilize_complex_rotor,
};
pub use scalar::{ScalarBackend, ScalarVec4};
pub use spec_math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, FP16_MAX_FINITE_F32, INV_TAU_256_LE, InvalidFloatInput,
    LOG2_MANTISSA_LUT_129_BITS, Log2EnvelopeDecomp, MAX_ROTOR_ANGLE_F32_BITS,
    PITCH_LN2_OVER_12_F64_BITS, Q0_192Phase, SAFETY_CEILING_F32, contractive_rotor_f32,
    contractive_rotor_f32_bits, decompose_positive_f32_bits, eval_monotone_log2_f32_bits,
    eval_pitch_profile_f32, f32_bits_at_rank, f32_coordinate_count, f32_nearest_admissible_bits,
    f32_rank_boundary_for_f64, f32_rank_intersection, f32_rank_of_bits, f32_to_fp16_with_telemetry,
    f32_unbiased_exponent, lookup_log2_mantissa_f32, newton6_reciprocal_f64, next_up_pos_f64,
    normalize_declared_range_bits, project_rotor_components_ulp_guard,
    q0_192_phase_increment_from_f32_bits, q32_phase_to_phasor, round_f64_to_f32_bits,
    sanitize_input_sample, sanitize_internal_sample, sanitize_output_sample, saturation_flags,
    step_lfo_phasor_zero_hold, trust_clip_fast_candidate, trust_clip_outward_4d,
    trust_clip_outward_8d, trust_clip_outward_checked_bits, trust_clip_reference_bisection_bits,
    upper_product_f64, upper_sum_squares_f64, widen_f32_bits,
};

#[cfg(target_arch = "aarch64")]
pub use neon::{NeonBackend, NeonVec4};

#[cfg(target_arch = "aarch64")]
pub use arm_sme::ArmSmeBackend;

#[cfg(target_arch = "x86_64")]
pub use x86_simd::{X86SimdBackend, X86Vec4};

#[cfg(target_arch = "wasm32")]
pub use wasm_simd128::{WasmSimd128Backend, WasmVec4};

/// Compile-time default `MathBackend` for `aarch64` (`NeonBackend`, while runtime dispatch via
/// [`detect_best_backend`] / [`crate::DynamicDnamEngine`] automatically upgrades to [`ArmSmeBackend`]
/// when `FEAT_SME` is detected).
#[cfg(target_arch = "aarch64")]
pub type DefaultMathBackend = NeonBackend;

/// Compile-time optimal default `MathBackend` for `x86_64`.
#[cfg(target_arch = "x86_64")]
pub type DefaultMathBackend = X86SimdBackend;

/// Compile-time optimal default `MathBackend` for `wasm32`.
#[cfg(target_arch = "wasm32")]
pub type DefaultMathBackend = WasmSimd128Backend;

/// Compile-time fallback `MathBackend` for other architectures.
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
pub type DefaultMathBackend = ScalarBackend;

/// Identifier for available hardware math backends used by `DynamicDnamEngine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Portable 16-byte aligned scalar/auto-vectorized backend.
    Scalar,
    /// ARM64 128-bit NEON (`float32x4_t`, `vfmaq_f32`, hardware `CRC32C`).
    ArmNeon,
    /// Alias for [`Self::ArmNeon`].
    Neon,
    /// ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`, e.g., Apple Silicon M4+, Cortex-X925+).
    ArmSme,
    /// x86-64 SSE4.2 (`__m128` + hardware `_mm_crc32_u64`).
    X86Sse42,
    /// x86-64 AVX2 + FMA + F16C (`__m128` / `__m256` + `_mm_fmadd_ps`).
    X86Avx2Fma,
    /// Alias for [`Self::X86Avx2Fma`].
    X86Simd,
    /// WebAssembly 128-bit SIMD (`v128` `f32x4`).
    WasmSimd128,
}

impl BackendKind {
    /// Returns a concise human-readable name for benchmark tables and telemetry.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::ArmNeon | Self::Neon => "ARM64 NEON",
            Self::ArmSme => "ARM SME",
            Self::X86Sse42 => "x86 SSE4.2",
            Self::X86Avx2Fma | Self::X86Simd => "x86 AVX2",
            Self::WasmSimd128 => "WASM SIMD",
        }
    }
}

/// Detects the highest-performance hardware math backend supported by the current CPU at runtime.
#[inline]
pub fn detect_best_backend() -> BackendKind {
    #[cfg(target_arch = "aarch64")]
    {
        if ArmSmeBackend::is_supported() {
            BackendKind::ArmSme
        } else {
            BackendKind::Neon
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
            BackendKind::X86Avx2Fma
        } else {
            BackendKind::X86Sse42
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if cfg!(target_feature = "simd128") {
            BackendKind::WasmSimd128
        } else {
            BackendKind::Scalar
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )))]
    {
        BackendKind::Scalar
    }
}
