//! Multi-platform SIMD & hardware math abstraction (`SimdVec4` & `MathBackend`) plus
//! division-free algebraic DSP primitives (`CL-01..03`, `CL-06`).
//!
//! All real-time audio ($48\text{ kHz}$) and control-bus ($3\text{ kHz}$) kernels execute with
//! **zero `FSQRT`, zero `FDIV`, and zero `libm` transcendental calls**.

#[forbid(unsafe_code)]
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod arm_sme;

#[cfg(target_arch = "x86_64")]
pub mod x86_simd;

#[cfg(target_arch = "wasm32")]
#[forbid(unsafe_code)]
pub mod wasm_simd128;

pub use scalar::{ScalarBackend, ScalarVec4};

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

/// Level 1 Hardware Abstraction: 128-bit 4-lane `f32` SIMD register (`float32x4_t` / `__m128` / `v128`).
pub trait SimdVec4: Copy + Clone + Send + Sync + 'static {
    /// Broadcasts `val` across all 4 lanes.
    fn splat(val: f32) -> Self;
    /// Loads 4 `f32` lanes from a slice.
    fn load(src: &[f32; 4]) -> Self;
    /// Stores 4 `f32` lanes into a slice.
    fn store(self, dst: &mut [f32; 4]);
    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;
    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;
    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;
    /// Fused multiply-add (`self + a * b`).
    fn fma(self, a: Self, b: Self) -> Self;
    /// Fused multiply-subtract (`self - a * b`).
    fn fms(self, a: Self, b: Self) -> Self;
    /// Lane-wise maximum (`max(self, rhs)`).
    fn max(self, rhs: Self) -> Self;
    /// Lane-wise minimum (`min(self, rhs)`).
    fn min(self, rhs: Self) -> Self;
    /// Branchless lane-wise clamp to `[lo, hi]`.
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Lane-wise absolute value (`|self|`).
    fn abs(self) -> Self;
    /// Horizontal sum across all 4 lanes (`lane0 + lane1 + lane2 + lane3`).
    fn horizontal_sum(self) -> f32;
    /// 4-lane dot product (`sum(self * rhs)`).
    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        self.mul(rhs).horizontal_sum()
    }
    /// Returns a 4-lane vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }
    /// Constructs a 4-lane vector from a `[f32; 4]` value.
    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        Self::load(&arr)
    }
    /// Converts this 4-lane vector into a `[f32; 4]` array.
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut dst = [0.0; 4];
        self.store(&mut dst);
        dst
    }
}

/// Level 2 Hardware Abstraction: Domain-specific DSP, tile-matrix, and container kernels.
///
/// Every method provides a default zero-overhead implementation written in terms of `Self::Vec4`,
/// while hardware backends (`AppleAmxBackend`, `NeonBackend`, `X86SimdBackend`, `WasmSimd128Backend`)
/// override tile/matrix contractions or hardware instructions where platform silicon accelerates them.
pub trait MathBackend: Copy + Clone + Send + Sync + 'static {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// Core 1A (`CL-01`): Evaluates the exact division-free algebraic `PolyADAA` kernel across 4 channels:
    /// $\mathcal{A}_c[n] = \beta_c m_n(1.5 - 0.5(m_n^2 + d_n^2)) + \delta_c((m_n^2 + \tfrac{1}{3}d_n^2) - (m_n^4 + 2m_n^2 d_n^2 + \tfrac{1}{5}d_n^4))$
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: Self::Vec4,
        u_curr: Self::Vec4,
        beta: Self::Vec4,
        delta: Self::Vec4,
    ) -> Self::Vec4 {
        let half = Self::Vec4::splat(0.5);
        let one_pt_five = Self::Vec4::splat(1.5);
        let one_third = Self::Vec4::splat(1.0 / 3.0);
        let one_fifth = Self::Vec4::splat(0.2);
        let two = Self::Vec4::splat(2.0);

        let m = u_curr.add(u_prev).mul(half);
        let d = u_curr.sub(u_prev).mul(half);
        let m2 = m.mul(m);
        let d2 = d.mul(d);
        let m2_plus_d2 = m2.add(d2);

        // Odd soft-clipper term: beta * m * (1.5 - 0.5 * (m^2 + d^2))
        let odd_factor = one_pt_five.fms(half, m2_plus_d2);
        let odd_term = beta.mul(m).mul(odd_factor);

        // Even triode asymmetry term: delta * ((m^2 + d^2/3) - (m^4 + 2 m^2 d^2 + d^4/5))
        let quad_part = m2.fma(one_third, d2);
        let m4 = m2.mul(m2);
        let d4 = d2.mul(d2);
        let quart_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
        let even_term = delta.mul(quad_part.sub(quart_part));

        odd_term.add(even_term)
    }

    /// Core 1A (`CL-01`, `CL-02`): Evaluates a full 4-channel `PolyADAA` activation step with
    /// un-delayed linear path $\alpha_c x_c[n]$, $1\times$ passive B-spline blend $(0.875, 0.125)$,
    /// and $5\text{ Hz}$ AC coupling ($R_{\text{ac}}$).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn poly_adaa_layer_4ch(
        x: &[f32; 4],
        alpha: &[f32; 4],
        beta: &[f32; 4],
        delta: &[f32; 4],
        r_ac: f32,
        u_prev: &mut [f32; 4],
        a_prev: &mut [f32; 4],
        m_dc: &mut [f32; 4],
        y_out: &mut [f32; 4],
    ) {
        let vx = Self::Vec4::load(x);
        let valpha = Self::Vec4::load(alpha);
        let vbeta = Self::Vec4::load(beta);
        let vdelta = Self::Vec4::load(delta);
        let vu_prev = Self::Vec4::load(u_prev);
        let va_prev = Self::Vec4::load(a_prev);
        let vm_dc = Self::Vec4::load(m_dc);

        // CL-01: u_n = clamp((2/3) * x_c[n], -1.0, +1.0)
        let vu_curr = vx
            .mul(Self::Vec4::splat(2.0 / 3.0))
            .clamp(Self::Vec4::splat(-1.0), Self::Vec4::splat(1.0));

        let va_curr = Self::poly_adaa_algebraic_4ch(vu_prev, vu_curr, vbeta, vdelta);

        // CL-02: y_nl = 0.875 * A[n] + 0.125 * A[n-1]
        let vy_nl = va_curr
            .mul(Self::Vec4::splat(0.875))
            .fma(Self::Vec4::splat(0.125), va_prev);

        // 5 Hz AC coupling: m_dc[n] = r_ac * m_dc[n-1] + (1 - r_ac) * y_nl[n]
        let vr_ac = Self::Vec4::splat(r_ac);
        let vone_minus_rac = Self::Vec4::splat(1.0 - r_ac);
        let vm_dc_new = vm_dc.mul(vr_ac).fma(vone_minus_rac, vy_nl);

        // Un-delayed linear passband + AC-coupled nonlinear path
        let vy = vy_nl.sub(vm_dc_new).fma(valpha, vx);

        vu_curr.store(u_prev);
        va_curr.store(a_prev);
        vm_dc_new.store(m_dc);
        vy.store(y_out);
    }

    /// Core 1A (`CL-01`, `CL-02`): Register-resident 4-channel `PolyADAA` activation step with
    /// zero-DC linear bias removal (`- alpha_c * b_conv_c`), returning `(y_zero_dc, |u_curr|)` directly
    /// in SIMD registers without intermediate stack stores/loads.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn poly_adaa_step_vec4(
        vx: Self::Vec4,
        b_conv: &[f32; 4],
        alpha: &[f32; 4],
        beta: &[f32; 4],
        delta: &[f32; 4],
        r_ac: f32,
        u_prev: &mut [f32; 4],
        a_prev: &mut [f32; 4],
        m_dc: &mut [f32; 4],
    ) -> (Self::Vec4, Self::Vec4) {
        let vb_conv = Self::Vec4::load(b_conv);
        let valpha = Self::Vec4::load(alpha);
        let vbeta = Self::Vec4::load(beta);
        let vdelta = Self::Vec4::load(delta);
        let vu_prev = Self::Vec4::load(u_prev);
        let va_prev = Self::Vec4::load(a_prev);
        let vm_dc = Self::Vec4::load(m_dc);

        let vu_curr = vx
            .mul(Self::Vec4::splat(2.0 / 3.0))
            .clamp(Self::Vec4::splat(-1.0), Self::Vec4::splat(1.0));

        let va_curr = Self::poly_adaa_algebraic_4ch(vu_prev, vu_curr, vbeta, vdelta);

        let vy_nl = va_curr
            .mul(Self::Vec4::splat(0.875))
            .fma(Self::Vec4::splat(0.125), va_prev);

        let vr_ac = Self::Vec4::splat(r_ac);
        let vone_minus_rac = Self::Vec4::splat(1.0 - r_ac);
        let vm_dc_new = vm_dc.mul(vr_ac).fma(vone_minus_rac, vy_nl);

        let vy = vy_nl.sub(vm_dc_new).fma(valpha, vx).fms(valpha, vb_conv);

        vu_curr.store(u_prev);
        va_curr.store(a_prev);
        vm_dc_new.store(m_dc);
        (vy, vu_curr.abs())
    }

    /// Core 1A: 4x4 matrix-vector FMA tile `acc + sum_{c=0..3} x[c] * mat[c]`.
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Self::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Self::Vec4 {
        let mut out = acc;
        out = out.fma(Self::Vec4::splat(x[0]), Self::Vec4::load(&mat[0]));
        out = out.fma(Self::Vec4::splat(x[1]), Self::Vec4::load(&mat[1]));
        out = out.fma(Self::Vec4::splat(x[2]), Self::Vec4::load(&mat[2]));
        out = out.fma(Self::Vec4::splat(x[3]), Self::Vec4::load(&mat[3]));
        out
    }

    /// Core 1A (`Lite`, Tier 0): 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles)
    /// using dual accumulators (`acc0`, `acc1`) to hide FMA pipeline latency.
    #[inline(always)]
    fn conv6_taps_1group(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 6],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> Self::Vec4 {
        let mut acc0 = bias;
        let mut acc1 = Self::Vec4::zero();
        acc0 = Self::fma_mat4x4_vec4(acc0, &w_tiles[0], &h_taps[0][0]);
        acc1 = Self::fma_mat4x4_vec4(acc1, &w_tiles[1], &h_taps[1][0]);
        acc0 = Self::fma_mat4x4_vec4(acc0, &w_tiles[2], &h_taps[2][0]);
        acc1 = Self::fma_mat4x4_vec4(acc1, &w_tiles[3], &h_taps[3][0]);
        acc0 = Self::fma_mat4x4_vec4(acc0, &w_tiles[4], &h_taps[4][0]);
        acc1 = Self::fma_mat4x4_vec4(acc1, &w_tiles[5], &h_taps[5][0]);
        acc0.add(acc1)
    }

    /// Core 1A (`Full`, Tier 1): 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles)
    /// using quad accumulators (`acc0..acc3`) to saturate dual/quad SIMD FMA pipelines.
    #[inline(always)]
    fn conv6_taps_2groups(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 12],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> Self::Vec4 {
        let mut acc0 = bias;
        let mut acc1 = Self::Vec4::zero();
        let mut acc2 = Self::Vec4::zero();
        let mut acc3 = Self::Vec4::zero();
        for pair in 0..3 {
            let t_even = pair * 2;
            let t_odd = pair * 2 + 1;
            let h_e = h_taps[t_even];
            let h_o = h_taps[t_odd];
            acc0 = Self::fma_mat4x4_vec4(acc0, &w_tiles[t_even * 2], &h_e[0]);
            acc1 = Self::fma_mat4x4_vec4(acc1, &w_tiles[t_even * 2 + 1], &h_e[1]);
            acc2 = Self::fma_mat4x4_vec4(acc2, &w_tiles[t_odd * 2], &h_o[0]);
            acc3 = Self::fma_mat4x4_vec4(acc3, &w_tiles[t_odd * 2 + 1], &h_o[1]);
        }
        acc0.add(acc1).add(acc2.add(acc3))
    }

    /// Core 1A (`Ultra`, Tier 2): 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles)
    /// using 3-way group accumulators (`acc0..acc2`) to break FMA dependency chains.
    #[inline(always)]
    fn conv6_taps_3groups(
        bias: Self::Vec4,
        w_tiles: &[[[f32; 4]; 4]; 18],
        h_taps: &[&[[f32; 4]; 3]; 6],
    ) -> Self::Vec4 {
        let mut acc0 = bias;
        let mut acc1 = Self::Vec4::zero();
        let mut acc2 = Self::Vec4::zero();
        for tap in 0..6 {
            let ht = h_taps[tap];
            let base = tap * 3;
            acc0 = Self::fma_mat4x4_vec4(acc0, &w_tiles[base], &ht[0]);
            acc1 = Self::fma_mat4x4_vec4(acc1, &w_tiles[base + 1], &ht[1]);
            acc2 = Self::fma_mat4x4_vec4(acc2, &w_tiles[base + 2], &ht[2]);
        }
        acc0.add(acc1).add(acc2)
    }

    /// Core 1A: 6-tap dilated causal convolution tile for `Lite` (`4 out × 4 in`).
    /// `weights` is stored in `[tap: 6][in_lane: 4][out_lane: 4]` SIMD column-major order so
    /// each input sample broadcasts across 4 output lanes via a single vector FMA.
    #[inline(always)]
    fn conv6_tile_4x4(
        taps: &[[f32; 4]; 6],
        weights: &[[[f32; 4]; 4]; 6],
        bias: &[f32; 4],
        out: &mut [f32; 4],
    ) {
        let mut acc = Self::Vec4::load(bias);
        for tap_idx in 0..6 {
            let x_tap = &taps[tap_idx];
            let w_tap = &weights[tap_idx];
            for in_ch in 0..4 {
                let vx = Self::Vec4::splat(x_tap[in_ch]);
                let vw = Self::Vec4::load(&w_tap[in_ch]);
                acc = acc.fma(vx, vw);
            }
        }
        acc.store(out);
    }

    /// Core 1A: 6-tap dilated causal convolution tile for `Full Delta` (`4 out × 8 in`).
    #[inline(always)]
    fn conv6_tile_4x8(
        taps_0: &[[f32; 4]; 6],
        taps_1: &[[f32; 4]; 6],
        weights: &[[[f32; 4]; 8]; 6],
        bias: &[f32; 4],
        out: &mut [f32; 4],
    ) {
        let mut acc = Self::Vec4::load(bias);
        for tap_idx in 0..6 {
            let x0 = &taps_0[tap_idx];
            let x1 = &taps_1[tap_idx];
            let w_tap = &weights[tap_idx];
            for in_ch in 0..4 {
                acc = acc.fma(
                    Self::Vec4::splat(x0[in_ch]),
                    Self::Vec4::load(&w_tap[in_ch]),
                );
            }
            for in_ch in 0..4 {
                acc = acc.fma(
                    Self::Vec4::splat(x1[in_ch]),
                    Self::Vec4::load(&w_tap[4 + in_ch]),
                );
            }
        }
        acc.store(out);
    }

    /// Core 1A: 6-tap dilated causal convolution tile for `Ultra Delta` (`4 out × 12 in`).
    #[inline(always)]
    fn conv6_tile_4x12(
        taps_0: &[[f32; 4]; 6],
        taps_1: &[[f32; 4]; 6],
        taps_2: &[[f32; 4]; 6],
        weights: &[[[f32; 4]; 12]; 6],
        bias: &[f32; 4],
        out: &mut [f32; 4],
    ) {
        let mut acc = Self::Vec4::load(bias);
        for tap_idx in 0..6 {
            let x0 = &taps_0[tap_idx];
            let x1 = &taps_1[tap_idx];
            let x2 = &taps_2[tap_idx];
            let w_tap = &weights[tap_idx];
            for in_ch in 0..4 {
                acc = acc.fma(
                    Self::Vec4::splat(x0[in_ch]),
                    Self::Vec4::load(&w_tap[in_ch]),
                );
            }
            for in_ch in 0..4 {
                acc = acc.fma(
                    Self::Vec4::splat(x1[in_ch]),
                    Self::Vec4::load(&w_tap[4 + in_ch]),
                );
            }
            for in_ch in 0..4 {
                acc = acc.fma(
                    Self::Vec4::splat(x2[in_ch]),
                    Self::Vec4::load(&w_tap[8 + in_ch]),
                );
            }
        }
        acc.store(out);
    }

    /// Core 1B (`CL-03`): Branchless $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor
    /// across 4 channels (`0` `libm`, `0` `FSQRT`, `0` `FDIV`, $|u_{\text{rot}}| \le 1.0$).
    #[inline(always)]
    fn algebraic_rotor_4ch(theta_rad: &[f32; 4], cos_out: &mut [f32; 4], sin_out: &mut [f32; 4]) {
        let vtheta = Self::Vec4::load(theta_rad).clamp(
            Self::Vec4::splat(1.0e-4),
            Self::Vec4::splat(core::f32::consts::PI - 1.0e-4),
        );
        let phi = vtheta.sub(Self::Vec4::splat(core::f32::consts::FRAC_PI_2));
        let p2 = phi.mul(phi);

        // 4-FMA minimax polynomial for sin(phi)
        let s3 = Self::Vec4::splat(0.008_332_9).fms(p2, Self::Vec4::splat(0.000_195_07));
        let s2 = Self::Vec4::splat(0.166_666_5).fms(p2, s3);
        let s1 = Self::Vec4::splat(0.999_999_95).fms(p2, s2);
        let sin_phi = phi.mul(s1);

        // 4-FMA minimax polynomial for cos(phi)
        let c4 = Self::Vec4::splat(0.001_388_6).fms(p2, Self::Vec4::splat(0.000_024_38));
        let c3 = Self::Vec4::splat(0.041_666_3).fms(p2, c4);
        let c2 = Self::Vec4::splat(0.499_999_9).fms(p2, c3);
        let cos_phi = Self::Vec4::splat(0.999_999_95).fms(p2, c2);

        // (c_hat, s_hat) = (-sin_phi, cos_phi)
        let c_hat = Self::Vec4::splat(0.0).sub(sin_phi);
        let s_hat = cos_phi;

        // Algebraic Newton projection: y1 = 1.5 - 0.5 * (c_hat^2 + s_hat^2)
        let norm_sq = c_hat.mul(c_hat).fma(s_hat, s_hat);
        let y1 = Self::Vec4::splat(1.5).fms(Self::Vec4::splat(0.5), norm_sq);

        c_hat.mul(y1).store(cos_out);
        s_hat.mul(y1).store(sin_out);
    }

    /// Core 1B (`CL-04`, `CL-05`): $48\text{ kHz}$ Radial-Polar Selective Complex RG-LRU step
    /// across 4 channels.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn rglru_step_4ch(
        x: &[f32; 4],
        r_low: &[f32; 4],
        dr_span: &[f32; 4],
        b_gamma: &[f32; 4],
        w_sel: &[f32; 4],
        c_real: &[f32; 4],
        c_imag: &[f32; 4],
        u_rot_cos: &[f32; 4],
        u_rot_sin: &[f32; 4],
        z_re: &mut [f32; 4],
        z_im: &mut [f32; 4],
        y_out: &mut [f32; 4],
    ) {
        let vx = Self::Vec4::load(x);
        let vr_low = Self::Vec4::load(r_low);
        let vdr = Self::Vec4::load(dr_span);
        let vb_gamma = Self::Vec4::load(b_gamma);
        let vw_sel = Self::Vec4::load(w_sel);
        let vc_re = Self::Vec4::load(c_real);
        let vc_im = Self::Vec4::load(c_imag);
        let vcos = Self::Vec4::load(u_rot_cos);
        let vsin = Self::Vec4::load(u_rot_sin);
        let vz_re = Self::Vec4::load(z_re);
        let vz_im = Self::Vec4::load(z_im);

        // CL-05: g_c[n] = clamp(0.5 + w_sel * x_c[n], 0.0, 1.0)
        let vg = Self::Vec4::splat(0.5)
            .fma(vw_sel, vx)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(1.0));

        // r_c[n] = r_low + g_c[n] * (r_high - r_low)
        let vr = vr_low.fma(vg, vdr);

        // Complex pole a_c[n] = r_c[n] * (cos + i * sin)
        let va_re = vr.mul(vcos);
        let va_im = vr.mul(vsin);

        // Real input injection: b_tilde = g_c[n] * b_gamma * x_c[n]
        let vb_in = vg.mul(vb_gamma).mul(vx);

        // Complex recurrence: z[n] = a[n] * z[n-1] + b_tilde[n]
        let vz_re_new = vb_in.fma(va_re, vz_re).fms(va_im, vz_im);
        let vz_im_new = va_re.mul(vz_im).fma(va_im, vz_re);

        // Real projection + residual: y_c[n] = x_c[n] + Re(c_c^* z_c[n]) = x + c_re * z_re + c_im * z_im
        let vy = vx.fma(vc_re, vz_re_new).fma(vc_im, vz_im_new);

        vz_re_new.store(z_re);
        vz_im_new.store(z_im);
        vy.store(y_out);
    }

    /// Control Bus (`3 kHz`): Collapses the SVD stereo output head basis
    /// $\mathbf{W}_{\text{head}}(\mathbf{u}, \mathbf{s}) = \mathbf{W}_{\text{head},0} + \sum_{k=1}^{D_h} z_{\text{head},k} \mathbf{W}_{\text{head},k}$.
    /// Length of `base` and `out_collapsed` is a multiple of 4 (`128` floats per 4-channel slice).
    #[inline]
    fn collapse_svd_head_slice(
        base: &[f32],
        svd_modes: &[f32],
        z_head: &[f32; 4],
        active_rank: usize,
        out_collapsed: &mut [f32],
    ) {
        let len = base.len();
        debug_assert_eq!(out_collapsed.len(), len);
        debug_assert!(len.is_multiple_of(4));
        let rank = active_rank.min(4);
        let chunks = len / 4;

        for c in 0..chunks {
            let offset = c * 4;
            let mut b_arr = [0.0f32; 4];
            b_arr.copy_from_slice(&base[offset..offset + 4]);
            let mut acc = Self::Vec4::load(&b_arr);
            for (k, &zk) in z_head.iter().enumerate().take(rank) {
                let m_offset = k * len + offset;
                let mut m_arr = [0.0f32; 4];
                m_arr.copy_from_slice(&svd_modes[m_offset..m_offset + 4]);
                acc = acc.fma(Self::Vec4::splat(zk), Self::Vec4::load(&m_arr));
            }
            let mut out_arr = [0.0f32; 4];
            acc.store(&mut out_arr);
            out_collapsed[offset..offset + 4].copy_from_slice(&out_arr);
        }
    }

    /// Core 1 Output Head (`48 kHz`): Evaluates the 16-tap stereo FIR dot product for a 4-channel
    /// tier slice (`16 taps × 4 channels = 64 MACs` per output channel).
    #[inline(always)]
    fn stereo_head_dot_16x4(
        taps: &[[f32; 4]; 16],
        w_l: &[[f32; 4]; 16],
        w_r: &[[f32; 4]; 16],
    ) -> (f32, f32) {
        let mut acc_l = Self::Vec4::splat(0.0);
        let mut acc_r = Self::Vec4::splat(0.0);
        for t in 0..16 {
            let vt = Self::Vec4::load(&taps[t]);
            acc_l = acc_l.fma(vt, Self::Vec4::load(&w_l[t]));
            acc_r = acc_r.fma(vt, Self::Vec4::load(&w_r[t]));
        }
        (acc_l.horizontal_sum(), acc_r.horizontal_sum())
    }

    /// Core 1 Output Head (`48 kHz`, Mono fast-path when `STEREO_OUT == 0`): Evaluates only the
    /// left/mono channel of the 16-tap head, saving 50% of head MACs.
    #[inline(always)]
    fn mono_head_dot_16x4(taps: &[[f32; 4]; 16], w_l: &[[f32; 4]; 16]) -> f32 {
        let mut acc_l = Self::Vec4::splat(0.0);
        for t in 0..16 {
            let vt = Self::Vec4::load(&taps[t]);
            acc_l = acc_l.fma(vt, Self::Vec4::load(&w_l[t]));
        }
        acc_l.horizontal_sum()
    }

    /// Core 2 (`CL-06`): Stateless 4-tap Cubic Lagrange fractional delay interpolation across
    /// 4 delay lines simultaneously (`0 FDIV`).
    #[inline(always)]
    fn cubic_lagrange_4ch(
        mu: &[f32; 4],
        s_m1: &[f32; 4],
        s_0: &[f32; 4],
        s_1: &[f32; 4],
        s_2: &[f32; 4],
        out: &mut [f32; 4],
    ) {
        let vmu = Self::Vec4::load(mu);
        let one = Self::Vec4::splat(1.0);
        let two = Self::Vec4::splat(2.0);
        let inv_six = Self::Vec4::splat(1.0 / 6.0);
        let neg_inv_six = Self::Vec4::splat(-1.0 / 6.0);
        let half = Self::Vec4::splat(0.5);
        let neg_half = Self::Vec4::splat(-0.5);

        let mu_p1 = vmu.add(one);
        let mu_m1 = vmu.sub(one);
        let mu_m2 = vmu.sub(two);

        let c_m1 = vmu.mul(mu_m1).mul(mu_m2).mul(neg_inv_six);
        let c_0 = mu_p1.mul(mu_m1).mul(mu_m2).mul(half);
        let c_1 = mu_p1.mul(vmu).mul(mu_m2).mul(neg_half);
        let c_2 = mu_p1.mul(vmu).mul(mu_m1).mul(inv_six);

        let d = c_m1
            .mul(Self::Vec4::load(s_m1))
            .fma(c_0, Self::Vec4::load(s_0))
            .fma(c_1, Self::Vec4::load(s_1))
            .fma(c_2, Self::Vec4::load(s_2));
        d.store(out);
    }

    /// Core 2: $O(M)$ Givens stereo rotation $\mathbf{R}(\psi)$ + Householder reflection
    /// $\mathbf{H}(\mathbf{v}) = \mathbf{I} - 2\mathbf{v}\mathbf{v}^\top$ for `Lite` ($M = 4$).
    #[inline(always)]
    fn givens_householder_mix_4ch(
        p: &[f32; 4],
        cos_psi: f32,
        sin_psi: f32,
        v: &[f32; 4],
        q_out: &mut [f32; 4],
    ) {
        let r = [
            cos_psi * p[0] - sin_psi * p[1],
            sin_psi * p[0] + cos_psi * p[1],
            cos_psi * p[2] - sin_psi * p[3],
            sin_psi * p[2] + cos_psi * p[3],
        ];
        let vr = Self::Vec4::load(&r);
        let vv = Self::Vec4::load(v);
        let two_dot = Self::Vec4::splat(2.0 * vr.dot(vv));
        vr.fms(two_dot, vv).store(q_out);
    }

    /// Core 2: $O(M)$ Givens stereo rotation $\mathbf{R}(\psi)$ + Householder reflection
    /// $\mathbf{H}(\mathbf{v}) = \mathbf{I} - 2\mathbf{v}\mathbf{v}^\top$ for `Full`/`Ultra` ($M = 8$).
    #[inline(always)]
    fn givens_householder_mix_8ch(
        p: &[f32; 8],
        cos_psi: f32,
        sin_psi: f32,
        v: &[f32; 8],
        q_out: &mut [f32; 8],
    ) {
        let r_lo = [
            cos_psi * p[0] - sin_psi * p[1],
            sin_psi * p[0] + cos_psi * p[1],
            cos_psi * p[2] - sin_psi * p[3],
            sin_psi * p[2] + cos_psi * p[3],
        ];
        let r_hi = [
            cos_psi * p[4] - sin_psi * p[5],
            sin_psi * p[4] + cos_psi * p[5],
            cos_psi * p[6] - sin_psi * p[7],
            sin_psi * p[6] + cos_psi * p[7],
        ];
        let mut v_lo_arr = [0.0f32; 4];
        let mut v_hi_arr = [0.0f32; 4];
        v_lo_arr.copy_from_slice(&v[0..4]);
        v_hi_arr.copy_from_slice(&v[4..8]);

        let vr_lo = Self::Vec4::load(&r_lo);
        let vr_hi = Self::Vec4::load(&r_hi);
        let vv_lo = Self::Vec4::load(&v_lo_arr);
        let vv_hi = Self::Vec4::load(&v_hi_arr);

        let dot = vr_lo.mul(vv_lo).fma(vr_hi, vv_hi).horizontal_sum();
        let two_dot = Self::Vec4::splat(2.0 * dot);

        let mut q_lo = [0.0f32; 4];
        let mut q_hi = [0.0f32; 4];
        vr_lo.fms(two_dot, vv_lo).store(&mut q_lo);
        vr_hi.fms(two_dot, vv_hi).store(&mut q_hi);
        q_out[0..4].copy_from_slice(&q_lo);
        q_out[4..8].copy_from_slice(&q_hi);
    }

    /// Core 2 (`CL-10`, `CL-11`): Unity-Slope Normalized Cubic Coercive-Saturating Lyapunov Loop Law
    /// across 4 delay lines (`0 FDIV`).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn lyapunov_loop_4ch(
        q_ac: &[f32; 4],
        input_drive: &[f32; 4],
        g_lin: &[f32; 4],
        g_regen: &[f32; 4],
        inv_sigma_knee: &[f32; 4],
        two_over_3v_ceil: &[f32; 4],
        v_ceil: &[f32; 4],
        s_out: &mut [f32; 4],
    ) {
        let vq = Self::Vec4::load(q_ac);
        let vin = Self::Vec4::load(input_drive);
        let vg_lin =
            Self::Vec4::load(g_lin).clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(0.9990));
        let vg_regen = Self::Vec4::load(g_regen);
        let vinv_sigma = Self::Vec4::load(inv_sigma_knee);
        let vscale_sat = Self::Vec4::load(two_over_3v_ceil);
        let vv_ceil =
            Self::Vec4::load(v_ceil).clamp(Self::Vec4::splat(0.01), Self::Vec4::splat(0.9850));

        // CL-10: v_knee = clamp(|q_ac| * inv_sigma_knee, 0.0, 1.0)
        let v_knee = vq
            .abs()
            .mul(vinv_sigma)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(1.0));
        let smoothstep = v_knee
            .mul(v_knee)
            .mul(Self::Vec4::splat(3.0).fms(Self::Vec4::splat(2.0), v_knee));
        let g_eff = vg_lin.fma(vg_regen, smoothstep);

        // u_loop = g_eff * q_ac + input_drive
        let u_loop = vin.fma(g_eff, vq);

        // CL-11: v_sat = clamp((2 / (3 V_ceil)) * u_loop, -1.0, +1.0)
        let v_sat = u_loop
            .mul(vscale_sat)
            .clamp(Self::Vec4::splat(-1.0), Self::Vec4::splat(1.0));
        let v_sat2 = v_sat.mul(v_sat);
        let s_new = vv_ceil
            .mul(v_sat)
            .mul(Self::Vec4::splat(1.5).fms(Self::Vec4::splat(0.5), v_sat2));
        s_new.store(s_out);
    }

    /// Converts 4 `f32` samples to IEEE-754 half-precision `fp16` (`u16`).
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        [
            f32_to_fp16(src[0]),
            f32_to_fp16(src[1]),
            f32_to_fp16(src[2]),
            f32_to_fp16(src[3]),
        ]
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) samples to `f32`.
    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        [
            fp16_to_f32(src[0]),
            fp16_to_f32(src[1]),
            fp16_to_f32(src[2]),
            fp16_to_f32(src[3]),
        ]
    }

    /// Computes Castagnoli `CRC32C` (`0x1EDC6F41` / reflected `0x82F63B78`) over `data`.
    #[inline]
    fn crc32c(data: &[u8]) -> u32 {
        crc32c_slice(0, data)
    }

    /// Evaluates the division-free 4-FMA + 1-Step Newton unit-circle phasor $(\cos\theta, \sin\theta)$ (`CL-03`).
    #[inline(always)]
    fn fast_unit_phasor(theta_rad: f32) -> (f32, f32) {
        algebraic_unit_rotor(theta_rad)
    }

    /// Evaluates the 16-tap 4-channel (`64` `f32`) dot product for the collapsed output head.
    #[inline(always)]
    fn dot_product_16taps_4ch(taps_64: &[f32; 64], weights_64: &[f32; 64]) -> f32 {
        let (t_chunks, _) = taps_64.as_chunks::<4>();
        let (w_chunks, _) = weights_64.as_chunks::<4>();
        let mut acc0 = Self::Vec4::zero();
        let mut acc1 = Self::Vec4::zero();
        let mut acc2 = Self::Vec4::zero();
        let mut acc3 = Self::Vec4::zero();
        let mut c = 0;
        while c < 16 {
            acc0 = acc0.fma(
                Self::Vec4::load(&t_chunks[c]),
                Self::Vec4::load(&w_chunks[c]),
            );
            acc1 = acc1.fma(
                Self::Vec4::load(&t_chunks[c + 1]),
                Self::Vec4::load(&w_chunks[c + 1]),
            );
            acc2 = acc2.fma(
                Self::Vec4::load(&t_chunks[c + 2]),
                Self::Vec4::load(&w_chunks[c + 2]),
            );
            acc3 = acc3.fma(
                Self::Vec4::load(&t_chunks[c + 3]),
                Self::Vec4::load(&w_chunks[c + 3]),
            );
            c += 4;
        }
        acc0.add(acc1).add(acc2.add(acc3)).horizontal_sum()
    }

    /// Evaluates a fused stereo (`L` and `R`) 16-tap 4-channel (`64` `f32`) dot product in a single
    /// pass over `taps_64`, halving feature-buffer memory loads.
    #[inline(always)]
    fn dot_product_16taps_4ch_stereo(
        taps_64: &[f32; 64],
        weights_l: &[f32; 64],
        weights_r: &[f32; 64],
    ) -> [f32; 2] {
        let (t_chunks, _) = taps_64.as_chunks::<4>();
        let (wl_chunks, _) = weights_l.as_chunks::<4>();
        let (wr_chunks, _) = weights_r.as_chunks::<4>();
        let mut l0 = Self::Vec4::zero();
        let mut l1 = Self::Vec4::zero();
        let mut l2 = Self::Vec4::zero();
        let mut l3 = Self::Vec4::zero();
        let mut r0 = Self::Vec4::zero();
        let mut r1 = Self::Vec4::zero();
        let mut r2 = Self::Vec4::zero();
        let mut r3 = Self::Vec4::zero();
        let mut c = 0;
        while c < 16 {
            let vt0 = Self::Vec4::load(&t_chunks[c]);
            l0 = l0.fma(vt0, Self::Vec4::load(&wl_chunks[c]));
            r0 = r0.fma(vt0, Self::Vec4::load(&wr_chunks[c]));

            let vt1 = Self::Vec4::load(&t_chunks[c + 1]);
            l1 = l1.fma(vt1, Self::Vec4::load(&wl_chunks[c + 1]));
            r1 = r1.fma(vt1, Self::Vec4::load(&wr_chunks[c + 1]));

            let vt2 = Self::Vec4::load(&t_chunks[c + 2]);
            l2 = l2.fma(vt2, Self::Vec4::load(&wl_chunks[c + 2]));
            r2 = r2.fma(vt2, Self::Vec4::load(&wr_chunks[c + 2]));

            let vt3 = Self::Vec4::load(&t_chunks[c + 3]);
            l3 = l3.fma(vt3, Self::Vec4::load(&wl_chunks[c + 3]));
            r3 = r3.fma(vt3, Self::Vec4::load(&wr_chunks[c + 3]));
            c += 4;
        }
        [
            l0.add(l1).add(l2.add(l3)).horizontal_sum(),
            r0.add(r1).add(r2.add(r3)).horizontal_sum(),
        ]
    }

    /// Evaluates the 4-channel Radial-Polar `SelectiveRgLru` layer (`CL-03`..`CL-05`).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn selective_rglru_layer_4ch(
        x: &[f32; 4],
        r_low: &[f32; 4],
        dr_span: &[f32; 4],
        w_sel: &[f32; 4],
        b_gamma_norm: &[f32; 4],
        c_real: &[f32; 4],
        c_imag: &[f32; 4],
        u_rot_cos: &[f32; 4],
        u_rot_sin: &[f32; 4],
        z_re: &mut [f32; 4],
        z_im: &mut [f32; 4],
        y_out: &mut [f32; 4],
    ) {
        let vx = Self::Vec4::load(x);
        let vy = Self::selective_rglru_step_vec4(
            vx,
            r_low,
            dr_span,
            w_sel,
            b_gamma_norm,
            c_real,
            c_imag,
            u_rot_cos,
            u_rot_sin,
            z_re,
            z_im,
        );
        vy.store(y_out);
    }

    /// Register-resident 4-channel Radial-Polar `SelectiveRgLru` step (`CL-03`..`CL-05`),
    /// returning `y_out` directly in a SIMD register.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn selective_rglru_step_vec4(
        vx: Self::Vec4,
        r_low: &[f32; 4],
        dr_span: &[f32; 4],
        w_sel: &[f32; 4],
        b_gamma_norm: &[f32; 4],
        c_real: &[f32; 4],
        c_imag: &[f32; 4],
        u_rot_cos: &[f32; 4],
        u_rot_sin: &[f32; 4],
        z_re: &mut [f32; 4],
        z_im: &mut [f32; 4],
    ) -> Self::Vec4 {
        let vr_low = Self::Vec4::load(r_low);
        let vdr = Self::Vec4::load(dr_span);
        let vw_sel = Self::Vec4::load(w_sel);
        let vb_gamma = Self::Vec4::load(b_gamma_norm);
        let vc_re = Self::Vec4::load(c_real);
        let vc_im = Self::Vec4::load(c_imag);
        let vcos = Self::Vec4::load(u_rot_cos);
        let vsin = Self::Vec4::load(u_rot_sin);
        let vz_re = Self::Vec4::load(z_re);
        let vz_im = Self::Vec4::load(z_im);

        // Selectivity gate: q = min(|x| * w_sel, 1.0), g_sel = q^2 (3 - 2q)
        let vq = vx
            .abs()
            .mul(vw_sel)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(1.0));
        let vg_sel = vq
            .mul(vq)
            .mul(Self::Vec4::splat(3.0).fms(Self::Vec4::splat(2.0), vq));
        let vr = vr_low
            .fma(vdr, Self::Vec4::splat(1.0).sub(vg_sel))
            .clamp(Self::Vec4::splat(0.25), Self::Vec4::splat(0.9995));

        let va_re = vr.mul(vcos);
        let va_im = vr.mul(vsin);
        let vb_in = vb_gamma.mul(vx);

        let vz_re_new = vb_in.fma(va_re, vz_re).fms(va_im, vz_im);
        let vz_im_new = va_re.mul(vz_im).fma(va_im, vz_re);
        let vy = vc_re.mul(vz_re_new).fms(vc_im, vz_im_new);

        vz_re_new.store(z_re);
        vz_im_new.store(z_im);
        vy
    }

    /// Evaluates the 4-tap Cubic Lagrange basis weights $(w_{-1}, w_0, w_{+1}, w_{+2})$ across 4 lanes (`CL-06`).
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4, Self::Vec4, Self::Vec4) {
        let one = Self::Vec4::splat(1.0);
        let two = Self::Vec4::splat(2.0);
        let inv_six = Self::Vec4::splat(1.0 / 6.0);
        let neg_inv_six = Self::Vec4::splat(-1.0 / 6.0);
        let half = Self::Vec4::splat(0.5);
        let neg_half = Self::Vec4::splat(-0.5);

        let mu_p1 = vmu.add(one);
        let mu_m1 = vmu.sub(one);
        let mu_m2 = vmu.sub(two);

        let c_m1 = vmu.mul(mu_m1).mul(mu_m2).mul(neg_inv_six);
        let c_0 = mu_p1.mul(mu_m1).mul(mu_m2).mul(half);
        let c_1 = mu_p1.mul(vmu).mul(mu_m2).mul(neg_half);
        let c_2 = mu_p1.mul(vmu).mul(mu_m1).mul(inv_six);
        (c_m1, c_0, c_1, c_2)
    }

    /// Evaluates the Core 2 $5\text{ Hz}$ gain-normalized DC blocker (`CL-09`) and Coercive-Saturating
    /// Lyapunov loop (`CL-10`, `CL-11`) across 4 FDN lines with `0 FDIV`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn mufdn_coercive_loop_4ch(
        u_in: &[f32; 4],
        g_lin: &[f32; 4],
        inv_sigma_knee: &[f32; 4],
        two_over_3v_ceil: &[f32; 4],
        v_ceil: &[f32; 4],
        r_dc: f32,
        dc_x_prev: &mut [f32; 4],
        dc_y_prev: &mut [f32; 4],
        s_out: &mut [f32; 4],
    ) {
        let vu = Self::Vec4::load(u_in);
        let vx_prev = Self::Vec4::load(dc_x_prev);
        let vy_prev = Self::Vec4::load(dc_y_prev);

        // Gain-normalized 5 Hz DC blocker: alpha_dc = 0.5 * (1 + r_dc)
        let valpha_dc = Self::Vec4::splat(0.5 * (1.0 + r_dc));
        let vr_dc = Self::Vec4::splat(r_dc);
        let vy_dc = valpha_dc.mul(vu.sub(vx_prev)).fma(vr_dc, vy_prev);
        vu.store(dc_x_prev);
        vy_dc.store(dc_y_prev);

        // Coercive gain reduction + cubic soft-ceiling (CL-10, CL-11, 0 FDIV)
        let vg_lin =
            Self::Vec4::load(g_lin).clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(0.9990));
        let vinv_sigma = Self::Vec4::load(inv_sigma_knee);
        let vscale_sat = Self::Vec4::load(two_over_3v_ceil);
        let vv_ceil =
            Self::Vec4::load(v_ceil).clamp(Self::Vec4::splat(0.25), Self::Vec4::splat(2.0));

        let vratio = vy_dc
            .abs()
            .mul(vinv_sigma)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(4.0));
        // Division-free coercive damping factor 1 / (1 + r^2) approximated on [0, 4] or smoothstep knee
        let r2 = vratio.mul(vratio);
        let damp = Self::Vec4::splat(1.0)
            .fms(Self::Vec4::splat(0.05), r2)
            .clamp(Self::Vec4::splat(0.20), Self::Vec4::splat(1.0));
        let vs_pre = vg_lin.mul(damp).mul(vy_dc);

        let v_sat = vs_pre
            .mul(vscale_sat)
            .clamp(Self::Vec4::splat(-1.0), Self::Vec4::splat(1.0));
        let v_sat2 = v_sat.mul(v_sat);
        let s_new = vv_ceil
            .mul(v_sat)
            .mul(Self::Vec4::splat(1.5).fms(Self::Vec4::splat(0.5), v_sat2));
        s_new.store(s_out);
    }
}

pub use self::{f32_to_fp16 as f32_to_fp16_bits, fp16_to_f32 as fp16_bits_to_f32};

// ============================================================================
// Standalone Reference DSP & High-Precision Certification Oracle Functions
// ============================================================================

/// Clamps pre-activation input $u_n = \operatorname{clamp}(\frac{2}{3}x, -1.0, +1.0)$ (`CL-01`).
#[inline(always)]
pub fn clamp_pre_activation(x: f32) -> f32 {
    ((2.0 / 3.0) * x).clamp(-1.0, 1.0)
}

/// Evaluates the instantaneous `PolyADAA` waveshaper $g_{\text{nl}}(u) = \beta(1.5u - 0.5u^3) + \delta(u^2 - u^4)$
/// used to initialize the exact 3-state DC resting equilibrium.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper(u: f32, beta: f32, delta: f32) -> f32 {
    let u2 = u * u;
    beta * u * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2)
}

/// High-precision `f64` oracle for $g_{\text{nl}}(u) = \beta(1.5u - 0.5u^3) + \delta(u^2 - u^4)$.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    beta * u * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2)
}

/// Evaluates the analytical antiderivative $F(u) = \beta(\frac{3}{4}u^2 - \frac{1}{8}u^4) + \delta(\frac{1}{3}u^3 - \frac{1}{5}u^5)$ in `f64`.
#[inline(always)]
pub fn eval_poly_adaa_antiderivative_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u2 * u2;
    let u5 = u4 * u;
    beta * (0.75 * u2 - 0.125 * u4) + delta * (u3 / 3.0 - 0.2 * u5)
}

/// Exact division-free algebraic `PolyADAA` kernel in `f32` (`CL-01`, `0 FDIV`, `0 FSQRT`).
#[inline(always)]
pub fn eval_poly_adaa_algebraic(u_prev: f32, u_curr: f32, beta: f32, delta: f32) -> f32 {
    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd_term = beta * m * (1.5 - 0.5 * (m2 + d2));
    let even_term = delta * ((m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2));
    odd_term + even_term
}

/// High-precision `f64` oracle for the exact division-free algebraic `PolyADAA` kernel (`Test 1B`).
#[inline(always)]
pub fn eval_poly_adaa_algebraic_f64(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd_term = beta * m * (1.5 - 0.5 * (m2 + d2));
    let even_term = delta * ((m2 + (1.0 / 3.0) * d2) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2));
    odd_term + even_term
}

/// Branchless $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor in `f32` (`CL-03`).
/// Returns `(cos_theta, sin_theta)` with $|u_{\text{rot}}| \le 1.0$.
#[inline(always)]
pub fn algebraic_unit_rotor(theta_rad: f32) -> (f32, f32) {
    let theta = theta_rad.clamp(1.0e-4, core::f32::consts::PI - 1.0e-4);
    let phi = theta - core::f32::consts::FRAC_PI_2;
    let p2 = phi * phi;
    let sin_phi =
        phi * (0.999_999_95 - p2 * (0.166_666_5 - p2 * (0.008_332_9 - p2 * 0.000_195_07)));
    let cos_phi = 0.999_999_95
        - p2 * (0.499_999_9 - p2 * (0.041_666_3 - p2 * (0.001_388_6 - p2 * 0.000_024_38)));
    let c_hat = -sin_phi;
    let s_hat = cos_phi;
    let y1 = 1.5 - 0.5 * (c_hat * c_hat + s_hat * s_hat);
    (y1 * c_hat, y1 * s_hat)
}

/// High-precision `f64` oracle for the branchless 4-FMA + Algebraic Newton unit-circle rotor (`Test 2`).
#[inline(always)]
pub fn algebraic_unit_rotor_f64(theta_rad: f64) -> (f64, f64) {
    let theta = theta_rad.clamp(0.0, core::f64::consts::PI);
    let phi = theta - core::f64::consts::FRAC_PI_2;
    let p2 = phi * phi;
    let sin_phi =
        phi * (0.999_999_95 - p2 * (0.166_666_5 - p2 * (0.008_332_9 - p2 * 0.000_195_07)));
    let cos_phi = 0.999_999_95
        - p2 * (0.499_999_9 - p2 * (0.041_666_3 - p2 * (0.001_388_6 - p2 * 0.000_024_38)));
    let c_hat = -sin_phi;
    let s_hat = cos_phi;
    let y1 = 1.5 - 0.5 * (c_hat * c_hat + s_hat * s_hat);
    (y1 * c_hat, y1 * s_hat)
}

/// Stabilizes an autonomous LFO complex phasor on the unit circle via one Algebraic Newton step (`CL-03`).
#[inline(always)]
pub fn stabilize_complex_rotor(re: f32, im: f32) -> (f32, f32) {
    let y1 = 1.5 - 0.5 * (re * re + im * im);
    (y1 * re, y1 * im)
}

/// Evaluates the 4-tap Cubic Lagrange basis weights $[c_{-1}(\mu), c_0(\mu), c_1(\mu), c_2(\mu)]$
/// in `f32` (`CL-06`, `0 FDIV`).
#[inline(always)]
pub fn cubic_lagrange_weights(mu: f32) -> [f32; 4] {
    let c_m1 = -mu * (mu - 1.0) * (mu - 2.0) * (1.0 / 6.0);
    let c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) * 0.5;
    let c_1 = -(mu + 1.0) * mu * (mu - 2.0) * 0.5;
    let c_2 = (mu + 1.0) * mu * (mu - 1.0) * (1.0 / 6.0);
    [c_m1, c_0, c_1, c_2]
}

/// High-precision `f64` oracle for 4-tap Cubic Lagrange basis weights (`Test 3`).
#[inline(always)]
pub fn cubic_lagrange_weights_f64(mu: f64) -> [f64; 4] {
    let c_m1 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0;
    let c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) * 0.5;
    let c_1 = -(mu + 1.0) * mu * (mu - 2.0) * 0.5;
    let c_2 = (mu + 1.0) * mu * (mu - 1.0) / 6.0;
    [c_m1, c_0, c_1, c_2]
}

/// Branchless IEEE-754 `f32` to `fp16` (`u16`) conversion (round-to-nearest-even, flushes subnormals).
#[inline(always)]
pub fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf or NaN
        return sign | 0x7C00 | ((mant != 0) as u16);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Clamp overflow to max finite fp16
        return sign | 0x7BFF;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half_mant = (m >> shift) as u16;
        return sign | half_mant;
    }
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0x0FFF) != 0;
    let mut half_mant = (mant >> 13) as u16;
    if round_bit != 0 && (sticky || (half_mant & 1) != 0) {
        half_mant += 1;
    }
    sign + (((new_exp as u16) << 10) + half_mant)
}

/// Branchless IEEE-754 `fp16` (`u16`) to `f32` conversion.
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal fp16
        let mut m = mant;
        let mut e = 127u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    let f32_exp = (exp + 127 - 15) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

/// Lookup table for Castagnoli `CRC32C` (reflected polynomial `0x82F63B78`).
const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Updates a running Castagnoli `CRC32C` checksum over `data`.
#[inline]
pub fn crc32c_slice(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = !crc_init;
    for &byte in data {
        let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[idx];
    }
    !crc
}

/// Computes Castagnoli `CRC32C` over `data` starting from `0`.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_slice(0, data)
}

// ============================================================================
// Pure `#![no_std]` IEEE-754 Cold-Initialization Math Helpers
// (Used only during cold setup, ZOH `set_sample_rate`, and UI skew calculation)
// ============================================================================

/// Pure `#![no_std]` floor for `f32`.
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` floor for `f64`.
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    let i = x as i64;
    let f = i as f64;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` round-half-away-from-zero for `f32`.
#[inline(always)]
pub fn round_f32(x: f32) -> f32 {
    if x >= 0.0 {
        floor_f32(x + 0.5)
    } else {
        -floor_f32(-x + 0.5)
    }
}

/// Pure `#![no_std]` round-half-away-from-zero for `f64`.
#[inline(always)]
pub fn round_f64(x: f64) -> f64 {
    if x >= 0.0 {
        floor_f64(x + 0.5)
    } else {
        -floor_f64(-x + 0.5)
    }
}

/// Pure `#![no_std]` IEEE-754 + 5-step Newton-Raphson square root in `f64` (used only at cold init / ZOH).
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let approx_bits = (bits >> 1) + (0x3FFu64 << 51);
    let mut y = f64::from_bits(approx_bits);
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` via IEEE-754 exponent extraction +
/// Padé/atanh series (accurate to $< 10^{-15}$ across all positive normals; cold init only).
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mut mant = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    let mut k = exp;
    if mant > core::f64::consts::SQRT_2 {
        mant *= 0.5;
        k += 1;
    }
    let u = (mant - 1.0) / (mant + 1.0);
    let u2 = u * u;
    let mut term = u;
    let mut sum = u;
    let mut n = 3;
    while n <= 21 {
        term *= u2;
        sum += term / (n as f64);
        n += 2;
    }
    2.0 * sum + (k as f64) * core::f64::consts::LN_2
}

/// Pure `#![no_std]` exponential $\exp(x)$ in `f64` via range reduction $x = k \ln 2 + r$
/// ($|r| \le \frac{\ln 2}{2}$) and 12-term Taylor polynomial (accurate to $< 10^{-15}$; cold init only).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    if x < -708.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    let k = round_f64(x / core::f64::consts::LN_2) as i64;
    let r = x - (k as f64) * core::f64::consts::LN_2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=14 {
        term *= r / (i as f64);
        sum += term;
    }
    let scale_bits = ((k + 1023) as u64) << 52;
    sum * f64::from_bits(scale_bits)
}

/// Pure `#![no_std]` power function $x^y = \exp(y \ln x)$ in `f64` (cold init / ZOH rescaling only).
#[inline]
pub fn powf_f64(base: f64, exp: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    if (exp - 1.0).abs() < 1.0e-15 {
        return base;
    }
    exp_f64(exp * ln_f64(base))
}

/// Pure `#![no_std]` cosine $\cos(\theta)$ for $\theta \in [0, \pi]$ via our algebraic rotor in `f64`.
#[inline]
pub fn cos_pi_unit_f64(w: f64) -> f64 {
    let theta = (w.clamp(0.0, 1.0)) * core::f64::consts::PI;
    algebraic_unit_rotor_f64(theta).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_roundtrip_and_bounds() {
        for &val in &[0.0f32, -0.0, 1.0, -1.0, 0.5, -0.25, 0.985, -0.985] {
            let half = f32_to_fp16(val);
            let back = fp16_to_f32(half);
            assert!((back - val).abs() < 1.0e-3);
        }
    }

    #[test]
    fn test_crc32c_known_vector() {
        // Standard RFC 3720 / iSCSI CRC32C check value for "123456789" is 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn test_no_std_f64_init_math_accuracy() {
        for &x in &[0.25f64, 0.5, 0.995, 1.0, 2.0, 10.0, 640.0] {
            let s = sqrt_f64(x);
            assert!((s * s - x).abs() < 1.0e-12);
            let l = ln_f64(x);
            let e = exp_f64(l);
            assert!((e - x).abs() < 1.0e-12);
        }
        assert!((powf_f64(0.995, 48000.0 / 96000.0) - sqrt_f64(0.995)).abs() < 1.0e-14);
    }
}
