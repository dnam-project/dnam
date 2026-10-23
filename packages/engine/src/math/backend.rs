//! Hardware abstraction traits (`SimdVec4` and `MathBackend`) for 4-lane SIMD registers
//! and domain-specific Core 1, Core 2, and Control Bus DSP kernels.

use super::{BackendKind, algebraic_unit_rotor, crc32c_slice, f32_to_fp16, fp16_to_f32};

#[cfg(all(test, target_arch = "aarch64"))]
use super::ArmSmeBackend;
#[cfg(test)]
use super::{DefaultMathBackend, ScalarBackend, detect_best_backend};

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
pub trait MathBackend: Copy + Clone + Send + Sync + 'static {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// Core 1A (`CL-01`): Evaluates the exact division-free algebraic `PolyADAA` kernel across 4 channels.
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

        let odd_factor = one_pt_five.fms(half, m2_plus_d2);
        let odd_term = beta.mul(m).mul(odd_factor);

        let quad_part = m2.fma(one_third, d2);
        let m4 = m2.mul(m2);
        let d4 = d2.mul(d2);
        let quart_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
        let even_term = delta.mul(quad_part.sub(quart_part));

        odd_term.add(even_term)
    }

    /// Core 1A (`CL-01`, `CL-02`): Evaluates a full 4-channel `PolyADAA` activation step.
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

        let vy = vy_nl.sub(vm_dc_new).fma(valpha, vx);

        vu_curr.store(u_prev);
        va_curr.store(a_prev);
        vm_dc_new.store(m_dc);
        vy.store(y_out);
    }

    /// Core 1A (`CL-01`, `CL-02`): Register-resident 4-channel `PolyADAA` activation step.
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

    /// Core 1A (`Lite`, Tier 0): 6-tap dilated convolution over 1 input group (`6 × 4×4` tiles).
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

    /// Core 1A (`Full`, Tier 1): 6-tap dilated convolution over 2 input groups (`12 × 4×4` tiles).
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

    /// Core 1A (`Ultra`, Tier 2): 6-tap dilated convolution over 3 input groups (`18 × 4×4` tiles).
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

        let s3 = Self::Vec4::splat(0.008_332_9).fms(p2, Self::Vec4::splat(0.000_195_07));
        let s2 = Self::Vec4::splat(0.166_666_5).fms(p2, s3);
        let s1 = Self::Vec4::splat(0.999_999_95).fms(p2, s2);
        let sin_phi = phi.mul(s1);

        let c4 = Self::Vec4::splat(0.001_388_6).fms(p2, Self::Vec4::splat(0.000_024_38));
        let c3 = Self::Vec4::splat(0.041_666_3).fms(p2, c4);
        let c2 = Self::Vec4::splat(0.499_999_9).fms(p2, c3);
        let cos_phi = Self::Vec4::splat(0.999_999_95).fms(p2, c2);

        let c_hat = Self::Vec4::splat(0.0).sub(sin_phi);
        let s_hat = cos_phi;

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

        let vg = Self::Vec4::splat(0.5)
            .fma(vw_sel, vx)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(1.0));

        let vr = vr_low.fma(vg, vdr);
        let va_re = vr.mul(vcos);
        let va_im = vr.mul(vsin);
        let vb_in = vg.mul(vb_gamma).mul(vx);

        let vz_re_new = vb_in.fma(va_re, vz_re).fms(va_im, vz_im);
        let vz_im_new = va_re.mul(vz_im).fma(va_im, vz_re);
        let vy = vx.fma(vc_re, vz_re_new).fma(vc_im, vz_im_new);

        vz_re_new.store(z_re);
        vz_im_new.store(z_im);
        vy.store(y_out);
    }

    /// Control Bus (`3 kHz`): Collapses the SVD stereo output head basis.
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

    /// Core 1 Output Head (`48 kHz`): Evaluates the 16-tap stereo FIR dot product for a 4-channel tier slice.
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

    /// Core 1 Output Head (`48 kHz`, Mono fast-path): Evaluates only the left/mono channel of the 16-tap head.
    #[inline(always)]
    fn mono_head_dot_16x4(taps: &[[f32; 4]; 16], w_l: &[[f32; 4]; 16]) -> f32 {
        let mut acc_l = Self::Vec4::splat(0.0);
        for t in 0..16 {
            let vt = Self::Vec4::load(&taps[t]);
            acc_l = acc_l.fma(vt, Self::Vec4::load(&w_l[t]));
        }
        acc_l.horizontal_sum()
    }

    /// Core 2 (`CL-06`): Stateless 4-tap Cubic Lagrange fractional delay interpolation across 4 delay lines.
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
        let (c_m1, c_0, c_1, c_2) = Self::cubic_lagrange_weights_4ch(vmu);

        let d = c_m1
            .mul(Self::Vec4::load(s_m1))
            .fma(c_0, Self::Vec4::load(s_0))
            .fma(c_1, Self::Vec4::load(s_1))
            .fma(c_2, Self::Vec4::load(s_2));
        d.store(out);
    }

    /// Core 2: $O(M)$ Givens stereo rotation $\mathbf{R}(\psi)$ + Householder reflection for `Lite` ($M = 4$).
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

    /// Core 2: $O(M)$ Givens stereo rotation $\mathbf{R}(\psi)$ + Householder reflection for `Full`/`Ultra` ($M = 8$).
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

    /// Core 2 (`CL-10`, `CL-11`): Unity-Slope Normalized Cubic Coercive-Saturating Lyapunov Loop Law.
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

        let v_knee = vq
            .abs()
            .mul(vinv_sigma)
            .clamp(Self::Vec4::splat(0.0), Self::Vec4::splat(1.0));
        let smoothstep = v_knee
            .mul(v_knee)
            .mul(Self::Vec4::splat(3.0).fms(Self::Vec4::splat(2.0), v_knee));
        let g_eff = vg_lin.fma(vg_regen, smoothstep);

        let u_loop = vin.fma(g_eff, vq);
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

    /// Evaluates a fused stereo (`L` and `R`) 16-tap 4-channel (`64` `f32`) dot product in a single pass.
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

    /// Register-resident 4-channel Radial-Polar `SelectiveRgLru` step (`CL-03`..`CL-05`).
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

        let valpha_dc = Self::Vec4::splat(0.5 * (1.0 + r_dc));
        let vr_dc = Self::Vec4::splat(r_dc);
        let vy_dc = valpha_dc.mul(vu.sub(vx_prev)).fma(vr_dc, vy_prev);
        vu.store(dc_x_prev);
        vy_dc.store(dc_y_prev);

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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn exercise_math_backend<B: MathBackend>() {
        let _ = B::NAME;
        let _ = B::KIND.name();

        let v0 = B::Vec4::zero();
        assert_eq!(v0.to_array(), [0.0; 4]);
        let va = B::Vec4::from_array([1.0, -2.0, 3.0, -4.0]);
        let vb = B::Vec4::splat(2.0);
        let _ = va.add(vb);
        let _ = va.sub(vb);
        let _ = va.mul(vb);
        let _ = va.fma(vb, vb);
        let _ = va.fms(vb, vb);
        let _ = va.max(vb);
        let _ = va.min(vb);
        let _ = va.clamp(B::Vec4::splat(-1.0), B::Vec4::splat(1.0));
        let _ = va.abs();
        let _ = va.horizontal_sum();
        let _ = va.dot(vb);

        let mut u_prev = [0.0_f32; 4];
        let mut a_prev = [0.0_f32; 4];
        let mut m_dc = [0.0_f32; 4];
        let mut y_out = [0.0_f32; 4];
        B::poly_adaa_layer_4ch(
            &[0.5, -0.5, 0.25, -0.25],
            &[0.8; 4],
            &[0.4; 4],
            &[0.05; 4],
            0.9993,
            &mut u_prev,
            &mut a_prev,
            &mut m_dc,
            &mut y_out,
        );
        assert!(y_out[0].is_finite());

        let (vy, vabs) = B::poly_adaa_step_vec4(
            B::Vec4::from_array([0.5, -0.5, 0.25, -0.25]),
            &[0.01; 4],
            &[0.8; 4],
            &[0.4; 4],
            &[0.05; 4],
            0.9993,
            &mut u_prev,
            &mut a_prev,
            &mut m_dc,
        );
        assert!(vy.horizontal_sum().is_finite() && vabs.horizontal_sum() >= 0.0);

        let mat = [[0.1_f32; 4]; 4];
        let v_mat = B::fma_mat4x4_vec4(B::Vec4::zero(), &mat, &[1.0, 2.0, 3.0, 4.0]);
        assert!(v_mat.horizontal_sum().is_finite());

        let h_frame = [[0.2_f32; 4]; 3];
        let h_taps = [&h_frame; 6];
        let w1 = [[[0.1_f32; 4]; 4]; 6];
        let w2 = [[[0.1_f32; 4]; 4]; 12];
        let w3 = [[[0.1_f32; 4]; 4]; 18];
        let _ = B::conv6_taps_1group(B::Vec4::zero(), &w1, &h_taps);
        let _ = B::conv6_taps_2groups(B::Vec4::zero(), &w2, &h_taps);
        let _ = B::conv6_taps_3groups(B::Vec4::zero(), &w3, &h_taps);

        let taps6 = [[0.2_f32; 4]; 6];
        let w4x4 = [[[0.1_f32; 4]; 4]; 6];
        let w4x8 = [[[0.1_f32; 4]; 8]; 6];
        let w4x12 = [[[0.1_f32; 4]; 12]; 6];
        let mut c_out = [0.0_f32; 4];
        B::conv6_tile_4x4(&taps6, &w4x4, &[0.01; 4], &mut c_out);
        B::conv6_tile_4x8(&taps6, &taps6, &w4x8, &[0.01; 4], &mut c_out);
        B::conv6_tile_4x12(&taps6, &taps6, &taps6, &w4x12, &[0.01; 4], &mut c_out);
        assert!(c_out[0].is_finite());

        let mut cos_o = [0.0_f32; 4];
        let mut sin_o = [0.0_f32; 4];
        B::algebraic_rotor_4ch(&[0.5, 1.0, 1.5, 2.0], &mut cos_o, &mut sin_o);
        assert!((cos_o[0] * cos_o[0] + sin_o[0] * sin_o[0] - 1.0).abs() < 1e-4);

        let mut z_re = [0.0_f32; 4];
        let mut z_im = [0.0_f32; 4];
        B::rglru_step_4ch(
            &[0.3; 4], &[0.9; 4], &[0.05; 4], &[0.2; 4], &[0.5; 4], &[0.4; 4], &[0.1; 4], &cos_o,
            &sin_o, &mut z_re, &mut z_im, &mut y_out,
        );
        B::selective_rglru_layer_4ch(
            &[0.3; 4], &[0.9; 4], &[0.05; 4], &[0.5; 4], &[0.2; 4], &[0.4; 4], &[0.1; 4], &cos_o,
            &sin_o, &mut z_re, &mut z_im, &mut y_out,
        );
        assert!(y_out[0].is_finite());

        let base_head = [0.1_f32; 8];
        let svd_modes = [0.05_f32; 32];
        let mut collapsed = [0.0_f32; 8];
        B::collapse_svd_head_slice(
            &base_head,
            &svd_modes,
            &[0.5, -0.5, 0.25, -0.25],
            4,
            &mut collapsed,
        );
        assert!(collapsed[0].is_finite());

        let taps16 = [[0.2_f32; 4]; 16];
        let wl16 = [[0.1_f32; 4]; 16];
        let wr16 = [[-0.1_f32; 4]; 16];
        let (sl, sr) = B::stereo_head_dot_16x4(&taps16, &wl16, &wr16);
        let sm = B::mono_head_dot_16x4(&taps16, &wl16);
        assert!((sl - sm).abs() < 1e-6 && (sl + sr).abs() < 1e-5);

        let taps64 = [0.25_f32; 64];
        let wl64 = [0.1_f32; 64];
        let wr64 = [-0.1_f32; 64];
        let d_mono = B::dot_product_16taps_4ch(&taps64, &wl64);
        let d_stereo = B::dot_product_16taps_4ch_stereo(&taps64, &wl64, &wr64);
        assert!((d_mono - d_stereo[0]).abs() < 1e-5);

        let mut lag_out = [0.0_f32; 4];
        B::cubic_lagrange_4ch(
            &[0.25; 4],
            &[1.0; 4],
            &[2.0; 4],
            &[3.0; 4],
            &[4.0; 4],
            &mut lag_out,
        );
        assert!((lag_out[0] - 2.25).abs() < 1e-4);

        let mut q4 = [0.0_f32; 4];
        B::givens_householder_mix_4ch(&[1.0, 0.0, 0.5, -0.5], 0.8, 0.6, &[0.5; 4], &mut q4);
        let mut q8 = [0.0_f32; 8];
        B::givens_householder_mix_8ch(&[0.5; 8], 0.8, 0.6, &[0.35355338; 8], &mut q8);

        let mut s_out = [0.0_f32; 4];
        B::lyapunov_loop_4ch(
            &[0.4; 4], &[0.1; 4], &[0.9; 4], &[0.05; 4], &[2.0; 4], &[0.7; 4], &[0.95; 4],
            &mut s_out,
        );
        let mut dc_x = [0.0_f32; 4];
        let mut dc_y = [0.0_f32; 4];
        B::mufdn_coercive_loop_4ch(
            &[0.4; 4], &[0.9; 4], &[2.0; 4], &[0.7; 4], &[0.95; 4], 0.995, &mut dc_x, &mut dc_y,
            &mut s_out,
        );
        assert!(s_out[0].is_finite());

        let fp16_4 = B::f32x4_to_fp16x4(&[0.5, -0.5, 1.0, 0.0]);
        let f32_4 = B::fp16x4_to_f32x4(&fp16_4);
        assert!((f32_4[0] - 0.5).abs() < 1e-3);

        assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
        let (cp, sp) = B::fast_unit_phasor(0.7);
        assert!((cp * cp + sp * sp - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_all_math_backends_and_backend_kind_names() {
        for (kind, expected) in [
            (BackendKind::Scalar, "Scalar"),
            (BackendKind::ArmNeon, "ARM64 NEON"),
            (BackendKind::Neon, "ARM64 NEON"),
            (BackendKind::ArmSme, "ARM SME"),
            (BackendKind::X86Sse42, "x86 SSE4.2"),
            (BackendKind::X86Avx2Fma, "x86 AVX2"),
            (BackendKind::X86Simd, "x86 AVX2"),
            (BackendKind::WasmSimd128, "WASM SIMD"),
        ] {
            assert_eq!(kind.name(), expected);
        }
        let _ = detect_best_backend();

        exercise_math_backend::<ScalarBackend>();
        exercise_math_backend::<DefaultMathBackend>();
        #[cfg(target_arch = "aarch64")]
        {
            exercise_math_backend::<ArmSmeBackend>();
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_cross_backend_simd_equivalence(
            x in proptest::array::uniform4(-10.0_f32..10.0_f32),
            y in proptest::array::uniform4(-10.0_f32..10.0_f32),
            z in proptest::array::uniform4(-10.0_f32..10.0_f32),
        ) {
            let sx = <ScalarBackend as MathBackend>::Vec4::load(&x);
            let sy = <ScalarBackend as MathBackend>::Vec4::load(&y);
            let sz = <ScalarBackend as MathBackend>::Vec4::load(&z);

            let hx = <DefaultMathBackend as MathBackend>::Vec4::load(&x);
            let hy = <DefaultMathBackend as MathBackend>::Vec4::load(&y);
            let hz = <DefaultMathBackend as MathBackend>::Vec4::load(&z);

            let s_fma = sx.fma(sy, sz).to_array();
            let h_fma = hx.fma(hy, hz).to_array();
            let s_fms = sx.fms(sy, sz).to_array();
            let h_fms = hx.fms(hy, hz).to_array();
            for lane in 0..4 {
                prop_assert!((s_fma[lane] - h_fma[lane]).abs() <= 1e-4);
                prop_assert!((s_fms[lane] - h_fms[lane]).abs() <= 1e-4);
            }

            let s_fp16 = ScalarBackend::f32x4_to_fp16x4(&x);
            let h_fp16 = DefaultMathBackend::f32x4_to_fp16x4(&x);
            let s_back = ScalarBackend::fp16x4_to_f32x4(&s_fp16);
            let h_back = DefaultMathBackend::fp16x4_to_f32x4(&h_fp16);
            for lane in 0..4 {
                prop_assert!((s_back[lane] - h_back[lane]).abs() <= 2e-3);
            }
        }
    }
}
