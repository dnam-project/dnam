//! Standalone `f64` scalar reference DSP kernels for TORA Operators (Op-T, Op-M, Op-N, Control).
//!
//! Provides canonical mathematical references for bit-accurate parity testing across all SIMD backends.

/// Reference `f64` division-free algebraic `PolyADAA` kernel (`CL-01`):
#[inline]
pub fn poly_adaa_algebraic_f64(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd = beta * m * (1.5 - 0.5 * (m2 + d2));
    let even = delta * ((m2 + d2 / 3.0) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2));
    odd + even
}

/// Reference `f64` single-sample `PolyADAA` step with $1\times$ passive B-spline weights $(0.875, 0.125)$
/// and $5\text{ Hz}$ AC coupling ($R_{\text{ac}}$, `CL-01`, `CL-02`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_step_f64(
    x: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: &mut f64,
    a_prev: &mut f64,
    m_dc: &mut f64,
) -> f64 {
    let u_curr = (x * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(*u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * (*a_prev);
    let m_dc_curr = r_ac * (*m_dc) + (1.0 - r_ac) * y_nl;
    *u_prev = u_curr;
    *a_prev = a_curr;
    *m_dc = m_dc_curr;
    alpha * x + (y_nl - m_dc_curr)
}

/// Processes a contiguous `f64` audio slice through the reference `PolyADAA` kernel.
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_slice_f64(
    input: &[f64],
    output: &mut [f64],
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
) {
    let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let u2 = u_rest * u_rest;
    let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
    let mut u_prev = u_rest;
    let mut a_prev = g_rest;
    let mut m_dc = g_rest;

    for (x, y) in input.iter().zip(output.iter_mut()) {
        let u_curr = ((*x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
        let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
        let y_nl = 0.875 * a_curr + 0.125 * a_prev;
        let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
        u_prev = u_curr;
        a_prev = a_curr;
        m_dc = m_dc_curr;
        *y = alpha * (*x) + (y_nl - m_dc_curr);
    }
}

/// Reference `f64` Radial-Polar `Op-M` modal single-sample step (`CL-03`..`CL-05`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_step_f64(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: &mut f64,
    z_im: &mut f64,
) -> f64 {
    let q = (w_sel * x.abs()).min(1.0);
    let g_sel = q * q * (3.0 - 2.0 * q);
    let r = (r_low + dr_span * (1.0 - g_sel)).clamp(0.25, 0.9995);
    let a_re = r * u_rot_cos;
    let a_im = r * u_rot_sin;
    let b_in = b_gamma_norm * x;
    let next_re = a_re * (*z_re) - a_im * (*z_im) + b_in;
    let next_im = a_im * (*z_re) + a_re * (*z_im);
    *z_re = next_re;
    *z_im = next_im;
    c_real * next_re - c_imag * next_im
}

/// Reference `f64` 4-tap Catmull-Rom cubic delay interpolation (`CL-06`).
#[inline]
pub fn catmull_rom_hermite_f64(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c1 = -0.5 * s_m1 + 0.5 * s_1;
    let c0 = s_0;
    ((c3 * mu + c2) * mu + c1) * mu + c0
}

/// Reference `f64` 4-tap Cubic Lagrange fractional delay weights (`CL-06`):
#[inline]
pub fn cubic_lagrange_weights_f64(mu: f64) -> [f64; 4] {
    let mu_p1 = mu + 1.0;
    let mu_m1 = mu - 1.0;
    let mu_m2 = mu - 2.0;
    let prod_m1_m2 = mu_m1 * mu_m2;
    let prod_mu_p1 = mu * mu_p1;
    [
        (-1.0 / 6.0) * mu * prod_m1_m2,
        0.5 * mu_p1 * prod_m1_m2,
        -0.5 * prod_mu_p1 * mu_m2,
        (1.0 / 6.0) * prod_mu_p1 * mu_m1,
    ]
}

/// Reference `f64` 8-channel Orthogonal Givens + Householder mixing matrix (`CL-08`).
pub fn givens_householder_mix_8ch_f64(x: &[f64; 8], cos_psi: f64, sin_psi: f64) -> [f64; 8] {
    let mut g_rot = [0.0_f64; 8];
    for p in 0..4 {
        let i0 = p * 2;
        let i1 = i0 + 1;
        g_rot[i0] = cos_psi * x[i0] - sin_psi * x[i1];
        g_rot[i1] = sin_psi * x[i0] + cos_psi * x[i1];
    }
    let mut sum = 0.0_f64;
    for &v in &g_rot {
        sum += v;
    }
    let sub = 0.25 * sum;
    let mut out = [0.0_f64; 8];
    for i in 0..8 {
        out[i] = g_rot[i] - sub;
    }
    out
}

/// Reference `f64` 16-channel Orthogonal Givens + Householder mixing matrix (`CL-08`).
pub fn givens_householder_mix_16ch_f64(
    x: &[f64; 16],
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: &[f64; 8],
    cross_sin: &[f64; 8],
) -> [f64; 16] {
    let mut g_rot = [0.0_f64; 16];
    for p in 0..8 {
        let i0 = p * 2;
        let i1 = i0 + 1;
        g_rot[i0] = cos_psi * x[i0] - sin_psi * x[i1];
        g_rot[i1] = sin_psi * x[i0] + cos_psi * x[i1];
    }
    for m in 0..8 {
        let a = g_rot[m];
        let b = g_rot[m + 8];
        let c = cross_cos[m];
        let s = cross_sin[m];
        g_rot[m] = c * a - s * b;
        g_rot[m + 8] = s * a + c * b;
    }
    let sum: f64 = g_rot.iter().sum();
    let sub = 0.125 * sum;
    let mut out = [0.0_f64; 16];
    for i in 0..16 {
        out[i] = g_rot[i] - sub;
    }
    out
}

/// Reference `f64` single-sample Op-T Coercive-Saturating Lyapunov loop step with exact gain-normalized
/// $5\text{ Hz}$ DC blocker (`CL-09`, `CL-10`, `CL-11`).
#[inline]
pub fn op_t_coercive_step_f64(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: &mut f64,
    dc_y_prev: &mut f64,
) -> f64 {
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let y_dc = alpha_dc * (u_in - *dc_x_prev) + r_dc * (*dc_y_prev);
    *dc_x_prev = u_in;
    *dc_y_prev = y_dc;

    let ratio = y_dc.abs() / sigma_knee;
    let g_eff = g_lin / (1.0 + ratio * ratio);
    let s_pre = g_eff * y_dc;
    let ratio_ceil = (s_pre.abs() / v_ceil).min(1.0);
    s_pre * (1.0 - (1.0 / 3.0) * ratio_ceil * ratio_ceil)
}

/// Processes a contiguous `f64` audio slice through a single Op-T coercive tank loop channel.
pub fn op_t_tank_slice_f64(
    input: &[f64],
    output: &mut [f64],
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) {
    let mut dc_x_prev = 0.0_f64;
    let mut dc_y_prev = 0.0_f64;
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = op_t_coercive_step_f64(
            *x,
            g_lin,
            sigma_knee,
            v_ceil,
            r_dc,
            &mut dc_x_prev,
            &mut dc_y_prev,
        );
    }
}

/// Evaluates the 1D Node-Anchored Newton/Lobatto basis polynomials $\psi_k(u)$ (`CL-13`).
#[inline]
pub fn eval_newton_lobatto_basis_f64(u: f64, nodes: &[f64; 4], degree: usize) -> [f64; 4] {
    let mut psi = [0.0; 4];
    if degree >= 1 {
        psi[0] = u - nodes[0];
    }
    if degree >= 2 {
        psi[1] = psi[0] * (u - nodes[1]);
    }
    if degree >= 3 {
        psi[2] = psi[1] * (u - nodes[2]);
    }
    psi
}

/// Evaluates the Rank-4 Factorization Machine (FM) interaction term.
pub fn eval_rank4_fm_interaction_f64(v_matrix: &[[f64; 4]], u_norm: &[f64]) -> [f64; 4] {
    let p = u_norm.len().min(v_matrix.len());
    let mut sum_v = [0.0f64; 4];
    let mut sum_v2 = [0.0f64; 4];

    for i in 0..p {
        let ui = u_norm[i];
        let ui2 = ui * ui;
        for f in 0..4 {
            let vif = v_matrix[i][f];
            sum_v[f] += vif * ui;
            sum_v2[f] += vif * vif * ui2;
        }
    }

    let mut out = [0.0f64; 4];
    for f in 0..4 {
        out[f] = 0.5 * (sum_v[f] * sum_v[f] - sum_v2[f]);
    }
    out
}

/// Evaluates the 4-harmonic Chebyshev polynomial LFO waveshaper.
#[inline]
pub fn eval_chebyshev_lfo_4harm_f64(cos_phi: f64) -> [f64; 4] {
    let x = cos_phi.clamp(-1.0, 1.0);
    let x2 = x * x;
    let t1 = x;
    let t2 = 2.0 * x2 - 1.0;
    let t3 = 4.0 * x2 * x - 3.0 * x;
    let t4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0;
    [t1, t2, t3, t4]
}

/// Evaluates the Ellipsoidal Trust-Region Norm Soft-Clipping (`CL-15`, `CL-16`).
#[inline]
pub fn eval_trust_region_softclip_f64(z: &mut [f64; 4], trust_radius_mh: f64) {
    let norm_sq = z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    let mh_sq = trust_radius_mh * trust_radius_mh;
    if norm_sq > mh_sq && norm_sq > 0.0 {
        let scale = trust_radius_mh / norm_sq.sqrt();
        z[0] *= scale;
        z[1] *= scale;
        z[2] *= scale;
        z[3] *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use proptest::prelude::*;

    #[test]
    fn test_poly_adaa_reference() {
        let u_prev = 0.2;
        let u_curr = 0.5;
        let val = poly_adaa_algebraic_f64(u_prev, u_curr, 1.0, 0.5);
        assert!(val.is_finite());

        let mut up = 0.0;
        let mut ap = 0.0;
        let mut dc = 0.0;
        let out = poly_adaa_step_f64(0.3, 0.1, 1.0, 0.2, 0.999, &mut up, &mut ap, &mut dc);
        assert!(out.is_finite());
        assert_ne!(up, 0.0);

        let input = vec![0.1, -0.2, 0.5, -0.8, 0.4];
        let mut output = vec![0.0; 5];
        poly_adaa_slice_f64(&input, &mut output, 0.05, 0.2, 0.8, 0.3, 0.99);
        for &y in &output {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_op_m_modal_reference() {
        let mut z_re = 0.0;
        let mut z_im = 0.0;
        let rot_cos = (0.1_f64).cos();
        let rot_sin = (0.1_f64).sin();
        let y = op_m_modal_step_f64(
            0.5, 0.8, 0.15, 0.9, 0.5, 0.7, 0.3, rot_cos, rot_sin, &mut z_re, &mut z_im,
        );
        assert!(y.is_finite());
        assert!(z_re.is_finite());
        assert!(z_im.is_finite());
    }

    #[test]
    fn test_interpolations_and_weights_reference() {
        let s_m1 = 1.0;
        let s_0 = 2.0;
        let s_1 = 3.0;
        let s_2 = 4.0;
        assert!((catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, 0.0) - s_0).abs() < 1e-12);
        assert!((catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, 1.0) - s_1).abs() < 1e-12);

        let w = cubic_lagrange_weights_f64(0.4);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_givens_householder_orthogonal_reference() {
        let psi: f64 = 0.3;
        let c = psi.cos();
        let s = psi.sin();
        let x8 = [1.0, 2.0, -1.0, 0.5, 0.2, -0.4, 0.8, -0.6];
        let y8 = givens_householder_mix_8ch_f64(&x8, c, s);
        let norm_in: f64 = x8.iter().map(|v| v * v).sum();
        let norm_out: f64 = y8.iter().map(|v| v * v).sum();
        assert!((norm_in - norm_out).abs() < 1e-10);

        let x16 = [
            1.0, -0.5, 0.2, 0.8, -1.2, 0.4, -0.1, 0.9, 0.3, -0.7, 0.6, -0.2, 0.5, -0.4, 0.1, -0.8,
        ];
        let cross_c = [0.8; 8];
        let cross_s = [0.6; 8];
        let y16 = givens_householder_mix_16ch_f64(&x16, c, s, &cross_c, &cross_s);
        let norm_in16: f64 = x16.iter().map(|v| v * v).sum();
        let norm_out16: f64 = y16.iter().map(|v| v * v).sum();
        assert!((norm_in16 - norm_out16).abs() < 1e-10);
    }

    #[test]
    fn test_op_t_coercive_reference() {
        let mut dc_x = 0.0;
        let mut dc_y = 0.0;
        let y = op_t_coercive_step_f64(0.5, 0.9, 1.0, 2.0, 0.999, &mut dc_x, &mut dc_y);
        assert!(y.is_finite());
        assert_eq!(dc_x, 0.5);

        let input = vec![0.2, -0.4, 0.8, -0.1];
        let mut output = vec![0.0; 4];
        op_t_tank_slice_f64(&input, &mut output, 0.95, 1.2, 1.5, 0.998);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_control_math_reference() {
        let nodes = [0.0, 0.25, 0.75, 1.0];
        let psi0 = eval_newton_lobatto_basis_f64(0.5, &nodes, 0);
        assert_eq!(psi0, [0.0; 4]);
        let psi1 = eval_newton_lobatto_basis_f64(0.5, &nodes, 1);
        assert_eq!(psi1[0], 0.5);
        let psi2 = eval_newton_lobatto_basis_f64(0.5, &nodes, 2);
        assert_eq!(psi2[1], 0.5 * 0.25);
        let psi3 = eval_newton_lobatto_basis_f64(0.5, &nodes, 3);
        assert_eq!(psi3[2], 0.5 * 0.25 * (-0.25));

        let v_mat = [[1.0, 0.5, 0.2, 0.1], [0.3, 0.4, 0.5, 0.6]];
        let u_norm = [0.8, 0.6];
        let fm = eval_rank4_fm_interaction_f64(&v_mat, &u_norm);
        for &v in &fm {
            assert!(v.is_finite());
        }

        let cheb = eval_chebyshev_lfo_4harm_f64(0.5);
        assert!((cheb[0] - 0.5).abs() < 1e-12);
        assert!((cheb[1] - (2.0 * 0.25 - 1.0)).abs() < 1e-12);

        let mut z = [1.0, 1.0, 1.0, 1.0];
        eval_trust_region_softclip_f64(&mut z, 1.0);
        let norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);

        let mut z_in = [0.1, 0.1, 0.1, 0.1];
        eval_trust_region_softclip_f64(&mut z_in, 1.0);
        assert_eq!(z_in, [0.1, 0.1, 0.1, 0.1]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_givens_householder_8ch_isometry_and_coercive_bound(
            psi in 0.0f64..core::f64::consts::FRAC_PI_2,
            x8 in prop::array::uniform8(-5.0f64..5.0f64),
            u_in in -20.0f64..20.0f64,
            g_lin in 0.5f64..1.25f64,
            v_ceil in 0.5f64..2.0f64,
        ) {
            let (c, s) = (psi.cos(), psi.sin());
            let y8 = givens_householder_mix_8ch_f64(&x8, c, s);
            let norm_x: f64 = x8.iter().map(|v| v * v).sum();
            let norm_y: f64 = y8.iter().map(|v| v * v).sum();
            prop_assert!((norm_x - norm_y).abs() < 1e-10);

            let mut dc_x = 0.0;
            let mut dc_y = 0.0;
            let y_coercive = op_t_coercive_step_f64(u_in, g_lin, 0.5 * v_ceil, v_ceil, 0.995, &mut dc_x, &mut dc_y);
            prop_assert!(y_coercive.abs() <= v_ceil + 1e-9);
        }
    }
}
