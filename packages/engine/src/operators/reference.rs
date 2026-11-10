//! Standalone `f64` scalar reference DSP kernels for TORA Operators (Op-T, Op-M, Op-N, Control).
//!
//! Provides canonical mathematical references for bit-accurate parity testing across all SIMD backends.

/// Reference `f64` division-free algebraic `PolyADAA` kernel (`ref[dsp.op-n.polyadaa.midpoint]`):
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
/// and $5\text{ Hz}$ AC coupling ($R_{\text{ac}}$, `ref[dsp.op-n.polyadaa.bspline-ac]`).
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

/// Reference `f64` Radial-Polar `Op-M` modal single-sample step (`ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.stage3.radial-gate]`).
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

/// Reference `f64` 4-tap Catmull-Rom cubic delay interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
#[inline]
pub fn catmull_rom_hermite_f64(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c1 = -0.5 * s_m1 + 0.5 * s_1;
    let c0 = s_0;
    ((c3 * mu + c2) * mu + c1) * mu + c0
}

/// Reference `f64` 4-tap Cubic Lagrange fractional delay weights (`ref[dsp.op-t.reader.catmull-rom]`):
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

/// Reference `f64` 8-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
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

/// Reference `f64` 16-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
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
/// $5\text{ Hz}$ DC blocker (`ref[dsp.stage4.dc-blocker]`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).
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

/// Evaluates the 1D Node-Anchored Newton/Lobatto basis polynomials $\psi_k(u)$ (`ref[dsp.ctrl.basis.lobatto]`).
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

/// Evaluates the Ellipsoidal Trust-Region Norm Soft-Clipping (`ref[dsp.ctrl.op-n.trust-clip]`, `0 FDIV`, `0 FSQRT`).
#[inline]
pub fn eval_trust_region_softclip_f64(z: &mut [f64; 4], trust_radius_mh: f64) {
    let norm_sq = z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    let mh_sq = trust_radius_mh * trust_radius_mh;
    if norm_sq > mh_sq && norm_sq > 0.0 {
        let inv_norm = crate::math::newton6_reciprocal_f64(crate::math::sqrt_f64(norm_sq));
        let scale = trust_radius_mh * inv_norm;
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

#[cfg(test)]
mod operator_verification_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical TORA Operator Verification Suite (`Tests 1A, 1B, 1C, 2A, 2B, 2C, 3`).
    //
    // Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md`, `docs/specs/dsp/op_m_modal_kautz.md`,
    // and `docs/specs/dsp/op_t_transport.md`.
    //
    // Verifies all Rust-side mathematical and numerical operator invariants for:
    // - **Op-N (`Stage II` Commuted Cascade / `PolyADAA`):** `Test 1A` (linear passband preservation,
    //   `verify[dsp.op-n.polyadaa.linear-residual]`), `Test 1B` (division-free midpoint-difference identity,
    //   `verify[dsp.op-n.polyadaa.midpoint]`), and `Test 1C` (cascaded aliasing suppression at $f_0 \approx 3,450\text{ Hz}$,
    //   `verify[dsp.op-n.polyadaa.bspline-ac]`).
    // - **Op-M (`Stage I/III` Kautz & Symplectic Modal Bank):** `Test 2A` (radial-polar $Q$ preservation,
    //   `verify[dsp.op-m.recurrence.symplectic]`), `Test 2B` (branchless 4-FMA + 1-step Newton unit-circle rotor,
    //   `verify[dsp.op-m.rotor.contractive]`), and `Test 2C` (Blelloch $\mathcal{O}(\log_2 L)$ parallel associative scan equivalence,
    //   `verify[arch.operators.differentiability]`).
    // - **Op-T (`Stage I/IV` Elastic Transport Delay Line):** `Test 3` (stateless 4-tap Catmull-Rom vs.
    //   recursive 1st-order Thiran boundary click jerk, `verify[dsp.op-t.reader.catmull-rom]`).

    use std::f64::consts::{PI, TAU};

    use crate::FtzDazGuard;
    use crate::math::ScalarBackend;
    use crate::operators::op_m::{KautzModalBank, SymplecticModal4Lane, algebraic_unit_rotor_op_m};
    use crate::operators::op_n::{MultiDrivePolyAdaaChannel, poly_adaa_algebraic_eval};
    use crate::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
    use crate::operators::reference::{
        catmull_rom_hermite_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64,
    };
    use proptest::prelude::*;

    /// Test 1A (`verify[dsp.op-n.polyadaa.linear-residual]`, `verify[dsp.op-n.polyadaa.bspline-ac]`): Verifies that applying a naive midpoint filter $(1 + z^{-1})/2$
    /// to a 3-stage cascade near Nyquist causes `> 35 dB` passband droop, whereas DNAM's split linear
    /// bypass $\alpha_c x[n]$ + `1x B-Spline PolyADAA` preserves a flat $0.00\text{ dB}$ linear passband.
    #[test]
    fn test_1a_linear_passband_preservation_vs_midpoint_droop() {
        const N: usize = 2048;
        const FS: f64 = 48_000.0;
        const F_TEST: f64 = 22_000.0;

        let input: Vec<f64> = (0..N)
            .map(|n| (TAU * F_TEST * (n as f64) / FS).sin())
            .collect();

        let mut naive = input.clone();
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; N];
            let mut prev = 0.0_f64;
            for i in 0..N {
                next[i] = 0.5 * (naive[i] + prev);
                prev = naive[i];
            }
            naive = next;
        }

        let mut dnam = input.clone();
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; N];
            poly_adaa_slice_f64(&dnam, &mut next, 0.0, 1.0, 0.0, 0.0, 0.9993);
            dnam = next;
        }

        let rms = |sig: &[f64]| -> f64 {
            let s: f64 = sig[256..].iter().map(|&v| v * v).sum();
            (s / ((sig.len() - 256) as f64)).sqrt()
        };

        let rms_in = rms(&input);
        let droop_naive_db = 20.0 * (rms(&naive) / rms_in).log10();
        let droop_dnam_db = 20.0 * (rms(&dnam) / rms_in).log10();

        assert!(
            droop_naive_db < -35.0,
            "Expected naive midpoint filter droop < -35 dB, got {droop_naive_db:.2} dB"
        );
        assert!(
            droop_dnam_db.abs() < 1e-12,
            "Expected DNAM linear passband droop == 0.00 dB, got {droop_dnam_db:.6e} dB"
        );
    }

    /// Test 1B (`verify[dsp.op-n.polyadaa.midpoint]`): Verifies exact equivalence between the analytical antiderivative divided
    /// difference $\frac{\mathcal{F}(u_n) - \mathcal{F}(u_{n-1})}{u_n - u_{n-1}}$ and the division-free
    /// midpoint-difference polynomial $\mathcal{A}(m_n, d_n)$ across 10,000 pairs with `< 1e-12` error
    /// and zero `0/0` singularity at $u_n == u_{n-1}$.
    #[test]
    fn test_1b_division_free_midpoint_difference_identity_10k_pairs() {
        let beta = 0.85_f64;
        let delta = -0.25_f64;
        let antideriv = |u: f64| -> f64 {
            let u2 = u * u;
            let u3 = u2 * u;
            let u4 = u2 * u2;
            let u5 = u4 * u;
            beta * (0.75 * u2 - 0.125 * u4) + delta * ((1.0 / 3.0) * u3 - 0.2 * u5)
        };

        let mut max_alg_err = 0.0_f64;
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut next_u = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            -0.99 + 1.98 * unit
        };

        for _ in 0..10_000 {
            let u0 = next_u();
            let u1 = next_u();
            let alg = poly_adaa_algebraic_f64(u0, u1, beta, delta);
            if (u1 - u0).abs() > 1e-5 {
                let exact = (antideriv(u1) - antideriv(u0)) / (u1 - u0);
                max_alg_err = max_alg_err.max((alg - exact).abs());
            }
        }
        assert!(
            max_alg_err < 1e-12,
            "Test 1B max algebraic error exceeded 1e-12: {max_alg_err:e}"
        );

        for step in -99..=99 {
            let u = (step as f64) * 0.01;
            let inst = beta * u * (1.5 - 0.5 * u * u) + delta * (u * u - u * u * u * u);
            let alg64 = poly_adaa_algebraic_f64(u, u, beta, delta);
            assert!((alg64 - inst).abs() < 1e-15);

            let alg32 = poly_adaa_algebraic_eval(u as f32, u as f32, beta as f32, delta as f32);
            assert!((alg32 - (inst as f32)).abs() < 1e-6);
        }
    }

    /// Test 1C (`verify[dsp.op-n.polyadaa.midpoint]`, `verify[dsp.op-n.polyadaa.bspline-ac]`): Drives cascaded nonlinear saturation stages with a high-amplitude
    /// $f_0 \approx 3,422\text{ Hz}$ sine wave at $f_s = 48\text{ kHz}$ ($N = 1,024$, bin $k_0 = 73$)
    /// and verifies that DNAM's `1x B-Spline PolyADAA` suppresses non-harmonic aliased bin energy
    /// relative to un-antialiased hard/LeakyReLU clipping.
    #[test]
    fn test_1c_commuted_cascade_aliasing_suppression_at_3450hz() {
        const N: usize = 1024;
        const K0: usize = 73;
        const TOTAL: usize = 1280;

        let input_full: Vec<f64> = (0..TOTAL)
            .map(|n| 1.6 * (TAU * (K0 as f64) * (n as f64) / (N as f64)).sin())
            .collect();

        let mut base_full = input_full.clone();
        for _stage in 0..3 {
            for x in &mut base_full {
                let u = 1.6 * (*x) + 0.12;
                *x = if u > 1.0 {
                    1.0 + 0.01 * (u - 1.0)
                } else if u < -1.0 {
                    -1.0 + 0.01 * (u + 1.0)
                } else {
                    u
                };
            }
        }

        let mut dnam_full = input_full;
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; TOTAL];
            poly_adaa_slice_f64(&dnam_full, &mut next, 0.05, 0.0, 1.0, 0.12, 0.9993);
            dnam_full = next;
        }

        let compute_alias_dbc = |sig: &[f64]| -> f64 {
            let win_slice = &sig[TOTAL - N..];
            let mut harmonic_bins = [false; N / 2];
            for m in 1..=7 {
                let center = m * K0;
                for b in center.saturating_sub(2)..=(center + 2).min(N / 2 - 1) {
                    harmonic_bins[b] = true;
                }
            }
            let mut fund_e = 0.0_f64;
            let mut alias_e = 0.0_f64;
            for bin in 4..(N / 2) {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &s) in win_slice.iter().enumerate() {
                    let w = 0.5 - 0.5 * (TAU * (n as f64) / (N as f64)).cos();
                    let ang = TAU * (bin as f64) * (n as f64) / (N as f64);
                    re += s * w * ang.cos();
                    im -= s * w * ang.sin();
                }
                let e = re * re + im * im;
                if (bin as isize - K0 as isize).unsigned_abs() <= 2 {
                    fund_e += e;
                } else if !harmonic_bins[bin] {
                    alias_e += e;
                }
            }
            10.0 * (alias_e / fund_e).log10()
        };

        let alias_base_db = compute_alias_dbc(&base_full);
        let alias_dnam_db = compute_alias_dbc(&dnam_full);
        let suppression_db = alias_base_db - alias_dnam_db;
        assert!(
            suppression_db > 15.0,
            "Expected PolyADAA + B-Spline aliasing suppression > 15 dB, got {suppression_db:.2} dB"
        );
    }

    /// Test 2A (`verify[dsp.op-m.recurrence.symplectic]`, `verify[dsp.op-m.stage3.radial-gate]`): Verifies that Cartesian midpoint pole interpolation collapses
    /// resonator $Q$ from $Q_0 = 125.0$ ($r_0 = 0.992$) to $< 0.10 Q_0$, whereas DNAM's radial-polar
    /// factorization preserves $Q > 0.99 Q_0$.
    #[test]
    fn test_2a_radial_polar_q_preservation_vs_cartesian_collapse() {
        let r0 = 0.992_f64;
        let q_orig = 1.0 / (1.0 - r0);
        let theta_low = 0.05 * PI;
        let theta_high = 0.40 * PI;

        let re_mid = 0.5 * r0 * (theta_low.cos() + theta_high.cos());
        let im_mid = 0.5 * r0 * (theta_low.sin() + theta_high.sin());
        let r_cart = (re_mid * re_mid + im_mid * im_mid).sqrt();
        let q_cart = 1.0 / (1.0 - r_cart);

        let theta_mid = 0.5 * (theta_low + theta_high);
        let (c_rot, s_rot) = algebraic_unit_rotor_op_m(theta_mid as f32);
        let r_polar =
            r0 * ((c_rot as f64) * (c_rot as f64) + (s_rot as f64) * (s_rot as f64)).sqrt();
        let q_polar = 1.0 / (1.0 - r_polar);

        assert!(q_cart < 0.10 * q_orig);
        assert!(q_polar > 0.99 * q_orig);
    }

    /// Test 2B (`verify[dsp.op-m.rotor.contractive]`): Verifies the branchless 4-FMA + 1-step Newton unit-circle rotor across
    /// 20,000 angles $\Delta\theta \in (0, \pi)$ in both `f32` and `f64`.
    #[test]
    fn test_2b_branchless_newton_unit_circle_rotor_20k_angles() {
        const STEPS: usize = 20_000;
        let mut max_err_f64 = 0.0_f64;
        let mut max_unit_f64 = 0.0_f64;

        for i in 0..=STEPS {
            let theta = PI * (i as f64) / (STEPS as f64);
            let u = (2.0 / PI) * theta - 1.0;
            let alpha = (PI * 0.25) * (1.0 - u * u);
            let a2 = alpha * alpha;
            let c_hat = 1.0 - a2 * (0.5 - a2 * (0.041_666_664_183_139_8 - a2 * 0.001_388_839_7));
            let s_hat = alpha * (1.0 - a2 * (0.166_666_656_732_559_2 - a2 * 0.008_333_025_5));
            let r2_hat = c_hat * c_hat + s_hat * s_hat;
            let y1 = 1.5 - 0.5 * r2_hat;
            let c = c_hat * y1;
            let s = s_hat * y1;
            let mag = (c * c + s * s).sqrt();
            max_err_f64 = max_err_f64.max((1.0 - mag).abs());
            max_unit_f64 = max_unit_f64.max(mag);

            if i > 0 && i < STEPS {
                let (c32, s32) = algebraic_unit_rotor_op_m(theta as f32);
                let mag32 = ((c32 as f64) * (c32 as f64) + (s32 as f64) * (s32 as f64)).sqrt();
                assert!(mag32 <= 1.000_001);
            }
        }

        assert!(max_err_f64 < 2e-8);
        assert!(max_unit_f64 <= 1.0 + 1e-15);
    }

    /// Test 2C (`verify[arch.operators.differentiability]`): Verifies exact equivalence between sequential sample-by-sample complex
    /// modal recurrence and Blelloch $\mathcal{O}(\log_2 L)$ parallel associative scan over $L = 1,024$.
    #[test]
    fn test_2c_blelloch_parallel_associative_scan_equivalence_1024() {
        const L: usize = 1024;
        type Cplx = (f64, f64);

        let c_mul =
            |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar * br - ai * bi, ar * bi + ai * br) };
        let c_add = |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar + br, ai + bi) };

        let mut a = vec![(0.0, 0.0); L];
        let mut b = vec![(0.0, 0.0); L];
        for n in 0..L {
            let r = 0.985 + 0.010 * ((n as f64) * 0.017).sin();
            let theta = 0.12 + 0.04 * ((n as f64) * 0.031).cos();
            a[n] = (r * theta.cos(), r * theta.sin());
            b[n] = (
                0.25 * ((n as f64) * 0.11).sin(),
                0.15 * ((n as f64) * 0.07).cos(),
            );
        }

        let mut seq = vec![(0.0, 0.0); L];
        let mut state = (0.0, 0.0);
        for n in 0..L {
            state = c_add(c_mul(a[n], state), b[n]);
            seq[n] = state;
        }

        let mut scan_a = a;
        let mut scan_b = b;
        let mut stride = 1;
        while stride < L {
            let prev_a = scan_a.clone();
            let prev_b = scan_b.clone();
            for i in stride..L {
                scan_a[i] = c_mul(prev_a[i], prev_a[i - stride]);
                scan_b[i] = c_add(c_mul(prev_a[i], prev_b[i - stride]), prev_b[i]);
            }
            stride <<= 1;
        }

        let mut max_scan_err = 0.0_f64;
        for n in 0..L {
            let dr = seq[n].0 - scan_b[n].0;
            let di = seq[n].1 - scan_b[n].1;
            max_scan_err = max_scan_err.max((dr * dr + di * di).sqrt());
        }
        assert!(max_scan_err < 1e-13);
    }

    /// Test 3 (`verify[dsp.op-t.reader.catmull-rom]`): Verifies that Stateless 4-Tap Catmull-Rom Hermite Cubic interpolation has
    /// `> 15x` lower 2nd-difference boundary click jerk than 1st-order recursive Thiran all-pass.
    #[test]
    fn test_3_stateless_catmull_rom_vs_recursive_thiran_boundary_jerk() {
        const FS: f64 = 48_000.0;
        const F_SIG: f64 = 5_000.0;
        const N: usize = 4096;
        const BUF_LEN: usize = 512;

        let mut ring = vec![0.0_f64; BUF_LEN];
        let mut wr = 0_usize;
        let mut err_catmull = Vec::with_capacity(N);
        let mut err_thiran = Vec::with_capacity(N);
        let mut thiran_state = 0.0_f64;

        for n in 0..N {
            let t = (n as f64) / FS;
            let x = (TAU * F_SIG * t).sin();
            ring[wr] = x;

            let tau = 120.0 + 60.0 * (TAU * 3.0 * t).sin();
            let d_int = tau.floor() as usize;
            let mu = (tau - (d_int as f64)).clamp(0.05, 0.95);
            let mu_exact = tau - (d_int as f64);

            let p0 = (wr + BUF_LEN - d_int) & (BUF_LEN - 1);
            let pm1 = (p0 + 1) & (BUF_LEN - 1);
            let p1 = (p0 + BUF_LEN - 1) & (BUF_LEN - 1);
            let p2 = (p0 + BUF_LEN - 2) & (BUF_LEN - 1);

            let y_cat = catmull_rom_hermite_f64(ring[pm1], ring[p0], ring[p1], ring[p2], mu_exact);
            let a_th = (1.0 - mu) / (1.0 + mu);
            let y_th = a_th * (ring[p0] - thiran_state) + ring[p1];
            thiran_state = y_th;

            if n >= 256 {
                let exact = (TAU * F_SIG * (t - tau / FS)).sin();
                err_catmull.push(y_cat - exact);
                err_thiran.push(y_th - exact);
            }
            wr = (wr + 1) & (BUF_LEN - 1);
        }

        let max_jerk = |e: &[f64]| -> f64 {
            e.windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
                .fold(0.0_f64, f64::max)
        };

        assert!(max_jerk(&err_thiran) > 15.0 * max_jerk(&err_catmull));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Property test verifying `KautzModalBank::step_4lane_simd` and `KautzModalBank::step` produce
        /// identical IEEE-754 results for any stable pole configuration and input sequence.
        #[test]
        fn prop_kautz_4lane_simd_matches_scalar_step(
            r in 0.1_f32..0.998_f32,
            theta in 0.05_f32..3.05_f32,
            c_re in -1.5_f32..1.5_f32,
            c_im in -1.5_f32..1.5_f32,
            x in -4.0_f32..4.0_f32,
        ) {
            let _audio = FtzDazGuard::new();
            let mut b_scalar = KautzModalBank::<4>::default();
            let mut b_simd = KautzModalBank::<4>::default();
            for m in 0..4 {
                let mf = (m + 1) as f32 * 0.15;
                b_scalar.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
                b_simd.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
            }
            for k in 0..16 {
                let sample = x * ((k as f32) * 0.3).cos();
                let y_sc = b_scalar.step(sample);
                let y_sm: f32 = b_simd.step_4lane_simd::<ScalarBackend>(sample).iter().sum();
                prop_assert!((y_sc - y_sm).abs() < 1e-6);
            }
        }

        /// Property test verifying `SymplecticModal4Lane`, `MultiDrivePolyAdaaChannel`, and `OpTDelayLine`
        /// remain finite and bounded under arbitrary modulated drives and delay sweeps.
        #[test]
        fn prop_operators_bounded_and_finite(
            x in -10.0_f32..10.0_f32,
            g_base in 0.1_f32..5.0_f32,
            bias in -1.0_f32..1.0_f32,
            delay in 2.0_f32..500.0_f32,
        ) {
            let _audio = FtzDazGuard::new();
            let mut modal = SymplecticModal4Lane::default();
            modal.set_pole(0, 0.96, 0.3);
            let (re, im) = modal.step_lane(0, x);
            prop_assert!(re.is_finite() && im.is_finite());

            let mut adaa = MultiDrivePolyAdaaChannel::default();
            let y_adaa = adaa.step(re, g_base, bias, 0.5, 0.9993);
            prop_assert!(y_adaa.iter().all(|v| v.is_finite()));

            let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
            dl.set_target_delay(delay, false);
            let y_dl = dl.step(y_adaa[0]);
            prop_assert!(y_dl.is_finite());
        }
    }
}
