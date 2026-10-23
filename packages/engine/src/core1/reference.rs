//! Standalone `f64` reference DSP kernels for **Core 1** (`PolyADAA` and `SelectiveRgLru`).

/// Reference `f64` division-free algebraic `PolyADAA` kernel (`CL-01`, `Test 1`):
/// $$\mathcal{A}_c[n] = \beta m_n\big(1.5 - 0.5(m_n^2 + d_n^2)\big) + \delta\Big((m_n^2 + \tfrac{1}{3}d_n^2) - (m_n^4 + 2m_n^2 d_n^2 + \tfrac{1}{5}d_n^4)\Big)$$
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

/// Reference `f64` Radial-Polar `SelectiveRgLru` single-sample step (`CL-03`..`CL-05`, `Test 2`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_step_f64(
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

/// Processes a contiguous `f64` slice through the reference Radial-Polar `SelectiveRgLru` kernel.
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_slice_f64(
    input: &[f64],
    output: &mut [f64],
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
) {
    let mut z_re = 0.0;
    let mut z_im = 0.0;
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = selective_rglru_step_f64(
            *x,
            r_low,
            dr_span,
            w_sel,
            b_gamma_norm,
            c_real,
            c_imag,
            u_rot_cos,
            u_rot_sin,
            &mut z_re,
            &mut z_im,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Division-Free Algebraic `PolyADAA` & Exact 3-State DC Resting Equilibrium (`CL-01`, `CL-02`).
    #[test]
    fn test_1_division_free_algebraic_poly_adaa_and_dc_equilibrium() {
        let beta = 0.65_f64;
        let delta = 0.25_f64;

        // 1. Exact agreement with analytical antiderivative difference quotient for separated points
        let pairs = [
            (-0.9, -0.3),
            (-0.4, 0.2),
            (0.1, 0.85),
            (-0.75, 0.75),
            (0.25, 0.250001),
        ];
        for (u0, u1) in pairs {
            let alg = poly_adaa_algebraic_f64(u0, u1, beta, delta);
            let f1 = |u: f64| -> f64 {
                beta * (0.75 * u * u - 0.125 * u.powi(4))
                    + delta * (u.powi(3) / 3.0 - u.powi(5) / 5.0)
            };
            let diff_quotient = (f1(u1) - f1(u0)) / (u1 - u0);
            assert!(
                (alg - diff_quotient).abs() < 1e-10,
                "PolyADAA algebraic mismatch: alg={alg}, diff={diff_quotient}"
            );
        }

        // 2. Exact zero-division convergence when u_curr == u_prev
        for u in [-1.0_f64, -0.5, 0.0, 0.37, 1.0] {
            let alg = poly_adaa_algebraic_f64(u, u, beta, delta);
            let g_nl = beta * u * (1.5 - 0.5 * u * u) + delta * (u * u - u.powi(4));
            assert!(
                (alg - g_nl).abs() < 1e-15,
                "Degenerate limit mismatch at u={u}"
            );
        }

        // 3. Exact 0.000000 Sample-0 and Sample-1 transient on silence with non-zero bias b_conv
        let input = [0.0_f64; 64];
        let mut output = [1.0_f64; 64];
        poly_adaa_slice_f64(&input, &mut output, 0.45, 1.0, beta, delta, 0.9993);
        for (i, &y) in output.iter().enumerate() {
            assert!(
                y.abs() == 0.0,
                "Non-zero transient at sample {i} on silence: {y:.12e}"
            );
        }

        // 4. Single-sample step function test
        let mut u_prev = 0.0;
        let mut a_prev = 0.0;
        let mut m_dc = 0.0;
        let y_step = poly_adaa_step_f64(
            0.5,
            1.0,
            beta,
            delta,
            0.9993,
            &mut u_prev,
            &mut a_prev,
            &mut m_dc,
        );
        assert!(y_step.is_finite() && y_step > 0.0);
    }

    /// Test 2: Radial-Polar `SelectiveRgLru` Stability & $\mathcal{H}_2$ Energy Normalization (`CL-03`..`CL-05`).
    #[test]
    fn test_2_radial_polar_selective_rglru_stability_and_h2_norm() {
        let theta = 0.15_f64;
        let (c_rot, s_rot) = (theta.cos(), theta.sin());

        for &r_bar in &[0.50_f64, 0.85, 0.95, 0.995, 0.9995] {
            let b_gamma_norm = (1.0 - r_bar * r_bar).sqrt();
            let mut impulse = [0.0_f64; 16_384];
            impulse[0] = 1.0;
            let mut out = [0.0_f64; 16_384];
            selective_rglru_slice_f64(
                &impulse,
                &mut out,
                r_bar,
                0.0,
                0.0,
                b_gamma_norm,
                1.0,
                0.0,
                c_rot,
                s_rot,
            );
            let h2_energy: f64 = out.iter().map(|v| v * v).sum();
            assert!(
                (0.45..=1.05).contains(&h2_energy),
                "H2 impulse energy out of bounds for r_bar={r_bar}: {h2_energy}"
            );
        }
    }
}
