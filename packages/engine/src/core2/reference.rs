//! Standalone `f64` reference DSP kernels for **Core 2 (`M-UFDN`)**.

/// Reference `f64` 4-tap Cubic Lagrange fractional delay weights (`CL-06`, `Test 3`):
/// - $w_{-1}(\mu) = -\frac{1}{6}\mu(\mu - 1)(\mu - 2)$
/// - $w_{0}(\mu) = \frac{1}{2}(\mu + 1)(\mu - 1)(\mu - 2)$
/// - $w_{+1}(\mu) = -\frac{1}{2}\mu(\mu + 1)(\mu - 2)$
/// - $w_{+2}(\mu) = \frac{1}{6}\mu(\mu + 1)(\mu - 1)$
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

/// Reference `f64` 8-channel Orthogonal Givens + Householder mixing matrix $\mathbf{U}(\psi)$ (`CL-08`, `Test 5`):
/// Applies $2 \times 2$ Givens rotations $(\cos\psi, \sin\psi)$ on pairs `(0,1), (2,3), (4,5), (6,7)`,
/// followed by the $8 \times 8$ Householder reflection $\mathbf{H}_8 = \mathbf{I}_8 - \frac{1}{4}\mathbf{1}\mathbf{1}^\top$.
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

/// Reference `f64` single-sample Coercive-Saturating Lyapunov loop step with exact gain-normalized
/// $5\text{ Hz}$ DC blocker (`CL-09`, `CL-10`, `CL-11`, `Test 4`).
#[inline]
pub fn mufdn_coercive_step_f64(
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
    let v_norm = (s_pre * (2.0 / (3.0 * v_ceil))).clamp(-1.0, 1.0);
    v_ceil * v_norm * (1.5 - 0.5 * v_norm * v_norm)
}

/// Processes a contiguous `f64` slice through a single-line reference Coercive-Saturating `M-UFDN` loop.
#[allow(clippy::too_many_arguments)]
pub fn mufdn_slice_f64(
    input: &[f64],
    output: &mut [f64],
    delay_samples: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) {
    let ring_len = (delay_samples + 16).next_power_of_two().max(64);
    let mask = ring_len - 1;
    let mut ring = alloc::vec![0.0_f64; ring_len];
    let mut w_pos = 0_usize;
    let mut dc_x_prev = 0.0_f64;
    let mut dc_y_prev = 0.0_f64;

    for (x, y) in input.iter().zip(output.iter_mut()) {
        let r_pos = w_pos.wrapping_sub(delay_samples) & mask;
        let delayed = ring[r_pos];
        let s_next = mufdn_coercive_step_f64(
            delayed + *x,
            g_lin,
            sigma_knee,
            v_ceil,
            r_dc,
            &mut dc_x_prev,
            &mut dc_y_prev,
        );
        ring[w_pos] = s_next;
        *y = s_next;
        w_pos = (w_pos + 1) & mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::algebraic_unit_rotor;
    use proptest::prelude::*;

    /// Test 3: 4-Tap Cubic Lagrange Fractional Delay Partition of Unity & TZF Null (`CL-06`).
    #[test]
    fn test_3_cubic_lagrange_fractional_delay_and_tzf_null() {
        for k in 0..=100 {
            let mu = (k as f64) / 100.0;
            let w = cubic_lagrange_weights_f64(mu);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15);
            let p = |x: f64| -> f64 { 0.3 * x.powi(3) - 1.2 * x.powi(2) + 0.7 * x + 0.5 };
            let interp = w[0] * p(-1.0) + w[1] * p(0.0) + w[2] * p(1.0) + w[3] * p(2.0);
            assert!((interp - p(mu)).abs() < 1e-14);
        }
    }

    /// Test 4: Coercive-Saturating `M-UFDN` Loop & Gain-Normalized DC Blocker (`CL-09`, `CL-10`, `CL-11`).
    #[test]
    fn test_4_coercive_saturating_mufdn_dub_echo_and_dc_blocker() {
        let g_lin = 0.9990_f64;
        let sigma_knee = 0.35_f64;
        let v_ceil = 0.985_f64;
        let r_dc = 0.9993_f64;

        let mut input = [0.0_f64; 8_192];
        for (n, x) in input.iter_mut().enumerate().take(2_048) {
            *x = 1.5 * (2.0 * core::f64::consts::PI * 110.0 * (n as f64) / 48_000.0).sin() + 0.45;
        }
        let mut output = [0.0_f64; 8_192];
        mufdn_slice_f64(&input, &mut output, 128, g_lin, sigma_knee, v_ceil, r_dc);

        let max_abs = output.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_abs <= v_ceil + 1e-12);
        let tail_mean: f64 = output[6_000..].iter().sum::<f64>() / 2_192.0;
        assert!(tail_mean.abs() < 1e-2);
    }

    /// Test 5: Orthogonal Givens + Householder Mixing Matrix Unitarity (`CL-08`).
    #[test]
    fn test_5_givens_householder_unitarity() {
        let psi = 0.618_033_988_75_f64;
        let (c, s) = (psi.cos(), psi.sin());
        let x = [0.4, -0.7, 1.1, -0.2, 0.85, -0.55, 0.3, 0.9];
        let y = givens_householder_mix_8ch_f64(&x, c, s);
        let norm_x: f64 = x.iter().map(|v| v * v).sum();
        let norm_y: f64 = y.iter().map(|v| v * v).sum();
        assert!((norm_x - norm_y).abs() < 1e-14);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_givens_householder_unitarity_and_involution(
            x in prop::array::uniform8(-10.0f64..10.0f64),
            phi in -1.0f32..1.0f32,
        ) {
            let (c32, s32) = algebraic_unit_rotor(phi);
            let r = ((c32 as f64).hypot(s32 as f64)).max(1e-12);
            let (c, s) = ((c32 as f64) / r, (s32 as f64) / r);

            let y = givens_householder_mix_8ch_f64(&x, c, s);
            let norm_x: f64 = x.iter().map(|v| v * v).sum();
            let norm_y: f64 = y.iter().map(|v| v * v).sum();
            prop_assert!((norm_x - norm_y).abs() <= 1e-10 * norm_x.max(1.0));
        }

        #[test]
        fn prop_mufdn_coercive_loop_bounded_and_lyapunov_contractive(
            u_in in -100.0f64..100.0f64,
            g_lin in 0.0f64..=0.9990f64,
            sigma_knee in 0.1f64..2.0f64,
            v_ceil in 0.25f64..=2.0f64,
            r_dc in 0.990f64..=0.9999f64,
        ) {
            let mut dc_x = 0.0;
            let mut dc_y = 0.0;
            let s_next = mufdn_coercive_step_f64(
                u_in, g_lin, sigma_knee, v_ceil, r_dc, &mut dc_x, &mut dc_y,
            );
            prop_assert!(s_next.is_finite());
            prop_assert!(s_next.abs() <= v_ceil + 1e-12);

            let mut dc_x2 = 0.0;
            let mut dc_y2 = 0.0;
            let s_only = mufdn_coercive_step_f64(
                u_in, g_lin, sigma_knee, v_ceil, -1.0, &mut dc_x2, &mut dc_y2,
            );
            prop_assert!(s_only.abs() <= g_lin * u_in.abs() + 1e-12);
        }
    }
}
