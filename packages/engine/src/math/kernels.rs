//! Standalone scalar (`f32`) and high-precision oracle (`f64`) DSP kernels for `PolyADAA` (`CL-01`),
//! Algebraic Newton Unit Rotor (`CL-03`), and 4-Tap Cubic Lagrange interpolation (`CL-06`).

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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_reference_dsp_kernels_f32_vs_f64() {
        assert_eq!(clamp_pre_activation(3.0), 1.0);
        assert_eq!(clamp_pre_activation(-3.0), -1.0);
        let ws32 = eval_poly_adaa_waveshaper(0.5, 0.4, 0.1);
        let ws64 = eval_poly_adaa_waveshaper_f64(0.5, 0.4, 0.1);
        assert!((ws32 as f64 - ws64).abs() < 1e-6);

        let anti = eval_poly_adaa_antiderivative_f64(0.5, 0.4, 0.1);
        assert!(anti.is_finite());

        let ad32 = eval_poly_adaa_algebraic(0.2, 0.4, 0.5, 0.1);
        let ad64 = eval_poly_adaa_algebraic_f64(0.2, 0.4, 0.5, 0.1);
        assert!((ad32 as f64 - ad64).abs() < 1e-6);

        let (r32_c, r32_s) = algebraic_unit_rotor(1.0);
        let (r64_c, r64_s) = algebraic_unit_rotor_f64(1.0);
        assert!((r32_c as f64 - r64_c).abs() < 1e-6);
        assert!((r32_s as f64 - r64_s).abs() < 1e-6);

        let (sc, ss) = stabilize_complex_rotor(0.99, 0.0);
        assert!((sc - 1.0).abs() < 1e-3 && ss.abs() < 1e-6);

        let cw32 = cubic_lagrange_weights(0.3);
        let cw64 = cubic_lagrange_weights_f64(0.3);
        assert!((cw32[0] as f64 - cw64[0]).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_poly_adaa_algebraic_matches_antiderivative_difference_quotient(
            u_prev in -1.0_f64..1.0_f64,
            u_curr in -1.0_f64..1.0_f64,
            beta in 0.0_f64..2.0_f64,
            delta in -0.25_f64..0.25_f64,
        ) {
            let alg = eval_poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
            let du = u_curr - u_prev;
            if du.abs() > 1e-5 {
                let f_curr = eval_poly_adaa_antiderivative_f64(u_curr, beta, delta);
                let f_prev = eval_poly_adaa_antiderivative_f64(u_prev, beta, delta);
                let quotient = (f_curr - f_prev) / du;
                prop_assert!((alg - quotient).abs() < 1e-9);
            } else {
                let u_mid = 0.5 * (u_curr + u_prev);
                let g_mid = eval_poly_adaa_waveshaper_f64(u_mid, beta, delta);
                prop_assert!((alg - g_mid).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_algebraic_unit_rotor_stays_on_unit_circle(
            theta_rad in 0.0001_f64..(std::f64::consts::PI - 0.0001),
        ) {
            let (c64, s64) = algebraic_unit_rotor_f64(theta_rad);
            let norm64 = (c64 * c64 + s64 * s64).sqrt();
            prop_assert!(norm64 <= 1.000_000_000_1);
            prop_assert!((norm64 - 1.0).abs() < 1e-7);

            let (c32, s32) = algebraic_unit_rotor(theta_rad as f32);
            let norm32 = ((c32 as f64) * (c32 as f64) + (s32 as f64) * (s32 as f64)).sqrt();
            prop_assert!(norm32 <= 1.000_001);
            prop_assert!((norm32 - 1.0).abs() < 1e-5);
        }

        #[test]
        fn prop_cubic_lagrange_partition_of_unity_and_cubic_exactness(
            mu in 0.0_f64..1.0_f64,
            a0 in -5.0_f64..5.0_f64,
            a1 in -5.0_f64..5.0_f64,
            a2 in -5.0_f64..5.0_f64,
            a3 in -5.0_f64..5.0_f64,
        ) {
            let w = cubic_lagrange_weights_f64(mu);
            let sum = w[0] + w[1] + w[2] + w[3];
            prop_assert!((sum - 1.0).abs() < 1e-14);

            let p = |x: f64| a0 + a1 * x + a2 * x * x + a3 * x * x * x;
            let interp = w[0] * p(-1.0) + w[1] * p(0.0) + w[2] * p(1.0) + w[3] * p(2.0);
            let exact = p(mu);
            prop_assert!((interp - exact).abs() < 1e-11);
        }
    }
}
