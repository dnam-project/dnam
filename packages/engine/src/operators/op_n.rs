//! Op-N: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//!
//! Canonical Reference: `docs/dsp/op_n_commuted_cascade.md` (`CL-01`, `CL-02`, `CL-13`, `CL-15`).

use crate::math::flush_recursive_state_f32;

/// Drive scales for the 4-lane Multi-Drive PolyADAA Bank.
pub const MULTI_DRIVE_SCALES: [f32; 4] = [0.25, 1.0, 4.0, 16.0];

/// Exact division-free algebraic PolyADAA segment evaluation (CL-01).
///
/// Evaluates:
/// `m = 0.5 * (u_curr + u_prev)`
/// `d = 0.5 * (u_curr - u_prev)`
/// `A_odd = m * (1.5 - 0.5 * (m^2 + d^2))`
/// `A_even = (m^2 + 1/3 * d^2) - (m^4 + 2 * m^2 * d^2 + 1/5 * d^4)`
/// `A = beta * A_odd + delta * A_even`
#[inline(always)]
pub fn poly_adaa_algebraic_eval(u_prev: f32, u_curr: f32, beta: f32, delta: f32) -> f32 {
    const ONE_THIRD: f32 = 1.0 / 3.0;
    const ONE_FIFTH: f32 = 0.2;

    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);

    let m2 = m * m;
    let d2 = d * d;

    let odd_factor = 1.5 - 0.5 * (m2 + d2);
    let a_odd = m * odd_factor;

    let quad = m2 + ONE_THIRD * d2;
    let quart = m2 * m2 + 2.0 * m2 * d2 + ONE_FIFTH * d2 * d2;
    let a_even = quad - quart;

    beta * a_odd + delta * a_even
}

/// Instantaneous C^2 compactly supported wavefolder.
///
/// Evaluates `w_odd(u) = u * (1 - u^2)^2` and `w_even(u) = u^2 * (1 - u^2)^2` on `u in [-1, 1]`.
#[inline(always)]
pub fn c2_wavefolder(u: f32) -> (f32, f32) {
    let u_clamped = u.clamp(-1.0, 1.0);
    let u2 = u_clamped * u_clamped;
    let one_minus_u2 = 1.0 - u2;
    let factor = one_minus_u2 * one_minus_u2;

    let w_odd = u_clamped * factor;
    let w_even = u2 * factor;
    (w_odd, w_even)
}

/// 2D Quadrature Phase-Space Hysteresis state.
///
/// Models magnetic transformer core memory and optocoupler decay.
#[derive(Clone, Copy, Debug)]
pub struct QuadratureHysteresis2D {
    pub prev_s_re: f32,
    pub kappa_quad: f32,
}

impl Default for QuadratureHysteresis2D {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadratureHysteresis2D {
    pub const fn new() -> Self {
        Self {
            prev_s_re: 0.0,
            kappa_quad: 0.15,
        }
    }

    pub fn reset(&mut self) {
        self.prev_s_re = 0.0;
    }

    /// Steps the 2D hysteresis model given quadrature phase-space coordinates `(s_re, s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, s_re: f32, s_im: f32) -> f32 {
        self.step_with_rate(s_re, s_im, 1.0)
    }

    /// Steps the 2D hysteresis model with continuous-time ZOH differential rate scaling `diff_scale = 1 / rho_sr`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_with_rate(&mut self, s_re: f32, s_im: f32, diff_scale: f32) -> f32 {
        let delta_re = (s_re - self.prev_s_re) * diff_scale;
        let hyst = s_re * delta_re + self.kappa_quad * s_re * s_im;
        self.prev_s_re = s_re;
        hyst
    }
}

/// Precomputed-reciprocal ZOH Sample-Rate Reducer & Bitcrusher (Family 5).
///
/// This standalone helper still uses a floating-point phase accumulator; it
/// does not yet implement the specified fixed-hertz timing profile.
#[derive(Clone, Copy, Debug)]
pub struct ZohBitcrusher {
    pub step_size: f32,
    pub step_inv: f32,
    pub phase: f32,
    pub phase_inc: f32,
    pub held_sample: f32,
}

impl Default for ZohBitcrusher {
    fn default() -> Self {
        Self::new()
    }
}

impl ZohBitcrusher {
    pub const fn new() -> Self {
        Self {
            step_size: 1.0 / 32768.0,
            step_inv: 32768.0,
            phase: 0.0,
            phase_inc: 1.0,
            held_sample: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.held_sample = 0.0;
    }

    /// Sets bit depth `bits in [2, 16]` and downsampled sample rate `target_fs_hz`.
    pub fn set_parameters(&mut self, bits: u32, target_fs_hz: f32, sample_rate_hz: f32) {
        let b = bits.clamp(2, 16);
        let levels = (1u32 << (b - 1)) as f32;
        self.step_inv = levels;
        self.step_size = 1.0 / levels;

        let sr = sample_rate_hz.max(1000.0);
        self.phase_inc = (target_fs_hz / sr).clamp(0.0, 1.0);
    }

    /// Steps the ZOH bitcrusher with zero audio-rate divisions.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.phase_inc == 0.0 {
            return self.held_sample;
        }

        let latch = if self.phase_inc == 1.0 {
            true
        } else {
            let mut next_phase = self.phase + self.phase_inc;
            let latch = next_phase >= 1.0;
            if latch {
                next_phase -= 1.0;
            }
            self.phase = flush_recursive_state_f32(next_phase);
            latch
        };

        if latch {
            // Above this threshold every finite f32 is already on the grid.
            if !x.is_finite() || x.abs() >= 8_388_608.0 * self.step_size {
                self.held_sample = x;
            } else {
                let scaled = x * self.step_inv;
                let truncated = scaled as i32;
                let lower = truncated - i32::from(scaled < truncated as f32);
                let fraction = scaled - lower as f32;
                let quantized = lower + i32::from(fraction >= 0.5);
                self.held_sample = quantized as f32 * self.step_size;
            }
        }

        self.held_sample
    }
}

/// Multi-Drive 1x B-Spline PolyADAA Bank Channel State (CL-01, CL-02).
#[derive(Clone, Copy, Debug)]
pub struct MultiDrivePolyAdaaChannel {
    pub u_prev: [f32; 4],
    pub a_prev: [f32; 4],
    pub m_dc: [f32; 4],
}

impl Default for MultiDrivePolyAdaaChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDrivePolyAdaaChannel {
    pub const fn new() -> Self {
        Self {
            u_prev: [0.0; 4],
            a_prev: [0.0; 4],
            m_dc: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.u_prev.fill(0.0);
        self.a_prev.fill(0.0);
        self.m_dc.fill(0.0);
    }

    /// Initializes resting DC equilibrium fixed point for grid bias `b_conv`.
    pub fn init_dc_equilibrium(&mut self, b_conv: f32, beta: f32, delta: f32) {
        for (i, &drive) in MULTI_DRIVE_SCALES.iter().enumerate() {
            let u_rest = ((2.0 / 3.0) * drive * b_conv).clamp(-1.0, 1.0);
            let a_rest = poly_adaa_algebraic_eval(u_rest, u_rest, beta, delta);
            self.u_prev[i] = u_rest;
            self.a_prev[i] = a_rest;
            self.m_dc[i] = flush_recursive_state_f32(a_rest);
        }
    }

    /// Steps the multi-drive bank for input `x`, returning 4-lane nonlinearly shaped basis vector.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, beta: f32, delta: f32, alpha: f32, r_ac: f32) -> [f32; 4] {
        let mut out = [0.0; 4];
        let one_minus_rac = 1.0 - r_ac;

        for (i, &drive) in MULTI_DRIVE_SCALES.iter().enumerate() {
            let u_curr = ((2.0 / 3.0) * drive * x).clamp(-1.0, 1.0);
            let a_curr = poly_adaa_algebraic_eval(self.u_prev[i], u_curr, beta, delta);

            // B-spline blend: 0.875 * A[n] + 0.125 * A[n-1]
            let y_nl = 0.875 * a_curr + 0.125 * self.a_prev[i];

            // 5 Hz AC coupling: m_dc[n] = r_ac * m_dc[n-1] + (1 - r_ac) * y_nl
            let next_m_dc = r_ac * self.m_dc[i] + one_minus_rac * y_nl;

            // Linear separation: alpha * x + (y_nl - m_dc)
            out[i] = alpha * x + (y_nl - next_m_dc);

            self.u_prev[i] = u_curr;
            self.a_prev[i] = a_curr;
            self.m_dc[i] = flush_recursive_state_f32(next_m_dc);
        }
        out
    }
}

/// 3-Node Newton/Lobatto Basis evaluation (CL-13).
///
/// Evaluates: `[u - u1, (u - u1)(u - u2), (u - u1)(u - u2)(u - u3), s_sw]`.
#[inline(always)]
pub fn eval_lobatto_basis_3node(
    u: f32,
    nodes: &[f32; 4],
    degree: usize,
    switch_opt: f32,
) -> [f32; 4] {
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
    psi[3] = switch_opt;
    psi
}

/// Chebyshev 4-harmonic LFO polynomial waveshaping.
///
/// `T1(x) = x`
/// `T2(x) = 2x^2 - 1`
/// `T3(x) = 4x^3 - 3x`
/// `T4(x) = 8x^4 - 8x^2 + 1`
#[inline(always)]
pub fn eval_chebyshev_4harm(x: f32) -> [f32; 4] {
    let x2 = x * x;
    let t1 = x;
    let t2 = 2.0 * x2 - 1.0;
    let t3 = 4.0 * x2 * x - 3.0 * x;
    let t4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0;
    [t1, t2, t3, t4]
}

/// Ellipsoidal Trust-Region Norm Soft-Clipping (`CL-15`, `CL-17`, `0 FDIV`, `0 FSQRT`).
///
/// Applies the $C^1$-smooth cubic trust-region soft-clipper
/// $S_3(v) = v_c (1.5 - 0.5 v_c^2)$ with $v_c = \min(\frac{2}{3} \|\mathbf{z}\| / \rho_{\max}, 1.0)$,
/// preserving unity small-signal slope ($S_3'(0) = 1.0$) while strictly bounding $\|\mathbf{z}\| \le \rho_{\max}$.
#[inline(always)]
pub fn trust_region_softclip_4d(z: &mut [f32; 4], rho_max: f32) {
    let norm_sq = z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    if norm_sq > 1.0e-24 && rho_max > 1.0e-12 {
        let inv_norm = crate::operators::op_m::fast_rsqrt_nr2(norm_sq);
        let norm = norm_sq * inv_norm;
        let inv_rho = crate::operators::op_m::fast_rsqrt_nr2(rho_max * rho_max);
        let u_norm = (norm * (2.0 / 3.0) * inv_rho).min(1.0);
        let scale = rho_max * u_norm * (1.5 - 0.5 * u_norm * u_norm) * inv_norm;
        z[0] *= scale;
        z[1] *= scale;
        z[2] *= scale;
        z[3] *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_poly_adaa_and_wavefolder() {
        // When u_prev == u_curr == m (d == 0), poly_adaa_algebraic_eval reduces to the instantaneous
        // polynomial beta * m * (1.5 - 0.5 * m^2) + delta * (m^2 - m^4) with zero 0/0 division.
        for &m in &[-0.8_f32, -0.4, 0.0, 0.3, 0.9] {
            let beta = 1.2_f32;
            let delta = -0.35_f32;
            let actual = poly_adaa_algebraic_eval(m, m, beta, delta);
            let expected = beta * m * (1.5 - 0.5 * m * m) + delta * (m * m - m * m * m * m);
            assert!((actual - expected).abs() < 1e-6);
        }

        // At symmetric zero-crossing u_prev = -u0, u_curr = +u0: m == 0 so A_odd == 0 and A_even == d^2/3 - d^4/5.
        let d = 0.6_f32;
        let a_sym = poly_adaa_algebraic_eval(-d, d, 1.0, 1.0);
        let expected_sym = (d * d) / 3.0 - 0.2 * d * d * d * d;
        assert!((a_sym - expected_sym).abs() < 1e-6);

        // C2 wavefolder exact values at u = 0.5: (1 - 0.25)^2 = (0.75)^2 = 0.5625
        let (w_odd, w_even) = c2_wavefolder(0.5);
        assert!((w_odd - 0.5 * 0.5625).abs() < 1e-6);
        assert!((w_even - 0.25 * 0.5625).abs() < 1e-6);
    }

    #[test]
    fn test_quadrature_hysteresis_2d() {
        let _audio = crate::FtzDazGuard::new();
        let mut hyst = QuadratureHysteresis2D::default();
        assert_eq!(hyst.kappa_quad, 0.15);
        hyst.reset();

        // Step 1 from prev_s_re = 0.0: s_re * (s_re - 0.0) + 0.15 * s_re * s_im
        let h1 = hyst.step(0.5, 0.2);
        let expected_h1 = 0.5 * 0.5 + 0.15 * 0.5 * 0.2;
        assert!((h1 - expected_h1).abs() < 1e-6);
        assert_eq!(hyst.prev_s_re, 0.5);

        // Step 2 from prev_s_re = 0.5: 0.8 * (0.8 - 0.5) + 0.15 * 0.8 * (-0.1)
        let h2 = hyst.step(0.8, -0.1);
        let expected_h2 = 0.8 * 0.3 + 0.15 * 0.8 * (-0.1);
        assert!((h2 - expected_h2).abs() < 1e-6);

        hyst.reset();
        assert_eq!(hyst.prev_s_re, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher() {
        let _audio = crate::FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.reset();
        // 4 bits => levels = 1 << 3 = 8.0, step_size = 0.125; target_sr = 12 kHz at 48 kHz => phase_inc = 0.25 (holds 4 samples)
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert!((cr.step_size - 0.125).abs() < 1e-6);
        assert!((cr.phase_inc - 0.25).abs() < 1e-6);

        // First 3 samples hold initial held_sample (0.0), 4th sample triggers ZOH latch & quantizes input
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        let latched = cr.step(0.5);
        let expected_q = (0.5 * 8.0_f32 + 0.5).floor() * 0.125;
        assert!((latched - expected_q).abs() < 1e-6);
        // Next sample continues holding latched value
        assert_eq!(cr.step(0.9), latched);

        cr.reset();
        assert_eq!(cr.phase, 0.0);
        assert_eq!(cr.held_sample, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher_full_and_zero_rate() {
        let _audio = crate::FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.set_parameters(4, 48_000.0, 48_000.0);
        for phase in [0.0_f32, 0.1, 0.25, f32::from_bits(1.0_f32.to_bits() - 1)] {
            cr.phase = phase;
            for sample in [0.125, -0.25, 0.5, -0.75] {
                assert_eq!(cr.step(sample), sample);
                assert_eq!(cr.phase.to_bits(), phase.to_bits());
            }
        }

        cr.phase = 0.25;
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.phase, 0.25);
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), 0.5);

        cr.reset();
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.phase, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher_quantization_boundaries() {
        let _audio = crate::FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        for bits in 2..=16 {
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            for midpoint in [-1.5_f32, -0.5, 0.5, 1.5] {
                let encoded = midpoint.to_bits();
                for scaled in [
                    f32::from_bits(encoded - 1),
                    midpoint,
                    f32::from_bits(encoded + 1),
                ] {
                    let input = scaled * cr.step_size;
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }

            for input in [-100.0, 100.0, -f32::MAX, f32::MAX] {
                assert_eq!(cr.step(input), input, "bits={bits}, input={input}");
            }

            let threshold = 8_388_608.0 * cr.step_size;
            for value in [
                f32::from_bits(threshold.to_bits() - 1),
                threshold,
                f32::from_bits(threshold.to_bits() + 1),
            ] {
                for input in [value, -value] {
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_zoh_bitcrusher_quantization_against_f64(
            bits in 2u32..=16,
            input in -100.0_f32..100.0,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut cr = ZohBitcrusher::default();
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
            prop_assert_eq!(cr.step(input), expected);
        }
    }

    #[test]
    fn test_multi_drive_poly_adaa_channel() {
        let _audio = crate::FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        chan.reset();
        // Verify pure linear bypass when beta = 0, delta = 0: out[i] == alpha * x across all 4 lanes
        let out_lin = chan.step(0.4, 0.0, 0.0, 0.75, 0.9993);
        for &v in &out_lin {
            assert!((v - 0.3).abs() < 1e-6);
        }

        // Verify higher drive lanes saturate earlier than low drive lanes for nonlinear path (alpha = 0)
        chan.reset();
        let out_nl = chan.step(0.2, 1.0, 0.0, 0.0, 0.9993);
        assert!(out_nl[0] < out_nl[1]);
        assert!(out_nl[1] < out_nl[2]);
        assert!(out_nl[2] < out_nl[3]);

        chan.reset();
        assert_eq!(chan.u_prev, [0.0; 4]);
    }

    #[test]
    fn test_lobatto_chebyshev_and_trust_region() {
        let nodes = [0.1, 0.3, 0.7, 0.9];
        // Node vanishing property: psi[0] == 0 at nodes[0], psi[1] == 0 at nodes[1], psi[2] == 0 at nodes[2]
        let at_n0 = eval_lobatto_basis_3node(nodes[0], &nodes, 3, 1.0);
        assert_eq!(at_n0, [0.0, 0.0, 0.0, 1.0]);
        let at_n1 = eval_lobatto_basis_3node(nodes[1], &nodes, 3, 0.0);
        assert!((at_n1[0] - 0.2).abs() < 1e-6);
        assert!(at_n1[1].abs() < 1e-7);
        assert!(at_n1[2].abs() < 1e-7);
        let at_n2 = eval_lobatto_basis_3node(nodes[2], &nodes, 3, 0.0);
        assert!(at_n2[2].abs() < 1e-7);

        // Chebyshev exact trigonometric identity: T_k(cos(theta)) == cos(k * theta)
        let theta = 0.6_f32;
        let cheb = eval_chebyshev_4harm(theta.cos());
        assert!((cheb[0] - theta.cos()).abs() < 1e-6);
        assert!((cheb[1] - (2.0 * theta).cos()).abs() < 1e-6);
        assert!((cheb[2] - (3.0 * theta).cos()).abs() < 1e-6);
        assert!((cheb[3] - (4.0 * theta).cos()).abs() < 1e-5);

        let mut z = [10.0, 10.0, 10.0, 10.0];
        trust_region_softclip_4d(&mut z, 2.0);
        let norm = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
        assert!((norm - 2.0).abs() < 1e-3);

        let mut z_in = [0.1, 0.1, 0.1, 0.1];
        trust_region_softclip_4d(&mut z_in, 2.0);
        for &v in &z_in {
            assert!((v - 0.1).abs() < 1e-3);
        }
    }

    #[test]
    fn test_c2_wavefolder_smoothness_and_rails() {
        // C2 compact support: w_odd(+-1) == 0, w_even(+-1) == 0
        let (w_odd_pos, w_even_pos) = c2_wavefolder(1.0);
        assert!((w_odd_pos).abs() < 1e-7);
        assert!((w_even_pos).abs() < 1e-7);

        let (w_odd_neg, w_even_neg) = c2_wavefolder(-1.0);
        assert!((w_odd_neg).abs() < 1e-7);
        assert!((w_even_neg).abs() < 1e-7);

        // Derivative vanishes at rails: w'(1 - eps) -> 0
        let eps = 1e-4_f32;
        let (w_near, _) = c2_wavefolder(1.0 - eps);
        let deriv = (w_near - w_odd_pos) / (-eps);
        assert!(deriv.abs() < 1e-3);
    }

    #[test]
    fn test_poly_adaa_dc_equilibrium_startup() {
        let _audio = crate::FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        let b_conv = 0.2_f32;
        chan.init_dc_equilibrium(b_conv, 1.0, 0.25);

        // First step with quiescent bias should produce near-zero DC pop in nonlinear path
        let out = chan.step(b_conv, 1.0, 0.25, 0.0, 0.9993);
        for &val in &out {
            assert!(
                val.abs() < 1e-4,
                "DC pop detected at startup from quiescent state: {}",
                val
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_poly_adaa_algebraic_symmetry_and_quotient(
            u0 in -0.95f32..0.95f32,
            u1 in -0.95f32..0.95f32,
            beta in 0.1f32..2.0f32,
            delta in -0.5f32..0.5f32,
        ) {
            let a01 = poly_adaa_algebraic_eval(u0, u1, beta, delta);
            let a10 = poly_adaa_algebraic_eval(u1, u0, beta, delta);
            prop_assert!(a01.is_finite());
            prop_assert!((a01 - a10).abs() < 1e-6);

            let diff = (u1 as f64) - (u0 as f64);
            if diff.abs() > 1e-3 {
                let antideriv = |u: f64| -> f64 {
                    let u2 = u * u;
                    let u3 = u2 * u;
                    let u4 = u2 * u2;
                    let u5 = u4 * u;
                    (beta as f64) * (0.75 * u2 - 0.125 * u4)
                        + (delta as f64) * ((1.0 / 3.0) * u3 - 0.2 * u5)
                };
                let exact = (antideriv(u1 as f64) - antideriv(u0 as f64)) / diff;
                prop_assert!((a01 as f64 - exact).abs() < 1e-5);
            }
        }

        #[test]
        fn prop_c2_wavefolder_bounds_and_symmetry(u in -2.0f32..2.0f32) {
            let (w_odd, w_even) = c2_wavefolder(u);
            let (w_odd_neg, w_even_neg) = c2_wavefolder(-u);
            prop_assert!(w_odd.abs() <= 1.0001);
            prop_assert!(w_even.abs() <= 1.0001);
            prop_assert!((w_odd + w_odd_neg).abs() < 1e-6);
            prop_assert!((w_even - w_even_neg).abs() < 1e-6);
            if u.abs() >= 1.0 {
                prop_assert_eq!(w_odd, 0.0);
                prop_assert_eq!(w_even, 0.0);
            }
        }

        #[test]
        fn prop_trust_region_softclip_4d_cl15_bounds(
            z0 in -8.0f32..8.0f32,
            z1 in -8.0f32..8.0f32,
            z2 in -8.0f32..8.0f32,
            z3 in -8.0f32..8.0f32,
            rho_max in 0.5f32..5.0f32,
        ) {
            let mut z = [z0, z1, z2, z3];
            let orig_norm = (z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3).sqrt();
            trust_region_softclip_4d(&mut z, rho_max);
            let new_norm = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
            prop_assert!(new_norm <= rho_max + 1e-4);
            if orig_norm <= 0.45 * rho_max {
                prop_assert!((new_norm - orig_norm).abs() < 2e-2 * rho_max);
            }
        }
    }
}
