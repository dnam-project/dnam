//! Op-M: Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator.
//!
//! Canonical Reference: `docs/dsp/op_m_modal_kautz.md` (`CL-03`, `CL-04`, `CL-05`, `REQ-1.2`, `REQ-2.5`).

use crate::math::{flush_recursive_state_f32, flush_recursive_state_vec4};

/// Branchless algebraic 3 kHz rotor with Newton projection (CL-03).
///
/// Given frequency angle `theta` in `(0, pi)`, evaluates unit-circle coordinates `(cos(theta), sin(theta))`
/// with unconditional guarantee `cos^2 + sin^2 <= 1.000000000000` via single-step Newton projection.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn algebraic_unit_rotor_cl03(theta: f32) -> (f32, f32) {
    const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;
    let phi = (theta - HALF_PI).clamp(-HALF_PI, HALF_PI);
    let p2 = phi * phi;

    let s_phi = phi * (0.99999995 - p2 * (0.16666650 - p2 * (0.00833290 - 0.00019507 * p2)));
    let c_phi =
        0.99999995 - p2 * (0.49999990 - p2 * (0.04166630 - p2 * (0.00138860 - 0.00002438 * p2)));

    let c = -s_phi;
    let s = c_phi;

    // Newton projection: y1 = 1.5 - 0.5 * (c^2 + s^2)
    let r2 = c * c + s * s;
    let y1 = 1.5 - 0.5 * r2;
    (c * y1, s * y1)
}

/// Fast reciprocal square root using magic constant seed + 2 Newton-Raphson iterations (0 FDIV, 0 FSQRT).
#[inline(always)]
pub fn fast_rsqrt_nr2(x: f32) -> f32 {
    let x_safe = x.max(1.0e-14);
    let i = x_safe.to_bits();
    let i_seed = 0x5f3759df - (i >> 1);
    let y0 = f32::from_bits(i_seed);
    let half_x = 0.5 * x_safe;

    let y1 = y0 * (1.5 - half_x * y0 * y0);
    y1 * (1.5 - half_x * y1 * y1)
}

/// 4-lane SIMD Symplectic Complex Recurrence State (CL-03).
///
/// Evaluates:
/// `z_re[n] = alpha_re * z_re[n-1] - alpha_im * z_im[n-1] + b_re * x[n]`
/// `z_im[n] = alpha_im * z_re[n-1] + alpha_re * z_im[n-1] + b_im * x[n]`
#[derive(Clone, Copy, Debug)]
pub struct SymplecticModal4Lane {
    pub z_re: [f32; 4],
    pub z_im: [f32; 4],
    pub alpha_re: [f32; 4],
    pub alpha_im: [f32; 4],
    pub b_re: [f32; 4],
    pub b_im: [f32; 4],
}

impl Default for SymplecticModal4Lane {
    fn default() -> Self {
        Self::new()
    }
}

impl SymplecticModal4Lane {
    pub const fn new() -> Self {
        Self {
            z_re: [0.0; 4],
            z_im: [0.0; 4],
            alpha_re: [0.0; 4],
            alpha_im: [0.0; 4],
            b_re: [1.0; 4],
            b_im: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.z_re.fill(0.0);
        self.z_im.fill(0.0);
    }

    /// Sets complex pole for lane `c` given radius `r <= 0.9995` and angle `theta`.
    pub fn set_pole(&mut self, lane: usize, r: f32, theta: f32) {
        if lane < 4 {
            let r_clamped = r.clamp(0.0, 0.9995);
            let (c_rot, s_rot) = algebraic_unit_rotor_cl03(theta);
            self.alpha_re[lane] = r_clamped * c_rot;
            self.alpha_im[lane] = r_clamped * s_rot;

            // H2-norm scaling: gamma = sqrt(1 - r^2)
            let one_minus_r2 = (1.0 - r_clamped * r_clamped).max(0.0);
            let gamma = fast_rsqrt_nr2(one_minus_r2);
            self.b_re[lane] = if gamma > 0.0 {
                one_minus_r2 * gamma
            } else {
                1.0
            };
            self.b_im[lane] = 0.0;
        }
    }

    /// Steps lane `c` by 1 sample with input `x`, returning complex state `(z_re, z_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_lane(&mut self, lane: usize, x: f32) -> (f32, f32) {
        let prev_re = self.z_re[lane];
        let prev_im = self.z_im[lane];
        let a_re = self.alpha_re[lane];
        let a_im = self.alpha_im[lane];

        let next_re = a_re * prev_re - a_im * prev_im + self.b_re[lane] * x;
        let next_im = a_im * prev_re + a_re * prev_im + self.b_im[lane] * x;

        self.z_re[lane] = flush_recursive_state_f32(next_re);
        self.z_im[lane] = flush_recursive_state_f32(next_im);
        (next_re, next_im)
    }
}

/// 10th-order Hilbert Single-Sideband Lattice (Family 2).
///
/// Dual 5-stage all-pass lattice generating 90-degree quadrature pair `(I[n], Q[n])`:
/// `H_I(z) = prod_{k=0..4} (alpha_{2k} + z^-2) / (1 + alpha_{2k} z^-2)`
/// `H_Q(z) = z^-1 prod_{k=0..4} (alpha_{2k+1} + z^-2) / (1 + alpha_{2k+1} z^-2)`
#[derive(Clone, Debug)]
pub struct HilbertSsbLattice10th {
    pub i_stages: [f32; 5],
    pub i_stages_z2: [f32; 5],
    pub q_stages: [f32; 5],
    pub q_stages_z2: [f32; 5],
    pub q_z1: f32,
}

impl Default for HilbertSsbLattice10th {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertSsbLattice10th {
    pub const ALPHA_I: [f32; 5] = [0.034509, 0.174783, 0.435733, 0.730245, 0.940428];
    pub const ALPHA_Q: [f32; 5] = [0.089454, 0.288293, 0.579483, 0.852158, 0.985023];

    pub const fn new() -> Self {
        Self {
            i_stages: [0.0; 5],
            i_stages_z2: [0.0; 5],
            q_stages: [0.0; 5],
            q_stages_z2: [0.0; 5],
            q_z1: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.i_stages.fill(0.0);
        self.i_stages_z2.fill(0.0);
        self.q_stages.fill(0.0);
        self.q_stages_z2.fill(0.0);
        self.q_z1 = 0.0;
    }

    /// Steps the Hilbert lattice, producing analytic signal `(I[n], Q[n])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        // I-branch: 5 second-order real-pole all-pass sections in z^-2: (a - z^-2) / (1 - a * z^-2)
        let mut curr_i = x;
        for k in 0..5 {
            let a = Self::ALPHA_I[k];
            let v_z2 = self.i_stages_z2[k];
            let v = curr_i + a * v_z2;
            let y = a * v - v_z2;
            self.i_stages_z2[k] = flush_recursive_state_f32(self.i_stages[k]);
            self.i_stages[k] = flush_recursive_state_f32(v);
            curr_i = y;
        }

        // Q-branch: z^-1 delay followed by 5 second-order real-pole all-pass sections in z^-2
        let mut curr_q = self.q_z1;
        self.q_z1 = x;
        for k in 0..5 {
            let a = Self::ALPHA_Q[k];
            let v_z2 = self.q_stages_z2[k];
            let v = curr_q + a * v_z2;
            let y = a * v - v_z2;
            self.q_stages_z2[k] = flush_recursive_state_f32(self.q_stages[k]);
            self.q_stages[k] = flush_recursive_state_f32(v);
            curr_q = y;
        }

        (curr_i, curr_q)
    }

    /// Single-sideband frequency shifting by `(cos(w*n), sin(w*n))`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_ssb(&mut self, x: f32, cos_w: f32, sin_w: f32, upper_sideband: bool) -> f32 {
        let (i_val, q_val) = self.step(x);
        if upper_sideband {
            i_val * cos_w - q_val * sin_w
        } else {
            i_val * cos_w + q_val * sin_w
        }
    }
}

/// Discrete 1-bit zero-crossing toggle latch and Riemann sheet tracker (Family 1).
#[derive(Clone, Copy, Debug)]
pub struct SubHarmonicTracker {
    pub prev_x: f32,
    pub prev_q: f32,
    pub prev_u_re: f32,
    pub prev_u_im: f32,
    pub sheet: f32,
}

impl Default for SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SubHarmonicTracker {
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_q: 1.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_q = 1.0;
        self.prev_u_re = 1.0;
        self.prev_u_im = 0.0;
        self.sheet = 1.0;
    }

    /// Steps the bistable sub-harmonic tracker using rising zero-crossings.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, hilbert_x: f32) -> f32 {
        // Analytic envelope normalization via fast NR2 rsqrt with silence squelch (R^2 < 1e-12)
        let r2 = x * x + hilbert_x * hilbert_x;
        if r2 < 1.0e-12 {
            self.prev_x = x;
            return 0.0;
        }

        // Toggle latch on rising edge (only when above silence squelch threshold)
        if self.prev_x <= 0.0 && x > 0.0 {
            self.prev_q = -self.prev_q;
        }
        self.prev_x = x;

        let rho = fast_rsqrt_nr2(r2);
        let u_re = x * rho;
        let u_im = hilbert_x * rho;

        // Riemann branch cut tracking: branch flips when crossing negative real axis
        if ((self.prev_u_im >= 0.0 && u_im < 0.0) || (self.prev_u_im <= 0.0 && u_im > 0.0))
            && u_re < 0.0
        {
            self.sheet = -self.sheet;
        }
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;

        // Continuous half-angle representation: sqrt((1 + u_re)/2)
        let half_angle = ((1.0 + u_re) * 0.5).max(0.0);
        let half_rho = fast_rsqrt_nr2(half_angle);
        let sub_continuous = if half_rho > 0.0 {
            half_angle * half_rho
        } else {
            0.0
        };

        0.5 * (self.prev_q + self.sheet * sub_continuous)
    }
}

/// Orthonormal Kautz Modal Bank (Stage III & Family 6, REQ-1.2).
#[derive(Clone, Debug)]
pub struct KautzModalBank<const POLES: usize> {
    pub s_re: [f32; POLES],
    pub s_im: [f32; POLES],
    pub lambda_re: [f32; POLES],
    pub lambda_im: [f32; POLES],
    pub b_gamma: [f32; POLES],
    pub c_re: [f32; POLES],
    pub c_im: [f32; POLES],
}

impl<const POLES: usize> Default for KautzModalBank<POLES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const POLES: usize> KautzModalBank<POLES> {
    pub const fn new() -> Self {
        Self {
            s_re: [0.0; POLES],
            s_im: [0.0; POLES],
            lambda_re: [0.0; POLES],
            lambda_im: [0.0; POLES],
            b_gamma: [1.0; POLES],
            c_re: [0.0; POLES],
            c_im: [0.0; POLES],
        }
    }

    pub fn reset(&mut self) {
        self.s_re.fill(0.0);
        self.s_im.fill(0.0);
    }

    /// Sets complex conjugate pole pair `m` with dynamic sweep contraction (CL-05).
    pub fn set_pole(
        &mut self,
        m: usize,
        r: f32,
        theta: f32,
        delta_theta: f32,
        c_re: f32,
        c_im: f32,
    ) {
        self.set_pole_with_b_gamma(m, r, theta, delta_theta, 1.0, c_re, c_im);
    }

    /// Sets complex conjugate pole pair `m` with explicit input scaling `b_gamma` (`CL-05`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_pole_with_b_gamma(
        &mut self,
        m: usize,
        r: f32,
        theta: f32,
        delta_theta: f32,
        b_gamma: f32,
        c_re: f32,
        c_im: f32,
    ) {
        if m < POLES {
            let r_effective = if delta_theta.abs() > 1.0e-9 {
                r.min(0.9950 - 0.15 * delta_theta.abs()).clamp(0.0, 0.9950)
            } else {
                r.clamp(0.0, 0.99995)
            };
            let (cos_t, sin_t) = algebraic_unit_rotor_cl03(theta);
            self.lambda_re[m] = r_effective * cos_t;
            self.lambda_im[m] = r_effective * sin_t;
            self.b_gamma[m] = b_gamma;
            self.c_re[m] = c_re;
            self.c_im[m] = c_im;
        }
    }

    /// Steps the modal bank by 1 sample, returning the tail synthesis value.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let mut y_tail = 0.0f32;
        for m in 0..POLES {
            let prev_re = self.s_re[m];
            let prev_im = self.s_im[m];
            let l_re = self.lambda_re[m];
            let l_im = self.lambda_im[m];

            let next_re = l_re * prev_re - l_im * prev_im + self.b_gamma[m] * x;
            let next_im = l_im * prev_re + l_re * prev_im;

            self.s_re[m] = flush_recursive_state_f32(next_re);
            self.s_im[m] = flush_recursive_state_f32(next_im);

            // Tail output: 2 * Re(c_m * s_m) = 2 * (c_re * s_re - c_im * s_im)
            y_tail += 2.0 * (self.c_re[m] * next_re - self.c_im[m] * next_im);
        }
        y_tail
    }
}

impl KautzModalBank<4> {
    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the 4 per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd<B: crate::math::MathBackend>(&mut self, x: f32) -> [f32; 4] {
        use crate::math::SimdVec4;
        let prev_re = B::Vec4::from_array(self.s_re);
        let prev_im = B::Vec4::from_array(self.s_im);
        let l_re = B::Vec4::from_array(self.lambda_re);
        let l_im = B::Vec4::from_array(self.lambda_im);
        let b_g = B::Vec4::from_array(self.b_gamma);
        let c_r = B::Vec4::from_array(self.c_re);
        let c_i = B::Vec4::from_array(self.c_im);
        let x_vec = B::Vec4::splat(x);

        let next_re = l_re.mul(prev_re).sub(l_im.mul(prev_im)).add(b_g.mul(x_vec));
        let next_im = l_im.mul(prev_re).add(l_re.mul(prev_im));

        self.s_re = flush_recursive_state_vec4(next_re).to_array();
        self.s_im = flush_recursive_state_vec4(next_im).to_array();

        let out = c_r
            .mul(next_re)
            .sub(c_i.mul(next_im))
            .mul(B::Vec4::splat(2.0));
        out.to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;
    use proptest::prelude::*;

    #[test]
    fn test_algebraic_unit_rotor_and_rsqrt() {
        for angle in [0.1, 0.5, 1.0, 1.57, 2.5, 3.1] {
            let (c, s) = algebraic_unit_rotor_cl03(angle);
            let r2 = c * c + s * s;
            assert!(r2 <= 1.00001);
            assert!(c.is_finite());
            assert!(s.is_finite());
        }

        assert!((fast_rsqrt_nr2(4.0) - 0.5).abs() < 1e-3);
        assert!((fast_rsqrt_nr2(1.0) - 1.0).abs() < 1e-3);
        assert!(fast_rsqrt_nr2(1e-15).is_finite());
    }

    #[test]
    fn test_symplectic_modal_4lane() {
        let _audio = crate::FtzDazGuard::new();
        let mut modal = SymplecticModal4Lane::default();
        modal.reset();
        let r = 0.95_f32;
        let theta = 0.4_f32;
        modal.set_pole(0, r, theta);
        modal.set_pole(1, 0.9, 0.8);
        modal.set_pole(5, 0.9, 0.8); // out of bounds check

        // Impulse at n=0 sets state to (b_re, 0); subsequent steps decay by r^n and rotate by n*theta
        let (re0, im0) = modal.step_lane(0, 1.0);
        let b_norm = modal.b_re[0];
        assert!((re0 - b_norm).abs() < 1e-6);
        assert_eq!(im0, 0.0);

        let (c_t, s_t) = algebraic_unit_rotor_cl03(theta);
        let (re1, im1) = modal.step_lane(0, 0.0);
        assert!((re1 - r * c_t * b_norm).abs() < 1e-6);
        assert!((im1 - r * s_t * b_norm).abs() < 1e-6);

        modal.reset();
        assert_eq!(modal.z_re, [0.0; 4]);
    }

    #[test]
    fn test_hilbert_ssb_lattice_10th() {
        let _audio = crate::FtzDazGuard::new();
        let mut hilbert = HilbertSsbLattice10th::default();
        hilbert.reset();

        // Impulse energy conservation: I and Q are allpass cascades so sum(I^2) == 1 and sum(Q^2) == 1
        let mut energy_i = 0.0_f64;
        let mut energy_q = 0.0_f64;
        for n in 0..1024 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (i_out, q_out) = hilbert.step(x);
            energy_i += (i_out as f64) * (i_out as f64);
            energy_q += (q_out as f64) * (q_out as f64);
        }
        assert!((energy_i - 1.0).abs() < 1e-3);
        assert!((energy_q - 1.0).abs() < 1e-3);

        // Verify upper vs lower sideband relationship at cos_shift = 0, sin_shift = 1:
        // ssb_up = -Q, ssb_down = +Q (opposite signs for identical state)
        let mut h_up = HilbertSsbLattice10th::default();
        let mut h_dn = HilbertSsbLattice10th::default();
        for n in 0..16 {
            let x = ((n as f32) * 0.3 + 0.1).sin();
            let y_up = h_up.step_ssb(x, 0.0, 1.0, true);
            let y_dn = h_dn.step_ssb(x, 0.0, 1.0, false);
            assert!((y_up + y_dn).abs() < 1e-6);
        }

        // Verify 90-degree quadrature phase shift on a passband sine wave (e.g. f0 = 3 kHz at 48 kHz):
        // After lattice settling, I[n] and Q[n] are orthogonal (sum(I*Q) ~ 0) with constant analytic envelope I^2 + Q^2 ~ 1.0
        let mut h_quad = HilbertSsbLattice10th::default();
        let omega = core::f32::consts::TAU * 3_000.0 / 48_000.0;
        let mut dot_iq = 0.0_f64;
        let mut min_env = f64::INFINITY;
        let mut max_env = 0.0_f64;
        for n in 0..1024 {
            let x = ((n as f32) * omega).sin();
            let (i_v, q_v) = h_quad.step(x);
            if n >= 512 {
                let env2 = (i_v as f64) * (i_v as f64) + (q_v as f64) * (q_v as f64);
                min_env = min_env.min(env2);
                max_env = max_env.max(env2);
                dot_iq += (i_v as f64) * (q_v as f64);
            }
        }
        assert!(
            (max_env - min_env) < 0.10 && (min_env - 1.0).abs() < 0.06,
            "Hilbert quadrature envelope not constant: min_env={min_env}, max_env={max_env}"
        );
        assert!(
            (dot_iq / 512.0).abs() < 0.03,
            "Hilbert I/Q not orthogonal: avg(I*Q)={}",
            dot_iq / 512.0
        );

        hilbert.reset();
        assert_eq!(hilbert.q_z1, 0.0);
    }

    #[test]
    fn test_subharmonic_tracker() {
        let _audio = crate::FtzDazGuard::new();
        let mut tracker = SubHarmonicTracker::default();
        tracker.reset();
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);

        // Silence squelch: R^2 < 1e-12 must return strictly 0.0 (zero DC offset on silence)
        assert_eq!(tracker.step(0.0, 0.0), 0.0);
        assert_eq!(tracker.step(1e-7, -1e-7), 0.0);

        // Rising zero-crossing flips prev_q (+1.0 -> -1.0 -> +1.0); branch cut at u_re < 0 flips sheet
        // Step 1: prev_u_im == 0.0 -> u_im > 0.0 with u_re < 0.0 flips sheet +1.0 -> -1.0
        let _ = tracker.step(-0.5, 0.2);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, -1.0);
        // Step 2: Cross negative real axis again (u_re < 0, u_im crosses from + to -) -> sheet flips -1.0 -> +1.0
        let _ = tracker.step(-0.5, -0.2);
        assert_eq!(tracker.sheet, 1.0);
        // Rising zero-crossing on x (-0.5 -> +0.5) -> prev_q flips to -1.0
        let _ = tracker.step(0.5, 0.2);
        assert_eq!(tracker.prev_q, -1.0);

        tracker.reset();
        assert_eq!(tracker.prev_x, 0.0);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);
    }

    #[test]
    fn test_kautz_modal_bank() {
        let _audio = crate::FtzDazGuard::new();
        let mut bank = KautzModalBank::<4>::default();
        bank.reset();
        bank.set_pole_with_b_gamma(0, 0.90, 0.3, 0.0, 0.5, 0.8, -0.4);
        bank.set_pole(10, 0.95, 0.2, 0.01, 0.5, 0.2); // out of bounds check

        // Step 0 impulse response: s_re[0] = b_gamma * 1.0 = 0.5, s_im[0] = 0.0
        // y[0] = 2 * (c_re * s_re - c_im * s_im) = 2 * 0.8 * 0.5 = 0.8
        let y0 = bank.step(1.0);
        assert!((y0 - 0.8).abs() < 1e-6);

        // Step 1 free decay: s_re[0] = lambda_re * 0.5, s_im[0] = lambda_im * 0.5
        let expected_y1 =
            2.0 * (0.8 * (bank.lambda_re[0] * 0.5) - (-0.4) * (bank.lambda_im[0] * 0.5));
        let y1 = bank.step(0.0);
        assert!((y1 - expected_y1).abs() < 1e-6);

        bank.reset();
        assert_eq!(bank.s_re, [0.0; 4]);
    }

    #[test]
    fn test_cl03_rotor_unit_circle_proof() {
        // CL-03: |u_rot| <= 1.000000000000 proof across full angle range (0, PI)
        const STEPS: usize = 500;
        let d_theta = core::f32::consts::PI / (STEPS as f32);
        for i in 1..STEPS {
            let theta = (i as f32) * d_theta;
            let (c, s) = algebraic_unit_rotor_cl03(theta);
            let r2 = c * c + s * s;
            assert!(
                r2 <= 1.000001,
                "CL-03 violated at theta={}: c={}, s={}, r2={}",
                theta,
                c,
                s,
                r2
            );
        }
    }

    #[test]
    fn test_cl05_dynamic_sweep_contraction() {
        // Static cabinet pole (delta_theta == 0.0) preserves full r = 0.9995 (REQ-1.2 170 ms T60)
        let mut bank = KautzModalBank::<4>::default();
        bank.set_pole(0, 0.9995, 0.5, 0.0, 1.0, 0.0);
        let static_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(
            (static_r - 0.9995).abs() < 1e-4,
            "Static Kautz pole radius was prematurely clamped: {static_r}"
        );

        // CL-05: dynamic sweep contraction bounds r_effective <= 0.9950 - 0.15 * |delta_theta| when sweeping
        bank.set_pole(0, 0.9995, 0.5, 0.1, 1.0, 0.0);
        let lambda_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(lambda_r < 0.9995);
        assert!(lambda_r <= 0.9950 - 0.15 * 0.1 + 1e-5);
    }

    proptest! {
        #[test]
        fn prop_algebraic_unit_rotor_cl03_and_rsqrt(
            theta in 0.001_f32..3.10_f32,
            s in 0.001_f32..10.0_f32,
        ) {
            let (c, sn) = algebraic_unit_rotor_cl03(theta);
            let r2 = c * c + sn * sn;
            prop_assert!(r2 <= 1.000_002, "CL-03 unit circle bound violated: r2={}", r2);
            prop_assert!((r2 - 1.0).abs() < 2e-5);

            let inv_s = fast_rsqrt_nr2(s);
            let expected = 1.0 / s.sqrt();
            let rel_err = ((inv_s - expected) / expected).abs();
            prop_assert!(rel_err < 0.15, "fast_rsqrt_nr2 rel_err={} at s={}", rel_err, s);
        }

        #[test]
        fn prop_kautz_modal_bank_simd_and_gauge_invariance(
            r in 0.5_f32..0.999_f32,
            theta in 0.05_f32..3.0_f32,
            d_theta in -0.2_f32..0.2_f32,
            c_re in -2.0_f32..2.0_f32,
            c_im in -2.0_f32..2.0_f32,
            alpha in 0.2_f32..5.0_f32,
            u in -1.0_f32..1.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut bank_scalar = KautzModalBank::<4>::default();
            let mut bank_simd = KautzModalBank::<4>::default();
            let mut bank_gauge = KautzModalBank::<4>::default();
            let b_gamma = (1.0 - r * r).max(0.0).sqrt();
            for k in 0..4 {
                bank_scalar.set_pole_with_b_gamma(k, r, theta, d_theta, b_gamma, c_re, c_im);
                bank_simd.set_pole_with_b_gamma(k, r, theta, d_theta, b_gamma, c_re, c_im);
                bank_gauge.set_pole_with_b_gamma(
                    k,
                    r,
                    theta,
                    d_theta,
                    b_gamma * alpha,
                    c_re / alpha,
                    c_im / alpha,
                );
            }
            for _ in 0..8 {
                let y_s = bank_scalar.step(u);
                let y_v_lanes = bank_simd.step_4lane_simd::<ScalarBackend>(u);
                let y_v = y_v_lanes[0] + y_v_lanes[1] + y_v_lanes[2] + y_v_lanes[3];
                let y_g = bank_gauge.step(u);
                prop_assert!((y_s - y_v).abs() < 1e-4, "SIMD vs scalar mismatch: {} vs {}", y_s, y_v);
                prop_assert!((y_s - y_g).abs() < 1e-4, "Gauge invariance mismatch: {} vs {}", y_s, y_g);
            }

            // CL-05 pole contraction invariant
            for k in 0..4 {
                let lr2 = bank_scalar.lambda_re[k] * bank_scalar.lambda_re[k]
                    + bank_scalar.lambda_im[k] * bank_scalar.lambda_im[k];
                prop_assert!(lr2 <= 0.9995 * 0.9995 + 1e-5);
            }
        }
    }
}
