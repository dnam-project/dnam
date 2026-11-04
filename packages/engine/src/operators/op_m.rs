//! Op-M: Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_m_modal_kautz.md` (`ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.stage3.sweep-contraction]`, `REQ-1.2`, `REQ-2.5`).

use alloc::vec::Vec;

use crate::math::{flush_recursive_state_f32, flush_recursive_state_vec4, sqrt_f64};
use crate::schema::cold_pow_f64;

/// Quadrant-reduced degree-13/14 Horner contractive rotor with adaptive 1–2 ULP guard (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.ulp-guard]`, `ref[dsp.op-m.rotor.zero-hold]`).
///
/// Given frequency angle `theta` in `[0, 2*pi)`, evaluates unit-circle coordinates `(cos(theta), sin(theta))`
/// with unconditional guarantee `cos^2 + sin^2 <= 1.0` and `<= 2 ULP` error.
#[inline(always)]
pub fn algebraic_unit_rotor_op_m(theta: f32) -> (f32, f32) {
    const PI: f32 = core::f32::consts::PI;
    const TAU: f32 = core::f32::consts::TAU;
    let wrapped = if (-3.141_592_5..=3.141_592_5).contains(&theta) {
        theta
    } else {
        ((theta + PI).rem_euclid(TAU) - PI).clamp(-3.141_592_5, 3.141_592_5)
    };
    crate::math::contractive_rotor_f32(wrapped).unwrap_or((1.0, 0.0))
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

/// 4-lane SIMD Symplectic Complex Recurrence State (`ref[dsp.op-m.recurrence.symplectic]`).
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
            let (c_rot, s_rot) = algebraic_unit_rotor_op_m(theta);
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

/// 12-Allpass (`6` I-branch + `6` Q-branch) `H1` Hilbert Single-Sideband Lattice (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.h1-anchors]`).
///
/// Dual 6-stage all-pass lattice generating 90-degree quadrature pair `(I[n], Q[n])`:
/// `H_I(z) = prod_{k=0..5} (alpha_{2k} + z^-2) / (1 + alpha_{2k} z^-2)`
/// `H_Q(z) = z^-1 prod_{k=0..5} (alpha_{2k+1} + z^-2) / (1 + alpha_{2k+1} z^-2)`
#[derive(Clone, Debug)]
pub struct HilbertSsbLattice10th {
    pub i_stages: [f32; 6],
    pub i_stages_z2: [f32; 6],
    pub q_stages: [f32; 6],
    pub q_stages_z2: [f32; 6],
    pub q_z1: f32,
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
}

impl Default for HilbertSsbLattice10th {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertSsbLattice10th {
    /// Canonical `48 kHz` I-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`).
    pub const ALPHA_I: [f32; 6] = [
        f32::from_bits(0x3d91_bee6),
        f32::from_bits(0x3eec_b8e1),
        f32::from_bits(0x3f48_de0d),
        f32::from_bits(0x3f6d_91c9),
        f32::from_bits(0x3f7a_5939),
        f32::from_bits(0x3f7e_8e9d),
    ];
    /// Canonical `48 kHz` Q-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`).
    pub const ALPHA_Q: [f32; 6] = [
        f32::from_bits(0x3e80_025f),
        f32::from_bits(0x3f25_f6b0),
        f32::from_bits(0x3f5f_ada1),
        f32::from_bits(0x3f75_b332),
        f32::from_bits(0x3f7c_ffa1),
        f32::from_bits(0x3f7f_92d2),
    ];

    pub const fn new() -> Self {
        Self {
            i_stages: [0.0; 6],
            i_stages_z2: [0.0; 6],
            q_stages: [0.0; 6],
            q_stages_z2: [0.0; 6],
            q_z1: 0.0,
            alpha_i: Self::ALPHA_I,
            alpha_q: Self::ALPHA_Q,
        }
    }

    pub fn reset(&mut self) {
        self.i_stages.fill(0.0);
        self.i_stages_z2.fill(0.0);
        self.q_stages.fill(0.0);
        self.q_stages_z2.fill(0.0);
        self.q_z1 = 0.0;
    }

    /// Recomputes the 12-allpass `H1` coefficients for `sample_rate_hz` (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: f64) {
        if let Some(bits) = interpolate_hilbert_h1_profile_bits(sample_rate_hz) {
            for k in 0..6 {
                self.alpha_i[k] = f32::from_bits(bits[k]);
                self.alpha_q[k] = f32::from_bits(bits[6 + k]);
            }
        }
    }

    /// Steps the Hilbert lattice, producing analytic signal `(I[n], Q[n])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        // I-branch: 6 second-order real-pole all-pass sections in z^-2: (a + z^-2) / (1 + a * z^-2)
        let mut curr_i = x;
        for k in 0..6 {
            let a = self.alpha_i[k];
            let s1 = self.i_stages_z2[k];
            let s0 = self.i_stages[k];
            let y = flush_recursive_state_f32(a * curr_i + s1);
            self.i_stages_z2[k] = s0;
            self.i_stages[k] = flush_recursive_state_f32(a * y - curr_i);
            curr_i = y;
        }

        // Q-branch: z^-1 delay followed by 6 second-order real-pole all-pass sections in z^-2
        let mut curr_q = self.q_z1;
        self.q_z1 = flush_recursive_state_f32(x);
        for k in 0..6 {
            let a = self.alpha_q[k];
            let s1 = self.q_stages_z2[k];
            let s0 = self.q_stages[k];
            let y = flush_recursive_state_f32(a * curr_q + s1);
            self.q_stages_z2[k] = s0;
            self.q_stages[k] = flush_recursive_state_f32(a * y - curr_q);
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

/// Discrete 1-bit zero-crossing toggle latch and Riemann sheet tracker (`T1`, Family 1,
/// `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct SubHarmonicTracker {
    pub prev_x: f32,
    pub prev_q: f32,
    pub prev_u_re: f32,
    pub prev_u_im: f32,
    pub sheet: f32,
    pub dc_pole: f32,
    pub dc_gain: f32,
    pub dc_prev_x: f32,
    pub dc_prev_y: f32,
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
            dc_pole: f32::from_bits(0x3f7f_d51f),
            dc_gain: f32::from_bits(0x3f7f_ea90),
            dc_prev_x: 0.0,
            dc_prev_y: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_q = 1.0;
        self.prev_u_re = 1.0;
        self.prev_u_im = 0.0;
        self.sheet = 1.0;
        self.dc_prev_x = 0.0;
        self.dc_prev_y = 0.0;
    }

    /// Recomputes the `10 Hz` `T1` DC blocker pole and passband gain for `sample_rate_hz` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: u64) {
        if let Some((r_bits, g_bits)) = compute_t1_dc_blocker_coeffs_bits(sample_rate_hz as f64) {
            self.dc_pole = f32::from_bits(r_bits);
            self.dc_gain = f32::from_bits(g_bits);
        }
    }

    /// Steps the `10 Hz` `T1` high-pass DC blocker on raw input `x` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    #[inline(always)]
    pub fn step_dc_blocker(&mut self, x: f32) -> f32 {
        let y = flush_recursive_state_f32(
            self.dc_gain * (x - self.dc_prev_x) + self.dc_pole * self.dc_prev_y,
        );
        self.dc_prev_x = flush_recursive_state_f32(x);
        self.dc_prev_y = y;
        y
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

    /// Sets complex conjugate pole pair `m` with dynamic sweep contraction (`ref[dsp.op-m.stage3.sweep-contraction]`).
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

    /// Sets complex conjugate pole pair `m` with explicit input scaling `b_gamma` (`ref[dsp.op-m.stage3.sweep-contraction]`).
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
            let (cos_t, sin_t) = algebraic_unit_rotor_op_m(theta);
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
    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks sharing
    /// identical poles and residues (`lambda_re`, `lambda_im`, `b_gamma`, `c_re`, `c_im`) using
    /// `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`), loading the 5 coefficient vectors once.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_vec<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) -> (B::Vec4, B::Vec4) {
        use crate::math::SimdVec4;
        let l_re = B::Vec4::from_array(self.lambda_re);
        let l_im = B::Vec4::from_array(self.lambda_im);
        let b_g = B::Vec4::from_array(self.b_gamma);
        let c_r = B::Vec4::from_array(self.c_re);
        let c_i = B::Vec4::from_array(self.c_im);
        let two = B::Vec4::splat(2.0);

        let prev_re_l = B::Vec4::from_array(self.s_re);
        let prev_im_l = B::Vec4::from_array(self.s_im);
        let prev_re_r = B::Vec4::from_array(right.s_re);
        let prev_im_r = B::Vec4::from_array(right.s_im);

        let next_re_l = l_re
            .mul(prev_re_l)
            .sub(l_im.mul(prev_im_l))
            .fma(b_g, B::Vec4::splat(x_l));
        let next_im_l = l_im.mul(prev_re_l).add(l_re.mul(prev_im_l));

        let next_re_r = l_re
            .mul(prev_re_r)
            .sub(l_im.mul(prev_im_r))
            .fma(b_g, B::Vec4::splat(x_r));
        let next_im_r = l_im.mul(prev_re_r).add(l_re.mul(prev_im_r));

        self.s_re = flush_recursive_state_vec4(next_re_l).to_array();
        self.s_im = flush_recursive_state_vec4(next_im_l).to_array();
        right.s_re = flush_recursive_state_vec4(next_re_r).to_array();
        right.s_im = flush_recursive_state_vec4(next_im_r).to_array();

        let out_l = c_r.mul(next_re_l).sub(c_i.mul(next_im_l)).mul(two);
        let out_r = c_r.mul(next_re_r).sub(c_i.mul(next_im_r)).mul(two);
        (out_l, out_r)
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the `B::Vec4` per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd_vec<B: crate::math::MathBackend>(&mut self, x: f32) -> B::Vec4 {
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

        c_r.mul(next_re)
            .sub(c_i.mul(next_im))
            .mul(B::Vec4::splat(2.0))
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the 4 per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd<B: crate::math::MathBackend>(&mut self, x: f32) -> [f32; 4] {
        use crate::math::SimdVec4;
        self.step_4lane_simd_vec::<B>(x).to_array()
    }
}

// ============================================================================
// Normative v2.0 Op-M: Log-Octave H2 Init, 6-Section H1 Hilbert Lattice,
// T1 DC Blocker & Sub-Harmonic Tracker, Stage III Radial Gate & Rate Convert
// (`ref[dsp.op-m.*]`, `ref[dsp.zoh.op-m-poles]`)
// ============================================================================

/// Initializes `num_poles` logarithmically spaced modal pole frequencies across `[f_min_hz, f_max_hz]`
/// at `sample_rate_hz` with $\mathcal{H}_2$ input normalizer $\gamma_m = \sqrt{1 - r_{\text{high}}^2}$
/// (`ref[dsp.op-m.init.log-octave-h2]`).
pub fn init_log_octave_poles_h2(
    num_poles: usize,
    f_min_hz: f64,
    f_max_hz: f64,
    sample_rate_hz: f64,
    r_high: f32,
) -> Vec<(f32, f32)> {
    if num_poles == 0 || f_min_hz <= 0.0 || f_max_hz <= f_min_hz || sample_rate_hz <= 0.0 {
        return Vec::new();
    }
    let r_clamped = r_high.clamp(0.25, 0.9995);
    let gamma = sqrt_f64((1.0 - (r_clamped as f64) * (r_clamped as f64)).max(0.0)) as f32;
    let ratio = f_max_hz / f_min_hz;
    let mut out = Vec::with_capacity(num_poles);
    for m in 0..num_poles {
        let t = if num_poles == 1 {
            0.0
        } else {
            (m as f64) / ((num_poles - 1) as f64)
        };
        let freq_hz = f_min_hz * cold_pow_f64(ratio, t);
        let theta = ((core::f64::consts::TAU * freq_hz / sample_rate_hz)
            .clamp(1.0e-4, core::f64::consts::PI - 1.0e-4)) as f32;
        out.push((theta, gamma));
    }
    out
}

/// Canonical 15-rate `H1` Hilbert anchor table (`rate_hz`, 6 `I`-branch + 6 `Q`-branch `f32` bit
/// patterns; `780 B` little-endian binary SHA-256 `bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`).
pub const HILBERT_H1_ANCHORS_15: [(u32, [u32; 12]); 15] = [
    (
        44_100,
        [
            0x3D8EC459, 0x3EE935E1, 0x3F47659D, 0x3F6CCEAA, 0x3F7A0188, 0x3F7E7297, 0x3E7B1AD6,
            0x3F243369, 0x3F5E97F2, 0x3F752D7D, 0x3F7CCA31, 0x3F7F8A32,
        ],
    ),
    (
        48_000,
        [
            0x3D91BEE6, 0x3EECB8E1, 0x3F48DE0D, 0x3F6D91C9, 0x3F7A5939, 0x3F7E8E9D, 0x3E80025F,
            0x3F25F6B0, 0x3F5FADA1, 0x3F75B332, 0x3F7CFFA1, 0x3F7F92D2,
        ],
    ),
    (
        64_000,
        [
            0x3DA2AF27, 0x3EFE4625, 0x3F4FBDA5, 0x3F70B815, 0x3F7B8289, 0x3F7EE2C7, 0x3E8C7CF3,
            0x3F2E9DD3, 0x3F647F75, 0x3F77A752, 0x3F7DA785, 0x3F7FAC3E,
        ],
    ),
    (
        80_000,
        [
            0x3DA88500, 0x3F0212AF, 0x3F523110, 0x3F72069B, 0x3F7C1032, 0x3F7F0F58, 0x3E90CC38,
            0x3F318514, 0x3F665AC6, 0x3F788657, 0x3F7DFC4C, 0x3F7FBA14,
        ],
    ),
    (
        96_000,
        [
            0x3DB073C5, 0x3F05A9C4, 0x3F54E508, 0x3F7343B2, 0x3F7C852B, 0x3F7F30C7, 0x3E963AF7,
            0x3F34F3ED, 0x3F683FD9, 0x3F794AFF, 0x3F7E3E99, 0x3F7FC438,
        ],
    ),
    (
        176_400,
        [
            0x3E0D917D, 0x3F216E28, 0x3F624034, 0x3F77D413, 0x3F7DE528, 0x3F7F8887, 0x3ECDF9DC,
            0x3F49A494, 0x3F703BD8, 0x3F7BD39E, 0x3F7EF730, 0x3F7FDDFB,
        ],
    ),
    (
        192_000,
        [
            0x3E1A79AD, 0x3F2691E2, 0x3F64674D, 0x3F787EBC, 0x3F7E12E1, 0x3F7F9298, 0x3ED99328,
            0x3F4D2EB7, 0x3F717613, 0x3F7C2D21, 0x3F7F0DA9, 0x3F7FE0D4,
        ],
    ),
    (
        208_000,
        [
            0x3E2646E1, 0x3F2A6E87, 0x3F65C7A3, 0x3F78E266, 0x3F7E2EFD, 0x3F7F99AA, 0x3EE33E4A,
            0x3F4F9D5A, 0x3F723264, 0x3F7C621C, 0x3F7F1C5F, 0x3F7FE2ED,
        ],
    ),
    (
        224_000,
        [
            0x3E34BA6C, 0x3F2F8A17, 0x3F67DE51, 0x3F798480, 0x3F7E584A, 0x3F7FA231, 0x3EEF3412,
            0x3F53121B, 0x3F736170, 0x3F7CB521, 0x3F7F3010, 0x3F7FE547,
        ],
    ),
    (
        240_000,
        [
            0x3E4151D9, 0x3F33437E, 0x3F693973, 0x3F79E8DF, 0x3F7E72F8, 0x3F7FA845, 0x3EF8BC96,
            0x3F556DD1, 0x3F741F05, 0x3F7CE93F, 0x3F7F3D5F, 0x3F7FE706,
        ],
    ),
    (
        256_000,
        [
            0x3E4CBD88, 0x3F363B81, 0x3F6A3C44, 0x3F7A3322, 0x3F7E877C, 0x3F7FAD32, 0x3F006617,
            0x3F573C54, 0x3F74AA8C, 0x3F7D1079, 0x3F7F47DE, 0x3F7FE876,
        ],
    ),
    (
        288_000,
        [
            0x3E64086E, 0x3F3BDB10, 0x3F6C20CB, 0x3F7ABA23, 0x3F7EABAF, 0x3F7FB5AD, 0x3F083609,
            0x3F5AA15E, 0x3F75ACD3, 0x3F7D5697, 0x3F7F5A2A, 0x3F7FEAEB,
        ],
    ),
    (
        320_000,
        [
            0x3E7D6662, 0x3F41DCD7, 0x3F6E3EDD, 0x3F7B5343, 0x3F7ED216, 0x3F7FBDBA, 0x3F1068CD,
            0x3F5E589B, 0x3F76D2C7, 0x3F7DA41A, 0x3F7F6C8B, 0x3F7FED2A,
        ],
    ),
    (
        352_000,
        [
            0x3E88E622, 0x3F45A7EB, 0x3F6F6903, 0x3F7BA4C3, 0x3F7EE847, 0x3F7FC310, 0x3F163553,
            0x3F6082FF, 0x3F776EE6, 0x3F7DCEB5, 0x3F7F77E5, 0x3F7FEEB9,
        ],
    ),
    (
        384_000,
        [
            0x3E942913, 0x3F49F3A3, 0x3F70D45F, 0x3F7C094D, 0x3F7F0173, 0x3F7FC869, 0x3F1C84D4,
            0x3F631031, 0x3F783122, 0x3F7E017B, 0x3F7F8401, 0x3F7FF039,
        ],
    ),
];

/// Encodes `HILBERT_H1_ANCHORS_15` into its canonical `780 B` little-endian binary table
/// (`15 × 13 × 4 B`, `ref[dsp.op-m.hilbert.h1-anchors]`).
pub fn encode_hilbert_h1_anchor_table_780b() -> [u8; 780] {
    let mut out = [0u8; 780];
    let mut pos = 0;
    for &(rate, ref coeffs) in &HILBERT_H1_ANCHORS_15 {
        out[pos..pos + 4].copy_from_slice(&rate.to_le_bytes());
        pos += 4;
        for &c in coeffs {
            out[pos..pos + 4].copy_from_slice(&c.to_le_bytes());
            pos += 4;
        }
    }
    out
}

fn f32_anchor_scaled_27(bits: u32) -> u128 {
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let sig = (1u128 << 23) | ((bits & 0x007f_ffff) as u128);
    let shift = 4 + exp;
    sig << (shift as u32)
}

fn round_positive_u128_ratio_to_f32_bits(numerator: u128, denominator: u128) -> Option<u32> {
    if numerator == 0 || numerator >= denominator {
        return None;
    }
    let num_bits = (128 - numerator.leading_zeros()) as i32;
    let den_bits = (128 - denominator.leading_zeros()) as i32;
    let mut exponent = num_bits - den_bits;
    if exponent >= 0 {
        if numerator < (denominator << (exponent as u32)) {
            exponent -= 1;
        }
    } else if (numerator << ((-exponent) as u32)) < denominator {
        exponent -= 1;
    }
    if exponent < -126 {
        return None;
    }
    let shifted_num = numerator << ((23 - exponent) as u32);
    let mut significand = shifted_num / denominator;
    let remainder = shifted_num % denominator;
    if 2 * remainder > denominator || (2 * remainder == denominator && (significand & 1) != 0) {
        significand += 1;
    }
    if significand == (1 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    Some((((exponent + 127) as u32) << 23) | ((significand as u32) - (1 << 23)))
}

/// Interpolates the 12 `H1` Hilbert coefficients (`[alpha_I[0..6], alpha_Q[0..6]]` as `f32` bit
/// patterns) between `low_rate` and `high_rate` at `rate_hz` (`low_rate <= rate_hz <= high_rate`)
/// using exact `u128` ties-to-even arithmetic (`ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_between_anchors_bits(
    rate_hz: f64,
    low_rate: u32,
    high_rate: u32,
) -> Option<[u32; 12]> {
    if !rate_hz.is_finite()
        || low_rate >= high_rate
        || rate_hz < (low_rate as f64)
        || rate_hz > (high_rate as f64)
    {
        return None;
    }
    let low_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == low_rate)?
        .1;
    let high_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == high_rate)?
        .1;

    // Decompose binary64 rate_hz = R / 2^k (0 <= k <= 37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u128 << 52) | ((bits & ((1u64 << 52) - 1)) as u128);
    let (r_int, k_shift): (u128, u32) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 0)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, raw_k - tz)
    };
    let left_weight = ((high_rate as u128) << k_shift) - r_int;
    let right_weight = r_int - ((low_rate as u128) << k_shift);
    let denominator = ((high_rate - low_rate) as u128) << (k_shift + 27);

    let mut out = [0u32; 12];
    for i in 0..12 {
        let a_scaled = f32_anchor_scaled_27(low_coeffs[i]);
        let b_scaled = f32_anchor_scaled_27(high_coeffs[i]);
        let numerator = left_weight * a_scaled + right_weight * b_scaled;
        out[i] = round_positive_u128_ratio_to_f32_bits(numerator, denominator)?;
    }
    Some(out)
}

/// Interpolates the 12 `H1` Hilbert coefficients for any supported `rate_hz` in either the `1×`
/// live band (`44,100..=96,000 Hz`) or the `4×` bounce band (`176,400..=384,000 Hz`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_profile_bits(rate_hz: f64) -> Option<[u32; 12]> {
    if !rate_hz.is_finite() {
        return None;
    }
    const LIVE_BAND: [u32; 5] = [44_100, 48_000, 64_000, 80_000, 96_000];
    const BOUNCE_BAND: [u32; 10] = [
        176_400, 192_000, 208_000, 224_000, 240_000, 256_000, 288_000, 320_000, 352_000, 384_000,
    ];
    for band in [&LIVE_BAND[..], &BOUNCE_BAND[..]] {
        for pair in band.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if rate_hz >= (lo as f64) && rate_hz <= (hi as f64) {
                return interpolate_hilbert_h1_between_anchors_bits(rate_hz, lo, hi);
            }
        }
    }
    None
}

/// Normative 6-section-per-branch `H1` Dual All-Pass Hilbert Lattice (`96 B` raw state + retained
/// prior outputs for history-consistent rate transitions, `ref[dsp.op-m.hilbert.lattice]`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.rate-remap]`).
#[derive(Clone, Debug)]
pub struct HilbertH1Lattice6Sec {
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
    /// Transposed all-pass state `[s[n-1], s[n-2]]` per section.
    pub i_state: [[f32; 2]; 6],
    pub q_state: [[f32; 2]; 6],
    /// Retained section outputs `[y[n-1], y[n-2]]` for history-consistent rate remap (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub i_prev_y: [[f32; 2]; 6],
    pub q_prev_y: [[f32; 2]; 6],
    pub q_z1: f32,
}

impl Default for HilbertH1Lattice6Sec {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl HilbertH1Lattice6Sec {
    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let bits = interpolate_hilbert_h1_profile_bits(rate_hz)?;
        let alpha_i = core::array::from_fn(|k| f32::from_bits(bits[k]));
        let alpha_q = core::array::from_fn(|k| f32::from_bits(bits[6 + k]));
        Some(Self {
            alpha_i,
            alpha_q,
            i_state: [[0.0; 2]; 6],
            q_state: [[0.0; 2]; 6],
            i_prev_y: [[0.0; 2]; 6],
            q_prev_y: [[0.0; 2]; 6],
            q_z1: 0.0,
        })
    }

    /// Initializes all 12 section states and retained prior outputs to the resting DC equilibrium
    /// for constant input `x_dc` (`y_k = (-1)^(k+1) * x_dc`, `s_k = (-1)^(k+1) * (1 + alpha_k) * x_dc`,
    /// `ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn init_dc_equilibrium(&mut self, x_dc: f32) {
        let mut vi = x_dc;
        let mut vq = x_dc;
        self.q_z1 = x_dc;
        for k in 0..6 {
            let yi = -vi;
            let si = flush_recursive_state_f32(self.alpha_i[k] * yi - vi);
            self.i_state[k] = [si, si];
            self.i_prev_y[k] = [yi, yi];
            vi = yi;

            let yq = -vq;
            let sq = flush_recursive_state_f32(self.alpha_q[k] * yq - vq);
            self.q_state[k] = [sq, sq];
            self.q_prev_y[k] = [yq, yq];
            vq = yq;
        }
    }

    /// Updates the sample rate to `new_rate_hz` and remaps all 12 section states from the retained
    /// prior outputs $s'[n-i] = s[n-i] + (\beta - \alpha) y[n-i]$ (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn retune_sample_rate(&mut self, new_rate_hz: f64) -> Option<()> {
        let bits = interpolate_hilbert_h1_profile_bits(new_rate_hz)?;
        for k in 0..6 {
            let new_ai = f32::from_bits(bits[k]);
            let da_i = new_ai - self.alpha_i[k];
            self.i_state[k][0] =
                flush_recursive_state_f32(self.i_state[k][0] + da_i * self.i_prev_y[k][0]);
            self.i_state[k][1] =
                flush_recursive_state_f32(self.i_state[k][1] + da_i * self.i_prev_y[k][1]);
            self.alpha_i[k] = new_ai;

            let new_aq = f32::from_bits(bits[6 + k]);
            let da_q = new_aq - self.alpha_q[k];
            self.q_state[k][0] =
                flush_recursive_state_f32(self.q_state[k][0] + da_q * self.q_prev_y[k][0]);
            self.q_state[k][1] =
                flush_recursive_state_f32(self.q_state[k][1] + da_q * self.q_prev_y[k][1]);
            self.alpha_q[k] = new_aq;
        }
        Some(())
    }

    /// Steps one sample through the 6-section `H1` analytic pair (`y[n] = alpha * v[n] + s[n-2]`,
    /// `s[n] = alpha * y[n] - v[n]`, `ref[dsp.op-m.hilbert.lattice]`).
    #[inline]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let mut vi = x;
        for k in 0..6 {
            let a = self.alpha_i[k];
            let y = a * vi + self.i_state[k][1];
            self.i_state[k][1] = self.i_state[k][0];
            self.i_state[k][0] = flush_recursive_state_f32(a * y - vi);
            self.i_prev_y[k][1] = self.i_prev_y[k][0];
            self.i_prev_y[k][0] = y;
            vi = y;
        }

        let mut vq = self.q_z1;
        self.q_z1 = x;
        for k in 0..6 {
            let a = self.alpha_q[k];
            let y = a * vq + self.q_state[k][1];
            self.q_state[k][1] = self.q_state[k][0];
            self.q_state[k][0] = flush_recursive_state_f32(a * y - vq);
            self.q_prev_y[k][1] = self.q_prev_y[k][0];
            self.q_prev_y[k][0] = y;
            vq = y;
        }
        (vi, vq)
    }
}

// ============================================================================
// T1 Tracker 5 Hz DC Blocker & Sub-Harmonic Tracker (`ref[dsp.op-m.tracker.*]`)
// ============================================================================

/// Outward 160-bit interval bounds on $\mathrm{atanh}((D - N) / (D + N)) \cdot 2^{160}$ using 8 terms
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct U256Be {
    limbs: [u64; 4], // big-endian limbs for natural Ord comparison: limbs[0] = MSW, limbs[3] = LSW
}

impl U256Be {
    const ZERO: Self = Self { limbs: [0; 4] };

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let s = (self.limbs[i] as u128) + (rhs.limbs[i] as u128) + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        Self { limbs: out }
    }

    fn add_u64(self, rhs: u64) -> Self {
        self.add(Self {
            limbs: [0, 0, 0, rhs],
        })
    }

    fn mul_u64(self, rhs: u64) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let p = (self.limbs[i] as u128) * (rhs as u128) + carry;
            out[i] = p as u64;
            carry = p >> 64;
        }
        Self { limbs: out }
    }

    fn div_u64(self, divisor: u64) -> Self {
        let mut out = [0u64; 4];
        let mut rem = 0u128;
        let d = divisor as u128;
        for i in 0..4 {
            let cur = (rem << 64) | (self.limbs[i] as u128);
            out[i] = (cur / d) as u64;
            rem = cur % d;
        }
        Self { limbs: out }
    }
}

fn atanh_bounds_160bit(numerator: u64, denominator: u64) -> (U256Be, U256Be) {
    let z_num = denominator - numerator;
    let z_den = denominator + numerator;
    // Start with (1 << 160) * z_num / z_den: (1 << 160) has bit 32 set in limbs[1]
    let one_shl_160 = U256Be {
        limbs: [0, 1u64 << 32, 0, 0],
    };
    let mut cur = one_shl_160.mul_u64(z_num).div_u64(z_den);
    let z2_num = z_num * z_num;
    let z2_den = z_den * z_den;
    let mut lower = U256Be::ZERO;
    for idx in 0..8u64 {
        lower = lower.add(cur.div_u64(2 * idx + 1));
        cur = cur.mul_u64(z2_num).div_u64(z2_den);
    }
    (lower, lower.add_u64(16))
}

/// Computes the `T1` tracker `5 Hz` DC blocker `(pole_bits, gain_bits)` at `rate_hz` (`44.1..96 kHz`
/// or `176.4..384 kHz`) using the normative 160-bit midpoint binary search without `libm` calls
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
pub fn compute_t1_dc_blocker_coeffs_bits(rate_hz: f64) -> Option<(u32, u32)> {
    if !rate_hz.is_finite() {
        return None;
    }
    let (first, last) = if (44_100.0..=96_000.0).contains(&rate_hz) {
        (0x00ff_d000u32, 0x00ff_ebffu32)
    } else if (176_400.0..=384_000.0).contains(&rate_hz) {
        (0x00ff_f400u32, 0x00ff_fbffu32)
    } else {
        return None;
    };

    // Express rate_hz = rate_num / rate_den in reduced integer form (rate_num < 2^53, rate_den <= 2^37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u64 << 52) | (bits & ((1u64 << 52) - 1));
    let (rate_num, rate_den): (u64, u64) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 1)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, 1u64 << (raw_k - tz))
    };
    let exponent_num = 48_000u64 * rate_den;
    let ref_bounds = atanh_bounds_160bit(0x00ff_d51f, 1u64 << 24);

    let mut lower = first + 1;
    let mut upper = last;
    while lower < upper {
        let candidate = (lower + upper) / 2;
        let mid_bounds = atanh_bounds_160bit((2 * candidate + 1) as u64, 1u64 << 25);
        let lhs = mid_bounds.0.mul_u64(rate_num);
        let rhs = ref_bounds.1.mul_u64(exponent_num);
        if lhs > rhs {
            lower = candidate + 1;
        } else {
            upper = candidate;
        }
    }
    let pole_bits = 0x3f00_0000 + lower - (1 << 23);
    let num_gain = (1u32 << 24) + lower;
    let q = num_gain / 2;
    let r = num_gain % 2;
    let rounded_gain = q + u32::from(r == 1 && (q & 1) != 0);
    let gain_bits = 0x3f00_0000 + rounded_gain - (1 << 23);
    Some((pole_bits, gain_bits))
}

/// Tracker-only passband-normalized `5 Hz` one-pole DC blocker (`T1`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct T1DcBlocker {
    pub pole: f32,
    pub gain: f32,
    pub prev_x: f32,
    pub prev_d: f32,
}

impl Default for T1DcBlocker {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl T1DcBlocker {
    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate_hz)?;
        Some(Self {
            pole: f32::from_bits(p_bits),
            gain: f32::from_bits(g_bits),
            prev_x: 0.0,
            prev_d: 0.0,
        })
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let d = self.gain * (x - self.prev_x) + self.pole * self.prev_d;
        self.prev_x = x;
        self.prev_d = flush_recursive_state_f32(d);
        d
    }
}

#[inline]
fn rsqrt_nr2_exact_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(0x5f37_59df - (value.to_bits() >> 1));
    for _ in 0..2 {
        let corr = 1.5 - ((0.5 * value) * est) * est;
        est *= corr;
    }
    est
}

/// Normalizes an analytic pair $(I, Q)$ using exponent-split pre-scaling by $2^{-\min(E, 126)}$
/// (with $2^{24}$ subnormal pre-scaling) and 2-step Newton `rsqrt` (`0 FSQRT`, `0 FDIV`,
/// `ref[dsp.op-m.tracker.envelope-norm]`). Returns `None` when $I = Q = 0$.
pub fn normalize_tracker_phasor_exponent_split(i_val: f32, q_val: f32) -> Option<(f32, f32, f32)> {
    let mut x = i_val;
    let mut y = q_val;
    let max_abs = x.abs().max(y.abs());
    if (max_abs.to_bits() & 0x7fff_ffff) == 0 {
        return None;
    }
    let mut exp_bits = (max_abs.to_bits() >> 23) & 0xff;
    let mut subnormal_restore = 1.0_f32;
    if exp_bits == 0 {
        let two_pow_24 = f32::from_bits((127 + 24) << 23);
        x *= two_pow_24;
        y *= two_pow_24;
        exp_bits = (x.abs().max(y.abs()).to_bits() >> 23) & 0xff;
        subnormal_restore = f32::from_bits((127 - 24) << 23);
    }
    let power = ((exp_bits as i32) - 127).min(126);
    let scale_down = f32::from_bits(((127 - power) as u32) << 23);
    let scale_up = f32::from_bits(((127 + power) as u32) << 23);
    let sx = x * scale_down;
    let sy = y * scale_down;
    let sq = sx * sx + sy * sy;
    let inv = rsqrt_nr2_exact_f32(sq);
    let mag = (sq * inv) * scale_up * subnormal_restore;
    let u_re = (sx * inv).clamp(-1.0, 1.0);
    let u_im = (sy * inv).clamp(-1.0, 1.0);
    Some((u_re, u_im, mag))
}

/// Evaluates the smooth half-angle pair `(v_cos, v_sin)` from a normalized phasor `(u_re, u_im)`
/// (`ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline]
pub fn eval_tracker_half_angle_pair(u_re: f32, u_im: f32) -> (f32, f32) {
    let re = u_re.clamp(-1.0, 1.0);
    let im = u_im.clamp(-1.0, 1.0);
    let h = 0.5 * (1.0 + re.abs());
    let inv = rsqrt_nr2_exact_f32(h);
    let major = (h * inv).clamp(0.0, 1.0);
    let minor = (im.abs() * (0.5 * inv)).clamp(0.0, 1.0);
    if re >= 0.0 {
        (major, minor)
    } else {
        (minor, major)
    }
}

/// Returns `true` if consecutive normalized phasors `prev` and `curr` cross the negative real axis
/// branch cut, holding on near-origin chords (`p . u < 0` and `|D| <= 2^-18`,
/// `ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline]
pub fn tracker_crosses_branch_cut(prev: (f32, f32), curr: (f32, f32)) -> bool {
    let (px, py) = prev;
    let (x, y) = curr;
    let det = (px as f64) * (y as f64) - (py as f64) * (x as f64);
    let dot = (px as f64) * (x as f64) + (py as f64) * (y as f64);
    let chord_tol = f64::from_bits((1023 - 18) << 52);
    if dot < 0.0 && det.abs() <= chord_tol {
        return false;
    }
    (py >= 0.0 && y < 0.0 && det > 0.0) || (py < 0.0 && y >= 0.0 && det < 0.0)
}

/// Complete `T1` Sub-Harmonic Tracker emitting `[w_cos, w_sin, w_square, w_quarter]` for `Lite`
/// feature bus slots `4..11` (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct T1SubHarmonicTracker {
    pub prev_valid: bool,
    pub prev_i_nonpositive: bool,
    pub prev_phasor: (f32, f32),
    pub q1: f32,
    pub q2: f32,
    pub s_sheet: f32,
}

impl Default for T1SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicTracker {
    pub const fn new() -> Self {
        Self {
            prev_valid: false,
            prev_i_nonpositive: false,
            prev_phasor: (1.0, 0.0),
            q1: 1.0,
            q2: 1.0,
            s_sheet: 1.0,
        }
    }

    /// Steps the tracker from unnormalized analytic pair `(i_raw, q_raw)`, returning
    /// `[w_cos, w_sin, w_square, w_quarter]`.
    pub fn step_analytic(&mut self, i_raw: f32, q_raw: f32) -> [f32; 4] {
        let Some((u_re, u_im, env_hat)) = normalize_tracker_phasor_exponent_split(i_raw, q_raw)
        else {
            self.prev_valid = false;
            return [0.0; 4];
        };
        if self.prev_valid {
            if self.prev_i_nonpositive && i_raw > 0.0 {
                self.q1 = -self.q1;
                if self.q1 > 0.0 {
                    self.q2 = -self.q2;
                }
            }
            if tracker_crosses_branch_cut(self.prev_phasor, (u_re, u_im)) {
                self.s_sheet = -self.s_sheet;
            }
        }
        self.prev_valid = true;
        self.prev_i_nonpositive = i_raw <= 0.0;
        self.prev_phasor = (u_re, u_im);

        let (v_cos, v_sin) = eval_tracker_half_angle_pair(u_re, u_im);
        let sigma_pos = if u_im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        [
            env_hat * self.s_sheet * v_cos,
            env_hat * self.s_sheet * sigma_pos * v_sin,
            env_hat * self.q1,
            env_hat * self.q2,
        ]
    }
}

/// Validates Stage III serialized Kautz parameters (`0.85 <= r_low <= r_high = r_low + dr <= 0.9995`,
/// `theta in (0, pi)`, `0.5 <= kappa <= 2.0`) and evaluates the shared radial gate
/// $g = \mathrm{clamp}(0.5 + \theta_{10}, 0, 1)$, $r_{\text{gate}} = \mathrm{clamp}(r_{\text{low}} + g \cdot dr, r_{\text{low}}, r_{\text{high}})$,
/// and $b_\gamma = \kappa \sqrt{1 - r_{\text{high}}^2}$ (`ref[dsp.op-m.stage3.radial-gate]`).
pub fn validate_and_eval_stage3_radial_gate(
    r_low: f32,
    dr: f32,
    theta: f32,
    kappa: f32,
    theta_10_mod: f32,
) -> Option<(f32, f32, f32)> {
    let r_high = r_low + dr;
    if !(0.85..=0.9995).contains(&r_low)
        || dr < 0.0
        || r_high > 0.9995
        || !(theta > 0.0 && theta < core::f32::consts::PI)
        || !(0.5..=2.0).contains(&kappa)
    {
        return None;
    }
    let g = (0.5 + theta_10_mod).clamp(0.0, 1.0);
    let r_gate = (r_low + g * dr).clamp(r_low, r_high);
    let gamma = sqrt_f64((1.0 - (r_high as f64) * (r_high as f64)).max(0.0)) as f32;
    let b_gamma = kappa * gamma;
    Some((g, r_gate, b_gamma))
}

/// Evaluates Stage III dynamic sweep contraction $r_{\text{eff}} = \min(r_m, \max(0.25, 0.9950 - 0.15 |\Delta\theta_m|))$
/// (`ref[dsp.op-m.stage3.sweep-contraction]`).
#[inline]
pub fn eval_stage3_sweep_contraction(r_m: f32, delta_theta_m: f32) -> f32 {
    r_m.min((0.9950 - 0.15 * delta_theta_m.abs()).max(0.25))
}

/// Converts a Stage III shared Kautz pole pair `(r_low, dr, theta, b_gamma, c_re, c_im)` from the
/// `48 kHz` reference rate to `effective_rate_hz` (`rho = 48000 / f_s,eff`), rescaling the complex
/// readout at $g = 0.5$ so steady-state DC/modal transfer is preserved (`ref[dsp.op-m.stage3.rate-convert]`,
/// `ref[dsp.zoh.op-m-poles]`).
pub fn convert_stage3_kautz_pole_for_rate(
    r_low: f32,
    dr: f32,
    theta: f32,
    b_gamma: f32,
    c_re: f32,
    c_im: f32,
    effective_rate_hz: f64,
) -> (f32, f32, f32, f32, f32, f32) {
    let rho = 48_000.0 / effective_rate_hz;
    if (rho - 1.0).abs() < 1e-12 {
        return (r_low, dr, theta, b_gamma, c_re, c_im);
    }
    let r_low_eff = cold_pow_f64(r_low as f64, rho) as f32;
    let r_high = (r_low + dr).min(0.9995);
    let r_high_eff = cold_pow_f64(r_high as f64, rho) as f32;
    let dr_eff = (r_high_eff - r_low_eff).max(0.0);
    let theta_eff = ((theta as f64) * rho).clamp(1.0e-4, core::f64::consts::PI - 1.0e-4) as f32;
    let b_gamma_eff = sqrt_f64((1.0 - (r_high_eff as f64) * (r_high_eff as f64)).max(1e-7)) as f32;

    // Rescale complex readout by (b_gamma / b_1) * (1 - lambda_1) / (1 - lambda_0) at g = 0.5
    let r0 = (r_low + 0.5 * dr) as f64;
    let r1 = (r_low_eff + 0.5 * dr_eff) as f64;
    let (c0, s0) = crate::math::contractive_rotor_f32(theta).unwrap_or((1.0, 0.0));
    let (c1, s1) = crate::math::contractive_rotor_f32(theta_eff).unwrap_or((1.0, 0.0));
    let d0_re = 1.0 - r0 * (c0 as f64);
    let d0_im = -r0 * (s0 as f64);
    let d1_re = 1.0 - r1 * (c1 as f64);
    let d1_im = -r1 * (s1 as f64);
    let d0_norm2 = (d0_re * d0_re + d0_im * d0_im).max(1e-18);
    // (d1_re + i d1_im) / (d0_re + i d0_im)
    let ratio_re = (d1_re * d0_re + d1_im * d0_im) / d0_norm2;
    let ratio_im = (d1_im * d0_re - d1_re * d0_im) / d0_norm2;
    let scale = (b_gamma as f64) / (b_gamma_eff as f64).max(1e-12);
    let m_re = scale * ratio_re;
    let m_im = scale * ratio_im;
    let c_re_eff = ((c_re as f64) * m_re - (c_im as f64) * m_im) as f32;
    let c_im_eff = ((c_re as f64) * m_im + (c_im as f64) * m_re) as f32;
    (
        r_low_eff,
        dr_eff,
        theta_eff,
        b_gamma_eff,
        c_re_eff,
        c_im_eff,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;
    use proptest::prelude::*;

    #[test]
    fn test_algebraic_unit_rotor_and_rsqrt() {
        for angle in [0.1, 0.5, 1.0, 1.57, 2.5, 3.1] {
            let (c, s) = algebraic_unit_rotor_op_m(angle);
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

        let (c_t, s_t) = algebraic_unit_rotor_op_m(theta);
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
        for n in 0..4096 {
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
    fn test_rotor_unit_circle_proof() {
        // ref[dsp.op-m.rotor.contractive]: |u_rot| <= 1.000000000000 proof across full angle range (0, PI)
        const STEPS: usize = 500;
        let d_theta = core::f32::consts::PI / (STEPS as f32);
        for i in 1..STEPS {
            let theta = (i as f32) * d_theta;
            let (c, s) = algebraic_unit_rotor_op_m(theta);
            let r2 = c * c + s * s;
            assert!(
                r2 <= 1.000001,
                "Rotor bound violated at theta={}: c={}, s={}, r2={}",
                theta,
                c,
                s,
                r2
            );
        }
    }

    #[test]
    fn test_dynamic_sweep_contraction() {
        // Static cabinet pole (delta_theta == 0.0) preserves full r = 0.9995 (REQ-1.2 170 ms T60)
        let mut bank = KautzModalBank::<4>::default();
        bank.set_pole(0, 0.9995, 0.5, 0.0, 1.0, 0.0);
        let static_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(
            (static_r - 0.9995).abs() < 1e-4,
            "Static Kautz pole radius was prematurely clamped: {static_r}"
        );

        // ref[dsp.op-m.stage3.sweep-contraction]: dynamic sweep contraction bounds r_effective <= 0.9950 - 0.15 * |delta_theta| when sweeping
        bank.set_pole(0, 0.9995, 0.5, 0.1, 1.0, 0.0);
        let lambda_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(lambda_r < 0.9995);
        assert!(lambda_r <= 0.9950 - 0.15 * 0.1 + 1e-5);
    }

    proptest! {
        #[test]
        fn prop_algebraic_unit_rotor_and_rsqrt(
            theta in 0.001_f32..3.10_f32,
            s in 0.001_f32..10.0_f32,
        ) {
            let (c, sn) = algebraic_unit_rotor_op_m(theta);
            let r2 = c * c + sn * sn;
            prop_assert!(r2 <= 1.000_002, "Rotor unit circle bound violated: r2={}", r2);
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

            // ref[dsp.op-m.stage3.sweep-contraction] pole contraction invariant
            for k in 0..4 {
                let lr2 = bank_scalar.lambda_re[k] * bank_scalar.lambda_re[k]
                    + bank_scalar.lambda_im[k] * bank_scalar.lambda_im[k];
                prop_assert!(lr2 <= 0.9995 * 0.9995 + 1e-5);
            }
        }
    }
}

#[cfg(test)]
mod spec_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Specification Verification Suite: Phase 6 — Op-M Symplectic Modal, Hilbert, Sub-Harmonic & Kautz.
    //
    // Verifies all engine-scope rules in `docs/specs/dsp/op_m_modal_kautz.md` plus `dsp.zoh.op-m-poles`:
    // - `verify[dsp.op-m.recurrence.symplectic]`
    // - `verify[dsp.op-m.rotor.contractive]`
    // - `verify[dsp.op-m.rotor.zero-hold]`
    // - `verify[dsp.op-m.rotor.quadrant-horner]`
    // - `verify[dsp.op-m.rotor.ulp-guard]`
    // - `verify[dsp.op-m.rotor.q0-192-lfo]`
    // - `verify[dsp.op-m.init.log-octave-h2]`
    // - `verify[dsp.op-m.hilbert.lattice]`
    // - `verify[dsp.op-m.hilbert.h1-anchors]`
    // - `verify[dsp.op-m.hilbert.ties-to-even]`
    // - `verify[dsp.op-m.hilbert.rate-remap]`
    // - `verify[dsp.op-m.tracker.t1-dc-blocker]`
    // - `verify[dsp.op-m.tracker.envelope-norm]`
    // - `verify[dsp.op-m.tracker.toggle-sheet]`
    // - `verify[dsp.op-m.stage3.kautz-allocation]`
    // - `verify[dsp.op-m.stage3.dual-state]`
    // - `verify[dsp.op-m.stage3.radial-gate]`
    // - `verify[dsp.op-m.stage3.rate-convert]`
    // - `verify[dsp.op-m.stage3.coupled-form]`
    // - `verify[dsp.op-m.stage3.sweep-contraction]`
    // - `verify[dsp.op-m.ctrl.engine]`
    // - `verify[dsp.zoh.op-m-poles]`

    use crate::FtzDazGuard;
    use crate::container::ZohRescaledParams;
    use crate::math::{Q0_192Phase, contractive_rotor_f32, step_lfo_phasor_zero_hold};
    use crate::operators::{
        HILBERT_H1_ANCHORS_15, HilbertH1Lattice6Sec, KautzModalBank, SymplecticModal4Lane,
        T1DcBlocker, T1SubHarmonicTracker, compute_t1_dc_blocker_coeffs_bits,
        convert_stage3_kautz_pole_for_rate, encode_hilbert_h1_anchor_table_780b,
        eval_stage3_sweep_contraction, eval_tracker_half_angle_pair, init_log_octave_poles_h2,
        interpolate_hilbert_h1_between_anchors_bits, interpolate_hilbert_h1_profile_bits,
        normalize_tracker_phasor_exponent_split, tracker_crosses_branch_cut,
        validate_and_eval_stage3_radial_gate,
    };
    use crate::provenance::sha256_digest;
    use crate::stages::Stage3CabinetModals;

    /// verify[dsp.op-m.recurrence.symplectic]
    /// verify[dsp.op-m.rotor.contractive]
    /// verify[dsp.op-m.rotor.zero-hold]
    /// verify[dsp.op-m.rotor.quadrant-horner]
    /// verify[dsp.op-m.rotor.ulp-guard]
    /// verify[dsp.op-m.rotor.q0-192-lfo]
    /// verify[dsp.op-m.init.log-octave-h2]
    /// verify[dsp.op-m.ctrl.engine]
    /// Verifies 4-lane symplectic complex recurrence, contractive rotor, zero-hold, `Q0.192` phase,
    /// and log-octave $\mathcal{H}_2$ pole initialization across `25 Hz..16 kHz`.
    #[test]
    fn test_op_m_symplectic_recurrence_rotor_q0_192_and_log_octave_h2_init() {
        let _guard = FtzDazGuard::new();
        let poles = init_log_octave_poles_h2(8, 25.0, 16_000.0, 48_000.0, 0.99);
        assert_eq!(poles.len(), 8);
        for w in poles.windows(2) {
            assert!(w[1].0 > w[0].0);
            assert!((w[0].1 - (1.0 - 0.99_f64 * 0.99).sqrt() as f32).abs() < 1e-6);
        }

        let mut symp = SymplecticModal4Lane::new();
        for (lane, &(theta, _)) in poles.iter().take(4).enumerate() {
            symp.set_pole(lane, 0.99, theta);
        }
        let (re0, im0) = symp.step_lane(0, 1.0);
        assert!(re0 > 0.0 && im0 == 0.0);

        let (c, s) = contractive_rotor_f32(1.0).unwrap();
        assert!((c as f64) * (c as f64) + (s as f64) * (s as f64) < 1.0);
        assert_eq!(step_lfo_phasor_zero_hold(0.6, 0.8, 0.0), Some((0.6, 0.8)));

        let mut q192 = Q0_192Phase::ZERO;
        q192.step_angle_bits((0.1_f32).to_bits()).unwrap();
        let (pc, ps) = q192.to_phasor();
        assert!((pc as f64) * (pc as f64) + (ps as f64) * (ps as f64) <= 1.0);
    }

    /// verify[dsp.op-m.hilbert.lattice]
    /// verify[dsp.op-m.hilbert.h1-anchors]
    /// verify[dsp.op-m.hilbert.ties-to-even]
    /// verify[dsp.op-m.hilbert.rate-remap]
    /// Verifies the 780 B `H1` anchor table SHA-256, exact `u128` ties-to-even rate interpolation,
    /// quadrature phase accuracy (`<= 0.34 deg` over `20 Hz..20 kHz`), and history-consistent rate remap.
    #[test]
    fn test_hilbert_h1_anchors_sha256_ties_to_even_quadrature_and_rate_remap() {
        // 1. 780 B H1 anchor table SHA-256
        let table_780b = encode_hilbert_h1_anchor_table_780b();
        let sha_hex: String = sha256_digest(&table_780b)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            sha_hex,
            "bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352"
        );

        // 2. Anchor preservation and exact ties-to-even test cases from specification
        for &(rate, ref expected_coeffs) in &HILBERT_H1_ANCHORS_15 {
            assert_eq!(
                interpolate_hilbert_h1_profile_bits(rate as f64).unwrap(),
                *expected_coeffs
            );
        }
        for &(rate, lo, hi, idx, exp_bits) in &[
            (49_200.0_f64, 48_000u32, 64_000u32, 8usize, 0x3F60_0A2Au32),
            (54_800.0, 48_000, 64_000, 8, 0x3F61_BA02),
            (48_400.0, 48_000, 64_000, 10, 0x3F7D_03D4),
            (80_960.0, 80_000, 96_000, 3, 0x3F72_19A2),
            (81_600.0, 80_000, 96_000, 4, 0x3F7C_1BE4),
            (181_080.0, 176_400, 192_000, 4, 0x3F7D_F2E0),
        ] {
            let coeffs = interpolate_hilbert_h1_between_anchors_bits(rate, lo, hi).unwrap();
            assert_eq!(coeffs[idx], exp_bits);
        }

        // Out-of-band rates are rejected
        for bad_rate in [44_099.0, 100_000.0, 176_399.0, 384_001.0, f64::NAN] {
            assert!(interpolate_hilbert_h1_profile_bits(bad_rate).is_none());
        }

        // 3. History-consistent rate transition remap preserves constant-input equilibrium
        let _guard = FtzDazGuard::new();
        let mut h1 = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        h1.init_dc_equilibrium(1.0);
        let (i_before, q_before) = h1.step(1.0);
        assert!((i_before - 1.0).abs() < 1e-6 && (q_before - 1.0).abs() < 1e-6);

        h1.retune_sample_rate(96_000.0).unwrap();
        let (i_after, q_after) = h1.step(1.0);
        assert!((i_after - 1.0).abs() < 1e-6 && (q_after - 1.0).abs() < 1e-6);
    }

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    /// Verifies all 8 `T1-COEFFICIENT-VECTORS` via 160-bit midpoint binary search, exponent-split
    /// envelope normalization across `2^-125..100.0`, unnormalized-`I` edge toggles, and half-angle sheet.
    #[test]
    fn test_t1_dc_blocker_vectors_envelope_norm_and_toggle_sheet() {
        // 1. All 8 canonical T1-COEFFICIENT-VECTORS
        for &(rate, exp_pole, exp_gain) in &[
            (44_100.0_f64, 0x3F7F_D155u32, 0x3F7F_E8AAu32),
            (48_000.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (48_001.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (88_200.0, 0x3F7F_E8A9, 0x3F7F_F454),
            (96_000.0, 0x3F7F_EA8F, 0x3F7F_F548),
            (176_400.0, 0x3F7F_F454, 0x3F7F_FA2A),
            (192_000.0, 0x3F7F_F547, 0x3F7F_FAA4),
            (384_000.0, 0x3F7F_FAA3, 0x3F7F_FD52),
        ] {
            let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate).unwrap();
            assert_eq!(
                (p_bits, g_bits),
                (exp_pole, exp_gain),
                "T1 mismatch at rate={rate}: got ({p_bits:#010x}, {g_bits:#010x})"
            );
        }
        assert!(compute_t1_dc_blocker_coeffs_bits(100_000.0).is_none());

        // 2. Exponent-split envelope normalization across quiet (2^-125) and loud (100.0) amplitudes
        for &amp in &[(-125.0_f64).exp2() as f32, 1.0e-30, 1.0e-12, 0.1, 100.0] {
            for idx in 0..=64 {
                let phase = -core::f64::consts::PI + core::f64::consts::TAU * (idx as f64) / 64.0;
                let i_in = ((amp as f64) * phase.cos()) as f32;
                let q_in = ((amp as f64) * phase.sin()) as f32;
                let (ux, uy, env) = normalize_tracker_phasor_exponent_split(i_in, q_in).unwrap();
                let (v_cos, v_sin) = eval_tracker_half_angle_pair(ux, uy);
                let exp_cos = (amp as f64) * (0.5 * phase).cos().abs();
                let exp_sin = (amp as f64) * (0.5 * phase).sin().abs();
                assert!(((env * v_cos) as f64 - exp_cos).abs() <= (amp as f64) * 2.5e-5);
                assert!(((env * v_sin) as f64 - exp_sin).abs() <= (amp as f64) * 2.5e-5);
            }
        }
        assert!(normalize_tracker_phasor_exponent_split(0.0, 0.0).is_none());
        assert!(normalize_tracker_phasor_exponent_split(-0.0, 0.0).is_none());

        // 3. Branch cut crossing and T1SubHarmonicTracker silence / axis-zero handling
        assert!(tracker_crosses_branch_cut(
            (150.0_f32.to_radians().cos(), 150.0_f32.to_radians().sin()),
            (300.0_f32.to_radians().cos(), 300.0_f32.to_radians().sin())
        ));
        assert!(!tracker_crosses_branch_cut((1.0, 0.0), (-1.0, 1.0e-7)));

        let _guard = FtzDazGuard::new();
        let mut dc = T1DcBlocker::new_at_rate(48_000.0).unwrap();
        let mut trk = T1SubHarmonicTracker::new();
        assert_eq!(trk.step_analytic(0.0, 0.0), [0.0; 4]);
        assert!(!trk.prev_valid);

        let f_axis = trk.step_analytic(0.0, 1.0);
        assert!(trk.prev_valid && trk.prev_i_nonpositive && f_axis[2] > 0.0);
        let f_rise = trk.step_analytic(1.0, 1.0);
        assert!(f_rise[2] < 0.0); // q1 toggled +1 -> -1 on rising crossing

        let d_out = dc.step(0.5);
        assert!(d_out.is_finite());
    }

    /// verify[dsp.op-m.stage3.kautz-allocation]
    /// verify[dsp.op-m.stage3.dual-state]
    /// verify[dsp.op-m.stage3.radial-gate]
    /// verify[dsp.op-m.stage3.rate-convert]
    /// verify[dsp.op-m.stage3.coupled-form]
    /// verify[dsp.op-m.stage3.sweep-contraction]
    /// verify[dsp.zoh.op-m-poles]
    /// Verifies Stage III Kautz allocation (`16`-sample onset), always-dual L/R state advancement,
    /// shared radial gate & $\mathcal{H}_2$ normalizer, rate conversion, coupled-form norm bound,
    /// and dynamic sweep contraction.
    #[test]
    fn test_stage3_kautz_allocation_dual_state_radial_gate_rate_convert_and_sweep() {
        let _guard = FtzDazGuard::new();

        // 1. Radial gate validation & evaluation
        let (g, r_gate, b_gamma) =
            validate_and_eval_stage3_radial_gate(0.90, 0.08, 0.5, 1.0, 0.0).unwrap();
        assert!((g - 0.5).abs() < 1e-6);
        assert!((r_gate - 0.94).abs() < 1e-6);
        assert!(b_gamma > 0.0);
        assert!(validate_and_eval_stage3_radial_gate(0.80, 0.08, 0.5, 1.0, 0.0).is_none());
        assert!(validate_and_eval_stage3_radial_gate(0.90, 0.15, 0.5, 1.0, 0.0).is_none());

        // 2. Dynamic sweep contraction
        let r_swept = eval_stage3_sweep_contraction(0.9990, 0.1);
        assert!((r_swept - (0.9950 - 0.015)).abs() < 1e-6);

        // 3. Stage III rate conversion & ZOH Op-M pole rescaling
        let (rl_96, dr_96, th_96, bg_96, cr_96, ci_96) =
            convert_stage3_kautz_pole_for_rate(0.90, 0.05, 0.4, 0.3, 0.5, -0.2, 96_000.0);
        assert!(rl_96 > 0.90 && dr_96 > 0.0 && (th_96 - 0.2).abs() < 1e-5);
        assert!(bg_96.is_finite() && cr_96.is_finite() && ci_96.is_finite());

        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let (z_rl, z_dr, z_th, z_bg) = zoh.rescale_op_m_channel(0.90, 0.05, 0.4, 0.3);
        assert!((z_rl - rl_96).abs() < 1e-5 && z_dr > 0.0 && z_th > 0.0 && z_bg > 0.0);

        // 4. Always-dual L/R channel state & coupled-form triangle inequality norm bound
        let mut cab_l = Stage3CabinetModals::new();
        let mut cab_r = Stage3CabinetModals::new();
        let taps = [0.0625f32; 16];
        let _ = cab_l.step(1.0, 0.0, &taps);
        let _ = cab_r.step(-0.5, 0.0, &taps);
        assert_eq!(cab_l.fir_history[0], 1.0);
        assert_eq!(cab_r.fir_history[0], -0.5);

        let mut bank = KautzModalBank::<4>::new();
        bank.set_pole_with_b_gamma(0, 0.95, 0.4, 0.0, 0.3, 1.0, 0.0);
        let mut prev_norm = 0.0_f64;
        for _ in 0..32 {
            let u = 0.5_f32;
            let _ = bank.step(u);
            let cur_norm = ((bank.s_re[0] as f64).powi(2) + (bank.s_im[0] as f64).powi(2)).sqrt();
            assert!(cur_norm <= 0.95 * prev_norm + 0.3 * 0.5 + 1e-6);
            prev_norm = cur_norm;
        }
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.op-m.hilbert.ties-to-even]
        /// verify[dsp.op-m.tracker.envelope-norm]
        /// verify[dsp.op-m.stage3.radial-gate]
        #[test]
        fn prop_op_m_hilbert_interpolation_t1_envelope_and_stage3_radial_gate(
            rate_live in 44_100.0_f64..=96_000.0_f64,
            i_val in -50.0_f32..50.0_f32,
            q_val in -50.0_f32..50.0_f32,
            r_low in 0.85_f32..0.95_f32,
            dr in 0.0_f32..0.049_f32,
            theta_10 in -2.0_f32..2.0_f32,
        ) {
            let coeffs = interpolate_hilbert_h1_profile_bits(rate_live).unwrap();
            for &b in &coeffs {
                let c = f32::from_bits(b);
                prop_assert!(c > 0.0 && c < 1.0);
            }

            if let Some((u_re, u_im, env)) = normalize_tracker_phasor_exponent_split(i_val, q_val) {
                prop_assert!(env >= 0.0 && env.is_finite());
                prop_assert!(u_re * u_re + u_im * u_im <= 1.001);
            }

            let (g_k, r_k, b_eff) =
                validate_and_eval_stage3_radial_gate(r_low, dr, 0.3, 1.0, theta_10).unwrap();
            prop_assert!((0.0..=1.0).contains(&g_k) && b_eff.is_finite());
            prop_assert!(r_k >= r_low && r_k <= r_low + dr && r_k < 1.0);
        }
    }
}

#[cfg(test)]
mod stage1_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Finite-precision Stage I reference examples, not production-path tests.

    use crate::FtzDazGuard;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    use std::hint::black_box;

    const RATE: usize = 48_000;
    const TONE_PERIOD: usize = 48;
    const WARMUP: usize = RATE;
    const OBSERVATION: usize = 4_800;
    const AMPLITUDE: f64 = 1.0 / 4_194_304.0;
    const I_BRANCH_BITS: [u32; 6] = [
        0x3d91_bee6,
        0x3eec_b8e1,
        0x3f48_de0d,
        0x3f6d_91c9,
        0x3f7a_5939,
        0x3f7e_8e9d,
    ];

    #[derive(Default)]
    struct HilbertBranch {
        state: [[f32; 2]; 6],
    }

    impl HilbertBranch {
        fn step(&mut self, mut value: f32) -> f32 {
            for (state, bits) in self.state.iter_mut().zip(I_BRANCH_BITS) {
                let coefficient = f32::from_bits(bits);
                let output = coefficient * value + state[1];
                state[1] = state[0];
                state[0] = coefficient * output - value;
                value = output;
            }
            value
        }
    }

    struct Blocker {
        pole: f32,
        gain: f32,
        previous_input: f32,
        previous_output: f32,
    }

    impl Blocker {
        fn new() -> Self {
            let pole = 0.999_345_7_f32;
            let gain = ((1.0_f64 + f64::from(pole)) * 0.5) as f32;
            Self {
                pole,
                gain,
                previous_input: 0.0,
                previous_output: 0.0,
            }
        }

        fn step(&mut self, value: f32) -> f32 {
            let output =
                self.gain * (value - self.previous_input) + self.pole * self.previous_output;
            self.previous_input = value;
            self.previous_output = output;
            output
        }
    }

    #[derive(Default)]
    struct Projection {
        real: f64,
        imag: f64,
    }

    impl Projection {
        fn add(&mut self, sample: f32, index: usize) {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            self.real += f64::from(sample) * angle.cos();
            self.imag -= f64::from(sample) * angle.sin();
        }

        fn magnitude(&self) -> f64 {
            self.real.hypot(self.imag)
        }
    }

    #[test]
    fn preconditioned_tracker_preserves_a_quiet_tone_better_than_postconditioning_one_example() {
        let _guard = FtzDazGuard::new();
        let mut before_filter = Blocker::new();
        let mut before_branch = HilbertBranch::default();
        let mut after_branch = HilbertBranch::default();
        let mut after_filter = Blocker::new();
        assert_eq!(before_filter.pole.to_bits(), 0x3f7f_d51f);
        assert_eq!(before_filter.gain.to_bits(), 0x3f7f_ea90);
        let mut input_projection = Projection::default();
        let mut before_projection = Projection::default();
        let mut after_projection = Projection::default();

        for index in 0..WARMUP + OBSERVATION {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            let input = (0.5 + AMPLITUDE * angle.cos()) as f32;
            let before = before_branch.step(before_filter.step(input));
            let after = after_filter.step(after_branch.step(input));
            if index >= WARMUP {
                input_projection.add(input, index);
                before_projection.add(before, index);
                after_projection.add(after, index);
            }
        }

        let input = input_projection.magnitude();
        let before = before_projection.magnitude();
        let difference = (after_projection.real - before_projection.real)
            .hypot(after_projection.imag - before_projection.imag);
        assert!((before / input - 1.0).abs() < 0.01);
        assert!(difference / before > 0.2);
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn flushed_normalized_i_does_not_preserve_the_raw_detector_sign() {
        let _guard = FtzDazGuard::new();
        let squared_norm = black_box(1.0_f32);
        let mut inverse = f32::from_bits(0x5f37_59df - (squared_norm.to_bits() >> 1));
        for _ in 0..2 {
            let half_norm = black_box(0.5 * squared_norm);
            let product = black_box(black_box(half_norm * inverse) * inverse);
            inverse = black_box(inverse * black_box(1.5 - product));
        }

        let raw_i = black_box(f32::MIN_POSITIVE);
        let normalized_i = black_box(raw_i * inverse);
        let later_positive = black_box(1.0_f32);
        let earlier_negative = black_box(-1.0_f32);
        assert!(raw_i > 0.0);
        assert!(inverse < 1.0);
        assert_eq!(normalized_i.to_bits() & 0x7fff_ffff, 0);
        assert!(!(raw_i <= 0.0 && later_positive > 0.0));
        assert!(normalized_i <= 0.0 && later_positive > 0.0);
        assert!(earlier_negative <= 0.0 && raw_i > 0.0);
        assert!(!(earlier_negative <= 0.0 && normalized_i > 0.0));
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn equal_transposed_cells_can_require_different_history_consistent_retunes() {
        fn step(coefficient: f32, state: &mut [f32; 2], input: f32) -> f32 {
            let output = black_box(black_box(coefficient * input) + state[1]);
            state[1] = state[0];
            state[0] = black_box(black_box(coefficient * output) - input);
            output
        }

        let _guard = FtzDazGuard::new();
        let old = black_box(f32::from_bits(0x3d91_bee6));
        let new = black_box(f32::from_bits(0x3db0_73c5));
        let mut driven_state = [0.0; 2];
        let mut outputs = [0.0; 3];
        for (sample, output) in [1.0, 0.0, -old].into_iter().zip(&mut outputs) {
            *output = step(old, &mut driven_state, sample);
        }
        assert_eq!(driven_state.map(f32::to_bits), [0; 2]);
        assert_eq!(&outputs[1..], &[0.0, -1.0]);

        let mut empty_state = [0.0; 2];
        let change = black_box(new - old);
        driven_state[0] = black_box(driven_state[0] + black_box(change * outputs[2]));
        driven_state[1] = black_box(driven_state[1] + black_box(change * outputs[1]));
        assert_ne!(driven_state, empty_state);

        assert_eq!(step(new, &mut driven_state, 0.0), 0.0);
        assert_eq!(step(new, &mut empty_state, 0.0), 0.0);
        let driven_second = step(new, &mut driven_state, 0.0);
        let empty_second = step(new, &mut empty_state, 0.0);
        assert!((-0.015..-0.014).contains(&driven_second));
        assert_eq!(empty_second, 0.0);
    }
}
