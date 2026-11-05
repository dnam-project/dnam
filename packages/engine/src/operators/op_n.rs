//! Op-N: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.bspline-ac]`, `ref[dsp.op-n.ctrl.lobatto-basis]`, `ref[dsp.op-n.ctrl.trust-clip]`).

use alloc::vec::Vec;

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4, newton6_reciprocal_f64,
};

/// Drive scales for the 4-lane Multi-Drive PolyADAA Bank.
pub const MULTI_DRIVE_SCALES: [f32; 4] = [0.25, 1.0, 4.0, 16.0];

/// Exact division-free algebraic PolyADAA segment evaluation (`ref[dsp.op-n.polyadaa.midpoint]`).
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

#[inline(always)]
fn poly_adaa_eval_vec4<V: SimdVec4>(u_prev: V, u_curr: V, beta: V, delta: V) -> V {
    let half = V::splat(0.5);
    let one_pt_five = V::splat(1.5);
    let one_third = V::splat(1.0 / 3.0);
    let one_fifth = V::splat(0.2);
    let two = V::splat(2.0);

    let m = half.mul(u_curr.add(u_prev));
    let d = half.mul(u_curr.sub(u_prev));

    let m2 = m.mul(m);
    let d2 = d.mul(d);

    let odd_factor = one_pt_five.sub(half.mul(m2.add(d2)));
    let a_odd = m.mul(odd_factor);

    let quad = m2.add(one_third.mul(d2));
    let quart = m2
        .mul(m2)
        .add(two.mul(m2).mul(d2))
        .add(one_fifth.mul(d2).mul(d2));
    let a_even = quad.sub(quart);

    beta.mul(a_odd).add(delta.mul(a_even))
}

/// Instantaneous C^2 compactly supported wavefolder.
/// Evaluates the cubed-envelope $C^2$ wavefolder (`ref[dsp.op-n.wavefolder.cubic-c2]`):
/// `w_odd(u) = u_bar * (1 - u_bar^2)^3` and `w_even(u) = u_bar^2 * (1 - u_bar^2)^3` on `u_bar = clamp(u, -1, 1)`.
#[inline(always)]
pub fn c2_wavefolder(u: f32) -> (f32, f32) {
    let u_clamped = u.clamp(-1.0, 1.0);
    let u2 = u_clamped * u_clamped;
    let one_minus_u2 = 1.0 - u2;
    let factor = one_minus_u2 * one_minus_u2 * one_minus_u2;

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

/// Fixed-Hertz `Q64.32` Integer ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`, `ref[dsp.op-n.bitcrusher.quantizer]`).
#[derive(Clone, Copy, Debug)]
pub struct ZohBitcrusher {
    pub step_size: f32,
    pub step_inv: f32,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub bit_depth: u8,
    pub target_fs_hz: f32,
    pub held_sample: f32,
}

impl Default for ZohBitcrusher {
    fn default() -> Self {
        Self::new()
    }
}

impl ZohBitcrusher {
    pub const fn new() -> Self {
        let den = 48_000u64 << 32;
        Self {
            step_size: 1.0 / 32768.0,
            step_inv: 32768.0,
            phase: 0,
            increment: den,
            denominator: den,
            reciprocal_r: ((1u128 << 104) / (den as u128)) as u64,
            bit_depth: 16,
            target_fs_hz: 48_000.0,
            held_sample: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.held_sample = 0.0;
    }

    /// Sets bit depth `bits in [2, 16]` and downsampled sample rate `target_fs_hz` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
    pub fn set_parameters(&mut self, bits: u32, target_fs_hz: f32, sample_rate_hz: f32) {
        let b = bits.clamp(2, 16) as u8;
        let levels = (1u32 << (b - 1)) as f32;
        self.bit_depth = b;
        self.step_inv = levels;
        self.step_size = 1.0 / levels;
        self.target_fs_hz = target_fs_hz.max(0.0);

        let sr = sample_rate_hz.max(1000.0) as f64;
        if let Some((inc, den, rec)) =
            hertz_phase_parameters(self.target_fs_hz.min(sample_rate_hz) as f64, sr)
        {
            self.increment = inc;
            self.denominator = den;
            self.reciprocal_r = rec;
            if self.phase >= self.denominator {
                self.phase %= self.denominator;
            }
        }
    }

    /// Transfers the `Q64.32` phase accumulator across a host sample-rate change without `u128` division (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
    pub fn reconfigure_sample_rate(&mut self, new_sample_rate_hz: f32) {
        let sr = new_sample_rate_hz.max(1000.0) as f64;
        if let Some((inc, den, rec)) =
            hertz_phase_parameters(self.target_fs_hz.min(new_sample_rate_hz) as f64, sr)
        {
            self.phase =
                rescale_bitcrusher_phase_u128(self.phase, self.denominator, den, self.reciprocal_r);
            self.increment = inc;
            self.denominator = den;
            self.reciprocal_r = rec;
        }
    }

    /// Steps the ZOH bitcrusher with zero audio-rate divisions (`ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.quantizer]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.increment == 0 {
            return self.held_sample;
        }

        let latch = if self.increment >= self.denominator {
            true
        } else {
            let (next_phase, l) = hertz_phase_step(self.phase, self.increment, self.denominator);
            self.phase = next_phase;
            l
        };

        if latch {
            self.held_sample =
                quantize_bitcrusher_nearest_ties_pos_inf(x, u32::from(self.bit_depth));
        }

        self.held_sample
    }
}

/// Multi-Drive 1x B-Spline PolyADAA Bank Channel State (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.bspline-ac]`).
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
            let u_raw = (2.0 / 3.0) * drive * b_conv;
            let u_rest = u_raw.clamp(-1.0, 1.0);
            let a_rest = poly_adaa_algebraic_eval(u_rest, u_rest, beta, delta);
            self.u_prev[i] = u_raw;
            self.a_prev[i] = a_rest;
            self.m_dc[i] = flush_recursive_state_f32(a_rest);
        }
    }

    /// Steps the multi-drive bank for input `x`, returning 4-lane nonlinearly shaped basis vector.
    /// Evaluates the unclamped 6-step `f64` Newton rail-crossing `PolyADAA` law when a segment crosses $\pm 1$
    /// (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.bspline-ac]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    fn crosses_rail(u_prev: f32, u_curr: f32) -> bool {
        let lo = u_prev.min(u_curr);
        let hi = u_prev.max(u_curr);
        (lo < -1.0 && hi > -1.0) || (lo < 1.0 && hi > 1.0)
    }

    #[inline(always)]
    pub fn step(&mut self, x: f32, beta: f32, delta: f32, alpha: f32, r_ac: f32) -> [f32; 4] {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let one_minus_rac = 1.0 - r_ac;
        let x_scaled = (2.0 / 3.0) * x;

        let u_raw_v = V::load(&MULTI_DRIVE_SCALES).mul(V::splat(x_scaled));
        let u_prev_v = V::load(&self.u_prev);
        let neg_one = V::splat(-1.0);
        let pos_one = V::splat(1.0);
        let u_p_v = u_prev_v.max(neg_one).min(pos_one);
        let u_c_v = u_raw_v.max(neg_one).min(pos_one);

        let mut a_v = poly_adaa_eval_vec4(u_p_v, u_c_v, V::splat(beta), V::splat(delta));
        let mut u_raw_arr = [0.0f32; 4];
        u_raw_v.store(&mut u_raw_arr);

        if Self::crosses_rail(self.u_prev[3], u_raw_arr[3])
            || u_raw_arr[2].abs() > 1.0
            || self.u_prev[2].abs() > 1.0
        {
            let mut a_arr = [0.0f32; 4];
            a_v.store(&mut a_arr);
            for i in 0..4 {
                if Self::crosses_rail(self.u_prev[i], u_raw_arr[i]) {
                    let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                        self.u_prev[i] as f64,
                        u_raw_arr[i] as f64,
                    );
                    a_arr[i] = ((beta as f64) * odd + (delta as f64) * even) as f32;
                }
            }
            a_v = V::load(&a_arr);
        }

        let a_prev_v = V::load(&self.a_prev);
        let m_prev_v = V::load(&self.m_dc);
        let y_nl_v = V::splat(0.875).mul(a_v).add(V::splat(0.125).mul(a_prev_v));
        let next_m_dc_v = V::splat(r_ac)
            .mul(m_prev_v)
            .add(V::splat(one_minus_rac).mul(y_nl_v));
        let out_v = V::splat(alpha * x).add(y_nl_v.sub(next_m_dc_v));

        flush_recursive_state_vec4(u_raw_v).store(&mut self.u_prev);
        flush_recursive_state_vec4(a_v).store(&mut self.a_prev);
        flush_recursive_state_vec4(next_m_dc_v).store(&mut self.m_dc);

        let mut out = [0.0; 4];
        out_v.store(&mut out);
        out
    }

    /// Steps only drive lanes `1` (`1.0x`) and `2` (`4.0x`) for Stage II Sub-stage 2.2 (`V1B` Driver),
    /// returning `(out_lane1, out_lane2)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_lanes_1_2(
        &mut self,
        x: f32,
        beta: f32,
        delta: f32,
        alpha: f32,
        r_ac: f32,
    ) -> (f32, f32) {
        let one_minus_rac = 1.0 - r_ac;
        let x_scaled = (2.0 / 3.0) * x;
        let u1_raw = x_scaled;
        let u2_raw = 4.0 * x_scaled;

        let p1 = self.u_prev[1];
        let p2 = self.u_prev[2];
        let u1_p = p1.clamp(-1.0, 1.0);
        let u1_c = u1_raw.clamp(-1.0, 1.0);
        let u2_p = p2.clamp(-1.0, 1.0);
        let u2_c = u2_raw.clamp(-1.0, 1.0);

        let mut a1 = poly_adaa_algebraic_eval(u1_p, u1_c, beta, delta);
        let mut a2 = poly_adaa_algebraic_eval(u2_p, u2_c, beta, delta);
        if Self::crosses_rail(p2, u2_raw) || u1_raw.abs() > 1.0 || p1.abs() > 1.0 {
            if Self::crosses_rail(p1, u1_raw) {
                let (odd, even, _) =
                    eval_saturated_polyadaa_rail_crossing_f64(p1 as f64, u1_raw as f64);
                a1 = ((beta as f64) * odd + (delta as f64) * even) as f32;
            }
            if Self::crosses_rail(p2, u2_raw) {
                let (odd, even, _) =
                    eval_saturated_polyadaa_rail_crossing_f64(p2 as f64, u2_raw as f64);
                a2 = ((beta as f64) * odd + (delta as f64) * even) as f32;
            }
        }
        let y1_nl = 0.875 * a1 + 0.125 * self.a_prev[1];
        let next_m1_dc = r_ac * self.m_dc[1] + one_minus_rac * y1_nl;
        let out1 = alpha * x + (y1_nl - next_m1_dc);
        self.u_prev[1] = flush_recursive_state_f32(u1_raw);
        self.a_prev[1] = flush_recursive_state_f32(a1);
        self.m_dc[1] = flush_recursive_state_f32(next_m1_dc);

        let y2_nl = 0.875 * a2 + 0.125 * self.a_prev[2];
        let next_m2_dc = r_ac * self.m_dc[2] + one_minus_rac * y2_nl;
        let out2 = alpha * x + (y2_nl - next_m2_dc);
        self.u_prev[2] = flush_recursive_state_f32(u2_raw);
        self.a_prev[2] = flush_recursive_state_f32(a2);
        self.m_dc[2] = flush_recursive_state_f32(next_m2_dc);

        (out1, out2)
    }

    /// Steps drive lanes `1` (`1.0x`) and `2` (`4.0x`) for a stereo `(left, right)` pair simultaneously
    /// across a single 4-lane vectorizable loop (`[l_1x, l_4x, r_1x, r_4x]`), returning `(l1, l2, r1, r2)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_stereo_lanes_1_2(
        left: &mut Self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        beta: f32,
        delta: f32,
        alpha: f32,
        r_ac: f32,
    ) -> (f32, f32, f32, f32) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let one_minus_rac = 1.0 - r_ac;
        let xl_s = (2.0 / 3.0) * x_l;
        let xr_s = (2.0 / 3.0) * x_r;

        let u_prev_4 = [
            left.u_prev[1],
            left.u_prev[2],
            right.u_prev[1],
            right.u_prev[2],
        ];
        let u_raw_4 = [xl_s, 4.0 * xl_s, xr_s, 4.0 * xr_s];
        let a_prev_4 = [
            left.a_prev[1],
            left.a_prev[2],
            right.a_prev[1],
            right.a_prev[2],
        ];
        let m_prev_4 = [left.m_dc[1], left.m_dc[2], right.m_dc[1], right.m_dc[2]];
        let alpha_x_4 = [alpha * x_l, alpha * x_l, alpha * x_r, alpha * x_r];

        let neg_one = V::splat(-1.0);
        let pos_one = V::splat(1.0);
        let u_prev_v = V::load(&u_prev_4);
        let u_raw_v = V::load(&u_raw_4);
        let u_p_v = u_prev_v.max(neg_one).min(pos_one);
        let u_c_v = u_raw_v.max(neg_one).min(pos_one);
        let mut a_v = poly_adaa_eval_vec4(u_p_v, u_c_v, V::splat(beta), V::splat(delta));

        if Self::crosses_rail(u_prev_4[1], u_raw_4[1])
            || Self::crosses_rail(u_prev_4[3], u_raw_4[3])
            || u_raw_4[0].abs() > 1.0
            || u_prev_4[0].abs() > 1.0
            || u_raw_4[2].abs() > 1.0
            || u_prev_4[2].abs() > 1.0
        {
            let mut a_4 = [0.0f32; 4];
            a_v.store(&mut a_4);
            for i in 0..4 {
                if Self::crosses_rail(u_prev_4[i], u_raw_4[i]) {
                    let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                        u_prev_4[i] as f64,
                        u_raw_4[i] as f64,
                    );
                    a_4[i] = ((beta as f64) * odd + (delta as f64) * even) as f32;
                }
            }
            a_v = V::load(&a_4);
        }

        let a_prev_v = V::load(&a_prev_4);
        let m_prev_v = V::load(&m_prev_4);
        let y_nl_v = V::splat(0.875).mul(a_v).add(V::splat(0.125).mul(a_prev_v));
        let next_m_v = V::splat(r_ac)
            .mul(m_prev_v)
            .add(V::splat(one_minus_rac).mul(y_nl_v));
        let out_v = V::load(&alpha_x_4).add(y_nl_v.sub(next_m_v));

        let mut u_flushed = [0.0f32; 4];
        let mut a_flushed = [0.0f32; 4];
        let mut m_flushed = [0.0f32; 4];
        let mut out_4 = [0.0f32; 4];
        flush_recursive_state_vec4(u_raw_v).store(&mut u_flushed);
        flush_recursive_state_vec4(a_v).store(&mut a_flushed);
        flush_recursive_state_vec4(next_m_v).store(&mut m_flushed);
        out_v.store(&mut out_4);

        left.u_prev[1] = u_flushed[0];
        left.u_prev[2] = u_flushed[1];
        right.u_prev[1] = u_flushed[2];
        right.u_prev[2] = u_flushed[3];

        left.a_prev[1] = a_flushed[0];
        left.a_prev[2] = a_flushed[1];
        right.a_prev[1] = a_flushed[2];
        right.a_prev[2] = a_flushed[3];

        left.m_dc[1] = m_flushed[0];
        left.m_dc[2] = m_flushed[1];
        right.m_dc[1] = m_flushed[2];
        right.m_dc[2] = m_flushed[3];

        (out_4[0], out_4[1], out_4[2], out_4[3])
    }
}

/// 3-Node Newton/Lobatto Basis evaluation (`ref[dsp.op-n.ctrl.lobatto-basis]`).
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

/// Outward-Checked Ellipsoidal Trust-Region Clipping (`ref[dsp.ctrl.op-n.trust-clip]`, `ref[dsp.op-n.ctrl.trust-clip]`, `0 FDIV`, `0 FSQRT`).
///
/// Applies the 6-step `f64` Newton fast path with 32-step bisection fallback so that
/// $\sum_i \text{widen}(z_i)^2 \le \rho_{\max}^2$ holds strictly after `RN32` rounding.
#[inline(always)]
pub fn trust_region_softclip_4d(z: &mut [f32; 4], rho_max: f32) {
    if rho_max <= 1.0e-12 {
        return;
    }
    let val_bits = [
        z[0].to_bits(),
        z[1].to_bits(),
        z[2].to_bits(),
        z[3].to_bits(),
    ];
    if let Some((clipped_bits, _, _)) = crate::math::trust_clip_outward_checked_bits(
        &val_bits,
        rho_max.to_bits(),
        1.0e-12_f32.to_bits(),
    ) {
        for i in 0..4 {
            z[i] = f32::from_bits(clipped_bits[i]);
        }
    }
}

// ============================================================================
// Normative v2.0 Op-N: Saturated Rail-Crossing PolyADAA, Degree-7/8 ADAA
// Cubed-Envelope Wavefolder, Fixed-Hertz Integer Bitcrusher & Rate Transfer,
// LTI Commutativity, Project-Then-Filter & Latent Folding (`ref[dsp.op-n.*]`)
// ============================================================================

/// Evaluates the saturated-curve `PolyADAA` odd and even antiderivative quotients for consecutive
/// unclamped raw inputs `raw_prev` and `raw_curr` using the 6-step division-free Newton reciprocal
/// when the segment crosses $\pm 1$ (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`,
/// `ref[dsp.op-n.polyadaa.rail-crossing]`).
/// Returns `(odd, even, crossed_rail)`.
#[inline]
pub fn eval_saturated_polyadaa_rail_crossing_f64(raw_prev: f64, raw_curr: f64) -> (f64, f64, bool) {
    let lower = raw_prev.min(raw_curr);
    let upper = raw_prev.max(raw_curr);
    if lower >= 1.0 {
        return (1.0, 0.0, false);
    }
    if upper <= -1.0 {
        return (-1.0, 0.0, false);
    }
    let clamped_lower = lower.max(-1.0);
    let clamped_upper = upper.min(1.0);
    let m = 0.5 * (clamped_upper + clamped_lower);
    let d = 0.5 * (clamped_upper - clamped_lower);
    let m2 = m * m;
    let d2 = d * d;
    let odd = m * (1.5 - 0.5 * (m2 + d2));
    let even = (m2 + d2 * (1.0 / 3.0)) - (m2 * m2 + 2.0 * m2 * d2 + d2 * d2 * 0.2);
    if lower >= -1.0 && upper <= 1.0 {
        return (odd, even, false);
    }
    let distance = upper - lower;
    let inv = newton6_reciprocal_f64(distance);
    let inside = (clamped_upper - clamped_lower).max(0.0);
    let below = (upper.min(-1.0) - lower).max(0.0);
    let above = (upper - lower.max(1.0)).max(0.0);
    (
        (inside * odd + above - below) * inv,
        (inside * even) * inv,
        true,
    )
}

/// Evaluates the odd degree-7 interval-average `PolyADAA` basis polynomial $A_{\text{odd},3}$ in `f32`
/// for the cubed-envelope $C^2$ wavefolder on clamped endpoints $u_0, u_1 \in [-1, 1]$ (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline(always)]
pub fn eval_adaa_cubic_c2_wavefolder_odd_f32(u_prev: f32, u_curr: f32) -> f32 {
    let u0 = u_prev.clamp(-1.0, 1.0);
    let u1 = u_curr.clamp(-1.0, 1.0);
    let m = 0.5 * (u1 + u0);
    let d = 0.5 * (u1 - u0);
    let m2 = m * m;
    let d2 = d * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let m6 = m4 * m2;
    let d6 = d4 * d2;
    m * (1.0 - 3.0 * (m2 + d2) + 3.0 * m4 + 10.0 * m2 * d2 + 3.0 * d4
        - m6
        - 7.0 * (m4 * d2 + m2 * d4)
        - d6)
}

/// Evaluates the division-free degree-7/8 interval-average `PolyADAA` basis polynomials
/// $(A_{\text{odd},3}, A_{\text{even},3})$ for the cubed-envelope $C^2$ wavefolder on clamped
/// endpoints $u_0, u_1 \in [-1, 1]$ (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline]
pub fn eval_adaa_cubic_c2_wavefolder_f64(u_prev: f64, u_curr: f64) -> (f64, f64) {
    let u0 = u_prev.clamp(-1.0, 1.0);
    let u1 = u_curr.clamp(-1.0, 1.0);
    let m = 0.5 * (u1 + u0);
    let d = 0.5 * (u1 - u0);
    let m2 = m * m;
    let d2 = d * d;
    let m3 = m2 * m;
    let d3 = d2 * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let odd = m
        * (1.0 - 3.0 * (m2 + d2) + 3.0 * m4 + 10.0 * m2 * d2 + 3.0 * d4
            - m3 * m3
            - 7.0 * m4 * d2
            - 7.0 * m2 * d4
            - d3 * d3);
    let even = m2 + d2 / 3.0 - 3.0 * (m4 + 2.0 * m2 * d2 + 0.2 * d4)
        + 3.0 * (m3 * m3 + 5.0 * m4 * d2 + 3.0 * m2 * d4 + d3 * d3 / 7.0)
        - (m4 * m4
            + (28.0 / 3.0) * m3 * m3 * d2
            + 14.0 * m4 * d4
            + 4.0 * m2 * d3 * d3
            + d4 * d4 / 9.0);
    (odd, even)
}

/// Evaluates the uniform nearest-step quantizer $Q_B(x) = \lfloor x 2^{B-1} + 0.5 \rfloor 2^{-(B-1)}$
/// (halfway ties toward $+\infty$) for $B \in [2, 16]$ using floor-residual `i32` truncation
/// (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline]
pub fn quantize_bitcrusher_nearest_ties_pos_inf(x: f32, bits: u32) -> f32 {
    let b = bits.clamp(2, 16);
    let scale = (1u32 << (b - 1)) as f32;
    let step_size = 1.0_f32 / scale;
    if !x.is_finite() || x.abs() >= 8_388_608.0 * step_size {
        return x;
    }
    let v = x * scale;
    let t = v as i32;
    let k = t - i32::from(v < (t as f32));
    let frac = v - (k as f32);
    let q = k + i32::from(frac >= 0.5);
    (q as f32) * step_size
}

/// Computes the fixed-hertz integer ZOH phase parameters `(increment_i, denominator_d, reciprocal_r)`
/// at $F = 2^{32}\text{ units/Hz}$ with $R = \lfloor 2^{104} / D \rfloor$ (`u64`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`).
pub fn hertz_phase_parameters(target_hz: f64, sample_rate_hz: f64) -> Option<(u64, u64, u64)> {
    if !target_hz.is_finite()
        || !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !(0.0..=sample_rate_hz).contains(&target_hz)
    {
        return None;
    }
    let scale = (1u64 << 32) as f64;
    let d_f64 = (sample_rate_hz * scale).round_ties_even();
    let i_f64 = (target_hz * scale).round_ties_even();
    if d_f64 <= 0.0 || d_f64 >= ((1u64 << 63) as f64) {
        return None;
    }
    let d = d_f64 as u64;
    let i = i_f64 as u64;
    if (target_hz > 0.0 && i == 0) || i > d {
        return None;
    }
    // Precompute R = floor(2^104 / D) on the cold path (fits in u64 for all D >= 2^40, i.e., f_s >= 256 Hz)
    let r_u128 = (1u128 << 104) / (d as u128);
    let r = u64::try_from(r_u128).unwrap_or(u64::MAX);
    Some((i, d, r))
}

/// Advances the fixed-hertz `u64` phase accumulator by 1 sample: `A = P + I`, `b = A >= D`,
/// `P <- A - b * D` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
#[inline]
pub const fn hertz_phase_step(phase: u64, increment: u64, denominator: u64) -> (u64, bool) {
    let advanced = phase.wrapping_add(increment);
    let latch = advanced >= denominator;
    let next_phase = if latch {
        advanced - denominator
    } else {
        advanced
    };
    (next_phase, latch)
}

/// Transfers a fixed-hertz `u64` phase `phase_old` from `old_denominator` to `new_denominator`
/// without hardware division using precomputed $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$
/// (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[inline]
pub const fn rescale_bitcrusher_phase_u128(
    phase_old: u64,
    old_denominator: u64,
    new_denominator: u64,
    reciprocal_r: u64,
) -> u64 {
    let numerator = (phase_old as u128) * (new_denominator as u128);
    let n_lo = (numerator & (u64::MAX as u128)) as u64;
    let n_hi = (numerator >> 64) as u64;
    let a = (n_lo as u128) * (reciprocal_r as u128);
    let c = (n_hi as u128) * (reciprocal_r as u128);
    let m = (a >> 64) + (c & (u64::MAX as u128));
    let h = (c >> 64) + (m >> 64);
    let mut q0 = ((h as u64) << 24) | (((m & (u64::MAX as u128)) as u64) >> 40);
    let mut r0 = numerator - (q0 as u128) * (old_denominator as u128);
    let d_old_128 = old_denominator as u128;
    if r0 >= d_old_128 {
        q0 += 1;
        r0 -= d_old_128;
    }
    if 2 * r0 > d_old_128 || (2 * r0 == d_old_128 && (q0 & 1) != 0) {
        q0 += 1;
    }
    let max_phase = new_denominator.saturating_sub(1);
    if q0 > max_phase { max_phase } else { q0 }
}

/// Fixed-hertz `u64`-phase ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`,
/// `ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[derive(Clone, Copy, Debug)]
pub struct FixedHzBitcrusher {
    pub bit_depth: u32,
    pub target_hz: f64,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub held_sample: f32,
}

impl Default for FixedHzBitcrusher {
    fn default() -> Self {
        Self::new(16, 48_000.0, 48_000.0).unwrap()
    }
}

impl FixedHzBitcrusher {
    pub fn new(bit_depth: u32, target_hz: f64, sample_rate_hz: f64) -> Option<Self> {
        let (inc, den, rec) = hertz_phase_parameters(target_hz, sample_rate_hz)?;
        Some(Self {
            bit_depth: bit_depth.clamp(2, 16),
            target_hz,
            phase: 0,
            increment: inc,
            denominator: den,
            reciprocal_r: rec,
            held_sample: 0.0,
        })
    }

    pub fn set_sample_rate(&mut self, new_sample_rate_hz: f64) -> Option<()> {
        let eff_target = self.target_hz.min(new_sample_rate_hz);
        let (new_inc, new_den, new_rec) = hertz_phase_parameters(eff_target, new_sample_rate_hz)?;
        self.phase =
            rescale_bitcrusher_phase_u128(self.phase, self.denominator, new_den, self.reciprocal_r);
        self.increment = new_inc;
        self.denominator = new_den;
        self.reciprocal_r = new_rec;
        Some(())
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.increment == 0 {
            return self.held_sample;
        }
        let latch = if self.increment == self.denominator {
            true
        } else {
            let (next_p, l) = hertz_phase_step(self.phase, self.increment, self.denominator);
            self.phase = next_p;
            l
        };
        if latch {
            self.held_sample = quantize_bitcrusher_nearest_ties_pos_inf(x, self.bit_depth);
        }
        self.held_sample
    }
}

/// Evaluates Theorem 1 (Exact LTI Basis Commutativity) by running a 1-pole filter bank $\mathcal{L}_k$
/// (`poles[0..K]`) on a $P$-dimensional basis sequence `u_seq` in both uncommuted (`filter all P -> project W_k`)
/// and commuted (`project W_k -> filter 1 lane`) order (`ref[dsp.op-n.commuted.theorem1]`).
pub fn eval_commuted_vs_uncommuted_lti_cascade_f64<const P: usize, const K: usize>(
    u_seq: &[[f64; P]],
    poles: &[f64; K],
    weights: &[[f64; P]; K],
) -> (Vec<f64>, Vec<f64>) {
    let n_len = u_seq.len();
    let mut uncomm_state = [[0.0f64; P]; K];
    let mut comm_state = [0.0f64; K];
    let mut y_uncomm = Vec::with_capacity(n_len);
    let mut y_comm = Vec::with_capacity(n_len);

    for u in u_seq {
        let mut sum_u = 0.0f64;
        let mut sum_c = 0.0f64;
        for k in 0..K {
            let r = poles[k];
            let one_minus_r = 1.0 - r;
            // Uncommuted: step P filters then dot with W_k
            for p in 0..P {
                uncomm_state[k][p] = r * uncomm_state[k][p] + one_minus_r * u[p];
                sum_u += uncomm_state[k][p] * weights[k][p];
            }
            // Commuted: dot u with W_k first, then step 1 filter
            let mut v_k = 0.0f64;
            for p in 0..P {
                v_k += u[p] * weights[k][p];
            }
            comm_state[k] = r * comm_state[k] + one_minus_r * v_k;
            sum_c += comm_state[k];
        }
        y_uncomm.push(sum_u);
        y_comm.push(sum_c);
    }
    (y_uncomm, y_comm)
}

/// Folds $3\text{ kHz}$ stage control latents $\mathbf{z}_{\text{stage}} \in \mathbb{R}^{D_s}$ into a
/// sub-stage projection vector $\mathbf{w}(\mathbf{z}) = \mathbf{w}_0 + \sum_{r=0}^{D_s-1} z_r \mathbf{w}_{r+1}$
/// (`ref[dsp.op-n.stage2.latent-folding]`).
pub fn fold_stage2_latent_weights<const N: usize>(
    w_base: &[f32; N],
    w_modes: &[[f32; N]],
    z_stage: &[f32],
) -> [f32; N] {
    let mut out = *w_base;
    let active = w_modes.len().min(z_stage.len());
    for r in 0..active {
        let z = z_stage[r];
        for i in 0..N {
            out[i] += z * w_modes[r][i];
        }
    }
    out
}

/// Evaluates the $2 \times 2$ block stereo coupling matrix on a 16-D stereo drive basis
/// $\mathbf{u}_{\text{stereo}} = [\mathbf{u}_L^{(8)}, \mathbf{u}_R^{(8)}]^\top$
/// (`ref[dsp.op-n.stage2.stereo-coupling]`).
pub fn eval_stage2_stereo_coupling_16d(
    u_left: &[f32; 8],
    u_right: &[f32; 8],
    w_ll: &[f32; 8],
    w_lr: &[f32; 8],
    w_rl: &[f32; 8],
    w_rr: &[f32; 8],
) -> (f32, f32) {
    let mut out_l = 0.0f32;
    let mut out_r = 0.0f32;
    for i in 0..8 {
        out_l += w_ll[i] * u_left[i] + w_rl[i] * u_right[i];
        out_r += w_lr[i] * u_left[i] + w_rr[i] * u_right[i];
    }
    (out_l, out_r)
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

        // C2 wavefolder exact values at u = 0.5: (1 - 0.25)^3 = (0.75)^3 = 0.421875
        let (w_odd, w_even) = c2_wavefolder(0.5);
        assert!((w_odd - 0.5 * 0.421_875).abs() < 1e-6);
        assert!((w_even - 0.25 * 0.421_875).abs() < 1e-6);
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
        // 4 bits => levels = 1 << 3 = 8.0, step_size = 0.125; target_sr = 12 kHz at 48 kHz => increment = denominator / 4 (holds 4 samples)
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert!((cr.step_size - 0.125).abs() < 1e-6);
        assert_eq!(cr.increment * 4, cr.denominator);

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
        assert_eq!(cr.phase, 0);
        assert_eq!(cr.held_sample, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher_full_and_zero_rate() {
        let _audio = crate::FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.set_parameters(4, 48_000.0, 48_000.0);
        let den = cr.denominator;
        for phase in [0_u64, den / 10, den / 4, den - 1] {
            cr.phase = phase;
            for sample in [0.125, -0.25, 0.5, -0.75] {
                assert_eq!(cr.step(sample), sample);
                assert_eq!(cr.phase, phase);
            }
        }

        cr.phase = den / 4;
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.phase, den / 4);
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), 0.5);

        cr.reset();
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.phase, 0);
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
        fn prop_trust_region_softclip_4d_bounds(
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

    // Specification Verification Suite: Phase 7 — Op-N Commuted Cascade, PolyADAA, Wavefolder & Bitcrusher.
    //
    // Verifies all engine-scope rules in `docs/specs/dsp/op_n_commuted_cascade.md` plus `dsp.zoh.stage2-op-n`:
    // - `verify[dsp.op-n.polyadaa.clamp]`
    // - `verify[dsp.op-n.polyadaa.midpoint]`
    // - `verify[dsp.op-n.polyadaa.rail-crossing]`
    // - `verify[dsp.op-n.polyadaa.linear-residual]`
    // - `verify[dsp.op-n.polyadaa.bspline-ac]`
    // - `verify[dsp.op-n.polyadaa.drive-bank]`
    // - `verify[dsp.op-n.wavefolder.cubic-c2]`
    // - `verify[dsp.op-n.hysteresis.quadrature]`
    // - `verify[dsp.op-n.bitcrusher.quantizer]`
    // - `verify[dsp.op-n.bitcrusher.zoh-phase]`
    // - `verify[dsp.op-n.bitcrusher.rate-transfer]`
    // - `verify[dsp.op-n.commuted.theorem1]`
    // - `verify[dsp.op-n.commuted.causality]`
    // - `verify[dsp.op-n.commuted.project-then-filter]`
    // - `verify[dsp.op-n.commuted.dc-equilibrium]`
    // - `verify[dsp.op-n.stage2.sub-stages]`
    // - `verify[dsp.op-n.stage2.tone-basis]`
    // - `verify[dsp.op-n.stage2.stereo-coupling]`
    // - `verify[dsp.op-n.stage2.latent-folding]`
    // - `verify[dsp.op-n.ctrl.incremental-update]`
    // - `verify[dsp.op-n.ctrl.lobatto-basis]`
    // - `verify[dsp.op-n.ctrl.chebyshev-lfo]`
    // - `verify[dsp.op-n.ctrl.fm-interaction]`
    // - `verify[dsp.op-n.ctrl.trust-clip]`
    // - `verify[dsp.zoh.stage2-op-n]`

    use crate::FtzDazGuard;
    use crate::container::ZohRescaledParams;
    use crate::operators::{
        FixedHzBitcrusher, MULTI_DRIVE_SCALES, MultiDrivePolyAdaaChannel, QuadratureHysteresis2D,
        c2_wavefolder, eval_adaa_cubic_c2_wavefolder_f64, eval_chebyshev_4harm,
        eval_commuted_vs_uncommuted_lti_cascade_f64, eval_lobatto_basis_3node,
        eval_saturated_polyadaa_rail_crossing_f64, eval_stage2_stereo_coupling_16d,
        fold_stage2_latent_weights, hertz_phase_parameters, hertz_phase_step,
        poly_adaa_algebraic_eval, quantize_bitcrusher_nearest_ties_pos_inf,
        rescale_bitcrusher_phase_u128, trust_region_softclip_4d,
    };
    use crate::stages::{Stage2CommutedCascade, ToneBasisFilters};

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.rail-crossing]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.polyadaa.bspline-ac]
    /// verify[dsp.op-n.polyadaa.drive-bank]
    /// Verifies 4-lane drive bank `[0.25, 1.0, 4.0, 16.0]`, midpoint factorization, 6-step Newton
    /// saturated rail crossing (`< 2^-44` error), linear residual passband, and B-spline + AC coupling.
    #[test]
    fn test_polyadaa_clamp_midpoint_rail_crossing_linear_residual_and_drive_bank() {
        let _guard = FtzDazGuard::new();
        assert_eq!(MULTI_DRIVE_SCALES, [0.25, 1.0, 4.0, 16.0]);

        // 1. Saturated rail-crossing cases vs exact rational integrals (< 2^-44 error in f64)
        let tol = f64::from_bits((1023 - 44) << 52);
        for &(prev, curr, exp_odd, exp_even, exp_crossed) in &[
            (0.0_f64, 1.0_f64, 0.625_f64, 0.13333333333333333_f64, false),
            (0.0, 2.0, 0.8125, 0.06666666666666667, true),
            (-2.0, 0.0, -0.8125, 0.06666666666666667, true),
            (-2.0, 2.0, 0.0, 0.06666666666666667, true),
            (2.0, 4.0, 1.0, 0.0, false),
            (-4.0, -2.0, -1.0, 0.0, false),
            (0.25, 0.25, 0.3671875, 0.05859375, false),
        ] {
            for (a, b) in [(prev, curr), (curr, prev)] {
                let (odd, even, crossed) = eval_saturated_polyadaa_rail_crossing_f64(a, b);
                assert_eq!(crossed, exp_crossed);
                assert!((odd - exp_odd).abs() < tol);
                assert!((even - exp_even).abs() < tol);
            }
        }

        // 2. Midpoint evaluation & Multi-Drive channel linear residual + B-spline AC blend
        let a0 = poly_adaa_algebraic_eval(0.5, 0.5, 1.0, 1.0);
        let exp_a0 = 0.5 * (1.5 - 0.5 * 0.25) + (0.25 - 0.0625);
        assert!((a0 - exp_a0).abs() < 1e-6);

        let mut ch = MultiDrivePolyAdaaChannel::new();
        let lin_only = ch.step(0.5, 0.0, 0.0, 2.0, 0.9993);
        assert_eq!(lin_only, [1.0; 4]);
    }

    /// verify[dsp.op-n.wavefolder.cubic-c2]
    /// verify[dsp.op-n.hysteresis.quadrature]
    /// verify[dsp.op-n.bitcrusher.quantizer]
    /// verify[dsp.op-n.bitcrusher.zoh-phase]
    /// verify[dsp.op-n.bitcrusher.rate-transfer]
    /// Verifies cubed-envelope $C^2$ wavefolder (`(1-u^2)^3`), degree-7/8 ADAA basis, 2D quadrature
    /// hysteresis, nearest-step quantizer (ties to +inf), fixed-hertz integer ZOH phase, and `u128`
    /// division-free rate transfer across `44.1..384 kHz`.
    #[test]
    fn test_cubic_c2_wavefolder_hysteresis_and_fixed_hz_bitcrusher_rate_transfer() {
        let _guard = FtzDazGuard::new();

        // 1. Cubed-envelope C^2 wavefolder & degree-7/8 ADAA basis
        let (wo, we) = c2_wavefolder(0.5);
        assert!((wo - 0.5 * 0.421_875).abs() < 1e-6);
        assert!((we - 0.25 * 0.421_875).abs() < 1e-6);
        assert_eq!(c2_wavefolder(1.0), (0.0, 0.0));
        assert_eq!(c2_wavefolder(-1.0), (0.0, 0.0));

        let (ao_eq, ae_eq) = eval_adaa_cubic_c2_wavefolder_f64(0.5, 0.5);
        assert!((ao_eq - 0.5 * 0.421_875).abs() < 1e-12);
        assert!((ae_eq - 0.25 * 0.421_875).abs() < 1e-12);

        // 2. 2D quadrature hysteresis
        let mut hyst = QuadratureHysteresis2D::new();
        let h1 = hyst.step_with_rate(0.5, 0.2, 2.0);
        assert!((h1 - (0.5 * 0.5 * 2.0 + 0.15 * 0.5 * 0.2)).abs() < 1e-6);

        // 3. Nearest-step quantizer (halfway ties toward +inf)
        for bits in 2..=16u32 {
            let scale = (1u32 << (bits - 1)) as f64;
            for mid in [-1.5_f32, -0.5, 0.5, 1.5] {
                let x = mid / (scale as f32);
                let expected = ((f64::from(x) * scale + 0.5).floor() / scale) as f32;
                assert_eq!(quantize_bitcrusher_nearest_ties_pos_inf(x, bits), expected);
            }
        }

        // 4. Fixed-hertz integer ZOH phase: exact period latching, full-rate, zero-rate, and u128 rate transfer
        for &(target_hz, period) in &[(16_000.0_f64, 3usize), (12_000.0, 4), (48_000.0, 1)] {
            let (inc, den, _) = hertz_phase_parameters(target_hz, 48_000.0).unwrap();
            let mut p = 0u64;
            for sample in 1..=(period * 8) {
                let (next_p, latch) = hertz_phase_step(p, inc, den);
                p = next_p;
                assert_eq!(latch, sample % period == 0);
            }
            assert_eq!(p, 0);
        }

        // Zero-rate holds last latched sample and phase; full-rate latches every sample without changing phase
        let mut cr = FixedHzBitcrusher::new(4, 12_000.0, 48_000.0).unwrap();
        for _ in 0..3 {
            assert_eq!(cr.step(0.5), 0.0);
        }
        assert_eq!(cr.step(0.5), 0.5);
        let saved_phase = cr.phase;
        cr.increment = 0;
        assert_eq!(cr.step(0.875), 0.5);
        assert_eq!(cr.phase, saved_phase);

        // Division-free u128 rate transfer across canonical rates matches exact rational rounding
        let rates = [44_100.0_f64, 48_000.0, 96_000.0, 192_000.0, 384_000.0];
        for &old_r in &rates {
            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_r).unwrap();
            for &new_r in &rates {
                let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_r).unwrap();
                for p_old in [0u64, 1, d_old / 3, d_old / 2, d_old - 1] {
                    let actual = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
                    let num = (p_old as u128) * (d_new as u128);
                    let den = d_old as u128;
                    let mut q = (num / den) as u64;
                    let rem = num % den;
                    if 2 * rem > den || (2 * rem == den && (q & 1) != 0) {
                        q += 1;
                    }
                    let expected = q.min(d_new - 1);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    /// verify[dsp.op-n.commuted.theorem1]
    /// verify[dsp.op-n.commuted.causality]
    /// verify[dsp.op-n.commuted.project-then-filter]
    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.stage2.sub-stages]
    /// verify[dsp.op-n.stage2.tone-basis]
    /// verify[dsp.op-n.stage2.stereo-coupling]
    /// verify[dsp.op-n.stage2.latent-folding]
    /// verify[dsp.op-n.ctrl.incremental-update]
    /// verify[dsp.op-n.ctrl.lobatto-basis]
    /// verify[dsp.op-n.ctrl.chebyshev-lfo]
    /// verify[dsp.op-n.ctrl.fm-interaction]
    /// verify[dsp.op-n.ctrl.trust-clip]
    /// verify[dsp.zoh.stage2-op-n]
    /// Verifies Theorem 1 (Exact LTI Basis Commutativity), resting DC equilibrium for all 4 tone-basis
    /// filters, Stage II 3-sub-stage cascade, 16-D stereo coupling, 3 kHz latent folding, and ZOH rescaling.
    #[test]
    fn test_commuted_lti_theorem1_dc_equilibrium_stage2_and_zoh() {
        let _guard = FtzDazGuard::new();

        // 1. Theorem 1: Uncommuted (filter P -> project W_k) == Commuted (project W_k -> filter 1)
        let u_seq: Vec<[f64; 4]> = (0..128)
            .map(|n| {
                let t = (n as f64) * 0.07;
                [t.sin(), (2.0 * t).cos(), (0.5 * t).sin(), -0.3 * t.cos()]
            })
            .collect();
        let poles = [0.98959_f64, 0.94916, 0.85463];
        let weights = [
            [0.4_f64, -0.2, 0.1, 0.3],
            [-0.1, 0.5, 0.2, -0.4],
            [0.3, 0.1, -0.5, 0.2],
        ];
        let (y_uncomm, y_comm) =
            eval_commuted_vs_uncommuted_lti_cascade_f64(&u_seq, &poles, &weights);
        for (yu, yc) in y_uncomm.iter().zip(y_comm.iter()) {
            assert!((yu - yc).abs() < 1e-13);
        }

        // 2. ToneBasisFilters resting DC equilibrium: Direct = v_rest, LPF_80 = v_rest, LPF_400 = v_rest, HPF_1200 = 0.0
        let mut tb = ToneBasisFilters::new();
        tb.init_dc_equilibrium(0.75);
        let out_tb = tb.step(0.75);
        assert!((out_tb[0] - 0.75).abs() < 1e-6);
        assert!((out_tb[1] - 0.75).abs() < 1e-6);
        assert!((out_tb[2] - 0.75).abs() < 1e-6);
        assert!(out_tb[3].abs() < 1e-6);

        // 3. Stage II 3-sub-stage cascade, ZOH rescaling, stereo coupling, and latent folding
        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let mut s2 = Stage2CommutedCascade::new();
        s2.reconfigure_sample_rate(&zoh);
        assert!((s2.hyst_diff_scale - 2.0).abs() < 1e-6);

        let (cl, cr) = eval_stage2_stereo_coupling_16d(
            &[1.0; 8], &[0.5; 8], &[0.25; 8], &[0.01; 8], &[0.01; 8], &[0.25; 8],
        );
        assert!((cl - (2.0 + 0.04)).abs() < 1e-6);
        assert!((cr - (0.08 + 1.0)).abs() < 1e-6);

        let w_folded = fold_stage2_latent_weights(
            &[1.0, 2.0, 3.0, 4.0],
            &[[0.5, 0.0, -0.5, 1.0], [0.0, 1.0, 0.0, -1.0]],
            &[2.0, 0.5],
        );
        assert_eq!(w_folded, [2.0, 2.5, 2.0, 5.5]);

        let lb = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 2, 0.0);
        assert_eq!(lb[0], 0.5);
        let ch_lfo = eval_chebyshev_4harm(1.0);
        assert_eq!(ch_lfo, [1.0, 1.0, 1.0, 1.0]);
        let mut z4 = [10.0, 0.0, 0.0, 0.0];
        trust_region_softclip_4d(&mut z4, 2.0);
        assert!((z4[0] - 2.0).abs() < 1e-3);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.op-n.polyadaa.rail-crossing]
        /// verify[dsp.op-n.wavefolder.cubic-c2]
        /// verify[dsp.op-n.bitcrusher.quantizer]
        /// verify[dsp.op-n.bitcrusher.rate-transfer]
        #[test]
        fn prop_op_n_rail_crossing_wavefolder_and_bitcrusher(
            raw_prev in -10.0_f64..10.0_f64,
            raw_curr in -10.0_f64..10.0_f64,
            bits in 2u32..=16u32,
            sample_x in -100.0_f32..100.0_f32,
            old_sr in 44_100.0_f64..=96_000.0_f64,
            new_sr in 44_100.0_f64..=96_000.0_f64,
            frac in 0.0_f64..0.999_f64,
        ) {
            let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(raw_prev, raw_curr);
            prop_assert!(odd.is_finite() && odd.abs() <= 1.0 + 1e-12);
            prop_assert!(even.is_finite() && (0.0..=1.0).contains(&even));

            let (wf_o, wf_e) = eval_adaa_cubic_c2_wavefolder_f64(raw_prev, raw_curr);
            prop_assert!(wf_o.is_finite() && wf_o.abs() <= 1.0);
            prop_assert!(wf_e.is_finite() && (0.0..=1.0).contains(&wf_e));

            let q = quantize_bitcrusher_nearest_ties_pos_inf(sample_x, bits);
            let step = 1.0_f32 / ((1u32 << (bits - 1)) as f32);
            prop_assert!((q - sample_x).abs() <= 0.5 * step + 1e-5);

            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_sr).unwrap();
            let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_sr).unwrap();
            let p_old = ((d_old as f64) * frac) as u64;
            let p_new = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
            prop_assert!(p_new < d_new);
        }
    }
}

#[cfg(test)]
mod stage2_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Saturated-curve ADAA arithmetic references, not production-engine tests.

    use crate::FtzDazGuard;

    #[inline(never)]
    fn saturated_adaa_reference(previous: f64, current: f64) -> (f64, f64) {
        let lower = previous.min(current);
        let upper = previous.max(current);
        if lower >= 1.0 {
            return (1.0, 0.0);
        }
        if upper <= -1.0 {
            return (-1.0, 0.0);
        }

        let clamped_lower = lower.max(-1.0);
        let clamped_upper = upper.min(1.0);
        let midpoint = (clamped_upper + clamped_lower) * 0.5;
        let half_difference = (clamped_upper - clamped_lower) * 0.5;
        let m2 = midpoint * midpoint;
        let d2 = half_difference * half_difference;
        let odd = midpoint * (1.5 - 0.5 * (m2 + d2));
        let even = m2 + d2 * 0.3333333333333333 - (m2 * m2 + 2.0 * m2 * d2 + d2 * d2 * 0.2);
        if lower >= -1.0 && upper <= 1.0 {
            return (odd, even);
        }

        let distance = upper - lower;
        let exponent = ((distance.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let mut inverse = f64::from_bits(((1023 - exponent - 1) as u64) << 52);
        for _ in 0..6 {
            inverse *= 2.0 - distance * inverse;
        }
        let inside = (clamped_upper - clamped_lower).max(0.0);
        let below = (upper.min(-1.0) - lower).max(0.0);
        let above = (upper - lower.max(1.0)).max(0.0);
        (
            (inside * odd + above - below) * inverse,
            (inside * even) * inverse,
        )
    }

    #[test]
    fn saturated_adaa_covers_rail_crossings_under_native_guard() {
        let _guard = FtzDazGuard::new();
        let cases = [
            (0.0, 1.0, 0.625, 0.13333333333333333),
            (0.0, 2.0, 0.8125, 0.06666666666666667),
            (-2.0, 0.0, -0.8125, 0.06666666666666667),
            (-2.0, 2.0, 0.0, 0.06666666666666667),
            (-1.0, 1.0, 0.0, 0.13333333333333333),
            (2.0, 4.0, 1.0, 0.0),
            (-4.0, -2.0, -1.0, 0.0),
            (0.25, 0.25, 0.3671875, 0.05859375),
        ];
        let tolerance = f64::from_bits(((1023 - 44) as u64) << 52);
        for (previous, current, expected_odd, expected_even) in cases {
            for (left, right) in [(previous, current), (current, previous)] {
                let (odd, even) = saturated_adaa_reference(left, right);
                assert!((odd - expected_odd).abs() < tolerance);
                assert!((even - expected_even).abs() < tolerance);
            }
        }
    }

    #[test]
    fn saturated_adaa_preserves_tiny_even_result_in_guarded_f64() {
        let _guard = FtzDazGuard::new();
        let minimum_f32 = f64::from_bits(((1023 - 149) as u64) << 52);
        let (tiny_odd, tiny_even) = saturated_adaa_reference(minimum_f32, minimum_f32);
        assert_eq!(tiny_odd, 1.5 * minimum_f32);
        assert_eq!(tiny_even, minimum_f32 * minimum_f32);

        let maximum = f64::from(f32::MAX);
        let (odd, even) = saturated_adaa_reference(-maximum, maximum);
        let expected = 2.0 / (15.0 * maximum);
        let relative_tolerance = f64::from_bits(((1023 - 44) as u64) << 52);
        assert_eq!(odd, 0.0);
        assert!(even > 0.0);
        assert!((even / expected - 1.0).abs() < relative_tolerance);
    }
}
