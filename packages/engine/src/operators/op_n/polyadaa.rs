//! Op-N Multi-Drive 1x B-Spline PolyADAA Bank & Saturated Rail-Crossing Evaluation.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.linear-residual]`, `ref[dsp.op-n.polyadaa.bspline-ac]`, `ref[dsp.op-n.polyadaa.drive-bank]`).

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use proptest::prelude::*;

    #[test]
    fn test_poly_adaa_algebraic() {
        for &m in &[-0.8_f32, -0.4, 0.0, 0.3, 0.9] {
            let beta = 1.2_f32;
            let delta = -0.35_f32;
            let actual = poly_adaa_algebraic_eval(m, m, beta, delta);
            let expected = beta * m * (1.5 - 0.5 * m * m) + delta * (m * m - m * m * m * m);
            assert!((actual - expected).abs() < 1e-6);
        }

        let d = 0.6_f32;
        let a_sym = poly_adaa_algebraic_eval(-d, d, 1.0, 1.0);
        let expected_sym = (d * d) / 3.0 - 0.2 * d * d * d * d;
        assert!((a_sym - expected_sym).abs() < 1e-6);
    }

    #[test]
    fn test_multi_drive_poly_adaa_channel() {
        let _audio = FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        chan.reset();
        let out_lin = chan.step(0.4, 0.0, 0.0, 0.75, 0.9993);
        for &v in &out_lin {
            assert!((v - 0.3).abs() < 1e-6);
        }

        chan.reset();
        let out_nl = chan.step(0.2, 1.0, 0.0, 0.0, 0.9993);
        assert!(out_nl[0] < out_nl[1]);
        assert!(out_nl[1] < out_nl[2]);
        assert!(out_nl[2] < out_nl[3]);

        chan.reset();
        assert_eq!(chan.u_prev, [0.0; 4]);
    }

    #[test]
    fn test_poly_adaa_dc_equilibrium_startup() {
        let _audio = FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        let b_conv = 0.2_f32;
        chan.init_dc_equilibrium(b_conv, 1.0, 0.25);

        let out = chan.step(b_conv, 1.0, 0.25, 0.0, 0.9993);
        for &val in &out {
            assert!(
                val.abs() < 1e-4,
                "DC pop detected at startup from quiescent state: {}",
                val
            );
        }
    }

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.rail-crossing]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.polyadaa.bspline-ac]
    /// verify[dsp.op-n.polyadaa.drive-bank]
    #[test]
    fn test_polyadaa_clamp_midpoint_rail_crossing_linear_residual_and_drive_bank() {
        let _guard = FtzDazGuard::new();
        assert_eq!(MULTI_DRIVE_SCALES, [0.25, 1.0, 4.0, 16.0]);

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

        let a0 = poly_adaa_algebraic_eval(0.5, 0.5, 1.0, 1.0);
        let exp_a0 = 0.5 * (1.5 - 0.5 * 0.25) + (0.25 - 0.0625);
        assert!((a0 - exp_a0).abs() < 1e-6);

        let mut ch = MultiDrivePolyAdaaChannel::new();
        let lin_only = ch.step(0.5, 0.0, 0.0, 2.0, 0.9993);
        assert_eq!(lin_only, [1.0; 4]);
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
    }
}

#[cfg(test)]
mod stage2_reference_tests {
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
