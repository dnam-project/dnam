//! Default algorithmic implementations for MathBackend methods.
//!
//! Canonical Reference: `docs/dsp/op_n_commuted_cascade.md` and `docs/overview/tora_architecture.md`.

use super::{MathBackend, SimdVec4};

#[inline(always)]
pub fn poly_adaa_algebraic_4ch<B: MathBackend>(
    u_prev: B::Vec4,
    u_curr: B::Vec4,
    beta: B::Vec4,
    delta: B::Vec4,
) -> B::Vec4 {
    let half = B::Vec4::splat(0.5);
    let one_pt_five = B::Vec4::splat(1.5);
    let one_third = B::Vec4::splat(1.0 / 3.0);
    let one_fifth = B::Vec4::splat(0.2);
    let two = B::Vec4::splat(2.0);

    let m = u_curr.add(u_prev).mul(half);
    let d = u_curr.sub(u_prev).mul(half);
    let m2 = m.mul(m);
    let d2 = d.mul(d);
    let m2_plus_d2 = m2.add(d2);

    let odd_factor = one_pt_five.fms(half, m2_plus_d2);
    let odd_term = beta.mul(m).mul(odd_factor);

    let quad_part = m2.fma(one_third, d2);
    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let quart_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let even_term = delta.mul(quad_part.sub(quart_part));

    odd_term.add(even_term)
}

#[inline(always)]
pub fn fma_mat4x4_vec4<B: MathBackend>(acc: B::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> B::Vec4 {
    let mut out = acc;
    out = out.fma(B::Vec4::splat(x[0]), B::Vec4::load(&mat[0]));
    out = out.fma(B::Vec4::splat(x[1]), B::Vec4::load(&mat[1]));
    out = out.fma(B::Vec4::splat(x[2]), B::Vec4::load(&mat[2]));
    out = out.fma(B::Vec4::splat(x[3]), B::Vec4::load(&mat[3]));
    out
}

#[inline(always)]
pub fn conv6_taps_1group<B: MathBackend>(
    bias: B::Vec4,
    w_tiles: &[[[f32; 4]; 4]; 6],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> B::Vec4 {
    let mut acc0 = bias;
    let mut acc1 = B::Vec4::zero();
    acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[0], &h_taps[0][0]);
    acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[1], &h_taps[1][0]);
    acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[2], &h_taps[2][0]);
    acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[3], &h_taps[3][0]);
    acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[4], &h_taps[4][0]);
    acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[5], &h_taps[5][0]);
    acc0.add(acc1)
}

#[inline(always)]
pub fn conv6_taps_2groups<B: MathBackend>(
    bias: B::Vec4,
    w_tiles: &[[[f32; 4]; 4]; 12],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> B::Vec4 {
    let mut acc0 = bias;
    let mut acc1 = B::Vec4::zero();
    let mut acc2 = B::Vec4::zero();
    let mut acc3 = B::Vec4::zero();
    for pair in 0..3 {
        let t_even = pair * 2;
        let t_odd = pair * 2 + 1;
        let h_e = h_taps[t_even];
        let h_o = h_taps[t_odd];
        acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[t_even * 2], &h_e[0]);
        acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[t_even * 2 + 1], &h_e[1]);
        acc2 = fma_mat4x4_vec4::<B>(acc2, &w_tiles[t_odd * 2], &h_o[0]);
        acc3 = fma_mat4x4_vec4::<B>(acc3, &w_tiles[t_odd * 2 + 1], &h_o[1]);
    }
    acc0.add(acc1).add(acc2.add(acc3))
}

#[inline(always)]
pub fn conv6_taps_3groups<B: MathBackend>(
    bias: B::Vec4,
    w_tiles: &[[[f32; 4]; 4]; 18],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> B::Vec4 {
    let mut acc = bias;
    for t in 0..6 {
        acc = fma_mat4x4_vec4::<B>(acc, &w_tiles[t * 3], &h_taps[t][0]);
        acc = fma_mat4x4_vec4::<B>(acc, &w_tiles[t * 3 + 1], &h_taps[t][1]);
        acc = fma_mat4x4_vec4::<B>(acc, &w_tiles[t * 3 + 2], &h_taps[t][2]);
    }
    acc
}

#[inline(always)]
pub fn conv6_taps_4groups<B: MathBackend>(
    bias: B::Vec4,
    w_tiles: &[[[f32; 4]; 4]; 24],
    h_taps: &[&[[f32; 4]; 4]; 6],
) -> B::Vec4 {
    let mut acc0 = bias;
    let mut acc1 = B::Vec4::zero();
    for t in 0..6 {
        acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[t * 4], &h_taps[t][0]);
        acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[t * 4 + 1], &h_taps[t][1]);
        acc0 = fma_mat4x4_vec4::<B>(acc0, &w_tiles[t * 4 + 2], &h_taps[t][2]);
        acc1 = fma_mat4x4_vec4::<B>(acc1, &w_tiles[t * 4 + 3], &h_taps[t][3]);
    }
    acc0.add(acc1)
}

#[inline(always)]
pub fn cubic_lagrange_weights_4ch<B: MathBackend>(
    vmu: B::Vec4,
) -> (B::Vec4, B::Vec4, B::Vec4, B::Vec4) {
    let one = B::Vec4::splat(1.0);
    let two = B::Vec4::splat(2.0);
    let inv_six = B::Vec4::splat(1.0 / 6.0);
    let neg_inv_six = B::Vec4::splat(-1.0 / 6.0);
    let half = B::Vec4::splat(0.5);
    let neg_half = B::Vec4::splat(-0.5);

    let mu_p1 = vmu.add(one);
    let mu_m1 = vmu.sub(one);
    let mu_m2 = vmu.sub(two);

    let c_m1 = vmu.mul(mu_m1).mul(mu_m2).mul(neg_inv_six);
    let c_0 = mu_p1.mul(mu_m1).mul(mu_m2).mul(half);
    let c_1 = mu_p1.mul(vmu).mul(mu_m2).mul(neg_half);
    let c_2 = mu_p1.mul(vmu).mul(mu_m1).mul(inv_six);
    (c_m1, c_0, c_1, c_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::{ScalarBackend, ScalarVec4};

    #[test]
    fn test_default_ops_scalar_backend() {
        let u_prev = ScalarVec4::splat(0.2);
        let u_curr = ScalarVec4::splat(0.5);
        let beta = ScalarVec4::splat(1.0);
        let delta = ScalarVec4::splat(0.5);
        let adaa = poly_adaa_algebraic_4ch::<ScalarBackend>(u_prev, u_curr, beta, delta);
        assert!(adaa.0[0].is_finite());

        let (c_m1, c_0, c_1, c_2) =
            cubic_lagrange_weights_4ch::<ScalarBackend>(ScalarVec4::splat(0.4));
        let sum = c_m1.add(c_0).add(c_1).add(c_2);
        assert!((sum.0[0] - 1.0).abs() < 1e-6);

        let bias = ScalarVec4::splat(0.1);
        let w_tile = [[0.1f32; 4]; 4];
        let h_tap = [[0.2f32; 4]; 4];

        // 1 group
        let w1 = [w_tile; 6];
        let h1 = [&h_tap; 6];
        let r1 = conv6_taps_1group::<ScalarBackend>(bias, &w1, &h1);
        assert!(r1.0[0].is_finite());

        // 2 groups
        let w2 = [w_tile; 12];
        let r2 = conv6_taps_2groups::<ScalarBackend>(bias, &w2, &h1);
        assert!(r2.0[0].is_finite());

        // 3 groups
        let w3 = [w_tile; 18];
        let r3 = conv6_taps_3groups::<ScalarBackend>(bias, &w3, &h1);
        assert!(r3.0[0].is_finite());

        // 4 groups
        let w4 = [w_tile; 24];
        let r4 = conv6_taps_4groups::<ScalarBackend>(bias, &w4, &h1);
        assert!(r4.0[0].is_finite());
    }
}
