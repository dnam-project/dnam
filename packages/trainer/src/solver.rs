#![forbid(unsafe_code)]
//! Analytical Variable-Projection (`VarPro`), `uplo='U'` Cholesky & Noise-Subtracted `B-ESR` Solver.

pub mod stages;

use dnam_engine::Tier;

/// Returns the canonical feature-bus dimension $N_T \in \{64, 128, 192, 256\}$ for `tier`.
#[must_use]
pub const fn tier_feature_dimension(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 64,
        Tier::Full => 128,
        Tier::Pro => 192,
        Tier::Ultra => 256,
    }
}

/// Returns the packed upper-triangular length $N(N + 1)/2$ for a symmetric $N \times N$ Gram matrix
/// (`ref[container.ckpt.uplo-u]`), avoiding `dim * (dim + 1)` intermediate overflow on 32-bit targets.
#[must_use]
pub const fn uplo_u_packed_len(dim: usize) -> usize {
    if dim.is_multiple_of(2) {
        (dim / 2).saturating_mul(dim.saturating_add(1))
    } else {
        dim.saturating_mul(dim.div_ceil(2))
    }
}

#[inline]
const fn checked_uplo_u_packed_len(dim: usize) -> Option<usize> {
    if dim.is_multiple_of(2) {
        match dim.checked_add(1) {
            Some(d1) => (dim / 2).checked_mul(d1),
            None => None,
        }
    } else {
        dim.checked_mul(dim.div_ceil(2))
    }
}

/// Maps upper-triangular coordinates `(i, j)` with `0 <= i <= j < dim` to their column-major
/// `uplo='U'` packed index $i + j(j + 1)/2$.
#[must_use]
pub const fn uplo_u_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    i + if j.is_multiple_of(2) {
        (j / 2) * (j + 1)
    } else {
        j * j.div_ceil(2)
    }
}

/// Accumulates a weighted feature outer product $\mathbf{G} \mathrel{+}= w \boldsymbol{\phi} \boldsymbol{\phi}^\top$
/// and cross-correlation $\mathbf{C} \mathrel{+}= w \boldsymbol{\phi} \mathbf{y}^\top$ in `f64`
/// (`ref[train.gram.accuracy-contract]`, `ref[container.ckpt.standalone-full]`).
pub fn accumulate_uplo_u_gram_and_cross(
    packed_gram: &mut [f64],
    cross_corr: &mut [[f64; 2]],
    features: &[f64],
    target_lr: [f64; 2],
    weight: f64,
) {
    let n = features.len();
    assert_eq!(Some(packed_gram.len()), checked_uplo_u_packed_len(n));
    assert_eq!(cross_corr.len(), n);
    if !weight.is_finite()
        || weight <= 0.0
        || !target_lr[0].is_finite()
        || !target_lr[1].is_finite()
        || !features.iter().all(|v| v.is_finite())
    {
        return;
    }
    for j in 0..n {
        let w_fj = weight * features[j];
        let base = uplo_u_index(0, j);
        for i in 0..=j {
            packed_gram[base + i] += w_fj * features[i];
        }
        cross_corr[j][0] += w_fj * target_lr[0];
        cross_corr[j][1] += w_fj * target_lr[1];
    }
}

/// Solves the regularized symmetric positive-definite linear system
/// $(\mathbf{G} + \lambda \mathbf{I}) \mathbf{W} = \mathbf{C}$ from a `uplo='U'` packed Gram matrix
/// via Cholesky factorization (`ref[train.pipeline.sweep2-solve]`, `ref[train.resume.kronecker-reassembly]`).
pub fn solve_uplo_u_cholesky_2ch(
    packed_gram: &[f64],
    cross_corr: &[[f64; 2]],
    ridge_lambda: f64,
) -> Result<Vec<[f64; 2]>, &'static str> {
    let n = cross_corr.len();
    if Some(packed_gram.len()) != checked_uplo_u_packed_len(n) {
        return Err("packed_gram length mismatch");
    }
    if !ridge_lambda.is_finite()
        || ridge_lambda < 0.0
        || !packed_gram.iter().all(|v| v.is_finite())
        || !cross_corr
            .iter()
            .all(|c| c[0].is_finite() && c[1].is_finite())
    {
        return Err("non-finite solver input");
    }
    let lambda = ridge_lambda.max(1.0e-12);
    let mut r_upper = packed_gram.to_vec();
    for j in 0..n {
        r_upper[uplo_u_index(j, j)] += lambda;
    }

    // Upper-triangular Cholesky: G + lambda*I = R^T R
    for j in 0..n {
        for i in 0..j {
            let mut sum = r_upper[uplo_u_index(i, j)];
            for k in 0..i {
                sum -= r_upper[uplo_u_index(k, i)] * r_upper[uplo_u_index(k, j)];
            }
            r_upper[uplo_u_index(i, j)] = sum / r_upper[uplo_u_index(i, i)];
        }
        let mut diag = r_upper[uplo_u_index(j, j)];
        for k in 0..j {
            let r_kj = r_upper[uplo_u_index(k, j)];
            diag -= r_kj * r_kj;
        }
        if !diag.is_finite() || diag <= 0.0 {
            return Err("matrix is not positive definite");
        }
        r_upper[uplo_u_index(j, j)] = diag.sqrt();
    }

    // Forward substitution: R^T Y = C
    let mut y = vec![[0.0_f64; 2]; n];
    for i in 0..n {
        let mut s0 = cross_corr[i][0];
        let mut s1 = cross_corr[i][1];
        for k in 0..i {
            let r_ki = r_upper[uplo_u_index(k, i)];
            s0 -= r_ki * y[k][0];
            s1 -= r_ki * y[k][1];
        }
        let diag = r_upper[uplo_u_index(i, i)];
        y[i][0] = s0 / diag;
        y[i][1] = s1 / diag;
    }

    // Back substitution: R W = Y
    let mut w = vec![[0.0_f64; 2]; n];
    for i in (0..n).rev() {
        let mut s0 = y[i][0];
        let mut s1 = y[i][1];
        for k in (i + 1)..n {
            let r_ik = r_upper[uplo_u_index(i, k)];
            s0 -= r_ik * w[k][0];
            s1 -= r_ik * w[k][1];
        }
        let diag = r_upper[uplo_u_index(i, i)];
        w[i][0] = s0 / diag;
        w[i][1] = s1 / diag;
    }

    if !w.iter().all(|r| r[0].is_finite() && r[1].is_finite()) {
        return Err("non-finite solver output");
    }

    Ok(w)
}

#[inline]
fn eval_bezout_degree7_poly(h: f64) -> f64 {
    let u = if h.is_finite() {
        h.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if u <= 0.5 {
        let u2 = u * u;
        let u4 = u2 * u2;
        u4 * (35.0 - 84.0 * u + 70.0 * u2 - 20.0 * u * u2)
    } else {
        let v = 1.0 - u;
        let v2 = v * v;
        let v4 = v2 * v2;
        1.0 - v4 * (35.0 - 84.0 * v + 70.0 * v2 - 20.0 * v * v2)
    }
}

/// Evaluates the canonical degree-7 Binomial/Bézout 3-band partition
/// $B_7(h) = 35 h^4 - 84 h^5 + 70 h^6 - 20 h^7$ ($B_7(h) + B_7(1 - h) \equiv 1$) across ordered
/// normalized responses $h_1 = (1 - s)^2 \le h_2 = 1 - s^2$ and returns `(h_low, h_mid, h_high)`
/// (`ref[qual.besr.bezout-crossover]`).
#[must_use]
pub fn bezout_three_band_weights(s: f64) -> (f64, f64, f64) {
    let s_clamped = if s.is_finite() {
        s.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let one_minus_s = 1.0 - s_clamped;
    let l1 = eval_bezout_degree7_poly(one_minus_s * one_minus_s);
    let l2 = eval_bezout_degree7_poly(1.0 - s_clamped * s_clamped);
    let h_low = l1;
    let h_mid = (l2 - l1).max(0.0);
    let h_high = eval_bezout_degree7_poly(s_clamped * s_clamped);
    (h_low, h_mid, h_high)
}

/// Computes the canonical §1.2 sum-denominator noise-subtracted Error-to-Signal Ratio
/// $\max(0, R - F) / (Y + F)$ with exact zero-denominator semantics (`0.0` when $Y + F = 0, R = 0$;
/// `+inf` when $Y + F = 0, R > 0$ or on invalid inputs; `ref[qual.besr.noise-subtracted]`,
/// `ref[qual.besr.zero-denominator]`).
#[must_use]
pub fn compute_noise_subtracted_besr(
    err_energy: f64,
    target_energy: f64,
    noise_energy: f64,
) -> f64 {
    if !err_energy.is_finite()
        || !target_energy.is_finite()
        || !noise_energy.is_finite()
        || err_energy < 0.0
        || target_energy < 0.0
        || noise_energy < 0.0
    {
        return f64::INFINITY;
    }
    let denom = target_energy + noise_energy;
    if denom.is_finite() {
        if denom > 0.0 {
            (err_energy - noise_energy).max(0.0) / denom
        } else if err_energy == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        let half_denom = 0.5 * target_energy + 0.5 * noise_energy;
        (0.5 * err_energy - 0.5 * noise_energy).max(0.0) / half_denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_cholesky_solver_and_error_paths() {
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[1.0, 2.0], [3.0, 4.0]], 1.0e-6).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[-10.0], &[[1.0, 2.0]], 1.0e-6).is_err());

        let mut g = vec![0.0_f64; uplo_u_packed_len(2)];
        let mut c = vec![[0.0_f64; 2]; 2];
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[1.0, 0.0], [2.5, -1.5], 0.0);
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[1.0, 0.0], [2.5, -1.5], f64::NAN);
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[f64::NAN, 0.0], [2.5, -1.5], 1.0);
        assert_eq!(g, vec![0.0, 0.0, 0.0]);
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[f64::NAN, 0.0]], 1.0e-6).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[1.0, 0.0]], f64::NAN).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[0.0], &[[1.0e305, 0.0]], 1.0e-12).is_err());
        assert_eq!(bezout_three_band_weights(f64::NAN), (1.0, 0.0, 0.0));

        assert_eq!(compute_noise_subtracted_besr(0.0, 0.0, 0.0), 0.0);
        assert!(compute_noise_subtracted_besr(1.0e-15, 0.0, 0.0).is_infinite());
        assert!(compute_noise_subtracted_besr(-1.0, 1.0, 0.0).is_infinite());
        assert!((compute_noise_subtracted_besr(0.011, 0.999, 0.001) - 0.01).abs() <= 1.0e-15);
    }

    proptest! {
        #[test]
        fn prop_bezout_identity_sums_to_unity(s in 0.0_f64..=1.0_f64) {
            let (low, mid, high) = bezout_three_band_weights(s);
            prop_assert!(low >= 0.0 && mid >= 0.0 && high >= 0.0);
            prop_assert!(((low + mid + high) - 1.0).abs() <= 1.0e-14);
        }

        #[test]
        fn prop_uplo_u_cholesky_recovers_exact_linear_weights(
            w00 in -5.0_f64..5.0_f64,
            w01 in -5.0_f64..5.0_f64,
            w10 in -5.0_f64..5.0_f64,
            w11 in -5.0_f64..5.0_f64,
        ) {
            let mut g = vec![0.0_f64; uplo_u_packed_len(2)];
            let mut c = vec![[0.0_f64; 2]; 2];
            let basis = [[1.0_f64, 0.2_f64], [-0.3_f64, 1.1_f64], [0.7_f64, -0.8_f64]];
            for row in basis {
                let y0 = row[0] * w00 + row[1] * w10;
                let y1 = row[0] * w01 + row[1] * w11;
                accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &row, [y0, y1], 1.0);
            }
            let solved = solve_uplo_u_cholesky_2ch(&g, &c, 1.0e-12).expect("SPD system should solve");
            prop_assert!((solved[0][0] - w00).abs() <= 1.0e-7);
            prop_assert!((solved[0][1] - w01).abs() <= 1.0e-7);
            prop_assert!((solved[1][0] - w10).abs() <= 1.0e-7);
            prop_assert!((solved[1][1] - w11).abs() <= 1.0e-7);
        }
    }
}
