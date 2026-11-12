#![forbid(unsafe_code)]
//! Analytical Variable-Projection (`VarPro`), `uplo='U'` Cholesky & Noise-Subtracted `B-ESR` Solver.

pub mod bezout_gram;
pub mod bezout_schur;
pub mod stage_a;
pub mod stage_a_id;
pub mod stage_b_c;
pub mod stage_d;
pub mod stage_e;
pub mod stages;
pub mod wide_gram;

pub use bezout_gram::{
    BezoutHarvestStats, canonical_besr_warmup, contract_bezout_3band_stats,
    contract_bezout_3band_stats_4tier, eval_4tier_besr_quad_1ch, eval_4tier_post_trim_besr_lin,
};
pub use bezout_schur::solve_multitier_besr_4tier;

pub use stage_a::{
    StageAConfig, StageAReport, has_populated_physical_backbone, lift_stage_a_backbone,
    zero_all_role_b_cascade_weights,
};
pub use stage_b_c::{
    StageBHarvest, StageCSolveOutput, compute_role_b_multipliers, harvest_star_center_256d,
    serialize_stat_chunk_ultra, solve_stage_c_nested_schur_4tier, write_effective_mode0_weights,
};
pub use stage_d::{
    StageDSolveOutput, solve_stage_d_cp1_and_audit, solve_stage_d_cp1_and_audit_with_harvest,
    star_center_control_values, write_inline_cp1_if_fits,
};
pub use stage_e::{
    BuyerWatermarkConfig, StageESolveOutput, compute_integrated_lufs, derive_buyer_gauge_params,
    derive_buyer_phase_seed, derive_buyer_tardos_codebook, export_runtime_dnam_watermarked,
    extract_arena_pole_angles_34, finalize_stage_e_and_telemetry,
    identify_pole_angles_blackbox_from_audio, render_blackbox_probe_response,
};
pub use wide_gram::{Wide192Accum, accumulate_uplo_u_wide192, contract_256d_gram_tiled_wide192};

/// Tuning options for the 5-Stage `< 25 ms` Closed-Form + `VarPro` Solver (`Stage A..E`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverOptions {
    /// Stage A anchor-stem lifting configuration (`preserve_existing_backbone` for white-box oracles).
    pub stage_a: StageAConfig,
    /// Stride when contracting `256 × 256` `Wide192Accum` Gram outer products (`default = 4` -> `3,000` frames).
    pub gram_frame_stride: usize,
    /// Clears `W_ctrl^base` (`9,536..10,560`) during Stage D unless preserving a white-box dynamic backbone.
    pub clear_dynamic_w_ctrl: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            stage_a: StageAConfig::default(),
            gram_frame_stride: 4,
            clear_dynamic_w_ctrl: true,
        }
    }
}

impl SolverOptions {
    /// Returns white-box oracle options that preserve an existing populated `64,000 B` physical
    /// backbone (`Stage 0..IV` + `W_ctrl`) while solving `W_head`, `CP1`, `U_int`, `STAT`, and trims.
    #[must_use]
    pub const fn preserve_backbone() -> Self {
        Self {
            stage_a: StageAConfig {
                preserve_existing_backbone: true,
                vp_gn_steps: 1,
            },
            gram_frame_stride: 2,
            clear_dynamic_w_ctrl: false,
        }
    }
}

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
/// (`ref[container.proj.stat-uplo-u]`), avoiding `dim * (dim + 1)` intermediate overflow on 32-bit targets.
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
/// (`ref[train.gram.accuracy-contract]`, `ref[container.proj.stat-uplo-u]`).
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

pub(crate) fn factorize_uplo_u_cholesky(
    dim: usize,
    packed_gram: &[f64],
    ridge_lambda: f64,
) -> Result<Vec<f64>, &'static str> {
    if Some(packed_gram.len()) != checked_uplo_u_packed_len(dim) {
        return Err("packed_gram length mismatch");
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 || !packed_gram.iter().all(|v| v.is_finite())
    {
        return Err("non-finite solver input");
    }
    let lambda = ridge_lambda.max(1.0e-12);
    let mut r_upper = packed_gram.to_vec();
    for j in 0..dim {
        r_upper[uplo_u_index(j, j)] += lambda;
    }

    // Upper-triangular Cholesky: G + lambda*I = R^T R (4-way unrolled column dot products)
    for j in 0..dim {
        let col_j = uplo_u_index(0, j);
        for i in 0..j {
            let col_i = uplo_u_index(0, i);
            let (ci_chunks, ci_rem) = r_upper[col_i..col_i + i].as_chunks::<4>();
            let (cj_chunks, cj_rem) = r_upper[col_j..col_j + i].as_chunks::<4>();
            let (mut d0, mut d1, mut d2, mut d3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for (a, b) in ci_chunks.iter().zip(cj_chunks) {
                d0 += a[0] * b[0];
                d1 += a[1] * b[1];
                d2 += a[2] * b[2];
                d3 += a[3] * b[3];
            }
            let mut dot = (d0 + d1) + (d2 + d3);
            for (&r_ki, &r_kj) in ci_rem.iter().zip(cj_rem) {
                dot += r_ki * r_kj;
            }
            r_upper[col_j + i] = (r_upper[col_j + i] - dot) / r_upper[col_i + i];
        }
        let mut diag = r_upper[col_j + j];
        for &r_kj in &r_upper[col_j..col_j + j] {
            diag -= r_kj * r_kj;
        }
        if !diag.is_finite() || diag <= 0.0 {
            return Err("matrix is not positive definite");
        }
        r_upper[col_j + j] = diag.sqrt();
    }
    Ok(r_upper)
}

pub(crate) fn solve_factorized_uplo_u_2ch(
    n: usize,
    r_upper: &[f64],
    cross_corr: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, &'static str> {
    if cross_corr.len() != n
        || !cross_corr
            .iter()
            .all(|c| c[0].is_finite() && c[1].is_finite())
    {
        return Err("non-finite solver input");
    }

    // Forward substitution: R^T Y = C
    let mut y = vec![[0.0_f64; 2]; n];
    for i in 0..n {
        let col_i = uplo_u_index(0, i);
        let mut s0 = cross_corr[i][0];
        let mut s1 = cross_corr[i][1];
        for (&r_ki, yk) in r_upper[col_i..col_i + i].iter().zip(&y[..i]) {
            s0 -= r_ki * yk[0];
            s1 -= r_ki * yk[1];
        }
        let inv_diag = 1.0 / r_upper[col_i + i];
        y[i][0] = s0 * inv_diag;
        y[i][1] = s1 * inv_diag;
    }

    // Back substitution: R W = Y (column-sweep updates)
    let mut w = y;
    for k in (0..n).rev() {
        let col_k = uplo_u_index(0, k);
        let inv_diag = 1.0 / r_upper[col_k + k];
        let wk0 = w[k][0] * inv_diag;
        let wk1 = w[k][1] * inv_diag;
        w[k] = [wk0, wk1];
        for (wi, &r_ik) in w[..k].iter_mut().zip(&r_upper[col_k..col_k + k]) {
            wi[0] -= r_ik * wk0;
            wi[1] -= r_ik * wk1;
        }
    }

    if !w.iter().all(|r| r[0].is_finite() && r[1].is_finite()) {
        return Err("non-finite solver output");
    }
    Ok(w)
}

/// Solves the regularized symmetric positive-definite linear system
/// $(\mathbf{G} + \lambda \mathbf{I}) \mathbf{W} = \mathbf{C}$ from a `uplo='U'` packed Gram matrix
/// via Cholesky factorization (`ref[train.pipeline.stage-b-harvest]`, `ref[train.resume.kronecker-reassembly]`).
pub fn solve_uplo_u_cholesky_2ch(
    packed_gram: &[f64],
    cross_corr: &[[f64; 2]],
    ridge_lambda: f64,
) -> Result<Vec<[f64; 2]>, &'static str> {
    let n = cross_corr.len();
    let r_upper = factorize_uplo_u_cholesky(n, packed_gram, ridge_lambda)?;
    solve_factorized_uplo_u_2ch(n, &r_upper, cross_corr)
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

/// Cyclic Jacobi symmetric eigensolver for small $L \times L$ positive-semidefinite matrices.
fn jacobi_top_eigenvectors_desc(sym_mat: &[Vec<f64>], max_modes: usize) -> Vec<(f64, Vec<f64>)> {
    let n = sym_mat.len();
    if n == 0 || max_modes == 0 {
        return Vec::new();
    }
    let mut a = sym_mat.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _sweep in 0..24 {
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                max_off = max_off.max(apq.abs());
                if apq.abs() <= 1.0e-15 {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * arp + c * arq;
                        a[q][r] = a[r][q];
                    }
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
        if max_off <= 1.0e-13 {
            break;
        }
    }

    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|col| {
            let eval = a[col][col];
            let evec: Vec<f64> = (0..n).map(|row| v[row][col]).collect();
            (eval, evec)
        })
        .collect();
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    pairs.truncate(max_modes.min(n));
    pairs
}

/// Extracts up to `4` orthonormal right singular vectors $\mathbf{m}_k \in \mathbb{R}^{512}$ and
/// left projection coordinates `coords[l][k]` from `u_mat` (`L × 512`).
pub(crate) fn extract_rank4_svd_modes(u_mat: &[Vec<f64>]) -> (Vec<[f64; 512]>, Vec<[f64; 4]>) {
    let l_cnt = u_mat.len();
    let mut coords = vec![[0.0_f64; 4]; l_cnt];
    if l_cnt == 0 {
        return (Vec::new(), coords);
    }
    let mut cov = vec![vec![0.0_f64; l_cnt]; l_cnt];
    for i in 0..l_cnt {
        for j in i..l_cnt {
            let mut dot = 0.0_f64;
            for (&a, &b) in u_mat[i].iter().zip(&u_mat[j]) {
                dot += a * b;
            }
            cov[i][j] = dot;
            cov[j][i] = dot;
        }
    }

    let eig = jacobi_top_eigenvectors_desc(&cov, 4);
    let mut modes = Vec::with_capacity(4);
    for (k, (eval, left_vec)) in eig.into_iter().enumerate() {
        if eval <= 1.0e-16 {
            continue;
        }
        let mut right = [0.0_f64; 512];
        for (row_idx, &lv) in left_vec.iter().enumerate() {
            if lv.abs() <= 1.0e-16 {
                continue;
            }
            for (d, r_val) in right.iter_mut().enumerate() {
                *r_val += lv * u_mat[row_idx][d];
            }
        }
        let norm = right.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm <= 1.0e-12 {
            continue;
        }
        let unit_inv = 1.0 / norm;
        for v in &mut right {
            *v *= unit_inv;
        }
        let mut max_coord = 0.0_f64;
        for row in u_mat {
            let dot: f64 = row.iter().zip(&right).map(|(&a, &b)| a * b).sum();
            max_coord = max_coord.max(dot.abs());
        }
        let mode_scale = (max_coord / 4.0).max(1.0);
        for v in &mut right {
            *v *= mode_scale;
        }
        let right_norm_sq: f64 = right.iter().map(|&v| v * v).sum();
        for l in 0..l_cnt {
            let dot: f64 = u_mat[l].iter().zip(&right).map(|(&a, &b)| a * b).sum();
            coords[l][k] = dot / right_norm_sq.max(1.0e-18);
        }
        modes.push(right);
    }
    (modes, coords)
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
