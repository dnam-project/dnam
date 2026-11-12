#![forbid(unsafe_code)]
//! Post-`LUFS`-Trim Multi-Tier Nested SPD Normal Equations & Selective `> 1.5 kHz` `MM-Cholesky`
//! Polish (`solver::bezout_schur`, `r[train.pipeline.stage-c-schur]`).

use dnam_engine::schema::DnamError;

use super::bezout_gram::{
    BezoutHarvestStats, eval_4tier_besr_quad_1ch, eval_4tier_post_trim_besr_lin,
};
use super::stage_b_c::{StageBHarvest, eval_all_4_prefix_residuals, solve_cholesky_robust};
use super::{
    factorize_uplo_u_cholesky, solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len,
};

#[inline]
fn quantize_effective_1ch(w_eff: &[f64], role_b_mult: &[[f64; 2]; 256], ch: usize) -> [f64; 256] {
    let mut out = [0.0_f64; 256];
    for idx in 0..256 {
        let m = role_b_mult[idx][ch];
        let inv_m = if m.abs() > 1.0e-6 { 1.0 / m } else { 1.0 };
        let stored = ((w_eff[idx] * inv_m) as f32).clamp(-16.0, 16.0);
        out[idx] = f64::from(stored) * m;
    }
    out
}

#[inline]
fn head_l1_norm(w_1ch: &[f64; 256]) -> f64 {
    let mut s = 0.0_f64;
    for &v in w_1ch {
        s += v.abs();
    }
    s
}

fn solve_1ch_uplo_u_robust(
    dim: usize,
    packed_gram: &[f64],
    rhs: &[f64],
    base_ridge: f64,
) -> Result<Vec<f64>, DnamError> {
    let mut rhs_2ch = vec![[0.0_f64; 2]; dim];
    for (dst, &r) in rhs_2ch.iter_mut().zip(rhs.iter().take(dim)) {
        dst[0] = r;
    }
    let mut ridge = base_ridge.max(1.0e-12);
    for _ in 0..6 {
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(packed_gram, &rhs_2ch, ridge) {
            return Ok(sol.into_iter().map(|v| v[0]).collect());
        }
        ridge *= 10.0;
    }
    let sol = solve_uplo_u_cholesky_2ch(packed_gram, &rhs_2ch, ridge.max(1.0e-3))
        .map_err(|e| DnamError::SchemaValidationError(format!("bESR Cholesky failed: {e}")))?;
    Ok(sol.into_iter().map(|v| v[0]).collect())
}

fn solve_factorized_1ch(dim: usize, r_up: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    for j in 0..dim {
        let col = uplo_u_index(0, j);
        let mut s = rhs[j];
        for i in 0..j {
            s -= r_up[col + i] * y[i];
        }
        y[j] = s / r_up[col + j];
    }
    let mut x = vec![0.0_f64; dim];
    for i in (0..dim).rev() {
        let mut s = y[i];
        for j in (i + 1)..dim {
            s -= r_up[uplo_u_index(i, j)] * x[j];
        }
        x[i] = s / r_up[uplo_u_index(i, i)];
    }
    x
}

/// Result of the 3-band Bézout Multi-Tier Nested SPD Normal Equations (`r[train.pipeline.stage-c-schur]`).
#[derive(Debug, Clone)]
pub struct MultitierBesrSolution {
    pub effective_w0_256: Vec<[f64; 2]>,
    pub max_homotopy_eta: f64,
    pub prefix_besr_db: [f32; 4],
}

/// Solves the 3-band Bézout Multi-Tier Nested SPD Normal Equations across `[Lite, Full, Pro, Ultra]`
/// (`r[train.pipeline.stage-c-schur]`).
pub fn solve_multitier_besr_4tier(
    stats: &BezoutHarvestStats,
    role_b_multipliers: &[[f64; 2]; 256],
    eval_stereo: bool,
) -> Result<MultitierBesrSolution, DnamError> {
    let n_pack = uplo_u_packed_len(256);
    let n_ch = if eval_stereo { 2 } else { 1 };
    let is_post_monotone =
        |b: &[f64; 4]| b[3] <= b[2] + 1.0e-13 && b[2] <= b[1] + 1.0e-13 && b[1] <= b[0] + 1.0e-13;

    let g_bb = &stats.tier_lufs_gram[3];
    let mut trace_bb = 0.0_f64;
    for d in 0..256 {
        trace_bb += g_bb[uplo_u_index(d, d)];
    }
    let mean_bb_diag = (trace_bb / 256.0).max(1.0e-9);
    let base_r = (1.0e-10 * mean_bb_diag).max(1.0e-13);

    // Scale regularization by the unweighted feature-bus Gram diagonal `mean_bb_diag` (not the
    // stopband-inflated trace of `tier_gram_lr`) so steep stopband weights never damp passbands!
    let mut reg_ridge = [1.0e-10_f64; 2];
    let mut cond_ridge = [1.0e-9_f64; 2];
    let mut y_tot = [1.0_f64; 2];
    let mut base_lin = [1.0_f64; 4];
    let mut opt_tier_lin = [0.0_f64; 4];

    for ch in 0..n_ch {
        let mut tr_w = 0.0_f64;
        for d in 0..256 {
            tr_w += stats.tier_gram_lr[3][ch][uplo_u_index(d, d)];
        }
        let mean_diag = (tr_w / 256.0).min(mean_bb_diag * 250.0).max(1.0e-12);
        reg_ridge[ch] = (5.0e-6 * mean_diag).max(1.0e-10);
        cond_ridge[ch] = (2.0e-5 * mean_diag).max(1.0e-9);
        y_tot[ch] = (stats.sampled_target_band_energy_lr[ch][0]
            + stats.sampled_target_band_energy_lr[ch][1]
            + stats.sampled_target_band_energy_lr[ch][2])
            .max(1.0e-15);
        for t in 0..4 {
            let b0 = (stats.tier_base_res_lr[t][ch] / y_tot[ch]).max(1.0);
            base_lin[t] = if ch == 0 { b0 } else { base_lin[t].max(b0) };
            let dim = (t + 1) * 64;
            let rhs_t: Vec<f64> = (0..dim).map(|d| stats.tier_cross_256[t][d][ch]).collect();
            let sol_t =
                solve_1ch_uplo_u_robust(dim, &stats.tier_gram_lr[t][ch], &rhs_t, reg_ridge[ch])?;
            let mut pad = [0.0_f64; 256];
            pad[..dim].copy_from_slice(&sol_t);
            let q_pad = quantize_effective_1ch(&pad, role_b_multipliers, ch);
            let ol = eval_4tier_besr_quad_1ch(stats, ch, &q_pad)[t] / y_tot[ch];
            opt_tier_lin[t] = if ch == 0 { ol } else { opt_tier_lin[t].max(ol) };
        }
    }

    let pack_lr = |ch_arr: &[[f64; 256]; 2]| -> [[f64; 2]; 256] {
        let mut out = [[0.0_f64; 2]; 256];
        for d in 0..256 {
            out[d][0] = ch_arr[0][d];
            out[d][1] = if eval_stereo {
                ch_arr[1][d]
            } else {
                ch_arr[0][d]
            };
        }
        out
    };

    // 0. Silent / muted target (`Y_tot <= 1e-12`): cancel backbone via sequential 64-D Schur
    // so all 4 tiers (`Lite..Ultra`) output `-70.000 LUFS` silence (`0.000 LU` spread).
    let raw_y_max = (0..n_ch)
        .map(|ch| stats.sampled_target_band_energy_lr[ch].iter().sum::<f64>())
        .fold(0.0_f64, f64::max);
    if raw_y_max <= 1.0e-12 {
        let silent_ridge = (1.0e-6 * mean_bb_diag).max(1.0e-11);
        let mut ch_ws = [[0.0_f64; 256]; 2];
        for ch in 0..n_ch {
            let mut ws = [0.0_f64; 256];
            for t in 0..4 {
                let start = t * 64;
                let g_t = &stats.tier_lufs_gram[t];
                let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                for j in 0..64 {
                    let src_col = uplo_u_index(0, start + j) + start;
                    let dst_col = uplo_u_index(0, j);
                    sub_g[dst_col..dst_col + j + 1].copy_from_slice(&g_t[src_col..src_col + j + 1]);
                }
                if let Ok(r_up) = factorize_uplo_u_cholesky(64, &sub_g, silent_ridge)
                    .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, silent_ridge * 100.0))
                {
                    let mut sub_rhs = [0.0_f64; 64];
                    for i in 0..64 {
                        let row = start + i;
                        let col_base = uplo_u_index(0, row);
                        let mut c = stats.tier_bb_res_cross_256[t][row][ch];
                        for (prev, &g_ij) in g_t[col_base..col_base + start].iter().enumerate() {
                            c -= g_ij * ws[prev];
                        }
                        sub_rhs[i] = c;
                    }
                    let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                    ws[start..start + 64].copy_from_slice(&blk_w);
                    let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                    ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
                }
            }
            ch_ws[ch] = ws;
        }
        return Ok(MultitierBesrSolution {
            effective_w0_256: pack_lr(&ch_ws).to_vec(),
            max_homotopy_eta: 0.0,
            prefix_besr_db: [-120.0; 4],
        });
    }
    let eval_bb_mse_ultra = |ch: usize, q: &[f64; 256]| -> f64 {
        let mut quad = 0.0_f64;
        let mut lin = 0.0_f64;
        for (j, &qj) in q.iter().enumerate() {
            lin += qj * stats.tier_bb_res_cross_256[3][j][ch];
            if qj == 0.0 {
                continue;
            }
            let col = uplo_u_index(0, j);
            let mut off = 0.0_f64;
            for i in 0..j {
                off += g_bb[col + i] * q[i];
            }
            quad += qj * (2.0 * off + g_bb[col + j] * qj);
        }
        (stats.tier_bb_res_energy_lr[3][ch] - 2.0 * lin + quad).max(0.0)
    };
    let mut q_in_ch = [[0.0_f64; 256]; 2];
    let mut inclass_ok = true;
    for ch in 0..n_ch {
        let rhs_u: [f64; 256] = std::array::from_fn(|d| stats.tier_bb_res_cross_256[3][d][ch]);
        let sol = solve_1ch_uplo_u_robust(256, g_bb, &rhs_u, base_r)?;
        q_in_ch[ch] = quantize_effective_1ch(&sol, role_b_multipliers, ch);
        let bb_u = eval_bb_mse_ultra(ch, &q_in_ch[ch]);
        if bb_u > 1.0e-5 * y_tot[ch] || head_l1_norm(&q_in_ch[ch]) > 48.0 {
            inclass_ok = false;
        }
    }
    let w_in_lr = pack_lr(&q_in_ch);
    let (b_in, _) = eval_4tier_post_trim_besr_lin(stats, &w_in_lr, eval_stereo);
    if inclass_ok
        && opt_tier_lin[0] > 1.0e-3
        && is_post_monotone(&b_in)
        && b_in[0] <= 0.75
        && b_in[0] <= opt_tier_lin[0] * 12.0 + 0.05
    {
        let db = b_in.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
        return Ok(MultitierBesrSolution {
            effective_w0_256: w_in_lr.to_vec(),
            max_homotopy_eta: 0.0,
            prefix_besr_db: db,
        });
    }

    let max_l1 = 42.0_f64;
    let no_prefix_blowup = |b: &[f64; 4]| {
        (0..3).all(|t| b[t] <= base_lin[t] * 1.01 + 1.0e-12 && b[t] <= opt_tier_lin[t] * 1.6 + 0.01)
    };

    // 1b. Rank-deficient / band-limited in-class targets (e.g. 2-tone sinusoids with distinct
    // per-tier backbones): sequential 64-D Gauss-Markov broadband Schur cascade, gated on
    // monotone post-trim 3-band bESR, zero prefix blowup, and non-degraded Ultra bESR.
    if inclass_ok {
        let seq_bb_ridge = (1.0e-6 * mean_bb_diag).max(1.0e-11);
        let mut ch_ws = [[0.0_f64; 256]; 2];
        let mut seq_bb_ok = true;
        for ch in 0..n_ch {
            let mut ws = [0.0_f64; 256];
            for t in 0..4 {
                let start = t * 64;
                let g_t = &stats.tier_lufs_gram[t];
                let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                for j in 0..64 {
                    let row = start + j;
                    let src_col = uplo_u_index(0, row) + start;
                    let dst_col = uplo_u_index(0, j);
                    sub_g[dst_col..dst_col + j + 1].copy_from_slice(&g_t[src_col..src_col + j + 1]);
                    let is_late_tank = matches!(row, 116..=125 | 184..=189 | 246..=253);
                    if is_late_tank {
                        sub_g[dst_col + j] += 1.0e5 * seq_bb_ridge;
                    }
                }
                let Ok(r_up) = factorize_uplo_u_cholesky(64, &sub_g, seq_bb_ridge)
                    .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, seq_bb_ridge * 100.0))
                else {
                    seq_bb_ok = false;
                    break;
                };
                let mut sub_rhs = [0.0_f64; 64];
                for i in 0..64 {
                    let row = start + i;
                    let col_base = uplo_u_index(0, row);
                    let mut c = stats.tier_bb_res_cross_256[t][row][ch];
                    for (prev, &g_ij) in g_t[col_base..col_base + start].iter().enumerate() {
                        c -= g_ij * ws[prev];
                    }
                    sub_rhs[i] = c;
                }
                let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                ws[start..start + 64].copy_from_slice(&blk_w);
                let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
            }
            if !seq_bb_ok
                || head_l1_norm(&ws) > max_l1
                || eval_bb_mse_ultra(ch, &ws) > 1.0e-5 * y_tot[ch]
            {
                seq_bb_ok = false;
                break;
            }
            ch_ws[ch] = ws;
        }
        if seq_bb_ok {
            let w_seq_bb = pack_lr(&ch_ws);
            let (b_si, _) = eval_4tier_post_trim_besr_lin(stats, &w_seq_bb, eval_stereo);
            if is_post_monotone(&b_si)
                && no_prefix_blowup(&b_si)
                && b_si[3] <= opt_tier_lin[3] * 2.0 + 1.0e-5
            {
                let db = b_si.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
                return Ok(MultitierBesrSolution {
                    effective_w0_256: w_seq_bb.to_vec(),
                    max_homotopy_eta: 1.0,
                    prefix_besr_db: db,
                });
            }
        }
    }

    // 2. Out-of-class physical targets: exact per-tier sequential 64-D Schur cascade (eta = 1.0)

    let build_seq_lr = |ridge_scale: f64, g_est: [f64; 4]| -> Result<[[f64; 2]; 256], DnamError> {
        let mut ch_ws = [[0.0_f64; 256]; 2];
        for ch in 0..n_ch {
            let blk_ridge = cond_ridge[ch] * ridge_scale;
            let mut ws = [0.0_f64; 256];
            for t in 0..4 {
                let start = t * 64;
                let g_tier = &stats.tier_gram_lr[t][ch];
                let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                for j in 0..64 {
                    let src_col = uplo_u_index(0, start + j) + start;
                    let dst_col = uplo_u_index(0, j);
                    sub_g[dst_col..dst_col + j + 1]
                        .copy_from_slice(&g_tier[src_col..src_col + j + 1]);
                }
                let r_up = factorize_uplo_u_cholesky(64, &sub_g, blk_ridge)
                    .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, blk_ridge * 100.0))
                    .map_err(|e| DnamError::SchemaValidationError(format!("64-D bESR: {e}")))?;
                let inv_g_m1 = (1.0 / g_est[t].clamp(0.25, 4.0)) - 1.0;
                let mut sub_rhs = [0.0_f64; 64];
                for i in 0..64 {
                    let row = start + i;
                    let col_base = uplo_u_index(0, row);
                    let mut c = stats.tier_cross_256[t][row][ch]
                        + inv_g_m1 * stats.tier_tgt_fb_cross[t][row][ch];
                    for (prev, &g_ij) in g_tier[col_base..col_base + start].iter().enumerate() {
                        c -= g_ij * ws[prev];
                    }
                    sub_rhs[i] = c;
                }
                let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                ws[start..start + 64].copy_from_slice(&blk_w);
                let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
            }
            ch_ws[ch] = ws;
        }
        Ok(pack_lr(&ch_ws))
    };

    let l1_ok = |w_lr: &[[f64; 2]; 256]| -> bool {
        (0..n_ch).all(|ch| {
            let s: f64 = w_lr.iter().map(|v| v[ch].abs()).sum();
            s <= max_l1
        })
    };

    let mut seq_scale = 1.0_f64;
    let mut w_seq = build_seq_lr(seq_scale, [1.0; 4])?;
    if !l1_ok(&w_seq) {
        seq_scale = 1_000.0;
        w_seq = build_seq_lr(seq_scale, [1.0; 4])?;
    }
    let (mut b_seq, mut g_seq) = eval_4tier_post_trim_besr_lin(stats, &w_seq, eval_stereo);
    for _ in 0..2 {
        let w_next = build_seq_lr(seq_scale, g_seq)?;
        if !l1_ok(&w_next) {
            break;
        }
        let (b_next, g_next) = eval_4tier_post_trim_besr_lin(stats, &w_next, eval_stereo);
        if (is_post_monotone(&b_next) && (!is_post_monotone(&b_seq) || b_next[3] <= b_seq[3]))
            || (!is_post_monotone(&b_seq) && b_next[2] - b_next[1] < b_seq[2] - b_seq[1])
        {
            w_seq = w_next;
            b_seq = b_next;
        }
        for t in 0..4 {
            g_seq[t] = (0.5 * g_seq[t] + 0.5 * g_next[t]).clamp(0.25, 4.0);
        }
    }

    let mut best_w = w_seq;
    let mut best_eta = 1.0_f64;
    let mut best_b = b_seq;
    let mut has_valid_best = is_post_monotone(&b_seq) && no_prefix_blowup(&b_seq);
    let mut best_mono_fallback: Option<([[f64; 2]; 256], f64, [f64; 4])> =
        is_post_monotone(&b_seq).then_some((w_seq, 1.0, b_seq));
    let mut g_weighted = vec![0.0_f64; n_pack];
    let mut rhs_weighted = [0.0_f64; 256];

    // Sweep exact Multi-Tier SPD Normal Equations summing exact per-tier Hessians H^(gamma, g) w = b^(gamma, g)
    for (eta, gammas) in [
        (0.0_f64, [0.0_f64, 0.0, 0.0, 1.0]),
        (0.08, [0.15, 0.30, 0.55, 1.0]),
        (0.16, [0.35, 0.55, 0.75, 1.0]),
        (0.28, [0.75, 0.85, 0.95, 1.0]),
        (0.45, [2.00, 1.50, 1.20, 1.0]),
        (0.68, [8.00, 3.50, 1.80, 1.0]),
        (0.88, [36.0, 10.0, 3.00, 1.0]),
        (0.97, [144.0, 28.0, 5.50, 1.0]),
        (0.99, [200.0, 80.0, 32.0, 1.0]),
    ] {
        for ridge_mult in [1.0_f64, 4.0, 16.0, 128.0, 1_024.0] {
            let mut g_kkt = [1.0_f64; 4];
            let mut ridge_exceeded_l1 = false;
            for kkt_pass in 0..3 {
                let g_sq = g_kkt.map(|g| g * g);
                let big_gamma = [
                    gammas[0] * g_sq[0]
                        + gammas[1] * g_sq[1]
                        + gammas[2] * g_sq[2]
                        + gammas[3] * g_sq[3],
                    gammas[1] * g_sq[1] + gammas[2] * g_sq[2] + gammas[3] * g_sq[3],
                    gammas[2] * g_sq[2] + gammas[3] * g_sq[3],
                    gammas[3] * g_sq[3],
                ];
                let mut cand_ch = [[0.0_f64; 256]; 2];
                let mut solve_ok = true;
                for ch in 0..n_ch {
                    let cur_ridge = reg_ridge[ch] * ridge_mult;
                    g_weighted.fill(0.0);
                    rhs_weighted.fill(0.0);
                    for t in 0..4 {
                        let w_t = gammas[t] * g_sq[t];
                        if w_t <= 0.0 {
                            continue;
                        }
                        let dim = (t + 1) * 64;
                        let p_len = uplo_u_packed_len(dim);
                        let g_tier = &stats.tier_gram_lr[t][ch];
                        for idx in 0..p_len {
                            g_weighted[idx] += w_t * g_tier[idx];
                        }
                        let gt = g_kkt[t];
                        let w_k = gammas[t] * gt * (1.0 - gt);
                        for d in 0..dim {
                            rhs_weighted[d] += w_t * stats.tier_cross_256[t][d][ch]
                                + w_k * stats.tier_tgt_fb_cross[t][d][ch];
                        }
                    }
                    for d in 0..256 {
                        g_weighted[uplo_u_index(d, d)] += big_gamma[d / 64] * cur_ridge;
                    }
                    let Ok(cand_raw) = solve_1ch_uplo_u_robust(
                        256,
                        &g_weighted,
                        &rhs_weighted,
                        cur_ridge * 1.0e-3,
                    ) else {
                        solve_ok = false;
                        break;
                    };
                    cand_ch[ch] = quantize_effective_1ch(&cand_raw, role_b_multipliers, ch);
                }
                if !solve_ok {
                    break;
                }
                let cand_lr = pack_lr(&cand_ch);
                if !l1_ok(&cand_lr) {
                    ridge_exceeded_l1 = true;
                    break;
                }
                let (b_cand, g_next) = eval_4tier_post_trim_besr_lin(stats, &cand_lr, eval_stereo);
                if is_post_monotone(&b_cand) {
                    if best_mono_fallback
                        .as_ref()
                        .is_none_or(|(_, _, mb)| b_cand[3] < mb[3])
                    {
                        best_mono_fallback = Some((cand_lr, eta, b_cand));
                    }
                    if no_prefix_blowup(&b_cand) && (!has_valid_best || b_cand[3] <= best_b[3]) {
                        best_w = cand_lr;
                        best_eta = eta;
                        best_b = b_cand;
                        has_valid_best = true;
                    }
                }
                if kkt_pass == 0 && g_next.iter().all(|&g| (g - 1.0).abs() < 1.0e-3) {
                    break;
                }
                for t in 0..4 {
                    g_kkt[t] = (0.5 * g_kkt[t] + 0.5 * g_next[t]).clamp(0.25, 4.0);
                }
            }
            if !ridge_exceeded_l1 {
                break;
            }
        }
        if best_eta < 1.0 {
            break;
        }
    }

    if !is_post_monotone(&best_b)
        && let Some((mw, meta, mb)) = best_mono_fallback
    {
        best_w = mw;
        best_eta = meta;
        best_b = mb;
    }

    let prefix_besr_db = best_b.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
    Ok(MultitierBesrSolution {
        effective_w0_256: best_w.to_vec(),
        max_homotopy_eta: best_eta,
        prefix_besr_db,
    })
}

/// Applies Hann-windowed Theorem 5 Selective `1/12`-Octave (`> 1.5 kHz`, $|\Delta L_b| > 0.25\text{ dB}$)
/// Majorize-Minimize polish on `w_cand`, strictly gated on non-regressing post-trim 3-band `bESR` & 4-tier monotonicity.
pub(super) fn apply_selective_hf_mm_and_ltas_polish(
    harvest: &StageBHarvest,
    packed_gram_256: &[f64],
    base_ridge: f64,
    role_b_multipliers: &[[f64; 2]; 256],
    w_cand: &mut [[f64; 2]],
) -> usize {
    const WIN_N: usize = 256;
    const HF_BINS: [usize; 16] = [
        9, 11, 13, 16, 19, 23, 27, 32, 38, 45, 53, 63, 74, 84, 94, 105,
    ];
    let n_total = harvest.residual_lr.len();
    let use_polish_win = harvest.polish_win_features.len() >= WIN_N;
    if harvest.weighted_target_energy <= 1.0e-12
        || n_total < WIN_N
        || (!use_polish_win && harvest.features.len() < WIN_N)
    {
        return 0;
    }
    let e_init = eval_all_4_prefix_residuals(
        packed_gram_256,
        &harvest.cross_256,
        w_cand,
        harvest.weighted_residual_energy,
    )[3];
    if e_init <= 1.0e-5 * harvest.weighted_target_energy.max(1.0e-12) {
        return 0;
    }

    let win_start = (n_total / 3).min(n_total - WIN_N);
    let mut hann = [0.0_f64; WIN_N];
    let mut pred = [[0.0_f64; 2]; WIN_N];
    for t in 0..WIN_N {
        hann[t] = 0.5 * (1.0 - (std::f64::consts::TAU * (t as f64) / (WIN_N as f64)).cos());
        let fb = if use_polish_win {
            &harvest.polish_win_features[t]
        } else {
            &harvest.features[win_start + t]
        };
        let (mut sl, mut sr) = (0.0_f64, 0.0_f64);
        for d in 0..256 {
            let fd = f64::from(fb[d]);
            sl += fd * w_cand[d][0];
            sr += fd * w_cand[d][1];
        }
        pred[t] = [sl, sr];
    }

    let mut relaxed_count = 0usize;
    let mut delta_time = [[0.0_f64; 2]; WIN_N];
    for &k_bin in &HF_BINS {
        let (s_step, c_step) = (std::f64::consts::TAU * (k_bin as f64) / (WIN_N as f64)).sin_cos();
        let (mut tgt_re_l, mut tgt_im_l, mut prd_re_l, mut prd_im_l) = (0.0, 0.0, 0.0, 0.0);
        let (mut tgt_re_r, mut tgt_im_r, mut prd_re_r, mut prd_im_r) = (0.0, 0.0, 0.0, 0.0);
        let (mut c, mut s) = (1.0_f64, 0.0_f64);
        for t in 0..WIN_N {
            let w_h = hann[t];
            let ([tl, tr], [pl, pr]) = (harvest.residual_lr[win_start + t], pred[t]);
            tgt_re_l += (w_h * tl) * c;
            tgt_im_l -= (w_h * tl) * s;
            prd_re_l += (w_h * pl) * c;
            prd_im_l -= (w_h * pl) * s;
            tgt_re_r += (w_h * tr) * c;
            tgt_im_r -= (w_h * tr) * s;
            prd_re_r += (w_h * pr) * c;
            prd_im_r -= (w_h * pr) * s;
            (c, s) = (c * c_step - s * s_step, s * c_step + c * s_step);
        }
        let e_tgt =
            (tgt_re_l * tgt_re_l + tgt_im_l * tgt_im_l + tgt_re_r * tgt_re_r + tgt_im_r * tgt_im_r)
                .max(1.0e-15);
        let e_prd =
            (prd_re_l * prd_re_l + prd_im_l * prd_im_l + prd_re_r * prd_re_r + prd_im_r * prd_im_r)
                .max(1.0e-15);
        if e_tgt < 1.0e-6 || (10.0 * (e_prd / e_tgt).log10()).abs() <= 0.25 {
            continue;
        }
        relaxed_count += 1;
        let norm = ((e_tgt / e_prd).sqrt().clamp(0.5, 2.0) - 1.0) * (4.0 / (WIN_N as f64));
        let (dl_re, dl_im, dr_re, dr_im) = (
            norm * prd_re_l,
            norm * prd_im_l,
            norm * prd_re_r,
            norm * prd_im_r,
        );
        let (mut c, mut s) = (1.0_f64, 0.0_f64);
        for (t, dt_lr) in delta_time.iter_mut().enumerate() {
            let w_h = 0.12 * hann[t];
            dt_lr[0] += w_h * (dl_re * c - dl_im * s);
            dt_lr[1] += w_h * (dr_re * c - dr_im * s);
            (c, s) = (c * c_step - s * s_step, s * c_step + c * s_step);
        }
    }
    if relaxed_count == 0 {
        return 0;
    }

    let mut delta_c = vec![[0.0_f64; 2]; 256];
    let scale = (n_total as f64) / (WIN_N as f64);
    for (t, &[dl, dr]) in delta_time.iter().enumerate() {
        let fb = if use_polish_win {
            &harvest.polish_win_features[t]
        } else {
            &harvest.features[win_start + t]
        };
        let w = harvest.frame_weights[win_start + t] * scale;
        for d in 0..256 {
            let wfd = w * f64::from(fb[d]);
            delta_c[d][0] += wfd * dl;
            delta_c[d][1] += wfd * dr;
        }
    }

    let mut accepted = false;
    if let Ok(dw) = solve_cholesky_robust(packed_gram_256, &delta_c, base_ridge * 10.0) {
        if let Some(ref stats) = harvest.besr_stats {
            let n_ch = if harvest.eval_stereo { 2 } else { 1 };
            let mut cur_lr = [[0.0_f64; 2]; 256];
            cur_lr.copy_from_slice(&w_cand[..256]);
            let (b_before, _) = eval_4tier_post_trim_besr_lin(stats, &cur_lr, harvest.eval_stereo);
            for step in [0.5_f64, 0.15, 0.04] {
                let mut raw_ch = [[0.0_f64; 256]; 2];
                let mut q_ch = [[0.0_f64; 256]; 2];
                for ch in 0..n_ch {
                    for d in 0..256 {
                        raw_ch[ch][d] = cur_lr[d][ch] + step * dw[d][ch];
                    }
                    q_ch[ch] = quantize_effective_1ch(&raw_ch[ch], role_b_multipliers, ch);
                }
                if !(0..n_ch).all(|ch| head_l1_norm(&q_ch[ch]) <= 42.0) {
                    continue;
                }
                let mut cand_lr = [[0.0_f64; 2]; 256];
                for d in 0..256 {
                    cand_lr[d][0] = q_ch[0][d];
                    cand_lr[d][1] = q_ch[usize::from(harvest.eval_stereo)][d];
                }
                let (ba, _) = eval_4tier_post_trim_besr_lin(stats, &cand_lr, harvest.eval_stereo);
                let lower_ok = (0..3).all(|t| {
                    ba[t] <= b_before[t] * 1.02 + 1.0e-5 && (b_before[t] >= 1.0 || ba[t] < 1.0)
                });
                if ba[3] <= b_before[3]
                    && ba[3] <= ba[2]
                    && ba[2] <= ba[1]
                    && ba[1] <= ba[0]
                    && lower_ok
                {
                    w_cand[..256].copy_from_slice(&cand_lr);
                    accepted = true;
                    break;
                }
            }
        } else {
            let eb = eval_all_4_prefix_residuals(
                packed_gram_256,
                &harvest.cross_256,
                w_cand,
                harvest.weighted_residual_energy,
            );
            for step in [0.5_f64, 0.15, 0.04] {
                let mut cand = w_cand.to_vec();
                for (w, d) in cand.iter_mut().zip(&dw) {
                    w[0] += step * d[0];
                    w[1] += step * d[1];
                }
                if !(0..2).all(|ch| cand.iter().map(|v| v[ch].abs()).sum::<f64>() <= 42.0) {
                    continue;
                }
                let ea = eval_all_4_prefix_residuals(
                    packed_gram_256,
                    &harvest.cross_256,
                    &cand,
                    harvest.weighted_residual_energy,
                );
                let mono = ea[3] <= ea[2] + 1.0e-12
                    && ea[2] <= ea[1] + 1.0e-12
                    && ea[1] <= ea[0] + 1.0e-12;
                let lower_ok = (0..3).all(|t| ea[t] <= eb[t] * 1.02 + 1.0e-9);
                if ea[3] <= eb[3] && mono && lower_ok {
                    w_cand.copy_from_slice(&cand);
                    accepted = true;
                    break;
                }
            }
        }
    }
    if accepted { relaxed_count } else { 0 }
}
