#![forbid(unsafe_code)]
//! Dual-Rate Steiglitz–McBride Rational Modal Pole Identification, Deconvolved Stage III
//! Stereo FIR + Nested Kautz Residue Fitting, and Channel-Aligned Stage IV Reflection Extraction
//! (`solver::stage_a_id`).
//!
//! Canonical Reference: `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-a-poles]`).

use std::f64::consts::TAU;

use dnam_engine::operators::op_m::algebraic_unit_rotor_op_m;
use dnam_engine::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER;

use super::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

/// Deconvolved Stage III stereo `64`-tap Early FIR, nested `32`-pole Kautz stereo residues,
/// and simulated `(s3_l, s3_r)` backbone signal feeding Stage IV.
pub(crate) struct Stage3DeconvolvedFit {
    pub fir_l: [f32; 64],
    pub fir_r: [f32; 64],
    pub kautz_c_l: [(f32, f32); 32],
    pub kautz_c_r: [(f32, f32); 32],
    pub s3_lr: Vec<[f32; 2]>,
}

/// Finds all `P` complex roots of monic polynomial $z^P + a_1 z^{P-1} + \dots + a_P$ via
/// Durand–Kerner iterations in `f64` and reflects unstable roots inside `max_radius`.
pub(crate) fn find_and_stabilize_monic_roots(
    a: &[f64],
    max_radius: f64,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let p = a.len();
    let mut roots: Vec<(f64, f64)> = (0..p)
        .map(|k| {
            let ang = TAU * (k as f64 + 0.37) / (p as f64);
            let rad = 0.86 - 0.04 * ((k % 3) as f64);
            (rad * ang.cos(), rad * ang.sin())
        })
        .collect();

    for _ in 0..28 {
        for i in 0..p {
            let (zr, zi) = roots[i];
            let (mut pr, mut pi) = (1.0_f64, 0.0_f64);
            for &ak in a {
                let nr = pr * zr - pi * zi + ak;
                let ni = pr * zi + pi * zr;
                pr = nr;
                pi = ni;
            }
            let (mut dr, mut di) = (1.0_f64, 0.0_f64);
            for (j, &(or, oi)) in roots.iter().enumerate() {
                if i != j {
                    let (vr, vi) = (zr - or, zi - oi);
                    let nr = dr * vr - di * vi;
                    let ni = dr * vi + di * vr;
                    dr = nr;
                    di = ni;
                }
            }
            let den = (dr * dr + di * di).max(1.0e-24);
            let step_r = (pr * dr + pi * di) / den;
            let step_i = (pi * dr - pr * di) / den;
            roots[i] = (
                (zr - step_r).clamp(-2.0, 2.0),
                (zi - step_i).clamp(-2.0, 2.0),
            );
        }
    }

    for (zr, zi) in &mut roots {
        let mag = (*zr * *zr + *zi * *zi).sqrt();
        if mag > max_radius && mag > 1.0e-12 {
            let s = max_radius / mag;
            *zr *= s;
            *zi *= s;
        }
    }

    let mut poly_re = vec![0.0_f64; p + 1];
    let mut poly_im = vec![0.0_f64; p + 1];
    poly_re[0] = 1.0;
    for (deg, &(rr, ri)) in roots.iter().enumerate() {
        for k in (1..=deg + 1).rev() {
            let (prev_r, prev_i) = (poly_re[k - 1], poly_im[k - 1]);
            poly_re[k] -= prev_r * rr - prev_i * ri;
            poly_im[k] -= prev_r * ri + prev_i * rr;
        }
    }
    let a_stab: Vec<f64> = poly_re[1..=p]
        .iter()
        .map(|&v| v.clamp(-64.0, 64.0))
        .collect();
    (roots, a_stab)
}

fn run_steiglitz_mcbride_arma(
    x: &[f64],
    y: &[f64],
    p_ar: usize,
    q_ma: usize,
    max_r: f64,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let n = x.len().min(y.len());
    let dim = p_ar + q_ma;
    let mut a_cur = vec![0.0_f64; p_ar];
    let mut roots_final = Vec::new();
    let mut lin_arx_res = vec![0.0_f64; n];
    if n <= p_ar + 8 {
        return (roots_final, lin_arx_res);
    }

    for sm_iter in 0..3 {
        let mut xf = vec![0.0_f64; n];
        let mut yf = vec![0.0_f64; n];
        if sm_iter == 0 {
            xf.copy_from_slice(&x[..n]);
            yf.copy_from_slice(&y[..n]);
        } else {
            for i in 0..n {
                let (mut sx, mut sy) = (x[i], y[i]);
                for k in 1..=i.min(p_ar) {
                    sx -= a_cur[k - 1] * xf[i - k];
                    sy -= a_cur[k - 1] * yf[i - k];
                }
                xf[i] = sx.clamp(-16.0, 16.0);
                yf[i] = sy.clamp(-16.0, 16.0);
            }
        }

        let mut g = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c = vec![[0.0_f64; 2]; dim];
        let mut phi = vec![0.0_f64; dim];
        let i_start = if n <= 128 { 0 } else { p_ar };
        for i in i_start..n {
            for k in 0..p_ar {
                phi[k] = if i > k { -yf[i - 1 - k] } else { 0.0 };
            }
            for m in 0..q_ma {
                phi[p_ar + m] = if i >= m { xf[i - m] } else { 0.0 };
            }
            let target = yf[i];
            for col in 0..dim {
                let pc = phi[col];
                let base = uplo_u_index(0, col);
                for row in 0..=col {
                    g[base + row] += phi[row] * pc;
                }
                c[col][0] += pc * target;
            }
        }

        let mut d_inv = vec![0.0_f64; dim];
        let mut g_eq = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c_eq = vec![[0.0_f64; 2]; dim];
        for col in 0..dim {
            let diag = g[uplo_u_index(col, col)];
            d_inv[col] = if diag > 1.0e-20 {
                1.0 / diag.sqrt()
            } else {
                0.0
            };
        }
        for col in 0..dim {
            let base = uplo_u_index(0, col);
            let dc = d_inv[col];
            for row in 0..=col {
                g_eq[base + row] = g[base + row] * d_inv[row] * dc;
            }
            if dc == 0.0 {
                g_eq[base + col] = 1.0;
            }
            c_eq[col][0] = c[col][0] * dc;
        }
        let Ok(sol_eq) = solve_uplo_u_cholesky_2ch(&g_eq, &c_eq, 1.0e-5) else {
            break;
        };
        let sol: Vec<[f64; 2]> = (0..dim).map(|k| [sol_eq[k][0] * d_inv[k], 0.0]).collect();

        if sm_iter == 0 {
            for i in p_ar.max(q_ma)..n {
                let mut pred = 0.0_f64;
                for k in 0..p_ar {
                    pred -= sol[k][0] * y[i - 1 - k];
                }
                for m in 0..q_ma {
                    pred += sol[p_ar + m][0] * x[i - m];
                }
                lin_arx_res[i] = y[i] - pred;
            }
        }

        let a_raw: Vec<f64> = (0..p_ar).map(|k| sol[k][0]).collect();
        let (roots, a_stab) = find_and_stabilize_monic_roots(&a_raw, max_r);
        a_cur = a_stab;
        roots_final = roots;
    }

    (roots_final, lin_arx_res)
}

/// Runs dual-rate (`48 kHz` broadband + `6 kHz` `8x`-decimated LF) prefiltered Steiglitz–McBride
/// rational pole identification AND measures nonlinear distortion variance (`r[train.pipeline.stage-a-poles]`).
pub(crate) fn identify_steiglitz_mcbride_and_linearity(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    hf_growth: f32,
) -> (f32, Vec<(f32, f32)>) {
    const P_AR: usize = 18;
    const Q_MA: usize = 17;
    let n = dry_lr.len().min(wet_lr.len()).min(12_000);
    if n < 96 {
        return ((hf_growth - 1.0).clamp(0.0, 1.0), Vec::new());
    }

    let mut x = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];
    let mut wet_e = 0.0_f64;
    let nl_end = n.min(4_096);
    let mut mean_abs_x = 0.0_f64;
    for i in 0..n {
        let xi = 0.5 * (f64::from(dry_lr[i][0]) + f64::from(dry_lr[i][1]));
        let (wl, wr) = (f64::from(wet_lr[i][0]), f64::from(wet_lr[i][1]));
        // Asymmetric stereo blend (1.0*L + 0.37*R) never cancels same-sign or opposite-sign modes.
        let yi = if (wl - wr).abs() <= 1.0e-7 {
            wl
        } else {
            (wl + 0.37 * wr) * (1.0 / 1.37)
        };
        x[i] = if xi.is_finite() { xi } else { 0.0 };
        y[i] = if yi.is_finite() { yi } else { 0.0 };
        wet_e += y[i] * y[i];
        if i < nl_end {
            mean_abs_x += x[i].abs();
        }
    }
    mean_abs_x /= nl_end as f64;
    if wet_e <= 1.0e-12 {
        return (0.0, Vec::new());
    }

    // Deconvolve early 104-tap causal impulse response and gate before discrete room reflections
    // (> 24 samples post-onset) so 48 kHz SM identifies true modal poles rather than comb echoes.
    let bb_start = if n >= 11_000 { (n * 3) / 4 - 256 } else { 0 };
    let h_ir = solve_causal_fir_1ch(&x[bb_start..], &y[bb_start..], 0, 104);
    let k_peak = (0..24.min(h_ir.len()))
        .max_by(|&a, &b| h_ir[a].abs().total_cmp(&h_ir[b].abs()))
        .unwrap_or(0);
    let mut gate_end = (k_peak + 78).min(h_ir.len());
    for k in (k_peak + 24)..h_ir.len().saturating_sub(2) {
        let prev_rms = (h_ir[k - 12..k - 2].iter().map(|v| v * v).sum::<f64>() * 0.1)
            .sqrt()
            .max(1.0e-9);
        if h_ir[k].abs() > 3.2 * prev_rms {
            gate_end = k - 2;
            break;
        }
    }
    let mut delta_hp = vec![0.0_f64; gate_end];
    let mut h_hp = vec![0.0_f64; gate_end];
    if gate_end >= 2 {
        delta_hp[0] = 1.0;
        delta_hp[1] = -0.965;
        h_hp[0] = h_ir[0];
        for k in 1..gate_end {
            h_hp[k] = h_ir[k] - 0.965 * h_ir[k - 1];
        }
    }
    let (roots_48k, _) =
        run_steiglitz_mcbride_arma(&delta_hp, &h_hp, 16, (k_peak + 8).min(24), 0.992);
    let (_, lin_arx_res) =
        run_steiglitz_mcbride_arma(&x[..nl_end], &y[..nl_end], P_AR, Q_MA, 0.992);

    const NL_LAGS: [usize; 4] = [0, 3, 7, 13];
    const DIM_AUG: usize = 16;
    let mut g_aug = vec![0.0_f64; uplo_u_packed_len(DIM_AUG)];
    let mut c_aug = vec![[0.0_f64; 2]; DIM_AUG];
    let mut psi = [0.0_f64; DIM_AUG];
    let mut wet_nl_e = 0.0_f64;
    for i in P_AR..nl_end {
        wet_nl_e += y[i] * y[i];
        for (idx_l, &lag) in NL_LAGS.iter().enumerate() {
            let xd = x[i - lag];
            psi[idx_l] = xd;
            psi[4 + idx_l * 3] = xd * xd.abs();
            psi[4 + idx_l * 3 + 1] = xd * xd * xd - 0.5 * xd;
            psi[4 + idx_l * 3 + 2] = xd.abs() - mean_abs_x;
        }
        let r = lin_arx_res[i];
        for col in 0..DIM_AUG {
            let pc = psi[col];
            let base = uplo_u_index(0, col);
            for row in 0..=col {
                g_aug[base + row] += psi[row] * pc;
            }
            c_aug[col][0] += pc * r;
        }
    }
    let mut g_lin4 = vec![0.0_f64; uplo_u_packed_len(4)];
    for col in 0..4 {
        let src = uplo_u_index(0, col);
        g_lin4[src..src + col + 1].copy_from_slice(&g_aug[src..src + col + 1]);
    }
    let e_lin4 = solve_uplo_u_cholesky_2ch(&g_lin4, &c_aug[..4], 1.0e-6).map_or(0.0, |w| {
        (0..4).map(|k| w[k][0] * c_aug[k][0]).sum::<f64>().max(0.0)
    });
    let e_aug16 = solve_uplo_u_cholesky_2ch(&g_aug, &c_aug, 1.0e-6).map_or(0.0, |w| {
        (0..DIM_AUG)
            .map(|k| w[k][0] * c_aug[k][0])
            .sum::<f64>()
            .max(0.0)
    });
    let lin_res_e: f64 = lin_arx_res[P_AR.max(Q_MA)..nl_end]
        .iter()
        .map(|r| r * r)
        .sum();
    let lin_unexplained = (lin_res_e / wet_nl_e.max(1.0e-12)).clamp(0.0, 1.0) as f32;
    let nl_gate = ((lin_unexplained - 0.004) * 35.0).clamp(0.0, 1.0);
    let nl_excess_ratio = ((e_aug16 - e_lin4).max(0.0) / wet_nl_e.max(1.0e-12)).clamp(0.0, 1.0);
    let nl_from_res = ((nl_excess_ratio - 0.0045).max(0.0).sqrt() * 3.4) as f32;
    let nl_from_hf = nl_gate * (0.55 * (hf_growth - 1.06).max(0.0)).clamp(0.0, 0.85);
    let nl_drive = (nl_from_res + nl_from_hf).clamp(0.0, 1.0);

    // Guard against Froissart pole-zero doublets on memoryless / short-FIR / tonal targets where
    // a 16-tap (0.33 ms) causal FIR already explains >= 99.9% of the transfer (< -30 dB residual).
    let h16 = solve_causal_fir_1ch(&x[bb_start..], &y[bb_start..], 0, 16);
    let (mut fir16_res_e, mut fir16_tot_e) = (0.0_f64, 0.0_f64);
    for i in (bb_start + 16)..n {
        let mut p = 0.0_f64;
        for (k, &hk) in h16.iter().enumerate() {
            p += hk * x[i - k];
        }
        let d = y[i] - p;
        fir16_res_e += d * d;
        fir16_tot_e += y[i] * y[i];
    }
    if fir16_tot_e > 1.0e-12 && fir16_res_e <= 1.0e-3 * fir16_tot_e {
        return (nl_drive, Vec::new());
    }

    let mut candidates: Vec<(f64, f32, f32)> = Vec::new();
    // Low-frequency 8x-decimated (6 kHz) pass across all 12k samples resolves 35..900 Hz resonances
    if n >= 1_024 {
        let n_low = n / 8;
        let mut x_low = vec![0.0_f64; n_low];
        let mut y_low = vec![0.0_f64; n_low];
        for k in 0..n_low {
            let b = k * 8;
            x_low[k] = 0.125 * x[b..b + 8].iter().sum::<f64>();
            y_low[k] = 0.125 * y[b..b + 8].iter().sum::<f64>();
        }
        let mut x_lp_hp = vec![0.0_f64; n_low];
        let mut y_lp_hp = vec![0.0_f64; n_low];
        for k in 1..n_low {
            x_lp_hp[k] = x_low[k] - 0.88 * x_low[k - 1];
            y_lp_hp[k] = y_low[k] - 0.88 * y_low[k - 1];
        }
        let (roots_6k, _) = run_steiglitz_mcbride_arma(&x_lp_hp, &y_lp_hp, 12, 50, 0.985);
        for (zr, zi) in roots_6k {
            if zi <= 1.0e-3 {
                continue;
            }
            let rad_6k = (zr * zr + zi * zi).sqrt();
            let th_6k = zi.atan2(zr);
            if rad_6k >= 0.40 && (0.036..=1.25).contains(&th_6k) {
                let r_48 = rad_6k.powf(0.125).clamp(0.88, 0.994);
                let th_48 = th_6k * 0.125;
                let q_score = 1.25 * (r_48 / (1.004 - r_48.min(0.995)));
                candidates.push((q_score, r_48 as f32, th_48 as f32));
            }
        }
    }

    for (zr, zi) in roots_48k {
        if zi <= 1.5e-3 {
            continue;
        }
        let rad = (zr * zr + zi * zi).sqrt();
        let theta = zi.atan2(zr);
        if rad >= 0.65 && (0.085..=1.35).contains(&theta) {
            let r_clamped = rad.clamp(0.85, 0.994) as f32;
            candidates.push((rad / (1.005 - rad.min(0.995)), r_clamped, theta as f32));
        }
    }

    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut sm_poles: Vec<(f32, f32)> = Vec::new();
    for (_, r_m, th_m) in candidates {
        let distinct = sm_poles
            .iter()
            .all(|&(_, eth)| (eth - th_m).abs() >= 0.003 && (eth.ln() - th_m.ln()).abs() >= 0.14);
        if distinct {
            sm_poles.push((r_m, th_m));
            if sm_poles.len() >= 16 {
                break;
            }
        }
    }

    (nl_drive, sm_poles)
}

/// Estimates the causal `64`-tap (`1.33 ms` at `48 kHz`) stereo Early FIR impulse response via
/// regularized autocorrelation/cross-correlation Cholesky (`r[train.pipeline.stage-a-poles]`).
pub(crate) fn solve_early_fir_64_stereo(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> ([f32; 64], [f32; 64]) {
    const TAPS: usize = 64;
    let n = dry_lr.len().min(wet_lr.len());
    if n <= TAPS {
        let mut hl = [0.0_f32; TAPS];
        let mut hr = [0.0_f32; TAPS];
        hl[0] = 0.25;
        hr[0] = 0.25;
        return (hl, hr);
    }

    let mut r_xx = [0.0_f64; TAPS];
    let mut r_xy = [[0.0_f64; 2]; TAPS];
    let step = if n > 4_000 { 2 } else { 1 };
    let mut idx = TAPS;
    while idx < n {
        let x0 = 0.5 * (f64::from(dry_lr[idx][0]) + f64::from(dry_lr[idx][1]));
        let yl = f64::from(wet_lr[idx][0]);
        let yr = f64::from(wet_lr[idx][1]);
        if x0.is_finite() && yl.is_finite() && yr.is_finite() {
            for lag in 0..TAPS {
                let x_lag =
                    0.5 * (f64::from(dry_lr[idx - lag][0]) + f64::from(dry_lr[idx - lag][1]));
                r_xx[lag] += x0 * x_lag;
                r_xy[lag][0] += x_lag * yl;
                r_xy[lag][1] += x_lag * yr;
            }
        }
        idx += step;
    }

    let mut g = vec![0.0_f64; uplo_u_packed_len(TAPS)];
    for j in 0..TAPS {
        for i in 0..=j {
            g[uplo_u_index(i, j)] = r_xx[j - i];
        }
    }
    let ridge = (0.01 * r_xx[0]).max(1.0e-5);
    let solved =
        solve_uplo_u_cholesky_2ch(&g, &r_xy, ridge).unwrap_or_else(|_| vec![[0.0; 2]; TAPS]);

    let mut hl = [0.0_f32; TAPS];
    let mut hr = [0.0_f32; TAPS];
    for k in 0..TAPS {
        hl[k] = (solved[k][0] as f32).clamp(-1.5, 1.5);
        hr[k] = (solved[k][1] as f32).clamp(-1.5, 1.5);
    }
    (hl, hr)
}

fn solve_equilibrated_1ch(g: &[f64], c: &[[f64; 2]], dim: usize, rel_ridge: f64) -> Vec<f64> {
    let mut d_inv = vec![0.0_f64; dim];
    let mut g_eq = vec![0.0_f64; uplo_u_packed_len(dim)];
    let mut c_eq = vec![[0.0_f64; 2]; dim];
    for col in 0..dim {
        let diag = g[uplo_u_index(col, col)];
        d_inv[col] = if diag > 1.0e-20 {
            diag.sqrt().recip()
        } else {
            0.0
        };
    }
    for col in 0..dim {
        let (base, dc) = (uplo_u_index(0, col), d_inv[col]);
        for row in 0..=col {
            g_eq[base + row] = g[base + row] * d_inv[row] * dc;
        }
        if dc == 0.0 {
            g_eq[base + col] = 1.0;
        }
        g_eq[base + col] += 2.0e-3 * (col as f64);
        c_eq[col][0] = c[col][0] * dc;
    }
    solve_uplo_u_cholesky_2ch(&g_eq, &c_eq, rel_ridge).map_or_else(
        |_| vec![0.0_f64; dim],
        |sol| (0..dim).map(|k| sol[k][0] * d_inv[k]).collect(),
    )
}

fn solve_causal_fir_1ch(u: &[f64], target: &[f64], lag_offset: usize, taps: usize) -> Vec<f64> {
    let n = u.len().min(target.len());
    if n <= lag_offset + taps + 1 {
        return vec![0.0_f64; taps];
    }
    let mut g = vec![0.0_f64; uplo_u_packed_len(taps)];
    let mut c = vec![[0.0_f64; 2]; taps];
    let q3 = (3 * n) / 4;
    let mut i = lag_offset + taps + 1;
    while i < n {
        let w = if n >= 8_000 && i >= q3 { 4.0 } else { 1.0 };
        let ti = w * target[i];
        let ti_hp = 16.0 * w * (target[i] - 0.96 * target[i - 1]);
        for col in 0..taps {
            let idx = i - lag_offset - col;
            let uc = u[idx];
            let uc_hp = uc - 0.96 * u[idx - 1];
            let base = uplo_u_index(0, col);
            for row in 0..=col {
                let r_idx = i - lag_offset - row;
                let ur = u[r_idx];
                let ur_hp = ur - 0.96 * u[r_idx - 1];
                g[base + row] += w * (ur * uc + 16.0 * ur_hp * uc_hp);
            }
            c[col][0] += uc * ti + uc_hp * ti_hp;
        }
        i += if n >= 8_000 && i < q3 { 2 } else { 1 };
    }
    solve_equilibrated_1ch(&g, &c, taps, 5.0e-4)
}

/// Deconvolves Stage III (`64`-tap Early FIR + nested `4`-tier `32`-pole Kautz stereo residues)
/// directly from the simulated Stage II `(s2_l, s2_r)` and Stage 0 `(clean_sub_l, clean_sub_r)`
/// signals so `(s3_l, s3_r)` matches stereo cabinet modals and feeds clean signal into Stage IV.
pub(crate) fn fit_deconvolved_stage3_stereo(
    s2_lr: &[[f32; 2]],
    clean_sub_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    pole_schedule: &[(f32, f32); 32],
    mono_only: bool,
    r_chan_scale: f32,
) -> Stage3DeconvolvedFit {
    let n = s2_lr.len().min(clean_sub_lr.len()).min(wet_lr.len());
    let mut u_l = vec![0.0_f64; n];
    let mut u_r = vec![0.0_f64; n];
    let mut rem_l = vec![0.0_f64; n];
    let mut rem_r = vec![0.0_f64; n];
    for i in 0..n {
        u_l[i] = f64::from(s2_lr[i][0]);
        u_r[i] = f64::from(s2_lr[i][1]);
        rem_l[i] = f64::from(wet_lr[i][0]) - f64::from(clean_sub_lr[i][0]);
        let wr = if mono_only {
            wet_lr[i][0]
        } else {
            wet_lr[i][1]
        };
        rem_r[i] = f64::from(wr) - f64::from(clean_sub_lr[i][1]);
    }

    // 1. Fit 64-tap FIR first so onset segment 0 (lags 0..15) is free of IIR truncation bias
    let h64_l = solve_causal_fir_1ch(&u_l, &rem_l, 0, 64);
    let h64_r = if mono_only {
        h64_l.clone()
    } else {
        solve_causal_fir_1ch(&u_r, &rem_r, 0, 64)
    };

    let mut fir_l = [0.0_f32; 64];
    let mut fir_r = [0.0_f32; 64];
    for k in 0..16 {
        fir_l[k] = (h64_l[k] as f32).clamp(-2.0, 2.0);
        fir_r[k] = (h64_r[k] as f32).clamp(-2.0, 2.0);
    }

    let mut s3_lr = vec![[0.0_f32; 2]; n];
    for i in 0..n {
        let mut dot_l = 0.0_f64;
        let mut dot_r = 0.0_f64;
        for k in 0..=i.min(15) {
            dot_l += f64::from(fir_l[k]) * u_l[i - k];
            dot_r += f64::from(fir_r[k]) * u_r[i - k];
        }
        s3_lr[i][0] = (f64::from(clean_sub_lr[i][0]) + dot_l) as f32;
        s3_lr[i][1] = (f64::from(clean_sub_lr[i][1]) + dot_r) as f32;
        rem_l[i] -= dot_l;
        rem_r[i] -= dot_r;
    }

    // 2. Simulate all 32 Kautz states driven by 16-sample-delayed Stage II output s2[n - 16]
    let mut s_re_l = vec![[0.0_f64; 32]; n];
    let mut s_im_l = vec![[0.0_f64; 32]; n];
    let mut s_re_r = vec![[0.0_f64; 32]; n];
    let mut s_im_r = vec![[0.0_f64; 32]; n];
    for m in 0..32 {
        let (r_low, theta) = pole_schedule[m];
        let (cos_t, sin_t) = algebraic_unit_rotor_op_m(theta);
        let l_re = f64::from(r_low * cos_t);
        let l_im = f64::from(r_low * sin_t);
        let b_g = (1.0_f64 - f64::from(r_low) * f64::from(r_low))
            .max(1.0e-6)
            .sqrt();
        let (mut pr_l, mut pi_l, mut pr_r, mut pi_r) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n {
            let xl = if i >= 16 { u_l[i - 16] } else { 0.0 };
            let xr = if i >= 16 { u_r[i - 16] } else { 0.0 };
            let nr_l = l_re * pr_l - l_im * pi_l + b_g * xl;
            let ni_l = l_im * pr_l + l_re * pi_l;
            let nr_r = l_re * pr_r - l_im * pi_r + b_g * xr;
            let ni_r = l_im * pr_r + l_re * pi_r;
            pr_l = nr_l;
            pi_l = ni_l;
            pr_r = nr_r;
            pi_r = ni_r;
            s_re_l[i][m] = nr_l;
            s_im_l[i][m] = ni_l;
            s_re_r[i][m] = nr_r;
            s_im_r[i][m] = ni_r;
        }
    }

    // 3. Nested per-tier (Lite -> Full -> Pro -> Ultra) stereo Kautz residue fitting
    let mut kautz_c_l = [(0.0_f32, 0.0_f32); 32];
    let mut kautz_c_r = [(0.0_f32, 0.0_f32); 32];
    let mut pole_start = 0usize;
    for &n_blocks in &STAGE3_KAUTZ_BLOCKS_PER_TIER {
        let p_cnt = n_blocks * 4;
        let dim = p_cnt * 2;
        let mut g_l = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c_l = vec![[0.0_f64; 2]; dim];
        let mut g_r = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c_r = vec![[0.0_f64; 2]; dim];
        let mut phi_l = vec![0.0_f64; dim];
        let mut phi_r = vec![0.0_f64; dim];
        let mut phi_hp_l = vec![0.0_f64; dim];
        let mut phi_hp_r = vec![0.0_f64; dim];

        let q3 = (3 * n) / 4;
        let mut i = 33usize;
        while i < n {
            let w = if n >= 8_000 && i >= q3 {
                4.0_f64
            } else {
                1.0_f64
            };
            let rl_hp = rem_l[i] - 0.96 * rem_l[i - 1];
            let rr_hp = rem_r[i] - 0.96 * rem_r[i - 1];
            for k in 0..p_cnt {
                let m = pole_start + k;
                phi_l[2 * k] = 0.25 * s_re_l[i][m];
                phi_l[2 * k + 1] = -0.25 * s_im_l[i][m];
                phi_r[2 * k] = 0.25 * s_re_r[i][m];
                phi_r[2 * k + 1] = -0.25 * s_im_r[i][m];
                phi_hp_l[2 * k] = 0.25 * (s_re_l[i][m] - 0.96 * s_re_l[i - 1][m]);
                phi_hp_l[2 * k + 1] = -0.25 * (s_im_l[i][m] - 0.96 * s_im_l[i - 1][m]);
                phi_hp_r[2 * k] = 0.25 * (s_re_r[i][m] - 0.96 * s_re_r[i - 1][m]);
                phi_hp_r[2 * k + 1] = -0.25 * (s_im_r[i][m] - 0.96 * s_im_r[i - 1][m]);
            }
            for col in 0..dim {
                let base = uplo_u_index(0, col);
                for row in 0..=col {
                    g_l[base + row] +=
                        w * (phi_l[row] * phi_l[col] + 16.0 * phi_hp_l[row] * phi_hp_l[col]);
                    g_r[base + row] +=
                        w * (phi_r[row] * phi_r[col] + 16.0 * phi_hp_r[row] * phi_hp_r[col]);
                }
                c_l[col][0] += w * (phi_l[col] * rem_l[i] + 16.0 * phi_hp_l[col] * rl_hp);
                c_r[col][0] += w * (phi_r[col] * rem_r[i] + 16.0 * phi_hp_r[col] * rr_hp);
            }
            i += if n >= 8_000 && i < q3 { 2 } else { 1 };
        }

        let sol_l = solve_equilibrated_1ch(&g_l, &c_l, dim, 2.0e-4);
        let sol_r = if mono_only {
            sol_l.clone()
        } else {
            solve_equilibrated_1ch(&g_r, &c_r, dim, 2.0e-4)
        };

        for k in 0..p_cnt {
            let m = pole_start + k;
            const C_FLOOR: f32 = 0.002;
            const C_CAP: f32 = 1.60;

            let (mut fl_re, mut fl_im) = (sol_l[2 * k] as f32, sol_l[2 * k + 1] as f32);
            let (mut fr_re, mut fr_im) = (sol_r[2 * k] as f32, sol_r[2 * k + 1] as f32);
            let nl_mag = fl_re.hypot(fl_im);
            let nr_mag = fr_re.hypot(fr_im);
            if nl_mag > C_CAP {
                let s = C_CAP / nl_mag;
                fl_re *= s;
                fl_im *= s;
            }
            if nr_mag > C_CAP {
                let s = C_CAP / nr_mag;
                fr_re *= s;
                fr_im *= s;
            }

            let (mut cre_l, cim_l, cre_r, cim_r) = if nl_mag.max(nr_mag) >= C_FLOOR {
                (fl_re, fl_im, fr_re, fr_im)
            } else {
                let ph = 0.37_f32 * ((m + 1) as f32);
                let cl = (C_FLOOR * ph.cos(), C_FLOOR * ph.sin());
                (cl.0, cl.1, cl.0 * r_chan_scale, cl.1 * r_chan_scale)
            };
            if (cre_l + cre_r).hypot(cim_l + cim_r) < C_FLOOR {
                cre_l += C_FLOOR;
            }
            kautz_c_l[m] = (cre_l, cim_l);
            kautz_c_r[m] = (cre_r, cim_r);

            for i in 0..n {
                let dl = 0.25 * (f64::from(cre_l) * s_re_l[i][m] - f64::from(cim_l) * s_im_l[i][m]);
                let dr = 0.25 * (f64::from(cre_r) * s_re_r[i][m] - f64::from(cim_r) * s_im_r[i][m]);
                rem_l[i] -= dl;
                rem_r[i] -= dr;
                s3_lr[i][0] += dl as f32;
                s3_lr[i][1] += dr as f32;
            }
        }
        pole_start += p_cnt;
    }

    // 4. Fit post-Kautz Early FIR segments 1..3 (lags 16..63) to capture remaining fine detail
    let h48_l = solve_causal_fir_1ch(&u_l, &rem_l, 16, 48);
    let h48_r = if mono_only {
        h48_l.clone()
    } else {
        solve_causal_fir_1ch(&u_r, &rem_r, 16, 48)
    };
    for k in 0..48 {
        fir_l[16 + k] = (h48_l[k] as f32).clamp(-2.0, 2.0);
        fir_r[16 + k] = (h48_r[k] as f32).clamp(-2.0, 2.0);
    }

    Stage3DeconvolvedFit {
        fir_l,
        fir_r,
        kautz_c_l,
        kautz_c_r,
        s3_lr,
    }
}

/// Identifies stereo reflection lag anchors via pre-whitened cross-correlation of `(wet - s3)`
/// against `s3`, mapping Left peaks to EVEN lanes (`lines[0, 2]`) and Right peaks to ODD lanes
/// (`lines[1, 3]`), plus estimating reverberant feedback gain `g_loop` (`r[train.pipeline.stage-a-poles]`).
pub(crate) fn extract_stage4_reflection_delays(
    s3_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 8], f32) {
    let def_lite = [53.0_f32, 89.0, 137.0, 199.0];
    let def_full = [257.0_f32, 313.0, 389.0, 463.0];
    let def_pro = [523.0_f32, 593.0, 661.0, 743.0];
    let def_ultra = [773.0_f32, 809.0, 839.0, 877.0, 907.0, 941.0, 971.0, 997.0];

    let n_all = s3_lr.len().min(wet_lr.len()).min(12_000);
    if n_all < 1_100 {
        return (def_lite, def_full, def_pro, def_ultra, 0.08);
    }
    // Use the broadband transient + clean release decay quarter [75%..100%] when full 12k coreset is present
    let start_off = if n_all >= 11_000 {
        (n_all * 3) / 4 - 128
    } else {
        0
    };
    let s3_lr = &s3_lr[start_off..n_all];
    let wet_lr = &wet_lr[start_off..n_all];
    let n = s3_lr.len();

    let (mut ds_l, mut ds_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    let (mut dr_l, mut dr_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    let (mut es_l, mut es_r, mut ew_tot) = (0.0, 0.0, 0.0);
    for i in 1..n {
        let sl = f64::from(s3_lr[i][0]) - 0.94 * f64::from(s3_lr[i - 1][0]);
        let sr = f64::from(s3_lr[i][1]) - 0.94 * f64::from(s3_lr[i - 1][1]);
        let rl_cur = f64::from(wet_lr[i][0]) - f64::from(s3_lr[i][0]);
        let rl_prv = f64::from(wet_lr[i - 1][0]) - f64::from(s3_lr[i - 1][0]);
        let rr_cur = f64::from(wet_lr[i][1]) - f64::from(s3_lr[i][1]);
        let rr_prv = f64::from(wet_lr[i - 1][1]) - f64::from(s3_lr[i - 1][1]);
        let rl = rl_cur - 0.94 * rl_prv;
        let rr = rr_cur - 0.94 * rr_prv;
        ds_l[i] = sl;
        ds_r[i] = sr;
        dr_l[i] = rl;
        dr_r[i] = rr;
        if i >= 64 {
            es_l += sl * sl;
            es_r += sr * sr;
            let wl = f64::from(wet_lr[i][0]) - 0.94 * f64::from(wet_lr[i - 1][0]);
            let wr = f64::from(wet_lr[i][1]) - 0.94 * f64::from(wet_lr[i - 1][1]);
            ew_tot += 0.5 * (wl * wl + wr * wr);
        }
    }
    let norm_l = (es_l * ew_tot).sqrt().max(1.0e-12);
    let norm_r = (es_r * ew_tot).sqrt().max(1.0e-12);
    let es_l_safe = es_l.max(1.0e-12);
    let es_r_safe = es_r.max(1.0e-12);

    let mut corr_l = vec![0.0_f64; 1_000];
    let mut corr_r = vec![0.0_f64; 1_000];
    for lag in 24..998 {
        let (mut sl, mut sr, mut al, mut ar) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for idx in lag.max(64)..n {
            sl += dr_l[idx] * ds_l[idx - lag];
            sr += dr_r[idx] * ds_r[idx - lag];
            al += ds_l[idx] * ds_l[idx - lag];
            ar += ds_r[idx] * ds_r[idx - lag];
        }
        let pen_l = (1.0 - 0.85 * (al / es_l_safe).powi(2)).clamp(0.0, 1.0);
        let pen_r = (1.0 - 0.85 * (ar / es_r_safe).powi(2)).clamp(0.0, 1.0);
        corr_l[lag] = (sl.abs() / norm_l) * pen_l;
        corr_r[lag] = (sr.abs() / norm_r) * pen_r;
    }

    let pick_ch =
        |corr: &[f64], lo: usize, hi: usize, min_sep: usize, defs: &[f32]| -> Vec<usize> {
            let k = defs.len();
            let mut chosen: Vec<usize> = Vec::with_capacity(k);
            let mut peaks: Vec<(f64, usize)> = (lo + 1..hi)
                .filter(|&l| corr[l] > 0.040 && corr[l] >= corr[l - 1] && corr[l] >= corr[l + 1])
                .map(|l| (corr[l], l))
                .collect();
            peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, lag) in peaks {
                if chosen.iter().all(|&c| c.abs_diff(lag) >= min_sep) {
                    chosen.push(lag);
                    if chosen.len() == k {
                        break;
                    }
                }
            }
            for &d in defs {
                if chosen.len() == k {
                    break;
                }
                let di = d as usize;
                if chosen.iter().all(|&c| c.abs_diff(di) >= min_sep / 2) {
                    chosen.push(di);
                }
            }
            while chosen.len() < k {
                chosen.push(defs[chosen.len()] as usize);
            }
            chosen.sort_unstable();
            chosen
        };

    let l_lite = pick_ch(&corr_l, 24, 252, 18, &[def_lite[0], def_lite[2]]);
    let r_lite = pick_ch(&corr_r, 24, 252, 18, &[def_lite[1], def_lite[3]]);
    let mut l_full = pick_ch(&corr_l, 255, 508, 24, &[def_full[0], def_full[2]]);
    let mut r_full = pick_ch(&corr_r, 255, 508, 24, &[def_full[1], def_full[3]]);
    let l_pro = pick_ch(&corr_l, 514, 764, 28, &[def_pro[0], def_pro[2]]);
    let r_pro = pick_ch(&corr_r, 514, 764, 28, &[def_pro[1], def_pro[3]]);
    let mut l_ultra = pick_ch(
        &corr_l,
        768,
        997,
        16,
        &[def_ultra[0], def_ultra[2], def_ultra[4], def_ultra[6]],
    );
    let mut r_ultra = pick_ch(
        &corr_r,
        768,
        997,
        16,
        &[def_ultra[1], def_ultra[3], def_ultra[5], def_ultra[7]],
    );

    // Mirror strong Lite (< 252) reflections into unused Full/Ultra raw slots to bypass TankDcBlocker LF phase shift
    for (slot_idx, &lag_l) in l_lite.iter().enumerate() {
        if corr_l[lag_l] > 0.065 {
            if corr_l[l_full[slot_idx]] <= 0.040 {
                l_full[slot_idx] = lag_l;
            } else if corr_l[l_ultra[slot_idx]] <= 0.040 {
                l_ultra[slot_idx] = lag_l;
            }
        }
    }
    for (slot_idx, &lag_r) in r_lite.iter().enumerate() {
        if corr_r[lag_r] > 0.065 {
            if corr_r[r_full[slot_idx]] <= 0.040 {
                r_full[slot_idx] = lag_r;
            } else if corr_r[r_ultra[slot_idx]] <= 0.040 {
                r_ultra[slot_idx] = lag_r;
            }
        }
    }

    let late_mean: f64 = (450..995)
        .map(|l| 0.5 * (corr_l[l] + corr_r[l]))
        .sum::<f64>()
        / 545.0;
    let g_loop = (0.05 + 4.5 * (late_mean - 0.022).max(0.0)).clamp(0.05, 0.85) as f32;

    let pair4 = |l: &[usize], r: &[usize]| [l[0] as f32, r[0] as f32, l[1] as f32, r[1] as f32];
    let u8_arr = std::array::from_fn(|k| {
        if k % 2 == 0 {
            l_ultra[k / 2] as f32
        } else {
            r_ultra[k / 2] as f32
        }
    });
    (
        pair4(&l_lite, &r_lite),
        pair4(&l_full, &r_full),
        pair4(&l_pro, &r_pro),
        u8_arr,
        g_loop,
    )
}
