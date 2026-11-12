#![forbid(unsafe_code)]
//! Exact 4-Tier 3-Band Binomial/Bézout (`LF < 250 Hz`, `MF 250..2,500 Hz`, `HF > 2,500 Hz`)
//! Feature-Bus Contraction & Post-`LUFS`-Trim `bESR` Evaluation (`solver::bezout_gram`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-c-schur]`)
//! - `docs/specs/architecture/psychoacoustic_quality_and_tiers.md`
//!   (`ref[qual.besr.bezout-crossover]`, `ref[qual.besr.finite-timeline]`, `ref[qual.besr.per-output]`)

use std::f64::consts::PI;

use super::wide_gram::{Wide192Accum, contract_256d_gram_tiled_wide192};
use super::{uplo_u_index, uplo_u_packed_len};

/// Canonical `48 kHz` sample rate for TORA Stage B / Stage C coreset contraction.
const SOLVER_SR_HZ: f64 = 48_000.0;

/// Relative band-energy floor `1e-4 * Y_tot` (`-40 dB`) preventing silent-band ill-conditioning
/// (`ref[train.pipeline.stage-c-schur]`).
const BAND_WEIGHT_REL_FLOOR: f64 = 1.0e-4;

/// Harvested per-tier (`Lite 64-D, Full 128-D, Pro 192-D, Ultra 256-D`) 3-band Binomial/Bézout
/// and post-warmup `LUFS` sufficient statistics.
#[derive(Debug, Clone)]
pub struct BezoutHarvestStats {
    /// Post-warmup target 3-band energies `[LF, MF, HF]` per stereo output channel `[L, R]`.
    pub target_band_energy_lr: [[f64; 3]; 2],
    /// Post-warmup sampled target 3-band energies `[LF, MF, HF]` sharing the exact inner-product stride of `tier_gram_lr`.
    pub sampled_target_band_energy_lr: [[f64; 3]; 2],
    /// Normalized 3-band weights $w_{b,c} = \frac{Y_{\text{tot},c}}{3 \max(Y_{b,c}, \text{floor}_{b,c})}$ per channel.
    pub band_weights_lr: [[f64; 3]; 2],
    /// Per-tier (`0..4`), per-channel `uplo='U'` (`64, 128, 192, 256`-D) Bézout-weighted Gram matrices.
    pub tier_gram_lr: [[Vec<f64>; 2]; 4],
    /// Per-tier (`0..4`) `256 × 2` Bézout-weighted cross-correlations $\mathbf{C}_{\text{bESR}}^{(T)}$.
    pub tier_cross_256: [[[f64; 2]; 256]; 4],
    /// Per-tier (`0..4`) Bézout-weighted zero-head base residual energies $R_{\text{bESR},0}^{(T)}$ per channel `[L, R]`.
    pub tier_base_res_lr: [[f64; 2]; 4],
    /// Per-tier (`0..4`) Bézout-weighted target-residual dot products $\sum_b w_{b,c} \langle \mathcal{H}_b y_c, \mathcal{H}_b r_{t,c} \rangle$.
    pub tier_tgt_res_dot_lr: [[f64; 2]; 4],
    /// Per-tier (`0..4`) Bézout-weighted feature-target cross-correlations $\sum_b w_{b,c} \langle \mathcal{H}_b \phi^{(T)}_d, \mathcal{H}_b y_c \rangle$.
    pub tier_tgt_fb_cross: [[[f64; 2]; 256]; 4],
    /// Per-tier (`0..4`) post-warmup unweighted `uplo='U'` (`64, 128, 192, 256`-D) Gram matrices for `Stage E` `LUFS`.
    pub tier_lufs_gram: [Vec<f64>; 4],
    /// Per-tier (`0..4`) post-warmup backbone broadband energy $\sum b_{t,c}^2$ per channel `[L, R]`.
    pub tier_lufs_bb_energy_lr: [[f64; 2]; 4],
    /// Per-tier (`0..4`) post-warmup backbone-feature cross-correlation $\sum b_{t,c} \phi^{(T)}_d$.
    pub tier_lufs_bb_cross: [[[f64; 2]; 256]; 4],
    /// Per-tier (`0..4`) post-warmup unweighted broadband residual energy $\sum r_{t,c}^2$ per channel.
    pub tier_bb_res_energy_lr: [[f64; 2]; 4],
    /// Per-tier (`0..4`) post-warmup unweighted broadband cross-correlation $\sum \phi^{(T)}_d r_{t,c}$.
    pub tier_bb_res_cross_256: [[[f64; 2]; 256]; 4],
    /// Post-warmup frame count (`n - warmup`) for exact `LUFS` mean-square normalization.
    pub post_warmup_frames: usize,
    /// Post-warmup sample index (`min(512, n / 4)` when `n > 64`, else `0`).
    pub warmup_samples: usize,
}

#[inline]
fn step_bezout7_lp_scalar(x: f64, r: f64, omr: f64, u: &mut [f64; 7]) -> f64 {
    let mut prev = x;
    for stage in u.iter_mut() {
        *stage = omr * prev + r * *stage;
        prev = *stage;
    }
    35.0 * u[3] - 84.0 * u[4] + 70.0 * u[5] - 20.0 * u[6]
}

#[inline]
fn advance_bezout7_lp_dim(fb: &[f32; 256], dim: usize, r: f64, omr: f64, u: &mut [[f64; 256]; 7]) {
    for d in 0..dim {
        let mut prev = f64::from(fb[d]);
        for st in u.iter_mut() {
            prev = omr * prev + r * st[d];
            st[d] = prev;
        }
    }
}

#[inline]
fn readout_bezout7_bands_dim(
    fb: &[f32; 256],
    dim: usize,
    u1: &[[f64; 256]; 7],
    u2: &[[f64; 256]; 7],
    out_lf: &mut [f32; 256],
    out_mf: &mut [f32; 256],
    out_hf: &mut [f32; 256],
) {
    for d in 0..dim {
        let x = f64::from(fb[d]);
        let l250 = 35.0 * u1[3][d] - 84.0 * u1[4][d] + 70.0 * u1[5][d] - 20.0 * u1[6][d];
        let l2500 = 35.0 * u2[3][d] - 84.0 * u2[4][d] + 70.0 * u2[5][d] - 20.0 * u2[6][d];
        out_lf[d] = l250 as f32;
        out_mf[d] = (l2500 - l250) as f32;
        out_hf[d] = (x - l2500) as f32;
    }
}

/// Computes canonical post-warmup offset matching Stage E and certification gates (`512` on `12,000` frames).
#[must_use]
pub const fn canonical_besr_warmup(n_frames: usize) -> usize {
    if n_frames > 64 {
        let q = n_frames / 4;
        if q < 512 { q } else { 512 }
    } else {
        0
    }
}

/// Contracts exact 3-band Binomial/Bézout (`LF`, `MF`, `HF`) `Wide192Accum` Gram and cross-correlation
/// statistics when all 4 tiers share a single `256-D` feature trajectory.
#[must_use]
pub fn contract_bezout_3band_stats(
    full_features_256: &[[f32; 256]],
    wet_lr: &[[f32; 2]],
    tier_backbone_lr: &[Vec<[f32; 2]>; 4],
    eval_stereo: bool,
    gram_frame_stride: usize,
) -> BezoutHarvestStats {
    contract_bezout_3band_stats_4tier(
        &[
            full_features_256,
            full_features_256,
            full_features_256,
            full_features_256,
        ],
        wet_lr,
        tier_backbone_lr,
        eval_stereo,
        gram_frame_stride,
    )
}

/// Contracts exact per-tier (`Lite 64-D, Full 128-D, Pro 192-D, Ultra 256-D`) 3-band Binomial/Bézout
/// and post-warmup `LUFS` sufficient statistics via `Wide192Accum`.
#[must_use]
pub fn contract_bezout_3band_stats_4tier(
    tier_features_256: &[&[[f32; 256]]; 4],
    wet_lr: &[[f32; 2]],
    tier_backbone_lr: &[Vec<[f32; 2]>; 4],
    eval_stereo: bool,
    gram_frame_stride: usize,
) -> BezoutHarvestStats {
    let n = tier_features_256
        .iter()
        .fold(wet_lr.len(), |acc, s| acc.min(s.len()));
    let warmup = canonical_besr_warmup(n);
    let empty_tier_grams = || -> [[Vec<f64>; 2]; 4] {
        std::array::from_fn(|t| {
            let plen = uplo_u_packed_len((t + 1) * 64);
            [vec![0.0; plen], vec![0.0; plen]]
        })
    };
    let empty_lufs_grams =
        || -> [Vec<f64>; 4] { std::array::from_fn(|t| vec![0.0; uplo_u_packed_len((t + 1) * 64)]) };
    if n == 0 || warmup >= n {
        return BezoutHarvestStats {
            target_band_energy_lr: [[0.0; 3]; 2],
            sampled_target_band_energy_lr: [[0.0; 3]; 2],
            band_weights_lr: [[1.0 / 3.0; 3]; 2],
            tier_gram_lr: empty_tier_grams(),
            tier_cross_256: [[[0.0; 2]; 256]; 4],
            tier_base_res_lr: [[0.0; 2]; 4],
            tier_tgt_res_dot_lr: [[0.0; 2]; 4],
            tier_tgt_fb_cross: [[[0.0; 2]; 256]; 4],
            tier_lufs_gram: empty_lufs_grams(),
            tier_lufs_bb_energy_lr: [[0.0; 2]; 4],
            tier_lufs_bb_cross: [[[0.0; 2]; 256]; 4],
            tier_bb_res_energy_lr: [[0.0; 2]; 4],
            tier_bb_res_cross_256: [[[0.0; 2]; 256]; 4],
            post_warmup_frames: 0,
            warmup_samples: warmup,
        };
    }

    let r1 = (-2.0 * PI * 250.0 / SOLVER_SR_HZ).exp();
    let omr1 = 1.0 - r1;
    let r2 = (-2.0 * PI * 2_500.0 / SOLVER_SR_HZ).exp();
    let omr2 = 1.0 - r2;

    let eff_stride = if n <= 48_000 {
        1
    } else {
        gram_frame_stride.max(1)
    };
    let stride_lm = eff_stride;
    let stride_hf = if n <= 96_000 { 1 } else { eff_stride.min(2) };
    let post_n = n - warmup;
    let m_lm = post_n.div_ceil(stride_lm).max(1);
    let m_hf = post_n.div_ceil(stride_hf).max(1);
    let scale_lm = (post_n as f64) / (m_lm as f64);
    let scale_hf = (post_n as f64) / (m_hf as f64);
    let weights_lm = vec![scale_lm; m_lm];
    let weights_hf = vec![scale_hf; m_hf];

    // 1. Pre-filter target wet_lr through 3-band Bézout crossover once across 0..n
    let mut u1_tgt = [[0.0_f64; 7]; 2];
    let mut u2_tgt = [[0.0_f64; 7]; 2];
    let mut tgt_bands_seq = vec![[[0.0_f64; 3]; 2]; post_n];
    let mut tgt_e_acc = [[Wide192Accum::default(); 3]; 2];
    let mut samp_tgt_e_acc = [[Wide192Accum::default(); 3]; 2];
    for i in 0..n {
        let yl = f64::from(wet_lr[i][0]);
        let yr = if eval_stereo {
            f64::from(wet_lr[i][1])
        } else {
            yl
        };
        let tgt_lr = [yl, yr];
        let mut tb = [[0.0_f64; 3]; 2];
        for ch in 0..2 {
            let l250 = step_bezout7_lp_scalar(tgt_lr[ch], r1, omr1, &mut u1_tgt[ch]);
            let l2500 = step_bezout7_lp_scalar(tgt_lr[ch], r2, omr2, &mut u2_tgt[ch]);
            tb[ch] = [l250, l2500 - l250, tgt_lr[ch] - l2500];
        }
        if i >= warmup {
            let k = i - warmup;
            tgt_bands_seq[k] = tb;
            let take_lm = k.is_multiple_of(stride_lm);
            let take_hf = k.is_multiple_of(stride_hf);
            for ch in 0..2 {
                for b in 0..3 {
                    let sq = tb[ch][b] * tb[ch][b];
                    tgt_e_acc[ch][b].add_f64(sq);
                    if b < 2 && take_lm {
                        samp_tgt_e_acc[ch][b].add_f64(scale_lm * sq);
                    } else if b == 2 && take_hf {
                        samp_tgt_e_acc[ch][2].add_f64(scale_hf * sq);
                    }
                }
            }
        }
    }
    let target_band_energy_lr =
        std::array::from_fn(|ch| std::array::from_fn(|b| tgt_e_acc[ch][b].to_f64()));
    let sampled_target_band_energy_lr =
        std::array::from_fn(|ch| std::array::from_fn(|b| samp_tgt_e_acc[ch][b].to_f64()));

    let mut band_weights_lr = [[1.0 / 3.0; 3]; 2];
    for ch in 0..2 {
        let y_tot = sampled_target_band_energy_lr[ch][0]
            + sampled_target_band_energy_lr[ch][1]
            + sampled_target_band_energy_lr[ch][2];
        if y_tot > 1.0e-15 {
            for b in 0..3 {
                let e_b = sampled_target_band_energy_lr[ch][b];
                let floor = if e_b > 0.0 {
                    1.0e-7 * y_tot
                } else {
                    BAND_WEIGHT_REL_FLOOR * y_tot
                };
                band_weights_lr[ch][b] = y_tot / (3.0 * e_b.max(floor));
            }
        }
    }

    // 2. Per-tier exact feature-bus & backbone contraction (dim = 64, 128, 192, 256)
    let mut tier_gram_lr = empty_tier_grams();
    let mut tier_cross_256 = [[[0.0_f64; 2]; 256]; 4];
    let mut tier_base_res_lr = [[0.0_f64; 2]; 4];
    let mut tier_tgt_res_dot_lr = [[0.0_f64; 2]; 4];
    let mut tier_tgt_fb_cross = [[[0.0_f64; 2]; 256]; 4];
    let mut tier_lufs_gram = empty_lufs_grams();
    let mut tier_lufs_bb_energy_lr = [[0.0_f64; 2]; 4];
    let mut tier_lufs_bb_cross = [[[0.0_f64; 2]; 256]; 4];
    let mut tier_bb_res_energy_lr = [[0.0_f64; 2]; 4];
    let mut tier_bb_res_cross_256 = [[[0.0_f64; 2]; 256]; 4];

    for t in 0..4 {
        let dim = (t + 1) * 64;
        let p_len = uplo_u_packed_len(dim);
        let feats_t = tier_features_256[t];
        let mut u1_fb = Box::new([[0.0_f64; 256]; 7]);
        let mut u2_fb = Box::new([[0.0_f64; 256]; 7]);
        let mut u1_res = [[0.0_f64; 7]; 2];
        let mut u2_res = [[0.0_f64; 7]; 2];

        let mut raw_feats = Vec::with_capacity(m_lm);
        let mut raw_bb = Vec::with_capacity(m_lm);
        let mut lf_feats = Vec::with_capacity(m_lm);
        let mut mf_feats = Vec::with_capacity(m_lm);
        let mut hf_feats = Vec::with_capacity(m_hf);
        let mut lf_res = Vec::with_capacity(m_lm);
        let mut mf_res = Vec::with_capacity(m_lm);
        let mut hf_res = Vec::with_capacity(m_hf);

        let mut band_res_acc = [[Wide192Accum::default(); 3]; 2];
        let mut tgt_res_acc = [[Wide192Accum::default(); 3]; 2];
        let mut tgt_fb_acc = vec![[[Wide192Accum::default(); 2]; 3]; dim];
        let mut bb_energy_acc = [Wide192Accum::default(); 2];
        let mut bb_res_e_acc = [Wide192Accum::default(); 2];
        let mut bb_res_c_acc = vec![[Wide192Accum::default(); 2]; dim];

        let mut fb_raw = [0.0_f32; 256];
        let mut fb_lf = [0.0_f32; 256];
        let mut fb_mf = [0.0_f32; 256];
        let mut fb_hf = [0.0_f32; 256];

        for i in 0..n {
            advance_bezout7_lp_dim(&feats_t[i], dim, r1, omr1, &mut u1_fb);
            advance_bezout7_lp_dim(&feats_t[i], dim, r2, omr2, &mut u2_fb);

            let yl = f64::from(wet_lr[i][0]);
            let yr = if eval_stereo {
                f64::from(wet_lr[i][1])
            } else {
                yl
            };
            let [bl_f32, br_f32] = tier_backbone_lr[t].get(i).copied().unwrap_or([0.0, 0.0]);
            let bl = f64::from(bl_f32);
            let br = if eval_stereo { f64::from(br_f32) } else { bl };
            let r_lr = [yl - bl, yr - br];
            let mut rb = [[0.0_f64; 3]; 2];
            for ch in 0..2 {
                let l250 = step_bezout7_lp_scalar(r_lr[ch], r1, omr1, &mut u1_res[ch]);
                let l2500 = step_bezout7_lp_scalar(r_lr[ch], r2, omr2, &mut u2_res[ch]);
                rb[ch] = [l250, l2500 - l250, r_lr[ch] - l2500];
            }

            if i < warmup {
                continue;
            }
            let k = i - warmup;
            let take_lm = k.is_multiple_of(stride_lm);
            let take_hf = k.is_multiple_of(stride_hf);
            if !take_lm && !take_hf {
                continue;
            }

            let tb = tgt_bands_seq[k];
            readout_bezout7_bands_dim(
                &feats_t[i],
                dim,
                &u1_fb,
                &u2_fb,
                &mut fb_lf,
                &mut fb_mf,
                &mut fb_hf,
            );

            if take_lm {
                fb_raw[..dim].copy_from_slice(&feats_t[i][..dim]);
                raw_feats.push(fb_raw);
                raw_bb.push([bl, br]);
                lf_feats.push(fb_lf);
                mf_feats.push(fb_mf);
                lf_res.push([rb[0][0], rb[1][0]]);
                mf_res.push([rb[0][1], rb[1][1]]);
                bb_energy_acc[0].add_f64(scale_lm * bl * bl);
                bb_energy_acc[1].add_f64(scale_lm * br * br);
                for ch in 0..2 {
                    for b in 0..2 {
                        band_res_acc[ch][b].add_f64(scale_lm * rb[ch][b] * rb[ch][b]);
                        tgt_res_acc[ch][b].add_f64(scale_lm * tb[ch][b] * rb[ch][b]);
                    }
                }
                let (tl_lf, tr_lf) = (scale_lm * tb[0][0], scale_lm * tb[1][0]);
                let (tl_mf, tr_mf) = (scale_lm * tb[0][1], scale_lm * tb[1][1]);
                let (rl_w, rr_w) = (scale_lm * r_lr[0], scale_lm * r_lr[1]);
                bb_res_e_acc[0].add_f64(rl_w * r_lr[0]);
                bb_res_e_acc[1].add_f64(rr_w * r_lr[1]);
                for d in 0..dim {
                    let fr = f64::from(fb_raw[d]);
                    bb_res_c_acc[d][0].add_f64(fr * rl_w);
                    bb_res_c_acc[d][1].add_f64(fr * rr_w);
                }
                for d in 0..dim {
                    let flf = f64::from(fb_lf[d]);
                    let fmf = f64::from(fb_mf[d]);
                    tgt_fb_acc[d][0][0].add_f64(flf * tl_lf);
                    tgt_fb_acc[d][0][1].add_f64(flf * tr_lf);
                    tgt_fb_acc[d][1][0].add_f64(fmf * tl_mf);
                    tgt_fb_acc[d][1][1].add_f64(fmf * tr_mf);
                }
            }

            if take_hf {
                hf_feats.push(fb_hf);
                hf_res.push([rb[0][2], rb[1][2]]);
                for ch in 0..2 {
                    band_res_acc[ch][2].add_f64(scale_hf * rb[ch][2] * rb[ch][2]);
                    tgt_res_acc[ch][2].add_f64(scale_hf * tb[ch][2] * rb[ch][2]);
                }
                let (tl_hf, tr_hf) = (scale_hf * tb[0][2], scale_hf * tb[1][2]);
                for d in 0..dim {
                    let fhf = f64::from(fb_hf[d]);
                    tgt_fb_acc[d][2][0].add_f64(fhf * tl_hf);
                    tgt_fb_acc[d][2][1].add_f64(fhf * tr_hf);
                }
            }
        }

        let (g_lufs_256, c_bb) =
            contract_256d_gram_tiled_wide192(&raw_feats, &raw_bb, &weights_lm, 1);
        let (g_lf, c_lf) = contract_256d_gram_tiled_wide192(&lf_feats, &lf_res, &weights_lm, 1);
        let (g_mf, c_mf) = contract_256d_gram_tiled_wide192(&mf_feats, &mf_res, &weights_lm, 1);
        let (g_hf, c_hf) = contract_256d_gram_tiled_wide192(&hf_feats, &hf_res, &weights_hf, 1);

        tier_lufs_gram[t].copy_from_slice(&g_lufs_256[..p_len]);
        for ch in 0..2 {
            let [w0, w1, w2] = band_weights_lr[ch];
            for idx in 0..p_len {
                tier_gram_lr[t][ch][idx] = w0 * g_lf[idx] + w1 * g_mf[idx] + w2 * g_hf[idx];
            }
            for d in 0..dim {
                tier_cross_256[t][d][ch] = w0 * c_lf[d][ch] + w1 * c_mf[d][ch] + w2 * c_hf[d][ch];
                tier_tgt_fb_cross[t][d][ch] = w0 * tgt_fb_acc[d][0][ch].to_f64()
                    + w1 * tgt_fb_acc[d][1][ch].to_f64()
                    + w2 * tgt_fb_acc[d][2][ch].to_f64();
                tier_lufs_bb_cross[t][d][ch] = c_bb[d][ch];
            }
            tier_base_res_lr[t][ch] = w0 * band_res_acc[ch][0].to_f64()
                + w1 * band_res_acc[ch][1].to_f64()
                + w2 * band_res_acc[ch][2].to_f64();
            tier_tgt_res_dot_lr[t][ch] = w0 * tgt_res_acc[ch][0].to_f64()
                + w1 * tgt_res_acc[ch][1].to_f64()
                + w2 * tgt_res_acc[ch][2].to_f64();
            tier_lufs_bb_energy_lr[t][ch] = bb_energy_acc[ch].to_f64();
            tier_bb_res_energy_lr[t][ch] = bb_res_e_acc[ch].to_f64();
            for d in 0..dim {
                tier_bb_res_cross_256[t][d][ch] = bb_res_c_acc[d][ch].to_f64();
            }
        }
    }

    BezoutHarvestStats {
        target_band_energy_lr,
        sampled_target_band_energy_lr,
        band_weights_lr,
        tier_gram_lr,
        tier_cross_256,
        tier_base_res_lr,
        tier_tgt_res_dot_lr,
        tier_tgt_fb_cross,
        tier_lufs_gram,
        tier_lufs_bb_energy_lr,
        tier_lufs_bb_cross,
        tier_bb_res_energy_lr,
        tier_bb_res_cross_256,
        post_warmup_frames: post_n,
        warmup_samples: warmup,
    }
}

/// Evaluates exact per-tier (`Lite 64-D, Full 128-D, Pro 192-D, Ultra 256-D`) untrimmed quadratic
/// 3-band `bESR` objective values (`Y_tot * bESR_lin`) for `ch`.
#[must_use]
pub fn eval_4tier_besr_quad_1ch(
    stats: &BezoutHarvestStats,
    ch: usize,
    w_eff_1ch: &[f64; 256],
) -> [f64; 4] {
    let mut out = [0.0_f64; 4];
    for t in 0..4 {
        let dim = (t + 1) * 64;
        let g = &stats.tier_gram_lr[t][ch];
        let mut quad_g = 0.0_f64;
        let mut lin = 0.0_f64;
        for j in 0..dim {
            let wj = w_eff_1ch[j];
            lin += wj * stats.tier_cross_256[t][j][ch];
            let col_base = uplo_u_index(0, j);
            quad_g += g[col_base + j] * wj * wj;
            for (i, &g_ij) in g[col_base..col_base + j].iter().enumerate() {
                quad_g += 2.0 * g_ij * w_eff_1ch[i] * wj;
            }
        }
        out[t] = (stats.tier_base_res_lr[t][ch] - 2.0 * lin + quad_g).max(0.0);
    }
    out
}

#[inline]
fn eval_tier_lufs_energies_1ch(
    stats: &BezoutHarvestStats,
    ch: usize,
    w_1ch: &[f64; 256],
) -> [f64; 4] {
    let mut out = [0.0_f64; 4];
    for t in 0..4 {
        let dim = (t + 1) * 64;
        let g = &stats.tier_lufs_gram[t];
        let mut quad_g = 0.0_f64;
        let mut lin = 0.0_f64;
        for j in 0..dim {
            let wj = w_1ch[j];
            lin += wj * stats.tier_lufs_bb_cross[t][j][ch];
            let col_base = uplo_u_index(0, j);
            quad_g += g[col_base + j] * wj * wj;
            for (i, &g_ij) in g[col_base..col_base + j].iter().enumerate() {
                quad_g += 2.0 * g_ij * w_1ch[i] * wj;
            }
        }
        out[t] = (stats.tier_lufs_bb_energy_lr[t][ch] + 2.0 * lin + quad_g).max(0.0);
    }
    out
}

/// Evaluates exact post-`Stage E` `LUFS`-trimmed 4-tier 3-band `bESR` linear ratios (`worst_ch`)
/// and linear gain trims `g_trim[0..4]`.
#[must_use]
pub fn eval_4tier_post_trim_besr_lin(
    stats: &BezoutHarvestStats,
    w_lr: &[[f64; 2]; 256],
    eval_stereo: bool,
) -> ([f64; 4], [f64; 4]) {
    let n_ch = if eval_stereo { 2 } else { 1 };
    let mut w_ch = [[0.0_f64; 256]; 2];
    for d in 0..256 {
        w_ch[0][d] = w_lr[d][0];
        w_ch[1][d] = if eval_stereo { w_lr[d][1] } else { w_lr[d][0] };
    }
    let e0 = eval_tier_lufs_energies_1ch(stats, 0, &w_ch[0]);
    let e1 = if eval_stereo {
        eval_tier_lufs_energies_1ch(stats, 1, &w_ch[1])
    } else {
        e0
    };
    let post_n = (stats.post_warmup_frames.max(1)) as f64;
    let mut lufs = [-70.0_f32; 4];
    for t in 0..4 {
        let sum_sq = e0[t] + e1[t];
        let mean_sq = (sum_sq / post_n).max(1.0e-14);
        let raw_lufs = (-0.691 + 10.0 * mean_sq.log10()).clamp(-70.0, 24.0) as f32;
        lufs[t] = (raw_lufs * 1_000.0).round() / 1_000.0;
    }
    let ultra_lufs = lufs[3];
    let mut g_trim = [1.0_f64; 4];
    for t in 0..4 {
        let raw_trim = (ultra_lufs - lufs[t]).clamp(-12.0, 12.0);
        g_trim[t] = f64::from(10.0_f32.powf(raw_trim / 20.0));
    }

    let mut worst_lin = [0.0_f64; 4];
    for ch in 0..n_ch {
        let y_tot = (stats.sampled_target_band_energy_lr[ch][0]
            + stats.sampled_target_band_energy_lr[ch][1]
            + stats.sampled_target_band_energy_lr[ch][2])
            .max(1.0e-15);
        let y_w = stats.band_weights_lr[ch][0] * stats.sampled_target_band_energy_lr[ch][0]
            + stats.band_weights_lr[ch][1] * stats.sampled_target_band_energy_lr[ch][1]
            + stats.band_weights_lr[ch][2] * stats.sampled_target_band_energy_lr[ch][2];
        let j_untrimmed = eval_4tier_besr_quad_1ch(stats, ch, &w_ch[ch]);
        for t in 0..4 {
            let dim = (t + 1) * 64;
            let mut fk = 0.0_f64;
            for d in 0..dim {
                fk += stats.tier_tgt_fb_cross[t][d][ch] * w_ch[ch][d];
            }
            let p_tc = stats.tier_tgt_res_dot_lr[t][ch] - fk;
            let g = g_trim[t];
            let om_g = 1.0 - g;
            let j_post =
                (om_g * om_g * y_w + 2.0 * g * om_g * p_tc + g * g * j_untrimmed[t]).max(0.0);
            let ratio = j_post / y_tot;
            worst_lin[t] = if ch == 0 {
                ratio
            } else {
                worst_lin[t].max(ratio)
            };
        }
    }
    (worst_lin, g_trim)
}
