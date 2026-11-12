#![forbid(unsafe_code)]
//! Stage A (`< 5 ms` Anchor-Stem Lifting): Stage 0 Routing/Crossover, Deconvolved Contiguous
//! `64`-Tap Early FIR, Dual-Rate Steiglitz–McBride `32`-Pole Shared Kautz Modal Lifting, Adaptive
//! Stage I/II/IV Physical Backbone Seeding, and Role A `VP-GN` Refinement (`solver::stage_a`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-a-poles]`)
//! - `docs/specs/dsp/op_m_modal_kautz.md`
//! - `docs/specs/dsp/op_n_commuted_cascade.md`

use std::f64::consts::TAU;

use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container_with_cp1};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{STAGE3_KAUTZ_BASES, STAGE3_KAUTZ_BLOCKS_PER_TIER};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::DnamProjManifest;

use super::stage_a_id::{
    extract_stage4_reflection_delays, fit_deconvolved_stage3_stereo,
    identify_steiglitz_mcbride_and_linearity, solve_early_fir_64_stereo,
};

/// Stage III contiguous `64`-tap (`1.33 ms`) Early FIR base offsets (`16 L + 16 R` `f32` per tier).
pub const STAGE3_EARLY_FIR_BASES: [usize; 4] = [4_288, 17_408, 31_552, 48_512];

/// Stage II Commuted Cascade weight base offsets (`[48, 96, 144, 192] × 64 B`, `7,680` `f32` total).
pub const CASCADE_W_BASES: [usize; 4] = [1_024, 11_264, 22_336, 36_224];

/// Number of `64 B` (`16`-float) cascade blocks per tier (`Role A = block 0`, `Role B = blocks 1..K_t`).
pub const CASCADE_BLOCKS_PER_TIER: [usize; 4] = [48, 96, 144, 192];

/// Configuration for Stage A (`< 5 ms` Anchor-Stem Backbone Lifting, `r[train.pipeline.stage-a-poles]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageAConfig {
    /// Preserve an already-populated TORA physical backbone (`true` for ground-truth TORA oracles
    /// or incremental vault edits; `false` to re-lift from `StarCenter` dry/wet).
    pub preserve_existing_backbone: bool,
    /// Number of coordinate Variable-Projection Gauss-Newton (`VP-GN`) Role A steps (`1..=2`).
    pub vp_gn_steps: usize,
}

impl Default for StageAConfig {
    fn default() -> Self {
        Self {
            preserve_existing_backbone: true,
            vp_gn_steps: 2,
        }
    }
}

/// Telemetry emitted by [`lift_stage_a_backbone`] (`r[train.pipeline.stage-a-poles]`).
#[derive(Debug, Clone, PartialEq)]
pub struct StageAReport {
    /// Wall-clock Stage A execution time in milliseconds (`< 5.0 ms`).
    pub elapsed_ms: f32,
    /// True if an existing non-trivial TORA backbone was preserved rather than re-synthesized.
    pub preserved_backbone: bool,
    /// Stage 0 clean-sub / TPT LR4 crossover frequency in Hz (`80..260 Hz`).
    pub crossover_hz: f32,
    /// Energy of the lifted `64`-tap Early FIR impulse response.
    pub early_fir_energy: f64,
    /// Number of lifted shared Kautz stereo pole pairs (`32` across `Lite..Ultra`).
    pub kautz_poles_lifted: usize,
    /// Confirms all `7,616` Role B cascade floats were hard-zeroed (`0.0`) on lifted backbones.
    pub role_b_zeroed: bool,
    /// Optional Phase-2 `[CLAMP-RAIL]` prescription emitted when `PolyADAA` rail occupancy exceeded `95%`.
    pub phase2_rail_warning: Option<String>,
}

#[derive(Debug, Clone)]
struct PhysicalDescriptors {
    crossover_hz: f32,
    hf_growth: f32,
    nl_drive: f32,
    g_loop: f32,
    amp_scale: f32,
    band_q_bias: [f32; 32],
    sm_poles: Vec<(f32, f32)>,
}

#[inline]
fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    if off + 4 <= buf.len() {
        let clean = if val.is_finite() {
            if val == 0.0 { 0.0 } else { val }
        } else {
            0.0
        };
        buf[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
}

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    if off + 4 <= buf.len() {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    } else {
        0.0
    }
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

/// Returns `true` if `arena_64k` already contains a non-trivial Stage II/III physical backbone
/// (as opposed to a zeroed arena or the minimal 4-impulse bootstrap seed).
#[must_use]
pub fn has_populated_physical_backbone(arena_64k: &[u8]) -> bool {
    if arena_64k.len() < ARENA_OFFSET_ULTRA_END as usize {
        return false;
    }
    let has_kautz = STAGE3_KAUTZ_BASES
        .iter()
        .zip(&STAGE3_KAUTZ_BLOCKS_PER_TIER)
        .any(|(&base, &blks)| arena_64k[base..base + blks * 128].iter().any(|&b| b != 0));
    let has_cascade = CASCADE_W_BASES
        .iter()
        .any(|&base| arena_64k[base..base + 64].iter().any(|&b| b != 0));
    has_kautz || has_cascade
}

/// Estimates physical crossover, dual-rate Steiglitz–McBride modal poles, and nonlinearity ratio.
fn estimate_physical_descriptors(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spec_4096: Option<&[u8; 4_096]>,
) -> PhysicalDescriptors {
    let n = dry_lr.len().min(wet_lr.len()).max(1);
    let mut dry_e = 0.0_f64;
    let mut wet_e = 0.0_f64;
    let mut diff_dry = 0.0_f64;
    let mut diff_wet = 0.0_f64;
    let mut lp_dry = 0.0_f64;
    let mut lp2_dry = 0.0_f64;
    let mut low_dry_e = 0.0_f64;
    let mut lp_wet = 0.0_f64;
    let mut lp2_wet = 0.0_f64;
    let mut low_wet_e = 0.0_f64;
    let mut low2_wet_e = 0.0_f64;

    for i in 0..n {
        let x = f64::from(dry_lr[i][0]);
        let y = f64::from(wet_lr[i][0]);
        dry_e += x * x;
        wet_e += y * y;
        if i > 0 {
            let dx = x - f64::from(dry_lr[i - 1][0]);
            let dy = y - f64::from(wet_lr[i - 1][0]);
            diff_dry += dx * dx;
            diff_wet += dy * dy;
        }
        lp_dry = 0.975 * lp_dry + 0.025 * x;
        lp2_dry = 0.975 * lp2_dry + 0.025 * lp_dry;
        low_dry_e += lp2_dry * lp2_dry;
        lp_wet = 0.975 * lp_wet + 0.025 * y;
        lp2_wet = 0.975 * lp2_wet + 0.025 * lp_wet;
        low_wet_e += lp_wet * lp_wet;
        low2_wet_e += lp2_wet * lp2_wet;
    }

    let gain_ratio = wet_e / dry_e.max(1.0e-9);
    let hf_growth = ((diff_wet / diff_dry.max(1.0e-9)) / gain_ratio.max(1.0e-6))
        .sqrt()
        .clamp(0.25, 3.5) as f32;
    let low2_rel_dry = low2_wet_e / (low_dry_e * gain_ratio.max(1.0e-6)).max(1.0e-12);
    let low_ratio = (low_wet_e / wet_e.max(1.0e-9)).clamp(0.02, 0.85) as f32;
    let (nl_drive, sm_poles) = identify_steiglitz_mcbride_and_linearity(dry_lr, wet_lr, hf_growth);
    let crossover_hz = if nl_drive < 0.05 && low2_rel_dry < 0.10 {
        (20.0 + 600.0 * (low2_rel_dry as f32)).clamp(20.0, 80.0)
    } else {
        (110.0 + 140.0 * (1.0 - low_ratio)).clamp(80.0, 260.0)
    };
    let min_amp = if nl_drive < 0.03 { 0.002 } else { 0.15 };
    let amp_scale = if wet_e <= 1.0e-12 {
        0.0
    } else {
        (gain_ratio.sqrt() as f32).clamp(min_amp, 1.0)
    };

    let mut lag_wet = 0.0_f64;
    let mut lag_dry = 0.0_f64;
    if n > 400 {
        let end = n.min(4_000);
        for i in 300..end {
            lag_wet += (f64::from(wet_lr[i][0]) * f64::from(wet_lr[i - 179][0])).abs();
            lag_dry += (f64::from(dry_lr[i][0]) * f64::from(dry_lr[i - 179][0])).abs();
        }
        let inv_cnt = 1.0 / ((end - 300) as f64);
        lag_wet *= inv_cnt;
        lag_dry *= inv_cnt;
    }
    let norm_wet_tail = lag_wet / (wet_e / (n as f64)).max(1.0e-9);
    let norm_dry_tail = lag_dry / (dry_e / (n as f64)).max(1.0e-9);
    let norm_tail = (norm_wet_tail - norm_dry_tail * 1.05).clamp(0.0, 0.85) as f32;
    let g_loop = (0.06 + 0.76 * norm_tail).clamp(0.06, 0.88);

    let mut band_q_bias = [0.0_f32; 32];
    if let Some(spec) = spec_4096 {
        for (m, dst) in band_q_bias.iter_mut().enumerate() {
            let off = (m * 16).min(4_092);
            let raw = read_f32_le(spec, off);
            if raw.is_finite() {
                *dst = raw.abs().clamp(0.0, 0.06);
            }
        }
    }

    PhysicalDescriptors {
        crossover_hz,
        hf_growth,
        nl_drive,
        g_loop,
        amp_scale,
        band_q_bias,
        sm_poles,
    }
}

#[inline]
fn snap_pole_theta_hash_safe(raw_theta: f32) -> f32 {
    let clamped = raw_theta.clamp(0.007, std::f32::consts::PI - 0.025);
    let x100 = clamped * 100.0;
    let xr = x100.round();
    ((xr + (x100 - xr).clamp(-0.30, 0.30)) * 0.01).clamp(0.007, std::f32::consts::PI - 0.025)
}

/// Builds the 32-pole Kautz modal schedule (`[Lite 4, Full 4, Pro 8, Ultra 16]`), anchoring slots
/// onto dual-rate Steiglitz–McBride rational poles (`55 Hz..16 kHz`) while preserving full rank.
fn build_kautz_32_pole_schedule(
    sm_poles: &[(f32, f32)],
    band_q_bias: &[f32; 32],
) -> [(f32, f32); 32] {
    const POLE_PERM_32: [usize; 32] = [
        2, 9, 17, 25, // Lite (4 coarse poles across 80 Hz .. 9 kHz)
        5, 13, 21, 29, // Full (4 mid-spaced poles)
        0, 4, 8, 12, 16, 20, 24, 28, // Pro (8 fine cabinet breakup & port modes)
        1, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, // Ultra (16 dense poles)
    ];
    let f_min_hz = 55.0_f64;
    let f_max_hz = 16_200.0_f64;
    let ratio = f_max_hz / f_min_hz;
    let mut grid = [(0.92_f32, 0.01_f32); 32];
    let mut prev_th = 0.0_f32;
    for g_idx in 0..32 {
        let frac = (g_idx as f64 + 0.35) / 32.0;
        let raw_th = (TAU * (f_min_hz * ratio.powf(frac)) / 48_000.0) as f32;
        let min_th = if g_idx == 0 {
            0.007
        } else if g_idx <= 5 {
            prev_th + 0.0105
        } else {
            prev_th + 0.014
        };
        let snapped = snap_pole_theta_hash_safe(raw_th.max(min_th));
        prev_th = snapped;
        let base_r = 0.895_f32 + 0.075_f32 * (1.0 - (frac as f32) * 0.48);
        let r_low = (base_r + band_q_bias[g_idx]).clamp(0.895, 0.994);
        grid[g_idx] = (r_low, snapped);
    }

    let mut replaced = [false; 32];
    for &(sm_r, sm_th) in sm_poles {
        let snapped_th = snap_pole_theta_hash_safe(sm_th.max(0.007));
        let mut best_slot = None;
        let mut best_dist = f32::INFINITY;
        for g_idx in 0..32 {
            if replaced[g_idx] {
                continue;
            }
            let min_sep = 0.0025_f32;
            let d = (grid[g_idx].1.ln() - snapped_th.ln()).abs();
            if d < best_dist && d < 0.35 {
                let no_collision = (0..32)
                    .all(|other| other == g_idx || (grid[other].1 - snapped_th).abs() >= min_sep);
                if no_collision {
                    best_dist = d;
                    best_slot = Some(g_idx);
                }
            }
        }
        if let Some(slot) = best_slot {
            grid[slot] = (sm_r.clamp(0.85, 0.994), snapped_th);
            replaced[slot] = true;
        }
    }

    let mut out = [(0.90_f32, 0.10_f32); 32];
    for (cursor, &g_idx) in POLE_PERM_32.iter().enumerate() {
        out[cursor] = grid[g_idx];
    }
    out
}

/// Simulates Stage 0..II on `dry_lr[..n_fit]` via `DnamEngine` (`Tier::Full`) to extract the exact
/// `(s2_lr, clean_sub_lr)` trajectories feeding Stage III.
fn simulate_stage2_and_sub_trajectories(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    n_fit: usize,
) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let mut probe_arena = arena_64k.to_vec();
    let _ = super::stages::sync_clamp_descriptors_to_arena(&manifest.base, &mut probe_arena);
    let mut hdr = manifest.base.clone();
    if let Ok(dnam_bytes) =
        build_dnam_container_with_cp1(&mut hdr, &probe_arena, &probe_arena, false, &[])
        && let Ok(mut eng) =
            DnamEngine::<DefaultMathBackend>::from_bytes(&dnam_bytes, Tier::Full, 48_000)
    {
        let star_ctrls = super::star_center_control_values(manifest);
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let mut s2_lr = Vec::with_capacity(n_fit);
        let mut sub_lr = Vec::with_capacity(n_fit);
        let mut fb = [0.0_f32; 256];
        let _ftz = FtzDazGuard::new();
        for &[dl, dr] in &dry_lr[..n_fit] {
            let _ = eng.step_frame_with_feature_bus(dl, dr, &mut fb);
            sub_lr.push([fb[60], fb[61]]);
            s2_lr.push([fb[58] - fb[76], fb[59] - fb[77]]);
        }
        return (s2_lr, sub_lr);
    }
    (dry_lr[..n_fit].to_vec(), vec![[0.0_f32; 2]; n_fit])
}

fn cancel_discrete_early_reflections(
    s3_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    lags_l: [usize; 4],
    lags_r: [usize; 4],
) -> Vec<[f32; 2]> {
    let n = s3_lr.len().min(wet_lr.len());
    let mut out = wet_lr[..n].to_vec();
    if n < 512 {
        return out;
    }
    for (ch, lags) in [(0usize, lags_l), (1usize, lags_r)] {
        let mut uniq: Vec<usize> = Vec::with_capacity(4);
        for &d in &lags {
            if (24..n / 2).contains(&d) && uniq.iter().all(|&u| u.abs_diff(d) >= 8) {
                uniq.push(d);
            }
        }
        let m = uniq.len();
        if m == 0 {
            continue;
        }
        let mut g = vec![0.0_f64; crate::solver::uplo_u_packed_len(m)];
        let mut c = vec![[0.0_f64; 2]; m];
        let max_d = *uniq.iter().max().unwrap_or(&24);
        for i in (max_d + 16)..n {
            let r_hp = (f64::from(wet_lr[i][ch]) - f64::from(s3_lr[i][ch]))
                - 0.94 * (f64::from(wet_lr[i - 1][ch]) - f64::from(s3_lr[i - 1][ch]));
            for col in 0..m {
                let dc = uniq[col];
                let sc = f64::from(s3_lr[i - dc][ch]) - 0.94 * f64::from(s3_lr[i - 1 - dc][ch]);
                let base = crate::solver::uplo_u_index(0, col);
                for row in 0..=col {
                    let dr = uniq[row];
                    let sr = f64::from(s3_lr[i - dr][ch]) - 0.94 * f64::from(s3_lr[i - 1 - dr][ch]);
                    g[base + row] += sr * sc;
                }
                c[col][0] += sc * r_hp;
            }
        }
        let tr: f64 = (0..m).map(|k| g[crate::solver::uplo_u_index(k, k)]).sum();
        if let Ok(sol) = crate::solver::solve_uplo_u_cholesky_2ch(&g, &c, (1.0e-3 * tr).max(1.0e-9))
        {
            for (k, &d) in uniq.iter().enumerate() {
                let alpha = (sol[k][0] as f32).clamp(-0.65, 0.65);
                if alpha.abs() >= 0.03 {
                    for i in d..n {
                        out[i][ch] -= alpha * s3_lr[i - d][ch];
                    }
                }
            }
        }
    }
    out
}

/// Zeroes all `7,616` Role B cascade floats (`b = 1..K_t-1` across `CASCADE_W_BASES[0..4]`) so
/// `cached_role_b_slices` evaluates to exact `1.0` (`0 ULP`) before buyer gauge watermarking.
pub fn zero_all_role_b_cascade_weights(arena_64k: &mut [u8]) {
    for (&cw_base, &k_t) in CASCADE_W_BASES.iter().zip(&CASCADE_BLOCKS_PER_TIER) {
        let start = cw_base + 64;
        let end = (cw_base + k_t * 64).min(arena_64k.len());
        if start < end {
            arena_64k[start..end].fill(0);
        }
    }
}

/// Executes Stage A (`< 5 ms` Anchor-Stem Lifting, `r[train.pipeline.stage-a-poles]`):
/// populates Stage 0 routing/crossover, adaptive Stage I/II, deconvolved `64`-tap Early FIR,
/// dual-rate Steiglitz–McBride `32`-pole Kautz bank, and stereo Stage IV reflection delays.
pub fn lift_stage_a_backbone(
    manifest: &DnamProjManifest,
    arena_64k: &mut [u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spec_4096: Option<&[u8; 4_096]>,
    config: StageAConfig,
) -> StageAReport {
    let timer = now_ms_token();
    debug_assert!(arena_64k.len() >= ARENA_OFFSET_ULTRA_END as usize);

    if config.preserve_existing_backbone && has_populated_physical_backbone(arena_64k) {
        let xover = read_f32_le(arena_64k, 128);
        return StageAReport {
            elapsed_ms: elapsed_ms_since(timer),
            preserved_backbone: true,
            crossover_hz: if xover > 20.0 { xover } else { 200.0 },
            early_fir_energy: 1.0,
            kautz_poles_lifted: 32,
            role_b_zeroed: false,
            phase2_rail_warning: None,
        };
    }

    let n_common = dry_lr.len().min(wet_lr.len());
    let (dry_lr, wet_lr) = if n_common > 12_000 {
        let win = 12_000usize;
        let mut best_off = 0usize;
        let mut best_score = -1.0_f64;
        for off in [0, (n_common - win) / 2, n_common - win] {
            let (mut we, mut d_hf) = (0.0_f64, 0.0_f64);
            for i in (off + 1)..(off + win) {
                let y = 0.5 * (f64::from(wet_lr[i][0]) + f64::from(wet_lr[i][1]));
                let dx = f64::from(dry_lr[i][0] - dry_lr[i - 1][0]);
                we += y * y;
                d_hf += dx * dx;
            }
            let score = we.sqrt() * d_hf.sqrt();
            if score > best_score {
                best_score = score;
                best_off = off;
            }
        }
        (
            &dry_lr[best_off..best_off + win],
            &wet_lr[best_off..best_off + win],
        )
    } else {
        (&dry_lr[..n_common], &wet_lr[..n_common])
    };

    let desc = estimate_physical_descriptors(dry_lr, wet_lr, spec_4096);
    let nl = desc.nl_drive;
    let lti_w = (1.0 - 18.0 * nl).clamp(0.0, 1.0);
    let nl_w = 1.0 - lti_w;

    // 1. Stage 0 Router (0..132): canonical S_router isolation (S_ret->send == 0) + crossover
    arena_64k[0..132].fill(0);
    write_f32_le(arena_64k, 0, 1.0);
    write_f32_le(arena_64k, 20, 1.0);
    write_f32_le(arena_64k, 32, 1.0);
    write_f32_le(arena_64k, 40, 1.0);
    write_f32_le(arena_64k, 52, 1.0);
    write_f32_le(arena_64k, 60, 1.0);
    write_f32_le(arena_64k, 116, 0.0); // mix_dry_gain
    write_f32_le(arena_64k, 120, 1.0); // mix_wet_gain
    write_f32_le(arena_64k, 124, 0.9993); // r_ac
    write_f32_le(arena_64k, 128, desc.crossover_hz);

    // 2. Stage I Pre-Dynamics (256..360) + Arena 1 Parallel Delta Op-M (10,752..10,832)
    let stage1_r: [f32; 4] = [0.040 * lti_w + 0.580 * nl_w, 0.935, 0.905, 0.875];
    let stage1_theta: [f32; 4] = [0.350, 0.155, 0.460, 1.080];
    let stage1_cre: [f32; 4] = [
        0.90 * lti_w + 0.62 * nl_w,
        0.025 * lti_w + 0.22 * nl_w,
        -0.020 * lti_w - 0.14 * nl_w,
        0.015 * lti_w + 0.10 * nl_w,
    ];
    let stage1_cim: [f32; 4] = [
        0.010 * lti_w + 0.08 * nl_w,
        -0.020 * lti_w - 0.16 * nl_w,
        0.015 * lti_w + 0.11 * nl_w,
        -0.010 * lti_w - 0.07 * nl_w,
    ];
    for lane in 0..4 {
        write_f32_le(arena_64k, 256 + lane * 4, stage1_r[lane]);
        write_f32_le(arena_64k, 272 + lane * 4, 0.0);
        write_f32_le(arena_64k, 288 + lane * 4, stage1_theta[lane]);
        write_f32_le(arena_64k, 304 + lane * 4, stage1_cre[lane]);
        write_f32_le(arena_64k, 320 + lane * 4, stage1_cim[lane]);
    }
    write_f32_le(arena_64k, 352, 0.80 + 0.20 * nl);
    let topo_mode: u32 = u32::from(manifest.base.io_config.inputs >= 3);
    arena_64k[356..360].copy_from_slice(&topo_mode.to_le_bytes());

    let delta_r: [f32; 4] = [0.968, 0.948, 0.922, 0.895];
    let delta_theta: [f32; 4] = [0.095, 0.285, 0.640, 1.280];
    for lane in 0..4 {
        write_f32_le(arena_64k, 10_752 + lane * 4, delta_r[lane]);
        write_f32_le(arena_64k, 10_768 + lane * 4, 0.0);
        write_f32_le(arena_64k, 10_784 + lane * 4, delta_theta[lane]);
        write_f32_le(arena_64k, 10_800 + lane * 4, 0.35);
        write_f32_le(arena_64k, 10_816 + lane * 4, -0.18);
    }

    // 3. Stage II Adaptive Bussgang Saturation & Inter-Stage Poles (4,096..4,288)
    arena_64k[4_096..4_288].fill(0);
    write_f32_le(arena_64k, 4_096, 0.9896);
    write_f32_le(arena_64k, 4_100, 0.9490);
    write_f32_le(arena_64k, 4_104, 0.8546);
    write_f32_le(arena_64k, 4_108, 0.9992);
    let v1a_vals = [
        lti_w * 0.55 + nl_w * (0.025 + 1.125 * nl),
        lti_w * 0.55 + nl_w * (0.024 + 1.096 * nl),
        0.08 * nl,
        -0.06 * nl,
        0.24 + 0.64 * nl,
        0.24 + 0.64 * nl,
    ];
    let v1b_vals = [
        lti_w * 1.25 + nl_w * (0.025 + 1.425 * nl),
        lti_w * 1.25 + nl_w * (0.024 + 1.376 * nl),
        -0.10 * nl,
        0.09 * nl,
        0.42 + 0.40 * nl,
        0.42 + 0.40 * nl,
    ];
    for (off, vals) in [(4_112usize, v1a_vals), (4_136usize, v1b_vals)] {
        for (k, v) in vals.into_iter().enumerate() {
            write_f32_le(arena_64k, off + k * 4, v);
        }
    }
    write_f32_le(arena_64k, 4_160, 0.9920);
    write_f32_le(arena_64k, 4_164, 0.6200);
    write_f32_le(arena_64k, 4_168, 0.2000 * nl);
    write_f32_le(arena_64k, 4_172, 0.9995);
    write_f32_le(arena_64k, 4_176, 0.8500);
    write_f32_le(arena_64k, 4_180, 0.1200);
    write_f32_le(arena_64k, 4_184, 0.0200 * lti_w + 0.2600 * nl_w);
    write_f32_le(arena_64k, 4_188, 0.0800 * nl);
    write_f32_le(arena_64k, 4_192, 0.9880);
    write_f32_le(arena_64k, 4_196, 0.5400);
    write_f32_le(arena_64k, 4_200, 0.1800 * nl);
    write_f32_le(arena_64k, 4_204, 0.9992);
    write_f32_le(arena_64k, 4_208, 0.8500);
    write_f32_le(arena_64k, 4_212, 0.2100);
    write_f32_le(arena_64k, 4_216, 0.0200 * lti_w + 0.2200 * nl_w);
    write_f32_le(arena_64k, 4_220, 0.0600 * nl);
    write_f32_le(arena_64k, 4_272, 0.10 + 0.90 * nl);
    write_f32_le(arena_64k, 4_276, 1.0);
    write_f32_le(arena_64k, 4_280, 0.980);
    write_f32_le(arena_64k, 4_284, 0.999);

    zero_all_role_b_cascade_weights(arena_64k);
    let vp_steps = config.vp_gn_steps.clamp(1, 2);
    let mut drive_scale = (0.04 + 0.75 * nl * desc.hf_growth.clamp(0.5, 1.6)).clamp(0.04, 0.95);
    for _ in 0..vp_steps {
        let target_ratio = nl * (desc.hf_growth - 1.0).clamp(-0.3, 1.2);
        drive_scale = (drive_scale + 0.12 * target_ratio).clamp(0.04, 1.10);
    }
    drive_scale *= desc.amp_scale;
    for (t, &cw_base) in CASCADE_W_BASES.iter().enumerate() {
        let tier_scale = drive_scale * (1.0 - 0.12 * (t as f32));
        for k in 0..16 {
            let sign = if (k + t) % 2 == 0 { 1.0_f32 } else { -0.75_f32 };
            let harmonic_w = tier_scale * sign * (0.28 + 0.04 * ((k % 5) as f32));
            write_f32_le(arena_64k, cw_base + k * 4, harmonic_w);
        }
    }

    // 7. 3 kHz OU analog drift & micro-dither (7,496..7,524) + default trust_radius_mh = 10.0
    write_f32_le(arena_64k, 7_496, 0.0012);
    write_f32_le(arena_64k, 7_500, 0.9950);
    write_f32_le(arena_64k, 7_516, 1.0e-7);
    write_f32_le(arena_64k, 7_520, 10.0);
    write_f32_le(arena_64k, 10_560, 10.0);
    write_f32_le(arena_64k, 10_688, 0.0625);
    zero_arena_b_head_dc_bias(arena_64k);

    // Enforce Phase-2 [CLAMP-RAIL] on Stage II V1A BEFORE simulating Stage 0..II for Stage III fit
    let (occ, safe_scale, locked) =
        crate::audit::audit_phase2_polyadaa_rail_lockup(arena_64k, 3.0, 0.25);
    let phase2_rail_warning = if locked && arena_64k.len() >= 4_136 {
        let cur_bl = read_f32_le(arena_64k, 4_112).max(0.1);
        let cur_br = read_f32_le(arena_64k, 4_116);
        write_f32_le(arena_64k, 4_112, (cur_bl * safe_scale).clamp(0.1, 4.0));
        if cur_br > 0.0 {
            write_f32_le(arena_64k, 4_116, (cur_br * safe_scale).clamp(0.1, 4.0));
        }
        Some(format!(
            "[CLAMP-RAIL] PolyADAA corner rail occupancy {:.1}% > 95.0%; scaled V1A beta by {:.3}",
            occ * 100.0,
            safe_scale
        ))
    } else {
        None
    };

    // 4 & 5. Adaptive Stage III: Deconvolved Stereo Fit on LTI/Clean vs. Broadband Basis on Driven
    let pole_schedule = build_kautz_32_pole_schedule(&desc.sm_poles, &desc.band_q_bias);
    let mono_only = manifest.base.io_config.output_mode == "mono";
    let r_chan_scale = if mono_only { 1.0_f32 } else { 0.96_f32 };
    let n_fit = dry_lr.len().min(wet_lr.len()).min(12_000);

    let (raw_fir_l, raw_fir_r) = solve_early_fir_64_stereo(dry_lr, wet_lr);
    let (mut fir_l, mut fir_r, kautz_c_l, kautz_c_r, s3_ref_lr) = if desc.amp_scale == 0.0 {
        (
            [0.0_f32; 64],
            [0.0_f32; 64],
            [(0.0_f32, 0.0_f32); 32],
            [(0.0_f32, 0.0_f32); 32],
            vec![[0.0_f32; 2]; n_fit],
        )
    } else if lti_w > 0.01 {
        let (s2_lr, clean_sub_lr) =
            simulate_stage2_and_sub_trajectories(manifest, arena_64k, dry_lr, n_fit);
        let s3_fit0 = fit_deconvolved_stage3_stereo(
            &s2_lr,
            &clean_sub_lr,
            &wet_lr[..n_fit],
            &pole_schedule,
            mono_only,
            r_chan_scale,
        );
        let (ld0, fd0, _, _, _) =
            extract_stage4_reflection_delays(&s3_fit0.s3_lr, &wet_lr[..n_fit]);
        let wet_cab_lr = cancel_discrete_early_reflections(
            &s3_fit0.s3_lr,
            &wet_lr[..n_fit],
            [
                ld0[0] as usize,
                ld0[2] as usize,
                fd0[0] as usize,
                fd0[2] as usize,
            ],
            [
                ld0[1] as usize,
                ld0[3] as usize,
                fd0[1] as usize,
                fd0[3] as usize,
            ],
        );
        let s3_fit = fit_deconvolved_stage3_stereo(
            &s2_lr,
            &clean_sub_lr,
            &wet_cab_lr,
            &pole_schedule,
            mono_only,
            r_chan_scale,
        );
        let mut bl = [0.0_f32; 64];
        let mut br = [0.0_f32; 64];
        for t in 0..4 {
            let nl_seg_gain = if t == 0 { 0.15_f32 } else { 0.65_f32 };
            for i in 0..16 {
                let idx = t * 16 + i;
                bl[idx] = lti_w * s3_fit.fir_l[idx] + (1.0 - lti_w) * nl_seg_gain * raw_fir_l[idx];
                br[idx] = lti_w * s3_fit.fir_r[idx] + (1.0 - lti_w) * nl_seg_gain * raw_fir_r[idx];
            }
        }
        let mut cl = [(0.0_f32, 0.0_f32); 32];
        let mut cr = [(0.0_f32, 0.0_f32); 32];
        for m in 0..32 {
            let base_c = if m < 4 { 0.30_f32 } else { 0.24_f32 };
            let b_g = (1.0_f64 - f64::from(pole_schedule[m].0).powi(2))
                .max(1.0e-6)
                .sqrt() as f32;
            let c_sc = (base_c * b_g).clamp(0.055, 0.125);
            let ph = 0.37_f32 * ((m + 1) as f32);
            let nl_cl = (c_sc * ph.cos(), c_sc * ph.sin());
            cl[m] = (
                lti_w * s3_fit.kautz_c_l[m].0 + (1.0 - lti_w) * nl_cl.0,
                lti_w * s3_fit.kautz_c_l[m].1 + (1.0 - lti_w) * nl_cl.1,
            );
            cr[m] = (
                lti_w * s3_fit.kautz_c_r[m].0 + (1.0 - lti_w) * nl_cl.0 * r_chan_scale,
                lti_w * s3_fit.kautz_c_r[m].1 + (1.0 - lti_w) * nl_cl.1 * r_chan_scale,
            );
        }
        (bl, br, cl, cr, s3_fit.s3_lr)
    } else {
        let mut bl = [0.0_f32; 64];
        let mut br = [0.0_f32; 64];
        for t in 0..4 {
            let nl_seg_gain = if t == 0 { 0.15_f32 } else { 0.65_f32 };
            for i in 0..16 {
                let idx = t * 16 + i;
                bl[idx] = nl_seg_gain * raw_fir_l[idx];
                br[idx] = nl_seg_gain * raw_fir_r[idx];
            }
        }
        let mut cl = [(0.0_f32, 0.0_f32); 32];
        let mut cr = [(0.0_f32, 0.0_f32); 32];
        for m in 0..32 {
            let base_c = if m < 4 { 0.30_f32 } else { 0.24_f32 };
            let b_g = (1.0_f64 - f64::from(pole_schedule[m].0).powi(2))
                .max(1.0e-6)
                .sqrt() as f32;
            let c_sc = (base_c * b_g).clamp(0.055, 0.125);
            let ph = 0.37_f32 * ((m + 1) as f32);
            cl[m] = (c_sc * ph.cos(), c_sc * ph.sin());
            cr[m] = (cl[m].0 * r_chan_scale, cl[m].1 * r_chan_scale);
        }
        let mut pred = vec![[0.0_f32; 2]; n_fit];
        for i in 0..n_fit {
            let mut dl = 0.0_f32;
            let mut dr = 0.0_f32;
            for k in 0..=i.min(63) {
                dl += raw_fir_l[k] * dry_lr[i - k][0];
                dr += raw_fir_r[k] * dry_lr[i - k][1];
            }
            pred[i] = [dl, dr];
        }
        (bl, br, cl, cr, pred)
    };

    let mut fir_energy: f64 = fir_l
        .iter()
        .chain(&fir_r)
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    if fir_energy < 1.0e-6 && desc.amp_scale > 0.0 {
        fir_l[0] = 0.20 * desc.amp_scale;
        fir_r[0] = 0.20 * desc.amp_scale;
        fir_energy = f64::from(fir_l[0]) * f64::from(fir_l[0]) * 2.0;
    }
    for (t, &f_base) in STAGE3_EARLY_FIR_BASES.iter().enumerate() {
        let seg_slice_l = &fir_l[t * 16..(t + 1) * 16];
        let seg_slice_r = &fir_r[t * 16..(t + 1) * 16];
        let seg_e: f64 = seg_slice_l
            .iter()
            .chain(seg_slice_r)
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        for i in 0..16 {
            let fallback_seed = if t > 0 && seg_e < 1.0e-8 && i == (t * 3 + 1) % 16 {
                0.015_f32 * desc.amp_scale
            } else {
                0.0
            };
            let tl = (seg_slice_l[i] + fallback_seed).clamp(-2.0, 2.0);
            let tr = (seg_slice_r[i] + fallback_seed).clamp(-2.0, 2.0);
            write_f32_le(arena_64k, f_base + i * 4, tl);
            write_f32_le(arena_64k, f_base + 64 + i * 4, tr);
        }
    }

    let mut pole_cursor = 0usize;
    for (&k_base, &n_blocks) in STAGE3_KAUTZ_BASES.iter().zip(&STAGE3_KAUTZ_BLOCKS_PER_TIER) {
        for blk in 0..n_blocks {
            let blk_off = k_base + blk * 128;
            for lane in 0..4 {
                let (r_low, theta) = pole_schedule[pole_cursor];
                let b_gamma = (1.0_f64 - f64::from(r_low) * f64::from(r_low))
                    .max(1.0e-6)
                    .sqrt() as f32;
                let (cre_l, cim_l) = kautz_c_l[pole_cursor];
                let (cre_r, cim_r) = kautz_c_r[pole_cursor];

                write_f32_le(arena_64k, blk_off + lane * 4, r_low);
                write_f32_le(arena_64k, blk_off + (4 + lane) * 4, 0.0);
                write_f32_le(arena_64k, blk_off + (8 + lane) * 4, theta);
                write_f32_le(arena_64k, blk_off + (12 + lane) * 4, b_gamma);
                write_f32_le(arena_64k, blk_off + (16 + lane) * 4, cre_l);
                write_f32_le(arena_64k, blk_off + (20 + lane) * 4, cim_l);
                write_f32_le(arena_64k, blk_off + (24 + lane) * 4, cre_r);
                write_f32_le(arena_64k, blk_off + (28 + lane) * 4, cim_r);
                pole_cursor += 1;
            }
        }
    }

    // 6. Stage IV Channel-Aligned Spatial Tank Delays & Reverberant Loop Gain
    let (lite_delays, full_delays, pro_delays, ultra_delays, refl_g_loop) =
        extract_stage4_reflection_delays(&s3_ref_lr, &wet_lr[..n_fit]);
    let eff_g_loop = lti_w * refl_g_loop + (1.0 - lti_w) * desc.g_loop;
    let lite_allpass: [f32; 4] = [0.52, -0.47, 0.43, -0.39];
    for l in 0..4 {
        write_f32_le(arena_64k, 4_672 + l * 4, lite_delays[l]);
        write_f32_le(arena_64k, 4_700 + l * 4, lite_allpass[l]);
    }
    write_f32_le(arena_64k, 4_688, eff_g_loop.clamp(0.05, 0.9990));
    write_f32_le(arena_64k, 4_692, 1.0); // sigma_knee
    write_f32_le(arena_64k, 4_696, 0.36); // psi

    for (l, &d) in full_delays.iter().enumerate() {
        write_f32_le(arena_64k, 17_792 + l * 4, d);
    }
    for (l, &d) in pro_delays.iter().enumerate() {
        write_f32_le(arena_64k, 34_496 + l * 4, d);
    }
    for (k, &d) in ultra_delays.iter().enumerate() {
        write_f32_le(arena_64k, 48_896 + k * 4, d);
    }

    StageAReport {
        elapsed_ms: elapsed_ms_since(timer),
        preserved_backbone: false,
        crossover_hz: desc.crossover_hz,
        early_fir_energy: fir_energy,
        kautz_poles_lifted: pole_cursor,
        role_b_zeroed: true,
        phase2_rail_warning,
    }
}
