#![forbid(unsafe_code)]
//! Canonical Stage Decomposition, Step-0 Variance Gating & 3-Phase Control-Space Audit Helpers.

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::control::{ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::{ControlCalibrationRole, DnamError, DnamJsonHeader};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::DnamProjManifest;

use super::stage_b_c::{HEAD_BASES, ULTRA_MODES_5_TO_8_BASE, compute_role_b_multipliers};
use super::wide_gram::{Wide192Accum, contract_256d_gram_tiled_wide192};
use super::{uplo_u_index, uplo_u_packed_len};

/// Stability and clamping bounds enforced across Sweep 1 and Sweep 2 (`ref[train.audit.phase1.clamp-stability]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditBounds {
    /// Maximum discrete-time pole radius $r_c \le 0.9995$.
    pub max_pole_radius: f64,
    /// Maximum Stage IV linear feedback gain $g_{\mathrm{lin}} \le 0.9990$.
    pub max_tank_feedback_gain: f64,
    /// Maximum Schroeder all-pass coefficient $|a_{\mathrm{diff}}| \le 0.75$.
    pub max_allpass_coeff: f64,
    /// Minimum causal loop delay $\tau_m \ge 4.0$ samples.
    pub min_causal_delay_samples: f64,
}

impl Default for AuditBounds {
    fn default() -> Self {
        Self {
            max_pole_radius: 0.9995,
            max_tank_feedback_gain: 0.9990,
            max_allpass_coeff: 0.75,
            min_causal_delay_samples: 4.0,
        }
    }
}

impl AuditBounds {
    /// Clamps a candidate modal pole radius, tank feedback gain, all-pass coefficient, and loop delay
    /// into the certified stability envelope (`ref[train.audit.phase1.clamp-stability]`).
    #[must_use]
    pub fn clamp_parameters(
        &self,
        pole_radius: f64,
        tank_gain: f64,
        allpass_coeff: f64,
        delay_samples: f64,
    ) -> (f64, f64, f64, f64) {
        let max_r = if self.max_pole_radius.is_finite() {
            self.max_pole_radius.clamp(0.0, 0.9995)
        } else {
            0.9995
        };
        let max_g = if self.max_tank_feedback_gain.is_finite() {
            self.max_tank_feedback_gain.clamp(0.0, 0.9990)
        } else {
            0.9990
        };
        let max_a = if self.max_allpass_coeff.is_finite() {
            self.max_allpass_coeff.abs().min(0.75)
        } else {
            0.75
        };
        let min_d = if self.min_causal_delay_samples.is_finite() {
            self.min_causal_delay_samples.max(4.0)
        } else {
            4.0
        };
        let r = if pole_radius.is_finite() {
            pole_radius.clamp(0.0, max_r)
        } else {
            0.0
        };
        let g = if tank_gain.is_finite() {
            tank_gain.clamp(0.0, max_g)
        } else {
            0.0
        };
        let a = if allpass_coeff.is_finite() {
            allpass_coeff.clamp(-max_a, max_a)
        } else {
            0.0
        };
        let d = if delay_samples.is_finite() {
            delay_samples.max(min_d)
        } else {
            min_d
        };
        (r, g, a, d)
    }
}

/// Evaluates the Step-0 variance gate for control parameter `p` (`ref[dsp.ctrl.step0.variance-gating]`).
/// Returns `true` when the control is active across stems and `false` when it must be locked to $K_p = 0$.
#[must_use]
pub fn is_control_variance_active(observed_values: &[f64], min_spread: f64) -> bool {
    if observed_values.len() < 2 || !min_spread.is_finite() {
        return false;
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in observed_values {
        if !v.is_finite() {
            return false;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let spread = 0.5 * max_v - 0.5 * min_v;
    spread > 0.5 * min_spread.max(1.0e-9)
}

/// Smooth periodic cosine surrogate for Family-5 bitcrusher quantization step size $\Delta$
/// during gradient/coordinate refinement (`ref[dsp.op-n.bitcrusher.surrogate]`).
#[must_use]
pub fn bitcrusher_smooth_surrogate(x: f64, step_delta: f64, temp_tau: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let delta = if step_delta.is_finite() {
        step_delta.max(1.0e-6)
    } else {
        1.0e-6
    };
    let tau = if temp_tau.is_finite() {
        temp_tau.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q_hard = (x / delta).round() * delta;
    let q_smooth =
        x - (delta / (2.0 * std::f64::consts::PI)) * (2.0 * std::f64::consts::PI * x / delta).sin();
    (1.0 - tau) * q_hard + tau * q_smooth
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

/// Encodes `cp1_slots` into `Arena 0 [7,524..7,552, 7,680..8,512]` and `Arena 3 [61,312..63,424]`
/// when `1 <= cp1_slots.len() <= 8` and within the `832 B` / `2,112 B` inline budgets, or zeroes
/// `7,524..7,552` so Region-4 `CP1` handles arbitrary $P$ (`P = 16, 31, 64+`).
pub fn write_inline_cp1_if_fits(arena_64k: &mut [u8], cp1_slots: &[Cp1ContributorSlot]) {
    if arena_64k.len() >= 8_512 {
        arena_64k[7_524..7_552].fill(0);
        arena_64k[7_680..8_512].fill(0);
    }
    if arena_64k.len() >= 63_424 {
        arena_64k[61_312..63_424].fill(0);
    }
    if cp1_slots.is_empty() || cp1_slots.len() > 8 || arena_64k.len() < 8_512 {
        return;
    }
    if cp1_slots
        .iter()
        .any(|s| s.control_idx > 255 || !(1..=3).contains(&s.lanes))
    {
        return;
    }
    let total_base_bytes: usize = cp1_slots
        .iter()
        .map(|s| 4 * (9 * usize::from(s.lanes) + 4))
        .sum();
    let total_ultra_bytes: usize = cp1_slots
        .iter()
        .map(|s| 32 * usize::from(s.lanes) + 16)
        .sum();
    if total_base_bytes > 832 || total_ultra_bytes > 2_112 {
        return;
    }

    let count = cp1_slots.len() as u16;
    arena_64k[7_524..7_526].copy_from_slice(&count.to_le_bytes());
    arena_64k[7_526] = 0;
    arena_64k[7_527] = 0;

    let mut b_cursor = 7_680usize;
    let mut u_cursor = 61_312usize;
    let has_ultra = arena_64k.len() >= 63_424;

    for (c, slot) in cp1_slots.iter().enumerate() {
        let dir_off = 7_528 + c * 3;
        arena_64k[dir_off] = slot.control_idx as u8;
        arena_64k[dir_off + 1] = ((slot.ctrl_kind & 0x0F) << 4) | (slot.lanes & 0x0F);
        arena_64k[dir_off + 2] = if slot.ctrl_kind == 1 {
            slot.option_idx
        } else if slot.ctrl_kind == 2 {
            slot.partner_control_id as u8
        } else {
            0
        };

        let l_cnt = usize::from(slot.lanes);
        for row in slot.main_base.iter().take(l_cnt) {
            for &val in row {
                write_f32_le(arena_64k, b_cursor, val);
                b_cursor += 4;
            }
        }
        for &val in slot.emb.iter().take(l_cnt) {
            write_f32_le(arena_64k, b_cursor, val);
            b_cursor += 4;
        }
        for &val in &slot.v_base {
            write_f32_le(arena_64k, b_cursor, val);
            b_cursor += 4;
        }
        if has_ultra {
            for row in slot.main_ultra.iter().take(l_cnt) {
                for &val in row {
                    write_f32_le(arena_64k, u_cursor, val);
                    u_cursor += 4;
                }
            }
            for &val in &slot.v_ultra {
                write_f32_le(arena_64k, u_cursor, val);
                u_cursor += 4;
            }
        }
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

/// Compiles `header.controls_schema` clamp descriptors and writes the first `8` binary
/// descriptors into `arena_64k[8_512..9_024]` so all solver stages match exported `WGHT`.
pub(crate) fn sync_clamp_descriptors_to_arena(
    header: &DnamJsonHeader,
    arena_64k: &mut [u8],
) -> Result<Vec<ControlClampDescriptor>, DnamError> {
    let descriptors = header.compile_clamp_descriptors()?;
    if arena_64k.len() >= 9_024 {
        arena_64k[8_512..9_024].fill(0);
        for (idx, desc) in descriptors.iter().take(8).enumerate() {
            let off = 8_512 + idx * 64;
            arena_64k[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }
    Ok(descriptors)
}

/// Evaluates `Mode 0` (`StarCenter`) predictions directly from `StageBHarvest`'s `features` and
/// `backbone_out_lr` when standard unit-wet / non-WDW routing applies (`0` redundant engine passes).
pub(crate) fn compute_mode0_pred_from_backbone_harvest(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    features: &[[f32; 256]],
    backbone_out_lr: &[[f32; 2]],
) -> Option<Vec<[f32; 2]>> {
    if features.len() != backbone_out_lr.len() || arena_64k.len() < 60_160 {
        return None;
    }
    if manifest.base.io_config.output_mode == "wdw" || (arena_64k[112] & 1) != 0 {
        return None;
    }
    if read_f32_le(arena_64k, 9_024) != 0.0 || read_f32_le(arena_64k, 9_028) != 0.0 {
        return None;
    }
    if manifest.base.controls_schema.iter().any(|c| {
        c.calibration.as_ref().is_some_and(|cal| {
            matches!(
                cal.role,
                ControlCalibrationRole::Mix
                    | ControlCalibrationRole::StereoMode
                    | ControlCalibrationRole::StereoWidth
                    | ControlCalibrationRole::FreezeLatch
            )
        })
    }) {
        return None;
    }
    let dry_g = read_f32_le(arena_64k, 116);
    let wet_g = read_f32_le(arena_64k, 120);
    if dry_g.abs() > 1.0e-7 || (wet_g - 1.0).abs() > 1.0e-6 {
        return None;
    }
    let ultra_trim = read_f32_le(arena_64k, 10_624 + 3 * 4);
    if (0.0625..=16.0).contains(&ultra_trim) && (ultra_trim - 1.0).abs() > 1.0e-6 {
        return None;
    }
    if backbone_out_lr.iter().any(|&[bl, br]| {
        !bl.is_finite() || !br.is_finite() || bl.abs() >= 5.0e8 || br.abs() >= 5.0e8
    }) {
        return None;
    }

    let role_b = compute_role_b_multipliers(arena_64k);
    let mut w0_eff = [[0.0_f32; 2]; 256];
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        for i in 0..64 {
            let idx = t * 64 + i;
            let wl = read_f32_le(arena_64k, h_base + i * 4);
            let wr = read_f32_le(arena_64k, h_base + 256 + i * 4);
            w0_eff[idx][0] = wl * (role_b[idx][0] as f32);
            w0_eff[idx][1] = wr * (role_b[idx][1] as f32);
        }
    }
    let mono_only = manifest.base.io_config.output_mode == "mono";
    let mut out = Vec::with_capacity(features.len());
    for (fb, &[bl, br]) in features.iter().zip(backbone_out_lr.iter()) {
        // Match `eval_tier_feature_bus_head_dot` 4xVec4 SIMD reduction tree order
        let mut acc_l = [[0.0_f32; 4]; 4];
        let mut acc_r = [[0.0_f32; 4]; 4];
        for g4 in 0..16 {
            let base = g4 * 16;
            for b in 0..4 {
                let d0 = base + b * 4;
                for lane in 0..4 {
                    let fd = fb[d0 + lane];
                    acc_l[b][lane] += fd * w0_eff[d0 + lane][0];
                    acc_r[b][lane] += fd * w0_eff[d0 + lane][1];
                }
            }
        }
        let sl: [f32; 4] =
            std::array::from_fn(|c| (acc_l[0][c] + acc_l[1][c]) + (acc_l[2][c] + acc_l[3][c]));
        let sr: [f32; 4] =
            std::array::from_fn(|c| (acc_r[0][c] + acc_r[1][c]) + (acc_r[2][c] + acc_r[3][c]));
        let raw_l = bl + ((sl[0] + sl[1]) + (sl[2] + sl[3]));
        let raw_r = br + ((sr[0] + sr[1]) + (sr[2] + sr[3]));
        let yl = if raw_l.is_finite() {
            raw_l.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        let yr_s = if raw_r.is_finite() {
            raw_r.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        out.push([yl, if mono_only { yl } else { yr_s }]);
    }
    Some(out)
}

/// Validates and resolves `cached_packed_gram_256` against `features` (auto-detecting the exact
/// `gram_frame_stride` used to build `STAT` in `< 0.01 ms`), or re-contracts via `Wide192Accum`.
pub(crate) fn resolve_cached_gram_or_recontract(
    cached_packed_gram_256: Option<&[f64]>,
    features: &[[f32; 256]],
    star_emr_w: Option<&[f64]>,
    requested_stride: usize,
) -> (Vec<f64>, usize) {
    let n_frames = features.len();
    let req_s = requested_stride.max(1);
    if n_frames == 0 {
        return (vec![0.0_f64; uplo_u_packed_len(256)], req_s);
    }

    if let Some(g) = cached_packed_gram_256
        && g.len() == uplo_u_packed_len(256)
        && let Some(emr_w) = star_emr_w
        && emr_w.len() >= n_frames
    {
        if let Some(d_probe) = (0..256).find(|&d| g[uplo_u_index(d, d)] > 1.0e-12) {
            let target_diag = g[uplo_u_index(d_probe, d_probe)];
            let mut candidates = [req_s, 4, 2, 1, 8];
            let mut seen = [0usize; 5];
            let mut seen_len = 0usize;
            for cand in &mut candidates {
                if !seen[..seen_len].contains(cand) {
                    seen[seen_len] = *cand;
                    seen_len += 1;
                }
            }
            for &s in &seen[..seen_len] {
                let mut sampled = 0usize;
                let mut idx = 0usize;
                while idx < n_frames {
                    if emr_w[idx].is_finite()
                        && emr_w[idx] > 0.0
                        && features[idx][d_probe].is_finite()
                    {
                        sampled += 1;
                    }
                    idx += s;
                }
                if sampled == 0 {
                    continue;
                }
                let scale = (n_frames as f64) / (sampled as f64);
                let mut acc = Wide192Accum::zero();
                let mut idx2 = 0usize;
                while idx2 < n_frames {
                    let w = emr_w[idx2] * scale;
                    let f = f64::from(features[idx2][d_probe]);
                    if w.is_finite() && w > 0.0 && f.is_finite() {
                        acc.add_product(f, w * f);
                    }
                    idx2 += s;
                }
                let est = acc.to_f64();
                if (est - target_diag).abs() <= 1.0e-6 * (1.0 + target_diag.abs()) {
                    return (g.to_vec(), s);
                }
            }
        } else {
            return (g.to_vec(), req_s);
        }
    }

    let weights: Vec<f64> = match star_emr_w {
        Some(w) if w.len() >= n_frames => w[..n_frames].to_vec(),
        _ => vec![1.0_f64; n_frames],
    };
    let zero_targets = vec![[0.0_f64; 2]; n_frames];
    let (gram, _) = contract_256d_gram_tiled_wide192(features, &zero_targets, &weights, req_s);
    (gram, req_s)
}

#[inline]
pub(crate) fn decode_coreset_frame_lr_at(fp16: &[u8], channels: usize, idx: usize) -> [f32; 2] {
    if channels == 2 {
        let o = idx * 4;
        let l = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o], fp16[o + 1]]));
        let r = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o + 2], fp16[o + 3]]));
        [l, r]
    } else {
        let o = idx * 2;
        let v = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o], fp16[o + 1]]));
        [v, v]
    }
}

pub(crate) struct StageDMode0Resolved {
    pub features: Vec<[f32; 256]>,
    pub pred0_lr: Vec<[f32; 2]>,
    pub compact_strided: bool,
    pub n_total_frames: usize,
    pub gram_256: Vec<f64>,
    pub sample_stride: usize,
}

pub(crate) type PrecomputedBackboneRefs<'a> = (&'a [[f32; 256]], &'a [[f32; 2]]);

/// Harvests or reuses `Mode 0` predictions and `256-D` features for Stage D, avoiding `12.28 MB`
/// full-frame feature allocations on Path 1 when `cached_packed_gram_256` matches `gram_frame_stride`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn harvest_and_resolve_mode0_for_stage_d(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    star_ctrls: &[f32],
    precomputed_backbone_harvest: Option<PrecomputedBackboneRefs<'_>>,
    cached_packed_gram_256: Option<&[f64]>,
    star_emr_w: Option<&[f64]>,
    gram_frame_stride: usize,
) -> Result<StageDMode0Resolved, DnamError> {
    if let Some((pre_f, pre_back)) = precomputed_backbone_harvest {
        let req_s = gram_frame_stride.max(1);
        if req_s > 1
            && pre_f.len() == pre_back.len().div_ceil(req_s)
            && let Some(g) = cached_packed_gram_256
            && g.len() == uplo_u_packed_len(256)
        {
            let back_sub: Vec<[f32; 2]> = (0..pre_f.len()).map(|k| pre_back[k * req_s]).collect();
            if let Some(pred0) =
                compute_mode0_pred_from_backbone_harvest(manifest, arena_64k, pre_f, &back_sub)
            {
                return Ok(StageDMode0Resolved {
                    features: pre_f.to_vec(),
                    pred0_lr: pred0,
                    compact_strided: true,
                    n_total_frames: pre_back.len(),
                    gram_256: g.to_vec(),
                    sample_stride: req_s,
                });
            }
        }
        if let Some(pred0) =
            compute_mode0_pred_from_backbone_harvest(manifest, arena_64k, pre_f, pre_back)
        {
            let n_total = pre_f.len();
            let (gram_256, sample_stride) = resolve_cached_gram_or_recontract(
                cached_packed_gram_256,
                pre_f,
                star_emr_w,
                gram_frame_stride,
            );
            if sample_stride > 1 {
                let m = n_total.div_ceil(sample_stride);
                let mut f_sub = Vec::with_capacity(m);
                let mut p_sub = Vec::with_capacity(m);
                let mut idx = 0usize;
                while idx < n_total {
                    f_sub.push(pre_f[idx]);
                    p_sub.push(pred0[idx]);
                    idx += sample_stride;
                }
                return Ok(StageDMode0Resolved {
                    features: f_sub,
                    pred0_lr: p_sub,
                    compact_strided: true,
                    n_total_frames: n_total,
                    gram_256,
                    sample_stride,
                });
            }
            return Ok(StageDMode0Resolved {
                features: pre_f.to_vec(),
                pred0_lr: pred0,
                compact_strided: false,
                n_total_frames: n_total,
                gram_256,
                sample_stride,
            });
        }
    }

    let mut mode0_arena = arena_64k.to_vec();
    for &h_base in &HEAD_BASES {
        if mode0_arena.len() >= h_base + 5 * 512 {
            mode0_arena[h_base + 512..h_base + 5 * 512].fill(0);
        }
    }
    if mode0_arena.len() >= ULTRA_MODES_5_TO_8_BASE + 8_192 {
        mode0_arena[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    }
    if mode0_arena.len() >= 10_560 {
        mode0_arena[7_524..8_512].fill(0);
    }
    if mode0_arena.len() >= 63_424 {
        mode0_arena[60_928..63_424].fill(0);
    }
    zero_arena_b_head_dc_bias(&mut mode0_arena);
    let _ = sync_clamp_descriptors_to_arena(&manifest.base, &mut mode0_arena)?;

    let mut temp_hdr = manifest.base.clone();
    let dnam_bytes =
        build_dnam_container_with_cp1(&mut temp_hdr, &mode0_arena, &mode0_arena, false, &[])?;
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(&dnam_bytes, Tier::Ultra, 48_000)?;
    for (p, &val) in star_ctrls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }

    let n_total = dry_lr.len();
    let req_s = gram_frame_stride.max(1);
    if req_s > 1
        && n_total > 0
        && let Some(g) = cached_packed_gram_256
        && g.len() == uplo_u_packed_len(256)
    {
        let m_sub = n_total.div_ceil(req_s);
        let mut f_sub = Vec::with_capacity(m_sub);
        let mut p_sub = Vec::with_capacity(m_sub);
        let mut fb = [0.0_f32; 256];
        {
            let _ftz = FtzDazGuard::new();
            for (n, &[il, ir]) in dry_lr.iter().enumerate() {
                if n % req_s == 0 {
                    let y = engine.step_frame_with_feature_bus(il, ir, &mut fb);
                    f_sub.push(fb);
                    p_sub.push(y);
                } else {
                    let _ = engine.step_frame_tora(il, ir, false);
                }
            }
        }

        let stride_matches = if let Some(emr_w) = star_emr_w
            && emr_w.len() >= n_total
            && let Some(d_probe) = (0..256).find(|&d| g[uplo_u_index(d, d)] > 1.0e-12)
        {
            let target_diag = g[uplo_u_index(d_probe, d_probe)];
            let mut sampled = 0usize;
            for (k, feat) in f_sub.iter().enumerate() {
                let w = emr_w[k * req_s];
                if w.is_finite() && w > 0.0 && feat[d_probe].is_finite() {
                    sampled += 1;
                }
            }
            if sampled > 0 {
                let scale = (n_total as f64) / (sampled as f64);
                let mut acc = Wide192Accum::zero();
                for (k, feat) in f_sub.iter().enumerate() {
                    let w = emr_w[k * req_s] * scale;
                    let f = f64::from(feat[d_probe]);
                    if w.is_finite() && w > 0.0 && f.is_finite() {
                        acc.add_product(f, w * f);
                    }
                }
                (acc.to_f64() - target_diag).abs() <= 1.0e-6 * (1.0 + target_diag.abs())
            } else {
                false
            }
        } else {
            true
        };

        if stride_matches {
            return Ok(StageDMode0Resolved {
                features: f_sub,
                pred0_lr: p_sub,
                compact_strided: true,
                n_total_frames: n_total,
                gram_256: g.to_vec(),
                sample_stride: req_s,
            });
        }

        // Fallback when STAT was built with a different stride than req_s (e.g. stride = 2 vs 4)
        engine = DnamEngine::<DefaultMathBackend>::from_bytes(&dnam_bytes, Tier::Ultra, 48_000)?;
        for (p, &val) in star_ctrls.iter().enumerate() {
            engine.set_control_immediate(p, val);
        }
    }

    let mut in_l = Vec::with_capacity(n_total);
    let mut in_r = Vec::with_capacity(n_total);
    for &[il, ir] in dry_lr {
        in_l.push(il);
        in_r.push(ir);
    }
    let mut features = vec![[0.0_f32; 256]; n_total];
    let mut out_l = vec![0.0_f32; n_total];
    let mut out_r = vec![0.0_f32; n_total];
    engine.process_block_with_feature_bus(&in_l, &in_r, &mut out_l, &mut out_r, &mut features);
    let pred0_lr = out_l.into_iter().zip(out_r).map(|(l, r)| [l, r]).collect();
    let (gram_256, sample_stride) = resolve_cached_gram_or_recontract(
        cached_packed_gram_256,
        &features,
        star_emr_w,
        gram_frame_stride,
    );
    Ok(StageDMode0Resolved {
        features,
        pred0_lr,
        compact_strided: false,
        n_total_frames: n_total,
        gram_256,
        sample_stride,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_variance_gating_edge_cases() {
        assert!(!is_control_variance_active(&[], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5, 0.5], 1.0e-4));
        assert!(is_control_variance_active(&[0.0, 1.0], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::INFINITY], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::NAN], 1.0e-4));
        let neg_bounds = AuditBounds {
            max_pole_radius: -1.0,
            max_tank_feedback_gain: -1.0,
            max_allpass_coeff: -0.5,
            min_causal_delay_samples: -2.0,
        };
        assert_eq!(
            neg_bounds.clamp_parameters(f64::NAN, f64::INFINITY, f64::NAN, f64::NAN),
            (0.0, 0.0, 0.0, 4.0)
        );
        let nan_bounds = AuditBounds {
            max_pole_radius: f64::NAN,
            max_tank_feedback_gain: f64::NAN,
            max_allpass_coeff: f64::NAN,
            min_causal_delay_samples: f64::NAN,
        };
        assert_eq!(
            nan_bounds.clamp_parameters(2.0, 2.0, 2.0, 1.0),
            (0.9995, 0.9990, 0.75, 4.0)
        );
        assert_eq!(bitcrusher_smooth_surrogate(f64::NAN, 0.1, 0.5), 0.0);
        assert!(bitcrusher_smooth_surrogate(0.5, f64::NAN, f64::NAN).is_finite());
    }

    #[test]
    fn test_inline_cp1_mode0_fastpath_and_cached_gram_branches() {
        let mut tiny_arena = vec![0u8; 256];
        let s_ok = Cp1ContributorSlot {
            lanes: 2,
            ctrl_kind: 1,
            option_idx: 2,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        let s_pair = Cp1ContributorSlot {
            lanes: 3,
            ctrl_kind: 2,
            partner_control_id: 1,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        write_inline_cp1_if_fits(&mut tiny_arena, &[s_ok]);
        let mut lite_arena = vec![0u8; 10_752];
        write_inline_cp1_if_fits(&mut lite_arena, &[s_ok, s_pair]);
        assert_eq!(
            u16::from_le_bytes([lite_arena[7_524], lite_arena[7_525]]),
            2
        );
        let mut bad_slot = s_ok;
        bad_slot.lanes = 0;
        write_inline_cp1_if_fits(&mut lite_arena, &[bad_slot]);
        assert_eq!(
            u16::from_le_bytes([lite_arena[7_524], lite_arena[7_525]]),
            0
        );
        // Exceed 832 B base budget with 7 x 3-lane slots (7 * 124 = 868 > 832)
        write_inline_cp1_if_fits(&mut lite_arena, &[s_pair; 7]);
        assert_eq!(
            u16::from_le_bytes([lite_arena[7_524], lite_arena[7_525]]),
            0
        );
        write_inline_cp1_if_fits(&mut lite_arena, &[s_ok; 9]);

        // Mode-0 fastpath guard branches
        let mut m = DnamProjManifest::default_reference();
        let mut arena_64k = vec![0u8; 64_000];
        write_f32_le(&mut arena_64k, 116, 0.0);
        write_f32_le(&mut arena_64k, 120, 1.0);
        assert!(
            compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &[], &[[0.0; 2]]).is_none()
        );
        let feats = vec![[0.25_f32; 256]; 4];
        let bbone = vec![[0.1_f32, -0.1]; 4];
        m.base.controls_schema.clear();
        assert_eq!(
            compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone)
                .unwrap()
                .len(),
            4
        );
        // Also cover harvest_and_resolve_mode0_for_stage_d precomputed strided & unstrided branches
        let zero_g = vec![0.0_f64; uplo_u_packed_len(256)];
        let res_pre_strided = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &[[0.1_f32, -0.1]; 4],
            &[],
            Some((&feats[..2], &bbone)),
            Some(&zero_g),
            None,
            2,
        )
        .unwrap();
        assert!(res_pre_strided.compact_strided);
        let res_pre_sub = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &[[0.1_f32, -0.1]; 4],
            &[],
            Some((&feats, &bbone)),
            Some(&zero_g),
            None,
            2,
        )
        .unwrap();
        assert_eq!(res_pre_sub.features.len(), 2);
        let res_pre_full = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &[[0.1_f32, -0.1]; 4],
            &[],
            Some((&feats, &bbone)),
            Some(&zero_g),
            None,
            1,
        )
        .unwrap();
        assert!(!res_pre_full.compact_strided);
        m.base.controls_schema.push(
            serde_json::from_value(serde_json::json!({
                "id": "width",
                "name": "Width",
                "type": "range",
                "declared_range": [0.0, 2.0],
                "default": 1.0,
                "calibration": { "role": "stereo_width", "nodes": [0.0, 1.0, 2.0] }
            }))
            .unwrap(),
        );
        assert!(compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone).is_none());
        m.base.controls_schema.clear();
        m.base.io_config.output_mode = "mono".to_string();
        let mono_pred =
            compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone).unwrap();
        assert_eq!(mono_pred[0][0], mono_pred[0][1]);
        m.base.io_config.output_mode = "wdw".to_string();
        assert!(compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone).is_none());
        m.base.io_config.output_mode = "stereo".to_string();
        write_f32_le(&mut arena_64k, 10_624 + 3 * 4, 1.5);
        assert!(compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone).is_none());
        write_f32_le(&mut arena_64k, 116, 0.2);
        assert!(compute_mode0_pred_from_backbone_harvest(&m, &arena_64k, &feats, &bbone).is_none());

        // Cached Gram resolution branches
        let (empty_g, s0) = resolve_cached_gram_or_recontract(None, &[], None, 4);
        assert_eq!(empty_g.len(), uplo_u_packed_len(256));
        assert_eq!(s0, 4);
        let zero_g = vec![0.0_f64; uplo_u_packed_len(256)];
        let emr = vec![1.0_f64; 4];
        let (_, s_zero) = resolve_cached_gram_or_recontract(Some(&zero_g), &feats, Some(&emr), 2);
        assert_eq!(s_zero, 2);
        let mut mismatch_g = zero_g;
        mismatch_g[0] = 999.0;
        let (recontracted, s_re) =
            resolve_cached_gram_or_recontract(Some(&mismatch_g), &feats, Some(&emr), 2);
        assert_eq!(s_re, 2);
        assert!((recontracted[0] - 0.25).abs() < 1.0e-5);

        // Cover solve_stage_d_cp1_and_audit wrapper + zero-weight off-center early-return branch
        let mut m_d = DnamProjManifest::default_reference();
        m_d.base.controls_schema.push(
            serde_json::from_value(serde_json::json!({
                "id": "drive",
                "name": "Drive",
                "type": "range",
                "declared_range": [0.0, 1.0],
                "default": 0.5
            }))
            .unwrap(),
        );
        m_d.stems_registry.push(crate::proj::StemRegistryEntry {
            stem_id: "off0".to_string(),
            chunk_id: 99,
            enabled: true,
            is_star_center: false,
            controls_meta: std::collections::BTreeMap::from([(
                "drive".to_string(),
                serde_json::json!(0.9),
            )]),
            provenance: crate::proj::StemProvenance {
                input_pcm_sha256: "0".repeat(64),
                target_pcm_sha256: "1".repeat(64),
                num_frames: 16,
            },
            user_metadata: std::collections::BTreeMap::new(),
            extracted_physics: None,
        });
        let mut a64 = crate::proj::vault::initialize_backbone_arena_if_needed(None);
        let stmc_empty = std::collections::BTreeMap::new();
        let d_out = crate::solver::solve_stage_d_cp1_and_audit(
            &mut m_d,
            &mut a64,
            &stmc_empty,
            &[[0.05_f32, -0.05]; 16],
            None,
            1,
            false,
        )
        .expect("zero-weight Stage D should succeed cleanly");
        assert_eq!(d_out.active_head_modes, 0);
    }

    proptest! {
        #[test]
        fn prop_audit_bounds_always_enforce_invariants(
            r in -2.0_f64..2.0_f64,
            g in -2.0_f64..2.0_f64,
            a in -2.0_f64..2.0_f64,
            d in -10.0_f64..100.0_f64,
        ) {
            let bounds = AuditBounds::default();
            let (rc, gc, ac, dc) = bounds.clamp_parameters(r, g, a, d);
            prop_assert!((0.0..=0.9995).contains(&rc));
            prop_assert!((0.0..=0.9990).contains(&gc));
            prop_assert!((-0.75..=0.75).contains(&ac));
            prop_assert!(dc >= 4.0);
        }

        #[test]
        fn prop_bitcrusher_surrogate_bounded_error(
            x in -10.0_f64..10.0_f64,
            delta in 0.01_f64..1.0_f64,
            tau in 0.0_f64..=1.0_f64,
        ) {
            let y = bitcrusher_smooth_surrogate(x, delta, tau);
            prop_assert!((y - x).abs() <= delta);
        }
    }
}
