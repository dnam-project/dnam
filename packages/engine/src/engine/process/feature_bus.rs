//! Canonical 256-D Multi-Tier Feature Bus Assembly (`Lite` `0..64`, `Full` `64..128`, `Pro` `128..192`, `Ultra` `192..256`).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` §7 (`CANONICAL-256D-FEATURE-BUS-MAP`, `r[dsp.head.feature-bus-slots]`).

use crate::stages::Stage2CascadeFrameOut;
use crate::stages::stage1_pre::Stage1PreDynamics;

/// Packs the canonical 256-D (`64` features/tier) feature bus up to `loop_tiers` (`1..=4`).
///
/// Guarantees:
/// - **Zero duplicate columns across all 256 slots (`Lite..Ultra`)**: Every slot is backed by a
///   distinct physical state or nonlinear branch readout across Stages 0, I, II, III, and IV.
/// - **Strict lower-triangular tier causality (`REQ-5.6`)**: `Lite` (`0..64`) depends exclusively
///   on `Arena 0`, and higher-tier slots (`64..256`) never mutate lower-tier state.
/// - **Exact DC resting equilibrium (`[STAB-ZERO-DC]`)**: Evaluates to `0.0` across all 256 slots
///   at zero input after `reset_to_dc_equilibrium()`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn pack_tier_feature_bus(
    stage1: &Stage1PreDynamics,
    s2_out: &Stage2CascadeFrameOut,
    kautz_modal_feats: &[[f32; 4]; 8],
    fir_seg_l: &[f32; 4],
    fir_seg_r: &[f32; 4],
    tank_readouts: &[f32; 4],
    s4_full_taps: &[f32; 10],
    s4_pro_taps: &[f32; 6],
    s4_ultra_taps: &[f32; 8],
    down_l: f32,
    down_r: f32,
    clean_sub_l: f32,
    clean_sub_r: f32,
    s1_l: f32,
    s1_r: f32,
    s2_in_l: f32,
    s2_in_r: f32,
    send_l: f32,
    send_r: f32,
    tank_l: f32,
    tank_r: f32,
    loop_tiers: usize,
) -> [f32; 256] {
    let mut fb = [0.0_f32; 256];

    // Tier 0 — Lite (0..64):
    // 0..4: Stage III Kautz Block 0 (4 shared pole pairs)
    fb[0..4].copy_from_slice(&kautz_modal_feats[0]);
    // 4..12: Stage I T1 Sub-Harmonic Tracker L/R (8 floats)
    fb[4..8].copy_from_slice(&stage1.last_tracker_l);
    fb[8..12].copy_from_slice(&stage1.last_tracker_r);
    // 12..20: Stage I Pre-FX Stereo Op-M[Pre] z_re L/R (8 floats)
    fb[12..16].copy_from_slice(&stage1.op_m_pre.z_re);
    fb[16..20].copy_from_slice(&stage1.op_m_pre_r.z_re);
    // 20..52: Stage II Commuted Cascade Lite (32 floats: basis_l[0..8], basis_r[0..8], lite_tail[0..16])
    fb[20..28].copy_from_slice(&s2_out.basis_l);
    fb[28..36].copy_from_slice(&s2_out.basis_r);
    fb[36..52].copy_from_slice(&s2_out.lite_tail);
    // 52..60: Stage IV Spatial Tank Lite (8 floats)
    fb[52..56].copy_from_slice(tank_readouts);
    fb[56] = tank_l;
    fb[57] = tank_r;
    fb[58] = s1_l + s2_out.s2_l;
    fb[59] = s1_r + s2_out.s2_r;
    // 60..62: Stage 0 Clean Sub L/R (2 floats)
    fb[60] = clean_sub_l;
    fb[61] = clean_sub_r;
    // 62..64: Stage III Early FIR Segment 0 L/R (2 floats)
    fb[62] = fir_seg_l[0];
    fb[63] = fir_seg_r[0];

    if loop_tiers >= 2 {
        // Tier 1 — Full (64..128):
        // 64..68: Stage III Kautz Block 1 (4 shared pole pairs)
        fb[64..68].copy_from_slice(&kautz_modal_feats[1]);
        // 68..84: Stage I Pre-FX Full Group 1 (16 floats: delta Op-M z_re L/R + Stage I/0 intermediate taps)
        fb[68..72].copy_from_slice(&stage1.op_m_pre_delta_l.z_re);
        fb[72..76].copy_from_slice(&stage1.op_m_pre_delta_r.z_re);
        fb[76..80].copy_from_slice(&[s1_l, s1_r, down_l, down_r]);
        fb[80..84].copy_from_slice(&[s2_in_l, s2_in_r, send_l, send_r]);
        // 84..116: Stage II Commuted Cascade Full (32 floats: V2A branch + interstage L^(1)/L^(2) + even RC bias)
        fb[84..116].copy_from_slice(&s2_out.full_features);
        // 116..126: Stage IV Spatial Tank Full (10 floats: M=8 tank tap differences + diffuser states)
        fb[116..126].copy_from_slice(s4_full_taps);
        // 126..128: Stage III Early FIR Segment 1 L/R (2 floats, taps 16..31)
        fb[126] = fir_seg_l[1];
        fb[127] = fir_seg_r[1];

        if loop_tiers >= 3 {
            // Tier 2 — Pro (128..192):
            // 128..136: Stage III Kautz Blocks 2..3 (8 shared pole pairs)
            fb[128..132].copy_from_slice(&kautz_modal_feats[2]);
            fb[132..136].copy_from_slice(&kautz_modal_feats[3]);
            // 136..152: Stage I Pre-FX Pro Group 2 (16 floats: delta Op-M z_im L/R + Op-N/Op-T intermediate taps)
            fb[136..140].copy_from_slice(&stage1.op_m_pre_delta_l.z_im);
            fb[140..144].copy_from_slice(&stage1.op_m_pre_delta_r.z_im);
            fb[144..148].copy_from_slice(&[
                stage1.last_op_n_l,
                stage1.last_op_n_r,
                stage1.last_op_t_l,
                stage1.last_op_t_r,
            ]);
            fb[148..152].copy_from_slice(&[
                stage1.last_op_t_l - stage1.prev_op_t_l,
                stage1.last_op_t_r - stage1.prev_op_t_r,
                stage1.op_m_pre.z_im[0],
                stage1.op_m_pre_r.z_im[0],
            ]);
            // 152..184: Stage II Commuted Cascade Pro (32 floats: V2B branch + cross-stage Volterra products)
            fb[152..184].copy_from_slice(&s2_out.pro_features);
            // 184..190: Stage IV Spatial Tank Pro (6 floats: Pro delay taps + ducking/loop products)
            fb[184..190].copy_from_slice(s4_pro_taps);
            // 190..192: Stage III Early FIR Segment 2 L/R (2 floats, taps 32..47)
            fb[190] = fir_seg_l[2];
            fb[191] = fir_seg_r[2];

            if loop_tiers >= 4 {
                // Tier 3 — Ultra (192..256):
                // 192..208: Stage III Kautz Blocks 4..7 (16 shared pole pairs)
                fb[192..196].copy_from_slice(&kautz_modal_feats[4]);
                fb[196..200].copy_from_slice(&kautz_modal_feats[5]);
                fb[200..204].copy_from_slice(&kautz_modal_feats[6]);
                fb[204..208].copy_from_slice(&kautz_modal_feats[7]);
                // 208..224: Stage I Pre-FX Ultra Group 3 (16 floats: Lite Op-M z_im L/R + Hilbert & T1 DC states)
                fb[208..212].copy_from_slice(&stage1.op_m_pre.z_im);
                fb[212..216].copy_from_slice(&stage1.op_m_pre_r.z_im);
                fb[216..220].copy_from_slice(&[
                    stage1.hilbert_l.i_state[5][0],
                    stage1.hilbert_l.q_state[5][0],
                    stage1.hilbert_r.i_state[5][0],
                    stage1.hilbert_r.q_state[5][0],
                ]);
                fb[220..224].copy_from_slice(&[
                    stage1.dc_blocker_l.prev_d,
                    stage1.dc_blocker_r.prev_d,
                    stage1.prev_op_t_l,
                    stage1.prev_op_t_r,
                ]);
                // 224..246: Stage II Commuted Cascade Ultra (22 floats: V3 branch + hysteresis cross-products)
                fb[224..246].copy_from_slice(&s2_out.ultra_features);
                // 246..254: Stage IV Spatial Tank Ultra (8 floats: M=16 deep late-reflection tap differences)
                fb[246..254].copy_from_slice(s4_ultra_taps);
                // 254..256: Stage III Early FIR Segment 3 L/R (2 floats, taps 48..63)
                fb[254] = fir_seg_l[3];
                fb[255] = fir_seg_r[3];
            }
        }
    }

    fb
}

/// Evaluates the 16-group per-tier SIMD `Vec4` head readout dot product across `0..active_tiers`
/// (`Lite` `0..64`, `Full` `64..128`, `Pro` `128..192`, `Ultra` `192..256`), shared between
/// `step_frame_tora` and `step_frame_4port_tora`.
#[inline(always)]
pub(crate) fn eval_tier_feature_bus_head_dot<B: crate::math::MathBackend>(
    fb: &[f32; 256],
    collapsed_head_wl: &[[B::Vec4; 16]; 4],
    collapsed_head_wr: &[[B::Vec4; 16]; 4],
    active_tiers: usize,
    mono_only: bool,
) -> (f32, f32) {
    use crate::math::SimdVec4;
    let mut head_acc_l = B::Vec4::zero();
    let mut head_acc_r = B::Vec4::zero();
    for t in 0..active_tiers.min(4) {
        let t_off = t * 64;
        let wl_t = &collapsed_head_wl[t];
        let wr_t = &collapsed_head_wr[t];
        for g in 0..16 {
            let off = t_off + g * 4;
            let phi_g = B::Vec4::from_array([fb[off], fb[off + 1], fb[off + 2], fb[off + 3]]);
            head_acc_l = head_acc_l.fma(wl_t[g], phi_g);
            if !mono_only {
                head_acc_r = head_acc_r.fma(wr_t[g], phi_g);
            }
        }
    }
    let head_sum_l = head_acc_l.horizontal_sum();
    let head_sum_r = if mono_only {
        head_sum_l
    } else {
        head_acc_r.horizontal_sum()
    };
    (head_sum_l, head_sum_r)
}

/// Evaluates the Stage IV Spatial Tank contribution to `(head_sum_l, head_sum_r)` across all active tiers
/// (`Lite` slots `52..58`, `Full` slots `116..126`, `Pro` slots `184..190`, `Ultra` slots `246..254`)
/// so `WDW` mode cleanly separates dry center core (`core_l/r`) from wet spatial tank (`w_wet_l/r`).
#[inline(always)]
pub(crate) fn eval_tier_tank_head_contribution<B: crate::math::MathBackend>(
    fb: &[f32; 256],
    collapsed_head_wl: &[[B::Vec4; 16]; 4],
    collapsed_head_wr: &[[B::Vec4; 16]; 4],
    active_tiers: usize,
) -> (f32, f32) {
    use crate::math::SimdVec4;
    const TANK_RANGES: [(usize, usize, usize); 4] =
        [(0, 52, 58), (1, 116, 126), (2, 184, 190), (3, 246, 254)];
    let mut tank_head_l = 0.0_f32;
    let mut tank_head_r = 0.0_f32;
    for &(t, start, end) in TANK_RANGES.iter().take(active_tiers.min(4)) {
        for idx in start..end {
            let in_tier = idx - t * 64;
            let g = in_tier / 4;
            let lane = in_tier % 4;
            let wl = collapsed_head_wl[t][g].to_array()[lane];
            let wr = collapsed_head_wr[t][g].to_array()[lane];
            let val = fb[idx];
            tank_head_l += wl * val;
            tank_head_r += wr * val;
        }
    }
    (tank_head_l, tank_head_r)
}
