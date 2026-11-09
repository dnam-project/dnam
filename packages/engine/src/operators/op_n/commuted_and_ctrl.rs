//! Op-N Commuted LTI Cascade, 3 kHz Latent Folding & Control-Plane Primitives.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.commuted.theorem1]`, `ref[dsp.op-n.stage2.latent-folding]`, `ref[dsp.op-n.stage2.stereo-coupling]`, `ref[dsp.op-n.ctrl.lobatto-basis]`, `ref[dsp.op-n.ctrl.chebyshev-lfo]`, `ref[dsp.op-n.ctrl.trust-clip]`).

use alloc::vec::Vec;

/// 3-Node Newton/Lobatto Basis evaluation (`ref[dsp.op-n.ctrl.lobatto-basis]`).
///
/// Evaluates: `[u - u1, (u - u1)(u - u2), (u - u1)(u - u2)(u - u3), s_sw]`.
#[inline(always)]
pub fn eval_lobatto_basis_3node(
    u: f32,
    nodes: &[f32; 4],
    degree: usize,
    switch_opt: f32,
) -> [f32; 4] {
    let mut psi = [0.0; 4];
    if degree >= 1 {
        psi[0] = u - nodes[0];
    }
    if degree >= 2 {
        psi[1] = psi[0] * (u - nodes[1]);
    }
    if degree >= 3 {
        psi[2] = psi[1] * (u - nodes[2]);
    }
    psi[3] = switch_opt;
    psi
}

/// Chebyshev 4-harmonic LFO polynomial waveshaping.
///
/// `T1(x) = x`
/// `T2(x) = 2x^2 - 1`
/// `T3(x) = 4x^3 - 3x`
/// `T4(x) = 8x^4 - 8x^2 + 1`
#[inline(always)]
pub fn eval_chebyshev_4harm(x: f32) -> [f32; 4] {
    let x2 = x * x;
    let t1 = x;
    let t2 = 2.0 * x2 - 1.0;
    let t3 = 4.0 * x2 * x - 3.0 * x;
    let t4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0;
    [t1, t2, t3, t4]
}

/// Outward-Checked Ellipsoidal Trust-Region Clipping (`ref[dsp.ctrl.op-n.trust-clip]`, `ref[dsp.op-n.ctrl.trust-clip]`, `0 FDIV`, `0 FSQRT`).
///
/// Applies the 6-step `f64` Newton fast path with 32-step bisection fallback so that
/// $\sum_i \text{widen}(z_i)^2 \le \rho_{\max}^2$ holds strictly after `RN32` rounding.
#[inline(always)]
pub fn trust_region_softclip_4d(z: &mut [f32; 4], rho_max: f32) {
    if rho_max <= 1.0e-12 {
        return;
    }
    let val_bits = [
        z[0].to_bits(),
        z[1].to_bits(),
        z[2].to_bits(),
        z[3].to_bits(),
    ];
    if let Some((clipped_bits, _, _)) = crate::math::trust_clip_outward_checked_bits(
        &val_bits,
        rho_max.to_bits(),
        1.0e-12_f32.to_bits(),
    ) {
        for i in 0..4 {
            z[i] = f32::from_bits(clipped_bits[i]);
        }
    }
}

/// Evaluates Theorem 1 (Exact LTI Basis Commutativity) by running a 1-pole filter bank $\mathcal{L}_k$
/// (`poles[0..K]`) on a $P$-dimensional basis sequence `u_seq` in both uncommuted (`filter all P -> project W_k`)
/// and commuted (`project W_k -> filter 1 lane`) order (`ref[dsp.op-n.commuted.theorem1]`).
pub fn eval_commuted_vs_uncommuted_lti_cascade_f64<const P: usize, const K: usize>(
    u_seq: &[[f64; P]],
    poles: &[f64; K],
    weights: &[[f64; P]; K],
) -> (Vec<f64>, Vec<f64>) {
    let n_len = u_seq.len();
    let mut uncomm_state = [[0.0f64; P]; K];
    let mut comm_state = [0.0f64; K];
    let mut y_uncomm = Vec::with_capacity(n_len);
    let mut y_comm = Vec::with_capacity(n_len);

    for u in u_seq {
        let mut sum_u = 0.0f64;
        let mut sum_c = 0.0f64;
        for k in 0..K {
            let r = poles[k];
            let one_minus_r = 1.0 - r;
            for p in 0..P {
                uncomm_state[k][p] = r * uncomm_state[k][p] + one_minus_r * u[p];
                sum_u += uncomm_state[k][p] * weights[k][p];
            }
            let mut v_k = 0.0f64;
            for p in 0..P {
                v_k += u[p] * weights[k][p];
            }
            comm_state[k] = r * comm_state[k] + one_minus_r * v_k;
            sum_c += comm_state[k];
        }
        y_uncomm.push(sum_u);
        y_comm.push(sum_c);
    }
    (y_uncomm, y_comm)
}

/// Folds $3\text{ kHz}$ stage control latents $\mathbf{z}_{\text{stage}} \in \mathbb{R}^{D_s}$ into a
/// sub-stage projection vector $\mathbf{w}(\mathbf{z}) = \mathbf{w}_0 + \sum_{r=0}^{D_s-1} z_r \mathbf{w}_{r+1}$
/// (`ref[dsp.op-n.stage2.latent-folding]`).
pub fn fold_stage2_latent_weights<const N: usize>(
    w_base: &[f32; N],
    w_modes: &[[f32; N]],
    z_stage: &[f32],
) -> [f32; N] {
    let mut out = *w_base;
    let active = w_modes.len().min(z_stage.len());
    for r in 0..active {
        let z = z_stage[r];
        for i in 0..N {
            out[i] += z * w_modes[r][i];
        }
    }
    out
}

/// Evaluates the $2 \times 2$ block stereo coupling matrix on a 16-D stereo drive basis
/// $\mathbf{u}_{\text{stereo}} = [\mathbf{u}_L^{(8)}, \mathbf{u}_R^{(8)}]^\top$
/// (`ref[dsp.op-n.stage2.stereo-coupling]`).
pub fn eval_stage2_stereo_coupling_16d(
    u_left: &[f32; 8],
    u_right: &[f32; 8],
    w_ll: &[f32; 8],
    w_lr: &[f32; 8],
    w_rl: &[f32; 8],
    w_rr: &[f32; 8],
) -> (f32, f32) {
    let mut out_l = 0.0f32;
    let mut out_r = 0.0f32;
    for i in 0..8 {
        out_l += w_ll[i] * u_left[i] + w_rl[i] * u_right[i];
        out_r += w_lr[i] * u_left[i] + w_rr[i] * u_right[i];
    }
    (out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::container::ZohRescaledParams;
    use crate::stages::{Stage2CommutedCascade, ToneBasisFilters};
    use alloc::vec::Vec;
    use proptest::prelude::*;

    #[test]
    fn test_lobatto_chebyshev_and_trust_region() {
        let nodes = [0.1, 0.3, 0.7, 0.9];
        let at_n0 = eval_lobatto_basis_3node(nodes[0], &nodes, 3, 1.0);
        assert_eq!(at_n0, [0.0, 0.0, 0.0, 1.0]);
        let at_n1 = eval_lobatto_basis_3node(nodes[1], &nodes, 3, 0.0);
        assert!((at_n1[0] - 0.2).abs() < 1e-6);
        assert!(at_n1[1].abs() < 1e-7);
        assert!(at_n1[2].abs() < 1e-7);
        let at_n2 = eval_lobatto_basis_3node(nodes[2], &nodes, 3, 0.0);
        assert!(at_n2[2].abs() < 1e-7);

        let theta = 0.6_f32;
        let cheb = eval_chebyshev_4harm(theta.cos());
        assert!((cheb[0] - theta.cos()).abs() < 1e-6);
        assert!((cheb[1] - (2.0 * theta).cos()).abs() < 1e-6);
        assert!((cheb[2] - (3.0 * theta).cos()).abs() < 1e-6);
        assert!((cheb[3] - (4.0 * theta).cos()).abs() < 1e-5);

        let mut z = [10.0, 10.0, 10.0, 10.0];
        trust_region_softclip_4d(&mut z, 2.0);
        let norm = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
        assert!((norm - 2.0).abs() < 1e-3);

        let mut z_in = [0.1, 0.1, 0.1, 0.1];
        trust_region_softclip_4d(&mut z_in, 2.0);
        for &v in &z_in {
            assert!((v - 0.1).abs() < 1e-3);
        }
    }

    /// verify[dsp.op-n.commuted.theorem1]
    /// verify[dsp.op-n.commuted.causality]
    /// verify[dsp.op-n.commuted.project-then-filter]
    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.stage2.sub-stages]
    /// verify[dsp.op-n.stage2.tone-basis]
    /// verify[dsp.op-n.stage2.stereo-coupling]
    /// verify[dsp.op-n.stage2.latent-folding]
    /// verify[dsp.op-n.ctrl.incremental-update]
    /// verify[dsp.op-n.ctrl.lobatto-basis]
    /// verify[dsp.op-n.ctrl.chebyshev-lfo]
    /// verify[dsp.op-n.ctrl.fm-interaction]
    /// verify[dsp.op-n.ctrl.trust-clip]
    /// verify[dsp.zoh.stage2-op-n]
    #[test]
    fn test_commuted_lti_theorem1_dc_equilibrium_stage2_and_zoh() {
        let _guard = FtzDazGuard::new();

        let u_seq: Vec<[f64; 4]> = (0..128)
            .map(|n| {
                let t = (n as f64) * 0.07;
                [t.sin(), (2.0 * t).cos(), (0.5 * t).sin(), -0.3 * t.cos()]
            })
            .collect();
        let poles = [0.98959_f64, 0.94916, 0.85463];
        let weights = [
            [0.4_f64, -0.2, 0.1, 0.3],
            [-0.1, 0.5, 0.2, -0.4],
            [0.3, 0.1, -0.5, 0.2],
        ];
        let (y_uncomm, y_comm) =
            eval_commuted_vs_uncommuted_lti_cascade_f64(&u_seq, &poles, &weights);
        for (yu, yc) in y_uncomm.iter().zip(y_comm.iter()) {
            assert!((yu - yc).abs() < 1e-13);
        }

        let mut tb = ToneBasisFilters::new();
        tb.init_dc_equilibrium(0.75);
        let out_tb = tb.step(0.75);
        assert!((out_tb[0] - 0.75).abs() < 1e-6);
        assert!((out_tb[1] - 0.75).abs() < 1e-6);
        assert!((out_tb[2] - 0.75).abs() < 1e-6);
        assert!(out_tb[3].abs() < 1e-6);

        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let mut s2 = Stage2CommutedCascade::new();
        s2.reconfigure_sample_rate(&zoh);
        assert!((s2.hyst_diff_scale - 2.0).abs() < 1e-6);

        let (cl, cr) = eval_stage2_stereo_coupling_16d(
            &[1.0; 8], &[0.5; 8], &[0.25; 8], &[0.01; 8], &[0.01; 8], &[0.25; 8],
        );
        assert!((cl - (2.0 + 0.04)).abs() < 1e-6);
        assert!((cr - (0.08 + 1.0)).abs() < 1e-6);

        let w_folded = fold_stage2_latent_weights(
            &[1.0, 2.0, 3.0, 4.0],
            &[[0.5, 0.0, -0.5, 1.0], [0.0, 1.0, 0.0, -1.0]],
            &[2.0, 0.5],
        );
        assert_eq!(w_folded, [2.0, 2.5, 2.0, 5.5]);

        let lb = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 2, 0.0);
        assert_eq!(lb[0], 0.5);
        let ch_lfo = eval_chebyshev_4harm(1.0);
        assert_eq!(ch_lfo, [1.0, 1.0, 1.0, 1.0]);
        let mut z4 = [10.0, 0.0, 0.0, 0.0];
        trust_region_softclip_4d(&mut z4, 2.0);
        assert!((z4[0] - 2.0).abs() < 1e-3);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_trust_region_softclip_4d_bounds(
            z0 in -8.0f32..8.0f32,
            z1 in -8.0f32..8.0f32,
            z2 in -8.0f32..8.0f32,
            z3 in -8.0f32..8.0f32,
            rho_max in 0.5f32..5.0f32,
        ) {
            let mut z = [z0, z1, z2, z3];
            let orig_norm = (z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3).sqrt();
            trust_region_softclip_4d(&mut z, rho_max);
            let new_norm = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]).sqrt();
            prop_assert!(new_norm <= rho_max + 1e-4);
            if orig_norm <= 0.45 * rho_max {
                prop_assert!((new_norm - orig_norm).abs() < 2e-2 * rho_max);
            }
        }
    }
}
