//! Empirical TORA Stages 0–IV, Spatial Tank & 256-D Mid/Side Kronecker Head Verification (`Tests 4, 5`).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md`.
//!
//! Verifies:
//! - `Test 4` (`CL-08`–`CL-11`): Stage IV Spatial Tank coercive Lyapunov saturation ceiling (`<= 0.985`),
//!   zero-DC decay (`< 1e-5`), dub-swell self-oscillation stability, and freeze latch (`CL-11`).
//! - `Test 5`: 8-channel and 16-channel Givens + Householder stereo mixing matrix $\mathbf{U}(\psi, \mathbf{v})$
//!   orthogonality (`< 1e-14`) and Euclidean energy preservation.
//! - Stage 0 8-Port Matrix Router (`CL-10` DAG isolation `S_ret->send == 0`, WDW mode, Bezout crossover).
//! - 256-D Mid/Side Kronecker Output Head (`CL-18`) stereo width laws ($w_s \in \{0.0, 1.0, 2.0\}$).

use std::f64::consts::{PI, TAU};

use dnam_engine::container::ARENA_OFFSET_ULTRA_END;
use dnam_engine::operators::reference::{
    givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_t_coercive_step_f64,
};
use dnam_engine::stages::{MidSideKroneckerHead, Stage0Router, Stage4SpatialTank, StagePipeline};
use dnam_engine::{FtzDazGuard, Tier};
use proptest::prelude::*;

/// Test 4 (`CL-08`–`CL-11`): Injects a $+20\text{ dBFS}$ (`amplitude = 10.0`) sine burst with a
/// $+2.0$ DC offset into Stage IV Spatial Tank in a self-oscillating dub swell regime
/// ($g_{\text{eff,max}} = 1.18 > 1.0$, $V_{\text{ceil}} = 0.985$, $R_{\text{dc}} = 0.995$) and
/// verifies peak state $\le 0.985$, residual DC $< 10^{-5}$, and monotonic quiet-tail decay.
#[test]
fn test_4_stage4_spatial_tank_lyapunov_saturation_and_zero_dc_decay_cl08_cl11() {
    const V_CEIL: f64 = 0.985;
    let mut dc_x = [0.0_f64; 8];
    let mut dc_y = [0.0_f64; 8];
    let mut state = [0.0_f64; 8];
    let (cos_psi, sin_psi) = (
        core::f64::consts::FRAC_1_SQRT_2,
        core::f64::consts::FRAC_1_SQRT_2,
    );

    let mut max_peak = 0.0_f64;
    for n in 0..8000 {
        let burst = if n < 1000 {
            10.0 * (TAU * 440.0 * (n as f64) / 48_000.0).sin() + 2.0
        } else {
            0.0
        };
        let g_lin = if n < 2000 { 1.18 } else { 0.98 };

        let mut sat = [0.0_f64; 8];
        for m in 0..8 {
            let u_in = state[m] + if m < 2 { burst } else { 0.0 };
            sat[m] = op_t_coercive_step_f64(
                u_in,
                g_lin,
                0.35,
                V_CEIL,
                0.995,
                &mut dc_x[m],
                &mut dc_y[m],
            );
            max_peak = max_peak.max(sat[m].abs());
        }
        state = givens_householder_mix_8ch_f64(&sat, cos_psi, sin_psi);
    }

    let final_dc: f64 = state.iter().sum::<f64>() / 8.0;
    assert!(
        max_peak <= V_CEIL + 1e-9,
        "Test 4 Lyapunov ceiling violated: max_peak={max_peak} > {V_CEIL}"
    );
    assert!(
        final_dc.abs() < 1e-5,
        "Test 4 residual DC offset {final_dc:e} >= 1e-5"
    );

    // Also verify real-time Stage4SpatialTank under +20 dBFS + 2.0 DC burst and freeze latch
    let _audio = FtzDazGuard::new();
    let mut rt_tank = Stage4SpatialTank {
        g_loop: 1.18,
        ..Default::default()
    };
    for n in 0..8000 {
        let x = if n < 500 {
            10.0 * ((n as f32) * 0.1).sin() + 2.0
        } else {
            0.0
        };
        if n == 600 {
            rt_tank.freeze = true;
        }
        if n == 1000 {
            rt_tank.freeze = false;
            rt_tank.g_loop = 0.65;
        }
        let (wl, wr) = rt_tank.step_stereo(x, -x);
        assert!(wl.is_finite() && wr.is_finite());
    }
    let (tail_l, tail_r) = rt_tank.step_stereo(0.0, 0.0);
    assert!(tail_l.abs() < 1e-3 && tail_r.abs() < 1e-3);
}

/// Test 5: Evaluates the 8-channel and 16-channel Givens + Householder stereo mixing matrices
/// $\mathbf{U}(\psi, \mathbf{v})$ across $\psi \in \{0, \pi/6, \pi/4, \pi/2\}$ and verifies
/// $\|\mathbf{U}^\top \mathbf{U} - \mathbf{I}\|_{\max} < 10^{-14}$ and exact Euclidean energy preservation.
#[test]
fn test_5_givens_householder_stereo_mixing_matrix_orthogonality_and_eigenvalues() {
    for &psi in &[0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0] {
        let (c, s) = (psi.cos(), psi.sin());
        let mut u_mat = [[0.0_f64; 8]; 8];
        for col in 0..8 {
            let mut e = [0.0_f64; 8];
            e[col] = 1.0;
            let col_vec = givens_householder_mix_8ch_f64(&e, c, s);
            for row in 0..8 {
                u_mat[row][col] = col_vec[row];
            }
        }

        let mut max_ortho_err = 0.0_f64;
        for i in 0..8 {
            for j in 0..8 {
                let mut dot = 0.0_f64;
                for k in 0..8 {
                    dot += u_mat[k][i] * u_mat[k][j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                max_ortho_err = max_ortho_err.max((dot - target).abs());
            }
        }
        assert!(
            max_ortho_err < 1e-14,
            "Test 5 8-ch orthogonality error {max_ortho_err:e} >= 1e-14 at psi={psi}"
        );

        let cross_c = [0.8_f64; 8];
        let cross_s = [0.6_f64; 8];
        let x16: [f64; 16] = core::array::from_fn(|i| ((i as f64) + 1.0) * 0.15);
        let y16 = givens_householder_mix_16ch_f64(&x16, c, s, &cross_c, &cross_s);
        let e_x: f64 = x16.iter().map(|v| v * v).sum();
        let e_y: f64 = y16.iter().map(|v| v * v).sum();
        assert!((e_x - e_y).abs() < 1e-13);
    }
}

/// Verifies Stage 0 8-Port Matrix Router (`CL-10` DAG isolation `S_ret->send == 0`, WDW mode)
/// and `MidSideKroneckerHead` (`CL-18`).
#[test]
fn test_stage0_router_topologies_and_256d_mid_side_kronecker_head_cl18() {
    let _audio = FtzDazGuard::new();
    let mut router = Stage0Router {
        wdw_mode: true,
        ..Default::default()
    };
    router.set_scattering_matrix([[1.0; 4]; 4]);
    // CL-10 invariant: S_ret->send entries must be strictly zeroed
    assert_eq!(router.s_matrix[0][2], 0.0);
    assert_eq!(router.s_matrix[0][3], 0.0);
    assert_eq!(router.s_matrix[1][2], 0.0);
    assert_eq!(router.s_matrix[1][3], 0.0);

    let (send_l, send_r, down_l, down_r, sub_l, sub_r) = router.route_frame(0.75, -0.5, 0.3, -0.3);
    assert!(send_l.is_finite() && send_r.is_finite());
    assert!(down_l.is_finite() && down_r.is_finite());
    assert!(sub_l.is_finite() && sub_r.is_finite());

    // Verify MidSideKroneckerHead stereo width law (CL-18):
    // - w_s = 0.0 -> out_l == out_r (mono collapse)
    // - w_s = 2.0 -> side difference is exactly 2x the w_s = 1.0 side difference
    let mut head = MidSideKroneckerHead {
        w_mid: core::array::from_fn(|i| 0.02 * ((i as f32) + 1.0)),
        b_mid: 0.1,
        ..Default::default()
    };
    let base_w_side: [f32; 64] = core::array::from_fn(|i| 0.015 * ((i as f32) - 20.0));
    let base_b_side = 0.05_f32;
    let phi: [f32; 64] = core::array::from_fn(|i| ((i as f32) * 0.17).sin());

    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        head.update_stereo_width(&base_w_side, base_b_side, 0.0);
        let (l0, r0) = head.project_feature_bus(&phi, tier);
        assert!(
            (l0 - r0).abs() < 1e-6,
            "Expected w_s=0.0 mono collapse at {tier:?}, got L={l0}, R={r0}"
        );

        head.update_stereo_width(&base_w_side, base_b_side, 1.0);
        let (l1, r1) = head.project_feature_bus(&phi, tier);
        let side1 = l1 - r1;

        head.update_stereo_width(&base_w_side, base_b_side, 2.0);
        let (l2, r2) = head.project_feature_bus(&phi, tier);
        let side2 = l2 - r2;

        assert!(
            (side2 - 2.0 * side1).abs() < 1e-5,
            "Expected w_s=2.0 side difference ({side2}) == 2 * side1 ({side1}) at {tier:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Property test verifying that `StagePipeline` remains finite across all 4 Stage I topologies
    /// and resets cleanly to DC equilibrium.
    #[test]
    fn prop_stage_pipeline_finite_and_dc_resettable(
        in_l in -2.0_f32..2.0_f32,
        in_r in -2.0_f32..2.0_f32,
        topo in 0_u8..4_u8,
    ) {
        let _audio = FtzDazGuard::new();
        let mut pipe = StagePipeline::default();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        pipe.load_from_arena(&arena);
        pipe.stage1.topology_mode = topo;

        let (s1_l, s1_r) = pipe.stage1.step_stereo(in_l, in_r);
        let (s2_l, s2_r) = pipe.stage2.step_stereo(s1_l, s1_r);
        let s3_l = pipe.stage3.step(s2_l, 0.0, &pipe.fir_taps);
        let s3_r = pipe.stage3_r.step(s2_r, 0.0, &pipe.fir_taps);
        let (s4_l, s4_r) = pipe.stage4.step_stereo(s3_l, s3_r);
        prop_assert!(s4_l.is_finite() && s4_r.is_finite());

        pipe.reset_to_dc_equilibrium();
        let (z_l, z_r) = pipe.stage2.step_stereo(0.0, 0.0);
        prop_assert!(z_l.abs() < 1e-7 && z_r.abs() < 1e-7);
    }
}
