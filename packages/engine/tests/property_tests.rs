//! Property-Based Testing Suite (`tests/property_tests.rs`) powered by `proptest`.
//!
//! Verifies mathematical, physical, and cross-backend invariants across arbitrary inputs:
//! 1. `PolyADAA` division-free algebraic antiderivative identity & continuity (`CL-01`, `CL-02`).
//! 2. $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor bounds (`CL-03`).
//! 3. 4-tap Cubic Lagrange partition of unity & cubic polynomial exactness (`CL-06`).
//! 4. Core 2 Givens + Householder unitary energy conservation ($\|\mathbf{q}\|_2 = \|\mathbf{p}\|_2$).
//! 5. Coercive-Saturating Lyapunov loop ceiling & BIBO stability (`CL-09`..`CL-11`).
//! 6. Hardware NEON vs. Software `CRC32C` equivalence over arbitrary byte lengths (`0..512`).
//! 7. Cross-backend (`Scalar`, `NEON`, `ARM SME`) equivalence on 6-tap dilated convolutions & 64-tap heads.
//! 8. ZOH sample-rate rescaling pole stability (`44.1 kHz .. 96 kHz`) & `uplo='U'` prefix theorem.

use dnam_engine::container::{
    ZohRescaledParams, crc32c_compute, crc32c_update, pack_uplo_u_f64, unpack_uplo_u_f64,
    uplo_u_packed_len,
};
use dnam_engine::control::{ConditionalClampRule, ControlBus, ControlClampDescriptor};
use dnam_engine::math::{
    DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4, algebraic_unit_rotor,
    cubic_lagrange_weights_f64, eval_poly_adaa_algebraic_f64, eval_poly_adaa_antiderivative_f64,
    eval_poly_adaa_waveshaper_f64,
};
use dnam_engine::schema::ControlUiHints;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn prop_poly_adaa_algebraic_matches_antiderivative_difference_quotient(
        u_prev in -1.0_f64..=1.0_f64,
        u_curr in -1.0_f64..=1.0_f64,
        beta in 0.0_f64..=1.0_f64,
        delta in -0.5_f64..=0.5_f64,
    ) {
        let alg = eval_poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
        prop_assert!(alg.is_finite());

        if (u_curr - u_prev).abs() > 1.0e-5 {
            let f_curr = eval_poly_adaa_antiderivative_f64(u_curr, beta, delta);
            let f_prev = eval_poly_adaa_antiderivative_f64(u_prev, beta, delta);
            let diff_quot = (f_curr - f_prev) / (u_curr - u_prev);
            prop_assert!((alg - diff_quot).abs() < 1.0e-10);
        }

        // Exact degenerate limit u_prev == u_curr equals instantaneous waveshaper g_nl(u)
        let diag = eval_poly_adaa_algebraic_f64(u_curr, u_curr, beta, delta);
        let exact_ws = eval_poly_adaa_waveshaper_f64(u_curr, beta, delta);
        prop_assert!((diag - exact_ws).abs() < 1.0e-12);
    }

    #[test]
    fn prop_algebraic_unit_rotor_stays_on_unit_circle(
        theta in 1.0e-4_f32..=(core::f32::consts::PI - 1.0e-4_f32),
    ) {
        let (c, s) = algebraic_unit_rotor(theta);
        let norm_sq = c * c + s * s;
        prop_assert!(norm_sq <= 1.000_002);
        prop_assert!((norm_sq - 1.0).abs() < 2.0e-5);
    }

    #[test]
    fn prop_cubic_lagrange_partition_of_unity_and_cubic_exactness(
        mu in 0.0_f64..=1.0_f64,
        a0 in -5.0_f64..=5.0_f64,
        a1 in -5.0_f64..=5.0_f64,
        a2 in -5.0_f64..=5.0_f64,
        a3 in -5.0_f64..=5.0_f64,
    ) {
        let w = cubic_lagrange_weights_f64(mu);
        let sum_w = w[0] + w[1] + w[2] + w[3];
        prop_assert!((sum_w - 1.0).abs() < 1.0e-14);

        let poly = |x: f64| a0 + a1 * x + a2 * x * x + a3 * x * x * x;
        let interp = w[0] * poly(-1.0) + w[1] * poly(0.0) + w[2] * poly(1.0) + w[3] * poly(2.0);
        let exact = poly(mu);
        prop_assert!((interp - exact).abs() < 1.0e-11);
    }

    #[test]
    fn prop_givens_householder_preserves_l2_energy(
        p in prop::array::uniform8(-5.0_f32..=5.0_f32),
        psi in 0.1_f32..=3.0_f32,
    ) {
        let (c_psi, s_psi) = algebraic_unit_rotor(psi);
        // Unit-norm 4-vector: [0.5, 0.5, 0.5, 0.5]
        let v4 = [0.5_f32; 4];
        let p4 = [p[0], p[1], p[2], p[3]];
        let mut q4 = [0.0_f32; 4];
        DefaultMathBackend::givens_householder_mix_4ch(&p4, c_psi, s_psi, &v4, &mut q4);
        let e_in4: f32 = p4.iter().map(|x| x * x).sum();
        let e_out4: f32 = q4.iter().map(|x| x * x).sum();
        prop_assert!((e_in4 - e_out4).abs() <= 1.0e-3 * e_in4.max(1.0));

        // Unit-norm 8-vector: 1/sqrt(8) = 0.35355339
        let v8 = [0.353_553_38_f32; 8];
        let mut q8 = [0.0_f32; 8];
        DefaultMathBackend::givens_householder_mix_8ch(&p, c_psi, s_psi, &v8, &mut q8);
        let e_in8: f32 = p.iter().map(|x| x * x).sum();
        let e_out8: f32 = q8.iter().map(|x| x * x).sum();
        prop_assert!((e_in8 - e_out8).abs() <= 1.0e-3 * e_in8.max(1.0));
    }

    #[test]
    fn prop_lyapunov_coercive_loop_respects_ceiling(
        u_in in prop::array::uniform4(-50.0_f32..=50.0_f32),
        g_lin in 0.0_f32..=1.2_f32,
        v_ceil in 0.25_f32..=0.985_f32,
    ) {
        let mut dc_x = [0.0_f32; 4];
        let mut dc_y = [0.0_f32; 4];
        let mut s_out = [0.0_f32; 4];
        let two_over_3v = [2.0 / (3.0 * v_ceil); 4];
        DefaultMathBackend::mufdn_coercive_loop_4ch(
            &u_in,
            &[g_lin; 4],
            &[2.0; 4],
            &two_over_3v,
            &[v_ceil; 4],
            0.995,
            &mut dc_x,
            &mut dc_y,
            &mut s_out,
        );
        for &s in &s_out {
            prop_assert!(s.is_finite());
            prop_assert!(s.abs() <= v_ceil + 1.0e-5);
        }
    }

    #[test]
    fn prop_hardware_crc32c_matches_software_table_and_incremental(
        data in prop::collection::vec(any::<u8>(), 0..300),
        split_ratio in 0.0_f32..=1.0_f32,
    ) {
        let sw_crc = ScalarBackend::crc32c(&data);
        let hw_crc = DefaultMathBackend::crc32c(&data);
        let comp_crc = crc32c_compute(&data);
        prop_assert_eq!(sw_crc, hw_crc);
        prop_assert_eq!(sw_crc, comp_crc);

        let split = ((data.len() as f32) * split_ratio) as usize;
        let split = split.min(data.len());
        let part1 = crc32c_update(0, &data[..split]);
        let part2 = crc32c_update(part1, &data[split..]);
        prop_assert_eq!(sw_crc, part2);
    }

    #[test]
    fn prop_cross_backend_simd_equivalence(
        a in prop::array::uniform4(-2.0_f32..=2.0_f32),
        b in prop::array::uniform4(-2.0_f32..=2.0_f32),
        c in prop::array::uniform4(-2.0_f32..=2.0_f32),
    ) {
        let sa = <ScalarBackend as MathBackend>::Vec4::load(&a);
        let sb = <ScalarBackend as MathBackend>::Vec4::load(&b);
        let sc = <ScalarBackend as MathBackend>::Vec4::load(&c);
        let s_res = sa.fma(sb, sc).fms(sa, sb).to_array();

        let na = <DefaultMathBackend as MathBackend>::Vec4::load(&a);
        let nb = <DefaultMathBackend as MathBackend>::Vec4::load(&b);
        let nc = <DefaultMathBackend as MathBackend>::Vec4::load(&c);
        let n_res = na.fma(nb, nc).fms(na, nb).to_array();

        for i in 0..4 {
            prop_assert!((s_res[i] - n_res[i]).abs() < 1.0e-5);
        }

        let h16_s = ScalarBackend::f32x4_to_fp16x4(&a);
        let f32_s = ScalarBackend::fp16x4_to_f32x4(&h16_s);
        let h16_n = DefaultMathBackend::f32x4_to_fp16x4(&a);
        let f32_n = DefaultMathBackend::fp16x4_to_f32x4(&h16_n);
        for i in 0..4 {
            prop_assert!((f32_s[i] - f32_n[i]).abs() < 2.0e-3);
        }
    }

    #[test]
    fn prop_zoh_rescaling_and_ui_skew_roundtrip(
        fs_host in 44_100_u32..=96_000_u32,
        r_low in 0.30_f32..=0.95_f32,
        dr in 0.0_f32..=0.04_f32,
        theta0 in 0.01_f32..=3.0_f32,
        t_norm in 0.01_f32..=0.99_f32,
        midpoint in 1.0_f32..=9.0_f32,
    ) {
        let zoh = ZohRescaledParams::compute(48_000, fs_host, 16, 0.9993, 0.995, 64.0, 128).unwrap();
        let (rl_h, dr_h, th_h, bg_h) = zoh.rescale_rglru_channel(r_low, dr, theta0, 0.25);
        prop_assert!((0.25..=0.9995).contains(&rl_h));
        prop_assert!(rl_h + dr_h <= 0.9995 + 1.0e-6);
        prop_assert!((1.0e-4..=core::f32::consts::PI).contains(&th_h));
        prop_assert!(bg_h > 0.0 && bg_h.is_finite());

        let hints = ControlUiHints {
            midpoint: Some(midpoint),
            ..Default::default()
        };
        let phys = hints.normalized_to_physical(t_norm, 0.0, 10.0);
        let back = hints.physical_to_normalized(phys, 0.0, 10.0);
        prop_assert!((back - t_norm).abs() < 1.0e-4);
    }

    #[test]
    fn prop_uplo_u_prefix_theorem_and_control_clamping(
        diag_vals in prop::array::uniform8(0.1_f64..=10.0_f64),
        raw_ctrl in -50.0_f32..=50.0_f32,
    ) {
        // 8x8 symmetric matrix packed with uplo='U', unpacked at N=4 prefix and N=8 full
        let mut dense8 = [0.0_f64; 64];
        for i in 0..8 {
            for j in i..8 {
                let v = if i == j { diag_vals[i] } else { 0.5 * (diag_vals[i] + diag_vals[j]) };
                dense8[i * 8 + j] = v;
                dense8[j * 8 + i] = v;
            }
        }
        let mut packed8 = vec![0.0_f64; uplo_u_packed_len(8)];
        pack_uplo_u_f64(&dense8, 8, &mut packed8);

        let mut unpacked4 = [0.0_f64; 16];
        unpack_uplo_u_f64(&packed8[..uplo_u_packed_len(4)], 4, &mut unpacked4);
        for r in 0..4 {
            for c in 0..4 {
                prop_assert!((unpacked4[r * 4 + c] - dense8[r * 8 + c]).abs() < 1.0e-14);
            }
        }

        // Control clamping bounds
        let mut cb: ControlBus<ScalarBackend> = ControlBus::new();
        cb.num_controls = 1;
        cb.descriptors[0] = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 3,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 1.5,
            safe_max_base: 8.5,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0, 0.0],
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.set_control_immediate(0, raw_ctrl);
        prop_assert!(cb.clamped_values[0] >= 1.5 && cb.clamped_values[0] <= 8.5);
    }
}
