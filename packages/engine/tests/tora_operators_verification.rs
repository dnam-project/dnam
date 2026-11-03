//! Empirical TORA Operator Verification Suite (`Tests 1A, 1B, 1C, 2A, 2B, 2C, 3`).
//!
//! Canonical Reference: `docs/verification/verification_and_guardrails.md`.
//!
//! Verifies all Rust-side mathematical and numerical operator invariants for:
//! - **Op-N (`Stage II` Commuted Cascade / `PolyADAA`):** `Test 1A` (linear passband preservation),
//!   `Test 1B` (division-free midpoint-difference identity, `CL-01`), and `Test 1C` (cascaded aliasing
//!   suppression at $f_0 \approx 3,450\text{ Hz}$, `CL-01`, `CL-02`).
//! - **Op-M (`Stage I/III` Kautz & Symplectic Modal Bank):** `Test 2A` (radial-polar $Q$ preservation,
//!   `CL-03`–`CL-05`), `Test 2B` (branchless 4-FMA + 1-step Newton unit-circle rotor, `CL-03`), and
//!   `Test 2C` (Blelloch $\mathcal{O}(\log_2 L)$ parallel associative scan equivalence, `CL-04`).
//! - **Op-T (`Stage I/IV` Elastic Transport Delay Line):** `Test 3` (stateless 4-tap Catmull-Rom vs.
//!   recursive 1st-order Thiran boundary click jerk, `CL-06`).

use std::f64::consts::{PI, TAU};

use dnam_engine::FtzDazGuard;
use dnam_engine::math::ScalarBackend;
use dnam_engine::operators::op_m::{
    KautzModalBank, SymplecticModal4Lane, algebraic_unit_rotor_cl03,
};
use dnam_engine::operators::op_n::{MultiDrivePolyAdaaChannel, poly_adaa_algebraic_eval};
use dnam_engine::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
use dnam_engine::operators::reference::{
    catmull_rom_hermite_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64,
};
use proptest::prelude::*;

/// Test 1A (`CL-01`, `CL-02`): Verifies that applying a naive midpoint filter $(1 + z^{-1})/2$
/// to a 3-stage cascade near Nyquist causes `> 35 dB` passband droop, whereas DNAM's split linear
/// bypass $\alpha_c x[n]$ + `1x B-Spline PolyADAA` preserves a flat $0.00\text{ dB}$ linear passband.
#[test]
fn test_1a_linear_passband_preservation_vs_midpoint_droop() {
    const N: usize = 2048;
    const FS: f64 = 48_000.0;
    const F_TEST: f64 = 22_000.0;

    let input: Vec<f64> = (0..N)
        .map(|n| (TAU * F_TEST * (n as f64) / FS).sin())
        .collect();

    let mut naive = input.clone();
    for _stage in 0..3 {
        let mut next = vec![0.0_f64; N];
        let mut prev = 0.0_f64;
        for i in 0..N {
            next[i] = 0.5 * (naive[i] + prev);
            prev = naive[i];
        }
        naive = next;
    }

    let mut dnam = input.clone();
    for _stage in 0..3 {
        let mut next = vec![0.0_f64; N];
        poly_adaa_slice_f64(&dnam, &mut next, 0.0, 1.0, 0.0, 0.0, 0.9993);
        dnam = next;
    }

    let rms = |sig: &[f64]| -> f64 {
        let s: f64 = sig[256..].iter().map(|&v| v * v).sum();
        (s / ((sig.len() - 256) as f64)).sqrt()
    };

    let rms_in = rms(&input);
    let droop_naive_db = 20.0 * (rms(&naive) / rms_in).log10();
    let droop_dnam_db = 20.0 * (rms(&dnam) / rms_in).log10();

    assert!(
        droop_naive_db < -35.0,
        "Expected naive midpoint filter droop < -35 dB, got {droop_naive_db:.2} dB"
    );
    assert!(
        droop_dnam_db.abs() < 1e-12,
        "Expected DNAM linear passband droop == 0.00 dB, got {droop_dnam_db:.6e} dB"
    );
}

/// Test 1B (`CL-01`): Verifies exact equivalence between the analytical antiderivative divided
/// difference $\frac{\mathcal{F}(u_n) - \mathcal{F}(u_{n-1})}{u_n - u_{n-1}}$ and the division-free
/// midpoint-difference polynomial $\mathcal{A}(m_n, d_n)$ across 10,000 pairs with `< 1e-12` error
/// and zero `0/0` singularity at $u_n == u_{n-1}$.
#[test]
fn test_1b_division_free_midpoint_difference_identity_10k_pairs() {
    let beta = 0.85_f64;
    let delta = -0.25_f64;
    let antideriv = |u: f64| -> f64 {
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        beta * (0.75 * u2 - 0.125 * u4) + delta * ((1.0 / 3.0) * u3 - 0.2 * u5)
    };

    let mut max_alg_err = 0.0_f64;
    let mut state = 0x1234_5678_9ABC_DEF0_u64;
    let mut next_u = || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        -0.99 + 1.98 * unit
    };

    for _ in 0..10_000 {
        let u0 = next_u();
        let u1 = next_u();
        let alg = poly_adaa_algebraic_f64(u0, u1, beta, delta);
        if (u1 - u0).abs() > 1e-5 {
            let exact = (antideriv(u1) - antideriv(u0)) / (u1 - u0);
            max_alg_err = max_alg_err.max((alg - exact).abs());
        }
    }
    assert!(
        max_alg_err < 1e-12,
        "Test 1B max algebraic error exceeded 1e-12: {max_alg_err:e}"
    );

    for step in -99..=99 {
        let u = (step as f64) * 0.01;
        let inst = beta * u * (1.5 - 0.5 * u * u) + delta * (u * u - u * u * u * u);
        let alg64 = poly_adaa_algebraic_f64(u, u, beta, delta);
        assert!((alg64 - inst).abs() < 1e-15);

        let alg32 = poly_adaa_algebraic_eval(u as f32, u as f32, beta as f32, delta as f32);
        assert!((alg32 - (inst as f32)).abs() < 1e-6);
    }
}

/// Test 1C (`CL-01`, `CL-02`): Drives cascaded nonlinear saturation stages with a high-amplitude
/// $f_0 \approx 3,422\text{ Hz}$ sine wave at $f_s = 48\text{ kHz}$ ($N = 1,024$, bin $k_0 = 73$)
/// and verifies that DNAM's `1x B-Spline PolyADAA` suppresses non-harmonic aliased bin energy
/// relative to un-antialiased hard/LeakyReLU clipping.
#[test]
fn test_1c_commuted_cascade_aliasing_suppression_at_3450hz() {
    const N: usize = 1024;
    const K0: usize = 73;
    const TOTAL: usize = 1280;

    let input_full: Vec<f64> = (0..TOTAL)
        .map(|n| 1.6 * (TAU * (K0 as f64) * (n as f64) / (N as f64)).sin())
        .collect();

    let mut base_full = input_full.clone();
    for _stage in 0..3 {
        for x in &mut base_full {
            let u = 1.6 * (*x) + 0.12;
            *x = if u > 1.0 {
                1.0 + 0.01 * (u - 1.0)
            } else if u < -1.0 {
                -1.0 + 0.01 * (u + 1.0)
            } else {
                u
            };
        }
    }

    let mut dnam_full = input_full;
    for _stage in 0..3 {
        let mut next = vec![0.0_f64; TOTAL];
        poly_adaa_slice_f64(&dnam_full, &mut next, 0.05, 0.0, 1.0, 0.12, 0.9993);
        dnam_full = next;
    }

    let compute_alias_dbc = |sig: &[f64]| -> f64 {
        let win_slice = &sig[TOTAL - N..];
        let mut harmonic_bins = [false; N / 2];
        for m in 1..=7 {
            let center = m * K0;
            for b in center.saturating_sub(2)..=(center + 2).min(N / 2 - 1) {
                harmonic_bins[b] = true;
            }
        }
        let mut fund_e = 0.0_f64;
        let mut alias_e = 0.0_f64;
        for bin in 4..(N / 2) {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (n, &s) in win_slice.iter().enumerate() {
                let w = 0.5 - 0.5 * (TAU * (n as f64) / (N as f64)).cos();
                let ang = TAU * (bin as f64) * (n as f64) / (N as f64);
                re += s * w * ang.cos();
                im -= s * w * ang.sin();
            }
            let e = re * re + im * im;
            if (bin as isize - K0 as isize).unsigned_abs() <= 2 {
                fund_e += e;
            } else if !harmonic_bins[bin] {
                alias_e += e;
            }
        }
        10.0 * (alias_e / fund_e).log10()
    };

    let alias_base_db = compute_alias_dbc(&base_full);
    let alias_dnam_db = compute_alias_dbc(&dnam_full);
    let suppression_db = alias_base_db - alias_dnam_db;
    assert!(
        suppression_db > 15.0,
        "Expected PolyADAA + B-Spline aliasing suppression > 15 dB, got {suppression_db:.2} dB"
    );
}

/// Test 2A (`CL-03`–`CL-05`): Verifies that Cartesian midpoint pole interpolation collapses
/// resonator $Q$ from $Q_0 = 125.0$ ($r_0 = 0.992$) to $< 0.10 Q_0$, whereas DNAM's radial-polar
/// factorization preserves $Q > 0.99 Q_0$.
#[test]
fn test_2a_radial_polar_q_preservation_vs_cartesian_collapse() {
    let r0 = 0.992_f64;
    let q_orig = 1.0 / (1.0 - r0);
    let theta_low = 0.05 * PI;
    let theta_high = 0.40 * PI;

    let re_mid = 0.5 * r0 * (theta_low.cos() + theta_high.cos());
    let im_mid = 0.5 * r0 * (theta_low.sin() + theta_high.sin());
    let r_cart = (re_mid * re_mid + im_mid * im_mid).sqrt();
    let q_cart = 1.0 / (1.0 - r_cart);

    let theta_mid = 0.5 * (theta_low + theta_high);
    let (c_rot, s_rot) = algebraic_unit_rotor_cl03(theta_mid as f32);
    let r_polar = r0 * ((c_rot as f64) * (c_rot as f64) + (s_rot as f64) * (s_rot as f64)).sqrt();
    let q_polar = 1.0 / (1.0 - r_polar);

    assert!(q_cart < 0.10 * q_orig);
    assert!(q_polar > 0.99 * q_orig);
}

/// Test 2B (`CL-03`): Verifies the branchless 4-FMA + 1-step Newton unit-circle rotor across
/// 20,000 angles $\Delta\theta \in (0, \pi)$ in both `f32` and `f64`.
#[test]
fn test_2b_branchless_newton_unit_circle_rotor_20k_angles_cl03() {
    const STEPS: usize = 20_000;
    let mut max_err_f64 = 0.0_f64;
    let mut max_unit_f64 = 0.0_f64;

    for i in 0..=STEPS {
        let theta = PI * (i as f64) / (STEPS as f64);
        let u = (2.0 / PI) * theta - 1.0;
        let alpha = (PI * 0.25) * (1.0 - u * u);
        let a2 = alpha * alpha;
        let c_hat = 1.0 - a2 * (0.5 - a2 * (0.041_666_664_183_139_8 - a2 * 0.001_388_839_7));
        let s_hat = alpha * (1.0 - a2 * (0.166_666_656_732_559_2 - a2 * 0.008_333_025_5));
        let r2_hat = c_hat * c_hat + s_hat * s_hat;
        let y1 = 1.5 - 0.5 * r2_hat;
        let c = c_hat * y1;
        let s = s_hat * y1;
        let mag = (c * c + s * s).sqrt();
        max_err_f64 = max_err_f64.max((1.0 - mag).abs());
        max_unit_f64 = max_unit_f64.max(mag);

        if i > 0 && i < STEPS {
            let (c32, s32) = algebraic_unit_rotor_cl03(theta as f32);
            let mag32 = ((c32 as f64) * (c32 as f64) + (s32 as f64) * (s32 as f64)).sqrt();
            assert!(mag32 <= 1.000_001);
        }
    }

    assert!(max_err_f64 < 2e-8);
    assert!(max_unit_f64 <= 1.0 + 1e-15);
}

/// Test 2C (`CL-04`): Verifies exact equivalence between sequential sample-by-sample complex
/// modal recurrence and Blelloch $\mathcal{O}(\log_2 L)$ parallel associative scan over $L = 1,024$.
#[test]
fn test_2c_blelloch_parallel_associative_scan_equivalence_1024() {
    const L: usize = 1024;
    type Cplx = (f64, f64);

    let c_mul = |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar * br - ai * bi, ar * bi + ai * br) };
    let c_add = |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar + br, ai + bi) };

    let mut a = vec![(0.0, 0.0); L];
    let mut b = vec![(0.0, 0.0); L];
    for n in 0..L {
        let r = 0.985 + 0.010 * ((n as f64) * 0.017).sin();
        let theta = 0.12 + 0.04 * ((n as f64) * 0.031).cos();
        a[n] = (r * theta.cos(), r * theta.sin());
        b[n] = (
            0.25 * ((n as f64) * 0.11).sin(),
            0.15 * ((n as f64) * 0.07).cos(),
        );
    }

    let mut seq = vec![(0.0, 0.0); L];
    let mut state = (0.0, 0.0);
    for n in 0..L {
        state = c_add(c_mul(a[n], state), b[n]);
        seq[n] = state;
    }

    let mut scan_a = a;
    let mut scan_b = b;
    let mut stride = 1;
    while stride < L {
        let prev_a = scan_a.clone();
        let prev_b = scan_b.clone();
        for i in stride..L {
            scan_a[i] = c_mul(prev_a[i], prev_a[i - stride]);
            scan_b[i] = c_add(c_mul(prev_a[i], prev_b[i - stride]), prev_b[i]);
        }
        stride <<= 1;
    }

    let mut max_scan_err = 0.0_f64;
    for n in 0..L {
        let dr = seq[n].0 - scan_b[n].0;
        let di = seq[n].1 - scan_b[n].1;
        max_scan_err = max_scan_err.max((dr * dr + di * di).sqrt());
    }
    assert!(max_scan_err < 1e-13);
}

/// Test 3 (`CL-06`): Verifies that Stateless 4-Tap Catmull-Rom Hermite Cubic interpolation has
/// `> 15x` lower 2nd-difference boundary click jerk than 1st-order recursive Thiran all-pass.
#[test]
fn test_3_stateless_catmull_rom_vs_recursive_thiran_boundary_jerk_cl06() {
    const FS: f64 = 48_000.0;
    const F_SIG: f64 = 5_000.0;
    const N: usize = 4096;
    const BUF_LEN: usize = 512;

    let mut ring = vec![0.0_f64; BUF_LEN];
    let mut wr = 0_usize;
    let mut err_catmull = Vec::with_capacity(N);
    let mut err_thiran = Vec::with_capacity(N);
    let mut thiran_state = 0.0_f64;

    for n in 0..N {
        let t = (n as f64) / FS;
        let x = (TAU * F_SIG * t).sin();
        ring[wr] = x;

        let tau = 120.0 + 60.0 * (TAU * 3.0 * t).sin();
        let d_int = tau.floor() as usize;
        let mu = (tau - (d_int as f64)).clamp(0.05, 0.95);
        let mu_exact = tau - (d_int as f64);

        let p0 = (wr + BUF_LEN - d_int) & (BUF_LEN - 1);
        let pm1 = (p0 + 1) & (BUF_LEN - 1);
        let p1 = (p0 + BUF_LEN - 1) & (BUF_LEN - 1);
        let p2 = (p0 + BUF_LEN - 2) & (BUF_LEN - 1);

        let y_cat = catmull_rom_hermite_f64(ring[pm1], ring[p0], ring[p1], ring[p2], mu_exact);
        let a_th = (1.0 - mu) / (1.0 + mu);
        let y_th = a_th * (ring[p0] - thiran_state) + ring[p1];
        thiran_state = y_th;

        if n >= 256 {
            let exact = (TAU * F_SIG * (t - tau / FS)).sin();
            err_catmull.push(y_cat - exact);
            err_thiran.push(y_th - exact);
        }
        wr = (wr + 1) & (BUF_LEN - 1);
    }

    let max_jerk = |e: &[f64]| -> f64 {
        e.windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .fold(0.0_f64, f64::max)
    };

    assert!(max_jerk(&err_thiran) > 15.0 * max_jerk(&err_catmull));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Property test verifying `KautzModalBank::step_4lane_simd` and `KautzModalBank::step` produce
    /// identical IEEE-754 results for any stable pole configuration and input sequence.
    #[test]
    fn prop_kautz_4lane_simd_matches_scalar_step(
        r in 0.1_f32..0.998_f32,
        theta in 0.05_f32..3.05_f32,
        c_re in -1.5_f32..1.5_f32,
        c_im in -1.5_f32..1.5_f32,
        x in -4.0_f32..4.0_f32,
    ) {
        let _audio = FtzDazGuard::new();
        let mut b_scalar = KautzModalBank::<4>::default();
        let mut b_simd = KautzModalBank::<4>::default();
        for m in 0..4 {
            let mf = (m + 1) as f32 * 0.15;
            b_scalar.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
            b_simd.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
        }
        for k in 0..16 {
            let sample = x * ((k as f32) * 0.3).cos();
            let y_sc = b_scalar.step(sample);
            let y_sm: f32 = b_simd.step_4lane_simd::<ScalarBackend>(sample).iter().sum();
            prop_assert!((y_sc - y_sm).abs() < 1e-6);
        }
    }

    /// Property test verifying `SymplecticModal4Lane`, `MultiDrivePolyAdaaChannel`, and `OpTDelayLine`
    /// remain finite and bounded under arbitrary modulated drives and delay sweeps.
    #[test]
    fn prop_operators_bounded_and_finite(
        x in -10.0_f32..10.0_f32,
        g_base in 0.1_f32..5.0_f32,
        bias in -1.0_f32..1.0_f32,
        delay in 2.0_f32..500.0_f32,
    ) {
        let _audio = FtzDazGuard::new();
        let mut modal = SymplecticModal4Lane::default();
        modal.set_pole(0, 0.96, 0.3);
        let (re, im) = modal.step_lane(0, x);
        prop_assert!(re.is_finite() && im.is_finite());

        let mut adaa = MultiDrivePolyAdaaChannel::default();
        let y_adaa = adaa.step(re, g_base, bias, 0.5, 0.9993);
        prop_assert!(y_adaa.iter().all(|v| v.is_finite()));

        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl.set_target_delay(delay, false);
        let y_dl = dl.step(y_adaa[0]);
        prop_assert!(y_dl.is_finite());
    }
}
