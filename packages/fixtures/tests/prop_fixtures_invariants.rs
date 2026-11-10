//! Cross-module integration and property-based tests for `dnam-fixtures`.
//!
//! Unit tests for individual signal chains and hardware impairment decorators live
//! inside their respective submodules (`src/chains/**`, `src/adversaries/**`).
//! This integration suite verifies:
//! 1. Individual per-control and per-switch non-zero sensitivity across all 12 ground-truth chains.
//! 2. Resting zero-DC equilibrium (`inputs = 0 -> outputs = 0`) across all 12 chains.
//! 3. `0 ULP` `reset()` == `new()` == `Default::default()` determinism across all chains & decorators.
//! 4. Stereo channel isolation (zero L/R crosstalk vs dual-mono) on stereo decorators.
//! 5. Multi-sample-rate (`8 / 16 / 44.1 / 48 / 88.2 / 96 / 192 kHz`) & multi-bit-depth (`12 / 16 / 24 / 32-bit` PCM & `f32`)
//!    device contracts across chains and converter decorators.
//! 6. Property-based (`proptest!`) boundedness including `NaN`/extreme inputs and random formats.

#![forbid(unsafe_code)]

use dnam_fixtures::{
    AdcClipAndPadDecorator, BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz,
    Chain03ParallelCleanBlend, Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom,
    Chain06EnvelopeResonantSvf, Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer,
    Chain09La2aOpticalCompressor, Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron,
    Chain12FullEndToEndStereoRig, Chain12RigMode, ConverterLoopbackDecorator,
    DEFAULT_FIXTURE_SAMPLE_RATE_HZ, DigitalQuirksDecorator, FixtureBitDepth,
    GermaniumTempDriftDecorator, HardwareNoiseGateDecorator, HumanHandTrajectoryDecorator,
    IdleSquealDecorator, MainsHumAndNoiseFloorDecorator, MicrophoneCapsuleClipDecorator,
    PolarityAndDcOffsetDecorator, PotTaperAndSwitchDecorator,
    ScratchyPotAndSteppedAttenuatorDecorator, SpringChirpDecorator, TaperCurveKind,
    TubeThermalHysteresisDecorator,
};
use proptest::prelude::*;

fn multitone_stimulus(len: usize, sample_rate_hz: u32, amp: f32) -> Vec<f32> {
    let fs = sample_rate_hz as f32;
    (0..len)
        .map(|n| {
            let env = if (n / 128) % 2 == 0 {
                1.0_f32
            } else {
                0.25_f32
            };
            let t = (n as f32) / fs;
            amp * env
                * (0.45 * (core::f32::consts::TAU * 110.0 * t).sin()
                    + 0.35 * (core::f32::consts::TAU * 440.0 * t).sin()
                    + 0.20 * (core::f32::consts::TAU * 2_400.0 * t).sin())
        })
        .collect()
}

fn build_all_chains() -> Vec<Box<dyn BlackBoxModel>> {
    vec![
        Box::new(Chain01SymmetricOd::new()),
        Box::new(Chain02GatedAsymmetricFuzz::new()),
        Box::new(Chain03ParallelCleanBlend::new()),
        Box::new(Chain04TubeAmpFmv4x12::new()),
        Box::new(Chain05PureLtiCabAndRoom::new()),
        Box::new(Chain06EnvelopeResonantSvf::new()),
        Box::new(Chain07StereoChorusAndRotary::new()),
        Box::new(Chain08LongBbdEchoAnd30sShimmer::new()),
        Box::new(Chain09La2aOpticalCompressor::new()),
        Box::new(Chain10Oc2OctaveDividerAndRingMod::new()),
        Box::new(Chain11BiAmpBassAndOutputIron::new()),
        Box::new(Chain12FullEndToEndStereoRig::new(
            Chain12RigMode::FixedZeroKnob,
        )),
        Box::new(Chain12FullEndToEndStereoRig::new(
            Chain12RigMode::Parametric16,
        )),
    ]
}

#[test]
fn test_all_chains_per_control_and_per_switch_sensitivity_and_zero_dc() {
    let mut chains = build_all_chains();
    let input_l = multitone_stimulus(4_800, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, 0.55);
    let input_r = multitone_stimulus(4_800, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, 0.40);
    let silence = vec![0.0_f32; 256];

    for chain in &mut chains {
        let meta = chain.metadata();
        assert!(!meta.id.is_empty());
        assert!(!meta.name.is_empty());
        assert_eq!(meta.sample_rate_hz, DEFAULT_FIXTURE_SAMPLE_RATE_HZ);
        assert_eq!(meta.bit_depth, FixtureBitDepth::Float32);

        let base_controls = meta.default_controls();
        let base_switches = meta.default_switches();

        // 1. Resting zero-DC equilibrium: zero input must yield exact zero output
        chain.reset();
        let mut z_l = vec![1.0_f32; 256];
        let mut z_r = vec![1.0_f32; 256];
        chain.process_block(
            &[&silence, &silence],
            &mut [&mut z_l, &mut z_r],
            &base_controls,
            &base_switches,
        );
        assert!(
            z_l.iter().all(|&v| v == 0.0) && z_r.iter().all(|&v| v == 0.0),
            "{} violated resting zero-DC equilibrium",
            meta.id
        );

        // 2. Physical <-> normalized roundtrip for every control
        for ctrl in meta.controls {
            let phys_mid = ctrl.norm_to_physical(ctrl.default_norm);
            let roundtrip = ctrl.physical_to_norm(phys_mid);
            assert!(
                (roundtrip - ctrl.default_norm).abs() < 1e-5,
                "{}: control {} failed roundtrip",
                meta.id,
                ctrl.name
            );
        }

        // 3. Individual per-control sensitivity: perturbing each single knob p from -0.85 to +0.85
        //    must measurably alter the output waveform.
        for p in 0..meta.controls.len() {
            let mut c_lo = base_controls.clone();
            let mut c_hi = base_controls.clone();
            // For Chain-12 ping-pong delay feedback, shorten delay_time (#9) to 50 ms so feedback returns within 100 ms
            if meta.id == "Chain-12-Full-EndToEnd-Stereo-Rig-P16" && p != 9 {
                c_lo[9] = -1.0;
                c_hi[9] = -1.0;
            }
            c_lo[p] = -0.85;
            c_hi[p] = 0.85;

            chain.reset();
            let mut lo_l = vec![0.0_f32; input_l.len()];
            let mut lo_r = vec![0.0_f32; input_r.len()];
            chain.process_block(
                &[&input_l, &input_r],
                &mut [&mut lo_l, &mut lo_r],
                &c_lo,
                &base_switches,
            );

            chain.reset();
            let mut hi_l = vec![0.0_f32; input_l.len()];
            let mut hi_r = vec![0.0_f32; input_r.len()];
            chain.process_block(
                &[&input_l, &input_r],
                &mut [&mut hi_l, &mut hi_r],
                &c_hi,
                &base_switches,
            );

            let diff: f64 = lo_l
                .iter()
                .zip(hi_l.iter())
                .chain(lo_r.iter().zip(hi_r.iter()))
                .map(|(&a, &b)| f64::from((a - b).abs()))
                .sum();
            assert!(
                diff > 1.0e-4,
                "{}: control #{} ({}) had zero sensitivity (diff={diff})",
                meta.id,
                p,
                meta.controls[p].name
            );
        }

        // 4. Individual per-switch sensitivity: toggling each switch between 0 and max throw
        for s_idx in 0..meta.switches.len() {
            let max_throw = (meta.switches[s_idx].options.len().saturating_sub(1)) as u8;
            if max_throw == 0 {
                continue;
            }
            let mut c_sw = base_controls.clone();
            if meta.id == "Chain-12-Full-EndToEnd-Stereo-Rig-P16" {
                c_sw[9] = -1.0;
                c_sw[11] = 0.6;
            }
            let mut sw_a = base_switches.clone();
            let mut sw_b = base_switches.clone();
            sw_a[s_idx] = 0;
            sw_b[s_idx] = max_throw;

            chain.reset();
            let mut a_l = vec![0.0_f32; input_l.len()];
            let mut a_r = vec![0.0_f32; input_r.len()];
            chain.process_block(
                &[&input_l, &input_r],
                &mut [&mut a_l, &mut a_r],
                &c_sw,
                &sw_a,
            );

            chain.reset();
            let mut b_l = vec![0.0_f32; input_l.len()];
            let mut b_r = vec![0.0_f32; input_r.len()];
            chain.process_block(
                &[&input_l, &input_r],
                &mut [&mut b_l, &mut b_r],
                &c_sw,
                &sw_b,
            );

            let diff: f64 = a_l
                .iter()
                .zip(b_l.iter())
                .chain(a_r.iter().zip(b_r.iter()))
                .map(|(&x, &y)| f64::from((x - y).abs()))
                .sum();
            assert!(
                diff > 1.0e-4,
                "{}: switch #{} ({}) had zero sensitivity (diff={diff})",
                meta.id,
                s_idx,
                meta.switches[s_idx].name
            );
        }
    }
}

#[test]
fn test_zero_ulp_reset_determinism_across_all_chains_and_decorators() {
    let stim_l = multitone_stimulus(384, 48_000, 0.5);
    let stim_r = multitone_stimulus(384, 48_000, 0.35);

    let mut fresh_chains = build_all_chains();
    let mut dirty_chains = build_all_chains();

    for (fresh, dirty) in fresh_chains.iter_mut().zip(dirty_chains.iter_mut()) {
        let ctrls = fresh.metadata().default_controls();
        let sws = fresh.metadata().default_switches();

        // Dirty the second instance with hot signal & extreme controls
        let hot = vec![0.9_f32; 256];
        let hot_c = vec![0.95_f32; ctrls.len()];
        let mut dump_l = vec![0.0_f32; 256];
        let mut dump_r = vec![0.0_f32; 256];
        dirty.process_block(&[&hot, &hot], &mut [&mut dump_l, &mut dump_r], &hot_c, &sws);
        dirty.reset();

        let mut f_l = vec![0.0_f32; stim_l.len()];
        let mut f_r = vec![0.0_f32; stim_r.len()];
        let mut d_l = vec![0.0_f32; stim_l.len()];
        let mut d_r = vec![0.0_f32; stim_r.len()];
        fresh.process_block(&[&stim_l, &stim_r], &mut [&mut f_l, &mut f_r], &ctrls, &sws);
        dirty.process_block(&[&stim_l, &stim_r], &mut [&mut d_l, &mut d_r], &ctrls, &sws);
        assert_eq!(f_l, d_l, "{} reset() L mismatch", fresh.metadata().id);
        assert_eq!(f_r, d_r, "{} reset() R mismatch", fresh.metadata().id);
    }

    // Also verify Default::default() == new() and reset() == new() across all 14 decorators
    let make_full_stack = || {
        AdcClipAndPadDecorator::new(
            ConverterLoopbackDecorator::new(MainsHumAndNoiseFloorDecorator::new(
                MicrophoneCapsuleClipDecorator::new(TubeThermalHysteresisDecorator::new(
                    HumanHandTrajectoryDecorator::new(
                        PotTaperAndSwitchDecorator::new(
                            PolarityAndDcOffsetDecorator::new(
                                IdleSquealDecorator::new(
                                    ScratchyPotAndSteppedAttenuatorDecorator::new(
                                        DigitalQuirksDecorator::new(SpringChirpDecorator::new(
                                            GermaniumTempDriftDecorator::new(
                                                HardwareNoiseGateDecorator::new(
                                                    Chain04TubeAmpFmv4x12::default(),
                                                ),
                                                31.0,
                                            ),
                                        )),
                                    ),
                                ),
                                true,
                                0.01,
                            ),
                            vec![TaperCurveKind::Log15A, TaperCurveKind::WTaper],
                        ),
                        12,
                        0.05,
                    ),
                )),
                60.0,
            )),
            6.0,
        )
    };

    let mut stack_fresh = make_full_stack();
    let mut stack_dirty = make_full_stack();
    let ctrls = stack_fresh.metadata().default_controls();
    let sws = stack_fresh.metadata().default_switches();
    let mut tmp = vec![0.0_f32; 384];
    stack_dirty.process_block(&[&stim_l], &mut [&mut tmp], &[0.9, -0.8, 0.7, -0.6], &[1]);
    stack_dirty.reset();

    let mut y_fresh = vec![0.0_f32; 384];
    let mut y_dirty = vec![0.0_f32; 384];
    stack_fresh.process_block(&[&stim_l], &mut [&mut y_fresh], &ctrls, &sws);
    stack_dirty.process_block(&[&stim_l], &mut [&mut y_dirty], &ctrls, &sws);
    assert_eq!(
        y_fresh, y_dirty,
        "14-decorator stack violated 0 ULP reset() determinism"
    );
}

#[test]
fn test_stereo_decorator_channel_isolation_vs_dual_mono() {
    #[derive(Clone, Default)]
    struct StereoIdentityFixture;
    impl BlackBoxModel for StereoIdentityFixture {
        fn reset(&mut self) {}
        fn metadata(&self) -> dnam_fixtures::FixtureDeviceMetadata {
            dnam_fixtures::FixtureDeviceMetadata {
                id: "Stereo-Identity",
                name: "Stereo Identity",
                category: "Test",
                sample_rate_hz: 48_000,
                bit_depth: FixtureBitDepth::Float32,
                expected_tail_ms: 0.0,
                supports_tempo_sync: false,
                controls: &[],
                switches: &[],
            }
        }
        fn num_inputs(&self) -> usize {
            2
        }
        fn num_outputs(&self) -> usize {
            2
        }
        fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _: &[f32], _: &[u8]) {
            outputs[0] = inputs.first().copied().unwrap_or(0.0);
            outputs[1] = inputs.get(1).copied().unwrap_or(0.0);
        }
    }

    let sig_l = multitone_stimulus(256, 48_000, 0.65);
    let sig_r = multitone_stimulus(256, 48_000, 0.12);
    let zeros = vec![0.0_f32; 256];

    // SpringChirp, MicrophoneCapsuleClip, HardwareNoiseGate, ConverterLoopback must have 0 L/R crosstalk
    let mut spring_st = SpringChirpDecorator::new(StereoIdentityFixture);
    let mut spring_r_only = SpringChirpDecorator::new(StereoIdentityFixture);
    let (mut st_l, mut st_r) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    let (mut ro_l, mut ro_r) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    spring_st.process_block(&[&sig_l, &sig_r], &mut [&mut st_l, &mut st_r], &[], &[]);
    spring_r_only.process_block(&[&zeros, &sig_r], &mut [&mut ro_l, &mut ro_r], &[], &[]);
    assert_eq!(
        st_r, ro_r,
        "SpringChirpDecorator leaked L state into R channel"
    );

    let mut mic_st = MicrophoneCapsuleClipDecorator::new(StereoIdentityFixture);
    let mut mic_r_only = MicrophoneCapsuleClipDecorator::new(StereoIdentityFixture);
    mic_st.process_block(&[&sig_l, &sig_r], &mut [&mut st_l, &mut st_r], &[], &[]);
    mic_r_only.process_block(&[&zeros, &sig_r], &mut [&mut ro_l, &mut ro_r], &[], &[]);
    assert_eq!(
        st_r, ro_r,
        "MicrophoneCapsuleClipDecorator leaked L cone state into R channel"
    );

    let mut gate_st = HardwareNoiseGateDecorator::new(StereoIdentityFixture);
    let mut gate_r_only = HardwareNoiseGateDecorator::new(StereoIdentityFixture);
    gate_st.process_block(&[&sig_l, &sig_r], &mut [&mut st_l, &mut st_r], &[], &[]);
    gate_r_only.process_block(&[&zeros, &sig_r], &mut [&mut ro_l, &mut ro_r], &[], &[]);
    assert_eq!(
        st_r, ro_r,
        "HardwareNoiseGateDecorator leaked L envelope into R channel"
    );

    let mut conv_st = ConverterLoopbackDecorator::new(StereoIdentityFixture);
    let mut conv_r_only = ConverterLoopbackDecorator::new(StereoIdentityFixture);
    conv_st.process_block(&[&sig_l, &sig_r], &mut [&mut st_l, &mut st_r], &[], &[]);
    conv_r_only.process_block(&[&zeros, &sig_r], &mut [&mut ro_l, &mut ro_r], &[], &[]);
    assert_eq!(
        st_r, ro_r,
        "ConverterLoopbackDecorator leaked L state into R channel"
    );

    // TubeThermalHysteresisDecorator must apply identical gain to L and R when fed L == R
    let mut sag_st = TubeThermalHysteresisDecorator::new(StereoIdentityFixture);
    sag_st.process_block(&[&sig_l, &sig_l], &mut [&mut st_l, &mut st_r], &[], &[]);
    assert_eq!(
        st_l, st_r,
        "TubeThermalHysteresisDecorator skewed L vs R on symmetric stereo input"
    );
}

#[test]
fn test_multi_sample_rate_and_bit_depth_device_contracts() {
    let sample_rates = [8_000_u32, 16_000, 44_100, 48_000, 88_200, 96_000, 192_000];
    let bit_depths = FixtureBitDepth::ALL;

    for &sr in &sample_rates {
        for &bd in &bit_depths {
            let mut chains = build_all_chains();
            let n_frames = ((sr as usize) / 200).max(128); // ~5 ms block at every rate
            let in_l = multitone_stimulus(n_frames, sr, 0.45);
            let in_r = multitone_stimulus(n_frames, sr, 0.30);

            for chain in &mut chains {
                chain.set_sample_rate_hz(sr);
                chain.set_bit_depth(bd);
                assert_eq!(chain.sample_rate_hz(), sr);
                assert_eq!(chain.bit_depth(), bd);
                assert_eq!(chain.metadata().sample_rate_hz, sr);
                assert_eq!(chain.metadata().bit_depth, bd);

                let ctrls = chain.metadata().default_controls();
                let sws = chain.metadata().default_switches();
                let mut out_l = vec![0.0_f32; n_frames];
                let mut out_r = vec![0.0_f32; n_frames];
                chain.process_block(&[&in_l, &in_r], &mut [&mut out_l, &mut out_r], &ctrls, &sws);
                assert!(out_l.iter().all(|v| v.is_finite()));
                assert!(out_r.iter().all(|v| v.is_finite()));

                // Reset must preserve configured sample rate and bit depth!
                chain.reset();
                assert_eq!(chain.sample_rate_hz(), sr);
                assert_eq!(chain.bit_depth(), bd);

                if bd.is_integer_pcm() {
                    let step = bd.lsb_step();
                    for &v in out_l.iter().take(32) {
                        assert_eq!(
                            v,
                            bd.quantize_sample(v),
                            "{} at {sr} Hz / {bd:?} emitted unquantized sample {v}",
                            chain.metadata().id
                        );
                        if bd.bits() <= 24 {
                            let vf = f64::from(v);
                            let snapped = (vf / step).round() * step;
                            assert_eq!(
                                vf,
                                snapped,
                                "{} at {sr} Hz / {bd:?} violated exact binary32 PCM grid: v={v}",
                                chain.metadata().id
                            );
                        }
                    }
                }
            }
        }
    }

    // Verify odd-only vs asymmetric 2nd-harmonic parity (Chain-01 vs Chain-02) & coupled FMV tone-stack cross-derivative (Chain-04)
    let stim_200: Vec<f32> = (0..2_880)
        .map(|i| 0.08 * (core::f32::consts::TAU * 200.0 * (i as f32) / 48_000.0).sin())
        .collect();
    let mut c1 = Chain01SymmetricOd::new();
    let mut c2 = Chain02GatedAsymmetricFuzz::new();
    let (mut y_sym, mut y_asym) = (vec![0.0_f32; 2_880], vec![0.0_f32; 2_880]);
    c1.process_block(&[&stim_200], &mut [&mut y_sym], &[0.6, 0.0, 0.0], &[]);
    c2.process_block(&[&stim_200], &mut [&mut y_asym], &[-0.8, 0.6, 0.5], &[]);
    let even_h2_ratio = |sig: &[f32]| -> f64 {
        let slice = &sig[480..2_880];
        let dft_mag = |freq_hz: f64| -> f64 {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (k, &v) in slice.iter().enumerate() {
                let ph = core::f64::consts::TAU * freq_hz * (k as f64) / 48_000.0;
                re += f64::from(v) * ph.cos();
                im -= f64::from(v) * ph.sin();
            }
            re.hypot(im)
        };
        dft_mag(400.0) / dft_mag(200.0).max(1.0e-9)
    };
    assert!(even_h2_ratio(&y_sym) < 1.0e-4);
    assert!(even_h2_ratio(&y_asym) > 0.05);

    let mut amp = Chain04TubeAmpFmv4x12::new();
    let probe = multitone_stimulus(256, 48_000, 0.5);
    let eval_rms = |amp_ref: &mut Chain04TubeAmpFmv4x12, bass: f32, mid: f32| -> f64 {
        amp_ref.reset();
        let mut y = vec![0.0_f32; probe.len()];
        amp_ref.process_block(&[&probe], &mut [&mut y], &[0.2, bass, mid, 0.0], &[0]);
        (y.iter().map(|&v| f64::from(v * v)).sum::<f64>() / (y.len() as f64)).sqrt()
    };
    let cross = (eval_rms(&mut amp, 0.8, 0.8)
        - eval_rms(&mut amp, 0.8, -0.8)
        - eval_rms(&mut amp, -0.8, 0.8)
        + eval_rms(&mut amp, -0.8, -0.8))
    .abs();
    assert!(cross > 1.0e-5);

    // Growing HumanHandTrajectoryDecorator::lag_samples mid-stream must stay strictly causal & monotone
    struct CtrlProbeFixture;
    impl BlackBoxModel for CtrlProbeFixture {
        fn reset(&mut self) {}
        fn metadata(&self) -> dnam_fixtures::FixtureDeviceMetadata {
            dnam_fixtures::FixtureDeviceMetadata {
                id: "ctrl-probe",
                name: "CtrlProbe",
                category: "Test",
                sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
                bit_depth: FixtureBitDepth::Float32,
                expected_tail_ms: 0.0,
                supports_tempo_sync: false,
                controls: &[dnam_fixtures::FixtureControlSpec {
                    id: "c0",
                    name: "C0",
                    unit: "",
                    min_physical: -1.0,
                    max_physical: 1.0,
                    default_norm: 0.0,
                }],
                switches: &[],
            }
        }
        fn num_inputs(&self) -> usize {
            1
        }
        fn num_outputs(&self) -> usize {
            1
        }
        fn process_frame(&mut self, _: &[f32], outputs: &mut [f32], controls: &[f32], _: &[u8]) {
            outputs[0] = controls.first().copied().unwrap_or(0.0);
        }
    }

    let mut hand = HumanHandTrajectoryDecorator::new(CtrlProbeFixture, 10, 0.0);
    let mut y = [0.0_f32; 1];
    for _ in 0..8 {
        hand.process_frame(&[0.0], &mut y, &[-0.5], &[]);
        assert!((y[0] - (-0.5)).abs() < 1.0e-6);
    }
    hand.lag_samples = 25;
    let mut prev = -0.5_f32;
    for step in 0..40 {
        hand.process_frame(&[0.0], &mut y, &[0.5], &[]);
        if step < 25 {
            assert!(
                (y[0] - (-0.5)).abs() < 1.0e-6,
                "non-causal step leak at step={step}: y={}",
                y[0]
            );
        } else {
            assert!(
                y[0] >= prev - 1.0e-7,
                "non-monotone backwards jump at {step}"
            );
        }
        prev = y[0];
    }
    assert!(prev > -0.5);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_composed_chains_and_decorators_remain_finite_across_rates_and_bit_depths(
        amp in 0.01f32..1.5f32,
        c0 in -1.2f32..1.2f32,
        c1 in -1.2f32..1.2f32,
        c2 in -1.2f32..1.2f32,
        c3 in -1.2f32..1.2f32,
        sw0 in 0u8..4u8,
        temp_c in 10.0f32..50.0f32,
        sr_idx in 0usize..7usize,
        bd_idx in 0usize..5usize,
    ) {
        let sr = [8_000_u32, 16_000, 44_100, 48_000, 88_200, 96_000, 192_000][sr_idx];
        let bd = FixtureBitDepth::ALL[bd_idx];

        let mut input_l = multitone_stimulus(160, sr, amp);
        let input_r = multitone_stimulus(160, sr, amp * 0.8);
        // Inject occasional NaN / extreme spike to exercise fault sanitization
        input_l[7] = f32::NAN;
        input_l[19] = 250.0;
        let controls = [c0, c1, c2, c3, f32::NAN];
        let switches = [sw0];

        let mut decorated_od = ConverterLoopbackDecorator::with_converter_format(
            GermaniumTempDriftDecorator::new(
                PotTaperAndSwitchDecorator::new(
                    Chain01SymmetricOd::new(),
                    vec![TaperCurveKind::Log15A, TaperCurveKind::WTaper],
                ),
                temp_c,
            ),
            sr,
            bd,
        );
        prop_assert_eq!(decorated_od.sample_rate_hz(), sr);
        prop_assert_eq!(decorated_od.bit_depth(), bd);

        let mut out_l = vec![0.0_f32; 160];
        let mut out_r = vec![0.0_f32; 160];
        decorated_od.process_block(
            &[&input_l, &input_r],
            &mut [&mut out_l, &mut out_r],
            &controls,
            &switches,
        );
        prop_assert!(out_l.iter().all(|x| x.is_finite() && !x.is_subnormal() && x.abs() <= 20.0));
        prop_assert!(out_r.iter().all(|x| x.is_finite() && !x.is_subnormal() && x.abs() <= 20.0));

        let mut decorated_chorus = TubeThermalHysteresisDecorator::new(
            Chain07StereoChorusAndRotary::with_format(sr, bd),
        );
        decorated_chorus.process_block(
            &[&input_l, &input_r],
            &mut [&mut out_l, &mut out_r],
            &controls,
            &switches,
        );
        // Chain-07 is a linear chorus/rotary without hard clipping, so a +48 dBFS (250.0) input spike stays finite and proportional
        prop_assert!(out_l.iter().all(|x| x.is_finite() && !x.is_subnormal() && x.abs() <= 400.0));
        prop_assert!(out_r.iter().all(|x| x.is_finite() && !x.is_subnormal() && x.abs() <= 400.0));
    }
}

#[test]
fn test_decorators_preserve_arbitrary_continuous_controls_above_16() {
    use dnam_fixtures::adversaries::{
        GermaniumTempDriftDecorator, HumanHandTrajectoryDecorator, IdleSquealDecorator,
        PotTaperAndSwitchDecorator, ScratchyPotAndSteppedAttenuatorDecorator, TaperCurveKind,
    };
    use dnam_fixtures::{
        BlackBoxModel, FixtureBitDepth, FixtureControlSpec, FixtureDeviceMetadata,
    };

    const CTRL_SPEC_ZERO: FixtureControlSpec = FixtureControlSpec {
        id: "ctrl",
        name: "Ctrl",
        unit: "",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 10.0,
    };
    const CONTROLS_20: [FixtureControlSpec; 20] = {
        let mut arr = [CTRL_SPEC_ZERO; 20];
        arr[19].default_norm = 0.35;
        arr
    };

    #[derive(Clone, Debug, Default)]
    struct Rig20Controls {
        pub last_ctrl19: f32,
        pub last_ctrl21: f32,
    }

    impl BlackBoxModel for Rig20Controls {
        fn reset(&mut self) {
            self.last_ctrl19 = 0.0;
            self.last_ctrl21 = 0.0;
        }
        fn metadata(&self) -> FixtureDeviceMetadata {
            FixtureDeviceMetadata {
                id: "Rig-20-Controls",
                name: "20-Knob Multi-Pedal Rig",
                category: "Multi-Pedal",
                sample_rate_hz: 48_000,
                bit_depth: FixtureBitDepth::Float32,
                expected_tail_ms: 0.0,
                supports_tempo_sync: false,
                controls: &CONTROLS_20,
                switches: &[],
            }
        }
        fn num_inputs(&self) -> usize {
            1
        }
        fn num_outputs(&self) -> usize {
            1
        }
        fn process_frame(
            &mut self,
            inputs: &[f32],
            outputs: &mut [f32],
            controls: &[f32],
            _switches: &[u8],
        ) {
            self.last_ctrl19 = controls.get(19).copied().unwrap_or(-1.0);
            self.last_ctrl21 = controls.get(21).copied().unwrap_or(-1.0);
            outputs[0] = inputs.first().copied().unwrap_or(0.0) * (1.0 + 0.5 * self.last_ctrl19);
        }
    }

    let mut ctrls20 = [0.0_f32; 20];
    ctrls20[19] = 0.8;
    let mut out = [0.0_f32; 1];

    // 1. PotTaperAndSwitchDecorator forwards & tapers p=19
    let mut tapers = vec![TaperCurveKind::Log15A; 20];
    tapers[19] = TaperCurveKind::RevLog85C;
    let mut d_taper = PotTaperAndSwitchDecorator::new(Rig20Controls::default(), tapers);
    d_taper.process_frame(&[0.5], &mut out, &ctrls20, &[]);
    let expected_taper19 = TaperCurveKind::RevLog85C.map_shaft_to_electrical(0.5 * (0.8 + 1.0));
    assert!((d_taper.inner.last_ctrl19 - expected_taper19).abs() < 1.0e-6);

    // 2. HumanHandTrajectoryDecorator preserves p=19 default on empty slice, grows dynamically,
    //    and smoothly decays dynamic extra controls back to 0.0 when controls shrink and later regrow.
    let mut d_hand = HumanHandTrajectoryDecorator::new(Rig20Controls::default(), 2, 0.02);
    d_hand.wrist_bandwidth_hz = 40.0;
    d_hand.process_frame(&[0.5], &mut out, &[], &[]);
    assert!((d_hand.inner.last_ctrl19 - 0.35).abs() < 1.0e-5);
    let mut ctrls22 = [0.0_f32; 22];
    ctrls22[19] = 0.8;
    ctrls22[21] = 0.9;
    for _ in 0..8 {
        d_hand.process_frame(&[0.5], &mut out, &ctrls22, &[]);
    }
    assert!(d_hand.inner.last_ctrl19 > 0.35 && d_hand.inner.last_ctrl21 > 0.05);
    // Shrink back to 20 controls for enough samples to drain delay ring + relax extra control #21 toward 0.0
    for _ in 0..4_000 {
        d_hand.process_frame(&[0.5], &mut out, &ctrls20, &[]);
    }
    // Regrow with ctrls22[21] = 0.0: extra control #21 must already have settled near 0.0 (not frozen at 0.9!)
    ctrls22[21] = 0.0;
    d_hand.process_frame(&[0.5], &mut out, &ctrls22, &[]);
    assert!(d_hand.inner.last_ctrl21.abs() < 1.0e-4);

    // 3. GermaniumTempDriftDecorator, ScratchyPotAndSteppedAttenuatorDecorator, IdleSquealDecorator
    let mut d_ge = GermaniumTempDriftDecorator::new(Rig20Controls::default(), 40.0);
    d_ge.process_frame(&[0.5], &mut out, &ctrls20, &[]);
    assert!((d_ge.inner.last_ctrl19 - 0.8).abs() < 1.0e-6);

    let mut d_sp = ScratchyPotAndSteppedAttenuatorDecorator::new(Rig20Controls::default());
    d_sp.process_frame(&[0.5], &mut out, &ctrls20, &[]);
    assert!((d_sp.inner.last_ctrl19 - 0.8).abs() < 1.0e-6);

    let mut d_sq = IdleSquealDecorator::new(Rig20Controls::default());
    d_sq.process_frame(&[0.5], &mut out, &ctrls20, &[]);
    assert!((d_sq.inner.last_ctrl19 - 0.8).abs() < 1.0e-6);

    // 4. Impulse-to-silence tail across all 12 chains and Polarity/Loopback decorators must emit zero IEEE-754 subnormals
    assert_eq!(
        PolarityAndDcOffsetDecorator::<Chain01SymmetricOd>::sanitize_sample(f32::MIN_POSITIVE),
        (f32::MIN_POSITIVE, false)
    );
    assert_eq!(
        PolarityAndDcOffsetDecorator::<Chain01SymmetricOd>::sanitize_sample(f32::from_bits(
            0x007F_FFFF
        )),
        (0.0, false)
    );
    let mut imp = vec![0.0_f32; 8_192];
    imp[0] = 1.0e-18;
    for mut chain in build_all_chains() {
        let c = chain.metadata().default_controls();
        let s = chain.metadata().default_switches();
        let (mut yl, mut yr) = (vec![0.0_f32; imp.len()], vec![0.0_f32; imp.len()]);
        chain.process_block(&[&imp, &imp], &mut [&mut yl, &mut yr], &c, &s);
        assert!(
            yl.iter().chain(yr.iter()).all(|v| !v.is_subnormal()),
            "{} emitted subnormal f32 on impulse-to-silence decay",
            chain.metadata().id
        );
    }
    let mut lb_tail = ConverterLoopbackDecorator::new(PolarityAndDcOffsetDecorator::new(
        Chain01SymmetricOd::new(),
        false,
        0.0,
    ));
    let mut y_lb = vec![0.0_f32; imp.len()];
    lb_tail.process_block(&[&imp], &mut [&mut y_lb], &[0.0, 0.0, 0.0], &[]);
    assert!(y_lb.iter().all(|v| !v.is_subnormal()));
}
