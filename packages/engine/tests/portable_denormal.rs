//! Portable gradual-underflow and recursive-state writeback regression checks.

#[cfg(target_arch = "wasm32")]
pub fn verify_portable_denormal_policy() {
    use dnam_engine::control::{ControlBus, ControlLatentProjection, LatentAccumulator64B};
    use dnam_engine::math::{
        ScalarBackend, ScalarVec4, SimdVec4, WasmSimd128Backend, WasmVec4, flush_subnormal_f32_bits,
    };
    use dnam_engine::operators::op_m::{
        HilbertSsbLattice10th, KautzModalBank, SymplecticModal4Lane,
    };
    use dnam_engine::operators::op_n::MultiDrivePolyAdaaChannel;
    use dnam_engine::operators::op_t::{BallisticDuckingVca, OpTDelayLine};
    use dnam_engine::stages::head::MidSideKroneckerHead;
    use dnam_engine::stages::router::{BezoutCleanCrossover, ReturnAligner};
    use dnam_engine::stages::stage2_cascade::{Stage2CommutedCascade, ToneBasisFilters};
    use dnam_engine::stages::stage3_cabinet::Stage3CabinetModals;
    use dnam_engine::stages::stage4_tank::TankDcBlocker;
    use dnam_engine::{FtzDazGuard, Tier};

    let _audio = FtzDazGuard::new();
    let cases = [
        (
            [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001],
            [0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000],
        ),
        (
            [0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000],
            [0x0000_0000, 0x8000_0000, 0x0080_0000, 0x8080_0000],
        ),
        (
            [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
            [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
        ),
    ];
    for (input, expected) in cases {
        let values = std::hint::black_box(input).map(f32::from_bits);
        assert_eq!(
            values.map(flush_subnormal_f32_bits).map(f32::to_bits),
            expected
        );
        assert_eq!(
            ScalarVec4::from_array(values)
                .flush_subnormal()
                .to_array()
                .map(f32::to_bits),
            expected
        );
        assert_eq!(
            WasmVec4::from_array(values)
                .flush_subnormal()
                .to_array()
                .map(f32::to_bits),
            expected
        );
    }

    let min = std::hint::black_box(f32::MIN_POSITIVE);
    let next = std::hint::black_box(f32::from_bits(0x0080_0001));
    let smallest = f32::from_bits(1);
    assert_eq!((next - min).to_bits(), 1);

    let mut modal = SymplecticModal4Lane::new();
    modal.z_re[0] = min;
    modal.alpha_re[0] = 0.5;
    let (modal_out, _) = modal.step_lane(0, 0.0);
    assert_eq!(modal_out.to_bits(), 0x0040_0000);
    assert_eq!(modal.z_re[0].to_bits(), 0);

    let mut scalar_bank = KautzModalBank::<1>::new();
    scalar_bank.s_re[0] = -min;
    scalar_bank.lambda_re[0] = 0.5;
    scalar_bank.c_re[0] = 1.0;
    assert_eq!(scalar_bank.step(0.0).to_bits(), 0x8080_0000);
    assert_eq!(scalar_bank.s_re[0].to_bits(), 0x8000_0000);
    assert_eq!(scalar_bank.step(0.0), 0.0);

    let mut vector_bank = KautzModalBank::<4>::new();
    vector_bank.s_re = [min, -min, 2.0 * min, -2.0 * min];
    vector_bank.lambda_re = [0.5; 4];
    vector_bank.c_re = [1.0; 4];
    let mut scalar_vector_bank = vector_bank.clone();
    let scalar_vector_out = scalar_vector_bank.step_4lane_simd::<ScalarBackend>(0.0);
    let vector_out = vector_bank.step_4lane_simd::<WasmSimd128Backend>(0.0);
    let expected_out = [0x0080_0000, 0x8080_0000, 0x0100_0000, 0x8100_0000];
    assert_eq!(vector_out.map(f32::to_bits), expected_out);
    assert_eq!(scalar_vector_out.map(f32::to_bits), expected_out);
    let expected_state = [0x0000_0000, 0x8000_0000, 0x0080_0000, 0x8080_0000];
    assert_eq!(vector_bank.s_re.map(f32::to_bits), expected_state);
    assert_eq!(scalar_vector_bank.s_re.map(f32::to_bits), expected_state);

    let mut hilbert = HilbertSsbLattice10th::new();
    hilbert.i_stages[0] = -smallest;
    hilbert.i_stages_z2[0] = min;
    let _ = hilbert.step(smallest);
    assert_eq!(hilbert.i_stages[0].to_bits(), 0);
    assert_eq!(hilbert.i_stages_z2[0].to_bits(), 0x8000_0000);
    assert_eq!(hilbert.q_z1.to_bits(), 1);

    let mut diffuser = OpTDelayLine::<2048>::new();
    diffuser.diff_state[255] = min;
    let diffuser_out = diffuser.step_diffuser(min, 1, 0.5);
    assert_eq!(diffuser.diff_state[0].to_bits(), 0);
    assert_eq!(diffuser_out.to_bits(), 0x00a0_0000);

    let mut vca = BallisticDuckingVca::new();
    vca.env = min;
    vca.alpha_rel = 0.5;
    assert_eq!(vca.step(0.0, 1.0), 1.0);
    assert_eq!(vca.env.to_bits(), 0);
    vca.env = min;
    assert_eq!(vca.step_stereo(0.0, 0.0, 1.0, -1.0), (1.0, -1.0));
    assert_eq!(vca.env.to_bits(), 0);

    let mut crossover = BezoutCleanCrossover::new();
    crossover.pole = 0.5;
    crossover.y_hist[0] = min;
    let _ = crossover.step(0.0);
    assert_eq!(crossover.y_hist.map(f32::to_bits), [0; 4]);

    let mut tone = ToneBasisFilters::new();
    tone.lpf80_state = min;
    tone.lpf400_state = -min;
    tone.hpf1200_state = min;
    tone.alpha_80 = 0.5;
    tone.alpha_400 = 0.5;
    tone.alpha_1200 = 0.5;
    let tone_out = tone.step(0.0);
    assert_eq!(tone_out[1].to_bits(), 0x0040_0000);
    assert_eq!(tone_out[2].to_bits(), 0x8040_0000);
    assert_eq!(tone.lpf80_state.to_bits(), 0);
    assert_eq!(tone.lpf400_state.to_bits(), 0x8000_0000);
    assert_eq!(tone.hpf1200_state.to_bits(), 0);

    let mut cascade = Stage2CommutedCascade::new();
    cascade.sag_envelope = min;
    cascade.r_sag = 0.5;
    let _ = cascade.step_frame(0.0, 0.0, 1.0, 0.2);
    assert_eq!(cascade.sag_envelope.to_bits(), 0);

    let mut dc = TankDcBlocker::new();
    dc.y_prev = -min;
    dc.r_dc = 0.5;
    assert_eq!(dc.step(0.0).to_bits(), 0x8040_0000);
    assert_eq!(dc.y_prev.to_bits(), 0x8000_0000);

    let mut bus = ControlBus::<WasmSimd128Backend>::new();
    bus.env_states = [min, -min, 2.0 * min, 0.0];
    bus.env_poles = [0.5; 4];
    let _ = bus.tick_3khz(0.0, &[]);
    assert_eq!(
        bus.env_states[..3]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        [0, 0x8000_0000, 0x0080_0000]
    );

    let mut accumulator = LatentAccumulator64B::new();
    let mut projection = ControlLatentProjection::new();
    accumulator.z_main[0] = min;
    projection.w_main[0] = -0.5;
    projection.v_factors[0] = 0.5;
    accumulator.update_delta(&projection, 0.0, min, min);
    assert_eq!(accumulator.z_main[0].to_bits(), 0);
    assert_eq!(accumulator.s_vx[0].to_bits(), 0);

    let mut aligner = ReturnAligner::new();
    let _ = aligner.step(smallest);
    assert_eq!(aligner.ring[0].to_bits(), 1);
    let mut cabinet = Stage3CabinetModals::new();
    let _ = cabinet.step(smallest, 0.0, &[0.0; 16]);
    assert_eq!(cabinet.fir_history[0].to_bits(), 1);
    let mut poly = MultiDrivePolyAdaaChannel::new();
    let _ = poly.step(min, 0.0, 0.0, 0.0, 0.5);
    assert_ne!(poly.u_prev[0].to_bits(), 0);
    assert_eq!(poly.u_prev[0].to_bits() & 0x7f80_0000, 0);

    let mut head = MidSideKroneckerHead::new();
    head.b_mid = smallest;
    let (left, right) = head.project_feature_bus(&[0.0; 64], Tier::Lite);
    assert_eq!(left.to_bits(), 1);
    assert_eq!(right.to_bits(), 1);
}

#[cfg(target_arch = "wasm32")]
#[test]
fn portable_denormal_policy() {
    verify_portable_denormal_policy();
}
