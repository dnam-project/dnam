//! Isolated TORA Operator & Spec-Math Criterion Benchmarks (`benches/operators.rs`).
//!
//! Benchmarks `Op-T` (Hermite Delay Line, Partitioned Schroeder Diffuser, Ballistic VCA),
//! `Op-M` (Symplectic 4-Lane, 4-Lane SIMD Kautz Modal Bank, 12-Allpass Hilbert `H1`, Sub-Harmonic `T1`,
//! `Q0.192` LFO Phase + Contractive Rotor), and `Op-N` (6-Step Rail-Crossing `MultiDrivePolyAdaaChannel`,
//! `Q64.32` `ZohBitcrusher`, `QuadratureHysteresis2D`, `c2_wavefolder`).

use std::hint::black_box;

use dnam_engine::FtzDazGuard;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::math::spec_math::{Q0_192Phase, contractive_rotor_f32};
use dnam_engine::operators::{
    BallisticDuckingVca, HilbertSsbLattice10th, KautzModalBank, MultiDrivePolyAdaaChannel,
    OpTDelayLine, PartitionedSchroederDiffuserBank, QuadratureHysteresis2D, SubHarmonicTracker,
    SymplecticModal4Lane, ZohBitcrusher, c2_wavefolder,
};
use hotpath::criterion::{Criterion, criterion_group, criterion_main};

fn bench_op_t_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("op_t_transport");

    let mut delay = OpTDelayLine::<256>::new();
    delay.set_target_delay(24.375, false);
    group.bench_function("hermite_delay_line_step", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(delay.step(black_box(0.25))));
    });

    let mut schroeder = PartitionedSchroederDiffuserBank::new_for_rate(48_000);
    group.bench_function("partitioned_schroeder_diffuser_step", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(schroeder.step_line(0, black_box(0.25), black_box(0.5))));
    });

    let mut vca = BallisticDuckingVca::new();
    group.bench_function("ballistic_ducking_vca_step", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(vca.step(black_box(0.35), black_box(0.25))));
    });

    group.finish();
}

fn bench_op_m_and_rotor_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("op_m_modal_and_rotor");

    let mut h1 = HilbertSsbLattice10th::default();
    group.bench_function("hilbert_ssb_10th_12allpass", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(h1.step(black_box(0.25))));
    });

    let mut t1 = SubHarmonicTracker::new();
    group.bench_function("subharmonic_tracker_t1", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(t1.step(black_box(0.25), black_box(0.4))));
    });

    let mut op_m_4lane = SymplecticModal4Lane::new();
    op_m_4lane.set_pole(0, 0.992, 0.11);
    group.bench_function("symplectic_modal_4lane_step", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(op_m_4lane.step_lane(0, black_box(0.25))));
    });

    let mut kautz_bank = KautzModalBank::<4>::new();
    for m in 0..4 {
        kautz_bank.set_pole(m, 0.99, 0.1 + 0.05 * (m as f32), 0.0, 0.25, -0.15);
    }
    group.bench_function("kautz_modal_bank_4lane_simd", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(kautz_bank.step_4lane_simd::<DefaultMathBackend>(black_box(0.25))));
    });

    let mut phase = Q0_192Phase::default();
    let step_words = [0, 0, 1u64 << 52];
    group.bench_function("q0_192_phase_and_contractive_rotor", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            phase.advance_signed(false, black_box(step_words));
            let (cos_p, sin_p) = phase.to_phasor();
            black_box((cos_p, sin_p, contractive_rotor_f32(black_box(0.25))))
        });
    });

    group.finish();
}

fn bench_op_n_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("op_n_nonlinear");

    let mut op_n_ch = MultiDrivePolyAdaaChannel::new();
    group.bench_function("multidrive_poly_adaa_6step_rail_crossing", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(op_n_ch.step(
                black_box(0.45),
                black_box(0.25),
                black_box(0.10),
                black_box(0.85),
                black_box(0.9993),
            ))
        });
    });

    let mut bitcrusher = ZohBitcrusher::new();
    group.bench_function("zoh_bitcrusher_q64_32", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(bitcrusher.step(black_box(0.25))));
    });

    let mut hyst = QuadratureHysteresis2D::new();
    group.bench_function("quadrature_hysteresis_2d", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(hyst.step(black_box(0.35), black_box(0.995))));
    });

    group.bench_function("c2_wavefolder", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(c2_wavefolder(black_box(0.65))));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_op_t_kernels,
    bench_op_m_and_rotor_kernels,
    bench_op_n_kernels
);
criterion_main!(benches);
