//! Example 5: Standalone TORA Zero-`FDIV` Operator Chain (`examples/custom_operator_chain.rs`).
//!
//! Demonstrates composing the standalone real-time TORA operators (`Op-T`, `Op-M`, `Op-N`)
//! and Stage 0 `TptLr4Crossover` directly inside a caller-held [`FtzDazGuard`]:
//! - `TptLr4Crossover` (`200 Hz` Linkwitz-Riley 4th-order bass-mono / drive split, `ref[dsp.router.lr4-crossover]`)
//! - `OpTDelayLine` + `HilbertSsbLattice10th` (`H1`) + `SubHarmonicTracker` (`T1`) (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.tracker.toggle-sheet]`)
//! - `Q0_192Phase` + `contractive_rotor_f32` (`ref[dsp.op-m.rotor.q0-192-lfo]`, `ref[dsp.op-m.rotor.contractive]`)
//! - `MultiDrivePolyAdaaChannel` (6-step rail-crossing exact cubic spline PolyADAA, `ref[dsp.op-n.polyadaa.rail-crossing]`)
//! - `ZohBitcrusher` (`Q64.32` zero-order hold + TPDF dither, `ref[dsp.op-n.bitcrusher.zoh-phase]`)
//! - `KautzModalBank::<4>::step_4lane_simd` (`ref[dsp.op-m.recurrence.symplectic]`)
//!
//! Run with:
//!   `cargo run -p dnam-engine --example custom_operator_chain`

use dnam_engine::FtzDazGuard;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::math::spec_math::{Q0_192Phase, contractive_rotor_f32};
use dnam_engine::operators::{
    HilbertSsbLattice10th, KautzModalBank, MultiDrivePolyAdaaChannel, OpTDelayLine,
    SubHarmonicTracker, ZohBitcrusher,
};
use dnam_engine::stages::topology::TptLr4Crossover;

fn main() {
    let _audio_guard = FtzDazGuard::new();

    let mut lr4 = TptLr4Crossover::new_48khz();
    let mut delay = OpTDelayLine::<256>::new();
    delay.set_target_delay(18.5, false);

    let mut hilbert = HilbertSsbLattice10th::new();
    let mut sub_tracker = SubHarmonicTracker::new();
    let mut lfo_phase = Q0_192Phase::default();

    let mut poly_adaa = MultiDrivePolyAdaaChannel::new();
    let mut bitcrusher = ZohBitcrusher::new();

    let mut kautz = KautzModalBank::<4>::new();
    for m in 0..4 {
        kautz.set_pole(m, 0.985, 0.08 + 0.04 * (m as f32), 0.0, 0.25, -0.12);
    }

    let mut peak_out = 0.0_f32;
    let mut max_rotor_norm = 0.0_f32;

    for n in 0..256 {
        let x = 0.35 * ((n as f32) * 0.09).sin();

        // 1. Stage 0 200 Hz LR4 crossover split into low clean bypass + high drive path
        let (x_low, x_high) = lr4.step(x);

        // 2. Advance Q0.192 LFO phase and evaluate contractive rotor (c^2 + s^2 <= 1.0)
        lfo_phase.step_angle_bits(0.015_f32.to_bits()).unwrap();
        let (lfo_cos, lfo_sin) = contractive_rotor_f32((0.015 * (n as f32)) % 3.0).unwrap();
        max_rotor_norm = max_rotor_norm.max(lfo_cos * lfo_cos + lfo_sin * lfo_sin);

        // 3. Op-T fractional delay + Op-M 12-allpass Hilbert SSB & sub-octave tracker
        let delayed = delay.step(x_high + 0.02 * lfo_sin);
        let (h_i, h_q) = hilbert.step(delayed);
        let sub_oct = sub_tracker.step(h_i, h_q);

        // 4. Op-N 6-step rail-crossing MultiDrive PolyADAA + Q64.32 ZOH Bitcrusher
        let drives = poly_adaa.step(delayed + 0.15 * sub_oct, 0.30, 0.08, 0.85, 0.9993);
        let crushed = bitcrusher.step(drives[1]);

        // 5. Op-M 4-lane SIMD Kautz modal cabinet resonator
        let modal_lanes = kautz.step_4lane_simd::<DefaultMathBackend>(crushed);
        let y = x_low + modal_lanes.iter().sum::<f32>();
        peak_out = peak_out.max(y.abs());
    }

    println!(
        "Custom TORA operator chain processed 256 frames | Peak Output: {:.6} | Max Rotor Norm (<= 1.0): {:.8}",
        peak_out, max_rotor_norm,
    );
    assert!(max_rotor_norm <= 1.0);
}
