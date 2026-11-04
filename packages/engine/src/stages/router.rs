//! Stage 0: 8-Port Matrix Router, Lower-Block Triangular DAG Routing & Sub Crossover.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` (`ref[dsp.router.return-to-send-isolation]`, `REQ-7.4`).

use crate::math::flush_recursive_state_f32;
use crate::operators::op_t::catmull_rom_hermite_cubic;

/// Memory footprint for Stage 0 State: strictly 1,536 B.
pub const ROUTER_STATE_RAM_BYTES: usize = 1536;

/// Dual Catmull-Rom return aligner buffer capacity (samples per return channel).
pub const RETURN_ALIGN_CAPACITY: usize = 128;
const RETURN_ALIGN_MASK: usize = RETURN_ALIGN_CAPACITY - 1;

/// 4th-order Bezout low-pass clean crossover (< 200 Hz).
///
/// Implemented as a 4-stage cascade of 1st-order bilinear low-pass sections with pole `p`:
/// `H(z) = ((0.5 * (1 - p) * (1 + z^-1)) / (1 - p * z^-1))^4`, guaranteeing exact `1.000000`
/// (`0.000000 dB`) steady-state DC gain, a 4th-order zero at Nyquist (`z = -1`), unconditional
/// BIBO stability in `f32`, and exact continuous-time ZOH sample-rate rescaling.
#[derive(Clone, Copy, Debug)]
pub struct BezoutCleanCrossover {
    pub x_hist: [f32; 4],
    pub y_hist: [f32; 4],
    pub base_pole: f32,
    pub pole: f32,
}

impl Default for BezoutCleanCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl BezoutCleanCrossover {
    /// Base real pole per 1st-order section at 48 kHz for a 4th-order 200 Hz crossover.
    pub const POLE_200HZ_48K: f32 = 0.941568;

    pub const fn new() -> Self {
        Self {
            x_hist: [0.0; 4],
            y_hist: [0.0; 4],
            base_pole: Self::POLE_200HZ_48K,
            pole: Self::POLE_200HZ_48K,
        }
    }

    pub fn reset(&mut self) {
        self.x_hist.fill(0.0);
        self.y_hist.fill(0.0);
    }

    /// Sets the training-rate crossover frequency in Hz (default `200.0`).
    pub fn set_crossover_hz(&mut self, freq_hz: f32) {
        let f = freq_hz.clamp(20.0, 2000.0);
        if (f - 200.0).abs() < 1e-4 {
            self.base_pole = Self::POLE_200HZ_48K;
        } else {
            self.base_pole =
                (crate::schema::cold_pow_f64(Self::POLE_200HZ_48K as f64, (f as f64) / 200.0)
                    as f32)
                    .clamp(0.5, 0.9995);
        }
        self.pole = self.base_pole;
    }

    /// Rescales the section pole for the host sample rate under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.pole = zoh.rescale_real_pole(self.base_pole);
    }

    /// Advances the crossover by one sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let g = 0.5 * (1.0 - self.pole);
        let mut u = x;
        for i in 0..4 {
            let y = self.pole * self.y_hist[i] + g * (u + self.x_hist[i]);
            self.x_hist[i] = flush_recursive_state_f32(u);
            self.y_hist[i] = flush_recursive_state_f32(y);
            u = y;
        }
        u
    }
}

/// Dual Catmull-Rom return latency aligner (0.000000 dB DC gain).
#[derive(Clone, Debug)]
pub struct ReturnAligner {
    pub ring: [f32; RETURN_ALIGN_CAPACITY],
    pub write_ptr: usize,
    pub delay_samples: f32,
}

impl Default for ReturnAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl ReturnAligner {
    pub const fn new() -> Self {
        Self {
            ring: [0.0; RETURN_ALIGN_CAPACITY],
            write_ptr: 0,
            delay_samples: 4.0,
        }
    }

    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.write_ptr = 0;
    }

    pub fn set_delay(&mut self, delay: f32) {
        self.delay_samples = delay.clamp(0.0, (RETURN_ALIGN_CAPACITY - 4) as f32);
    }

    /// Advances the return aligner by one sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        self.ring[self.write_ptr] = x;

        let d_clamped = self
            .delay_samples
            .clamp(0.0, (RETURN_ALIGN_CAPACITY - 4) as f32);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);

        let p0 = (self.write_ptr + RETURN_ALIGN_CAPACITY - d_int) & RETURN_ALIGN_MASK;
        let pm1 = if d_int == 0 {
            p0
        } else {
            (p0 + 1) & RETURN_ALIGN_MASK
        };
        let p1 = (p0 + RETURN_ALIGN_CAPACITY - 1) & RETURN_ALIGN_MASK;
        let p2 = (p0 + RETURN_ALIGN_CAPACITY - 2) & RETURN_ALIGN_MASK;

        let s_m1 = self.ring[pm1];
        let s_0 = self.ring[p0];
        let s_1 = self.ring[p1];
        let s_2 = self.ring[p2];

        self.write_ptr = (self.write_ptr + 1) & RETURN_ALIGN_MASK;
        catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
    }
}

/// Stage 0: 8-Port Matrix Router state (1,536 B RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage0Router {
    /// Scattering matrix S_router (4x4): [inst_l, inst_r, ret_l, ret_r] -> [send_l, send_r, down_l, down_r].
    /// Invariant (`ref[dsp.router.return-to-send-isolation]`): S_ret->send == 0 (S[0][2] == S[0][3] == S[1][2] == S[1][3] == 0.0).
    pub s_matrix: [[f32; 4]; 4],
    /// Dual return aligners for Left and Right external loop returns.
    pub return_aligner_l: ReturnAligner,
    pub return_aligner_r: ReturnAligner,
    /// Dual Bezout parallel clean sub crossovers (< 200 Hz).
    pub sub_crossover_l: BezoutCleanCrossover,
    pub sub_crossover_r: BezoutCleanCrossover,
    /// Dual TPT Linkwitz-Riley 4th-order (`200 Hz`) crossovers (`ref[dsp.router.lr4-crossover]`).
    pub tpt_crossover_l: super::topology::TptLr4Crossover,
    pub tpt_crossover_r: super::topology::TptLr4Crossover,
    /// WDW dry tap routing mode (`PreSpatialCore` vs `DirectDry`, `ref[dsp.router.wdw-contract]`).
    pub wdw_tap_mode: super::topology::WdwDryTapMode,
    /// WDW mode: whether Port Out 3 outputs dedicated dry center feed.
    pub wdw_mode: bool,
    /// Padding to strictly match 1,536 B footprint.
    pub _pad: [u8; 64],
}

const _: () = assert!(core::mem::size_of::<Stage0Router>() <= ROUTER_STATE_RAM_BYTES);

impl Default for Stage0Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage0Router {
    pub const fn new() -> Self {
        Self {
            s_matrix: [
                [1.0, 0.0, 0.0, 0.0], // send_l = 1.0 * inst_l + 0.0 * ret
                [0.0, 1.0, 0.0, 0.0], // send_r = 1.0 * inst_r + 0.0 * ret
                [1.0, 0.0, 1.0, 0.0], // down_l = inst_l + ret_l
                [0.0, 1.0, 0.0, 1.0], // down_r = inst_r + ret_r
            ],
            return_aligner_l: ReturnAligner::new(),
            return_aligner_r: ReturnAligner::new(),
            sub_crossover_l: BezoutCleanCrossover::new(),
            sub_crossover_r: BezoutCleanCrossover::new(),
            tpt_crossover_l: super::topology::TptLr4Crossover::new_48khz(),
            tpt_crossover_r: super::topology::TptLr4Crossover::new_48khz(),
            wdw_tap_mode: super::topology::WdwDryTapMode::PreSpatialCore,
            wdw_mode: false,
            _pad: [0; 64],
        }
    }

    pub fn reset(&mut self) {
        self.return_aligner_l.reset();
        self.return_aligner_r.reset();
        self.sub_crossover_l.reset();
        self.sub_crossover_r.reset();
        self.tpt_crossover_l.reset();
        self.tpt_crossover_r.reset();
    }

    /// Sets scattering matrix coefficients with strict enforcement of `ref[dsp.router.return-to-send-isolation]` DAG constraint:
    /// `S_ret->send == 0`.
    pub fn set_scattering_matrix(&mut self, s: [[f32; 4]; 4]) {
        self.s_matrix = s;
        // Enforce acyclic DAG return-to-send isolation
        self.s_matrix[0][2] = 0.0;
        self.s_matrix[0][3] = 0.0;
        self.s_matrix[1][2] = 0.0;
        self.s_matrix[1][3] = 0.0;
    }

    /// Voltage-normalized Mid/Side transformation.
    #[inline(always)]
    pub fn voltage_normalize_ms(l: f32, r: f32) -> (f32, f32) {
        (0.5 * (l + r), 0.5 * (l - r))
    }

    /// Voltage-normalized Mid/Side inverse reconstruction.
    #[inline(always)]
    pub fn voltage_reconstruct_ms(mid: f32, side: f32) -> (f32, f32) {
        (mid + side, mid - side)
    }

    /// Routes 4 input ports [inst_l, inst_r, ret_l, ret_r] to sends, downstream feeds, and clean sub paths.
    ///
    /// Returns:
    /// `(send_l, send_r, down_l, down_r, clean_sub_l, clean_sub_r)`
    ///
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn route_frame(
        &mut self,
        inst_l: f32,
        inst_r: f32,
        ret_l: f32,
        ret_r: f32,
    ) -> (f32, f32, f32, f32, f32, f32) {
        // 1. Align return latency via 4-tap Catmull-Rom
        let ret_l_aligned = self.return_aligner_l.step(ret_l);
        let ret_r_aligned = self.return_aligner_r.step(ret_r);

        // 2. Scattering matrix evaluation (ref[dsp.router.return-to-send-isolation]: S_ret->send == 0)
        let send_l = self.s_matrix[0][0] * inst_l + self.s_matrix[0][1] * inst_r;
        let send_r = self.s_matrix[1][0] * inst_l + self.s_matrix[1][1] * inst_r;

        let down_l = self.s_matrix[2][0] * inst_l
            + self.s_matrix[2][1] * inst_r
            + self.s_matrix[2][2] * ret_l_aligned
            + self.s_matrix[2][3] * ret_r_aligned;

        let down_r = self.s_matrix[3][0] * inst_l
            + self.s_matrix[3][1] * inst_r
            + self.s_matrix[3][2] * ret_l_aligned
            + self.s_matrix[3][3] * ret_r_aligned;

        // 3. Parallel clean Bezout sub crossover (< 200 Hz)
        let clean_sub_l = self.sub_crossover_l.step(inst_l);
        let clean_sub_r = self.sub_crossover_r.step(inst_r);

        (send_l, send_r, down_l, down_r, clean_sub_l, clean_sub_r)
    }

    /// Fast-path stereo routing when external FX returns `(ret_l, ret_r)` are unconnected (`0.0`),
    /// returning `(down_l, down_r, clean_sub_l, clean_sub_r)`.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn route_frame_no_return(&mut self, inst_l: f32, inst_r: f32) -> (f32, f32, f32, f32) {
        let down_l = self.s_matrix[2][0] * inst_l + self.s_matrix[2][1] * inst_r;
        let down_r = self.s_matrix[3][0] * inst_l + self.s_matrix[3][1] * inst_r;
        let clean_sub_l = self.sub_crossover_l.step(inst_l);
        let clean_sub_r = self.sub_crossover_r.step(inst_r);
        (down_l, down_r, clean_sub_l, clean_sub_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_bezout_crossover() {
        let _audio = crate::FtzDazGuard::new();
        let mut filter = BezoutCleanCrossover::default();
        // Impulse response at n=0 equals g^4 where g = 0.5 * (1 - pole)
        let g = 0.5 * (1.0 - BezoutCleanCrossover::POLE_200HZ_48K);
        let g4 = g * g * g * g;
        let y0 = filter.step(1.0);
        assert!((y0 - g4).abs() < 1e-10);

        filter.reset();
        assert_eq!(filter.x_hist, [0.0; 4]);
        assert_eq!(filter.y_hist, [0.0; 4]);

        // Steady-state DC gain must be identically 1.000000 (0.000000 dB) across all ZOH sample rates
        for &sr in &[44_100_u32, 48_000, 96_000] {
            let zoh = crate::container::ZohRescaledParams::new(48_000, sr);
            filter.set_crossover_hz(200.0);
            filter.reconfigure_sample_rate(&zoh);
            filter.reset();
            let mut dc_out = 0.0_f32;
            for _ in 0..800 {
                dc_out = filter.step(1.0);
            }
            assert!(
                (dc_out - 1.0).abs() < 1e-4,
                "Expected 1.0 DC gain at {sr} Hz, got {dc_out}"
            );
        }
        filter.set_crossover_hz(150.0);
        assert!(filter.base_pole > BezoutCleanCrossover::POLE_200HZ_48K);
    }

    #[test]
    fn test_return_aligner() {
        let _audio = crate::FtzDazGuard::new();
        let mut aligner = ReturnAligner::default();
        aligner.set_delay(8.5);
        assert_eq!(aligner.delay_samples, 8.5);
        // Feed linear ramp x[n] = n as f32; after 12 samples, Catmull-Rom reproduces (n - 8.5)
        for n in 0..24 {
            let out = aligner.step(n as f32);
            if n >= 12 {
                assert!((out - ((n as f32) - 8.5)).abs() < 1e-4);
            }
        }
        aligner.reset();
        assert_eq!(aligner.write_ptr, 0);
        assert_eq!(aligner.delay_samples, 8.5);

        // Sub-sample delay (< 1.0) must remain causal and never wrap to write_ptr + 1
        aligner.set_delay(0.5);
        aligner.reset();
        aligner.ring[1] = 100.0;
        let out0 = aligner.step(1.0);
        assert!((out0 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_stage0_router_routing_and_ms() {
        let _audio = crate::FtzDazGuard::new();
        let (mid, side) = Stage0Router::voltage_normalize_ms(1.0, 0.5);
        assert_eq!(mid, 0.75);
        assert_eq!(side, 0.25);
        let (l, r) = Stage0Router::voltage_reconstruct_ms(mid, side);
        assert_eq!(l, 1.0);
        assert_eq!(r, 0.5);

        let mut router = Stage0Router::default();
        let custom_s = [
            [0.8, 0.1, 0.9, 0.9], // ret terms should be forced to 0
            [0.1, 0.8, 0.9, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [0.2, 0.2, 0.8, 0.8],
        ];
        router.set_scattering_matrix(custom_s);
        assert_eq!(router.s_matrix[0][2], 0.0);
        assert_eq!(router.s_matrix[0][3], 0.0);
        assert_eq!(router.s_matrix[1][2], 0.0);
        assert_eq!(router.s_matrix[1][3], 0.0);

        let (send_l, send_r, down_l, down_r, sub_l, sub_r) = router.route_frame(1.0, 0.5, 0.2, 0.1);
        assert!(send_l.is_finite());
        assert!(send_r.is_finite());
        assert!(down_l.is_finite());
        assert!(down_r.is_finite());
        assert!(sub_l.is_finite());
        assert!(sub_r.is_finite());

        router.reset();
    }

    #[test]
    fn test_theorem_2_1_drive_invariance() {
        // Theorem 2.1: Drive-Invariant Voltage-Normalized Mid/Side Input Transformation
        // For centered stereo signal x_L = x_R = x, mid = x and side = 0.0f32
        for &x in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let (mid, side) = Stage0Router::voltage_normalize_ms(x, x);
            assert_eq!(
                mid, x,
                "Voltage ratio x_mid / x must be identically 1.000000 (0.000000 dB)"
            );
            assert_eq!(
                side, 0.0,
                "Side signal must be identically 0.0 for centered input"
            );

            let (recon_l, recon_r) = Stage0Router::voltage_reconstruct_ms(mid, side);
            assert_eq!(recon_l, x);
            assert_eq!(recon_r, x);
        }
    }

    proptest! {
        #[test]
        fn prop_ms_roundtrip_and_dag_isolation(
            in_l in -10.0_f32..10.0_f32,
            in_r in -10.0_f32..10.0_f32,
            ret_l in -10.0_f32..10.0_f32,
            ret_r in -10.0_f32..10.0_f32,
            s_leak in -5.0_f32..5.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let (mid, side) = Stage0Router::voltage_normalize_ms(in_l, in_r);
            let (recon_l, recon_r) = Stage0Router::voltage_reconstruct_ms(mid, side);
            prop_assert!((recon_l - in_l).abs() < 1e-5);
            prop_assert!((recon_r - in_r).abs() < 1e-5);

            // ref[dsp.router.return-to-send-isolation]: send_l and send_r must be strictly independent of ret_l and ret_r
            let mut r1 = Stage0Router::default();
            let mut r2 = Stage0Router::default();
            let s_mat = [
                [0.9, 0.1, s_leak, s_leak],
                [0.1, 0.9, s_leak, s_leak],
                [0.5, 0.5, 0.5, 0.5],
                [0.3, 0.3, 0.7, 0.7],
            ];
            r1.set_scattering_matrix(s_mat);
            r2.set_scattering_matrix(s_mat);
            let (send_l1, send_r1, _, _, _, _) = r1.route_frame(in_l, in_r, 0.0, 0.0);
            let (send_l2, send_r2, _, _, _, _) = r2.route_frame(in_l, in_r, ret_l, ret_r);
            prop_assert_eq!(send_l1, send_l2);
            prop_assert_eq!(send_r1, send_r2);
        }
    }
}
