//! Stage II: 3-Stage Commuted Nonlinear Amp Cascade (288 B Lite RAM).
//!
//! Canonical Reference: `docs/dsp/op_n_commuted_cascade.md` §4 & `docs/dsp/router_and_stages_topology.md` §2.2.

use crate::math::flush_recursive_state_f32;
use crate::operators::op_n::{
    MultiDrivePolyAdaaChannel, QuadratureHysteresis2D, c2_wavefolder,
    eval_adaa_cubic_c2_wavefolder_odd_f32,
};

/// State RAM footprint for Stage II Lite: strictly 288 B.
pub const STAGE2_CASCADE_RAM_BYTES: usize = 288;

/// 4-Filter Tone Basis Filter State: [Direct, LPF_80, LPF_400, HPF_1200].
#[derive(Clone, Copy, Debug)]
pub struct ToneBasisFilters {
    pub lpf80_state: f32,
    pub lpf400_state: f32,
    pub hpf1200_state: f32,
    pub hpf1200_x1: f32,
    pub alpha_80: f32,
    pub alpha_400: f32,
    pub alpha_1200: f32,
}

impl Default for ToneBasisFilters {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneBasisFilters {
    // 1-pole filter coefficients at 48 kHz
    // 80 Hz LPF: alpha = exp(-2*pi*80/48000) ~ 0.98959
    pub const ALPHA_80: f32 = 0.98959;
    // 400 Hz LPF: alpha = exp(-2*pi*400/48000) ~ 0.94916
    pub const ALPHA_400: f32 = 0.94916;
    // 1200 Hz HPF: alpha = exp(-2*pi*1200/48000) ~ 0.85463
    pub const ALPHA_1200: f32 = 0.85463;

    pub const fn new() -> Self {
        Self {
            lpf80_state: 0.0,
            lpf400_state: 0.0,
            hpf1200_state: 0.0,
            hpf1200_x1: 0.0,
            alpha_80: Self::ALPHA_80,
            alpha_400: Self::ALPHA_400,
            alpha_1200: Self::ALPHA_1200,
        }
    }

    /// Initializes the 4-filter tone basis (`Direct`, `LPF_80`, `LPF_400`, `HPF_1200`) to exact
    /// resting DC equilibrium for constant input `v_rest` (`LPF y[-1] = v_rest`,
    /// `HPF y[-1] = 0, x[-1] = v_rest`, `ref[dsp.op-n.commuted.dc-equilibrium]`).
    pub fn init_dc_equilibrium(&mut self, v_rest: f32) {
        let vr = flush_recursive_state_f32(v_rest);
        self.lpf80_state = vr;
        self.lpf400_state = vr;
        self.hpf1200_state = 0.0;
        self.hpf1200_x1 = vr;
    }

    pub fn reset(&mut self) {
        self.lpf80_state = 0.0;
        self.lpf400_state = 0.0;
        self.hpf1200_state = 0.0;
        self.hpf1200_x1 = 0.0;
    }

    /// Reconfigures 1-pole tone basis cutoffs (`80 Hz`, `400 Hz`, `1200 Hz`) under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.alpha_80 = zoh.rescale_real_pole(Self::ALPHA_80);
        self.alpha_400 = zoh.rescale_real_pole(Self::ALPHA_400);
        self.alpha_1200 = zoh.rescale_real_pole(Self::ALPHA_1200);
    }

    /// Advances the tone-basis filters by one sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> [f32; 4] {
        // Direct
        let s_direct = x;

        // LPF 80 Hz
        let s_lpf80 = self.alpha_80 * self.lpf80_state + (1.0 - self.alpha_80) * x;
        self.lpf80_state = flush_recursive_state_f32(s_lpf80);

        // LPF 400 Hz
        let s_lpf400 = self.alpha_400 * self.lpf400_state + (1.0 - self.alpha_400) * x;
        self.lpf400_state = flush_recursive_state_f32(s_lpf400);

        // HPF 1200 Hz (y[n] = alpha * y[n-1] + alpha * (x[n] - x[n-1]))
        let s_hpf1200 =
            self.alpha_1200 * self.hpf1200_state + self.alpha_1200 * (x - self.hpf1200_x1);
        self.hpf1200_state = flush_recursive_state_f32(s_hpf1200);
        self.hpf1200_x1 = flush_recursive_state_f32(x);

        [s_direct, s_lpf80, s_lpf400, s_hpf1200]
    }
}

/// Stage II 3-Stage Commuted Nonlinear Cascade (288 B Lite RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage2CommutedCascade {
    /// 3 Commuted sub-stages:
    /// Sub-stage 2.1 (V1A Preamp)
    pub v1a_left: MultiDrivePolyAdaaChannel,
    pub v1a_right: MultiDrivePolyAdaaChannel,
    /// Sub-stage 2.2 (V1B Driver)
    pub v1b_left: MultiDrivePolyAdaaChannel,
    pub v1b_right: MultiDrivePolyAdaaChannel,
    /// Sub-stage 2.3 (Power Amp sag & hysteresis)
    pub hyst_left: QuadratureHysteresis2D,
    pub hyst_right: QuadratureHysteresis2D,
    /// 4-Filter Tone Basis filters for Left and Right channels
    pub tone_basis_l: ToneBasisFilters,
    pub tone_basis_r: ToneBasisFilters,
    /// Power supply sag integrator
    pub sag_envelope: f32,
    /// ZOH-rescaled 5 Hz AC-coupling pole (default `0.9993` at 48 kHz).
    pub r_ac: f32,
    /// ZOH-rescaled power-supply sag pole (default `0.995` at 48 kHz).
    pub r_sag: f32,
    /// ZOH-rescaled 2D quadrature hysteresis differential rate scale (`1 / rho_sr`).
    pub hyst_diff_scale: f32,
    /// Previous V1A output for degree-7/8 ADAA `C^2` wavefolding (`ref[dsp.op-n.wavefolder.cubic-c2]`).
    pub prev_v1a_l: f32,
    pub prev_v1a_r: f32,
    pub _pad: [u8; 4],
}

impl Default for Stage2CommutedCascade {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage2CommutedCascade {
    pub const fn new() -> Self {
        Self {
            v1a_left: MultiDrivePolyAdaaChannel::new(),
            v1a_right: MultiDrivePolyAdaaChannel::new(),
            v1b_left: MultiDrivePolyAdaaChannel::new(),
            v1b_right: MultiDrivePolyAdaaChannel::new(),
            hyst_left: QuadratureHysteresis2D::new(),
            hyst_right: QuadratureHysteresis2D::new(),
            tone_basis_l: ToneBasisFilters::new(),
            tone_basis_r: ToneBasisFilters::new(),
            sag_envelope: 0.0,
            r_ac: 0.9993,
            r_sag: 0.995,
            hyst_diff_scale: 1.0,
            prev_v1a_l: 0.0,
            prev_v1a_r: 0.0,
            _pad: [0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.v1a_left.reset();
        self.v1a_right.reset();
        self.v1b_left.reset();
        self.v1b_right.reset();
        self.hyst_left.reset();
        self.hyst_right.reset();
        self.tone_basis_l.reset();
        self.tone_basis_r.reset();
        self.sag_envelope = 0.0;
        self.prev_v1a_l = 0.0;
        self.prev_v1a_r = 0.0;
    }

    /// Reconfigures Stage II AC-coupling pole, sag integrator pole, hysteresis rate scale,
    /// and 4-filter tone basis cutoffs for host sample rate under continuous-time ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.r_ac = zoh.r_ac_host;
        self.r_sag = zoh.rescale_real_pole(0.995);
        self.hyst_diff_scale = zoh.hysteresis_diff_scale();
        self.tone_basis_l.reconfigure_sample_rate(zoh);
        self.tone_basis_r.reconfigure_sample_rate(zoh);
    }

    /// Initializes exact DC resting equilibrium for sample-0 pop-free startup.
    pub fn init_dc_equilibrium(&mut self, b_conv: f32, beta: f32, delta: f32) {
        self.v1a_left.init_dc_equilibrium(b_conv, beta, delta);
        self.v1a_right.init_dc_equilibrium(b_conv, beta, delta);
        self.v1b_left.init_dc_equilibrium(b_conv, beta, delta);
        self.v1b_right.init_dc_equilibrium(b_conv, beta, delta);
        self.hyst_left.reset();
        self.hyst_right.reset();
        self.tone_basis_l.reset();
        self.tone_basis_r.reset();
        self.sag_envelope = 0.0;
        self.prev_v1a_l = 0.0;
        self.prev_v1a_r = 0.0;
    }

    /// Steps Stage II for stereo frame `(in_l, in_r)` through 2.1 Preamp -> 2.2 Driver -> 2.3 Power Amp.
    ///
    /// Returns stereo tone basis channels: `([f32; 8], [f32; 8])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_frame(
        &mut self,
        in_l: f32,
        in_r: f32,
        drive_gain: f32,
        sag_depth: f32,
    ) -> ([f32; 8], [f32; 8]) {
        let r_ac = self.r_ac;
        // 1. Sub-stage 2.1: Preamp V1A
        let v1a_out_l = self.v1a_left.step(in_l * drive_gain, 1.0, 0.25, 0.5, r_ac);
        let v1a_out_r = self.v1a_right.step(in_r * drive_gain, 1.0, 0.25, 0.5, r_ac);
        let sig_v1a_l = v1a_out_l[1]; // 1.0x drive
        let sig_v1a_r = v1a_out_r[1];

        // 2. Sub-stage 2.2: Driver V1B + degree-7/8 ADAA C2 wavefolding (`ref[dsp.op-n.wavefolder.cubic-c2]`)
        let wf_adaa_l = eval_adaa_cubic_c2_wavefolder_odd_f32(self.prev_v1a_l, sig_v1a_l);
        let wf_adaa_r = eval_adaa_cubic_c2_wavefolder_odd_f32(self.prev_v1a_r, sig_v1a_r);
        self.prev_v1a_l = flush_recursive_state_f32(sig_v1a_l);
        self.prev_v1a_r = flush_recursive_state_f32(sig_v1a_r);
        let (wf_inst_l, _) = c2_wavefolder(sig_v1a_l);
        let (wf_inst_r, _) = c2_wavefolder(sig_v1a_r);
        let wf_l = 0.5 * wf_adaa_l + 0.5 * wf_inst_l;
        let wf_r = 0.5 * wf_adaa_r + 0.5 * wf_inst_r;
        let v1b_in_l = 0.7 * sig_v1a_l + 0.3 * wf_l;
        let v1b_in_r = 0.7 * sig_v1a_r + 0.3 * wf_r;

        let (v1b_l1, v1b_l2, v1b_r1, v1b_r2) = MultiDrivePolyAdaaChannel::step_stereo_lanes_1_2(
            &mut self.v1b_left,
            &mut self.v1b_right,
            v1b_in_l,
            v1b_in_r,
            1.0,
            0.2,
            0.5,
            r_ac,
        );

        // 3. Sub-stage 2.3: Power amp dynamic sag & magnetic hysteresis
        let pwr = 0.5 * (v1b_l2.abs() + v1b_r2.abs());
        let next_sag = self.r_sag * self.sag_envelope + (1.0 - self.r_sag) * pwr;
        self.sag_envelope = flush_recursive_state_f32(next_sag);
        let sag_compression = (1.0 - sag_depth * next_sag).max(0.2);

        let h_l = self
            .hyst_left
            .step_with_rate(v1b_l1, v1b_l2, self.hyst_diff_scale);
        let h_r = self
            .hyst_right
            .step_with_rate(v1b_r1, v1b_r2, self.hyst_diff_scale);

        let pwr_l = (v1b_l1 + 0.1 * h_l) * sag_compression;
        let pwr_r = (v1b_r1 + 0.1 * h_r) * sag_compression;

        // 4. Tone basis filters [Direct, LPF_80, LPF_400, HPF_1200]
        let tone_l = self.tone_basis_l.step(pwr_l);
        let tone_r = self.tone_basis_r.step(pwr_r);

        // Form 8L and 8R basis
        let mut basis_l = [0.0; 8];
        let mut basis_r = [0.0; 8];
        basis_l[0..4].copy_from_slice(&tone_l);
        basis_l[4..8].copy_from_slice(&v1a_out_l);

        basis_r[0..4].copy_from_slice(&tone_r);
        basis_r[4..8].copy_from_slice(&v1a_out_r);

        (basis_l, basis_r)
    }

    /// Evaluates Stage II returning primary stereo audio channels `(out_l, out_r)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let (basis_l, basis_r) = self.step_frame(in_l, in_r, 1.0, 0.2);
        (basis_l[0], basis_r[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage2_cascade_and_tone_basis() {
        let _audio = crate::FtzDazGuard::new();
        let mut filters = ToneBasisFilters::default();
        // Steady-state DC input (1.0): Direct = 1.0, LPF_80 -> 1.0, LPF_400 -> 1.0, HPF_1200 -> 0.0
        let mut b = [0.0_f32; 4];
        for _ in 0..2000 {
            b = filters.step(1.0);
        }
        assert_eq!(b[0], 1.0);
        assert!((b[1] - 1.0).abs() < 0.05);
        assert!((b[2] - 1.0).abs() < 1e-3);
        assert!(b[3].abs() < 1e-3);
        filters.reset();
        assert_eq!(filters.lpf80_state, 0.0);

        let mut cascade = Stage2CommutedCascade::default();
        cascade.init_dc_equilibrium(0.1, 0.5, 0.2);
        let (bl, br) = cascade.step_frame(0.3, -0.3, 1.5, 0.4);
        // Antisymmetric stereo input (0.3, -0.3) produces non-zero opposite-sign direct outputs
        assert!(bl[0] > 0.0 && br[0] < 0.0);
        assert!(cascade.sag_envelope > 0.0);

        let (sl, sr) = cascade.step_stereo(0.2, -0.2);
        assert!(sl > 0.0 && sr < 0.0);

        cascade.reset();
        assert_eq!(cascade.sag_envelope, 0.0);
    }

    proptest! {
        #[test]
        fn prop_stage2_cascade_dc_equilibrium_and_bounded(
            b0 in -0.5_f32..0.5_f32,
            b1 in 0.1_f32..1.0_f32,
            b2 in -0.5_f32..0.5_f32,
            in_l in -2.0_f32..2.0_f32,
            in_r in -2.0_f32..2.0_f32,
            drive in 0.1_f32..4.0_f32,
            asym in -0.5_f32..0.5_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            // 1. MultiDrivePolyAdaaChannel exact DC equilibrium for arbitrary bias b0
            let mut ch = MultiDrivePolyAdaaChannel::new();
            ch.init_dc_equilibrium(b0, b1, b2);
            let out_dc = ch.step(b0, b1, b2, 0.5, 0.9993);
            for k in 0..4 {
                prop_assert!((out_dc[k] - 0.5 * b0).abs() < 1e-6);
            }

            // 2. Stage2CommutedCascade zero-bias resting equilibrium and bounded stereo stepping
            let mut cascade = Stage2CommutedCascade::default();
            cascade.init_dc_equilibrium(0.0, b1, b2);
            let (bl_zero, br_zero) = cascade.step_frame(0.0, 0.0, drive, asym);
            for k in 0..8 {
                prop_assert!(bl_zero[k].abs() < 1e-6, "DC equilibrium violated on L[{}]: {}", k, bl_zero[k]);
                prop_assert!(br_zero[k].abs() < 1e-6, "DC equilibrium violated on R[{}]: {}", k, br_zero[k]);
            }

            let (bl, br) = cascade.step_frame(in_l, in_r, drive, asym);
            for k in 0..8 {
                prop_assert!(bl[k].is_finite());
                prop_assert!(br[k].is_finite());
            }
        }
    }
}
