//! Stage II: 3-Stage Commuted Nonlinear Amp Cascade (288 B Lite RAM).
//!
//! Canonical Reference: `docs/dsp/op_n_commuted_cascade.md` §4 & `docs/dsp/router_and_stages_topology.md` §2.2.

pub mod conv6;
pub mod filters;
pub use conv6::{
    Stage2Backend, conv6_taps_1group_vec4, conv6_taps_2groups_vec4, conv6_taps_3groups_vec4,
    conv6_taps_4groups_vec4,
};
pub use filters::ToneBasisFilters;

use crate::math::flush_recursive_state_f32;
use crate::operators::op_n::{
    MultiDrivePolyAdaaChannel, QuadratureHysteresis2D, c2_wavefolder,
    eval_adaa_cubic_c2_wavefolder_odd_f32,
};

/// State RAM footprint for Stage II Lite: strictly 288 B.
pub const STAGE2_CASCADE_RAM_BYTES: usize = 288;

/// Full 4-tier Stage II `L-N-L-N` cascade frame output (`Lite` 32 + `Full` 32 + `Pro` 32 + `Ultra` 22 = 118 features).
#[derive(Clone, Copy, Debug)]
pub struct Stage2CascadeFrameOut {
    pub s2_l: f32,
    pub s2_r: f32,
    pub basis_l: [f32; 8],
    pub basis_r: [f32; 8],
    pub lite_tail: [f32; 16],
    pub full_features: [f32; 32],
    pub pro_features: [f32; 32],
    pub ultra_features: [f32; 22],
}

/// Stage II 3-Stage Commuted Nonlinear Cascade (288 B Lite nominal RAM budget).
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
    /// Base 48 kHz V1A/V1B nonlinear parameters loaded from `Arena 0 [4,096..4,160]`.
    pub v1a_beta: f32,
    pub v1a_delta: f32,
    pub v1a_comp: f32,
    pub v1a_beta_r: f32,
    pub v1a_delta_r: f32,
    pub v1a_comp_r: f32,
    pub v1b_beta: f32,
    pub v1b_delta: f32,
    pub v1b_comp: f32,
    pub v1b_beta_r: f32,
    pub v1b_delta_r: f32,
    pub v1b_comp_r: f32,
    /// Base 48 kHz power-supply sag pole (default `0.995`).
    pub base_r_sag: f32,
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
    /// Latest V1B + Hysteresis + Wavefolder taps (`[v1b_l1, v1b_l2, h_l, wf_l, v1b_r1, v1b_r2, h_r, wf_r]`).
    pub last_v1b_hyst: [f32; 8],
    /// Inter-stage `L^{(1)}` (`4,160..4,192`) and `L^{(2)}` (`4,192..4,224`) base & host parameters.
    pub base_l1_poles: [f32; 5], // [alpha_hpf_1, alpha_shelf_1, alpha_bias_rc_1, r_mid_1, theta_mid_1]
    pub base_l1_gains: [f32; 3], // [g_shelf_1, g_mid_1, g_bias_shift_1]
    pub base_l2_poles: [f32; 5], // [alpha_hpf_2, alpha_shelf_2, alpha_bias_rc_2, r_mid_2, theta_mid_2]
    pub base_l2_gains: [f32; 3], // [g_shelf_2, g_mid_2, g_bias_shift_2]
    pub l1_poles: [f32; 6], // [alpha_hpf_1, alpha_shelf_1, alpha_bias_rc_1, mid1_a_re, mid1_a_im, mid1_b_gamma]
    pub l2_poles: [f32; 6], // [alpha_hpf_2, alpha_shelf_2, alpha_bias_rc_2, mid2_a_re, mid2_a_im, mid2_b_gamma]
    /// Inter-stage filter & RC bias states for `L^{(1)}` and `L^{(2)}` (`[hpf_x, hpf_y, shelf_y, mid_re, mid_im, b_rc]`).
    pub l1_state_l: [f32; 6],
    pub l1_state_r: [f32; 6],
    pub l2_state_l: [f32; 6],
    pub l2_state_r: [f32; 6],
    /// Higher-tier branch parameters (`Full` `V2A` at `4,224..4,240`, `Pro` `V2B` at `4,240..4,256`, `Ultra` `V3` at `4,256..4,272`).
    pub base_branch_params: [[f32; 4]; 3], // [alpha_lpf, alpha_hpf, drive, asym] per branch
    pub branch_poles: [[f32; 2]; 3], // ZOH-rescaled [alpha_lpf, alpha_hpf] per branch
    pub branch_state_l: [[f32; 3]; 3], // [lpf_y, hpf_x, hpf_y] per branch (L)
    pub branch_state_r: [[f32; 3]; 3], // [lpf_y, hpf_x, hpf_y] per branch (R)
    /// Global cascade scales (`4,272..4,288`: `[hyst_drive_scale, hyst_diff_scale_override, sag_attack_alpha, sag_release_alpha]`).
    pub base_global_scales: [f32; 4],
    pub sag_attack_alpha: f32,
    pub sag_release_alpha: f32,
    /// Anchor-locked Role A (`b = 0`, 16 floats/tier at `CASCADE_W_BASES[t][0..16]`) inter-stage coupling scales `(1.0 + delta_a).clamp(0.5, 1.5)`.
    pub role_a_scales: [[f32; 16]; 4],
}

impl Default for Stage2CommutedCascade {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage2CommutedCascade {
    pub const DEFAULT_L1_POLES: [f32; 5] = [0.9955, 0.855, 0.9980, 0.85, 0.1047];
    pub const DEFAULT_L2_POLES: [f32; 5] = [0.9955, 0.920, 0.9990, 0.85, 0.0785];
    pub const DEFAULT_BRANCH_PARAMS: [[f32; 4]; 3] = [
        [0.78, 0.994, 1.25, 0.20],
        [0.88, 0.988, 1.50, 0.30],
        [0.94, 0.976, 1.75, 0.35],
    ];
    pub const DEFAULT_GLOBAL_SCALES: [f32; 4] = [1.0, 1.0, 0.980, 0.999];

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
            v1a_beta: 1.0,
            v1a_delta: 0.25,
            v1a_comp: 0.5,
            v1a_beta_r: 1.0,
            v1a_delta_r: 0.25,
            v1a_comp_r: 0.5,
            v1b_beta: 1.0,
            v1b_delta: 0.2,
            v1b_comp: 0.5,
            v1b_beta_r: 1.0,
            v1b_delta_r: 0.2,
            v1b_comp_r: 0.5,
            base_r_sag: 0.995,
            sag_envelope: 0.0,
            r_ac: 0.9993,
            r_sag: 0.995,
            hyst_diff_scale: 1.0,
            prev_v1a_l: 0.0,
            prev_v1a_r: 0.0,
            last_v1b_hyst: [0.0; 8],
            base_l1_poles: Self::DEFAULT_L1_POLES,
            base_l1_gains: [0.0; 3],
            base_l2_poles: Self::DEFAULT_L2_POLES,
            base_l2_gains: [0.0; 3],
            l1_poles: [0.9955, 0.855, 0.9980, 0.84534, 0.08883, 0.52678],
            l2_poles: [0.9955, 0.920, 0.9990, 0.84738, 0.06665, 0.52678],
            l1_state_l: [0.0; 6],
            l1_state_r: [0.0; 6],
            l2_state_l: [0.0; 6],
            l2_state_r: [0.0; 6],
            base_branch_params: Self::DEFAULT_BRANCH_PARAMS,
            branch_poles: [[0.78, 0.994], [0.88, 0.988], [0.94, 0.976]],
            branch_state_l: [[0.0; 3]; 3],
            branch_state_r: [[0.0; 3]; 3],
            base_global_scales: Self::DEFAULT_GLOBAL_SCALES,
            sag_attack_alpha: 0.980,
            sag_release_alpha: 0.999,
            role_a_scales: [[1.0; 16]; 4],
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
        self.last_v1b_hyst = [0.0; 8];
        self.l1_state_l = [0.0; 6];
        self.l1_state_r = [0.0; 6];
        self.l2_state_l = [0.0; 6];
        self.l2_state_r = [0.0; 6];
        self.branch_state_l = [[0.0; 3]; 3];
        self.branch_state_r = [[0.0; 3]; 3];
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
        self.last_v1b_hyst = [0.0; 8];
        self.l1_state_l = [0.0; 6];
        self.l1_state_r = [0.0; 6];
        self.l2_state_l = [0.0; 6];
        self.l2_state_r = [0.0; 6];
        self.branch_state_l = [[0.0; 3]; 3];
        self.branch_state_r = [[0.0; 3]; 3];
    }

    #[allow(clippy::type_complexity)]
    #[inline(always)]
    fn step_frame_lite_core(
        &mut self,
        in_l: f32,
        in_r: f32,
        drive_gain: f32,
        sag_depth: f32,
        beta_tone: f32,
        pwr_drive: f32,
    ) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], f32, f32) {
        let r_ac = self.r_ac;
        let ra0 = self.role_a_scales[0];

        // 1. Sub-stage 2.1: Preamp V1A
        let (v1a_out_l, v1a_out_r) = MultiDrivePolyAdaaChannel::step_stereo_pair(
            &mut self.v1a_left,
            &mut self.v1a_right,
            in_l * drive_gain,
            in_r * drive_gain,
            self.v1a_beta,
            self.v1a_delta,
            self.v1a_comp,
            self.v1a_beta_r,
            self.v1a_delta_r,
            self.v1a_comp_r,
            r_ac,
        );
        let sig_v1a_l = v1a_out_l[1] * ra0[1];
        let sig_v1a_r = v1a_out_r[1] * ra0[5];
        let ae1_l = 0.25
            * (self.v1a_left.last_a_even[0] * ra0[0]
                + self.v1a_left.last_a_even[1] * ra0[1]
                + self.v1a_left.last_a_even[2] * ra0[2]
                + self.v1a_left.last_a_even[3] * ra0[3]);
        let ae1_r = 0.25
            * (self.v1a_right.last_a_even[0] * ra0[4]
                + self.v1a_right.last_a_even[1] * ra0[5]
                + self.v1a_right.last_a_even[2] * ra0[6]
                + self.v1a_right.last_a_even[3] * ra0[7]);

        // Inter-stage L^(1) filter + RC grid-leak bias shift
        let (l1_out_l, l1_out_r) = Self::step_interstage_filter_stereo(
            &mut self.l1_state_l,
            &mut self.l1_state_r,
            &self.l1_poles,
            &self.base_l1_gains,
            sig_v1a_l,
            ae1_l,
            sig_v1a_r,
            ae1_r,
        );

        // 2. Sub-stage 2.2: Driver V1B + degree-7/8 ADAA C2 wavefolding (`ref[dsp.op-n.wavefolder.cubic-c2]`)
        let prev_l = self.prev_v1a_l;
        let prev_r = self.prev_v1a_r;
        self.prev_v1a_l = flush_recursive_state_f32(l1_out_l);
        self.prev_v1a_r = flush_recursive_state_f32(l1_out_r);

        let (wf_inst_l, _) = c2_wavefolder(l1_out_l);
        let (wf_inst_r, _) = c2_wavefolder(l1_out_r);
        let wf_adaa_l = eval_adaa_cubic_c2_wavefolder_odd_f32(prev_l, l1_out_l);
        let wf_adaa_r = eval_adaa_cubic_c2_wavefolder_odd_f32(prev_r, l1_out_r);

        let wf_l = 0.5 * (wf_adaa_l + wf_inst_l);
        let wf_r = 0.5 * (wf_adaa_r + wf_inst_r);
        let v1b_in_l = 0.7 * l1_out_l + 0.3 * wf_l;
        let v1b_in_r = 0.7 * l1_out_r + 0.3 * wf_r;

        let (v1b_l1_raw, v1b_l2_raw, v1b_r1_raw, v1b_r2_raw) =
            MultiDrivePolyAdaaChannel::step_stereo_lanes_1_2_lr(
                &mut self.v1b_left,
                &mut self.v1b_right,
                v1b_in_l,
                v1b_in_r,
                self.v1b_beta,
                self.v1b_delta,
                self.v1b_comp,
                self.v1b_beta_r,
                self.v1b_delta_r,
                self.v1b_comp_r,
                r_ac,
            );

        let ae2_l =
            0.5 * (self.v1b_left.last_a_even[1] * ra0[8] + self.v1b_left.last_a_even[2] * ra0[9]);
        let ae2_r = 0.5
            * (self.v1b_right.last_a_even[1] * ra0[12] + self.v1b_right.last_a_even[2] * ra0[13]);

        // Inter-stage L^(2) filter
        let (v1b_l1, v1b_r1) = Self::step_interstage_filter_stereo(
            &mut self.l2_state_l,
            &mut self.l2_state_r,
            &self.l2_poles,
            &self.base_l2_gains,
            v1b_l1_raw * ra0[10],
            ae2_l,
            v1b_r1_raw * ra0[14],
            ae2_r,
        );
        let v1b_l2 = v1b_l2_raw * ra0[11];
        let v1b_r2 = v1b_r2_raw * ra0[15];

        // 3. Sub-stage 2.3: Power amp dynamic sag & magnetic hysteresis
        let pwr = 0.5 * (v1b_l2.abs() + v1b_r2.abs());
        let r_sag_eff = if (self.base_global_scales[2] - Self::DEFAULT_GLOBAL_SCALES[2]).abs()
            > 1e-6
            || (self.base_global_scales[3] - Self::DEFAULT_GLOBAL_SCALES[3]).abs() > 1e-6
        {
            if pwr > self.sag_envelope {
                self.sag_attack_alpha
            } else {
                self.sag_release_alpha
            }
        } else {
            self.r_sag
        };
        let next_sag = r_sag_eff * self.sag_envelope + (1.0 - r_sag_eff) * pwr;
        self.sag_envelope = flush_recursive_state_f32(next_sag);
        let sag_compression = (1.0 - sag_depth * next_sag).max(0.2);

        let h_drv = self.base_global_scales[0] * pwr_drive;
        let h_l =
            self.hyst_left
                .step_with_rate(v1b_l1 * h_drv, v1b_l2 * h_drv, self.hyst_diff_scale);
        let h_r =
            self.hyst_right
                .step_with_rate(v1b_r1 * h_drv, v1b_r2 * h_drv, self.hyst_diff_scale);

        let pwr_l = (v1b_l1 * pwr_drive + 0.1 * h_l) * sag_compression;
        let pwr_r = (v1b_r1 * pwr_drive + 0.1 * h_r) * sag_compression;
        self.last_v1b_hyst = [v1b_l1, v1b_l2, h_l, wf_l, v1b_r1, v1b_r2, h_r, wf_r];

        // 4. Tone basis filters [Direct, LPF_80, LPF_400, HPF_1200]
        let (mut tone_l, mut tone_r) =
            self.tone_basis_l
                .step_stereo(&mut self.tone_basis_r, pwr_l, pwr_r);
        let tone_tilt = beta_tone.clamp(0.0, 1.0) - 0.5;
        if tone_tilt.abs() > 1e-6 {
            tone_l[0] += 0.25 * tone_tilt * (tone_l[3] - tone_l[1]);
            tone_r[0] += 0.25 * tone_tilt * (tone_r[3] - tone_r[1]);
        }

        (
            tone_l,
            v1a_out_l,
            tone_r,
            v1a_out_r,
            wf_adaa_l - wf_inst_l,
            wf_adaa_r - wf_inst_r,
        )
    }

    /// Steps Stage II `L-N-L-N` cascade and feedforward tier branches (`Lite..Ultra`),
    /// writing all Stage II features directly into `fb` (`20..52` Lite, `84..116` Full,
    /// `152..184` Pro, `224..246` Ultra) and returning `(s2_l, s2_r)`.
    /// Keeps primary `(s2_l, s2_r)` and `Lite` state strictly `Arena 0`-only (`REQ-5.6`).
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_frame_into_feature_bus(
        &mut self,
        in_l: f32,
        in_r: f32,
        drive_gain: f32,
        sag_depth: f32,
        beta_tone: f32,
        pwr_drive: f32,
        branch_tiers: usize,
        fb: &mut [f32; 256],
    ) -> (f32, f32) {
        let (tone_l, v1a_out_l, tone_r, v1a_out_r, wf_diff_l, wf_diff_r) =
            self.step_frame_lite_core(in_l, in_r, drive_gain, sag_depth, beta_tone, pwr_drive);
        let brc1_l = self.l1_state_l[5];
        let brc1_r = self.l1_state_r[5];

        // Write Lite Stage II slots directly into fb[20..52]
        fb[20..24].copy_from_slice(&tone_l);
        fb[24..28].copy_from_slice(&v1a_out_l);
        fb[28..32].copy_from_slice(&tone_r);
        fb[32..36].copy_from_slice(&v1a_out_r);
        fb[36..44].copy_from_slice(&self.last_v1b_hyst);
        fb[44..52].copy_from_slice(&[
            self.v1a_left.last_a_even[0],
            self.v1a_left.last_a_even[1],
            self.v1a_left.last_a_even[2],
            brc1_l,
            self.v1a_right.last_a_even[0],
            self.v1a_right.last_a_even[1],
            self.v1a_right.last_a_even[2],
            brc1_r,
        ]);

        if branch_tiers >= 2 {
            let [x1_l, hpf1_l, lpf1_l, mid1_re_l, mid1_im_l, _] = self.l1_state_l;
            let [x1_r, hpf1_r, lpf1_r, mid1_re_r, mid1_im_r, _] = self.l1_state_r;
            let [x2_l, hpf2_l, lpf2_l, mid2_re_l, mid2_im_l, brc2_l] = self.l2_state_l;
            let [x2_r, hpf2_r, lpf2_r, mid2_re_r, mid2_im_r, brc2_r] = self.l2_state_r;
            let sh1_l = x1_l - lpf1_l;
            let sh1_r = x1_r - lpf1_r;
            let sh2_l = x2_l - lpf2_l;
            let sh2_r = x2_r - lpf2_r;
            let [v1b_l1, v1b_l2, h_l, wf_l, v1b_r1, v1b_r2, h_r, wf_r] = self.last_v1b_hyst;
            let next_sag = self.sag_envelope;

            let ra1 = &self.role_a_scales[1];
            let ((v2a_nl_l, v2a_lpf_l, v2a_hpf_l), (v2a_nl_r, v2a_lpf_r, v2a_hpf_r)) =
                Self::step_branch_shaper_stereo(
                    &mut self.branch_state_l[0],
                    &mut self.branch_state_r[0],
                    &self.branch_poles[0],
                    &self.base_branch_params[0],
                    hpf1_l + 0.5 * mid1_re_l,
                    hpf1_r + 0.5 * mid1_re_r,
                );
            fb[84..116].copy_from_slice(&[
                v2a_nl_l * ra1[0],
                v2a_lpf_l * ra1[1],
                v2a_hpf_l * ra1[2],
                hpf1_l * ra1[3],
                sh1_l * ra1[4],
                mid1_re_l * ra1[5],
                mid1_im_l * ra1[6],
                hpf2_l * ra1[7],
                v2a_nl_r * ra1[8],
                v2a_lpf_r * ra1[9],
                v2a_hpf_r * ra1[10],
                hpf1_r * ra1[11],
                sh1_r * ra1[12],
                mid1_re_r * ra1[13],
                mid1_im_r * ra1[14],
                hpf2_r * ra1[15],
                brc2_l,
                self.v1b_left.last_a_even[1],
                self.v1b_left.last_a_even[2],
                self.v1a_left.last_a_even[3],
                brc2_r,
                self.v1b_right.last_a_even[1],
                self.v1b_right.last_a_even[2],
                self.v1a_right.last_a_even[3],
                sh2_l,
                mid2_re_l,
                mid2_im_l,
                wf_diff_l,
                sh2_r,
                mid2_re_r,
                mid2_im_r,
                wf_diff_r,
            ]);

            if branch_tiers >= 3 {
                let ra2 = &self.role_a_scales[2];
                let ((v2b_nl_l, v2b_lpf_l, v2b_hpf_l), (v2b_nl_r, v2b_lpf_r, v2b_hpf_r)) =
                    Self::step_branch_shaper_stereo(
                        &mut self.branch_state_l[1],
                        &mut self.branch_state_r[1],
                        &self.branch_poles[1],
                        &self.base_branch_params[1],
                        v2a_nl_l + 0.5 * hpf2_l,
                        v2a_nl_r + 0.5 * hpf2_r,
                    );
                let (wf2_l, _) = c2_wavefolder(v2b_nl_l);
                let (wf2_r, _) = c2_wavefolder(v2b_nl_r);
                fb[152..184].copy_from_slice(&[
                    v2b_nl_l * ra2[0],
                    v2b_lpf_l * ra2[1],
                    v2b_hpf_l * ra2[2],
                    wf2_l * ra2[3],
                    v1a_out_l[0] * v1b_l1 * ra2[4],
                    v1a_out_l[1] * v1b_l2 * ra2[5],
                    v1a_out_l[2] * wf_l * ra2[6],
                    v1a_out_l[3] * h_l * ra2[7],
                    v2b_nl_r * ra2[8],
                    v2b_lpf_r * ra2[9],
                    v2b_hpf_r * ra2[10],
                    wf2_r * ra2[11],
                    v1a_out_r[0] * v1b_r1 * ra2[12],
                    v1a_out_r[1] * v1b_r2 * ra2[13],
                    v1a_out_r[2] * wf_r * ra2[14],
                    v1a_out_r[3] * h_r * ra2[15],
                    hpf1_l * v1b_l1,
                    sh1_l * v1b_l2,
                    mid1_re_l * wf_l,
                    brc1_l * v1b_l1,
                    tone_l[1] * h_l,
                    tone_l[2] * wf_l,
                    tone_l[3] * v2a_nl_l,
                    next_sag * v1b_l1,
                    hpf1_r * v1b_r1,
                    sh1_r * v1b_r2,
                    mid1_re_r * wf_r,
                    brc1_r * v1b_r1,
                    tone_r[1] * h_r,
                    tone_r[2] * wf_r,
                    tone_r[3] * v2a_nl_r,
                    next_sag * v1b_r1,
                ]);

                if branch_tiers >= 4 {
                    let ra3 = &self.role_a_scales[3];
                    let ((v3_nl_l, v3_lpf_l, v3_hpf_l), (v3_nl_r, v3_lpf_r, v3_hpf_r)) =
                        Self::step_branch_shaper_stereo(
                            &mut self.branch_state_l[2],
                            &mut self.branch_state_r[2],
                            &self.branch_poles[2],
                            &self.base_branch_params[2],
                            v2b_nl_l + 0.5 * mid2_re_l,
                            v2b_nl_r + 0.5 * mid2_re_r,
                        );
                    let (wf3_l, _) = c2_wavefolder(v3_nl_l);
                    let (wf3_r, _) = c2_wavefolder(v3_nl_r);
                    fb[224..246].copy_from_slice(&[
                        v3_nl_l * ra3[0],
                        v3_lpf_l * ra3[1],
                        v3_hpf_l * ra3[2],
                        wf3_l * ra3[3],
                        v3_nl_l * h_l * ra3[4],
                        v2b_nl_l * brc2_l * ra3[5],
                        v3_nl_r * ra3[6],
                        v3_lpf_r * ra3[7],
                        v3_hpf_r * ra3[8],
                        wf3_r * ra3[9],
                        v3_nl_r * h_r * ra3[10],
                        v2b_nl_r * brc2_r * ra3[11],
                        h_l * hpf2_l * ra3[12],
                        h_l * mid2_re_l * ra3[13],
                        h_r * hpf2_r * ra3[14],
                        h_r * mid2_re_r * ra3[15],
                        h_l * sh2_l,
                        wf2_l * mid1_im_l,
                        v3_hpf_l * next_sag,
                        h_r * sh2_r,
                        wf2_r * mid1_im_r,
                        v3_hpf_r * next_sag,
                    ]);
                }
            }
        }

        (tone_l[0], tone_r[0])
    }

    /// Steps Stage II `L-N-L-N` cascade and feedforward tier branches (`Lite..Ultra`),
    /// keeping primary `(s2_l, s2_r)` and `Lite` state strictly `Arena 0`-only (`REQ-5.6`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_frame_with_branches(
        &mut self,
        in_l: f32,
        in_r: f32,
        drive_gain: f32,
        sag_depth: f32,
        beta_tone: f32,
        pwr_drive: f32,
        branch_tiers: usize,
    ) -> Stage2CascadeFrameOut {
        let mut fb = [0.0f32; 256];
        let (s2_l, s2_r) = self.step_frame_into_feature_bus(
            in_l,
            in_r,
            drive_gain,
            sag_depth,
            beta_tone,
            pwr_drive,
            branch_tiers,
            &mut fb,
        );
        let mut basis_l = [0.0f32; 8];
        let mut basis_r = [0.0f32; 8];
        let mut lite_tail = [0.0f32; 16];
        let mut full_features = [0.0f32; 32];
        let mut pro_features = [0.0f32; 32];
        let mut ultra_features = [0.0f32; 22];
        basis_l.copy_from_slice(&fb[20..28]);
        basis_r.copy_from_slice(&fb[28..36]);
        lite_tail.copy_from_slice(&fb[36..52]);
        full_features.copy_from_slice(&fb[84..116]);
        pro_features.copy_from_slice(&fb[152..184]);
        ultra_features.copy_from_slice(&fb[224..246]);
        Stage2CascadeFrameOut {
            s2_l,
            s2_r,
            basis_l,
            basis_r,
            lite_tail,
            full_features,
            pro_features,
            ultra_features,
        }
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
        let (tone_l, v1a_out_l, tone_r, v1a_out_r, _, _) =
            self.step_frame_lite_core(in_l, in_r, drive_gain, sag_depth, 0.5, 1.0);
        (
            [
                tone_l[0],
                tone_l[1],
                tone_l[2],
                tone_l[3],
                v1a_out_l[0],
                v1a_out_l[1],
                v1a_out_l[2],
                v1a_out_l[3],
            ],
            [
                tone_r[0],
                tone_r[1],
                tone_r[2],
                tone_r[3],
                v1a_out_r[0],
                v1a_out_r[1],
                v1a_out_r[2],
                v1a_out_r[3],
            ],
        )
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

        // Verify 48-float Arena 0 [4,096..4,288] loading including 0.999995 thermal pole and Role A
        let mut arena = [0u8; 4_288];
        arena[4_096..4_100].copy_from_slice(&0.999995_f32.to_le_bytes());
        arena[4_172..4_176].copy_from_slice(&0.999995_f32.to_le_bytes());
        arena[1_024..1_028].copy_from_slice(&0.25_f32.to_le_bytes());
        cascade.load_from_arena(&arena);
        assert!((cascade.tone_basis_l.base_alpha_80 - 0.999995).abs() < 1e-7);
        assert!((cascade.base_l1_poles[2] - 0.999995).abs() < 1e-7);
        assert!((cascade.role_a_scales[0][0] - 1.25).abs() < 1e-6);

        let out_full = cascade.step_frame_with_branches(0.35, -0.25, 1.2, 0.3, 0.6, 1.1, 4);
        assert!(out_full.full_features.iter().any(|&v| v.abs() > 1e-6));
        assert!(out_full.pro_features.iter().any(|&v| v.abs() > 1e-6));
        assert!(out_full.ultra_features.iter().any(|&v| v.abs() > 1e-6));

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
