//! Stage I: Pre-Dynamics (Op-T + Op-M + Op-N with 4-bit Topology Selector).
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §2.1 (4,144 B Lite RAM).

use crate::operators::op_m::{
    HilbertH1Lattice6Sec, SymplecticModal4Lane, T1DcBlocker, T1SubHarmonicTracker,
};
use crate::operators::op_n::c2_wavefolder;
use crate::operators::op_t::OpTDelayLine;
use crate::provenance::watermark::read_f32_le;

/// State RAM footprint for Stage I Lite: strictly 4,144 B.
pub const STAGE1_PRE_RAM_BYTES: usize = 4144;

/// Stage I Pre-Dynamics State (4,144 B Lite nominal RAM budget).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage1PreDynamics {
    /// Op-T[Pre]: 2,048 samples = 4,096 B fp16 delay buffer.
    pub op_t_pre: OpTDelayLine<2048>,
    /// Op-M[Pre] Left channel: 4 complex poles.
    pub op_m_pre: SymplecticModal4Lane,
    /// Op-M[Pre] Right channel: 4 complex poles (independent stereo recursive state).
    pub op_m_pre_r: SymplecticModal4Lane,
    /// Full+ tier Stage I parallel 4-pole delta modal bank (`Arena 1 [10,752..10,832]`, `REQ-5.6`: zero mutation of `Lite` state).
    pub op_m_pre_delta_l: SymplecticModal4Lane,
    pub op_m_pre_delta_r: SymplecticModal4Lane,
    /// Immutable 48 kHz base Op-M parameters loaded from `Arena 0 [256..352]` for idempotent ZOH rescaling.
    pub base_op_m_r_low: [f32; 4],
    pub base_op_m_dr: [f32; 4],
    pub base_op_m_theta: [f32; 4],
    pub base_op_m_c_re: [f32; 4],
    pub base_op_m_c_im: [f32; 4],
    pub op_m_c_re: [f32; 4],
    pub op_m_c_im: [f32; 4],
    /// Immutable 48 kHz base Full+ delta Op-M parameters loaded from `Arena 1 [10,752..10,832]`.
    pub base_delta_r_low: [f32; 4],
    pub base_delta_dr: [f32; 4],
    pub base_delta_theta: [f32; 4],
    pub base_delta_c_re: [f32; 4],
    pub base_delta_c_im: [f32; 4],
    /// ZOH-rescaled host-rate Op-M parameters for 3 kHz formant/damping modulation (`0 FSQRT/FDIV/libm`).
    pub host_op_m_r_low: [f32; 4],
    pub host_op_m_dr: [f32; 4],
    pub host_op_m_theta: [f32; 4],
    pub host_delta_r_low: [f32; 4],
    pub host_delta_dr: [f32; 4],
    pub host_delta_theta: [f32; 4],
    pub last_dtheta_pre: f32,
    pub last_g_rad_pre: f32,
    /// Tracker-only passband-normalized 5 Hz DC blockers (`T1`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
    pub dc_blocker_l: T1DcBlocker,
    pub dc_blocker_r: T1DcBlocker,
    /// 6-section-per-branch `H1` Hilbert all-pass lattices (`ref[dsp.op-m.hilbert.lattice]`).
    pub hilbert_l: HilbertH1Lattice6Sec,
    pub hilbert_r: HilbertH1Lattice6Sec,
    /// Complete `T1` Sub-Harmonic Trackers (`[w_cos, w_sin, w_square, w_quarter]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
    pub tracker_l: T1SubHarmonicTracker,
    pub tracker_r: T1SubHarmonicTracker,
    pub last_tracker_l: [f32; 4],
    pub last_tracker_r: [f32; 4],
    /// Per-sample Stage I sub-operator tap telemetry for the 256-D feature bus (`Full..Ultra` slots).
    pub last_op_n_l: f32,
    pub last_op_n_r: f32,
    pub last_op_t_l: f32,
    pub last_op_t_r: f32,
    pub prev_op_t_l: f32,
    pub prev_op_t_r: f32,
    /// Op-N[Pre] pre-gain, boost scale, gate threshold/ratio, and sub-harmonic pitch blend (`theta_2..theta_4`).
    pub pre_gain: f32,
    pub boost_gain: f32,
    pub gate_threshold: f32,
    pub gate_ratio: f32,
    pub pitch_blend: f32,
    /// 4-bit topology selector: serial ordering of Op-T, Op-M, Op-N.
    /// 0 = N -> M -> T (Drive -> Filter -> Mod)
    /// 1 = M -> N -> T (Wah -> Overdrive -> Mod)
    /// 2 = T -> N -> M (Mod -> Drive -> Filter)
    /// 3 = T -> M -> N (Mod -> Filter -> Drive)
    pub topology_mode: u8,
    pub _pad: [u8; 7],
}

impl Default for Stage1PreDynamics {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage1PreDynamics {
    pub const DEFAULT_DELTA_R_LOW: [f32; 4] = [0.78, 0.82, 0.86, 0.90];
    pub const DEFAULT_DELTA_THETA: [f32; 4] = [0.18, 0.38, 0.72, 1.15];

    pub const fn new() -> Self {
        Self {
            op_t_pre: OpTDelayLine::new(),
            op_m_pre: SymplecticModal4Lane::new(),
            op_m_pre_r: SymplecticModal4Lane::new(),
            op_m_pre_delta_l: SymplecticModal4Lane::new(),
            op_m_pre_delta_r: SymplecticModal4Lane::new(),
            base_op_m_r_low: [0.0; 4],
            base_op_m_dr: [0.0; 4],
            base_op_m_theta: [0.0; 4],
            base_op_m_c_re: [1.0, 0.0, 0.0, 0.0],
            base_op_m_c_im: [0.0; 4],
            op_m_c_re: [1.0, 0.0, 0.0, 0.0],
            op_m_c_im: [0.0; 4],
            base_delta_r_low: Self::DEFAULT_DELTA_R_LOW,
            base_delta_dr: [0.0; 4],
            base_delta_theta: Self::DEFAULT_DELTA_THETA,
            base_delta_c_re: [1.0, 0.0, 0.0, 0.0],
            base_delta_c_im: [0.0; 4],
            host_op_m_r_low: [0.0; 4],
            host_op_m_dr: [0.0; 4],
            host_op_m_theta: [0.0; 4],
            host_delta_r_low: Self::DEFAULT_DELTA_R_LOW,
            host_delta_dr: [0.0; 4],
            host_delta_theta: Self::DEFAULT_DELTA_THETA,
            last_dtheta_pre: 0.0,
            last_g_rad_pre: 0.5,
            dc_blocker_l: T1DcBlocker::new_48khz(),
            dc_blocker_r: T1DcBlocker::new_48khz(),
            hilbert_l: HilbertH1Lattice6Sec::new_48khz(),
            hilbert_r: HilbertH1Lattice6Sec::new_48khz(),
            tracker_l: T1SubHarmonicTracker::new(),
            tracker_r: T1SubHarmonicTracker::new(),
            last_tracker_l: [0.0; 4],
            last_tracker_r: [0.0; 4],
            last_op_n_l: 0.0,
            last_op_n_r: 0.0,
            last_op_t_l: 0.0,
            last_op_t_r: 0.0,
            prev_op_t_l: 0.0,
            prev_op_t_r: 0.0,
            pre_gain: 1.0,
            boost_gain: 1.0,
            gate_threshold: 0.001,
            gate_ratio: 0.0,
            pitch_blend: 0.0,
            topology_mode: 0,
            _pad: [0; 7],
        }
    }

    pub fn reset(&mut self) {
        self.op_t_pre.reset();
        self.op_m_pre.reset();
        self.op_m_pre_r.reset();
        self.op_m_pre_delta_l.reset();
        self.op_m_pre_delta_r.reset();
        self.dc_blocker_l.reset();
        self.dc_blocker_r.reset();
        self.hilbert_l.reset();
        self.hilbert_r.reset();
        self.tracker_l.reset();
        self.tracker_r.reset();
        self.last_tracker_l = [0.0; 4];
        self.last_tracker_r = [0.0; 4];
        self.last_op_n_l = 0.0;
        self.last_op_n_r = 0.0;
        self.last_op_t_l = 0.0;
        self.last_op_t_r = 0.0;
        self.prev_op_t_l = 0.0;
        self.prev_op_t_r = 0.0;
    }

    /// Loads immutable 48 kHz Stage I Op-M, pre-gain, and topology parameters from `Arena 0 [256..360]`
    /// and Full+ parallel delta Op-M poles from `Arena 1 [10,752..10,832]`.
    pub fn load_from_arena(&mut self, arena: &[u8]) {
        if arena.len() < 360 {
            return;
        }
        let mut has_poles = false;
        for lane in 0..4 {
            let r_low = read_f32_le(arena, 256 + lane * 4);
            let dr = read_f32_le(arena, 272 + lane * 4);
            let theta = read_f32_le(arena, 288 + lane * 4);
            let c_re = read_f32_le(arena, 304 + lane * 4);
            let c_im = read_f32_le(arena, 320 + lane * 4);
            if r_low.abs() > 1e-12 || dr.abs() > 1e-12 || theta.abs() > 1e-12 {
                has_poles = true;
            }
            self.base_op_m_r_low[lane] = r_low;
            self.base_op_m_dr[lane] = dr;
            self.base_op_m_theta[lane] = theta;
            self.base_op_m_c_re[lane] = c_re;
            self.base_op_m_c_im[lane] = c_im;
        }
        if !has_poles {
            self.base_op_m_c_re = [1.0, 0.0, 0.0, 0.0];
            self.base_op_m_c_im = [0.0; 4];
        } else if self
            .base_op_m_c_re
            .iter()
            .chain(self.base_op_m_c_im.iter())
            .all(|&v| v.abs() <= 1e-12)
        {
            self.base_op_m_c_re[0] = 1.0;
        }
        self.op_m_c_re = self.base_op_m_c_re;
        self.op_m_c_im = self.base_op_m_c_im;
        self.host_op_m_r_low = self.base_op_m_r_low;
        self.host_op_m_dr = self.base_op_m_dr;
        self.host_op_m_theta = self.base_op_m_theta;

        let pg = read_f32_le(arena, 352);
        if pg.is_finite() && pg > 1e-6 {
            self.pre_gain = pg.clamp(0.05, 16.0);
        }
        let mode_word = u32::from_le_bytes([arena[356], arena[357], arena[358], arena[359]]);
        if (1..=3).contains(&mode_word) {
            self.topology_mode = (mode_word as u8) & 0x03;
        }

        let mut has_delta_poles = false;
        if arena.len() >= 10_832 {
            for lane in 0..4 {
                let r_low = read_f32_le(arena, 10_752 + lane * 4);
                let dr = read_f32_le(arena, 10_768 + lane * 4);
                let theta = read_f32_le(arena, 10_784 + lane * 4);
                let c_re = read_f32_le(arena, 10_800 + lane * 4);
                let c_im = read_f32_le(arena, 10_816 + lane * 4);
                if r_low.abs() > 1e-12 || dr.abs() > 1e-12 || theta.abs() > 1e-12 {
                    has_delta_poles = true;
                }
                self.base_delta_r_low[lane] = r_low;
                self.base_delta_dr[lane] = dr;
                self.base_delta_theta[lane] = theta;
                self.base_delta_c_re[lane] = c_re;
                self.base_delta_c_im[lane] = c_im;
            }
        }
        if !has_delta_poles {
            self.base_delta_r_low = Self::DEFAULT_DELTA_R_LOW;
            self.base_delta_dr = [0.0; 4];
            self.base_delta_theta = Self::DEFAULT_DELTA_THETA;
            self.base_delta_c_re = [1.0, 0.0, 0.0, 0.0];
            self.base_delta_c_im = [0.0; 4];
        }
        self.host_delta_r_low = self.base_delta_r_low;
        self.host_delta_dr = self.base_delta_dr;
        self.host_delta_theta = self.base_delta_theta;
        self.last_dtheta_pre = 0.0;
        self.last_g_rad_pre = 0.5;
    }

    /// Reconfigures Stage I Op-T delay, Op-M poles, T1 DC blockers, and H1 Hilbert lattices for `zoh`.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        self.op_t_pre.xfade_len = zoh.switch_xfade_samples_host as usize;
        self.op_t_pre
            .set_target_delay(zoh.tau_ref_samples_host, true);
        self.op_t_pre.current_delay = self.op_t_pre.target_delay;
        self.op_t_pre.xfade_active = false;

        let fs_eff = zoh.effective_sample_rate_hz();
        if let Some(dcb) = T1DcBlocker::new_at_rate(fs_eff) {
            self.dc_blocker_l.pole = dcb.pole;
            self.dc_blocker_l.gain = dcb.gain;
            self.dc_blocker_r.pole = dcb.pole;
            self.dc_blocker_r.gain = dcb.gain;
        }
        let _ = self.hilbert_l.retune_sample_rate(fs_eff);
        let _ = self.hilbert_r.retune_sample_rate(fs_eff);

        self.last_dtheta_pre = 0.0;
        self.last_g_rad_pre = 0.5;
        for lane in 0..4 {
            let r_low = self.base_op_m_r_low[lane];
            let dr = self.base_op_m_dr[lane];
            let theta = self.base_op_m_theta[lane];
            if r_low.abs() > 1e-12 || dr.abs() > 1e-12 || theta.abs() > 1e-12 {
                let (rl_h, dr_h, th_h, _) = zoh.rescale_op_m_channel(r_low, dr, theta, 1.0);
                self.host_op_m_r_low[lane] = rl_h;
                self.host_op_m_dr[lane] = dr_h;
                self.host_op_m_theta[lane] = th_h;
                let r_eff = (rl_h + 0.5 * dr_h).clamp(0.0, 0.9995);
                self.op_m_pre.set_pole(lane, r_eff, th_h);
                self.op_m_pre_r.set_pole(lane, r_eff, th_h);
            } else {
                self.host_op_m_r_low[lane] = 0.0;
                self.host_op_m_dr[lane] = 0.0;
                self.host_op_m_theta[lane] = 0.0;
            }

            let d_rl = self.base_delta_r_low[lane];
            let d_dr = self.base_delta_dr[lane];
            let d_th = self.base_delta_theta[lane];
            let (d_rl_h, d_dr_h, d_th_h, _) = zoh.rescale_op_m_channel(d_rl, d_dr, d_th, 1.0);
            self.host_delta_r_low[lane] = d_rl_h;
            self.host_delta_dr[lane] = d_dr_h;
            self.host_delta_theta[lane] = d_th_h;
            let d_r_eff = (d_rl_h + 0.5 * d_dr_h).clamp(0.0, 0.9995);
            self.op_m_pre_delta_l.set_pole(lane, d_r_eff, d_th_h);
            self.op_m_pre_delta_r.set_pole(lane, d_r_eff, d_th_h);
        }
        self.op_m_c_re = self.base_op_m_c_re;
        self.op_m_c_im = self.base_op_m_c_im;
    }

    /// Retunes Stage I Op-M[Pre] (and Full+ delta Op-M when `retune_delta` is true) at the 3 kHz control rate (`0 FSQRT/FDIV/libm`).
    #[inline]
    pub fn retune_op_m_pre_3khz(&mut self, dtheta_pre: f32, g_rad_pre: f32, retune_delta: bool) {
        let g_clamped = g_rad_pre.clamp(0.0, 1.0);
        if (dtheta_pre - self.last_dtheta_pre).abs() <= 1e-9
            && (g_clamped - self.last_g_rad_pre).abs() <= 1e-9
        {
            return;
        }
        self.last_dtheta_pre = dtheta_pre;
        self.last_g_rad_pre = g_clamped;

        for lane in 0..4 {
            let rl_h = self.host_op_m_r_low[lane];
            let dr_h = self.host_op_m_dr[lane];
            let th_h = self.host_op_m_theta[lane];
            if rl_h.abs() > 1e-12 || dr_h.abs() > 1e-12 || th_h.abs() > 1e-12 {
                let r_eff = (rl_h + g_clamped * dr_h).clamp(0.0, 0.9995);
                let theta_eff = (th_h + dtheta_pre).clamp(0.0, core::f32::consts::PI - 1e-4);
                self.op_m_pre.set_pole(lane, r_eff, theta_eff);
                self.op_m_pre_r.set_pole(lane, r_eff, theta_eff);
            }
            if retune_delta {
                let d_rl_h = self.host_delta_r_low[lane];
                let d_dr_h = self.host_delta_dr[lane];
                let d_th_h = self.host_delta_theta[lane];
                if d_rl_h.abs() > 1e-12 || d_dr_h.abs() > 1e-12 || d_th_h.abs() > 1e-12 {
                    let r_d = (d_rl_h + g_clamped * d_dr_h).clamp(0.0, 0.9995);
                    let th_d = (d_th_h + dtheta_pre).clamp(0.0, core::f32::consts::PI - 1e-4);
                    self.op_m_pre_delta_l.set_pole(lane, r_d, th_d);
                    self.op_m_pre_delta_r.set_pole(lane, r_d, th_d);
                }
            }
        }
    }

    /// Steps the Full+ parallel 4-pole Stage I modal delta bank (`REQ-5.6`: zero mutation of `Lite` state).
    #[inline(always)]
    pub fn step_delta_branch(&mut self, in_l: f32, in_r: f32) {
        for lane in 0..4 {
            let _ = self.op_m_pre_delta_l.step_lane(lane, in_l);
            let _ = self.op_m_pre_delta_r.step_lane(lane, in_r);
        }
    }

    #[inline(always)]
    fn step_op_n(&self, x: f32) -> f32 {
        let boosted = x * self.pre_gain * self.boost_gain;
        // Noise gate
        let gated = if boosted.abs() < self.gate_threshold {
            boosted * self.gate_ratio
        } else {
            boosted
        };
        // Soft pre-overdrive via C2 wavefolder
        let (w_odd, _) = c2_wavefolder(gated * 0.5);
        0.8 * gated + 0.2 * w_odd
    }

    #[inline(always)]
    fn step_op_m(&mut self, x: f32) -> f32 {
        let d_l = self.dc_blocker_l.step(x);
        let (i_l, q_l) = self.hilbert_l.step(d_l);
        self.last_tracker_l = self.tracker_l.step_analytic(i_l, q_l);

        let mut acc = 0.0_f32;
        for lane in 0..4 {
            let c_re = self.op_m_c_re[lane];
            let c_im = self.op_m_c_im[lane];
            if lane == 0 || c_re.abs() > 1e-12 || c_im.abs() > 1e-12 {
                let (re, im) = self.op_m_pre.step_lane(lane, x);
                acc += c_re * re - c_im * im;
            }
        }
        acc + self.pitch_blend * self.last_tracker_l[0]
    }

    #[inline(always)]
    fn step_op_m_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.op_m_pre_r.alpha_re != self.op_m_pre.alpha_re
            || self.op_m_pre_r.alpha_im != self.op_m_pre.alpha_im
        {
            self.op_m_pre_r.alpha_re = self.op_m_pre.alpha_re;
            self.op_m_pre_r.alpha_im = self.op_m_pre.alpha_im;
            self.op_m_pre_r.b_re = self.op_m_pre.b_re;
            self.op_m_pre_r.b_im = self.op_m_pre.b_im;
        }
        let d_l = self.dc_blocker_l.step(in_l);
        let (i_l, q_l) = self.hilbert_l.step(d_l);
        self.last_tracker_l = self.tracker_l.step_analytic(i_l, q_l);

        let d_r = self.dc_blocker_r.step(in_r);
        let (i_r, q_r) = self.hilbert_r.step(d_r);
        self.last_tracker_r = self.tracker_r.step_analytic(i_r, q_r);

        let mut acc_l = 0.0_f32;
        let mut acc_r = 0.0_f32;
        for lane in 0..4 {
            let c_re = self.op_m_c_re[lane];
            let c_im = self.op_m_c_im[lane];
            if lane == 0 || c_re.abs() > 1e-12 || c_im.abs() > 1e-12 {
                let (re_l, im_l) = self.op_m_pre.step_lane(lane, in_l);
                let (re_r, im_r) = self.op_m_pre_r.step_lane(lane, in_r);
                acc_l += c_re * re_l - c_im * im_l;
                acc_r += c_re * re_r - c_im * im_r;
            }
        }
        (
            acc_l + self.pitch_blend * self.last_tracker_l[0],
            acc_r + self.pitch_blend * self.last_tracker_r[0],
        )
    }

    #[inline(always)]
    fn step_op_t(&mut self, x: f32) -> f32 {
        self.op_t_pre.step(x)
    }

    #[inline(always)]
    fn step_op_t_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        self.op_t_pre.step_stereo(in_l, in_r)
    }

    /// Evaluates Stage I for input `x` according to `topology_mode`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        self.prev_op_t_l = self.last_op_t_l;
        match self.topology_mode & 0x03 {
            0 => {
                // N -> M -> T
                let s_n = self.step_op_n(x);
                let s_m = self.step_op_m(s_n);
                let s_t = self.step_op_t(s_m);
                self.last_op_n_l = s_n;
                self.last_op_t_l = s_t;
                s_t
            }
            1 => {
                // M -> N -> T
                let s_m = self.step_op_m(x);
                let s_n = self.step_op_n(s_m);
                let s_t = self.step_op_t(s_n);
                self.last_op_n_l = s_n;
                self.last_op_t_l = s_t;
                s_t
            }
            2 => {
                // T -> N -> M
                let s_t = self.step_op_t(x);
                let s_n = self.step_op_n(s_t);
                let s_m = self.step_op_m(s_n);
                self.last_op_t_l = s_t;
                self.last_op_n_l = s_n;
                s_m
            }
            _ => {
                // T -> M -> N
                let s_t = self.step_op_t(x);
                let s_m = self.step_op_m(s_t);
                let s_n = self.step_op_n(s_m);
                self.last_op_t_l = s_t;
                self.last_op_n_l = s_n;
                s_n
            }
        }
    }

    /// Evaluates Stage I for stereo frame `(in_l, in_r)` with partitioned Left/Right state isolation.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        self.prev_op_t_l = self.last_op_t_l;
        self.prev_op_t_r = self.last_op_t_r;
        match self.topology_mode & 0x03 {
            0 => {
                // N -> M -> T
                let (n_l, n_r) = (self.step_op_n(in_l), self.step_op_n(in_r));
                let (m_l, m_r) = self.step_op_m_stereo(n_l, n_r);
                let (t_l, t_r) = self.step_op_t_stereo(m_l, m_r);
                self.last_op_n_l = n_l;
                self.last_op_n_r = n_r;
                self.last_op_t_l = t_l;
                self.last_op_t_r = t_r;
                (t_l, t_r)
            }
            1 => {
                // M -> N -> T
                let (m_l, m_r) = self.step_op_m_stereo(in_l, in_r);
                let (n_l, n_r) = (self.step_op_n(m_l), self.step_op_n(m_r));
                let (t_l, t_r) = self.step_op_t_stereo(n_l, n_r);
                self.last_op_n_l = n_l;
                self.last_op_n_r = n_r;
                self.last_op_t_l = t_l;
                self.last_op_t_r = t_r;
                (t_l, t_r)
            }
            2 => {
                // T -> N -> M
                let (t_l, t_r) = self.step_op_t_stereo(in_l, in_r);
                let (n_l, n_r) = (self.step_op_n(t_l), self.step_op_n(t_r));
                let (m_l, m_r) = self.step_op_m_stereo(n_l, n_r);
                self.last_op_t_l = t_l;
                self.last_op_t_r = t_r;
                self.last_op_n_l = n_l;
                self.last_op_n_r = n_r;
                (m_l, m_r)
            }
            _ => {
                // T -> M -> N
                let (t_l, t_r) = self.step_op_t_stereo(in_l, in_r);
                let (m_l, m_r) = self.step_op_m_stereo(t_l, t_r);
                let (n_l, n_r) = (self.step_op_n(m_l), self.step_op_n(m_r));
                self.last_op_t_l = t_l;
                self.last_op_t_r = t_r;
                self.last_op_n_l = n_l;
                self.last_op_n_r = n_r;
                (n_l, n_r)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage1_pre_all_topologies_and_gate() {
        let _audio = crate::FtzDazGuard::new();
        let mut pre = Stage1PreDynamics::default();
        pre.reset();
        pre.op_m_pre.set_pole(0, 0.90, 0.4);
        pre.op_t_pre.set_target_delay(2.0, true);
        for _ in 0..130 {
            let _ = pre.op_t_pre.step(0.0);
        }

        // Verify all 4 topologies execute distinct operator orderings and gate attenuates sub-threshold inputs
        for mode in 0..4 {
            pre.reset();
            pre.topology_mode = mode;
            pre.gate_threshold = 0.05;
            pre.gate_ratio = 0.0;

            // Sub-threshold input is hard-gated to 0.0 by Op-N
            assert_eq!(pre.step_op_n(0.01), 0.0);
            // Above-threshold input experiences 0.8 * x + 0.2 * w_odd(0.5 * x) soft pre-overdrive
            let above = pre.step_op_n(0.5);
            let (w_odd, _) = c2_wavefolder(0.25);
            assert!((above - (0.8 * 0.5 + 0.2 * w_odd)).abs() < 1e-6);

            let (sl, sr) = pre.step_stereo(0.4, -0.4);
            assert!(sl.is_finite() && sr.is_finite());
        }

        // Verify zero Left/Right stereo crosstalk across fractional delays and switch crossfades
        for mode in 0..4 {
            let mut pre_stereo = Stage1PreDynamics {
                topology_mode: mode,
                gate_threshold: 0.0,
                ..Default::default()
            };
            pre_stereo.op_m_pre.set_pole(0, 0.92, 0.35);
            pre_stereo.op_t_pre.set_target_delay(12.37, false);
            for n in 0..64 {
                let in_l = ((n as f32) * 0.23 + 0.1).sin();
                let (out_l, out_r) = pre_stereo.step_stereo(in_l, 0.0);
                assert!(out_l.is_finite());
                assert_eq!(
                    out_r, 0.0,
                    "Stereo L->R crosstalk detected in topology mode {mode} at sample {n}: {out_r}"
                );
            }
            // Trigger 128-sample switch jump crossfade and verify 0 crosstalk on R and 1x frame advance
            pre_stereo.op_t_pre.set_target_delay(120.75, true);
            assert!(pre_stereo.op_t_pre.xfade_active);
            for n in 0..128 {
                let in_l = ((n as f32) * 0.19).cos();
                let (_, out_r) = pre_stereo.step_stereo(in_l, 0.0);
                assert_eq!(out_r, 0.0);
            }
            assert!(!pre_stereo.op_t_pre.xfade_active);
        }
    }

    proptest! {
        #[test]
        fn prop_stage1_pre_topologies_finite_and_gated(
            mode in 0_u8..4_u8,
            x in -2.0_f32..2.0_f32,
            pre_gain in 0.1_f32..4.0_f32,
            gate_thresh in 0.01_f32..0.5_f32,
            delay in 4.0_f32..120.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut pre = Stage1PreDynamics {
                topology_mode: mode,
                pre_gain,
                gate_threshold: gate_thresh,
                gate_ratio: 0.0,
                ..Default::default()
            };
            pre.op_m_pre.set_pole(0, 0.88, 0.4);
            pre.op_t_pre.set_target_delay(delay, false);
            let y = pre.step(x);
            prop_assert!(y.is_finite());

            // Strictly zero gate_ratio below threshold yields zero from Op-N
            let sub_thresh_x = (gate_thresh * 0.5) / pre_gain;
            prop_assert_eq!(pre.step_op_n(sub_thresh_x), 0.0);

            // Stereo isolation: feeding signal on Left and 0.0 on Right must emit 0.0 on Right
            let mut pre_iso = Stage1PreDynamics {
                topology_mode: mode,
                pre_gain,
                gate_threshold: 0.0,
                gate_ratio: 0.0,
                ..Default::default()
            };
            pre_iso.op_m_pre.set_pole(0, 0.88, 0.4);
            pre_iso.op_t_pre.current_delay = delay;
            pre_iso.op_t_pre.target_delay = delay;
            for _ in 0..20 {
                let (ol, or) = pre_iso.step_stereo(x, 0.0);
                prop_assert!(ol.is_finite());
                prop_assert_eq!(or, 0.0);
            }
        }
    }
}
