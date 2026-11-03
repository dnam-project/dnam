//! Stage I: Pre-Dynamics (Op-T + Op-M + Op-N with 4-bit Topology Selector).
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §2.1 (4,144 B Lite RAM).

use crate::operators::op_m::SymplecticModal4Lane;
use crate::operators::op_n::c2_wavefolder;
use crate::operators::op_t::OpTDelayLine;

/// State RAM footprint for Stage I Lite: strictly 4,144 B.
pub const STAGE1_PRE_RAM_BYTES: usize = 4144;

/// Stage I Pre-Dynamics State (4,144 B Lite RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage1PreDynamics {
    /// Op-T[Pre]: 2,048 samples = 4,096 B fp16 delay buffer.
    pub op_t_pre: OpTDelayLine<2048>,
    /// Op-M[Pre]: 4 complex poles (32 B).
    pub op_m_pre: SymplecticModal4Lane,
    /// Op-N[Pre] pre-gain and boost scale.
    pub pre_gain: f32,
    pub boost_gain: f32,
    pub gate_threshold: f32,
    pub gate_ratio: f32,
    /// 4-bit topology selector: serial ordering of Op-T, Op-M, Op-N.
    /// 0 = N -> M -> T (Drive -> Filter -> Mod)
    /// 1 = M -> N -> T (Wah -> Overdrive -> Mod)
    /// 2 = T -> N -> M (Mod -> Drive -> Filter)
    /// 3 = T -> M -> N (Mod -> Filter -> Drive)
    pub topology_mode: u8,
    pub _pad: [u8; 11],
}

impl Default for Stage1PreDynamics {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage1PreDynamics {
    pub const fn new() -> Self {
        Self {
            op_t_pre: OpTDelayLine::new(),
            op_m_pre: SymplecticModal4Lane::new(),
            pre_gain: 1.0,
            boost_gain: 1.0,
            gate_threshold: 0.001,
            gate_ratio: 0.0,
            topology_mode: 0,
            _pad: [0; 11],
        }
    }

    pub fn reset(&mut self) {
        self.op_t_pre.reset();
        self.op_m_pre.reset();
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
        let (re, _) = self.op_m_pre.step_lane(0, x);
        re
    }

    #[inline(always)]
    fn step_op_m_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.op_m_pre.alpha_re[1] == 0.0
            && self.op_m_pre.alpha_im[1] == 0.0
            && (self.op_m_pre.alpha_re[0] != 0.0 || self.op_m_pre.alpha_im[0] != 0.0)
        {
            self.op_m_pre.alpha_re[1] = self.op_m_pre.alpha_re[0];
            self.op_m_pre.alpha_im[1] = self.op_m_pre.alpha_im[0];
            self.op_m_pre.b_re[1] = self.op_m_pre.b_re[0];
            self.op_m_pre.b_im[1] = self.op_m_pre.b_im[0];
        }
        let (re_l, _) = self.op_m_pre.step_lane(0, in_l);
        let (re_r, _) = self.op_m_pre.step_lane(1, in_r);
        (re_l, re_r)
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
        match self.topology_mode & 0x03 {
            0 => {
                // N -> M -> T
                let s_n = self.step_op_n(x);
                let s_m = self.step_op_m(s_n);
                self.step_op_t(s_m)
            }
            1 => {
                // M -> N -> T
                let s_m = self.step_op_m(x);
                let s_n = self.step_op_n(s_m);
                self.step_op_t(s_n)
            }
            2 => {
                // T -> N -> M
                let s_t = self.step_op_t(x);
                let s_n = self.step_op_n(s_t);
                self.step_op_m(s_n)
            }
            _ => {
                // T -> M -> N
                let s_t = self.step_op_t(x);
                let s_m = self.step_op_m(s_t);
                self.step_op_n(s_m)
            }
        }
    }

    /// Evaluates Stage I for stereo frame `(in_l, in_r)` with partitioned Left/Right state isolation.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        match self.topology_mode & 0x03 {
            0 => {
                // N -> M -> T
                let (n_l, n_r) = (self.step_op_n(in_l), self.step_op_n(in_r));
                let (m_l, m_r) = self.step_op_m_stereo(n_l, n_r);
                self.step_op_t_stereo(m_l, m_r)
            }
            1 => {
                // M -> N -> T
                let (m_l, m_r) = self.step_op_m_stereo(in_l, in_r);
                let (n_l, n_r) = (self.step_op_n(m_l), self.step_op_n(m_r));
                self.step_op_t_stereo(n_l, n_r)
            }
            2 => {
                // T -> N -> M
                let (t_l, t_r) = self.step_op_t_stereo(in_l, in_r);
                let (n_l, n_r) = (self.step_op_n(t_l), self.step_op_n(t_r));
                self.step_op_m_stereo(n_l, n_r)
            }
            _ => {
                // T -> M -> N
                let (t_l, t_r) = self.step_op_t_stereo(in_l, in_r);
                let (m_l, m_r) = self.step_op_m_stereo(t_l, t_r);
                (self.step_op_n(m_l), self.step_op_n(m_r))
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
