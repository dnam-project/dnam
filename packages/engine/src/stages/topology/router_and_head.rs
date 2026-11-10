//! Stage 0 8-Port Router (WDW, Kaiser-Sinc Sub-Sample Alignment, 200 Hz TPT LR4 Crossover,
//! Stereo-Linked & Aux-Key Detector), Stage IV Blended Quiet-Feedback Knee & Outer-Clamped Loop
//! Saturator, and 256-D Feature Bus Slot Allocation.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` (`ref[dsp.router.wdw-contract]`, `ref[dsp.router.parallel-alignment]`, `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.stereo-link-detector]`, `ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`, `ref[dsp.head.feature-bus-slots]`).

use alloc::vec::Vec;

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4, newton6_reciprocal_f64, sqrt_f64, widen_f32_bits,
};
use crate::operators::op_t::catmull_rom_hermite_cubic;
use crate::schema::cold_exp_f64;
use crate::types::Tier;

/// Wet-Dry-Wet (`WDW`) center dry tap mode (`ref[dsp.router.wdw-contract]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WdwDryTapMode {
    /// Center dry on `Port Out 1` is the Mid contribution `m_c` of Stages I–III core;
    /// `Ports Out 2/3` carry Stage IV spatial contribution `m_w ± alpha * s_w`.
    PreSpatialCore,
    /// Center dry on `Port Out 1` is voltage-normalized Mid `0.5 * (x_L + x_R)` of the selected
    /// direct source; `Ports Out 2/3` carry the wet effect response.
    DirectDry,
}

/// Evaluates the WDW output ports `(port_out1_dry, port_out2_wet_l, port_out3_wet_r)`
/// (`ref[dsp.router.wdw-contract]`).
#[inline]
pub fn eval_wdw_routing_outputs(
    mode: WdwDryTapMode,
    core_mid: f32,
    wet_mid: f32,
    wet_side: f32,
    width_alpha: f32,
    direct_l: f32,
    direct_r: f32,
) -> (f32, f32, f32) {
    let wet_l = wet_mid + width_alpha * wet_side;
    let wet_r = wet_mid - width_alpha * wet_side;
    let dry = match mode {
        WdwDryTapMode::PreSpatialCore => core_mid,
        WdwDryTapMode::DirectDry => 0.5 * (direct_l + direct_r),
    };
    (dry, wet_l, wet_r)
}

/// Parallel-path sub-sample alignment mode (`ref[dsp.router.parallel-alignment]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelAlignmentMode {
    /// 64-tap Kaiser-derived windowed sinc (`beta = 8.6`, `G = 31` host samples).
    FullBand64Tap,
    /// 16-tap Kaiser-derived windowed sinc (`beta = 6.5`, `G = 7` host samples).
    LowLatency16Tap,
    /// Centered 4-tap Catmull-Rom cubic Hermite (`G = 1` host sample).
    MinLatency4Tap,
}

impl ParallelAlignmentMode {
    /// Returns the host-rate centering offset `G` in samples (`31`, `7`, or `1`).
    #[inline]
    pub const fn centering_offset_host_samples(self) -> usize {
        match self {
            Self::FullBand64Tap => 31,
            Self::LowLatency16Tap => 7,
            Self::MinLatency4Tap => 1,
        }
    }

    /// Returns `(base_offset_b, effective_delay_samples, combined_pdc_host_samples)` at oversampling
    /// factor `oversampling` (`1` or `4`, `ref[dsp.router.parallel-alignment]`, `ref[arch.bounce.oversampling]`).
    #[inline]
    pub const fn timing_at_oversampling(
        self,
        oversampling: usize,
        delta_samples_eff: f64,
    ) -> (usize, f64, usize) {
        let g = self.centering_offset_host_samples();
        let b = (oversampling - 1) * g;
        let d_eff = (oversampling * g) as f64 + delta_samples_eff;
        let pdc_host = if oversampling == 4 { g + 12 } else { g };
        (b, d_eff, pdc_host)
    }
}

fn bessel_i0_f64(z: f64) -> f64 {
    let half = 0.5 * z;
    let half2 = half * half;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for m in 1..=28 {
        term *= half2 / ((m * m) as f64);
        sum += term;
    }
    sum
}

#[inline]
fn normalized_sinc_f64(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        1.0
    } else {
        let pix = core::f64::consts::PI * x;
        let k = (x.round()) as i64;
        let rem = x - (k as f64);
        let sign = if (k & 1) != 0 { -1.0 } else { 1.0 };
        let u = core::f64::consts::PI * rem;
        let u2 = u * u;
        let sin_u = u
            * (1.0
                - u2 * (1.0 / 6.0
                    - u2 * (1.0 / 120.0
                        - u2 * (1.0 / 5040.0
                            - u2 * (1.0 / 362_880.0 - u2 * (1.0 / 39_916_800.0))))));
        (sign * sin_u) / pix
    }
}

/// Prepares normalized zero-ended Kaiser-derived sinc weights $h_k(\mu)$ for $K \in \{16, 64\}$
/// (`beta = 8.6` for `K = 64`, `beta = 6.5` for `K = 16`, `0 <= mu < 1`,
/// `ref[dsp.router.parallel-alignment]`).
pub fn prepare_kaiser_sinc_alignment_weights(
    num_taps: usize,
    beta: f64,
    mu: f64,
) -> Option<Vec<f64>> {
    if !matches!(num_taps, 16 | 64) || !(0.0..1.0).contains(&mu) || beta <= 0.0 {
        return None;
    }
    let g = ((num_taps - 2) / 2) as f64;
    let r = (num_taps as f64) * 0.5;
    let denom_i0 = (bessel_i0_f64(beta) - 1.0).max(1.0e-15);
    let mut raw = Vec::with_capacity(num_taps);
    let mut sum = 0.0_f64;
    for k in 0..num_taps {
        let s = (k as f64) - g - mu;
        let ratio = s / r;
        let w = if ratio.abs() >= 1.0 {
            0.0
        } else {
            let arg = beta * sqrt_f64((1.0 - ratio * ratio).max(0.0));
            ((bessel_i0_f64(arg) - 1.0) / denom_i0).max(0.0)
        };
        let h = normalized_sinc_f64(s) * w;
        raw.push(h);
        sum += h;
    }
    for v in &mut raw {
        *v /= sum;
    }
    Some(raw)
}

/// Evaluates `MinLatency4Tap` Catmull-Rom alignment for `[s_{-1}, s_0, s_1, s_2]` (`G = 1`,
/// `ref[dsp.router.parallel-alignment]`).
#[inline]
pub fn eval_min_latency_4tap_alignment(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
}

/// 4th-order Linkwitz–Riley (`LR4`) Clean/Drive Crossover implemented as two cascaded 2nd-order
/// Butterworth trapezoidal state-variable filter (`TPT SVF`) sections (`Q = 1/sqrt(2)`,
/// `ref[dsp.router.lr4-crossover]`).
#[derive(Clone, Copy, Debug)]
pub struct TptLr4Crossover {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k_damp: f32,
    pub lp_s1: [f32; 2],
    pub lp_s2: [f32; 2],
    pub hp_s1: [f32; 2],
    pub hp_s2: [f32; 2],
    pub sec2_ic1: [f32; 4],
    pub sec2_ic2: [f32; 4],
}

impl Default for TptLr4Crossover {
    fn default() -> Self {
        Self::new_at_rate(200.0, 48_000.0)
    }
}

impl TptLr4Crossover {
    /// Const-initialized `200 Hz` at `48 kHz` `TPT SVF` LR4 crossover (`ref[dsp.router.lr4-crossover]`).
    pub const fn new_48khz() -> Self {
        Self {
            a1: 0.981_658_2,
            a2: 0.012_850_65,
            a3: 0.000_168_23,
            k_damp: core::f32::consts::SQRT_2,
            lp_s1: [0.0; 2],
            lp_s2: [0.0; 2],
            hp_s1: [0.0; 2],
            hp_s2: [0.0; 2],
            sec2_ic1: [0.0; 4],
            sec2_ic2: [0.0; 4],
        }
    }

    /// Prepares cold-path `TPT SVF` coefficients in `f64` for cutoff `fc_hz` (default `200 Hz`) at
    /// `effective_rate_hz` (`ref[dsp.router.lr4-crossover]`).
    pub fn new_at_rate(fc_hz: f64, effective_rate_hz: f64) -> Self {
        let angle = (core::f64::consts::PI * fc_hz / effective_rate_hz).clamp(1.0e-5, 1.4);
        let u2 = angle * angle;
        let sin_a = angle
            * (1.0 - u2 * (1.0 / 6.0 - u2 * (1.0 / 120.0 - u2 * (1.0 / 5040.0 - u2 / 362_880.0))));
        let cos_a = 1.0 - u2 * (0.5 - u2 * (1.0 / 24.0 - u2 * (1.0 / 720.0 - u2 / 40_320.0)));
        let g = sin_a / cos_a;
        let k = core::f64::consts::SQRT_2;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        Self {
            a1: a1 as f32,
            a2: a2 as f32,
            a3: a3 as f32,
            k_damp: k as f32,
            lp_s1: [0.0; 2],
            lp_s2: [0.0; 2],
            hp_s1: [0.0; 2],
            hp_s2: [0.0; 2],
            sec2_ic1: [0.0; 4],
            sec2_ic2: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.lp_s1 = [0.0; 2];
        self.lp_s2 = [0.0; 2];
        self.hp_s1 = [0.0; 2];
        self.hp_s2 = [0.0; 2];
        self.sec2_ic1 = [0.0; 4];
        self.sec2_ic2 = [0.0; 4];
    }

    #[inline(always)]
    fn step_svf_section(
        a1: f32,
        a2: f32,
        a3: f32,
        k: f32,
        x: f32,
        s1: &mut f32,
        s2: &mut f32,
    ) -> (f32, f32) {
        let v3 = x - *s2;
        let v1 = a1 * (*s1) + a2 * v3;
        let v2 = *s2 + a2 * (*s1) + a3 * v3;
        let low = v2;
        let high = x - k * v1 - v2;
        *s1 = flush_recursive_state_f32(2.0 * v1 - *s1);
        *s2 = flush_recursive_state_f32(2.0 * v2 - *s2);
        (low, high)
    }

    /// Steps the `LR4` crossover by 1 sample without real-time division, returning
    /// `(low_clean_bypass, high_drive_path)` (`ref[dsp.router.lr4-crossover]`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let (l2_first, h2_first) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            x,
            &mut self.lp_s1[0],
            &mut self.lp_s2[0],
        );
        self.hp_s1[0] = self.lp_s1[0];
        self.hp_s2[0] = self.lp_s2[0];
        let (l4, _) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            l2_first,
            &mut self.lp_s1[1],
            &mut self.lp_s2[1],
        );
        let (_, h4) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            h2_first,
            &mut self.hp_s1[1],
            &mut self.hp_s2[1],
        );
        (l4, h4)
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of `LR4` crossovers simultaneously,
    /// returning `(low_l, high_l, low_r, high_r)` (`ref[dsp.router.lr4-crossover]`).
    #[inline(always)]
    pub fn step_stereo(&mut self, right: &mut Self, x_l: f32, x_r: f32) -> (f32, f32, f32, f32) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let (a1, a2, a3, k) = (self.a1, self.a2, self.a3, self.k_damp);
        let (l2_l, h2_l) =
            Self::step_svf_section(a1, a2, a3, k, x_l, &mut self.lp_s1[0], &mut self.lp_s2[0]);
        let (l2_r, h2_r) =
            Self::step_svf_section(a1, a2, a3, k, x_r, &mut right.lp_s1[0], &mut right.lp_s2[0]);

        let x_arr = [l2_l, h2_l, l2_r, h2_r];
        let x = V::load(&x_arr);
        let ic1 = V::load(&self.sec2_ic1);
        let ic2 = V::load(&self.sec2_ic2);
        let v3 = x.sub(ic2);
        let v1 = ic1.mul(V::splat(a1)).fma(v3, V::splat(a2));
        let v2 = ic2.fma(ic1, V::splat(a2)).fma(v3, V::splat(a3));
        let two = V::splat(2.0);
        flush_recursive_state_vec4(v1.mul(two).sub(ic1)).store(&mut self.sec2_ic1);
        flush_recursive_state_vec4(v2.mul(two).sub(ic2)).store(&mut self.sec2_ic2);

        let mut v1_a = [0.0f32; 4];
        let mut v2_a = [0.0f32; 4];
        v1.store(&mut v1_a);
        v2.store(&mut v2_a);
        (
            v2_a[0],
            h2_l - k * v1_a[1] - v2_a[1],
            v2_a[2],
            h2_r - k * v1_a[3] - v2_a[3],
        )
    }
}

/// Evaluates the programmable stereo-linked & aux-key sidechain detector levels `(e_det_l, e_det_r)`
/// with zero `-6.02 dB` hard-panned peak loss (`ref[dsp.router.stereo-link-detector]`).
#[inline(always)]
pub fn eval_stereo_linked_aux_detector(
    x_l: f32,
    x_r: f32,
    x_aux: f32,
    alpha_link: f32,
    beta_key: f32,
) -> (f32, f32) {
    let a = alpha_link.clamp(0.0, 1.0);
    let b = beta_key.clamp(0.0, 1.0);
    let abs_l = x_l.abs();
    let abs_r = x_r.abs();
    let e_link = abs_l.max(abs_r);
    let e_key = x_aux.abs();
    let one_minus_b = 1.0 - b;
    let one_minus_a = 1.0 - a;
    let det_l = one_minus_b * (one_minus_a * abs_l + a * e_link) + b * e_key;
    let det_r = one_minus_b * (one_minus_a * abs_r + a * e_link) + b * e_key;
    (det_l, det_r)
}

/// Canonical 4-band asymmetric ballistic envelope time constants `(tau_att_sec, tau_rel_sec)`
/// (`Low`, `Mid`, `High`, `Aux Key`, `ref[dsp.router.ballistic-envelope-bank]`).
pub const BALLISTIC_ENVELOPE_BANK_TAUS_SEC: [(f64, f64); 4] = [
    (0.0100, 0.1500),
    (0.0020, 0.0400),
    (0.0002, 0.0150),
    (0.0005, 0.0500),
];

/// Prepares cold-path 1-pole step coefficients `(alpha_att, alpha_rel)` for band `band_idx` (`0..4`)
/// at `rate_hz` (`ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.zoh.router-detector]`).
pub fn prepare_ballistic_band_alphas(band_idx: usize, rate_hz: f64) -> Option<(f32, f32)> {
    let &(tau_att, tau_rel) = BALLISTIC_ENVELOPE_BANK_TAUS_SEC.get(band_idx)?;
    if rate_hz <= 0.0 || !rate_hz.is_finite() {
        return None;
    }
    let a_att = (1.0 - cold_exp_f64(-1.0 / (rate_hz * tau_att))) as f32;
    let a_rel = (1.0 - cold_exp_f64(-1.0 / (rate_hz * tau_rel))) as f32;
    Some((a_att, a_rel))
}

/// Largest `f32` bit pattern $\le 0.985$ (`0x3F7C28F5`, since `0x3F7C28F6 > 0.985`,
/// `ref[dsp.stage4.loop-saturator]`).
pub const STAGE4_MAX_V_CEIL_BITS: u32 = 0x3F7C_28F5;

/// Largest `f32` bit pattern $\le 0.9990$ (`0x3F7FBE76`, since `0x3F7FBE77 > 0.9990`,
/// `ref[dsp.stage4.quiet-feedback-knee]`).
pub const STAGE4_MAX_G_LIN_BITS: u32 = 0x3F7F_BE76;

/// Evaluates the Stage IV unity-slope cubic loop saturator with explicit outer clamp to $\pm V_{\text{ceil}}$
/// (`0 < V_ceil <= f32::from_bits(0x3F7C28F5)`, `ref[dsp.stage4.loop-saturator]`).
#[inline]
pub fn eval_stage4_loop_saturator(u: f32, v_ceil: f32) -> Option<f32> {
    let max_ceil = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
    if !u.is_finite() || !v_ceil.is_finite() || !(v_ceil > 0.0 && v_ceil <= max_ceil) {
        return None;
    }
    let two_over_3v = if v_ceil.to_bits() == STAGE4_MAX_V_CEIL_BITS {
        (2.0_f64 / (3.0_f64 * (f32::from_bits(STAGE4_MAX_V_CEIL_BITS) as f64))) as f32
    } else {
        (2.0_f64 * newton6_reciprocal_f64(3.0_f64 * (v_ceil as f64))) as f32
    };
    let v = (u * two_over_3v).clamp(-1.0, 1.0);
    let prod = v_ceil * v;
    let half_sq = (0.5 * v) * v;
    let raw = prod * (1.5 - half_sq);
    Some(raw.clamp(-v_ceil, v_ceil))
}

/// Fast hot-path Stage IV loop saturator at the canonical `STAGE4_MAX_V_CEIL_BITS` ceiling (`0 FDIV`).
#[inline(always)]
pub fn eval_stage4_loop_saturator_fast(u: f32) -> f32 {
    const V_CEIL: f32 = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
    const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;
    let v = (u * TWO_OVER_3V).clamp(-1.0, 1.0);
    let prod = V_CEIL * v;
    let half_sq = (0.5 * v) * v;
    (prod * (1.5 - half_sq)).clamp(-V_CEIL, V_CEIL)
}

/// Fast hot-path Stage IV blended quiet-feedback knee with cold-prepared `inv_sigma = 1.0 / sigma` (`0 FDIV`).
#[inline(always)]
pub fn eval_stage4_blended_feedback_knee_fast(
    u: f32,
    g_lin: f32,
    inv_sigma: f64,
    alpha: f64,
) -> f32 {
    let u64_val = u as f64;
    let g64 = g_lin as f64;
    let ratio = u64_val.abs() * inv_sigma;
    let denom = 1.0 + ratio * ratio;
    let inv_denom = newton6_reciprocal_f64(denom);
    let factor = (1.0 - alpha) + alpha * inv_denom;
    ((g64 * u64_val) * factor) as f32
}

/// Evaluates the Stage IV blended quiet-feedback knee $K(u) = g_{\text{lin}} u ((1 - \alpha) + \alpha / (1 + (u/\sigma)^2))$
/// using a cold-prepared $1/\sigma$ and 6-step `f64` Newton reciprocal (`0 FDIV` in hot loop,
/// `< 2^-40` pre-rounding relative error, `ref[dsp.stage4.quiet-feedback-knee]`).
#[inline]
pub fn eval_stage4_blended_feedback_knee(
    u: f32,
    g_lin: f32,
    sigma: f32,
    alpha: f32,
) -> Option<f32> {
    let max_gain = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
    if !u.is_finite()
        || !g_lin.is_finite()
        || !sigma.is_finite()
        || !alpha.is_finite()
        || !(0.0..=max_gain).contains(&g_lin)
        || !(0.0..=1.0).contains(&alpha)
        || sigma <= 0.0
    {
        return None;
    }
    let u64_val = widen_f32_bits(u.to_bits())?;
    let g64 = widen_f32_bits(g_lin.to_bits())?;
    let s64 = widen_f32_bits(sigma.to_bits())?;
    let a64 = widen_f32_bits(alpha.to_bits())?;

    let inv_sigma = newton6_reciprocal_f64(s64);
    let ratio = u64_val.abs() * inv_sigma;
    let denom = 1.0 + ratio * ratio;
    let inv_denom = newton6_reciprocal_f64(denom);
    let factor = (1.0 - a64) + a64 * inv_denom;
    let inter = (g64 * u64_val) * factor;
    Some(inter as f32)
}

/// Returns `(modal_slot_range, other_stage_slot_range, fir_lr_slots)` for `tier` inside the
/// 256-D feature bus $\boldsymbol{\Phi}[n]$ (`ref[dsp.head.feature-bus-slots]`).
pub const fn stage3_feature_bus_slot_ranges(
    tier: Tier,
) -> (core::ops::Range<usize>, core::ops::Range<usize>, [usize; 2]) {
    match tier {
        Tier::Lite => (0..4, 4..62, [62, 63]),
        Tier::Full => (64..68, 68..126, [126, 127]),
        Tier::Pro => (128..136, 136..190, [190, 191]),
        Tier::Ultra => (192..208, 208..254, [254, 255]),
    }
}

/// Pair-groups 4 complex modal outputs `[y0, y1, y2, y3]` for L and R into the 4 Stage III feature
/// bus slots `[y0_L + y2_L, y1_L + y3_L, y0_R + y2_R, y1_R + y3_R]` (`ref[dsp.head.feature-bus-slots]`).
#[inline]
pub fn pack_stage3_modal_block_features(y_l: &[f32; 4], y_r: &[f32; 4]) -> [f32; 4] {
    [
        y_l[0] + y_l[2],
        y_l[1] + y_l[3],
        y_r[0] + y_r[2],
        y_r[1] + y_r[3],
    ]
}

/// Evaluates the closed-form per-sample stage modulation laws (`ref[dsp.ctrl.dispatch.stage-laws]`)
/// mapping micro-ramped `theta[0..16]` and base parameters into clamped stage parameters (`0 FDIV`, `0 FSQRT`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn eval_stage_modulation_laws(
    theta: &[f32; 16],
    base_send_gains: [f32; 2],
    base_tau_pre: f32,
    max_tau_pre: f32,
    base_tau_tank: f32,
    max_tau_tank: f32,
    base_g_lin: f32,
    base_a_diff: f32,
    base_width: f32,
) -> [f32; 16] {
    let max_g_lin = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
    [
        (base_send_gains[0] + theta[0]).clamp(0.0, 2.0),
        (base_send_gains[1] + theta[1]).clamp(0.0, 2.0),
        (base_tau_pre + theta[2]).clamp(4.0, max_tau_pre.max(4.0)),
        (24.0 * theta[3]).clamp(-24.0, 24.0),
        (1.0 + theta[4]).clamp(0.0, 16.0),
        (1.0 + 0.25 * theta[5]).clamp(0.25, 4.0),
        (0.5 + theta[6]).clamp(0.0, 1.0),
        (1.0 + 0.25 * theta[7]).clamp(0.25, 4.0),
        (0.2 + 0.1 * theta[8]).clamp(0.0, 0.8),
        theta[9].clamp(-core::f32::consts::FRAC_PI_4, core::f32::consts::FRAC_PI_4),
        (0.5 + theta[10]).clamp(0.0, 1.0),
        (1.0 + theta[11]).clamp(0.0, 2.0),
        (base_tau_tank + theta[12]).clamp(4.0, max_tau_tank.max(4.0)),
        (base_g_lin + theta[13]).clamp(0.0, max_g_lin),
        (base_a_diff + theta[14]).clamp(-0.75, 0.75),
        (base_width + theta[15]).clamp(0.0, 2.0),
    ]
}

/// Validates a 64-byte-aligned `tensor_directory` slice entry (`byte_offset`, `byte_length`) against
/// its owning tier prefix boundary `tier_arena_bytes` (`ref[container.dnam.tensor-directory]`).
#[inline]
pub const fn validate_tensor_directory_slice(
    byte_offset: usize,
    byte_length: usize,
    tier_arena_bytes: usize,
) -> bool {
    if !byte_offset.is_multiple_of(64) || !byte_length.is_multiple_of(64) {
        return false;
    }
    match byte_offset.checked_add(byte_length) {
        Some(end) => end <= tier_arena_bytes,
        None => false,
    }
}

#[cfg(test)]
mod spec_tests {
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container, strip_container_to_tier};
    use crate::math::DefaultMathBackend;
    use crate::schema::DnamJsonHeader;
    use crate::stages::{
        MidSideKroneckerHead, ParallelAlignmentMode, STAGE4_MAX_G_LIN_BITS, STAGE4_MAX_V_CEIL_BITS,
        Stage0Router, Stage1PreDynamics, Stage2CommutedCascade, Stage3CabinetModals,
        Stage4SpatialTank, TankDcBlocker, TptLr4Crossover, WdwDryTapMode,
        eval_min_latency_4tap_alignment, eval_stage4_blended_feedback_knee,
        eval_stage4_loop_saturator, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
        pack_stage3_modal_block_features, prepare_ballistic_band_alphas,
        prepare_kaiser_sinc_alignment_weights, stage3_feature_bus_slot_ranges,
    };
    use crate::{DnamEngine, FtzDazGuard, Tier};
    use alloc::vec;
    use proptest::prelude::*;

    /// verify[dsp.router.signal-flow]
    /// verify[dsp.router.ports]
    /// verify[dsp.router.wdw-contract]
    /// verify[dsp.router.return-to-send-isolation]
    /// verify[dsp.router.parallel-alignment]
    /// verify[dsp.router.lr4-crossover]
    /// verify[dsp.router.mid-side-input]
    /// verify[dsp.router.stereo-link-detector]
    /// verify[dsp.router.ballistic-envelope-bank]
    /// verify[dsp.router.modulation-bus]
    #[test]
    fn test_stage0_router_wdw_isolation_alignment_lr4_ms_and_detector() {
        let _guard = FtzDazGuard::new();

        let (dry_core, wet_l, wet_r) =
            eval_wdw_routing_outputs(WdwDryTapMode::PreSpatialCore, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_core - 0.4).abs() < 1e-6);
        assert!((wet_l - 0.35).abs() < 1e-6);
        assert!((wet_r - 0.05).abs() < 1e-6);

        let (dry_dir, _, _) =
            eval_wdw_routing_outputs(WdwDryTapMode::DirectDry, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_dir - 0.5).abs() < 1e-6);

        let mut router = Stage0Router::new();
        router.set_scattering_matrix([[1.0; 4]; 4]);
        assert_eq!(router.s_matrix[0][2], 0.0);
        assert_eq!(router.s_matrix[0][3], 0.0);
        assert_eq!(router.s_matrix[1][2], 0.0);
        assert_eq!(router.s_matrix[1][3], 0.0);
        let (s_l1, s_r1, _, _, _, _) = router.route_frame(0.5, -0.25, 0.0, 0.0);
        let (s_l2, s_r2, _, _, _, _) = router.route_frame(0.5, -0.25, 10.0, -10.0);
        assert_eq!((s_l1, s_r1), (s_l2, s_r2));

        let (mid, side) = Stage0Router::voltage_normalize_ms(0.75, 0.75);
        assert_eq!((mid, side), (0.75, 0.0));
        assert_eq!(
            Stage0Router::voltage_reconstruct_ms(mid, side),
            (0.75, 0.75)
        );

        assert_eq!(
            ParallelAlignmentMode::FullBand64Tap.centering_offset_host_samples(),
            31
        );
        assert_eq!(
            ParallelAlignmentMode::LowLatency16Tap.centering_offset_host_samples(),
            7
        );
        assert_eq!(
            ParallelAlignmentMode::MinLatency4Tap.centering_offset_host_samples(),
            1
        );
        let (b_4x, d_eff_4x, pdc_4x) =
            ParallelAlignmentMode::FullBand64Tap.timing_at_oversampling(4, 5.5);
        assert_eq!(b_4x, 93);
        assert!((d_eff_4x - 129.5).abs() < 1e-12);
        assert_eq!(pdc_4x, 43);

        for (k, beta) in [(64usize, 8.6_f64), (16, 6.5)] {
            let w = prepare_kaiser_sinc_alignment_weights(k, beta, 0.35).unwrap();
            assert_eq!(w.len(), k);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14);
        }
        assert!((eval_min_latency_4tap_alignment(1.0, 1.0, 1.0, 1.0, 0.4) - 1.0).abs() < 1e-6);

        let mut lr4 = TptLr4Crossover::new_at_rate(200.0, 48_000.0);
        let mut low_dc = 0.0f32;
        let mut high_dc = 0.0f32;
        for _ in 0..4_000 {
            let (l, h) = lr4.step(1.0);
            low_dc = l;
            high_dc = h;
        }
        assert!((low_dc - 1.0).abs() < 1e-3);
        assert!(high_dc.abs() < 1e-3);

        assert_eq!(
            eval_stereo_linked_aux_detector(-0.75, 0.0, 0.0, 1.0, 0.0),
            (0.75, 0.75)
        );
        assert_eq!(
            eval_stereo_linked_aux_detector(0.75, -0.5, 0.25, 1.0, 0.5),
            (0.5, 0.5)
        );
        for b in 0..4 {
            let (a_att, a_rel) = prepare_ballistic_band_alphas(b, 48_000.0).unwrap();
            assert!(a_att > a_rel && a_rel > 0.0 && a_att < 1.0);
        }
    }

    /// verify[dsp.stage1.pre-bus]
    /// verify[dsp.stage2.commuted-cascade]
    /// verify[dsp.stage3.cabinet-bank]
    /// verify[dsp.stage4.spatial-tank]
    /// verify[dsp.stage4.captured-freeze]
    /// verify[dsp.stage4.dc-blocker]
    /// verify[dsp.stage4.loop-saturator]
    /// verify[dsp.stage4.quiet-feedback-knee]
    #[test]
    fn test_stages_1_to_4_loop_saturator_blended_knee_dc_blocker_and_freeze() {
        let _guard = FtzDazGuard::new();

        let mut s1 = Stage1PreDynamics::new();
        let (s1_l, s1_r) = s1.step_stereo(0.25, -0.25);
        assert!(s1_l.is_finite() && s1_r.is_finite());

        let mut s2 = Stage2CommutedCascade::new();
        let (s2_l, s2_r) = s2.step_stereo(s1_l, s1_r);
        assert!(s2_l.is_finite() && s2_r.is_finite());

        let mut s3 = Stage3CabinetModals::new();
        let s3_out = s3.step(s2_l, 0.0, &[0.0625; 16]);
        assert!(s3_out.is_finite());

        let mut s4 = Stage4SpatialTank::new();
        s4.freeze = true;
        let (w_l, w_r) = s4.step_stereo(0.5, -0.5);
        assert!(w_l.is_finite() && w_r.is_finite());

        let v_ceil = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
        assert!((v_ceil as f64) <= 0.985);
        assert!((f32::from_bits(STAGE4_MAX_V_CEIL_BITS + 1) as f64) > 0.985);

        for u in [0.985_f32, 1.5, 10.0, f32::from_bits(0x3f7f_f4d1)] {
            let sat_pos = eval_stage4_loop_saturator(u, v_ceil).unwrap();
            let sat_neg = eval_stage4_loop_saturator(-u, v_ceil).unwrap();
            assert!(sat_pos.to_bits() <= STAGE4_MAX_V_CEIL_BITS);
            assert_eq!(sat_pos, -sat_neg);
        }
        assert!(
            eval_stage4_loop_saturator(1.0, f32::from_bits(STAGE4_MAX_V_CEIL_BITS + 1)).is_none()
        );

        let g_max = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
        assert!((g_max as f64) <= 0.999);
        assert!((f32::from_bits(STAGE4_MAX_G_LIN_BITS + 1) as f64) > 0.999);

        let k_lin = eval_stage4_blended_feedback_knee(2.0, 0.8, 0.5, 0.0).unwrap();
        assert!((k_lin - 1.6).abs() < 1e-6);
        let k_rat = eval_stage4_blended_feedback_knee(2.0, 0.8, 0.5, 1.0).unwrap();
        assert!((k_rat - (1.6 / 17.0)).abs() < 1e-6);
        assert!(eval_stage4_blended_feedback_knee(1.0e9, 0.5, 0.05, 1.0).unwrap() > 0.0);
        assert!(
            eval_stage4_blended_feedback_knee(
                1.0,
                f32::from_bits(STAGE4_MAX_G_LIN_BITS + 1),
                0.5,
                0.5
            )
            .is_none()
        );

        let mut dc = TankDcBlocker::new();
        let mut dc_out = 0.0f32;
        for _ in 0..8_000 {
            dc_out = dc.step(1.0);
        }
        assert!(dc_out.abs() < 1e-5);
    }

    /// verify[dsp.head.mid-side-factorization]
    /// verify[dsp.head.feature-bus-slots]
    /// verify[dsp.head.tier-equivalence]
    /// verify[dsp.head.lower-triangular]
    /// verify[dsp.router.topology-dispatch]
    #[test]
    fn test_head_feature_bus_slots_topologies_and_effective_tier_equivalence() {
        let _guard = FtzDazGuard::new();

        assert_eq!(
            stage3_feature_bus_slot_ranges(Tier::Lite),
            (0..4, 4..62, [62, 63])
        );
        assert_eq!(
            stage3_feature_bus_slot_ranges(Tier::Full),
            (64..68, 68..126, [126, 127])
        );
        assert_eq!(
            stage3_feature_bus_slot_ranges(Tier::Pro),
            (128..136, 136..190, [190, 191])
        );
        assert_eq!(
            stage3_feature_bus_slot_ranges(Tier::Ultra),
            (192..208, 208..254, [254, 255])
        );
        assert_eq!(
            pack_stage3_modal_block_features(&[1.0, 2.0, 3.0, 4.0], &[-1.0, -2.0, -3.0, -4.0]),
            [4.0, 6.0, -4.0, -6.0]
        );

        let mut head = MidSideKroneckerHead {
            w_mid: [0.1; 64],
            b_mid: 0.2,
            ..Default::default()
        };
        let phi = [0.5f32; 64];
        head.update_stereo_width(&[0.05; 64], -0.1, 0.0);
        let (l0, r0) = head.project_feature_bus(&phi, Tier::Lite);
        assert_eq!(l0, r0);

        let mut header = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        for (i, chunk) in arena.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let val = 0.01_f32 * (((i % 13) as f32) - 6.0);
            *chunk = val.to_le_bytes();
        }
        arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
        let full_container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let in_l: [f32; 32] = core::array::from_fn(|n| 0.25 * ((n as f32) * 0.2).sin());
        let in_r: [f32; 32] = core::array::from_fn(|n| -0.20 * ((n as f32) * 0.3).cos());

        for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let stripped = strip_container_to_tier(&full_container, tier).unwrap();
            let mut eng_unstripped =
                DnamEngine::<DefaultMathBackend>::from_bytes(&full_container, tier, 48_000)
                    .unwrap();
            let mut eng_stripped =
                DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, tier, 48_000).unwrap();

            let mut out_ul = [0.0f32; 32];
            let mut out_ur = [0.0f32; 32];
            let mut out_sl = [0.0f32; 32];
            let mut out_sr = [0.0f32; 32];
            eng_unstripped.process_block(&in_l, &in_r, &mut out_ul, &mut out_ur);
            eng_stripped.process_block(&in_l, &in_r, &mut out_sl, &mut out_sr);
            for n in 0..32 {
                assert_eq!(out_ul[n].to_bits(), out_sl[n].to_bits());
                assert_eq!(out_ur[n].to_bits(), out_sr[n].to_bits());
            }
        }
    }

    /// verify[dsp.ctrl.dispatch.stage-laws]
    /// verify[container.dnam.tensor-directory]
    #[test]
    fn test_stage_modulation_laws_and_tensor_directory_slices() {
        let theta_zero = [0.0f32; 16];
        let base = super::eval_stage_modulation_laws(
            &theta_zero,
            [1.0, 1.0],
            64.0,
            2048.0,
            149.0,
            4096.0,
            0.85,
            0.5,
            1.0,
        );
        assert_eq!(base[0], 1.0);
        assert_eq!(base[2], 64.0);
        assert_eq!(base[10], 0.5);
        assert_eq!(base[13], 0.85);
        assert_eq!(base[14], 0.5);
        assert_eq!(base[15], 1.0);

        let theta_extreme = [5000.0f32; 16];
        let clamped = super::eval_stage_modulation_laws(
            &theta_extreme,
            [1.0, 1.0],
            64.0,
            2048.0,
            149.0,
            4096.0,
            0.85,
            0.5,
            1.0,
        );
        assert_eq!(clamped[0], 2.0);
        assert_eq!(clamped[2], 2048.0);
        assert_eq!(clamped[3], 24.0);
        assert_eq!(clamped[4], 16.0);
        assert_eq!(clamped[5], 4.0);
        assert_eq!(clamped[10], 1.0);
        assert_eq!(clamped[13].to_bits(), STAGE4_MAX_G_LIN_BITS);
        assert_eq!(clamped[14], 0.75);
        assert_eq!(clamped[15], 2.0);

        assert!(super::validate_tensor_directory_slice(0, 64, 10_752));
        assert!(super::validate_tensor_directory_slice(128, 256, 10_752));
        assert!(!super::validate_tensor_directory_slice(32, 64, 10_752));
        assert!(!super::validate_tensor_directory_slice(64, 100, 10_752));
        assert!(!super::validate_tensor_directory_slice(10_752, 64, 10_752));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.router.stereo-link-detector]
        /// verify[dsp.stage4.loop-saturator]
        /// verify[dsp.stage4.quiet-feedback-knee]
        /// verify[dsp.ctrl.dispatch.stage-laws]
        #[test]
        fn prop_topology_detector_stage4_saturator_knee_and_stage_laws(
            x_l in -10.0_f32..10.0_f32,
            x_r in -10.0_f32..10.0_f32,
            x_aux in -10.0_f32..10.0_f32,
            alpha_link in 0.0_f32..=1.0_f32,
            beta_key in 0.0_f32..=1.0_f32,
            u_in in -5.0_f32..5.0_f32,
            g_lin in 0.0_f32..=0.9990_f32,
            v_ceil in 0.1_f32..=0.985_f32,
            theta_val in -20.0_f32..20.0_f32,
        ) {
            let (e_l, e_r) = eval_stereo_linked_aux_detector(x_l, x_r, x_aux, alpha_link, beta_key);
            prop_assert!(e_l >= 0.0 && e_r >= 0.0);

            let knee = eval_stage4_blended_feedback_knee(u_in, g_lin, 0.25, 0.5).unwrap();
            prop_assert!(knee.abs() <= g_lin * u_in.abs() + 1e-6);

            let sat = eval_stage4_loop_saturator(u_in, v_ceil).unwrap();
            prop_assert!(sat.abs() <= v_ceil);
            prop_assert!(sat.abs().to_bits() <= STAGE4_MAX_V_CEIL_BITS);

            let laws = super::eval_stage_modulation_laws(
                &[theta_val; 16],
                [1.0, 1.0],
                64.0,
                2048.0,
                149.0,
                4096.0,
                0.85,
                0.5,
                1.0,
            );
            prop_assert!((0.0..=2.0).contains(&laws[0]));
            prop_assert!((4.0..=2048.0).contains(&laws[2]));
            prop_assert!((-24.0..=24.0).contains(&laws[3]));
            prop_assert!((0.0..=1.0).contains(&laws[10]));
            prop_assert!(laws[13].to_bits() <= STAGE4_MAX_G_LIN_BITS);
            prop_assert!((-0.75..=0.75).contains(&laws[14]));
            prop_assert!((0.0..=2.0).contains(&laws[15]));
        }
    }
}
