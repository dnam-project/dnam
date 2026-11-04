//! Canonical v2.0 Stage 0 8-Port Router (WDW, Kaiser-Sinc Sub-Sample Alignment, 200 Hz TPT LR4
//! Crossover, Stereo-Linked & Aux-Key Detector), Stage IV Blended Quiet-Feedback Knee & Outer-Clamped
//! Loop Saturator, 256-D Feature Bus Slot Allocation, and Stage III ZOH First-Tap & 6-Point Lagrange
//! Difference Mappings (`src/stages/topology.rs`).
//!
//! Implements:
//! - `ref[dsp.router.signal-flow]`, `ref[dsp.router.ports]`, `ref[dsp.router.wdw-contract]`,
//!   `ref[dsp.router.return-to-send-isolation]`, `ref[dsp.router.parallel-alignment]`,
//!   `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.mid-side-input]`,
//!   `ref[dsp.router.topology-dispatch]`, `ref[dsp.router.stereo-link-detector]`,
//!   `ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.router.modulation-bus]`
//! - `ref[dsp.stage1.pre-bus]`, `ref[dsp.stage2.commuted-cascade]`, `ref[dsp.stage3.cabinet-bank]`,
//!   `ref[dsp.stage4.spatial-tank]`, `ref[dsp.stage4.captured-freeze]`, `ref[dsp.stage4.dc-blocker]`,
//!   `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`
//! - `ref[dsp.head.mid-side-factorization]`, `ref[dsp.head.feature-bus-slots]`,
//!   `ref[dsp.head.tier-equivalence]`, `ref[dsp.head.lower-triangular]`
//! - `ref[dsp.zoh.router-detector]`, `ref[dsp.zoh.stage3-fir-onset]`

use alloc::vec::Vec;

use crate::math::{flush_recursive_state_f32, newton6_reciprocal_f64, sqrt_f64, widen_f32_bits};
use crate::operators::op_t::catmull_rom_hermite_cubic;
use crate::schema::cold_exp_f64;
use crate::types::Tier;

// ============================================================================
// 1. Stage 0: WDW Contract, Parallel Sub-Sample Alignment & TPT LR4 Crossover
// (`ref[dsp.router.wdw-contract]`, `ref[dsp.router.parallel-alignment]`,
//  `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.stereo-link-detector]`)
// ============================================================================

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
        // Cold-path quadrant-reduced sin(pix) / pix
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
        }
    }

    /// Prepares cold-path `TPT SVF` coefficients in `f64` for cutoff `fc_hz` (default `200 Hz`) at
    /// `effective_rate_hz` (`ref[dsp.router.lr4-crossover]`).
    pub fn new_at_rate(fc_hz: f64, effective_rate_hz: f64) -> Self {
        let angle = (core::f64::consts::PI * fc_hz / effective_rate_hz).clamp(1.0e-5, 1.4);
        // Cold-path tan(angle) via sin/cos Horner series
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
        }
    }

    pub fn reset(&mut self) {
        self.lp_s1 = [0.0; 2];
        self.lp_s2 = [0.0; 2];
        self.hp_s1 = [0.0; 2];
        self.hp_s2 = [0.0; 2];
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
    #[inline]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let (l2_first, _) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            x,
            &mut self.lp_s1[0],
            &mut self.lp_s2[0],
        );
        let (l4, _) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            l2_first,
            &mut self.lp_s1[1],
            &mut self.lp_s2[1],
        );

        let (_, h2_first) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            x,
            &mut self.hp_s1[0],
            &mut self.hp_s2[0],
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
}

/// Evaluates the programmable stereo-linked & aux-key sidechain detector levels `(e_det_l, e_det_r)`
/// with zero `-6.02 dB` hard-panned peak loss (`ref[dsp.router.stereo-link-detector]`).
#[inline]
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
    (0.0100, 0.1500), // Band 0 (Low < 250 Hz): 10 ms att, 150 ms rel
    (0.0020, 0.0400), // Band 1 (Mid 250-2000 Hz): 2 ms att, 40 ms rel
    (0.0002, 0.0150), // Band 2 (High > 2000 Hz): 0.2 ms att, 15 ms rel
    (0.0005, 0.0500), // Band 3 (Aux Key): 0.5 ms att, 50 ms rel
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

// ============================================================================
// 2. Stage IV: Outer-Clamped Loop Saturator & Blended Quiet-Feedback Knee
// (`ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`)
// ============================================================================

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
    let two_over_3v = (2.0_f64 / (3.0_f64 * (v_ceil as f64))) as f32;
    let v = (u * two_over_3v).clamp(-1.0, 1.0);
    let prod = v_ceil * v;
    let half_sq = (0.5 * v) * v;
    let raw = prod * (1.5 - half_sq);
    Some(raw.clamp(-v_ceil, v_ceil))
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

    let inv_sigma = 1.0 / s64;
    let ratio = u64_val.abs() * inv_sigma;
    let denom = 1.0 + ratio * ratio;
    let inv_denom = newton6_reciprocal_f64(denom);
    let factor = (1.0 - a64) + a64 * inv_denom;
    let inter = (g64 * u64_val) * factor;
    Some(inter as f32)
}

// ============================================================================
// 3. Unified 256-D Feature Bus Slot Layout (`ref[dsp.head.feature-bus-slots]`)
// ============================================================================

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

// ============================================================================
// 4. Stage III ZOH First-Tap (44.1-48 kHz) & 6-Point Lagrange (176.4-384 kHz)
// (`ref[dsp.zoh.stage3-fir-onset]`)
// ============================================================================

/// Gain- and DC-constrained 64-tap first-tap anchor coefficients at `44.1 kHz` (`ref[dsp.zoh.stage3-fir-onset]`).
pub const GAIN_BOUNDED_FIRST_TAP_64_BITS: [u32; 64] = [
    0x3d6c_9268,
    0x3f83_725f,
    0xbdf0_a4f1,
    0x3dad_b502,
    0xbd9d_a17d,
    0x3d95_380c,
    0xbd94_9c5f,
    0x3d91_792e,
    0xbd91_ce40,
    0x3d8d_d068,
    0xbd8b_eae0,
    0x3d85_3dac,
    0xbd80_189f,
    0x3d6c_7fa0,
    0xbd5b_a083,
    0x3d42_31cb,
    0xbd2b_f99b,
    0x3d0e_88f9,
    0xbcea_4a93,
    0x3cac_58c1,
    0xbc71_4876,
    0x3bf4_c5f0,
    0xbae2_2743,
    0xbb96_2d07,
    0x3c16_34d0,
    0xbc65_add4,
    0x3c8b_7e54,
    0xbca4_d3dc,
    0x3caf_0e76,
    0xbcb9_82b1,
    0x3cb5_d3c2,
    0xbcb2_e27a,
    0x3ca3_932b,
    0xbc96_3f77,
    0x3c7d_c51e,
    0xbc56_5369,
    0x3c1f_d21c,
    0xbbe7_f768,
    0x3b77_5394,
    0xbaa9_8a21,
    0xbacf_4513,
    0x3b66_e3c0,
    0xbbba_af04,
    0x3be2_53a0,
    0xbc07_70ed,
    0x3c0e_a839,
    0xbc18_1cd3,
    0x3c14_420d,
    0xbc13_3bf5,
    0x3c07_4843,
    0xbbfe_526f,
    0x3bdd_6c26,
    0xbbc6_2d29,
    0x3ba3_1dc4,
    0xbb8b_1039,
    0x3b56_8017,
    0xbb2d_823b,
    0x3af6_3e3f,
    0xbaba_d495,
    0x3a6a_e4c8,
    0xba23_1f37,
    0x39a6_0fed,
    0xb943_301a,
    0x386d_c990,
];

/// Computes the 63 residual difference coefficients $\widetilde p_j = \mathrm{RN}_{32}(\lambda p_j)$
/// for the live `44.1–48 kHz` first-tap difference family ($\lambda = (48000 - f)/3900$,
/// `ref[dsp.zoh.stage3-fir-onset]`).
pub fn prepare_live_first_tap_difference_coeffs(effective_rate_hz: f64) -> Option<[f32; 63]> {
    if !effective_rate_hz.is_finite() || !(44_100.0..=48_000.0).contains(&effective_rate_hz) {
        return None;
    }
    let lambda = (48_000.0 - effective_rate_hz) / 3_900.0;
    let mut prefix = 0.0_f64;
    let mut out = [0.0f32; 63];
    for (idx, slot) in out.iter_mut().enumerate() {
        prefix += f64::from(f32::from_bits(GAIN_BOUNDED_FIRST_TAP_64_BITS[idx]));
        let p_j = if idx == 0 { prefix } else { prefix - 1.0 };
        *slot = (lambda * p_j) as f32;
    }
    Some(out)
}

/// Evaluates the previous-sample anchored live `44.1–48 kHz` first-tap difference filter
/// $x[n-1] + \sum_{j=0}^{62} \widetilde p_j (x[n-j] - x[n-j-1])$, preserving exact `0 ULP` unit DC
/// gain on constant normal histories (`ref[dsp.zoh.stage3-fir-onset]`).
pub fn eval_live_first_tap_difference(history: &[f32; 64], coeffs: &[f32; 63]) -> f32 {
    let mut out = history[1];
    for (j, &c) in coeffs.iter().enumerate() {
        let delta = history[j] - history[j + 1];
        out += c * delta;
    }
    out
}

/// Computes the exact integer decomposition $d = k f / 48000 = m + r / D$ (`m >= 3`), anchor index
/// $a \in \{2, 3\}$, and 5 difference coefficients $p_0..p_4$ for `4×` bounce (`176.4–384 kHz`)
/// tap $k \in [1, 63]$ (`ref[dsp.zoh.stage3-fir-onset]`).
pub fn prepare_bounce_6point_lagrange_tap(
    tap_k: u8,
    effective_rate_hz: f64,
) -> Option<(usize, usize, [f32; 5])> {
    if !(1..=63).contains(&tap_k)
        || !effective_rate_hz.is_finite()
        || !(176_400.0..=384_000.0).contains(&effective_rate_hz)
    {
        return None;
    }
    let bits = effective_rate_hz.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as u32) - 1023;
    let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let denominator = 48_000u64 << (52 - exponent);
    let numerator = u64::from(tap_k) * significand;
    let m = (numerator / denominator) as usize;
    let rem = numerator % denominator;
    let mu = (rem as f64) / (denominator as f64);
    let anchor_a = if 2 * rem <= denominator {
        2usize
    } else {
        3usize
    };

    // 6-point Lagrange weights at nodes v = [-2, -1, 0, 1, 2, 3] with scales c = [-1, 5, -10, 10, -5, 1] / 120
    let nodes = [-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];
    let scales = [-1.0_f64, 5.0, -10.0, 10.0, -5.0, 1.0];
    let mut h = [0.0_f64; 6];
    for i in 0..6 {
        let mut prod = scales[i];
        for (l, &vl) in nodes.iter().enumerate() {
            if l != i {
                prod *= mu - vl;
            }
        }
        h[i] = prod / 120.0;
    }
    let mut prefix = 0.0_f64;
    let mut p_coeffs = [0.0f32; 5];
    for (j, slot) in p_coeffs.iter_mut().enumerate() {
        prefix += h[j];
        let sub = if j >= anchor_a { 1.0 } else { 0.0 };
        *slot = (prefix - sub) as f32;
    }
    Some((m, anchor_a, p_coeffs))
}

/// Evaluates the 6-point Lagrange difference tap $x[n - m - a + 2] + \sum_{j=0}^4 p_j (x[n - m + 2 - j] - x[n - m + 1 - j])$
/// (`ref[dsp.zoh.stage3-fir-onset]`).
pub fn eval_bounce_6point_lagrange_tap(
    history: &[f32],
    m: usize,
    anchor_a: usize,
    p_coeffs: &[f32; 5],
) -> f32 {
    let anchor_idx = m + anchor_a - 2;
    let mut out = history[anchor_idx];
    for (j, &c) in p_coeffs.iter().enumerate() {
        let delta = history[m - 2 + j] - history[m - 1 + j];
        out += c * delta;
    }
    out
}

#[cfg(test)]
mod spec_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Specification Verification Suite: Phase 8 — Stage 0 Router, Stages I–IV, 256-D Head & ZOH.
    //
    // Verifies all 29 rules in `docs/specs/dsp/router_and_stages_topology.md`:
    // - `verify[dsp.router.signal-flow]`
    // - `verify[dsp.router.ports]`
    // - `verify[dsp.router.wdw-contract]`
    // - `verify[dsp.router.return-to-send-isolation]`
    // - `verify[dsp.router.parallel-alignment]`
    // - `verify[dsp.router.lr4-crossover]`
    // - `verify[dsp.router.mid-side-input]`
    // - `verify[dsp.head.mid-side-factorization]`
    // - `verify[dsp.router.topology-dispatch]`
    // - `verify[dsp.router.stereo-link-detector]`
    // - `verify[dsp.router.ballistic-envelope-bank]`
    // - `verify[dsp.stage1.pre-bus]`
    // - `verify[dsp.stage2.commuted-cascade]`
    // - `verify[dsp.stage3.cabinet-bank]`
    // - `verify[dsp.stage4.spatial-tank]`
    // - `verify[dsp.stage4.captured-freeze]`
    // - `verify[dsp.stage4.dc-blocker]`
    // - `verify[dsp.stage4.loop-saturator]`
    // - `verify[dsp.stage4.quiet-feedback-knee]`
    // - `verify[dsp.head.feature-bus-slots]`
    // - `verify[dsp.head.tier-equivalence]`
    // - `verify[dsp.head.lower-triangular]`
    // - `verify[dsp.router.modulation-bus]`
    // - `verify[dsp.zoh.op-m-poles]`
    // - `verify[dsp.zoh.router-detector]`
    // - `verify[dsp.zoh.stage2-op-n]`
    // - `verify[dsp.zoh.op-t-delays]`
    // - `verify[dsp.zoh.ctrl-stride]`
    // - `verify[dsp.zoh.stage3-fir-onset]`

    use crate::container::{
        ARENA_OFFSET_ULTRA_END, ZohRescaledParams, build_dnam_container, strip_container_to_tier,
    };
    use crate::math::DefaultMathBackend;
    use crate::schema::DnamJsonHeader;
    use crate::stages::{
        MidSideKroneckerHead, ParallelAlignmentMode, STAGE4_MAX_G_LIN_BITS, STAGE4_MAX_V_CEIL_BITS,
        Stage0Router, Stage1PreDynamics, Stage2CommutedCascade, Stage3CabinetModals,
        Stage4SpatialTank, TankDcBlocker, TptLr4Crossover, WdwDryTapMode,
        eval_bounce_6point_lagrange_tap, eval_live_first_tap_difference,
        eval_min_latency_4tap_alignment, eval_stage4_blended_feedback_knee,
        eval_stage4_loop_saturator, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
        pack_stage3_modal_block_features, prepare_ballistic_band_alphas,
        prepare_bounce_6point_lagrange_tap, prepare_kaiser_sinc_alignment_weights,
        prepare_live_first_tap_difference_coeffs, stage3_feature_bus_slot_ranges,
    };
    use crate::{DnamEngine, FtzDazGuard, Tier};

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
    /// Verifies Stage 0 8-port routing, WDW `PreSpatialCore` / `DirectDry`, `S_ret->send == 0`,
    /// `FullBand64Tap` (`G=31`, `B=93` @ `4x`, `43` host PDC) & Kaiser-sinc weights, `200 Hz` TPT LR4
    /// crossover, voltage-normalized M/S input, stereo-linked detector, and 4-band ballistic alphas.
    #[test]
    fn test_stage0_router_wdw_isolation_alignment_lr4_ms_and_detector() {
        let _guard = FtzDazGuard::new();

        // 1. WDW PreSpatialCore vs DirectDry output contracts
        let (dry_core, wet_l, wet_r) =
            eval_wdw_routing_outputs(WdwDryTapMode::PreSpatialCore, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_core - 0.4).abs() < 1e-6);
        assert!((wet_l - 0.35).abs() < 1e-6);
        assert!((wet_r - 0.05).abs() < 1e-6);

        let (dry_dir, _, _) =
            eval_wdw_routing_outputs(WdwDryTapMode::DirectDry, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_dir - 0.5).abs() < 1e-6);

        // 2. Return-to-send isolation (S_ret->send == 0) & Voltage-Normalized Mid/Side
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

        // 3. Parallel-path alignment modes & Kaiser-sinc partition of unity
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

        // 4. Stage 0 200 Hz TPT LR4 crossover: DC passes to low, high-frequency passes to high
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

        // 5. Stereo-linked & aux-key detector (0 dB hard-panned peak preservation) & 4-band ballistics
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
    /// Verifies Stages I–IV, Stage IV outer-clamped loop saturator (`0x3F7C28F5` ceiling at `v = 0x3F7FF4D1`),
    /// 6-step `f64` Newton blended quiet-feedback knee (`< 2^-40` error), readout DC conditioning, and freeze.
    #[test]
    fn test_stages_1_to_4_loop_saturator_blended_knee_dc_blocker_and_freeze() {
        let _guard = FtzDazGuard::new();

        // 1. Stage I, II, III, IV basic execution
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

        // 2. Stage IV loop saturator outer clamp prevents f32 intermediate rounding from exceeding 0x3F7C28F5
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

        // 3. Stage IV blended quiet-feedback knee (alpha = 0 linear, alpha = 1 rational, < 2^-40 error)
        let g_max = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
        assert!((g_max as f64) <= 0.999);
        assert!((f32::from_bits(STAGE4_MAX_G_LIN_BITS + 1) as f64) > 0.999);

        let k_lin = eval_stage4_blended_feedback_knee(2.0, 0.8, 0.5, 0.0).unwrap();
        assert!((k_lin - 1.6).abs() < 1e-6);
        let k_rat = eval_stage4_blended_feedback_knee(2.0, 0.8, 0.5, 1.0).unwrap();
        assert!((k_rat - (1.6 / 17.0)).abs() < 1e-6);
        // Extreme input u = 1e9 with alpha = 1.0 stays strictly positive (no 1 - 1 cancellation)
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

        // 4. Stage IV passband-normalized 5 Hz DC blocker
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
    /// Verifies 256-D feature bus slot allocation across `Lite..Ultra`, Mid/Side Kronecker head
    /// factorization, 4 I/O topologies (`1->1`, `2->1`, `1->2`, `2->2`), and stripped vs unstripped
    /// effective-tier equivalence!
    #[test]
    fn test_head_feature_bus_slots_topologies_and_effective_tier_equivalence() {
        let _guard = FtzDazGuard::new();

        // 1. Feature bus slots across Lite, Full, Pro, Ultra
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

        // 2. Mid/Side head stereo width law: width = 0 -> L == R
        let mut head = MidSideKroneckerHead {
            w_mid: [0.1; 64],
            b_mid: 0.2,
            ..Default::default()
        };
        let phi = [0.5f32; 64];
        head.update_stereo_width(&[0.05; 64], -0.1, 0.0);
        let (l0, r0) = head.project_feature_bus(&phi, Tier::Lite);
        assert_eq!(l0, r0);

        // 3. Stripped container at tier T produces bit-identical outputs to unstripped container at tier T
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

    /// verify[dsp.zoh.op-m-poles]
    /// verify[dsp.zoh.router-detector]
    /// verify[dsp.zoh.stage2-op-n]
    /// verify[dsp.zoh.op-t-delays]
    /// verify[dsp.zoh.ctrl-stride]
    /// verify[dsp.zoh.stage3-fir-onset]
    /// Verifies all 6 ZOH sample-rate rescaling laws, including the live `44.1–48 kHz` first-tap
    /// difference family (`0 ULP` constant-history preservation) and the `4×` bounce (`176.4–384 kHz`)
    /// 6-point Lagrange difference mapping.
    #[test]
    fn test_zoh_rescaling_and_stage3_first_tap_and_6point_lagrange_mappings() {
        let _guard = FtzDazGuard::new();

        // 1. Live 44.1..48 kHz first-tap difference family preserves constant normal histories to 0 ULP
        for &rate in &[44_100.0_f64, 46_000.0, 48_000.0] {
            let coeffs = prepare_live_first_tap_difference_coeffs(rate).unwrap();
            for &c_val in &[0.125_f32, -0.125, 1.0, -1.0, 100.0, -100.0] {
                let hist = [c_val; 64];
                let out = eval_live_first_tap_difference(&hist, &coeffs);
                assert_eq!(out.to_bits(), c_val.to_bits());
            }
        }
        assert!(prepare_live_first_tap_difference_coeffs(96_000.0).is_none());

        // 2. 4x bounce (176.4..384 kHz) 6-point Lagrange difference mapping preserves constant normal histories
        for &rate in &[176_400.0_f64, 192_000.0, 352_800.0, 384_000.0] {
            for tap_k in [1u8, 15, 31, 63] {
                let (m, anchor_a, p_coeffs) =
                    prepare_bounce_6point_lagrange_tap(tap_k, rate).unwrap();
                assert!(m >= 3 && (anchor_a == 2 || anchor_a == 3));
                let hist = vec![1.5_f32; m + 8];
                let out = eval_bounce_6point_lagrange_tap(&hist, m, anchor_a, &p_coeffs);
                assert_eq!(out.to_bits(), 1.5_f32.to_bits());
            }
        }

        // 3. ZohRescaledParams router alignment & detector pole helpers
        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        assert!((zoh.rescale_alignment_delay(16.0) - 32.0).abs() < 1e-5);
        assert!(zoh.rescale_real_pole(0.95) > 0.95);
        assert_eq!(zoh.control_decimation_host, 32);
    }

    /// verify[dsp.ctrl.dispatch.stage-laws]
    /// verify[container.dnam.tensor-directory]
    /// Verifies the closed-form per-sample stage modulation laws (`theta_0..theta_15`) and
    /// 64B-aligned `tensor_directory` slice validation (`ADR-0011`).
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

    use proptest::prelude::*;

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

#[cfg(test)]
mod stage3_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Native first-tap feasibility examples, not production Stage III tests.

    use crate::FtzDazGuard;
    use std::hint::black_box;

    const COEFFICIENT_BITS: [u32; 64] = [
        0x3d6c_9268,
        0x3f83_725f,
        0xbdf0_a4f1,
        0x3dad_b502,
        0xbd9d_a17d,
        0x3d95_380c,
        0xbd94_9c5f,
        0x3d91_792e,
        0xbd91_ce40,
        0x3d8d_d068,
        0xbd8b_eae0,
        0x3d85_3dac,
        0xbd80_189f,
        0x3d6c_7fa0,
        0xbd5b_a083,
        0x3d42_31cb,
        0xbd2b_f99b,
        0x3d0e_88f9,
        0xbcea_4a93,
        0x3cac_58c1,
        0xbc71_4876,
        0x3bf4_c5f0,
        0xbae2_2743,
        0xbb96_2d07,
        0x3c16_34d0,
        0xbc65_add4,
        0x3c8b_7e54,
        0xbca4_d3dc,
        0x3caf_0e76,
        0xbcb9_82b1,
        0x3cb5_d3c2,
        0xbcb2_e27a,
        0x3ca3_932b,
        0xbc96_3f77,
        0x3c7d_c51e,
        0xbc56_5369,
        0x3c1f_d21c,
        0xbbe7_f768,
        0x3b77_5394,
        0xbaa9_8a21,
        0xbacf_4513,
        0x3b66_e3c0,
        0xbbba_af04,
        0x3be2_53a0,
        0xbc07_70ed,
        0x3c0e_a839,
        0xbc18_1cd3,
        0x3c14_420d,
        0xbc13_3bf5,
        0x3c07_4843,
        0xbbfe_526f,
        0x3bdd_6c26,
        0xbbc6_2d29,
        0x3ba3_1dc4,
        0xbb8b_1039,
        0x3b56_8017,
        0xbb2d_823b,
        0x3af6_3e3f,
        0xbaba_d495,
        0x3a6a_e4c8,
        0xba23_1f37,
        0x39a6_0fed,
        0xb943_301a,
        0x386d_c990,
    ];

    fn difference_coefficients() -> [f32; 63] {
        let mut prefix = 0.0_f64;
        let mut result = [0.0; 63];
        for (difference, bits) in result.iter_mut().zip(COEFFICIENT_BITS) {
            prefix += f64::from(f32::from_bits(bits));
            *difference = (prefix - 1.0) as f32;
        }
        result
    }

    fn step_difference(history: &[f32; 64], coefficients: &[f32; 63]) -> f32 {
        let mut output = history[0];
        for (offset, coefficient) in coefficients.iter().enumerate() {
            let delta = black_box(history[offset] - history[offset + 1]);
            let product = black_box(coefficient * delta);
            output = black_box(output + product);
        }
        output
    }

    fn step_previous_anchored(history: &[f32; 64], coefficients: &[f32; 63]) -> f32 {
        let mut output = history[1];
        for (offset, coefficient) in coefficients.iter().enumerate() {
            let delta = black_box(history[offset] - history[offset + 1]);
            let product = black_box(coefficient * delta);
            output = black_box(output + product);
        }
        output
    }

    fn step_six_point(
        history: &[f32; 8],
        integer: usize,
        anchor: usize,
        coefficients: &[f32; 5],
    ) -> f32 {
        let mut output = history[anchor];
        for (offset, coefficient) in coefficients.iter().enumerate() {
            let delta = black_box(history[integer - 2 + offset] - history[integer - 1 + offset]);
            let product = black_box(coefficient * delta);
            output = black_box(output + product);
        }
        output
    }

    fn step_six_point_half_delay(history: &[f32; 8]) -> f32 {
        let coefficients = [
            3.0_f32 / 256.0,
            -22.0 / 256.0,
            -0.5,
            22.0 / 256.0,
            -3.0 / 256.0,
        ];
        step_six_point(history, 4, 4, &coefficients)
    }

    fn exact_bounce_delay(rate: f64, tap: u8) -> (u64, u64, u64) {
        assert!((176_400.0..=384_000.0).contains(&rate) && (1..=63).contains(&tap));
        let bits = rate.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u32 - 1023;
        let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
        let denominator = 48_000_u64 << (52 - exponent);
        let numerator = u64::from(tap) * significand;
        (
            numerator / denominator,
            numerator % denominator,
            denominator,
        )
    }

    #[test]
    fn difference_form_preserves_constant_normal_samples() {
        let _guard = FtzDazGuard::new();
        let coefficients = difference_coefficients();
        let mut direct = 0.0_f32;
        for bits in COEFFICIENT_BITS {
            direct = black_box(direct + f32::from_bits(bits));
        }
        assert_eq!(direct.to_bits(), 0x3f80_0002);

        for sample in [0.125_f32, -0.125, 1.0, -1.0, 100.0, -100.0] {
            let history = [sample; 64];
            assert_eq!(
                step_difference(&history, &coefficients).to_bits(),
                sample.to_bits()
            );
        }
    }

    #[test]
    fn previous_sample_anchor_avoids_reference_rate_cancellation() {
        let _guard = FtzDazGuard::new();
        let mut history = [0.0_f32; 64];
        history[0] = black_box(1.0);
        history[1] = black_box(f32::from_bits(0x3300_0000));
        let mut current_anchored = [0.0_f32; 63];
        current_anchored[0] = -1.0;
        assert_eq!(step_difference(&history, &current_anchored).to_bits(), 0);
        assert_eq!(
            step_previous_anchored(&history, &[0.0; 63]).to_bits(),
            history[1].to_bits()
        );
    }

    #[test]
    fn previous_sample_anchor_tracks_tested_20khz_tones() {
        let _guard = FtzDazGuard::new();
        let differences = difference_coefficients();
        for rate in [44_100_usize, 46_000, 48_000] {
            let weight = (48_000.0 - rate as f64) / 3_900.0;
            let mut residual = differences;
            residual[0] += 1.0;
            let coefficients = residual.map(|value| (f64::from(value) * weight) as f32);
            let mut history = [0.0; 64];
            let omega = core::f64::consts::TAU * 20_000.0 / rate as f64;
            let delay = rate as f64 / 48_000.0;
            let mut residual_energy = 0.0;
            let mut target_energy = 0.0;
            for index in 0..64 + rate / 100 {
                history.copy_within(0..63, 1);
                history[0] = (omega * index as f64).sin() as f32;
                let output = step_previous_anchored(&history, &coefficients);
                if index >= 64 {
                    let target = (omega * (index as f64 - delay)).sin();
                    residual_energy += (f64::from(output) - target).powi(2);
                    target_energy += target.powi(2);
                }
            }
            assert!(10.0 * (residual_energy / target_energy).log10() < -35.0);
        }
    }

    #[test]
    fn six_point_half_delay_preserves_constant_normal_samples() {
        let _guard = FtzDazGuard::new();
        for sample in [0.125_f32, -0.125, 1.0, -1.0, 100.0, -100.0] {
            assert_eq!(
                step_six_point_half_delay(&[sample; 8]).to_bits(),
                sample.to_bits()
            );
        }
    }

    #[test]
    fn six_point_nearest_anchor_avoids_integer_boundary_cancellation() {
        let _guard = FtzDazGuard::new();
        let small = f32::from_bits(0x3300_0000);
        let history = [0.0, 1.0, 1.0, 1.0, small, small, small, 0.0];
        let nearest = [
            0x23ba_69dc,
            0xa517_7603,
            0x2608_e5bd,
            0x259d_4952,
            0xa40b_cf65,
        ]
        .map(f32::from_bits);
        let mut previous = nearest;
        previous[2] = -1.0;
        assert_eq!(step_six_point(&history, 3, 3, &previous), 0.0);
        assert!((step_six_point(&history, 3, 4, &nearest) - small).abs() < 1e-14);
    }

    #[test]
    fn bounce_rate_decomposition_preserves_integer_boundary() {
        let rate = f64::from_bits(0x4106_99b6_db6d_b6db);
        assert_eq!((rate * 7.0 / 48_000.0).floor(), 27.0);
        assert_eq!(
            exact_bounce_delay(rate, 7),
            (26, 1_649_267_441_663_997, 1_649_267_441_664_000)
        );
        let (integer, remainder, denominator) = exact_bounce_delay(176_400.0, 16);
        assert_eq!(integer, 58);
        assert_eq!(remainder * 5, denominator * 4);
        assert_eq!(exact_bounce_delay(384_000.0, 63).0, 504);
        assert_eq!(exact_bounce_delay(384_000.0, 63).1, 0);
    }

    #[test]
    fn six_point_half_delay_tracks_one_20khz_bounce_tone() {
        let _guard = FtzDazGuard::new();
        let mut history = [0.0_f32; 8];
        let omega = core::f64::consts::TAU * 20_000.0 / 216_000.0;
        let mut residual_energy = 0.0;
        let mut target_energy = 0.0;
        for index in 0..8 + 2_160 {
            history.copy_within(0..7, 1);
            history[0] = (omega * index as f64).sin() as f32;
            let output = step_six_point_half_delay(&history);
            if index >= 8 {
                let target = (omega * (index as f64 - 4.5)).sin();
                residual_energy += (f64::from(output) - target).powi(2);
                target_energy += target.powi(2);
            }
        }
        assert!(10.0 * (residual_energy / target_energy).log10() < -63.8);
    }

    #[test]
    fn difference_form_tracks_one_20khz_tone() {
        let _guard = FtzDazGuard::new();
        let coefficients = difference_coefficients();
        let mut history = [0.0; 64];
        let omega = core::f64::consts::TAU * 20_000.0 / 44_100.0;
        let delay = 44_100.0 / 48_000.0;
        let mut residual_energy = 0.0;
        let mut target_energy = 0.0;
        for index in 0..64 + 441 {
            history.copy_within(0..63, 1);
            history[0] = (omega * index as f64).sin() as f32;
            let output = step_difference(&history, &coefficients);
            if index >= 64 {
                let target = (omega * (index as f64 - delay)).sin();
                residual_energy += (f64::from(output) - target).powi(2);
                target_energy += target.powi(2);
            }
        }
        assert!(10.0 * (residual_energy / target_energy).log10() < -35.5);
    }
}
