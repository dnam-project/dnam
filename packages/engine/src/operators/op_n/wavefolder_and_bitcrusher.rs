//! Op-N Cubed-Envelope C^2 Wavefolder, 2D Quadrature Hysteresis & Fixed-Hertz ZOH Bitcrusher.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.wavefolder.cubic-c2]`, `ref[dsp.op-n.hysteresis.quadrature]`, `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`).

/// Instantaneous C^2 compactly supported wavefolder.
/// Evaluates the cubed-envelope $C^2$ wavefolder (`ref[dsp.op-n.wavefolder.cubic-c2]`):
/// `w_odd(u) = u_bar * (1 - u_bar^2)^3` and `w_even(u) = u_bar^2 * (1 - u_bar^2)^3` on `u_bar = clamp(u, -1, 1)`.
#[inline(always)]
pub fn c2_wavefolder(u: f32) -> (f32, f32) {
    let u_clamped = u.clamp(-1.0, 1.0);
    let u2 = u_clamped * u_clamped;
    let one_minus_u2 = 1.0 - u2;
    let factor = one_minus_u2 * one_minus_u2 * one_minus_u2;

    let w_odd = u_clamped * factor;
    let w_even = u2 * factor;
    (w_odd, w_even)
}

/// 2D Quadrature Phase-Space Hysteresis state.
///
/// Models magnetic transformer core memory and optocoupler decay.
#[derive(Clone, Copy, Debug)]
pub struct QuadratureHysteresis2D {
    pub prev_s_re: f32,
    pub kappa_quad: f32,
}

impl Default for QuadratureHysteresis2D {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadratureHysteresis2D {
    pub const fn new() -> Self {
        Self {
            prev_s_re: 0.0,
            kappa_quad: 0.15,
        }
    }

    pub fn reset(&mut self) {
        self.prev_s_re = 0.0;
    }

    /// Steps the 2D hysteresis model given quadrature phase-space coordinates `(s_re, s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, s_re: f32, s_im: f32) -> f32 {
        self.step_with_rate(s_re, s_im, 1.0)
    }

    /// Steps the 2D hysteresis model with continuous-time ZOH differential rate scaling `diff_scale = 1 / rho_sr`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_with_rate(&mut self, s_re: f32, s_im: f32, diff_scale: f32) -> f32 {
        let delta_re = (s_re - self.prev_s_re) * diff_scale;
        let hyst = s_re * delta_re + self.kappa_quad * s_re * s_im;
        self.prev_s_re = s_re;
        hyst
    }
}

/// Fixed-Hertz `Q64.32` Integer ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`, `ref[dsp.op-n.bitcrusher.quantizer]`).
#[derive(Clone, Copy, Debug)]
pub struct ZohBitcrusher {
    pub step_size: f32,
    pub step_inv: f32,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub bit_depth: u8,
    pub target_fs_hz: f32,
    pub held_sample: f32,
}

impl Default for ZohBitcrusher {
    fn default() -> Self {
        Self::new()
    }
}

impl ZohBitcrusher {
    pub const fn new() -> Self {
        let den = 48_000u64 << 32;
        Self {
            step_size: 1.0 / 32768.0,
            step_inv: 32768.0,
            phase: 0,
            increment: den,
            denominator: den,
            reciprocal_r: ((1u128 << 104) / (den as u128)) as u64,
            bit_depth: 16,
            target_fs_hz: 48_000.0,
            held_sample: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.held_sample = 0.0;
    }

    /// Sets bit depth `bits in [2, 16]` and downsampled sample rate `target_fs_hz` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
    pub fn set_parameters(&mut self, bits: u32, target_fs_hz: f32, sample_rate_hz: f32) {
        let b = bits.clamp(2, 16) as u8;
        let levels = (1u32 << (b - 1)) as f32;
        self.bit_depth = b;
        self.step_inv = levels;
        self.step_size = 1.0 / levels;
        self.target_fs_hz = target_fs_hz.max(0.0);

        let sr = sample_rate_hz.max(1000.0) as f64;
        if let Some((inc, den, rec)) =
            hertz_phase_parameters(self.target_fs_hz.min(sample_rate_hz) as f64, sr)
        {
            self.increment = inc;
            self.denominator = den;
            self.reciprocal_r = rec;
            if self.phase >= self.denominator {
                self.phase %= self.denominator;
            }
        }
    }

    /// Transfers the `Q64.32` phase accumulator across a host sample-rate change without `u128` division (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
    pub fn reconfigure_sample_rate(&mut self, new_sample_rate_hz: f32) {
        let sr = new_sample_rate_hz.max(1000.0) as f64;
        if let Some((inc, den, rec)) =
            hertz_phase_parameters(self.target_fs_hz.min(new_sample_rate_hz) as f64, sr)
        {
            self.phase =
                rescale_bitcrusher_phase_u128(self.phase, self.denominator, den, self.reciprocal_r);
            self.increment = inc;
            self.denominator = den;
            self.reciprocal_r = rec;
        }
    }

    /// Steps the ZOH bitcrusher with zero audio-rate divisions (`ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.quantizer]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.increment == 0 {
            return self.held_sample;
        }

        let latch = if self.increment >= self.denominator {
            true
        } else {
            let (next_phase, l) = hertz_phase_step(self.phase, self.increment, self.denominator);
            self.phase = next_phase;
            l
        };

        if latch {
            self.held_sample =
                quantize_bitcrusher_nearest_ties_pos_inf(x, u32::from(self.bit_depth));
        }

        self.held_sample
    }
}

/// Evaluates the odd degree-7 interval-average `PolyADAA` basis polynomial $A_{\text{odd},3}$ in `f32`
/// for the cubed-envelope $C^2$ wavefolder on clamped endpoints $u_0, u_1 \in [-1, 1]$ (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline(always)]
pub fn eval_adaa_cubic_c2_wavefolder_odd_f32(u_prev: f32, u_curr: f32) -> f32 {
    let u0 = u_prev.clamp(-1.0, 1.0);
    let u1 = u_curr.clamp(-1.0, 1.0);
    let m = 0.5 * (u1 + u0);
    let d = 0.5 * (u1 - u0);
    let m2 = m * m;
    let d2 = d * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let m6 = m4 * m2;
    let d6 = d4 * d2;
    m * (1.0 - 3.0 * (m2 + d2) + 3.0 * m4 + 10.0 * m2 * d2 + 3.0 * d4
        - m6
        - 7.0 * (m4 * d2 + m2 * d4)
        - d6)
}

/// Evaluates the division-free degree-7/8 interval-average `PolyADAA` basis polynomials
/// $(A_{\text{odd},3}, A_{\text{even},3})$ for the cubed-envelope $C^2$ wavefolder on clamped
/// endpoints $u_0, u_1 \in [-1, 1]$ (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline]
pub fn eval_adaa_cubic_c2_wavefolder_f64(u_prev: f64, u_curr: f64) -> (f64, f64) {
    let u0 = u_prev.clamp(-1.0, 1.0);
    let u1 = u_curr.clamp(-1.0, 1.0);
    let m = 0.5 * (u1 + u0);
    let d = 0.5 * (u1 - u0);
    let m2 = m * m;
    let d2 = d * d;
    let m3 = m2 * m;
    let d3 = d2 * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let odd = m
        * (1.0 - 3.0 * (m2 + d2) + 3.0 * m4 + 10.0 * m2 * d2 + 3.0 * d4
            - m3 * m3
            - 7.0 * m4 * d2
            - 7.0 * m2 * d4
            - d3 * d3);
    let even = m2 + d2 / 3.0 - 3.0 * (m4 + 2.0 * m2 * d2 + 0.2 * d4)
        + 3.0 * (m3 * m3 + 5.0 * m4 * d2 + 3.0 * m2 * d4 + d3 * d3 / 7.0)
        - (m4 * m4
            + (28.0 / 3.0) * m3 * m3 * d2
            + 14.0 * m4 * d4
            + 4.0 * m2 * d3 * d3
            + d4 * d4 / 9.0);
    (odd, even)
}

/// Evaluates the uniform nearest-step quantizer $Q_B(x) = \lfloor x 2^{B-1} + 0.5 \rfloor 2^{-(B-1)}$
/// (halfway ties toward $+\infty$) for $B \in [2, 16]$ using floor-residual `i32` truncation
/// (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline]
pub fn quantize_bitcrusher_nearest_ties_pos_inf(x: f32, bits: u32) -> f32 {
    let b = bits.clamp(2, 16);
    let scale = (1u32 << (b - 1)) as f32;
    let step_size = 1.0_f32 / scale;
    if !x.is_finite() || x.abs() >= 8_388_608.0 * step_size {
        return x;
    }
    let v = x * scale;
    let t = v as i32;
    let k = t - i32::from(v < (t as f32));
    let frac = v - (k as f32);
    let q = k + i32::from(frac >= 0.5);
    (q as f32) * step_size
}

/// Computes the fixed-hertz integer ZOH phase parameters `(increment_i, denominator_d, reciprocal_r)`
/// at $F = 2^{32}\text{ units/Hz}$ with $R = \lfloor 2^{104} / D \rfloor$ (`u64`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`).
pub fn hertz_phase_parameters(target_hz: f64, sample_rate_hz: f64) -> Option<(u64, u64, u64)> {
    if !target_hz.is_finite()
        || !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !(0.0..=sample_rate_hz).contains(&target_hz)
    {
        return None;
    }
    let scale = (1u64 << 32) as f64;
    let d_f64 = (sample_rate_hz * scale).round_ties_even();
    let i_f64 = (target_hz * scale).round_ties_even();
    if d_f64 <= 0.0 || d_f64 >= ((1u64 << 63) as f64) {
        return None;
    }
    let d = d_f64 as u64;
    let i = i_f64 as u64;
    if (target_hz > 0.0 && i == 0) || i > d {
        return None;
    }
    let r_u128 = (1u128 << 104) / (d as u128);
    let r = u64::try_from(r_u128).unwrap_or(u64::MAX);
    Some((i, d, r))
}

/// Advances the fixed-hertz `u64` phase accumulator by 1 sample: `A = P + I`, `b = A >= D`,
/// `P <- A - b * D` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
#[inline]
pub const fn hertz_phase_step(phase: u64, increment: u64, denominator: u64) -> (u64, bool) {
    let advanced = phase.wrapping_add(increment);
    let latch = advanced >= denominator;
    let next_phase = if latch {
        advanced - denominator
    } else {
        advanced
    };
    (next_phase, latch)
}

/// Transfers a fixed-hertz `u64` phase `phase_old` from `old_denominator` to `new_denominator`
/// without hardware division using precomputed $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$
/// (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[inline]
pub const fn rescale_bitcrusher_phase_u128(
    phase_old: u64,
    old_denominator: u64,
    new_denominator: u64,
    reciprocal_r: u64,
) -> u64 {
    let numerator = (phase_old as u128) * (new_denominator as u128);
    let n_lo = (numerator & (u64::MAX as u128)) as u64;
    let n_hi = (numerator >> 64) as u64;
    let a = (n_lo as u128) * (reciprocal_r as u128);
    let c = (n_hi as u128) * (reciprocal_r as u128);
    let m = (a >> 64) + (c & (u64::MAX as u128));
    let h = (c >> 64) + (m >> 64);
    let mut q0 = ((h as u64) << 24) | (((m & (u64::MAX as u128)) as u64) >> 40);
    let mut r0 = numerator - (q0 as u128) * (old_denominator as u128);
    let d_old_128 = old_denominator as u128;
    if r0 >= d_old_128 {
        q0 += 1;
        r0 -= d_old_128;
    }
    if 2 * r0 > d_old_128 || (2 * r0 == d_old_128 && (q0 & 1) != 0) {
        q0 += 1;
    }
    let max_phase = new_denominator.saturating_sub(1);
    if q0 > max_phase { max_phase } else { q0 }
}

/// Fixed-hertz `u64`-phase ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`,
/// `ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[derive(Clone, Copy, Debug)]
pub struct FixedHzBitcrusher {
    pub bit_depth: u32,
    pub target_hz: f64,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub held_sample: f32,
}

impl Default for FixedHzBitcrusher {
    fn default() -> Self {
        Self::new(16, 48_000.0, 48_000.0).unwrap()
    }
}

impl FixedHzBitcrusher {
    pub fn new(bit_depth: u32, target_hz: f64, sample_rate_hz: f64) -> Option<Self> {
        let (inc, den, rec) = hertz_phase_parameters(target_hz, sample_rate_hz)?;
        Some(Self {
            bit_depth: bit_depth.clamp(2, 16),
            target_hz,
            phase: 0,
            increment: inc,
            denominator: den,
            reciprocal_r: rec,
            held_sample: 0.0,
        })
    }

    pub fn set_sample_rate(&mut self, new_sample_rate_hz: f64) -> Option<()> {
        let eff_target = self.target_hz.min(new_sample_rate_hz);
        let (new_inc, new_den, new_rec) = hertz_phase_parameters(eff_target, new_sample_rate_hz)?;
        self.phase =
            rescale_bitcrusher_phase_u128(self.phase, self.denominator, new_den, self.reciprocal_r);
        self.increment = new_inc;
        self.denominator = new_den;
        self.reciprocal_r = new_rec;
        Some(())
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.increment == 0 {
            return self.held_sample;
        }
        let latch = if self.increment == self.denominator {
            true
        } else {
            let (next_p, l) = hertz_phase_step(self.phase, self.increment, self.denominator);
            self.phase = next_p;
            l
        };
        if latch {
            self.held_sample = quantize_bitcrusher_nearest_ties_pos_inf(x, self.bit_depth);
        }
        self.held_sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::operators::eval_saturated_polyadaa_rail_crossing_f64;
    use proptest::prelude::*;

    #[test]
    fn test_wavefolder_and_hysteresis() {
        let (w_odd, w_even) = c2_wavefolder(0.5);
        assert!((w_odd - 0.5 * 0.421_875).abs() < 1e-6);
        assert!((w_even - 0.25 * 0.421_875).abs() < 1e-6);

        let _audio = FtzDazGuard::new();
        let mut hyst = QuadratureHysteresis2D::default();
        assert_eq!(hyst.kappa_quad, 0.15);
        hyst.reset();

        let h1 = hyst.step(0.5, 0.2);
        let expected_h1 = 0.5 * 0.5 + 0.15 * 0.5 * 0.2;
        assert!((h1 - expected_h1).abs() < 1e-6);
        assert_eq!(hyst.prev_s_re, 0.5);

        let h2 = hyst.step(0.8, -0.1);
        let expected_h2 = 0.8 * 0.3 + 0.15 * 0.8 * (-0.1);
        assert!((h2 - expected_h2).abs() < 1e-6);

        hyst.reset();
        assert_eq!(hyst.prev_s_re, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.reset();
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert!((cr.step_size - 0.125).abs() < 1e-6);
        assert_eq!(cr.increment * 4, cr.denominator);

        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        let latched = cr.step(0.5);
        let expected_q = (0.5 * 8.0_f32 + 0.5).floor() * 0.125;
        assert!((latched - expected_q).abs() < 1e-6);
        assert_eq!(cr.step(0.9), latched);

        cr.reset();
        assert_eq!(cr.phase, 0);
        assert_eq!(cr.held_sample, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher_full_and_zero_rate() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.set_parameters(4, 48_000.0, 48_000.0);
        let den = cr.denominator;
        for phase in [0_u64, den / 10, den / 4, den - 1] {
            cr.phase = phase;
            for sample in [0.125, -0.25, 0.5, -0.75] {
                assert_eq!(cr.step(sample), sample);
                assert_eq!(cr.phase, phase);
            }
        }

        cr.phase = den / 4;
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.phase, den / 4);
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), 0.5);

        cr.reset();
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.phase, 0);
    }

    #[test]
    fn test_zoh_bitcrusher_quantization_boundaries() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        for bits in 2..=16 {
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            for midpoint in [-1.5_f32, -0.5, 0.5, 1.5] {
                let encoded = midpoint.to_bits();
                for scaled in [
                    f32::from_bits(encoded - 1),
                    midpoint,
                    f32::from_bits(encoded + 1),
                ] {
                    let input = scaled * cr.step_size;
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }

            for input in [-100.0, 100.0, -f32::MAX, f32::MAX] {
                assert_eq!(cr.step(input), input, "bits={bits}, input={input}");
            }

            let threshold = 8_388_608.0 * cr.step_size;
            for value in [
                f32::from_bits(threshold.to_bits() - 1),
                threshold,
                f32::from_bits(threshold.to_bits() + 1),
            ] {
                for input in [value, -value] {
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }
        }
    }

    #[test]
    fn test_c2_wavefolder_smoothness_and_rails() {
        let (w_odd_pos, w_even_pos) = c2_wavefolder(1.0);
        assert!((w_odd_pos).abs() < 1e-7);
        assert!((w_even_pos).abs() < 1e-7);

        let (w_odd_neg, w_even_neg) = c2_wavefolder(-1.0);
        assert!((w_odd_neg).abs() < 1e-7);
        assert!((w_even_neg).abs() < 1e-7);

        let eps = 1e-4_f32;
        let (w_near, _) = c2_wavefolder(1.0 - eps);
        let deriv = (w_near - w_odd_pos) / (-eps);
        assert!(deriv.abs() < 1e-3);
    }

    /// verify[dsp.op-n.wavefolder.cubic-c2]
    /// verify[dsp.op-n.hysteresis.quadrature]
    /// verify[dsp.op-n.bitcrusher.quantizer]
    /// verify[dsp.op-n.bitcrusher.zoh-phase]
    /// verify[dsp.op-n.bitcrusher.rate-transfer]
    #[test]
    fn test_cubic_c2_wavefolder_hysteresis_and_fixed_hz_bitcrusher_rate_transfer() {
        let _guard = FtzDazGuard::new();

        let (wo, we) = c2_wavefolder(0.5);
        assert!((wo - 0.5 * 0.421_875).abs() < 1e-6);
        assert!((we - 0.25 * 0.421_875).abs() < 1e-6);
        assert_eq!(c2_wavefolder(1.0), (0.0, 0.0));
        assert_eq!(c2_wavefolder(-1.0), (0.0, 0.0));

        let (ao_eq, ae_eq) = eval_adaa_cubic_c2_wavefolder_f64(0.5, 0.5);
        assert!((ao_eq - 0.5 * 0.421_875).abs() < 1e-12);
        assert!((ae_eq - 0.25 * 0.421_875).abs() < 1e-12);

        let mut hyst = QuadratureHysteresis2D::new();
        let h1 = hyst.step_with_rate(0.5, 0.2, 2.0);
        assert!((h1 - (0.5 * 0.5 * 2.0 + 0.15 * 0.5 * 0.2)).abs() < 1e-6);

        for bits in 2..=16u32 {
            let scale = (1u32 << (bits - 1)) as f64;
            for mid in [-1.5_f32, -0.5, 0.5, 1.5] {
                let x = mid / (scale as f32);
                let expected = ((f64::from(x) * scale + 0.5).floor() / scale) as f32;
                assert_eq!(quantize_bitcrusher_nearest_ties_pos_inf(x, bits), expected);
            }
        }

        for &(target_hz, period) in &[(16_000.0_f64, 3usize), (12_000.0, 4), (48_000.0, 1)] {
            let (inc, den, _) = hertz_phase_parameters(target_hz, 48_000.0).unwrap();
            let mut p = 0u64;
            for sample in 1..=(period * 8) {
                let (next_p, latch) = hertz_phase_step(p, inc, den);
                p = next_p;
                assert_eq!(latch, sample % period == 0);
            }
            assert_eq!(p, 0);
        }

        let mut cr = FixedHzBitcrusher::new(4, 12_000.0, 48_000.0).unwrap();
        for _ in 0..3 {
            assert_eq!(cr.step(0.5), 0.0);
        }
        assert_eq!(cr.step(0.5), 0.5);
        let saved_phase = cr.phase;
        cr.increment = 0;
        assert_eq!(cr.step(0.875), 0.5);
        assert_eq!(cr.phase, saved_phase);

        let rates = [44_100.0_f64, 48_000.0, 96_000.0, 192_000.0, 384_000.0];
        for &old_r in &rates {
            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_r).unwrap();
            for &new_r in &rates {
                let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_r).unwrap();
                for p_old in [0u64, 1, d_old / 3, d_old / 2, d_old - 1] {
                    let actual = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
                    let num = (p_old as u128) * (d_new as u128);
                    let den = d_old as u128;
                    let mut q = (num / den) as u64;
                    let rem = num % den;
                    if 2 * rem > den || (2 * rem == den && (q & 1) != 0) {
                        q += 1;
                    }
                    let expected = q.min(d_new - 1);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_zoh_bitcrusher_quantization_against_f64(
            bits in 2u32..=16,
            input in -100.0_f32..100.0,
        ) {
            let _audio = FtzDazGuard::new();
            let mut cr = ZohBitcrusher::default();
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
            prop_assert_eq!(cr.step(input), expected);
        }

        #[test]
        fn prop_c2_wavefolder_bounds_and_symmetry(u in -2.0f32..2.0f32) {
            let (w_odd, w_even) = c2_wavefolder(u);
            let (w_odd_neg, w_even_neg) = c2_wavefolder(-u);
            prop_assert!(w_odd.abs() <= 1.0001);
            prop_assert!(w_even.abs() <= 1.0001);
            prop_assert!((w_odd + w_odd_neg).abs() < 1e-6);
            prop_assert!((w_even - w_even_neg).abs() < 1e-6);
            if u.abs() >= 1.0 {
                prop_assert_eq!(w_odd, 0.0);
                prop_assert_eq!(w_even, 0.0);
            }
        }

        /// verify[dsp.op-n.polyadaa.rail-crossing]
        /// verify[dsp.op-n.wavefolder.cubic-c2]
        /// verify[dsp.op-n.bitcrusher.quantizer]
        /// verify[dsp.op-n.bitcrusher.rate-transfer]
        #[test]
        fn prop_op_n_rail_crossing_wavefolder_and_bitcrusher(
            raw_prev in -10.0_f64..10.0_f64,
            raw_curr in -10.0_f64..10.0_f64,
            bits in 2u32..=16u32,
            sample_x in -100.0_f32..100.0_f32,
            old_sr in 44_100.0_f64..=96_000.0_f64,
            new_sr in 44_100.0_f64..=96_000.0_f64,
            frac in 0.0_f64..0.999_f64,
        ) {
            let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(raw_prev, raw_curr);
            prop_assert!(odd.is_finite() && odd.abs() <= 1.0 + 1e-12);
            prop_assert!(even.is_finite() && (0.0..=1.0).contains(&even));

            let (wf_o, wf_e) = eval_adaa_cubic_c2_wavefolder_f64(raw_prev, raw_curr);
            prop_assert!(wf_o.is_finite() && wf_o.abs() <= 1.0);
            prop_assert!(wf_e.is_finite() && (0.0..=1.0).contains(&wf_e));

            let q = quantize_bitcrusher_nearest_ties_pos_inf(sample_x, bits);
            let step = 1.0_f32 / ((1u32 << (bits - 1)) as f32);
            prop_assert!((q - sample_x).abs() <= 0.5 * step + 1e-5);

            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_sr).unwrap();
            let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_sr).unwrap();
            let p_old = ((d_old as f64) * frac) as u64;
            let p_new = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
            prop_assert!(p_new < d_new);
        }
    }
}
