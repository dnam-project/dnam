//! `T1` Sub-Harmonic Tracker, 160-Bit Midpoint `5 Hz` DC Blocker, Exponent-Split Phasor
//! Normalization, and Riemann Branch-Cut Toggle Sheet (`ref[dsp.op-m.tracker.t1-dc-blocker]`,
//! `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).

use super::kautz::fast_rsqrt_nr2;
use crate::math::flush_recursive_state_f32;

/// Discrete 1-bit zero-crossing toggle latch and Riemann sheet tracker (`T1`, Family 1,
/// `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct SubHarmonicTracker {
    pub prev_x: f32,
    pub prev_q: f32,
    pub prev_u_re: f32,
    pub prev_u_im: f32,
    pub sheet: f32,
    pub dc_pole: f32,
    pub dc_gain: f32,
    pub dc_prev_x: f32,
    pub dc_prev_y: f32,
}

impl Default for SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SubHarmonicTracker {
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_q: 1.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
            dc_pole: f32::from_bits(0x3f7f_d51f),
            dc_gain: f32::from_bits(0x3f7f_ea90),
            dc_prev_x: 0.0,
            dc_prev_y: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_q = 1.0;
        self.prev_u_re = 1.0;
        self.prev_u_im = 0.0;
        self.sheet = 1.0;
        self.dc_prev_x = 0.0;
        self.dc_prev_y = 0.0;
    }

    /// Recomputes the `10 Hz` `T1` DC blocker pole and passband gain for `sample_rate_hz` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: u64) {
        if let Some((r_bits, g_bits)) = compute_t1_dc_blocker_coeffs_bits(sample_rate_hz as f64) {
            self.dc_pole = f32::from_bits(r_bits);
            self.dc_gain = f32::from_bits(g_bits);
        }
    }

    /// Steps the `10 Hz` `T1` high-pass DC blocker on raw input `x` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    #[inline(always)]
    pub fn step_dc_blocker(&mut self, x: f32) -> f32 {
        let y = flush_recursive_state_f32(
            self.dc_gain * (x - self.dc_prev_x) + self.dc_pole * self.dc_prev_y,
        );
        self.dc_prev_x = flush_recursive_state_f32(x);
        self.dc_prev_y = y;
        y
    }

    /// Steps the bistable sub-harmonic tracker using rising zero-crossings.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, hilbert_x: f32) -> f32 {
        // Analytic envelope normalization via fast NR2 rsqrt with silence squelch (R^2 < 1e-12)
        let r2 = x * x + hilbert_x * hilbert_x;
        if r2 < 1.0e-12 {
            self.prev_x = x;
            return 0.0;
        }

        // Toggle latch on rising edge (only when above silence squelch threshold)
        if self.prev_x <= 0.0 && x > 0.0 {
            self.prev_q = -self.prev_q;
        }
        self.prev_x = x;

        let rho = fast_rsqrt_nr2(r2);
        let u_re = x * rho;
        let u_im = hilbert_x * rho;

        // Riemann branch cut tracking: branch flips when crossing negative real axis
        if ((self.prev_u_im >= 0.0 && u_im < 0.0) || (self.prev_u_im <= 0.0 && u_im > 0.0))
            && u_re < 0.0
        {
            self.sheet = -self.sheet;
        }
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;

        // Continuous half-angle representation: sqrt((1 + u_re)/2)
        let half_angle = ((1.0 + u_re) * 0.5).max(0.0);
        let half_rho = fast_rsqrt_nr2(half_angle);
        let sub_continuous = if half_rho > 0.0 {
            half_angle * half_rho
        } else {
            0.0
        };

        0.5 * (self.prev_q + self.sheet * sub_continuous)
    }
}

/// Outward 160-bit interval bounds on $\mathrm{atanh}((D - N) / (D + N)) \cdot 2^{160}$ using 8 terms
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct U256Be {
    limbs: [u64; 4], // big-endian limbs for natural Ord comparison: limbs[0] = MSW, limbs[3] = LSW
}

impl U256Be {
    const ZERO: Self = Self { limbs: [0; 4] };

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let s = (self.limbs[i] as u128) + (rhs.limbs[i] as u128) + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        Self { limbs: out }
    }

    fn add_u64(self, rhs: u64) -> Self {
        self.add(Self {
            limbs: [0, 0, 0, rhs],
        })
    }

    fn mul_u64(self, rhs: u64) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let p = (self.limbs[i] as u128) * (rhs as u128) + carry;
            out[i] = p as u64;
            carry = p >> 64;
        }
        Self { limbs: out }
    }

    fn div_u64(self, divisor: u64) -> Self {
        let mut out = [0u64; 4];
        let mut rem = 0u128;
        let d = divisor as u128;
        for i in 0..4 {
            let cur = (rem << 64) | (self.limbs[i] as u128);
            out[i] = (cur / d) as u64;
            rem = cur % d;
        }
        Self { limbs: out }
    }
}

fn atanh_bounds_160bit(numerator: u64, denominator: u64) -> (U256Be, U256Be) {
    let z_num = denominator - numerator;
    let z_den = denominator + numerator;
    // Start with (1 << 160) * z_num / z_den: (1 << 160) has bit 32 set in limbs[1]
    let one_shl_160 = U256Be {
        limbs: [0, 1u64 << 32, 0, 0],
    };
    let mut cur = one_shl_160.mul_u64(z_num).div_u64(z_den);
    let z2_num = z_num * z_num;
    let z2_den = z_den * z_den;
    let mut lower = U256Be::ZERO;
    for idx in 0..8u64 {
        lower = lower.add(cur.div_u64(2 * idx + 1));
        cur = cur.mul_u64(z2_num).div_u64(z2_den);
    }
    (lower, lower.add_u64(16))
}

/// Computes the `T1` tracker `5 Hz` DC blocker `(pole_bits, gain_bits)` at `rate_hz` (`44.1..96 kHz`
/// or `176.4..384 kHz`) using the normative 160-bit midpoint binary search without `libm` calls
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
pub fn compute_t1_dc_blocker_coeffs_bits(rate_hz: f64) -> Option<(u32, u32)> {
    if !rate_hz.is_finite() {
        return None;
    }
    let (first, last) = if (44_100.0..=96_000.0).contains(&rate_hz) {
        (0x00ff_d000u32, 0x00ff_ebffu32)
    } else if (176_400.0..=384_000.0).contains(&rate_hz) {
        (0x00ff_f400u32, 0x00ff_fbffu32)
    } else {
        return None;
    };

    // Express rate_hz = rate_num / rate_den in reduced integer form (rate_num < 2^53, rate_den <= 2^37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u64 << 52) | (bits & ((1u64 << 52) - 1));
    let (rate_num, rate_den): (u64, u64) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 1)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, 1u64 << (raw_k - tz))
    };
    let exponent_num = 48_000u64 * rate_den;
    let ref_bounds = atanh_bounds_160bit(0x00ff_d51f, 1u64 << 24);

    let mut lower = first + 1;
    let mut upper = last;
    while lower < upper {
        let candidate = (lower + upper) / 2;
        let mid_bounds = atanh_bounds_160bit((2 * candidate + 1) as u64, 1u64 << 25);
        let lhs = mid_bounds.0.mul_u64(rate_num);
        let rhs = ref_bounds.1.mul_u64(exponent_num);
        if lhs > rhs {
            lower = candidate + 1;
        } else {
            upper = candidate;
        }
    }
    let pole_bits = 0x3f00_0000 + lower - (1 << 23);
    let num_gain = (1u32 << 24) + lower;
    let q = num_gain / 2;
    let r = num_gain % 2;
    let rounded_gain = q + u32::from(r == 1 && (q & 1) != 0);
    let gain_bits = 0x3f00_0000 + rounded_gain - (1 << 23);
    Some((pole_bits, gain_bits))
}

/// Tracker-only passband-normalized `5 Hz` one-pole DC blocker (`T1`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct T1DcBlocker {
    pub pole: f32,
    pub gain: f32,
    pub prev_x: f32,
    pub prev_d: f32,
}

impl Default for T1DcBlocker {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl T1DcBlocker {
    pub const fn new_48khz() -> Self {
        Self {
            pole: f32::from_bits(0x3f7f_d51f),
            gain: f32::from_bits(0x3f7f_ea90),
            prev_x: 0.0,
            prev_d: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_d = 0.0;
    }

    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate_hz)?;
        Some(Self {
            pole: f32::from_bits(p_bits),
            gain: f32::from_bits(g_bits),
            prev_x: 0.0,
            prev_d: 0.0,
        })
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let d = self.gain * (x - self.prev_x) + self.pole * self.prev_d;
        self.prev_x = x;
        self.prev_d = flush_recursive_state_f32(d);
        d
    }
}

#[inline]
fn rsqrt_nr2_exact_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(0x5f37_59df - (value.to_bits() >> 1));
    for _ in 0..2 {
        let corr = 1.5 - ((0.5 * value) * est) * est;
        est *= corr;
    }
    est
}

/// Normalizes an analytic pair $(I, Q)$ using exponent-split pre-scaling by $2^{-\min(E, 126)}$
/// (with $2^{24}$ subnormal pre-scaling) and 2-step Newton `rsqrt` (`0 FSQRT`, `0 FDIV`,
/// `ref[dsp.op-m.tracker.envelope-norm]`). Returns `None` when $I = Q = 0$.
pub fn normalize_tracker_phasor_exponent_split(i_val: f32, q_val: f32) -> Option<(f32, f32, f32)> {
    let mut x = i_val;
    let mut y = q_val;
    let max_abs = x.abs().max(y.abs());
    if (max_abs.to_bits() & 0x7fff_ffff) == 0 {
        return None;
    }
    let mut exp_bits = (max_abs.to_bits() >> 23) & 0xff;
    let mut subnormal_restore = 1.0_f32;
    if exp_bits == 0 {
        let two_pow_24 = f32::from_bits((127 + 24) << 23);
        x *= two_pow_24;
        y *= two_pow_24;
        exp_bits = (x.abs().max(y.abs()).to_bits() >> 23) & 0xff;
        subnormal_restore = f32::from_bits((127 - 24) << 23);
    }
    let power = ((exp_bits as i32) - 127).min(126);
    let scale_down = f32::from_bits(((127 - power) as u32) << 23);
    let scale_up = f32::from_bits(((127 + power) as u32) << 23);
    let sx = x * scale_down;
    let sy = y * scale_down;
    let sq = sx * sx + sy * sy;
    let inv = rsqrt_nr2_exact_f32(sq);
    let mag = (sq * inv) * scale_up * subnormal_restore;
    let u_re = (sx * inv).clamp(-1.0, 1.0);
    let u_im = (sy * inv).clamp(-1.0, 1.0);
    Some((u_re, u_im, mag))
}

/// Evaluates the smooth half-angle pair `(v_cos, v_sin)` from a normalized phasor `(u_re, u_im)`
/// (`ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline]
pub fn eval_tracker_half_angle_pair(u_re: f32, u_im: f32) -> (f32, f32) {
    let re = u_re.clamp(-1.0, 1.0);
    let im = u_im.clamp(-1.0, 1.0);
    let h = 0.5 * (1.0 + re.abs());
    let inv = rsqrt_nr2_exact_f32(h);
    let major = (h * inv).clamp(0.0, 1.0);
    let minor = (im.abs() * (0.5 * inv)).clamp(0.0, 1.0);
    if re >= 0.0 {
        (major, minor)
    } else {
        (minor, major)
    }
}

/// Returns `true` if consecutive normalized phasors `prev` and `curr` cross the negative real axis
/// branch cut, holding on near-origin chords (`p . u < 0` and `|D| <= 2^-18`,
/// `ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline]
pub fn tracker_crosses_branch_cut(prev: (f32, f32), curr: (f32, f32)) -> bool {
    let (px, py) = prev;
    let (x, y) = curr;
    let det = (px as f64) * (y as f64) - (py as f64) * (x as f64);
    let dot = (px as f64) * (x as f64) + (py as f64) * (y as f64);
    let chord_tol = f64::from_bits((1023 - 18) << 52);
    if dot < 0.0 && det.abs() <= chord_tol {
        return false;
    }
    (py >= 0.0 && y < 0.0 && det > 0.0) || (py < 0.0 && y >= 0.0 && det < 0.0)
}

/// Complete `T1` Sub-Harmonic Tracker emitting `[w_cos, w_sin, w_square, w_quarter]` for `Lite`
/// feature bus slots `4..11` (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct T1SubHarmonicTracker {
    pub prev_valid: bool,
    pub prev_i_nonpositive: bool,
    pub prev_phasor: (f32, f32),
    pub q1: f32,
    pub q2: f32,
    pub s_sheet: f32,
}

impl Default for T1SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicTracker {
    pub const fn new() -> Self {
        Self {
            prev_valid: false,
            prev_i_nonpositive: false,
            prev_phasor: (1.0, 0.0),
            q1: 1.0,
            q2: 1.0,
            s_sheet: 1.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Steps the tracker from unnormalized analytic pair `(i_raw, q_raw)`, returning
    /// `[w_cos, w_sin, w_square, w_quarter]`.
    pub fn step_analytic(&mut self, i_raw: f32, q_raw: f32) -> [f32; 4] {
        let Some((u_re, u_im, env_hat)) = normalize_tracker_phasor_exponent_split(i_raw, q_raw)
        else {
            self.prev_valid = false;
            return [0.0; 4];
        };
        if self.prev_valid {
            if self.prev_i_nonpositive && i_raw > 0.0 {
                self.q1 = -self.q1;
                if self.q1 > 0.0 {
                    self.q2 = -self.q2;
                }
            }
            if tracker_crosses_branch_cut(self.prev_phasor, (u_re, u_im)) {
                self.s_sheet = -self.s_sheet;
            }
        }
        self.prev_valid = true;
        self.prev_i_nonpositive = i_raw <= 0.0;
        self.prev_phasor = (u_re, u_im);

        let (v_cos, v_sin) = eval_tracker_half_angle_pair(u_re, u_im);
        let sigma_pos = if u_im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        [
            env_hat * self.s_sheet * v_cos,
            env_hat * self.s_sheet * sigma_pos * v_sin,
            env_hat * self.q1,
            env_hat * self.q2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;

    #[test]
    fn test_subharmonic_tracker() {
        let _audio = crate::FtzDazGuard::new();
        let mut tracker = SubHarmonicTracker::default();
        tracker.reset();
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);

        // Silence squelch: R^2 < 1e-12 must return strictly 0.0 (zero DC offset on silence)
        assert_eq!(tracker.step(0.0, 0.0), 0.0);
        assert_eq!(tracker.step(1e-7, -1e-7), 0.0);

        // Rising zero-crossing flips prev_q (+1.0 -> -1.0 -> +1.0); branch cut at u_re < 0 flips sheet
        // Step 1: prev_u_im == 0.0 -> u_im > 0.0 with u_re < 0.0 flips sheet +1.0 -> -1.0
        let _ = tracker.step(-0.5, 0.2);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, -1.0);
        // Step 2: Cross negative real axis again (u_re < 0, u_im crosses from + to -) -> sheet flips -1.0 -> +1.0
        let _ = tracker.step(-0.5, -0.2);
        assert_eq!(tracker.sheet, 1.0);
        // Rising zero-crossing on x (-0.5 -> +0.5) -> prev_q flips to -1.0
        let _ = tracker.step(0.5, 0.2);
        assert_eq!(tracker.prev_q, -1.0);

        tracker.reset();
        assert_eq!(tracker.prev_x, 0.0);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);
    }

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    #[test]
    fn test_t1_dc_blocker_vectors_envelope_norm_and_toggle_sheet() {
        // 1. All 8 canonical T1-COEFFICIENT-VECTORS
        for &(rate, exp_pole, exp_gain) in &[
            (44_100.0_f64, 0x3F7F_D155u32, 0x3F7F_E8AAu32),
            (48_000.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (48_001.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (88_200.0, 0x3F7F_E8A9, 0x3F7F_F454),
            (96_000.0, 0x3F7F_EA8F, 0x3F7F_F548),
            (176_400.0, 0x3F7F_F454, 0x3F7F_FA2A),
            (192_000.0, 0x3F7F_F547, 0x3F7F_FAA4),
            (384_000.0, 0x3F7F_FAA3, 0x3F7F_FD52),
        ] {
            let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate).unwrap();
            assert_eq!(
                (p_bits, g_bits),
                (exp_pole, exp_gain),
                "T1 mismatch at rate={rate}: got ({p_bits:#010x}, {g_bits:#010x})"
            );
        }
        assert!(compute_t1_dc_blocker_coeffs_bits(100_000.0).is_none());

        // 2. Exponent-split envelope normalization across quiet (2^-125) and loud (100.0) amplitudes
        for &amp in &[(-125.0_f64).exp2() as f32, 1.0e-30, 1.0e-12, 0.1, 100.0] {
            for idx in 0..=64 {
                let phase = -core::f64::consts::PI + core::f64::consts::TAU * (idx as f64) / 64.0;
                let i_in = ((amp as f64) * phase.cos()) as f32;
                let q_in = ((amp as f64) * phase.sin()) as f32;
                let (ux, uy, env) = normalize_tracker_phasor_exponent_split(i_in, q_in).unwrap();
                let (v_cos, v_sin) = eval_tracker_half_angle_pair(ux, uy);
                let exp_cos = (amp as f64) * (0.5 * phase).cos().abs();
                let exp_sin = (amp as f64) * (0.5 * phase).sin().abs();
                assert!(((env * v_cos) as f64 - exp_cos).abs() <= (amp as f64) * 2.5e-5);
                assert!(((env * v_sin) as f64 - exp_sin).abs() <= (amp as f64) * 2.5e-5);
            }
        }
        assert!(normalize_tracker_phasor_exponent_split(0.0, 0.0).is_none());
        assert!(normalize_tracker_phasor_exponent_split(-0.0, 0.0).is_none());

        // 3. Branch cut crossing and T1SubHarmonicTracker silence / axis-zero handling
        assert!(tracker_crosses_branch_cut(
            (150.0_f32.to_radians().cos(), 150.0_f32.to_radians().sin()),
            (300.0_f32.to_radians().cos(), 300.0_f32.to_radians().sin())
        ));
        assert!(!tracker_crosses_branch_cut((1.0, 0.0), (-1.0, 1.0e-7)));

        let _guard = FtzDazGuard::new();
        let mut dc = T1DcBlocker::new_at_rate(48_000.0).unwrap();
        let mut trk = T1SubHarmonicTracker::new();
        assert_eq!(trk.step_analytic(0.0, 0.0), [0.0; 4]);
        assert!(!trk.prev_valid);

        let f_axis = trk.step_analytic(0.0, 1.0);
        assert!(trk.prev_valid && trk.prev_i_nonpositive && f_axis[2] > 0.0);
        let f_rise = trk.step_analytic(1.0, 1.0);
        assert!(f_rise[2] < 0.0); // q1 toggled +1 -> -1 on rising crossing

        let d_out = dc.step(0.5);
        assert!(d_out.is_finite());
    }
}

#[cfg(test)]
mod stage1_reference_tests {
    use crate::FtzDazGuard;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    use std::hint::black_box;

    const RATE: usize = 48_000;
    const TONE_PERIOD: usize = 48;
    const WARMUP: usize = RATE;
    const OBSERVATION: usize = 4_800;
    const AMPLITUDE: f64 = 1.0 / 4_194_304.0;
    const I_BRANCH_BITS: [u32; 6] = [
        0x3d91_bee6,
        0x3eec_b8e1,
        0x3f48_de0d,
        0x3f6d_91c9,
        0x3f7a_5939,
        0x3f7e_8e9d,
    ];

    #[derive(Default)]
    struct HilbertBranch {
        state: [[f32; 2]; 6],
    }

    impl HilbertBranch {
        fn step(&mut self, mut value: f32) -> f32 {
            for (state, bits) in self.state.iter_mut().zip(I_BRANCH_BITS) {
                let coefficient = f32::from_bits(bits);
                let output = coefficient * value + state[1];
                state[1] = state[0];
                state[0] = coefficient * output - value;
                value = output;
            }
            value
        }
    }

    struct Blocker {
        pole: f32,
        gain: f32,
        previous_input: f32,
        previous_output: f32,
    }

    impl Blocker {
        fn new() -> Self {
            let pole = 0.999_345_7_f32;
            let gain = ((1.0_f64 + f64::from(pole)) * 0.5) as f32;
            Self {
                pole,
                gain,
                previous_input: 0.0,
                previous_output: 0.0,
            }
        }

        fn step(&mut self, value: f32) -> f32 {
            let output =
                self.gain * (value - self.previous_input) + self.pole * self.previous_output;
            self.previous_input = value;
            self.previous_output = output;
            output
        }
    }

    #[derive(Default)]
    struct Projection {
        real: f64,
        imag: f64,
    }

    impl Projection {
        fn add(&mut self, sample: f32, index: usize) {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            self.real += f64::from(sample) * angle.cos();
            self.imag -= f64::from(sample) * angle.sin();
        }

        fn magnitude(&self) -> f64 {
            self.real.hypot(self.imag)
        }
    }

    #[test]
    fn preconditioned_tracker_preserves_a_quiet_tone_better_than_postconditioning_one_example() {
        let _guard = FtzDazGuard::new();
        let mut before_filter = Blocker::new();
        let mut before_branch = HilbertBranch::default();
        let mut after_branch = HilbertBranch::default();
        let mut after_filter = Blocker::new();
        assert_eq!(before_filter.pole.to_bits(), 0x3f7f_d51f);
        assert_eq!(before_filter.gain.to_bits(), 0x3f7f_ea90);
        let mut input_projection = Projection::default();
        let mut before_projection = Projection::default();
        let mut after_projection = Projection::default();

        for index in 0..WARMUP + OBSERVATION {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            let input = (0.5 + AMPLITUDE * angle.cos()) as f32;
            let before = before_branch.step(before_filter.step(input));
            let after = after_filter.step(after_branch.step(input));
            if index >= WARMUP {
                input_projection.add(input, index);
                before_projection.add(before, index);
                after_projection.add(after, index);
            }
        }

        let input = input_projection.magnitude();
        let before = before_projection.magnitude();
        let difference = (after_projection.real - before_projection.real)
            .hypot(after_projection.imag - before_projection.imag);
        assert!((before / input - 1.0).abs() < 0.01);
        assert!(difference / before > 0.2);
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn flushed_normalized_i_does_not_preserve_the_raw_detector_sign() {
        let _guard = FtzDazGuard::new();
        let squared_norm = black_box(1.0_f32);
        let mut inverse = f32::from_bits(0x5f37_59df - (squared_norm.to_bits() >> 1));
        for _ in 0..2 {
            let half_norm = black_box(0.5 * squared_norm);
            let product = black_box(black_box(half_norm * inverse) * inverse);
            inverse = black_box(inverse * black_box(1.5 - product));
        }

        let raw_i = black_box(f32::MIN_POSITIVE);
        let normalized_i = black_box(raw_i * inverse);
        let later_positive = black_box(1.0_f32);
        let earlier_negative = black_box(-1.0_f32);
        assert!(raw_i > 0.0);
        assert!(inverse < 1.0);
        assert_eq!(normalized_i.to_bits() & 0x7fff_ffff, 0);
        assert!(!(raw_i <= 0.0 && later_positive > 0.0));
        assert!(normalized_i <= 0.0 && later_positive > 0.0);
        assert!(earlier_negative <= 0.0 && raw_i > 0.0);
        assert!(!(earlier_negative <= 0.0 && normalized_i > 0.0));
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn equal_transposed_cells_can_require_different_history_consistent_retunes() {
        fn step(coefficient: f32, state: &mut [f32; 2], input: f32) -> f32 {
            let output = black_box(black_box(coefficient * input) + state[1]);
            state[1] = state[0];
            state[0] = black_box(black_box(coefficient * output) - input);
            output
        }

        let _guard = FtzDazGuard::new();
        let old = black_box(f32::from_bits(0x3d91_bee6));
        let new = black_box(f32::from_bits(0x3db0_73c5));
        let mut driven_state = [0.0; 2];
        let mut outputs = [0.0; 3];
        for (sample, output) in [1.0, 0.0, -old].into_iter().zip(&mut outputs) {
            *output = step(old, &mut driven_state, sample);
        }
        assert_eq!(driven_state.map(f32::to_bits), [0; 2]);
        assert_eq!(&outputs[1..], &[0.0, -1.0]);

        let mut empty_state = [0.0; 2];
        let change = black_box(new - old);
        driven_state[0] = black_box(driven_state[0] + black_box(change * outputs[2]));
        driven_state[1] = black_box(driven_state[1] + black_box(change * outputs[1]));
        assert_ne!(driven_state, empty_state);

        assert_eq!(step(new, &mut driven_state, 0.0), 0.0);
        assert_eq!(step(new, &mut empty_state, 0.0), 0.0);
        let driven_second = step(new, &mut driven_state, 0.0);
        let empty_second = step(new, &mut empty_state, 0.0);
        assert!((-0.015..-0.014).contains(&driven_second));
        assert_eq!(empty_second, 0.0);
    }
}
