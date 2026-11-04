//! Op-T: Universal Transport, Granular Windowing & All-Pass Diffusion Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_t_transport.md` (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.diffuser.recurrence]`).

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::container::align64_u64;
use crate::math::{
    flush_recursive_state_f32,
    fp16::{f32_to_fp16, fp16_to_f32},
    widen_f32_bits,
};
use crate::schema::DnamError;

/// Maximum delay line capacity for Op-T Lite buffer (`2,048` samples = `4,096` bytes fp16).
pub const OP_T_LITE_RING_CAPACITY: usize = 2048;
pub const OP_T_FULL_RING_CAPACITY: usize = 4096;
pub const OP_T_ULTRA_RING_CAPACITY: usize = 8192;

/// Co-prime Schroeder all-pass diffuser delays (`ref[dsp.op-t.diffuser.recurrence]`).
pub const SCHROEDER_COPRIME_DELAYS: [usize; 8] = [19, 31, 47, 73, 103, 139, 179, 223];

/// Stateless 4-tap Catmull-Rom cubic Hermite interpolation in Horner form (`ref[dsp.op-t.reader.catmull-rom]`).
///
/// Evaluates:
/// `c3 = -0.5*s[-1] + 1.5*s[0] - 1.5*s[1] + 0.5*s[2]`
/// `c2 = s[-1] - 2.5*s[0] + 2.0*s[1] - 0.5*s[2]`
/// `c1 = -0.5*s[-1] + 0.5*s[1]`
/// `c0 = s[0]`
/// `d = ((c3 * mu + c2) * mu + c1) * mu + c0`
#[inline(always)]
pub fn catmull_rom_hermite_cubic(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c1 = -0.5 * s_m1 + 0.5 * s_1;
    let c0 = s_0;
    ((c3 * mu + c2) * mu + c1) * mu + c0
}

/// Computes partition-of-unity grain weights for Hermite-Hann grains (Mode M = 2, Family 3).
///
/// Exact identity: `H(phi_A) + H(phi_B) == 1.000000000000`.
#[inline(always)]
pub fn hermite_hann_partition_of_unity(phi_a: f32) -> (f32, f32) {
    let u_a = 1.0 - (2.0 * phi_a - 1.0).abs();
    let u_a_clamped = u_a.clamp(0.0, 1.0);
    let h_a = u_a_clamped * u_a_clamped * (3.0 - 2.0 * u_a_clamped);
    let h_b = 1.0 - h_a;
    (h_a, h_b)
}

/// Computes partition-of-unity 3-phase raised-cosine weights (Mode M = 3, Family 6).
///
/// Exact identity: `W0 + W1 + W2 == 1.000000000000` with non-negative weights.
#[inline(always)]
pub fn raised_cosine_barberpole_3phase(phi: f32) -> [f32; 3] {
    const TWO_PI: f32 = core::f32::consts::TAU;
    const ONE_THIRD: f32 = 1.0 / 3.0;

    let phi0 = phi.fract();
    let phi1 = (phi + ONE_THIRD).fract();
    let phi2 = (phi + 2.0 * ONE_THIRD).fract();

    // Use fast minimax cosine: cos(2*pi*f)
    let c0 = fast_cos_unit(phi0 * TWO_PI);
    let c1 = fast_cos_unit(phi1 * TWO_PI);
    let _c2 = fast_cos_unit(phi2 * TWO_PI);

    let w0 = ONE_THIRD * (1.0 + c0).max(0.0);
    let w1 = ONE_THIRD * (1.0 + c1).max(0.0);
    let w2 = (1.0 - w0 - w1).max(0.0);
    [w0, w1, w2]
}

/// Approximate cosine for phase in `[0, 2*pi]` via minimax polynomial.
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn fast_cos_unit(mut x: f32) -> f32 {
    const PI: f32 = core::f32::consts::PI;
    const TWO_PI: f32 = core::f32::consts::TAU;
    while x < 0.0 {
        x += TWO_PI;
    }
    while x >= TWO_PI {
        x -= TWO_PI;
    }
    if x > PI {
        x = TWO_PI - x;
    }
    // x in [0, pi], map to z in [-pi/2, pi/2]
    let z = x - PI * 0.5;
    let z2 = z * z;
    // cos(x) = -sin(z) ~ -z * (1.0 - z2/6 + z4/120)
    -z * (0.99999995 - z2 * (0.16666650 - z2 * 0.00833290))
}

/// Single-delay line with power-of-2 ring buffer mask, Catmull-Rom reader,
/// partitioned Schroeder diffuser, and 128-sample switch crossfade.
#[derive(Clone, Debug)]
pub struct OpTDelayLine<const CAPACITY: usize> {
    pub ring: [u16; CAPACITY],
    pub write_ptr: usize,
    pub current_delay: f32,
    pub target_delay: f32,
    pub xfade_active: bool,
    pub xfade_idx: usize,
    pub xfade_len: usize,
    pub old_delay: f32,
    pub diff_state: [f32; 256],
    pub diff_idx: usize,
}

impl<const CAPACITY: usize> Default for OpTDelayLine<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> OpTDelayLine<CAPACITY> {
    pub const MASK: usize = CAPACITY - 1;
    pub const HALF_CAPACITY: usize = CAPACITY / 2;
    pub const HALF_MASK: usize = (CAPACITY / 2) - 1;

    pub const fn new() -> Self {
        Self {
            ring: [0; CAPACITY],
            write_ptr: 0,
            current_delay: 64.0,
            target_delay: 64.0,
            xfade_active: false,
            xfade_idx: 0,
            xfade_len: 128,
            old_delay: 64.0,
            diff_state: [0.0; 256],
            diff_idx: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ring.fill(0);
        self.write_ptr = 0;
        self.current_delay = self.target_delay;
        self.xfade_active = false;
        self.xfade_idx = 0;
        self.old_delay = self.target_delay;
        self.diff_state.fill(0.0);
        self.diff_idx = 0;
    }

    /// Sets delay target with dual-mode transition (`ref[dsp.op-t.transition.dual-mode]`).
    pub fn set_target_delay(&mut self, target: f32, is_switch_jump: bool) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(4).max(4) as f32;
        let clamped = target.clamp(4.0, max_delay);
        if is_switch_jump && (clamped - self.current_delay).abs() > 16.0 {
            self.old_delay = self.current_delay;
            self.target_delay = clamped;
            self.xfade_active = true;
            self.xfade_idx = 0;
        } else {
            self.target_delay = clamped;
        }
    }

    /// Reads fractional delay via 4-tap Catmull-Rom interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn read_delay(&self, delay: f32) -> f32 {
        let max_delay = CAPACITY.saturating_sub(4).max(2) as f32;
        let d_clamped = delay.clamp(2.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);

        let p_0 = (self.write_ptr + CAPACITY - d_int) & Self::MASK;
        let p_m1 = (p_0 + 1) & Self::MASK;
        let p_1 = (p_0 + CAPACITY - 1) & Self::MASK;
        let p_2 = (p_0 + CAPACITY - 2) & Self::MASK;

        let s_m1 = fp16_to_f32(self.ring[p_m1]);
        let s_0 = fp16_to_f32(self.ring[p_0]);
        let s_1 = fp16_to_f32(self.ring[p_1]);
        let s_2 = fp16_to_f32(self.ring[p_2]);

        catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
    }

    /// Advances the delay line by 1 audio sample: writes `input` and returns delayed output.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, input: f32) -> f32 {
        // Continuous slew rate limit: |dtau/dn| <= 0.25 (`ref[dsp.op-t.transition.dual-mode]`)
        if !self.xfade_active {
            let diff = self.target_delay - self.current_delay;
            if diff.abs() > 0.0001 {
                self.current_delay += diff.clamp(-0.25, 0.25);
            } else {
                self.current_delay = self.target_delay;
            }
        }

        let out = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let d_old = self.read_delay(self.old_delay);
            let d_new = self.read_delay(self.target_delay);
            let w = (self.xfade_idx as f32) / ((len - 1) as f32);
            let alpha_xf = 0.5 * (1.0 - fast_cos_unit(w * core::f32::consts::PI));
            let mixed = (1.0 - alpha_xf) * d_old + alpha_xf * d_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            mixed
        } else {
            self.read_delay(self.current_delay)
        };

        // Write to ring buffer
        self.ring[self.write_ptr] = f32_to_fp16(input);
        self.write_ptr = (self.write_ptr + 1) & Self::MASK;
        out
    }

    /// Reads fractional delay for a partitioned stereo pair (`ring[0..CAPACITY/2]` Left,
    /// `ring[CAPACITY/2..CAPACITY]` Right) via 4-tap Catmull-Rom interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn read_delay_stereo(&self, delay: f32) -> (f32, f32) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(4).max(2) as f32;
        let d_clamped = delay.clamp(2.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);
        let w_ptr = self.write_ptr & Self::HALF_MASK;

        let p_0 = (w_ptr + Self::HALF_CAPACITY - d_int) & Self::HALF_MASK;
        let p_m1 = (p_0 + 1) & Self::HALF_MASK;
        let p_1 = (p_0 + Self::HALF_CAPACITY - 1) & Self::HALF_MASK;
        let p_2 = (p_0 + Self::HALF_CAPACITY - 2) & Self::HALF_MASK;

        let l_m1 = fp16_to_f32(self.ring[p_m1]);
        let l_0 = fp16_to_f32(self.ring[p_0]);
        let l_1 = fp16_to_f32(self.ring[p_1]);
        let l_2 = fp16_to_f32(self.ring[p_2]);

        let r_off = Self::HALF_CAPACITY;
        let r_m1 = fp16_to_f32(self.ring[r_off + p_m1]);
        let r_0 = fp16_to_f32(self.ring[r_off + p_0]);
        let r_1 = fp16_to_f32(self.ring[r_off + p_1]);
        let r_2 = fp16_to_f32(self.ring[r_off + p_2]);

        (
            catmull_rom_hermite_cubic(l_m1, l_0, l_1, l_2, mu),
            catmull_rom_hermite_cubic(r_m1, r_0, r_1, r_2, mu),
        )
    }

    /// Advances the partitioned stereo delay line by 1 stereo audio frame (`in_l`, `in_r`)
    /// with zero L/R crosstalk and single-frame slew/crossfade progression (`ref[dsp.op-t.transition.dual-mode]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if !self.xfade_active {
            let diff = self.target_delay - self.current_delay;
            if diff.abs() > 0.0001 {
                self.current_delay += diff.clamp(-0.25, 0.25);
            } else {
                self.current_delay = self.target_delay;
            }
        }

        let (out_l, out_r) = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let (l_old, r_old) = self.read_delay_stereo(self.old_delay);
            let (l_new, r_new) = self.read_delay_stereo(self.target_delay);
            let w = (self.xfade_idx as f32) / ((len - 1) as f32);
            let alpha_xf = 0.5 * (1.0 - fast_cos_unit(w * core::f32::consts::PI));
            let one_minus_a = 1.0 - alpha_xf;
            let mix_l = one_minus_a * l_old + alpha_xf * l_new;
            let mix_r = one_minus_a * r_old + alpha_xf * r_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            (mix_l, mix_r)
        } else {
            self.read_delay_stereo(self.current_delay)
        };

        let w_ptr = self.write_ptr & Self::HALF_MASK;
        self.ring[w_ptr] = f32_to_fp16(in_l);
        self.ring[Self::HALF_CAPACITY + w_ptr] = f32_to_fp16(in_r);
        self.write_ptr = (w_ptr + 1) & Self::HALF_MASK;
        (out_l, out_r)
    }

    /// Steps the partitioned in-buffer Schroeder all-pass diffuser (`ref[dsp.op-t.diffuser.recurrence]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_diffuser(&mut self, input: f32, delay_samples: usize, a_diff: f32) -> f32 {
        let d = delay_samples.clamp(1, 255);
        let a = a_diff.clamp(-0.75, 0.75);

        let read_idx = (self.diff_idx + 256 - d) & 0xFF;
        let u_delayed = self.diff_state[read_idx];
        let v = input - a * u_delayed;
        let y = a * v + u_delayed;
        self.diff_state[self.diff_idx] = flush_recursive_state_f32(v);
        self.diff_idx = (self.diff_idx + 1) & 0xFF;
        y
    }
}

/// 1-pole ballistic envelope ducking VCA (Family 4).
#[derive(Clone, Copy, Debug)]
pub struct BallisticDuckingVca {
    pub env: f32,
    pub alpha_att: f32,
    pub alpha_rel: f32,
    pub threshold: f32,
    pub kappa: f32,
    pub min_gain: f32,
}

impl Default for BallisticDuckingVca {
    fn default() -> Self {
        Self::new()
    }
}

impl BallisticDuckingVca {
    pub const fn new() -> Self {
        Self {
            env: 0.0,
            alpha_att: 0.99,
            alpha_rel: 0.9995,
            threshold: 0.1,
            kappa: 2.0,
            min_gain: 0.1,
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.env = 0.0;
    }

    /// Applies ducking to one wet sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, sidechain: f32, wet: f32) -> f32 {
        let abs_sc = sidechain.abs();
        let alpha = if abs_sc > self.env {
            self.alpha_att
        } else {
            self.alpha_rel
        };
        let next_env = alpha * self.env + (1.0 - alpha) * abs_sc;
        self.env = flush_recursive_state_f32(next_env);
        let excess = (next_env - self.threshold).max(0.0);
        let g = (1.0 - self.kappa * excess).clamp(self.min_gain, 1.0);
        g * wet
    }

    /// Evaluates a stereo-linked ballistic ducking step once per stereo frame (`1x fs`),
    /// preserving the stereo image `(wet_l, wet_r)` with identical L/R ducking gain.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        sidechain_l: f32,
        sidechain_r: f32,
        wet_l: f32,
        wet_r: f32,
    ) -> (f32, f32) {
        let abs_sc = 0.5 * (sidechain_l.abs() + sidechain_r.abs());
        let alpha = if abs_sc > self.env {
            self.alpha_att
        } else {
            self.alpha_rel
        };
        let next_env = alpha * self.env + (1.0 - alpha) * abs_sc;
        self.env = flush_recursive_state_f32(next_env);
        let excess = (next_env - self.threshold).max(0.0);
        let g = (1.0 - self.kappa * excess).clamp(self.min_gain, 1.0);
        (g * wet_l, g * wet_r)
    }
}

// ============================================================================
// Normative v2.0 Op-T Ring Capacity, Branchless Wrap, Q0.32 Pointer,
// Max-Tree Feedback Guard, M=2/M=3 Windows & Partitioned Schroeder Diffusers
// (`ref[dsp.op-t.*]`, `ref[dsp.zoh.op-t-delays]`)
// ============================================================================

/// Computes the exact `u128` ceiling capacity $D_{\text{eff}} = \lceil D_{\text{ref}} f_{\text{eff}} / 48000 \rceil$,
/// logical sample length $L = D_{\text{eff}} + 2$, and 64-byte aligned allocation size $B = 64 \lceil a L / 64 \rceil$
/// (`bytes_per_sample` $a \in \{2, 4\}$, `ref[dsp.op-t.ring.capacity]`).
pub fn op_t_fractional_history_capacity(
    reference_delay: u64,
    effective_rate_hz: f64,
    bytes_per_sample: u64,
) -> Option<(u64, u64)> {
    if reference_delay < 4
        || !effective_rate_hz.is_finite()
        || !(44_100.0..=384_000.0).contains(&effective_rate_hz)
        || !matches!(bytes_per_sample, 2 | 4)
    {
        return None;
    }
    let bits = effective_rate_hz.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let significand = (1u128 << 52) | ((bits & ((1u64 << 52) - 1)) as u128);
    let shift = (52 - exponent) as u32;
    let num = (reference_delay as u128).checked_mul(significand)?;
    let den = 48_000u128 << shift;
    let d_eff = num.div_ceil(den);
    let logical_samples = u64::try_from(d_eff.checked_add(2)?).ok()?;
    let raw_bytes = logical_samples.checked_mul(bytes_per_sample)?;
    let aligned_bytes = align64_u64(raw_bytes)?;
    Some((logical_samples, aligned_bytes))
}

/// Cold-path host memory-ceiling admission check (`ref[dsp.op-t.ring.admission]`).
/// Accounts for both live and transition allocations before activation and fails fast on exhaustion.
pub fn admit_op_t_memory_budget(
    current_live_bytes: u64,
    requested_new_bytes: u64,
    host_ceiling_bytes: u64,
) -> Result<u64, DnamError> {
    let total = current_live_bytes
        .checked_add(requested_new_bytes)
        .ok_or_else(|| {
            DnamError::SchemaValidationError("Op-T memory budget overflow".to_string())
        })?;
    if total > host_ceiling_bytes {
        return Err(DnamError::SchemaValidationError(
            "Op-T allocation exceeds host memory ceiling".to_string(),
        ));
    }
    Ok(total)
}

/// Branchless single-wrap index for arbitrary ring length $L > 0$ and raw index $i \in [-L, L)$:
/// $r = i + (L \mathbin{\mathrm{AND}} -[i < 0])$ (`ref[dsp.op-t.ring.branchless-wrap]`).
#[inline]
pub const fn op_t_branchless_wrap_index(raw_index: isize, length: usize) -> Option<usize> {
    let len_i = length as isize;
    if length == 0 || raw_index < -len_i || raw_index >= len_i {
        return None;
    }
    let mask = if raw_index < 0 { -1isize } else { 0isize };
    Some((raw_index + (len_i & mask)) as usize)
}

/// Branchless write-pointer increment for arbitrary ring length $L > 0$ and $0 \le w < L$:
/// $w' = (w + 1) - (L \mathbin{\mathrm{AND}} -[w + 1 == L])$ (`ref[dsp.op-t.ring.branchless-wrap]`).
#[inline]
pub const fn op_t_branchless_next_write(write_index: usize, length: usize) -> Option<usize> {
    if length == 0 || write_index >= length {
        return None;
    }
    let next = write_index + 1;
    let mask = if next == length { usize::MAX } else { 0usize };
    Some(next - (length & mask))
}

/// Converts an integer sample count `integer` and unsigned `Q0.32` fraction `frac_q32` into
/// `(eff_integer, mu_f32)` with $\mu \in [0.0, 1.0)$, normalizing a `1.0f32` rounding carry to
/// `(integer + 1, 0.0f32)` (`ref[dsp.op-t.ring.q0-32-pointer]`).
#[inline]
pub fn q32_decode_reader_position(integer: u64, frac_q32: u32) -> (u64, f32) {
    let scale = f32::from_bits(0x2f80_0000); // 2^-32
    let rounded = (frac_q32 as f32) * scale;
    if rounded >= 1.0 {
        (integer + 1, 0.0)
    } else {
        (integer, rounded)
    }
}

/// Converts a `(integer, frac_q32)` target inside the exact `Q32.32` interval `[low, high]`
/// (`low = (low_int, low_q32)`, `high = (high_int, high_q32)`) into a safe `(eff_int, mu_f32)`
/// reader position, stepping inward by `±1` `f32` ULP if `RN32` crosses `low` or `high`
/// (`ref[dsp.op-t.ring.q0-32-pointer]`).
pub fn bounded_q32_reader_position(
    integer: u64,
    frac_q32: u32,
    low: (u64, u32),
    high: (u64, u32),
) -> Option<(u64, f32)> {
    let to_q32_u128 = |(i, f): (u64, u32)| -> u128 { ((i as u128) << 32) | (f as u128) };
    let target_q = to_q32_u128((integer, frac_q32));
    let low_q = to_q32_u128(low);
    let high_q = to_q32_u128(high);
    if !(low_q <= target_q && target_q <= high_q) {
        return None;
    }

    let pos_to_q32_u128 = |int_part: u64, mu: f32| -> Option<u128> {
        let mu64 = widen_f32_bits(mu.to_bits())?;
        let scaled = (mu64 * ((1u64 << 32) as f64)) as u128;
        Some(((int_part as u128) << 32) + scaled)
    };

    let (eff_int, rounded) = q32_decode_reader_position(integer, frac_q32);
    let pos_q = pos_to_q32_u128(eff_int, rounded)?;
    if low_q <= pos_q && pos_q <= high_q {
        return Some((eff_int, rounded));
    }

    // If carry to (integer + 1, 0.0) crossed `high`, step down from 1.0 to 1.0 - 2^-24 (`0x3f7f_ffff`)
    let (adj_int, adj_mu) = if pos_q > high_q {
        if rounded == 0.0 && eff_int > integer {
            (integer, f32::from_bits(0x3f7f_ffff))
        } else {
            (eff_int, f32::from_bits(rounded.to_bits().saturating_sub(1)))
        }
    } else {
        let next_bits = rounded.to_bits() + 1;
        let next_f = f32::from_bits(next_bits);
        if next_f >= 1.0 {
            (eff_int + 1, 0.0)
        } else {
            (eff_int, next_f)
        }
    };
    let adj_q = pos_to_q32_u128(adj_int, adj_mu)?;
    if low_q <= adj_q && adj_q <= high_q {
        Some((adj_int, adj_mu))
    } else {
        None
    }
}

/// Directed `fp16` decay multiplier `0.999f32` (`0x3F7FBE76`, `ref[dsp.op-t.guard.fp16-decay]`).
pub const FP16_DECAY_GAIN_BITS: u32 = 0x3F7F_BE76;

/// Clamps `value` against the directed `fp16` decay ceiling `0.999f32 * max_bound` and rounds the
/// `fp16` magnitude toward zero so every positive `fp16` history value strictly contracts under
/// unforced feedback (`ref[dsp.op-t.guard.fp16-decay]`).
#[inline]
pub fn directed_fp16_decay_write_bits(value: f32, max_bound: f32) -> u16 {
    let gain = f32::from_bits(FP16_DECAY_GAIN_BITS);
    let ceiling = gain * max_bound.abs();
    let clamped = value.clamp(-ceiling, ceiling);
    let nearest_bits = f32_to_fp16(clamped);
    let nearest_val = fp16_to_f32(nearest_bits);
    if nearest_val.abs() > clamped.abs() && (nearest_bits & 0x7fff) > 0 {
        nearest_bits - 1
    } else {
        nearest_bits
    }
}

/// Grouped block-maxima + binary max-tree feedback guard over $M$ delay lines (`ref[dsp.op-t.guard.bounded-history]`,
/// `ref[dsp.op-t.guard.max-tree]`, `ref[dsp.op-t.guard.bibo-decay]`).
#[derive(Clone, Debug)]
pub struct OpTBlockMaxTreeGuard {
    pub block_size: usize,
    pub histories: Vec<Vec<f64>>,
    pub positions: Vec<usize>,
    pub fresh_max: Vec<f64>,
    pub block_maxima: Vec<Vec<f64>>,
}

impl OpTBlockMaxTreeGuard {
    pub fn new(initial_histories: Vec<Vec<f64>>, block_size: usize) -> Self {
        assert!(block_size > 0 && !initial_histories.is_empty());
        let n_lines = initial_histories.len();
        let mut block_maxima = Vec::with_capacity(n_lines);
        for h in &initial_histories {
            assert!(!h.is_empty());
            let mut bm = Vec::new();
            for chunk in h.chunks(block_size) {
                let m = chunk.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                bm.push(m);
            }
            block_maxima.push(bm);
        }
        Self {
            block_size,
            histories: initial_histories,
            positions: vec![0; n_lines],
            fresh_max: vec![0.0; n_lines],
            block_maxima,
        }
    }

    /// Returns the upper bound $M_n$ across all active blocks and completed block maxima.
    pub fn bound(&self) -> f64 {
        let mut m = 0.0_f64;
        for &f in &self.fresh_max {
            m = m.max(f);
        }
        for bm in &self.block_maxima {
            for &b in bm {
                m = m.max(b);
            }
        }
        m
    }

    /// Writes `value` to `line` at its current write position and updates its block maximum.
    pub fn write(&mut self, line: usize, value: f64) {
        let pos = self.positions[line];
        let len = self.histories[line].len();
        self.histories[line][pos] = value;
        self.fresh_max[line] = self.fresh_max[line].max(value.abs());
        let end_of_block = (pos + 1).is_multiple_of(self.block_size) || (pos + 1) == len;
        if end_of_block {
            self.block_maxima[line][pos / self.block_size] = self.fresh_max[line];
            self.fresh_max[line] = 0.0;
        }
        self.positions[line] = op_t_branchless_next_write(pos, len).unwrap();
    }
}

/// Evaluates the 3-phase cardinal quadratic B-spline barberpole window ($M = 3$, `0 FDIV`, `0 libm`,
/// $\sum_{m=0}^2 B_2(3 \cdot \mathrm{frac}(\phi + m/3)) \equiv 1.0$, `ref[dsp.op-t.window.m3-barberpole]`,
/// `ref[dsp.op-t.window.partition-of-unity]`).
#[inline]
pub fn bspline_barberpole_3phase(phi: f32) -> [f32; 3] {
    let p = phi - libm_free_floor_01(phi);
    let t0 = 3.0 * p;
    let k = (t0 as usize).min(2);
    let u = (t0 - (k as f32)).clamp(0.0, 1.0);
    // On any unit sub-interval u in [0, 1), the 3 pieces of B2 at u, u+1, u+2 are:
    // w_a = 0.5 * u^2
    // w_b = -u^2 + u + 0.5
    // w_c = 0.5 * (1 - u)^2 = 0.5 - u + 0.5 * u^2
    // Notice w_a + w_b + w_c = (0.5 - 1 + 0.5)*u^2 + (1 - 1)*u + (0.5 + 0.5) == 1.0 identically!
    let u2 = u * u;
    let w_a = 0.5 * u2;
    let w_c = 0.5 * (1.0 - u) * (1.0 - u);
    let w_b = 1.0 - w_a - w_c;
    match k {
        0 => [w_a, w_b, w_c],
        1 => [w_b, w_c, w_a],
        _ => [w_c, w_a, w_b],
    }
}

#[inline]
fn libm_free_floor_01(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if f > x { f - 1.0 } else { f }
}

/// Evaluates the two grain phases $(\phi_A, \phi_B)$, Hermite-Hann weights $(H_A, H_B)$, and read
/// delays $(\tau_A, \tau_B)$ for Mode $M = 2$ granular pitch shifting at sample `frame` with even
/// integer grain duration `grain_duration`, center delay `center_delay`, and time-warp `warp`
/// (`ref[dsp.op-t.window.m2-pitch]`).
pub fn eval_m2_pitch_grain_delays(
    frame: u64,
    grain_duration: u64,
    center_delay: f64,
    warp: f64,
) -> Option<([f64; 2], [f64; 2], [f64; 2])> {
    if grain_duration == 0 || !grain_duration.is_multiple_of(2) {
        return None;
    }
    let g_f64 = grain_duration as f64;
    let phi_a = ((frame % grain_duration) as f64) / g_f64;
    let phi_b = if phi_a + 0.5 >= 1.0 {
        phi_a - 0.5
    } else {
        phi_a + 0.5
    };
    let u_a = 1.0 - (2.0 * phi_a - 1.0).abs();
    let h_a = u_a * u_a * (3.0 - 2.0 * u_a);
    let h_b = 1.0 - h_a;
    let tau_a = (center_delay + warp * g_f64 * (phi_a - 0.5)).max(4.0);
    let tau_b = (center_delay + warp * g_f64 * (phi_b - 0.5)).max(4.0);
    Some(([phi_a, phi_b], [h_a, h_b], [tau_a, tau_b]))
}

/// Converts a reference `48 kHz` Schroeder all-pass delay length to `effective_rate_hz` via
/// $\max(1, \mathrm{round}(D_{\text{ap}} \cdot f_{s,\text{eff}} / 48000))$ without capping at `255`
/// (`ref[dsp.op-t.diffuser.partition]`, `ref[dsp.op-t.diffuser.recurrence]`, `ref[dsp.zoh.op-t-delays]`).
#[inline]
pub const fn convert_schroeder_delay_for_rate(ref_delay: usize, effective_rate_hz: u32) -> usize {
    let num = (ref_delay as u64) * (effective_rate_hz as u64);
    let q = num / 48_000;
    let r = num % 48_000;
    let rounded = if 2 * r >= 48_000 { q + 1 } else { q };
    if rounded < 1 { 1 } else { rounded as usize }
}

/// Partitioned 8-line Schroeder all-pass diffuser bank with dedicated per-line segments and
/// arbitrary-length branchless wrap (no 255-sample cap, `ref[dsp.op-t.diffuser.partition]`,
/// `ref[dsp.op-t.diffuser.recurrence]`).
#[derive(Clone, Debug)]
pub struct PartitionedSchroederDiffuserBank {
    pub delays: [usize; 8],
    pub write_indices: [usize; 8],
    pub segments: [Vec<f32>; 8],
}

impl PartitionedSchroederDiffuserBank {
    /// Creates an 8-line Schroeder all-pass bank scaled for `effective_rate_hz`.
    pub fn new_for_rate(effective_rate_hz: u32) -> Self {
        let delays: [usize; 8] = core::array::from_fn(|i| {
            convert_schroeder_delay_for_rate(SCHROEDER_COPRIME_DELAYS[i], effective_rate_hz)
        });
        let segments: [Vec<f32>; 8] = core::array::from_fn(|i| vec![0.0_f32; delays[i]]);
        Self {
            delays,
            write_indices: [0; 8],
            segments,
        }
    }

    /// Steps line `line_idx` (`0..8`) by 1 sample with clamped all-pass coefficient `a_diff in [-0.75, 0.75]`.
    #[inline]
    pub fn step_line(&mut self, line_idx: usize, x_in: f32, a_diff: f32) -> f32 {
        let a = a_diff.clamp(-0.75, 0.75);
        let d = self.delays[line_idx];
        let w = self.write_indices[line_idx];
        let u_delayed = self.segments[line_idx][w];
        let v = x_in - a * u_delayed;
        let y = a * v + u_delayed;
        self.segments[line_idx][w] = flush_recursive_state_f32(v);
        self.write_indices[line_idx] = op_t_branchless_next_write(w, d).unwrap();
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_interpolation_and_windowing() {
        let v = catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 0.5);
        assert!((v - 2.5).abs() < 1e-4);

        for phi in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (ha, hb) = hermite_hann_partition_of_unity(phi);
            assert!((ha + hb - 1.0).abs() < 1e-6);
        }

        for phi in [0.1, 0.33, 0.5, 0.8] {
            let [w0, w1, w2] = raised_cosine_barberpole_3phase(phi);
            assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
        }

        for x in [
            -core::f32::consts::PI,
            0.0,
            1.5,
            core::f32::consts::PI,
            core::f32::consts::TAU,
            10.0,
        ] {
            let c = fast_cos_unit(x);
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_op_t_delay_line_and_diffuser() {
        let _audio = crate::FtzDazGuard::new();
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl.reset();

        // Feed some samples
        for i in 0..100 {
            let _ = dl.step(i as f32);
        }

        // Test smooth slew
        dl.set_target_delay(80.0, false);
        for _ in 0..200 {
            let _ = dl.step(0.5);
        }
        assert!((dl.current_delay - 80.0).abs() < 0.1);

        // Test switch jump crossfade
        dl.set_target_delay(150.0, true);
        assert!(dl.xfade_active);
        for _ in 0..150 {
            let _ = dl.step(0.5);
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 150.0);

        // Test diffuser
        let d_out = dl.step_diffuser(0.8, 31, 0.5);
        assert!(d_out.is_finite());

        dl.reset();
        assert_eq!(dl.write_ptr, 0);
    }

    #[test]
    fn test_ballistic_ducking_vca() {
        let _audio = crate::FtzDazGuard::new();
        let mut vca = BallisticDuckingVca::default();
        // Low sidechain
        let out_low = vca.step(0.01, 1.0);
        assert!((out_low - 1.0).abs() < 0.05);

        // High sidechain: ducking active
        for _ in 0..50 {
            let _ = vca.step(0.9, 1.0);
        }
        let out_ducked = vca.step(0.9, 1.0);
        assert!(out_ducked < 0.9);

        // Stereo-linked ducking preserves L/R ratio under asymmetric sidechain
        let mut vca_s = BallisticDuckingVca::default();
        let (dl, dr) = vca_s.step_stereo(1.0, 0.0, 0.8, 0.4);
        assert!((dl - 2.0 * dr).abs() < 1e-6);
    }

    #[test]
    fn test_catmull_rom_exact_dc_gain_and_diffuser_energy() {
        let _audio = crate::FtzDazGuard::new();
        // Theorem 4.2: Exact 0.000000 dB DC Gain Preservation for all mu in [0, 1]
        for step in 0..=100 {
            let mu = (step as f32) / 100.0;
            let val = catmull_rom_hermite_cubic(1.0, 1.0, 1.0, 1.0, mu);
            assert!(
                (val - 1.0).abs() < 1e-6,
                "Catmull-Rom DC gain diverged at mu={}: val={}",
                mu,
                val
            );
        }

        // ref[dsp.op-t.diffuser.recurrence]: Schroeder all-pass energy conservation (|H_ap|^2 == 1.0)
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let a_diff = 0.65_f32;
        let d = 19_usize;
        let mut energy_in = 0.0_f64;
        let mut energy_out = 0.0_f64;

        // Run impulse through diffuser
        for n in 0..1000 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let y = dl.step_diffuser(x, d, a_diff);
            energy_in += (x as f64) * (x as f64);
            energy_out += (y as f64) * (y as f64);
        }
        assert!(
            (energy_in - energy_out).abs() < 1e-4,
            "Diffuser energy unconserved: in={}, out={}",
            energy_in,
            energy_out
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_catmull_rom_linear_reproduction_and_windows(
            a in -5.0f32..5.0f32,
            b in -5.0f32..5.0f32,
            mu in 0.0f32..1.0f32,
        ) {
            let ym1 = a - b;
            let y0 = a;
            let y1 = a + b;
            let y2 = a + 2.0 * b;
            let interp = catmull_rom_hermite_cubic(ym1, y0, y1, y2, mu);
            let exact = a + b * mu;
            prop_assert!((interp - exact).abs() < 1e-4);

            let (ha, hb) = hermite_hann_partition_of_unity(mu);
            prop_assert!((ha + hb - 1.0).abs() < 1e-6);
            prop_assert!(ha >= -1e-6 && hb >= -1e-6);

            let [w0, w1, w2] = raised_cosine_barberpole_3phase(mu);
            prop_assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
            prop_assert!(w0 >= -1e-5 && w1 >= -1e-5 && w2 >= -1e-5);
        }

        #[test]
        fn prop_schroeder_diffuser_parseval_energy(
            a_diff in -0.7f32..0.7f32,
            d in 4usize..32usize,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
            let mut e_in = 0.0_f64;
            let mut e_out = 0.0_f64;
            for n in 0..512 {
                let x = if n < 8 { ((n as f32) * 0.7 + 0.2).sin() } else { 0.0 };
                let y = dl.step_diffuser(x, d, a_diff);
                e_in += (x as f64) * (x as f64);
                e_out += (y as f64) * (y as f64);
            }
            prop_assert!((e_in - e_out).abs() < 5e-3);
        }
    }
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

    // Specification Verification Suite: Phase 5 — Op-T Universal Transport, Granular Windowing & All-Pass Diffusion.
    //
    // Verifies all 20 rules in `docs/specs/dsp/op_t_transport.md` plus `dsp.zoh.op-t-delays`:
    // - `verify[dsp.op-t.ring.declaration]`
    // - `verify[dsp.op-t.ring.capacity]`
    // - `verify[dsp.op-t.ring.admission]`
    // - `verify[dsp.op-t.ring.branchless-wrap]`
    // - `verify[dsp.op-t.ring.q0-32-pointer]`
    // - `verify[dsp.op-t.ring.fp16-saturation]`
    // - `verify[dsp.op-t.reader.catmull-rom]`
    // - `verify[dsp.op-t.guard.bounded-history]`
    // - `verify[dsp.op-t.guard.max-tree]`
    // - `verify[dsp.op-t.guard.bibo-decay]`
    // - `verify[dsp.op-t.guard.fp16-decay]`
    // - `verify[dsp.op-t.transition.dual-mode]`
    // - `verify[dsp.op-t.window.partition-of-unity]`
    // - `verify[dsp.op-t.window.m1-delay]`
    // - `verify[dsp.op-t.window.m2-pitch]`
    // - `verify[dsp.op-t.window.m3-barberpole]`
    // - `verify[dsp.op-t.diffuser.partition]`
    // - `verify[dsp.op-t.diffuser.recurrence]`
    // - `verify[dsp.op-t.ducking.vca]`
    // - `verify[dsp.op-t.zero-fdiv]`
    // - `verify[dsp.zoh.op-t-delays]`

    use crate::FtzDazGuard;
    use crate::container::ZohRescaledParams;
    use crate::math::{f32_to_fp16_with_telemetry, fp16_to_f32, saturation_flags};
    use crate::operators::{
        BallisticDuckingVca, OP_T_LITE_RING_CAPACITY, OpTBlockMaxTreeGuard, OpTDelayLine,
        PartitionedSchroederDiffuserBank, admit_op_t_memory_budget, bounded_q32_reader_position,
        bspline_barberpole_3phase, catmull_rom_hermite_cubic, convert_schroeder_delay_for_rate,
        directed_fp16_decay_write_bits, eval_m2_pitch_grain_delays,
        hermite_hann_partition_of_unity, op_t_branchless_next_write, op_t_branchless_wrap_index,
        op_t_fractional_history_capacity, q32_decode_reader_position,
        raised_cosine_barberpole_3phase,
    };

    /// verify[dsp.op-t.ring.declaration]
    /// verify[dsp.op-t.ring.capacity]
    /// verify[dsp.op-t.ring.admission]
    /// verify[dsp.op-t.ring.branchless-wrap]
    /// verify[dsp.op-t.ring.q0-32-pointer]
    /// verify[dsp.op-t.ring.fp16-saturation]
    /// Verifies 9-second (432,000-sample) `u128` capacity sizing across 44.1..384 kHz, cold-path
    /// memory-ceiling admission, arbitrary-length branchless wrap, `Q0.32` reader pointer inward
    /// bounds, and `fp16` saturation telemetry.
    #[test]
    fn test_op_t_ring_capacity_admission_branchless_wrap_q32_and_fp16_saturation() {
        // 1. 9-second (432,000 reference samples) history sizing across canonical rates
        for &(sr, exp_logical, exp_bytes) in &[
            (44_100.0_f64, 396_902u64, 793_856u64),
            (48_000.0, 432_002, 864_064),
            (96_000.0, 864_002, 1_728_064),
            (384_000.0, 3_456_002, 6_912_064),
        ] {
            let (logical, bytes) = op_t_fractional_history_capacity(432_000, sr, 2).unwrap();
            assert_eq!((logical, bytes), (exp_logical, exp_bytes));
            assert_eq!(bytes % 64, 0);
            assert!(bytes - 2 * logical < 64);
        }
        assert_eq!(
            op_t_fractional_history_capacity(30, 48_000.0, 2),
            Some((32, 64))
        );
        assert!(op_t_fractional_history_capacity(3, 48_000.0, 2).is_none());

        // 2. Transition admission accounts for both live and new allocations
        let (_, old_bytes) = op_t_fractional_history_capacity(432_000, 48_000.0, 2).unwrap();
        let (_, new_bytes) = op_t_fractional_history_capacity(432_000, 96_000.0, 2).unwrap();
        assert!(admit_op_t_memory_budget(0, new_bytes, 2_000_000).is_ok());
        assert!(admit_op_t_memory_budget(old_bytes, new_bytes, 2_000_000).is_err());

        // 3. Branchless wrap matches modulo for arbitrary non-power-of-two lengths & Catmull-Rom taps
        for len in [8usize, 13, 37, 79, 511] {
            for w in 0..len {
                assert_eq!(op_t_branchless_next_write(w, len), Some((w + 1) % len));
                for d in [4usize, len - 2] {
                    for offset in [1isize, 0, -1, -2] {
                        let raw = (w as isize) - (d as isize) + offset;
                        let wrapped = op_t_branchless_wrap_index(raw, len).unwrap();
                        assert_eq!(wrapped, raw.rem_euclid(len as isize) as usize);
                    }
                }
            }
        }

        // 4. Q0.32 pointer carry & inward reader bounds
        assert_eq!(
            q32_decode_reader_position(432_000, u32::MAX),
            (432_001, 0.0)
        );
        let half_word = 1u32 << 31;
        let up = bounded_q32_reader_position(4, half_word + 1, (4, half_word + 1), (5, 0)).unwrap();
        assert!(up.1 >= 0.5);
        let down =
            bounded_q32_reader_position(4, half_word - 1, (4, 0), (4, half_word - 1)).unwrap();
        assert!(down.1 <= 0.5);
        // Carry at u32::MAX steps inward when upper bound is (4, u32::MAX)
        let carry_clamped =
            bounded_q32_reader_position(4, u32::MAX, (4, 0), (4, u32::MAX)).unwrap();
        assert_eq!(carry_clamped, (4, f32::from_bits(0x3f7f_ffff)));

        // 5. fp16 write saturation telemetry
        let mut flags = 0u32;
        assert_eq!(f32_to_fp16_with_telemetry(100_000.0, &mut flags), 0x7bff);
        assert_eq!(flags, saturation_flags::FP16_SATURATED);
    }

    /// verify[dsp.op-t.reader.catmull-rom]
    /// verify[dsp.op-t.guard.bounded-history]
    /// verify[dsp.op-t.guard.max-tree]
    /// verify[dsp.op-t.guard.bibo-decay]
    /// verify[dsp.op-t.guard.fp16-decay]
    /// Verifies Catmull-Rom endpoint/passivity/1.25x L1 peak gain, grouped max-tree feedback guard
    /// BIBO bound & unforced geometric decay, and directed `fp16` decay contraction across all `1..0x7C00`.
    #[test]
    fn test_catmull_rom_reader_and_bounded_history_max_tree_feedback_guard() {
        // 1. Catmull-Rom endpoint exactness and L1 peak gain == 1.25 at mu = 0.5 on [-1, 1, 1, -1]
        assert_eq!(catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 0.0), 2.0);
        assert_eq!(catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 1.0), 3.0);
        assert_eq!(catmull_rom_hermite_cubic(-1.0, 1.0, 1.0, -1.0, 0.5), 1.25);

        // 2. Grouped block-maxima guard: driven BIBO bound and unforced geometric decay
        let init_h = vec![
            vec![0.5, -0.8, 0.3, -0.4, 0.9, -0.2, 0.1, -0.7],
            vec![-0.6, 0.4, -0.9, 0.2, -0.3, 0.8, -0.5, 0.1],
            vec![0.2, -0.3, 0.7, -0.6, 0.4, -0.5, 0.8, -0.1],
            vec![-0.4, 0.6, -0.2, 0.5, -0.7, 0.3, -0.9, 0.2],
        ];
        let mut guard = OpTBlockMaxTreeGuard::new(init_h, 4);
        let m0 = guard.bound();
        assert!((m0 - 0.9).abs() < 1e-12);
        let g_max = 0.9_f64;

        // Unforced decay: after 8 frames (1 full history replacement), bound <= g_max * m0
        for frame in 0..8 {
            let m_n = guard.bound();
            for line in 0..4 {
                let h = &guard.histories[line];
                let adv_read = -0.0625 * h[frame % 8]
                    + 0.5625 * h[(frame + 1) % 8]
                    + 0.5625 * h[(frame + 2) % 8]
                    - 0.0625 * h[(frame + 3) % 8];
                let clamped = adv_read.clamp(-m_n, m_n);
                guard.write(line, g_max * clamped);
            }
        }
        assert!(guard.bound() <= g_max * m0 + 1e-12);

        // 3. Directed fp16 decay strictly contracts EVERY positive finite fp16 value (bits 1..0x7C00)
        for bits in 1u16..0x7c00 {
            let max_val = fp16_to_f32(bits);
            let decayed_bits = directed_fp16_decay_write_bits(max_val, max_val);
            let decayed_val = fp16_to_f32(decayed_bits);
            assert!(
                decayed_val >= 0.0 && decayed_val < max_val,
                "fp16 bits {bits:#06x} did not strictly contract: {decayed_val} >= {max_val}"
            );
        }
    }

    /// verify[dsp.op-t.transition.dual-mode]
    /// verify[dsp.op-t.window.partition-of-unity]
    /// verify[dsp.op-t.window.m1-delay]
    /// verify[dsp.op-t.window.m2-pitch]
    /// verify[dsp.op-t.window.m3-barberpole]
    /// verify[dsp.op-t.zero-fdiv]
    /// Verifies dual-mode slew vs switch crossfade, `M = 1, 2, 3` partition of unity (including
    /// division-free `B_2` quadratic B-spline barberpole), and `M = 2` grain delay bounds (`7,204` / `14,404`).
    #[test]
    fn test_op_t_dual_mode_transitions_and_m1_m2_m3_partition_of_unity() {
        let _guard = FtzDazGuard::new();

        // 1. M = 2 Hermite-Hann & M = 3 B-Spline / Raised-Cosine exact partition of unity
        for step in 0..=300 {
            let phi = (step as f32) / 300.0;
            let (ha, hb) = hermite_hann_partition_of_unity(phi);
            assert!((ha + hb - 1.0).abs() < 1e-6);

            let [b0, b1, b2] = bspline_barberpole_3phase(phi);
            assert!(b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0);
            assert!((b0 + b1 + b2 - 1.0).abs() < 1e-6);

            let [r0, r1, r2] = raised_cosine_barberpole_3phase(phi);
            assert!(r0 >= -1e-6 && r1 >= -1e-6 && r2 >= -1e-6);
            assert!((r0 + r1 + r2 - 1.0).abs() < 1e-5);
        }

        // 2. M = 2 centered grain pitch shifter: 100 ms grain at 48 kHz (G = 4800), +24 st (warp = -3.0)
        // requires center_delay = 7,204 and spans [4.0, 14,404.0]
        let g = 4_800u64;
        let center = 7_204.0_f64;
        let warp = -3.0_f64;
        for frame in [0u64, 1, 1200, 2399, 2400, 3600, 4799] {
            let (_phases, weights, delays) =
                eval_m2_pitch_grain_delays(frame, g, center, warp).unwrap();
            assert!((weights[0] + weights[1] - 1.0).abs() < 1e-12);
            assert!((4.0..=14_404.0).contains(&delays[0]));
            assert!((4.0..=14_404.0).contains(&delays[1]));
        }
        // Odd grain duration is rejected
        assert!(eval_m2_pitch_grain_delays(0, 7, 20.0, -1.0).is_none());

        // 3. M = 1 delay line slew & switch crossfade
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY> {
            current_delay: 64.0,
            target_delay: 64.0,
            ..Default::default()
        };
        for n in 0..128 {
            let _ = dl.step((n as f32) * 0.01);
        }
        dl.set_target_delay(128.0, true);
        assert!(dl.xfade_active);
        for _ in 0..128 {
            let y = dl.step(0.5);
            assert!(y.is_finite());
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 128.0);
    }

    /// verify[dsp.op-t.diffuser.partition]
    /// verify[dsp.op-t.diffuser.recurrence]
    /// verify[dsp.op-t.ducking.vca]
    /// verify[dsp.zoh.op-t-delays]
    /// Verifies rate-converted 8-line partitioned Schroeder all-pass diffusers without a 255-sample cap
    /// (`223 -> 1,784` at `384 kHz`), unit energy conservation, and 1-pole ballistic ducking VCA.
    #[test]
    fn test_partitioned_schroeder_diffusers_rate_conversion_and_ducking_vca() {
        let _guard = FtzDazGuard::new();

        // 1. Rate conversion without 255 cap: 223 at 48 kHz -> 1,784 at 384 kHz
        assert_eq!(convert_schroeder_delay_for_rate(223, 384_000), 1_784);
        let bank_44k1 = PartitionedSchroederDiffuserBank::new_for_rate(44_100);
        assert_eq!(bank_44k1.delays, [17, 28, 43, 67, 95, 128, 164, 205]);

        let mut bank_384k = PartitionedSchroederDiffuserBank::new_for_rate(384_000);
        assert_eq!(bank_384k.delays[7], 1_784);

        // 2. Verify exact delay and all-pass Parseval energy conservation on line 7 (1,784 samples!)
        let mut e_in = 0.0_f64;
        let mut e_out = 0.0_f64;
        for n in 0..50_000 {
            let x = if n == 0 { 1.0_f32 } else { 0.0 };
            let y = bank_384k.step_line(7, x, 0.65);
            e_in += (x as f64) * (x as f64);
            e_out += (y as f64) * (y as f64);
        }
        assert!((e_in - e_out).abs() < 1e-5);

        // 3. ZOH Op-T delay rescaling helper
        let zoh_96k = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let (tau_host, _) = zoh_96k.rescale_op_t_delays(64.0, 19.0);
        assert!((tau_host - 128.0).abs() < 1e-4);

        // 4. Ballistic ducking VCA stereo image preservation
        let mut vca = BallisticDuckingVca::new();
        let (wl, wr) = vca.step_stereo(0.8, 0.2, 1.0, -0.5);
        assert!((wl + 2.0 * wr).abs() < 1e-6);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.op-t.ring.capacity]
        /// verify[dsp.op-t.ring.q0-32-pointer]
        /// verify[dsp.op-t.window.partition-of-unity]
        #[test]
        fn prop_op_t_ring_capacity_q32_pointer_and_partition_of_unity(
            d_ref in 4u64..100_000u64,
            rate_hz in 44_100u32..=384_000u32,
            d_int in 4usize..500usize,
            frac_q32 in any::<u32>(),
            phi in 0.0_f32..100.0_f32,
        ) {
            let (l_cap, bytes) =
                op_t_fractional_history_capacity(d_ref, rate_hz as f64, 2).unwrap();
            prop_assert!(l_cap >= 6 && bytes.is_multiple_of(64));

            let (norm_int, norm_mu) =
                bounded_q32_reader_position(d_int as u64, frac_q32, (4, 0), (1000, 0)).unwrap();
            prop_assert!((4..=1000).contains(&norm_int));
            prop_assert!((0.0..1.0).contains(&norm_mu));

            let (ha, hb) = hermite_hann_partition_of_unity(phi.fract());
            prop_assert!((ha + hb - 1.0).abs() < 1e-6);
            let w3 = bspline_barberpole_3phase(phi);
            prop_assert!((w3[0] + w3[1] + w3[2] - 1.0).abs() < 1e-6);
        }
    }
}
