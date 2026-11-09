//! Op-T Ring Capacity, Branchless Wrap, Q0.32 Pointer, Max-Tree Feedback Guard & Partitioned Diffusers.
//!
//! Canonical Reference: `docs/specs/dsp/op_t_transport.md` (`ref[dsp.op-t.ring.capacity]`, `ref[dsp.op-t.ring.branchless-wrap]`, `ref[dsp.op-t.ring.q0-32-pointer]`, `ref[dsp.op-t.guard.max-tree]`, `ref[dsp.op-t.diffuser.partition]`).

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::container::align64_u64;
use crate::math::{
    flush_recursive_state_f32,
    fp16::{f32_to_fp16, fp16_to_f32},
    widen_f32_bits,
};
use crate::operators::SCHROEDER_COPRIME_DELAYS;
use crate::schema::DnamError;

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
    let scale = f32::from_bits(0x2f80_0000);
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
mod spec_tests {
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
    use alloc::vec;
    use proptest::prelude::*;

    /// verify[dsp.op-t.ring.declaration]
    /// verify[dsp.op-t.ring.capacity]
    /// verify[dsp.op-t.ring.admission]
    /// verify[dsp.op-t.ring.branchless-wrap]
    /// verify[dsp.op-t.ring.q0-32-pointer]
    /// verify[dsp.op-t.ring.fp16-saturation]
    #[test]
    fn test_op_t_ring_capacity_admission_branchless_wrap_q32_and_fp16_saturation() {
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

        let (_, old_bytes) = op_t_fractional_history_capacity(432_000, 48_000.0, 2).unwrap();
        let (_, new_bytes) = op_t_fractional_history_capacity(432_000, 96_000.0, 2).unwrap();
        assert!(admit_op_t_memory_budget(0, new_bytes, 2_000_000).is_ok());
        assert!(admit_op_t_memory_budget(old_bytes, new_bytes, 2_000_000).is_err());

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
        let carry_clamped =
            bounded_q32_reader_position(4, u32::MAX, (4, 0), (4, u32::MAX)).unwrap();
        assert_eq!(carry_clamped, (4, f32::from_bits(0x3f7f_ffff)));

        let mut flags = 0u32;
        assert_eq!(f32_to_fp16_with_telemetry(100_000.0, &mut flags), 0x7bff);
        assert_eq!(flags, saturation_flags::FP16_SATURATED);
    }

    /// verify[dsp.op-t.reader.catmull-rom]
    /// verify[dsp.op-t.guard.bounded-history]
    /// verify[dsp.op-t.guard.max-tree]
    /// verify[dsp.op-t.guard.bibo-decay]
    /// verify[dsp.op-t.guard.fp16-decay]
    #[test]
    fn test_catmull_rom_reader_and_bounded_history_max_tree_feedback_guard() {
        assert_eq!(catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 0.0), 2.0);
        assert_eq!(catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 1.0), 3.0);
        assert_eq!(catmull_rom_hermite_cubic(-1.0, 1.0, 1.0, -1.0, 0.5), 1.25);

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
    #[test]
    fn test_op_t_dual_mode_transitions_and_m1_m2_m3_partition_of_unity() {
        let _guard = FtzDazGuard::new();

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
        assert!(eval_m2_pitch_grain_delays(0, 7, 20.0, -1.0).is_none());

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
    #[test]
    fn test_partitioned_schroeder_diffusers_rate_conversion_and_ducking_vca() {
        let _guard = FtzDazGuard::new();

        assert_eq!(convert_schroeder_delay_for_rate(223, 384_000), 1_784);
        let bank_44k1 = PartitionedSchroederDiffuserBank::new_for_rate(44_100);
        assert_eq!(bank_44k1.delays, [17, 28, 43, 67, 95, 128, 164, 205]);

        let mut bank_384k = PartitionedSchroederDiffuserBank::new_for_rate(384_000);
        assert_eq!(bank_384k.delays[7], 1_784);

        let mut e_in = 0.0_f64;
        let mut e_out = 0.0_f64;
        for n in 0..50_000 {
            let x = if n == 0 { 1.0_f32 } else { 0.0 };
            let y = bank_384k.step_line(7, x, 0.65);
            e_in += (x as f64) * (x as f64);
            e_out += (y as f64) * (y as f64);
        }
        assert!((e_in - e_out).abs() < 1e-5);

        let zoh_96k = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let (tau_host, _) = zoh_96k.rescale_op_t_delays(64.0, 19.0);
        assert!((tau_host - 128.0).abs() < 1e-4);

        let mut vca = BallisticDuckingVca::new();
        let (wl, wr) = vca.step_stereo(0.8, 0.2, 1.0, -0.5);
        assert!((wl + 2.0 * wr).abs() < 1e-6);
    }

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
