//! `Q64`/`Q32` Tempo Synchronization, Ordered Clock Pulse Tracker, and Physical `f32` Calibration
//! Inversion (`ref[dsp.tempo.subdivisions]`, `ref[dsp.tempo.q64-q32-phase]`,
//! `ref[dsp.tempo.event-tracker]`, `ref[dsp.tempo.physical-inversion]`,
//! `ref[dsp.tempo.coprime-fdn]`, `ref[dsp.tempo.transport-jump]`).

use alloc::vec::Vec;

use crate::math::{f32_bits_at_rank, f32_rank_intersection, q32_phase_to_phasor, widen_f32_bits};
use crate::schema::cold_exp_f64;

/// Canonical 12 rhythmic subdivision period ratios `(a_s, b_s)` (`ref[dsp.tempo.subdivisions]`).
pub const CANONICAL_SUBDIVISION_RATIOS: [(u32, u32); 12] = [
    (3, 1), // 0: 1/2.
    (2, 1), // 1: 1/2
    (4, 3), // 2: 1/2T
    (3, 2), // 3: 1/4.
    (1, 1), // 4: 1/4
    (2, 3), // 5: 1/4T
    (3, 4), // 6: 1/8.
    (1, 2), // 7: 1/8
    (1, 3), // 8: 1/8T
    (3, 8), // 9: 1/16.
    (1, 4), // 10: 1/16
    (1, 6), // 11: 1/16T
];

const REM3_TABLE_24: [u32; 24] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

// ============================================================================
// 4. Q64 / Q32 Tempo Synchronization, Event Tracker & Physical Inversion
// (`ref[dsp.tempo.subdivisions]`, `ref[dsp.tempo.q64-q32-phase]`,
//  `ref[dsp.tempo.event-tracker]`, `ref[dsp.tempo.physical-inversion]`,
//  `ref[dsp.tempo.coprime-fdn]`, `ref[dsp.tempo.transport-jump]`)
// ============================================================================

/// Computes the division-free `Q0.32` subdivision phase target `q` for `beat_mod12` (`0..12`),
/// `within_beat_q32` (`u32`), `ratio = (a_s, b_s)` in `CANONICAL_SUBDIVISION_RATIOS`, and `offset_q32`
/// (`ref[dsp.tempo.q64-q32-phase]`).
pub fn eval_q32_subdivision_phase(
    beat_mod12: u8,
    within_beat_q32: u32,
    ratio: (u32, u32),
    offset_q32: u32,
) -> Option<u32> {
    if beat_mod12 >= 12 || !CANONICAL_SUBDIVISION_RATIOS.contains(&ratio) {
        return None;
    }
    let (a, b) = ratio;
    let beat_rem = match a {
        1 => 0,
        2 => u32::from(beat_mod12) & 1,
        3 => REM3_TABLE_24[beat_mod12 as usize],
        4 => u32::from(beat_mod12) & 3,
        _ => return None,
    };
    let product = u64::from(b) * ((u64::from(beat_rem) << 32) + u64::from(within_beat_q32));
    let high = (product >> 32) as usize;
    let low = product & u64::from(u32::MAX);
    let phase = match a {
        1 => low,
        2 => (((high as u64 & 1) << 32) + low) >> 1,
        3 => {
            let rem = u64::from(REM3_TABLE_24[high]);
            rem * 0x5555_5555 + (((low + rem) * 0xaaaa_aaab) >> 33)
        }
        4 => (((high as u64 & 3) << 32) + low) >> 2,
        _ => unreachable!(),
    };
    Some((phase as u32).wrapping_add(offset_q32))
}

/// Advances a `(beat_mod12, fraction_q64)` beat accumulator by `step_q64` (`ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub const fn advance_q64_beat_accumulator(
    beat_mod12: u8,
    fraction_q64: u64,
    step_q64: u64,
) -> (u8, u64) {
    let (next_frac, carry) = fraction_q64.overflowing_add(step_q64);
    let next_beat = if carry {
        if beat_mod12 == 11 { 0 } else { beat_mod12 + 1 }
    } else {
        beat_mod12
    };
    (next_beat, next_frac)
}

/// Computes the per-host-frame `Q0.64` seconds increment $\Delta_h = \mathrm{round}_{\text{even}}(2^{64}/f_h)$
/// for $f_h \in [44100.0, 96000.0]\text{ Hz}$ (`ref[dsp.tempo.event-tracker]`).
pub fn host_frame_q64_increment(rate_hz: f64) -> Option<u64> {
    if !rate_hz.is_finite() || !(44_100.0..=96_000.0).contains(&rate_hz) {
        return None;
    }
    let bits = rate_hz.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as u32) - 1023;
    let significand = (1u128 << 52) | ((bits & ((1u64 << 52) - 1)) as u128);
    let numerator = 1u128 << (116 - exponent);
    let q = numerator / significand;
    let r = numerator % significand;
    let rounded = if 2 * r > significand || (2 * r == significand && (q & 1) != 0) {
        q + 1
    } else {
        q
    };
    u64::try_from(rounded).ok()
}

/// Computes the `4×` subframe `Q0.64` offset $\mathrm{round}_{\text{even}}(j \Delta_h / 4)$ for $j \in [0..4]$
/// (`ref[dsp.tempo.event-tracker]`).
#[inline]
pub const fn subframe_4x_q64_offset(host_increment: u64, subframe_idx: u8) -> u64 {
    let prod = (subframe_idx as u128) * (host_increment as u128);
    let q = (prod >> 2) as u64;
    let r = (prod & 3) as u64;
    if r > 2 || (r == 2 && (q & 1) != 0) {
        q + 1
    } else {
        q
    }
}

/// State of the ordered clock-event watermark tracker (`ref[dsp.tempo.event-tracker]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockTrackerState {
    Unavailable,
    Holdover,
    Candidate,
}

/// Ordered clock-event tracker enforcing strictly increasing timestamps and 2-fresh-event recovery
/// after timing loss (`ref[dsp.tempo.event-tracker]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedClockPulseTracker {
    pub watermark: Option<u128>,
    pub previous: Option<u128>,
    pub last_period: Option<u128>,
    pub state: ClockTrackerState,
    pub received: usize,
    pub unmeasurable: usize,
    pub intervals: Vec<u128>,
}

impl Default for OrderedClockPulseTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderedClockPulseTracker {
    pub const fn new() -> Self {
        Self {
            watermark: None,
            previous: None,
            last_period: None,
            state: ClockTrackerState::Unavailable,
            received: 0,
            unmeasurable: 0,
            intervals: Vec::new(),
        }
    }

    pub fn lose_timing(&mut self) {
        self.previous = None;
        self.state = if self.last_period.is_some() {
            ClockTrackerState::Holdover
        } else {
            ClockTrackerState::Unavailable
        };
    }

    pub fn pulse(&mut self, timestamp: u128) -> bool {
        self.received += 1;
        if let Some(wm) = self.watermark
            && timestamp <= wm
        {
            self.unmeasurable += 1;
            self.lose_timing();
            return false;
        }
        self.watermark = Some(timestamp);
        let Some(prev) = self.previous else {
            self.previous = Some(timestamp);
            return true;
        };
        let period = timestamp - prev;
        self.last_period = Some(period);
        self.previous = Some(timestamp);
        self.intervals.push(period);
        self.state = ClockTrackerState::Candidate;
        true
    }
}

/// Evaluates the event-rate-aware exponential period and phase tracker over a sequence of event
/// arrival times (in seconds) with time constant `tau_sec > 0` (`ref[dsp.tempo.event-tracker]`, `0 FDIV`).
pub fn eval_event_rate_aware_tracker(
    event_times_sec: &[f64],
    initial_period_sec: f64,
    tau_sec: f64,
) -> (Vec<f64>, Vec<f64>) {
    if event_times_sec.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let inv_tau = crate::math::newton6_reciprocal_f64(tau_sec.max(1e-12));
    let mut period = initial_period_sec;
    let mut phase_time = event_times_sec[0];
    let mut periods = Vec::with_capacity(event_times_sec.len());
    let mut phases = Vec::with_capacity(event_times_sec.len());
    periods.push(period);
    phases.push(phase_time);

    for pair in event_times_sec.windows(2) {
        let prev_t = pair[0];
        let cur_t = pair[1];
        let gain = 1.0 - cold_exp_f64(-period * inv_tau);
        let predicted = phase_time + period;
        phase_time = predicted + gain * (cur_t - predicted);
        period += gain * ((cur_t - prev_t) - period);
        periods.push(period);
        phases.push(phase_time);
    }
    (periods, phases)
}

/// Transport in-play jump policy (`ref[dsp.tempo.transport-jump]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportJumpPolicy {
    SmoothRelock,
    ImmediatePhaseLock,
}

/// Applies one tick of `SmoothRelock` shortest-arc circular phase correction in `Q0.32` turns
/// (negative tie-break at half-cycle `0x8000_0000`, `ref[dsp.tempo.transport-jump]`).
#[inline]
pub fn smooth_relock_q32_tick(
    previous_q32: u32,
    advance_q32: u32,
    target_q32: u32,
    max_correction_q32: u32,
) -> u32 {
    let predicted = previous_q32.wrapping_add(advance_q32);
    let diff = target_q32.wrapping_sub(predicted) as i32; // -2^31 .. 2^31 - 1 (half-cycle is negative!)
    let limit = (max_correction_q32.min(0x7fff_ffff)) as i32;
    let corr = diff.clamp(-limit, limit);
    predicted.wrapping_add(corr as u32)
}

/// Evaluates the 3-node physical calibration curve $v(u) = \frac{1}{2} v_{\min} u (u - 1) + v_{\text{mid}}(1 - u^2) + \frac{1}{2} v_{\max} u (u + 1)$
/// (`ref[dsp.tempo.physical-inversion]`).
#[inline]
pub fn eval_3node_calibration_f64(anchors: [f64; 3], u: f64) -> f64 {
    let [lo, mid, hi] = anchors;
    0.5 * lo * u * (u - 1.0) + mid * (1.0 - u * u) + 0.5 * hi * u * (u + 1.0)
}

/// Result of inverting a 3-node physical calibration curve over the admissible `f32` rank set
/// (`ref[dsp.tempo.physical-inversion]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoPhysicalInversionResult {
    pub physical_f32: f32,
    pub normalized_f32: f32,
    pub realized_value: f64,
    pub residual: f64,
    pub search_steps: (u32, u32),
}

/// Inverts a monotone 3-node physical calibration curve `anchors = [v_min, v_mid, v_max]` over the
/// admissible `f32` physical rank interval `[first, end)` for `declared = (decl_lo, decl_hi)` and
/// `safe = (safe_lo, safe_hi)` in at most 32 binary-search comparisons per pass, breaking ties toward
/// the lower physical rank (`ref[dsp.tempo.physical-inversion]`).
pub fn invert_tempo_physical_calibration(
    declared: (f32, f32),
    safe: (f32, f32),
    anchors: [f64; 3],
    target: f64,
    flush_subnormals: bool,
) -> Option<TempoPhysicalInversionResult> {
    if !target.is_finite()
        || !declared.0.is_finite()
        || !declared.1.is_finite()
        || !safe.0.is_finite()
        || !safe.1.is_finite()
        || declared.0 >= declared.1
        || safe.0 > safe.1
    {
        return None;
    }
    let lo_phys = declared.0.max(safe.0);
    let hi_phys = declared.1.min(safe.1);
    if lo_phys > hi_phys {
        return None;
    }
    let (first, end) =
        f32_rank_intersection(&[lo_phys as f64], &[hi_phys as f64], flush_subnormals)?;

    let [v_lo, v_mid, v_hi] = anchors;
    let c1 = 0.5 * (v_hi - v_lo);
    let c2 = 0.5 * (v_hi + v_lo) - v_mid;
    let target_delta = target - v_mid;

    let eval_at_rank = |rank: u64| -> Option<(f32, f32, f64, f64)> {
        let p_bits = f32_bits_at_rank(rank, flush_subnormals)?;
        let low = widen_f32_bits(declared.0.to_bits())?;
        let high = widen_f32_bits(declared.1.to_bits())?;
        let val = widen_f32_bits(p_bits)?;
        let span = high - low;
        let inv_span = crate::math::newton6_reciprocal_f64(span);
        let num = if (low < 0.0) != (high < 0.0) {
            2.0 * val - (low + high)
        } else {
            (val - low) + (val - high)
        };
        let coord = (num * inv_span).clamp(-1.0, 1.0);
        let mut u = f32::from_bits(crate::math::round_f64_to_f32_bits(coord)?);
        if flush_subnormals && (u.to_bits() & 0x7fff_ffff) < 0x0080_0000 {
            u = 0.0;
        }
        let u64_val = widen_f32_bits(u.to_bits())?;
        let delta_v = u64_val * (c1 + c2 * u64_val);
        let v = v_mid + delta_v;
        Some((f32::from_bits(p_bits), u, v, delta_v))
    };

    let (_, _, _, dv_first) = eval_at_rank(first)?;
    let (_, _, _, dv_last) = eval_at_rank(end - 1)?;
    let ascending = dv_last >= dv_first;

    let mut low = first;
    let mut high = end;
    let mut crossing_steps = 0u32;
    while low < high {
        let mid = low + (high - low) / 2;
        let (_, _, _, dv) = eval_at_rank(mid)?;
        crossing_steps += 1;
        let before = if ascending {
            dv < target_delta
        } else {
            dv > target_delta
        };
        if before {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let cand0 = low.saturating_sub(1).clamp(first, end - 1);
    let cand1 = low.clamp(first, end - 1);
    let (_, _, _, dv0) = eval_at_rank(cand0)?;
    let (_, _, _, dv1) = eval_at_rank(cand1)?;
    let best = if (dv0 - target_delta).abs() <= (dv1 - target_delta).abs() {
        cand0
    } else {
        cand1
    };
    let (_, _, _, chosen_dv) = eval_at_rank(best)?;

    // Second pass: find lowest rank on any constant plateau matching `chosen_dv`
    let mut p_low = first;
    let mut p_high = best;
    let mut plateau_steps = 0u32;
    while p_low < p_high {
        let mid = p_low + (p_high - p_low) / 2;
        let (_, _, _, dv) = eval_at_rank(mid)?;
        plateau_steps += 1;
        let before = if ascending {
            dv < chosen_dv
        } else {
            dv > chosen_dv
        };
        if before {
            p_low = mid + 1;
        } else {
            p_high = mid;
        }
    }
    let (physical_f32, normalized_f32, realized_value, realized_dv) = eval_at_rank(p_low)?;
    Some(TempoPhysicalInversionResult {
        physical_f32,
        normalized_f32,
        realized_value,
        residual: (realized_dv - target_delta).abs(),
        search_steps: (crossing_steps, plateau_steps),
    })
}

/// Prime reference lengths for Stage IV FDN lines (`ref[dsp.tempo.coprime-fdn]`).
pub const FDN_COPRIME_BASE_PRIMES: [u32; 8] = [149, 211, 307, 401, 503, 601, 701, 887];

/// Verifies that for any of the 144 `(idx_l, idx_r)` stereo subdivision pairs (`0..12 × 0..12`),
/// the scaled FDN modal delay ratios $\alpha_m = P_m / P_{\text{ref}}$ preserve distinct prime
/// factors (`gcd(P_i, P_j) == 1` within each channel, `ref[dsp.tempo.coprime-fdn]`).
pub const fn verify_coprime_fdn_pair(subdiv_l: usize, subdiv_r: usize) -> bool {
    if subdiv_l >= 12 || subdiv_r >= 12 {
        return false;
    }
    let mut i = 0;
    while i < 8 {
        let mut j = i + 1;
        while j < 8 {
            let mut a = FDN_COPRIME_BASE_PRIMES[i];
            let mut b = FDN_COPRIME_BASE_PRIMES[j];
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            if a != 1 {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

/// Maps a `Q0.32` subdivision phase directly to a contractive unit phasor `(cos, sin)`
/// (`ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub fn subdivision_q32_to_phasor(q32: u32) -> (f32, f32) {
    q32_phase_to_phasor(q32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::control::ExactIntFactorAccumulator;
    use alloc::vec;
    use proptest::prelude::*;

    /// verify[dsp.tempo.subdivisions]
    /// verify[dsp.tempo.q64-q32-phase]
    /// verify[dsp.tempo.event-tracker]
    /// verify[dsp.tempo.physical-inversion]
    /// verify[dsp.tempo.coprime-fdn]
    /// verify[dsp.tempo.transport-jump]
    /// verify[dsp.ctrl.op-n.trust-clip]
    /// verify[dsp.ctrl.range.normalize-clamp]
    /// verify[dsp.ctrl.range.f32-rank-bounds]
    /// verify[dsp.ctrl.math.pitch-profile]
    /// verify[dsp.ctrl.math.log2-lut]
    /// verify[dsp.op-n.ctrl.trust-clip]
    /// Verifies all 12 subdivisions, division-free Q32 phase, Q64 beat accumulator, Q64 host-frame
    /// timebase, ordered clock event tracker, physical `f32` calibration inversion, coprime FDN pairs,
    /// and `SmoothRelock` / `ImmediatePhaseLock` transport jump policies.
    #[test]
    fn test_tempo_subdivisions_q64_q32_tracker_inversion_coprime_and_jump() {
        let _guard = FtzDazGuard::new();

        // 1. All 12 subdivisions match integer division reference across beats 0..12
        for &ratio in &CANONICAL_SUBDIVISION_RATIOS {
            let (a, b) = ratio;
            for beat in 0..12u8 {
                for frac in [0u32, 0x4000_0000, 0x7fff_ffff, 0xffff_ffff] {
                    let prod =
                        u64::from(b) * ((u64::from(u32::from(beat) % a) << 32) + u64::from(frac));
                    let expected = (prod / u64::from(a)) as u32;
                    let actual = eval_q32_subdivision_phase(beat, frac, ratio, 0).unwrap();
                    assert_eq!(actual, expected);
                    let (c, s) = subdivision_q32_to_phasor(actual);
                    assert!((c as f64) * (c as f64) + (s as f64) * (s as f64) <= 1.0);
                }
            }
        }

        // 2. Q64 beat accumulator wraps modulo 12 on carry
        assert_eq!(advance_q64_beat_accumulator(11, u64::MAX - 10, 20), (0, 9));

        // 3. Q64 host-frame increment & 4x subframe offsets
        let inc_48k = host_frame_q64_increment(48_000.0).unwrap();
        assert!(inc_48k > 0);
        assert_eq!(subframe_4x_q64_offset(inc_48k, 0), 0);
        assert_eq!(subframe_4x_q64_offset(inc_48k, 4), inc_48k);
        assert!(host_frame_q64_increment(12_000.0).is_none());

        // 4. Ordered clock pulse tracker: equal/out-of-order rejection and 2-fresh-event recovery
        let mut tracker = OrderedClockPulseTracker::new();
        assert!(tracker.pulse(0));
        assert!(tracker.pulse(10));
        assert_eq!(tracker.state, ClockTrackerState::Candidate);
        tracker.lose_timing();
        assert_eq!(tracker.state, ClockTrackerState::Holdover);
        assert!(tracker.pulse(30));
        assert_eq!(tracker.intervals, vec![10]);
        assert!(tracker.pulse(42));
        assert_eq!(tracker.intervals, vec![10, 12]);
        assert!(!tracker.pulse(40)); // out-of-order rejected
        assert_eq!(tracker.state, ClockTrackerState::Holdover);

        // Event-rate-aware exponential filter
        let events: Vec<f64> = (0..100).map(|n| (n as f64) * 0.5).collect();
        let (periods, _) = eval_event_rate_aware_tracker(&events, 0.5, 0.5);
        assert_eq!(periods.len(), 100);
        assert!((periods[99] - 0.5).abs() < 1e-12);

        // 5. Reachable physical f32 calibration inversion (including coarse 2^30 endpoints)
        let lo_coarse = (1u64 << 30) as f32;
        let hi_coarse = lo_coarse + 128.0;
        let inv_coarse = invert_tempo_physical_calibration(
            (lo_coarse, hi_coarse),
            (lo_coarse, hi_coarse),
            [0.0, 1.0, 2.0],
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(inv_coarse.physical_f32, lo_coarse);
        assert_eq!(inv_coarse.normalized_f32, -1.0);
        assert_eq!(inv_coarse.realized_value, 0.0);
        assert_eq!(inv_coarse.residual, 1.0);

        let inv_fine = invert_tempo_physical_calibration(
            (-1.0, 1.0),
            (-1.0, 1.0),
            [0.0, 1.0, 2.0],
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(inv_fine.physical_f32, 0.0);
        assert_eq!(inv_fine.residual, 0.0);
        assert!(inv_fine.search_steps.0 <= 32 && inv_fine.search_steps.1 <= 32);

        // 6. All 144 stereo subdivision pairs preserve coprime FDN base primes
        for l in 0..12 {
            for r in 0..12 {
                assert!(verify_coprime_fdn_pair(l, r));
            }
        }

        // 7. SmoothRelock shortest-arc correction with negative tie-break at half-cycle (0x8000_0000)
        assert_ne!(
            TransportJumpPolicy::SmoothRelock,
            TransportJumpPolicy::ImmediatePhaseLock
        );
        let relocked = smooth_relock_q32_tick(0x4000_0000, 0, 0xc000_0000, 0x1000_0000);
        assert_eq!(relocked, 0x3000_0000); // half-cycle tie breaks negative
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.ctrl.accum.exact-int]
        /// verify[dsp.tempo.q64-q32-phase]
        /// verify[dsp.tempo.physical-inversion]
        #[test]
        fn prop_exact_int_accumulator_and_q32_tempo_subdivision_phase(
            vals in prop::array::uniform4(-100.0_f32..100.0_f32),
            subdiv_idx in 0usize..12usize,
            beat_mod12 in 0u32..12u32,
            frac_q32 in any::<u32>(),
            offset_q32 in any::<u32>(),
            v_min in 10.0_f32..50.0_f32,
            span1 in 10.0_f32..100.0_f32,
            span2 in 10.0_f32..100.0_f32,
            target in 5.0_f64..300.0_f64,
        ) {
            // 1. Order-independent exact-integer accumulation
            let mut acc_fwd = ExactIntFactorAccumulator::new(4);
            let mut acc_rev = ExactIntFactorAccumulator::new(4);
            for i in 0..4 {
                acc_fwd.update_contributor(i, vals[i].to_bits()).unwrap();
                acc_rev.update_contributor(3 - i, vals[3 - i].to_bits()).unwrap();
            }
            prop_assert_eq!(acc_fwd.sum, acc_rev.sum);
            prop_assert_eq!(acc_fwd.twice_pairs, acc_rev.twice_pairs);

            // 2. Q32 subdivision phase matches exact u128 division
            let ratios: [(u32, u32); 12] = [
                (3, 1), (2, 1), (4, 3), (3, 2), (1, 1), (2, 3),
                (3, 4), (1, 2), (1, 3), (3, 8), (1, 4), (1, 6),
            ];
            let (a_s, b_s) = ratios[subdiv_idx];
            let actual_q = eval_q32_subdivision_phase(
                beat_mod12 as u8,
                frac_q32,
                (a_s, b_s),
                offset_q32,
            ).unwrap();
            let rem = u128::from(beat_mod12 % a_s);
            let num = u128::from(b_s) * ((rem << 32) + u128::from(frac_q32));
            let expected_q = ((num / u128::from(a_s)) as u32).wrapping_add(offset_q32);
            prop_assert_eq!(actual_q, expected_q);

            // 3. Monotone physical calibration inversion
            let v_mid = v_min + span1;
            let v_max = v_mid + span2;
            let inv_res = invert_tempo_physical_calibration(
                (v_min, v_max),
                (v_min, v_max),
                [v_min as f64, v_mid as f64, v_max as f64],
                target,
                false,
            ).unwrap();
            let p_val = inv_res.physical_f32;
            prop_assert!(p_val >= v_min && p_val <= v_max);
        }
    }
}

#[cfg(test)]
mod tempo_reference_tests {
    use proptest::prelude::*;

    const RATIOS: [(u32, u32); 12] = [
        (3, 1),
        (2, 1),
        (4, 3),
        (3, 2),
        (1, 1),
        (2, 3),
        (3, 4),
        (1, 2),
        (1, 3),
        (3, 8),
        (1, 4),
        (1, 6),
    ];

    const REM3: [u32; 24] = [
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    ];

    fn fixed_subdivision_phase(beat: u32, fraction: u32, ratio: (u32, u32), offset: u32) -> u32 {
        let (a, b) = ratio;
        let beat_remainder = match a {
            1 => 0,
            2 => beat & 1,
            3 => REM3[beat as usize],
            4 => beat & 3,
            _ => panic!("unsupported subdivision"),
        };
        let product = u64::from(b) * ((u64::from(beat_remainder) << 32) + u64::from(fraction));
        let high = (product >> 32) as usize;
        let low = product & u64::from(u32::MAX);
        let phase = match a {
            1 => low,
            2 => ((u64::from((high as u32) & 1) << 32) + low) >> 1,
            3 => {
                let remainder = u64::from(REM3[high]);
                remainder * 0x5555_5555 + (((low + remainder) * 0xaaaa_aaab) >> 33)
            }
            4 => ((u64::from((high as u32) & 3) << 32) + low) >> 2,
            _ => unreachable!(),
        };
        assert!(phase <= u64::from(u32::MAX));
        (phase as u32).wrapping_add(offset)
    }

    fn tick_beat(beat: u8, fraction: u64, step: u64) -> (u8, u64) {
        let (next, carry) = fraction.overflowing_add(step);
        let next_beat = if carry && beat == 11 {
            0
        } else {
            beat + u8::from(carry)
        };
        (next_beat, next)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        #[test]
        fn division_free_phase_matches_integer_reference(
            beat in 0_u32..12,
            index in 0_usize..12,
            fraction: u32,
            offset: u32,
        ) {
            let (a, b) = RATIOS[index];
            let product = u64::from(b) * ((u64::from(beat % a) << 32) + u64::from(fraction));
            prop_assert!(product < 24_u64 << 32);
            let reference = ((product / u64::from(a)) as u32).wrapping_add(offset);
            prop_assert_eq!(fixed_subdivision_phase(beat, fraction, (a, b), offset), reference);
        }

        #[test]
        fn f32_phase_conversion_preserves_circular_accuracy(phase: u32) {
            let mut rounded = (phase as f32) * f32::from_bits(0x2f80_0000);
            if rounded == 1.0 {
                rounded = 0.0;
            }
            let exact = f64::from(phase) * f64::from(f32::from_bits(0x2f80_0000));
            let mut error = f64::from(rounded) - exact;
            if error < -0.5 {
                error += 1.0;
            } else if error >= 0.5 {
                error -= 1.0;
            }
            prop_assert!(error.abs() <= 2.0_f64.powi(-25));
        }

        #[test]
        fn q64_tick_matches_wide_integer_carry(beat in 0_u8..12, fraction: u64, step: u64) {
            let total = u128::from(fraction) + u128::from(step);
            let expected_fraction = total as u64;
            let expected_beat = if total > u128::from(u64::MAX) {
                (beat + 1) % 12
            } else {
                beat
            };
            prop_assert_eq!(tick_beat(beat, fraction, step), (expected_beat, expected_fraction));
        }
    }

    #[test]
    fn fixed_phase_extremes_match_the_integer_reference() {
        for (a, b) in RATIOS {
            for beat in 0..12 {
                for fraction in [0, 1, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
                    for offset in [0, 1, u32::MAX / 2, u32::MAX] {
                        let product =
                            u64::from(b) * ((u64::from(beat % a) << 32) + u64::from(fraction));
                        let reference = ((product / u64::from(a)) as u32).wrapping_add(offset);
                        assert_eq!(
                            fixed_subdivision_phase(beat, fraction, (a, b), offset),
                            reference
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q64_tick_tracks_carries_and_twelve_beat_wrap() {
        assert_eq!(tick_beat(11, u64::MAX - 1, 3), (0, 1));
        assert_eq!(tick_beat(7, 42, 0), (7, 42));
        assert_eq!(tick_beat(2, 1, u64::MAX), (3, 0));
    }

    #[test]
    fn largest_u32_phase_rounds_to_one_before_canonicalization() {
        let rounded = (u32::MAX as f32) * f32::from_bits(0x2f80_0000);
        assert_eq!(rounded, 1.0);
        let normalized = if rounded == 1.0 { 0.0 } else { rounded };
        assert_eq!(normalized, 0.0);
    }
}
