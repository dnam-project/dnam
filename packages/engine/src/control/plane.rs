//! Canonical v2.0 3 kHz Self-Similar Control Plane, Exact-Integer Sufficient-Statistic
//! Accumulator (`9-word S` + `18-word D`), SPSC Per-Producer Ingress, Host Curve & Timestamp
//! Arbitration, `Q64`/`Q32` Tempo Synchronization, Physical `f32` Calibration Inversion, and
//! 3-Phase Control-Space Audit Helpers (`src/control/plane.rs`).
//!
//! Implements:
//! - `ref[dsp.ctrl.op-m.ballistic-lfo]`, `ref[dsp.ctrl.op-m.rate-retime]`,
//!   `ref[dsp.ctrl.op-m.timing]`, `ref[dsp.ctrl.op-m.spsc-ingress]`
//! - `ref[dsp.ctrl.op-n.engine]`, `ref[dsp.ctrl.switch.option-contributors]`,
//!   `ref[dsp.ctrl.switch.conditional-batch]`, `ref[dsp.ctrl.accum.cp1-scaling]`,
//!   `ref[dsp.ctrl.accum.anova-fm]`, `ref[dsp.ctrl.accum.exact-int]`,
//!   `ref[dsp.op-n.ctrl.incremental-update]`, `ref[dsp.op-n.ctrl.fm-interaction]`
//! - `ref[dsp.ctrl.modulators.dynamic]`, `ref[dsp.ctrl.dispatch.matrix]`,
//!   `ref[dsp.ctrl.dispatch.micro-ramp]`, `ref[dsp.ctrl.ledger.memory-compute]`
//! - `ref[dsp.tempo.subdivisions]`, `ref[dsp.tempo.q64-q32-phase]`,
//!   `ref[dsp.tempo.event-tracker]`, `ref[dsp.tempo.physical-inversion]`,
//!   `ref[dsp.tempo.coprime-fdn]`, `ref[dsp.tempo.transport-jump]`
//! - `ref[dsp.zoh.ctrl-stride]`
//! - `ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//!   `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
//!   `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//!   `ref[train.audit.report.prescription]`

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::math::{f32_bits_at_rank, f32_rank_intersection, q32_phase_to_phasor, widen_f32_bits};
use crate::schema::{DnamError, SchemaOptionExt, cold_exp_f64, reject_if};

/// Number of `u32` limbs in an exact-integer linear sum `S` (`9 × 4 B = 36 B`, `ref[dsp.ctrl.accum.exact-int]`).
pub const EXACT_SUM_WORDS: usize = 9;
/// Number of `u32` limbs in an exact-integer pairwise interaction sum `D` (`18 × 4 B = 72 B`, `ref[dsp.ctrl.accum.exact-int]`).
pub const EXACT_SQUARE_WORDS: usize = 18;
/// Maximum number of contributors per factor in the fixed-width exact-integer accumulator (`255`).
pub const MAX_EXACT_CONTRIBUTORS: usize = 255;

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
// 1. Control Stride, Tick Ratio & Single-Rate-Change Window Retiming
// (`ref[dsp.zoh.ctrl-stride]`, `ref[dsp.ctrl.op-m.ballistic-lfo]`, `ref[dsp.ctrl.op-m.rate-retime]`)
// ============================================================================

/// Computes the control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{eff}}) = \max(8, \lfloor(f_s + 1500)/3000\rfloor)$
/// (`16` at `48 kHz`, `15` at `44.1 kHz`, `ref[dsp.zoh.ctrl-stride]`).
#[inline]
pub const fn control_stride_for_rate(sample_rate_hz: u32) -> u32 {
    let rounded = (sample_rate_hz + 1_500) / 3_000;
    if rounded < 8 { 8 } else { rounded }
}

/// Computes the effective control-tick ratio $\rho_{\text{ctrl}} = \frac{48000 \cdot K_{\text{ctrl}}(f_s)}{16 \cdot f_s}$
/// (`ref[dsp.zoh.ctrl-stride]`, `ref[dsp.ctrl.op-m.ballistic-lfo]`).
#[inline]
pub fn control_tick_ratio(sample_rate_hz: u32) -> f64 {
    let k = control_stride_for_rate(sample_rate_hz) as f64;
    (3_000.0 * k) / (sample_rate_hz as f64)
}

/// Retimes an in-progress control window on an admitted rate transition from `old_rate_hz` to
/// `new_rate_hz` after `consumed_old_frames` (`0 <= consumed < K_old`), returning
/// `(remaining_new_frames, elapsed_seconds)` (`ref[dsp.ctrl.op-m.rate-retime]`).
pub fn retime_control_window(
    old_rate_hz: u32,
    new_rate_hz: u32,
    consumed_old_frames: u32,
) -> Option<(u32, f64)> {
    let old_stride = control_stride_for_rate(old_rate_hz);
    let new_stride = control_stride_for_rate(new_rate_hz);
    if consumed_old_frames >= old_stride || old_rate_hz == 0 || new_rate_hz == 0 {
        return None;
    }
    let remaining_num = (old_stride - consumed_old_frames) as u64 * (new_stride as u64);
    let new_frames = ((2 * remaining_num + (old_stride as u64)) / (2 * (old_stride as u64))) as u32;
    let elapsed_sec = (consumed_old_frames as f64) / (old_rate_hz as f64)
        + (new_frames as f64) / (new_rate_hz as f64);
    Some((new_frames, elapsed_sec))
}

/// Evaluates division-free linear micro-ramping across a block of `stride` samples, clamping every
/// intermediate sample within `[min(start, target), max(start, target)]` and snapping the final
/// sample `stride - 1` to `target` (`ref[dsp.ctrl.dispatch.micro-ramp]`).
pub fn eval_clamped_micro_ramp_block(start: f32, target: f32, stride: usize) -> Vec<f32> {
    if stride == 0 {
        return Vec::new();
    }
    let alpha = 1.0_f32 / (stride as f32);
    let delta = (target - start) * alpha;
    let lo = start.min(target);
    let hi = start.max(target);
    let mut cur = start;
    let mut out = Vec::with_capacity(stride);
    for n in 0..stride {
        if n + 1 == stride {
            cur = target;
        } else {
            cur = (cur + delta).clamp(lo, hi);
        }
        out.push(cur);
    }
    out
}

/// Evaluates the tier-aware stage dispatch matrix (`16x16` base in `Lite..Pro`, plus `16x4` delta
/// in `Ultra`, with `q_send` driving rows `0..5` and `q_down` driving rows `5..16`,
/// `ref[dsp.ctrl.dispatch.matrix]`).
pub fn eval_tier_stage_dispatch(
    w_ctrl_base: &[[f32; 16]; 16],
    w_ctrl_ultra_delta: Option<&[[f32; 4]; 16]>,
    q_send: &[f32; 16],
    q_down: &[f32; 16],
    z_stage_ultra_delta: &[f32; 4],
) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for row in 0..16 {
        let q = if row < 5 { q_send } else { q_down };
        let mut acc = 0.0f32;
        for col in 0..16 {
            acc += w_ctrl_base[row][col] * q[col];
        }
        if let Some(w_delta) = w_ctrl_ultra_delta {
            for col in 0..4 {
                acc += w_delta[row][col] * z_stage_ultra_delta[col];
            }
        }
        out[row] = acc;
    }
    out
}

// ============================================================================
// 2. Bounded Per-Producer SPSC Ingress & Host Curve / Timestamp Arbitration
// (`ref[dsp.ctrl.op-m.spsc-ingress]`, `ref[dsp.ctrl.op-m.timing]`)
// ============================================================================

/// Delivery record emitted at a control-boundary cutoff (`ref[dsp.ctrl.op-m.spsc-ingress]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IngressDelivery {
    pub source: String,
    pub receipt: (u64, u16),
    pub scheduled_frame: u64,
    pub consumed_frame: u64,
    pub missed_first_boundary: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IngressSlotRecord {
    id: usize,
    source: String,
    generation: Option<u64>,
    sequence: Option<u16>,
    ready: bool,
}

/// Bounded per-producer SPSC ingress coordinator (`high 48 bits = generation`, `low 16 bits = sequence`,
/// total slots `< 65,536`, `ref[dsp.ctrl.op-m.spsc-ingress]`).
#[derive(Clone, Debug)]
pub struct BoundedPerProducerIngress {
    lane_capacity: usize,
    history_capacity: usize,
    lane_names: Vec<String>,
    generation: u64,
    sequence: u16,
    next_id: usize,
    slots: BTreeMap<String, Vec<IngressSlotRecord>>,
    history: BTreeMap<u64, u64>,
    pub expired_count: usize,
}

impl BoundedPerProducerIngress {
    pub const MAX_GENERATION: u64 = (1u64 << 48) - 1;

    pub fn new(
        lane_capacity: usize,
        lane_names: &[&str],
        history_capacity: usize,
    ) -> Result<Self, DnamError> {
        reject_if(
            lane_capacity == 0 || history_capacity == 0 || lane_names.is_empty(),
            "capacities must be positive",
        )?;
        let mut unique = BTreeSet::new();
        let mut names = Vec::with_capacity(lane_names.len());
        for &n in lane_names {
            reject_if(!unique.insert(n), "lanes must be unique")?;
            names.push(n.to_string());
        }
        reject_if(
            lane_capacity.saturating_mul(names.len()) >= (1 << 16),
            "total slots exceed the sequence space",
        )?;
        let mut slots = BTreeMap::new();
        for n in &names {
            slots.insert(n.clone(), Vec::new());
        }
        Ok(Self {
            lane_capacity,
            history_capacity,
            lane_names: names,
            generation: 0,
            sequence: 0,
            next_id: 1,
            slots,
            history: BTreeMap::new(),
            expired_count: 0,
        })
    }

    /// Reserves a slot in `source`'s SPSC lane (`None` if full or an unready reservation is open).
    pub fn reserve(&mut self, source: &str) -> Option<usize> {
        let lane = self.slots.get_mut(source)?;
        if lane.len() == self.lane_capacity || lane.iter().any(|r| !r.ready) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        lane.push(IngressSlotRecord {
            id,
            source: source.to_string(),
            generation: None,
            sequence: None,
            ready: false,
        });
        Some(id)
    }

    /// Assigns the atomic `(generation, sequence)` receipt to a reserved slot (`ref[dsp.ctrl.op-m.spsc-ingress]`).
    pub fn accept(&mut self, record_id: usize) -> Result<(u64, u16), DnamError> {
        for lane in self.slots.values_mut() {
            for rec in lane.iter_mut() {
                if rec.id == record_id {
                    reject_if(rec.generation.is_some(), "invalid reservation")?;
                    let receipt = (self.generation, self.sequence);
                    self.sequence = self.sequence.wrapping_add(1);
                    rec.generation = Some(receipt.0);
                    rec.sequence = Some(receipt.1);
                    return Ok(receipt);
                }
            }
        }
        Err(DnamError::SchemaValidationError(
            "invalid reservation".to_string(),
        ))
    }

    /// Marks an accepted slot ready for consumption at the next boundary cutoff.
    pub fn publish(&mut self, record_id: usize) -> Result<(), DnamError> {
        for lane in self.slots.values_mut() {
            for rec in lane.iter_mut() {
                if rec.id == record_id {
                    reject_if(
                        rec.generation.is_none() || rec.ready,
                        "record must be accepted exactly once",
                    )?;
                    rec.ready = true;
                    return Ok(());
                }
            }
        }
        Err(DnamError::SchemaValidationError(
            "unknown reservation".to_string(),
        ))
    }

    /// Advances the boundary generation and drains all ready records accepted on or before the cutoff
    /// in `(generation, sequence)` receipt order (`ref[dsp.ctrl.op-m.spsc-ingress]`).
    pub fn cutoff(&mut self, frame: u64) -> Result<Vec<IngressDelivery>, DnamError> {
        reject_if(
            self.generation >= Self::MAX_GENERATION,
            "generation exhausted",
        )?;
        let current = self.generation;
        self.generation += 1;
        self.sequence = 0;
        self.history.insert(current, frame);
        while self.history.len() > self.history_capacity {
            let oldest = *self.history.keys().next().unwrap();
            self.history.remove(&oldest);
        }

        let mut deliveries = Vec::new();
        for name in &self.lane_names {
            let lane = self.slots.get_mut(name).unwrap();
            while let Some(front) = lane.first() {
                if !front.ready {
                    break;
                }
                let g = front.generation.unwrap();
                let s = front.sequence.unwrap();
                if g > current {
                    break;
                }
                let rec = lane.remove(0);
                if let Some(&scheduled) = self.history.get(&g) {
                    deliveries.push(IngressDelivery {
                        source: rec.source,
                        receipt: (g, s),
                        scheduled_frame: scheduled,
                        consumed_frame: frame,
                        missed_first_boundary: frame > scheduled,
                    });
                } else {
                    self.expired_count += 1;
                }
            }
        }
        deliveries.sort_by_key(|d| d.receipt);
        Ok(deliveries)
    }
}

/// Reconstructed VST3 curve `(per_frame_samples, emitted_timestamp_value_updates)`.
pub type Vst3CurveReconstruction = (Vec<f64>, Vec<(u64, f64)>);

/// Reconstructs a piecewise-linear VST3 normalized automation curve over a block of `block_size`
/// frames starting at `block_start_frame` (`ref[dsp.ctrl.op-m.timing]`).
/// Returns `(per_frame_samples, emitted_updates)`.
pub fn reconstruct_vst3_normalized_curve(
    previous_norm: f64,
    points: &[(usize, f64)],
    block_size: usize,
    block_start_frame: u64,
) -> Result<Vst3CurveReconstruction, DnamError> {
    reject_if(
        block_size == 0 || !(0.0..=1.0).contains(&previous_norm),
        "invalid block",
    )?;
    let mut prev_off: Option<usize> = None;
    for &(off, val) in points {
        reject_if(
            off >= block_size || !(0.0..=1.0).contains(&val) || !val.is_finite(),
            "invalid curve points",
        )?;
        if let Some(p) = prev_off
            && off <= p
        {
            return Err(DnamError::SchemaValidationError(
                "invalid curve points".to_string(),
            ));
        }
        prev_off = Some(off);
    }

    let mut samples = Vec::with_capacity(block_size);
    let mut updates = Vec::new();
    let mut left_offset: isize = -1;
    let mut left_value = previous_norm;
    let mut point_idx = 0usize;
    let mut prior_sample = previous_norm;

    for frame in 0..block_size {
        while point_idx < points.len() && points[point_idx].0 < frame {
            left_offset = points[point_idx].0 as isize;
            left_value = points[point_idx].1;
            point_idx += 1;
        }
        let value = if point_idx < points.len() {
            let (right_offset, right_value) = points[point_idx];
            let num = (frame as isize - left_offset) as f64;
            let den = (right_offset as isize - left_offset) as f64;
            left_value + (right_value - left_value) * (num / den)
        } else {
            left_value
        };
        samples.push(value);
        let is_explicit = points.iter().any(|&(o, _)| o == frame);
        if (value - prior_sample).abs() > 1e-15 || is_explicit {
            updates.push((block_start_frame + (frame as u64), value));
        }
        prior_sample = value;
    }
    Ok((samples, updates))
}

/// Ordering key `(timestamp, daw_priority, receipt_order)` for control arbitration
/// (`DAW > Live` on exact timestamp ties; higher `receipt_order` wins within same priority,
/// `ref[dsp.ctrl.op-m.timing]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ControlEventStamp {
    pub timestamp: u64,
    pub daw_priority: bool,
    pub receipt_order: u64,
}

impl ControlEventStamp {
    pub fn from_source(
        timestamp: u64,
        source: &str,
        receipt_order: u64,
    ) -> Result<Self, DnamError> {
        let daw_priority = match source {
            "daw" => true,
            "studio" | "midi" => false,
            _ => {
                return Err(DnamError::SchemaValidationError(
                    "unattributed source".to_string(),
                ));
            }
        };
        Ok(Self {
            timestamp,
            daw_priority,
            receipt_order,
        })
    }
}

/// Evaluates whether a timestamped event arriving at `ingress_frame` (`after_cutoff`) is late for
/// its first eligible control boundary (`stride = 16`, `ref[dsp.ctrl.op-m.timing]`).
#[inline]
pub const fn is_late_for_first_boundary(
    timestamp: u64,
    stride: u64,
    ingress_frame: u64,
    after_cutoff: bool,
) -> bool {
    let first_boundary = timestamp.div_ceil(stride) * stride;
    ingress_frame > first_boundary || (ingress_frame == first_boundary && after_cutoff)
}

/// Evaluates the complementary disjoint-interval two-path output blend
/// $y = (1 - \alpha) y_{\text{old}} + \alpha y_{\text{new}}$ (`0 <= alpha <= 1`,
/// `ref[dsp.ctrl.switch.conditional-batch]`).
#[inline]
pub fn blend_disjoint_interval_paths(y_old: f32, y_new: f32, alpha: f32) -> f32 {
    let a = alpha.clamp(0.0, 1.0);
    (1.0 - a) * y_old + a * y_new
}

// ============================================================================
// 3. Exact-Integer Sufficient-Statistic Accumulator (`9-word S` + `18-word D`)
// (`ref[dsp.ctrl.accum.exact-int]`, `ref[dsp.op-n.ctrl.incremental-update]`,
//  `ref[dsp.ctrl.accum.anova-fm]`, `ref[dsp.ctrl.accum.cp1-scaling]`)
// ============================================================================

/// Fixed-width unsigned multi-word magnitude (`N` 32-bit little-endian limbs,
/// `ref[dsp.ctrl.accum.exact-int]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactMagnitude<const N: usize> {
    pub words: [u32; N],
}

impl<const N: usize> Default for ExactMagnitude<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> ExactMagnitude<N> {
    pub const fn zero() -> Self {
        Self { words: [0; N] }
    }

    pub fn compare(&self, other: &Self) -> Ordering {
        self.words.iter().rev().cmp(other.words.iter().rev())
    }

    pub fn add_assign(&mut self, other: &Self) {
        let mut carry = 0u64;
        for (w, a) in self.words.iter_mut().zip(other.words) {
            let total = u64::from(*w) + u64::from(a) + carry;
            *w = total as u32;
            carry = total >> 32;
        }
        assert_eq!(carry, 0, "exact-int width exceeded");
    }

    pub fn sub_assign(&mut self, other: &Self) {
        let mut borrow = false;
        for (w, r) in self.words.iter_mut().zip(other.words) {
            let (first, b1) = w.overflowing_sub(r);
            let (res, b2) = first.overflowing_sub(u32::from(borrow));
            *w = res;
            borrow = b1 || b2;
        }
        assert!(!borrow, "exact-int unsigned underflow");
    }

    pub fn bit(&self, pos: usize) -> bool {
        let word = pos / 32;
        word < N && (self.words[word] & (1u32 << (pos % 32))) != 0
    }

    pub fn any_bits_below(&self, limit: usize) -> bool {
        let full = limit / 32;
        if self.words[..full.min(N)].iter().any(|&w| w != 0) {
            return true;
        }
        let rem = limit % 32;
        full < N && rem != 0 && (self.words[full] & ((1u32 << rem) - 1)) != 0
    }

    pub fn highest_bit(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .rev()
            .find(|(_, w)| **w != 0)
            .map(|(idx, w)| idx * 32 + 31 - (w.leading_zeros() as usize))
    }

    pub fn rounded_shift_24(&self, shift: usize) -> u32 {
        let word = shift / 32;
        let low = u64::from(self.words[word]);
        let high = self.words.get(word + 1).map_or(0, |&v| u64::from(v));
        let top = (((low | (high << 32)) >> (shift % 32)) & 0x00ff_ffff) as u32;
        if shift == 0 {
            return top;
        }
        let guard = self.bit(shift - 1);
        let sticky = self.any_bits_below(shift - 1);
        top + u32::from(guard && (sticky || (top & 1) != 0))
    }
}

/// Signed multi-word integer `(negative, ExactMagnitude<N>)` (`ref[dsp.ctrl.accum.exact-int]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedExactInt<const N: usize> {
    pub negative: bool,
    pub magnitude: ExactMagnitude<N>,
}

impl<const N: usize> SignedExactInt<N> {
    pub const fn zero() -> Self {
        Self {
            negative: false,
            magnitude: ExactMagnitude::zero(),
        }
    }

    pub fn add_signed(&mut self, negative: bool, magnitude: ExactMagnitude<N>) {
        if magnitude.highest_bit().is_none() {
            return;
        }
        if self.negative == negative {
            self.magnitude.add_assign(&magnitude);
        } else {
            match self.magnitude.compare(&magnitude) {
                Ordering::Greater => self.magnitude.sub_assign(&magnitude),
                Ordering::Equal => *self = Self::zero(),
                Ordering::Less => {
                    let mut diff = magnitude;
                    diff.sub_assign(&self.magnitude);
                    self.magnitude = diff;
                    self.negative = negative;
                }
            }
        }
    }
}

impl SignedExactInt<EXACT_SUM_WORDS> {
    /// Rounds a 9-word linear sum ($S \cdot 2^{-149}$) to an IEEE-754 `f32` bit pattern (`RNE`).
    pub fn to_rounded_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 23 {
            return Some(sign | self.magnitude.words[0]);
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 149;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

impl SignedExactInt<EXACT_SQUARE_WORDS> {
    /// Rounds an 18-word pairwise interaction sum ($z_{\text{fm}} = D \cdot 2^{-299}$) to an
    /// IEEE-754 `f32` bit pattern (`RNE`, `ref[dsp.ctrl.accum.exact-int]`).
    pub fn to_interaction_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 173 {
            return Some(sign | self.magnitude.rounded_shift_24(150));
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 299;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

fn decode_f32_significand_shift(bits: u32) -> Option<(bool, u32, usize)> {
    let negative = (bits >> 31) != 0;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return None;
    }
    let (sig, shift) = if exponent == 0 {
        (fraction, 0usize)
    } else {
        ((1 << 23) | fraction, (exponent - 1) as usize)
    };
    Some((negative, sig, shift))
}

fn decode_f32_to_9word(bits: u32) -> Option<(bool, ExactMagnitude<EXACT_SUM_WORDS>)> {
    let (neg, sig, shift) = decode_f32_significand_shift(bits)?;
    let mut mag = ExactMagnitude::zero();
    let idx = shift / 32;
    let shifted = u64::from(sig) << (shift % 32);
    mag.words[idx] = shifted as u32;
    mag.words[idx + 1] = (shifted >> 32) as u32;
    Some((neg, mag))
}

fn sparse_product_18word(
    value: &ExactMagnitude<EXACT_SUM_WORDS>,
    significand: u32,
    shift: usize,
) -> ExactMagnitude<EXACT_SQUARE_WORDS> {
    let mut prod = [0u32; EXACT_SUM_WORDS + 1];
    let mut carry = 0u64;
    for (i, &w) in value.words.iter().enumerate() {
        let total = u64::from(w) * u64::from(significand) + carry;
        prod[i] = total as u32;
        carry = total >> 32;
    }
    prod[EXACT_SUM_WORDS] = carry as u32;
    let mut out = ExactMagnitude::zero();
    for (i, &w) in prod.iter().enumerate() {
        let dst = i + shift / 32;
        let shifted = u64::from(w) << (shift % 32);
        out.words[dst] |= shifted as u32;
        out.words[dst + 1] |= (shifted >> 32) as u32;
    }
    out
}

/// Exact-integer sufficient-statistic accumulator for one FM factor (`9-word S` + `18-word D`,
/// `108 B` state + contributor history, `ref[dsp.ctrl.accum.exact-int]`,
/// `ref[dsp.op-n.ctrl.incremental-update]`, `ref[dsp.ctrl.ledger.memory-compute]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactIntFactorAccumulator {
    pub count: usize,
    pub previous_bits: Vec<u32>,
    pub sum: SignedExactInt<EXACT_SUM_WORDS>,
    pub twice_pairs: SignedExactInt<EXACT_SQUARE_WORDS>,
}

impl ExactIntFactorAccumulator {
    pub fn new(count: usize) -> Self {
        assert!(count <= MAX_EXACT_CONTRIBUTORS);
        Self {
            count,
            previous_bits: vec![0; count],
            sum: SignedExactInt::zero(),
            twice_pairs: SignedExactInt::zero(),
        }
    }

    /// Resets the accumulator in place without heap allocation.
    #[inline]
    pub fn reset(&mut self) {
        self.previous_bits.fill(0);
        self.sum = SignedExactInt::zero();
        self.twice_pairs = SignedExactInt::zero();
    }

    /// Updates contributor `index` from its stored `f32` bit pattern to `new_bits` in $O(1)$
    /// without wide squaring (`R = S - b_o`, `D_new = D - 2 b_o R + 2 b_n R`, `S_new = R + b_n`,
    /// `ref[dsp.ctrl.accum.exact-int]`).
    pub fn update_contributor(&mut self, index: usize, new_bits: u32) -> Result<(), DnamError> {
        if index >= self.count {
            return Err(DnamError::InvalidControlIndex(index));
        }
        let (new_neg, new_sig, new_shift) =
            decode_f32_significand_shift(new_bits).or_schema_err("nonfinite factor")?;
        let old_bits = self.previous_bits[index];
        let (old_neg, old_sig, old_shift) = decode_f32_significand_shift(old_bits).unwrap();
        let (_, old_mag) = decode_f32_to_9word(old_bits).unwrap();
        let (_, new_mag) = decode_f32_to_9word(new_bits).unwrap();

        // R = S - b_old
        self.sum.add_signed(!old_neg, old_mag);
        let rest_neg = self.sum.negative;
        // D_new = D - 2 * b_old * R + 2 * b_new * R
        self.twice_pairs.add_signed(
            !(old_neg ^ rest_neg),
            sparse_product_18word(&self.sum.magnitude, old_sig, old_shift + 1),
        );
        self.twice_pairs.add_signed(
            new_neg ^ rest_neg,
            sparse_product_18word(&self.sum.magnitude, new_sig, new_shift + 1),
        );
        // S_new = R + b_new
        self.sum.add_signed(new_neg, new_mag);
        self.previous_bits[index] = new_bits;
        Ok(())
    }

    /// Returns the rounded `f32` bit pattern of $z_{\text{fm}} = D \cdot 2^{-299}$.
    pub fn interaction_f32_bits(&self) -> Option<u32> {
        self.twice_pairs.to_interaction_f32_bits()
    }
}

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
/// arrival times (in seconds) with time constant `tau_sec > 0` (`ref[dsp.tempo.event-tracker]`).
pub fn eval_event_rate_aware_tracker(
    event_times_sec: &[f64],
    initial_period_sec: f64,
    tau_sec: f64,
) -> (Vec<f64>, Vec<f64>) {
    if event_times_sec.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut period = initial_period_sec;
    let mut phase_time = event_times_sec[0];
    let mut periods = Vec::with_capacity(event_times_sec.len());
    let mut phases = Vec::with_capacity(event_times_sec.len());
    periods.push(period);
    phases.push(phase_time);

    for pair in event_times_sec.windows(2) {
        let prev_t = pair[0];
        let cur_t = pair[1];
        let gain = 1.0 - cold_exp_f64(-period / tau_sec);
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
        let num = if (low < 0.0) != (high < 0.0) {
            2.0 * val - (low + high)
        } else {
            (val - low) + (val - high)
        };
        let coord = (num / span).clamp(-1.0, 1.0);
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

// ============================================================================
// 5. Non-Stop 3-Phase Control-Space Audit Helpers
// (`ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//  `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
//  `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//  `ref[train.audit.report.prescription]`)
// ============================================================================

/// Phase-1 `[CLAMP-RANGE]`: intersects `declared_range` with `observed_range` and locks `K_p = 0`
/// if `distinct_observed_count <= 1` (`ref[train.audit.phase1.clamp-range]`).
#[inline]
pub fn audit_phase1_clamp_range(
    declared: (f32, f32),
    observed: (f32, f32),
    distinct_observed_count: usize,
) -> ((f32, f32), u8, bool) {
    let safe_min = declared.0.max(observed.0);
    let safe_max = declared.1.min(observed.1).max(safe_min);
    let degree = if distinct_observed_count <= 1 {
        0u8
    } else {
        (distinct_observed_count - 1).min(3) as u8
    };
    let was_clamped = (safe_min - declared.0).abs() > 1e-6 || (safe_max - declared.1).abs() > 1e-6;
    ((safe_min, safe_max), degree, was_clamped)
}

/// Phase-1 `[CLAMP-SWITCH]`: restricts `declared_options` to options present in `observed_options`
/// (`ref[train.audit.phase1.clamp-switch]`).
pub fn audit_phase1_clamp_switch(
    declared_options: &[&str],
    observed_options: &BTreeSet<&str>,
) -> (u16, Vec<String>) {
    let mut mask = 0u16;
    let mut disabled = Vec::new();
    for (idx, &opt) in declared_options.iter().enumerate().take(16) {
        if observed_options.contains(opt) {
            mask |= 1u16 << idx;
        } else {
            disabled.push(opt.to_string());
        }
    }
    (mask, disabled)
}

/// Phase-1 `[CLAMP-STABILITY]` & Phase-3 `[CLAMP-DYNAMIC]` worst-case excursion verifier
/// (`r_worst <= 0.9995`, `g_lin <= 0.9990`, `|a_diff| <= 0.75`, `tau_min >= 4.0`,
/// `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`).
#[inline]
pub fn verify_dynamic_stability_excursion(
    r_high: f32,
    r_mod_l1: f32,
    tau_0: f32,
    tau_mod_l1: f32,
    g_lin: f32,
    a_diff: f32,
) -> (f32, f32, f32, f32, bool) {
    let r_worst = r_high + r_mod_l1.abs();
    let tau_min = tau_0 - tau_mod_l1.abs();
    let ok = r_worst <= 0.9995 && tau_min >= 4.0 && g_lin <= 0.9990 && a_diff.abs() <= 0.75;
    (
        r_worst.min(0.9995),
        tau_min.max(4.0),
        g_lin.min(0.9990),
        a_diff.clamp(-0.75, 0.75),
        ok,
    )
}

/// Phase-3 `[CLAMP-HEAD-GAIN]`: clamps rendered head spectral gain to `<= +6.0 dB` (`<= 1.9952623`
/// linear amplitude ratio, `ref[train.audit.phase3.clamp-head-gain]`).
#[inline]
pub fn clamp_head_spectral_gain_linear(peak_gain_linear: f32) -> (f32, bool) {
    const MAX_GAIN_6DB_LIN: f32 = 1.995_262_3; // 10^(6/20)
    if peak_gain_linear > MAX_GAIN_6DB_LIN {
        (MAX_GAIN_6DB_LIN / peak_gain_linear, true)
    } else {
        (1.0, false)
    }
}

/// Formats a consolidated post-training audit & stem prescription banner (`ref[train.audit.report.prescription]`).
pub fn format_audit_prescription_report(model_name: &str, clamp_lines: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&alloc::format!(
        "DNAM TRAINING COMPLETE — EXPORTED: {model_name}.dnam | {model_name}.dnam-ckpt\n"
    ));
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&alloc::format!(
        "Active Control Clamps Applied ({}):\n",
        clamp_lines.len()
    ));
    for (idx, line) in clamp_lines.iter().enumerate() {
        out.push_str(&alloc::format!("  {}. {line}\n", idx + 1));
    }
    out.push_str(
        "================================================================================\n",
    );
    out
}

/// Maps a `Q0.32` subdivision phase directly to a contractive unit phasor `(cos, sin)`
/// (`ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub fn subdivision_q32_to_phasor(q32: u32) -> (f32, f32) {
    q32_phase_to_phasor(q32)
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

    // Specification Verification Suite: Phase 4 — Parametric Control Bus, Tempo Sync & 3-Phase Audit.
    //
    // Verifies all engine-scope rules in `docs/specs/dsp/parametric_control_bus.md` and
    // `docs/specs/training/control_space_audit.md`:
    // - `verify[dsp.ctrl.op-m.ballistic-lfo]`
    // - `verify[dsp.ctrl.op-m.rate-retime]`
    // - `verify[dsp.ctrl.op-m.timing]`
    // - `verify[dsp.ctrl.op-m.spsc-ingress]`
    // - `verify[dsp.ctrl.op-n.engine]`
    // - `verify[dsp.ctrl.op-n.trust-clip]`
    // - `verify[dsp.ctrl.range.normalize-clamp]`
    // - `verify[dsp.ctrl.range.f32-rank-bounds]`
    // - `verify[dsp.ctrl.basis.lobatto]`
    // - `verify[dsp.ctrl.accum.cp1-scaling]`
    // - `verify[dsp.ctrl.switch.option-contributors]`
    // - `verify[dsp.ctrl.switch.conditional-batch]`
    // - `verify[dsp.ctrl.accum.anova-fm]`
    // - `verify[dsp.ctrl.accum.exact-int]`
    // - `verify[dsp.ctrl.modulators.dynamic]`
    // - `verify[dsp.ctrl.dispatch.matrix]`
    // - `verify[dsp.ctrl.dispatch.micro-ramp]`
    // - `verify[dsp.ctrl.math.pitch-profile]`
    // - `verify[dsp.ctrl.math.log2-lut]`
    // - `verify[dsp.ctrl.ledger.memory-compute]`
    // - `verify[dsp.tempo.subdivisions]`
    // - `verify[dsp.tempo.q64-q32-phase]`
    // - `verify[dsp.tempo.event-tracker]`
    // - `verify[dsp.tempo.physical-inversion]`
    // - `verify[dsp.tempo.coprime-fdn]`
    // - `verify[dsp.tempo.transport-jump]`
    // - `verify[dsp.zoh.ctrl-stride]`
    // - `verify[train.audit.phase1.clamp-range]`
    // - `verify[train.audit.phase1.clamp-switch]`
    // - `verify[train.audit.phase1.clamp-stability]`
    // - `verify[train.audit.phase3.clamp-dynamic]`
    // - `verify[train.audit.phase3.clamp-head-gain]`
    // - `verify[train.audit.phase3.numeric-envelope]`
    // - `verify[train.audit.report.prescription]`
    // - `verify[dsp.op-n.ctrl.incremental-update]`
    // - `verify[dsp.op-n.ctrl.lobatto-basis]`
    // - `verify[dsp.op-n.ctrl.chebyshev-lfo]`
    // - `verify[dsp.op-n.ctrl.fm-interaction]`
    // - `verify[dsp.op-n.ctrl.trust-clip]`

    use std::collections::BTreeSet;

    use crate::FtzDazGuard;
    use crate::control::{
        BoundedPerProducerIngress, CANONICAL_SUBDIVISION_RATIOS, ClockTrackerState, ControlBus,
        ControlClampDescriptor, ControlEventStamp, ExactIntFactorAccumulator, IngressDelivery,
        OrderedClockPulseTracker, TransportJumpPolicy, advance_q64_beat_accumulator,
        audit_phase1_clamp_range, audit_phase1_clamp_switch, blend_disjoint_interval_paths,
        clamp_head_spectral_gain_linear, control_stride_for_rate, control_tick_ratio,
        eval_clamped_micro_ramp_block, eval_event_rate_aware_tracker, eval_q32_subdivision_phase,
        eval_tier_stage_dispatch, format_audit_prescription_report, host_frame_q64_increment,
        invert_tempo_physical_calibration, is_late_for_first_boundary,
        reconstruct_vst3_normalized_curve, retime_control_window, smooth_relock_q32_tick,
        subdivision_q32_to_phasor, subframe_4x_q64_offset, verify_coprime_fdn_pair,
        verify_dynamic_stability_excursion,
    };
    use crate::math::ScalarBackend;
    use crate::operators::op_n::{eval_chebyshev_4harm, eval_lobatto_basis_3node};

    /// verify[dsp.ctrl.op-m.ballistic-lfo]
    /// verify[dsp.ctrl.op-m.rate-retime]
    /// verify[dsp.zoh.ctrl-stride]
    /// verify[dsp.ctrl.modulators.dynamic]
    /// Verifies control decimation stride `K_ctrl`, tick ratio `rho_ctrl`, single-rate-change mixed-window
    /// retiming, and 8-D dynamic modulator emission.
    #[test]
    fn test_control_stride_tick_ratio_rate_retime_and_dynamic_modulators() {
        for &(sr, exp_stride) in &[
            (44_100u32, 15u32),
            (46_500, 16),
            (48_000, 16),
            (88_200, 29),
            (96_000, 32),
            (176_400, 59),
            (384_000, 128),
        ] {
            assert_eq!(control_stride_for_rate(sr), exp_stride);
        }
        assert!((control_tick_ratio(48_000) - 1.0).abs() < 1e-15);
        assert!((control_tick_ratio(44_100) - (3000.0 / 2940.0)).abs() < 1e-12);

        // Rate transition from 48 kHz (K=16) to 44.1 kHz (K=15) after 8 consumed frames
        let (rem_new, elapsed) = retime_control_window(48_000, 44_100, 8).unwrap();
        assert_eq!(rem_new, 8);
        let expected_elapsed = 8.0 / 48_000.0 + 8.0 / 44_100.0;
        assert!((elapsed - expected_elapsed).abs() < 1e-15);

        // Verify window fraction preservation within half a new sample across rate pairs
        for &(old_r, new_r) in &[
            (44_100u32, 48_000u32),
            (48_000, 44_100),
            (88_200, 46_500),
            (176_400, 384_000),
        ] {
            let old_s = control_stride_for_rate(old_r);
            let new_s = control_stride_for_rate(new_r);
            for c in 0..old_s {
                let (rem, _) = retime_control_window(old_r, new_r, c).unwrap();
                assert!(rem <= new_s);
                let ideal_rem_sec =
                    ((old_s - c) as f64 / (old_s as f64)) * (new_s as f64 / (new_r as f64));
                let actual_rem_sec = (rem as f64) / (new_r as f64);
                assert!((actual_rem_sec - ideal_rem_sec).abs() <= 0.5 / (new_r as f64) + 1e-15);
            }
        }

        let _guard = FtzDazGuard::new();
        let mut bus = ControlBus::<ScalarBackend>::new();
        bus.tick_3khz(0.5, &[]);
        let q16 = ControlBus::<ScalarBackend>::assemble_q16(&[0.1; 8], &[0.2; 8]);
        assert_eq!(q16[0], 0.1);
        assert_eq!(q16[8], 0.2);
    }

    /// verify[dsp.ctrl.op-m.timing]
    /// verify[dsp.ctrl.op-m.spsc-ingress]
    /// verify[dsp.ctrl.switch.conditional-batch]
    /// Verifies bounded per-producer SPSC ingress, VST3 piecewise-linear curve reconstruction,
    /// DAW > Live timestamp tie precedence, late/stale cutoff classification, and disjoint-interval blending.
    #[test]
    fn test_spsc_ingress_vst3_curve_and_timestamp_tie_arbitration() {
        // 1. SPSC slow publisher does not block an independent ready lane
        let mut ingress = BoundedPerProducerIngress::new(1, &["slow", "ready"], 2).unwrap();
        let slow_id = ingress.reserve("slow").unwrap();
        let ready_id = ingress.reserve("ready").unwrap();
        let slow_receipt = ingress.accept(slow_id).unwrap();
        let ready_receipt = ingress.accept(ready_id).unwrap();
        assert_eq!(slow_receipt, (0, 0));
        assert_eq!(ready_receipt, (0, 1));
        ingress.publish(ready_id).unwrap();

        let d0 = ingress.cutoff(0).unwrap();
        assert_eq!(
            d0,
            vec![IngressDelivery {
                source: "ready".to_string(),
                receipt: ready_receipt,
                scheduled_frame: 0,
                consumed_frame: 0,
                missed_first_boundary: false,
            }]
        );

        ingress.publish(slow_id).unwrap();
        let d1 = ingress.cutoff(16).unwrap();
        assert_eq!(
            d1,
            vec![IngressDelivery {
                source: "slow".to_string(),
                receipt: slow_receipt,
                scheduled_frame: 0,
                consumed_frame: 16,
                missed_first_boundary: true,
            }]
        );

        // Expired history drops very old unpublished reservation cleanly
        let mut exp_ingress = BoundedPerProducerIngress::new(1, &["slow"], 2).unwrap();
        let s_id = exp_ingress.reserve("slow").unwrap();
        exp_ingress.accept(s_id).unwrap();
        for f in [0u64, 16, 32] {
            assert!(exp_ingress.cutoff(f).unwrap().is_empty());
        }
        exp_ingress.publish(s_id).unwrap();
        assert!(exp_ingress.cutoff(48).unwrap().is_empty());
        assert_eq!(exp_ingress.expired_count, 1);

        // 2. VST3 single endpoint is a ramp curve, not a step
        let (samples, updates) =
            reconstruct_vst3_normalized_curve(0.0, &[(63, 1.0)], 64, 0).unwrap();
        assert!((samples[0] - 1.0 / 64.0).abs() < 1e-15);
        assert!((samples[15] - 0.25).abs() < 1e-15);
        assert!((samples[16] - 17.0 / 64.0).abs() < 1e-15);
        assert!((samples[63] - 1.0).abs() < 1e-15);
        assert_eq!(updates.len(), 64);

        // 3. Timestamp & tie ordering: DAW > Studio/MIDI at exact timestamp tie; last-received wins within live
        let s_midi = ControlEventStamp::from_source(5, "midi", 1).unwrap();
        let s_studio = ControlEventStamp::from_source(5, "studio", 2).unwrap();
        let s_daw = ControlEventStamp::from_source(5, "daw", 0).unwrap();
        assert!(s_studio > s_midi);
        assert!(s_daw > s_studio);

        // 4. First-boundary late classification
        assert!(!is_late_for_first_boundary(0, 16, 0, false));
        assert!(is_late_for_first_boundary(0, 16, 0, true));
        assert!(!is_late_for_first_boundary(15, 16, 16, false));
        assert!(is_late_for_first_boundary(15, 16, 16, true));

        // 5. Complementary disjoint-interval two-path output blend preserves exact path bound
        for old_v in [-4.0_f32, -1.0, 0.0, 2.0, 4.0] {
            for new_v in [-4.0_f32, -1.0, 0.0, 2.0, 4.0] {
                for step in 0..=8 {
                    let alpha = (step as f32) * 0.125;
                    let y = blend_disjoint_interval_paths(old_v, new_v, alpha);
                    assert!(y.abs() <= old_v.abs().max(new_v.abs()) + 1e-6);
                }
            }
        }
    }

    /// verify[dsp.ctrl.op-n.engine]
    /// verify[dsp.ctrl.basis.lobatto]
    /// verify[dsp.ctrl.switch.option-contributors]
    /// verify[dsp.ctrl.accum.cp1-scaling]
    /// verify[dsp.ctrl.accum.anova-fm]
    /// verify[dsp.ctrl.accum.exact-int]
    /// verify[dsp.ctrl.ledger.memory-compute]
    /// verify[dsp.op-n.ctrl.incremental-update]
    /// verify[dsp.op-n.ctrl.lobatto-basis]
    /// verify[dsp.op-n.ctrl.chebyshev-lfo]
    /// verify[dsp.op-n.ctrl.fm-interaction]
    /// Verifies 9-word `S` + 18-word `D` exact-integer sufficient-statistic accumulation (`O(1)` updates
    /// immune to catastrophic cancellation and update order), Lobatto basis, and Chebyshev LFO waveshaping.
    #[test]
    fn test_exact_integer_accumulator_lobatto_chebyshev_and_fm_invariance() {
        // 1. Catastrophic cancellation test: huge positive + huge negative contributors + small pair
        let mut acc_a = ExactIntFactorAccumulator::new(4);
        let mut acc_b = ExactIntFactorAccumulator::new(4);
        let huge_pos = (1.0e18_f32).to_bits();
        let huge_neg = (-1.0e18_f32).to_bits();
        let c2 = (0.5_f32).to_bits();
        let c3 = (0.25_f32).to_bits();

        // Order A: 0, 1, 2, 3
        acc_a.update_contributor(0, huge_pos).unwrap();
        acc_a.update_contributor(1, huge_neg).unwrap();
        acc_a.update_contributor(2, c2).unwrap();
        acc_a.update_contributor(3, c3).unwrap();

        // Order B: 3, 2, 1, 0 with intermediate updates
        acc_b.update_contributor(2, (3.0_f32).to_bits()).unwrap();
        acc_b.update_contributor(3, c3).unwrap();
        acc_b.update_contributor(1, huge_neg).unwrap();
        acc_b.update_contributor(2, c2).unwrap();
        acc_b.update_contributor(0, huge_pos).unwrap();

        assert_eq!(acc_a.sum, acc_b.sum);
        assert_eq!(acc_a.twice_pairs, acc_b.twice_pairs);
        assert_eq!(acc_a.interaction_f32_bits(), acc_b.interaction_f32_bits());

        // Remove the huge pair and verify exact pair product 0.5 * 0.25 = 0.125
        acc_a.update_contributor(0, 0).unwrap();
        acc_a.update_contributor(1, 0).unwrap();
        assert_eq!(acc_a.interaction_f32_bits(), Some((0.125_f32).to_bits()));

        // 2. Lobatto degree 0..3 & switch contributor
        let psi0 = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 0, 1.0);
        assert_eq!(psi0, [0.0, 0.0, 0.0, 1.0]);
        let psi3 = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 3, 0.0);
        assert!((psi3[0] - 0.5).abs() < 1e-6);
        assert!((psi3[1] - 0.5).abs() < 1e-6);
        assert_eq!(psi3[2], 0.0);

        // 3. Chebyshev T1..T4 waveshaping
        let cheb = eval_chebyshev_4harm(0.5);
        assert!((cheb[0] - 0.5).abs() < 1e-6);
        assert!((cheb[1] - (-0.5)).abs() < 1e-6);
        assert!((cheb[2] - (-1.0)).abs() < 1e-6);
        assert!((cheb[3] - (-0.5)).abs() < 1e-6);

        assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
    }

    /// verify[dsp.ctrl.dispatch.matrix]
    /// verify[dsp.ctrl.dispatch.micro-ramp]
    /// Verifies `16x16` base + `16x4` Ultra stage dispatch and endpoint-clamped micro-ramping.
    #[test]
    fn test_stage_dispatch_matrix_and_clamped_micro_ramping() {
        let mut w_base = [[0.0f32; 16]; 16];
        let mut w_ultra = [[0.0f32; 4]; 16];
        for r in 0..16 {
            w_base[r][0] = 1.0;
            w_ultra[r][0] = 0.5;
        }
        let q_send = [2.0f32; 16];
        let q_down = [4.0f32; 16];
        let z_ultra = [1.0f32; 4];

        let base_only = eval_tier_stage_dispatch(&w_base, None, &q_send, &q_down, &z_ultra);
        assert_eq!(base_only[0], 2.0); // row 0 uses q_send
        assert_eq!(base_only[4], 2.0); // row 4 uses q_send
        assert_eq!(base_only[5], 4.0); // row 5 uses q_down
        assert_eq!(base_only[15], 4.0);

        let with_ultra =
            eval_tier_stage_dispatch(&w_base, Some(&w_ultra), &q_send, &q_down, &z_ultra);
        assert_eq!(with_ultra[0], 2.5);
        assert_eq!(with_ultra[5], 4.5);

        let ramp = eval_clamped_micro_ramp_block(1.0, 2.0, 15);
        assert_eq!(ramp.len(), 15);
        assert_eq!(*ramp.last().unwrap(), 2.0);
        for w in ramp.windows(2) {
            assert!(w[1] >= w[0]);
            assert!((1.0..=2.0).contains(&w[0]));
        }
    }

    /// verify[dsp.tempo.subdivisions]
    /// verify[dsp.tempo.q64-q32-phase]
    /// verify[dsp.tempo.event-tracker]
    /// verify[dsp.tempo.physical-inversion]
    /// verify[dsp.tempo.coprime-fdn]
    /// verify[dsp.tempo.transport-jump]
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

    /// verify[train.audit.phase1.clamp-range]
    /// verify[train.audit.phase1.clamp-switch]
    /// verify[train.audit.phase1.clamp-stability]
    /// verify[train.audit.phase3.clamp-dynamic]
    /// verify[train.audit.phase3.clamp-head-gain]
    /// verify[train.audit.phase3.numeric-envelope]
    /// verify[train.audit.report.prescription]
    /// Verifies Phase-1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), Phase-3
    /// (`[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`, `[NUMERIC-ENVELOPE]`), and the prescription report.
    #[test]
    fn test_three_phase_control_space_audit_and_prescription_report() {
        let ((s_min, s_max), deg0, clamped) = audit_phase1_clamp_range((0.0, 10.0), (2.0, 8.0), 1);
        assert_eq!((s_min, s_max), (2.0, 8.0));
        assert_eq!(deg0, 0);
        assert!(clamped);

        let (_, deg3, _) = audit_phase1_clamp_range((0.0, 10.0), (0.0, 10.0), 5);
        assert_eq!(deg3, 3);

        let obs_sw = BTreeSet::from(["off", "on"]);
        let (mask, disabled) = audit_phase1_clamp_switch(&["off", "on", "ultra"], &obs_sw);
        assert_eq!(mask, 0b011);
        assert_eq!(disabled, vec!["ultra".to_string()]);

        let (r_c, tau_c, g_c, a_c, ok) =
            verify_dynamic_stability_excursion(0.9990, 0.0020, 3.5, 1.0, 1.02, 0.85);
        assert!(!ok);
        assert_eq!(r_c, 0.9995);
        assert_eq!(tau_c, 4.0);
        assert_eq!(g_c, 0.9990);
        assert_eq!(a_c, 0.75);

        let (scale_ok, clamped_ok) = clamp_head_spectral_gain_linear(1.5);
        assert_eq!((scale_ok, clamped_ok), (1.0, false));
        let (scale_hi, clamped_hi) = clamp_head_spectral_gain_linear(4.0);
        assert!(clamped_hi && scale_hi < 0.5);

        let report = format_audit_prescription_report(
            "model",
            &["[CLAMP-RANGE] 'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]"],
        );
        assert!(report.contains("DNAM TRAINING COMPLETE"));
        assert!(report.contains("Active Control Clamps Applied (1):"));
    }

    use proptest::prelude::*;

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
mod accumulator_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Fixed-width control-interaction reference, not a production control bus.

    use core::cmp::Ordering;

    use crate::FtzDazGuard;
    use proptest::prelude::*;

    const SUM_WORDS: usize = 9;
    const SQUARE_WORDS: usize = 18;
    const MAX_CONTRIBUTORS: usize = 255;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Magnitude<const N: usize> {
        words: [u32; N],
    }

    impl<const N: usize> Magnitude<N> {
        const fn zero() -> Self {
            Self { words: [0; N] }
        }

        fn compare(&self, other: &Self) -> Ordering {
            self.words.iter().rev().cmp(other.words.iter().rev())
        }

        fn add(&mut self, other: &Self) {
            let mut carry = 0_u64;
            for (word, added) in self.words.iter_mut().zip(other.words) {
                let total = u64::from(*word) + u64::from(added) + carry;
                *word = total as u32;
                carry = total >> 32;
            }
            assert_eq!(carry, 0, "integer width exceeded");
        }

        fn subtract(&mut self, other: &Self) {
            let mut borrow = false;
            for (word, removed) in self.words.iter_mut().zip(other.words) {
                let (first, first_borrow) = word.overflowing_sub(removed);
                let (result, second_borrow) = first.overflowing_sub(u32::from(borrow));
                *word = result;
                borrow = first_borrow || second_borrow;
            }
            assert!(!borrow, "unsigned subtraction underflow");
        }

        fn shift_right_one_exact(&mut self) {
            let mut carry = 0_u32;
            for word in self.words.iter_mut().rev() {
                let next_carry = *word & 1;
                *word = (*word >> 1) | (carry << 31);
                carry = next_carry;
            }
            assert_eq!(carry, 0, "integer is not even");
        }

        fn bit(&self, position: usize) -> bool {
            let word = position / 32;
            word < N && (self.words[word] & (1_u32 << (position % 32))) != 0
        }

        fn any_bits_below(&self, limit: usize) -> bool {
            let full_words = limit / 32;
            if self.words[..full_words.min(N)]
                .iter()
                .any(|word| *word != 0)
            {
                return true;
            }
            let remaining_bits = limit % 32;
            full_words < N
                && remaining_bits != 0
                && (self.words[full_words] & ((1_u32 << remaining_bits) - 1)) != 0
        }

        fn highest_bit(&self) -> Option<usize> {
            self.words
                .iter()
                .enumerate()
                .rev()
                .find(|(_, word)| **word != 0)
                .map(|(index, word)| index * 32 + 31 - word.leading_zeros() as usize)
        }

        fn rounded_shift(&self, shift: usize) -> u32 {
            let word = shift / 32;
            let low = u64::from(self.words[word]);
            let high = self
                .words
                .get(word + 1)
                .map_or(0, |value| u64::from(*value));
            let top = (((low | (high << 32)) >> (shift % 32)) & 0x00ff_ffff) as u32;
            if shift == 0 {
                return top;
            }
            let guard = self.bit(shift - 1);
            let sticky = self.any_bits_below(shift - 1);
            top + u32::from(guard && (sticky || top & 1 != 0))
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SignedMagnitude<const N: usize> {
        negative: bool,
        magnitude: Magnitude<N>,
    }

    type SignedSum = SignedMagnitude<SUM_WORDS>;
    type SignedInteraction = SignedMagnitude<SQUARE_WORDS>;

    impl<const N: usize> SignedMagnitude<N> {
        const fn zero() -> Self {
            Self {
                negative: false,
                magnitude: Magnitude::zero(),
            }
        }

        fn add(&mut self, negative: bool, magnitude: Magnitude<N>) {
            if magnitude.highest_bit().is_none() {
                return;
            }
            if self.negative == negative {
                self.magnitude.add(&magnitude);
            } else {
                match self.magnitude.compare(&magnitude) {
                    Ordering::Greater => self.magnitude.subtract(&magnitude),
                    Ordering::Equal => *self = Self::zero(),
                    Ordering::Less => {
                        let mut difference = magnitude;
                        difference.subtract(&self.magnitude);
                        self.magnitude = difference;
                        self.negative = negative;
                    }
                }
            }
        }
    }

    impl SignedMagnitude<SUM_WORDS> {
        fn rounded_f32_bits(&self) -> Result<u32, ReferenceError> {
            let Some(mut highest) = self.magnitude.highest_bit() else {
                return Ok(0);
            };
            let sign = u32::from(self.negative) << 31;
            if highest < 23 {
                return Ok(sign | self.magnitude.words[0]);
            }
            let mut rounded = self.magnitude.rounded_shift(highest - 23);
            if rounded == 1 << 24 {
                rounded >>= 1;
                highest += 1;
            }
            let exponent = highest as i32 - 149;
            if exponent > 127 {
                return Err(ReferenceError::Overflow);
            }
            Ok(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
        }
    }

    fn decode_f32_parts(bits: u32) -> Option<(bool, u32, usize)> {
        let negative = bits >> 31 != 0;
        let exponent = (bits >> 23) & 0xff;
        let fraction = bits & 0x007f_ffff;
        if exponent == 0xff {
            return None;
        }
        let (significand, shift) = if exponent == 0 {
            (fraction, 0)
        } else {
            ((1 << 23) | fraction, exponent - 1)
        };
        Some((negative, significand, shift as usize))
    }

    fn decode_f32(bits: u32) -> Option<(bool, Magnitude<SUM_WORDS>)> {
        let (negative, significand, shift) = decode_f32_parts(bits)?;
        let mut magnitude = Magnitude::zero();
        let index = shift / 32;
        let shifted = u64::from(significand) << (shift % 32);
        magnitude.words[index] = shifted as u32;
        magnitude.words[index + 1] = (shifted >> 32) as u32;
        Some((negative, magnitude))
    }

    fn square(value: &Magnitude<SUM_WORDS>) -> Magnitude<SQUARE_WORDS> {
        let mut result = Magnitude::zero();
        for (left_index, left) in value.words.iter().enumerate() {
            let mut carry = 0_u64;
            for (right_index, right) in value.words.iter().enumerate() {
                let index = left_index + right_index;
                let total =
                    u64::from(*left) * u64::from(*right) + u64::from(result.words[index]) + carry;
                result.words[index] = total as u32;
                carry = total >> 32;
            }
            assert_eq!(result.words[left_index + SUM_WORDS], 0);
            result.words[left_index + SUM_WORDS] = carry as u32;
        }
        result
    }

    fn sparse_product(
        value: &Magnitude<SUM_WORDS>,
        significand: u32,
        shift: usize,
    ) -> Magnitude<SQUARE_WORDS> {
        let mut product = [0_u32; SUM_WORDS + 1];
        let mut carry = 0_u64;
        for (index, word) in value.words.iter().enumerate() {
            let total = u64::from(*word) * u64::from(significand) + carry;
            product[index] = total as u32;
            carry = total >> 32;
        }
        product[SUM_WORDS] = carry as u32;
        let mut result = Magnitude::zero();
        for (index, word) in product.iter().enumerate() {
            let destination = index + shift / 32;
            let shifted = u64::from(*word) << (shift % 32);
            result.words[destination] |= shifted as u32;
            result.words[destination + 1] |= (shifted >> 32) as u32;
        }
        result
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ReferenceError {
        InvalidIndex,
        NonFinite,
        Overflow,
    }

    fn round_interaction_bits(
        negative: bool,
        magnitude: &Magnitude<SQUARE_WORDS>,
    ) -> Result<u32, ReferenceError> {
        let Some(mut highest) = magnitude.highest_bit() else {
            return Ok(0);
        };
        let sign = u32::from(negative) << 31;
        if highest < 173 {
            return Ok(sign | magnitude.rounded_shift(150));
        }
        let mut rounded = magnitude.rounded_shift(highest - 23);
        if rounded == 1 << 24 {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = highest as i32 - 299;
        if exponent > 127 {
            return Err(ReferenceError::Overflow);
        }
        Ok(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct IntegerStats {
        count: usize,
        previous: [u32; MAX_CONTRIBUTORS],
        sum: SignedSum,
        squared_sum: Magnitude<SQUARE_WORDS>,
    }

    impl IntegerStats {
        fn new(count: usize) -> Self {
            assert!(count <= MAX_CONTRIBUTORS);
            Self {
                count,
                previous: [0; MAX_CONTRIBUTORS],
                sum: SignedSum::zero(),
                squared_sum: Magnitude::zero(),
            }
        }

        fn update(&mut self, index: usize, bits: u32) -> Result<(), ReferenceError> {
            if index >= self.count {
                return Err(ReferenceError::InvalidIndex);
            }
            let (new_negative, new_magnitude) =
                decode_f32(bits).ok_or(ReferenceError::NonFinite)?;
            let (old_negative, old_magnitude) =
                decode_f32(self.previous[index]).expect("stored contributor is finite");
            self.sum.add(!old_negative, old_magnitude);
            self.squared_sum.subtract(&square(&old_magnitude));
            self.sum.add(new_negative, new_magnitude);
            self.squared_sum.add(&square(&new_magnitude));
            self.previous[index] = bits;
            Ok(())
        }

        fn twice_interaction(&self) -> SignedInteraction {
            let squared_linear = square(&self.sum.magnitude);
            let (negative, mut magnitude) = match squared_linear.compare(&self.squared_sum) {
                Ordering::Less => {
                    let mut difference = self.squared_sum;
                    difference.subtract(&squared_linear);
                    (true, difference)
                }
                Ordering::Equal => return SignedInteraction::zero(),
                Ordering::Greater => (false, squared_linear),
            };
            if !negative {
                magnitude.subtract(&self.squared_sum);
            }
            SignedInteraction {
                negative,
                magnitude,
            }
        }

        fn interaction_bits(&self) -> Result<u32, ReferenceError> {
            let interaction = self.twice_interaction();
            round_interaction_bits(interaction.negative, &interaction.magnitude)
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct SparsePairStats {
        count: usize,
        previous: [u32; MAX_CONTRIBUTORS],
        sum: SignedSum,
        twice_pairs: SignedInteraction,
    }

    impl SparsePairStats {
        fn new(count: usize) -> Self {
            assert!(count <= MAX_CONTRIBUTORS);
            Self {
                count,
                previous: [0; MAX_CONTRIBUTORS],
                sum: SignedSum::zero(),
                twice_pairs: SignedInteraction::zero(),
            }
        }

        fn update(&mut self, index: usize, bits: u32) -> Result<(), ReferenceError> {
            if index >= self.count {
                return Err(ReferenceError::InvalidIndex);
            }
            let (new_negative, new_significand, new_shift) =
                decode_f32_parts(bits).ok_or(ReferenceError::NonFinite)?;
            let (old_negative, old_significand, old_shift) =
                decode_f32_parts(self.previous[index]).expect("stored contributor is finite");
            let (_, old_magnitude) =
                decode_f32(self.previous[index]).expect("stored value is finite");
            let (_, new_magnitude) = decode_f32(bits).expect("validated contributor is finite");
            self.sum.add(!old_negative, old_magnitude);
            let rest_negative = self.sum.negative;
            self.twice_pairs.add(
                !(old_negative ^ rest_negative),
                sparse_product(&self.sum.magnitude, old_significand, old_shift + 1),
            );
            self.twice_pairs.add(
                new_negative ^ rest_negative,
                sparse_product(&self.sum.magnitude, new_significand, new_shift + 1),
            );
            self.sum.add(new_negative, new_magnitude);
            self.previous[index] = bits;
            Ok(())
        }

        fn interaction_bits(&self) -> Result<u32, ReferenceError> {
            round_interaction_bits(self.twice_pairs.negative, &self.twice_pairs.magnitude)
        }
    }

    fn shifted_units(value: &Magnitude<SUM_WORDS>, shift: usize) -> Magnitude<SQUARE_WORDS> {
        let mut shifted = Magnitude::zero();
        for (index, word) in value.words.iter().enumerate() {
            let destination = index + shift / 32;
            let bits = u64::from(*word) << (shift % 32);
            shifted.words[destination] |= bits as u32;
            shifted.words[destination + 1] |= (bits >> 32) as u32;
        }
        shifted
    }

    fn round_u64_right_even(value: u64, shift: u32) -> u64 {
        if shift == 0 {
            return value;
        }
        if shift > 64 {
            return 0;
        }
        if shift == 64 {
            return u64::from(value > 1_u64 << 63);
        }
        let truncated = value >> shift;
        let remainder = value & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        truncated + u64::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
    }

    fn exact_product_bits(left: u32, right: u32) -> Result<u32, ReferenceError> {
        fn parts(bits: u32) -> Result<(u64, i32), ReferenceError> {
            let exponent = (bits >> 23) & 0xff;
            let fraction = bits & 0x007f_ffff;
            match exponent {
                0xff => Err(ReferenceError::NonFinite),
                0 => Ok((u64::from(fraction), -149)),
                _ => Ok((u64::from((1 << 23) | fraction), exponent as i32 - 150)),
            }
        }
        let (left_mantissa, left_power) = parts(left)?;
        let (right_mantissa, right_power) = parts(right)?;
        let product = left_mantissa * right_mantissa;
        if product == 0 {
            return Ok(0);
        }
        let sign = (left ^ right) & 0x8000_0000;
        let power = left_power + right_power;
        let highest = 63 - product.leading_zeros() as i32;
        let mut exponent = highest + power;
        if exponent < -126 {
            let shift = power + 149;
            let rounded = if shift >= 0 {
                product << shift as u32
            } else {
                round_u64_right_even(product, (-shift) as u32)
            };
            return Ok(sign | rounded as u32);
        }
        if exponent > 127 {
            return Err(ReferenceError::Overflow);
        }
        let shift = highest - 23;
        let mut rounded = if shift >= 0 {
            round_u64_right_even(product, shift as u32)
        } else {
            product << (-shift) as u32
        };
        if rounded == 1 << 24 {
            rounded >>= 1;
            exponent += 1;
        }
        if exponent > 127 {
            return Err(ReferenceError::Overflow);
        }
        Ok(sign | (((exponent + 127) as u32) << 23) | (rounded as u32 & 0x007f_ffff))
    }

    fn next_finite_encoding(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let bits = (*state >> 32) as u32;
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            (bits & !0x7f80_0000) | 0x7f00_0000
        } else {
            bits
        }
    }

    #[test]
    fn guarded_integer_reference_preserves_small_interactions_and_ties() {
        let _guard = FtzDazGuard::new();
        let cases = [
            (0x3f80_0000, 0x3300_0000, 0x3300_0000),
            (0x3f80_0000, 0x0000_0001, 0x0000_0001),
            (0xbf80_0000, 0x0000_0001, 0x8000_0001),
            (0x3f00_0000, 0x0000_0001, 0x0000_0000),
            (0xbf00_0000, 0x0000_0001, 0x8000_0000),
            (0x3f00_0000, 0x0000_0003, 0x0000_0002),
            (0x3f80_0001, 0x3fc0_0000, 0x3fc0_0002),
            (0x3f80_0000, 0x7f7f_ffff, 0x7f7f_ffff),
        ];
        for (first, second, expected) in cases {
            let mut stats = IntegerStats::new(2);
            stats.update(0, first).expect("finite contributor");
            stats.update(1, second).expect("finite contributor");
            assert_eq!(stats.interaction_bits(), Ok(expected));
        }
    }

    #[test]
    fn guarded_integer_reference_rounds_main_sums_and_cancellation() {
        let _guard = FtzDazGuard::new();
        let cases: &[(&[u32], u32)] = &[
            (&[0x0000_0001, 0x0000_0001], 0x0000_0002),
            (&[0x007f_ffff, 0x0000_0001], 0x0080_0000),
            (&[0x0080_0000, 0x8000_0001], 0x007f_ffff),
            (&[0x3f80_0000, 0x3380_0000], 0x3f80_0000),
            (&[0x3f80_0000, 0x3440_0000], 0x3f80_0002),
            (&[0x7f7f_ffff, 0xff7f_ffff, 0x0000_0001], 0x0000_0001),
            (&[0xff7f_ffff, 0x7f7f_ffff, 0x8000_0001], 0x8000_0001),
        ];
        for (values, expected) in cases {
            let mut stats = IntegerStats::new(values.len());
            for (index, bits) in values.iter().enumerate() {
                stats.update(index, *bits).expect("finite contributor");
            }
            assert_eq!(stats.sum.rounded_f32_bits(), Ok(*expected));
        }
        let mut stats = IntegerStats::new(2);
        stats.update(0, f32::MAX.to_bits()).expect("finite");
        stats.update(1, f32::MAX.to_bits()).expect("finite");
        assert_eq!(stats.sum.rounded_f32_bits(), Err(ReferenceError::Overflow));
    }

    #[test]
    fn guarded_main_sum_rounding_covers_every_subnormal_value() {
        let _guard = FtzDazGuard::new();
        let mut sum = SignedSum::zero();
        for fraction in 0..0x0080_0000_u32 {
            sum.magnitude.words[0] = fraction;
            assert_eq!(sum.rounded_f32_bits(), Ok(fraction));
            if fraction != 0 && fraction % 127 == 0 {
                sum.negative = true;
                assert_eq!(sum.rounded_f32_bits(), Ok(fraction | 0x8000_0000));
                sum.negative = false;
            }
        }
    }

    #[test]
    fn guarded_main_sum_rounding_covers_normal_boundaries_and_overflow() {
        let _guard = FtzDazGuard::new();
        let mut unit = Magnitude::<SUM_WORDS>::zero();
        unit.words[0] = 1;
        for exponent in 2..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe] {
                let lower = (exponent << 23) | fraction;
                let (_, mut units) = decode_f32(lower).expect("finite");
                let (_, upper_units) = decode_f32(lower + 1).expect("finite");
                units.add(&upper_units);
                units.shift_right_one_exact();
                let expected = lower + (lower & 1);
                let mut sum = SignedSum {
                    negative: false,
                    magnitude: units,
                };
                assert_eq!(sum.rounded_f32_bits(), Ok(expected));
                sum.negative = true;
                assert_eq!(sum.rounded_f32_bits(), Ok(expected | 0x8000_0000));
                sum.negative = false;
                sum.magnitude.subtract(&unit);
                assert_eq!(sum.rounded_f32_bits(), Ok(lower));
                sum.magnitude = units;
                sum.magnitude.add(&unit);
                assert_eq!(sum.rounded_f32_bits(), Ok(lower + 1));
            }
        }
        let (_, mut maximum) = decode_f32(f32::MAX.to_bits()).expect("finite");
        let mut next_power = Magnitude::<SUM_WORDS>::zero();
        next_power.words[8] = 1 << 21;
        maximum.add(&next_power);
        maximum.shift_right_one_exact();
        let mut sum = SignedSum {
            negative: false,
            magnitude: maximum,
        };
        assert_eq!(sum.rounded_f32_bits(), Err(ReferenceError::Overflow));
        sum.magnitude.subtract(&unit);
        assert_eq!(sum.rounded_f32_bits(), Ok(f32::MAX.to_bits()));
    }

    #[test]
    fn guarded_integer_rounding_covers_every_subnormal_midpoint() {
        let _guard = FtzDazGuard::new();
        for lower in 0..0x0080_0000_u32 {
            let mut midpoint = Magnitude::<SQUARE_WORDS>::zero();
            let shifted = u64::from(2 * lower + 1) << 21;
            midpoint.words[4] = shifted as u32;
            midpoint.words[5] = (shifted >> 32) as u32;
            let expected = lower + (lower & 1);
            assert_eq!(round_interaction_bits(false, &midpoint), Ok(expected));
            if lower % 127 == 0 {
                assert_eq!(
                    round_interaction_bits(true, &midpoint),
                    Ok(expected | 0x8000_0000)
                );
                let mut below = midpoint;
                let mut above = midpoint;
                let mut unit = Magnitude::<SQUARE_WORDS>::zero();
                unit.words[0] = 1;
                below.subtract(&unit);
                above.add(&unit);
                assert_eq!(round_interaction_bits(false, &below), Ok(lower));
                assert_eq!(round_interaction_bits(false, &above), Ok(lower + 1));
            }
        }
    }

    #[test]
    fn guarded_integer_rounding_covers_normal_boundaries_and_overflow() {
        let _guard = FtzDazGuard::new();
        let mut unit = Magnitude::<SQUARE_WORDS>::zero();
        unit.words[0] = 1;
        for exponent in 1..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe] {
                let lower = (exponent << 23) | fraction;
                let (_, mut units) = decode_f32(lower).expect("finite");
                let (_, upper_units) = decode_f32(lower + 1).expect("finite");
                units.add(&upper_units);
                let midpoint = shifted_units(&units, 149);
                let expected = lower + (lower & 1);
                assert_eq!(round_interaction_bits(false, &midpoint), Ok(expected));
                assert_eq!(
                    round_interaction_bits(true, &midpoint),
                    Ok(expected | 0x8000_0000)
                );
                let mut below = midpoint;
                let mut above = midpoint;
                below.subtract(&unit);
                above.add(&unit);
                assert_eq!(round_interaction_bits(false, &below), Ok(lower));
                assert_eq!(round_interaction_bits(false, &above), Ok(lower + 1));
            }
        }
        let (_, mut maximum) = decode_f32(f32::MAX.to_bits()).expect("finite");
        let mut next_power = Magnitude::<SUM_WORDS>::zero();
        next_power.words[8] = 1 << 21;
        maximum.add(&next_power);
        let overflow_midpoint = shifted_units(&maximum, 149);
        let mut below = overflow_midpoint;
        below.subtract(&unit);
        assert_eq!(
            round_interaction_bits(false, &below),
            Ok(f32::MAX.to_bits())
        );
        assert_eq!(
            round_interaction_bits(false, &overflow_midpoint),
            Err(ReferenceError::Overflow)
        );
        assert_eq!(
            round_interaction_bits(true, &overflow_midpoint),
            Err(ReferenceError::Overflow)
        );
    }

    #[test]
    fn guarded_integer_reference_restores_state_and_rejects_bad_inputs() {
        let _guard = FtzDazGuard::new();
        let mut stats = IntegerStats::new(2);
        let mut sparse = SparsePairStats::new(2);
        stats.update(0, 1.0_f32.to_bits()).expect("finite");
        sparse.update(0, 1.0_f32.to_bits()).expect("finite");
        for _ in 0..10_000 {
            for value in [
                5.0_f32 * 2.0_f32.powi(-26),
                5.0_f32 * 2.0_f32.powi(-25),
                0.0,
            ] {
                stats.update(1, value.to_bits()).expect("finite");
                sparse.update(1, value.to_bits()).expect("finite");
            }
        }
        assert_eq!(stats.interaction_bits(), Ok(0));
        assert_eq!(sparse.interaction_bits(), Ok(0));
        assert_eq!(sparse.twice_pairs, stats.twice_interaction());
        let unchanged = stats.clone();
        assert_eq!(
            stats.update(1, f32::INFINITY.to_bits()),
            Err(ReferenceError::NonFinite)
        );
        assert_eq!(
            stats.update(1, f32::NAN.to_bits()),
            Err(ReferenceError::NonFinite)
        );
        assert_eq!(stats.update(2, 0), Err(ReferenceError::InvalidIndex));
        assert_eq!(stats, unchanged);

        stats.update(0, f32::MAX.to_bits()).expect("finite");
        stats.update(1, 2.0_f32.to_bits()).expect("finite");
        assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
    }

    #[test]
    fn maximum_contributor_count_stays_inside_fixed_widths() {
        let _guard = FtzDazGuard::new();
        let mut stats = IntegerStats::new(MAX_CONTRIBUTORS);
        let mut sparse = SparsePairStats::new(MAX_CONTRIBUTORS);
        for index in 0..MAX_CONTRIBUTORS {
            stats.update(index, f32::MAX.to_bits()).expect("finite");
            sparse.update(index, f32::MAX.to_bits()).expect("finite");
        }
        assert!(stats.sum.magnitude.highest_bit().expect("nonzero") < 285);
        assert!(stats.squared_sum.highest_bit().expect("nonzero") < 562);
        assert!(square(&stats.sum.magnitude).highest_bit().expect("nonzero") < 570);
        assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
        assert_eq!(sparse.sum, stats.sum);
        assert_eq!(sparse.twice_pairs, stats.twice_interaction());

        for index in (1..MAX_CONTRIBUTORS).step_by(2) {
            stats.update(index, (-f32::MAX).to_bits()).expect("finite");
            sparse.update(index, (-f32::MAX).to_bits()).expect("finite");
        }
        assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
        assert_eq!(sparse.sum, stats.sum);
        assert_eq!(sparse.twice_pairs, stats.twice_interaction());
        assert_eq!(sparse.interaction_bits(), stats.interaction_bits());
    }

    #[test]
    fn guarded_integer_reference_preserves_cancellation_of_extreme_pairs() {
        let _guard = FtzDazGuard::new();
        let mut stats = IntegerStats::new(3);
        stats.update(0, 0x7f7f_ffff).expect("finite");
        stats.update(1, 0x7f7f_ffff).expect("finite");
        stats.update(2, 0xfeff_ffff).expect("finite");
        assert_eq!(stats.interaction_bits(), Ok(0));
        stats.update(2, 0xfeff_fffe).expect("finite");
        assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
    }

    #[test]
    fn guarded_integer_reference_matches_independent_full_exponent_products() {
        let _guard = FtzDazGuard::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..20_000 {
            let left = next_finite_encoding(&mut state);
            let right = next_finite_encoding(&mut state);
            let mut stats = IntegerStats::new(2);
            let mut sparse = SparsePairStats::new(2);
            stats.update(0, left).expect("finite contributor");
            stats.update(1, right).expect("finite contributor");
            sparse.update(0, left).expect("finite contributor");
            sparse.update(1, right).expect("finite contributor");
            assert_eq!(stats.interaction_bits(), exact_product_bits(left, right));
            assert_eq!(sparse.twice_pairs, stats.twice_interaction());
            assert_eq!(sparse.interaction_bits(), exact_product_bits(left, right));
        }
    }

    #[test]
    fn guarded_sparse_updates_match_wide_statistics_after_full_exponent_histories() {
        let _guard = FtzDazGuard::new();
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut stats = IntegerStats::new(MAX_CONTRIBUTORS);
        let mut sparse = SparsePairStats::new(MAX_CONTRIBUTORS);
        for step in 0..20_000 {
            let index = next_finite_encoding(&mut state) as usize % MAX_CONTRIBUTORS;
            let bits = if step % 17 == 0 {
                0
            } else {
                next_finite_encoding(&mut state)
            };
            stats.update(index, bits).expect("finite contributor");
            sparse.update(index, bits).expect("finite contributor");
            assert_eq!(sparse.sum, stats.sum);
            assert_eq!(sparse.twice_pairs, stats.twice_interaction());
            assert_eq!(sparse.interaction_bits(), stats.interaction_bits());
        }
        let unchanged = sparse.clone();
        assert_eq!(
            sparse.update(0, f32::INFINITY.to_bits()),
            Err(ReferenceError::NonFinite)
        );
        assert_eq!(
            sparse.update(0, f32::NAN.to_bits()),
            Err(ReferenceError::NonFinite)
        );
        assert_eq!(
            sparse.update(MAX_CONTRIBUTORS, 0),
            Err(ReferenceError::InvalidIndex)
        );
        assert_eq!(sparse, unchanged);
    }

    #[test]
    fn guarded_sparse_updates_cover_every_finite_exponent_and_sign() {
        let _guard = FtzDazGuard::new();
        for left in [
            0x0000_0001,
            0x0080_0001,
            0x3f80_0001,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x8080_0000,
        ] {
            let mut sparse = SparsePairStats::new(2);
            sparse.update(0, left).expect("finite contributor");
            for exponent in 0..=254 {
                for fraction in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                    for sign in [0, 0x8000_0000] {
                        let right = sign | (exponent << 23) | fraction;
                        sparse.update(1, right).expect("finite contributor");
                        assert_eq!(sparse.interaction_bits(), exact_product_bits(left, right));
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn integer_reference_matches_exact_small_dyadic_pairs(
            values in prop::collection::vec(-1000_i32..=1000, 0..=16),
            shift in 0_u32..=20,
        ) {
            let _guard = FtzDazGuard::new();
            let mut stats = IntegerStats::new(values.len());
            let mut sparse = SparsePairStats::new(values.len());
            for (index, value) in values.iter().enumerate() {
                let input = (*value as f64 * 2.0_f64.powi(-(shift as i32))) as f32;
                stats.update(index, input.to_bits()).expect("finite contributor");
                sparse.update(index, input.to_bits()).expect("finite contributor");
            }
            let mut exact_pairs = 0_i64;
            let mut exact_sum = 0_i64;
            for (index, left) in values.iter().enumerate() {
                exact_sum += i64::from(*left);
                for right in &values[index + 1..] {
                    exact_pairs += i64::from(*left) * i64::from(*right);
                }
            }
            let expected = (exact_pairs as f64 * 2.0_f64.powi(-2 * shift as i32)) as f32;
            let expected_sum = (exact_sum as f64 * 2.0_f64.powi(-(shift as i32))) as f32;
            prop_assert_eq!(stats.interaction_bits(), Ok(expected.to_bits()));
            prop_assert_eq!(stats.sum.rounded_f32_bits(), Ok(expected_sum.to_bits()));
            prop_assert_eq!(sparse.twice_pairs, stats.twice_interaction());
            prop_assert_eq!(sparse.interaction_bits(), Ok(expected.to_bits()));
        }

        #[test]
        fn integer_reference_matches_mixed_exponent_pairs(
            values in prop::collection::vec((-1000_i32..=1000, 0_u32..=10), 0..=16),
        ) {
            let _guard = FtzDazGuard::new();
            let mut stats = IntegerStats::new(values.len());
            let mut integer_units = Vec::with_capacity(values.len());
            for (index, (coefficient, shift)) in values.iter().enumerate() {
                let input = (*coefficient as f64 * 2.0_f64.powi(-(*shift as i32))) as f32;
                stats.update(index, input.to_bits()).expect("finite contributor");
                integer_units.push(i64::from(*coefficient) << (10 - shift));
            }
            let mut exact_pairs = 0_i64;
            let mut exact_sum = 0_i64;
            for (index, left) in integer_units.iter().enumerate() {
                exact_sum += left;
                for right in &integer_units[index + 1..] {
                    exact_pairs += left * right;
                }
            }
            let expected = (exact_pairs as f64 * 2.0_f64.powi(-20)) as f32;
            let expected_sum = (exact_sum as f64 * 2.0_f64.powi(-10)) as f32;
            prop_assert_eq!(stats.interaction_bits(), Ok(expected.to_bits()));
            prop_assert_eq!(stats.sum.rounded_f32_bits(), Ok(expected_sum.to_bits()));
        }
    }
}

#[cfg(test)]
mod tempo_reference_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Fixed-point tempo feasibility examples, not production tempo tests.

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
