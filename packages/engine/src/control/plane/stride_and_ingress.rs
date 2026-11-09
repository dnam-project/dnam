//! Control-bus decimation stride, window retiming, stage dispatch matrix, bounded per-producer
//! SPSC ingress, and host curve / timestamp arbitration (`ref[dsp.zoh.ctrl-stride]`,
//! `ref[dsp.ctrl.op-m.ballistic-lfo]`, `ref[dsp.ctrl.op-m.rate-retime]`,
//! `ref[dsp.ctrl.op-m.spsc-ingress]`, `ref[dsp.ctrl.op-m.timing]`,
//! `ref[dsp.ctrl.dispatch.matrix]`, `ref[dsp.ctrl.dispatch.micro-ramp]`,
//! `ref[dsp.ctrl.switch.conditional-batch]`).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::schema::{DnamError, reject_if};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::control::ControlBus;
    use crate::math::ScalarBackend;
    use alloc::vec;

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
}
