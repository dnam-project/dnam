//! Consumer endpoint (`ControlConsumer`) of the 3-Lane SPSC control ingress
//! (`ref[dsp.ctrl.op-m.spsc-ingress]`).

use alloc::sync::Arc;
use core::sync::atomic::{Ordering, fence};

use crate::dynamic::DynamicDnamEngine;

use super::ring::{
    ControlCommand, ControlUpdateMode, ControlValueKind, FALLBACK_WORDS, FallbackSnapshot,
    Lane1Entry, MAX_CONTROLS, SPSC_RING_MASK, SpscControlRing, unpack_lane1_slot,
};

/// Tracks any explicit `offline_bounce` or `maximize_tier_on_bounce` transitions drained during
/// an offline bounce so temporary bounce overrides are never confused with user state changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrainedBounceOverrides {
    /// Latest explicit `SetOfflineBounce` (or `FallbackSnapshot::offline_bounce`) drained, if any.
    pub offline_bounce: Option<bool>,
    /// Latest explicit `SetMaximizeTierOnBounce` (or `FallbackSnapshot::maximize_tier_on_bounce`) drained, if any.
    pub maximize_tier_on_bounce: Option<bool>,
}

/// Consumer endpoint of the 3-Lane SPSC control ingress (`ref[dsp.ctrl.op-m.spsc-ingress]`).
///
/// Owned by the real-time audio context and drained at the start of every audio block with zero
/// heap allocations and zero locks.
#[derive(Debug)]
pub struct ControlConsumer {
    pub(crate) ring: Arc<SpscControlRing>,
    pub(crate) drained_causal_epoch: u64,
}

impl ControlConsumer {
    /// Returns `true` if any of the 3 SPSC lanes has pending control updates.
    #[inline]
    pub fn has_pending(&self) -> bool {
        if self.ring.fallback_epoch.load(Ordering::Acquire) != 0 {
            return true;
        }
        let tail = self.ring.tail.value.load(Ordering::Relaxed);
        let head = self.ring.head.value.load(Ordering::Acquire);
        if head != tail {
            return true;
        }
        for s in &self.ring.lane1_slots {
            if s.load(Ordering::Acquire) != 0 {
                return true;
            }
        }
        for s in &self.ring.lane1_ext {
            if s.load(Ordering::Acquire) != 0 {
                return true;
            }
        }
        false
    }

    /// Returns the last causal epoch drained into the audio engine.
    #[inline]
    pub const fn drained_causal_epoch(&self) -> u64 {
        self.drained_causal_epoch
    }

    /// Advances `drained_causal_epoch` to at least `epoch` (e.g., when a staged engine swap is
    /// activated or after a synchronous sample-rate reconfiguration).
    #[inline]
    pub fn advance_causal_epoch(&mut self, epoch: u64) {
        if epoch > self.drained_causal_epoch {
            self.drained_causal_epoch = epoch;
            self.ring
                .drained_causal_epoch
                .value
                .store(self.drained_causal_epoch, Ordering::Release);
        }
    }

    /// Publishes the engine's current `tempo_total_frames` to the shared atomic counter.
    #[inline]
    pub fn publish_total_frames(&self, engine: &DynamicDnamEngine<'_>) {
        self.ring
            .published_total_frames
            .value
            .store(engine.tempo_total_frames(), Ordering::Release);
    }

    /// Drains Lane 3 (overflow fallback latch), Lane 2 (64-slot SPSC ring), and Lane 1
    /// (coalescing parameter mailbox) in deterministic causal epoch order into `engine`.
    ///
    /// Returns the number of discrete and parameter updates applied.
    #[inline]
    pub fn drain_into(&mut self, engine: &mut DynamicDnamEngine<'_>) -> usize {
        self.drain_into_tracking_bounce(engine).0
    }

    /// Drains all 3 SPSC lanes in causal epoch order into `engine` and returns both the count of
    /// applied updates and any explicit bounce-mode overrides encountered.
    pub fn drain_into_tracking_bounce(
        &mut self,
        engine: &mut DynamicDnamEngine<'_>,
    ) -> (usize, DrainedBounceOverrides) {
        let mut applied_count = 0_usize;
        let mut max_drained_epoch_29 = 0_u32;
        let mut bounce_overrides = DrainedBounceOverrides::default();

        // 1. Drain Lane 3 (Overflow-Proof Full-Reset Epoch Latch) if active.
        let mut min_epoch_after_latch = 0_u32;
        let fb_opt = self.take_lane3_fallback();
        if let Some(ref fb) = fb_opt {
            min_epoch_after_latch = fb.epoch;
            max_drained_epoch_29 = max_drained_epoch_29.max(fb.epoch);
            bounce_overrides.offline_bounce = Some(fb.offline_bounce);
            bounce_overrides.maximize_tier_on_bounce = Some(fb.maximize_tier_on_bounce);
        }

        // 2. Collect active Lane 1 inline (`0..32`) and extended (`32..256`) slots on the stack
        //    and sort by ascending epoch.
        let default_entry = Lane1Entry {
            epoch: 0,
            mode: ControlUpdateMode::Slewed,
            kind: ControlValueKind::Range,
            payload_bits: 0,
        };
        let mut lane1_buf = [(0_u16, default_entry); 256];
        let mut lane1_len = 0_usize;
        let mut ext_fb_buf = [(0_u16, default_entry); 256 - MAX_CONTROLS];
        let mut ext_fb_len = 0_usize;

        for (idx, slot) in self.ring.lane1_slots.iter().enumerate() {
            let word = slot.swap(0, Ordering::AcqRel);
            if let Some(entry) = unpack_lane1_slot(word) {
                max_drained_epoch_29 = max_drained_epoch_29.max(entry.epoch);
                if entry.epoch > min_epoch_after_latch {
                    lane1_buf[lane1_len] = (idx as u16, entry);
                    lane1_len += 1;
                }
            }
        }
        for (ext_i, slot) in self.ring.lane1_ext.iter().enumerate() {
            let word = slot.swap(0, Ordering::AcqRel);
            if let Some(entry) = unpack_lane1_slot(word) {
                let ctrl_idx = (MAX_CONTROLS + ext_i) as u16;
                max_drained_epoch_29 = max_drained_epoch_29.max(entry.epoch);
                if entry.epoch > min_epoch_after_latch {
                    if lane1_len < lane1_buf.len() {
                        lane1_buf[lane1_len] = (ctrl_idx, entry);
                        lane1_len += 1;
                    }
                } else if fb_opt.is_some() && ext_fb_len < ext_fb_buf.len() {
                    ext_fb_buf[ext_fb_len] = (ctrl_idx, entry);
                    ext_fb_len += 1;
                }
            }
        }

        if let Some(ref fb) = fb_opt {
            self.apply_fallback_snapshot(engine, fb, &ext_fb_buf[..ext_fb_len]);
            applied_count += 1;
        }

        // Insertion sort stack buffer by ascending `epoch` (k <= 256, zero heap allocation).
        for i in 1..lane1_len {
            let key = lane1_buf[i];
            let mut j = i;
            while j > 0 && lane1_buf[j - 1].1.epoch > key.1.epoch {
                lane1_buf[j] = lane1_buf[j - 1];
                j -= 1;
            }
            lane1_buf[j] = key;
        }

        // 3. Two-way causal merge of Lane 2 FIFO ring and sorted Lane 1 stack buffer.
        let mut tail = self.ring.tail.value.load(Ordering::Relaxed);
        let head = self.ring.head.value.load(Ordering::Acquire);
        let mut l1_idx = 0_usize;

        while tail != head || l1_idx < lane1_len {
            let next_l2 = if tail != head {
                Some(self.ring.lane2_slots[tail & SPSC_RING_MASK].load())
            } else {
                None
            };

            if let Some(l2_entry) = next_l2
                && l2_entry.epoch <= min_epoch_after_latch
            {
                max_drained_epoch_29 = max_drained_epoch_29.max(l2_entry.epoch);
                tail = tail.wrapping_add(1);
                continue;
            }

            let take_l1 = match (next_l2, l1_idx < lane1_len) {
                (Some(l2), true) => lane1_buf[l1_idx].1.epoch < l2.epoch,
                (None, true) => true,
                (Some(_), false) => false,
                (None, false) => break,
            };

            if take_l1 {
                let (ctrl_idx, entry) = lane1_buf[l1_idx];
                l1_idx += 1;
                self.apply_lane1_entry(engine, ctrl_idx as usize, entry);
                applied_count += 1;
            } else if let Some(l2_entry) = next_l2 {
                tail = tail.wrapping_add(1);
                max_drained_epoch_29 = max_drained_epoch_29.max(l2_entry.epoch);
                match l2_entry.cmd {
                    ControlCommand::SetOfflineBounce { enabled } => {
                        bounce_overrides.offline_bounce = Some(enabled);
                    }
                    ControlCommand::SetMaximizeTierOnBounce { enabled } => {
                        bounce_overrides.maximize_tier_on_bounce = Some(enabled);
                    }
                    _ => {}
                }
                self.apply_lane2_command(engine, l2_entry.cmd);
                applied_count += 1;
            }
        }

        self.ring.tail.value.store(tail, Ordering::Release);

        if max_drained_epoch_29 > 0 {
            let base = self.ring.epoch_base.load(Ordering::Acquire);
            let target_causal = base.wrapping_add(max_drained_epoch_29 as u64);
            self.advance_causal_epoch(target_causal);
        }
        self.publish_total_frames(engine);
        (applied_count, bounce_overrides)
    }

    fn take_lane3_fallback(&self) -> Option<FallbackSnapshot> {
        let ep = self.ring.fallback_epoch.load(Ordering::Acquire);
        if ep == 0 {
            return None;
        }
        loop {
            let s1 = self.ring.fallback_seq.value.load(Ordering::Acquire);
            if (s1 & 1) != 0 {
                core::hint::spin_loop();
                continue;
            }
            let mut words = [0_u64; FALLBACK_WORDS];
            for (dst, slot) in words.iter_mut().zip(self.ring.fallback_words.iter()) {
                *dst = slot.load(Ordering::Relaxed);
            }
            fence(Ordering::Acquire);
            let s2 = self.ring.fallback_seq.value.load(Ordering::Relaxed);
            if s1 == s2 {
                let snap = FallbackSnapshot::from_words(&words);
                if snap.epoch == 0 {
                    return None;
                }
                if self
                    .ring
                    .fallback_epoch
                    .compare_exchange(snap.epoch, 0, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    self.ring.fallback_reset_engine.store(0, Ordering::Release);
                }
                return Some(snap);
            }
            core::hint::spin_loop();
        }
    }

    fn apply_fallback_snapshot(
        &self,
        engine: &mut DynamicDnamEngine<'_>,
        fb: &FallbackSnapshot,
        ext_entries: &[(u16, Lane1Entry)],
    ) {
        if fb.reset_engine_requested {
            engine.reset_to_dc_equilibrium();
        }
        engine.set_max_supported_tier(fb.max_supported_tier);
        engine.set_tier(fb.tier);
        engine.set_offline_bounce(fb.offline_bounce);
        engine.set_maximize_tier_on_bounce(fb.maximize_tier_on_bounce);
        engine.set_bypass(fb.bypassed, fb.trails_enabled);
        engine.set_bpm(fb.current_bpm);
        if let Some((blend_wet, output_gain_db, apply_unity)) = fb.host_mix {
            engine.configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity);
        }
        let n = (fb.num_controls as usize)
            .min(MAX_CONTROLS)
            .min(engine.num_controls());
        for idx in 0..n {
            let is_sw = engine
                .control_descriptors()
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 1);
            if is_sw {
                engine.set_control_immediate(idx, fb.raw_values[idx]);
            }
        }
        for &(ctrl_idx, entry) in ext_entries {
            let idx = ctrl_idx as usize;
            if engine
                .control_descriptors()
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 1)
            {
                self.apply_lane1_entry(engine, idx, entry);
            }
        }
        for idx in 0..n {
            let is_sw = engine
                .control_descriptors()
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 1);
            if !is_sw {
                if (fb.immediate_mask & (1_u32 << idx)) != 0 {
                    engine.set_control_immediate(idx, fb.raw_values[idx]);
                } else {
                    engine.set_control_value(idx, fb.raw_values[idx]);
                }
            }
        }
        for &(ctrl_idx, entry) in ext_entries {
            let idx = ctrl_idx as usize;
            if !engine
                .control_descriptors()
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 1)
            {
                self.apply_lane1_entry(engine, idx, entry);
            }
        }
        engine.set_freeze(fb.is_frozen);
        engine.set_stereo_width(fb.stereo_width);
        engine.or_active_clamp_mask(fb.tempo_clamp_mask);
    }

    #[inline]
    fn apply_lane1_entry(
        &self,
        engine: &mut DynamicDnamEngine<'_>,
        ctrl_idx: usize,
        entry: Lane1Entry,
    ) {
        let val = entry.physical_value();
        match entry.mode {
            ControlUpdateMode::Slewed => engine.set_control_value(ctrl_idx, val),
            ControlUpdateMode::Immediate => engine.set_control_immediate(ctrl_idx, val),
        }
    }

    fn apply_lane2_command(&self, engine: &mut DynamicDnamEngine<'_>, cmd: ControlCommand) {
        match cmd {
            ControlCommand::ApplyPreset { preset_index } => {
                let _ = engine.apply_preset_by_index_rt(preset_index as usize);
            }
            ControlCommand::SyncTempo {
                bpm,
                subdivision,
                fold_octave,
            } => {
                let _ = engine.sync_tempo(bpm, subdivision, fold_octave);
            }
            ControlCommand::TapTempo {
                resolved_bpm,
                subdivision,
                fold_octave,
            } => {
                let _ = engine.sync_tempo(resolved_bpm, subdivision, fold_octave);
            }
            ControlCommand::SetBypass { bypassed, trails } => {
                engine.set_bypass(bypassed, trails);
            }
            ControlCommand::SetFreeze { frozen } => {
                engine.set_freeze(frozen);
            }
            ControlCommand::SetFreezeAmount { alpha } => {
                engine.set_freeze_amount(alpha);
            }
            ControlCommand::SetStereoWidth { width } => {
                engine.set_stereo_width(width);
            }
            ControlCommand::ConfigureHostMix {
                blend_wet,
                output_gain_db,
                apply_unity_leveling,
            } => {
                engine.configure_host_mix_and_leveling(
                    blend_wet,
                    output_gain_db,
                    apply_unity_leveling,
                );
            }
            ControlCommand::SetTier { tier } => {
                engine.set_tier(tier);
            }
            ControlCommand::SetMaxSupportedTier { tier } => {
                engine.set_max_supported_tier(tier);
            }
            ControlCommand::SetOfflineBounce { enabled } => {
                engine.set_offline_bounce(enabled);
            }
            ControlCommand::SetMaximizeTierOnBounce { enabled } => {
                engine.set_maximize_tier_on_bounce(enabled);
            }
            ControlCommand::ResetControl { index } => {
                let idx = index as usize;
                if let Some(def_val) = engine.control_descriptors().get(idx).map(|d| d.default_val)
                {
                    engine.set_control_immediate(idx, def_val);
                }
            }
            ControlCommand::ResetAllControls => {
                engine.reset_controls_to_defaults();
            }
            ControlCommand::ResetEngine => {
                engine.reset_to_dc_equilibrium();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::host::spsc::spsc_control_channel;
    use crate::math::BackendKind;
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_consumer_test_container(num_controls: usize) -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        let mut controls = Vec::with_capacity(num_controls);
        for i in 0..num_controls {
            if i == 1 || i == 34 {
                controls.push(ControlSchemaDef {
                    id: alloc::format!("sw_ctrl_{i}"),
                    name: alloc::format!("Switch {i}"),
                    kind: ControlKind::Switch,
                    unit: None,
                    lobatto_degree: None,
                    anchor_nodes_norm: None,
                    declared_range: None,
                    safe_clamped_range: None,
                    default: serde_json::json!("A"),
                    conditional_min: BTreeMap::new(),
                    conditional_max: BTreeMap::new(),
                    declared_options: Some(vec![String::from("A"), String::from("B")]),
                    active_options: Some(vec![String::from("A"), String::from("B")]),
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                });
            } else {
                controls.push(ControlSchemaDef {
                    id: alloc::format!("ctrl_{i}"),
                    name: alloc::format!("Ctrl {i}"),
                    kind: ControlKind::Range,
                    unit: None,
                    lobatto_degree: Some(2),
                    anchor_nodes_norm: None,
                    declared_range: Some([0.0, 10.0]),
                    safe_clamped_range: Some([0.0, 10.0]),
                    default: serde_json::json!(2.0),
                    conditional_min: BTreeMap::new(),
                    conditional_max: BTreeMap::new(),
                    declared_options: None,
                    active_options: None,
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                });
            }
        }
        header.controls_schema = controls;
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_consumer_insertion_sort_extended_controls_and_lane3_fallback() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_consumer_test_container(36);
        let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
            &container,
            Tier::Ultra,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        let (mut producer, mut consumer) = spsc_control_channel(&engine);

        assert!(!consumer.has_pending());
        // Update controls in descending index order so slot 0 has a higher epoch than slot 3
        // (exercises the insertion sort swap loop in drain_into_tracking_bounce)
        producer.set_control_value(3, 4.0);
        producer.set_control_immediate(2, 5.0);
        producer.set_control_immediate(1, 1.0);
        producer.set_control_value(0, 6.0);
        assert!(consumer.has_pending());
        assert_eq!(consumer.drain_into(&mut engine), 4);
        assert!(!consumer.has_pending());

        // Extended Lane 1 slots (`idx >= 32`)
        producer.set_control_value(33, 7.5);
        producer.set_control_immediate(34, 1.0);
        assert!(consumer.has_pending());
        consumer.drain_into(&mut engine);
        assert!((engine.clamped_control_values()[33] - 7.5).abs() < 1e-6);
        assert!((engine.clamped_control_values()[34] - 1.0).abs() < 1e-6);

        // Overflow Lane 2 (> 64 commands) with early reset_to_dc_equilibrium (at i == 5, before
        // ring overflow at i == 64), extended controls (idx >= 32), host_mix, and bounce flags
        producer.configure_host_mix_and_leveling(0.6, -2.0, true);
        producer.set_offline_bounce(true);
        producer.set_maximize_tier_on_bounce(false);
        producer.set_control_immediate(0, 8.0);
        producer.set_control_value(2, 3.0);
        producer.set_control_value(32, 6.25);
        producer.set_control_immediate(34, 0.0);
        producer.set_control_immediate(35, 4.5);
        for i in 0..70 {
            if i == 5 {
                producer.reset_to_dc_equilibrium();
            } else {
                producer.set_bypass(i % 2 == 0, true);
            }
        }
        assert!(consumer.has_pending());
        assert_eq!(
            producer.ring.fallback_reset_engine.load(Ordering::Acquire),
            1
        );
        // Latch stereo_width into Lane 3, then queue post-latch Lane 1 updates (both inline and extended)
        producer.set_stereo_width(1.3);
        producer.set_control_value(0, 9.0);
        producer.set_control_value(33, 8.75);
        let (applied, overrides) = consumer.drain_into_tracking_bounce(&mut engine);
        assert_eq!(applied, 3);
        assert_eq!(overrides.offline_bounce, Some(true));
        assert_eq!(overrides.maximize_tier_on_bounce, Some(false));
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );
        assert!((engine.stereo_width() - 1.3).abs() < 1e-6);

        // Simultaneous 29-bit epoch rollover (MAX_EPOCH_29 -> 2) with extended Lane 1 controls
        producer.set_control_value(32, 3.25);
        producer.set_control_immediate(34, 1.0);
        producer.next_epoch_29 = crate::host::spsc::ring::MAX_EPOCH_29;
        producer.set_control_value(35, 9.5);
        producer.set_control_value(0, 4.25);
        consumer.drain_into(&mut engine);
        assert_eq!(consumer.drained_causal_epoch(), producer.causal_epoch());
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_consumer_drains_all_lanes_to_exact_shadow_parity(
            ops in proptest::collection::vec((0u8..16, 0.0f32..10.0f32), 1..80)
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            let container = build_consumer_test_container(36);
            let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Ultra,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();
            let (mut producer, mut consumer) = spsc_control_channel(&engine);

            for (op, val) in ops {
                match op {
                    0 => producer.set_control_value(0, val),
                    1 => producer.set_control_immediate(2, val),
                    2 => producer.set_control_immediate(1, ((val as u32) % 2) as f32),
                    3 => { let _ = producer.reset_control(0); }
                    4 => producer.reset_controls_to_defaults(),
                    5 => producer.reset_to_dc_equilibrium(),
                    6 => producer.set_bypass(val > 5.0, val > 2.5),
                    7 => producer.set_freeze(val > 5.0),
                    8 => producer.set_freeze_amount(val * 0.1),
                    9 => producer.set_stereo_width(val * 0.2),
                    10 => producer.set_tier(Tier::from_u8((val as u8) % 4)),
                    11 => producer.set_max_supported_tier(Tier::from_u8(((val as u8) % 3) + 1)),
                    12 => producer.set_control_value(33, val),
                    13 => producer.set_control_immediate(34, ((val as u32) % 2) as f32),
                    14 => { let _ = producer.reset_control(33); }
                    _ => {
                        producer.next_epoch_29 = crate::host::spsc::ring::MAX_EPOCH_29;
                        producer.set_control_value(35, val);
                    }
                }
            }

            consumer.drain_into(&mut engine);
            prop_assert_eq!(consumer.drained_causal_epoch(), producer.causal_epoch());
            prop_assert_eq!(
                engine.clamped_control_values(),
                producer.shadow().clamped_control_values()
            );
            prop_assert_eq!(engine.tier(), producer.shadow().tier());
            prop_assert_eq!(
                engine.max_supported_tier(),
                producer.shadow().max_supported_tier()
            );
            prop_assert_eq!(engine.bypass_state(), producer.shadow().bypass_state());
            prop_assert_eq!(engine.is_frozen(), producer.shadow().is_frozen());
            prop_assert!((engine.stereo_width() - producer.shadow().stereo_width()).abs() < 1e-6);
        }
    }
}
