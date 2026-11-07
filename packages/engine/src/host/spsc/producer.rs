//! Producer endpoint (`ControlProducer`) of the 3-Lane SPSC control ingress
//! (`ref[dsp.ctrl.op-m.spsc-ingress]`).

use alloc::string::String;
use alloc::sync::Arc;
use core::sync::atomic::{Ordering, fence};

use crate::container::ZohRescaledParams;
use crate::dynamic::DynamicDnamEngine;
use crate::engine::{TempoSyncState, resolve_preset_batch_by_id};
use crate::schema::{DnamError, TempoSubdivision};
use crate::types::Tier;

use super::ring::{
    ControlCommand, ControlUpdateMode, ControlValueKind, MAX_CONTROLS, MAX_EPOCH_29, RingEntry,
    SPSC_RING_CAPACITY, SPSC_RING_MASK, SpscControlRing, pack_lane1_slot,
};
use super::shadow::AcceptedStateShadow;

/// Producer endpoint of the 3-Lane SPSC control ingress (`ref[dsp.ctrl.op-m.spsc-ingress]`).
#[derive(Debug)]
pub struct ControlProducer {
    pub(crate) ring: Arc<SpscControlRing>,
    pub(crate) shadow: AcceptedStateShadow,
    pub(crate) next_epoch_29: u32,
    pub(crate) causal_epoch: u64,
}

impl ControlProducer {
    /// Returns an immutable reference to the synchronous [`AcceptedStateShadow`].
    #[inline]
    pub const fn shadow(&self) -> &AcceptedStateShadow {
        &self.shadow
    }

    /// Returns the monotonic 64-bit causal epoch of the producer.
    #[inline]
    pub const fn causal_epoch(&self) -> u64 {
        self.causal_epoch
    }

    /// Returns the monotonic 64-bit causal epoch last drained by the audio thread.
    #[inline]
    pub fn drained_causal_epoch(&self) -> u64 {
        self.ring.drained_causal_epoch.value.load(Ordering::Acquire)
    }

    /// Returns the number of times the Lane 3 overflow fallback latch has been activated.
    #[inline]
    pub fn overflow_latch_count(&self) -> usize {
        self.ring.overflow_latch_count.load(Ordering::Acquire)
    }

    /// Re-synchronizes the producer shadow with `engine` (e.g. after model reload or sample-rate change).
    pub fn resync_from_engine(&mut self, engine: &DynamicDnamEngine<'_>) {
        self.ring.clear_all_lane1();
        let _ = self.ring.tombstone_unconsumed_lane2();
        let h = self.ring.head.value.load(Ordering::Relaxed);
        self.ring.tail.value.store(h, Ordering::Release);
        self.ring.fallback_epoch.store(0, Ordering::Release);
        self.ring.fallback_reset_engine.store(0, Ordering::Release);
        self.ring
            .published_total_frames
            .value
            .store(engine.tempo_total_frames(), Ordering::Release);
        self.shadow = AcceptedStateShadow::from_engine(engine);
        self.causal_epoch = self.causal_epoch.wrapping_add(1);
        self.ring
            .epoch_base
            .store(self.causal_epoch, Ordering::Release);
        self.next_epoch_29 = 1;
    }

    #[inline]
    pub(crate) fn allocate_epoch(&mut self) -> u32 {
        self.causal_epoch = self.causal_epoch.wrapping_add(1);
        let ep = self.next_epoch_29;
        if ep >= MAX_EPOCH_29 {
            self.latch_into_lane3(1, false);
            self.ring
                .epoch_base
                .store(self.causal_epoch.wrapping_sub(2), Ordering::Release);
            self.next_epoch_29 = 3;
            2
        } else {
            self.next_epoch_29 = ep + 1;
            ep
        }
    }

    pub(crate) fn latch_into_lane3(&mut self, epoch: u32, reset_engine: bool) {
        let saw_unconsumed_reset = self.ring.tombstone_unconsumed_lane2();
        if reset_engine || saw_unconsumed_reset {
            self.ring.fallback_reset_engine.store(1, Ordering::Release);
        }
        let reset_latched = self.ring.fallback_reset_engine.load(Ordering::Relaxed) != 0;
        let snap = self.shadow.to_fallback_snapshot(epoch, reset_latched);
        let packed = snap.to_words();
        let s0 = self.ring.fallback_seq.value.load(Ordering::Relaxed);
        let s_odd = s0.wrapping_add(1) | 1;
        self.ring.fallback_seq.value.store(s_odd, Ordering::Release);
        fence(Ordering::AcqRel);
        for (slot, &word) in self.ring.fallback_words.iter().zip(packed.iter()) {
            slot.store(word, Ordering::Relaxed);
        }
        self.ring.fallback_epoch.store(epoch, Ordering::Release);
        self.ring
            .fallback_seq
            .value
            .store(s_odd.wrapping_add(1), Ordering::Release);
        self.ring.clear_all_lane1();
        for idx in MAX_CONTROLS..self.shadow.num_controls {
            let mode = if self.shadow.is_control_immediate(idx) {
                ControlUpdateMode::Immediate
            } else {
                ControlUpdateMode::Slewed
            };
            self.write_lane1_slot(idx, epoch, self.shadow.raw_values[idx], mode);
        }
        self.ring
            .overflow_latch_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn push_lane2_or_latch(&mut self, epoch: u32, cmd: ControlCommand) {
        let head = self.ring.head.value.load(Ordering::Relaxed);
        let tail = self.ring.tail.value.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= SPSC_RING_CAPACITY {
            let is_reset_engine = matches!(cmd, ControlCommand::ResetEngine);
            self.latch_into_lane3(epoch, is_reset_engine);
            return;
        }
        self.ring.lane2_slots[head & SPSC_RING_MASK].store(RingEntry { epoch, cmd });
        self.ring
            .head
            .value
            .store(head.wrapping_add(1), Ordering::Release);
    }

    #[inline]
    fn write_lane1_slot(
        &self,
        ctrl_idx: usize,
        epoch: u32,
        physical_value: f32,
        mode: ControlUpdateMode,
    ) {
        let Some(slot) = self.ring.lane1_slot(ctrl_idx) else {
            return;
        };
        let is_switch = self
            .shadow
            .descriptors
            .get(ctrl_idx)
            .is_some_and(|d| d.ctrl_kind == 1);
        let (kind, payload_bits) = if is_switch
            && physical_value >= 0.0
            && (physical_value as u32 as f32) == physical_value
        {
            (ControlValueKind::Switch, physical_value as u32)
        } else {
            (ControlValueKind::Range, physical_value.to_bits())
        };
        let word = pack_lane1_slot(epoch, mode, kind, payload_bits);
        slot.store(word, Ordering::Release);
    }

    /// Validates `ctrl_idx` and `physical_value` synchronously, updates [`AcceptedStateShadow`],
    /// and coalesces the target into Lane 1 (`ControlUpdateMode::Slewed`).
    pub fn try_set_control_value(
        &mut self,
        ctrl_idx: usize,
        physical_value: f32,
    ) -> Result<f32, DnamError> {
        if ctrl_idx >= self.shadow.num_controls {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "control index {ctrl_idx} out of range (num_controls={})",
                self.shadow.num_controls
            )));
        }
        if !physical_value.is_finite() {
            return Err(DnamError::SchemaValidationError(String::from(
                "control physical_value must be finite",
            )));
        }
        self.set_control_value(ctrl_idx, physical_value);
        Ok(self.shadow.clamped_values[ctrl_idx])
    }

    /// Validates `ctrl_idx` and `physical_value` synchronously, updates [`AcceptedStateShadow`],
    /// and coalesces the target into Lane 1 (`ControlUpdateMode::Immediate`).
    pub fn try_set_control_immediate(
        &mut self,
        ctrl_idx: usize,
        physical_value: f32,
    ) -> Result<f32, DnamError> {
        if ctrl_idx >= self.shadow.num_controls {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "control index {ctrl_idx} out of range (num_controls={})",
                self.shadow.num_controls
            )));
        }
        if !physical_value.is_finite() {
            return Err(DnamError::SchemaValidationError(String::from(
                "control physical_value must be finite",
            )));
        }
        self.set_control_immediate(ctrl_idx, physical_value);
        Ok(self.shadow.clamped_values[ctrl_idx])
    }

    /// Sets a user control value with smooth $3\text{ kHz}$ interpolation via Lane 1 coalescing mailbox.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.shadow.num_controls {
            return;
        }
        let epoch = self.allocate_epoch();
        self.shadow
            .apply_control_update(ctrl_idx, physical_value, false);
        self.write_lane1_slot(ctrl_idx, epoch, physical_value, ControlUpdateMode::Slewed);
    }

    /// Sets a user control value and snaps immediately via Lane 1 coalescing mailbox.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        if ctrl_idx >= self.shadow.num_controls {
            return;
        }
        let epoch = self.allocate_epoch();
        self.shadow
            .apply_control_update(ctrl_idx, physical_value, true);
        self.write_lane1_slot(
            ctrl_idx,
            epoch,
            physical_value,
            ControlUpdateMode::Immediate,
        );
    }

    /// Applies a batch of immediate control updates (`ref[dsp.ctrl.switch.conditional-batch]`),
    /// ordering switch controls before continuous range controls.
    pub fn set_controls_immediate_batch(&mut self, updates: &[(usize, f32)]) {
        for &(idx, val) in updates {
            if self
                .shadow
                .descriptors
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 1)
            {
                self.set_control_immediate(idx, val);
            }
        }
        for &(idx, val) in updates {
            if self
                .shadow
                .descriptors
                .get(idx)
                .is_some_and(|d| d.ctrl_kind == 0)
            {
                self.set_control_immediate(idx, val);
            }
        }
    }

    /// Resets a single control `ctrl_idx` immediately back to its schema default.
    pub fn reset_control(&mut self, ctrl_idx: usize) -> Result<f32, DnamError> {
        if ctrl_idx >= self.shadow.num_controls {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "control index {ctrl_idx} out of range (num_controls={})",
                self.shadow.num_controls
            )));
        }
        let epoch = self.allocate_epoch();
        self.shadow.reset_single_control(ctrl_idx);
        if let Some(slot) = self.ring.lane1_slot(ctrl_idx) {
            slot.store(0, Ordering::Release);
        }
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::ResetControl {
                index: ctrl_idx as u16,
            },
        );
        Ok(self.shadow.clamped_values[ctrl_idx])
    }

    /// Snaps all controls immediately back to their schema defaults (`reset_controls_to_defaults`).
    pub fn reset_controls_to_defaults(&mut self) {
        let epoch = self.allocate_epoch();
        self.shadow.reset_all_controls();
        self.ring.clear_all_lane1();
        self.push_lane2_or_latch(epoch, ControlCommand::ResetAllControls);
    }

    /// Enqueues a full analytical DC equilibrium reset (`reset_to_dc_equilibrium`).
    pub fn reset_to_dc_equilibrium(&mut self) {
        let epoch = self.allocate_epoch();
        self.shadow.reset_engine();
        self.ring
            .published_total_frames
            .value
            .store(0, Ordering::Release);
        self.push_lane2_or_latch(epoch, ControlCommand::ResetEngine);
    }

    /// Validates `preset_id` synchronously against `DnamJsonHeader::presets`, updates
    /// [`AcceptedStateShadow`], and enqueues `ControlCommand::ApplyPreset` in Lane 2.
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        let batch = resolve_preset_batch_by_id(
            self.shadow.header.as_ref(),
            &self.shadow.descriptors,
            preset_id,
        )?;
        let epoch = self.allocate_epoch();
        for &(idx, val, is_sw) in &batch.default_restores[..batch.restore_count] {
            self.shadow.apply_control_update(idx, val, is_sw);
        }
        for &(idx, val) in &batch.switch_updates[..batch.switch_count] {
            self.shadow.apply_control_update(idx, val, true);
        }
        for &(idx, val) in &batch.range_updates[..batch.range_count] {
            self.shadow.apply_control_update(idx, val, false);
        }
        self.ring.clear_all_lane1();
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::ApplyPreset {
                preset_index: batch.preset_index,
            },
        );
        Ok(())
    }

    /// Synchronizes tempo-calibrated controls to `bpm` and optional `subdivision_override`.
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        let epoch = self.allocate_epoch();
        let count = self.apply_sync_tempo_to_shadow(bpm, subdivision_override, fold_octave);
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::SyncTempo {
                bpm,
                subdivision: subdivision_override,
                fold_octave,
            },
        );
        count
    }

    /// Registers a tap-tempo event using the latest audio frame counter, updates BPM, and syncs controls.
    pub fn register_tap_tempo(
        &mut self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        let published_frames = self
            .ring
            .published_total_frames
            .value
            .load(Ordering::Acquire);
        if published_frames > self.shadow.tempo_state.total_frames {
            self.shadow.tempo_state.total_frames = published_frames;
        }
        let bpm = self.shadow.tempo_state.register_tap();
        let epoch = self.allocate_epoch();
        let count = self.apply_sync_tempo_to_shadow(bpm, subdivision_override, fold_octave);
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::TapTempo {
                resolved_bpm: bpm,
                subdivision: subdivision_override,
                fold_octave,
            },
        );
        (bpm, count)
    }

    fn apply_sync_tempo_to_shadow(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        self.shadow.tempo_state.set_bpm(bpm);
        let mut updates = [(0usize, 0.0f32); 512];
        let mut tempo_clamp_mask = 0u16;
        let count = if let Some(ref hdr) = self.shadow.header {
            TempoSyncState::resolve_sync_updates_with_clamp_mask_into(
                hdr,
                self.shadow.tempo_state.current_bpm,
                subdivision_override,
                fold_octave,
                &mut updates,
                &mut tempo_clamp_mask,
            )
        } else {
            0
        };
        for &(idx, val) in &updates[..count] {
            self.shadow.apply_control_update(idx, val, true);
            if let Some(slot) = self.ring.lane1_slot(idx) {
                slot.store(0, Ordering::Release);
            }
        }
        self.shadow.tempo_clamp_mask = tempo_clamp_mask;
        self.shadow.active_clamp_mask |= tempo_clamp_mask;
        count
    }

    /// Sets bypass state and spillover trails flag via Lane 2.
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        let epoch = self.allocate_epoch();
        self.shadow.bypassed = bypassed;
        self.shadow.trails_enabled = trails;
        self.push_lane2_or_latch(epoch, ControlCommand::SetBypass { bypassed, trails });
    }

    /// Engages or disengages Unitary Spectral Freeze via Lane 2.
    pub fn set_freeze(&mut self, freeze: bool) {
        let epoch = self.allocate_epoch();
        self.shadow.set_freeze(freeze);
        self.push_lane2_or_latch(epoch, ControlCommand::SetFreeze { frozen: freeze });
    }

    /// Sets continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$ via Lane 2.
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        let epoch = self.allocate_epoch();
        self.shadow.set_freeze_amount(alpha);
        self.push_lane2_or_latch(epoch, ControlCommand::SetFreezeAmount { alpha });
    }

    /// Sets Mid/Side stereo width $\alpha_{\text{width}} \in [0.0, 2.0]$ via Lane 2.
    pub fn set_stereo_width(&mut self, width: f32) {
        let epoch = self.allocate_epoch();
        self.shadow.stereo_width = width.clamp(0.0, 2.0);
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::SetStereoWidth {
                width: self.shadow.stereo_width,
            },
        );
    }

    /// Configures host dry/wet blend, output gain trim (dB), and unity loudness leveling via Lane 2.
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        let epoch = self.allocate_epoch();
        self.shadow.host_mix = Some((blend_wet, output_gain_db, apply_unity_leveling));
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::ConfigureHostMix {
                blend_wet,
                output_gain_db,
                apply_unity_leveling,
            },
        );
    }

    /// Switches the active compute [`Tier`] via Lane 2.
    pub fn set_tier(&mut self, tier: Tier) {
        let epoch = self.allocate_epoch();
        self.shadow.tier = tier
            .min(self.shadow.max_available_tier)
            .min(self.shadow.max_supported_tier);
        self.push_lane2_or_latch(epoch, ControlCommand::SetTier { tier });
    }

    /// Sets the host ceiling [`Tier`] (`max_supported_tier`) via Lane 2.
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        let epoch = self.allocate_epoch();
        self.shadow.max_supported_tier = max_tier;
        if self.shadow.tier > max_tier {
            self.shadow.tier = max_tier.min(self.shadow.max_available_tier);
        }
        self.push_lane2_or_latch(
            epoch,
            ControlCommand::SetMaxSupportedTier { tier: max_tier },
        );
    }

    /// Enables or disables $4\times$ oversampled Offline Bounce mode via Lane 2.
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        if self.shadow.offline_bounce == enabled {
            return;
        }
        let epoch = self.allocate_epoch();
        self.shadow.offline_bounce = enabled;
        let oversample_factor = if enabled { 4 } else { 1 };
        let (r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr) = self.shadow.training_zoh_poles;
        if let Ok(new_zoh) = ZohRescaledParams::compute_oversampled(
            self.shadow.sample_rate_train_hz,
            self.shadow.sample_rate_hz,
            oversample_factor,
            r_ac_tr,
            r_dc_tr,
            tau_ref_tr,
            switch_xfade_tr,
        ) {
            self.shadow.zoh = new_zoh;
        }
        self.push_lane2_or_latch(epoch, ControlCommand::SetOfflineBounce { enabled });
    }

    /// Enables or disables automatic tier promotion during Offline Bounce via Lane 2.
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        let epoch = self.allocate_epoch();
        self.shadow.maximize_tier_on_bounce = enabled;
        self.push_lane2_or_latch(epoch, ControlCommand::SetMaximizeTierOnBounce { enabled });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::host::spsc::spsc_control_channel;
    use crate::math::BackendKind;
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
        DnamJsonHeader, PresetDef,
    };
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_producer_test_container() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        let mut cond_max = BTreeMap::new();
        cond_max.insert(String::from("channel=Clean"), 6.0);
        header.controls_schema = vec![
            ControlSchemaDef {
                id: String::from("gain"),
                name: String::from("Gain"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([1.0, 9.0]),
                default: serde_json::json!(5.0),
                conditional_min: BTreeMap::new(),
                conditional_max: cond_max,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("channel"),
                name: String::from("Channel"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Clean"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec![String::from("Clean"), String::from("Crunch")]),
                active_options: Some(vec![String::from("Clean"), String::from("Crunch")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("delay_ms"),
                name: String::from("Delay"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([20.0, 1000.0]),
                safe_clamped_range: Some([50.0, 800.0]),
                default: serde_json::json!(250.0),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: Some([20.0, 510.0, 1000.0]),
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: Some(TempoSubdivision::Quarter),
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let mut p1 = BTreeMap::new();
        p1.insert(String::from("gain"), serde_json::json!(8.0));
        p1.insert(String::from("channel"), serde_json::json!("Crunch"));
        header.presets = vec![PresetDef {
            id: String::from("crunch"),
            name: String::from("Crunch"),
            description: None,
            controls: p1,
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_producer_validation_batch_tempo_epoch_rollover_and_resync() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        let container = build_producer_test_container();
        let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
            &container,
            Tier::Ultra,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        let (mut producer, mut consumer) = spsc_control_channel(&engine);

        // Fallible validation errors and successes
        assert!(producer.try_set_control_value(99, 1.0).is_err());
        assert!(producer.try_set_control_value(0, f32::NAN).is_err());
        assert!(producer.try_set_control_immediate(99, 1.0).is_err());
        assert!(
            producer
                .try_set_control_immediate(0, f32::INFINITY)
                .is_err()
        );
        assert!(producer.reset_control(99).is_err());
        assert!(producer.apply_preset("nonexistent").is_err());

        // Out-of-bounds infallible setters are no-ops
        producer.set_control_value(99, 1.0);
        producer.set_control_immediate(99, 1.0);

        // Batch immediate update orders switch before range
        producer.set_controls_immediate_batch(&[(0, 8.0), (1, 0.0)]);
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 6.0).abs() < 1e-6);

        // Reset single, all, and engine
        assert!((producer.reset_control(0).unwrap() - 5.0).abs() < 1e-6);
        producer.reset_controls_to_defaults();
        producer.reset_to_dc_equilibrium();

        // Preset, sync_tempo, tap_tempo, host mix, tier ceiling, offline_bounce
        producer.apply_preset("crunch").unwrap();
        assert!((producer.shadow().clamped_control_value(0).unwrap() - 8.0).abs() < 1e-6);
        assert_eq!(
            producer.sync_tempo(120.0, Some(TempoSubdivision::Quarter), true),
            1
        );
        producer
            .ring
            .published_total_frames
            .value
            .store(24_000, Ordering::Release);
        let _ = producer.register_tap_tempo(Some(TempoSubdivision::Quarter), true);
        producer
            .ring
            .published_total_frames
            .value
            .store(48_000, Ordering::Release);
        let (bpm, cnt) = producer.register_tap_tempo(Some(TempoSubdivision::Quarter), true);
        assert!((bpm - 120.0).abs() < 1.0);
        assert_eq!(cnt, 1);

        producer.set_bypass(true, true);
        producer.set_freeze(true);
        producer.set_freeze_amount(0.2);
        producer.set_stereo_width(1.5);
        producer.configure_host_mix_and_leveling(0.8, -1.0, true);
        producer.set_max_supported_tier(Tier::Full);
        assert_eq!(producer.shadow().tier(), Tier::Full);
        producer.set_tier(Tier::Pro);
        assert_eq!(producer.shadow().tier(), Tier::Full);
        producer.set_offline_bounce(true);
        producer.set_offline_bounce(true); // idempotent early return
        producer.set_maximize_tier_on_bounce(false);

        // 29-bit epoch rollover in allocate_epoch followed by consumer drain parity check
        producer.next_epoch_29 = MAX_EPOCH_29;
        producer.set_control_value(0, 5.5);
        assert!(producer.overflow_latch_count() >= 1);
        consumer.drain_into(&mut engine);
        assert_eq!(consumer.drained_causal_epoch(), producer.causal_epoch());
        assert_eq!(producer.drained_causal_epoch(), producer.causal_epoch());
        assert!((engine.clamped_control_values()[0] - 5.5).abs() < 1e-6);
        assert_eq!(
            engine.clamped_control_values(),
            producer.shadow().clamped_control_values()
        );

        // Resync and missing header branch
        producer.resync_from_engine(&engine);
        assert_eq!(
            producer.drained_causal_epoch(),
            consumer.drained_causal_epoch()
        );
        producer.shadow.header = None;
        assert!(producer.apply_preset("crunch").is_err());
        assert_eq!(producer.sync_tempo(120.0, None, true), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_producer_causal_epoch_monotonic(
            vals in proptest::collection::vec(-10.0f32..20.0f32, 1..40)
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            let container = build_producer_test_container();
            let engine = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Lite,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();
            let (mut producer, _consumer) = spsc_control_channel(&engine);

            let mut prev_epoch = producer.causal_epoch();
            for v in vals {
                let res = producer.try_set_control_value(0, v);
                prop_assert!(res.is_ok());
                let cur = producer.causal_epoch();
                prop_assert!(cur > prev_epoch);
                prev_epoch = cur;
            }
        }
    }
}
