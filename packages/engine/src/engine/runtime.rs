//! State lifecycle, model reloading, and offline bounce rendering for [`DnamEngine`].

use super::{DnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES};
use crate::container::{DnamContainerView, DnamPreamble, ZohRescaledParams};
use crate::math::MathBackend;
use crate::schema::DnamError;
use crate::types::Tier;

impl<'a, B: MathBackend> DnamEngine<'a, B> {
    pub(crate) fn init_from_view(
        &mut self,
        view: DnamContainerView<'a>,
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let arena_len = view.arena_storage.as_bytes().len();
        let max_available_tier = Self::detect_max_tier(&view.preamble, arena_len);
        let active_tier = tier.min(max_available_tier).min(max_supported_tier);

        let zoh = ZohRescaledParams::compute(
            view.preamble.sample_rate_train_hz,
            sample_rate_hz,
            0.9993,
            0.995,
            64.0,
            128,
        )?;

        let freeze_mutes_dry = view.header.io_config.capabilities.freeze_mutes_dry;
        self.preamble = view.preamble;
        self.header = Some(view.header);
        self.verification_status = view.verification_status;
        self.protection_profile = view.protection_profile;
        self.arena_storage = view.arena_storage;
        self.control_bus.load_control_records(&view.control_records);
        self.tier = active_tier;
        self.max_available_tier = max_available_tier;
        self.max_supported_tier = max_supported_tier;
        self.offline_bounce = false;
        self.maximize_tier_on_bounce = true;
        self.oversampler.reset();
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        self.zoh = zoh;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        self.bypassed = false;
        self.trails_enabled = false;
        self.freeze_mutes_dry = freeze_mutes_dry;
        self.stereo_width = 1.0;

        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    pub(crate) fn detect_max_tier(preamble: &DnamPreamble, arena_len: usize) -> Tier {
        if (preamble.flags & crate::container::FLAG_HAS_ULTRA_DELTA) != 0
            && preamble.arena_ultra_end_cl > 0
            && arena_len >= (preamble.arena_offset_ultra_end() as usize)
        {
            Tier::Ultra
        } else if (preamble.flags & crate::container::FLAG_HAS_PRO_DELTA) != 0
            && preamble.arena_pro_end_cl > 0
            && arena_len >= (preamble.arena_offset_pro_end() as usize)
        {
            Tier::Pro
        } else if (preamble.flags & crate::container::FLAG_HAS_FULL_DELTA) != 0
            && preamble.arena_full_end_cl > 0
            && arena_len >= (preamble.arena_offset_full_end() as usize)
        {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    pub(crate) fn reconfigure_for_sample_rate(
        &mut self,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, false)
    }

    pub(crate) fn reconfigure_for_sample_rate_with_mode(
        &mut self,
        sample_rate_hz: u32,
        reset_controls: bool,
    ) -> Result<(), DnamError> {
        let arena_len = self.arena_storage.as_bytes().len();
        let hdr_ac_pole = self
            .header
            .as_ref()
            .map(|h| h.stage2_cascade.ac_coupling_pole)
            .filter(|&p| p > 0.8 && p < 1.0)
            .unwrap_or(0.9993);
        let r_ac_tr = if arena_len >= 128 {
            let arena = self.arena_storage.as_bytes();
            let v = f32::from_le_bytes([arena[124], arena[125], arena[126], arena[127]]);
            if v > 0.9 && v < 1.0 { v } else { hdr_ac_pole }
        } else {
            hdr_ac_pole
        };
        let r_dc_tr = self
            .header
            .as_ref()
            .map(|h| h.stage4_tank.dc_blocker_pole)
            .filter(|&p| p > 0.8 && p < 1.0)
            .unwrap_or(0.995);
        let tau_ref_tr = self
            .header
            .as_ref()
            .map(|h| h.stage4_tank.tzf_reference_tap_samples)
            .filter(|&t| t > 0.0)
            .unwrap_or(64.0);
        let switch_xfade_tr = self
            .header
            .as_ref()
            .map(|h| h.stage4_tank.switch_xfade_samples)
            .filter(|&x| x > 0)
            .unwrap_or(128);

        let oversample_factor = if self.offline_bounce { 4 } else { 1 };
        self.zoh = ZohRescaledParams::compute_oversampled(
            self.preamble.sample_rate_train_hz,
            sample_rate_hz,
            oversample_factor,
            r_ac_tr,
            r_dc_tr,
            tau_ref_tr,
            switch_xfade_tr as u32,
        )?;

        if reset_controls {
            let arena = self.arena_storage.as_bytes();
            let has_arena_mix = if arena.len() >= 128 {
                let dry = f32::from_le_bytes([arena[116], arena[117], arena[118], arena[119]]);
                let wet = f32::from_le_bytes([arena[120], arena[121], arena[122], arena[123]]);
                if wet.abs() > 1e-6 || dry.abs() > 1e-6 {
                    self.mix_dry_gain = dry;
                    self.mix_wet_gain = wet;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !has_arena_mix {
                if let Some(ref hdr) = self.header {
                    let default_blend = hdr.io_config.capabilities.default_blend;
                    self.configure_host_mix_and_leveling(default_blend, 0.0, true);
                } else {
                    self.mix_dry_gain = 0.0;
                    self.mix_wet_gain = 1.0;
                }
            }
            let arena = self.arena_storage.as_bytes();
            if let Some(ref hdr) = self.header
                && !hdr.controls_schema.is_empty()
                && let Ok(compiled) = hdr.compile_clamp_descriptors()
            {
                self.control_bus.ensure_control_capacity(compiled.len());
                for (i, d) in compiled.into_iter().enumerate() {
                    self.control_bus.descriptors[i] = d;
                }
            }
            self.control_bus.load_from_arena0(
                arena,
                self.preamble.num_controls as usize,
                self.zoh.rho_ctrl,
            );
            if let Some(ref hdr) = self.header {
                let mh = hdr.control_bus.head_trust_region_radius;
                if mh > 1e-3 {
                    self.control_bus.inv_trust_radius_mh2 = 1.0 / (mh * mh);
                }
                self.stage_pipeline
                    .stage0
                    .sub_crossover_l
                    .set_crossover_hz(hdr.stage0_router.sub_crossover_hz);
                self.stage_pipeline
                    .stage0
                    .sub_crossover_r
                    .set_crossover_hz(hdr.stage0_router.sub_crossover_hz);
                self.stage_pipeline.stage1.topology_mode = hdr.stage1_pre.topology_selector & 0x03;
                self.stage_pipeline.stage1.gate_threshold = if hdr.stage1_pre.noise_gate_enabled {
                    crate::schema::cold_pow_f64(
                        10.0,
                        (hdr.stage1_pre.gate_threshold_db as f64) / 20.0,
                    ) as f32
                } else {
                    0.0
                };
            }
            self.control_bus.refresh_static_state(arena);
            self.stage_pipeline
                .load_from_arena_with_zoh(arena, &self.zoh);
        } else {
            let arena = self.arena_storage.as_bytes();
            self.control_bus
                .update_sample_rate_params(arena, self.zoh.rho_ctrl);
            if let Some(ref hdr) = self.header {
                let mh = hdr.control_bus.head_trust_region_radius;
                if mh > 1e-3 {
                    self.control_bus.inv_trust_radius_mh2 = 1.0 / (mh * mh);
                }
            }
            self.control_bus.reset_dynamic_state();
            self.control_bus.refresh_static_state(arena);
            self.stage_pipeline
                .load_from_arena_with_zoh(arena, &self.zoh);
        }
        if self.control_bus.tank_mod_dirty {
            self.apply_tank_modulations();
            self.control_bus.tank_mod_dirty = false;
        }

        let num_ctrls = self.control_bus.num_controls;
        for ctrl_idx in 0..num_ctrls {
            let raw_val = self.control_bus.raw_values[ctrl_idx];
            self.refresh_calibrated_control_effects(ctrl_idx, raw_val);
        }

        self.oversampler.reset();
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        self.stage_pipeline.stage1.op_t_pre.current_delay =
            self.stage_pipeline.stage1.op_t_pre.target_delay;
        self.stage_pipeline.stage1.op_t_pre.old_delay =
            self.stage_pipeline.stage1.op_t_pre.target_delay;
        self.stage_pipeline.stage1.op_t_pre.xfade_active = false;
        self.stage_pipeline.stage1.op_t_pre.xfade_idx = 0;
        self.stage_pipeline.stage4.current_psi = self.stage_pipeline.stage4.target_psi;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        Ok(())
    }

    /// Hot-swaps a new `.dnam` model in-place with zero heap allocations.
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let preamble = DnamPreamble::from_bytes(bytes)?;
        if preamble.is_encrypted_arena() {
            return Err(DnamError::EncryptedArenaRequiresDeviceKey);
        }
        let arena_off = preamble.tensor_arena_offset as usize;
        let arena_len = preamble.tensor_arena_byte_len as usize;
        if bytes.len() < arena_off + arena_len {
            return Err(DnamError::BufferTooSmall {
                expected: arena_off + arena_len,
                actual: bytes.len(),
            });
        }
        let computed_hdr_crc = crate::container::compute_header_crc32c(bytes, arena_off);
        if computed_hdr_crc != preamble.header_crc32c {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: preamble.header_crc32c,
                computed: computed_hdr_crc,
            });
        }

        let max_avail = Self::detect_max_tier(&preamble, arena_len);
        let active_tier = tier.min(max_avail).min(self.max_supported_tier);

        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena)?;
        if !self.arena_storage.reload_in_place(raw_arena) {
            self.arena_storage = crate::memory::ArenaStorage::from_slice_or_copy(raw_arena);
        }
        let p_ctrls = preamble.num_controls as usize;
        if arena_off >= 64 + 128 * p_ctrls {
            let ctrl_off = arena_off - 128 * p_ctrls;
            self.control_bus
                .load_control_records_from_bytes(&bytes[ctrl_off..arena_off], p_ctrls);
        }
        self.preamble = preamble;
        self.header = None;
        self.tier = active_tier;
        self.max_available_tier = max_avail;
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    /// Hot-swaps a model in-place into an owned engine.
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let preamble = DnamPreamble::from_bytes(bytes)?;
        if preamble.is_encrypted_arena() {
            return Err(DnamError::EncryptedArenaRequiresDeviceKey);
        }
        let arena_off = preamble.tensor_arena_offset as usize;
        let arena_len = preamble.tensor_arena_byte_len as usize;
        if bytes.len() < arena_off + arena_len {
            return Err(DnamError::BufferTooSmall {
                expected: arena_off + arena_len,
                actual: bytes.len(),
            });
        }
        let computed_hdr_crc = crate::container::compute_header_crc32c(bytes, arena_off);
        if computed_hdr_crc != preamble.header_crc32c {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: preamble.header_crc32c,
                computed: computed_hdr_crc,
            });
        }

        let max_avail = Self::detect_max_tier(&preamble, arena_len);
        let active_tier = tier.min(max_avail).min(self.max_supported_tier);

        let raw_arena = &bytes[arena_off..arena_off + arena_len];
        DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena)?;
        if !self.arena_storage.reload_in_place(raw_arena) {
            self.arena_storage = crate::memory::ArenaStorage::from_slice_owned(raw_arena);
        }
        let p_ctrls = preamble.num_controls as usize;
        if arena_off >= 64 + 128 * p_ctrls {
            let ctrl_off = arena_off - 128 * p_ctrls;
            self.control_bus
                .load_control_records_from_bytes(&bytes[ctrl_off..arena_off], p_ctrls);
        }
        let json_len = preamble.json_header_byte_len as usize;
        let parsed_header = if json_len > 0 && 64 + json_len <= arena_off {
            serde_json::from_slice::<crate::schema::DnamJsonHeader>(&bytes[64..64 + json_len]).ok()
        } else {
            None
        };
        if let Some(ref hdr) = parsed_header {
            self.freeze_mutes_dry = hdr.io_config.capabilities.freeze_mutes_dry;
        }
        self.preamble = preamble;
        self.header = parsed_header;
        self.tier = active_tier;
        self.max_available_tier = max_avail;
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    /// Resets all states to analytical DC resting equilibrium.
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.control_bus.reset_dynamic_state();
        let arena = self.arena_storage.as_bytes();
        self.control_bus.refresh_static_state(arena);
        self.stage_pipeline.reset_to_dc_equilibrium();
        if self.control_bus.tank_mod_dirty {
            self.apply_tank_modulations();
            self.control_bus.tank_mod_dirty = false;
        }
        let num_ctrls = self.control_bus.num_controls;
        for ctrl_idx in 0..num_ctrls {
            let raw_val = self.control_bus.raw_values[ctrl_idx];
            self.refresh_calibrated_control_effects(ctrl_idx, raw_val);
        }
        self.stage_pipeline.stage1.op_t_pre.current_delay =
            self.stage_pipeline.stage1.op_t_pre.target_delay;
        self.stage_pipeline.stage1.op_t_pre.old_delay =
            self.stage_pipeline.stage1.op_t_pre.target_delay;
        self.stage_pipeline.stage1.op_t_pre.xfade_active = false;
        self.stage_pipeline.stage1.op_t_pre.xfade_idx = 0;
        self.stage_pipeline.stage4.current_psi = self.stage_pipeline.stage4.target_psi;
        self.oversampler.reset();
        self.tempo_state.reset();
        self.current_stage_params = self.control_bus.stage_targets;
        self.stage_ramp_deltas = [0.0; 16];
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
    }

    /// Renders a complete stereo buffer in Offline Bounce mode, automatically stripping the 12-sample PDC lead-in.
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let prev_bounce = self.offline_bounce;
        if !prev_bounce {
            self.set_offline_bounce(true);
        }
        self.reset_to_dc_equilibrium();

        let n = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        let pdc = OFFLINE_BOUNCE_PDC_SAMPLES;
        let mut sink_l = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
        let mut sink_r = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];

        if n <= pdc {
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &left_in[..n],
                &right_in[..n],
                &mut sink_l[..n],
                &mut sink_r[..n],
            );
            let rem = pdc - n;
            if rem > 0 {
                self.process_block(
                    &zeros[..rem],
                    &zeros[..rem],
                    &mut sink_l[..rem],
                    &mut sink_r[..rem],
                );
            }
            self.process_block(
                &zeros[..n],
                &zeros[..n],
                &mut left_out[..n],
                &mut right_out[..n],
            );
        } else {
            self.process_block(&left_in[..pdc], &right_in[..pdc], &mut sink_l, &mut sink_r);
            let body_len = n - pdc;
            self.process_block(
                &left_in[pdc..n],
                &right_in[pdc..n],
                &mut left_out[..body_len],
                &mut right_out[..body_len],
            );
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &zeros,
                &zeros,
                &mut left_out[body_len..n],
                &mut right_out[body_len..n],
            );
        }

        if !prev_bounce {
            self.set_offline_bounce(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DnamPreamble, FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA,
        FLAG_HAS_ULTRA_DELTA, build_dnam_container,
    };
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_detect_max_tier() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut preamble = DnamPreamble::from_bytes(&container).unwrap();
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 10752),
            Tier::Lite
        );

        preamble.flags |= FLAG_HAS_FULL_DELTA;
        preamble.arena_full_end_cl = 349;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 22336),
            Tier::Full
        );

        preamble.flags |= FLAG_HAS_PRO_DELTA;
        preamble.arena_pro_end_cl = 566;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 36224),
            Tier::Pro
        );

        preamble.flags |= FLAG_HAS_ULTRA_DELTA;
        preamble.arena_ultra_end_cl = 1000;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 64000),
            Tier::Ultra
        );
    }

    #[test]
    fn test_runtime_render_offline_bounce_and_reloads() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
            &container,
            Tier::Ultra,
            Tier::Ultra,
            48000,
        )
        .unwrap();

        // 1. Reset to DC equilibrium
        engine.reset_to_dc_equilibrium();

        // 2. Offline bounce with small buffer (n <= 12 PDC)
        let short_in = [0.1f32; 8];
        let mut short_l = [0.0f32; 8];
        let mut short_r = [0.0f32; 8];
        engine.render_offline_bounce(&short_in, &short_in, &mut short_l, &mut short_r);

        // 3. Offline bounce with longer buffer (n > 12 PDC)
        let long_in = [0.1f32; 32];
        let mut long_l = [0.0f32; 32];
        let mut long_r = [0.0f32; 32];
        engine.render_offline_bounce(&long_in, &long_in, &mut long_l, &mut long_r);

        // 4. Reload in place owned success and error
        let mut owned_engine =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&container, Tier::Lite, 48000).unwrap();
        assert!(
            owned_engine
                .reload_model_in_place_owned(&container, Tier::Lite, 48000)
                .is_ok()
        );

        // Error: corrupt preamble
        assert!(
            owned_engine
                .reload_model_in_place_owned(&[0u8; 32], Tier::Lite, 48000)
                .is_err()
        );

        // Error: buffer too small
        assert!(
            owned_engine
                .reload_model_in_place_owned(&container[..container.len() - 100], Tier::Lite, 48000)
                .is_err()
        );

        // Error: header crc mismatch
        let mut bad_crc = container.clone();
        bad_crc[60] ^= 0xAA;
        assert!(
            owned_engine
                .reload_model_in_place_owned(&bad_crc, Tier::Lite, 48000)
                .is_err()
        );

        // Error: arena crc mismatch in reload_model_in_place and reload_model_in_place_owned
        let preamble = DnamPreamble::from_bytes(&container).unwrap();
        let arena_off = preamble.tensor_arena_offset as usize;
        let mut bad_arena = container.clone();
        bad_arena[arena_off] ^= 0xFF;
        assert!(
            owned_engine
                .reload_model_in_place_owned(&bad_arena, Tier::Lite, 48000)
                .is_err()
        );
        assert!(
            engine
                .reload_model_in_place(&bad_arena, Tier::Lite, 48000)
                .is_err()
        );

        // Verify reload_model_in_place_owned preserves parsed JSON header
        assert!(owned_engine.header().is_some());

        // Verify process_block advances tempo_state.total_frames and reset_to_dc_equilibrium preserves ZOH delays
        owned_engine.set_sample_rate(96_000).unwrap();
        let _ = owned_engine.register_tap_tempo(None, true);
        owned_engine.process_block(&short_in, &short_in, &mut short_l, &mut short_r);
        assert_eq!(owned_engine.tempo_state.total_frames, 8);
        let (bpm_tap, _) = owned_engine.register_tap_tempo(None, true);
        assert!((20.0..=400.0).contains(&bpm_tap));
        let d_before = owned_engine
            .stage_pipeline
            .stage0
            .return_aligner_l
            .delay_samples;
        assert!((d_before - 8.0).abs() < 1e-5);
        owned_engine.reset_to_dc_equilibrium();
        assert_eq!(
            owned_engine
                .stage_pipeline
                .stage0
                .return_aligner_l
                .delay_samples,
            d_before
        );
    }
}
