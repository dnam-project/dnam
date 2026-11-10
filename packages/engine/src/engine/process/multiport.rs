//! 4-port (`4CM`, `WDW`, `Sidechain`) and 256-D Feature Bus processing loops for [`DnamEngine`].

use super::{DnamAudioIo, feature_bus};
use crate::engine::{DnamEngine, EngineBackend};
use crate::math::{SimdVec4, flush_recursive_state_f32};
use crate::provenance::watermark::max_tiers_for_arena_len;
use crate::types::FtzDazGuard;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Low-level 4-port (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_or_wdw_3, send_4]`)
    /// TORA stepping with optional 256-D feature bus capture (`ref[dsp.router.ports]`).
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(crate) fn step_frame_4port_tora(
        &mut self,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
        mono_only: bool,
        mut feature_bus_out: Option<&mut [f32; 256]>,
    ) -> [f32; 4] {
        let in_l = crate::math::sanitize_input_sample(
            in_l,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = crate::math::sanitize_input_sample(
            in_r,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_l = crate::math::sanitize_input_sample(
            ret_l,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_r = crate::math::sanitize_input_sample(
            ret_r,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        if self.bypassed && !self.trails_enabled {
            if let Some(ref mut fb) = feature_bus_out {
                fb.fill(0.0);
            }
            return if mono_only {
                [in_l, in_l, 0.0, 0.0]
            } else {
                [in_l, in_r, 0.0, 0.0]
            };
        }

        let is_frozen = self.stage_pipeline.stage4.freeze;
        let live_scale = if !self.bypassed && self.freeze_mutes_dry && is_frozen {
            0.0
        } else {
            1.0
        };

        let (eff_in_l, eff_in_r, eff_ret_l, eff_ret_r) = if self.bypassed {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                in_l * live_scale,
                in_r * live_scale,
                ret_l * live_scale,
                ret_r * live_scale,
            )
        };

        let arena_len = self.arena_storage.as_bytes().len();
        let active_tiers =
            (self.effective_tier() as usize + 1).min(max_tiers_for_arena_len(arena_len));
        let loop_tiers = if feature_bus_out.is_some() {
            4
        } else {
            active_tiers
        };

        let k_ctrl = self.zoh.control_decimation_host;
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            self.step_control_tick_3khz(loop_tiers);
        }
        crate::control::ControlBus::<B>::step_micro_ramp_12(
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
        );

        let theta = &self.current_stage_params;
        self.stage_pipeline.stage1.boost_gain = (1.0 + theta[4]).clamp(0.0, 16.0);
        self.stage_pipeline.stage1.pitch_blend = theta[3].clamp(-1.0, 1.0);
        let drive_gain = ((1.0 + 0.25 * theta[5]).clamp(0.25, 4.0)
            + 0.25 * self.control_bus.ctrl_basis_4[0])
            .clamp(0.25, 4.0);
        let beta_tone = (0.5 + theta[6]).clamp(0.0, 1.0);
        let pwr_drive = (1.0 + 0.25 * theta[7]).clamp(0.25, 4.0);
        let sag_depth = ((0.2 + 0.1 * theta[8]).clamp(0.0, 0.8)
            + 0.1 * self.control_bus.ctrl_basis_4[1])
            .clamp(0.0, 0.8);
        let g_fir = (1.0 + theta[11]).clamp(0.0, 2.0);

        let (down_l, down_r, clean_sub_l, clean_sub_r) = self
            .stage_pipeline
            .stage0
            .route_frame_no_return(eff_in_l, eff_in_r);

        let (s1_l, s1_r) = self.stage_pipeline.stage1.step_stereo(down_l, down_r);
        if loop_tiers >= 2 {
            self.stage_pipeline.stage1.step_delta_branch(down_l, down_r);
        }

        // 4CM contract (`r[dsp.router.return-to-send-isolation]`, `R15-E3`):
        // - FX Send (`send_l, send_r`) taps after Stage I (`s1_l, s1_r`) with S_{ret->send} == 0.
        // - FX Return (`eff_ret_l, eff_ret_r`) aligns via `return_aligner_l/r` and injects before Stage II.
        let send_l = self.stage_pipeline.stage0.s_matrix[0][0] * s1_l
            + self.stage_pipeline.stage0.s_matrix[0][1] * s1_r;
        let send_r = self.stage_pipeline.stage0.s_matrix[1][0] * s1_l
            + self.stage_pipeline.stage0.s_matrix[1][1] * s1_r;
        let (s2_in_l, s2_in_r) = if has_return {
            let ret_l_aligned = self.stage_pipeline.stage0.return_aligner_l.step(eff_ret_l);
            let ret_r_aligned = self.stage_pipeline.stage0.return_aligner_r.step(eff_ret_r);
            let s = &self.stage_pipeline.stage0.s_matrix;
            (
                s1_l + s[2][2] * ret_l_aligned + s[2][3] * ret_r_aligned,
                s1_r + s[3][2] * ret_l_aligned + s[3][3] * ret_r_aligned,
            )
        } else {
            (s1_l, s1_r)
        };

        let (s2_l, s2_r) = self.stage_pipeline.stage2.step_frame_into_feature_bus(
            s2_in_l,
            s2_in_r,
            drive_gain,
            sag_depth,
            beta_tone,
            pwr_drive,
            loop_tiers,
            &mut self.feature_bus_scratch,
        );

        let (s3_fir_raw_l, s3_fir_raw_r, s2_delayed_16_l, s2_delayed_16_r) = self
            .stage_pipeline
            .stage3
            .step_fir_stereo_lr_taps_with_delayed_tap::<B>(
                &mut self.stage_pipeline.stage3_r,
                s2_l,
                s2_r,
                clean_sub_l,
                clean_sub_r,
                &self.stage_pipeline.fir_taps_l[0],
                &self.stage_pipeline.fir_taps_r[0],
            );
        let (fir_seg_l, fir_seg_r) = self
            .stage_pipeline
            .stage3
            .eval_higher_tier_fir_segments::<B>(
                &self.stage_pipeline.stage3_r,
                s3_fir_raw_l,
                s3_fir_raw_r,
                clean_sub_l,
                clean_sub_r,
                &self.stage_pipeline.fir_taps_l,
                &self.stage_pipeline.fir_taps_r,
                loop_tiers,
                g_fir,
            );
        let s3_fir_l = fir_seg_l[0];
        let s3_fir_r = fir_seg_r[0];

        if self.cached_head_tiers != active_tiers
            || self.cached_z_head != self.control_bus.z_head
            || self.cached_z_head_ultra != self.control_bus.z_head_ultra
        {
            self.refresh_collapsed_head_weights(active_tiers);
        }

        const KAUTZ_FB_OFFSETS: [usize; 8] = [0, 64, 128, 132, 192, 196, 200, 204];
        const TIER_BLK_BASE: [usize; 4] = [0, 1, 2, 4];
        let mut kautz_acc_l = B::Vec4::zero();
        let mut kautz_acc_r = B::Vec4::zero();
        let xl_vec = B::Vec4::splat(s2_delayed_16_l);
        let xr_vec = B::Vec4::splat(s2_delayed_16_r);
        for t in 0..loop_tiers {
            let tiers_l = &mut self.stage_pipeline.kautz_tiers_l[t];
            let tiers_r = &mut self.stage_pipeline.kautz_tiers_r[t];
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
            let blk_base = TIER_BLK_BASE[t];
            for blk in 0..n_blocks {
                let (vl, vr) = tiers_l[blk].step_4lane_stereo_simd_unscaled::<B>(
                    &mut tiers_r[blk],
                    xl_vec,
                    xr_vec,
                );
                let fb_off = KAUTZ_FB_OFFSETS[blk_base + blk];
                vl.add(vr).store(
                    (&mut self.feature_bus_scratch[fb_off..fb_off + 4])
                        .try_into()
                        .unwrap(),
                );
                if t < active_tiers {
                    kautz_acc_l = kautz_acc_l.add(vl);
                    kautz_acc_r = kautz_acc_r.add(vr);
                }
            }
        }
        let s3_l = s3_fir_l + 0.25 * kautz_acc_l.horizontal_sum();
        let s3_r = s3_fir_r + 0.25 * kautz_acc_r.horizontal_sum();

        let (tank_l, tank_r, tank_readouts, tank_loops) = self
            .stage_pipeline
            .stage4
            .step_stereo_with_readouts(s3_l, s3_r);
        self.stage_pipeline.stage4.read_multi_tier_taps_into(
            loop_tiers,
            tank_readouts,
            tank_loops,
            &mut self.feature_bus_scratch,
        );

        feature_bus::pack_tier_feature_bus_into(
            &mut self.feature_bus_scratch,
            &self.stage_pipeline.stage1,
            &fir_seg_l,
            &fir_seg_r,
            &tank_readouts,
            down_l,
            down_r,
            clean_sub_l,
            clean_sub_r,
            s1_l,
            s1_r,
            s2_in_l,
            s2_in_r,
            s2_l,
            s2_r,
            send_l,
            send_r,
            tank_l,
            tank_r,
            loop_tiers,
        );
        if let Some(fb) = feature_bus_out.as_mut() {
            **fb = *self.feature_bus_scratch;
        }

        let (head_sum_l, head_sum_r) = feature_bus::eval_tier_feature_bus_head_dot::<B>(
            &self.feature_bus_scratch,
            &self.collapsed_head_wl,
            &self.collapsed_head_wr,
            active_tiers,
            mono_only,
        );

        let [b_head_l, b_head_r] = self.collapsed_b_head;

        let mut wet_l = s3_l + tank_l + head_sum_l + b_head_l;
        let mut wet_r = s3_r + tank_r + head_sum_r + b_head_r;

        if !mono_only && (self.stereo_width - 1.0).abs() > 1e-6 {
            let mid = 0.5 * (wet_l + wet_r);
            let side = 0.5 * (wet_l - wet_r);
            wet_l = mid + self.stereo_width * side;
            wet_r = mid - self.stereo_width * side;
        }

        let eff_dry_g = if self.bypassed {
            1.0
        } else {
            self.mix_dry_gain
        };
        let eff_wet_g = if self.bypassed && !self.trails_enabled {
            0.0
        } else {
            self.mix_wet_gain
        };

        if self.stage_pipeline.stage0.wdw_mode {
            let (tank_head_l, tank_head_r) = feature_bus::eval_tier_tank_head_contribution::<B>(
                &self.feature_bus_scratch,
                &self.collapsed_head_wl,
                &self.collapsed_head_wr,
                active_tiers,
            );
            let w_wet_l = tank_l + tank_head_l;
            let w_wet_r = tank_r + tank_head_r;
            let core_l = s3_l + (head_sum_l - tank_head_l) + b_head_l;
            let core_r = s3_r + (head_sum_r - tank_head_r) + b_head_r;
            let core_mid = 0.5 * (core_l + core_r);
            let wet_mid = 0.5 * (w_wet_l + w_wet_r);
            let wet_side = 0.5 * (w_wet_l - w_wet_r);
            let (wdw_dry_c, wdw_wet_l, wdw_wet_r) = crate::stages::eval_wdw_routing_outputs(
                self.stage_pipeline.stage0.wdw_tap_mode,
                core_mid,
                wet_mid,
                wet_side,
                self.stereo_width,
                in_l,
                in_r,
            );
            let out_l = crate::math::sanitize_output_sample(
                eff_wet_g * wdw_wet_l,
                &mut self.saturation_flags,
            );
            let out_r = if mono_only {
                out_l
            } else {
                crate::math::sanitize_output_sample(
                    eff_wet_g * wdw_wet_r,
                    &mut self.saturation_flags,
                )
            };
            let out_3 = crate::math::sanitize_output_sample(wdw_dry_c, &mut self.saturation_flags);
            return [out_l, out_r, out_3, 0.0];
        }

        let out_l = crate::math::sanitize_output_sample(
            eff_dry_g * in_l + eff_wet_g * wet_l,
            &mut self.saturation_flags,
        );
        let out_r = if mono_only {
            out_l
        } else {
            crate::math::sanitize_output_sample(
                eff_dry_g * in_r + eff_wet_g * wet_r,
                &mut self.saturation_flags,
            )
        };
        let out_3 = crate::math::sanitize_output_sample(send_l, &mut self.saturation_flags);
        let out_4 = crate::math::sanitize_output_sample(send_r, &mut self.saturation_flags);

        [out_l, out_r, out_3, out_4]
    }

    /// Internal 4x oversampled 4-port stepping using `self.oversampler` and `self.oversampler_aux`.
    #[inline(never)]
    pub(crate) fn step_frame_4x_4port_tora(
        &mut self,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
        mono_only: bool,
    ) -> [f32; 4] {
        let in_l = crate::math::sanitize_input_sample(
            in_l,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = crate::math::sanitize_input_sample(
            in_r,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_l = crate::math::sanitize_input_sample(
            ret_l,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_r = crate::math::sanitize_input_sample(
            ret_r,
            crate::math::CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let up_main = self.oversampler.upsample_4x(in_l, in_r);
        let up_aux = if has_return {
            self.oversampler_aux.upsample_4x(ret_l, ret_r)
        } else {
            [[0.0_f32; 2]; 4]
        };
        let mut proc_main = [[0.0_f32; 2]; 4];
        let mut last_aux = [0.0_f32; 2];
        for m in 0..4 {
            let [ol, or, o3, o4] = self.step_frame_4port_tora(
                up_main[m][0],
                up_main[m][1],
                up_aux[m][0],
                up_aux[m][1],
                has_return,
                mono_only,
                None,
            );
            proc_main[m] = [ol, or];
            last_aux = [o3, o4];
        }
        let [dl, dr] = self.oversampler.downsample_4x(proc_main);
        [dl, dr, last_aux[0], last_aux[1]]
    }

    /// Steps a single stereo frame and writes the 256-D multi-tier feature bus $\phi[n] \in \mathbb{R}^{256}$
    /// (`64` features per tier for `Lite`, `Full`, `Pro`, `Ultra`) for Stage A / Stage D trainer distillation.
    #[inline]
    pub fn step_frame_with_feature_bus(
        &mut self,
        in_l: f32,
        in_r: f32,
        feature_bus_256: &mut [f32; 256],
    ) -> [f32; 2] {
        let [out_l, out_r, _, _] =
            self.step_frame_4port_tora(in_l, in_r, 0.0, 0.0, false, false, Some(feature_bus_256));
        [out_l, out_r]
    }

    /// Processes a stereo audio block and captures the per-frame 256-D feature bus $\phi[n] \in \mathbb{R}^{256}$.
    pub fn process_block_with_feature_bus(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        feature_bus_frames: &mut [[f32; 256]],
    ) {
        let _ftz = FtzDazGuard::new();
        let num_frames = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len())
            .min(feature_bus_frames.len());
        self.tempo_state.advance_frames(num_frames);
        for i in 0..num_frames {
            let [l, r] = self.step_frame_with_feature_bus(
                left_in[i],
                right_in[i],
                &mut feature_bus_frames[i],
            );
            left_out[i] = l;
            right_out[i] = r;
        }
    }

    /// Processes a multi-port audio block (`Stage 0` 8-port router contract: `in_1..4`, `out_1..4`).
    pub fn process_block_io(&mut self, io: DnamAudioIo<'_>) {
        let _ftz = FtzDazGuard::new();
        let mut num_frames = io.in_1.len().min(io.out_1.len());
        if let Some(s) = io.in_2 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(s) = io.in_3 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(s) = io.in_4 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_2 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_3 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_4 {
            num_frames = num_frames.min(s.len());
        }
        self.tempo_state.advance_frames(num_frames);

        let has_return = io.in_3.is_some() || io.in_4.is_some();
        let mono_only = io.out_2.is_none() && io.out_3.is_none() && io.out_4.is_none();
        let DnamAudioIo {
            in_1,
            in_2,
            in_3,
            in_4,
            out_1,
            mut out_2,
            mut out_3,
            mut out_4,
        } = io;

        for i in 0..num_frames {
            let il = in_1[i];
            let ir = in_2.map_or(il, |s| s[i]);
            let rl = in_3.map_or(0.0, |s| s[i]);
            let rr = in_4.map_or(rl, |s| s[i]);
            let [ol, or, o3, o4] = if self.offline_bounce {
                self.step_frame_4x_4port_tora(il, ir, rl, rr, has_return, mono_only)
            } else {
                self.step_frame_4port_tora(il, ir, rl, rr, has_return, mono_only, None)
            };
            out_1[i] = ol;
            if let Some(ref mut s2) = out_2 {
                s2[i] = or;
            }
            if let Some(ref mut s3) = out_3 {
                s3[i] = o3;
            }
            if let Some(ref mut s4) = out_4 {
                s4[i] = o4;
            }
        }
    }

    /// Processes a 4-Cable Method (`4CM`) block (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_l, send_r]`).
    #[allow(clippy::too_many_arguments)]
    pub fn process_block_4cm(
        &mut self,
        inst_l: &[f32],
        inst_r: &[f32],
        ret_l: &[f32],
        ret_r: &[f32],
        main_l: &mut [f32],
        main_r: &mut [f32],
        send_l: &mut [f32],
        send_r: &mut [f32],
    ) {
        let prev_wdw = self.stage_pipeline.stage0.wdw_mode;
        self.stage_pipeline.stage0.wdw_mode = false;
        self.process_block_io(DnamAudioIo {
            in_1: inst_l,
            in_2: Some(inst_r),
            in_3: Some(ret_l),
            in_4: Some(ret_r),
            out_1: main_l,
            out_2: Some(main_r),
            out_3: Some(send_l),
            out_4: Some(send_r),
        });
        self.stage_pipeline.stage0.wdw_mode = prev_wdw;
    }

    /// Processes a Wet/Dry/Wet (`WDW`) 3-output block (`[in_l, in_r] -> [wet_l, wet_r, dry_center]`).
    pub fn process_block_wdw(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        wet_l: &mut [f32],
        wet_r: &mut [f32],
        dry_center: &mut [f32],
    ) {
        let prev_wdw = self.stage_pipeline.stage0.wdw_mode;
        self.stage_pipeline.stage0.wdw_mode = true;
        self.process_block_io(DnamAudioIo {
            in_1: in_l,
            in_2: Some(in_r),
            in_3: None,
            in_4: None,
            out_1: wet_l,
            out_2: Some(wet_r),
            out_3: Some(dry_center),
            out_4: None,
        });
        self.stage_pipeline.stage0.wdw_mode = prev_wdw;
    }

    /// Processes a sidechain / auxiliary return stereo block (`[main_l, main_r, aux_l, aux_r] -> [out_l, out_r]`).
    pub fn process_block_sidechain(
        &mut self,
        main_l: &[f32],
        main_r: &[f32],
        aux_l: &[f32],
        aux_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.process_block_io(DnamAudioIo {
            in_1: main_l,
            in_2: Some(main_r),
            in_3: Some(aux_l),
            in_4: Some(aux_r),
            out_1: out_l,
            out_2: Some(out_r),
            out_3: None,
            out_4: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::engine::DnamEngine;
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    use crate::types::Tier;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_feature_bus_zero_ulp_parity_and_4cm_wdw_routing() {
        let _ftz = crate::types::FtzDazGuard::new();
        let mut header = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        // Populate Kautz poles in all 4 tiers so all 4 tiers are active in feature_bus_256
        for t in 0..4 {
            let k_base = crate::provenance::kautz_soa_byte_offset(t, 0);
            arena[k_base..k_base + 4].copy_from_slice(&0.88_f32.to_le_bytes());
            arena[k_base + 16..k_base + 20].copy_from_slice(&0.04_f32.to_le_bytes());
            arena[k_base + 32..k_base + 36].copy_from_slice(&0.35_f32.to_le_bytes());
            arena[k_base + 48..k_base + 52].copy_from_slice(&0.25_f32.to_le_bytes());
            arena[k_base + 64..k_base + 68].copy_from_slice(&0.20_f32.to_le_bytes());
            arena[k_base + 80..k_base + 84].copy_from_slice(&(-0.15_f32).to_le_bytes());
        }
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let mut eng_rt = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
                &container,
                tier,
                Tier::Ultra,
                48_000,
            )
            .unwrap();
            let mut eng_fb = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
                &container,
                tier,
                Tier::Ultra,
                48_000,
            )
            .unwrap();

            let mut fb = [0.0f32; 256];
            for n in 0..96 {
                let xl = 0.25 * ((n as f32) * 0.2).sin();
                let xr = -0.20 * ((n as f32) * 0.17).cos();
                let [rt_l, rt_r] = eng_rt.step_frame_tora(xl, xr, false);
                let [fb_l, fb_r] = eng_fb.step_frame_with_feature_bus(xl, xr, &mut fb);
                assert_eq!(
                    rt_l.to_bits(),
                    fb_l.to_bits(),
                    "0 ULP L mismatch at {tier:?}"
                );
                assert_eq!(
                    rt_r.to_bits(),
                    fb_r.to_bits(),
                    "0 ULP R mismatch at {tier:?}"
                );
            }
            // All 4 tiers (0..4) must populate their canonical Kautz modal blocks and Stage II taps even when tier == Tier::Lite
            assert!(fb[0..4].iter().any(|&v| v.abs() > 1e-6));
            assert!(fb[20..52].iter().any(|&v| v.abs() > 1e-6));
            assert!(fb[64..68].iter().any(|&v| v.abs() > 1e-6));
            assert!(fb[128..136].iter().any(|&v| v.abs() > 1e-6));
            assert!(fb[192..208].iter().any(|&v| v.abs() > 1e-6));
            assert!(fb.iter().all(|v| v.is_finite()));
        }

        // Verify 4CM return-to-send isolation (S_{ret->send} == 0) and Return -> Stage II injection
        let mut eng_4cm_a =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_4cm_b =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let inst_l = [0.2f32; 16];
        let inst_r = [-0.15f32; 16];
        let ret_zero = [0.0f32; 16];
        let ret_sig = [0.4f32; 16];
        let (mut main_l_a, mut main_r_a, mut send_l_a, mut send_r_a) =
            ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        let (mut main_l_b, mut main_r_b, mut send_l_b, mut send_r_b) =
            ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        eng_4cm_a.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_zero,
            &ret_zero,
            &mut main_l_a,
            &mut main_r_a,
            &mut send_l_a,
            &mut send_r_a,
        );
        eng_4cm_b.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut main_l_b,
            &mut main_r_b,
            &mut send_l_b,
            &mut send_r_b,
        );
        // Send output must be 100% invariant to Return input (S_{ret->send} == 0)
        assert_eq!(send_l_a, send_l_b);
        assert_eq!(send_r_a, send_r_b);
        // Main output must reflect Return injection after alignment delay
        assert_ne!(main_l_a, main_l_b);

        // Verify WDW routing outputs
        let mut eng_wdw =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let (mut wdw_l, mut wdw_r, mut wdw_dry) = ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        eng_wdw.process_block_wdw(&inst_l, &inst_r, &mut wdw_l, &mut wdw_r, &mut wdw_dry);
        assert!(wdw_dry.iter().any(|&v| v.abs() > 1e-5));

        // Verify sidechain block processing, process_block_with_feature_bus, offline 4-port, and bypassed feature_bus
        let (mut sc_out_l, mut sc_out_r) = ([0.0f32; 16], [0.0f32; 16]);
        eng_wdw.process_block_sidechain(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut sc_out_l,
            &mut sc_out_r,
        );
        assert!(sc_out_l[0].is_finite());

        let mut fb_seq = [[0.0f32; 256]; 16];
        eng_wdw.process_block_with_feature_bus(
            &inst_l,
            &inst_r,
            &mut sc_out_l,
            &mut sc_out_r,
            &mut fb_seq,
        );
        assert!(fb_seq[15][20..52].iter().any(|&v| v.abs() > 1e-6));

        eng_wdw.set_offline_bounce(true);
        eng_wdw.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut main_l_a,
            &mut main_r_a,
            &mut send_l_a,
            &mut send_r_a,
        );
        eng_wdw.process_block_wdw(&inst_l, &inst_r, &mut wdw_l, &mut wdw_r, &mut wdw_dry);
        eng_wdw.process_block_sidechain(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut sc_out_l,
            &mut sc_out_r,
        );
        assert!(sc_out_l[0].is_finite());

        eng_wdw.set_offline_bounce(false);
        eng_wdw.set_bypass(true, false);
        let mut fb_bypass = [1.0f32; 256];
        let [b_l, b_r] = eng_wdw.step_frame_with_feature_bus(0.25, -0.25, &mut fb_bypass);
        assert_eq!([b_l, b_r], [0.25, -0.25]);
        assert_eq!(fb_bypass[0], 0.0);
    }
}
