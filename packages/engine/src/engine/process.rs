//! Real-time audio processing loops for [`DnamEngine`].
//!
//! Submodules:
//! - [`multiport`]: 4-port (`4CM`, `WDW`, `Sidechain`) and 256-D Feature Bus processing loops.

pub mod feature_bus;
pub mod multiport;

use super::DnamEngine;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};
use crate::provenance::watermark::{max_tiers_for_arena_len, read_f32_le};
use crate::types::FtzDazGuard;

#[inline(always)]
fn read_vec4_le<B: MathBackend>(bytes: &[u8], off: usize) -> B::Vec4 {
    #[cfg(target_endian = "little")]
    {
        if let Ok(chunk) = <&[u8; 16]>::try_from(&bytes[off..off + 16]) {
            let arr: [f32; 4] = zerocopy::transmute!(*chunk);
            return B::Vec4::load(&arr);
        }
    }
    let f0 = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    let f1 = f32::from_le_bytes([
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ]);
    let f2 = f32::from_le_bytes([
        bytes[off + 8],
        bytes[off + 9],
        bytes[off + 10],
        bytes[off + 11],
    ]);
    let f3 = f32::from_le_bytes([
        bytes[off + 12],
        bytes[off + 13],
        bytes[off + 14],
        bytes[off + 15],
    ]);
    B::Vec4::from_array([f0, f1, f2, f3])
}

impl<'a, B: MathBackend> DnamEngine<'a, B> {
    /// Collapses the `3 kHz` SVD Kronecker Output Head weights (`ref[dsp.head.mid-side-factorization]`)
    /// across all `16` `Vec4` feature groups (`g = 0..16`) per tier and folds Role B (`b = 1..K_t-1`)
    /// of `CASCADE_W_BASES[t]` surjectively onto the `32` `Vec4` slices $(c \in \{0,1\}, g \in 0..16)$.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline]
    pub(crate) fn refresh_collapsed_head_weights(&mut self, active_tiers: usize) {
        const CASCADE_W_BASES: [usize; 4] = [1_024, 11_264, 22_336, 36_224];
        const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];

        let arena = self.arena_storage.as_bytes();
        let z_h = self.control_bus.z_head;
        let z_u = self.control_bus.z_head_ultra;
        let z_vecs = [
            B::Vec4::splat(z_h[0]),
            B::Vec4::splat(z_h[1]),
            B::Vec4::splat(z_h[2]),
            B::Vec4::splat(z_h[3]),
        ];
        let zu_vecs = [
            B::Vec4::splat(z_u[0]),
            B::Vec4::splat(z_u[1]),
            B::Vec4::splat(z_u[2]),
            B::Vec4::splat(z_u[3]),
        ];
        let one_vec = B::Vec4::splat(1.0);

        for t in 0..active_tiers {
            let cw_base = CASCADE_W_BASES[t];
            let k_t = 48 * (t + 1);
            let mut m_slices = [one_vec; 32];
            if arena.len() >= cw_base + k_t * 64 {
                for r in 0..(k_t - 1) {
                    let blk_off = cw_base + (r + 1) * 64;
                    for q in 0..4 {
                        let s = (4 * r + q) & 31;
                        m_slices[s] = m_slices[s].add(read_vec4_le::<B>(arena, blk_off + q * 16));
                    }
                }
            }

            let h_base = HEAD_BASES[t];
            let u_base = 49_408 + t * 2048;
            for g in 0..16 {
                let g_off_l = g * 16;
                let g_off_r = 256 + g * 16;
                let mut wl = read_vec4_le::<B>(arena, h_base + g_off_l);
                let mut wr = read_vec4_le::<B>(arena, h_base + g_off_r);
                for k in 0..4 {
                    let mode_off = h_base + (k + 1) * 512;
                    wl = wl.fma(z_vecs[k], read_vec4_le::<B>(arena, mode_off + g_off_l));
                    wr = wr.fma(z_vecs[k], read_vec4_le::<B>(arena, mode_off + g_off_r));
                }
                if active_tiers >= 4 {
                    for k in 0..4 {
                        let u_mode_off = u_base + k * 512;
                        wl = wl.fma(zu_vecs[k], read_vec4_le::<B>(arena, u_mode_off + g_off_l));
                        wr = wr.fma(zu_vecs[k], read_vec4_le::<B>(arena, u_mode_off + g_off_r));
                    }
                }
                self.collapsed_head_wl[t][g] = wl.mul(m_slices[g]);
                self.collapsed_head_wr[t][g] = wr.mul(m_slices[16 + g]);
            }
        }

        let mut b_head_l = 0.0_f32;
        let mut b_head_r = 0.0_f32;
        if active_tiers >= 1 && arena.len() >= 9_064 {
            b_head_l = read_f32_le(arena, 9_024);
            b_head_r = read_f32_le(arena, 9_028);
            for k in 0..4 {
                b_head_l += z_h[k] * read_f32_le(arena, 9_024 + ((k + 1) * 2) * 4);
                b_head_r += z_h[k] * read_f32_le(arena, 9_024 + ((k + 1) * 2 + 1) * 4);
            }
            if active_tiers >= 4 && arena.len() >= 63_456 {
                for k in 0..4 {
                    b_head_l += z_u[k] * read_f32_le(arena, 63_424 + (k * 2) * 4);
                    b_head_r += z_u[k] * read_f32_le(arena, 63_424 + (k * 2 + 1) * 4);
                }
            }
        }
        self.collapsed_b_head = [b_head_l, b_head_r];
        self.cached_z_head = z_h;
        self.cached_z_head_ultra = z_u;
        self.cached_head_tiers = active_tiers;
    }

    /// Executes the decimated `3 kHz` Control Bus tick and modal formant retuning.
    #[inline]
    pub(crate) fn step_control_tick_3khz(&mut self, loop_tiers: usize) {
        let avg_energy = self.ctrl_energy_accum * self.zoh.alpha_ramp_host; // Division-free
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        let det_env = self.stage_pipeline.stage0.detector_envelopes;
        let arena = self.arena_storage.as_bytes();
        let _changed = self
            .control_bus
            .tick_3khz_with_detectors(avg_energy, Some(det_env), arena);
        self.stage_ramp_deltas = crate::control::ControlBus::<B>::compute_micro_ramp_deltas(
            &self.current_stage_params,
            &self.control_bus.stage_targets,
            self.zoh.alpha_ramp_host,
        );
        if self.control_bus.tank_mod_dirty {
            self.apply_tank_modulations();
            self.control_bus.tank_mod_dirty = false;
        }
        let dtheta_tick = self.current_stage_params[9]
            .clamp(-core::f32::consts::FRAC_PI_4, core::f32::consts::FRAC_PI_4);
        let g_rad_pre_tick = (0.5 + self.current_stage_params[6]).clamp(0.0, 1.0);
        let g_kautz_tick = (0.5 + self.control_bus.ctrl_basis_4[2] + self.current_stage_params[10])
            .clamp(0.0, 1.0);
        self.stage_pipeline
            .stage1
            .retune_op_m_pre_3khz(dtheta_tick, g_rad_pre_tick, true);
        self.stage_pipeline
            .retune_kautz_formant_3khz(loop_tiers, dtheta_tick, g_kautz_tick);
    }

    /// Applies $3\text{ kHz}$ `ControlBus` state-space modulations (`W_ctrl`) to Stage I and Stage IV.
    #[inline(always)]
    pub(crate) fn apply_tank_modulations(&mut self) {
        let g_mod = 0.25
            * (self.control_bus.tank_gain_targets[0]
                + self.control_bus.tank_gain_targets[1]
                + self.control_bus.tank_gain_targets[2]
                + self.control_bus.tank_gain_targets[3])
            + self.control_bus.tank_state_mod[2];
        self.stage_pipeline.stage4.g_loop = (0.85 + g_mod).clamp(0.1, 1.18);
        let psi_target = (core::f32::consts::FRAC_PI_4 + self.control_bus.tank_state_mod[1])
            .clamp(0.05, core::f32::consts::PI - 0.05);
        self.stage_pipeline.stage4.set_target_psi(psi_target);
        let d_mod0 =
            self.control_bus.tank_delay_mod_samples[0] * self.stage_pipeline.stage2.hyst_diff_scale;
        self.stage_pipeline
            .stage1
            .op_t_pre
            .set_target_delay(self.zoh.tau_ref_samples_host + d_mod0, false);
    }

    /// Low-level single-frame TORA stepping (`0 FDIV`, `0 FSQRT`).
    ///
    /// Direct callers must hold a [`FtzDazGuard`] for the enclosing batch. This
    /// method does not perform block-level preview accounting, tempo advancement,
    /// or offline oversampling; use the block APIs when those behaviors are needed.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_frame_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
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
        if self.bypassed && !self.trails_enabled {
            return if mono_only {
                [in_l, in_l]
            } else {
                [in_l, in_r]
            };
        }

        let is_frozen = self.stage_pipeline.stage4.freeze;
        let live_scale = if !self.bypassed && self.freeze_mutes_dry && is_frozen {
            0.0
        } else {
            1.0
        };

        let (eff_in_l, eff_in_r) = if self.bypassed {
            (0.0, 0.0)
        } else {
            (in_l * live_scale, in_r * live_scale)
        };

        let arena_len = self.arena_storage.as_bytes().len();
        let active_tiers =
            (self.effective_tier() as usize + 1).min(max_tiers_for_arena_len(arena_len));

        // Dual-rate 3 kHz Control Bus stepping: decimate by K_ctrl samples with precomputed alpha_ramp
        let k_ctrl = self.zoh.control_decimation_host;
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            self.step_control_tick_3khz(active_tiers);
        }
        crate::control::ControlBus::<B>::step_micro_ramp(
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
        );

        let laws = crate::stages::eval_stage_modulation_laws(
            &self.current_stage_params,
            [
                self.stage_pipeline.stage0.s_matrix[0][0],
                self.stage_pipeline.stage0.s_matrix[1][1],
            ],
            self.zoh.tau_ref_samples_host,
            256.0,
            self.stage_pipeline.stage4.delay_lengths[0] as f32,
            1004.0,
            self.stage_pipeline.stage4.g_loop,
            0.5,
            self.stereo_width,
        );
        self.stage_pipeline.stage1.boost_gain = laws[4];
        self.stage_pipeline.stage1.pitch_blend = self.current_stage_params[3].clamp(-1.0, 1.0);
        let drive_gain = (laws[5] + 0.25 * self.control_bus.ctrl_basis_4[0]).clamp(0.25, 4.0);
        let beta_tone = laws[6];
        let pwr_drive = laws[7];
        let sag_depth = (laws[8] + 0.1 * self.control_bus.ctrl_basis_4[1]).clamp(0.0, 0.8);
        let g_fir = laws[11];

        // Stage 0: 8-Port Matrix Router
        let (down_l, down_r, clean_sub_l, clean_sub_r) = self
            .stage_pipeline
            .stage0
            .route_frame_no_return(eff_in_l, eff_in_r);

        // Stage I: Pre-Dynamics (+ Full+ parallel modal delta bank)
        let (s1_l, s1_r) = self.stage_pipeline.stage1.step_stereo(down_l, down_r);
        if active_tiers >= 2 {
            self.stage_pipeline.stage1.step_delta_branch(down_l, down_r);
        }
        let send_l = self.stage_pipeline.stage0.s_matrix[0][0] * s1_l
            + self.stage_pipeline.stage0.s_matrix[0][1] * s1_r;
        let send_r = self.stage_pipeline.stage0.s_matrix[1][0] * s1_l
            + self.stage_pipeline.stage0.s_matrix[1][1] * s1_r;

        // Stage II: Commuted Cascade (L-N-L-N + feedforward tier branches)
        let s2_out = self.stage_pipeline.stage2.step_frame_with_branches(
            s1_l,
            s1_r,
            drive_gain,
            sag_depth,
            beta_tone,
            pwr_drive,
            active_tiers,
        );
        let s2_l = s2_out.s2_l;
        let s2_r = s2_out.s2_r;

        // Stage III: Cabinet Modals (64-tap contiguous Early FIR + Stereo Kautz Modal Bank)
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
                active_tiers,
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

        const TIER_BLK_BASE: [usize; 4] = [0, 1, 2, 4];
        let mut kautz_modal_feats = [[0.0_f32; 4]; 8];
        let mut kautz_acc_l = B::Vec4::zero();
        let mut kautz_acc_r = B::Vec4::zero();
        let xl_vec = B::Vec4::splat(s2_delayed_16_l);
        let xr_vec = B::Vec4::splat(s2_delayed_16_r);
        for t in 0..active_tiers {
            let tiers_l = &mut self.stage_pipeline.kautz_tiers_l[t];
            let tiers_r = &mut self.stage_pipeline.kautz_tiers_r[t];
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
            let blk_base = TIER_BLK_BASE[t];
            for blk in 0..n_blocks {
                let (vl, vr) = tiers_l[blk].step_4lane_stereo_simd_splatted::<B>(
                    &mut tiers_r[blk],
                    xl_vec,
                    xr_vec,
                );
                kautz_modal_feats[blk_base + blk] =
                    crate::stages::pack_stage3_modal_block_features(&vl.to_array(), &vr.to_array());
                kautz_acc_l = kautz_acc_l.add(vl);
                kautz_acc_r = kautz_acc_r.add(vr);
            }
        }
        let s3_l = s3_fir_l + 0.125 * kautz_acc_l.horizontal_sum();
        let s3_r = s3_fir_r + 0.125 * kautz_acc_r.horizontal_sum();

        // Stage IV: Unitary Spatial Tank + Multi-Tier Delay Taps
        let (tank_l, tank_r, tank_readouts, tank_loops) = self
            .stage_pipeline
            .stage4
            .step_stereo_with_readouts(s3_l, s3_r);
        let (s4_full_taps, s4_pro_taps, s4_ultra_taps) = self
            .stage_pipeline
            .stage4
            .read_multi_tier_taps(active_tiers, tank_readouts, tank_loops);

        let fb_local = feature_bus::pack_tier_feature_bus(
            &self.stage_pipeline.stage1,
            &s2_out,
            &kautz_modal_feats,
            &fir_seg_l,
            &fir_seg_r,
            &tank_readouts,
            &s4_full_taps,
            &s4_pro_taps,
            &s4_ultra_taps,
            down_l,
            down_r,
            clean_sub_l,
            clean_sub_r,
            s1_l,
            s1_r,
            s1_l,
            s1_r,
            send_l,
            send_r,
            tank_l,
            tank_r,
            active_tiers,
        );

        let (head_sum_l, head_sum_r) = feature_bus::eval_tier_feature_bus_head_dot::<B>(
            &fb_local,
            &self.collapsed_head_wl,
            &self.collapsed_head_wr,
            active_tiers,
            mono_only,
        );

        let [b_head_l, b_head_r] = self.collapsed_b_head;

        let mut wet_l = s3_l + tank_l + head_sum_l + b_head_l;
        let mut wet_r = s3_r + tank_r + head_sum_r + b_head_r;

        // Stereo width folding (0.0 = mono, 1.0 = reference stereo, 2.0 = hyper-wide)
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

        let dry_l = in_l;
        let dry_r = in_r;

        let out_l = crate::math::sanitize_output_sample(
            eff_dry_g * dry_l + eff_wet_g * wet_l,
            &mut self.saturation_flags,
        );
        let out_r = if mono_only {
            out_l
        } else {
            crate::math::sanitize_output_sample(
                eff_dry_g * dry_r + eff_wet_g * wet_r,
                &mut self.saturation_flags,
            )
        };

        [out_l, out_r]
    }

    /// Internal 4x oversampled stepping.
    #[inline(always)]
    pub(crate) fn step_frame_4x_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
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
        let up_frames = self.oversampler.upsample_4x(in_l, in_r);
        let mut proc_frames = [[0.0_f32; 2]; 4];
        for (m, frame) in up_frames.iter().enumerate() {
            proc_frames[m] = self.step_frame_tora(frame[0], frame[1], mono_only);
        }
        self.oversampler.downsample_4x(proc_frames)
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        let num_frames = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        self.tempo_state.advance_frames(num_frames);

        if self.offline_bounce {
            for i in 0..num_frames {
                let [l, r] = self.step_frame_4x_tora(left_in[i], right_in[i], false);
                left_out[i] = l;
                right_out[i] = r;
            }
        } else {
            for i in 0..num_frames {
                let [l, r] = self.step_frame_tora(left_in[i], right_in[i], false);
                left_out[i] = l;
                right_out[i] = r;
            }
        }
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        let num_frames = mono_in.len().min(left_out.len()).min(right_out.len());
        self.tempo_state.advance_frames(num_frames);

        if self.offline_bounce {
            for i in 0..num_frames {
                let s = mono_in[i];
                let [l, r] = self.step_frame_4x_tora(s, s, false);
                left_out[i] = l;
                right_out[i] = r;
            }
        } else {
            for i in 0..num_frames {
                let s = mono_in[i];
                let [l, r] = self.step_frame_tora(s, s, false);
                left_out[i] = l;
                right_out[i] = r;
            }
        }
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let num_frames = mono_in.len().min(mono_out.len());
        self.tempo_state.advance_frames(num_frames);

        if self.offline_bounce {
            for i in 0..num_frames {
                let s = mono_in[i];
                let [l, _] = self.step_frame_4x_tora(s, s, true);
                mono_out[i] = l;
            }
        } else {
            for i in 0..num_frames {
                let s = mono_in[i];
                let [l, _] = self.step_frame_tora(s, s, true);
                mono_out[i] = l;
            }
        }
    }

    /// Processes in-place stereo audio blocks.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let num_frames = left_io.len().min(right_io.len());
        self.tempo_state.advance_frames(num_frames);

        if self.offline_bounce {
            for i in 0..num_frames {
                let [l, r] = self.step_frame_4x_tora(left_io[i], right_io[i], false);
                left_io[i] = l;
                right_io[i] = r;
            }
        } else {
            for i in 0..num_frames {
                let [l, r] = self.step_frame_tora(left_io[i], right_io[i], false);
                left_io[i] = l;
                right_io[i] = r;
            }
        }
    }
}

/// Unified multi-port audio I/O descriptor (`Stage 0` 8-port router contract: `in_1..4`, `out_1..4`).
#[derive(Debug)]
pub struct DnamAudioIo<'b> {
    pub in_1: &'b [f32],
    pub in_2: Option<&'b [f32]>,
    pub in_3: Option<&'b [f32]>,
    pub in_4: Option<&'b [f32]>,
    pub out_1: &'b mut [f32],
    pub out_2: Option<&'b mut [f32]>,
    pub out_3: Option<&'b mut [f32]>,
    pub out_4: Option<&'b mut [f32]>,
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
    fn test_process_modes_and_in_place() {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48000).unwrap();

        let mut l = [0.1f32; 16];
        let mut r = [-0.1f32; 16];
        engine.process_block_in_place(&mut l, &mut r);
        assert!(l[0].is_finite());
        assert!(r[0].is_finite());

        // Offline bounce mode in all process functions
        engine.set_offline_bounce(true);
        let mut l_out = [0.0f32; 16];
        let mut r_out = [0.0f32; 16];
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        engine.process_mono_to_stereo(&l, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        engine.process_mono(&l, &mut l_out);
        assert!(l_out[0].is_finite());

        engine.process_block_in_place(&mut l, &mut r);
        assert!(l[0].is_finite());

        // Bypassed with trails
        engine.set_offline_bounce(false);
        engine.set_bypass(true, true);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        // Bypassed without trails (stereo & mono)
        engine.set_bypass(true, false);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert_eq!(l_out, l);
        engine.process_mono(&l, &mut l_out);
        assert_eq!(l_out, l);

        // Freeze mutes dry
        engine.set_bypass(false, false);
        engine.set_freeze(true);
        engine.freeze_mutes_dry = true;
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());
    }
}
