//! Real-time audio processing loops for [`DnamEngine`].

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
    /// and pre-folds `CASCADE_W_BASES` and cross-tier shared feature groups (`g = 8..15`) for `active_tiers`.
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
        let mut shared_wl = [B::Vec4::zero(); 7];
        let mut shared_wr = [B::Vec4::zero(); 7];

        for t in 0..active_tiers {
            let cw_base = CASCADE_W_BASES[t];
            let w_c = [
                read_vec4_le::<B>(arena, cw_base).add(one_vec),
                read_vec4_le::<B>(arena, cw_base + 16).add(one_vec),
                read_vec4_le::<B>(arena, cw_base + 32).add(one_vec),
                read_vec4_le::<B>(arena, cw_base + 48).add(one_vec),
            ];

            let h_base = HEAD_BASES[t];
            let u_base = 49_408 + t * 2048;
            for g in 0..15 {
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
                if g < 8 {
                    self.collapsed_kautz_wl[t][g] = wl;
                    self.collapsed_kautz_wr[t][g] = wr;
                } else if g < 12 {
                    let scale = w_c[g - 8];
                    shared_wl[g - 8] = shared_wl[g - 8].fma(wl, scale);
                    shared_wr[g - 8] = shared_wr[g - 8].fma(wr, scale);
                } else {
                    shared_wl[g - 8] = shared_wl[g - 8].add(wl);
                    shared_wr[g - 8] = shared_wr[g - 8].add(wr);
                }
            }
        }
        self.collapsed_shared_wl = shared_wl;
        self.collapsed_shared_wr = shared_wr;

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

        // Dual-rate 3 kHz Control Bus stepping: decimate by K_ctrl samples with precomputed alpha_ramp
        let k_ctrl = self.zoh.control_decimation_host;
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            let avg_energy = self.ctrl_energy_accum * self.zoh.alpha_ramp_host; // Division-free
            self.ctrl_sub_counter = 0;
            self.ctrl_energy_accum = 0.0;
            let det_env = self.stage_pipeline.stage0.detector_envelopes;
            let arena = self.arena_storage.as_bytes();
            let _changed =
                self.control_bus
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
        }
        crate::control::ControlBus::<B>::step_micro_ramp(
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
        );

        // Stage 0: 8-Port Matrix Router
        let (down_l, down_r, clean_sub_l, clean_sub_r) = self
            .stage_pipeline
            .stage0
            .route_frame_no_return(eff_in_l, eff_in_r);

        // Stage I: Pre-Dynamics
        let (s1_l, s1_r) = self.stage_pipeline.stage1.step_stereo(down_l, down_r);

        // Stage II: Commuted Cascade (Tone Basis + Non-linear PolyADAA)
        let stage_law_5 = (1.0 + 0.25 * self.current_stage_params[5]).clamp(0.25, 4.0);
        let stage_law_8 = (0.2 + 0.1 * self.current_stage_params[8]).clamp(0.0, 0.8);
        let drive_gain = (stage_law_5 + 0.25 * self.control_bus.ctrl_basis_4[0]).clamp(0.25, 4.0);
        let sag_depth = (stage_law_8 + 0.1 * self.control_bus.ctrl_basis_4[1]).clamp(0.0, 0.8);
        let (basis_l, basis_r) = self
            .stage_pipeline
            .stage2
            .step_frame(s1_l, s1_r, drive_gain, sag_depth);
        let s2_l = basis_l[0];
        let s2_r = basis_r[0];

        // Stage III: Cabinet Modals (Early FIR + Stereo Kautz Modal Bank driven by 16-sample-delayed tap + Clean Sub crossover)
        let (s3_fir_l, s3_fir_r, s2_delayed_16_l, s2_delayed_16_r) = self
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

        let arena_len = self.arena_storage.as_bytes().len();
        let active_tiers =
            (self.effective_tier() as usize + 1).min(max_tiers_for_arena_len(arena_len));

        if self.cached_head_tiers != active_tiers
            || self.cached_z_head != self.control_bus.z_head
            || self.cached_z_head_ultra != self.control_bus.z_head_ultra
        {
            self.refresh_collapsed_head_weights(active_tiers);
        }

        let mut head_acc_l = B::Vec4::zero();
        let mut head_acc_r = B::Vec4::zero();
        let mut kautz_acc_l = B::Vec4::zero();
        let mut kautz_acc_r = B::Vec4::zero();
        let xl_vec = B::Vec4::splat(s2_delayed_16_l);
        let xr_vec = B::Vec4::splat(s2_delayed_16_r);
        for t in 0..active_tiers {
            let kwl = &self.collapsed_kautz_wl[t];
            let kwr = &self.collapsed_kautz_wr[t];
            let tiers_l = &mut self.stage_pipeline.kautz_tiers_l[t];
            let tiers_r = &mut self.stage_pipeline.kautz_tiers_r[t];
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
            for blk in 0..n_blocks {
                let (vl, vr) = tiers_l[blk].step_4lane_stereo_simd_splatted::<B>(
                    &mut tiers_r[blk],
                    xl_vec,
                    xr_vec,
                );
                kautz_acc_l = kautz_acc_l.add(vl);
                kautz_acc_r = kautz_acc_r.add(vr);
                head_acc_l = head_acc_l.fma(kwl[blk], vl).fma(kwl[4 + blk], vr);
                if !mono_only {
                    head_acc_r = head_acc_r.fma(kwr[blk], vl).fma(kwr[4 + blk], vr);
                }
            }
        }
        let s3_l = s3_fir_l + 0.125 * kautz_acc_l.horizontal_sum();
        let s3_r = s3_fir_r + 0.125 * kautz_acc_r.horizontal_sum();

        // Stage IV: Unitary Spatial Tank (Givens+Householder + Diffusers + DC Blocker + Ducking)
        let (tank_l, tank_r) = self.stage_pipeline.stage4.step_stereo(s3_l, s3_r);

        // Multi-Tier 64-D / 256-D Feature Bus & 3 kHz SVD-Collapsed Kronecker Output Head (`ref[dsp.head.mid-side-factorization]`)
        let shared_phi: [B::Vec4; 7] = [
            B::Vec4::from_array([basis_l[0], basis_l[1], basis_l[2], basis_l[3]]),
            B::Vec4::from_array([basis_l[4], basis_l[5], basis_l[6], basis_l[7]]),
            B::Vec4::from_array([basis_r[0], basis_r[1], basis_r[2], basis_r[3]]),
            B::Vec4::from_array([basis_r[4], basis_r[5], basis_r[6], basis_r[7]]),
            B::Vec4::from_array([s1_l, s1_r, s3_fir_l, s3_fir_r]),
            B::Vec4::from_array([tank_l, tank_r, clean_sub_l, clean_sub_r]),
            B::Vec4::from_array([s2_l, s2_r, s3_l, s3_r]),
        ];
        if mono_only {
            for (g, &phi_g) in shared_phi.iter().enumerate() {
                head_acc_l = head_acc_l.fma(self.collapsed_shared_wl[g], phi_g);
            }
        } else {
            for (g, &phi_g) in shared_phi.iter().enumerate() {
                head_acc_l = head_acc_l.fma(self.collapsed_shared_wl[g], phi_g);
                head_acc_r = head_acc_r.fma(self.collapsed_shared_wr[g], phi_g);
            }
        }

        let [b_head_l, b_head_r] = self.collapsed_b_head;

        let mut wet_l = s3_l + tank_l + head_acc_l.horizontal_sum() + b_head_l;
        let mut wet_r = s3_r + tank_r + head_acc_r.horizontal_sum() + b_head_r;

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

    /// Low-level 4-port (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_or_wdw_3, send_4]`)
    /// TORA stepping with optional 256-D feature bus capture (`ref[dsp.router.ports]`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
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

        let k_ctrl = self.zoh.control_decimation_host;
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            let avg_energy = self.ctrl_energy_accum * self.zoh.alpha_ramp_host;
            self.ctrl_sub_counter = 0;
            self.ctrl_energy_accum = 0.0;
            let det_env = self.stage_pipeline.stage0.detector_envelopes;
            let arena = self.arena_storage.as_bytes();
            let _changed =
                self.control_bus
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
        }
        crate::control::ControlBus::<B>::step_micro_ramp(
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
        );

        let (down_l, down_r, clean_sub_l, clean_sub_r) = self
            .stage_pipeline
            .stage0
            .route_frame_no_return(eff_in_l, eff_in_r);

        let (s1_l, s1_r) = self.stage_pipeline.stage1.step_stereo(down_l, down_r);

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

        let stage_law_5 = (1.0 + 0.25 * self.current_stage_params[5]).clamp(0.25, 4.0);
        let stage_law_8 = (0.2 + 0.1 * self.current_stage_params[8]).clamp(0.0, 0.8);
        let drive_gain = (stage_law_5 + 0.25 * self.control_bus.ctrl_basis_4[0]).clamp(0.25, 4.0);
        let sag_depth = (stage_law_8 + 0.1 * self.control_bus.ctrl_basis_4[1]).clamp(0.0, 0.8);
        let (basis_l, basis_r) = self
            .stage_pipeline
            .stage2
            .step_frame(s2_in_l, s2_in_r, drive_gain, sag_depth);
        let s2_l = basis_l[0];
        let s2_r = basis_r[0];

        let (s3_fir_l, s3_fir_r, s2_delayed_16_l, s2_delayed_16_r) = self
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

        let arena_len = self.arena_storage.as_bytes().len();
        let active_tiers =
            (self.effective_tier() as usize + 1).min(max_tiers_for_arena_len(arena_len));

        if self.cached_head_tiers != active_tiers
            || self.cached_z_head != self.control_bus.z_head
            || self.cached_z_head_ultra != self.control_bus.z_head_ultra
        {
            self.refresh_collapsed_head_weights(active_tiers);
        }

        if let Some(ref mut fb) = feature_bus_out {
            fb.fill(0.0);
        }

        let mut head_acc_l = B::Vec4::zero();
        let mut head_acc_r = B::Vec4::zero();
        let mut kautz_acc_l = B::Vec4::zero();
        let mut kautz_acc_r = B::Vec4::zero();
        let xl_vec = B::Vec4::splat(s2_delayed_16_l);
        let xr_vec = B::Vec4::splat(s2_delayed_16_r);
        let kautz_loop_tiers = if feature_bus_out.is_some() {
            4
        } else {
            active_tiers
        };
        for t in 0..kautz_loop_tiers {
            let kwl = &self.collapsed_kautz_wl[t];
            let kwr = &self.collapsed_kautz_wr[t];
            let tiers_l = &mut self.stage_pipeline.kautz_tiers_l[t];
            let tiers_r = &mut self.stage_pipeline.kautz_tiers_r[t];
            let n_blocks = crate::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
            for blk in 0..n_blocks {
                let (vl, vr) = tiers_l[blk].step_4lane_stereo_simd_splatted::<B>(
                    &mut tiers_r[blk],
                    xl_vec,
                    xr_vec,
                );
                if let Some(ref mut fb) = feature_bus_out {
                    let modal_feat = crate::stages::pack_stage3_modal_block_features(
                        &vl.to_array(),
                        &vr.to_array(),
                    );
                    let m_off = t * 64 + blk * 4;
                    fb[m_off..m_off + 4].copy_from_slice(&modal_feat);
                }
                if t < active_tiers {
                    kautz_acc_l = kautz_acc_l.add(vl);
                    kautz_acc_r = kautz_acc_r.add(vr);
                    head_acc_l = head_acc_l.fma(kwl[blk], vl).fma(kwl[4 + blk], vr);
                    if !mono_only {
                        head_acc_r = head_acc_r.fma(kwr[blk], vl).fma(kwr[4 + blk], vr);
                    }
                }
            }
        }
        let s3_l = s3_fir_l + 0.125 * kautz_acc_l.horizontal_sum();
        let s3_r = s3_fir_r + 0.125 * kautz_acc_r.horizontal_sum();

        let (tank_l, tank_r, tank_readouts, tank_loops) = self
            .stage_pipeline
            .stage4
            .step_stereo_with_readouts(s3_l, s3_r);

        let shared_phi: [B::Vec4; 7] = [
            B::Vec4::from_array([basis_l[0], basis_l[1], basis_l[2], basis_l[3]]),
            B::Vec4::from_array([basis_l[4], basis_l[5], basis_l[6], basis_l[7]]),
            B::Vec4::from_array([basis_r[0], basis_r[1], basis_r[2], basis_r[3]]),
            B::Vec4::from_array([basis_r[4], basis_r[5], basis_r[6], basis_r[7]]),
            B::Vec4::from_array([s1_l, s1_r, s3_fir_l, s3_fir_r]),
            B::Vec4::from_array([tank_l, tank_r, clean_sub_l, clean_sub_r]),
            B::Vec4::from_array([s2_l, s2_r, s3_l, s3_r]),
        ];
        if let Some(fb) = feature_bus_out.as_mut() {
            // Assemble CANONICAL-256D-FEATURE-BUS-MAP (`r[dsp.head.feature-bus-slots]`, `R15-E2`, `R16-E3`)
            let fir_0_l = s3_fir_l - clean_sub_l;
            let fir_0_r = s3_fir_r - clean_sub_r;
            let latest_l = (self.stage_pipeline.stage3.fir_write_ptr + 63) & 63;
            let latest_r = (self.stage_pipeline.stage3_r.fir_write_ptr + 63) & 63;
            let mut fir_seg_l = [fir_0_l, 0.0, 0.0, 0.0];
            let mut fir_seg_r = [fir_0_r, 0.0, 0.0, 0.0];
            for t in 1..4 {
                let mut acc_l = 0.0_f32;
                let mut acc_r = 0.0_f32;
                for i in 0..16 {
                    let h_l = self.stage_pipeline.stage3.fir_history[(latest_l + 64 - i) & 63];
                    let h_r = self.stage_pipeline.stage3_r.fir_history[(latest_r + 64 - i) & 63];
                    acc_l += self.stage_pipeline.fir_taps_l[t][i] * h_l;
                    acc_r += self.stage_pipeline.fir_taps_r[t][i] * h_r;
                }
                fir_seg_l[t] = acc_l;
                fir_seg_r[t] = acc_r;
            }
            let s1 = &self.stage_pipeline.stage1;
            let s2 = &self.stage_pipeline.stage2;
            let s4 = &self.stage_pipeline.stage4;

            // Lite (0..64):
            // 0..4 already filled by Stage III Kautz Block 0
            fb[4..8].copy_from_slice(&s1.last_tracker_l);
            fb[8..12].copy_from_slice(&s1.last_tracker_r);
            fb[12..16].copy_from_slice(&s1.op_m_pre.z_re);
            fb[16..20].copy_from_slice(&s1.op_m_pre_r.z_re);
            fb[20..24].copy_from_slice(&basis_l[0..4]);
            fb[24..28].copy_from_slice(&basis_l[4..8]);
            fb[28..32].copy_from_slice(&basis_r[0..4]);
            fb[32..36].copy_from_slice(&basis_r[4..8]);
            fb[36..44].copy_from_slice(&s2.last_v1b_hyst);
            fb[44..48].copy_from_slice(&s2.v1a_left.u_prev);
            fb[48..52].copy_from_slice(&s2.v1a_right.u_prev);
            fb[52..56].copy_from_slice(&tank_readouts);
            fb[56] = tank_l;
            fb[57] = tank_r;
            fb[58] = s1_l + s2_l;
            fb[59] = s1_r + s2_r;
            fb[60] = clean_sub_l;
            fb[61] = clean_sub_r;
            fb[62] = fir_seg_l[0];
            fb[63] = fir_seg_r[0];

            // Full (64..128):
            // 64..68 already filled by Stage III Kautz Block 1
            fb[68..72].copy_from_slice(&s1.op_m_pre.z_im);
            fb[72..76].copy_from_slice(&s1.op_m_pre_r.z_im);
            fb[76..80].copy_from_slice(&[s1_l, s1_r, down_l, down_r]);
            fb[80..84].copy_from_slice(&[s2_in_l, s2_in_r, send_l, send_r]);
            fb[84..92].copy_from_slice(&basis_l);
            fb[92..100].copy_from_slice(&basis_r);
            fb[100..108].copy_from_slice(&s2.last_v1b_hyst);
            fb[108..112].copy_from_slice(&s2.v1b_left.u_prev);
            fb[112..116].copy_from_slice(&s2.v1b_right.u_prev);
            fb[116..120].copy_from_slice(&tank_loops);
            fb[120..124].copy_from_slice(&tank_readouts);
            fb[124] = s4.ducking_vca.env;
            fb[125] = s2.sag_envelope;
            fb[126] = fir_seg_l[1];
            fb[127] = fir_seg_r[1];

            // Pro (128..192):
            // 128..136 already filled by Stage III Kautz Blocks 2..3
            fb[136..140].copy_from_slice(&s1.last_tracker_l);
            fb[140..144].copy_from_slice(&s1.last_tracker_r);
            fb[144..148].copy_from_slice(&s1.op_m_pre.z_re);
            fb[148..152].copy_from_slice(&s1.op_m_pre_r.z_re);
            fb[152..160].copy_from_slice(&basis_l);
            fb[160..168].copy_from_slice(&basis_r);
            fb[168..176].copy_from_slice(&s2.last_v1b_hyst);
            fb[176..180].copy_from_slice(&s2.v1a_left.a_prev);
            fb[180..184].copy_from_slice(&s2.v1a_right.a_prev);
            fb[184..188].copy_from_slice(&tank_readouts);
            fb[188] = tank_l;
            fb[189] = tank_r;
            fb[190] = fir_seg_l[2];
            fb[191] = fir_seg_r[2];

            // Ultra (192..256):
            // 192..208 already filled by Stage III Kautz Blocks 4..7
            fb[208..212].copy_from_slice(&s1.op_m_pre.z_im);
            fb[212..216].copy_from_slice(&s1.op_m_pre_r.z_im);
            fb[216..220].copy_from_slice(&s1.last_tracker_l);
            fb[220..224].copy_from_slice(&s1.last_tracker_r);
            fb[224..232].copy_from_slice(&basis_l);
            fb[232..240].copy_from_slice(&basis_r);
            fb[240..246].copy_from_slice(&s2.last_v1b_hyst[0..6]);
            fb[246..250].copy_from_slice(&tank_loops);
            fb[250..254].copy_from_slice(&tank_readouts);
            fb[254] = fir_seg_l[3];
            fb[255] = fir_seg_r[3];
        }

        if mono_only {
            for (g, &phi_g) in shared_phi.iter().enumerate() {
                head_acc_l = head_acc_l.fma(self.collapsed_shared_wl[g], phi_g);
            }
        } else {
            for (g, &phi_g) in shared_phi.iter().enumerate() {
                head_acc_l = head_acc_l.fma(self.collapsed_shared_wl[g], phi_g);
                head_acc_r = head_acc_r.fma(self.collapsed_shared_wr[g], phi_g);
            }
        }

        let [b_head_l, b_head_r] = self.collapsed_b_head;

        let head_sum_l = head_acc_l.horizontal_sum();
        let head_sum_r = head_acc_r.horizontal_sum();
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
            let w5_l = self.collapsed_shared_wl[5].to_array();
            let w5_r = self.collapsed_shared_wr[5].to_array();
            let tank_head_l = w5_l[0] * tank_l + w5_l[1] * tank_r;
            let tank_head_r = w5_r[0] * tank_l + w5_r[1] * tank_r;
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
    #[inline]
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
