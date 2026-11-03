//! Real-time audio processing loops for [`DnamEngine`].

use super::DnamEngine;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};
use crate::provenance::watermark::{max_tiers_for_arena_len, read_f32_le};
use crate::types::FtzDazGuard;

#[inline(always)]
fn read_vec4_le<B: MathBackend>(bytes: &[u8], off: usize) -> B::Vec4 {
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
    #[inline(always)]
    pub fn step_frame_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
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
            let arena = self.arena_storage.as_bytes();
            let _changed = self.control_bus.tick_3khz(avg_energy, arena);
            if self.control_bus.tank_mod_dirty {
                self.apply_tank_modulations();
                self.control_bus.tank_mod_dirty = false;
            }
        }

        // Stage 0: 8-Port Matrix Router
        let (_send_l, _send_r, down_l, down_r, clean_sub_l, clean_sub_r) = self
            .stage_pipeline
            .stage0
            .route_frame(eff_in_l, eff_in_r, 0.0, 0.0);

        // Stage I: Pre-Dynamics
        let (s1_l, s1_r) = self.stage_pipeline.stage1.step_stereo(down_l, down_r);

        // Stage II: Commuted Cascade (Tone Basis + Non-linear PolyADAA)
        let drive_gain = (1.0 + 0.25 * self.control_bus.ctrl_basis_4[0]).clamp(0.25, 4.0);
        let sag_depth = (0.2 + 0.1 * self.control_bus.ctrl_basis_4[1]).clamp(0.0, 0.8);
        let (basis_l, basis_r) = self
            .stage_pipeline
            .stage2
            .step_frame(s1_l, s1_r, drive_gain, sag_depth);
        let s2_l = basis_l[0];
        let s2_r = basis_r[0];

        // Stage III: Cabinet Modals (Early FIR + Stereo Kautz Modal Bank + Clean Sub crossover)
        let fir_taps = self.stage_pipeline.fir_taps;
        let s3_fir_l = self
            .stage_pipeline
            .stage3
            .step(s2_l, clean_sub_l, &fir_taps);
        let s3_fir_r = self
            .stage_pipeline
            .stage3_r
            .step(s2_r, clean_sub_r, &fir_taps);

        let arena = self.arena_storage.as_bytes();
        let active_tiers =
            (self.effective_tier() as usize + 1).min(max_tiers_for_arena_len(arena.len()));

        let mut kautz_vecs_l = [[B::Vec4::zero(); 4]; 4];
        let mut kautz_vecs_r = [[B::Vec4::zero(); 4]; 4];
        let mut kautz_sum_l = 0.0_f32;
        let mut kautz_sum_r = 0.0_f32;
        for t in 0..active_tiers {
            for blk in 0..4 {
                let kl = self.stage_pipeline.kautz_tiers_l[t][blk].step_4lane_simd::<B>(s2_l);
                let kr = self.stage_pipeline.kautz_tiers_r[t][blk].step_4lane_simd::<B>(s2_r);
                let vl = B::Vec4::from_array(kl);
                let vr = B::Vec4::from_array(kr);
                kautz_sum_l += vl.horizontal_sum();
                kautz_sum_r += vr.horizontal_sum();
                kautz_vecs_l[t][blk] = vl;
                kautz_vecs_r[t][blk] = vr;
            }
        }
        let s3_l = s3_fir_l + 0.125 * kautz_sum_l;
        let s3_r = s3_fir_r + 0.125 * kautz_sum_r;

        // Stage IV: Unitary Spatial Tank (Givens+Householder + Diffusers + DC Blocker + Ducking)
        let (tank_l, tank_r) = self.stage_pipeline.stage4.step_stereo(s3_l, s3_r);

        // Multi-Tier 64-D / 256-D Feature Bus & SVD-Collapsed Kronecker Output Head
        let mut head_acc_l = B::Vec4::zero();
        let mut head_acc_r = B::Vec4::zero();
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

        const CASCADE_W_BASES: [usize; 4] = [1_024, 11_264, 22_336, 36_224];
        const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];

        let b_l0 = B::Vec4::from_array([basis_l[0], basis_l[1], basis_l[2], basis_l[3]]);
        let b_l1 = B::Vec4::from_array([basis_l[4], basis_l[5], basis_l[6], basis_l[7]]);
        let b_r0 = B::Vec4::from_array([basis_r[0], basis_r[1], basis_r[2], basis_r[3]]);
        let b_r1 = B::Vec4::from_array([basis_r[4], basis_r[5], basis_r[6], basis_r[7]]);
        let stage_vec12 = B::Vec4::from_array([s1_l, s1_r, s3_fir_l, s3_fir_r]);
        let stage_vec13 = B::Vec4::from_array([tank_l, tank_r, clean_sub_l, clean_sub_r]);
        let stage_vec14 = B::Vec4::from_array([s2_l, s2_r, s3_l, s3_r]);
        let stage_vec15 = B::Vec4::zero();

        for t in 0..active_tiers {
            let cw_base = CASCADE_W_BASES[t];
            let w_c0 = read_vec4_le::<B>(arena, cw_base);
            let w_c1 = read_vec4_le::<B>(arena, cw_base + 16);
            let w_c2 = read_vec4_le::<B>(arena, cw_base + 32);
            let w_c3 = read_vec4_le::<B>(arena, cw_base + 48);

            let phi_t: [B::Vec4; 16] = [
                kautz_vecs_l[t][0],
                kautz_vecs_l[t][1],
                kautz_vecs_l[t][2],
                kautz_vecs_l[t][3],
                kautz_vecs_r[t][0],
                kautz_vecs_r[t][1],
                kautz_vecs_r[t][2],
                kautz_vecs_r[t][3],
                b_l0.fma(w_c0, b_l0),
                b_l1.fma(w_c1, b_l1),
                b_r0.fma(w_c2, b_r0),
                b_r1.fma(w_c3, b_r1),
                stage_vec12,
                stage_vec13,
                stage_vec14,
                stage_vec15,
            ];

            let h_base = HEAD_BASES[t];
            let u_base = 49_408 + t * 2048;
            for (g, &phi_g) in phi_t.iter().enumerate() {
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
                head_acc_l = head_acc_l.fma(wl, phi_g);
                head_acc_r = head_acc_r.fma(wr, phi_g);
            }
        }

        let mut b_head_l = 0.0_f32;
        let mut b_head_r = 0.0_f32;
        if active_tiers >= 1 {
            b_head_l = read_f32_le(arena, 9_024);
            b_head_r = read_f32_le(arena, 9_028);
            for k in 0..4 {
                b_head_l += z_h[k] * read_f32_le(arena, 9_024 + ((k + 1) * 2) * 4);
                b_head_r += z_h[k] * read_f32_le(arena, 9_024 + ((k + 1) * 2 + 1) * 4);
            }
            if active_tiers >= 4 {
                for k in 0..4 {
                    b_head_l += z_u[k] * read_f32_le(arena, 63_424 + (k * 2) * 4);
                    b_head_r += z_u[k] * read_f32_le(arena, 63_424 + (k * 2 + 1) * 4);
                }
            }
        }

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

        let out_l = eff_dry_g * dry_l + eff_wet_g * wet_l;
        let out_r = if mono_only {
            out_l
        } else {
            eff_dry_g * dry_r + eff_wet_g * wet_r
        };

        [out_l, out_r]
    }

    /// Internal 4x oversampled stepping.
    #[inline(always)]
    pub(crate) fn step_frame_4x_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
        let up_frames = self.oversampler.upsample_4x(in_l, in_r);
        let mut proc_frames = [[0.0_f32; 2]; 4];
        for (m, frame) in up_frames.iter().enumerate() {
            proc_frames[m] = self.step_frame_tora(frame[0], frame[1], mono_only);
        }
        self.oversampler.downsample_4x(proc_frames)
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
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
        if !self.preview_guard.advance_block(num_frames) {
            left_out[..num_frames].fill(0.0);
            right_out[..num_frames].fill(0.0);
            return;
        }

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
        if !self.preview_guard.advance_block(num_frames) {
            left_out[..num_frames].fill(0.0);
            right_out[..num_frames].fill(0.0);
            return;
        }

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
        if !self.preview_guard.advance_block(num_frames) {
            mono_out[..num_frames].fill(0.0);
            return;
        }

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
        if !self.preview_guard.advance_block(num_frames) {
            left_io[..num_frames].fill(0.0);
            right_io[..num_frames].fill(0.0);
            return;
        }

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

        // Expired preview guard mutes all process variants
        engine.preview_guard.is_expired = true;
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert_eq!(l_out, [0.0f32; 16]);
        l_out = [1.0f32; 16];
        r_out = [1.0f32; 16];
        engine.process_mono_to_stereo(&l, &mut l_out, &mut r_out);
        assert_eq!(l_out, [0.0f32; 16]);
        l_out = [1.0f32; 16];
        engine.process_mono(&l, &mut l_out);
        assert_eq!(l_out, [0.0f32; 16]);
        let mut l_ip = [1.0f32; 16];
        let mut r_ip = [1.0f32; 16];
        engine.process_block_in_place(&mut l_ip, &mut r_ip);
        assert_eq!(l_ip, [0.0f32; 16]);
    }
}
