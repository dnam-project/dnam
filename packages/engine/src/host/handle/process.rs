//! Sub-block audio rendering (`process_*`) and offline bounce (`render_offline_bounce*`)
//! for [`DnamHostHandle`] (`ref[dsp.ctrl.op-m.spsc-ingress]`, `ref[arch.telemetry.seqlock-publication]`).

use crate::container::ZohRescaledParams;
use crate::engine::OFFLINE_BOUNCE_PDC_SAMPLES;
use crate::host::telemetry::RawTelemetryRecord;

use super::{DEFAULT_HOST_SUBBLOCK_FRAMES, DnamHostHandle};

impl DnamHostHandle {
    /// Processes a stereo audio buffer (`Stereo -> Stereo`), draining SPSC control updates and
    /// publishing wait-free Seqlock telemetry at every sub-block boundary.
    pub fn process_block_chunked(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        block_size: Option<usize>,
    ) {
        let total = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        let step = block_size
            .unwrap_or(DEFAULT_HOST_SUBBLOCK_FRAMES)
            .max(1)
            .min(total.max(1));
        let mut audio_ctx = self.audio.lock();
        if total == 0 {
            audio_ctx.drain_and_publish();
            return;
        }
        let mut offset = 0_usize;
        while offset < total {
            let end = (offset + step).min(total);
            audio_ctx.poll_and_drain();
            audio_ctx.swap_audio.engine_mut().process_block(
                &left_in[offset..end],
                &right_in[offset..end],
                &mut left_out[offset..end],
                &mut right_out[offset..end],
            );
            audio_ctx.publish_telemetry();
            offset = end;
        }
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
    #[inline]
    pub fn process_block(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        self.process_block_chunked(left_in, right_in, left_out, right_out, None);
    }

    /// Processes a stereo audio block in-place (`left_io`, `right_io`) in sub-blocks, draining
    /// SPSC control updates and publishing telemetry at every sub-block boundary.
    pub fn process_block_in_place_chunked(
        &self,
        left_io: &mut [f32],
        right_io: &mut [f32],
        block_size: Option<usize>,
    ) {
        let total = left_io.len().min(right_io.len());
        let step = block_size
            .unwrap_or(DEFAULT_HOST_SUBBLOCK_FRAMES)
            .max(1)
            .min(total.max(1));
        let mut audio_ctx = self.audio.lock();
        if total == 0 {
            audio_ctx.drain_and_publish();
            return;
        }
        let mut offset = 0_usize;
        while offset < total {
            let end = (offset + step).min(total);
            audio_ctx.poll_and_drain();
            audio_ctx
                .swap_audio
                .engine_mut()
                .process_block_in_place(&mut left_io[offset..end], &mut right_io[offset..end]);
            audio_ctx.publish_telemetry();
            offset = end;
        }
    }

    /// Processes a stereo audio block in-place (`left_io`, `right_io`), draining SPSC control
    /// updates and publishing telemetry at every sub-block boundary.
    #[inline]
    pub fn process_block_in_place(&self, left_io: &mut [f32], right_io: &mut [f32]) {
        self.process_block_in_place_chunked(left_io, right_io, None);
    }

    /// Processes a mono audio buffer into true stereo (`Mono -> Stereo`), draining SPSC control
    /// updates and publishing telemetry at every sub-block boundary.
    pub fn process_mono_to_stereo_chunked(
        &self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        block_size: Option<usize>,
    ) {
        let total = mono_in.len().min(left_out.len()).min(right_out.len());
        let step = block_size
            .unwrap_or(DEFAULT_HOST_SUBBLOCK_FRAMES)
            .max(1)
            .min(total.max(1));
        let mut audio_ctx = self.audio.lock();
        if total == 0 {
            audio_ctx.drain_and_publish();
            return;
        }
        let mut offset = 0_usize;
        while offset < total {
            let end = (offset + step).min(total);
            audio_ctx.poll_and_drain();
            audio_ctx.swap_audio.engine_mut().process_mono_to_stereo(
                &mono_in[offset..end],
                &mut left_out[offset..end],
                &mut right_out[offset..end],
            );
            audio_ctx.publish_telemetry();
            offset = end;
        }
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    #[inline]
    pub fn process_mono_to_stereo(
        &self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        self.process_mono_to_stereo_chunked(mono_in, left_out, right_out, None);
    }

    /// Processes a mono audio buffer (`Mono -> Mono`), draining SPSC control updates and
    /// publishing telemetry at every sub-block boundary.
    pub fn process_mono_chunked(
        &self,
        mono_in: &[f32],
        mono_out: &mut [f32],
        block_size: Option<usize>,
    ) {
        let total = mono_in.len().min(mono_out.len());
        let step = block_size
            .unwrap_or(DEFAULT_HOST_SUBBLOCK_FRAMES)
            .max(1)
            .min(total.max(1));
        let mut audio_ctx = self.audio.lock();
        if total == 0 {
            audio_ctx.drain_and_publish();
            return;
        }
        let mut offset = 0_usize;
        while offset < total {
            let end = (offset + step).min(total);
            audio_ctx.poll_and_drain();
            audio_ctx
                .swap_audio
                .engine_mut()
                .process_mono(&mono_in[offset..end], &mut mono_out[offset..end]);
            audio_ctx.publish_telemetry();
            offset = end;
        }
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    #[inline]
    pub fn process_mono(&self, mono_in: &[f32], mono_out: &mut [f32]) {
        self.process_mono_chunked(mono_in, mono_out, None);
    }

    /// Processes a mono audio buffer in-place (`mono_io`), draining SPSC control updates and
    /// publishing telemetry at every sub-block boundary.
    pub fn process_mono_in_place_chunked(&self, mono_io: &mut [f32], block_size: Option<usize>) {
        let total = mono_io.len();
        let step = block_size
            .unwrap_or(DEFAULT_HOST_SUBBLOCK_FRAMES)
            .clamp(1, DEFAULT_HOST_SUBBLOCK_FRAMES)
            .min(total.max(1));
        let mut audio_ctx = self.audio.lock();
        if total == 0 {
            audio_ctx.drain_and_publish();
            return;
        }
        let mut scratch = [0.0_f32; DEFAULT_HOST_SUBBLOCK_FRAMES];
        let mut offset = 0_usize;
        while offset < total {
            let end = (offset + step).min(total);
            let len = end - offset;
            scratch[..len].copy_from_slice(&mono_io[offset..end]);
            audio_ctx.poll_and_drain();
            audio_ctx
                .swap_audio
                .engine_mut()
                .process_mono(&scratch[..len], &mut mono_io[offset..end]);
            audio_ctx.publish_telemetry();
            offset = end;
        }
    }

    /// Processes a mono audio buffer in-place (`mono_io`).
    #[inline]
    pub fn process_mono_in_place(&self, mono_io: &mut [f32]) {
        self.process_mono_in_place_chunked(mono_io, None);
    }

    /// Renders a complete stereo buffer in Offline Bounce mode with an optional per-call
    /// `maximize_tier` override, stripping the 12-sample PDC lead-in and draining SPSC control
    /// updates at every sub-block boundary.
    pub fn render_offline_bounce_with_options(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
        maximize_tier: Option<bool>,
    ) {
        let mut audio_ctx = self.audio.lock();
        audio_ctx.poll_and_drain();

        let mut prev_max = audio_ctx.swap_audio.engine().maximize_tier_on_bounce();
        let mut prev_bounce = audio_ctx.swap_audio.engine().offline_bounce();
        let mut non_bounce_zoh = if prev_bounce {
            let (r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr) =
                audio_ctx.swap_audio.engine().training_zoh_poles();
            ZohRescaledParams::compute_oversampled(
                audio_ctx
                    .swap_audio
                    .engine()
                    .preamble()
                    .sample_rate_train_hz,
                audio_ctx.swap_audio.engine().sample_rate_hz(),
                1,
                r_ac_tr,
                r_dc_tr,
                tau_ref_tr,
                switch_xfade_tr,
            )
            .unwrap_or(*audio_ctx.swap_audio.engine().zoh())
        } else {
            *audio_ctx.swap_audio.engine().zoh()
        };

        if let Some(max_t) = maximize_tier {
            audio_ctx
                .swap_audio
                .engine_mut()
                .set_maximize_tier_on_bounce(max_t);
        }
        if !prev_bounce {
            audio_ctx.swap_audio.engine_mut().set_offline_bounce(true);
        }
        audio_ctx.swap_audio.engine_mut().reset_to_dc_equilibrium();

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
            audio_ctx.swap_audio.engine_mut().process_block(
                &left_in[..n],
                &right_in[..n],
                &mut sink_l[..n],
                &mut sink_r[..n],
            );
            let rem = pdc - n;
            if rem > 0 {
                audio_ctx.swap_audio.engine_mut().process_block(
                    &zeros[..rem],
                    &zeros[..rem],
                    &mut sink_l[..rem],
                    &mut sink_r[..rem],
                );
            }
            audio_ctx.swap_audio.engine_mut().process_block(
                &zeros[..n],
                &zeros[..n],
                &mut left_out[..n],
                &mut right_out[..n],
            );
        } else {
            audio_ctx.swap_audio.engine_mut().process_block(
                &left_in[..pdc],
                &right_in[..pdc],
                &mut sink_l,
                &mut sink_r,
            );
            let body_len = n - pdc;
            let mut cur_max = prev_max;
            let mut cur_bounce = prev_bounce;
            let mut offset = 0_usize;
            while offset < body_len {
                let end = (offset + DEFAULT_HOST_SUBBLOCK_FRAMES).min(body_len);
                let swap_epoch = audio_ctx.swap_audio.poll_swap_epoch();
                if let Some(ep) = swap_epoch {
                    audio_ctx.consumer.advance_causal_epoch(ep);
                    cur_max = audio_ctx.swap_audio.engine().maximize_tier_on_bounce();
                    cur_bounce = audio_ctx.swap_audio.engine().offline_bounce();
                    non_bounce_zoh = *audio_ctx.swap_audio.engine().zoh();
                }
                if swap_epoch.is_some() || audio_ctx.consumer.has_pending() {
                    let ctx = &mut *audio_ctx;
                    let (_, overrides) = ctx
                        .consumer
                        .drain_into_tracking_bounce(ctx.swap_audio.engine_mut());
                    if let Some(m) = overrides.maximize_tier_on_bounce {
                        cur_max = m;
                    }
                    if let Some(b) = overrides.offline_bounce {
                        cur_bounce = b;
                    }
                    let bounce_max_t = maximize_tier.unwrap_or(cur_max);
                    audio_ctx
                        .swap_audio
                        .engine_mut()
                        .set_maximize_tier_on_bounce(bounce_max_t);
                    audio_ctx.swap_audio.engine_mut().set_offline_bounce(true);
                }
                audio_ctx.swap_audio.engine_mut().process_block(
                    &left_in[pdc + offset..pdc + end],
                    &right_in[pdc + offset..pdc + end],
                    &mut left_out[offset..end],
                    &mut right_out[offset..end],
                );
                {
                    let ctx = &mut *audio_ctx;
                    ctx.consumer.publish_total_frames(ctx.swap_audio.engine());
                }
                let mut rec = RawTelemetryRecord::from_engine(
                    audio_ctx.swap_audio.engine(),
                    audio_ctx.consumer.drained_causal_epoch(),
                );
                rec.offline_bounce = cur_bounce;
                rec.maximize_tier_on_bounce = cur_max;
                rec.latency_samples = if cur_bounce {
                    OFFLINE_BOUNCE_PDC_SAMPLES
                } else {
                    0
                };
                if !cur_bounce {
                    rec.zoh = non_bounce_zoh;
                    rec.snapshot.control_decimation_host =
                        non_bounce_zoh.control_decimation_host as u16;
                }
                let eff_tier = if cur_bounce && cur_max {
                    rec.max_available_tier.min(rec.max_supported_tier)
                } else {
                    rec.requested_tier
                        .min(rec.max_available_tier)
                        .min(rec.max_supported_tier)
                };
                rec.snapshot.active_tier = eff_tier as u8;
                audio_ctx.telemetry_writer.publish(rec);
                offset = end;
            }
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            audio_ctx.swap_audio.engine_mut().process_block(
                &zeros,
                &zeros,
                &mut left_out[body_len..n],
                &mut right_out[body_len..n],
            );
            prev_max = cur_max;
            prev_bounce = cur_bounce;
        }

        audio_ctx
            .swap_audio
            .engine_mut()
            .set_maximize_tier_on_bounce(prev_max);
        audio_ctx
            .swap_audio
            .engine_mut()
            .set_offline_bounce(prev_bounce);
        audio_ctx.poll_and_drain();
        audio_ctx.publish_telemetry();
    }

    /// Renders a complete stereo buffer in Offline Bounce mode, stripping the 12-sample PDC
    /// lead-in and draining SPSC control updates at every sub-block boundary.
    #[inline]
    pub fn render_offline_bounce(
        &self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        self.render_offline_bounce_with_options(left_in, right_in, left_out, right_out, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use crate::types::Tier;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    fn build_process_test_container() -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).unwrap()
    }

    #[test]
    fn test_all_five_audio_topologies_empty_buffers_and_offline_bounce_branches() {
        // verify[dsp.ctrl.op-m.spsc-ingress]
        // verify[arch.telemetry.seqlock-publication]
        let container = build_process_test_container();
        let host = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();

        let mut empty_a: [f32; 0] = [];
        let mut empty_b: [f32; 0] = [];
        host.process_block(&[], &[], &mut empty_a, &mut empty_b);
        host.process_block_in_place(&mut empty_a, &mut empty_b);
        host.process_mono_to_stereo(&[], &mut empty_a, &mut empty_b);
        host.process_mono(&[], &mut empty_a);
        host.process_mono_in_place(&mut empty_a);

        let in_l = vec![0.05_f32; 640];
        let in_r = vec![-0.05_f32; 640];
        let mut out_l = vec![0.0_f32; 640];
        let mut out_r = vec![0.0_f32; 640];
        host.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|x| x.is_finite()));

        let mut io_l = in_l.clone();
        let mut io_r = in_r.clone();
        host.process_block_in_place(&mut io_l, &mut io_r);
        host.process_block_in_place_chunked(&mut io_l, &mut io_r, Some(128));
        assert!(io_l.iter().all(|x| x.is_finite()));

        host.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
        host.process_mono_to_stereo_chunked(&in_l, &mut out_l, &mut out_r, Some(128));
        assert!(out_r.iter().all(|x| x.is_finite()));

        host.process_mono(&in_l, &mut out_l);
        host.process_mono_chunked(&in_l, &mut out_l, Some(128));
        assert!(out_l.iter().all(|x| x.is_finite()));

        let mut mono_io = in_l.clone();
        host.process_mono_in_place(&mut mono_io);
        host.process_mono_in_place_chunked(&mut mono_io, Some(128));
        assert!(mono_io.iter().all(|x| x.is_finite()));

        // Short offline bounce (n = 6 < 12 and n = 12 == pdc)
        let short_in_l = [0.1_f32; 6];
        let short_in_r = [-0.1_f32; 6];
        let mut short_out_l = [0.0_f32; 6];
        let mut short_out_r = [0.0_f32; 6];
        host.render_offline_bounce(&short_in_l, &short_in_r, &mut short_out_l, &mut short_out_r);
        assert!(short_out_l.iter().all(|x| x.is_finite()));

        let pdc_in = [0.1_f32; 12];
        let mut pdc_out_l = [0.0_f32; 12];
        let mut pdc_out_r = [0.0_f32; 12];
        host.render_offline_bounce(&pdc_in, &pdc_in, &mut pdc_out_l, &mut pdc_out_r);
        assert!(pdc_out_l.iter().all(|x| x.is_finite()));

        // Offline bounce when prev_bounce is already true AND with a staged engine swap + bounce overrides queued
        host.set_offline_bounce(true);
        host.set_maximize_tier_on_bounce(true);
        host.flush_pending_controls();
        assert!(host.offline_bounce());

        {
            let _guard = host.audio.lock();
            host.reload_model_in_place_owned(&container, Tier::Pro, 48_000)
                .unwrap();
            host.set_maximize_tier_on_bounce(false);
            host.set_offline_bounce(false);
        }
        host.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|x| x.is_finite()));
        assert!(!host.offline_bounce());
        assert!(!host.maximize_tier_on_bounce());
        assert_eq!(host.tier(), Tier::Pro);

        // Also test when cur_bounce && cur_max remain true during bounce
        host.set_offline_bounce(true);
        host.set_maximize_tier_on_bounce(true);
        host.flush_pending_controls();
        host.render_offline_bounce_with_options(&in_l, &in_r, &mut out_l, &mut out_r, Some(true));
        assert!(host.offline_bounce());
        assert!(host.maximize_tier_on_bounce());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_host_process_all_topologies_finite_outputs(
            topology in 0u8..6,
            block_len in 1usize..96,
        ) {
            // verify[dsp.ctrl.op-m.spsc-ingress]
            // verify[arch.telemetry.seqlock-publication]
            let container = build_process_test_container();
            let host = DnamHostHandle::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
            let input_l = vec![0.05_f32; block_len];
            let input_r = vec![-0.05_f32; block_len];
            let mut out_l = vec![0.0_f32; block_len];
            let mut out_r = vec![0.0_f32; block_len];

            match topology {
                0 => host.process_block(&input_l, &input_r, &mut out_l, &mut out_r),
                1 => {
                    out_l.copy_from_slice(&input_l);
                    out_r.copy_from_slice(&input_r);
                    host.process_block_in_place(&mut out_l, &mut out_r);
                }
                2 => host.process_mono_to_stereo(&input_l, &mut out_l, &mut out_r),
                3 => host.process_mono(&input_l, &mut out_l),
                4 => {
                    out_l.copy_from_slice(&input_l);
                    host.process_mono_in_place(&mut out_l);
                }
                _ => host.render_offline_bounce(&input_l, &input_r, &mut out_l, &mut out_r),
            }
            prop_assert!(out_l.iter().all(|x| x.is_finite()));
            prop_assert!(out_r.iter().all(|x| x.is_finite()));
        }
    }
}
