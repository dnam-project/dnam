#![forbid(unsafe_code)]
//! `INPC`/`STMC` Capsule Sealing Helpers, Freeze-Transition Stem Synthesis, Control-Meta
//! Mapping, and `400 ms` Differential Mic-Pad Re-Capture (`stimulus::sealing`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.planner.freeze-transition]`, `ref[train.capture.calib.clean-pad]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlKind, DnamError};

use crate::CORESET_FP16_FRAMES_PER_STEM;
use crate::align::{
    DifferentialPadCalibration, compute_pcm_sha256, estimate_multiregime_alignment,
    measure_differential_pad_ratio, realign_fractional_kaiser64, rescale_padded_stem,
};
use crate::proj::capsules::build_proj_inpc_capsule;
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::fp16_bytes_to_f32;
use crate::proj::schema::{DnamProjManifest, StemProvenance};
use crate::proj::vault::{VaultCaptureSpec, replace_capture};
use crate::resampler::{
    resample_48k_to_arbitrary_rate, resample_arbitrary_rate_to_48k, resampled_frame_len,
};
use crate::targets::StemStimulus;

use super::laws::TopologyProbeMode;
use super::waveforms::{
    ActivePrimitiveKind, compute_stmc_psd_88_and_spectral_4096, encode_coreset_12k_fp16,
    synthesize_a5_musical_transient_coreset, synthesize_active_primitive_frame,
};
use super::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, PEDESTAL_BIN_MONO, build_disjoint_comb_schedule,
    synthesize_cyclic_comb_frame,
};

/// `3 * 4,608 = 13,824` samples at `48 kHz` (`12,000` `A5` coreset frames + `1,824` settle zeros).
pub(crate) const CORESET_WITH_SETTLE_48K: usize = 3 * ADAPTIVE_FRAME_SAMPLES;
const SETTLE_PAD_SAMPLES_48K: usize = CORESET_WITH_SETTLE_48K - CORESET_FP16_FRAMES_PER_STEM;

pub(crate) fn control_declared_bounds(ctrl: &dnam_engine::schema::ControlSchemaDef) -> (f32, f32) {
    let [lo, hi] = ctrl.declared_range.unwrap_or([0.0, 1.0]);
    (lo, hi)
}

pub(crate) fn control_default_f32(ctrl: &dnam_engine::schema::ControlSchemaDef) -> f32 {
    let (lo, hi) = control_declared_bounds(ctrl);
    ctrl.default
        .as_f64()
        .map_or(0.5 * (lo + hi), |v| v as f32)
        .clamp(lo, hi)
}

pub(crate) fn build_controls_meta_map(
    manifest: &DnamProjManifest,
    values: &[f32],
    switch_overrides: &BTreeMap<usize, String>,
) -> BTreeMap<String, serde_json::Value> {
    let mut map = BTreeMap::new();
    for (idx, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        match ctrl.kind {
            ControlKind::Range => {
                let v = f64::from(
                    values
                        .get(idx)
                        .copied()
                        .unwrap_or_else(|| control_default_f32(ctrl)),
                );
                if let Some(num) = serde_json::Number::from_f64(v) {
                    map.insert(ctrl.id.clone(), serde_json::Value::Number(num));
                }
            }
            ControlKind::Switch => {
                let opt = switch_overrides.get(&idx).cloned().unwrap_or_else(|| {
                    ctrl.default
                        .as_str()
                        .map(str::to_string)
                        .or_else(|| {
                            ctrl.declared_options
                                .as_ref()
                                .and_then(|o| o.first().cloned())
                        })
                        .unwrap_or_else(|| "default".to_string())
                });
                map.insert(ctrl.id.clone(), serde_json::Value::String(opt));
            }
        }
    }
    map
}

pub(crate) fn default_control_vector(manifest: &DnamProjManifest) -> Vec<f32> {
    manifest
        .base
        .controls_schema
        .iter()
        .map(|c| match c.kind {
            ControlKind::Range => control_default_f32(c),
            ControlKind::Switch => 0.0,
        })
        .collect()
}

pub(crate) fn resample_48k_channels_to_sr(
    sr: u32,
    l_48k: &[f32],
    r_48k: Option<&[f32]>,
    aux_48k: Option<&[f32]>,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    if sr == 48_000 {
        return (
            l_48k.to_vec(),
            r_48k.map(<[f32]>::to_vec),
            aux_48k.map(<[f32]>::to_vec),
        );
    }
    let dst_len = resampled_frame_len(l_48k.len(), 48_000, sr);
    let l = resample_48k_to_arbitrary_rate(l_48k, sr, Some(dst_len));
    let r = r_48k.map(|ch| resample_48k_to_arbitrary_rate(ch, sr, Some(dst_len)));
    let aux = aux_48k.map(|ch| resample_48k_to_arbitrary_rate(ch, sr, Some(dst_len)));
    (l, r, aux)
}

pub(crate) fn normalize_stem_channel_to_48k(samples: &[f32], sr: u32) -> Vec<f32> {
    let clamped_sr = sr.clamp(8_000, 384_000);
    if clamped_sr == 48_000 {
        samples.to_vec()
    } else {
        let dst_len = resampled_frame_len(samples.len(), clamped_sr, 48_000);
        resample_arbitrary_rate_to_48k(samples, clamped_sr, Some(dst_len))
    }
}

/// Synthesizes a timed `freeze_latch` sustain-entry -> live play-over -> release [`StemStimulus`]
/// (`r[train.capture.planner.freeze-transition]`).
pub(crate) fn build_freeze_transition_stem_stimulus(
    manifest: &DnamProjManifest,
    freeze_ctrl_idx: usize,
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    seed_256: &[u8; 256],
) -> StemStimulus {
    let empty_mask = BTreeSet::new();
    let sched_l =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let bin_amps_l: Vec<(usize, f64)> = sched_l.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let comb_l = synthesize_cyclic_comb_frame(
        &bin_amps_l,
        Some(PEDESTAL_BIN_MONO),
        0.05,
        0.18,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; 64],
    );
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let (mut coreset_l, mut coreset_r) = synthesize_a5_musical_transient_coreset(
        seed_256,
        CORESET_FP16_FRAMES_PER_STEM,
        if need_stereo { 2 } else { 1 },
        48_000,
    );
    coreset_l.resize(CORESET_WITH_SETTLE_48K, 0.0);
    if let Some(ref mut cr) = coreset_r {
        cr.resize(CORESET_WITH_SETTLE_48K, 0.0);
    }

    // [0..9,216): unfrozen tank charge; [9,216..18,432): frozen play-over; [18,432..23,040): release;
    // [23,040..36,864): A5 transient coreset + settle.
    let mut wave_l_48k = Vec::with_capacity(8 * ADAPTIVE_FRAME_SAMPLES);
    wave_l_48k.extend_from_slice(&comb_l.samples);
    wave_l_48k.extend_from_slice(&synthesize_active_primitive_frame(
        ActivePrimitiveKind::P4TankDuckingBurst,
        0,
        false,
        ADC_PEAK_GOVERNOR_FS,
    ));
    let n_earliest_48k = wave_l_48k.len();
    wave_l_48k.extend_from_slice(&synthesize_active_primitive_frame(
        ActivePrimitiveKind::P1BranchCutChirp,
        0,
        false,
        ADC_PEAK_GOVERNOR_FS,
    ));
    wave_l_48k.extend_from_slice(&comb_l.samples);
    let n_latest_48k = wave_l_48k.len();
    wave_l_48k.resize(n_latest_48k + ADAPTIVE_FRAME_SAMPLES, 0.0);
    wave_l_48k.extend_from_slice(&coreset_l);

    let wave_r_48k = coreset_r.map(|cr| {
        let prefix_len = wave_l_48k.len() - CORESET_WITH_SETTLE_48K;
        let mut wr = wave_l_48k[..prefix_len].to_vec();
        wr.extend_from_slice(&cr);
        wr
    });
    let aux_48k = need_aux.then(|| wave_l_48k.clone());

    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let (in_1, in_2, in_3) =
        resample_48k_channels_to_sr(sr, &wave_l_48k, wave_r_48k.as_deref(), aux_48k.as_deref());
    let n_earliest = resampled_frame_len(n_earliest_48k, 48_000, sr).min(in_1.len());
    let n_latest = resampled_frame_len(n_latest_48k, 48_000, sr)
        .max(n_earliest + 1)
        .min(in_1.len());

    let base_ctrls = default_control_vector(manifest);
    let ctrl_def = &manifest.base.controls_schema[freeze_ctrl_idx];
    let (unfrozen_val, frozen_val) = match ctrl_def.kind {
        ControlKind::Switch => (0.0_f32, 1.0_f32),
        ControlKind::Range => control_declared_bounds(ctrl_def),
    };
    let mut trajs = Vec::with_capacity(base_ctrls.len());
    for (q, &def_v) in base_ctrls.iter().enumerate() {
        if q == freeze_ctrl_idx {
            let tr: Vec<f32> = (0..in_1.len())
                .map(|idx| {
                    if (n_earliest..n_latest).contains(&idx) {
                        frozen_val
                    } else {
                        unfrozen_val
                    }
                })
                .collect();
            trajs.push(tr);
        } else {
            trajs.push(vec![def_v; in_1.len()]);
        }
    }
    let mut active_ctrls = base_ctrls;
    if let Some(slot) = active_ctrls.get_mut(freeze_ctrl_idx) {
        *slot = frozen_val;
    }

    StemStimulus {
        stem_id: format!("freeze_transition_{freeze_ctrl_idx}"),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: active_ctrls,
        control_trajectories: Some(trajs),
        freeze_transition_window: Some([n_earliest, n_latest]),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn seal_coresets_to_inpc_and_vault(
    stem_id: &str,
    is_star_center: bool,
    seed_256: &[u8; 256],
    carrier_bins: &[usize],
    valid_frames: usize,
    coreset_dry_l: &[f32],
    coreset_dry_r: Option<&[f32]>,
    raw_wet_l: &[f32],
    raw_wet_r: Option<&[f32]>,
    windows: Vec<StmcTrajectoryWindow>,
    controls_meta: BTreeMap<String, serde_json::Value>,
) -> Result<(Vec<u8>, VaultCaptureSpec), DnamError> {
    let dry_12k_l = &coreset_dry_l[..CORESET_FP16_FRAMES_PER_STEM];
    let dry_12k_r = coreset_dry_r.map(|r| &r[..CORESET_FP16_FRAMES_PER_STEM]);
    let dry_fp16 = encode_coreset_12k_fp16(dry_12k_l, dry_12k_r);
    let inpc_channels = if dry_12k_r.is_some() { 2_u16 } else { 1_u16 };

    let mut schedule_320 = [0u8; 320];
    schedule_320[0..4].copy_from_slice(&(carrier_bins.len() as u32).to_le_bytes());
    schedule_320[4..8].copy_from_slice(&(valid_frames as u32).to_le_bytes());
    for (idx, &b) in carrier_bins.iter().take(76).enumerate() {
        let off = 16 + idx * 4;
        schedule_320[off..off + 4].copy_from_slice(&(b as u32).to_le_bytes());
    }
    let inpc_bytes = build_proj_inpc_capsule(inpc_channels, seed_256, &schedule_320, &dry_fp16)?;

    let dry_ref = &coreset_dry_l[..raw_wet_l.len().min(coreset_dry_l.len())];
    let wet_ref = &raw_wet_l[..dry_ref.len()];
    let (lag_samples, inv_pol) = if dry_ref.len() >= 64 {
        match estimate_multiregime_alignment(dry_ref, wet_ref, 48_000, 512) {
            Ok(al) => (al.total_lag_samples, al.polarity_inverted),
            Err(_) => (0.0, false),
        }
    } else {
        (0.0, false)
    };

    let aligned_wet_l = realign_fractional_kaiser64(raw_wet_l, lag_samples, 0.0, inv_pol);
    let aligned_wet_r = raw_wet_r
        .filter(|r| !r.is_empty())
        .map(|r| realign_fractional_kaiser64(r, lag_samples, 0.0, inv_pol));

    let wet_fp16 = encode_coreset_12k_fp16(&aligned_wet_l, aligned_wet_r.as_deref());
    let (psd_88, spec_4096) =
        compute_stmc_psd_88_and_spectral_4096(&aligned_wet_l, aligned_wet_r.as_deref());

    let provenance = StemProvenance {
        input_pcm_sha256: compute_pcm_sha256(dry_12k_l),
        target_pcm_sha256: compute_pcm_sha256(
            &aligned_wet_l[..aligned_wet_l.len().min(CORESET_FP16_FRAMES_PER_STEM)],
        ),
        num_frames: CORESET_FP16_FRAMES_PER_STEM as u64,
    };

    let spec = VaultCaptureSpec {
        stem_id: stem_id.to_string(),
        is_star_center,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: lag_samples as f32,
        controls_meta,
        provenance,
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows,
        band_psd_summary_88: psd_88,
        n_comb_bins: carrier_bins.len() as u32,
        spectral_summary_4096: spec_4096,
        wet_coreset_fp16: wet_fp16,
        optional_spcm: None,
    };
    Ok((inpc_bytes, spec))
}

/// Aligns a full `48 kHz`-normalized stem on its aperiodic `A5` coreset+settle window FIRST
/// (handling bulk converter/DSP latency without periodic comb aliases) before slicing `12,000` frames.
#[allow(clippy::too_many_arguments)]
pub(crate) fn align_full_stem_and_seal_vault(
    stem_id: &str,
    is_star_center: bool,
    seed_256: &[u8; 256],
    carrier_bins: &[usize],
    valid_frames: usize,
    dry_48k_l: &[f32],
    dry_48k_r: Option<&[f32]>,
    wet_48k_l: &[f32],
    wet_48k_r: Option<&[f32]>,
    windows: Vec<StmcTrajectoryWindow>,
    controls_meta: BTreeMap<String, serde_json::Value>,
    user_metadata: BTreeMap<String, serde_json::Value>,
) -> Result<(Vec<u8>, VaultCaptureSpec), DnamError> {
    let has_settle_pad = dry_48k_l.len() >= CORESET_WITH_SETTLE_48K
        && dry_48k_l[dry_48k_l.len() - SETTLE_PAD_SAMPLES_48K..]
            .iter()
            .all(|&v| v.abs() < 1.0e-3);
    let tail_span = if has_settle_pad {
        CORESET_WITH_SETTLE_48K
    } else {
        CORESET_FP16_FRAMES_PER_STEM
    };
    let core_start = dry_48k_l.len().saturating_sub(tail_span);

    let dry_align_slice = &dry_48k_l[core_start..];
    let wet_align_end = wet_48k_l.len().min(core_start + dry_align_slice.len());
    let wet_align_slice = if core_start < wet_align_end {
        &wet_48k_l[core_start..wet_align_end]
    } else {
        &[]
    };
    let common_len = dry_align_slice.len().min(wet_align_slice.len());
    let (lag_samples, inv_pol) = if common_len >= 64 {
        match estimate_multiregime_alignment(
            &dry_align_slice[..common_len],
            &wet_align_slice[..common_len],
            48_000,
            1_024,
        ) {
            Ok(al) => (al.total_lag_samples, al.polarity_inverted),
            Err(_) => (0.0, false),
        }
    } else {
        (0.0, false)
    };

    let full_aligned_l = realign_fractional_kaiser64(wet_48k_l, lag_samples, 0.0, inv_pol);
    let full_aligned_r = wet_48k_r
        .filter(|r| !r.is_empty())
        .map(|r| realign_fractional_kaiser64(r, lag_samples, 0.0, inv_pol));

    let dry_12k_l = &dry_48k_l[core_start..core_start + CORESET_FP16_FRAMES_PER_STEM];
    let dry_12k_r = dry_48k_r
        .filter(|r| r.len() >= core_start + CORESET_FP16_FRAMES_PER_STEM)
        .map(|r| &r[core_start..core_start + CORESET_FP16_FRAMES_PER_STEM]);
    let dry_fp16 = encode_coreset_12k_fp16(dry_12k_l, dry_12k_r);
    let inpc_channels = if dry_12k_r.is_some() { 2_u16 } else { 1_u16 };

    let mut schedule_320 = [0u8; 320];
    schedule_320[0..4].copy_from_slice(&(carrier_bins.len() as u32).to_le_bytes());
    schedule_320[4..8].copy_from_slice(&(valid_frames as u32).to_le_bytes());
    for (idx, &b) in carrier_bins.iter().take(76).enumerate() {
        let off = 16 + idx * 4;
        schedule_320[off..off + 4].copy_from_slice(&(b as u32).to_le_bytes());
    }
    let inpc_bytes = build_proj_inpc_capsule(inpc_channels, seed_256, &schedule_320, &dry_fp16)?;

    let wet_start = core_start.min(
        full_aligned_l
            .len()
            .saturating_sub(CORESET_FP16_FRAMES_PER_STEM),
    );
    let wet_end = (wet_start + CORESET_FP16_FRAMES_PER_STEM).min(full_aligned_l.len());
    let wet_12k_l = &full_aligned_l[wet_start..wet_end];
    let wet_12k_r = full_aligned_r.as_deref().map(|r| {
        let ws = core_start.min(r.len().saturating_sub(CORESET_FP16_FRAMES_PER_STEM));
        let we = (ws + CORESET_FP16_FRAMES_PER_STEM).min(r.len());
        &r[ws..we]
    });

    let wet_fp16 = encode_coreset_12k_fp16(wet_12k_l, wet_12k_r);
    let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(wet_12k_l, wet_12k_r);

    let provenance = StemProvenance {
        input_pcm_sha256: compute_pcm_sha256(dry_12k_l),
        target_pcm_sha256: compute_pcm_sha256(wet_12k_l),
        num_frames: CORESET_FP16_FRAMES_PER_STEM as u64,
    };

    let spec = VaultCaptureSpec {
        stem_id: stem_id.to_string(),
        is_star_center,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: lag_samples as f32,
        controls_meta,
        provenance,
        user_metadata,
        extracted_physics: None,
        windows,
        band_psd_summary_88: psd_88,
        n_comb_bins: carrier_bins.len() as u32,
        spectral_summary_4096: spec_4096,
        wet_coreset_fp16: wet_fp16,
        optional_spcm: None,
    };
    Ok((inpc_bytes, spec))
}

/// Re-normalizes and replaces a clipped stem after the user engages a physical `-10 dB` / `-20 dB`
/// mic/preamp pad (`Law 0.F`), using the `400 ms` `-18 dB` differential reference probe to recover
/// $\hat{G}_{\text{pad}}$ and rescale the re-captured stem by $\hat{G}_{\text{pad}}^{-1}$
/// (`r[train.capture.safeguards.mic-capsule-clip]`, `r[train.cache.proj-vault]`).
pub fn recapture_clipped_stem(
    proj_bytes: &[u8],
    stem_id: &str,
    prev_unpadded_probe_400ms: &[f32],
    new_padded_probe_400ms: &[f32],
    mut padded_capture: VaultCaptureSpec,
) -> Result<(Vec<u8>, DifferentialPadCalibration), DnamError> {
    let pad_cal =
        measure_differential_pad_ratio(prev_unpadded_probe_400ms, new_padded_probe_400ms, 128)
            .map_err(|e| {
                DnamError::SchemaValidationError(format!("mic-pad probe calibration failed: {e}"))
            })?;

    let is_stereo = padded_capture.wet_coreset_fp16.len() == 48_000;
    let decoded = fp16_bytes_to_f32(&padded_capture.wet_coreset_fp16);
    let rescaled = rescale_padded_stem(&decoded, pad_cal.pad_gain_linear).map_err(|e| {
        DnamError::SchemaValidationError(format!("mic-pad stem rescale failed: {e}"))
    })?;

    if is_stereo {
        let mut l = Vec::with_capacity(CORESET_FP16_FRAMES_PER_STEM);
        let mut r = Vec::with_capacity(CORESET_FP16_FRAMES_PER_STEM);
        for pair in rescaled.as_chunks::<2>().0 {
            l.push(pair[0]);
            r.push(pair[1]);
        }
        padded_capture.wet_coreset_fp16 = encode_coreset_12k_fp16(&l, Some(&r));
        let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(&l, Some(&r));
        padded_capture.band_psd_summary_88 = psd_88;
        padded_capture.spectral_summary_4096 = spec_4096;
        padded_capture.provenance.target_pcm_sha256 = compute_pcm_sha256(&l);
    } else {
        padded_capture.wet_coreset_fp16 = encode_coreset_12k_fp16(&rescaled, None);
        let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(&rescaled, None);
        padded_capture.band_psd_summary_88 = psd_88;
        padded_capture.spectral_summary_4096 = spec_4096;
        padded_capture.provenance.target_pcm_sha256 = compute_pcm_sha256(&rescaled);
    }
    padded_capture.pad_gain_f32 = pad_cal.pad_gain_linear as f32;

    let updated_proj = replace_capture(proj_bytes, stem_id, padded_capture)?;
    Ok((updated_proj, pad_cal))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn interpolate_frame_trajectories(
    start_vals: &[f32],
    end_vals_opt: Option<&[f32]>,
    active_control_id: u16,
    window_weight: f32,
    t0: f32,
    t1: f32,
    reset_coreset_knots: bool,
    is_coreset: bool,
    frame_len: usize,
    trajectory_windows: &mut Vec<StmcTrajectoryWindow>,
) -> (Vec<f32>, Option<Vec<Vec<f32>>>) {
    let mut cur_ctrls = start_vals.to_vec();
    let Some(end_vals) = end_vals_opt.filter(|e| e.len() == start_vals.len() && !e.is_empty())
    else {
        return (cur_ctrls, None);
    };

    if reset_coreset_knots {
        trajectory_windows.clear();
        let swept_indices: Vec<usize> = start_vals
            .iter()
            .zip(end_vals)
            .enumerate()
            .filter_map(|(p, (&v0, &v1))| ((v1 - v0).abs() > 1.0e-5).then_some(p))
            .collect();
        if !swept_indices.is_empty() {
            let knots = (36 / swept_indices.len()).clamp(2, 6);
            for k in 0..knots {
                let frac = (k as f32) / ((knots - 1) as f32);
                for &p in &swept_indices {
                    let v0 = start_vals[p];
                    let v1 = end_vals[p];
                    trajectory_windows.push(StmcTrajectoryWindow {
                        control_id: p as u16,
                        ctrl_kind: 0,
                        active_opt: 0,
                        u_phys: v0 + (v1 - v0) * frac,
                        window_weight,
                        emr_velocity: 1.0 / (knots as f32),
                    });
                }
            }
        }
    }

    let mut trajs = Vec::with_capacity(end_vals.len());
    let mut moved_any = false;
    let n_len = frame_len.max(1);
    for (p, (&v0, &v1)) in start_vals.iter().zip(end_vals).enumerate() {
        let s0 = v0 + (v1 - v0) * t0;
        let s1 = v0 + (v1 - v0) * t1;
        cur_ctrls[p] = s0;
        trajs.push(
            (0..frame_len)
                .map(|i| s0 + (s1 - s0) * (i as f32 / n_len as f32))
                .collect(),
        );
        if !is_coreset && (v1 - v0).abs() > 1.0e-5 && trajectory_windows.len() < 36 {
            moved_any = true;
            trajectory_windows.push(StmcTrajectoryWindow {
                control_id: p as u16,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: s0,
                window_weight,
                emr_velocity: (t1 - t0).abs(),
            });
        }
    }
    if !is_coreset
        && !moved_any
        && trajectory_windows.len() < 36
        && let Some(&u_val) = cur_ctrls.get(usize::from(active_control_id))
    {
        trajectory_windows.push(StmcTrajectoryWindow {
            control_id: active_control_id,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: u_val,
            window_weight,
            emr_velocity: (t1 - t0).abs(),
        });
    }
    (cur_ctrls, Some(trajs))
}
