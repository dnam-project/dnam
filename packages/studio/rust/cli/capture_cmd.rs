//! Capture subcommand handlers (`backends`, `devices`, `generate`, `calibrate`, `plan`, `record`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Subcommand;

use crate::capture::{
    CalibrationResult, Step1CalibrationMetrics, align_hardware_latency,
    analyze_hot_corner_nonlinearity, analyze_probe_for_excitation, apply_subsample_shift_fft,
    compute_pcm_sha256, create_capture_backend_with_gain, estimate_subsample_lag_and_snr,
    generate_calibration_probe, generate_excitation_signal,
    generate_excitation_signal_with_profile, list_backend_devices, list_capture_backends,
    measure_silent_window_noise_floor, query_cpal_devices, write_wav_24bit, write_wav_32bit_float,
};
use crate::cli::format::{self, StemPromptAction};
use crate::error::{StudioError, StudioResult};
use crate::project::{
    DnamProject, StemRecord, build_capture_plan_with_prescriptions,
    extract_checkpoint_prescriptions,
};

/// Subcommands under `dnam capture`.
#[derive(Debug, Subcommand)]
pub enum CaptureAction {
    /// List available capture backends (`software`, `usb`, `anagram`).
    Backends,
    /// List audio interfaces (lists all discovered `cpal` hardware devices when `--backend` is omitted).
    Devices {
        /// Optional backend filter (`anagram`, `usb`, `software`).
        #[arg(long)]
        backend: Option<String>,
    },
    /// Generate a 24-bit dry excitation WAV file.
    Generate {
        /// Output WAV file path.
        #[arg(long, default_value = "input.wav")]
        output: PathBuf,
        /// Number of samples to synthesize (overridden by `--duration-sec` if provided).
        #[arg(long, default_value_t = 4096)]
        samples: usize,
        /// Optional duration in seconds (overrides `--samples`).
        #[arg(long)]
        duration_sec: Option<f32>,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 48_000)]
        sample_rate_hz: u32,
    },
    /// Run 2-step Reference Unity & Engaged Hot-Corner calibration and synthesize probe-tailored `models/<model>/stems/input.wav`.
    Calibrate {
        /// Optional target model ID (`models/<model>/stems/input.wav`).
        #[arg(long)]
        model: Option<String>,
        /// Backend name (`software`, `usb`, `anagram`).
        #[arg(long, default_value = "software")]
        backend: String,
        /// Device identifier.
        #[arg(long, default_value = "loopback_default")]
        device: String,
        /// Optional initial output send gain in dB (Step 1 auto-pads if bypass clips).
        #[arg(long)]
        send_gain_db: Option<f32>,
        /// Optional Clean Level pad in dB applied on the device for Step-2 Hot Corner (e.g. `-40.0`).
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        clean_pad_db: f32,
        /// Optional individual calibration step (`1` = Bypass Unity, `2` = Engaged Hot-Corner).
        #[arg(long)]
        step: Option<u8>,
        /// Automatically confirm interactive calibration prompts.
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Display the Star + Switch + Corner + `.dnam-ckpt` Prescription capture schedule for a model.
    Plan {
        /// Target model ID (defaults to the sole model in the workspace if omitted).
        #[arg(long)]
        model: Option<String>,
        /// Disable multi-knob high-corner anchor stems.
        #[arg(long)]
        no_corners: bool,
    },
    /// Execute the planned capture stems for a model using the active backend.
    Record {
        /// Target model ID (defaults to the sole model in the workspace if omitted).
        #[arg(long)]
        model: Option<String>,
        /// Optional backend override (`software`, `usb`, `anagram`).
        #[arg(long)]
        backend: Option<String>,
        /// Number of samples per stem.
        #[arg(long, default_value_t = 2048)]
        samples: usize,
        /// Optional specific stem ID to record (`stem_00_star_center`, etc.).
        #[arg(long)]
        stem: Option<String>,
        /// Optional maximum number of pending stems to record in this invocation.
        #[arg(long)]
        max_stems: Option<usize>,
        /// Optional Clean Level pad in dB applied on the device (`-40.0` restores `+40.0 dB` in 32-bit float).
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        clean_pad_db: f32,
        /// Automatically confirm interactive stem prompts.
        #[arg(long, short = 'y')]
        yes: bool,
        /// Optional hands-free countdown in seconds before recording each stem (`0.0` = interactive prompt).
        #[arg(long, default_value_t = 0.0)]
        countdown_sec: f32,
        /// Re-record all planned stems even if already captured.
        #[arg(long)]
        all_stems: bool,
    },
}

fn is_real_hw_device(backend: &str, device_id: &str) -> bool {
    let trimmed = device_id.trim();
    backend != "software"
        && !trimmed.is_empty()
        && trimmed != "loopback_default"
        && trimmed != "usb_class_compliant_0"
        && trimmed != "anagram_rk3582_usb"
}

fn resolve_effective_device(backend: &str, device: &str) -> String {
    if backend != "software"
        && (device.trim().is_empty() || device.trim() == "loopback_default")
        && let Some(first) = query_cpal_devices(backend).into_iter().next()
    {
        return first.device_id;
    }
    device.to_string()
}

fn resolve_target_model_id(proj: &DnamProject, explicit: Option<&str>) -> StudioResult<String> {
    if let Some(m) = explicit {
        return Ok(m.to_string());
    }
    if proj.manifest().models.len() == 1 {
        return Ok(proj.manifest().models[0].clone());
    }
    Err(StudioError::Validation(
        "Specify `--model <id>` (workspace contains multiple or zero models)".to_string(),
    ))
}

fn is_step2_return_clipped(peak_dbfs: f32, clipped_fraction: f32) -> bool {
    clipped_fraction > 0.0 || peak_dbfs >= -0.05
}

fn calibrate_two_step_with_clipping_prompt(
    cap_backend: &dyn crate::capture::CaptureBackend,
    device_id: &str,
    sample_rate_hz: u32,
    auto_yes: bool,
) -> StudioResult<CalibrationResult> {
    loop {
        let cal = cap_backend.calibrate_with_prompts(
            device_id,
            sample_rate_hz,
            &mut |setup| format::prompt_calibration_step1(setup, auto_yes),
            &mut |step1| format::prompt_calibration_step2(step1, auto_yes),
        )?;
        if is_step2_return_clipped(cal.hot_corner_peak_dbfs, cal.hot_corner_clipped_fraction)
            && format::prompt_repeat_calibration_on_clipping(
                cal.hot_corner_peak_dbfs,
                cal.hot_corner_clipped_fraction,
                auto_yes,
            )?
        {
            continue;
        }
        return Ok(cal);
    }
}

/// Executes `dnam capture`.
pub fn run_capture(workspace: &Path, action: &CaptureAction) -> StudioResult<String> {
    match action {
        CaptureAction::Backends => {
            let list = list_capture_backends();
            Ok(serde_json::to_string_pretty(&list)?)
        }
        CaptureAction::Devices { backend } => {
            let devices = list_backend_devices(backend.as_deref().unwrap_or("all"));
            Ok(serde_json::to_string_pretty(&devices)?)
        }
        CaptureAction::Generate {
            output,
            samples,
            duration_sec,
            sample_rate_hz,
        } => {
            let sr = (*sample_rate_hz).max(8_000);
            let total_samples = duration_sec
                .map(|d| ((d.max(0.1) * (sr as f32)).round() as usize).max(64))
                .unwrap_or(*samples);
            let sig = generate_excitation_signal(sr, total_samples);
            write_wav_24bit(output, &sig, sr, 1)?;
            Ok(format!(
                "Generated 24-bit dry excitation WAV ({} samples @ {} Hz) at {}",
                total_samples,
                sr,
                output.display()
            ))
        }
        CaptureAction::Calibrate {
            model,
            backend,
            device,
            send_gain_db,
            clean_pad_db,
            step,
            yes,
        } => {
            let proj = DnamProject::open(workspace)?;
            let sr = proj.manifest().project.sample_rate_hz;
            let model_id = resolve_target_model_id(&proj, model.as_deref())?;
            let model_manifest = proj.load_model(&model_id)?;
            let resolved_dev = resolve_effective_device(backend, device);
            let initial_gain_db = send_gain_db.unwrap_or(model_manifest.capture.send_gain_db);
            let cap_backend =
                create_capture_backend_with_gain(backend, Some(&resolved_dev), sr, initial_gain_db);
            let auto_yes = *yes || backend == "software";
            let is_hw = is_real_hw_device(backend, &resolved_dev);
            let probe_len = if is_hw {
                ((sr as usize) * 4).max(2_048)
            } else {
                2_048
            };
            let dry_len = if is_hw {
                ((sr as usize) * 12).max(4_096)
            } else {
                4_096
            };

            if *step == Some(1) {
                format::prompt_calibration_step1(&cap_backend.setup_instructions(sr), auto_yes)?;
                let base_dry = generate_calibration_probe(sr, probe_len);
                let mut unity_ranges = BTreeMap::new();
                unity_ranges.insert("drive".to_string(), 0.0_f32);
                let mut locked_gain_db = initial_gain_db;
                let mut scaled_dry = base_dry.clone();
                let mut wet_step1 = Vec::new();
                for attempt in 0..5 {
                    let pad_lin = 10.0_f32.powf((locked_gain_db - initial_gain_db) / 20.0);
                    scaled_dry = base_dry.iter().map(|&s| s * pad_lin).collect();
                    wet_step1 =
                        cap_backend.capture_stem(&scaled_dry, &unity_ranges, &BTreeMap::new())?;
                    if !wet_step1.iter().any(|&s| s.abs() >= 0.994) || attempt == 4 {
                        break;
                    }
                    locked_gain_db -= 6.0;
                }
                let report = estimate_subsample_lag_and_snr(
                    &scaled_dry,
                    &wet_step1,
                    (scaled_dry.len() / 4).max(32),
                )?;
                let noise_1 = measure_silent_window_noise_floor(&wet_step1, sr, None);
                let dry_rms = (scaled_dry.iter().map(|&x| x * x).sum::<f32>()
                    / (scaled_dry.len().max(1) as f32))
                    .sqrt()
                    .max(1e-9);
                let wet1_rms = (wet_step1.iter().map(|&x| x * x).sum::<f32>()
                    / (wet_step1.len().max(1) as f32))
                    .sqrt()
                    .max(1e-9);
                let unity_ref_dbfs = 20.0 * (wet1_rms / dry_rms).log10();
                let s1 = Step1CalibrationMetrics {
                    backend: backend.clone(),
                    device_id: resolved_dev.clone(),
                    send_gain_db: locked_gain_db,
                    latency_samples: report.coarse_lag_samples,
                    unity_ref_dbfs,
                    noise_floor_dbfs: noise_1,
                    snr_db: report.snr_db,
                };
                proj.update_calibration_with_profile(
                    &model_id,
                    backend,
                    &resolved_dev,
                    s1.send_gain_db,
                    s1.latency_samples,
                    s1.noise_floor_dbfs,
                    s1.unity_ref_dbfs,
                    model_manifest
                        .capture
                        .excitation_profile
                        .unwrap_or_default(),
                )?;
                return Ok(serde_json::to_string_pretty(&s1)?);
            }

            if *step == Some(2) {
                let s1 = Step1CalibrationMetrics {
                    backend: backend.clone(),
                    device_id: resolved_dev.clone(),
                    send_gain_db: initial_gain_db,
                    latency_samples: model_manifest.capture.calibrated_latency_samples,
                    unity_ref_dbfs: model_manifest.capture.unity_ref_dbfs,
                    noise_floor_dbfs: model_manifest.capture.noise_floor_dbfs,
                    snr_db: 60.0,
                };
                let makeup_db = clean_pad_db.abs();
                let cal = loop {
                    format::prompt_calibration_step2(&s1, auto_yes)?;
                    let scaled_dry = generate_calibration_probe(sr, probe_len);
                    let mut hot_ranges = BTreeMap::new();
                    hot_ranges.insert("drive".to_string(), 1.0_f32);
                    let wet_step2 =
                        cap_backend.capture_stem(&scaled_dry, &hot_ranges, &BTreeMap::new())?;
                    let report = align_hardware_latency(
                        &scaled_dry,
                        &wet_step2,
                        (scaled_dry.len() / 4).max(32),
                        if s1.latency_samples != 0 {
                            Some(s1.latency_samples)
                        } else {
                            None
                        },
                        sr,
                        Some(s1.noise_floor_dbfs),
                    )?;
                    let aligned_wet2 =
                        apply_subsample_shift_fft(&wet_step2, report.subsample_lag_samples);
                    let hot_analysis = analyze_hot_corner_nonlinearity(
                        &scaled_dry,
                        &aligned_wet2,
                        s1.unity_ref_dbfs,
                    );
                    let profile = analyze_probe_for_excitation(&scaled_dry, &aligned_wet2, sr);
                    let raw_peak_dbfs = hot_analysis.measured_return_peak_dbfs;
                    let candidate = CalibrationResult {
                        backend: backend.clone(),
                        device_id: resolved_dev.clone(),
                        send_gain_db: initial_gain_db,
                        latency_samples: report.coarse_lag_samples,
                        noise_floor_dbfs: s1.noise_floor_dbfs,
                        snr_db: report.snr_db,
                        unity_ref_dbfs: s1.unity_ref_dbfs,
                        hot_corner_peak_dbfs: raw_peak_dbfs + makeup_db,
                        hot_corner_clipped_fraction: hot_analysis.clipped_sample_fraction,
                        excitation_profile: profile,
                    };
                    if is_step2_return_clipped(raw_peak_dbfs, candidate.hot_corner_clipped_fraction)
                        && format::prompt_repeat_calibration_on_clipping(
                            raw_peak_dbfs,
                            candidate.hot_corner_clipped_fraction,
                            auto_yes,
                        )?
                    {
                        continue;
                    }
                    break candidate;
                };
                let tailored = generate_excitation_signal_with_profile(
                    sr,
                    dry_len,
                    Some(&cal.excitation_profile),
                );
                let dry_path = proj.model_dry_input_wav_path(&model_id);
                write_wav_24bit(&dry_path, &tailored, sr, 1)?;
                proj.update_calibration_with_profile(
                    &model_id,
                    backend,
                    &resolved_dev,
                    cal.send_gain_db,
                    cal.latency_samples,
                    cal.noise_floor_dbfs,
                    cal.unity_ref_dbfs,
                    cal.excitation_profile.clone(),
                )?;
                return Ok(serde_json::to_string_pretty(&cal)?);
            }

            let mut cal = calibrate_two_step_with_clipping_prompt(
                cap_backend.as_ref(),
                &resolved_dev,
                sr,
                auto_yes,
            )?;
            cal.hot_corner_peak_dbfs += clean_pad_db.abs();
            let tailored =
                generate_excitation_signal_with_profile(sr, dry_len, Some(&cal.excitation_profile));
            let dry_path = proj.model_dry_input_wav_path(&model_id);
            write_wav_24bit(&dry_path, &tailored, sr, 1)?;
            proj.update_calibration_with_profile(
                &model_id,
                backend,
                &resolved_dev,
                cal.send_gain_db,
                cal.latency_samples,
                cal.noise_floor_dbfs,
                cal.unity_ref_dbfs,
                cal.excitation_profile.clone(),
            )?;
            Ok(serde_json::to_string_pretty(&cal)?)
        }
        CaptureAction::Plan { model, no_corners } => {
            let proj = DnamProject::open(workspace)?;
            let model_id = resolve_target_model_id(&proj, model.as_deref())?;
            let manifest = proj.load_model(&model_id)?;
            let ckpt_path = workspace
                .join("models")
                .join(&model_id)
                .join("checkpoint.dnam-ckpt");
            let prescriptions = extract_checkpoint_prescriptions(&ckpt_path, &manifest);
            let plan =
                build_capture_plan_with_prescriptions(&manifest, !*no_corners, &prescriptions);
            Ok(format::format_capture_plan(&plan))
        }
        CaptureAction::Record {
            model,
            backend,
            samples,
            stem,
            max_stems,
            clean_pad_db,
            yes,
            countdown_sec,
            all_stems,
        } => {
            let proj = DnamProject::open(workspace)?;
            let model_id = resolve_target_model_id(&proj, model.as_deref())?;
            let mut manifest = proj.load_model(&model_id)?;
            let ckpt_path = workspace
                .join("models")
                .join(&model_id)
                .join("checkpoint.dnam-ckpt");
            let prescriptions = extract_checkpoint_prescriptions(&ckpt_path, &manifest);
            let plan = build_capture_plan_with_prescriptions(&manifest, true, &prescriptions);
            let sr = proj.manifest().project.sample_rate_hz;
            let backend_name = backend
                .as_deref()
                .unwrap_or(&manifest.capture.backend)
                .to_string();
            let device_id = resolve_effective_device(&backend_name, &manifest.capture.device_id);
            let is_hw = is_real_hw_device(&backend_name, &device_id);
            let effective_samples = if is_hw && *samples == 2048 {
                ((sr as usize) * 12).max(*samples)
            } else {
                *samples
            };
            let auto_yes = *yes || backend_name == "software";
            let mut cap_backend = create_capture_backend_with_gain(
                &backend_name,
                Some(&device_id),
                sr,
                manifest.capture.send_gain_db,
            );

            // Always ensure target probe calibration & probe-optimized dry stem synthesis have run
            let dry_path = proj.model_dry_input_wav_path(&model_id);
            if manifest.capture.excitation_profile.is_none() || !dry_path.is_file() {
                let cal = calibrate_two_step_with_clipping_prompt(
                    cap_backend.as_ref(),
                    &device_id,
                    sr,
                    auto_yes,
                )?;
                cap_backend.set_send_gain_db(cal.send_gain_db);
                proj.update_calibration_with_profile(
                    &model_id,
                    &backend_name,
                    &device_id,
                    cal.send_gain_db,
                    cal.latency_samples,
                    cal.noise_floor_dbfs,
                    cal.unity_ref_dbfs,
                    cal.excitation_profile,
                )?;
                manifest = proj.load_model(&model_id)?;
            }

            let dry = generate_excitation_signal_with_profile(
                sr,
                effective_samples,
                manifest.capture.excitation_profile.as_ref(),
            );
            let dry_sha = compute_pcm_sha256(&dry);
            let stems_dir = workspace.join("models").join(&model_id).join("stems");
            let dry_rel = format!("models/{model_id}/stems/input.wav");
            write_wav_24bit(&dry_path, &dry, sr, 1)?;

            let locked_lag = if manifest.capture.calibrated_latency_samples != 0 {
                Some(manifest.capture.calibrated_latency_samples)
            } else {
                None
            };
            let baseline_noise = Some(manifest.capture.noise_floor_dbfs);

            let pending: Vec<_> = plan
                .steps
                .iter()
                .filter(|s| {
                    if let Some(target_stem) = stem.as_deref() {
                        s.stem_id == target_stem
                    } else {
                        !s.already_captured || *all_stems
                    }
                })
                .collect();
            let total = max_stems
                .map(|m| pending.len().min(m))
                .unwrap_or(pending.len());
            let mut prev_ranges: Option<BTreeMap<String, f32>> =
                manifest.stems.last().map(|s| s.normalized_ranges.clone());
            let mut prev_switches: Option<BTreeMap<String, usize>> =
                manifest.stems.last().map(|s| s.switches.clone());
            let mut prev_clean_pad_db: Option<f32> = manifest.stems.last().map(|s| s.clean_pad_db);
            let mut active_clean_pad_db = *clean_pad_db;
            let mut recorded = 0_usize;
            let mut idx = 0_usize;
            let max_lag = (dry.len() / 4).max(32);
            let clip_lin = 10.0_f32.powf(-0.05 / 20.0);
            let mut stem_summaries = Vec::new();

            while idx < total {
                let step = pending[idx];
                let action = format::prompt_next_stem_action_with_pad(
                    &manifest,
                    step,
                    idx + 1,
                    total,
                    prev_ranges.as_ref(),
                    prev_switches.as_ref(),
                    prev_clean_pad_db,
                    Some(active_clean_pad_db),
                    auto_yes,
                    *countdown_sec,
                )?;
                match action {
                    StemPromptAction::Skip => {
                        idx += 1;
                        continue;
                    }
                    StemPromptAction::Retake => {
                        idx = idx.saturating_sub(1);
                        continue;
                    }
                    StemPromptAction::Record => {}
                }

                let (aligned_wet, align, raw_peak_dbfs, raw_clipped_fraction) = loop {
                    let wet =
                        cap_backend.capture_stem(&dry, &step.normalized_ranges, &step.switches)?;
                    let align = align_hardware_latency(
                        &dry,
                        &wet,
                        max_lag,
                        locked_lag,
                        sr,
                        baseline_noise,
                    )?;
                    let aligned_wet = apply_subsample_shift_fft(&wet, align.subsample_lag_samples);
                    let raw_peak = aligned_wet
                        .iter()
                        .fold(1e-9_f32, |acc, &x| acc.max(x.abs()));
                    let raw_peak_dbfs = 20.0 * raw_peak.log10();
                    let raw_clipped_count =
                        aligned_wet.iter().filter(|&&x| x.abs() >= clip_lin).count();
                    let raw_clipped_fraction =
                        (raw_clipped_count as f32) / (aligned_wet.len().max(1) as f32);
                    if is_step2_return_clipped(raw_peak_dbfs, raw_clipped_fraction)
                        && let Some(new_pad_db) = format::prompt_stem_clean_pad_on_clipping(
                            &step.stem_id,
                            raw_peak_dbfs,
                            raw_clipped_fraction,
                            active_clean_pad_db,
                            auto_yes,
                        )?
                    {
                        active_clean_pad_db = new_pad_db;
                        continue;
                    }
                    if format::is_clean_pad_excessive(raw_peak_dbfs, active_clean_pad_db)
                        && let Some(new_pad_db) = format::prompt_stem_excessive_clean_pad(
                            &step.stem_id,
                            raw_peak_dbfs,
                            active_clean_pad_db,
                            auto_yes,
                        )?
                    {
                        active_clean_pad_db = new_pad_db;
                        continue;
                    }
                    break (aligned_wet, align, raw_peak_dbfs, raw_clipped_fraction);
                };

                let makeup_db = active_clean_pad_db.abs();
                let makeup_lin = if makeup_db > 1e-4 {
                    10.0_f32.powf(makeup_db / 20.0)
                } else {
                    1.0
                };
                let restored_wet: Vec<f32> = aligned_wet.iter().map(|&s| s * makeup_lin).collect();
                let restored_peak_dbfs = raw_peak_dbfs + makeup_db;
                let wet_rel = format!("models/{model_id}/stems/{}.wav", step.stem_id);
                let wet_abs = stems_dir.join(format!("{}.wav", step.stem_id));
                write_wav_32bit_float(&wet_abs, &restored_wet, sr, 1)?;
                let sha = compute_pcm_sha256(&restored_wet);
                proj.register_stem(
                    &model_id,
                    StemRecord {
                        stem_id: step.stem_id.clone(),
                        dry_wav_path: dry_rel.clone(),
                        wet_wav_path: wet_rel,
                        normalized_ranges: step.normalized_ranges.clone(),
                        switches: step.switches.clone(),
                        pcm_sha256: sha,
                        dry_pcm_sha256: dry_sha.clone(),
                        aligned_lag_samples: align.subsample_lag_samples,
                        snr_db: align.snr_db,
                        noise_floor_dbfs: align.noise_floor_dbfs,
                        clean_pad_db: active_clean_pad_db,
                        peak_dbfs: restored_peak_dbfs,
                    },
                )?;
                let clip_tag = if is_step2_return_clipped(raw_peak_dbfs, raw_clipped_fraction) {
                    let rec_pad =
                        format::recommend_clean_pad_db(raw_peak_dbfs, active_clean_pad_db);
                    format!(
                        " [CLIPPED {:.2}% -> rec_clean_level={rec_pad:+.1} dB]",
                        raw_clipped_fraction * 100.0
                    )
                } else if format::is_clean_pad_excessive(raw_peak_dbfs, active_clean_pad_db) {
                    let rec_pad =
                        format::recommend_clean_pad_db(raw_peak_dbfs, active_clean_pad_db);
                    format!(" [EXCESSIVE PAD -> rec_clean_level={rec_pad:+.1} dB]")
                } else {
                    String::new()
                };
                let line = format!(
                    "  - {}: usb_peak={raw_peak_dbfs:+.2} dBFS{clip_tag} | clean_pad={active_clean_pad_db:+.1} dB | f32_peak={restored_peak_dbfs:+.2} dBFS | lag={:.2} samp | snr={:.1} dB",
                    step.stem_id, align.subsample_lag_samples, align.snr_db
                );
                eprintln!("{line}");
                stem_summaries.push(line);
                prev_ranges = Some(step.normalized_ranges.clone());
                prev_switches = Some(step.switches.clone());
                prev_clean_pad_db = Some(active_clean_pad_db);
                recorded += 1;
                idx += 1;
            }
            Ok(format!(
                "Recorded and aligned {recorded} stems for model '{model_id}'\n{}",
                stem_summaries.join("\n")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::parse_compact_range_spec;
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_capcli_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn capture_cli_subcommands_cover_generate_calibrate_plan_and_record() {
        let dir = temp_dir("cap_cli");
        let mut proj = DnamProject::init(&dir, "WS", "Auth").expect("init");

        // 0 models -> resolve_target_model_id error
        assert!(
            run_capture(
                &dir,
                &CaptureAction::Plan {
                    model: None,
                    no_corners: false,
                }
            )
            .is_err()
        );

        let backends = run_capture(&dir, &CaptureAction::Backends).expect("backends");
        assert!(backends.contains("anagram"));
        let devs = run_capture(
            &dir,
            &CaptureAction::Devices {
                backend: Some("software".to_string()),
            },
        )
        .expect("devices");
        assert!(devs.contains("loopback_default"));

        let out_wav = dir.join("gen.wav");
        let gen_msg = run_capture(
            &dir,
            &CaptureAction::Generate {
                output: out_wav.clone(),
                samples: 512,
                duration_sec: Some(0.1),
                sample_rate_hz: 8_000,
            },
        )
        .expect("generate duration");
        assert!(gen_msg.contains("800 samples @ 8000 Hz"));
        assert!(out_wav.is_file());

        proj.add_model("m1", "M1", "Full").expect("add m1");
        proj.add_range_control(
            "m1",
            parse_compact_range_spec("gain=0..10:5").expect("range"),
        )
        .expect("add range");

        // Record before calibration -> auto-calibrates and records
        let rec1 = run_capture(
            &dir,
            &CaptureAction::Record {
                model: None,
                backend: Some("software".to_string()),
                samples: 1024,
                stem: None,
                max_stems: None,
                clean_pad_db: 0.0,
                yes: true,
                countdown_sec: 0.0,
                all_stems: false,
            },
        )
        .expect("record uncalibrated");
        assert!(rec1.contains("Recorded and aligned 3 stems"));

        // Explicit calibrate (both steps, step 1, and step 2)
        let cal_out = run_capture(
            &dir,
            &CaptureAction::Calibrate {
                model: Some("m1".to_string()),
                backend: "software".to_string(),
                device: "loopback_default".to_string(),
                send_gain_db: Some(0.0),
                clean_pad_db: 0.0,
                step: None,
                yes: true,
            },
        )
        .expect("calibrate");
        assert!(cal_out.contains("excitation_profile"));

        let cal_s1 = run_capture(
            &dir,
            &CaptureAction::Calibrate {
                model: Some("m1".to_string()),
                backend: "software".to_string(),
                device: "loopback_default".to_string(),
                send_gain_db: Some(0.0),
                clean_pad_db: 0.0,
                step: Some(1),
                yes: true,
            },
        )
        .expect("calibrate step 1");
        assert!(cal_s1.contains("unity_ref_dbfs"));

        let cal_s2 = run_capture(
            &dir,
            &CaptureAction::Calibrate {
                model: Some("m1".to_string()),
                backend: "software".to_string(),
                device: "loopback_default".to_string(),
                send_gain_db: Some(0.0),
                clean_pad_db: -40.0,
                step: Some(2),
                yes: true,
            },
        )
        .expect("calibrate step 2");
        assert!(cal_s2.contains("excitation_profile"));

        // Plan shows [CAPTURED]
        let plan_out = run_capture(
            &dir,
            &CaptureAction::Plan {
                model: None,
                no_corners: true,
            },
        )
        .expect("plan");
        assert!(plan_out.contains("[CAPTURED]"));

        // Record again with all_stems = true, max_stems = Some(1), and clean_pad_db = -40.0
        let rec2 = run_capture(
            &dir,
            &CaptureAction::Record {
                model: Some("m1".to_string()),
                backend: None,
                samples: 1024,
                stem: None,
                max_stems: Some(1),
                clean_pad_db: -40.0,
                yes: true,
                countdown_sec: 0.0,
                all_stems: true,
            },
        )
        .expect("record max_stems=1 with -40 dB pad");
        assert!(rec2.contains("Recorded and aligned 1 stems"));
        assert!(rec2.contains("clean_pad=-40.0 dB"));
        let padded_wav = crate::capture::read_wav(
            dir.join("models")
                .join("m1")
                .join("stems")
                .join("stem_00_star_center.wav"),
        )
        .expect("read 32-bit float padded stem");
        let max_abs = padded_wav
            .samples
            .iter()
            .fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        assert!(
            max_abs > 5.0,
            "Expected +40 dB 32-bit float rescaled stem peak > 5.0, got {max_abs}"
        );

        // Verify Step-2 clipping detection & high-gain software calibration warning path
        assert!(!is_step2_return_clipped(-3.0, 0.0));
        assert!(is_step2_return_clipped(0.0, 0.01));
        assert!(is_step2_return_clipped(-0.02, 0.0));
        let cal_hot = run_capture(
            &dir,
            &CaptureAction::Calibrate {
                model: Some("m1".to_string()),
                backend: "software".to_string(),
                device: "loopback_default".to_string(),
                send_gain_db: Some(48.0),
                clean_pad_db: 0.0,
                step: Some(2),
                yes: true,
            },
        )
        .expect("calibrate step 2 hot");
        assert!(cal_hot.contains("hot_corner_clipped_fraction"));
        let _ = fs::remove_dir_all(&dir);
    }
}
