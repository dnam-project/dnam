//! Capture subcommand handlers (`backends`, `devices`, `generate`, `calibrate`, `plan`, `record`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Subcommand;

use crate::capture::{
    align_hardware_latency, apply_subsample_shift_fft, compute_pcm_sha256,
    create_capture_backend_with_gain, generate_excitation_signal,
    generate_excitation_signal_with_profile, list_backend_devices, list_capture_backends,
    write_wav_24bit,
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
            yes,
        } => {
            let mut proj = DnamProject::open(workspace)?;
            let sr = proj.manifest().project.sample_rate_hz;
            let initial_gain_db = send_gain_db.unwrap_or(proj.manifest().capture.send_gain_db);
            let cap_backend =
                create_capture_backend_with_gain(backend, Some(device), sr, initial_gain_db);
            let auto_yes = *yes || backend == "software";
            let cal = cap_backend.calibrate_with_prompts(
                device,
                sr,
                &mut |setup| format::prompt_calibration_step1(setup, auto_yes),
                &mut |step1| format::prompt_calibration_step2(step1, auto_yes),
            )?;
            if let Ok(model_id) = resolve_target_model_id(&proj, model.as_deref()) {
                let tailored = generate_excitation_signal_with_profile(
                    sr,
                    4_096,
                    Some(&cal.excitation_profile),
                );
                let dry_path = proj.model_dry_input_wav_path(&model_id);
                write_wav_24bit(&dry_path, &tailored, sr, 1)?;
                if let Ok(mut m) = proj.load_model(&model_id) {
                    m.excitation_profile = Some(cal.excitation_profile.clone());
                    let _ = proj.save_model(&m);
                }
            }
            proj.update_calibration_with_profile(
                backend,
                device,
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
            yes,
            countdown_sec,
            all_stems,
        } => {
            let mut proj = DnamProject::open(workspace)?;
            let model_id = resolve_target_model_id(&proj, model.as_deref())?;
            let manifest = proj.load_model(&model_id)?;
            let ckpt_path = workspace
                .join("models")
                .join(&model_id)
                .join("checkpoint.dnam-ckpt");
            let prescriptions = extract_checkpoint_prescriptions(&ckpt_path, &manifest);
            let plan = build_capture_plan_with_prescriptions(&manifest, true, &prescriptions);
            let sr = proj.manifest().project.sample_rate_hz;
            let backend_name = backend
                .as_deref()
                .unwrap_or(&proj.manifest().capture.backend)
                .to_string();
            let device_id = proj.manifest().capture.device_id.clone();
            let auto_yes = *yes || backend_name == "software";
            let mut cap_backend = create_capture_backend_with_gain(
                &backend_name,
                Some(&device_id),
                sr,
                proj.manifest().capture.send_gain_db,
            );

            // Always ensure target probe calibration & probe-optimized dry stem synthesis have run
            let dry_path = proj.model_dry_input_wav_path(&model_id);
            let mut active_profile = manifest
                .excitation_profile
                .clone()
                .or_else(|| proj.manifest().capture.excitation_profile.clone());
            if active_profile.is_none() || !dry_path.is_file() {
                let cal = cap_backend.calibrate_with_prompts(
                    &device_id,
                    sr,
                    &mut |setup| format::prompt_calibration_step1(setup, auto_yes),
                    &mut |step1| format::prompt_calibration_step2(step1, auto_yes),
                )?;
                cap_backend.set_send_gain_db(cal.send_gain_db);
                active_profile = Some(cal.excitation_profile.clone());
                let mut m = proj.load_model(&model_id)?;
                m.excitation_profile = Some(cal.excitation_profile.clone());
                proj.save_model(&m)?;
                proj.update_calibration_with_profile(
                    &backend_name,
                    &device_id,
                    cal.send_gain_db,
                    cal.latency_samples,
                    cal.noise_floor_dbfs,
                    cal.unity_ref_dbfs,
                    cal.excitation_profile,
                )?;
            }

            let dry =
                generate_excitation_signal_with_profile(sr, *samples, active_profile.as_ref());
            let dry_sha = compute_pcm_sha256(&dry);
            let stems_dir = workspace.join("models").join(&model_id).join("stems");
            let dry_rel = format!("models/{model_id}/stems/input.wav");
            write_wav_24bit(&dry_path, &dry, sr, 1)?;

            let locked_lag = if proj.manifest().capture.calibrated_latency_samples != 0 {
                Some(proj.manifest().capture.calibrated_latency_samples)
            } else {
                None
            };
            let baseline_noise = Some(proj.manifest().capture.noise_floor_dbfs);

            let pending: Vec<_> = plan
                .steps
                .iter()
                .filter(|s| !s.already_captured || *all_stems)
                .collect();
            let total = pending.len();
            let mut prev_ranges: Option<BTreeMap<String, f32>> = None;
            let mut prev_switches: Option<BTreeMap<String, usize>> = None;
            let mut recorded = 0_usize;
            let mut idx = 0_usize;

            while idx < total {
                let step = pending[idx];
                let action = format::prompt_next_stem_action(
                    &manifest,
                    step,
                    idx + 1,
                    total,
                    prev_ranges.as_ref(),
                    prev_switches.as_ref(),
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

                let wet =
                    cap_backend.capture_stem(&dry, &step.normalized_ranges, &step.switches)?;
                let align = align_hardware_latency(&dry, &wet, 32, locked_lag, sr, baseline_noise)?;
                let aligned_wet = apply_subsample_shift_fft(&wet, align.subsample_lag_samples);
                let wet_rel = format!("models/{model_id}/stems/{}.wav", step.stem_id);
                let wet_abs = stems_dir.join(format!("{}.wav", step.stem_id));
                write_wav_24bit(&wet_abs, &aligned_wet, sr, 1)?;
                let sha = compute_pcm_sha256(&aligned_wet);
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
                    },
                )?;
                prev_ranges = Some(step.normalized_ranges.clone());
                prev_switches = Some(step.switches.clone());
                recorded += 1;
                idx += 1;
            }
            Ok(format!(
                "Recorded and aligned {recorded} stems for model '{model_id}'"
            ))
        }
    }
}
