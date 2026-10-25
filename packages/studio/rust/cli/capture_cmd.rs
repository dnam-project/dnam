//! Capture subcommand handlers (`backends`, `devices`, `generate`, `calibrate`, `plan`, `record`).

use std::path::{Path, PathBuf};

use clap::Subcommand;

use crate::capture::{
    CaptureBackend, SoftwareLoopbackBackend, compute_pcm_sha256, estimate_subsample_lag_and_snr,
    generate_excitation_signal, list_backend_devices, list_capture_backends, write_wav_24bit,
};
use crate::cli::format;
use crate::error::StudioResult;
use crate::project::{DnamProject, StemRecord, build_capture_plan};

/// Subcommands under `dnam capture`.
#[derive(Debug, Subcommand)]
pub enum CaptureAction {
    /// List available capture backends (`software`, `usb`, `anagram`).
    Backends,
    /// List audio interfaces for a capture backend.
    Devices {
        /// Backend name (`software`, `usb`, `anagram`).
        #[arg(long, default_value = "software")]
        backend: String,
    },
    /// Generate a 24-bit dry excitation WAV file.
    Generate {
        /// Output WAV file path.
        #[arg(long)]
        output: PathBuf,
        /// Number of samples to synthesize.
        #[arg(long, default_value_t = 4096)]
        samples: usize,
    },
    /// Run 2-step round-trip latency and noise floor calibration.
    Calibrate {
        /// Backend name.
        #[arg(long, default_value = "software")]
        backend: String,
        /// Device identifier.
        #[arg(long, default_value = "loopback_default")]
        device: String,
    },
    /// Display the Star + Switch + Corner capture schedule for a model.
    Plan {
        /// Target model ID.
        #[arg(long)]
        model: String,
    },
    /// Execute the planned capture stems for a model using the active backend.
    Record {
        /// Target model ID.
        #[arg(long)]
        model: String,
        /// Number of samples per stem.
        #[arg(long, default_value_t = 2048)]
        samples: usize,
    },
}

/// Executes `dnam capture`.
pub fn run_capture(workspace: &Path, action: &CaptureAction) -> StudioResult<String> {
    match action {
        CaptureAction::Backends => {
            let list = list_capture_backends();
            Ok(serde_json::to_string_pretty(&list)?)
        }
        CaptureAction::Devices { backend } => {
            let devices = list_backend_devices(backend);
            Ok(serde_json::to_string_pretty(&devices)?)
        }
        CaptureAction::Generate { output, samples } => {
            let sig = generate_excitation_signal(48_000, *samples);
            write_wav_24bit(output, &sig, 48_000, 1)?;
            Ok(format!(
                "Generated 24-bit dry excitation WAV ({} samples) at {}",
                samples,
                output.display()
            ))
        }
        CaptureAction::Calibrate { backend, device } => {
            let mut proj = DnamProject::open(workspace)?;
            let lo = SoftwareLoopbackBackend::default();
            let cal = lo.calibrate(device, proj.manifest().project.sample_rate_hz)?;
            proj.update_calibration(backend, device, cal.latency_samples, cal.noise_floor_dbfs)?;
            Ok(serde_json::to_string_pretty(&cal)?)
        }
        CaptureAction::Plan { model } => {
            let proj = DnamProject::open(workspace)?;
            let manifest = proj.load_model(model)?;
            let plan = build_capture_plan(&manifest, true);
            Ok(format::format_capture_plan(&plan))
        }
        CaptureAction::Record { model, samples } => {
            let proj = DnamProject::open(workspace)?;
            let manifest = proj.load_model(model)?;
            let plan = build_capture_plan(&manifest, true);
            let lo = SoftwareLoopbackBackend::default();
            let dry = generate_excitation_signal(proj.manifest().project.sample_rate_hz, *samples);
            let stems_dir = workspace.join("models").join(model).join("stems");
            let dry_rel = format!("models/{model}/stems/dry_excitation.wav");
            write_wav_24bit(
                workspace.join(&dry_rel),
                &dry,
                proj.manifest().project.sample_rate_hz,
                1,
            )?;

            let mut recorded = 0_usize;
            for step in &plan.steps {
                let wet = lo.capture_stem(&dry, &step.normalized_ranges, &step.switches)?;
                let align = estimate_subsample_lag_and_snr(&dry, &wet, 32)?;
                let wet_rel = format!("models/{model}/stems/{}.wav", step.stem_id);
                let wet_abs = stems_dir.join(format!("{}.wav", step.stem_id));
                write_wav_24bit(&wet_abs, &wet, proj.manifest().project.sample_rate_hz, 1)?;
                let sha = compute_pcm_sha256(&wet);
                proj.register_stem(
                    model,
                    StemRecord {
                        stem_id: step.stem_id.clone(),
                        dry_wav_path: dry_rel.clone(),
                        wet_wav_path: wet_rel,
                        normalized_ranges: step.normalized_ranges.clone(),
                        switches: step.switches.clone(),
                        pcm_sha256: sha,
                        aligned_lag_samples: align.subsample_lag_samples,
                        snr_db: align.snr_db,
                    },
                )?;
                recorded += 1;
            }
            Ok(format!(
                "Recorded and aligned {recorded} stems for model '{model}'"
            ))
        }
    }
}
