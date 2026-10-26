//! `dnam train` and `dnam trainer {status,install,uninstall}` subcommand handlers.

use std::fs;
use std::io::{self, Write as _};
use std::path::Path;

use clap::Subcommand;

use crate::error::{StudioError, StudioResult};
use crate::project::{DnamProject, default_user_config_dir, ensure_creator_ed25519_keypair};
use crate::trainer::{
    TrainerBridge, TrainerEnvManager, TrainerJobOptions, TrainingEventMessage,
    build_training_request_json,
};

/// Subcommands under `dnam trainer`.
#[derive(Debug, Subcommand)]
pub enum TrainerAction {
    /// Inspect `uv` discovery and `<user_config_dir>/trainer-venv` installation status.
    Status,
    /// Provision or update `<user_config_dir>/trainer-venv` via `uv`.
    Install {
        /// Simulate installation without downloading packages.
        #[arg(long)]
        dry_run: bool,
    },
    /// Remove the managed `<user_config_dir>/trainer-venv` directory.
    Uninstall,
}

/// Arguments for `dnam train`.
#[derive(Debug, Clone, Default)]
pub struct TrainCommandArgs {
    /// Target model ID (`None` when `--all-models` or when only one model exists).
    pub model_id: Option<String>,
    /// Train all models in the workspace with recorded stems.
    pub all_models: bool,
    /// Optional epoch count override.
    pub epochs: Option<u32>,
    /// Ignore any existing `.dnam-ckpt` and train from scratch.
    pub from_scratch: bool,
    /// Run Phase 1..3 control clamping audit without full gradient optimization.
    pub audit_only: bool,
    /// Force Tier-2 micro-polish during `.dnam-ckpt` resume.
    pub force_tier2: bool,
    /// Topology override (`auto`, `core1_only`, `full`).
    pub topology: Option<String>,
    /// Backend override (`local`, `remote`).
    pub backend: Option<String>,
    /// Device override (`auto`, `cpu`, `cuda`, `rocm`, `mps`, `tpu`).
    pub device: Option<String>,
    /// Precision override (`auto`, `fp32`, `bf16`, `fp16`).
    pub precision: Option<String>,
    /// Skip `Ed25519` creator signing.
    pub unsigned: bool,
    /// Emit only `training_request.json` without spawning `dnam-train`.
    pub dry_run: bool,
}

/// Executes `dnam train`.
pub fn run_train(workspace: &Path, args: TrainCommandArgs) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    let target_ids: Vec<String> = if args.all_models {
        proj.list_models()
            .into_iter()
            .filter(|m| !m.stems.is_empty())
            .map(|m| m.model_id)
            .collect()
    } else if let Some(mid) = &args.model_id {
        vec![mid.clone()]
    } else if proj.manifest().models.len() == 1 {
        vec![proj.manifest().models[0].clone()]
    } else {
        return Err(StudioError::Validation(
            "Specify `--model <id>` or pass `--all-models`".to_string(),
        ));
    };

    let creator_seed_hex = if args.unsigned {
        None
    } else {
        let explicit_key = proj
            .manifest()
            .project
            .signing_key_path
            .as_deref()
            .map(Path::new);
        let (_, seed_bytes, _) = ensure_creator_ed25519_keypair(explicit_key)?;
        let mut s = String::with_capacity(64);
        for b in seed_bytes {
            use std::fmt::Write as _;
            let _ = write!(&mut s, "{b:02x}");
        }
        Some(s)
    };

    let config_dir = default_user_config_dir();
    let mgr = TrainerEnvManager::new(&config_dir, Some(workspace.to_path_buf()));

    let mut summary_lines = Vec::new();
    for model_id in &target_ids {
        let model = proj.load_model(model_id)?;
        let req_json = build_training_request_json(proj.manifest(), &model, workspace, args.epochs);
        let model_dir = workspace.join("models").join(model_id);
        let reports_dir = model_dir.join("reports");
        let req_path = reports_dir.join("training_request.json");
        if let Some(parent) = req_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let encoded = serde_json::to_string_pretty(&req_json)?;
        fs::write(&req_path, &encoded)?;

        if args.dry_run {
            summary_lines.push(format!(
                "Wrote training request JSON for '{model_id}' to {}",
                req_path.display()
            ));
            continue;
        }

        let output_dnam = workspace.join("dist").join(format!("{model_id}.dnam"));
        let output_ckpt = model_dir.join("checkpoint.dnam-ckpt");

        if let Some(worker_bin) = mgr.discover_worker_binary() {
            let bridge = TrainerBridge::new(worker_bin.display().to_string(), Vec::new());
            let opts = TrainerJobOptions {
                checkpoint_in: Some(output_ckpt.clone()),
                from_scratch: args.from_scratch,
                audit_only: args.audit_only,
                force_tier2: args.force_tier2,
                epochs: args.epochs,
                topology: args.topology.clone(),
                backend: args.backend.clone(),
                device: args.device.clone(),
                precision: args.precision.clone(),
                unsigned: args.unsigned,
                creator_seed_hex: creator_seed_hex.clone(),
            };
            let events = bridge.run_job_with_options(
                &req_json,
                &req_path,
                &output_dnam,
                &output_ckpt,
                &reports_dir,
                &opts,
                |ev| {
                    if let Some(line) = format_training_event_line(ev) {
                        println!("{line}");
                        let _ = io::stdout().flush();
                    }
                },
            )?;
            summary_lines.push(format!(
                "Trained model '{}' -> {} ({} events, ckpt={})",
                model_id,
                output_dnam.display(),
                events.len(),
                output_ckpt.display()
            ));
        } else {
            summary_lines.push(format!(
                "Prepared training request JSON for '{model_id}' at {}",
                req_path.display()
            ));
        }
    }

    Ok(summary_lines.join("\n"))
}

/// Formats a live [`TrainingEventMessage`] into a human-readable progress line for CLI output.
pub fn format_training_event_line(ev: &TrainingEventMessage) -> Option<String> {
    let p = &ev.payload;
    match ev.event.as_str() {
        "training_stage" => {
            let stage = p.get("stage").and_then(|v| v.as_u64()).unwrap_or(0);
            let name = p
                .get("stage_name")
                .and_then(|v| v.as_str())
                .unwrap_or("Stage");
            let msg = p.get("message").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("  [Stage {stage}/4 — {name}] {msg}"))
        }
        "training_step" => {
            let stage = p.get("stage").and_then(|v| v.as_u64()).unwrap_or(0);
            let epoch = p.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0);
            let total = p.get("total_epochs").and_then(|v| v.as_u64()).unwrap_or(0);
            let lite = p.get("esr_lite_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let full = p.get("esr_full_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ultra = p
                .get("esr_ultra_db")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let elapsed = p.get("elapsed_sec").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let bands_suffix = match (
                p.get("esr_lf_db").and_then(|v| v.as_f64()),
                p.get("esr_mid_db").and_then(|v| v.as_f64()),
                p.get("esr_hf_db").and_then(|v| v.as_f64()),
            ) {
                (Some(lf), Some(mf), Some(hf)) => {
                    format!(" | Bands(LF/MF/HF)={lf:+.2}/{mf:+.2}/{hf:+.2} dB")
                }
                _ => String::new(),
            };
            Some(format!(
                "    • Stage {stage} Step {epoch:>2}/{total:<2} | bESR: Lite={lite:+.2} dB, Full={full:+.2} dB, Ultra={ultra:+.2} dB{bands_suffix} ({elapsed:.1}s)"
            ))
        }
        "audit_phase" => {
            let phase = p.get("phase").and_then(|v| v.as_u64()).unwrap_or(0);
            let clamps_count = p
                .get("audit_report")
                .and_then(|r| {
                    let k = match phase {
                        1 => "phase1_clamps",
                        2 => "phase2_rail_warnings",
                        _ => "phase3_clamps",
                    };
                    r.get(k).and_then(|arr| arr.as_array()).map(Vec::len)
                })
                .unwrap_or(0);
            Some(format!(
                "    ✓ Control Audit Phase {phase}/3 complete ({clamps_count} clamps/warnings)"
            ))
        }
        "training_complete" => {
            let mid = p
                .get("model_id")
                .and_then(|v| v.as_str())
                .unwrap_or("model");
            let lite = p.get("esr_lite_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let full = p.get("esr_full_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ultra = p
                .get("esr_ultra_db")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let mut psycho_suffix = String::new();
            if let (Some(lf), Some(mf), Some(hf)) = (
                p.get("esr_lf_db").and_then(|v| v.as_f64()),
                p.get("esr_mid_db").and_then(|v| v.as_f64()),
                p.get("esr_hf_db").and_then(|v| v.as_f64()),
            ) {
                psycho_suffix
                    .push_str(&format!(" | Bands(LF/MF/HF)={lf:+.2}/{mf:+.2}/{hf:+.2} dB"));
            }
            if let Some(mr_stft) = p.get("mr_stft_db").and_then(|v| v.as_f64()) {
                psycho_suffix.push_str(&format!(" | MR-STFT={mr_stft:.2} dB"));
            }
            if let (Some(rms_tilt), Some(hf_tilt)) = (
                p.get("ltas_rms_tilt_db").and_then(|v| v.as_f64()),
                p.get("ltas_hf_tilt_db").and_then(|v| v.as_f64()),
            ) {
                psycho_suffix.push_str(&format!(" | Tilt(RMS/HF)={rms_tilt:.2}/{hf_tilt:+.2} dB"));
            }
            if let Some(seg_esr) = p.get("seg_esr_db").and_then(|v| v.as_f64()) {
                psycho_suffix.push_str(&format!(" | SegESR={seg_esr:+.2} dB"));
            }
            let c2 = if p
                .get("core2_active")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "ON"
            } else {
                "BYPASS"
            };
            let elapsed = p.get("elapsed_sec").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(format!(
                "  ✓ Training Complete ({mid}) | Final bESR: Lite={lite:+.2} dB, Full={full:+.2} dB, Ultra={ultra:+.2} dB{psycho_suffix} | Core2={c2} ({elapsed:.1}s)"
            ))
        }
        _ => None,
    }
}

/// Executes `dnam trainer {status,install,uninstall}`.
pub fn run_trainer_env(workspace: &Path, action: &TrainerAction) -> StudioResult<String> {
    let config_dir = default_user_config_dir();
    let mgr = TrainerEnvManager::new(&config_dir, Some(workspace.to_path_buf()));
    match action {
        TrainerAction::Status => {
            let st = mgr.status();
            Ok(serde_json::to_string_pretty(&st)?)
        }
        TrainerAction::Install { dry_run } => {
            let st = mgr.install(*dry_run)?;
            Ok(serde_json::to_string_pretty(&st)?)
        }
        TrainerAction::Uninstall => {
            let st = mgr.uninstall()?;
            Ok(serde_json::to_string_pretty(&st)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::manifest::StemRecord;
    use std::collections::BTreeMap;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_traincli_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn train_cli_and_trainer_env_subcommands() {
        let dir = temp_dir("train_cli");
        let mut proj = DnamProject::init(&dir, "WS", "Auth").expect("init");

        // Zero models -> error when --model omitted
        assert!(run_train(&dir, TrainCommandArgs::default()).is_err());

        proj.add_model("m1", "M1", "Full").expect("add m1");
        proj.register_stem(
            "m1",
            StemRecord {
                stem_id: "s0".to_string(),
                dry_wav_path: "models/m1/stems/input.wav".to_string(),
                wet_wav_path: "models/m1/stems/s0.wav".to_string(),
                normalized_ranges: BTreeMap::new(),
                switches: BTreeMap::new(),
                pcm_sha256: "a".repeat(64),
                dry_pcm_sha256: "b".repeat(64),
                aligned_lag_samples: 0.0,
                snr_db: 40.0,
                noise_floor_dbfs: -90.0,
            },
        )
        .expect("reg stem");

        // Single model auto-select + dry_run + signed
        let out_dry = run_train(
            &dir,
            TrainCommandArgs {
                dry_run: true,
                ..TrainCommandArgs::default()
            },
        )
        .expect("dry_run single model");
        assert!(out_dry.contains("Wrote training request JSON for 'm1'"));

        // all_models + unsigned + dry_run = false with mock .venv/bin/dnam-train
        let venv_bin = dir.join(".venv/bin");
        fs::create_dir_all(&venv_bin).expect("create venv_bin");
        let mock_worker = venv_bin.join("dnam-train");
        fs::write(
            &mock_worker,
            "#!/bin/sh\nprintf '{\"event\":\"training_complete\"}\\n'\n",
        )
        .expect("write mock worker");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&mock_worker, fs::Permissions::from_mode(0o755))
                .expect("chmod mock worker");
        }

        let out_worker = run_train(
            &dir,
            TrainCommandArgs {
                all_models: true,
                unsigned: true,
                epochs: Some(2),
                ..TrainCommandArgs::default()
            },
        )
        .expect("run with mock worker");
        assert!(out_worker.contains("Trained model 'm1'"));

        // Trainer env status / install --dry-run / uninstall
        let st_json = run_trainer_env(&dir, &TrainerAction::Status).expect("status");
        assert!(st_json.contains("venv_dir"));
        let inst_json = run_trainer_env(&dir, &TrainerAction::Install { dry_run: true })
            .expect("install dry_run");
        assert!(inst_json.contains("\"installed\": true"));
        let _ = run_trainer_env(&dir, &TrainerAction::Uninstall).expect("uninstall");
        let _ = fs::remove_dir_all(&dir);
    }
}
