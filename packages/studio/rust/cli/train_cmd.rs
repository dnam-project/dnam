//! `dnam train` and `dnam trainer {status,install,uninstall}` subcommand handlers.

use std::fs;
use std::path::Path;

use clap::Subcommand;

use crate::error::{StudioError, StudioResult};
use crate::project::{DnamProject, default_user_config_dir, ensure_creator_ed25519_keypair};
use crate::trainer::{
    TrainerBridge, TrainerEnvManager, TrainerJobOptions, build_training_request_json,
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
                |_ev| {},
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
