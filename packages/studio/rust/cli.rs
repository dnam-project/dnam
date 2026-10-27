//! Subsystem 6: `clap` v4 Subcommand Router for the `dnam` CLI and Tauri v2 Studio.

pub mod capture_cmd;
pub mod format;
pub mod train_cmd;
pub mod workspace_cmd;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::error::StudioResult;

/// Unified DNAM (`dnam`) CLI and Tauri v2 Desktop Studio entrypoint.
#[derive(Debug, Parser)]
#[command(
    name = "dnam",
    version,
    about = "Dynamic Neural Audio Modeler (DNAM / Tri-Physics) CLI & Desktop Studio"
)]
pub struct DnamCli {
    /// Workspace root directory containing `dnam.toml` (defaults to current directory).
    #[arg(long, global = true, default_value = ".")]
    pub workspace: PathBuf,

    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Option<DnamCommand>,
}

/// Top-level `dnam` subcommands.
#[derive(Debug, Subcommand)]
pub enum DnamCommand {
    /// Initialize a new DNAM workspace (`dnam.toml`, `audio/input.wav`, `dist/`, `models/`).
    Init {
        /// Workspace project name.
        #[arg(long, default_value = "DNAM Studio Workspace")]
        name: String,
        /// Creator or studio handle.
        #[arg(long, default_value = "studio")]
        author: String,
        /// Optional initial model ID to create immediately.
        #[arg(long)]
        model_id: Option<String>,
    },
    /// Show workspace, models, capture calibration, and hardware SIMD status.
    Status,
    /// Inspect a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` container's preamble, tiers, controls, and provenance.
    Inspect {
        /// Path to the `.dnam`, `.dnam-ckpt`, or `.dnam-seal` container.
        container_path: PathBuf,
        /// Optional OEM device ID hash for `.dnam-seal` (`enclave_sealed`) or `device_allowlisted` unwrapping.
        #[arg(long)]
        device_id: Option<String>,
        /// Optional 64-character hex `X25519` device secret key for `.dnam-seal` or `device_allowlisted` unwrapping.
        #[arg(long)]
        device_key_hex: Option<String>,
        /// Optional 64-character hex `Ed25519` marketplace public key.
        #[arg(long)]
        marketplace_pubkey_hex: Option<String>,
    },
    /// Render `input_wav` through the native Rust SIMD `DnamEngine` and write a 24-bit WAV.
    Render {
        /// Path to the `.dnam`, `.dnam-ckpt`, or `.dnam-seal` container.
        container_path: PathBuf,
        /// Input dry WAV path.
        input_wav: PathBuf,
        /// Output rendered 24-bit WAV path.
        output_wav: PathBuf,
        /// Slimmable tier (`lite`, `full`, `ultra`).
        #[arg(long, default_value = "ultra")]
        tier: String,
        /// Optional factory preset ID to apply before rendering.
        #[arg(long)]
        preset: Option<String>,
        /// Control overrides (`--set drive=7.5 --set voice=Crunch`).
        #[arg(long = "set", short = 'c')]
        set_controls: Vec<String>,
        /// Optional OEM device ID hash for `.dnam-seal` (`enclave_sealed`) or `device_allowlisted` unwrapping.
        #[arg(long)]
        device_id: Option<String>,
        /// Optional 64-character hex `X25519` device secret key for `.dnam-seal` or `device_allowlisted` unwrapping.
        #[arg(long)]
        device_key_hex: Option<String>,
        /// Optional 64-character hex `Ed25519` marketplace public key.
        #[arg(long)]
        marketplace_pubkey_hex: Option<String>,
    },
    /// Manage models in `models/<id>/model.toml` (`add`, `list`, `show`, `update`, `remove`).
    Model {
        /// Model action (`add`, `list`, `show`, `update`, `remove`).
        #[command(subcommand)]
        action: workspace_cmd::ModelAction,
    },
    /// Manage continuous `range` and discrete `switch` controls on a model (`add`, `add-range`, `add-switch`, `list`, `update`, `remove`).
    Control {
        /// Control action.
        #[command(subcommand)]
        action: workspace_cmd::ControlAction,
    },
    /// Manage presets on a model (`add`, `list`, `update`, `remove`).
    Preset {
        /// Preset action.
        #[command(subcommand)]
        action: workspace_cmd::PresetAction,
    },
    /// Manage recorded stems on a model (`add`, `list`, `update`, `remove`).
    Stem {
        /// Stem action.
        #[command(subcommand)]
        action: workspace_cmd::StemAction,
    },
    /// Manage re-amp hardware/software capture, calibration, excitation generation, and schedules.
    Capture {
        /// Capture subcommand.
        #[command(subcommand)]
        action: capture_cmd::CaptureAction,
    },
    /// Train or iteratively resume a model (`dist/<id>.dnam` + `models/<id>/checkpoint.dnam-ckpt`).
    Train {
        /// Model ID to train (optional if `--all-models` or only one model exists).
        #[arg(long)]
        model: Option<String>,
        /// Train all models in the workspace that have recorded stems.
        #[arg(long)]
        all_models: bool,
        /// Optional epoch count override.
        #[arg(long)]
        epochs: Option<u32>,
        /// Ignore any existing `.dnam-ckpt` and train all 4 stages from scratch.
        #[arg(long)]
        from_scratch: bool,
        /// Run Phase 1..3 control clamping audit without full gradient optimization.
        #[arg(long)]
        audit_only: bool,
        /// Force Tier-2 micro-polish during `.dnam-ckpt` resume.
        #[arg(long)]
        force_tier2: bool,
        /// Architecture topology override (`auto`, `core1_only`, `full`).
        #[arg(long)]
        topology: Option<String>,
        /// Training backend (`local` or `remote`).
        #[arg(long)]
        backend: Option<String>,
        /// Accelerator device (`auto`, `cpu`, `cuda`, `rocm`, `mps`, `tpu`).
        #[arg(long)]
        device: Option<String>,
        /// Mixed-precision policy (`auto`, `fp32`, `bf16`, `fp16`).
        #[arg(long)]
        precision: Option<String>,
        /// Skip `Ed25519` creator signing.
        #[arg(long)]
        unsigned: bool,
        /// Emit only the generated `--request-json` file without spawning `dnam-train`.
        #[arg(long)]
        dry_run: bool,
    },
    /// Manage the isolated `uv` Python trainer virtual environment (`dnam-train`).
    Trainer {
        /// Trainer environment action (`status`, `install`, `uninstall`).
        #[command(subcommand)]
        action: train_cmd::TrainerAction,
    },
    /// Launch the Tauri v2 Desktop Studio GUI window (or inspect Studio launch config).
    Studio {
        /// Print Tauri v2 window & native audio host configuration as JSON without opening a window.
        #[arg(long)]
        headless_check: bool,
    },
}

/// Executes a parsed [`DnamCli`] invocation and returns formatted output lines.
pub fn execute_cli(cli: &DnamCli) -> StudioResult<String> {
    match &cli.command {
        Some(DnamCommand::Init {
            name,
            author,
            model_id,
        }) => workspace_cmd::run_init(&cli.workspace, name, author, model_id.as_deref()),
        Some(DnamCommand::Status) => workspace_cmd::run_status(&cli.workspace),
        Some(DnamCommand::Inspect {
            container_path,
            device_id,
            device_key_hex,
            marketplace_pubkey_hex,
        }) => workspace_cmd::run_inspect_with_crypto(
            container_path,
            device_id.as_deref(),
            device_key_hex.as_deref(),
            marketplace_pubkey_hex.as_deref(),
        ),
        Some(DnamCommand::Render {
            container_path,
            input_wav,
            output_wav,
            tier,
            preset,
            set_controls,
            device_id,
            device_key_hex,
            marketplace_pubkey_hex,
        }) => workspace_cmd::run_render_with_crypto(
            container_path,
            input_wav,
            output_wav,
            tier,
            preset.as_deref(),
            set_controls,
            device_id.as_deref(),
            device_key_hex.as_deref(),
            marketplace_pubkey_hex.as_deref(),
        ),
        Some(DnamCommand::Model { action }) => workspace_cmd::run_model(&cli.workspace, action),
        Some(DnamCommand::Control { action }) => workspace_cmd::run_control(&cli.workspace, action),
        Some(DnamCommand::Preset { action }) => workspace_cmd::run_preset(&cli.workspace, action),
        Some(DnamCommand::Stem { action }) => workspace_cmd::run_stem(&cli.workspace, action),
        Some(DnamCommand::Capture { action }) => capture_cmd::run_capture(&cli.workspace, action),
        Some(DnamCommand::Train {
            model,
            all_models,
            epochs,
            from_scratch,
            audit_only,
            force_tier2,
            topology,
            backend,
            device,
            precision,
            unsigned,
            dry_run,
        }) => train_cmd::run_train(
            &cli.workspace,
            train_cmd::TrainCommandArgs {
                model_id: model.clone(),
                all_models: *all_models,
                epochs: *epochs,
                from_scratch: *from_scratch,
                audit_only: *audit_only,
                force_tier2: *force_tier2,
                topology: topology.clone(),
                backend: backend.clone(),
                device: device.clone(),
                precision: precision.clone(),
                unsigned: *unsigned,
                dry_run: *dry_run,
            },
        ),
        Some(DnamCommand::Trainer { action }) => train_cmd::run_trainer_env(&cli.workspace, action),
        Some(DnamCommand::Studio { headless_check }) => {
            workspace_cmd::run_studio(&cli.workspace, *headless_check)
        }
        None => Ok(format::format_cli_banner()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::write_wav_24bit;
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::schema::DnamJsonHeader;
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn execute_cli_dispatches_all_top_level_commands() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_cli_router_{nanos}"));
        let _ = fs::create_dir_all(&dir);

        let banner = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: None,
        })
        .expect("banner");
        assert!(banner.contains("DNAM Studio & CLI"));

        let init_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Init {
                name: "TestWS".to_string(),
                author: "Tester".to_string(),
                model_id: Some("m1".to_string()),
            }),
        })
        .expect("init");
        assert!(init_out.contains("Initialized DNAM workspace"));

        let status_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Status),
        })
        .expect("status");
        assert!(status_out.contains("TestWS"));

        let model_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Model {
                action: workspace_cmd::ModelAction::List,
            }),
        })
        .expect("model list");
        assert!(model_out.contains("m1"));

        let ctrl_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Control {
                action: workspace_cmd::ControlAction::List {
                    model: "m1".to_string(),
                },
            }),
        })
        .expect("control list");
        assert!(ctrl_out.contains("ranges"));

        let preset_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Preset {
                action: workspace_cmd::PresetAction::List {
                    model: "m1".to_string(),
                },
            }),
        })
        .expect("preset list");
        assert!(!preset_out.is_empty());

        let stem_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Stem {
                action: workspace_cmd::StemAction::List {
                    model: "m1".to_string(),
                },
            }),
        })
        .expect("stem list");
        assert!(!stem_out.is_empty());

        let cap_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Capture {
                action: capture_cmd::CaptureAction::Backends,
            }),
        })
        .expect("capture backends");
        assert!(cap_out.contains("anagram"));

        let train_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Train {
                model: Some("m1".to_string()),
                all_models: false,
                epochs: Some(1),
                from_scratch: true,
                audit_only: false,
                force_tier2: false,
                topology: None,
                backend: None,
                device: None,
                precision: None,
                unsigned: true,
                dry_run: true,
            }),
        })
        .expect("train dry run");
        assert!(train_out.contains("Wrote training request JSON"));

        let studio_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Studio {
                headless_check: true,
            }),
        })
        .expect("studio headless_check");
        assert!(!studio_out.is_empty());

        let mut header = DnamJsonHeader::default_reference(false);
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let c_bytes =
            build_dnam_container(&mut header, &arena, &arena, false).expect("build container");
        let c_path = dir.join("m.dnam");
        fs::write(&c_path, &c_bytes).expect("write dnam");
        let in_wav = dir.join("in.wav");
        let out_wav = dir.join("out.wav");
        write_wav_24bit(&in_wav, &[0.1, -0.1, 0.2, -0.2], 48_000, 1).expect("write in wav");

        let insp = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Inspect {
                container_path: c_path.clone(),
                device_id: None,
                device_key_hex: None,
                marketplace_pubkey_hex: None,
            }),
        })
        .expect("inspect");
        assert!(insp.contains("Reference DNAM Model"));

        let rend = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Render {
                container_path: c_path,
                input_wav: in_wav,
                output_wav: out_wav,
                tier: "lite".to_string(),
                preset: None,
                set_controls: vec!["gain=5.0".to_string()],
                device_id: None,
                device_key_hex: None,
                marketplace_pubkey_hex: None,
            }),
        })
        .expect("render");
        assert!(rend.contains("Rendered 24-bit WAV"));

        let tr_st = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Trainer {
                action: train_cmd::TrainerAction::Status,
            }),
        })
        .expect("trainer status");
        assert!(tr_st.contains("venv_dir"));

        let _ = fs::remove_dir_all(&dir);
    }
}
