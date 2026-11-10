//! Subsystem 6: `clap` v4 Subcommand Router for the `dnam` CLI and Tauri v2 Studio.

pub mod format;
pub mod workspace_cmd;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use crate::error::StudioResult;

/// Unified DNAM (`dnam`) CLI and Tauri v2 Desktop Studio entrypoint.
#[derive(Debug, Parser)]
#[command(
    name = "dnam",
    version,
    about = "Dynamic Neural Audio Modeler (DNAM) CLI & Desktop Studio"
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
    /// Initialize a new DNAM workspace (`dnam.toml`, `dist/`, `models/`).
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
    /// Show workspace, models, and hardware SIMD status.
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
    /// Render `input_wav` through the native Rust SIMD `DnamEngine` and write a 32-bit float WAV.
    Render {
        /// Path to the `.dnam`, `.dnam-ckpt`, or `.dnam-seal` container.
        container_path: PathBuf,
        /// Input dry WAV path.
        input_wav: PathBuf,
        /// Output rendered 32-bit float WAV path.
        output_wav: PathBuf,
        /// Slimmable tier (`lite`, `full`, `pro`, `ultra`).
        #[arg(long, default_value = "ultra")]
        tier: String,
        /// Disable $4\times$ linear-phase ZOH oversampled offline bounce mode.
        #[arg(long)]
        no_offline_bounce: bool,
        /// Disable automatic promotion to the maximum available & supported tier during offline bounce.
        #[arg(long)]
        no_maximize_tier_on_bounce: bool,
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
    /// Strip higher slimmable tiers from a `.dnam` or `.dnam-ckpt` container without invalidating `Ed25519` signatures.
    Strip {
        /// Input `.dnam` or `.dnam-ckpt` container path.
        input_path: PathBuf,
        /// Output stripped container path.
        output_path: PathBuf,
        /// Maximum tier to retain (`lite`, `full`, `pro`, `ultra`).
        #[arg(long, default_value = "pro")]
        max_tier: String,
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
    /// Launch the Tauri v2 Desktop Studio GUI window (or inspect Studio launch config).
    Studio {
        /// Print Tauri v2 window & native audio host configuration as JSON without opening a window.
        #[arg(long)]
        headless_check: bool,
    },
}

/// Executes a parsed [`DnamCli`] invocation and returns formatted output lines.
pub fn execute_cli(cli: &DnamCli) -> StudioResult<String> {
    execute_cli_inner(cli, None)
}

/// Executes a parsed [`DnamCli`] invocation against an explicit user `config_dir`.
pub fn execute_cli_with_config_dir(cli: &DnamCli, config_dir: &Path) -> StudioResult<String> {
    execute_cli_inner(cli, Some(config_dir))
}

fn execute_cli_inner(cli: &DnamCli, config_dir: Option<&Path>) -> StudioResult<String> {
    match &cli.command {
        Some(DnamCommand::Init {
            name,
            author,
            model_id,
        }) => match config_dir {
            Some(cfg) => workspace_cmd::run_init_with_config_dir(
                &cli.workspace,
                cfg,
                name,
                author,
                model_id.as_deref(),
            ),
            None => workspace_cmd::run_init(&cli.workspace, name, author, model_id.as_deref()),
        },
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
            no_offline_bounce,
            no_maximize_tier_on_bounce,
            preset,
            set_controls,
            device_id,
            device_key_hex,
            marketplace_pubkey_hex,
        }) => workspace_cmd::run_render_with_bounce_and_crypto(
            container_path,
            input_wav,
            output_wav,
            tier,
            !*no_offline_bounce,
            !*no_maximize_tier_on_bounce,
            preset.as_deref(),
            set_controls,
            device_id.as_deref(),
            device_key_hex.as_deref(),
            marketplace_pubkey_hex.as_deref(),
        ),
        Some(DnamCommand::Strip {
            input_path,
            output_path,
            max_tier,
        }) => workspace_cmd::run_strip(input_path, output_path, max_tier),
        Some(DnamCommand::Model { action }) => workspace_cmd::run_model(&cli.workspace, action),
        Some(DnamCommand::Control { action }) => workspace_cmd::run_control(&cli.workspace, action),
        Some(DnamCommand::Preset { action }) => workspace_cmd::run_preset(&cli.workspace, action),
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

        let studio_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Studio {
                headless_check: true,
            }),
        })
        .expect("studio headless_check");
        assert!(!studio_out.is_empty());

        let mut header = DnamJsonHeader::default_reference();
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
                container_path: c_path.clone(),
                input_wav: in_wav,
                output_wav: out_wav,
                tier: "lite".to_string(),
                no_offline_bounce: false,
                no_maximize_tier_on_bounce: false,
                preset: None,
                set_controls: vec!["gain=5.0".to_string()],
                device_id: None,
                device_key_hex: None,
                marketplace_pubkey_hex: None,
            }),
        })
        .expect("render");
        assert!(rend.contains("Rendered 32-bit float WAV"));

        let stripped_path = dir.join("m_pro.dnam");
        let strip_out = execute_cli(&DnamCli {
            workspace: dir.clone(),
            command: Some(DnamCommand::Strip {
                input_path: c_path,
                output_path: stripped_path.clone(),
                max_tier: "pro".to_string(),
            }),
        })
        .expect("strip");
        assert!(strip_out.contains("Stripped container"));
        assert!(stripped_path.is_file());

        let _ = fs::remove_dir_all(&dir);
    }
}
