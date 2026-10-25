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
    /// Initialize a new DNAM workspace (`dnam.toml`, `keys/`, `models/`).
    Init {
        /// Workspace project name.
        #[arg(long, default_value = "DNAM Studio Workspace")]
        name: String,
        /// Creator or studio handle.
        #[arg(long, default_value = "studio")]
        author: String,
    },
    /// Show workspace, models, capture calibration, and hardware SIMD status.
    Status,
    /// Manage models in `models/<id>/model.toml`.
    Model {
        /// Action (`add` or `show`).
        #[command(subcommand)]
        action: workspace_cmd::ModelAction,
    },
    /// Add continuous `range` or discrete `switch` controls to a model.
    Control {
        /// Action (`add-range` or `add-switch`).
        #[command(subcommand)]
        action: workspace_cmd::ControlAction,
    },
    /// Manage re-amp hardware/software capture, calibration, excitation generation, and schedules.
    Capture {
        /// Capture subcommand.
        #[command(subcommand)]
        action: capture_cmd::CaptureAction,
    },
    /// Prepare and run the headless Python 4-stage trainer (`dnam-train`).
    Train {
        /// Model ID to train.
        #[arg(long)]
        model: String,
        /// Optional epoch count override.
        #[arg(long)]
        epochs: Option<u32>,
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
        Some(DnamCommand::Init { name, author }) => {
            workspace_cmd::run_init(&cli.workspace, name, author)
        }
        Some(DnamCommand::Status) => workspace_cmd::run_status(&cli.workspace),
        Some(DnamCommand::Model { action }) => workspace_cmd::run_model(&cli.workspace, action),
        Some(DnamCommand::Control { action }) => workspace_cmd::run_control(&cli.workspace, action),
        Some(DnamCommand::Capture { action }) => capture_cmd::run_capture(&cli.workspace, action),
        Some(DnamCommand::Train {
            model,
            epochs,
            dry_run,
        }) => train_cmd::run_train(&cli.workspace, model, *epochs, *dry_run),
        Some(DnamCommand::Trainer { action }) => train_cmd::run_trainer_env(&cli.workspace, action),
        Some(DnamCommand::Studio { headless_check }) => {
            workspace_cmd::run_studio(&cli.workspace, *headless_check)
        }
        None => Ok(format::format_cli_banner()),
    }
}
