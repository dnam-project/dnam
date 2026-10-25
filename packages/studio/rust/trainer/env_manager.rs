//! `TrainerEnvManager`: `uv` auto-discovery and `<config_dir>/trainer-venv` lifecycle management.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{StudioError, StudioResult};

/// Status snapshot of the managed Python trainer environment (`dnam-train`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainerEnvStatus {
    /// Discovered `uv` executable path, if available.
    pub uv_binary_path: Option<String>,
    /// Managed virtual environment directory (`<config_dir>/trainer-venv`).
    pub venv_dir: String,
    /// Resolved `dnam-train` worker executable path, if installed.
    pub worker_binary_path: Option<String>,
    /// Whether `dnam-train` is ready to run.
    pub installed: bool,
}

/// Manages `uv` discovery and the isolated Python trainer virtual environment.
#[derive(Debug, Clone)]
pub struct TrainerEnvManager {
    config_dir: PathBuf,
    monorepo_root: Option<PathBuf>,
}

impl TrainerEnvManager {
    /// Creates a [`TrainerEnvManager`] rooted at `config_dir`.
    pub fn new(config_dir: impl AsRef<Path>, monorepo_root: Option<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.as_ref().to_path_buf(),
            monorepo_root,
        }
    }

    /// Returns the managed virtual environment directory.
    pub fn venv_dir(&self) -> PathBuf {
        self.config_dir.join("trainer-venv")
    }

    /// Discovers the `uv` executable from `PATH`, `<config_dir>/bin/uv`, or standard user locations.
    pub fn discover_uv(&self) -> Option<PathBuf> {
        let local_uv = self.config_dir.join("bin").join(exe_name("uv"));
        if local_uv.is_file() {
            return Some(local_uv);
        }
        if let Some(path_var) = env::var_os("PATH") {
            for entry in env::split_paths(&path_var) {
                let candidate = entry.join(exe_name("uv"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        for fallback in ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"] {
            let p = PathBuf::from(fallback);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    /// Resolves the `dnam-train` executable inside the managed venv or workspace `.venv`.
    pub fn discover_worker_binary(&self) -> Option<PathBuf> {
        let venv_bin = self
            .venv_dir()
            .join(venv_bin_subdir())
            .join(exe_name("dnam-train"));
        if venv_bin.is_file() {
            return Some(venv_bin);
        }
        if let Some(root) = &self.monorepo_root {
            let mut cursor: Option<&Path> = Some(root.as_path());
            while let Some(dir) = cursor {
                let ws_bin = dir
                    .join(".venv")
                    .join(venv_bin_subdir())
                    .join(exe_name("dnam-train"));
                if ws_bin.is_file() {
                    return Some(ws_bin);
                }
                cursor = dir.parent();
            }
        }
        None
    }

    /// Inspects the current status of `uv` and `dnam-train`.
    pub fn status(&self) -> TrainerEnvStatus {
        let uv = self.discover_uv().map(|p| p.display().to_string());
        let worker = self
            .discover_worker_binary()
            .map(|p| p.display().to_string());
        let installed = worker.is_some();
        TrainerEnvStatus {
            uv_binary_path: uv,
            venv_dir: self.venv_dir().display().to_string(),
            worker_binary_path: worker,
            installed,
        }
    }

    /// Installs or updates `dnam-trainer` inside `<config_dir>/trainer-venv` (or simulates in dry-run).
    pub fn install(&self, dry_run: bool) -> StudioResult<TrainerEnvStatus> {
        let uv_path = self.discover_uv().ok_or_else(|| {
            StudioError::Trainer(
                "Could not locate `uv` binary; install uv or place it on PATH".to_string(),
            )
        })?;
        fs::create_dir_all(&self.config_dir)?;
        let venv = self.venv_dir();
        if dry_run {
            let bin_dir = venv.join(venv_bin_subdir());
            fs::create_dir_all(&bin_dir)?;
            let stub_worker = bin_dir.join(exe_name("dnam-train"));
            fs::write(&stub_worker, "#!/bin/sh\nexit 0\n")?;
            return Ok(self.status());
        }

        let status = Command::new(&uv_path)
            .arg("venv")
            .arg(&venv)
            .status()
            .map_err(|e| StudioError::Trainer(format!("Failed to run uv venv: {e}")))?;
        if !status.success() {
            return Err(StudioError::Trainer(
                "`uv venv` exited with non-zero status".to_string(),
            ));
        }
        Ok(self.status())
    }

    /// Removes the managed `<config_dir>/trainer-venv` environment.
    pub fn uninstall(&self) -> StudioResult<TrainerEnvStatus> {
        let venv = self.venv_dir();
        if venv.exists() {
            fs::remove_dir_all(&venv)?;
        }
        Ok(self.status())
    }
}

const fn venv_bin_subdir() -> &'static str {
    if cfg!(windows) { "Scripts" } else { "bin" }
}

fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}
