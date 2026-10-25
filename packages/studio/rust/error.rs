//! Typed error and result hierarchy for `dnam-studio`.

use std::fmt;

/// Canonical error variants returned by `dnam-studio` workspace, capture, trainer, and IPC operations.
#[derive(Debug)]
pub enum StudioError {
    /// Filesystem or subprocess I/O failure.
    Io(std::io::Error),
    /// TOML manifest parse or serialization error.
    Toml(String),
    /// JSON wire-contract parse or serialization error.
    Json(String),
    /// Invalid user control, preset, or stem specification.
    Validation(String),
    ///WAV format or alignment failure.
    Audio(String),
    /// Python trainer environment (`uv`) or worker execution error.
    Trainer(String),
    /// Underlying `dnam-engine` container or DSP error.
    Engine(String),
}

impl fmt::Display for StudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Toml(msg) => write!(f, "TOML manifest error: {msg}"),
            Self::Json(msg) => write!(f, "JSON protocol error: {msg}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
            Self::Audio(msg) => write!(f, "Audio capture error: {msg}"),
            Self::Trainer(msg) => write!(f, "Python trainer error: {msg}"),
            Self::Engine(msg) => write!(f, "DNAM engine error: {msg}"),
        }
    }
}

impl std::error::Error for StudioError {}

impl From<std::io::Error> for StudioError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for StudioError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err.to_string())
    }
}

/// Result alias for `dnam-studio`.
pub type StudioResult<T> = Result<T, StudioError>;
