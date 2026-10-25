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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_studio_error_display_and_conversions() {
        let io_err = StudioError::from(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing file",
        ));
        assert!(io_err.to_string().contains("I/O error: missing file"));

        let json_raw = serde_json::from_str::<serde_json::Value>("{bad").unwrap_err();
        let json_err = StudioError::from(json_raw);
        assert!(json_err.to_string().contains("JSON protocol error:"));

        assert_eq!(
            StudioError::Toml("bad toml".to_string()).to_string(),
            "TOML manifest error: bad toml"
        );
        assert_eq!(
            StudioError::Validation("bad spec".to_string()).to_string(),
            "Validation error: bad spec"
        );
        assert_eq!(
            StudioError::Audio("bad wav".to_string()).to_string(),
            "Audio capture error: bad wav"
        );
        assert_eq!(
            StudioError::Trainer("worker failed".to_string()).to_string(),
            "Python trainer error: worker failed"
        );
        assert_eq!(
            StudioError::Engine("invalid magic".to_string()).to_string(),
            "DNAM engine error: invalid magic"
        );
    }
}
