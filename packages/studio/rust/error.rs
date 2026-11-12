//! Typed error and result hierarchy for `dnam-studio`.

use std::fmt;

/// Canonical error variants returned by `dnam-studio` audio device, `.dnam-proj` project, and IPC operations.
#[derive(Debug)]
pub enum StudioError {
    /// Filesystem I/O failure.
    Io(std::io::Error),
    /// JSON wire-contract parse or serialization error.
    Json(String),
    /// Invalid user parameter, path, or IPC payload.
    Validation(String),
    /// Audio device or buffer failure.
    Audio(String),
    /// Underlying `dnam-capture` `.dnam-proj` container or fit error.
    Capture(String),
    /// Underlying `dnam-engine` `.dnam` container or DSP error.
    Engine(String),
}

impl fmt::Display for StudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Json(msg) => write!(f, "JSON protocol error: {msg}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
            Self::Audio(msg) => write!(f, "Audio device error: {msg}"),
            Self::Capture(msg) => write!(f, "Capture engine error: {msg}"),
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

impl From<dnam_engine::schema::DnamError> for StudioError {
    fn from(err: dnam_engine::schema::DnamError) -> Self {
        Self::Capture(err.to_string())
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

        let cap_err = StudioError::from(dnam_engine::schema::DnamError::InvalidMagic(*b"BAD!"));
        assert!(cap_err.to_string().contains("Capture engine error:"));

        assert_eq!(
            StudioError::Validation("bad spec".to_string()).to_string(),
            "Validation error: bad spec"
        );
        assert_eq!(
            StudioError::Audio("bad stream".to_string()).to_string(),
            "Audio device error: bad stream"
        );
        assert_eq!(
            StudioError::Engine("invalid magic".to_string()).to_string(),
            "DNAM engine error: invalid magic"
        );
    }
}
