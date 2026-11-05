//! Exhaustive [`DnamError`] tag mapping for `@dnam-project/dnam` (`ref[sdk.ts.errors.coded-error]`).

use dnam_engine::schema::DnamError;

/// Formats a [`DnamError`] with an exhaustive `[DNAM-*]` tag prefix so the TypeScript layer
/// (`src/errors.ts`) maps every variant to its exact `ErrorCode` without a wildcard arm
/// (`ref[sdk.ts.errors.coded-error]`).
pub fn format_tagged_dnam_error(err: &DnamError) -> String {
    match err {
        DnamError::InvalidMagic(_)
        | DnamError::UnsupportedVersion { .. }
        | DnamError::BufferTooSmall { .. }
        | DnamError::Crc32cMismatch { .. }
        | DnamError::HeaderCapacityExceeded { .. } => format!("[DNAM-CONTAINER] {err}"),
        DnamError::InvalidSampleRate(_) => format!("[DNAM-SAMPLERATE] {err}"),
        DnamError::InvalidControlIndex(_) => format!("[DNAM-CONTROL] {err}"),
        DnamError::CryptoVerificationFailed(_)
        | DnamError::EncryptedArenaRequiresDeviceKey
        | DnamError::PreviewContainerRejected => format!("[DNAM-PROVENANCE] {err}"),
        DnamError::UnalignedOffset { .. }
        | DnamError::InvalidJsonHeader(_)
        | DnamError::SchemaValidationError(_) => format!("[DNAM-VALIDATION] {err}"),
    }
}

#[inline]
pub fn validation_err(msg: impl core::fmt::Display) -> String {
    format!("[DNAM-VALIDATION] {msg}")
}

#[inline]
pub fn control_err(msg: impl core::fmt::Display) -> String {
    format!("[DNAM-CONTROL] {msg}")
}

#[inline]
pub fn busy_err(msg: impl core::fmt::Display) -> String {
    format!("[DNAM-BUSY] {msg}")
}
