//! `dnam-js` — Unified Dual-Target Rust FFI Core for `@dnam-project/engine` (`packages/js/`).
//!
//! - On host targets (`cfg(not(target_arch = "wasm32"))`), compiles to a zero-copy Node-API v8
//!   addon (`dnam_js.<platform>-<arch>.node`) via `napi-rs` (`#[forbid(unsafe_code)]`).
//! - On `wasm32-unknown-unknown` (`cfg(target_arch = "wasm32")`), compiles to a synchronous
//!   WebAssembly SIMD128 module (`dnam_engine.wasm`) from the exact same shared Rust core.

#[forbid(unsafe_code)]
pub mod container;
#[forbid(unsafe_code)]
pub mod dsp;
#[forbid(unsafe_code)]
pub mod engine;
#[forbid(unsafe_code)]
pub mod provenance;

#[cfg(not(target_arch = "wasm32"))]
#[deny(unsafe_code)]
pub mod napi_bindings;

#[cfg(target_arch = "wasm32")]
pub mod wasm_ffi;

use dnam_engine::schema::DnamError;

/// Formats a [`DnamError`] with a structured `[DNAM-*]` tag prefix so the TypeScript layer
/// (`src/errors.ts`) can map it to the exact `DnamError` subclass (`DnamContainerError`,
/// `DnamValidationError`, `DnamSampleRateError`).
pub(crate) fn format_tagged_dnam_error(err: &DnamError) -> String {
    match err {
        DnamError::InvalidMagic(_)
        | DnamError::UnsupportedVersion { .. }
        | DnamError::BufferTooSmall { .. }
        | DnamError::Crc32cMismatch { .. }
        | DnamError::HeaderCapacityExceeded { .. } => format!("[DNAM-CONTAINER] {err}"),
        DnamError::InvalidSampleRate(_) => format!("[DNAM-SAMPLERATE] {err}"),
        DnamError::UnalignedOffset { .. }
        | DnamError::InvalidJsonHeader(_)
        | DnamError::SchemaValidationError(_)
        | DnamError::InvalidControlIndex(_)
        | DnamError::CryptoVerificationFailed(_)
        | DnamError::EncryptedArenaRequiresDeviceKey => format!("[DNAM-VALIDATION] {err}"),
    }
}
