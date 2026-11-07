//! Node-API v10 (`napi-rs`) bindings for `@dnam-project/dnam` (`packages/js/rust/napi.rs`).
//!
//! Wraps the shared [`crate::core`] implementation with zero-copy `Float32Array`, `Float64Array`,
//! and `Uint8Array` views plus libuv worker-pool `AsyncTask` support (`ref[sdk.ts.runtime.dual-target-drivers]`,
//! `ref[sdk.ts.engine.zero-copy-and-async]`).

pub mod container;
pub mod control;
pub mod engine;
pub mod operators;
pub mod provenance;

use napi::bindgen_prelude::{Error as NapiError, Status};

pub(crate) type MutF32Slice<'a> = &'a mut [f32];
pub(crate) type MutF64Slice<'a> = &'a mut [f64];

#[inline]
pub(crate) fn napi_err(msg: String) -> NapiError {
    NapiError::new(Status::GenericFailure, msg)
}
