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

use napi::bindgen_prelude::{Error as NapiError, Float32Array, Float64Array, Status};

#[inline]
pub(crate) fn napi_err(msg: String) -> NapiError {
    NapiError::new(Status::GenericFailure, msg)
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn f32_slice_mut(arr: &mut Float32Array) -> &mut [f32] {
    // SAFETY: Synchronous N-API call on the JS thread; the output TypedArray is mutated in-place.
    unsafe { arr.as_mut() }
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn f64_slice_mut(arr: &mut Float64Array) -> &mut [f64] {
    // SAFETY: Synchronous N-API call on the JS thread; the output TypedArray is mutated in-place.
    unsafe { arr.as_mut() }
}
