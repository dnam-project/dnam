//! `dnam-js` — Unified Dual-Target Rust Core for `@dnam-project/dnam` (`packages/js/`).
//!
//! - On host targets (`cfg(not(target_arch = "wasm32"))`), compiles to a zero-copy Node-API v10
//!   addon (`dnam.<platform>-<arch>.node`) via `napi-rs` (`ref[sdk.ts.runtime.dual-target-drivers]`).
//! - On `wasm32-unknown-unknown` (`cfg(target_arch = "wasm32")`), compiles to a synchronous
//!   WebAssembly SIMD128 module (`dnam.wasm`) from the exact same shared Rust core.

#![deny(unsafe_code)]

#[forbid(unsafe_code)]
pub mod core;
#[forbid(unsafe_code)]
pub mod error;

#[cfg(not(target_arch = "wasm32"))]
pub mod napi;

#[cfg(all(not(target_arch = "wasm32"), feature = "testing"))]
pub mod testing;

#[cfg(any(target_arch = "wasm32", test))]
pub mod wasm;
