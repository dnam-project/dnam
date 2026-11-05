//! WebAssembly (`wasm32-unknown-unknown` + `simd128`) raw `extern "C"` ABI wrapping the shared
//! `dnam-js` core (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.web-audio.realtime-bridge]`).

#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod abi;
pub mod container;
pub mod control;
pub mod engine;
pub mod operators;
pub mod provenance;
