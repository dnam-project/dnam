//! Target-agnostic safe Rust core (`#![forbid(unsafe_code)]`) shared between the Node.js 26
//! `N-API v10` addon and the `wasm32-unknown-unknown` `SIMD128` module
//! (`ref[sdk.ts.layout.feature-modules]`, `ref[sdk.ts.runtime.dual-target-drivers]`).

pub mod container;
pub mod control;
pub mod engine;
pub mod operators;
pub mod provenance;
