#![forbid(unsafe_code)]
//! `.dnam-ckpt` (`DNCK`) Checkpoint Container, `Arena 4` `uplo='U'` Packing, `Arena 5` (`DNCA`)
//! Calibration Statistics, and `DCA1` Bundle Framing (`dnam_trainer::ckpt`).

pub mod arena;
pub mod bundle;
pub mod schema;
pub mod view;

pub use arena::*;
pub use bundle::*;
pub use schema::*;
