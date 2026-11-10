//! Envelope-Modulated Resonant SVF (`Chain-06`), Stereo Chorus & Rotary (`Chain-07`),
//! and Long BBD Echo + `30s` Shimmer FDN with Timed Freeze (`Chain-08`).

#![forbid(unsafe_code)]

pub mod bbd_shimmer_fdn;
pub mod chorus_rotary;
pub mod envelope_svf;

pub use bbd_shimmer_fdn::Chain08LongBbdEchoAnd30sShimmer;
pub use chorus_rotary::{CANONICAL_SUBDIVISION_RATIOS, Chain07StereoChorusAndRotary};
pub use envelope_svf::Chain06EnvelopeResonantSvf;
