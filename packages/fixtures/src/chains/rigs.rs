//! Bi-Amp Bass Crossover & Output Iron (`Chain-11`) and Full End-to-End Stereo Rig (`Chain-12`).

#![forbid(unsafe_code)]

pub mod biamp_bass_iron;
pub mod full_stereo_rig;

pub use biamp_bass_iron::Chain11BiAmpBassAndOutputIron;
pub use full_stereo_rig::{Chain12FullEndToEndStereoRig, Chain12RigMode};
