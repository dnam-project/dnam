//! Program-Dependent Optical Compressor (`Chain-09`) and Analog Octave Divider & Ring Modulator (`Chain-10`).

#![forbid(unsafe_code)]

pub mod octave_ringmod;
pub mod optical_comp;

pub use octave_ringmod::Chain10Oc2OctaveDividerAndRingMod;
pub use optical_comp::Chain09La2aOpticalCompressor;
