//! Subsystem 1: Single Owner of `dnam.toml`, `models/<id>/model.toml`, and Capture Planning.

pub mod manifest;
pub mod planner;
pub mod workspace;

pub use manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelManifest, PresetSpec, ProjectMetadata, StemRecord,
};
pub use planner::{CapturePlan, CapturePlanStep, CaptureTier, build_capture_plan};
pub use workspace::{DnamProject, parse_compact_range_spec, parse_compact_switch_spec};
