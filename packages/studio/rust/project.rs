//! Subsystem 1: Single Owner of `dnam.toml`, `models/<id>/model.toml`, and Capture Planning.

pub mod manifest;
pub mod planner;
pub mod workspace;

pub use manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelManifest, PresetSpec, ProjectMetadata, StemRecord,
};
pub use planner::{
    CapturePlan, CapturePlanStep, CaptureTier, ControlDelta, build_capture_plan,
    build_capture_plan_with_prescriptions, compute_control_deltas,
    extract_checkpoint_prescriptions,
};
pub use workspace::{
    DnamProject, default_user_config_dir, ensure_creator_ed25519_keypair, parse_compact_range_spec,
    parse_compact_switch_spec,
};
