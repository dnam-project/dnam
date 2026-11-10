//! Subsystem 1: Single Owner of `dnam.toml` and `models/<id>/model.toml`.

pub mod manifest;
pub mod workspace;

pub use manifest::{
    CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
    DnamProjectManifest, ModelManifest, PresetSpec, ProjectMetadata, StemRecord,
};
pub use workspace::{
    DnamProject, default_user_config_dir, ensure_creator_ed25519_keypair, parse_compact_range_spec,
    parse_compact_switch_spec, write_private_studio_export_file,
};
