//! `#![no_std]` + `alloc` JSON Schema, Control Descriptors, Signatures, and Error Types (`schema`).
//!
//! Parses and serializes the canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON headers (`0x40..`)
//! using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`).
//! Compiles human-readable `controls_schema` definitions into 64-byte binary
//! [`crate::control::ControlClampDescriptor`] cache lines for zero-`FDIV`, zero-string-parsing
//! $3\text{ kHz}$ execution.

pub mod controls;
pub mod error;
pub mod header;
pub mod signatures;

pub use controls::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    PresetDef, TempoSubdivision, TempoSyncResolution,
};
pub use error::{DnamError, cold_exp_f64, cold_ln_f64, cold_pow_f64, cold_sqrt_f64};
pub use header::{
    AuditReport, BlendMode, ControlBusConfig, Core1Config, Core2Config, DnamJsonHeader,
    ExtractedPhysics, IoConfig, LevelingConfig, LevelingMetric, ModelCapabilities, ModelMetadata,
    SchemaVersion, SlimmableTiersConfig, StemProvenance, StemRegistryEntry, TensorSliceEntry,
    TierProfile,
};
pub use signatures::{
    AuthorizedDevice, CreatorSignature, DatasetProvenance, DistributionPolicy,
    DnamProtectionProfile, MarketplaceReceipt, ModelSignatures, OemDeviceCertificate, RoyaltySplit,
};
