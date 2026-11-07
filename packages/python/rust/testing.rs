//! Private test-only parity harness (`dnam._native._testing`, gated on `#[cfg(feature = "testing")]`).
//!
//! Exposes direct Rust `DynamicDnamEngine` rendering, reference/fully-populated `DnamJsonHeader`
//! JSON emitters, and an exhaustive `DnamError` variant raiser so Python tests verify bit-exact
//! parity without golden files (`ref[sdk.python.engine.pedalboard-call]`,
//! `ref[sdk.python.schema.rust-parity]`, `ref[sdk.python.errors.typed-hierarchy]`).

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
};
use dnam_engine::schema::{
    AuditReport, AuthorizedDevice, BlendMode, ControlBusConfig, ControlCalibration,
    ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints, CreatorSignature,
    DatasetProvenance, DistributionPolicy, DnamError, DnamJsonHeader, DnamProtectionProfile,
    ExtractedPhysics, IoConfig, LevelingConfig, LevelingMetric, MarketplaceReceipt,
    ModelCapabilities, ModelMetadata, ModelSignatures, PresetDef, RouterConfig, RoyaltySplit,
    SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig, Stage3CabinetKautzConfig,
    Stage4SpatialTankConfig, StemProvenance, StemRegistryEntry, TempoSubdivision, TensorSliceEntry,
    TierProfile,
};
use dnam_engine::{DynamicDnamEngine, Tier};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::arrays::slice_1d;
use crate::engine::parse_backend_kind;
use crate::{dnam_err_to_py, validation_err};

/// Renders stereo audio directly through `DynamicDnamEngine::process_block` in Rust for bit-exact parity testing.
#[pyfunction]
#[pyo3(signature = (container_bytes, tier, backend, sample_rate_hz, left_in, right_in, block_size = None, reset = true))]
#[allow(clippy::too_many_arguments)]
pub fn render_reference_stereo<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    tier: u8,
    backend: &str,
    sample_rate_hz: u32,
    left_in: PyReadonlyArray1<'py, f32>,
    right_in: PyReadonlyArray1<'py, f32>,
    block_size: Option<usize>,
    reset: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = left_in.len();
    if right_in.len() != n {
        return Err(validation_err(
            "left_in and right_in must have identical length",
        ));
    }
    let l_in = slice_1d(&left_in, "left_in")?;
    let r_in = slice_1d(&right_in, "right_in")?;
    let backend_enum = parse_backend_kind(backend)?;
    let (l_out, r_out) = py
        .detach(|| {
            let mut eng = DynamicDnamEngine::from_bytes_with_backend(
                container_bytes,
                Tier::from_u8(tier),
                sample_rate_hz,
                backend_enum,
            )?;
            if reset {
                eng.reset_to_dc_equilibrium();
            }
            let mut l_out = vec![0.0_f32; n];
            let mut r_out = vec![0.0_f32; n];
            let chunk = block_size.unwrap_or(n).max(1);
            if chunk >= n {
                eng.process_block(l_in, r_in, &mut l_out, &mut r_out);
            } else {
                for (((li, ri), lo), ro) in l_in
                    .chunks(chunk)
                    .zip(r_in.chunks(chunk))
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    eng.process_block(li, ri, lo, ro);
                }
            }
            Ok((l_out, r_out))
        })
        .map_err(dnam_err_to_py)?;
    PyArray2::from_vec2(py, &[l_out, r_out]).map_err(validation_err)
}

/// Renders mono-to-stereo audio directly through `DynamicDnamEngine::process_mono_to_stereo` in Rust.
#[pyfunction]
#[pyo3(signature = (container_bytes, tier, backend, sample_rate_hz, mono_in, block_size = None, reset = true))]
#[allow(clippy::too_many_arguments)]
pub fn render_reference_mono_to_stereo<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    tier: u8,
    backend: &str,
    sample_rate_hz: u32,
    mono_in: PyReadonlyArray1<'py, f32>,
    block_size: Option<usize>,
    reset: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = mono_in.len();
    let m_in = slice_1d(&mono_in, "mono_in")?;
    let backend_enum = parse_backend_kind(backend)?;
    let (l_out, r_out) = py
        .detach(|| {
            let mut eng = DynamicDnamEngine::from_bytes_with_backend(
                container_bytes,
                Tier::from_u8(tier),
                sample_rate_hz,
                backend_enum,
            )?;
            if reset {
                eng.reset_to_dc_equilibrium();
            }
            let mut l_out = vec![0.0_f32; n];
            let mut r_out = vec![0.0_f32; n];
            let chunk = block_size.unwrap_or(n).max(1);
            if chunk >= n {
                eng.process_mono_to_stereo(m_in, &mut l_out, &mut r_out);
            } else {
                for ((mi, lo), ro) in m_in
                    .chunks(chunk)
                    .zip(l_out.chunks_mut(chunk))
                    .zip(r_out.chunks_mut(chunk))
                {
                    eng.process_mono_to_stereo(mi, lo, ro);
                }
            }
            Ok((l_out, r_out))
        })
        .map_err(dnam_err_to_py)?;
    PyArray2::from_vec2(py, &[l_out, r_out]).map_err(validation_err)
}

/// Renders mono audio directly through `DynamicDnamEngine::process_mono` in Rust.
#[pyfunction]
#[pyo3(signature = (container_bytes, tier, backend, sample_rate_hz, mono_in, block_size = None, reset = true))]
#[allow(clippy::too_many_arguments)]
pub fn render_reference_mono<'py>(
    py: Python<'py>,
    container_bytes: &[u8],
    tier: u8,
    backend: &str,
    sample_rate_hz: u32,
    mono_in: PyReadonlyArray1<'py, f32>,
    block_size: Option<usize>,
    reset: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = mono_in.len();
    let m_in = slice_1d(&mono_in, "mono_in")?;
    let backend_enum = parse_backend_kind(backend)?;
    let m_out = py
        .detach(|| {
            let mut eng = DynamicDnamEngine::from_bytes_with_backend(
                container_bytes,
                Tier::from_u8(tier),
                sample_rate_hz,
                backend_enum,
            )?;
            if reset {
                eng.reset_to_dc_equilibrium();
            }
            let mut m_out = vec![0.0_f32; n];
            let chunk = block_size.unwrap_or(n).max(1);
            if chunk >= n {
                eng.process_mono(m_in, &mut m_out);
            } else {
                for (mi, mo) in m_in.chunks(chunk).zip(m_out.chunks_mut(chunk)) {
                    eng.process_mono(mi, mo);
                }
            }
            Ok(m_out)
        })
        .map_err(dnam_err_to_py)?;
    Ok(PyArray1::from_vec(py, m_out))
}

/// Emits a serialized `DnamJsonHeader` JSON string (`kind = "default"` or `"full"`) directly from Rust.
#[pyfunction]
#[pyo3(signature = (kind = "default"))]
pub fn reference_header_json(kind: &str) -> PyResult<String> {
    let header = match kind {
        "default" => DnamJsonHeader::default_reference(),
        "full" => DnamJsonHeader {
            version: "1.1.0".to_string(),
            container_kind: "dnam-ckpt".to_string(),
            architecture: "DNAM_TORA".to_string(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: "Fully Populated Parity Model".to_string(),
                author: Some("DNAM Audio Lab".to_string()),
                description: Some("Full schema parity fixture".to_string()),
                license: Some("LicenseRef-DNAM-1.0.0".to_string()),
                created_at: "2026-11-04T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance {
                    dataset_manifest_sha256: "aa".repeat(32),
                    config_recipe_sha256: Some("bb".repeat(32)),
                    epochs_trained: 42,
                },
                distribution_policy: Some(DistributionPolicy {
                    allowed_protection_profiles: vec![
                        DnamProtectionProfile::Open,
                        DnamProtectionProfile::DeviceAllowlisted,
                        DnamProtectionProfile::EnclaveSealed,
                    ],
                    allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
                }),
                signatures: Some(ModelSignatures {
                    immutable_header_sha256: "cc".repeat(32),
                    master_arena_sha256: "dd".repeat(32),
                    tier_arena_sha256: Some([
                        "11".repeat(32),
                        "22".repeat(32),
                        "33".repeat(32),
                        "44".repeat(32),
                    ]),
                    creator_signature: CreatorSignature {
                        signer_id: "did:web:dnam.audio:creators:alice".to_string(),
                        pubkey_ed25519: "55".repeat(32),
                        sig_ed25519: "66".repeat(64),
                    },
                    marketplace_receipt: Some(MarketplaceReceipt {
                        marketplace_id: "dnam.store".to_string(),
                        license_id: "0192f6c0-0000-7000-8000-000000000001".to_string(),
                        buyer_hash: "77".repeat(32),
                        license_tier: "commercial_studio".to_string(),
                        issued_at: "2026-11-04T01:00:00Z".to_string(),
                        buyer_arena_sha256: "88".repeat(32),
                        buyer_tier_arena_sha256: Some([
                            "91".repeat(32),
                            "92".repeat(32),
                            "93".repeat(32),
                            "94".repeat(32),
                        ]),
                        encryption_mode: "x25519_chacha20_arena".to_string(),
                        arena_nonce_96: Some("ab".repeat(12)),
                        ephemeral_pubkey_x25519: Some("cd".repeat(32)),
                        authorized_devices: vec![AuthorizedDevice {
                            oem_vendor: "darkglass".to_string(),
                            model_code: "anagram_rk3582".to_string(),
                            device_id_hash: "ef".repeat(16),
                            wrapped_cek: Some("12".repeat(48)),
                        }],
                        royalty_splits: vec![
                            RoyaltySplit {
                                recipient_id: "alice".to_string(),
                                role: "amp_capture".to_string(),
                                share_bps: 7_000,
                            },
                            RoyaltySplit {
                                recipient_id: "bob".to_string(),
                                role: "ir_cabinet".to_string(),
                                share_bps: 3_000,
                            },
                        ],
                        receipt_sig_ed25519: "99".repeat(64),
                    }),
                }),
            },
            user_metadata: BTreeMap::from([("tag".to_string(), serde_json::json!("vintage"))]),
            io_config: IoConfig {
                inputs: 2,
                output_mode: "stereo".to_string(),
                latency_samples: 0,
                tail_time_ms: 350.0,
                capabilities: ModelCapabilities {
                    blend_mode: BlendMode::HostParallel,
                    default_blend: 0.75,
                    wet_polarity_inverted: false,
                    supports_trails: true,
                    tempo_sync_capable: true,
                    supports_freeze: true,
                    freeze_mutes_dry: false,
                },
                leveling: LevelingConfig {
                    auto_unity_enabled: true,
                    leveling_metric: LevelingMetric::KWeightedLufs,
                    reference_input_level_db: -20.5,
                    measured_default_output_level_db: -18.5,
                    unity_gain_offset_db: -2.0,
                    target_gain_db: 1.0,
                    calibrated_trim_db: -1.0,
                },
            },
            slimmable_tiers: SlimmableTiersConfig {
                lite: TierProfile {
                    feature_bus_dim: 64,
                    channels: 4,
                    early_fir_taps: 16,
                    kautz_modes: 4,
                    tank_lines: 4,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -42.0,
                    arena_end_byte_offset: ARENA_OFFSET_LITE_END,
                    prefix_crc32c: 101,
                },
                full: TierProfile {
                    feature_bus_dim: 128,
                    channels: 8,
                    early_fir_taps: 32,
                    kautz_modes: 8,
                    tank_lines: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -46.0,
                    arena_end_byte_offset: ARENA_OFFSET_FULL_END,
                    prefix_crc32c: 202,
                },
                pro: TierProfile {
                    feature_bus_dim: 192,
                    channels: 12,
                    early_fir_taps: 48,
                    kautz_modes: 16,
                    tank_lines: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -49.0,
                    arena_end_byte_offset: ARENA_OFFSET_PRO_END,
                    prefix_crc32c: 303,
                },
                ultra: TierProfile {
                    feature_bus_dim: 256,
                    channels: 16,
                    early_fir_taps: 64,
                    kautz_modes: 32,
                    tank_lines: 16,
                    head_svd_rank: 8,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -52.0,
                    arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                    prefix_crc32c: 404,
                },
            },
            controls_schema: vec![
                ControlSchemaDef {
                    id: "delay_time".to_string(),
                    name: "Delay Time".to_string(),
                    kind: ControlKind::Range,
                    unit: Some("ms".to_string()),
                    lobatto_degree: Some(3),
                    anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                    declared_range: Some([10.0, 1000.0]),
                    safe_clamped_range: Some([20.0, 950.0]),
                    default: serde_json::json!(250.0),
                    conditional_min: BTreeMap::from([("range_mode=short".to_string(), 20.0)]),
                    conditional_max: BTreeMap::from([("range_mode=short".to_string(), 300.0)]),
                    declared_options: None,
                    active_options: None,
                    clamp_warning: Some("[CLAMP-RANGE] Safe range [20, 950] ms".to_string()),
                    calibration: Some(ControlCalibration {
                        role: ControlCalibrationRole::DelayTimeMs,
                        lobatto_physical_anchors: Some([20.0, 350.0, 950.0]),
                        conditional_physical_anchors: BTreeMap::from([(
                            "range_mode=short".to_string(),
                            [20.0, 120.0, 300.0],
                        )]),
                        tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
                    }),
                    ui_hints: ControlUiHints {
                        section: Some("Delay".to_string()),
                        order: 0,
                        widget: Some("knob".to_string()),
                        size: Some("lg".to_string()),
                        polarity: Some("unipolar".to_string()),
                        midpoint: Some(250.0),
                        detents: vec![250.0, 500.0],
                        step: Some(1.0),
                        decimals: Some(1),
                        scale_labels: BTreeMap::from([("20.0".to_string(), "Slap".to_string())]),
                        option_labels: BTreeMap::new(),
                        color_hex: Some("#38BDF8".to_string()),
                    },
                    user_metadata: BTreeMap::from([("midi_cc".to_string(), serde_json::json!(14))]),
                },
                ControlSchemaDef {
                    id: "range_mode".to_string(),
                    name: "Range Mode".to_string(),
                    kind: ControlKind::Switch,
                    unit: None,
                    lobatto_degree: None,
                    anchor_nodes_norm: None,
                    declared_range: None,
                    safe_clamped_range: None,
                    default: serde_json::json!("short"),
                    conditional_min: BTreeMap::new(),
                    conditional_max: BTreeMap::new(),
                    declared_options: Some(vec!["short".to_string(), "long".to_string()]),
                    active_options: Some(vec!["short".to_string(), "long".to_string()]),
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                },
            ],
            presets: vec![PresetDef {
                id: "dotted_eighth".to_string(),
                name: "Dotted Eighth".to_string(),
                description: Some("Rhythmic delay".to_string()),
                controls: BTreeMap::from([
                    ("delay_time".to_string(), serde_json::json!(375.0)),
                    ("range_mode".to_string(), serde_json::json!("short")),
                ]),
                user_metadata: BTreeMap::new(),
            }],
            control_bus: ControlBusConfig::default(),
            stage0_router: RouterConfig::default(),
            stage1_pre: Stage1PreFxConfig::default(),
            stage2_cascade: Stage2CascadeConfig::default(),
            stage3_cabinet: Stage3CabinetKautzConfig::default(),
            stage4_tank: Stage4SpatialTankConfig::default(),
            tensor_directory: BTreeMap::from([(
                "stage3_early_fir".to_string(),
                TensorSliceEntry {
                    arena: 0,
                    dtype: "f32".to_string(),
                    packing: None,
                    feature_dim: None,
                    packed_len: None,
                    shape: Some(vec![2, 16]),
                    byte_offset: 128,
                    byte_length: 128,
                },
            )]),
            stems_registry: Some(vec![StemRegistryEntry {
                stem_id: "stem_00".to_string(),
                controls_meta: BTreeMap::from([(
                    "delay_time".to_string(),
                    serde_json::json!(250.0),
                )]),
                provenance: StemProvenance {
                    input_pcm_sha256: "11".repeat(32),
                    target_pcm_sha256: "22".repeat(32),
                    num_frames: 48_000,
                },
                stem_arena4_crc32c: 999,
                user_metadata: BTreeMap::new(),
                extracted_physics: Some(ExtractedPhysics {
                    delays_ms: vec![250.0],
                    lfo_freqs_hz: vec![1.2],
                    decay_gains_lin: vec![0.7],
                    detected_grain_size_ms: 45.0,
                    detected_pitch_shift_semitones: 12.0,
                }),
                g_audio_uplo_u_f64: TensorSliceEntry {
                    arena: 4,
                    dtype: "f64".to_string(),
                    packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
                    feature_dim: Some(64),
                    packed_len: Some(2_080),
                    shape: None,
                    byte_offset: 64_000,
                    byte_length: 16_640,
                },
                c_audio_tier_f64: TensorSliceEntry {
                    arena: 4,
                    dtype: "f64".to_string(),
                    packing: None,
                    feature_dim: None,
                    packed_len: None,
                    shape: Some(vec![64, 2]),
                    byte_offset: 80_640,
                    byte_length: 1_024,
                },
            }]),
            audit_report: Some(AuditReport {
                phase1_clamps: vec!["CLAMP-RANGE".to_string()],
                phase2_rail_warnings: vec!["Stage II rail occupancy 4%".to_string()],
                phase3_clamps: vec!["CLAMP-DYNAMIC".to_string()],
                prescriptions: vec!["Add stem at delay_time=1000 ms".to_string()],
            }),
        },
        other => return Err(validation_err(format!("Unknown header kind '{other}'"))),
    };
    serde_json::to_string(&header).map_err(validation_err)
}

/// Returns the list of all `DnamError` variant names for exhaustive error-mapping verification.
#[pyfunction]
pub fn error_variant_names() -> Vec<&'static str> {
    vec![
        "BufferTooSmall",
        "InvalidMagic",
        "UnsupportedVersion",
        "UnalignedOffset",
        "Crc32cMismatch",
        "InvalidJsonHeader",
        "SchemaValidationError",
        "HeaderCapacityExceeded",
        "CryptoVerificationFailed",
        "EncryptedArenaRequiresDeviceKey",
        "InvalidSampleRate",
        "InvalidControlIndex",
        "PreviewContainerRejected",
    ]
}

/// Raises a specific `DnamError` variant through `dnam_err_to_py`.
#[pyfunction]
pub fn raise_error_variant(name: &str) -> PyResult<()> {
    let err = match name {
        "BufferTooSmall" => DnamError::BufferTooSmall {
            expected: 64,
            actual: 12,
        },
        "InvalidMagic" => DnamError::InvalidMagic(*b"BAD!"),
        "UnsupportedVersion" => DnamError::UnsupportedVersion { major: 9, minor: 0 },
        "UnalignedOffset" => DnamError::UnalignedOffset {
            field: "tensor_arena_offset",
            offset: 4100,
        },
        "Crc32cMismatch" => DnamError::Crc32cMismatch {
            region: "header",
            expected: 1,
            computed: 2,
        },
        "InvalidJsonHeader" => DnamError::InvalidJsonHeader("bad json".to_string()),
        "SchemaValidationError" => DnamError::SchemaValidationError("bad schema".to_string()),
        "HeaderCapacityExceeded" => DnamError::HeaderCapacityExceeded {
            required: 5000,
            capacity: 4032,
        },
        "CryptoVerificationFailed" => DnamError::CryptoVerificationFailed("bad sig"),
        "EncryptedArenaRequiresDeviceKey" => DnamError::EncryptedArenaRequiresDeviceKey,
        "InvalidSampleRate" => DnamError::InvalidSampleRate(8_000),
        "InvalidControlIndex" => DnamError::InvalidControlIndex(99),
        "PreviewContainerRejected" => DnamError::PreviewContainerRejected,
        other => return Err(validation_err(format!("Unknown error variant '{other}'"))),
    };
    Err(dnam_err_to_py(err))
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(parent.py(), "_testing")?;
    sub.add_function(wrap_pyfunction!(render_reference_stereo, &sub)?)?;
    sub.add_function(wrap_pyfunction!(render_reference_mono_to_stereo, &sub)?)?;
    sub.add_function(wrap_pyfunction!(render_reference_mono, &sub)?)?;
    sub.add_function(wrap_pyfunction!(reference_header_json, &sub)?)?;
    sub.add_function(wrap_pyfunction!(error_variant_names, &sub)?)?;
    sub.add_function(wrap_pyfunction!(raise_error_variant, &sub)?)?;
    parent.add_submodule(&sub)?;
    Ok(())
}
