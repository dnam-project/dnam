//! `.dnam-ckpt` (`DNCK`) checkpoint validation, stem `Arena 4` / `Arena 5` (`DNCA`) verification,
//! and runtime `.dnam` export (`ref[sdk.python.model.containers-and-checkpoints]`).

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container, crc32c_compute,
    export_runtime_dnam_from_ckpt, validate_v2_dnck_checkpoint, verify_ckpt_arena5_calib_stats,
    verify_ckpt_stem_arena4_crc32c,
};
use dnam_engine::provenance::RustCryptoProvider;
use dnam_engine::schema::{
    AuditReport, DnamJsonHeader, ExtractedPhysics, StemProvenance, StemRegistryEntry,
    TensorSliceEntry,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use super::populate_reference_controls_and_arena;
use crate::{dnam_err_to_py, validation_err};

/// Synthesizes a reference `.dnam-ckpt` (`DNCK`) container with a verified `Arena 4` stem record
/// and an optional `Arena 5` (`DNCA`) calibration statistics record.
#[pyfunction]
#[pyo3(signature = (include_calib_stats = true))]
pub fn native_build_reference_checkpoint<'py>(
    py: Python<'py>,
    include_calib_stats: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let container = py.detach(|| -> PyResult<Vec<u8>> {
        let mut header = DnamJsonHeader::default_reference();
        header.container_kind = "dnam-ckpt".to_string();
        header.version = if include_calib_stats {
            "1.1.0".to_string()
        } else {
            "1.0.0".to_string()
        };

        let mut arena = populate_reference_controls_and_arena(&mut header, true)?;
        let stem_offset = arena.len(); // 64,000
        let stem_record_bytes = vec![0x2A_u8; 4_096];
        let stem_crc = crc32c_compute(&stem_record_bytes);
        arena.extend_from_slice(&stem_record_bytes);

        header.stems_registry = Some(vec![StemRegistryEntry {
            stem_id: "stem_00_noon".to_string(),
            controls_meta: BTreeMap::from([
                ("gain".to_string(), serde_json::json!(5.0)),
                ("boost".to_string(), serde_json::json!("off")),
            ]),
            provenance: StemProvenance {
                input_pcm_sha256: "11".repeat(32),
                target_pcm_sha256: "22".repeat(32),
                num_frames: 48_000,
            },
            stem_arena4_crc32c: stem_crc,
            user_metadata: BTreeMap::new(),
            extracted_physics: Some(ExtractedPhysics {
                delays_ms: vec![250.0],
                lfo_freqs_hz: vec![1.5],
                decay_gains_lin: vec![0.65],
                detected_grain_size_ms: 0.0,
                detected_pitch_shift_semitones: 0.0,
            }),
            g_audio_uplo_u_f64: TensorSliceEntry {
                arena: 4,
                dtype: "f64".to_string(),
                packing: Some("uplo_upper_col_major_tier_ordered".to_string()),
                feature_dim: Some(64),
                packed_len: Some(2_080),
                shape: None,
                byte_offset: stem_offset as u32,
                byte_length: 2_048,
            },
            c_audio_tier_f64: TensorSliceEntry {
                arena: 4,
                dtype: "f64".to_string(),
                packing: None,
                feature_dim: None,
                packed_len: None,
                shape: Some(vec![64, 2]),
                byte_offset: (stem_offset + 2_048) as u32,
                byte_length: 2_048,
            },
        }]);

        header.audit_report = Some(AuditReport {
            phase1_clamps: vec!["CLAMP-RANGE".to_string(), "CLAMP-SWITCH".to_string()],
            phase2_rail_warnings: Vec::new(),
            phase3_clamps: Vec::new(),
            prescriptions: vec!["Record stem with boost=ultra to unlock switch option".to_string()],
        });

        if include_calib_stats {
            let mut a5 = vec![0u8; 64];
            a5[0..4].copy_from_slice(b"DNCA");
            a5[4..6].copy_from_slice(&1u16.to_le_bytes());
            a5[8..10].copy_from_slice(&0u16.to_le_bytes()); // num_inputs = 0
            a5[10..12].copy_from_slice(&0u16.to_le_bytes()); // num_stems = 0
            a5[12..16].copy_from_slice(&64u32.to_le_bytes()); // total_byte_len = 64
            let body_crc = crc32c_compute(&[]);
            a5[16..20].copy_from_slice(&body_crc.to_le_bytes());
            arena.extend_from_slice(&a5);
        }

        build_dnam_container(&mut header, &arena, &arena, false).map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &container))
}

/// Verifies a `.dnam-ckpt` stem's `stem_arena4_crc32c` over its `Arena 4` slice.
#[pyfunction]
pub fn native_verify_stem_arena4_crc32c(
    py: Python<'_>,
    arena_bytes: &[u8],
    stem_byte_offset: usize,
    stem_byte_len: usize,
    expected_crc32c: u32,
) -> PyResult<()> {
    py.detach(|| {
        verify_ckpt_stem_arena4_crc32c(
            arena_bytes,
            stem_byte_offset,
            stem_byte_len,
            expected_crc32c,
        )
    })
    .map_err(dnam_err_to_py)
}

/// Verifies `.dnam-ckpt` `Arena 5` (`DNCA` calibration statistics) starting at `arena5_offset`,
/// returning the total verified byte length of `Arena 5`.
#[pyfunction]
pub fn native_verify_arena5_calib_stats(
    py: Python<'_>,
    arena_bytes: &[u8],
    arena5_offset: usize,
) -> PyResult<usize> {
    py.detach(|| verify_ckpt_arena5_calib_stats(arena_bytes, arena5_offset))
        .map_err(dnam_err_to_py)
}

/// Validates a v2.0 `.dnam-ckpt` (`DNCK`) checkpoint container and returns `(retained_tier, bundle_ends, mutable_boundary)`.
#[pyfunction]
pub fn native_validate_v2_dnck_checkpoint<'py>(
    py: Python<'py>,
    ckpt_bytes: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let (retained_tier, ends, boundary) = py
        .detach(|| validate_v2_dnck_checkpoint(ckpt_bytes))
        .map_err(dnam_err_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("retained_tier", retained_tier)?;
    dict.set_item("bundle_ends", ends)?;
    dict.set_item("mutable_boundary", boundary)?;
    Ok(dict)
}

/// Exports a deployable runtime `.dnam` container from a `.dnam-ckpt` (`DNCK`) checkpoint container
/// (supports both v1.x `DnamContainerView` checkpoints and v2.0 template-backed `DNCK` checkpoints).
#[pyfunction]
pub fn native_export_runtime_dnam_from_ckpt<'py>(
    py: Python<'py>,
    ckpt_bytes: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let exported = py.detach(|| -> PyResult<Vec<u8>> {
        if ckpt_bytes.len() >= 8
            && &ckpt_bytes[0..4] == b"DNCK"
            && u16::from_le_bytes([ckpt_bytes[4], ckpt_bytes[5]]) == 2
        {
            return export_runtime_dnam_from_ckpt(ckpt_bytes).map_err(dnam_err_to_py);
        }

        let crypto = RustCryptoProvider::new();
        let view = DnamContainerView::from_bytes(ckpt_bytes, &crypto).map_err(dnam_err_to_py)?;
        if view.preamble.magic != *b"DNCK" {
            return Err(validation_err(
                "native_export_runtime_dnam_from_ckpt requires a .dnam-ckpt (DNCK) container",
            ));
        }

        let mut header = view.header.clone();
        header.version = "1.0.0".to_string();
        header.container_kind = "dnam".to_string();
        header.stems_registry = None;
        header.audit_report = None;
        header.tensor_directory.retain(|_, entry| entry.arena < 4);
        header.metadata.signatures = None;

        let full_arena = view.arena_storage.as_bytes();
        let runtime_end = (view.preamble.arena_offset_ultra_end() as usize)
            .max(ARENA_OFFSET_ULTRA_END as usize)
            .min(full_arena.len());
        let runtime_arena = &full_arena[..runtime_end];

        build_dnam_container(&mut header, runtime_arena, runtime_arena, false)
            .map_err(dnam_err_to_py)
    })?;
    Ok(PyBytes::new(py, &exported))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_build_reference_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(native_verify_stem_arena4_crc32c, m)?)?;
    m.add_function(wrap_pyfunction!(native_verify_arena5_calib_stats, m)?)?;
    m.add_function(wrap_pyfunction!(native_validate_v2_dnck_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(native_export_runtime_dnam_from_ckpt, m)?)?;
    Ok(())
}
