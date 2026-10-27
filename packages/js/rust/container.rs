//! Shared container validation, building, 4 KB in-place header update, 3-Phase control clamping,
//! ZOH rescaling, `uplo='U'` Gram packing, and hardware Castagnoli `CRC32C` logic.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, DnamPreamble, ZohRescaledParams,
    build_dnam_container, crc32c_compute, pack_uplo_u_f64, unpack_uplo_u_f64,
    update_header_in_place, uplo_u_packed_len,
};
use dnam_engine::control::{ControlClampDescriptor, MAX_CONTROLS};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader, PresetDef,
};

use crate::format_tagged_dnam_error;

/// Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
/// avoiding per-call JSON header parsing.
#[derive(Debug, Clone)]
pub struct ControlSpaceCore {
    descriptors: [ControlClampDescriptor; MAX_CONTROLS],
    num_controls: usize,
}

impl ControlSpaceCore {
    pub fn new(header_json: &str) -> Result<Self, String> {
        let header: DnamJsonHeader =
            serde_json::from_str(header_json).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?;
        let descriptors = header
            .compile_clamp_descriptors()
            .map_err(|e| format_tagged_dnam_error(&e))?;
        let num_controls = header.controls_schema.len().min(MAX_CONTROLS);
        Ok(Self {
            descriptors,
            num_controls,
        })
    }

    #[inline]
    pub const fn num_controls(&self) -> usize {
        self.num_controls
    }

    /// Evaluates 3-Phase control clamping directly into caller-provided slices (`0` heap allocations).
    /// Returns a 16-bit bitmask where bit `p` is `1` if control `p` was clamped.
    #[inline]
    pub fn clamp_into(
        &self,
        raw_numeric_values: &[f32],
        clamped_out: &mut [f32],
        norm_out: &mut [f32],
    ) -> u32 {
        let n = self.num_controls.min(clamped_out.len()).min(norm_out.len());
        let mut active_switches = [0_u8; MAX_CONTROLS];
        let mut was_clamped_mask = 0_u32;

        // Pass 1: Switch controls first so conditional range rules see active switch states
        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 1 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches);
                active_switches[p] = (clamped + 0.5) as u8;
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                if was_clamped {
                    was_clamped_mask |= 1_u32 << p;
                }
            }
        }
        // Pass 2: Range controls with switch-conditioned rules active
        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 0 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches);
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                if was_clamped {
                    was_clamped_mask |= 1_u32 << p;
                }
            }
        }
        was_clamped_mask
    }

    /// Evaluates 3-Phase control clamping over `raw_numeric_values` in `< 100 ns` without JSON parsing.
    /// Returns `(clamped_numeric, normalized_u, was_clamped)`.
    pub fn clamp(&self, raw_numeric_values: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
        let n = self.num_controls;
        let mut clamped_out = vec![0.0_f32; n];
        let mut norm_out = vec![0.0_f32; n];
        let mask = self.clamp_into(raw_numeric_values, &mut clamped_out, &mut norm_out);
        let was_clamped_out = (0..n).map(|p| (mask & (1_u32 << p)) != 0).collect();
        (clamped_out, norm_out, was_clamped_out)
    }
}

/// Validated container metadata extracted by [`validate_container_core`].
#[derive(Debug, Clone)]
pub struct ValidatedContainerInfo {
    pub preamble: DnamPreamble,
    pub magic_str: String,
    pub verification_status: String,
    pub header_json: String,
    pub arena_bytes: Vec<u8>,
}

/// Parses and validates a `.dnam` or `.dnam-ckpt` container byte slice.
pub fn validate_container_core(container_bytes: &[u8]) -> Result<ValidatedContainerInfo, String> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container_bytes, &crypto)
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let json_end = 64_usize.saturating_add(view.preamble.json_header_byte_len as usize);
    let header_json = if container_bytes.len() >= json_end
        && let Ok(raw_str) = core::str::from_utf8(&container_bytes[64..json_end])
    {
        raw_str.to_string()
    } else {
        serde_json::to_string(&view.header).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?
    };
    let magic_str = core::str::from_utf8(&view.preamble.magic)
        .unwrap_or("DNAM")
        .to_string();
    let verification_status = match view.verification_status {
        dnam_engine::provenance::VerificationStatus::Unsigned => "Unsigned".to_string(),
        dnam_engine::provenance::VerificationStatus::CreatorVerified { .. } => {
            "CreatorVerified".to_string()
        }
        dnam_engine::provenance::VerificationStatus::MarketplaceVerified { .. } => {
            "MarketplaceVerified".to_string()
        }
    };

    Ok(ValidatedContainerInfo {
        preamble: view.preamble,
        magic_str,
        verification_status,
        header_json,
        arena_bytes: view.arena_storage.as_bytes().to_vec(),
    })
}

/// Builds a 4 KB page-aligned `.dnam` or `.dnam-ckpt` container from `header_json` and `arena_bytes`.
pub fn build_container_core(
    header_json: &str,
    arena_bytes: &[u8],
    plaintext_arena_crc_source: Option<&[u8]>,
    encrypted_arena: bool,
) -> Result<Vec<u8>, String> {
    let mut header: DnamJsonHeader =
        serde_json::from_str(header_json).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?;
    header
        .validate()
        .map_err(|e| format_tagged_dnam_error(&e))?;

    let mut stored_arena = arena_bytes.to_vec();
    if !encrypted_arena
        && stored_arena.len() >= 8064
        && !header.controls_schema.is_empty()
        && stored_arena[7040..7044] == [0, 0, 0, 0]
        && let Ok(descriptors) = header.compile_clamp_descriptors()
    {
        for (idx, desc) in descriptors.iter().enumerate() {
            let off = 7040 + idx * 64;
            stored_arena[off..off + 64].copy_from_slice(&desc.to_bytes());
        }
    }

    let crc_src = plaintext_arena_crc_source.unwrap_or(&stored_arena);
    build_dnam_container(&mut header, crc_src, &stored_arena, encrypted_arena)
        .map_err(|e| format_tagged_dnam_error(&e))
}

/// Synthesizes a canonical reference `.dnam` (or `.dnam-ckpt`) container with built-in `"gain"` (`range`)
/// and `"boost"` (`switch`) controls, a `"warm_slapback"` preset, and calibrated dry/wet pass-through weights.
pub fn build_reference_container_core(
    core2_active: bool,
    is_checkpoint: bool,
) -> Result<Vec<u8>, String> {
    let mut header = DnamJsonHeader::default_reference(core2_active);
    if is_checkpoint {
        header.container_kind = "dnam-ckpt".to_string();
    }

    let mut cond_max = BTreeMap::new();
    cond_max.insert("boost=on".to_string(), 8.0_f32);

    header.controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: Some("dB".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 9.5]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: cond_max,
            declared_options: None,
            active_options: None,
            clamp_warning: Some(
                "[CLAMP-RANGE] Clamped at 9.5 dB (8.0 dB when boost=on)".to_string(),
            ),
            ui_hints: ControlUiHints {
                section: Some("Preamp".to_string()),
                order: 0,
                widget: Some("knob".to_string()),
                size: Some("lg".to_string()),
                polarity: Some("unipolar".to_string()),
                midpoint: Some(4.0),
                detents: vec![5.0],
                step: Some(0.1),
                decimals: Some(1),
                scale_labels: BTreeMap::new(),
                option_labels: BTreeMap::new(),
                color_hex: Some("#F59E0B".to_string()),
            },
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "boost".to_string(),
            name: "Boost".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("off"),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: Some(vec![
                "off".to_string(),
                "on".to_string(),
                "ultra".to_string(),
            ]),
            active_options: Some(vec!["off".to_string(), "on".to_string()]),
            clamp_warning: Some("[CLAMP-SWITCH] Option 'ultra' unobserved".to_string()),
            ui_hints: ControlUiHints {
                section: Some("Preamp".to_string()),
                order: 1,
                widget: Some("toggle".to_string()),
                ..Default::default()
            },
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut preset_controls = BTreeMap::new();
    preset_controls.insert("gain".to_string(), serde_json::json!(6.5));
    preset_controls.insert("boost".to_string(), serde_json::json!("on"));
    header.presets = vec![PresetDef {
        id: "warm_slapback".to_string(),
        name: "Warm Slapback".to_string(),
        description: Some("DNAM Studio preset".to_string()),
        controls: preset_controls,
        user_metadata: BTreeMap::new(),
    }];

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    arena[116..120].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[120..124].copy_from_slice(&0.5_f32.to_le_bytes());
    arena[124..128].copy_from_slice(&0.9993_f32.to_le_bytes());
    arena[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[20..24].copy_from_slice(&1.0_f32.to_le_bytes());
    arena[4224..4228].copy_from_slice(&0.25_f32.to_le_bytes());
    arena[4480..4484].copy_from_slice(&0.25_f32.to_le_bytes());

    let descriptors = header
        .compile_clamp_descriptors()
        .map_err(|e| format_tagged_dnam_error(&e))?;
    for (idx, desc) in descriptors.iter().enumerate() {
        let off = 7040 + idx * 64;
        arena[off..off + 64].copy_from_slice(&desc.to_bytes());
    }

    build_dnam_container(&mut header, &arena, &arena, false)
        .map_err(|e| format_tagged_dnam_error(&e))
}

/// Updates the JSON header in-place inside the 4 KB page-aligned region (`0x40..tensor_arena_offset`).
pub fn update_header_in_place_core(
    container_bytes: &[u8],
    updated_header_json: &str,
) -> Result<Vec<u8>, String> {
    let updated_header: DnamJsonHeader =
        serde_json::from_str(updated_header_json).map_err(|e| format!("[DNAM-VALIDATION] {e}"))?;
    updated_header
        .validate()
        .map_err(|e| format_tagged_dnam_error(&e))?;
    let mut buf = container_bytes.to_vec();
    update_header_in_place(&mut buf, &updated_header).map_err(|e| format_tagged_dnam_error(&e))?;
    Ok(buf)
}

/// Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters for `sample_rate_host_hz`.
#[allow(clippy::too_many_arguments)]
pub fn compute_zoh_core(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    control_decimation_train: u8,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
) -> Result<ZohRescaledParams, String> {
    ZohRescaledParams::compute(
        sample_rate_train_hz,
        sample_rate_host_hz,
        control_decimation_train,
        r_ac_train,
        r_dc_train,
        tau_ref_train,
        switch_xfade_train,
    )
    .map_err(|e| format_tagged_dnam_error(&e))
}

/// Packs a flattened row-major $N \times N$ symmetric matrix into `LAPACK uplo='U'` column-major upper-triangular format.
pub fn pack_uplo_u_core(dense_flat: &[f64], n_dim: usize) -> Result<Vec<f64>, String> {
    let expected = n_dim.saturating_mul(n_dim);
    if n_dim == 0 || dense_flat.len() != expected {
        return Err(format!(
            "[DNAM-VALIDATION] Expected {expected} elements for {n_dim}x{n_dim} matrix, got {}",
            dense_flat.len()
        ));
    }
    let packed_len = uplo_u_packed_len(n_dim);
    let mut out = vec![0.0_f64; packed_len];
    pack_uplo_u_f64(dense_flat, n_dim, &mut out);
    Ok(out)
}

/// Unpacks a `LAPACK uplo='U'` column-major upper-triangular buffer into a flattened $N_{\text{tier}} \times N_{\text{tier}}$ matrix.
pub fn unpack_uplo_u_core(packed_u: &[f64], n_tier: usize) -> Result<Vec<f64>, String> {
    if n_tier == 0 {
        return Err("[DNAM-VALIDATION] n_tier must be >= 1".to_string());
    }
    let req_len = uplo_u_packed_len(n_tier);
    if packed_u.len() < req_len {
        return Err(format!(
            "[DNAM-VALIDATION] Packed buffer too short for n_tier={n_tier}: need {req_len}, got {}",
            packed_u.len()
        ));
    }
    let mut out = vec![0.0_f64; n_tier * n_tier];
    unpack_uplo_u_f64(packed_u, n_tier, &mut out);
    Ok(out)
}

/// Computes the Castagnoli `CRC32C` checksum over `data`.
#[inline]
pub fn crc32c_core(data: &[u8]) -> u32 {
    crc32c_compute(data)
}
