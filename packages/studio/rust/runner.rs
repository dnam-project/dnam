//! Subsystem 3: Native SIMD `.dnam` / `.dnam-ckpt` Container Inspector & 24-Bit WAV Renderer (`dnam-engine`).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dnam_engine::container::DnamContainerView;
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::{DynamicDnamEngine, Tier};
use serde::{Deserialize, Serialize};

use crate::capture::{read_wav, write_wav_32bit_float};
use crate::error::{StudioError, StudioResult};

/// Summary returned when inspecting a `.dnam` or `.dnam-ckpt` container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContainerInspectionReport {
    /// Container magic (`"DNAM"`).
    pub magic: String,
    /// Container format major version.
    pub version_major: u16,
    /// Container format minor version.
    pub version_minor: u16,
    /// Whether this container includes a `.dnam-ckpt` Kronecker stem expansion arena.
    pub is_checkpoint: bool,
    /// Model display name from the JSON header.
    pub model_name: String,
    /// Creator or studio author from the JSON header.
    pub author: String,
    /// Cryptographic provenance verification status (`"CreatorVerified"`, `"Unsigned"`, etc.).
    pub verification_status: String,
    /// Container protection profile (`"open"`, `"device_allowlisted"`, `"enclave_sealed"`).
    #[serde(default)]
    pub protection_profile: String,
    /// Whether Core 2 (`M-UFDN`) is active.
    pub core2_active: bool,
    /// Whether $3\text{ kHz}$ LFOs or envelopes are active.
    pub lfo_env_active: bool,
    /// Training sample rate in Hz.
    pub sample_rate_train_hz: u32,
    /// Control-bus decimation factor (`16` -> $3\text{ kHz}$).
    pub control_decimation: u8,
    /// Active hardware SIMD acceleration backend (`"arm-sme"`, `"arm64-neon"`, `"x86-avx2-fma"`, `"scalar-reference"`).
    pub detected_simd_backend: String,
    /// Number of user-definable controls in the JSON header.
    pub num_controls: usize,
    /// Number of `.dnam-ckpt` Kronecker stem statistics blocks in Arena 3.
    #[serde(default)]
    pub num_ckpt_stems: u16,
    /// SVD head rank from preamble.
    #[serde(default)]
    pub head_svd_rank: u8,
    /// Tail decay time in milliseconds from preamble.
    #[serde(default)]
    pub tail_time_ms: f32,
    /// Tensor arena byte length (`Arena 0..2` or `Arena 0..3`).
    #[serde(default)]
    pub tensor_arena_byte_len: u32,
    /// Number of factory presets in the JSON header.
    pub num_presets: usize,
    /// Total file size in bytes.
    pub byte_len: usize,
}

/// Returns the canonical string identifier of the best hardware SIMD backend on this host.
pub fn detect_native_backend_str() -> &'static str {
    match detect_best_backend() {
        BackendKind::ArmSme => "arm-sme",
        BackendKind::Neon | BackendKind::ArmNeon => "arm64-neon",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma | BackendKind::X86Sse42 => "x86-avx2-fma",
        BackendKind::WasmSimd128 => "wasm-simd128",
        BackendKind::Scalar => "scalar-reference",
    }
}

fn decode_hex_32(hex: &str) -> StudioResult<[u8; 32]> {
    let trimmed = hex.trim();
    if trimmed.len() != 64 {
        return Err(StudioError::Validation(format!(
            "Expected 64 hex chars (32 bytes), got {} chars",
            trimmed.len()
        )));
    }
    let mut out = [0_u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let pair = &trimmed[i * 2..i * 2 + 2];
        *slot = u8::from_str_radix(pair, 16)
            .map_err(|_| StudioError::Validation(format!("Invalid hex byte '{pair}'")))?;
    }
    Ok(out)
}

fn build_crypto_provider(
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<DefaultCryptoProvider> {
    let mut crypto = DefaultCryptoProvider::default();
    if let (Some(id), Some(key_hex)) = (device_id, device_key_hex) {
        let secret = decode_hex_32(key_hex)?;
        crypto = crypto.with_device_key(id.trim(), secret);
    }
    if let Some(mkt_hex) = marketplace_pubkey_hex {
        let pk = decode_hex_32(mkt_hex)?;
        crypto = crypto.with_marketplace_pubkey(pk);
    }
    Ok(crypto)
}

/// Inspects a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` file on disk using `dnam-engine`.
pub fn inspect_container(path: impl AsRef<Path>) -> StudioResult<ContainerInspectionReport> {
    inspect_container_with_crypto(path, None, None, None)
}

/// Inspects a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` file on disk with optional device/marketplace keys.
pub fn inspect_container_with_crypto(
    path: impl AsRef<Path>,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<ContainerInspectionReport> {
    let bytes = fs::read(path)?;
    inspect_container_bytes_with_crypto(&bytes, device_id, device_key_hex, marketplace_pubkey_hex)
}

/// Inspects in-memory `.dnam`, `.dnam-ckpt`, or `.dnam-seal` bytes.
pub fn inspect_container_bytes(bytes: &[u8]) -> StudioResult<ContainerInspectionReport> {
    inspect_container_bytes_with_crypto(bytes, None, None, None)
}

/// Inspects in-memory `.dnam`, `.dnam-ckpt`, or `.dnam-seal` bytes with optional device/marketplace keys.
pub fn inspect_container_bytes_with_crypto(
    bytes: &[u8],
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<ContainerInspectionReport> {
    let crypto = build_crypto_provider(device_id, device_key_hex, marketplace_pubkey_hex)?;
    let view = DnamContainerView::from_bytes(bytes, &crypto)
        .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
    let preamble = &view.preamble;
    let verif = format!("{:?}", view.verification_status);
    let profile_str = match view.protection_profile {
        dnam_engine::schema::DnamProtectionProfile::Open => "open",
        dnam_engine::schema::DnamProtectionProfile::DeviceAllowlisted => "device_allowlisted",
        dnam_engine::schema::DnamProtectionProfile::EnclaveSealed => "enclave_sealed",
    };

    Ok(ContainerInspectionReport {
        magic: String::from_utf8_lossy(&preamble.magic).into_owned(),
        version_major: preamble.version_major,
        version_minor: preamble.version_minor,
        is_checkpoint: preamble.magic == *b"DNCK" || (preamble.flags & (1 << 3)) != 0,
        model_name: view.header.metadata.name.clone(),
        author: view
            .header
            .metadata
            .author
            .clone()
            .unwrap_or_else(|| "Unknown".to_string()),
        verification_status: verif,
        protection_profile: profile_str.to_string(),
        core2_active: (preamble.flags & (1 << 6)) != 0 || view.header.core2_mufdn.active,
        lfo_env_active: (preamble.flags & (1 << 7)) != 0
            || view.header.control_bus.envelopes > 0
            || view.header.control_bus.lfo_oscillators > 0,
        sample_rate_train_hz: preamble.sample_rate_train_hz,
        control_decimation: preamble.control_decimation,
        detected_simd_backend: detect_native_backend_str().to_string(),
        num_controls: view.header.controls_schema.len(),
        num_ckpt_stems: preamble.num_ckpt_stems,
        head_svd_rank: preamble.head_svd_rank,
        tail_time_ms: preamble.tail_time_ms,
        tensor_arena_byte_len: preamble.tensor_arena_byte_len,
        num_presets: view.header.presets.len(),
        byte_len: bytes.len(),
    })
}

/// Renders an input WAV file through a `.dnam` container with tier, preset, and control overrides.
pub fn render_wav_with_options(
    container_path: impl AsRef<Path>,
    input_wav_path: impl AsRef<Path>,
    output_wav_path: impl AsRef<Path>,
    tier_str: &str,
    preset_id: Option<&str>,
    control_overrides: &BTreeMap<String, String>,
) -> StudioResult<()> {
    render_wav_with_crypto(
        container_path,
        input_wav_path,
        output_wav_path,
        tier_str,
        preset_id,
        control_overrides,
        None,
        None,
        None,
    )
}

/// Renders an input WAV file through a `.dnam` or `.dnam-seal` container with optional device/marketplace keys.
#[allow(clippy::too_many_arguments)]
pub fn render_wav_with_crypto(
    container_path: impl AsRef<Path>,
    input_wav_path: impl AsRef<Path>,
    output_wav_path: impl AsRef<Path>,
    tier_str: &str,
    preset_id: Option<&str>,
    control_overrides: &BTreeMap<String, String>,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<()> {
    let container_bytes = fs::read(container_path)?;
    let crypto = build_crypto_provider(device_id, device_key_hex, marketplace_pubkey_hex)?;
    let view = DnamContainerView::from_bytes(&container_bytes, &crypto)
        .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
    let input = read_wav(input_wav_path)?;
    let tier = match tier_str.to_ascii_lowercase().as_str() {
        "lite" => Tier::Lite,
        "full" => Tier::Full,
        _ => Tier::Ultra,
    };
    let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend_and_crypto(
        &container_bytes,
        tier,
        input.sample_rate_hz,
        detect_best_backend(),
        &crypto,
    )
    .map_err(|e| StudioError::Engine(format!("{e:?}")))?;

    if let Some(pid) = preset_id
        && let Some(preset) = view
            .header
            .presets
            .iter()
            .find(|p| p.id == pid || p.name == pid)
    {
        for (idx, ctrl) in view.header.controls_schema.iter().enumerate() {
            if let Some(val) = preset.controls.get(&ctrl.id) {
                if let Some(num) = val.as_f64() {
                    engine.set_control_immediate(idx, num as f32);
                } else if let Some(s) = val.as_str()
                    && let Some(opts) = &ctrl.declared_options
                    && let Some(pos) = opts.iter().position(|o| o == s)
                {
                    engine.set_control_immediate(idx, pos as f32);
                }
            }
        }
    }

    for (idx, ctrl) in view.header.controls_schema.iter().enumerate() {
        if let Some(raw_val) = control_overrides.get(&ctrl.id) {
            if let Ok(num_val) = raw_val.parse::<f32>() {
                if let Some([lo, hi]) = ctrl.declared_range {
                    let span = (hi - lo).max(1e-6);
                    let norm = ((num_val - lo) / span).clamp(0.0, 1.0);
                    engine.set_control_immediate(idx, norm);
                } else {
                    engine.set_control_immediate(idx, num_val);
                }
            } else if let Some(opts) = &ctrl.declared_options
                && let Some(pos) = opts.iter().position(|o| o == raw_val)
            {
                engine.set_control_immediate(idx, pos as f32);
            }
        }
    }

    let num_frames = input.samples.len() / usize::from(input.channels.max(1));
    let mut mono_in = Vec::with_capacity(num_frames);
    if input.channels <= 1 {
        mono_in.extend_from_slice(&input.samples);
    } else {
        for chunk in input.samples.chunks_exact(usize::from(input.channels)) {
            mono_in.push(chunk[0]);
        }
    }

    let mut out_l = vec![0.0_f32; num_frames];
    let mut out_r = vec![0.0_f32; num_frames];
    engine.process_mono_to_stereo(&mono_in, &mut out_l, &mut out_r);

    let mut stereo_interleaved = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        stereo_interleaved.push(out_l[i]);
        stereo_interleaved.push(out_r[i]);
    }
    write_wav_32bit_float(
        output_wav_path,
        &stereo_interleaved,
        input.sample_rate_hz,
        2,
    )
}

/// Renders an input WAV file through a `.dnam` container using native hardware SIMD (`dnam-engine`).
pub fn render_wav_with_container(
    container_path: impl AsRef<Path>,
    input_wav_path: impl AsRef<Path>,
    output_wav_path: impl AsRef<Path>,
    normalized_controls: &[f32],
) -> StudioResult<()> {
    let container_bytes = fs::read(container_path)?;
    let input = read_wav(input_wav_path)?;
    let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend(
        &container_bytes,
        Tier::Full,
        input.sample_rate_hz,
        detect_best_backend(),
    )
    .map_err(|e| StudioError::Engine(format!("{e:?}")))?;

    for (idx, &val) in normalized_controls.iter().take(16).enumerate() {
        engine.set_control_immediate(idx, val);
    }

    let num_frames = input.samples.len() / usize::from(input.channels.max(1));
    let mut mono_in = Vec::with_capacity(num_frames);
    if input.channels <= 1 {
        mono_in.extend_from_slice(&input.samples);
    } else {
        for chunk in input.samples.chunks_exact(usize::from(input.channels)) {
            mono_in.push(chunk[0]);
        }
    }

    let mut out_l = vec![0.0_f32; num_frames];
    let mut out_r = vec![0.0_f32; num_frames];
    engine.process_mono_to_stereo(&mono_in, &mut out_l, &mut out_r);

    let mut stereo_interleaved = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        stereo_interleaved.push(out_l[i]);
        stereo_interleaved.push(out_r[i]);
    }
    write_wav_32bit_float(
        output_wav_path,
        &stereo_interleaved,
        input.sample_rate_hz,
        2,
    )
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamJsonHeader, PresetDef};

    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_runner_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn make_test_container(core2_active: bool, is_ckpt: bool) -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference(core2_active);
        if is_ckpt {
            header.container_kind = "dnam-ckpt".to_string();
        }
        let mut ctrl = serde_json::json!({
            "id": "gain",
            "name": "Gain",
            "type": "range",
            "declared_range": [0.0, 10.0],
            "safe_clamped_range": [0.0, 10.0],
            "default": 5.0
        });
        let c1: ControlSchemaDef = serde_json::from_value(ctrl.take()).expect("c1");
        let c2: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": "voice",
            "name": "Voice",
            "type": "switch",
            "declared_options": ["clean", "lead"],
            "active_options": ["clean", "lead"],
            "default": "clean"
        }))
        .expect("c2");
        header.controls_schema = vec![c1, c2];
        let mut preset_ctrls = BTreeMap::new();
        preset_ctrls.insert("gain".to_string(), serde_json::json!(6.0));
        preset_ctrls.insert("voice".to_string(), serde_json::json!("lead"));
        header.presets = vec![PresetDef {
            id: "lead_preset".to_string(),
            name: "Lead".to_string(),
            description: None,
            controls: preset_ctrls,
            user_metadata: BTreeMap::new(),
        }];
        assert_eq!(header.controls_schema[0].kind, ControlKind::Range);
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        build_dnam_container(&mut header, &arena, &arena, false).expect("build container")
    }

    #[test]
    fn test_inspect_and_render_with_dnam_engine() {
        let dir = temp_dir("inspect_render");
        let dnam_bytes = make_test_container(true, false);
        let ckpt_bytes = make_test_container(false, true);

        let dnam_path = dir.join("ref.dnam");
        let ckpt_path = dir.join("ref.dnam-ckpt");
        fs::write(&dnam_path, &dnam_bytes).expect("write ref.dnam");
        fs::write(&ckpt_path, &ckpt_bytes).expect("write ref.dnam-ckpt");

        let summary = inspect_container(&dnam_path).expect("inspect dnam");
        assert_eq!(summary.magic, "DNAM");
        assert!(!summary.is_checkpoint);
        assert!(summary.core2_active);
        assert_eq!(summary.sample_rate_train_hz, 48_000);
        assert!(!summary.detected_simd_backend.is_empty());

        let ckpt_summary = inspect_container(&ckpt_path).expect("inspect ckpt");
        assert!(ckpt_summary.is_checkpoint);

        assert!(inspect_container_bytes(b"bad_magic").is_err());

        let mono_in: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.05).sin() * 0.25).collect();
        let in_mono_path = dir.join("in_mono.wav");
        write_wav_32bit_float(&in_mono_path, &mono_in, 48_000, 1).expect("write mono wav");

        let stereo_in: Vec<f32> = (0..1024)
            .map(|i| ((i as f32) * 0.03).sin() * 0.25)
            .collect();
        let in_stereo_path = dir.join("in_stereo.wav");
        write_wav_32bit_float(&in_stereo_path, &stereo_in, 48_000, 2).expect("write stereo wav");

        let out_wav = dir.join("out.wav");
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(&dnam_bytes, &crypto).expect("parse view");
        let first_preset = view.header.presets.first().map(|p| p.id.as_str());
        let mut overrides = BTreeMap::new();
        for ctrl in &view.header.controls_schema {
            if ctrl.declared_range.is_some() {
                overrides.insert(ctrl.id.clone(), "5.0".to_string());
            } else if let Some(opts) = &ctrl.declared_options
                && let Some(first_opt) = opts.first()
            {
                overrides.insert(ctrl.id.clone(), first_opt.clone());
            }
        }

        render_wav_with_options(
            &dnam_path,
            &in_mono_path,
            &out_wav,
            "lite",
            first_preset,
            &overrides,
        )
        .expect("render lite");
        render_wav_with_options(
            &dnam_path,
            &in_stereo_path,
            &out_wav,
            "full",
            None,
            &overrides,
        )
        .expect("render full stereo");
        render_wav_with_options(
            &dnam_path,
            &in_mono_path,
            &out_wav,
            "ultra",
            None,
            &BTreeMap::new(),
        )
        .expect("render ultra");

        let decoded = read_wav(&out_wav).expect("read rendered wav");
        assert_eq!(decoded.sample_rate_hz, 48_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples.len(), 1024);
        assert!(decoded.samples.iter().all(|x| x.is_finite()));

        let out_wav2 = dir.join("out2.wav");
        render_wav_with_container(&dnam_path, &in_mono_path, &out_wav2, &[0.5, 0.5])
            .expect("render_wav_with_container");
        render_wav_with_container(&dnam_path, &in_stereo_path, &out_wav2, &[0.5])
            .expect("render_wav_with_container stereo");
        assert!(out_wav2.is_file());

        let _ = fs::remove_dir_all(&dir);
    }
}
