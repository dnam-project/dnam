//! Subsystem 3: Native SIMD `.dnam` / `.dnam-ckpt` Container Inspector & 32-Bit Float WAV Renderer (`dnam-engine`).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dnam_engine::container::{
    DNSL_MAGIC, DnamContainerView, FLAG_HAS_CKPT_STEMS, strip_container_to_tier,
};
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::{DynamicDnamEngine, Tier};
use dnam_trainer::ckpt::{
    DnamCkptView, export_runtime_dnam_from_ckpt, export_runtime_dnam_from_ckpt_with_crypto,
    strip_dnck_checkpoint_to_tier_with_crypto,
};
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
        BackendKind::ArmSve2 => "arm-sve2",
        BackendKind::Neon | BackendKind::ArmNeon => "arm64-neon",
        BackendKind::X86Simd | BackendKind::X86Avx2Fma => "x86-avx2-fma",
        BackendKind::X86Sse42 => "x86-sse4.2",
        BackendKind::X86Sse2 => "x86-sse2",
        BackendKind::WasmSimd128 => "wasm-simd128",
        BackendKind::Scalar => "scalar-reference",
    }
}

fn decode_hex_32(hex: &str) -> StudioResult<[u8; 32]> {
    let trimmed = hex.trim();
    if !trimmed.is_ascii() || trimmed.len() != 64 {
        return Err(StudioError::Validation(format!(
            "Expected 64 ASCII hex chars (32 bytes), got {} bytes",
            trimmed.len()
        )));
    }
    let mut out = [0_u8; 32];
    for (slot, chunk) in out.iter_mut().zip(trimmed.as_bytes().as_chunks::<2>().0) {
        let pair = std::str::from_utf8(chunk)
            .map_err(|_| StudioError::Validation("Invalid ASCII hex byte".to_string()))?;
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
    match (device_id, device_key_hex) {
        (Some(id), Some(key_hex)) => {
            let trimmed_id = id.trim();
            if trimmed_id.is_empty() {
                return Err(StudioError::Validation(
                    "Device ID must not be empty".to_string(),
                ));
            }
            let secret = decode_hex_32(key_hex)?;
            crypto = crypto.with_device_key(trimmed_id, secret);
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(StudioError::Validation(
                "Both --device-id and --device-key-hex must be provided together".to_string(),
            ));
        }
        (None, None) => {}
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
    if bytes.starts_with(b"DNCK") {
        let view = DnamCkptView::from_bytes(bytes, &crypto)
            .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
        let preamble = &view.preamble;
        let verif = format!("{:?}", view.verification_status);
        let profile_str = match view.protection_profile {
            dnam_engine::schema::DnamProtectionProfile::Open => "open",
            dnam_engine::schema::DnamProtectionProfile::DeviceAllowlisted => "device_allowlisted",
            dnam_engine::schema::DnamProtectionProfile::EnclaveSealed => "enclave_sealed",
        };
        return Ok(ContainerInspectionReport {
            magic: String::from_utf8_lossy(&preamble.magic).into_owned(),
            version_major: preamble.version_major,
            version_minor: preamble.version_minor,
            is_checkpoint: true,
            model_name: view.header.base.metadata.name.clone(),
            author: view
                .header
                .base
                .metadata
                .author
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            verification_status: verif,
            protection_profile: profile_str.to_string(),
            lfo_env_active: view.header.base.control_bus.envelopes > 0
                || view.header.base.control_bus.lfo_oscillators > 0,
            sample_rate_train_hz: preamble.sample_rate_train_hz,
            control_decimation: 16,
            detected_simd_backend: detect_native_backend_str().to_string(),
            num_controls: view.header.base.controls_schema.len(),
            num_ckpt_stems: preamble.num_ckpt_stems,
            head_svd_rank: view.header.base.slimmable_tiers.ultra.head_svd_rank,
            tail_time_ms: preamble.tail_time_ms,
            tensor_arena_byte_len: preamble.tensor_arena_byte_len,
            num_presets: view.header.base.presets.len(),
            byte_len: bytes.len(),
        });
    }
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
        is_checkpoint: preamble.magic == *b"DNCK" || (preamble.flags & FLAG_HAS_CKPT_STEMS) != 0,
        model_name: view.header.metadata.name.clone(),
        author: view
            .header
            .metadata
            .author
            .clone()
            .unwrap_or_else(|| "Unknown".to_string()),
        verification_status: verif,
        protection_profile: profile_str.to_string(),
        lfo_env_active: view.header.control_bus.envelopes > 0
            || view.header.control_bus.lfo_oscillators > 0,
        sample_rate_train_hz: preamble.sample_rate_train_hz,
        control_decimation: 16,
        detected_simd_backend: detect_native_backend_str().to_string(),
        num_controls: view.header.controls_schema.len(),
        num_ckpt_stems: preamble.num_ckpt_stems,
        head_svd_rank: view.header.slimmable_tiers.ultra.head_svd_rank,
        tail_time_ms: preamble.tail_time_ms,
        tensor_arena_byte_len: preamble.tensor_arena_byte_len,
        num_presets: view.header.presets.len(),
        byte_len: bytes.len(),
    })
}

fn parse_tier_str(tier_str: &str) -> StudioResult<Tier> {
    match tier_str.trim().to_ascii_lowercase().as_str() {
        "lite" | "0" => Ok(Tier::Lite),
        "full" | "1" => Ok(Tier::Full),
        "pro" | "2" => Ok(Tier::Pro),
        "ultra" | "3" => Ok(Tier::Ultra),
        other => Err(StudioError::Validation(format!(
            "Unknown tier '{other}': expected lite, full, pro, or ultra (0..3)"
        ))),
    }
}

/// Strips higher slimmable tiers from a `.dnam` or `.dnam-ckpt` container on disk down to `max_tier_str`,
/// verifying cryptographic integrity/signatures before and after stripping.
pub fn strip_container_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    max_tier_str: &str,
) -> StudioResult<usize> {
    strip_container_file_with_crypto(input_path, output_path, max_tier_str, None, None, None)
}

/// Strips higher slimmable tiers from a `.dnam` or `.dnam-ckpt` container on disk down to `max_tier_str`
/// with optional OEM device / marketplace cryptographic verification keys.
pub fn strip_container_file_with_crypto(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    max_tier_str: &str,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<usize> {
    let out_ref = output_path.as_ref();
    let bytes = fs::read(input_path)?;
    let max_tier = parse_tier_str(max_tier_str)?;
    let crypto = build_crypto_provider(device_id, device_key_hex, marketplace_pubkey_hex)?;
    if bytes.len() >= 4 && bytes[0..4] == DNSL_MAGIC {
        return Err(StudioError::Validation(
            "Cannot strip an EnclaveSealed (.dnam-seal / DNSL) container without re-sealing"
                .to_string(),
        ));
    }
    let wants_ckpt = out_ref
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("dnam-ckpt"));
    let stripped = if bytes.starts_with(b"DNCK") {
        let ckpt_view = DnamCkptView::from_bytes(&bytes, &crypto)
            .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
        let eff_tier = ckpt_view.preamble.highest_tier().min(max_tier);
        if wants_ckpt {
            let out = strip_dnck_checkpoint_to_tier_with_crypto(&bytes, eff_tier, &crypto)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
            DnamCkptView::from_bytes(&out, &crypto)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
            out
        } else {
            let dnam_bytes = export_runtime_dnam_from_ckpt_with_crypto(&bytes, &crypto)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
            let out = strip_container_to_tier(&dnam_bytes, eff_tier)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
            DnamContainerView::from_bytes(&out, &crypto)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
            out
        }
    } else {
        if wants_ckpt {
            return Err(StudioError::Validation(
                "Cannot write a .dnam-ckpt checkpoint from a runtime .dnam input container"
                    .to_string(),
            ));
        }
        let view = DnamContainerView::from_bytes(&bytes, &crypto)
            .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
        let eff_tier = view.preamble.highest_tier().min(max_tier);
        let out = strip_container_to_tier(&bytes, eff_tier)
            .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
        DnamContainerView::from_bytes(&out, &crypto)
            .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
        out
    };
    let len = stripped.len();
    if let Some(parent) = out_ref.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_ref, stripped)?;
    Ok(len)
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
    render_wav_with_bounce_and_crypto(
        container_path,
        input_wav_path,
        output_wav_path,
        tier_str,
        preset_id,
        control_overrides,
        true,
        true,
        device_id,
        device_key_hex,
        marketplace_pubkey_hex,
    )
}

/// Renders an input WAV file through a `.dnam` or `.dnam-seal` container with explicit
/// `offline_bounce` ($4\times$ ZOH oversampling) and `maximize_tier_on_bounce` flags.
#[allow(clippy::too_many_arguments)]
pub fn render_wav_with_bounce_and_crypto(
    container_path: impl AsRef<Path>,
    input_wav_path: impl AsRef<Path>,
    output_wav_path: impl AsRef<Path>,
    tier_str: &str,
    preset_id: Option<&str>,
    control_overrides: &BTreeMap<String, String>,
    offline_bounce: bool,
    maximize_tier_on_bounce: bool,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<()> {
    render_wav_with_bounce_and_crypto_inner(
        container_path.as_ref(),
        input_wav_path.as_ref(),
        output_wav_path.as_ref(),
        tier_str,
        preset_id,
        control_overrides,
        offline_bounce,
        maximize_tier_on_bounce,
        device_id,
        device_key_hex,
        marketplace_pubkey_hex,
    )
}

#[allow(clippy::too_many_arguments)]
fn render_wav_with_bounce_and_crypto_inner(
    container_path: &Path,
    input_wav_path: &Path,
    output_wav_path: &Path,
    tier_str: &str,
    preset_id: Option<&str>,
    control_overrides: &BTreeMap<String, String>,
    offline_bounce: bool,
    maximize_tier_on_bounce: bool,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<()> {
    let raw_bytes = fs::read(container_path)?;
    let crypto = build_crypto_provider(device_id, device_key_hex, marketplace_pubkey_hex)?;
    let container_bytes: Cow<'_, [u8]> = if raw_bytes.starts_with(b"DNCK") {
        Cow::Owned(
            export_runtime_dnam_from_ckpt_with_crypto(&raw_bytes, &crypto)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?,
        )
    } else {
        Cow::Borrowed(raw_bytes.as_slice())
    };
    let view = DnamContainerView::from_bytes(&container_bytes, &crypto)
        .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
    let input = read_wav(input_wav_path)?;
    let tier = parse_tier_str(tier_str)?;
    let mut engine = DynamicDnamEngine::from_bytes_owned_with_backend_and_crypto(
        &container_bytes,
        tier,
        input.sample_rate_hz,
        detect_best_backend(),
        &crypto,
    )
    .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
    engine.set_offline_bounce(offline_bounce);
    engine.set_maximize_tier_on_bounce(maximize_tier_on_bounce);

    if let Some(pid) = preset_id {
        let preset = view
            .header
            .presets
            .iter()
            .find(|p| p.id == pid || p.name == pid)
            .ok_or_else(|| {
                StudioError::Validation(format!("Preset '{pid}' not found in container"))
            })?;
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
            if let Some(opts) = &ctrl.declared_options {
                if let Some(pos) = opts.iter().position(|o| o == raw_val) {
                    engine.set_control_immediate(idx, pos as f32);
                } else if let Ok(pos) = raw_val.parse::<usize>()
                    && pos < opts.len()
                {
                    engine.set_control_immediate(idx, pos as f32);
                } else {
                    return Err(StudioError::Validation(format!(
                        "Invalid switch value '{raw_val}' for control '{}'; expected one of {opts:?} or index < {}",
                        ctrl.id,
                        opts.len()
                    )));
                }
            } else {
                let num_val = raw_val
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        StudioError::Validation(format!(
                            "Invalid finite range value '{raw_val}' for control '{}'",
                            ctrl.id
                        ))
                    })?;
                engine.set_control_immediate(idx, num_val);
            }
        }
    }

    let num_frames = input.samples.len() / usize::from(input.channels.max(1));
    let mut in_l = Vec::with_capacity(num_frames);
    let mut in_r = Vec::new();
    if input.channels <= 1 {
        in_l.extend_from_slice(&input.samples);
    } else {
        in_r.reserve(num_frames);
        for chunk in input.samples.chunks_exact(usize::from(input.channels)) {
            in_l.push(chunk[0]);
            in_r.push(chunk[1]);
        }
    }

    let mut out_l = vec![0.0_f32; num_frames];
    let mut out_r = vec![0.0_f32; num_frames];
    if offline_bounce {
        let right_ref = if input.channels >= 2 { &in_r } else { &in_l };
        engine.render_offline_bounce(&in_l, right_ref, &mut out_l, &mut out_r);
    } else if input.channels >= 2 {
        engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    } else {
        engine.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
    }

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

/// Renders an input WAV file through a `.dnam` or `.dnam-ckpt` container using native hardware SIMD (`dnam-engine`).
pub fn render_wav_with_container(
    container_path: impl AsRef<Path>,
    input_wav_path: impl AsRef<Path>,
    output_wav_path: impl AsRef<Path>,
    normalized_controls: &[f32],
) -> StudioResult<()> {
    let raw_bytes = fs::read(container_path)?;
    let container_bytes: Cow<'_, [u8]> = if raw_bytes.starts_with(b"DNCK") {
        Cow::Owned(
            export_runtime_dnam_from_ckpt(&raw_bytes)
                .map_err(|e| StudioError::Engine(format!("{e:?}")))?,
        )
    } else {
        Cow::Borrowed(raw_bytes.as_slice())
    };
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
    let mut in_l = Vec::with_capacity(num_frames);
    let mut in_r = Vec::new();
    if input.channels <= 1 {
        in_l.extend_from_slice(&input.samples);
    } else {
        in_r.reserve(num_frames);
        for chunk in input.samples.chunks_exact(usize::from(input.channels)) {
            in_l.push(chunk[0]);
            in_r.push(chunk[1]);
        }
    }

    let mut out_l = vec![0.0_f32; num_frames];
    let mut out_r = vec![0.0_f32; num_frames];
    if input.channels >= 2 {
        engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    } else {
        engine.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
    }

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
    use dnam_trainer::ckpt::{DnamCkptHeader, build_dnam_ckpt_container};

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

    fn make_test_container(is_ckpt: bool) -> Vec<u8> {
        let mut header = DnamJsonHeader::default_reference();
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
        if is_ckpt {
            header.container_kind = "dnam-ckpt".to_string();
            let mut ckpt_hdr = DnamCkptHeader {
                base: header,
                stems_registry: None,
                audit_report: None,
            };
            build_dnam_ckpt_container(&mut ckpt_hdr, &arena, &arena, false)
                .expect("build ckpt container")
        } else {
            build_dnam_container(&mut header, &arena, &arena, false).expect("build container")
        }
    }

    #[test]
    fn test_inspect_and_render_with_dnam_engine() {
        let dir = temp_dir("inspect_render");
        let dnam_bytes = make_test_container(false);
        let ckpt_bytes = make_test_container(true);

        let dnam_path = dir.join("ref.dnam");
        let ckpt_path = dir.join("ref.dnam-ckpt");
        fs::write(&dnam_path, &dnam_bytes).expect("write ref.dnam");
        fs::write(&ckpt_path, &ckpt_bytes).expect("write ref.dnam-ckpt");

        let summary = inspect_container(&dnam_path).expect("inspect dnam");
        assert_eq!(summary.magic, "DNAM");
        assert!(!summary.is_checkpoint);
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
            "pro",
            None,
            &BTreeMap::new(),
        )
        .expect("render pro");
        render_wav_with_options(
            &dnam_path,
            &in_mono_path,
            &out_wav,
            "ultra",
            None,
            &BTreeMap::new(),
        )
        .expect("render ultra");
        let stripped_path = dir.join("nested_strip").join("ref_pro.dnam");
        let stripped_len =
            strip_container_file(&dnam_path, &stripped_path, "pro").expect("strip to pro");
        assert!(stripped_len < dnam_bytes.len());
        let stripped_summary = inspect_container(&stripped_path).expect("inspect stripped");
        assert_eq!(stripped_summary.tensor_arena_byte_len, 36_224);

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

        // Exercise realtime (non-bounce) path + crypto hex validation & provider paths
        let hex32 = "aa".repeat(32);
        let rep_crypto =
            inspect_container_with_crypto(&dnam_path, Some("dev-1"), Some(&hex32), Some(&hex32))
                .expect("inspect with crypto");
        assert_eq!(rep_crypto.magic, "DNAM");
        assert!(inspect_container_with_crypto(&dnam_path, Some("d"), Some("short"), None).is_err());
        assert!(
            inspect_container_with_crypto(&dnam_path, Some("d"), Some(&"0éa".repeat(16)), None)
                .is_err()
        );
        assert!(
            inspect_container_with_crypto(&dnam_path, Some("d"), Some(&"zz".repeat(32)), None)
                .is_err()
        );
        render_wav_with_bounce_and_crypto(
            &dnam_path,
            &in_stereo_path,
            &out_wav2,
            "0",
            first_preset,
            &overrides,
            false,
            false,
            Some("dev-1"),
            Some(&hex32),
            Some(&hex32),
        )
        .expect("render non-bounce stereo with crypto");
        render_wav_with_bounce_and_crypto(
            &dnam_path,
            &in_mono_path,
            &out_wav2,
            "3",
            None,
            &overrides,
            false,
            false,
            None,
            None,
            None,
        )
        .expect("render non-bounce mono");

        // Unknown preset_id, unknown tier name, or .dnam -> .dnam-ckpt must return an explicit validation error
        assert!(strip_container_file(&dnam_path, &stripped_path, "ltie").is_err());
        assert!(strip_container_file(&dnam_path, dir.join("from_dnam.dnam-ckpt"), "pro").is_err());
        assert!(
            render_wav_with_options(
                &dnam_path,
                &in_mono_path,
                &out_wav2,
                "ltie",
                None,
                &overrides
            )
            .is_err()
        );
        assert!(
            render_wav_with_options(
                &dnam_path,
                &in_mono_path,
                &out_wav2,
                "lite",
                Some("nonexistent_preset"),
                &overrides,
            )
            .is_err()
        );
        for ctrl in &view.header.controls_schema {
            let mut bad_ov = BTreeMap::new();
            bad_ov.insert(ctrl.id.clone(), "NaN_or_nonexistent_opt".to_string());
            assert!(
                render_wav_with_options(
                    &dnam_path,
                    &in_mono_path,
                    &out_wav2,
                    "lite",
                    None,
                    &bad_ov
                )
                .is_err()
            );
        }

        // .dnam-ckpt (`DNCK`) support in render and strip (both to .dnam-ckpt and to .dnam)
        render_wav_with_options(
            &ckpt_path,
            &in_stereo_path,
            &out_wav2,
            "pro",
            first_preset,
            &overrides,
        )
        .expect("render from .dnam-ckpt");
        render_wav_with_container(&ckpt_path, &in_stereo_path, &out_wav2, &[0.5, 0.5])
            .expect("render_wav_with_container from .dnam-ckpt");
        let stripped_ckpt = dir.join("ref_pro.dnam-ckpt");
        strip_container_file(&ckpt_path, &stripped_ckpt, "pro").expect("strip ckpt to ckpt");
        assert!(inspect_container(&stripped_ckpt).unwrap().is_checkpoint);
        let stripped_dnam_from_ckpt = dir.join("from_ckpt_pro.dnam");
        strip_container_file(&ckpt_path, &stripped_dnam_from_ckpt, "pro")
            .expect("strip ckpt to runtime dnam");
        assert!(
            !inspect_container(&stripped_dnam_from_ckpt)
                .unwrap()
                .is_checkpoint
        );

        assert!(inspect_container_with_crypto(&dnam_path, Some("dev1"), None, None).is_err());
        assert!(
            inspect_container_with_crypto(&dnam_path, None, Some("11".repeat(32).as_str()), None)
                .is_err()
        );
        assert!(
            inspect_container_with_crypto(
                &dnam_path,
                Some("   "),
                Some("11".repeat(32).as_str()),
                None
            )
            .is_err()
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
