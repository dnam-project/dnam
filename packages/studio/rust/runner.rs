//! Subsystem 3: Native SIMD `.dnam` / `.dnam-ckpt` Container Inspector & 24-Bit WAV Renderer (`dnam-engine`).

use std::fs;
use std::path::Path;

use dnam_engine::container::DnamContainerView;
use dnam_engine::math::{BackendKind, detect_best_backend};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::{DynamicDnamEngine, Tier};
use serde::{Deserialize, Serialize};

use crate::capture::{read_wav, write_wav_24bit};
use crate::error::{StudioError, StudioResult};

/// Summary returned when inspecting a `.dnam` or `.dnam-ckpt` container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerInspectionReport {
    /// Container magic (`"DNAM"`).
    pub magic: String,
    /// Container format major version.
    pub version_major: u16,
    /// Container format minor version.
    pub version_minor: u16,
    /// Whether this container includes a `.dnam-ckpt` Kronecker stem expansion arena.
    pub is_checkpoint: bool,
    /// Training sample rate in Hz.
    pub sample_rate_train_hz: u32,
    /// Control-bus decimation factor (`16` -> $3\text{ kHz}$).
    pub control_decimation: u8,
    /// Active hardware SIMD acceleration backend (`"arm-sme"`, `"arm64-neon"`, `"x86-avx2-fma"`, `"scalar-reference"`).
    pub detected_simd_backend: String,
    /// Number of user-definable controls in the JSON header.
    pub num_controls: usize,
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

/// Inspects a `.dnam` or `.dnam-ckpt` file on disk using `dnam-engine`.
pub fn inspect_container(path: impl AsRef<Path>) -> StudioResult<ContainerInspectionReport> {
    let bytes = fs::read(path)?;
    inspect_container_bytes(&bytes)
}

/// Inspects in-memory `.dnam` or `.dnam-ckpt` bytes.
pub fn inspect_container_bytes(bytes: &[u8]) -> StudioResult<ContainerInspectionReport> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(bytes, &crypto)
        .map_err(|e| StudioError::Engine(format!("{e:?}")))?;
    let preamble = &view.preamble;

    Ok(ContainerInspectionReport {
        magic: "DNAM".to_string(),
        version_major: preamble.version_major,
        version_minor: preamble.version_minor,
        is_checkpoint: (preamble.flags & 0x01) != 0,
        sample_rate_train_hz: preamble.sample_rate_train_hz,
        control_decimation: preamble.control_decimation,
        detected_simd_backend: detect_native_backend_str().to_string(),
        num_controls: view.header.controls_schema.len(),
        num_presets: view.header.presets.len(),
        byte_len: bytes.len(),
    })
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
    write_wav_24bit(
        output_wav_path,
        &stereo_interleaved,
        input.sample_rate_hz,
        2,
    )
}
