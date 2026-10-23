//! `#![no_std]` + `alloc` JSON Schema, Control Descriptors, and Error Types (`schema.rs`).
//!
//! Parses and serializes the canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON headers (`0x40..`)
//! using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`).
//! Compiles human-readable `controls_schema` definitions into 64-byte binary
//! [`ControlClampDescriptor`] cache lines for zero-`FDIV`, zero-string-parsing $3\text{ kHz}$ execution.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Serialize};

use crate::control::{ConditionalClampRule, ControlClampDescriptor, MAX_CONTROLS};

/// Comprehensive error enumeration for container parsing, schema validation,
/// cryptographic verification, and runtime operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnamError {
    /// Input byte slice is shorter than the 64-byte binary preamble or declared arena offsets.
    BufferTooSmall {
        /// Expected minimum byte length.
        expected: usize,
        /// Actual byte length provided.
        actual: usize,
    },
    /// Binary preamble magic bytes are neither `b"DNAM"` nor `b"DNCK"`.
    InvalidMagic([u8; 4]),
    /// Unsupported binary container major/minor schema version.
    UnsupportedVersion {
        /// Major version number.
        major: u16,
        /// Minor version number.
        minor: u16,
    },
    /// Binary offset is not aligned to a 64-byte cache-line boundary (`offset & 0x3F != 0`).
    UnalignedOffset {
        /// Field name that violated 64-byte alignment.
        field: &'static str,
        /// Unaligned byte offset.
        offset: u32,
    },
    /// Castagnoli `CRC32C` checksum mismatch on the JSON header or tier arena prefix.
    Crc32cMismatch {
        /// Region name (`"header"`, `"arena_lite"`, `"arena_full"`, `"arena_ultra"`, or `"arena3_stem"`).
        region: &'static str,
        /// Expected CRC32C stored in container.
        expected: u32,
        /// Computed CRC32C over byte slice.
        computed: u32,
    },
    /// UTF-8 decoding or `serde_json` parsing error in the JSON header.
    InvalidJsonHeader(String),
    /// Schema invariant or safety bound violation in the JSON header.
    SchemaValidationError(String),
    /// Updated JSON header exceeds the pre-allocated 4 KB page budget (`json_capacity_bytes`)
    /// for zero-copy in-place header updates.
    HeaderCapacityExceeded {
        /// Required serialized JSON byte length.
        required: usize,
        /// Available capacity before `arena_offset_start`.
        capacity: usize,
    },
    /// Cryptographic signature, digest, or device authorization failure.
    CryptoVerificationFailed(&'static str),
    /// Attempted to load an encrypted arena (`ENCRYPTED_ARENA == 1`) without a valid
    /// device key or `MarketplaceReceipt`.
    EncryptedArenaRequiresDeviceKey,
    /// Invalid sample rate outside $[44100.0, 96000.0]\text{ Hz}$.
    InvalidSampleRate(u32),
    /// Requested control index or option is out of bounds.
    InvalidControlIndex(usize),
}

impl fmt::Display for DnamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { expected, actual } => {
                write!(
                    f,
                    "buffer too small: expected at least {expected} bytes, got {actual}"
                )
            }
            Self::InvalidMagic(magic) => {
                write!(f, "invalid container magic bytes: {magic:?}")
            }
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported container version: {major}.{minor}")
            }
            Self::UnalignedOffset { field, offset } => {
                write!(
                    f,
                    "unaligned 64-byte offset for {field}: {offset:#x} (rem {})",
                    offset & 0x3F
                )
            }
            Self::Crc32cMismatch {
                region,
                expected,
                computed,
            } => {
                write!(
                    f,
                    "CRC32C mismatch in {region}: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::InvalidJsonHeader(msg) => write!(f, "invalid JSON header: {msg}"),
            Self::SchemaValidationError(msg) => write!(f, "schema validation error: {msg}"),
            Self::HeaderCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "in-place JSON header update requires {required} bytes, exceeding capacity {capacity} bytes"
                )
            }
            Self::CryptoVerificationFailed(msg) => {
                write!(f, "cryptographic verification failed: {msg}")
            }
            Self::EncryptedArenaRequiresDeviceKey => {
                write!(
                    f,
                    "encrypted tensor arena (ENCRYPTED_ARENA=1) requires an authorized device key"
                )
            }
            Self::InvalidSampleRate(sr) => {
                write!(
                    f,
                    "unsupported sample rate {sr} Hz (expected 44100..=96000 Hz)"
                )
            }
            Self::InvalidControlIndex(idx) => {
                write!(f, "invalid control index: {idx}")
            }
        }
    }
}

impl core::error::Error for DnamError {}

// ============================================================================
// Pure `#![no_std]` IEEE-754 `f64` Cold-Init Transcendentals (Zero `libm`/`std`)
// ============================================================================

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` for cold-init ZOH rescaling and UI skew.
/// Never called on the real-time audio or $3\text{ kHz}$ control thread.
#[inline]
pub fn cold_ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp_unbiased = ((bits >> 52) & 0x7FF) as i64 - 1023;
    // Normalize mantissa m into [1.0, 2.0)
    let m_bits = (bits & 0x000F_FFFF_FFFF_FFFF) | (1023_u64 << 52);
    let m = f64::from_bits(m_bits);
    // Fold m into [1/sqrt(2), sqrt(2)]
    let (m_adj, exp_adj) = if m > core::f64::consts::SQRT_2 {
        (m * 0.5, exp_unbiased + 1)
    } else {
        (m, exp_unbiased)
    };
    let t = (m_adj - 1.0) / (m_adj + 1.0);
    let t2 = t * t;
    let mut term = t;
    let mut sum = t;
    let mut k = 3;
    while k <= 21 {
        term *= t2;
        sum += term / (k as f64);
        k += 2;
    }
    (exp_adj as f64) * core::f64::consts::LN_2 + 2.0 * sum
}

/// Pure `#![no_std]` exponential $\exp(y)$ in `f64` for cold-init ZOH rescaling and UI skew.
/// Never called on the real-time audio or $3\text{ kHz}$ control thread.
#[inline]
pub fn cold_exp_f64(y: f64) -> f64 {
    if y > 709.0 {
        return f64::INFINITY;
    }
    if y < -709.0 {
        return 0.0;
    }
    let inv_ln2 = core::f64::consts::LOG2_E;
    let k = (y * inv_ln2 + if y >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let r = y - (k as f64) * core::f64::consts::LN_2;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut n = 1;
    while n <= 14 {
        term *= r / (n as f64);
        sum += term;
        n += 1;
    }
    let scale_bits = ((k + 1023) as u64) << 52;
    sum * f64::from_bits(scale_bits)
}

/// Pure `#![no_std]` power function $a^b = \exp(b \ln(a))$ in `f64` for cold-init ZOH rescaling and UI skew.
#[inline]
pub fn cold_pow_f64(base: f64, exp: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    if (exp - 1.0).abs() < 1e-15 {
        return base;
    }
    if exp.abs() < 1e-15 {
        return 1.0;
    }
    cold_exp_f64(exp * cold_ln_f64(base))
}

/// Pure `#![no_std]` square root $\sqrt{x} = \exp(0.5 \ln(x))$ in `f64` for cold-init $\mathcal{H}_2$ normalizers,
/// avoiding emitting hardware `FSQRT` instructions anywhere in the crate binary.
#[inline]
pub fn cold_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    cold_exp_f64(0.5 * cold_ln_f64(x))
}

// ============================================================================
// Canonical `.dnam` and `.dnam-ckpt` JSON Header Structs
// ============================================================================

/// Default schema version string (`"1.0.0"`).
fn default_version() -> String {
    "1.0.0".to_string()
}

/// Default container kind (`"dnam"`).
fn default_container_kind() -> String {
    "dnam".to_string()
}

/// Default architecture identifier (`"DNAM_TriPhysics"`).
fn default_architecture() -> String {
    "DNAM_TriPhysics".to_string()
}

/// Default training sample rate (`48000`).
const fn default_sample_rate() -> u32 {
    48_000
}

/// Top-level UTF-8 JSON header located at byte offset `0x40` of a `.dnam` or `.dnam-ckpt` container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DnamJsonHeader {
    /// Semantic version of the container schema (`"1.0.0"`).
    #[serde(default = "default_version")]
    pub version: String,
    /// Container kind: `"dnam"` or `"dnam-ckpt"`.
    #[serde(default = "default_container_kind")]
    pub container_kind: String,
    /// Architecture identifier (`"DNAM_TriPhysics"`).
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Native training sample rate in Hz (typically `48000`).
    #[serde(default = "default_sample_rate")]
    pub sample_rate_train_hz: u32,
    /// Essential model identity, calibration, provenance, and split-payload signatures.
    pub metadata: ModelMetadata,
    /// Open dictionary for storefront/host/user metadata (excluded from `immutable_header_sha256`).
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Audio I/O topology, latency, and tail time configuration.
    #[serde(default)]
    pub io_config: IoConfig,
    /// Slimmable tier configurations (`lite`, `full`, `ultra`).
    pub slimmable_tiers: SlimmableTiersConfig,
    /// Parametric `range` and `switch` control definitions (`0..=16` controls).
    #[serde(default)]
    pub controls_schema: Vec<ControlSchemaDef>,
    /// Built-in parametric control snapshots (excluded from `immutable_header_sha256`).
    #[serde(default)]
    pub presets: Vec<PresetDef>,
    /// Control-bus hyperparameters.
    #[serde(default)]
    pub control_bus: ControlBusConfig,
    /// Core 1 Hybrid Dilated-Conv + `PolyADAA` + `SelectiveRgLru` architecture parameters.
    #[serde(default)]
    pub core1_hybrid_ssm: Core1Config,
    /// Core 2 Diffusive Modulated Unitary FDN (`M-UFDN`) parameters.
    #[serde(default)]
    pub core2_mufdn: Core2Config,
    /// 64-byte aligned tensor directory mapping named slices in `Arena 0..2`.
    #[serde(default)]
    pub tensor_directory: BTreeMap<String, TensorSliceEntry>,
    /// `.dnam-ckpt` only: per-stem provenance, extracted physics, and `Arena 3` `uplo='U'` records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stems_registry: Option<Vec<StemRegistryEntry>>,
    /// `.dnam-ckpt` only: 3-phase control clamp audit report and actionable prescriptions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_report: Option<AuditReport>,
}

/// Essential model identity, calibration, dataset provenance, and cryptographic signatures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Human-readable model name.
    pub name: String,
    /// Optional creator / studio name.
    #[serde(default)]
    pub author: Option<String>,
    /// Optional description of the modeled hardware gear.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional SPDX or custom license identifier.
    #[serde(default)]
    pub license: Option<String>,
    /// ISO-8601 UTC creation timestamp.
    pub created_at: String,
    /// Reference input calibration level in dBFS (`1 dBu = 0.869 Vrms = -20.5 dBFS` default).
    #[serde(default = "default_input_ref_dbfs")]
    pub input_reference_dbfs: f32,
    /// Content-addressable training dataset provenance.
    #[serde(default)]
    pub provenance: DatasetProvenance,
    /// Optional split-payload `Ed25519` creator signature and marketplace receipt.
    #[serde(default)]
    pub signatures: Option<ModelSignatures>,
}

const fn default_input_ref_dbfs() -> f32 {
    -20.5
}

/// Content-addressable dataset and recipe provenance.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetProvenance {
    /// SHA-256 hex digest over sorted `(stem_id, input_pcm_sha256, target_pcm_sha256, controls_meta)`.
    #[serde(default)]
    pub dataset_manifest_sha256: String,
    /// Optional SHA-256 hex digest of the training recipe.
    #[serde(default)]
    pub config_recipe_sha256: Option<String>,
    /// Total training epochs completed.
    #[serde(default)]
    pub epochs_trained: u32,
}

/// Split-payload `Ed25519` signature envelope (`HAS_SIGNATURES` bit 9).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSignatures {
    /// Hex SHA-256 digest of the RFC 8785 JCS canonical JSON header excluding mutable fields.
    pub immutable_header_sha256: String,
    /// Hex SHA-256 digest of the creator's un-watermarked master `Arena 0..2` (`36,224 B`).
    pub master_arena_sha256: String,
    /// Creator's offline `Ed25519` signature over `immutable_header_sha256 || master_arena_sha256`.
    pub creator_signature: CreatorSignature,
    /// Optional marketplace counter-signature binding `master_arena_sha256 -> buyer_arena_sha256`.
    #[serde(default)]
    pub marketplace_receipt: Option<MarketplaceReceipt>,
}

/// Creator's `Ed25519` signature block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreatorSignature {
    /// Creator identifier or `did:web` URI.
    pub signer_id: String,
    /// 32-byte `Ed25519` public key encoded as 64 lowercase hex characters.
    pub pubkey_ed25519: String,
    /// 64-byte `Ed25519` signature encoded as 128 lowercase hex characters.
    pub sig_ed25519: String,
}

/// Marketplace download receipt (`< 0.1 ms` edge personalization & OEM hardware authorization).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarketplaceReceipt {
    /// Marketplace domain identifier (e.g., `"dnam.store"`).
    pub marketplace_id: String,
    /// Unique UUIDv7 license identifier.
    pub license_id: String,
    /// Hex SHA-256 digest of the buyer account identity.
    pub buyer_hash: String,
    /// License tier (e.g., `"personal"`, `"commercial"`, `"oem_bundle"`).
    pub license_tier: String,
    /// ISO-8601 UTC timestamp of receipt issuance.
    pub issued_at: String,
    /// Hex SHA-256 digest of the personalized buyer `Arena 0..2` (`36,224 B`, pre-encryption).
    pub buyer_arena_sha256: String,
    /// Arena encryption mode: `"none"` (`ENCRYPTED_ARENA = 0`) or `"x25519_chacha20_arena"` (`ENCRYPTED_ARENA = 1`).
    #[serde(default = "default_encryption_mode")]
    pub encryption_mode: String,
    /// 96-bit (`12 B`, 24 hex chars) ChaCha20 nonce when `encryption_mode == "x25519_chacha20_arena"`.
    #[serde(default)]
    pub arena_nonce_96: Option<String>,
    /// 32-byte (`64` hex chars) ephemeral `X25519` public key when `encryption_mode == "x25519_chacha20_arena"`.
    #[serde(default)]
    pub ephemeral_pubkey_x25519: Option<String>,
    /// Authorized OEM hardware devices and optional wrapped Content Encryption Keys (`wrapped_cek`).
    #[serde(default)]
    pub authorized_devices: Vec<AuthorizedDevice>,
    /// Optional multi-creator royalty split records (`sum(share_bps) == 10000`).
    #[serde(default)]
    pub royalty_splits: Vec<RoyaltySplit>,
    /// 64-byte `Ed25519` marketplace signature encoded as 128 lowercase hex characters.
    pub receipt_sig_ed25519: String,
}

fn default_encryption_mode() -> String {
    "none".to_string()
}

/// Authorized hardware device record inside `marketplace_receipt.authorized_devices`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorizedDevice {
    /// OEM vendor identifier (e.g., `"darkglass"`, `"apple"`, `"microsoft"`).
    pub oem_vendor: String,
    /// Hardware model code (e.g., `"anagram_rk3582"`, `"mac_secure_enclave"`, `"tpm2_pcr07"`).
    pub model_code: String,
    /// Hex digest of the hardware device unique identifier.
    pub device_id_hash: String,
    /// Optional 48-byte (`96` hex chars) `ChaCha20-Poly1305` wrapped CEK (`32 B` ciphertext + `16 B` tag).
    #[serde(default)]
    pub wrapped_cek: Option<String>,
}

/// Multi-creator royalty split entry (`10000 bps = 100.00%`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoyaltySplit {
    /// Recipient identifier or `did:web` URI.
    pub recipient_id: String,
    /// Creator role (e.g., `"amp_capture"`, `"ir_cabinet"`, `"preset_designer"`).
    pub role: String,
    /// Share in basis points (`0..=10000`).
    pub share_bps: u16,
}

/// Audio I/O topology and host PDC / tail configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IoConfig {
    /// Number of input channels (`1` or `2`).
    #[serde(default = "default_inputs")]
    pub inputs: u8,
    /// Output topology mode (`"stereo"` or `"mono"`).
    #[serde(default = "default_output_mode")]
    pub output_mode: String,
    /// Algorithmic lookahead latency in samples (always `0` for DNAM).
    #[serde(default)]
    pub latency_samples: u32,
    /// Estimated $T_{60}$ reverb/delay tail time in milliseconds for DAW sleep management.
    #[serde(default)]
    pub tail_time_ms: f32,
}

const fn default_inputs() -> u8 {
    2
}

fn default_output_mode() -> String {
    "stereo".to_string()
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            inputs: 2,
            output_mode: default_output_mode(),
            latency_samples: 0,
            tail_time_ms: 0.0,
        }
    }
}

/// Slimmable tier profiles (`lite`, `full`, `ultra`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlimmableTiersConfig {
    /// `Lite` ($C=4$, $M=4$) tier profile.
    pub lite: TierProfile,
    /// `Full` ($C=8$, $M=8$) tier profile.
    pub full: TierProfile,
    /// `Ultra` ($C=12$, $M=8$) tier profile.
    pub ultra: TierProfile,
}

/// Metadata and prefix checksum for an individual slimmable tier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TierProfile {
    /// Active Core 1 channels ($C \in \{4, 8, 12\}$).
    pub channels: u8,
    /// Active Core 2 FDN delay lines ($M \in \{0, 4, 8\}$).
    pub fdn_lines: u8,
    /// Active early-reflection taps ($M_{\text{early}} \in \{0, 4, 8\}$).
    #[serde(default)]
    pub early_taps: u8,
    /// SVD output head rank ($D_h = 4$).
    #[serde(default = "default_head_svd_rank")]
    pub head_svd_rank: u8,
    /// Integrated loudness in LUFS (`-20.5` target).
    #[serde(default = "default_input_ref_dbfs")]
    pub loudness_lufs: f32,
    /// Linear output loudness trim in dB (`0.0` default).
    #[serde(default)]
    pub output_trim_db: f32,
    /// Validation Error-to-Signal Ratio (ESR) in dB.
    #[serde(default)]
    pub validation_esr_db: f32,
    /// End byte offset of this tier's cumulative prefix relative to `arena_offset_start`.
    pub arena_end_byte_offset: u32,
    /// Castagnoli `CRC32C` over `arena[0 .. arena_end_byte_offset]`.
    pub prefix_crc32c: u32,
}

const fn default_head_svd_rank() -> u8 {
    4
}

/// Parametric control kind (`"range"` vs `"switch"`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ControlKind {
    /// Continuous knob / slider mapped via Node-Anchored Lobatto polynomials.
    Range,
    /// Discrete selector / toggle switch.
    Switch,
}

/// Schema definition for a single parametric control (`range` or `switch`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlSchemaDef {
    /// Machine-readable control identifier (e.g., `"gain"`, `"feedback"`, `"boost"`).
    pub id: String,
    /// Human-readable display label (e.g., `"Drive"`, `"Repeats"`).
    pub name: String,
    /// Control kind (`"range"` or `"switch"`).
    #[serde(rename = "type")]
    pub kind: ControlKind,
    /// Optional physical unit string (e.g., `"dB"`, `"%"`, `"Hz"`).
    #[serde(default)]
    pub unit: Option<String>,
    /// `range` only: Node-Anchored Lobatto polynomial degree $K_p \in \{0, 1, 2, 3\}$.
    #[serde(default)]
    pub lobatto_degree: Option<u8>,
    /// `range` only: Normalized Newton/Lobatto anchor nodes $[u_{p,1}, u_{p,2}, u_{p,3}]$ in $[-1, +1]$.
    #[serde(default)]
    pub anchor_nodes_norm: Option<Vec<f32>>,
    /// `range` only: Full physical dial range $[x_{\text{decl\_min}}, x_{\text{decl\_max}}]$.
    #[serde(default)]
    pub declared_range: Option<[f32; 2]>,
    /// `range` only: Phase-1 `[CLAMP-RANGE]` safe observed interval $[x_{\text{safe\_min}}, x_{\text{safe\_max}}]$.
    #[serde(default)]
    pub safe_clamped_range: Option<[f32; 2]>,
    /// Default value (a JSON number for `range` controls or a JSON string for `switch` controls).
    pub default: serde_json::Value,
    /// `range` only: Switch-conditioned minimum bounds (`"switch_id=option_id" -> min_val`).
    #[serde(default)]
    pub conditional_min: BTreeMap<String, f32>,
    /// `range` only: Switch-conditioned maximum bounds (`"switch_id=option_id" -> max_val`).
    #[serde(default)]
    pub conditional_max: BTreeMap<String, f32>,
    /// `switch` only: All declared switch option IDs (`1..=16`).
    #[serde(default)]
    pub declared_options: Option<Vec<String>>,
    /// `switch` only: Observed/active switch option IDs enabled by `[CLAMP-SWITCH]`.
    #[serde(default)]
    pub active_options: Option<Vec<String>>,
    /// Optional human-readable clamp badge warning (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, etc.).
    #[serde(default)]
    pub clamp_warning: Option<String>,
    /// Intrinsic visual and interaction hints for plugin/web/hardware UI rendering.
    #[serde(default)]
    pub ui_hints: ControlUiHints,
    /// Open dictionary for control-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
}

/// Intrinsic visual and interaction hints for a parametric control (`ui_hints`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ControlUiHints {
    /// Logical rack/pedal section name (e.g., `"Preamp"`, `"Tape Echo"`).
    #[serde(default)]
    pub section: Option<String>,
    /// Ordering index within `section`.
    #[serde(default)]
    pub order: i32,
    /// Widget archetype (`"knob"`, `"slider"`, `"trim"`, `"toggle"`, `"segmented"`, `"rotary_select"`, `"footswitch"`).
    #[serde(default)]
    pub widget: Option<String>,
    /// Relative visual size (`"sm"`, `"md"`, `"lg"`).
    #[serde(default)]
    pub size: Option<String>,
    /// Visual fill polarity (`"unipolar"` or `"bipolar"`).
    #[serde(default)]
    pub polarity: Option<String>,
    /// Physical value at $50\%$ ($12\text{ o'clock}$, $t = 0.5$) UI gesture position.
    #[serde(default)]
    pub midpoint: Option<f32>,
    /// Physical click/detent snap positions.
    #[serde(default)]
    pub detents: Vec<f32>,
    /// Optional UI step increment.
    #[serde(default)]
    pub step: Option<f32>,
    /// Numeric display precision in decimal places.
    #[serde(default)]
    pub decimals: Option<u8>,
    /// Landmark scale labels (`{"0.0": "Clean", "10.0": "Saturated"}`).
    #[serde(default)]
    pub scale_labels: BTreeMap<String, String>,
    /// Switch option display labels (`{"off": "Bypass", "on": "Vintage +6 dB"}`).
    #[serde(default)]
    pub option_labels: BTreeMap<String, String>,
    /// Optional CSS hex accent color (e.g., `"#F59E0B"`).
    #[serde(default)]
    pub color_hex: Option<String>,
}

impl ControlUiHints {
    /// Computes the closed-form power-skew exponent $\gamma_{\text{skew}}$ for `[x_min, x_max]`
    /// anchored at `midpoint`:
    /// $$\gamma_{\text{skew}} = \frac{\ln(0.5)}{\ln\!\left(\frac{x_{\text{mid}} - x_{\min}}{x_{\max} - x_{\min}}\right)}$$
    #[inline]
    pub fn skew_exponent(&self, x_min: f32, x_max: f32) -> f64 {
        let span = (x_max - x_min) as f64;
        if span <= 1e-9 {
            return 1.0;
        }
        if let Some(mid) = self.midpoint {
            let ratio = ((mid - x_min) as f64) / span;
            if ratio > 1e-4 && ratio < (1.0 - 1e-4) && (ratio - 0.5).abs() > 1e-6 {
                return cold_ln_f64(0.5) / cold_ln_f64(ratio);
            }
        }
        1.0
    }

    /// Maps a normalized host/UI gesture coordinate $t \in [0.0, 1.0]$ to physical parameter value $x \in [x_{\min}, x_{\max}]$:
    /// $$x(t) = x_{\min} + (x_{\max} - x_{\min})\, t^{1 / \gamma_{\text{skew}}}$$
    #[inline]
    pub fn normalized_to_physical(&self, t: f32, x_min: f32, x_max: f32) -> f32 {
        let t_clamped = t.clamp(0.0, 1.0) as f64;
        if t_clamped <= 0.0 {
            return x_min;
        }
        if t_clamped >= 1.0 {
            return x_max;
        }
        let gamma = self.skew_exponent(x_min, x_max);
        let curved = cold_pow_f64(t_clamped, 1.0 / gamma);
        x_min + ((x_max - x_min) as f64 * curved) as f32
    }

    /// Maps a physical parameter value $x \in [x_{\min}, x_{\max}]$ to normalized host/UI gesture coordinate $t \in [0.0, 1.0]$:
    /// $$t(x) = \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right)^{\gamma_{\text{skew}}}$$
    #[inline]
    pub fn physical_to_normalized(&self, x: f32, x_min: f32, x_max: f32) -> f32 {
        let span = (x_max - x_min) as f64;
        if span <= 1e-9 {
            return 0.0;
        }
        let ratio = (((x - x_min) as f64) / span).clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return 0.0;
        }
        if ratio >= 1.0 {
            return 1.0;
        }
        let gamma = self.skew_exponent(x_min, x_max);
        cold_pow_f64(ratio, gamma) as f32
    }
}

/// Built-in preset snapshot (`presets[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresetDef {
    /// Machine-readable preset ID (e.g., `"warm_slapback"`).
    pub id: String,
    /// Human-readable preset display name.
    pub name: String,
    /// Optional preset description.
    #[serde(default)]
    pub description: Option<String>,
    /// Control value assignments (`"control_id" -> number | string`).
    pub controls: BTreeMap<String, serde_json::Value>,
    /// Open dictionary for preset-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
}

/// Control-Bus configuration (`control_bus`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlBusConfig {
    /// Control-bus decimation stride at $f_{s,\text{train}}$ (`16` at $48\text{ kHz}$ $\to 3\text{ kHz}$).
    #[serde(default = "default_decimation")]
    pub decimation: u16,
    /// Number of envelope follower bands (`3`).
    #[serde(default = "default_envelopes")]
    pub envelopes: u8,
    /// Number of LFO oscillators (`2`).
    #[serde(default = "default_lfos")]
    pub lfo_oscillators: u8,
    /// Number of Chebyshev harmonics per LFO (`4`).
    #[serde(default = "default_lfo_harmonics")]
    pub lfo_harmonics: u8,
    /// Factorization Machine (FM) interaction rank ($R = 4$).
    #[serde(default = "default_fm_rank")]
    pub fm_interaction_rank: u8,
    /// Trust-region soft-clip radius $M_h$ (`CL-15`, default `2.5`).
    #[serde(default = "default_trust_radius")]
    pub head_trust_region_radius: f32,
}

const fn default_decimation() -> u16 {
    16
}
const fn default_envelopes() -> u8 {
    3
}
const fn default_lfos() -> u8 {
    2
}
const fn default_lfo_harmonics() -> u8 {
    4
}
const fn default_fm_rank() -> u8 {
    4
}
const fn default_trust_radius() -> f32 {
    2.5
}

impl Default for ControlBusConfig {
    fn default() -> Self {
        Self {
            decimation: 16,
            envelopes: 3,
            lfo_oscillators: 2,
            lfo_harmonics: 4,
            fm_interaction_rank: 4,
            head_trust_region_radius: 2.5,
        }
    }
}

/// Core 1 Hybrid SSM configuration (`core1_hybrid_ssm`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Core1Config {
    /// Number of interleaved Core 1 blocks (`4`).
    #[serde(default = "default_blocks")]
    pub blocks: u8,
    /// Causal dilated convolution kernel width ($K = 6$).
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: u8,
    /// Block dilation schedule (`[1, 3, 7, 19]`).
    #[serde(default = "default_dilations")]
    pub dilations: [u16; 4],
    /// Anti-aliasing mode (`"1x_bspline_algebraic"`).
    #[serde(default = "default_adaa_mode")]
    pub adaa_mode: String,
    /// $5\text{ Hz}$ AC-coupling pole at $f_{s,\text{train}}$ (`0.9993`).
    #[serde(default = "default_ac_pole")]
    pub ac_coupling_pole: f32,
    /// Memory layout tag (`"tier_delta_exec_order_soa_align64"`).
    #[serde(default = "default_weight_layout")]
    pub weight_layout: String,
}

const fn default_blocks() -> u8 {
    4
}
const fn default_conv_kernel() -> u8 {
    6
}
const fn default_dilations() -> [u16; 4] {
    [1, 3, 7, 19]
}
fn default_adaa_mode() -> String {
    "1x_bspline_algebraic".to_string()
}
const fn default_ac_pole() -> f32 {
    0.9993
}
fn default_weight_layout() -> String {
    "tier_delta_exec_order_soa_align64".to_string()
}

impl Default for Core1Config {
    fn default() -> Self {
        Self {
            blocks: 4,
            conv_kernel: 6,
            dilations: [1, 3, 7, 19],
            adaa_mode: default_adaa_mode(),
            ac_coupling_pole: 0.9993,
            weight_layout: default_weight_layout(),
        }
    }
}

/// Core 2 Diffusive Modulated Unitary FDN configuration (`core2_mufdn`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Core2Config {
    /// Whether Core 2 is active (`CORE2_ACTIVE` bit 6).
    #[serde(default)]
    pub active: bool,
    /// Delay buffer storage format (`"fp16"`).
    #[serde(default = "default_delay_storage")]
    pub delay_storage: String,
    /// $\log_2(N_{\text{ring}})$ per FDN delay line (`0` when inactive, `10..=15` when active).
    #[serde(default)]
    pub ring_buffer_pow2_log2: u8,
    /// Fractional delay interpolation kernel (`"cubic_lagrange_4tap"`).
    #[serde(default = "default_interp")]
    pub interp: String,
    /// Through-Zero Flanging (TZF) dry reference tap in samples (`64.0` at $48\text{ kHz}$).
    #[serde(default = "default_tzf_tap")]
    pub tzf_reference_tap_samples: f32,
    /// Raised-cosine switch crossfade window in samples (`128` at $48\text{ kHz}$).
    #[serde(default = "default_switch_xfade")]
    pub switch_xfade_samples: u16,
    /// Coprime Schroeder all-pass delay lengths in samples (`[19, 31, 47, 73, 103, 139, 179, 223]`).
    #[serde(default = "default_allpass_delays")]
    pub allpass_delays_samples: [u16; 8],
    /// In-loop $5\text{ Hz}$ DC blocker pole (`0.995` at $48\text{ kHz}$).
    #[serde(default = "default_dc_pole")]
    pub dc_blocker_pole: f32,
    /// Global loop ceiling (`0.985`).
    #[serde(default = "default_loop_ceiling")]
    pub loop_ceiling: f32,
}

fn default_delay_storage() -> String {
    "fp16".to_string()
}
fn default_interp() -> String {
    "cubic_lagrange_4tap".to_string()
}
const fn default_tzf_tap() -> f32 {
    64.0
}
const fn default_switch_xfade() -> u16 {
    128
}
const fn default_allpass_delays() -> [u16; 8] {
    [19, 31, 47, 73, 103, 139, 179, 223]
}
const fn default_dc_pole() -> f32 {
    0.995
}
const fn default_loop_ceiling() -> f32 {
    0.985
}

impl Default for Core2Config {
    fn default() -> Self {
        Self {
            active: false,
            delay_storage: default_delay_storage(),
            ring_buffer_pow2_log2: 0,
            interp: default_interp(),
            tzf_reference_tap_samples: 64.0,
            switch_xfade_samples: 128,
            allpass_delays_samples: default_allpass_delays(),
            dc_blocker_pole: 0.995,
            loop_ceiling: 0.985,
        }
    }
}

/// Tensor directory entry describing a 64-byte aligned slice in `Arena 0..3`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorSliceEntry {
    /// Arena index (`0` = Lite, `1` = Full Delta, `2` = Ultra Delta, `3` = `.dnam-ckpt` Gram stats).
    pub arena: u8,
    /// Data type (`"f32"` or `"f64"`).
    pub dtype: String,
    /// Optional packing tag (`"uplo_upper_col_major_tier_ordered"` for `Arena 3` Gram matrices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packing: Option<String>,
    /// Optional feature dimension $N \in \{64, 128, 192\}$ for `uplo='U'` packed Gram matrices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_dim: Option<u32>,
    /// Optional packed element count $\frac{N(N+1)}{2}$.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_len: Option<u32>,
    /// Tensor dimensions (for standard dense tensors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    /// 64-byte aligned byte offset relative to `arena_offset_start`.
    pub byte_offset: u32,
    /// 64-byte aligned byte length.
    pub byte_length: u32,
}

/// `.dnam-ckpt` per-stem registry record (`stems_registry[*]`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StemRegistryEntry {
    /// Unique stem identifier (e.g., `"stem_00_noon"`).
    pub stem_id: String,
    /// Control settings captured for this stem.
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Platform-agnostic IEEE-754 PCM SHA-256 provenance digests.
    pub provenance: StemProvenance,
    /// Castagnoli `CRC32C` over this stem's $151,296\text{ B}$ `Arena 3` record (`G_audio + C_audio`).
    pub stem_arena3_crc32c: u32,
    /// Open dictionary for stem-specific user metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Extracted Stage-1 physical anchors (delays, LFO frequencies, decay gains).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_physics: Option<ExtractedPhysics>,
    /// `Arena 3` slice descriptor for $\operatorname{pack}_{\text{U}}(\mathbf{G}_{\text{audio}}^{(s)})$ (`148,224 B`).
    pub g_audio_uplo_u_f64: TensorSliceEntry,
    /// `Arena 3` slice descriptor for tier-ordered $\mathbf{C}_{\text{audio}}^{(s)}$ (`3,072 B`).
    pub c_audio_tier_f64: TensorSliceEntry,
}

/// Platform-agnostic IEEE-754 PCM SHA-256 provenance for a training stem.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StemProvenance {
    /// SHA-256 hex digest of raw little-endian `f32` DI input frames.
    pub input_pcm_sha256: String,
    /// SHA-256 hex digest of raw little-endian `f32` target output frames.
    pub target_pcm_sha256: String,
    /// Number of stereo/mono frames in the stem.
    pub num_frames: u64,
}

/// Extracted Stage-1 physical system-identification anchors for a stem.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedPhysics {
    /// Extracted FDN delay times in milliseconds.
    #[serde(default)]
    pub delays_ms: Vec<f32>,
    /// Extracted LFO frequencies in Hz.
    #[serde(default)]
    pub lfo_freqs_hz: Vec<f32>,
    /// Extracted per-line linear feedback decay gains.
    #[serde(default)]
    pub decay_gains_lin: Vec<f32>,
}

/// `.dnam-ckpt` 3-phase control clamp audit report and actionable stem prescriptions.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditReport {
    /// Phase-1 dataset coverage clamps (`"CLAMP-RANGE"`, `"CLAMP-SWITCH"`, `"CLAMP-STABILITY"`).
    #[serde(default)]
    pub phase1_clamps: Vec<String>,
    /// Phase-2 interior rail saturation warnings.
    #[serde(default)]
    pub phase2_rail_warnings: Vec<String>,
    /// Phase-3 post-fit dynamic stability & head gain clamps (`"CLAMP-DYNAMIC"`, `"CLAMP-HEAD-GAIN"`).
    #[serde(default)]
    pub phase3_clamps: Vec<String>,
    /// Human-readable actionable stem prescriptions to lift active clamps.
    #[serde(default)]
    pub prescriptions: Vec<String>,
}

impl DnamJsonHeader {
    /// Validates all structural, alignment, and safety invariants of the JSON header.
    pub fn validate(&self) -> Result<(), DnamError> {
        if self.container_kind != "dnam" && self.container_kind != "dnam-ckpt" {
            return Err(DnamError::SchemaValidationError(
                "container_kind must be 'dnam' or 'dnam-ckpt'".to_string(),
            ));
        }
        if self.sample_rate_train_hz < 44_100 || self.sample_rate_train_hz > 96_000 {
            return Err(DnamError::InvalidSampleRate(self.sample_rate_train_hz));
        }
        if self.controls_schema.len() > MAX_CONTROLS {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds maximum of 16 controls".to_string(),
            ));
        }
        for entry in self.tensor_directory.values() {
            if (entry.byte_offset & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_offset",
                    offset: entry.byte_offset,
                });
            }
            if (entry.byte_length & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_length",
                    offset: entry.byte_length,
                });
            }
        }
        // Validate compiling control descriptors
        let _ = self.compile_clamp_descriptors()?;
        Ok(())
    }

    /// Compiles `controls_schema` into the 16-slot binary [`ControlClampDescriptor`] table
    /// (`1,024 B = 16 × 64 B` cache lines) used at $3\text{ kHz}$ with zero string parsing and zero `FDIV`.
    pub fn compile_clamp_descriptors(
        &self,
    ) -> Result<[ControlClampDescriptor; MAX_CONTROLS], DnamError> {
        if self.controls_schema.len() > MAX_CONTROLS {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds MAX_CONTROLS (16)".to_string(),
            ));
        }
        let mut table = [ControlClampDescriptor::empty(); MAX_CONTROLS];

        for (p, ctrl) in self.controls_schema.iter().enumerate() {
            match ctrl.kind {
                ControlKind::Range => {
                    let decl = ctrl.declared_range.ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "range control missing declared_range".to_string(),
                        )
                    })?;
                    let safe = ctrl.safe_clamped_range.unwrap_or(decl);
                    let span = decl[1] - decl[0];
                    if span <= 1e-6 {
                        return Err(DnamError::SchemaValidationError(
                            "range control declared_range span must be > 0".to_string(),
                        ));
                    }
                    let scale_norm = 2.0 / span;
                    let offset_norm = -1.0 - decl[0] * scale_norm;
                    let default_val = ctrl.default.as_f64().ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "range control default must be numeric".to_string(),
                        )
                    })? as f32;
                    let degree = ctrl.lobatto_degree.unwrap_or(2).min(3);
                    let mut anchors = [0.0_f32; 4];
                    if let Some(ref nodes) = ctrl.anchor_nodes_norm {
                        for (i, &val) in nodes.iter().take(3).enumerate() {
                            anchors[i] = val.clamp(-1.0, 1.0);
                        }
                    } else {
                        anchors[0] = (default_val * scale_norm + offset_norm).clamp(-1.0, 1.0);
                        anchors[1] = -1.0;
                        anchors[2] = 1.0;
                    }

                    // Compile up to 3 conditional clamp rules ("switch_id=option_id" -> bound)
                    let mut cond_rules = [ConditionalClampRule::empty(); 3];
                    let mut rule_idx = 0;

                    for (cond_key, &max_val) in &ctrl.conditional_max {
                        if rule_idx < 3
                            && let Some((sw_idx, opt_idx)) = self.resolve_switch_condition(cond_key)
                        {
                            cond_rules[rule_idx] = ConditionalClampRule {
                                switch_ctrl_idx: sw_idx,
                                switch_opt_idx: opt_idx,
                                rule_op: 1, // 1 = max_ceil
                                _pad: 0,
                                bound_val: max_val,
                            };
                            rule_idx += 1;
                        }
                    }
                    for (cond_key, &min_val) in &ctrl.conditional_min {
                        if rule_idx < 3
                            && let Some((sw_idx, opt_idx)) = self.resolve_switch_condition(cond_key)
                        {
                            cond_rules[rule_idx] = ConditionalClampRule {
                                switch_ctrl_idx: sw_idx,
                                switch_opt_idx: opt_idx,
                                rule_op: 2, // 2 = min_floor
                                _pad: 0,
                                bound_val: min_val,
                            };
                            rule_idx += 1;
                        }
                    }

                    table[p] = ControlClampDescriptor {
                        ctrl_kind: 0,
                        lobatto_degree: degree,
                        active_options_mask: 0x0001,
                        default_val: default_val.clamp(safe[0], safe[1]),
                        safe_min_base: safe[0].max(decl[0]),
                        safe_max_base: safe[1].min(decl[1]),
                        scale_norm,
                        offset_norm,
                        anchor_nodes_norm: anchors,
                        cond_rules,
                    };
                }
                ControlKind::Switch => {
                    let decl_opts = ctrl.declared_options.as_ref().ok_or_else(|| {
                        DnamError::SchemaValidationError(
                            "switch control missing declared_options".to_string(),
                        )
                    })?;
                    if decl_opts.is_empty() || decl_opts.len() > 16 {
                        return Err(DnamError::SchemaValidationError(
                            "switch control must have 1..=16 declared_options".to_string(),
                        ));
                    }
                    let active_opts = ctrl.active_options.as_ref().unwrap_or(decl_opts);
                    let mut mask: u16 = 0;
                    for (idx, opt_id) in decl_opts.iter().enumerate() {
                        if active_opts.iter().any(|a| a == opt_id) {
                            mask |= 1_u16 << idx;
                        }
                    }
                    if mask == 0 {
                        mask = 1;
                    }
                    let default_str = ctrl.default.as_str().unwrap_or(decl_opts[0].as_str());
                    let default_idx = decl_opts.iter().position(|o| o == default_str).unwrap_or(0);

                    table[p] = ControlClampDescriptor {
                        ctrl_kind: 1,
                        lobatto_degree: (decl_opts.len() as u8).saturating_sub(1),
                        active_options_mask: mask,
                        default_val: default_idx as f32,
                        safe_min_base: 0.0,
                        safe_max_base: (decl_opts.len() - 1) as f32,
                        scale_norm: 1.0,
                        offset_norm: 0.0,
                        anchor_nodes_norm: [0.0; 4],
                        cond_rules: [ConditionalClampRule::empty(); 3],
                    };
                }
            }
        }

        Ok(table)
    }

    /// Resolves a `"switch_id=option_id"` condition key into `(switch_ctrl_idx, switch_opt_idx)`.
    fn resolve_switch_condition(&self, cond_key: &str) -> Option<(u8, u8)> {
        let (sw_id, opt_id) = cond_key.split_once('=')?;
        for (idx, ctrl) in self.controls_schema.iter().enumerate() {
            if ctrl.id == sw_id
                && ctrl.kind == ControlKind::Switch
                && let Some(ref opts) = ctrl.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == opt_id)
            {
                return Some((idx as u8, opt_idx as u8));
            }
        }
        None
    }
}
