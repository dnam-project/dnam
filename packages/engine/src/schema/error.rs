//! Error types (`DnamError`) and `#![no_std]` cold-initialization math aliases for schema validation.

use alloc::string::String;
use core::fmt;

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

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` for cold-init ZOH rescaling and UI skew.
#[inline]
pub fn cold_ln_f64(x: f64) -> f64 {
    crate::math::ln_f64(x)
}

/// Pure `#![no_std]` exponential $\exp(y)$ in `f64` for cold-init ZOH rescaling and UI skew.
#[inline]
pub fn cold_exp_f64(y: f64) -> f64 {
    crate::math::exp_f64(y)
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

/// Pure `#![no_std]` square root $\sqrt{x} = \exp(0.5 \ln(x))$ in `f64` for cold-init $\mathcal{H}_2$ normalizers.
#[inline]
pub fn cold_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    cold_exp_f64(0.5 * cold_ln_f64(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn test_dnam_error_display_and_cold_math() {
        let errors = [
            DnamError::BufferTooSmall {
                expected: 64,
                actual: 12,
            },
            DnamError::InvalidMagic(*b"ABCD"),
            DnamError::UnsupportedVersion { major: 9, minor: 0 },
            DnamError::UnalignedOffset {
                field: "arena",
                offset: 65,
            },
            DnamError::Crc32cMismatch {
                region: "header",
                expected: 1,
                computed: 2,
            },
            DnamError::InvalidJsonHeader("bad json".to_string()),
            DnamError::SchemaValidationError("bad schema".to_string()),
            DnamError::HeaderCapacityExceeded {
                required: 5000,
                capacity: 4000,
            },
            DnamError::CryptoVerificationFailed("bad sig"),
            DnamError::EncryptedArenaRequiresDeviceKey,
            DnamError::InvalidSampleRate(12_000),
            DnamError::InvalidControlIndex(99),
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
        }

        assert_eq!(cold_ln_f64(0.0), f64::NEG_INFINITY);
        assert_eq!(cold_exp_f64(800.0), f64::INFINITY);
        assert_eq!(cold_exp_f64(-800.0), 0.0);
        assert_eq!(cold_pow_f64(0.0, 2.0), 0.0);
        assert_eq!(cold_pow_f64(4.0, 1.0), 4.0);
        assert_eq!(cold_pow_f64(4.0, 0.0), 1.0);
        assert!((cold_pow_f64(4.0, 2.0) - 16.0).abs() < 1e-12);
        assert_eq!(cold_sqrt_f64(0.0), 0.0);
        assert_eq!(cold_sqrt_f64(-1.0), 0.0);
        assert!((cold_sqrt_f64(9.0) - 3.0).abs() < 1e-12);
    }
}
