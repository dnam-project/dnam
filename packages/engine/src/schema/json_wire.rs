//! Tensor directory wire format for 64B-aligned slices in Arena 0..3.
//!
//! Canonical Reference: `docs/containers/dnam_binary_format.md`.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Tensor directory entry describing a 64-byte aligned slice in `Arena 0..3` (or `.dnam-ckpt` `Arena 4..5`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorSliceEntry {
    /// Arena index (`0` = Lite, `1` = Full Delta, `2` = Pro Delta, `3` = Ultra Delta, `4` = `.dnam-ckpt` Stem Records).
    pub arena: u8,
    /// Data type (`"f32"` or `"f64"`).
    pub dtype: String,
    /// Optional packing tag (`"uplo_upper_col_major_tier_ordered"` for `.dnam-ckpt` Stem Record Gram matrices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packing: Option<String>,
    /// Optional feature dimension $N \in \{64, 128, 192, 256\}$ for `uplo='U'` packed Gram matrices.
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
