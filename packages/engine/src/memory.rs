//! 64-byte cache-line aligned memory management, OS page pre-faulting, and zero-copy storage.
//!
//! Ensures all weight arenas (`Arena 0..2`) and Core 2 `fp16` delay ring buffers are strictly
//! 64-byte aligned (`ptr % 64 == 0`) across ARM64, x86-64, and WebAssembly, while eliminating
//! first-touch OS page faults on the real-time audio thread.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// 64-byte L1D cache-line aligned byte block (`sizeof == 64`, `alignof == 64`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct CacheLine64(pub [u8; 64]);

impl Default for CacheLine64 {
    #[inline]
    fn default() -> Self {
        Self([0u8; 64])
    }
}

impl CacheLine64 {
    /// Returns the 64-byte cache line as a slice of 16 `f32` values.
    #[inline(always)]
    pub fn as_f32_slice(&self) -> &[f32; 16] {
        zerocopy::transmute_ref!(self)
    }

    /// Returns the 64-byte cache line as a mutable slice of 16 `f32` values.
    #[inline(always)]
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32; 16] {
        zerocopy::transmute_mut!(self)
    }
}

/// 64-byte L1D cache-line aligned `fp16` (`u16`) block of 32 samples (`sizeof == 64`, `alignof == 64`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct AlignedFp16Block(pub [u16; 32]);

impl Default for AlignedFp16Block {
    #[inline]
    fn default() -> Self {
        Self([0u16; 32])
    }
}

/// Heap-allocated buffer of 64-byte aligned blocks `T` that pre-faults all OS pages on creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignedBuffer<T: Copy + Default + FromBytes + IntoBytes + Immutable> {
    blocks: Vec<T>,
}

impl<T: Copy + Default + FromBytes + IntoBytes + Immutable> AlignedBuffer<T> {
    /// Allocates `num_blocks` 64-byte aligned blocks and immediately pre-faults every cache line
    /// so the real-time audio thread never triggers a demand-paging trap.
    pub fn new_zeroed_prefaulted(num_blocks: usize) -> Self {
        const {
            assert!(core::mem::align_of::<T>() == 64);
            assert!(core::mem::size_of::<T>() == 64);
        }
        let mut blocks = vec![T::default(); num_blocks];
        Self::prefault_slice(&mut blocks);
        Self { blocks }
    }

    /// Alias for [`Self::new_zeroed_prefaulted`].
    #[inline]
    pub fn new_zeroed(num_blocks: usize) -> Self {
        Self::new_zeroed_prefaulted(num_blocks)
    }

    /// Pre-faults all 64-byte cache lines in `slice` using opaque `black_box` stores to commit
    /// backing physical pages and TLB entries during cold initialization.
    #[inline]
    pub fn prefault_slice(slice: &mut [T]) {
        let default_val = core::hint::black_box(T::default());
        for block in slice.iter_mut() {
            *core::hint::black_box(block) = default_val;
        }
    }

    /// Returns the number of 64-byte blocks in the buffer.
    #[inline(always)]
    pub fn len_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns `true` if the buffer contains zero blocks.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns the raw byte slice spanning all blocks.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.blocks.as_slice().as_bytes()
    }

    /// Returns the mutable raw byte slice spanning all blocks.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.blocks.as_mut_slice().as_mut_bytes()
    }
}

impl<T: Copy + Default + FromBytes + IntoBytes + Immutable> Deref for AlignedBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.blocks
    }
}

impl<T: Copy + Default + FromBytes + IntoBytes + Immutable> DerefMut for AlignedBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.blocks
    }
}

/// Unified weight arena storage supporting zero-copy 64B-aligned borrowing (`mmap` / static flash)
/// and owned 64B-aligned buffers (for unaligned slices, decrypted `ENCRYPTED_ARENA = 1` models,
/// or in-place hot-swapping).
#[derive(Clone, Debug)]
pub enum ArenaStorage<'a> {
    /// Zero-copy borrowed slice of 64-byte cache lines (`0` bytes copied).
    Borrowed(&'a [CacheLine64]),
    /// Owned, 64-byte aligned and page-prefaulted cache-line buffer.
    Owned {
        /// Underlying 64-byte cache line allocation (may be pre-allocated up to `Ultra` 566 blocks).
        buffer: AlignedBuffer<CacheLine64>,
        /// Active number of 64-byte blocks currently populated (`168` Lite, `349` Full, `566` Ultra).
        active_blocks: usize,
    },
}

impl<'a> ArenaStorage<'a> {
    /// Creates an `ArenaStorage` from `bytes`. If `bytes` is already 64-byte aligned in memory
    /// (`ptr as usize % 64 == 0` and `bytes.len() % 64 == 0`), borrows it with zero copies;
    /// otherwise copies it into a 64-byte aligned `Owned` buffer.
    pub fn from_slice_zero_copy_or_owned(bytes: &'a [u8]) -> Self {
        assert!(
            bytes.len().is_multiple_of(64),
            "Arena byte length must be a multiple of 64"
        );
        if let Ok(aligned_slice) = <[CacheLine64]>::ref_from_bytes(bytes) {
            Self::Borrowed(aligned_slice)
        } else {
            Self::from_slice_owned(bytes)
        }
    }

    /// Alias for [`Self::from_slice_zero_copy_or_owned`].
    #[inline]
    pub fn from_slice_or_copy(bytes: &'a [u8]) -> Self {
        Self::from_slice_zero_copy_or_owned(bytes)
    }

    /// Wraps an already-allocated [`AlignedBuffer<CacheLine64>`] into [`ArenaStorage::Owned`].
    #[inline]
    pub fn from_owned_buffer(buffer: AlignedBuffer<CacheLine64>) -> Self {
        let active_blocks = buffer.len_blocks();
        Self::Owned {
            buffer,
            active_blocks,
        }
    }

    /// Allocates an `Owned` 64-byte aligned buffer and copies `bytes` into it.
    pub fn from_slice_owned(bytes: &[u8]) -> Self {
        assert!(
            bytes.len().is_multiple_of(64),
            "Arena byte length must be a multiple of 64"
        );
        let num_blocks = bytes.len() / 64;
        let mut buffer = AlignedBuffer::<CacheLine64>::new_zeroed_prefaulted(num_blocks);
        buffer.as_bytes_mut().copy_from_slice(bytes);
        Self::Owned {
            buffer,
            active_blocks: num_blocks,
        }
    }

    /// Pre-allocates an `Owned` 64-byte aligned arena with capacity for `max_blocks` (e.g., `566`
    /// for `Ultra` = `36,224 B`) so models can be hot-swapped in-place with zero allocations.
    pub fn with_max_blocks(max_blocks: usize) -> Self {
        let buffer = AlignedBuffer::<CacheLine64>::new_zeroed_prefaulted(max_blocks);
        Self::Owned {
            buffer,
            active_blocks: 0,
        }
    }

    /// Overwrites the arena in-place from `bytes` without allocating if `self` is `Owned` with
    /// sufficient capacity. Returns `true` if updated in-place with zero allocations.
    #[inline]
    pub fn reload_in_place(&mut self, bytes: &[u8]) -> bool {
        if !bytes.len().is_multiple_of(64) {
            return false;
        }
        let req_blocks = bytes.len() / 64;
        if let Self::Owned {
            buffer,
            active_blocks,
        } = self
            && buffer.len_blocks() >= req_blocks
        {
            buffer.as_bytes_mut()[..bytes.len()].copy_from_slice(bytes);
            *active_blocks = req_blocks;
            return true;
        }
        false
    }

    /// Returns `true` if the arena is borrowing an external 64-byte aligned buffer with zero copies.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns the active 64-byte cache lines.
    #[inline(always)]
    pub fn as_cache_lines(&self) -> &[CacheLine64] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned {
                buffer,
                active_blocks,
            } => &buffer[..*active_blocks],
        }
    }

    /// Returns the active arena as a 64-byte aligned byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_cache_lines().as_bytes()
    }

    /// Returns the active arena as a 64-byte aligned `f32` slice.
    #[inline(always)]
    pub fn as_f32_slice(&self) -> &[f32] {
        <[f32]>::ref_from_bytes(self.as_bytes())
            .expect("CacheLine64 slice is always 4-byte aligned and a multiple of 4 bytes")
    }
}

/// Core 2 `M-UFDN` `fp16` (`u16`) delay ring-buffer storage supporting either owned 64B-aligned
/// heap memory (`Owned`) or caller-supplied static/shared scratch memory (`Borrowed`).
#[derive(Debug)]
pub enum DelayScratchStorage<'a> {
    /// Owned 64-byte aligned `fp16` blocks (allocated once on construction or `with_max_capacity`).
    Owned(AlignedBuffer<AlignedFp16Block>),
    /// Caller-supplied pre-allocated 64-byte aligned `fp16` blocks (`0` heap allocations).
    Borrowed(&'a mut [AlignedFp16Block]),
}

impl<'a> DelayScratchStorage<'a> {
    /// Returns an empty `DelayScratchStorage` with `0` bytes of heap allocation (`0 KB` RAM).
    #[inline]
    pub const fn empty() -> Self {
        Self::Owned(AlignedBuffer { blocks: Vec::new() })
    }

    /// Allocates an owned, page-prefaulted `fp16` delay buffer of `num_blocks` 64-byte [`AlignedFp16Block`]s.
    pub fn new_owned(num_blocks: usize) -> Self {
        Self::Owned(AlignedBuffer::new_zeroed_prefaulted(num_blocks))
    }

    /// Wraps a caller-provided slice of 64-byte aligned `AlignedFp16Block`s and zeroes it.
    pub fn from_borrowed_scratch(scratch: &'a mut [AlignedFp16Block]) -> Self {
        AlignedBuffer::prefault_slice(scratch);
        Self::Borrowed(scratch)
    }

    /// Returns the total capacity in bytes (`num_blocks * 64`).
    #[inline(always)]
    pub fn byte_len(&self) -> usize {
        self.as_blocks().len() * 64
    }

    /// Returns the total capacity in `u16` (`fp16`) samples.
    #[inline(always)]
    pub fn capacity_u16(&self) -> usize {
        self.as_blocks().len() * 32
    }

    /// Returns the underlying `AlignedFp16Block` slice.
    #[inline(always)]
    pub fn as_blocks(&self) -> &[AlignedFp16Block] {
        match self {
            Self::Owned(buf) => buf,
            Self::Borrowed(slice) => slice,
        }
    }

    /// Alias for [`Self::as_blocks`].
    #[inline(always)]
    pub fn as_slice(&self) -> &[AlignedFp16Block] {
        self.as_blocks()
    }

    /// Returns the mutable underlying `AlignedFp16Block` slice.
    #[inline(always)]
    pub fn as_blocks_mut(&mut self) -> &mut [AlignedFp16Block] {
        match self {
            Self::Owned(buf) => buf,
            Self::Borrowed(slice) => slice,
        }
    }

    /// Alias for [`Self::as_blocks_mut`].
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [AlignedFp16Block] {
        self.as_blocks_mut()
    }

    /// Returns the buffer as a contiguous slice of `u16` (`fp16` bits).
    #[inline(always)]
    pub fn as_u16_slice(&self) -> &[u16] {
        <[u16]>::ref_from_bytes(self.as_blocks().as_bytes())
            .expect("AlignedFp16Block slice is always 2-byte aligned and a multiple of 2 bytes")
    }

    /// Returns the buffer as a contiguous mutable slice of `u16` (`fp16` bits).
    #[inline(always)]
    pub fn as_u16_slice_mut(&mut self) -> &mut [u16] {
        <[u16]>::mut_from_bytes(self.as_blocks_mut().as_mut_bytes())
            .expect("AlignedFp16Block slice is always 2-byte aligned and a multiple of 2 bytes")
    }

    /// Clears all delay samples to `0` (`0.0` in IEEE-754 `fp16`) without reallocating.
    #[inline]
    pub fn clear(&mut self) {
        self.as_u16_slice_mut().fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_line_64_and_aligned_buffers() {
        let mut cl = CacheLine64::default();
        assert_eq!((&raw const cl as usize) % 64, 0);
        cl.as_f32_slice_mut()[0] = 3.25;
        assert_eq!(cl.as_f32_slice()[0], 3.25);

        let fp16_blk = AlignedFp16Block::default();
        assert_eq!(fp16_blk.0.len(), 32);

        let mut buf = AlignedBuffer::<CacheLine64>::new_zeroed(4);
        assert_eq!(buf.len_blocks(), 4);
        assert!(!buf.is_empty());
        assert_eq!((buf.as_bytes().as_ptr() as usize) % 64, 0);
        buf[0].as_f32_slice_mut()[1] = -1.5;
        assert_eq!(buf[0].as_f32_slice()[1], -1.5);
    }

    #[test]
    fn test_arena_storage_borrowed_owned_and_reload() {
        let aligned = AlignedBuffer::<CacheLine64>::new_zeroed(2);
        let borrowed = ArenaStorage::from_slice_or_copy(aligned.as_bytes());
        assert!(borrowed.is_borrowed());
        assert_eq!(borrowed.as_f32_slice().len(), 32);

        let mut unaligned_vec = alloc::vec![0u8; 129];
        unaligned_vec[1] = 0x3F;
        let owned = ArenaStorage::from_slice_or_copy(&unaligned_vec[1..129]);
        assert!(!owned.is_borrowed());
        assert_eq!(owned.as_bytes().len(), 128);

        let mut cap = ArenaStorage::with_max_blocks(4);
        assert_eq!(cap.as_bytes().len(), 0);
        assert!(!cap.reload_in_place(&[0u8; 63]));
        assert!(cap.reload_in_place(aligned.as_bytes()));
        assert_eq!(cap.as_bytes().len(), 128);
        assert!(!cap.reload_in_place(&[0u8; 64 * 5]));
    }

    #[test]
    fn test_delay_scratch_storage() {
        let empty = DelayScratchStorage::empty();
        assert_eq!(empty.byte_len(), 0);
        assert_eq!(empty.capacity_u16(), 0);

        let mut owned = DelayScratchStorage::new_owned(2);
        assert_eq!(owned.byte_len(), 128);
        assert_eq!(owned.capacity_u16(), 64);
        owned.as_u16_slice_mut()[0] = 0x3C00;
        assert_eq!(owned.as_u16_slice()[0], 0x3C00);
        owned.clear();
        assert_eq!(owned.as_u16_slice()[0], 0);

        let mut raw_blocks = [AlignedFp16Block::default(); 2];
        let mut borrowed = DelayScratchStorage::from_borrowed_scratch(&mut raw_blocks);
        assert_eq!(borrowed.as_slice().len(), 2);
        borrowed.as_slice_mut()[0].0[0] = 42;
        assert_eq!(borrowed.as_u16_slice()[0], 42);
    }
}
