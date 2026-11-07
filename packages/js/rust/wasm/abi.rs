//! Safe WASM linear-memory arena, handle tables, and zero-`unsafe` slice helpers
//! (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.runtime.zero-unsafe]`).

use std::cell::RefCell;

use dnam_engine::memory::CacheLine64;
use zerocopy::{FromBytes, IntoBytes};

use crate::core::control::ControlSpaceCore;
use crate::core::engine::EngineCore;

struct WasmAllocSlot {
    base: usize,
    blocks: Box<[CacheLine64]>,
}

impl WasmAllocSlot {
    #[inline]
    fn byte_len(&self) -> usize {
        self.blocks.len() * 64
    }

    #[inline]
    fn contains_range(&self, ptr: usize, len: usize) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        let Some(slot_end) = self.base.checked_add(self.byte_len()) else {
            return false;
        };
        ptr >= self.base && end <= slot_end
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.blocks.as_bytes()
    }

    #[inline]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.blocks.as_mut_bytes()
    }
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static LAST_RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LAST_ARENA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ALLOC_TABLE: RefCell<Vec<WasmAllocSlot>> = const { RefCell::new(Vec::new()) };
    static ENGINES: RefCell<Vec<Option<EngineCore>>> = const { RefCell::new(Vec::new()) };
    static CONTROL_SPACES: RefCell<Vec<Option<ControlSpaceCore>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub(crate) fn set_error(msg: String) -> i32 {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
    -1
}

#[inline]
pub(crate) fn set_result(bytes: Vec<u8>) -> i32 {
    LAST_RESULT.with(|r| {
        *r.borrow_mut() = bytes;
    });
    0
}

#[inline]
pub(crate) fn set_arena(bytes: Vec<u8>) {
    LAST_ARENA.with(|a| {
        *a.borrow_mut() = bytes;
    });
}

#[inline]
pub(crate) fn insert_engine(engine: EngineCore) -> u32 {
    ENGINES.with(|table| {
        let mut slots = table.borrow_mut();
        for (idx, slot) in slots.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(engine);
                return (idx + 1) as u32;
            }
        }
        slots.push(Some(engine));
        slots.len() as u32
    })
}

#[inline]
pub(crate) fn remove_engine(handle: u32) {
    if handle == 0 {
        return;
    }
    ENGINES.with(|table| {
        let mut slots = table.borrow_mut();
        if let Some(slot) = slots.get_mut((handle - 1) as usize) {
            *slot = None;
        }
    });
}

#[inline]
pub(crate) fn with_engine<R>(handle: u32, f: impl FnOnce(&EngineCore) -> R) -> Option<R> {
    if handle == 0 {
        return None;
    }
    ENGINES.with(|table| {
        let slots = table.borrow();
        let engine = slots.get((handle - 1) as usize)?.as_ref()?;
        Some(f(engine))
    })
}

#[inline]
pub(crate) fn with_engine_mut<R>(handle: u32, f: impl FnOnce(&mut EngineCore) -> R) -> Option<R> {
    if handle == 0 {
        return None;
    }
    ENGINES.with(|table| {
        let mut slots = table.borrow_mut();
        let engine = slots.get_mut((handle - 1) as usize)?.as_mut()?;
        Some(f(engine))
    })
}

#[inline]
pub(crate) fn insert_control_space(space: ControlSpaceCore) -> u32 {
    CONTROL_SPACES.with(|table| {
        let mut slots = table.borrow_mut();
        for (idx, slot) in slots.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(space);
                return (idx + 1) as u32;
            }
        }
        slots.push(Some(space));
        slots.len() as u32
    })
}

#[inline]
pub(crate) fn remove_control_space(handle: u32) {
    if handle == 0 {
        return;
    }
    CONTROL_SPACES.with(|table| {
        let mut slots = table.borrow_mut();
        if let Some(slot) = slots.get_mut((handle - 1) as usize) {
            *slot = None;
        }
    });
}

#[inline]
pub(crate) fn with_control_space<R>(
    handle: u32,
    f: impl FnOnce(&ControlSpaceCore) -> R,
) -> Option<R> {
    if handle == 0 {
        return None;
    }
    CONTROL_SPACES.with(|table| {
        let slots = table.borrow();
        let space = slots.get((handle - 1) as usize)?.as_ref()?;
        Some(f(space))
    })
}

#[inline]
pub(crate) fn read_bytes(ptr: usize, len: usize) -> Vec<u8> {
    if len == 0 || ptr == 0 {
        return Vec::new();
    }
    ALLOC_TABLE.with(|table| {
        let slots = table.borrow();
        for slot in slots.iter() {
            if slot.contains_range(ptr, len) {
                let rel = ptr - slot.base;
                return slot.as_bytes()[rel..rel + len].to_vec();
            }
        }
        Vec::new()
    })
}

#[inline]
pub(crate) fn read_opt_bytes(ptr: usize, len: usize) -> Option<Vec<u8>> {
    if len == 0 || ptr == 0 {
        None
    } else {
        Some(read_bytes(ptr, len))
    }
}

#[inline]
pub(crate) fn read_str(ptr: usize, len: usize) -> Result<String, String> {
    let bytes = read_bytes(ptr, len);
    String::from_utf8(bytes).map_err(|e| format!("[DNAM-VALIDATION] Invalid UTF-8: {e}"))
}

#[inline]
pub(crate) fn read_opt_str(ptr: usize, len: usize) -> Result<Option<String>, String> {
    if len == 0 || ptr == 0 {
        Ok(None)
    } else {
        read_str(ptr, len).map(Some)
    }
}

#[inline]
pub(crate) fn read_u32_vec(ptr: usize, len: usize) -> Vec<u32> {
    let Some(byte_len) = len.checked_mul(4) else {
        return Vec::new();
    };
    if byte_len == 0 || ptr == 0 {
        return Vec::new();
    }
    ALLOC_TABLE.with(|table| {
        let slots = table.borrow();
        for slot in slots.iter() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(slice) = <[u32]>::ref_from_bytes(&slot.as_bytes()[rel..rel + byte_len]) {
                    return slice.to_vec();
                }
            }
        }
        Vec::new()
    })
}

#[inline]
pub(crate) fn read_f32_vec(ptr: usize, len: usize) -> Vec<f32> {
    let Some(byte_len) = len.checked_mul(4) else {
        return Vec::new();
    };
    if byte_len == 0 || ptr == 0 {
        return Vec::new();
    }
    ALLOC_TABLE.with(|table| {
        let slots = table.borrow();
        for slot in slots.iter() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(slice) = <[f32]>::ref_from_bytes(&slot.as_bytes()[rel..rel + byte_len]) {
                    return slice.to_vec();
                }
            }
        }
        Vec::new()
    })
}

#[inline]
pub(crate) fn read_f64_vec(ptr: usize, len: usize) -> Vec<f64> {
    let Some(byte_len) = len.checked_mul(8) else {
        return Vec::new();
    };
    if byte_len == 0 || ptr == 0 {
        return Vec::new();
    }
    ALLOC_TABLE.with(|table| {
        let slots = table.borrow();
        for slot in slots.iter() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(slice) = <[f64]>::ref_from_bytes(&slot.as_bytes()[rel..rel + byte_len]) {
                    return slice.to_vec();
                }
            }
        }
        Vec::new()
    })
}

#[inline]
pub(crate) fn write_f32(ptr: usize, val: f32) -> bool {
    write_f32_slice(ptr, &[val])
}

#[inline]
pub(crate) fn write_f64(ptr: usize, val: f64) -> bool {
    write_f64_slice(ptr, &[val])
}

#[inline]
pub(crate) fn write_u32_slice(ptr: usize, src: &[u32]) -> bool {
    let Some(byte_len) = src.len().checked_mul(4) else {
        return false;
    };
    if byte_len == 0 {
        return ptr != 0;
    }
    if ptr == 0 {
        return false;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        for slot in slots.iter_mut() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(dst) =
                    <[u32]>::mut_from_bytes(&mut slot.as_bytes_mut()[rel..rel + byte_len])
                {
                    dst.copy_from_slice(src);
                    return true;
                }
            }
        }
        false
    })
}

#[inline]
pub(crate) fn write_f32_slice(ptr: usize, src: &[f32]) -> bool {
    let Some(byte_len) = src.len().checked_mul(4) else {
        return false;
    };
    if byte_len == 0 {
        return ptr != 0;
    }
    if ptr == 0 {
        return false;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        for slot in slots.iter_mut() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(dst) =
                    <[f32]>::mut_from_bytes(&mut slot.as_bytes_mut()[rel..rel + byte_len])
                {
                    dst.copy_from_slice(src);
                    return true;
                }
            }
        }
        false
    })
}

#[inline]
pub(crate) fn write_f64_slice(ptr: usize, src: &[f64]) -> bool {
    let Some(byte_len) = src.len().checked_mul(8) else {
        return false;
    };
    if byte_len == 0 {
        return ptr != 0;
    }
    if ptr == 0 {
        return false;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        for slot in slots.iter_mut() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                if let Ok(dst) =
                    <[f64]>::mut_from_bytes(&mut slot.as_bytes_mut()[rel..rel + byte_len])
                {
                    dst.copy_from_slice(src);
                    return true;
                }
            }
        }
        false
    })
}

#[inline]
pub(crate) fn with_f32_mut<R>(
    ptr: usize,
    len: usize,
    f: impl FnOnce(&mut [f32]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&mut []));
    }
    let byte_len = len.checked_mul(4)?;
    if ptr == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        for slot in slots.iter_mut() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                let dst =
                    <[f32]>::mut_from_bytes(&mut slot.as_bytes_mut()[rel..rel + byte_len]).ok()?;
                return Some(f(dst));
            }
        }
        None
    })
}

#[inline]
pub(crate) fn with_f64_mut<R>(
    ptr: usize,
    len: usize,
    f: impl FnOnce(&mut [f64]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&mut []));
    }
    let byte_len = len.checked_mul(8)?;
    if ptr == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        for slot in slots.iter_mut() {
            if slot.contains_range(ptr, byte_len) {
                let rel = ptr - slot.base;
                let dst =
                    <[f64]>::mut_from_bytes(&mut slot.as_bytes_mut()[rel..rel + byte_len]).ok()?;
                return Some(f(dst));
            }
        }
        None
    })
}

#[inline]
fn split_2_within(
    bytes: &mut [u8],
    rel0: usize,
    rel1: usize,
    byte_len: usize,
) -> Option<(&mut [f32], &mut [f32])> {
    if rel0.checked_add(byte_len)? <= rel1 && rel1.checked_add(byte_len)? <= bytes.len() {
        let (head, tail) = bytes.split_at_mut(rel1);
        let s0 = <[f32]>::mut_from_bytes(&mut head[rel0..rel0 + byte_len]).ok()?;
        let s1 = <[f32]>::mut_from_bytes(&mut tail[..byte_len]).ok()?;
        Some((s0, s1))
    } else if rel1.checked_add(byte_len)? <= rel0 && rel0.checked_add(byte_len)? <= bytes.len() {
        let (head, tail) = bytes.split_at_mut(rel0);
        let s1 = <[f32]>::mut_from_bytes(&mut head[rel1..rel1 + byte_len]).ok()?;
        let s0 = <[f32]>::mut_from_bytes(&mut tail[..byte_len]).ok()?;
        Some((s0, s1))
    } else {
        None
    }
}

/// Zero-copy access to two disjoint `&mut [f32]` slices in WASM linear memory.
pub(crate) fn with_f32_2_mut<R>(
    ptr0: usize,
    ptr1: usize,
    len: usize,
    f: impl FnOnce(&mut [f32], &mut [f32]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&mut [], &mut []));
    }
    let byte_len = len.checked_mul(4)?;
    if ptr0 == 0 || ptr1 == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        let idx0 = slots
            .iter()
            .position(|s| s.contains_range(ptr0, byte_len))?;
        let idx1 = slots
            .iter()
            .position(|s| s.contains_range(ptr1, byte_len))?;
        if idx0 == idx1 {
            let base = slots[idx0].base;
            let rel0 = ptr0 - base;
            let rel1 = ptr1 - base;
            let (s0, s1) = split_2_within(slots[idx0].as_bytes_mut(), rel0, rel1, byte_len)?;
            Some(f(s0, s1))
        } else if idx0 < idx1 {
            let (head, tail) = slots.split_at_mut(idx1);
            let slot0 = &mut head[idx0];
            let slot1 = &mut tail[0];
            let rel0 = ptr0 - slot0.base;
            let rel1 = ptr1 - slot1.base;
            let s0 =
                <[f32]>::mut_from_bytes(&mut slot0.as_bytes_mut()[rel0..rel0 + byte_len]).ok()?;
            let s1 =
                <[f32]>::mut_from_bytes(&mut slot1.as_bytes_mut()[rel1..rel1 + byte_len]).ok()?;
            Some(f(s0, s1))
        } else {
            let (head, tail) = slots.split_at_mut(idx0);
            let slot1 = &mut head[idx1];
            let slot0 = &mut tail[0];
            let rel0 = ptr0 - slot0.base;
            let rel1 = ptr1 - slot1.base;
            let s0 =
                <[f32]>::mut_from_bytes(&mut slot0.as_bytes_mut()[rel0..rel0 + byte_len]).ok()?;
            let s1 =
                <[f32]>::mut_from_bytes(&mut slot1.as_bytes_mut()[rel1..rel1 + byte_len]).ok()?;
            Some(f(s0, s1))
        }
    })
}

/// Zero-copy access to one `&[f32]` input slice and one `&mut [f32]` output slice,
/// falling back to a temporary input copy when `in_ptr` and `out_ptr` overlap.
#[inline]
pub(crate) fn with_f32_1_in_1_out<R>(
    in_ptr: usize,
    out_ptr: usize,
    len: usize,
    f: impl FnOnce(&[f32], &mut [f32]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&[], &mut []));
    }
    let byte_len = len.checked_mul(4)?;
    if in_ptr == 0 || out_ptr == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        let idx0 = slots
            .iter()
            .position(|s| s.contains_range(in_ptr, byte_len))?;
        let idx1 = slots
            .iter()
            .position(|s| s.contains_range(out_ptr, byte_len))?;
        if idx0 == idx1 {
            let base = slots[idx0].base;
            let rel0 = in_ptr - base;
            let rel1 = out_ptr - base;
            if let Some((s0, s1)) = split_2_within(slots[idx0].as_bytes_mut(), rel0, rel1, byte_len)
            {
                return Some(f(s0, s1));
            }
            drop(slots);
            let inp = read_f32_vec(in_ptr, len);
            if inp.len() != len {
                return None;
            }
            return with_f32_mut(out_ptr, len, |out| f(&inp, out));
        }
        if idx0 < idx1 {
            let (head, tail) = slots.split_at_mut(idx1);
            let slot0 = &mut head[idx0];
            let slot1 = &mut tail[0];
            let rel0 = in_ptr - slot0.base;
            let rel1 = out_ptr - slot1.base;
            let s0 = <[f32]>::ref_from_bytes(&slot0.as_bytes()[rel0..rel0 + byte_len]).ok()?;
            let s1 =
                <[f32]>::mut_from_bytes(&mut slot1.as_bytes_mut()[rel1..rel1 + byte_len]).ok()?;
            Some(f(s0, s1))
        } else {
            let (head, tail) = slots.split_at_mut(idx0);
            let slot1 = &mut head[idx1];
            let slot0 = &mut tail[0];
            let rel0 = in_ptr - slot0.base;
            let rel1 = out_ptr - slot1.base;
            let s0 = <[f32]>::ref_from_bytes(&slot0.as_bytes()[rel0..rel0 + byte_len]).ok()?;
            let s1 =
                <[f32]>::mut_from_bytes(&mut slot1.as_bytes_mut()[rel1..rel1 + byte_len]).ok()?;
            Some(f(s0, s1))
        }
    })
}

/// Zero-copy access to one `&[f32]` input slice and two disjoint `&mut [f32]` output slices.
pub(crate) fn with_f32_1_in_2_out<R>(
    in_ptr: usize,
    out0_ptr: usize,
    out1_ptr: usize,
    len: usize,
    f: impl FnOnce(&[f32], &mut [f32], &mut [f32]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&[], &mut [], &mut []));
    }
    let byte_len = len.checked_mul(4)?;
    if in_ptr == 0 || out0_ptr == 0 || out1_ptr == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        if let Some(idx) = slots.iter().position(|s| {
            s.contains_range(in_ptr, byte_len)
                && s.contains_range(out0_ptr, byte_len)
                && s.contains_range(out1_ptr, byte_len)
        }) {
            let base = slots[idx].base;
            let rel0 = in_ptr - base;
            let rel1 = out0_ptr - base;
            let rel2 = out1_ptr - base;
            let bytes = slots[idx].as_bytes_mut();
            if rel0.saturating_add(byte_len) <= rel1
                && rel1.saturating_add(byte_len) <= rel2
                && rel2.saturating_add(byte_len) <= bytes.len()
            {
                let (b01, b2_tail) = bytes.split_at_mut(rel2);
                let (b0_head, b1_tail) = b01.split_at_mut(rel1);
                if let (Ok(s0), Ok(s1), Ok(s2)) = (
                    <[f32]>::ref_from_bytes(&b0_head[rel0..rel0 + byte_len]),
                    <[f32]>::mut_from_bytes(&mut b1_tail[..byte_len]),
                    <[f32]>::mut_from_bytes(&mut b2_tail[..byte_len]),
                ) {
                    return Some(f(s0, s1, s2));
                }
            }
        }
        drop(slots);
        let inp = read_f32_vec(in_ptr, len);
        if inp.len() != len {
            return None;
        }
        with_f32_2_mut(out0_ptr, out1_ptr, len, |out0, out1| f(&inp, out0, out1))
    })
}

/// Zero-copy access to two `&[f32]` input slices and two disjoint `&mut [f32]` output slices.
pub(crate) fn with_f32_2_in_2_out<R>(
    in0_ptr: usize,
    in1_ptr: usize,
    out0_ptr: usize,
    out1_ptr: usize,
    len: usize,
    f: impl FnOnce(&[f32], &[f32], &mut [f32], &mut [f32]) -> R,
) -> Option<R> {
    if len == 0 {
        return Some(f(&[], &[], &mut [], &mut []));
    }
    let byte_len = len.checked_mul(4)?;
    if in0_ptr == 0 || in1_ptr == 0 || out0_ptr == 0 || out1_ptr == 0 {
        return None;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        if let Some(idx) = slots.iter().position(|s| {
            s.contains_range(in0_ptr, byte_len)
                && s.contains_range(in1_ptr, byte_len)
                && s.contains_range(out0_ptr, byte_len)
                && s.contains_range(out1_ptr, byte_len)
        }) {
            let base = slots[idx].base;
            let rel0 = in0_ptr - base;
            let rel1 = in1_ptr - base;
            let rel2 = out0_ptr - base;
            let rel3 = out1_ptr - base;
            let bytes = slots[idx].as_bytes_mut();
            if rel0.saturating_add(byte_len) <= rel1
                && rel1.saturating_add(byte_len) <= rel2
                && rel2.saturating_add(byte_len) <= rel3
                && rel3.saturating_add(byte_len) <= bytes.len()
            {
                let (b012, b3_tail) = bytes.split_at_mut(rel3);
                let (b01, b2_tail) = b012.split_at_mut(rel2);
                let (b0_head, b1_tail) = b01.split_at_mut(rel1);
                if let (Ok(s0), Ok(s1), Ok(s2), Ok(s3)) = (
                    <[f32]>::ref_from_bytes(&b0_head[rel0..rel0 + byte_len]),
                    <[f32]>::ref_from_bytes(&b1_tail[..byte_len]),
                    <[f32]>::mut_from_bytes(&mut b2_tail[..byte_len]),
                    <[f32]>::mut_from_bytes(&mut b3_tail[..byte_len]),
                ) {
                    return Some(f(s0, s1, s2, s3));
                }
            }
        }
        drop(slots);
        let in0 = read_f32_vec(in0_ptr, len);
        let in1 = read_f32_vec(in1_ptr, len);
        if in0.len() != len || in1.len() != len {
            return None;
        }
        with_f32_2_mut(out0_ptr, out1_ptr, len, |out0, out1| {
            f(&in0, &in1, out0, out1)
        })
    })
}

#[inline]
pub(crate) fn alloc_bytes(size: usize, _align: usize) -> usize {
    let num_blocks = size.max(1).div_ceil(64);
    let blocks = vec![CacheLine64::default(); num_blocks].into_boxed_slice();
    let base = blocks.as_bytes().as_ptr() as usize;
    ALLOC_TABLE.with(|table| {
        table.borrow_mut().push(WasmAllocSlot { base, blocks });
    });
    base
}

#[inline]
pub(crate) fn dealloc_bytes(ptr: usize, _size: usize, _align: usize) {
    if ptr == 0 {
        return;
    }
    ALLOC_TABLE.with(|table| {
        let mut slots = table.borrow_mut();
        if let Some(pos) = slots.iter().position(|s| s.base == ptr) {
            slots.swap_remove(pos);
        }
    });
}

#[inline]
pub(crate) fn last_error_ptr() -> usize {
    LAST_ERROR.with(|e| e.borrow().as_ptr() as usize)
}

#[inline]
pub(crate) fn last_error_len() -> usize {
    LAST_ERROR.with(|e| e.borrow().len())
}

#[inline]
pub(crate) fn result_ptr() -> usize {
    LAST_RESULT.with(|r| r.borrow().as_ptr() as usize)
}

#[inline]
pub(crate) fn result_len() -> usize {
    LAST_RESULT.with(|r| r.borrow().len())
}

#[inline]
pub(crate) fn arena_ptr() -> usize {
    LAST_ARENA.with(|a| a.borrow().as_ptr() as usize)
}

#[inline]
pub(crate) fn arena_len() -> usize {
    LAST_ARENA.with(|a| a.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{dnam_wasm_alloc, dnam_wasm_dealloc};

    // verify[sdk.ts.runtime.zero-unsafe]
    #[test]
    fn test_safe_wasm_arena_and_disjoint_slices() {
        let ptr = dnam_wasm_alloc(256, 64);
        assert_ne!(ptr, 0);
        assert_eq!(ptr % 64, 0);

        let src_f32 = [1.0f32, 2.0, 3.0, 4.0];
        assert!(write_f32_slice(ptr, &src_f32));
        assert!(write_f32_slice(ptr + 16, &[10.0f32, 20.0, 30.0, 40.0]));

        // Fast-path 4-slice disjoint split within the same 64B-aligned allocation
        let ok = with_f32_2_in_2_out(
            ptr,
            ptr + 16,
            ptr + 32,
            ptr + 48,
            4,
            |in0, in1, out0, out1| {
                for i in 0..4 {
                    out0[i] = in0[i] + in1[i];
                    out1[i] = in1[i] - in0[i];
                }
            },
        );
        assert!(ok.is_some());
        assert_eq!(read_f32_vec(ptr + 32, 4), vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(read_f32_vec(ptr + 48, 4), vec![9.0, 18.0, 27.0, 36.0]);

        // Cross-allocation disjoint split fallback
        let ptr2 = dnam_wasm_alloc(128, 64);
        assert!(write_f32_slice(ptr2, &[5.0f32, 6.0, 7.0, 8.0]));
        let ok_cross = with_f32_2_mut(ptr2, ptr, 4, |s0, s1| {
            s0.swap_with_slice(s1);
        });
        assert!(ok_cross.is_some());
        assert_eq!(read_f32_vec(ptr, 4), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(read_f32_vec(ptr2, 4), vec![1.0, 2.0, 3.0, 4.0]);

        // Overlapping range within the same slot must be rejected safely (`None`) for 2 mut
        assert!(with_f32_2_mut(ptr, ptr + 8, 4, |_, _| {}).is_none());

        // `with_f32_1_in_1_out` supports both disjoint and overlapping (in-place) slices safely
        assert!(
            with_f32_1_in_1_out(ptr, ptr + 16, 4, |inp, out| {
                for i in 0..4 {
                    out[i] = inp[i] * 2.0;
                }
            })
            .is_some()
        );
        assert_eq!(read_f32_vec(ptr + 16, 4), vec![10.0, 12.0, 14.0, 16.0]);
        assert!(
            with_f32_1_in_1_out(ptr + 16, ptr + 16, 4, |inp, out| {
                for i in 0..4 {
                    out[i] = inp[i] + 1.0;
                }
            })
            .is_some()
        );
        assert_eq!(read_f32_vec(ptr + 16, 4), vec![11.0, 13.0, 15.0, 17.0]);

        dnam_wasm_dealloc(ptr, 256, 64);
        dnam_wasm_dealloc(ptr2, 128, 64);
        assert!(read_f32_vec(ptr, 4).is_empty());
    }
}
