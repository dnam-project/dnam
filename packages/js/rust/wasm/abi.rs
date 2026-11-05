//! WASM linear-memory allocator, thread-local error/result buffers, and raw slice helpers
//! (`ref[sdk.ts.runtime.dual-target-drivers]`).

use std::alloc::{Layout, alloc, dealloc};
use std::cell::RefCell;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static LAST_RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LAST_ARENA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
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
pub(crate) fn slice_from_raw<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: WASM linear memory FFI boundary; caller passes `[ptr, ptr + len)` within linear memory.
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

#[inline]
pub(crate) fn slice_from_raw_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 || ptr.is_null() {
        &mut []
    } else {
        // SAFETY: WASM linear memory FFI boundary; caller passes non-overlapping `[ptr, ptr + len)` within linear memory.
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }
}

#[inline]
pub(crate) fn str_from_raw<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    let bytes = slice_from_raw(ptr, len);
    core::str::from_utf8(bytes).map_err(|e| format!("[DNAM-VALIDATION] Invalid UTF-8: {e}"))
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_alloc(size: usize, align: usize) -> *mut u8 {
    let safe_align = align.max(8).next_power_of_two();
    let safe_size = size.max(1);
    let Ok(layout) = Layout::from_size_align(safe_size, safe_align) else {
        return core::ptr::null_mut();
    };
    // SAFETY: `layout` has non-zero size and power-of-two alignment.
    unsafe { alloc(layout) }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    let safe_align = align.max(8).next_power_of_two();
    let safe_size = size.max(1);
    if let Ok(layout) = Layout::from_size_align(safe_size, safe_align) {
        // SAFETY: `ptr` was allocated with `dnam_wasm_alloc(size, align)`.
        unsafe { dealloc(ptr, layout) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_ptr() -> *const u8 {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_len() -> usize {
    LAST_ERROR.with(|e| e.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_ptr() -> *const u8 {
    LAST_RESULT.with(|r| r.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_len() -> usize {
    LAST_RESULT.with(|r| r.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_ptr() -> *const u8 {
    LAST_ARENA.with(|a| a.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_len() -> usize {
    LAST_ARENA.with(|a| a.borrow().len())
}
