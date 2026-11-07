//! WebAssembly (`wasm32-unknown-unknown` + `simd128`) safe `extern "C"` ABI wrapping the shared
//! `dnam-js` core (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.runtime.zero-unsafe]`,
//! `ref[sdk.ts.web-audio.realtime-bridge]`).

#[forbid(unsafe_code)]
pub mod abi;
#[allow(unsafe_code)]
pub mod container;
#[allow(unsafe_code)]
pub mod control;
#[allow(unsafe_code)]
pub mod engine;
#[allow(unsafe_code)]
pub mod operators;
#[allow(unsafe_code)]
pub mod provenance;

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_alloc(size: usize, align: usize) -> usize {
    abi::alloc_bytes(size, align)
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_dealloc(ptr: usize, size: usize, align: usize) {
    abi::dealloc_bytes(ptr, size, align);
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_ptr() -> usize {
    abi::last_error_ptr()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_last_error_len() -> usize {
    abi::last_error_len()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_ptr() -> usize {
    abi::result_ptr()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_result_len() -> usize {
    abi::result_len()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_ptr() -> usize {
    abi::arena_ptr()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_arena_len() -> usize {
    abi::arena_len()
}
