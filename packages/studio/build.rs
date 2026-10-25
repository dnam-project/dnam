//! Tauri v2 build script for `dnam-studio`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(coverage)");
    #[cfg(feature = "desktop")]
    tauri_build::build();
}
