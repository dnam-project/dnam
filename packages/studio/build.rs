//! Tauri v2 build script for `dnam-studio`.

#![forbid(unsafe_code)]

fn main() {
    println!("cargo::rustc-check-cfg=cfg(coverage)");
    #[cfg(feature = "desktop")]
    {
        let _ = std::fs::create_dir_all("dist-app");
        tauri_build::build();
    }
}
