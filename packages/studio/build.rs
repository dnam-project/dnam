//! Tauri v2 build script for `dnam-studio`.

fn main() {
    #[cfg(feature = "desktop")]
    tauri_build::build();
}
