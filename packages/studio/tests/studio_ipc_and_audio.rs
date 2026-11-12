//! Integration tests for `dnam-studio` standalone desktop binary, audio device ring buffer,
//! and `.dnam-proj` IPC bridge.

#![forbid(unsafe_code)]

use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use dnam_capture::proj::{DnamProjManifest, build_dnam_proj_container};
use dnam_studio::capture::{AtomicF32RingBuffer, list_backend_devices, list_capture_backends};
use dnam_studio::ipc::{StudioIpcHandler, StudioIpcRequest};
use serde_json::json;

#[test]
fn standalone_studio_binary_and_dnam_proj_ipc_bridge() {
    let bin = env!("CARGO_BIN_EXE_dnam-studio");
    let ok_out = Command::new(bin)
        .arg("--headless-check")
        .output()
        .expect("run dnam-studio --headless-check");
    assert!(ok_out.status.success());
    assert!(
        String::from_utf8_lossy(&ok_out.stdout).contains("dnam-studio desktop ready"),
        "expected readiness banner"
    );

    let err_out = Command::new(bin)
        .arg("--unknown-flag")
        .output()
        .expect("run dnam-studio --unknown-flag");
    assert!(!err_out.status.success());

    let rb = AtomicF32RingBuffer::with_capacity(8);
    assert_eq!(rb.push_slice(&[0.5, -0.5, 0.25]), 3);
    let mut out = [0.0_f32; 3];
    assert_eq!(rb.pop_slice(&mut out), 3);
    assert_eq!(out, [0.5, -0.5, 0.25]);
    assert!(!list_capture_backends().is_empty());
    assert!(!list_backend_devices("software").is_empty());

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("dnam_studio_integ_{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    let proj_path = dir.join("rig.dnam-proj");

    let mut manifest = DnamProjManifest::default_reference();
    manifest.project_meta.project_id = "b7k_ultra".to_string();
    let proj_bytes = build_dnam_proj_container(&manifest, &[]).expect("encode .dnam-proj");
    fs::write(&proj_path, proj_bytes).expect("write .dnam-proj");

    let mut handler = StudioIpcHandler::default();
    let ws_resp = handler
        .handle(
            &StudioIpcRequest {
                command: "get_workspace".to_string(),
                args: json!({}),
            },
            Some(&proj_path),
        )
        .expect("get_workspace from .dnam-proj");
    assert!(ws_resp.ok);
    assert_eq!(ws_resp.data["project_meta"]["project_id"], "b7k_ultra");

    let _ = fs::remove_dir_all(&dir);
}
