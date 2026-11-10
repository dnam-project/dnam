//! Integration tests for `dnam_studio::cli` and `dnam_studio::ipc`.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use dnam_studio::cli::{DnamCli, execute_cli_with_config_dir};
use dnam_studio::ipc::{
    StudioIpcHandler, StudioIpcRequest, official_tauri_plugins, parse_dnam_deep_link,
    validate_isolated_ipc_request,
};
use serde_json::json;

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = env::temp_dir().join(format!("dnam_studio_cli_{tag}_{nanos}"));
    let _ = fs::create_dir_all(&dir);
    dir
}

#[test]
fn test_cli_end_to_end_workflow_and_ipc_handler() {
    let dir = unique_temp_dir("workflow");
    let cfg_dir = dir.join("_test_config").join("dnam");
    let ws = dir.display().to_string();

    let run = |args: &[&str]| -> String {
        let mut full = vec!["dnam", "--workspace", &ws];
        full.extend_from_slice(args);
        let parsed = DnamCli::try_parse_from(full).expect("parse cli");
        execute_cli_with_config_dir(&parsed, &cfg_dir).expect("execute cli")
    };

    let init_out = run(&["init", "--name", "StudioRig", "--author", "peter"]);
    assert!(init_out.contains("Initialized DNAM workspace 'StudioRig'"));

    let add_out = run(&[
        "model",
        "add",
        "--id",
        "Plexi50",
        "--title",
        "Marshall Plexi 50W",
        "--tier",
        "Full",
        "-r",
        "drive=0..10:5",
    ]);
    assert!(add_out.contains("Created model 'Plexi50'"));

    let _ = run(&[
        "control",
        "add-range",
        "--model",
        "Plexi50",
        "--spec",
        "gain:Gain:0..10@6:dB",
    ]);
    let _ = run(&[
        "control",
        "add-switch",
        "--model",
        "Plexi50",
        "--spec",
        "bright:Bright:Off,On@0",
    ]);

    let ctrl_list = run(&["control", "list", "--model", "Plexi50"]);
    assert!(ctrl_list.contains("\"drive\""));
    assert!(ctrl_list.contains("\"gain\""));

    let _ = run(&["control", "remove", "--model", "Plexi50", "--id", "drive"]);

    let _ = run(&[
        "preset",
        "add",
        "--model",
        "Plexi50",
        "--id",
        "crunch",
        "--title",
        "Crunch Rhythm",
        "-c",
        "gain=7.5",
        "-c",
        "bright=On",
    ]);
    let preset_list = run(&["preset", "list", "--model", "Plexi50"]);
    assert!(preset_list.contains("Crunch Rhythm"));

    let show_out = run(&["model", "show", "--id", "Plexi50"]);
    assert!(show_out.contains("Marshall Plexi 50W"));

    let status_out = run(&["status"]);
    assert!(status_out.contains("Plexi50"));

    let studio_out = run(&["studio", "--headless-check"]);
    assert!(studio_out.contains("\"tauriVersion\": \"2.0\""));

    let mut ipc = StudioIpcHandler::new(48_000);
    let resp_param = ipc
        .handle(
            &StudioIpcRequest {
                command: "set_parameter".to_string(),
                args: json!({ "id": "gain", "value": 0.8 }),
            },
            Some(&dir),
        )
        .expect("ipc set_parameter");
    assert!(resp_param.ok);

    let resp_plan = ipc
        .handle(
            &StudioIpcRequest {
                command: "get_capture_plan".to_string(),
                args: json!({ "model_id": "Plexi50" }),
            },
            Some(&dir),
        )
        .expect("ipc get_capture_plan");
    assert!(resp_plan.ok);

    let plugins = official_tauri_plugins();
    assert_eq!(plugins.len(), 8);
    assert!(plugins.iter().any(|p| p.id == "dialog" && p.tier == 1));
    assert!(
        plugins
            .iter()
            .any(|p| p.id == "single-instance" && p.tier == 2)
    );

    let route_model = parse_dnam_deep_link("dnam://models/b7k_ultra").expect("parse model route");
    assert_eq!(route_model.kind, "model");
    assert_eq!(route_model.target, "b7k_ultra");

    assert!(
        validate_isolated_ipc_request(&StudioIpcRequest {
            command: "".to_string(),
            args: json!({}),
        })
        .is_err()
    );
    assert!(
        validate_isolated_ipc_request(&StudioIpcRequest {
            command: "get_capture_plan".to_string(),
            args: json!({ "model_id": "../secret" }),
        })
        .is_err()
    );

    let bin_ok = std::process::Command::new(env!("CARGO_BIN_EXE_dnam"))
        .args(["--workspace", &ws, "studio", "--headless-check"])
        .output()
        .expect("run dnam binary ok");
    assert!(bin_ok.status.success());

    let bin_err = std::process::Command::new(env!("CARGO_BIN_EXE_dnam"))
        .args(["--workspace", "/nonexistent_ws_xyz_999", "status"])
        .output()
        .expect("run dnam binary err");
    assert!(!bin_err.status.success());

    let _ = fs::remove_dir_all(&dir);
}
