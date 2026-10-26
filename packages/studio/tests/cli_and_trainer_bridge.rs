//! Integration tests for `dnam_studio::cli`, `dnam_studio::trainer`, and `dnam_studio::ipc`.

use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use dnam_studio::cli::{DnamCli, execute_cli};
use dnam_studio::ipc::{
    StudioIpcHandler, StudioIpcRequest, official_tauri_plugins, parse_dnam_deep_link,
    validate_isolated_ipc_request,
};
use dnam_studio::trainer::{TrainerEnvManager, parse_jsonl_events};
use serde_json::{Value, json};

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
    let ws = dir.display().to_string();

    let run = |args: &[&str]| -> String {
        let mut full = vec!["dnam", "--workspace", &ws];
        full.extend_from_slice(args);
        let parsed = DnamCli::try_parse_from(full).expect("parse cli");
        execute_cli(&parsed).expect("execute cli")
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

    let cal_out = run(&[
        "capture",
        "calibrate",
        "--backend",
        "software",
        "--device",
        "loopback_default",
    ]);
    assert!(cal_out.contains("excitation_profile"));

    let plan_out = run(&["capture", "plan", "--model", "Plexi50"]);
    assert!(plan_out.contains("stem_00_star_center"));

    let rec_out = run(&[
        "capture",
        "record",
        "--model",
        "Plexi50",
        "--samples",
        "1024",
    ]);
    assert!(rec_out.contains("Recorded and aligned 6 stems"));

    let stem_list = run(&["stem", "list", "--model", "Plexi50"]);
    assert!(stem_list.contains("stem_00_star_center"));

    let train_out = run(&["train", "--model", "Plexi50", "--epochs", "4", "--dry-run"]);
    assert!(train_out.contains("training_request.json"));

    let req_path = dir
        .join("models")
        .join("Plexi50")
        .join("reports")
        .join("training_request.json");
    let req_json: Value =
        serde_json::from_str(&fs::read_to_string(&req_path).expect("read training_request.json"))
            .expect("parse training_request.json");
    assert_eq!(req_json["model_name"], json!("Marshall Plexi 50W"));
    assert!(req_json["controls_schema"].is_array());
    assert!(req_json["stem_descriptors"].is_array());
    assert!(
        req_json["stem_descriptors"][0]["input_ref"]["path"]
            .as_str()
            .is_some()
    );

    let show_out = run(&["model", "show", "--id", "Plexi50"]);
    assert!(show_out.contains("Marshall Plexi 50W"));

    let status_out = run(&["status"]);
    assert!(status_out.contains("Plexi50"));

    let studio_out = run(&["studio", "--headless-check"]);
    assert!(studio_out.contains("\"tauriVersion\": \"2.0\""));

    // Verify StudioIpcHandler against the same workspace
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

    // Verify Tier 1 & Tier 2 Official Tauri v2 Plugins IPC & Deep-Link routes
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

    let route_market =
        parse_dnam_deep_link("dnam://marketplace/plexi_68").expect("parse marketplace route");
    assert_eq!(route_market.kind, "marketplace");
    assert_eq!(route_market.target, "plexi_68");

    let route_open =
        parse_dnam_deep_link("dnam://open?path=/tmp/rig.dnam").expect("parse open route");
    assert_eq!(route_open.kind, "open");
    assert_eq!(route_open.target, "/tmp/rig.dnam");

    assert!(parse_dnam_deep_link("https://example.com").is_err());
    assert!(parse_dnam_deep_link("dnam://models/").is_err());
    assert!(parse_dnam_deep_link("dnam://marketplace/").is_err());
    assert!(parse_dnam_deep_link("dnam://open?path=").is_err());
    assert!(parse_dnam_deep_link("dnam://unknown/route").is_err());

    let resp_plugins = ipc
        .handle(
            &StudioIpcRequest {
                command: "get_plugins".to_string(),
                args: json!({}),
            },
            None,
        )
        .expect("ipc get_plugins");
    assert!(resp_plugins.ok);
    assert_eq!(resp_plugins.data["singleInstanceLocked"], json!(true));

    let resp_dl = ipc
        .handle(
            &StudioIpcRequest {
                command: "parse_deep_link".to_string(),
                args: json!({ "uri": "dnam://models/Plexi50" }),
            },
            None,
        )
        .expect("ipc parse_deep_link");
    assert!(resp_dl.ok);
    assert_eq!(resp_dl.data["target"], json!("Plexi50"));

    let resp_win = ipc
        .handle(
            &StudioIpcRequest {
                command: "save_window_state".to_string(),
                args: json!({
                    "width": 1600,
                    "height": 960,
                    "x": 200,
                    "y": 100,
                    "maximized": true
                }),
            },
            None,
        )
        .expect("ipc save_window_state");
    assert!(resp_win.ok);
    assert_eq!(resp_win.data["width"], json!(1600));

    let resp_win_get = ipc
        .handle(
            &StudioIpcRequest {
                command: "get_window_state".to_string(),
                args: json!({}),
            },
            None,
        )
        .expect("ipc get_window_state");
    assert!(resp_win_get.ok);
    assert_eq!(resp_win_get.data["maximized"], json!(true));

    let resp_notify = ipc
        .handle(
            &StudioIpcRequest {
                command: "send_notification".to_string(),
                args: json!({
                    "title": "DNAM Trainer",
                    "body": "Stage 4 Kronecker export finished"
                }),
            },
            None,
        )
        .expect("ipc send_notification");
    assert!(resp_notify.ok);
    assert_eq!(resp_notify.data["delivered"], json!(true));
    assert_eq!(resp_plugins.data["ipcPattern"], json!("isolation"));
    assert_eq!(resp_plugins.data["freezePrototype"], json!(true));
    assert_eq!(resp_plugins.data["assetProtocolEnabled"], json!(false));

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

    // End-to-end binary execution check (`dnam` CLI entrypoint in `main.rs`)
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

#[test]
fn test_trainer_env_manager_and_jsonl_parser() {
    let dir = unique_temp_dir("trainer_env");
    let mgr = TrainerEnvManager::new(&dir, None);
    let initial = mgr.status();
    assert!(!initial.installed);

    let sample_jsonl = r#"
{"event":"stage_started","stage":"stage1","epochs":4}
{"event":"epoch_completed","epoch":1,"esr_loss":0.012}
"#;
    let parsed = parse_jsonl_events(sample_jsonl).expect("parse jsonl");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].event, "stage_started");
    assert_eq!(parsed[1].event, "epoch_completed");

    let _ = fs::remove_dir_all(&dir);
}
