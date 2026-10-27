//! File-backed `--request-json` builder and JSONL `TrainingEvent` stream parser for `dnam-train`.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::capture::{compute_pcm_sha256, read_wav};
use crate::error::{StudioError, StudioResult};
use crate::project::{DnamProjectManifest, ModelManifest};

/// Parsed JSONL event emitted by `dnam-train --event-format jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingEventMessage {
    /// Event discriminator (`"training_stage"`, `"training_step"`, `"audit_phase"`, `"training_complete"`, etc.).
    pub event: String,
    /// Full structured payload.
    pub payload: Value,
}

/// Optional execution flags forwarded to `dnam-train` by [`TrainerBridge::run_job_with_options`].
#[derive(Debug, Clone, Default)]
pub struct TrainerJobOptions {
    /// Existing `.dnam-ckpt` checkpoint path for Two-Tier Kronecker resume.
    pub checkpoint_in: Option<std::path::PathBuf>,
    /// Ignore any existing `.dnam-ckpt` and train all 4 stages from scratch.
    pub from_scratch: bool,
    /// Run Phase 1..3 control clamping audit without full gradient optimization.
    pub audit_only: bool,
    /// Force Tier-2 micro-polish during `.dnam-ckpt` resume.
    pub force_tier2: bool,
    /// Epoch count override.
    pub epochs: Option<u32>,
    /// Topology override (`"auto"`, `"core1_only"`, `"full"`).
    pub topology: Option<String>,
    /// Backend override (`"local"`, `"remote"`).
    pub backend: Option<String>,
    /// Device override (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"mps"`, `"tpu"`).
    pub device: Option<String>,
    /// Precision override (`"auto"`, `"fp32"`, `"bf16"`, `"fp16"`).
    pub precision: Option<String>,
    /// Skip `Ed25519` creator signing.
    pub unsigned: bool,
    /// Explicit 64-character hex `Ed25519` creator secret seed.
    pub creator_seed_hex: Option<String>,
}

/// Builds the canonical `TrainingJobRequest` JSON object consumed by `dnam-train --request-json`.
pub fn build_training_request_json(
    workspace: &DnamProjectManifest,
    model: &ModelManifest,
    workspace_root: &Path,
    epochs_override: Option<u32>,
) -> Value {
    let controls_schema: Vec<Value> = model
        .ranges
        .iter()
        .map(|r| {
            let mut obj = json!({
                "id": r.id,
                "name": r.label,
                "kind": "range",
                "declared_range": [r.min, r.max],
                "safe_clamped_range": [r.min, r.max],
                "default": r.default_value,
            });
            if let Some(u) = &r.unit {
                obj["unit"] = json!(u);
            }
            if let Some(cal) = &r.calibration {
                obj["calibration"] = serde_json::to_value(cal).unwrap_or(Value::Null);
            }
            obj
        })
        .chain(model.switches.iter().map(|s| {
            let labels: Vec<String> = s.options.iter().map(|o| o.label.clone()).collect();
            let default_label = s
                .options
                .iter()
                .find(|o| o.index == s.default_index)
                .or_else(|| s.options.first())
                .map(|o| o.label.clone())
                .unwrap_or_else(|| "default".to_string());
            let mut obj = json!({
                "id": s.id,
                "name": s.label,
                "kind": "switch",
                "declared_options": labels,
                "active_options": labels,
                "default": default_label,
            });
            if let Some(cal) = &s.calibration {
                obj["calibration"] = serde_json::to_value(cal).unwrap_or(Value::Null);
            }
            obj
        }))
        .collect();

    let presets: Vec<Value> = model
        .presets
        .iter()
        .map(|p| {
            let mut ctrl_map = Map::new();
            for (k, &v) in &p.ranges {
                ctrl_map.insert(k.clone(), json!(v));
            }
            for (k, &idx) in &p.switches {
                if let Some(sw) = model.switches.iter().find(|s| s.id == *k)
                    && let Some(opt) = sw.options.iter().find(|o| o.index == idx)
                {
                    ctrl_map.insert(k.clone(), json!(opt.label));
                }
            }
            json!({
                "id": p.id,
                "name": p.label,
                "controls": ctrl_map,
            })
        })
        .collect();

    let stem_descriptors: Vec<Value> = model
        .stems
        .iter()
        .map(|st| {
            let dry_abs = workspace_root.join(&st.dry_wav_path);
            let wet_abs = workspace_root.join(&st.wet_wav_path);

            let (dry_sha, dry_sr, dry_ch, dry_frames) = match read_wav(&dry_abs) {
                Ok(w) => {
                    let frames = w.samples.len() / usize::from(w.channels.max(1));
                    let sha = if st.dry_pcm_sha256.len() == 64 {
                        st.dry_pcm_sha256.clone()
                    } else {
                        compute_pcm_sha256(&w.samples)
                    };
                    (sha, w.sample_rate_hz, w.channels, frames.max(1))
                }
                Err(_) => (
                    if st.dry_pcm_sha256.len() == 64 {
                        st.dry_pcm_sha256.clone()
                    } else {
                        st.pcm_sha256.clone()
                    },
                    workspace.project.sample_rate_hz,
                    1,
                    1024,
                ),
            };

            let (wet_sha, wet_sr, wet_ch, wet_frames) = match read_wav(&wet_abs) {
                Ok(w) => {
                    let frames = w.samples.len() / usize::from(w.channels.max(1));
                    let sha = if st.pcm_sha256.len() == 64 {
                        st.pcm_sha256.clone()
                    } else {
                        compute_pcm_sha256(&w.samples)
                    };
                    (sha, w.sample_rate_hz, w.channels, frames.max(1))
                }
                Err(_) => (
                    st.pcm_sha256.clone(),
                    workspace.project.sample_rate_hz,
                    1,
                    dry_frames,
                ),
            };

            let mut controls_meta = Map::new();
            for r in &model.ranges {
                let u = st.normalized_ranges.get(&r.id).copied().unwrap_or(0.5);
                let phys = r.min + u * (r.max - r.min);
                controls_meta.insert(r.id.clone(), json!(phys));
            }
            for s in &model.switches {
                let idx = st.switches.get(&s.id).copied().unwrap_or(s.default_index);
                let label = s
                    .options
                    .iter()
                    .find(|o| o.index == idx)
                    .or_else(|| s.options.first())
                    .map(|o| o.label.clone())
                    .unwrap_or_else(|| "default".to_string());
                controls_meta.insert(s.id.clone(), json!(label));
            }

            json!({
                "stem_id": st.stem_id,
                "controls_meta": controls_meta,
                "input_ref": {
                    "ref_kind": "local_path",
                    "path": dry_abs.display().to_string(),
                    "pcm_sha256": dry_sha,
                    "sample_rate_hz": dry_sr,
                    "channels": dry_ch,
                    "num_samples": dry_frames,
                },
                "target_ref": {
                    "ref_kind": "local_path",
                    "path": wet_abs.display().to_string(),
                    "pcm_sha256": wet_sha,
                    "sample_rate_hz": wet_sr,
                    "channels": wet_ch,
                    "num_samples": wet_frames,
                },
                "noise_floor_dbfs": st.noise_floor_dbfs,
            })
        })
        .collect();

    let epochs = epochs_override.unwrap_or(12);
    let topology = if model.topology.is_empty() {
        "auto"
    } else {
        model.topology.as_str()
    };
    let output_mode = if model.output_mode.is_empty() {
        "stereo"
    } else {
        model.output_mode.as_str()
    };

    let mut config_obj = json!({
        "sample_rate_train_hz": workspace.project.sample_rate_hz,
        "topology": topology,
        "output_mode": output_mode,
        "epochs": epochs,
        "auto_unity_leveling": model.capture.auto_unity_leveling,
        "leveling_metric": if model.capture.leveling_metric.is_empty() {
            "rms_dbfs"
        } else {
            model.capture.leveling_metric.as_str()
        },
        "target_gain_db": model.capture.target_gain_db,
        "blend_mode": if model.capture.blend_mode.is_empty() {
            "auto"
        } else {
            model.capture.blend_mode.as_str()
        },
        "default_blend": model.capture.default_blend,
    });
    if let Some(trails) = model.capture.supports_trails {
        config_obj["supports_trails"] = json!(trails);
    }

    json!({
        "model_id": model.model_id,
        "model_name": model.title,
        "author": workspace.project.author,
        "description": model.description,
        "controls_schema": controls_schema,
        "presets": presets,
        "config": config_obj,
        "stem_descriptors": stem_descriptors,
    })
}

/// Parses newline-delimited JSONL `TrainingEvent` messages emitted by `dnam-train`.
pub fn parse_jsonl_events(jsonl_text: &str) -> StudioResult<Vec<TrainingEventMessage>> {
    let mut events = Vec::new();
    for line in jsonl_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(trimmed)?;
        let event_name = val
            .get("event")
            .or_else(|| val.get("event_type"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        events.push(TrainingEventMessage {
            event: event_name,
            payload: val,
        });
    }
    Ok(events)
}

/// Subprocess bridge for invoking `dnam-train` and streaming live JSONL progress events.
#[derive(Debug, Clone)]
pub struct TrainerBridge {
    worker_program: String,
    leading_args: Vec<String>,
}

impl TrainerBridge {
    /// Creates a [`TrainerBridge`] pointing at a specific command (`dnam-train` or `uv run dnam-train`).
    pub fn new(worker_program: impl Into<String>, leading_args: Vec<String>) -> Self {
        Self {
            worker_program: worker_program.into(),
            leading_args,
        }
    }

    /// Writes `request_json` to `request_path` and executes `dnam-train` with `--event-format jsonl`.
    pub fn run_job(
        &self,
        request_json: &Value,
        request_path: &Path,
        output_dnam: &Path,
        output_ckpt: &Path,
        reports_dir: &Path,
        on_event: impl FnMut(&TrainingEventMessage),
    ) -> StudioResult<Vec<TrainingEventMessage>> {
        self.run_job_with_options(
            request_json,
            request_path,
            output_dnam,
            output_ckpt,
            reports_dir,
            &TrainerJobOptions::default(),
            on_event,
        )
    }

    /// Writes `request_json` to `request_path` and executes `dnam-train` with [`TrainerJobOptions`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_job_with_options(
        &self,
        request_json: &Value,
        request_path: &Path,
        output_dnam: &Path,
        output_ckpt: &Path,
        reports_dir: &Path,
        opts: &TrainerJobOptions,
        mut on_event: impl FnMut(&TrainingEventMessage),
    ) -> StudioResult<Vec<TrainingEventMessage>> {
        self.run_job_inner(
            request_json,
            request_path,
            output_dnam,
            output_ckpt,
            reports_dir,
            opts,
            &mut on_event,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_job_inner(
        &self,
        request_json: &Value,
        request_path: &Path,
        output_dnam: &Path,
        output_ckpt: &Path,
        reports_dir: &Path,
        opts: &TrainerJobOptions,
        on_event: &mut dyn FnMut(&TrainingEventMessage),
    ) -> StudioResult<Vec<TrainingEventMessage>> {
        if let Some(parent) = request_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(request_path, serde_json::to_string_pretty(request_json)?)?;

        let mut cmd = Command::new(&self.worker_program);
        for arg in &self.leading_args {
            cmd.arg(arg);
        }
        cmd.arg("--request-json")
            .arg(request_path)
            .arg("--output-dnam")
            .arg(output_dnam)
            .arg("--output-ckpt")
            .arg(output_ckpt)
            .arg("--reports-dir")
            .arg(reports_dir)
            .arg("--event-format")
            .arg("jsonl");

        if let Some(ckpt_in) = &opts.checkpoint_in
            && ckpt_in.is_file()
            && !opts.from_scratch
            && !opts.audit_only
        {
            cmd.arg("--checkpoint-in").arg(ckpt_in);
        }
        if opts.from_scratch {
            cmd.arg("--from-scratch");
        }
        if opts.audit_only {
            cmd.arg("--audit-only");
        }
        if opts.force_tier2 {
            cmd.arg("--force-tier2");
        }
        if let Some(epochs) = opts.epochs {
            cmd.arg("--epochs").arg(epochs.to_string());
        }
        if let Some(top) = &opts.topology {
            cmd.arg("--topology").arg(top);
        }
        if let Some(backend) = &opts.backend {
            cmd.arg("--backend").arg(backend);
        }
        if let Some(dev) = &opts.device {
            cmd.arg("--device").arg(dev);
        }
        if let Some(prec) = &opts.precision {
            cmd.arg("--precision").arg(prec);
        }
        if opts.unsigned {
            cmd.arg("--unsigned");
        } else if let Some(seed_hex) = &opts.creator_seed_hex {
            cmd.arg("--creator-seed-hex").arg(seed_hex);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| StudioError::Trainer(format!("Failed to spawn dnam-train: {e}")))?;

        let mut collected = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line_str = line?;
                let parsed = parse_jsonl_events(&line_str)?;
                for ev in parsed {
                    on_event(&ev);
                    collected.push(ev);
                }
            }
        }

        let status = child
            .wait()
            .map_err(|e| StudioError::Trainer(format!("Failed waiting on dnam-train: {e}")))?;
        if !status.success() {
            let mut stderr_msg = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                use std::io::Read as _;
                let _ = stderr.read_to_string(&mut stderr_msg);
            }
            return Err(StudioError::Trainer(format!(
                "dnam-train exited with status {status}: {}",
                stderr_msg.trim()
            )));
        }
        Ok(collected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::write_wav_24bit;
    use crate::project::manifest::{
        CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec, PresetSpec,
        ProjectMetadata, StemRecord,
    };
    use std::collections::BTreeMap;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_bridge_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn build_training_request_json_and_parse_jsonl_cover_all_branches() {
        let dir = temp_dir("req_json");
        let dry_rel = "stems/dry.wav";
        let wet_rel = "stems/wet.wav";
        write_wav_24bit(dir.join(dry_rel), &[0.1, -0.1, 0.2, -0.2], 48_000, 1).expect("write dry");
        write_wav_24bit(dir.join(wet_rel), &[0.2, -0.2, 0.4, -0.4], 48_000, 1).expect("write wet");

        let ws = DnamProjectManifest {
            schema_version: "1.0".to_string(),
            project: ProjectMetadata {
                name: "Test".to_string(),
                author: "Tester".to_string(),
                sample_rate_hz: 48_000,
                creator_public_key_hex: String::new(),
                signing_key_path: None,
            },
            models: vec!["amp".to_string()],
        };
        let mut ranges_map = BTreeMap::new();
        ranges_map.insert("gain".to_string(), 0.5_f32);
        let mut sw_map = BTreeMap::new();
        sw_map.insert("bright".to_string(), 1_usize);

        let model = ModelManifest {
            model_id: "amp".to_string(),
            title: "Amp".to_string(),
            tier: "Full".to_string(),
            description: "Desc".to_string(),
            topology: String::new(),
            output_mode: String::new(),
            capture: CaptureSettings::default(),
            ranges: vec![ControlRangeSpec {
                id: "gain".to_string(),
                label: "Gain".to_string(),
                min: 0.0,
                max: 10.0,
                default_value: 5.0,
                unit: Some("dB".to_string()),
                calibration: None,
            }],
            switches: vec![ControlSwitchSpec {
                id: "bright".to_string(),
                label: "Bright".to_string(),
                options: vec![
                    ControlSwitchOptionSpec {
                        index: 0,
                        label: "off".to_string(),
                    },
                    ControlSwitchOptionSpec {
                        index: 1,
                        label: "on".to_string(),
                    },
                ],
                default_index: 0,
                calibration: None,
            }],
            presets: vec![PresetSpec {
                id: "lead".to_string(),
                label: "Lead".to_string(),
                ranges: ranges_map.clone(),
                switches: sw_map.clone(),
            }],
            stems: vec![
                StemRecord {
                    stem_id: "s0".to_string(),
                    dry_wav_path: dry_rel.to_string(),
                    wet_wav_path: wet_rel.to_string(),
                    dry_pcm_sha256: String::new(),
                    pcm_sha256: String::new(),
                    aligned_lag_samples: 0.0,
                    snr_db: 40.0,
                    noise_floor_dbfs: -90.0,
                    clean_pad_db: 0.0,
                    peak_dbfs: -3.0,
                    normalized_ranges: ranges_map.clone(),
                    switches: sw_map.clone(),
                },
                StemRecord {
                    stem_id: "s1_missing".to_string(),
                    dry_wav_path: "missing_dry.wav".to_string(),
                    wet_wav_path: "missing_wet.wav".to_string(),
                    dry_pcm_sha256: "a".repeat(64),
                    pcm_sha256: "b".repeat(64),
                    aligned_lag_samples: 0.0,
                    snr_db: 40.0,
                    noise_floor_dbfs: -90.0,
                    clean_pad_db: 0.0,
                    peak_dbfs: -3.0,
                    normalized_ranges: ranges_map,
                    switches: sw_map,
                },
            ],
        };

        let req = build_training_request_json(&ws, &model, &dir, Some(4));
        assert_eq!(req["config"]["topology"], "auto");
        assert_eq!(req["config"]["output_mode"], "stereo");
        assert_eq!(req["config"]["epochs"], 4);
        assert_eq!(req["presets"][0]["controls"]["bright"], "on");
        assert_eq!(req["stem_descriptors"].as_array().unwrap().len(), 2);

        let events = parse_jsonl_events(
            "\n{\"event\":\"training_stage\",\"stage\":1}\n{\"event_type\":\"training_complete\"}\n",
        )
        .expect("parse events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event, "training_stage");
        assert_eq!(events[1].event, "training_complete");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn trainer_bridge_runs_and_handles_failures() {
        let dir = temp_dir("bridge_run");
        let req_path = dir.join("sub/req.json");
        let out_dnam = dir.join("out.dnam");
        let out_ckpt = dir.join("out.dnam-ckpt");
        let reports_dir = dir.join("reports");
        let ckpt_in = dir.join("in.dnam-ckpt");
        fs::write(&ckpt_in, b"stub").expect("write ckpt_in");

        let bridge_ok = TrainerBridge::new(
            "sh",
            vec![
                "-c".to_string(),
                "printf '{\"event\":\"training_complete\"}\\n'".to_string(),
            ],
        );
        let mut seen = 0usize;
        let res = bridge_ok
            .run_job(
                &json!({"model_id": "m"}),
                &req_path,
                &out_dnam,
                &out_ckpt,
                &reports_dir,
                |_| seen += 1,
            )
            .expect("run_job ok");
        assert_eq!(res.len(), 1);
        assert_eq!(seen, 1);

        // Test with all TrainerJobOptions flags enabled
        let opts_resume = TrainerJobOptions {
            checkpoint_in: Some(ckpt_in),
            from_scratch: false,
            audit_only: false,
            force_tier2: true,
            epochs: Some(2),
            topology: Some("full".to_string()),
            backend: Some("local".to_string()),
            device: Some("cpu".to_string()),
            precision: Some("fp32".to_string()),
            unsigned: false,
            creator_seed_hex: Some("11".repeat(32)),
        };
        bridge_ok
            .run_job_with_options(
                &json!({"model_id": "m"}),
                &req_path,
                &out_dnam,
                &out_ckpt,
                &reports_dir,
                &opts_resume,
                |_| {},
            )
            .expect("run_job_with_options resume");

        let opts_scratch = TrainerJobOptions {
            checkpoint_in: None,
            from_scratch: true,
            audit_only: true,
            force_tier2: false,
            epochs: None,
            topology: None,
            backend: None,
            device: None,
            precision: None,
            unsigned: true,
            creator_seed_hex: None,
        };
        bridge_ok
            .run_job_with_options(
                &json!({"model_id": "m"}),
                &req_path,
                &out_dnam,
                &out_ckpt,
                &reports_dir,
                &opts_scratch,
                |_| {},
            )
            .expect("run_job_with_options scratch");

        // Non-zero exit
        let bridge_fail = TrainerBridge::new(
            "sh",
            vec!["-c".to_string(), "echo boom >&2; exit 2".to_string()],
        );
        let err = bridge_fail
            .run_job(
                &json!({}),
                &req_path,
                &out_dnam,
                &out_ckpt,
                &reports_dir,
                |_| {},
            )
            .expect_err("non-zero exit should error");
        assert!(err.to_string().contains("boom"));

        // Non-existent binary
        let bridge_missing = TrainerBridge::new("/nonexistent/dnam-train-xyz", Vec::new());
        assert!(
            bridge_missing
                .run_job(
                    &json!({}),
                    &req_path,
                    &out_dnam,
                    &out_ckpt,
                    &reports_dir,
                    |_| {}
                )
                .is_err()
        );
    }
}
