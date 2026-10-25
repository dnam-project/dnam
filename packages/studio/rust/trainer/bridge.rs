//! File-backed `--request-json` builder and JSONL `TrainingEvent` stream parser for `dnam-train`.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{StudioError, StudioResult};
use crate::project::{DnamProjectManifest, ModelManifest};

/// Parsed JSONL event emitted by `dnam-train --event-format jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingEventMessage {
    /// Event discriminator (`"stage_started"`, `"epoch_completed"`, `"audit_completed"`, `"training_completed"`, etc.).
    pub event: String,
    /// Full structured payload.
    pub payload: Value,
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
            json!({
                "id": r.id,
                "label": r.label,
                "kind": "range",
                "min_val": r.min,
                "max_val": r.max,
                "default_val": r.default_value,
                "unit": r.unit.clone().unwrap_or_default(),
            })
        })
        .chain(model.switches.iter().map(|s| {
            let labels: Vec<String> = s.options.iter().map(|o| o.label.clone()).collect();
            json!({
                "id": s.id,
                "label": s.label,
                "kind": "switch",
                "options": labels,
                "default_index": s.default_index,
            })
        }))
        .collect();

    let stems: Vec<Value> = model
        .stems
        .iter()
        .map(|st| {
            json!({
                "stem_id": st.stem_id,
                "dry_audio": {
                    "kind": "local_wav",
                    "path": workspace_root.join(&st.dry_wav_path).display().to_string(),
                },
                "wet_audio": {
                    "kind": "local_wav",
                    "path": workspace_root.join(&st.wet_wav_path).display().to_string(),
                },
                "normalized_ranges": st.normalized_ranges,
                "switches": st.switches,
                "pcm_sha256": st.pcm_sha256,
            })
        })
        .collect();

    let epochs = epochs_override.unwrap_or(12);
    json!({
        "model_id": model.model_id,
        "title": model.title,
        "author": workspace.project.author,
        "sample_rate_hz": workspace.project.sample_rate_hz,
        "tier": model.tier,
        "epochs": epochs,
        "controls_schema": controls_schema,
        "stems": stems,
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
        mut on_event: impl FnMut(&TrainingEventMessage),
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
            .arg("jsonl")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

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
            return Err(StudioError::Trainer(format!(
                "dnam-train exited with status {status}"
            )));
        }
        Ok(collected)
    }
}
