//! `dnam cache {status,clear}` subcommand handlers for workspace `.dnam-cache` sidecars.

use std::fs;
use std::path::Path;

use clap::Subcommand;
use serde_json::json;

use crate::error::StudioResult;
use crate::project::DnamProject;

/// Subcommands under `dnam cache`.
#[derive(Debug, Subcommand)]
pub enum CacheAction {
    /// Inspect `.dnam-cache` sidecar size, entry counts, and quota usage across workspace models.
    Status {
        /// Optional model ID filter.
        #[arg(long)]
        model: Option<String>,
    },
    /// Clear `.dnam-cache` sidecar entries for a model or all models in the workspace.
    Clear {
        /// Optional model ID to clear (defaults to all models in workspace).
        #[arg(long)]
        model: Option<String>,
    },
}

fn dir_stats(dir: &Path) -> (usize, u64) {
    let Ok(entries) = fs::read_dir(dir) else {
        return (0, 0);
    };
    let mut count = 0usize;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("dnce")
            && let Ok(meta) = entry.metadata()
            && meta.is_file()
        {
            count += 1;
            bytes = bytes.saturating_add(meta.len());
        }
    }
    (count, bytes)
}

/// Executes `dnam cache {status,clear}`.
pub fn run_cache(workspace: &Path, action: &CacheAction) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    match action {
        CacheAction::Status { model } => {
            let model_ids: Vec<String> = match model {
                Some(mid) => vec![mid.clone()],
                None => proj.manifest().models.clone(),
            };
            let mut total_entries = 0usize;
            let mut total_bytes = 0u64;
            let mut model_reports = Vec::new();
            for mid in &model_ids {
                let cache_dir = proj.model_cache_dir(mid);
                let (entries, bytes) = dir_stats(&cache_dir);
                total_entries += entries;
                total_bytes = total_bytes.saturating_add(bytes);
                model_reports.push(json!({
                    "model_id": mid,
                    "cache_dir": cache_dir.display().to_string(),
                    "entries": entries,
                    "bytes": bytes,
                }));
            }
            let report = json!({
                "workspace": workspace.display().to_string(),
                "total_entries": total_entries,
                "total_bytes": total_bytes,
                "models": model_reports,
            });
            Ok(serde_json::to_string_pretty(&report)?)
        }
        CacheAction::Clear { model } => {
            let model_ids: Vec<String> = match model {
                Some(mid) => vec![mid.clone()],
                None => proj.manifest().models.clone(),
            };
            let mut removed_entries = 0usize;
            let mut reclaimed_bytes = 0u64;
            for mid in &model_ids {
                let cache_dir = proj.model_cache_dir(mid);
                let (entries, bytes) = dir_stats(&cache_dir);
                if cache_dir.exists() {
                    let _ = fs::remove_dir_all(&cache_dir);
                }
                removed_entries += entries;
                reclaimed_bytes = reclaimed_bytes.saturating_add(bytes);
            }
            Ok(format!(
                "Cleared .dnam-cache ({removed_entries} entries, {reclaimed_bytes} B reclaimed)"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cache_status_and_clear_work_across_models() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_cache_cmd_{nanos}"));
        let mut proj = DnamProject::init(&dir, "WS", "Auth").expect("init");
        proj.add_model("m1", "M1", "Full").expect("add m1");
        let cdir = proj.model_cache_dir("m1");
        fs::create_dir_all(&cdir).expect("mkdir cache");
        fs::write(cdir.join("input_stage1_0001.dnce"), vec![0u8; 128]).expect("write dnce");

        let st = run_cache(&dir, &CacheAction::Status { model: None }).expect("status");
        assert!(st.contains("\"total_entries\": 1"));
        assert!(st.contains("\"total_bytes\": 128"));

        let st_m1 = run_cache(
            &dir,
            &CacheAction::Status {
                model: Some("m1".to_string()),
            },
        )
        .expect("status m1");
        assert!(st_m1.contains("\"entries\": 1"));

        let clr = run_cache(
            &dir,
            &CacheAction::Clear {
                model: Some("m1".to_string()),
            },
        )
        .expect("clear m1");
        assert!(clr.contains("1 entries"));
        assert!(!cdir.exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
