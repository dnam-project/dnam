//! Terminal formatting helpers for `dnam` CLI banners and workspace summaries.

use crate::project::{DnamProjectManifest, ModelManifest};
use crate::runner::detect_native_backend_str;

/// Formats the top-level `dnam` CLI banner and hardware SIMD summary.
pub fn format_cli_banner() -> String {
    format!(
        "DNAM Studio & CLI v0.1.0 (Native SIMD Backend: {})\nUse `dnam --help` for commands or `dnam studio` to launch the Tauri v2 Desktop Studio.",
        detect_native_backend_str()
    )
}

/// Formats workspace status (`dnam status`).
pub fn format_workspace_status(manifest: &DnamProjectManifest, models: &[ModelManifest]) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Workspace: {} (author: {}, {} Hz)",
        manifest.project.name, manifest.project.author, manifest.project.sample_rate_hz
    ));
    lines.push(format!(
        "Native SIMD Engine: {}",
        detect_native_backend_str()
    ));
    lines.push(format!("Models ({}):", models.len()));
    for m in models {
        lines.push(format!(
            "  - {} ({}, tier={}, backend={}:{}, send={:+.1} dB, latency={} samples, ranges={}, switches={}, stems={})",
            m.model_id,
            m.title,
            m.tier,
            m.capture.backend,
            m.capture.device_id,
            m.capture.send_gain_db,
            m.capture.calibrated_latency_samples,
            m.ranges.len(),
            m.switches.len(),
            m.stems.len()
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::manifest::{CaptureSettings, ProjectMetadata};

    #[test]
    fn format_banner_and_workspace_status() {
        assert!(format_cli_banner().contains("DNAM Studio & CLI"));
        let manifest = DnamProjectManifest {
            schema_version: "1.0".to_string(),
            project: ProjectMetadata {
                name: "WS".to_string(),
                author: "Auth".to_string(),
                sample_rate_hz: 48_000,
                creator_public_key_hex: String::new(),
                signing_key_path: None,
            },
            models: vec!["amp".to_string()],
        };
        let model = ModelManifest {
            model_id: "amp".to_string(),
            title: "Amp".to_string(),
            description: String::new(),
            tier: "Full".to_string(),
            topology: "auto".to_string(),
            output_mode: "stereo".to_string(),
            capture: CaptureSettings::default(),
            ranges: Vec::new(),
            switches: Vec::new(),
            presets: Vec::new(),
            stems: Vec::new(),
            distribution_policy: None,
        };
        let status = format_workspace_status(&manifest, &[model]);
        assert!(status.contains("Workspace: WS"));
        assert!(status.contains("amp (Amp, tier=Full"));
    }
}
