//! Terminal formatting helpers for `dnam` CLI tables and banners.

use crate::project::{CapturePlan, DnamProjectManifest, ModelManifest};
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
        "Capture: backend={} device={} latency={} samples noise={:.1} dBFS",
        manifest.capture.backend,
        manifest.capture.device_id,
        manifest.capture.calibrated_latency_samples,
        manifest.capture.noise_floor_dbfs
    ));
    lines.push(format!(
        "Native SIMD Engine: {}",
        detect_native_backend_str()
    ));
    lines.push(format!("Models ({}):", models.len()));
    for m in models {
        lines.push(format!(
            "  - {} ({}, tier={}, ranges={}, switches={}, stems={})",
            m.model_id,
            m.title,
            m.tier,
            m.ranges.len(),
            m.switches.len(),
            m.stems.len()
        ));
    }
    lines.join("\n")
}

/// Formats a model capture schedule (`dnam capture plan`).
pub fn format_capture_plan(plan: &CapturePlan) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Capture Plan for '{}' ({} steps):",
        plan.model_id,
        plan.steps.len()
    ));
    for (idx, step) in plan.steps.iter().enumerate() {
        let status = if step.already_captured {
            "[CAPTURED]"
        } else {
            "[PENDING]"
        };
        lines.push(format!(
            "  {:02}. {} {} ({:?}) -> {}",
            idx + 1,
            status,
            step.stem_id,
            step.tier,
            step.prompt
        ));
    }
    lines.join("\n")
}
