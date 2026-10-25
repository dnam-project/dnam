//! Terminal formatting helpers and interactive `dialoguer` prompts for `dnam` CLI tables, panels, and capture steps.

use std::collections::BTreeMap;
use std::time::Duration;

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Select};

use crate::capture::Step1CalibrationMetrics;
use crate::error::{StudioError, StudioResult};
use crate::project::{
    CapturePlan, CapturePlanStep, DnamProjectManifest, ModelManifest, compute_control_deltas,
};
use crate::runner::detect_native_backend_str;

/// Interactive action chosen by the user at a stem capture prompt (`dnam capture record`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemPromptAction {
    /// Proceed with recording and aligning the current stem.
    Record,
    /// Step back and re-record the previous stem.
    Retake,
    /// Skip the current stem without recording.
    Skip,
}

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
        "Capture: backend={} device={} send={:+.1} dB latency={} samples noise={:.1} dBFS",
        manifest.capture.backend,
        manifest.capture.device_id,
        manifest.capture.send_gain_db,
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

/// Formats the Calibration Step 1/2 (Bypass / Dry-Through Unity Reference) guidance panel.
pub fn format_calibration_step1_panel(setup_instructions: &str) -> String {
    format!(
        "╭─── DNAM Capture Calibration — Step 1/2 ──────────────────────────────────────╮\n\
         {setup_instructions}\n\n\
         Step 1/2 — Bypass / Dry-Through Unity Reference & Auto-Pad Probe\n\
         • Put the pedal/amp in Bypass (or for Delay / Reverb / Echo pedals without\n\
           hardware bypass, set Mix = 0% dry-only and Feedback = 0% so there are zero\n\
           echo repeats).\n\
         • Automatically probes and locks unclipped Send Gain / Pad (`send_gain_db`),\n\
           round-trip latency (`calibrated_latency_samples`), unity reference level\n\
           (`unity_ref_dbfs`), and baseline noise floor.\n\
         ╰──────────────────────────────────────────────────────────────────────────────╯"
    )
}

/// Formats the Calibration Step 2/2 (Engaged Hot-Corner & System-ID Probe) guidance panel.
pub fn format_calibration_step2_panel(step1: &Step1CalibrationMetrics) -> String {
    format!(
        "╭─── DNAM Capture Calibration — Step 2/2 ──────────────────────────────────────╮\n\
         ✓ Step 1 Locked: send={:+.1} dB | latency={} samples | unity_ref={:+.2} dBFS | noise={:.1} dBFS\n\n\
         Step 2/2 — Engaged Hot-Corner & System-Identification Probe\n\
         • Engage the pedal/device at its hottest setting (at locked {:+.1} dB Send Gain):\n\
           - Drive / Preamp / Amp: Set Drive / Gain / Level to maximum.\n\
           - Delay / Echo / Reverb: Set Feedback / Repeats and Mix to maximum.\n\
         • Measures soft-knee onset, saturation/clipping rail, resonant peaks, LFO rate,\n\
           and reverb/delay tail across the full input range to synthesize the model's\n\
           probe-optimized dry stem.\n\
         ╰──────────────────────────────────────────────────────────────────────────────╯",
        step1.send_gain_db,
        step1.latency_samples,
        step1.unity_ref_dbfs,
        step1.noise_floor_dbfs,
        step1.send_gain_db,
    )
}

/// Formats the per-stem knob/switch delta panel (`Control | Previous | Target Setting | Action`).
pub fn format_stem_prompt_panel(
    model: &ModelManifest,
    step: &CapturePlanStep,
    stem_index: usize,
    total_stems: usize,
    prev_ranges: Option<&BTreeMap<String, f32>>,
    prev_switches: Option<&BTreeMap<String, usize>>,
) -> String {
    let deltas = compute_control_deltas(
        model,
        prev_ranges,
        prev_switches,
        &step.normalized_ranges,
        &step.switches,
    );
    let mut lines = Vec::new();
    lines.push(format!(
        "╭─── Stem [{stem_index}/{total_stems}]: {} ({:?}) ───",
        step.stem_id, step.tier
    ));
    lines.push(format!("│ {}", step.prompt));
    lines.push(format!(
        "│ {:<16} {:<12} {:<16} {}",
        "Control", "Previous", "Target Setting", "Action"
    ));
    lines.push("│ ────────────────────────────────────────────────────────────".to_string());
    if deltas.is_empty() {
        lines.push(format!(
            "│ {:<16} {:<12} {:<16} {}",
            "(fixed)", "—", "default", "Hold steady"
        ));
    } else {
        for d in &deltas {
            let prev_s = d.previous_value.as_deref().unwrap_or("—");
            let action_s = if d.changed {
                "TURN / FLIP"
            } else {
                "Unchanged"
            };
            lines.push(format!(
                "│ {:<16} {:<12} {:<16} {}",
                d.label, prev_s, d.target_value, action_s
            ));
        }
    }
    lines.push("╰─────────────────────────────────────────────────────────────".to_string());
    lines.join("\n")
}

/// Displays Calibration Step 1/2 guidance and waits for confirmation via `dialoguer::Confirm` when `!auto_yes`.
pub fn prompt_calibration_step1(setup_instructions: &str, auto_yes: bool) -> StudioResult<()> {
    let panel = format_calibration_step1_panel(setup_instructions);
    eprintln!("{panel}");
    if auto_yes {
        return Ok(());
    }
    let confirmed = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Device is in Bypass / 0% Mix Dry-Through — ready to probe Step 1?")
        .default(true)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Calibration Step 1 prompt cancelled: {e}")))?;
    if !confirmed {
        return Err(StudioError::Audio(
            "Calibration cancelled by user at Step 1".to_string(),
        ));
    }
    Ok(())
}

/// Displays Calibration Step 2/2 guidance and waits for confirmation via `dialoguer::Confirm` when `!auto_yes`.
pub fn prompt_calibration_step2(
    step1: &Step1CalibrationMetrics,
    auto_yes: bool,
) -> StudioResult<()> {
    let panel = format_calibration_step2_panel(step1);
    eprintln!("{panel}");
    if auto_yes {
        return Ok(());
    }
    let confirmed = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Device is engaged at its Hot Corner — ready to probe Step 2?")
        .default(true)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Calibration Step 2 prompt cancelled: {e}")))?;
    if !confirmed {
        return Err(StudioError::Audio(
            "Calibration cancelled by user at Step 2".to_string(),
        ));
    }
    Ok(())
}

/// Displays the control-delta panel for `step` and prompts for [`StemPromptAction`] via `dialoguer::Select`.
#[allow(clippy::too_many_arguments)]
pub fn prompt_next_stem_action(
    model: &ModelManifest,
    step: &CapturePlanStep,
    stem_index: usize,
    total_stems: usize,
    prev_ranges: Option<&BTreeMap<String, f32>>,
    prev_switches: Option<&BTreeMap<String, usize>>,
    auto_yes: bool,
    countdown_sec: f32,
) -> StudioResult<StemPromptAction> {
    let panel = format_stem_prompt_panel(
        model,
        step,
        stem_index,
        total_stems,
        prev_ranges,
        prev_switches,
    );
    eprintln!("{panel}");

    if countdown_sec > 0.0 {
        std::thread::sleep(Duration::from_secs_f32(countdown_sec.min(10.0)));
        return Ok(StemPromptAction::Record);
    }
    if auto_yes {
        return Ok(StemPromptAction::Record);
    }

    if stem_index > 1 {
        let items = ["Record stem", "Retake previous stem", "Skip stem"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Stem [{stem_index}/{total_stems}] '{}' action",
                step.stem_id
            ))
            .items(items)
            .default(0)
            .interact()
            .map_err(|e| StudioError::Audio(format!("Stem capture prompt cancelled: {e}")))?;
        Ok(match selection {
            1 => StemPromptAction::Retake,
            2 => StemPromptAction::Skip,
            _ => StemPromptAction::Record,
        })
    } else {
        let items = ["Record stem", "Skip stem"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Stem [{stem_index}/{total_stems}] '{}' action",
                step.stem_id
            ))
            .items(items)
            .default(0)
            .interact()
            .map_err(|e| StudioError::Audio(format!("Stem capture prompt cancelled: {e}")))?;
        Ok(match selection {
            1 => StemPromptAction::Skip,
            _ => StemPromptAction::Record,
        })
    }
}
