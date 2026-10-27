//! Terminal formatting helpers and interactive `dialoguer` prompts for `dnam` CLI tables, panels, and capture steps.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static FORCE_NONINTERACTIVE: AtomicBool = AtomicBool::new(false);

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

/// Formats a warning panel when Calibration Step 2/2 (Engaged Hot-Corner) detects return clipping.
pub fn format_step2_clipping_warning(peak_dbfs: f32, clipped_fraction: f32) -> String {
    let pct = (clipped_fraction * 100.0).clamp(0.0, 100.0);
    format!(
        "╭─── ⚠ Calibration Step 2/2 Warning — Return Clipping Detected ───────────────╮\n\
         • Measured Hot-Corner Return Peak: {peak_dbfs:+.2} dBFS ({pct:.2}% clipped samples).\n\
         • Input send gain remains locked to preserve target drive headroom.\n\
         • If your signal chain has a clean output level adjustment (or audio interface\n\
           return input trim/pad), lower it (e.g. -40 dB with `--clean-pad-db -40`)\n\
           and repeat calibration.\n\
         ╰──────────────────────────────────────────────────────────────────────────────╯"
    )
}

/// Formats a warning panel when a recorded stem clips the raw USB return (`>= -0.05 dBFS`).
pub fn format_stem_clipping_warning(
    stem_id: &str,
    peak_dbfs: f32,
    clipped_fraction: f32,
    clean_pad_db: f32,
) -> String {
    let pct = (clipped_fraction * 100.0).clamp(0.0, 100.0);
    format!(
        "╭─── ⚠ Stem Return Clipping Detected: {stem_id} ──────────────────────────────╮\n\
         • Raw USB Return Peak: {peak_dbfs:+.2} dBFS ({pct:.2}% clipped samples, pad={clean_pad_db:+.1} dB).\n\
         • Lower your device's Clean Level (e.g. -40 dB) for this hot stem and pass\n\
           `--clean-pad-db -40`: `dnam capture` will automatically restore +40.0 dB\n\
           into an unclipped 32-bit IEEE-754 float WAV without lowering star_center SNR.\n\
         ╰──────────────────────────────────────────────────────────────────────────────╯"
    )
}

/// Prompts the user to re-record a clipped stem with a Clean Level pad (`Some(pad_db)`) or keep it (`None`).
pub fn prompt_stem_clean_pad_on_clipping(
    stem_id: &str,
    peak_dbfs: f32,
    clipped_fraction: f32,
    clean_pad_db: f32,
    auto_yes: bool,
) -> StudioResult<Option<f32>> {
    let warning = format_stem_clipping_warning(stem_id, peak_dbfs, clipped_fraction, clean_pad_db);
    eprintln!("{warning}");
    if auto_yes {
        return Ok(None);
    }
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Stem clipping pad prompt requires interactive terminal or --yes".to_string(),
        ));
    }
    let items = &[
        "Re-record with -40.0 dB Clean Level pad (+40.0 dB 32-bit float restore)",
        "Re-record with -20.0 dB Clean Level pad (+20.0 dB 32-bit float restore)",
        "Re-record with -12.0 dB Clean Level pad (+12.0 dB 32-bit float restore)",
        "Keep current recording as-is",
    ];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(format!(
            "Stem '{stem_id}' clipped USB return — select Clean Level pad action"
        ))
        .items(items)
        .default(0)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Stem clipping pad prompt cancelled: {e}")))?;
    Ok(match selection {
        0 => Some(-40.0),
        1 => Some(-20.0),
        2 => Some(-12.0),
        _ => None,
    })
}

/// Computes the recommended Clean Level pad in dB (`<= 0.0`) so the raw USB return peaks near
/// `-3.0 dBFS` (maximizing unclipped dynamic range and SNR).
pub fn recommend_clean_pad_db(raw_peak_dbfs: f32, active_clean_pad_db: f32) -> f32 {
    let restored_peak_dbfs = raw_peak_dbfs + active_clean_pad_db.abs();
    if restored_peak_dbfs <= -3.0 {
        0.0
    } else {
        -(restored_peak_dbfs + 3.0).ceil()
    }
}

/// Returns `true` when an active Clean Level pad leaves more than `12.0 dB` of unused USB headroom
/// (`raw_peak_dbfs < -12.0 dBFS`) and raising Clean Level to [`recommend_clean_pad_db`] improves
/// dynamic range by at least `6.0 dB`.
pub fn is_clean_pad_excessive(raw_peak_dbfs: f32, active_clean_pad_db: f32) -> bool {
    if active_clean_pad_db.abs() <= 1e-4 || raw_peak_dbfs >= -12.0 {
        return false;
    }
    let rec = recommend_clean_pad_db(raw_peak_dbfs, active_clean_pad_db);
    (rec - (-active_clean_pad_db.abs())) >= 6.0
}

/// Formats an advisory panel when `clean_pad_db` is excessive (`raw_peak_dbfs < -12.0 dBFS`),
/// showing the exact recommended Clean Level in dB to maximize USB dynamic range.
pub fn format_excessive_clean_pad_warning(
    stem_id: &str,
    raw_peak_dbfs: f32,
    active_clean_pad_db: f32,
) -> String {
    let restored_peak_dbfs = raw_peak_dbfs + active_clean_pad_db.abs();
    let rec_pad_db = recommend_clean_pad_db(raw_peak_dbfs, active_clean_pad_db);
    let expected_usb_peak_dbfs = restored_peak_dbfs + rec_pad_db;
    let snr_gain_db = rec_pad_db - (-active_clean_pad_db.abs());
    format!(
        "╭─── 💡 Excessive Clean Level Pad Detected: {stem_id} ─────────────────────────╮\n\
         • Raw USB Peak: {raw_peak_dbfs:+.2} dBFS at Clean Level = {active_clean_pad_db:+.1} dB (f32 Peak: {restored_peak_dbfs:+.2} dBFS).\n\
         • Recommended Clean Level: {rec_pad_db:+.1} dB (`--clean-pad-db {rec_pad_db:.1}`)\n\
           -> Places USB return peak at {expected_usb_peak_dbfs:+.2} dBFS (+{snr_gain_db:.1} dB higher dynamic range/SNR).\n\
         ╰──────────────────────────────────────────────────────────────────────────────╯"
    )
}

/// Displays the excessive Clean Level pad advisory panel and, when running interactively (`!auto_yes`),
/// offers to re-record the stem at the recommended Clean Level (`Some(rec_pad_db)`) or keep it (`None`).
pub fn prompt_stem_excessive_clean_pad(
    stem_id: &str,
    raw_peak_dbfs: f32,
    active_clean_pad_db: f32,
    auto_yes: bool,
) -> StudioResult<Option<f32>> {
    let rec_pad_db = recommend_clean_pad_db(raw_peak_dbfs, active_clean_pad_db);
    let warning = format_excessive_clean_pad_warning(stem_id, raw_peak_dbfs, active_clean_pad_db);
    eprintln!("{warning}");
    if auto_yes {
        return Ok(None);
    }
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Excessive clean pad prompt requires interactive terminal or --yes".to_string(),
        ));
    }
    let re_record_label = format!(
        "Re-record with recommended {rec_pad_db:+.1} dB Clean Level pad (+{:.1} dB dynamic range)",
        rec_pad_db - (-active_clean_pad_db.abs())
    );
    let items = [re_record_label.as_str(), "Keep current recording as-is"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(format!(
            "Stem '{stem_id}' has {:.1} dB unused headroom — adjust Clean Level to {rec_pad_db:+.1} dB?",
            -3.0 - raw_peak_dbfs
        ))
        .items(items)
        .default(0)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Excessive clean pad prompt cancelled: {e}")))?;
    Ok(if selection == 0 {
        Some(rec_pad_db)
    } else {
        None
    })
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
    format_stem_prompt_panel_with_pad(
        model,
        step,
        stem_index,
        total_stems,
        prev_ranges,
        prev_switches,
        None,
        None,
    )
}

/// Formats the per-stem knob/switch delta panel including an optional `Clean Level` hardware pad row.
#[allow(clippy::too_many_arguments)]
pub fn format_stem_prompt_panel_with_pad(
    model: &ModelManifest,
    step: &CapturePlanStep,
    stem_index: usize,
    total_stems: usize,
    prev_ranges: Option<&BTreeMap<String, f32>>,
    prev_switches: Option<&BTreeMap<String, usize>>,
    prev_clean_pad_db: Option<f32>,
    target_clean_pad_db: Option<f32>,
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
    if let Some(target_pad) = target_clean_pad_db {
        let prev_pad_s = prev_clean_pad_db
            .map(|p| format!("{p:+.1} dB"))
            .unwrap_or_else(|| "—".to_string());
        let target_pad_s = format!("{target_pad:+.1} dB");
        let pad_changed = prev_clean_pad_db.is_none_or(|p| (p - target_pad).abs() > 1e-3);
        let pad_action = if pad_changed {
            "TURN / FLIP"
        } else {
            "Unchanged"
        };
        lines.push(format!(
            "│ {:<16} {:<12} {:<16} {}",
            "Clean Level", prev_pad_s, target_pad_s, pad_action
        ));
    }
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

/// Maps a zero-based `dialoguer::Select` menu index to a [`StemPromptAction`].
pub const fn map_stem_prompt_selection(stem_index: usize, selection: usize) -> StemPromptAction {
    if stem_index > 1 {
        match selection {
            1 => StemPromptAction::Retake,
            2 => StemPromptAction::Skip,
            _ => StemPromptAction::Record,
        }
    } else {
        match selection {
            1 => StemPromptAction::Skip,
            _ => StemPromptAction::Record,
        }
    }
}

/// Displays Calibration Step 1/2 guidance and waits for confirmation via `dialoguer::Confirm` when `!auto_yes`.
pub fn prompt_calibration_step1(setup_instructions: &str, auto_yes: bool) -> StudioResult<()> {
    let panel = format_calibration_step1_panel(setup_instructions);
    eprintln!("{panel}");
    if auto_yes {
        return Ok(());
    }
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Calibration Step 1 prompt requires interactive terminal or --yes".to_string(),
        ));
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
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Calibration Step 2 prompt requires interactive terminal or --yes".to_string(),
        ));
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

/// Displays the Step-2 return clipping warning panel and, when running interactively (`!auto_yes`),
/// prompts the user (`Y/N`) whether to repeat calibration immediately without re-running the command.
pub fn prompt_repeat_calibration_on_clipping(
    peak_dbfs: f32,
    clipped_fraction: f32,
    auto_yes: bool,
) -> StudioResult<bool> {
    let warning = format_step2_clipping_warning(peak_dbfs, clipped_fraction);
    eprintln!("{warning}");
    if auto_yes {
        return Ok(false);
    }
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Calibration clipping retry prompt requires interactive terminal or --yes".to_string(),
        ));
    }
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt(
            "Repeat calibration now after adjusting your signal chain's clean output level?",
        )
        .default(true)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Calibration retry prompt cancelled: {e}")))
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
    prompt_next_stem_action_with_pad(
        model,
        step,
        stem_index,
        total_stems,
        prev_ranges,
        prev_switches,
        None,
        None,
        auto_yes,
        countdown_sec,
    )
}

/// Displays the control-delta panel (including `Clean Level` pad delta) and prompts for [`StemPromptAction`].
#[allow(clippy::too_many_arguments)]
pub fn prompt_next_stem_action_with_pad(
    model: &ModelManifest,
    step: &CapturePlanStep,
    stem_index: usize,
    total_stems: usize,
    prev_ranges: Option<&BTreeMap<String, f32>>,
    prev_switches: Option<&BTreeMap<String, usize>>,
    prev_clean_pad_db: Option<f32>,
    target_clean_pad_db: Option<f32>,
    auto_yes: bool,
    countdown_sec: f32,
) -> StudioResult<StemPromptAction> {
    let panel = format_stem_prompt_panel_with_pad(
        model,
        step,
        stem_index,
        total_stems,
        prev_ranges,
        prev_switches,
        prev_clean_pad_db,
        target_clean_pad_db,
    );
    eprintln!("{panel}");

    if countdown_sec > 0.0 {
        std::thread::sleep(Duration::from_secs_f32(countdown_sec.min(10.0)));
        return Ok(StemPromptAction::Record);
    }
    if auto_yes {
        return Ok(StemPromptAction::Record);
    }
    if FORCE_NONINTERACTIVE.load(Ordering::Relaxed)
        || std::env::var_os("DNAM_NONINTERACTIVE").is_some()
    {
        return Err(StudioError::Audio(
            "Stem capture prompt requires interactive terminal or --yes".to_string(),
        ));
    }

    let items: &[&str] = if stem_index > 1 {
        &["Record stem", "Retake previous stem", "Skip stem"]
    } else {
        &["Record stem", "Skip stem"]
    };
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt(format!(
            "Stem [{stem_index}/{total_stems}] '{}' action",
            step.stem_id
        ))
        .items(items)
        .default(0)
        .interact()
        .map_err(|e| StudioError::Audio(format!("Stem capture prompt cancelled: {e}")))?;
    Ok(map_stem_prompt_selection(stem_index, selection))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::manifest::{
        CaptureSettings, ControlRangeSpec, ControlSwitchOptionSpec, ControlSwitchSpec,
        ProjectMetadata,
    };
    use crate::project::planner::CaptureTier;

    #[test]
    fn format_panels_and_auto_yes_prompts() {
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
            ranges: vec![ControlRangeSpec {
                id: "gain".to_string(),
                label: "Gain".to_string(),
                min: 0.0,
                max: 10.0,
                default_value: 5.0,
                unit: None,
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
            presets: Vec::new(),
            stems: Vec::new(),
        };
        let status = format_workspace_status(&manifest, std::slice::from_ref(&model));
        assert!(status.contains("Workspace: WS"));
        assert!(status.contains("amp (Amp"));

        let mut ranges = BTreeMap::new();
        ranges.insert("gain".to_string(), 0.5_f32);
        let mut switches = BTreeMap::new();
        switches.insert("bright".to_string(), 0_usize);
        let step1 = CapturePlanStep {
            stem_id: "stem_00".to_string(),
            tier: CaptureTier::StarCenter,
            normalized_ranges: ranges.clone(),
            switches: switches.clone(),
            prompt: "Set Gain=5.0".to_string(),
            already_captured: true,
        };
        let mut ranges2 = BTreeMap::new();
        ranges2.insert("gain".to_string(), 1.0_f32);
        let step2 = CapturePlanStep {
            stem_id: "stem_01".to_string(),
            tier: CaptureTier::StarAxis,
            normalized_ranges: ranges2,
            switches,
            prompt: "Set Gain=10.0".to_string(),
            already_captured: false,
        };
        let plan = CapturePlan {
            model_id: "amp".to_string(),
            steps: vec![step1.clone(), step2.clone()],
        };
        let plan_txt = format_capture_plan(&plan);
        assert!(plan_txt.contains("[CAPTURED]"));
        assert!(plan_txt.contains("[PENDING]"));

        // Empty model stem prompt panel (`(fixed)`)
        let empty_model = ModelManifest {
            ranges: Vec::new(),
            switches: Vec::new(),
            ..model.clone()
        };
        let empty_panel = format_stem_prompt_panel(&empty_model, &step1, 1, 2, None, None);
        assert!(empty_panel.contains("(fixed)"));

        // Delta stem prompt panel
        let delta_panel = format_stem_prompt_panel(
            &model,
            &step2,
            2,
            2,
            Some(&step1.normalized_ranges),
            Some(&step1.switches),
        );
        assert!(delta_panel.contains("TURN / FLIP"));
        assert!(delta_panel.contains("Unchanged"));

        // Calibration step1 & step2 panels + auto_yes prompts + non-TTY error path
        FORCE_NONINTERACTIVE.store(true, Ordering::Relaxed);
        prompt_calibration_step1("Connect Send 1 -> Input", true).expect("step1 auto_yes");
        assert!(prompt_calibration_step1("Connect Send 1 -> Input", false).is_err());
        let s1_metrics = Step1CalibrationMetrics {
            backend: "software".to_string(),
            device_id: "loopback_default".to_string(),
            send_gain_db: -6.0,
            latency_samples: 12,
            unity_ref_dbfs: -18.0,
            noise_floor_dbfs: -92.0,
            snr_db: 50.0,
        };
        prompt_calibration_step2(&s1_metrics, true).expect("step2 auto_yes");
        assert!(prompt_calibration_step2(&s1_metrics, false).is_err());

        let clip_warn = format_step2_clipping_warning(0.0, 0.025);
        assert!(clip_warn.contains("Return Clipping Detected"));
        assert!(clip_warn.contains("clean output level adjustment"));
        assert!(!prompt_repeat_calibration_on_clipping(0.0, 0.025, true).expect("clip auto_yes"));
        assert!(prompt_repeat_calibration_on_clipping(0.0, 0.025, false).is_err());

        let stem_warn = format_stem_clipping_warning("stem_hot", 0.0, 0.03, 0.0);
        assert!(stem_warn.contains("Stem Return Clipping Detected: stem_hot"));
        assert!(stem_warn.contains("--clean-pad-db -40"));
        assert_eq!(
            prompt_stem_clean_pad_on_clipping("stem_hot", 0.0, 0.03, 0.0, true)
                .expect("stem clip auto_yes"),
            None
        );
        assert!(prompt_stem_clean_pad_on_clipping("stem_hot", 0.0, 0.03, 0.0, false).is_err());

        assert_eq!(recommend_clean_pad_db(-23.35, -40.0), -20.0);
        assert_eq!(recommend_clean_pad_db(-45.0, -40.0), 0.0);
        assert!(is_clean_pad_excessive(-23.35, -40.0));
        assert!(!is_clean_pad_excessive(-3.35, -20.0));
        assert!(!is_clean_pad_excessive(-20.0, 0.0));
        let exc_warn = format_excessive_clean_pad_warning("stem_hot", -23.35, -40.0);
        assert!(exc_warn.contains("Recommended Clean Level: -20.0 dB"));
        assert_eq!(
            prompt_stem_excessive_clean_pad("stem_hot", -23.35, -40.0, true)
                .expect("excessive pad auto_yes"),
            None
        );
        assert!(prompt_stem_excessive_clean_pad("stem_hot", -23.35, -40.0, false).is_err());

        // prompt_next_stem_action with countdown, auto_yes, non-TTY, and map_stem_prompt_selection
        let act_cd = prompt_next_stem_action(&model, &step1, 1, 2, None, None, false, 0.001)
            .expect("countdown");
        assert_eq!(act_cd, StemPromptAction::Record);
        let act_yes =
            prompt_next_stem_action(&model, &step1, 1, 2, None, None, true, 0.0).expect("auto_yes");
        assert_eq!(act_yes, StemPromptAction::Record);
        assert!(prompt_next_stem_action(&model, &step1, 1, 2, None, None, false, 0.0).is_err());
        assert!(prompt_next_stem_action(&model, &step2, 2, 2, None, None, false, 0.0).is_err());

        assert_eq!(map_stem_prompt_selection(1, 0), StemPromptAction::Record);
        assert_eq!(map_stem_prompt_selection(1, 1), StemPromptAction::Skip);
        assert_eq!(map_stem_prompt_selection(2, 0), StemPromptAction::Record);
        assert_eq!(map_stem_prompt_selection(2, 1), StemPromptAction::Retake);
        assert_eq!(map_stem_prompt_selection(2, 2), StemPromptAction::Skip);
    }
}
