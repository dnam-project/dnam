//! Workspace, model, control, preset, stem, inspect, render, and `dnam studio` subcommand handlers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Subcommand;
use serde_json::json;

use crate::cli::format;
use crate::error::{StudioError, StudioResult};
use crate::ipc::official_tauri_plugins;
use crate::project::{
    DnamProject, ModelManifest, parse_compact_range_spec, parse_compact_switch_spec,
};
use crate::runner::{
    detect_native_backend_str, inspect_container, inspect_container_with_crypto,
    render_wav_with_crypto, render_wav_with_options,
};

/// Subcommands under `dnam model`.
#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// Add a new model to `models/<id>/model.toml`.
    Add {
        /// Unique model identifier.
        #[arg(long)]
        id: String,
        /// Human-readable title (defaults to humanized `id`).
        #[arg(long, default_value = "")]
        title: String,
        /// Target tier (`Lite`, `Full`, `Ultra`).
        #[arg(long, default_value = "Full")]
        tier: String,
        /// Inline continuous range controls (`--range drive=0..10:5`).
        #[arg(long = "range", short = 'r')]
        ranges: Vec<String>,
        /// Inline discrete switch controls (`--switch attack=cut,flat,boost:flat`).
        #[arg(long = "switch", short = 's')]
        switches: Vec<String>,
    },
    /// List all models in the workspace.
    List,
    /// Show details of a model.
    Show {
        /// Model identifier.
        #[arg(long)]
        id: String,
    },
    /// Update model metadata, output mode, or topology.
    Update {
        /// Model identifier.
        #[arg(long)]
        id: String,
        /// Updated title.
        #[arg(long)]
        title: Option<String>,
        /// Updated tier (`Lite`, `Full`, `Ultra`).
        #[arg(long)]
        tier: Option<String>,
        /// Updated description.
        #[arg(long)]
        description: Option<String>,
        /// Updated output mode (`stereo`, `mono`, `dual_mono`).
        #[arg(long)]
        output_mode: Option<String>,
        /// Updated topology (`auto`, `core1_only`, `full`).
        #[arg(long)]
        topology: Option<String>,
        /// Allowed container protection profiles (`open`, `device_allowlisted`, `enclave_sealed`).
        #[arg(long = "allow-profile")]
        allow_profiles: Vec<String>,
        /// Allowed OEM vendors (`darkglass`, `darkglass:anagram_rk3582`).
        #[arg(long = "allow-oem")]
        allow_oems: Vec<String>,
    },
    /// Remove a model from the workspace.
    Remove {
        /// Model identifier.
        #[arg(long)]
        id: String,
        /// Delete `models/<id>` directory from disk.
        #[arg(long, default_value_t = true)]
        delete_files: bool,
    },
}

/// Subcommands under `dnam control`.
#[derive(Debug, Subcommand)]
pub enum ControlAction {
    /// Add a range (`--range`) or switch (`--switch`) control to a model.
    Add {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Compact range specification (`drive=0..10:5` or `id:Label:min..max@default[:unit]`).
        #[arg(long = "range", short = 'r')]
        range: Option<String>,
        /// Compact switch specification (`attack=cut,flat,boost:flat` or `id:Label:Opt0,Opt1@default_idx`).
        #[arg(long = "switch", short = 's')]
        switch: Option<String>,
    },
    /// Add a continuous range control using compact spec `"id=min..max[:default]"` or `"id:Label:min..max@default[:unit]"`.
    AddRange {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Compact range specification.
        #[arg(long)]
        spec: String,
    },
    /// Add a discrete switch control using compact spec `"id=opt0,opt1[:default]"` or `"id:Label:Opt0,Opt1@default_idx"`.
    AddSwitch {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Compact switch specification.
        #[arg(long)]
        spec: String,
    },
    /// List controls declared on a model.
    List {
        /// Target model identifier.
        #[arg(long)]
        model: String,
    },
    /// Update an existing control on a model.
    Update {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Control identifier.
        #[arg(long)]
        id: String,
        /// Updated label.
        #[arg(long)]
        label: Option<String>,
        /// Updated minimum value (for range controls).
        #[arg(long)]
        min: Option<f32>,
        /// Updated maximum value (for range controls).
        #[arg(long)]
        max: Option<f32>,
        /// Updated default value or option.
        #[arg(long)]
        default: Option<String>,
        /// Updated comma-separated options (for switch controls).
        #[arg(long)]
        options: Option<String>,
    },
    /// Remove a control from a model.
    Remove {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Control identifier.
        #[arg(long)]
        id: String,
    },
}

/// Subcommands under `dnam preset`.
#[derive(Debug, Subcommand)]
pub enum PresetAction {
    /// Add a named preset to a model.
    Add {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Preset identifier.
        #[arg(long)]
        id: String,
        /// Display label.
        #[arg(long, visible_alias = "title")]
        label: String,
        /// Control assignments (`--set drive=7.5 --set voice=Crunch`).
        #[arg(long = "set", short = 'c')]
        set_controls: Vec<String>,
    },
    /// List presets declared on a model.
    List {
        /// Target model identifier.
        #[arg(long)]
        model: String,
    },
    /// Update an existing preset on a model.
    Update {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Preset identifier.
        #[arg(long)]
        id: String,
        /// Updated display label.
        #[arg(long, visible_alias = "title")]
        label: Option<String>,
        /// Updated control assignments (`--set drive=8.0`).
        #[arg(long = "set", short = 'c')]
        set_controls: Vec<String>,
    },
    /// Remove a preset from a model.
    Remove {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Preset identifier.
        #[arg(long)]
        id: String,
    },
}

/// Subcommands under `dnam stem`.
#[derive(Debug, Subcommand)]
pub enum StemAction {
    /// Register an external wet WAV file as a stem with automatic latency alignment.
    Add {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Stem identifier.
        #[arg(long)]
        id: String,
        /// Path to the external wet WAV file.
        #[arg(long)]
        wet_wav: PathBuf,
        /// Control assignments (`--set drive=7.5 --set voice=Crunch`).
        #[arg(long = "set", short = 'c')]
        set_controls: Vec<String>,
    },
    /// List recorded stems on a model.
    List {
        /// Target model identifier.
        #[arg(long)]
        model: String,
    },
    /// Update control coordinates for an existing stem.
    Update {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Stem identifier.
        #[arg(long)]
        id: String,
        /// Updated control assignments (`--set drive=8.0`).
        #[arg(long = "set", short = 'c')]
        set_controls: Vec<String>,
    },
    /// Remove a stem from a model.
    Remove {
        /// Target model identifier.
        #[arg(long)]
        model: String,
        /// Stem identifier.
        #[arg(long)]
        id: String,
        /// Delete the stem WAV file from disk.
        #[arg(long, default_value_t = false)]
        delete_files: bool,
    },
}

fn parse_control_assignments_for_model(
    model: &ModelManifest,
    tokens: &[String],
    normalize_ranges: bool,
) -> StudioResult<(BTreeMap<String, f32>, BTreeMap<String, usize>)> {
    let mut ranges = BTreeMap::new();
    let mut switches = BTreeMap::new();
    for item in tokens {
        for part in item.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (k, v) = trimmed.split_once('=').ok_or_else(|| {
                StudioError::Validation(format!(
                    "Expected 'key=value' control assignment, got '{trimmed}'"
                ))
            })?;
            let key = k.trim();
            let val = v.trim();
            if let Some(r) = model.ranges.iter().find(|r| r.id == key) {
                let dial: f32 = val.parse().map_err(|_| {
                    StudioError::Validation(format!("Invalid numeric value '{val}' for '{key}'"))
                })?;
                if normalize_ranges {
                    let span = (r.max - r.min).max(1e-6);
                    ranges.insert(key.to_string(), ((dial - r.min) / span).clamp(0.0, 1.0));
                } else {
                    ranges.insert(key.to_string(), dial);
                }
            } else if let Some(s) = model.switches.iter().find(|s| s.id == key) {
                if let Some(opt) = s.options.iter().find(|o| o.label == val) {
                    switches.insert(key.to_string(), opt.index);
                } else if let Ok(idx) = val.parse::<usize>()
                    && idx < s.options.len()
                {
                    switches.insert(key.to_string(), idx);
                } else {
                    return Err(StudioError::Validation(format!(
                        "Unknown switch option '{val}' for '{key}'"
                    )));
                }
            } else if let Ok(num) = val.parse::<f32>() {
                ranges.insert(key.to_string(), num);
            }
        }
    }
    Ok((ranges, switches))
}

/// Executes `dnam init`.
pub fn run_init(
    workspace: &Path,
    name: &str,
    author: &str,
    initial_model_id: Option<&str>,
) -> StudioResult<String> {
    let mut proj = DnamProject::init(workspace, name, author)?;
    if let Some(mid) = initial_model_id
        && !mid.trim().is_empty()
    {
        proj.add_model(mid.trim(), mid.trim(), "Full")?;
    }
    Ok(format!(
        "Initialized DNAM workspace '{}' at {} (pubkey: {}...)",
        proj.manifest().project.name,
        proj.root_dir().display(),
        &proj.manifest().project.creator_public_key_hex[..16]
    ))
}

/// Executes `dnam status`.
pub fn run_status(workspace: &Path) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    let models = proj.list_models();
    Ok(format::format_workspace_status(proj.manifest(), &models))
}

/// Executes `dnam inspect`.
pub fn run_inspect(container_path: &Path) -> StudioResult<String> {
    let report = inspect_container(container_path)?;
    Ok(serde_json::to_string_pretty(&report)?)
}

/// Executes `dnam inspect` with optional device/marketplace keys.
pub fn run_inspect_with_crypto(
    container_path: &Path,
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<String> {
    let report = inspect_container_with_crypto(
        container_path,
        device_id,
        device_key_hex,
        marketplace_pubkey_hex,
    )?;
    Ok(serde_json::to_string_pretty(&report)?)
}

/// Executes `dnam render`.
pub fn run_render(
    container_path: &Path,
    input_wav: &Path,
    output_wav: &Path,
    tier: &str,
    preset: Option<&str>,
    set_controls: &[String],
) -> StudioResult<String> {
    let mut overrides = BTreeMap::new();
    for item in set_controls {
        for part in item.split(',') {
            if let Some((k, v)) = part.trim().split_once('=') {
                overrides.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    render_wav_with_options(
        container_path,
        input_wav,
        output_wav,
        tier,
        preset,
        &overrides,
    )?;
    Ok(format!(
        "Rendered 24-bit WAV to {} (tier={})",
        output_wav.display(),
        tier
    ))
}

/// Executes `dnam render` with optional device/marketplace keys.
#[allow(clippy::too_many_arguments)]
pub fn run_render_with_crypto(
    container_path: &Path,
    input_wav: &Path,
    output_wav: &Path,
    tier: &str,
    preset: Option<&str>,
    set_controls: &[String],
    device_id: Option<&str>,
    device_key_hex: Option<&str>,
    marketplace_pubkey_hex: Option<&str>,
) -> StudioResult<String> {
    let mut overrides = BTreeMap::new();
    for item in set_controls {
        for part in item.split(',') {
            if let Some((k, v)) = part.trim().split_once('=') {
                overrides.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    render_wav_with_crypto(
        container_path,
        input_wav,
        output_wav,
        tier,
        preset,
        &overrides,
        device_id,
        device_key_hex,
        marketplace_pubkey_hex,
    )?;
    Ok(format!(
        "Rendered 24-bit WAV to {} (tier={})",
        output_wav.display(),
        tier
    ))
}

/// Executes `dnam model`.
pub fn run_model(workspace: &Path, action: &ModelAction) -> StudioResult<String> {
    let mut proj = DnamProject::open(workspace)?;
    match action {
        ModelAction::Add {
            id,
            title,
            tier,
            ranges,
            switches,
        } => {
            let resolved_title = if title.trim().is_empty() {
                id.as_str()
            } else {
                title.as_str()
            };
            let model = proj.add_model(id, resolved_title, tier)?;
            for r_spec in ranges {
                let parsed = parse_compact_range_spec(r_spec)?;
                proj.add_range_control(id, parsed)?;
            }
            for s_spec in switches {
                let parsed = parse_compact_switch_spec(s_spec)?;
                proj.add_switch_control(id, parsed)?;
            }
            Ok(format!(
                "Created model '{}' ({}, tier={})",
                model.model_id, model.title, model.tier
            ))
        }
        ModelAction::List => {
            let models = proj.list_models();
            Ok(serde_json::to_string_pretty(&models)?)
        }
        ModelAction::Show { id } => {
            let model = proj.load_model(id)?;
            Ok(serde_json::to_string_pretty(&model)?)
        }
        ModelAction::Update {
            id,
            title,
            tier,
            description,
            output_mode,
            topology,
            allow_profiles,
            allow_oems,
        } => {
            let mut updated = proj.update_model(
                id,
                title.as_deref(),
                tier.as_deref(),
                description.as_deref(),
                output_mode.as_deref(),
                topology.as_deref(),
            )?;
            if !allow_profiles.is_empty() || !allow_oems.is_empty() {
                updated = proj.update_model_distribution_policy(id, allow_profiles, allow_oems)?;
            }
            Ok(format!("Updated model '{}'", updated.model_id))
        }
        ModelAction::Remove { id, delete_files } => {
            proj.remove_model(id, *delete_files)?;
            Ok(format!("Removed model '{id}'"))
        }
    }
}

/// Executes `dnam control`.
pub fn run_control(workspace: &Path, action: &ControlAction) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    match action {
        ControlAction::Add {
            model,
            range,
            switch,
        } => {
            if let Some(r_spec) = range {
                let parsed = parse_compact_range_spec(r_spec)?;
                let id = parsed.id.clone();
                proj.add_range_control(model, parsed)?;
                return Ok(format!("Added range control '{id}' to model '{model}'"));
            }
            if let Some(s_spec) = switch {
                let parsed = parse_compact_switch_spec(s_spec)?;
                let id = parsed.id.clone();
                proj.add_switch_control(model, parsed)?;
                return Ok(format!("Added switch control '{id}' to model '{model}'"));
            }
            Err(StudioError::Validation(
                "Provide `--range` or `--switch` to `dnam control add`".to_string(),
            ))
        }
        ControlAction::AddRange { model, spec } => {
            let parsed = parse_compact_range_spec(spec)?;
            let id = parsed.id.clone();
            proj.add_range_control(model, parsed)?;
            Ok(format!("Added range control '{id}' to model '{model}'"))
        }
        ControlAction::AddSwitch { model, spec } => {
            let parsed = parse_compact_switch_spec(spec)?;
            let id = parsed.id.clone();
            proj.add_switch_control(model, parsed)?;
            Ok(format!("Added switch control '{id}' to model '{model}'"))
        }
        ControlAction::List { model } => {
            let m = proj.load_model(model)?;
            let payload = json!({
                "ranges": m.ranges,
                "switches": m.switches,
            });
            Ok(serde_json::to_string_pretty(&payload)?)
        }
        ControlAction::Update {
            model,
            id,
            label,
            min,
            max,
            default,
            options,
        } => {
            proj.update_control(
                model,
                id,
                label.as_deref(),
                *min,
                *max,
                default.as_deref(),
                options.as_deref(),
            )?;
            Ok(format!("Updated control '{id}' on model '{model}'"))
        }
        ControlAction::Remove { model, id } => {
            proj.remove_control(model, id)?;
            Ok(format!("Removed control '{id}' from model '{model}'"))
        }
    }
}

/// Executes `dnam preset`.
pub fn run_preset(workspace: &Path, action: &PresetAction) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    match action {
        PresetAction::Add {
            model,
            id,
            label,
            set_controls,
        } => {
            let m = proj.load_model(model)?;
            let (ranges, switches) = parse_control_assignments_for_model(&m, set_controls, false)?;
            proj.add_preset(model, id, label, ranges, switches)?;
            Ok(format!("Added preset '{id}' to model '{model}'"))
        }
        PresetAction::List { model } => {
            let m = proj.load_model(model)?;
            Ok(serde_json::to_string_pretty(&m.presets)?)
        }
        PresetAction::Update {
            model,
            id,
            label,
            set_controls,
        } => {
            let m = proj.load_model(model)?;
            let (ranges, switches) = parse_control_assignments_for_model(&m, set_controls, false)?;
            proj.update_preset(
                model,
                id,
                label.as_deref(),
                if ranges.is_empty() {
                    None
                } else {
                    Some(ranges)
                },
                if switches.is_empty() {
                    None
                } else {
                    Some(switches)
                },
            )?;
            Ok(format!("Updated preset '{id}' on model '{model}'"))
        }
        PresetAction::Remove { model, id } => {
            proj.remove_preset(model, id)?;
            Ok(format!("Removed preset '{id}' from model '{model}'"))
        }
    }
}

/// Executes `dnam stem`.
pub fn run_stem(workspace: &Path, action: &StemAction) -> StudioResult<String> {
    let proj = DnamProject::open(workspace)?;
    match action {
        StemAction::Add {
            model,
            id,
            wet_wav,
            set_controls,
        } => {
            let m = proj.load_model(model)?;
            let (ranges, switches) = parse_control_assignments_for_model(&m, set_controls, true)?;
            let rec = proj.register_external_stem(model, id, wet_wav, ranges, switches)?;
            Ok(format!(
                "Registered stem '{}' on model '{}' (lag={:.2} samples, snr={:.1} dB)",
                rec.stem_id, model, rec.aligned_lag_samples, rec.snr_db
            ))
        }
        StemAction::List { model } => {
            let m = proj.load_model(model)?;
            Ok(serde_json::to_string_pretty(&m.stems)?)
        }
        StemAction::Update {
            model,
            id,
            set_controls,
        } => {
            let m = proj.load_model(model)?;
            let (ranges, switches) = parse_control_assignments_for_model(&m, set_controls, true)?;
            proj.update_stem(model, id, ranges, switches)?;
            Ok(format!("Updated stem '{id}' on model '{model}'"))
        }
        StemAction::Remove {
            model,
            id,
            delete_files,
        } => {
            proj.remove_stem(model, id, *delete_files)?;
            Ok(format!("Removed stem '{id}' from model '{model}'"))
        }
    }
}

/// Executes `dnam studio`.
pub fn run_studio(workspace: &Path, headless_check: bool) -> StudioResult<String> {
    let plugins = official_tauri_plugins();
    let tier1: Vec<&str> = plugins
        .iter()
        .filter(|p| p.tier == 1)
        .map(|p| p.id.as_str())
        .collect();
    let tier2: Vec<&str> = plugins
        .iter()
        .filter(|p| p.tier == 2)
        .map(|p| p.id.as_str())
        .collect();
    let payload = json!({
        "app": "dnam-studio",
        "tauriVersion": "2.0",
        "workspace": workspace.display().to_string(),
        "nativeAudioHost": detect_native_backend_str(),
        "headlessCheck": headless_check,
        "singleInstanceLocked": true,
        "deepLinkScheme": "dnam://",
        "ipcPattern": "isolation",
        "isolationDir": "./isolation",
        "freezePrototype": true,
        "assetProtocolEnabled": false,
        "officialPlugins": {
            "tier1": tier1,
            "tier2": tier2,
        }
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{generate_excitation_signal, write_wav_24bit};
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use dnam_engine::schema::DnamJsonHeader;
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_wscli_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn workspace_cli_handlers_cover_model_control_preset_stem_inspect_and_render() {
        let dir = temp_dir("ws_cli");
        let ws = dir.as_path();

        let init_out = run_init(ws, "MyWS", "Author", Some("init_mod")).expect("run_init");
        assert!(init_out.contains("MyWS"));

        let st_out = run_status(ws).expect("run_status");
        assert!(st_out.contains("init_mod"));

        // Model Add (empty title fallback + inline range & switch), List, Show, Update, Remove
        run_model(
            ws,
            &ModelAction::Add {
                id: "amp".to_string(),
                title: String::new(),
                tier: "Full".to_string(),
                ranges: vec!["gain=0..10:5".to_string()],
                switches: vec!["voice=clean,lead:clean".to_string()],
            },
        )
        .expect("model add");
        assert!(
            run_model(ws, &ModelAction::List)
                .expect("list")
                .contains("amp")
        );
        assert!(
            run_model(
                ws,
                &ModelAction::Show {
                    id: "amp".to_string()
                }
            )
            .expect("show")
            .contains("gain")
        );
        run_model(
            ws,
            &ModelAction::Update {
                id: "amp".to_string(),
                title: Some("Amp V2".to_string()),
                tier: Some("Ultra".to_string()),
                description: Some("Updated".to_string()),
                output_mode: Some("mono".to_string()),
                topology: Some("core1_only".to_string()),
                allow_profiles: vec!["enclave_sealed".to_string()],
                allow_oems: vec!["darkglass:anagram_rk3582".to_string()],
            },
        )
        .expect("model update");
        run_model(
            ws,
            &ModelAction::Remove {
                id: "init_mod".to_string(),
                delete_files: true,
            },
        )
        .expect("model remove");

        // Control Add, AddRange, AddSwitch, List, Update, Remove
        assert!(
            run_control(
                ws,
                &ControlAction::Add {
                    model: "amp".to_string(),
                    range: None,
                    switch: None,
                }
            )
            .is_err()
        );
        run_control(
            ws,
            &ControlAction::Add {
                model: "amp".to_string(),
                range: Some("bass=0..10:5".to_string()),
                switch: None,
            },
        )
        .expect("control add range");
        run_control(
            ws,
            &ControlAction::Add {
                model: "amp".to_string(),
                range: None,
                switch: Some("pad=off,on:off".to_string()),
            },
        )
        .expect("control add switch");
        run_control(
            ws,
            &ControlAction::AddRange {
                model: "amp".to_string(),
                spec: "treble=0..10:5".to_string(),
            },
        )
        .expect("control add-range");
        run_control(
            ws,
            &ControlAction::AddSwitch {
                model: "amp".to_string(),
                spec: "bright=off,on:off".to_string(),
            },
        )
        .expect("control add-switch");
        assert!(
            run_control(
                ws,
                &ControlAction::List {
                    model: "amp".to_string()
                }
            )
            .expect("control list")
            .contains("treble")
        );
        run_control(
            ws,
            &ControlAction::Update {
                model: "amp".to_string(),
                id: "treble".to_string(),
                label: Some("Treble2".to_string()),
                min: Some(1.0),
                max: Some(9.0),
                default: Some("4.0".to_string()),
                options: None,
            },
        )
        .expect("control update");
        run_control(
            ws,
            &ControlAction::Remove {
                model: "amp".to_string(),
                id: "bright".to_string(),
            },
        )
        .expect("control remove");

        // Preset Add, List, Update, Remove + assignment error branches
        run_preset(
            ws,
            &PresetAction::Add {
                model: "amp".to_string(),
                id: "p1".to_string(),
                label: "P1".to_string(),
                set_controls: vec!["gain=7.5,voice=lead,pad=1,extra_num=2.0".to_string()],
            },
        )
        .expect("preset add");
        assert!(
            run_preset(
                ws,
                &PresetAction::List {
                    model: "amp".to_string()
                }
            )
            .expect("preset list")
            .contains("p1")
        );
        run_preset(
            ws,
            &PresetAction::Update {
                model: "amp".to_string(),
                id: "p1".to_string(),
                label: Some("P1-Updated".to_string()),
                set_controls: vec!["gain=8.0".to_string()],
            },
        )
        .expect("preset update");
        assert!(
            run_preset(
                ws,
                &PresetAction::Update {
                    model: "amp".to_string(),
                    id: "p1".to_string(),
                    label: None,
                    set_controls: vec!["bad_token".to_string()],
                }
            )
            .is_err()
        );
        assert!(
            run_preset(
                ws,
                &PresetAction::Update {
                    model: "amp".to_string(),
                    id: "p1".to_string(),
                    label: None,
                    set_controls: vec!["gain=nan_val".to_string()],
                }
            )
            .is_err()
        );
        assert!(
            run_preset(
                ws,
                &PresetAction::Update {
                    model: "amp".to_string(),
                    id: "p1".to_string(),
                    label: None,
                    set_controls: vec!["voice=nonexistent".to_string()],
                }
            )
            .is_err()
        );
        run_preset(
            ws,
            &PresetAction::Remove {
                model: "amp".to_string(),
                id: "p1".to_string(),
            },
        )
        .expect("preset remove");

        // Stem Add, List, Update, Remove
        let dry_sig = generate_excitation_signal(48_000, 1024);
        let ext_wet = ws.join("ext_wet.wav");
        write_wav_24bit(&ext_wet, &dry_sig, 48_000, 1).expect("write ext_wet");
        run_stem(
            ws,
            &StemAction::Add {
                model: "amp".to_string(),
                id: "s_ext".to_string(),
                wet_wav: ext_wet,
                set_controls: vec!["gain=6.0,voice=lead".to_string()],
            },
        )
        .expect("stem add");
        assert!(
            run_stem(
                ws,
                &StemAction::List {
                    model: "amp".to_string()
                }
            )
            .expect("stem list")
            .contains("s_ext")
        );
        run_stem(
            ws,
            &StemAction::Update {
                model: "amp".to_string(),
                id: "s_ext".to_string(),
                set_controls: vec!["gain=9.0".to_string()],
            },
        )
        .expect("stem update");
        run_stem(
            ws,
            &StemAction::Remove {
                model: "amp".to_string(),
                id: "s_ext".to_string(),
                delete_files: true,
            },
        )
        .expect("stem remove");

        // Inspect & Render
        let mut header = DnamJsonHeader::default_reference(false);
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let c_bytes =
            build_dnam_container(&mut header, &arena, &arena, false).expect("build container");
        let c_path = ws.join("test.dnam");
        std::fs::write(&c_path, &c_bytes).expect("write dnam");
        assert!(
            run_inspect(&c_path)
                .expect("inspect")
                .contains("Reference DNAM Model")
        );
        let out_wav = ws.join("rendered.wav");
        let render_msg = run_render(
            &c_path,
            &ws.join("models/amp/stems/input.wav"),
            &out_wav,
            "lite",
            None,
            &["gain=5.0".to_string()],
        )
        .expect("render");
        assert!(render_msg.contains("Rendered 24-bit WAV"));

        // Studio headless check
        assert!(
            run_studio(ws, true)
                .expect("studio")
                .contains("dnam-studio")
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
