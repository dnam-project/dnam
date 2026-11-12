//! Recursive `faust -json` UI Metadata Parser & Out-of-Process / Analytical Twin `FaustRunner`.

use super::ipc::validate_runner_stimulus_layout;
use super::{DnwbCommand, DnwbError, DnwbWorkerSession};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus, StimulusFrame};
use dnam_engine::schema::DnamError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Parsed FAUST UI control descriptor extracted from `faust -json` (`hslider`, `vslider`,
/// `nentry`, `button`, `checkbox`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaustControlSpec {
    #[serde(rename = "type")]
    pub widget_type: String,
    pub label: String,
    #[serde(default)]
    pub init: f32,
    #[serde(default)]
    pub min: f32,
    #[serde(default = "default_faust_max")]
    pub max: f32,
    #[serde(default = "default_faust_step")]
    pub step: f32,
}

const fn default_faust_max() -> f32 {
    1.0
}

const fn default_faust_step() -> f32 {
    1.0
}

impl FaustControlSpec {
    /// Maps a normalized control coordinate `u in [-1, 1]` into the physical `[min, max]` FAUST widget domain.
    #[must_use]
    pub fn denormalize(&self, u_norm: f32) -> f64 {
        let u = if u_norm.is_finite() {
            f64::from(u_norm).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let init = if self.init.is_finite() {
            f64::from(self.init)
        } else {
            0.0
        };
        let min = if self.min.is_finite() {
            f64::from(self.min)
        } else {
            0.0
        };
        let max = if self.max.is_finite() {
            f64::from(self.max)
        } else {
            1.0
        };
        if max <= min {
            return init;
        }
        let val = min + 0.5 * (u + 1.0) * (max - min);
        if self.widget_type == "button" || self.widget_type == "checkbox" {
            if val >= 0.5 * (min + max) { max } else { min }
        } else {
            val
        }
    }
}

#[derive(Deserialize)]
struct FaustJsonDoc {
    ui: Vec<FaustUiNode>,
}

#[derive(Deserialize)]
struct FaustUiNode {
    #[serde(default, rename = "type")]
    node_type: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    init: Option<f32>,
    #[serde(default)]
    min: Option<f32>,
    #[serde(default)]
    max: Option<f32>,
    #[serde(default)]
    step: Option<f32>,
    #[serde(default)]
    items: Option<Vec<FaustUiNode>>,
}

fn collect_faust_ui_controls(
    nodes: Vec<FaustUiNode>,
    depth: usize,
    out: &mut Vec<FaustControlSpec>,
) -> Result<(), DnwbError> {
    if depth > 64 {
        return Err(DnwbError::InvalidJsonMetadata(String::from(
            "FAUST UI group nesting exceeds max depth 64",
        )));
    }
    for node in nodes {
        match node.node_type.as_str() {
            "vgroup" | "hgroup" | "tgroup" => {
                if let Some(children) = node.items {
                    collect_faust_ui_controls(children, depth + 1, out)?;
                }
            }
            "hslider" | "vslider" | "nentry" => {
                out.push(FaustControlSpec {
                    widget_type: node.node_type,
                    label: node.label,
                    init: node.init.unwrap_or(0.0),
                    min: node.min.unwrap_or(0.0),
                    max: node.max.unwrap_or(1.0),
                    step: node.step.unwrap_or(0.01),
                });
            }
            "button" | "checkbox" => {
                out.push(FaustControlSpec {
                    widget_type: node.node_type,
                    label: node.label,
                    init: node.init.unwrap_or(0.0),
                    min: node.min.unwrap_or(0.0),
                    max: node.max.unwrap_or(1.0),
                    step: node.step.unwrap_or(1.0),
                });
            }
            _ => {
                if let Some(children) = node.items {
                    collect_faust_ui_controls(children, depth + 1, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Recursively parses `faust -json` UI metadata (`vgroup`, `hgroup`, `tgroup`, sliders, `nentry`,
/// `button`, `checkbox`) into ordered [`FaustControlSpec`] descriptors.
pub fn parse_faust_json_controls(json_str: &str) -> Result<Vec<FaustControlSpec>, DnwbError> {
    let doc: FaustJsonDoc = serde_json::from_str(json_str)
        .map_err(|e| DnwbError::InvalidJsonMetadata(e.to_string()))?;
    let mut out = Vec::new();
    collect_faust_ui_controls(doc.ui, 0, &mut out)?;
    Ok(out)
}

/// Out-of-process / analytical twin FAUST DSP runner implementing [`DistillationTarget`].
#[derive(Clone, Debug)]
pub struct FaustRunner {
    pub dsp_name: String,
    pub controls: Vec<FaustControlSpec>,
    pub timeout: Duration,
    pub subprocess_cmd: Option<Vec<String>>,
    pub active_switches: Vec<u8>,
    lp_state: [f64; 4],
    worker_session: DnwbWorkerSession,
}

impl FaustRunner {
    /// Constructs a `FaustRunner` from a `faust -json` metadata string and watchdog timeout.
    pub fn from_faust_json(
        dsp_name: impl Into<String>,
        faust_json: &str,
        timeout_ms: u64,
    ) -> Result<Self, DnwbError> {
        let controls = parse_faust_json_controls(faust_json)?;
        Ok(Self {
            dsp_name: dsp_name.into(),
            controls,
            timeout: Duration::from_millis(timeout_ms),
            subprocess_cmd: None,
            active_switches: Vec::new(),
            lp_state: [0.0; 4],
            worker_session: DnwbWorkerSession::new(),
        })
    }

    /// Configures an external GPL-isolated FAUST binary communicating over `DNWB` pipes.
    #[must_use]
    pub fn with_subprocess_command(mut self, cmd: Vec<String>) -> Self {
        self.worker_session.terminate();
        self.subprocess_cmd = Some(cmd);
        self
    }

    /// Sets discrete switch states forwarded over `DNWB` IPC and applied during DSP execution.
    pub fn set_switches(&mut self, switches: &[u8]) {
        self.active_switches.clear();
        self.active_switches.extend_from_slice(switches);
    }

    fn resolve_control_unit(
        &self,
        idx: usize,
        sample_idx: usize,
        vals: &[f32],
        trajs: Option<&Vec<Vec<f32>>>,
    ) -> f64 {
        let raw_opt = trajs
            .and_then(|t| t.get(idx))
            .and_then(|tr| tr.get(sample_idx).copied())
            .or_else(|| vals.get(idx).copied());
        if let Some(spec) = self.controls.get(idx) {
            let init = if spec.init.is_finite() {
                f64::from(spec.init)
            } else {
                0.0
            };
            let min = if spec.min.is_finite() {
                f64::from(spec.min)
            } else {
                0.0
            };
            let max = if spec.max.is_finite() {
                f64::from(spec.max)
            } else {
                1.0
            };
            let physical = match raw_opt {
                Some(u) => spec.denormalize(u),
                None => init,
            };
            let span = (max - min).max(1.0e-9);
            ((physical - min) / span).clamp(0.0, 1.0)
        } else {
            let u = raw_opt.unwrap_or(0.0);
            let u_clean = if u.is_finite() {
                f64::from(u).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            (0.5 * (u_clean + 1.0)).clamp(0.0, 1.0)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &mut self,
        stem_id: String,
        sample_rate_hz: u32,
        in_1: &[f32],
        in_2: Option<&[f32]>,
        in_3: Option<&[f32]>,
        in_4: Option<&[f32]>,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<StemResponse, DnwbError> {
        if self.timeout.is_zero() {
            return Err(DnwbError::WatchdogTimeout { timeout_ms: 0 });
        }
        validate_runner_stimulus_layout(
            sample_rate_hz,
            in_1,
            in_2,
            in_3,
            in_4,
            control_values,
            control_trajectories,
            freeze_window,
        )?;
        let start = Instant::now();
        let n = in_1.len();
        let sr_ratio = 48_000.0 / f64::from(sample_rate_hz);
        let asym_bias = if self.active_switches.first() == Some(&1) {
            0.05
        } else {
            0.0
        };

        let mut wet_l = Vec::with_capacity(n);
        let mut pre_lp_l = Vec::with_capacity(n);
        let mut wet_r = in_2.map(|_| Vec::with_capacity(n));
        let mut pre_lp_r = in_2.map(|_| Vec::with_capacity(n));
        let mut wet_3 = in_3.map(|_| Vec::with_capacity(n));
        let mut wet_4 = in_4.map(|_| Vec::with_capacity(n));

        let extra_ctrl_count = self.controls.len().max(control_values.len());
        for i in 0..n {
            if (i & 255) == 0 && start.elapsed() >= self.timeout {
                return Err(DnwbError::WatchdogTimeout {
                    timeout_ms: self.timeout.as_millis() as u64,
                });
            }
            let frozen = freeze_window.is_some_and(|[fs, fe]| i >= fs && i < fe);
            let drive_01 = self.resolve_control_unit(0, i, control_values, control_trajectories);
            let tone_01 = self.resolve_control_unit(1, i, control_values, control_trajectories);
            let mut trim_gain = 1.0_f64;
            if extra_ctrl_count > 2 {
                let c2 = self.resolve_control_unit(2, i, control_values, control_trajectories);
                trim_gain = 0.75 + 0.5 * c2;
            }
            let drive = 2.0 + 4.0 * drive_01;
            let alpha_48k = (0.08 + 0.15 * tone_01).clamp(0.001, 0.99);
            let alpha = 1.0 - (1.0 - alpha_48k).powf(sr_ratio);

            let xl = if in_1[i].is_finite() {
                f64::from(in_1[i])
            } else {
                0.0
            };
            let nl_l = ((drive * (xl + asym_bias)).tanh() - (drive * asym_bias).tanh()) * trim_gain;
            pre_lp_l.push(nl_l as f32);
            if !frozen {
                self.lp_state[0] += alpha * (nl_l - self.lp_state[0]);
            }
            wet_l.push(self.lp_state[0] as f32);

            if let (Some(ch2), Some(out2), Some(pre2)) = (in_2, wet_r.as_mut(), pre_lp_r.as_mut()) {
                let xr_raw = ch2.get(i).copied().unwrap_or(0.0);
                let xr = if xr_raw.is_finite() {
                    f64::from(xr_raw)
                } else {
                    0.0
                };
                let nl_r =
                    ((drive * (xr + asym_bias)).tanh() - (drive * asym_bias).tanh()) * trim_gain;
                pre2.push(nl_r as f32);
                if !frozen {
                    self.lp_state[1] += alpha * (nl_r - self.lp_state[1]);
                }
                out2.push(self.lp_state[1] as f32);
            }

            if let (Some(ch3), Some(out3)) = (in_3, wet_3.as_mut()) {
                let x3 = ch3.get(i).copied().unwrap_or(0.0);
                let s3 = if x3.is_finite() { f64::from(x3) } else { 0.0 };
                if !frozen {
                    self.lp_state[2] += alpha * (s3 * trim_gain - self.lp_state[2]);
                }
                out3.push(self.lp_state[2].clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32);
            }
            if let (Some(ch4), Some(out4)) = (in_4, wet_4.as_mut()) {
                let x4 = ch4.get(i).copied().unwrap_or(0.0);
                let s4 = if x4.is_finite() { f64::from(x4) } else { 0.0 };
                if !frozen {
                    self.lp_state[3] += alpha * (s4 * trim_gain - self.lp_state[3]);
                }
                out4.push(self.lp_state[3].clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32);
            }
        }

        let mut internal_nodes = BTreeMap::new();
        internal_nodes.insert(String::from("faust_waveshaper"), pre_lp_l);
        if let Some(pre2) = pre_lp_r {
            internal_nodes.insert(String::from("faust_waveshaper_r"), pre2);
        }
        Ok(StemResponse {
            stem_id,
            sample_rate_hz,
            out_1: wet_l,
            out_2: wet_r,
            out_3: wet_3,
            out_4: wet_4,
            internal_nodes,
        })
    }
}

impl DistillationTarget for FaustRunner {
    fn reset_state(&mut self) {
        self.lp_state = [0.0; 4];
        if let Some(ref cmd) = self.subprocess_cmd {
            self.worker_session
                .reset_if_live(cmd, &self.active_switches, self.timeout);
        }
    }

    fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
        if let Some(ref cmd) = self.subprocess_cmd {
            return self
                .worker_session
                .execute(
                    cmd,
                    DnwbCommand::RenderStem,
                    stimulus,
                    &self.active_switches,
                    self.timeout,
                )
                .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")));
        }
        self.reset_state();
        self.render_internal(
            stimulus.stem_id.clone(),
            stimulus.sample_rate_hz,
            &stimulus.in_1,
            stimulus.in_2.as_deref(),
            stimulus.in_3.as_deref(),
            stimulus.in_4.as_deref(),
            &stimulus.control_values,
            stimulus.control_trajectories.as_ref(),
            stimulus.freeze_transition_window,
        )
        .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")))
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        if let Some(ref cmd) = self.subprocess_cmd {
            let stim = StemStimulus {
                stem_id: format!("frame_{}", frame.frame_index),
                sample_rate_hz: frame.sample_rate_hz,
                in_1: frame.in_1.clone(),
                in_2: frame.in_2.clone(),
                in_3: frame.in_3.clone(),
                in_4: frame.in_4.clone(),
                control_values: frame.control_values.clone(),
                control_trajectories: frame.control_trajectories.clone(),
                freeze_transition_window: None,
            };
            return self
                .worker_session
                .execute(
                    cmd,
                    DnwbCommand::RenderFrame,
                    &stim,
                    &self.active_switches,
                    self.timeout,
                )
                .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")));
        }
        self.render_internal(
            format!("frame_{}", frame.frame_index),
            frame.sample_rate_hz,
            &frame.in_1,
            frame.in_2.as_deref(),
            frame.in_3.as_deref(),
            frame.in_4.as_deref(),
            &frame.control_values,
            frame.control_trajectories.as_ref(),
            None,
        )
        .map_err(|e| DnamError::SchemaValidationError(format!("{e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faust_nested_ui_stereo_trajectories_and_multirate() {
        let nested_faust_json = r#"{
            "ui": [{
                "type": "vgroup",
                "label": "Pedal",
                "items": [
                    {
                        "type": "hgroup",
                        "label": "Preamp",
                        "items": [
                            {"type": "hslider", "label": "Drive", "init": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
                            {"type": "tgroup", "label": "EQ", "items": [
                                {"type": "vslider", "label": "Tone", "init": 2000.0, "min": 200.0, "max": 8000.0, "step": 10.0},
                                {"type": "nentry", "label": "Trim", "init": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}
                            ]}
                        ]
                    },
                    {"type": "checkbox", "label": "Bright"},
                    {"type": "button", "label": "Tap"},
                    {"type": "hbargraph", "label": "Meter", "min": -60.0, "max": 0.0}
                ]
            }]
        }"#;
        let mut faust =
            FaustRunner::from_faust_json("nested_overdrive.dsp", nested_faust_json, 5_000).unwrap();
        assert_eq!(faust.controls.len(), 5);
        assert_eq!(faust.controls[3].denormalize(0.8), 1.0);
        assert_eq!(faust.controls[3].denormalize(-0.8), 0.0);

        let stereo_stim = StemStimulus {
            stem_id: String::from("faust_stereo_96k"),
            sample_rate_hz: 96_000,
            in_1: (0..64).map(|i| 0.4 * ((i as f32) * 0.15).sin()).collect(),
            in_2: Some((0..64).map(|i| -0.3 * ((i as f32) * 0.15).cos()).collect()),
            in_3: None,
            in_4: None,
            control_values: vec![-0.5, 0.5, 0.0, 1.0, -1.0],
            control_trajectories: Some(vec![
                (0..64).map(|i| -1.0 + 2.0 * (i as f32) / 63.0).collect(),
                vec![0.25; 64],
                vec![0.0; 64],
                vec![1.0; 64],
                vec![-1.0; 64],
            ]),
            freeze_transition_window: None,
        };
        let f_resp = faust.render_stem(&stereo_stim).unwrap();
        assert_eq!(f_resp.sample_rate_hz, 96_000);
        assert_eq!(f_resp.out_1.len(), 64);
        assert_eq!(f_resp.out_2.as_ref().unwrap().len(), 64);
        assert!(f_resp.internal_nodes.contains_key("faust_waveshaper"));
        assert!(f_resp.internal_nodes.contains_key("faust_waveshaper_r"));

        let f_frame = faust
            .render_frame(&StimulusFrame {
                frame_index: 1,
                sample_rate_hz: 96_000,
                in_1: stereo_stim.in_1.clone(),
                in_2: stereo_stim.in_2.clone(),
                in_3: None,
                in_4: None,
                control_values: stereo_stim.control_values.clone(),
                control_trajectories: stereo_stim.control_trajectories.clone(),
            })
            .unwrap();
        assert_eq!(f_frame.out_1.len(), 64);
        assert!(f_frame.internal_nodes.contains_key("faust_waveshaper"));
        assert!(parse_faust_json_controls("{bad_json").is_err());

        let mut faust_timeout =
            FaustRunner::from_faust_json("timeout.dsp", nested_faust_json, 0).unwrap();
        assert!(faust_timeout.render_stem(&stereo_stim).is_err());

        let mut faust_sub = FaustRunner::from_faust_json("sub.dsp", nested_faust_json, 50)
            .unwrap()
            .with_subprocess_command(Vec::new());
        assert!(faust_sub.render_stem(&stereo_stim).is_err());
        assert!(
            faust_sub
                .render_frame(&StimulusFrame {
                    frame_index: 0,
                    sample_rate_hz: 48_000,
                    in_1: vec![0.1; 8],
                    in_2: None,
                    in_3: None,
                    in_4: None,
                    control_values: vec![0.0; 5],
                    control_trajectories: None,
                })
                .is_err()
        );

        // Serde defaults, degenerate max<=min / NaN denormalize, empty UI fallback, and NaN audio sanitization
        let spec_def: FaustControlSpec =
            serde_json::from_str(r#"{"type":"hslider","label":"Gain"}"#).unwrap();
        assert_eq!((spec_def.max, spec_def.step), (1.0, 1.0));
        assert_eq!(spec_def.denormalize(f32::NAN), 0.5);
        let degen_spec = FaustControlSpec {
            widget_type: String::from("hslider"),
            label: String::from("Fixed"),
            init: 3.5,
            min: 5.0,
            max: 2.0,
            step: 0.1,
        };
        assert_eq!(degen_spec.denormalize(0.8), 3.5);

        let mut faust_empty =
            FaustRunner::from_faust_json("empty.dsp", r#"{"ui":[]}"#, 500).unwrap();
        let nan_stim = StemStimulus {
            stem_id: String::from("nan_test"),
            sample_rate_hz: 48_000,
            in_1: vec![f32::NAN, 0.2],
            in_2: Some(vec![0.1, f32::INFINITY]),
            in_3: Some(vec![0.1, f32::NAN]),
            in_4: Some(vec![f32::INFINITY, -0.2]),
            control_values: vec![0.3],
            control_trajectories: None,
            freeze_transition_window: None,
        };
        let empty_resp = faust_empty.render_stem(&nan_stim).unwrap();
        assert!(empty_resp.out_1.iter().all(|v| v.is_finite()));
        assert!(empty_resp.out_2.unwrap().iter().all(|v| v.is_finite()));
        assert!(empty_resp.out_3.unwrap().iter().all(|v| v.is_finite()));
        assert!(empty_resp.out_4.unwrap().iter().all(|v| v.is_finite()));

        // Missing control_values falls back to spec.init cleanly
        let mut no_ctrl_stim = stereo_stim.clone();
        no_ctrl_stim.control_values.clear();
        no_ctrl_stim.control_trajectories = None;
        assert!(faust.render_stem(&no_ctrl_stim).is_ok());
    }
}
