//! Runtime configuration, control, preset, bypass, freeze, width & tempo helpers for [`PyEngine`]
//! (`ref[sdk.python.api.engine-facets]`).

use dnam_engine::Tier;
use dnam_engine::schema::TempoSubdivision;
use pyo3::prelude::*;

use super::PyEngine;
use crate::{control_err, dnam_err_to_py, validation_err};

fn parse_subdivision(subdivision: Option<&str>) -> PyResult<Option<TempoSubdivision>> {
    match subdivision {
        None => Ok(None),
        Some(s) => TempoSubdivision::from_str_label(s)
            .map(Some)
            .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{s}'"))),
    }
}

impl PyEngine {
    pub(crate) fn reload_impl(
        &self,
        py: Python<'_>,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> PyResult<()> {
        py.detach(|| {
            let target_tier = tier.map_or_else(|| self.host.tier(), Tier::from_u8);
            let target_sr = sample_rate_hz.unwrap_or_else(|| self.host.sample_rate_hz());
            self.host
                .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
        })
        .map_err(dnam_err_to_py)
    }

    pub(crate) fn set_sample_rate_impl(&self, py: Python<'_>, sample_rate_hz: u32) -> PyResult<()> {
        py.detach(|| self.host.set_sample_rate(sample_rate_hz))
            .map_err(dnam_err_to_py)
    }

    #[inline]
    pub(crate) fn clamped_control_value_impl(
        &self,
        py: Python<'_>,
        ctrl_idx: usize,
    ) -> PyResult<f32> {
        let (opt_val, num_controls) = py.detach(|| {
            (
                self.host.clamped_control_value(ctrl_idx),
                self.host.num_controls(),
            )
        });
        opt_val.ok_or_else(|| {
            control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={num_controls})"
            ))
        })
    }

    pub(crate) fn set_control_value_impl(
        &self,
        py: Python<'_>,
        ctrl_idx: usize,
        value: f32,
    ) -> PyResult<()> {
        let err_num = py.detach(|| {
            let num_controls = self.host.num_controls();
            if ctrl_idx >= num_controls {
                Some(num_controls)
            } else {
                self.host.set_control_value(ctrl_idx, value);
                None
            }
        });
        if let Some(num_controls) = err_num {
            return Err(control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={num_controls})"
            )));
        }
        Ok(())
    }

    pub(crate) fn set_control_immediate_impl(
        &self,
        py: Python<'_>,
        ctrl_idx: usize,
        value: f32,
    ) -> PyResult<()> {
        let err_num = py.detach(|| {
            let num_controls = self.host.num_controls();
            if ctrl_idx >= num_controls {
                Some(num_controls)
            } else {
                self.host.set_control_immediate(ctrl_idx, value);
                None
            }
        });
        if let Some(num_controls) = err_num {
            return Err(control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={num_controls})"
            )));
        }
        Ok(())
    }

    pub(crate) fn set_controls_immediate_batch_impl(
        &self,
        py: Python<'_>,
        updates: &[(usize, f32)],
    ) -> PyResult<()> {
        let err_pair = py.detach(|| {
            let num_controls = self.host.num_controls();
            for &(ctrl_idx, _) in updates {
                if ctrl_idx >= num_controls {
                    return Some((ctrl_idx, num_controls));
                }
            }
            self.host.set_controls_immediate_batch(updates);
            None
        });
        if let Some((ctrl_idx, num_controls)) = err_pair {
            return Err(control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={num_controls})"
            )));
        }
        Ok(())
    }

    pub(crate) fn sync_tempo_impl(
        &self,
        py: Python<'_>,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<usize> {
        let sub = parse_subdivision(subdivision)?;
        Ok(py.detach(|| self.host.sync_tempo(bpm, sub, fold_octave)))
    }

    pub(crate) fn register_tap_tempo_impl(
        &self,
        py: Python<'_>,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<(f32, usize)> {
        let sub = parse_subdivision(subdivision)?;
        Ok(py.detach(|| self.host.register_tap_tempo(sub, fold_octave)))
    }

    pub(crate) fn apply_preset_impl(&self, py: Python<'_>, preset_id: &str) -> PyResult<()> {
        py.detach(|| self.host.apply_preset(preset_id))
            .map_err(|e| control_err(format!("{e}")))
    }
}
