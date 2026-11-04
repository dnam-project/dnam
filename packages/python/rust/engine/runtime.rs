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
        &mut self,
        container_bytes: &[u8],
        tier: Option<u8>,
        sample_rate_hz: Option<u32>,
    ) -> PyResult<()> {
        let target_tier = tier.map_or_else(|| self.inner.tier(), Tier::from_u8);
        let target_sr = sample_rate_hz.unwrap_or_else(|| self.inner.sample_rate_hz());
        self.inner
            .reload_model_in_place_owned(container_bytes, target_tier, target_sr)
            .map_err(dnam_err_to_py)
    }

    pub(crate) fn set_sample_rate_impl(&mut self, sample_rate_hz: u32) -> PyResult<()> {
        self.inner
            .set_sample_rate(sample_rate_hz)
            .map_err(dnam_err_to_py)
    }

    #[inline]
    pub(crate) fn clamped_control_value_impl(&self, ctrl_idx: usize) -> PyResult<f32> {
        self.inner
            .clamped_control_values()
            .get(ctrl_idx)
            .copied()
            .ok_or_else(|| {
                control_err(format!(
                    "control index {ctrl_idx} out of bounds (num_controls={})",
                    self.inner.num_controls()
                ))
            })
    }

    pub(crate) fn set_control_value_impl(&mut self, ctrl_idx: usize, value: f32) -> PyResult<()> {
        if ctrl_idx >= self.inner.num_controls() {
            return Err(control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={})",
                self.inner.num_controls()
            )));
        }
        self.inner.set_control_value(ctrl_idx, value);
        Ok(())
    }

    pub(crate) fn set_control_immediate_impl(
        &mut self,
        ctrl_idx: usize,
        value: f32,
    ) -> PyResult<()> {
        if ctrl_idx >= self.inner.num_controls() {
            return Err(control_err(format!(
                "control index {ctrl_idx} out of bounds (num_controls={})",
                self.inner.num_controls()
            )));
        }
        self.inner.set_control_immediate(ctrl_idx, value);
        Ok(())
    }

    pub(crate) fn set_controls_immediate_batch_impl(
        &mut self,
        updates: &[(usize, f32)],
    ) -> PyResult<()> {
        let num_controls = self.inner.num_controls();
        for &(ctrl_idx, _) in updates {
            if ctrl_idx >= num_controls {
                return Err(control_err(format!(
                    "control index {ctrl_idx} out of bounds (num_controls={num_controls})"
                )));
            }
        }
        let Some((&(last_idx, last_val), init)) = updates.split_last() else {
            return Ok(());
        };
        for &(idx, val) in init {
            self.inner.set_control_value(idx, val);
        }
        self.inner.set_control_immediate(last_idx, last_val);
        Ok(())
    }

    pub(crate) fn sync_tempo_impl(
        &mut self,
        bpm: f32,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<usize> {
        let sub = parse_subdivision(subdivision)?;
        Ok(self.inner.sync_tempo(bpm, sub, fold_octave))
    }

    pub(crate) fn register_tap_tempo_impl(
        &mut self,
        subdivision: Option<&str>,
        fold_octave: bool,
    ) -> PyResult<(f32, usize)> {
        let sub = parse_subdivision(subdivision)?;
        Ok(self.inner.register_tap_tempo(sub, fold_octave))
    }

    pub(crate) fn apply_preset_impl(&mut self, preset_id: &str) -> PyResult<()> {
        self.inner
            .apply_preset(preset_id)
            .map_err(|e| control_err(format!("{e}")))
    }
}
