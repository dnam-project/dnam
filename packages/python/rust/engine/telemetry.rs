//! Telemetry snapshot serialization for [`PyEngine`] (`ref[sdk.python.api.engine-facets]`).

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{PyEngine, backend_kind_str};

impl PyEngine {
    pub(crate) fn telemetry_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (
            snap,
            zoh,
            (bypassed, trails_enabled),
            effective_tier,
            backend_str,
            sample_rate_hz,
            supports_freeze,
            stereo_width,
            current_bpm,
            num_ctrl,
        ) = py.detach(|| {
            let inner = self.lock_inner();
            (
                inner.telemetry_snapshot(),
                *inner.zoh(),
                inner.bypass_state(),
                inner.effective_tier() as u8,
                backend_kind_str(inner.backend_kind()),
                inner.sample_rate_hz(),
                inner.supports_freeze(),
                inner.stereo_width(),
                inner.current_bpm(),
                inner.num_controls().min(16),
            )
        });

        let dict = PyDict::new(py);
        dict.set_item(intern!(py, "active_tier"), effective_tier)?;
        dict.set_item(intern!(py, "backend"), backend_str)?;
        dict.set_item(intern!(py, "sample_rate_hz"), sample_rate_hz)?;
        dict.set_item(intern!(py, "stereo_out"), snap.stereo_out != 0)?;
        dict.set_item(intern!(py, "bypassed"), bypassed)?;
        dict.set_item(intern!(py, "trails_enabled"), trails_enabled)?;
        dict.set_item(intern!(py, "is_frozen"), snap.is_frozen != 0)?;
        dict.set_item(intern!(py, "supports_freeze"), supports_freeze)?;
        dict.set_item(intern!(py, "stereo_width"), stereo_width)?;
        dict.set_item(intern!(py, "current_bpm"), current_bpm)?;
        dict.set_item(intern!(py, "active_clamp_mask"), snap.active_clamp_mask)?;
        dict.set_item(
            intern!(py, "control_decimation_host"),
            snap.control_decimation_host,
        )?;
        dict.set_item(intern!(py, "max_op_n_drive"), snap.max_op_n_drive)?;
        dict.set_item(intern!(py, "max_pole_radius"), snap.max_pole_radius)?;
        dict.set_item(
            intern!(py, "max_effective_loop_gain"),
            snap.max_effective_loop_gain,
        )?;
        dict.set_item(intern!(py, "tail_time_ms"), snap.tail_time_ms)?;
        dict.set_item(
            intern!(py, "z_head"),
            (
                snap.z_head[0],
                snap.z_head[1],
                snap.z_head[2],
                snap.z_head[3],
            ),
        )?;
        dict.set_item(
            intern!(py, "env_states"),
            (
                snap.env_states[0],
                snap.env_states[1],
                snap.env_states[2],
                snap.env_states[3],
            ),
        )?;
        dict.set_item(
            intern!(py, "lfo_harmonics"),
            (
                snap.lfo_harmonics[0],
                snap.lfo_harmonics[1],
                snap.lfo_harmonics[2],
                snap.lfo_harmonics[3],
                snap.lfo_harmonics[4],
                snap.lfo_harmonics[5],
                snap.lfo_harmonics[6],
                snap.lfo_harmonics[7],
            ),
        )?;
        dict.set_item(
            intern!(py, "clamped_controls"),
            snap.clamped_controls[..num_ctrl].to_vec(),
        )?;
        dict.set_item(
            intern!(py, "overflow_or_saturation_flags"),
            snap.overflow_or_saturation_flags,
        )?;
        dict.set_item(intern!(py, "rho_sr"), zoh.rho_sr)?;
        dict.set_item(intern!(py, "rho_ctrl"), zoh.rho_ctrl)?;
        dict.set_item(intern!(py, "alpha_ramp_host"), zoh.alpha_ramp_host)?;
        dict.set_item(intern!(py, "r_ac_host"), zoh.r_ac_host)?;
        dict.set_item(intern!(py, "r_dc_host"), zoh.r_dc_host)?;
        dict.set_item(
            intern!(py, "tau_ref_samples_host"),
            zoh.tau_ref_samples_host,
        )?;
        dict.set_item(
            intern!(py, "switch_xfade_samples_host"),
            zoh.switch_xfade_samples_host,
        )?;
        Ok(dict)
    }
}
