//! Target-agnostic 3-Phase control clamping (`ControlSpaceCore`), ZOH rescaling, and tempo
//! calibration inversion (`ref[sdk.ts.parameters.audio-param-semantics]`,
//! `ref[sdk.ts.model.containers-and-checkpoints]`).

use dnam_engine::container::ZohRescaledParams;
use dnam_engine::control::{
    CANONICAL_SUBDIVISION_RATIOS, ControlClampDescriptor, control_stride_for_rate,
    control_tick_ratio, eval_3node_calibration_f64, eval_q32_subdivision_phase,
    invert_tempo_physical_calibration, subdivision_q32_to_phasor,
};
use dnam_engine::schema::{DnamJsonHeader, TempoSubdivision};

use crate::error::{control_err, format_tagged_dnam_error, validation_err};

/// Pre-compiled 3-Phase control clamping space (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`)
/// avoiding per-call JSON header parsing.
#[derive(Debug, Clone)]
pub struct ControlSpaceCore {
    descriptors: Vec<ControlClampDescriptor>,
    num_controls: usize,
}

impl ControlSpaceCore {
    pub fn new(header_json: &str) -> Result<Self, String> {
        let header: DnamJsonHeader =
            serde_json::from_str(header_json).map_err(|e| validation_err(e.to_string()))?;
        let descriptors = header
            .compile_clamp_descriptors()
            .map_err(|e| format_tagged_dnam_error(&e))?;
        let num_controls = descriptors.len();
        Ok(Self {
            descriptors,
            num_controls,
        })
    }

    #[inline]
    pub const fn num_controls(&self) -> usize {
        self.num_controls
    }

    #[inline]
    pub fn descriptors(&self) -> &[ControlClampDescriptor] {
        &self.descriptors
    }

    /// Evaluates 3-Phase control clamping directly into caller-provided slices (`0` heap allocations).
    /// Returns a 32-bit bitmask where bit `p` (`p < 32`) is `1` if control `p` was clamped.
    #[inline]
    pub fn clamp_into(
        &self,
        raw_numeric_values: &[f32],
        clamped_out: &mut [f32],
        norm_out: &mut [f32],
    ) -> u32 {
        let n = self.num_controls.min(clamped_out.len()).min(norm_out.len());
        let mut active_switches = [0_u8; 256];
        let mut was_clamped_mask = 0_u32;

        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 1 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches[..n]);
                active_switches[p] = (clamped + 0.5) as u8;
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                if was_clamped && p < 32 {
                    was_clamped_mask |= 1_u32 << p;
                }
            }
        }
        for p in 0..n {
            let desc = &self.descriptors[p];
            if desc.ctrl_kind == 0 {
                let raw_v = raw_numeric_values
                    .get(p)
                    .copied()
                    .unwrap_or(desc.default_val);
                let (clamped, norm, was_clamped) =
                    desc.clamp_and_normalize(raw_v, &active_switches[..n]);
                clamped_out[p] = clamped;
                norm_out[p] = norm;
                if was_clamped && p < 32 {
                    was_clamped_mask |= 1_u32 << p;
                }
            }
        }
        was_clamped_mask
    }

    pub fn clamp(&self, raw_numeric_values: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<bool>, u16) {
        let n = self.num_controls;
        let mut clamped_out = vec![0.0_f32; n];
        let mut norm_out = vec![0.0_f32; n];
        let mask = self.clamp_into(raw_numeric_values, &mut clamped_out, &mut norm_out);
        let was_clamped_out = (0..n).map(|p| (mask & (1_u32 << p)) != 0).collect();
        (
            clamped_out,
            norm_out,
            was_clamped_out,
            (mask & 0xFFFF) as u16,
        )
    }

    pub fn to_physical(&self, ctrl_idx: usize, normalized_u: f32) -> Result<f32, String> {
        let desc = self.descriptors.get(ctrl_idx).ok_or_else(|| {
            control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={})",
                self.num_controls
            ))
        })?;
        let u = normalized_u.clamp(-1.0, 1.0);
        if desc.scale_norm.abs() < 1e-12 {
            return Ok(desc.default_val);
        }
        Ok((u - desc.offset_norm) / desc.scale_norm)
    }

    pub fn to_normalized(&self, ctrl_idx: usize, physical_val: f32) -> Result<f32, String> {
        let desc = self.descriptors.get(ctrl_idx).ok_or_else(|| {
            control_err(format!(
                "parameter index {ctrl_idx} out of bounds (num_controls={})",
                self.num_controls
            ))
        })?;
        let active_switches = [0_u8; 256];
        let (_, norm, _) =
            desc.clamp_and_normalize(physical_val, &active_switches[..self.num_controls]);
        Ok(norm)
    }
}

pub fn compute_zoh_core(
    sample_rate_train_hz: u32,
    sample_rate_host_hz: u32,
    r_ac_train: f32,
    r_dc_train: f32,
    tau_ref_train: f32,
    switch_xfade_train: u32,
) -> Result<[f64; 9], String> {
    let zoh = ZohRescaledParams::compute(
        sample_rate_train_hz,
        sample_rate_host_hz,
        r_ac_train,
        r_dc_train,
        tau_ref_train,
        switch_xfade_train,
    )
    .map_err(|e| format_tagged_dnam_error(&e))?;
    Ok([
        f64::from(zoh.sample_rate_host_hz),
        zoh.rho_sr,
        zoh.control_decimation_host as f64,
        zoh.rho_ctrl,
        f64::from(zoh.alpha_ramp_host),
        f64::from(zoh.r_ac_host),
        f64::from(zoh.r_dc_host),
        f64::from(zoh.tau_ref_samples_host),
        f64::from(zoh.switch_xfade_samples_host),
    ])
}

#[inline]
pub fn control_stride_for_rate_core(sample_rate_hz: u32) -> u32 {
    control_stride_for_rate(sample_rate_hz)
}

#[inline]
pub fn control_tick_ratio_core(sample_rate_hz: u32) -> f64 {
    control_tick_ratio(sample_rate_hz)
}

pub fn eval_subdivision_phase_core(
    beat_mod12: u8,
    within_beat_q32: u32,
    subdivision: &str,
    offset_q32: u32,
) -> Result<(u32, f32, f32), String> {
    let sub = TempoSubdivision::from_str_label(subdivision)
        .ok_or_else(|| validation_err(format!("Unknown tempo subdivision '{subdivision}'")))?;
    let idx = TempoSubdivision::ALL
        .iter()
        .position(|&s| s == sub)
        .unwrap_or(4);
    let ratio = CANONICAL_SUBDIVISION_RATIOS[idx];
    let q32 = eval_q32_subdivision_phase(beat_mod12, within_beat_q32, ratio, offset_q32)
        .ok_or_else(|| {
            validation_err(format!("Invalid beatMod12={beat_mod12} (expected 0..12)"))
        })?;
    let (cos_phi, sin_phi) = subdivision_q32_to_phasor(q32);
    Ok((q32, cos_phi, sin_phi))
}

#[inline]
pub fn eval_3node_calibration_core(anchors: [f64; 3], u: f64) -> f64 {
    eval_3node_calibration_f64(anchors, u)
}

pub fn invert_tempo_calibration_core(
    declared_range: (f32, f32),
    safe_range: (f32, f32),
    anchors: [f64; 3],
    target: f64,
    flush_subnormals: bool,
) -> Result<[f64; 5], String> {
    let res = invert_tempo_physical_calibration(
        declared_range,
        safe_range,
        anchors,
        target,
        flush_subnormals,
    )
    .ok_or_else(|| validation_err("invalid calibration range or non-finite target"))?;
    Ok([
        f64::from(res.physical_f32),
        f64::from(res.normalized_f32),
        res.realized_value,
        res.residual,
        f64::from(res.search_steps.0),
    ])
}
