//! Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability ($44.1\text{--}96\text{ kHz}$).

use crate::schema::{DnamError, cold_pow_f64, cold_sqrt_f64};

/// Rescaled discrete-time parameters for a host sample rate $f_{s,\text{host}} \in [44100, 96000]\text{ Hz}$
/// per [`docs/dnam_container_schema.md` §4](../../../../docs/dnam_container_schema.md#4-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-441--96-khz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZohRescaledParams {
    /// Host sample rate $f_{s,\text{host}}$ in Hz.
    pub sample_rate_host_hz: u32,
    /// Sample-rate ratio $\rho_{\text{sr}} = f_{s,\text{train}} / f_{s,\text{host}}$.
    pub rho_sr: f64,
    /// Host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}}) = \max(8, \operatorname{round}(f_{s,\text{host}} / 3000))$.
    pub control_decimation_host: usize,
    /// Exact control-bus tick ratio $\rho_{\text{ctrl}} = \frac{f_{s,\text{train}} K_{\text{ctrl}}(f_{s,\text{host}})}{f_{s,\text{host}} K_{\text{ctrl}}(f_{s,\text{train}})}$
    /// keeping LFO frequencies in Hz 100% invariant across $44.1 / 48 / 88.2 / 96\text{ kHz}$.
    pub rho_ctrl: f64,
    /// Rescaled Core 1A $5\text{ Hz}$ AC-coupling pole $R_{\text{ac}}(f_{s,\text{host}}) = R_{\text{ac}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_ac_host: f32,
    /// Rescaled Core 2 $5\text{ Hz}$ DC-blocker pole $R_{\text{dc}}(f_{s,\text{host}}) = R_{\text{dc}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_dc_host: f32,
    /// Rescaled Core 2 TZF reference tap $\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{host}}) = \tau_{\text{ref}}(f_{s,\text{train}}) / \rho_{\text{sr}}$.
    pub tau_ref_samples_host: f32,
    /// Rescaled Core 2 switch crossfade length $L_{\text{xfade}}(f_{s,\text{host}}) = \operatorname{round}(L_{\text{xfade}}(f_{s,\text{train}}) / \rho_{\text{sr}})$.
    pub switch_xfade_samples_host: u32,
}

impl ZohRescaledParams {
    /// Convenience constructor with default DNAM training anchors (`k_ctrl=16`, `r_ac=0.9993`, `r_dc=0.995`, `tau_ref=64.0`, `xfade=128`).
    #[inline]
    pub fn new(fs_train_hz: u32, fs_host_hz: u32) -> Self {
        Self::compute(fs_train_hz, fs_host_hz, 16, 0.9993, 0.995, 64.0, 128).unwrap_or(Self {
            sample_rate_host_hz: fs_train_hz,
            rho_sr: 1.0,
            control_decimation_host: 16,
            rho_ctrl: 1.0,
            r_ac_host: 0.9993,
            r_dc_host: 0.995,
            tau_ref_samples_host: 64.0,
            switch_xfade_samples_host: 128,
        })
    }

    /// Computes the closed-form ZOH rescaled parameters for `fs_host_hz` given training anchors.
    /// When `fs_host_hz == fs_train_hz`, returns exact identity factors with `0` drift.
    pub fn compute(
        fs_train_hz: u32,
        fs_host_hz: u32,
        k_ctrl_train: u8,
        r_ac_train: f32,
        r_dc_train: f32,
        tau_ref_train: f32,
        xfade_train: u32,
    ) -> Result<Self, DnamError> {
        if !(44_100..=96_000).contains(&fs_host_hz) {
            return Err(DnamError::InvalidSampleRate(fs_host_hz));
        }
        if fs_host_hz == fs_train_hz {
            return Ok(Self {
                sample_rate_host_hz: fs_host_hz,
                rho_sr: 1.0,
                control_decimation_host: k_ctrl_train as usize,
                rho_ctrl: 1.0,
                r_ac_host: r_ac_train,
                r_dc_host: r_dc_train,
                tau_ref_samples_host: tau_ref_train,
                switch_xfade_samples_host: xfade_train,
            });
        }

        let fs_train = fs_train_hz as f64;
        let fs_host = fs_host_hz as f64;
        let rho_sr = fs_train / fs_host;

        let k_host_raw = ((fs_host / 3000.0) + 0.5) as usize;
        let k_host = k_host_raw.max(8);
        let rho_ctrl = (fs_train * (k_host as f64)) / (fs_host * (k_ctrl_train as f64));

        let r_ac_host = cold_pow_f64(r_ac_train as f64, rho_sr) as f32;
        let r_dc_host = cold_pow_f64(r_dc_train as f64, rho_sr) as f32;
        let tau_ref_samples_host = ((tau_ref_train as f64) / rho_sr) as f32;
        let switch_xfade_samples_host = (((xfade_train as f64) / rho_sr) + 0.5) as u32;

        Ok(Self {
            sample_rate_host_hz: fs_host_hz,
            rho_sr,
            control_decimation_host: k_host,
            rho_ctrl,
            r_ac_host,
            r_dc_host,
            tau_ref_samples_host,
            switch_xfade_samples_host: switch_xfade_samples_host.max(16),
        })
    }

    /// Rescales a Core 1B `SelectiveRgLru` pole radius $r_c$, angle $\theta_{0,c}$, and $\mathcal{H}_2$-norm
    /// input normalizer $b_{\gamma,c}$ under ZOH (`CL-03`, `CL-04`).
    #[inline]
    pub fn rescale_rglru_channel(
        &self,
        r_low_train: f32,
        dr_span_train: f32,
        theta0_train: f32,
        b_gamma_train: f32,
    ) -> (f32, f32, f32, f32) {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return (r_low_train, dr_span_train, theta0_train, b_gamma_train);
        }
        let r_high_train = (r_low_train + dr_span_train).clamp(0.25, 0.9995);
        let r_low_host = cold_pow_f64(r_low_train as f64, self.rho_sr).clamp(0.25, 0.9995) as f32;
        let r_high_host = cold_pow_f64(r_high_train as f64, self.rho_sr).clamp(0.25, 0.9995) as f32;
        let dr_span_host = (r_high_host - r_low_host).max(0.0);

        let theta_host =
            ((theta0_train as f64) * self.rho_sr).clamp(1e-4, core::f64::consts::PI - 1e-4) as f32;

        let r_bar_train = 0.5 * ((r_low_train + r_high_train) as f64);
        let r_bar_host = 0.5 * ((r_low_host + r_high_host) as f64);
        let norm_train = cold_sqrt_f64((1.0 - r_bar_train * r_bar_train).max(1e-7));
        let norm_host = cold_sqrt_f64((1.0 - r_bar_host * r_bar_host).max(1e-7));
        let b_gamma_host = ((b_gamma_train as f64) * (norm_host / norm_train)) as f32;

        (r_low_host, dr_span_host, theta_host, b_gamma_host)
    }

    /// Rescales a Core 2 base delay $\tau_{0,m}^{\text{samples}}$ and integer Schroeder all-pass delay $D_{\text{ap},m}$ (`CL-06`, `CL-08`).
    #[inline]
    pub fn rescale_core2_delays(&self, tau0_samples_train: f32, d_ap_train: f32) -> (f32, usize) {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return (
                tau0_samples_train.max(4.0),
                (d_ap_train as usize).clamp(4, 255),
            );
        }
        let tau0_host = (((tau0_samples_train as f64) / self.rho_sr) as f32).max(4.0);
        let d_ap_host = ((((d_ap_train as f64) / self.rho_sr) + 0.5) as usize).clamp(4, 255);
        (tau0_host, d_ap_host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_zoh_sample_rate_rescaling_and_lfo_tick_invariance() {
        assert!(ZohRescaledParams::compute(48_000, 12_000, 16, 0.9993, 0.995, 64.0, 128).is_err());

        let zoh48 =
            ZohRescaledParams::compute(48_000, 48_000, 16, 0.9993, 0.995, 64.0, 128).unwrap();
        assert_eq!(
            zoh48.rescale_rglru_channel(0.9, 0.05, 0.5, 0.2),
            (0.9, 0.05, 0.5, 0.2)
        );
        assert_eq!(zoh48.rescale_core2_delays(100.0, 20.0), (100.0, 20));

        let fs_train = 48_000_u32;
        let k_ctrl_train = 16_u8;
        let f_lfo_hz = 2.5_f64;
        let dphi_train =
            2.0 * core::f64::consts::PI * f_lfo_hz * (k_ctrl_train as f64) / (fs_train as f64);

        for &fs_host in &[44_100_u32, 48_000, 88_200, 96_000] {
            let zoh = ZohRescaledParams::compute(
                fs_train,
                fs_host,
                k_ctrl_train,
                0.9993,
                0.995,
                64.0,
                128,
            )
            .unwrap();
            let dphi_host = dphi_train * zoh.rho_ctrl;
            let ticks_per_sec = (fs_host as f64) / (zoh.control_decimation_host as f64);
            let f_lfo_host_hz = (dphi_host * ticks_per_sec) / (2.0 * core::f64::consts::PI);
            assert!((f_lfo_host_hz - f_lfo_hz).abs() < 1e-12);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_zoh_rescaling_bounds(
            fs_host in 44_100_u32..=96_000_u32,
            r_low in 0.30_f32..0.95_f32,
            dr in 0.01_f32..0.04_f32,
            theta in 0.01_f32..3.0_f32,
        ) {
            let zoh = ZohRescaledParams::compute(48_000, fs_host, 16, 0.9993, 0.995, 64.0, 128).unwrap();
            let (rl_h, dr_h, th_h, bg_h) = zoh.rescale_rglru_channel(r_low, dr, theta, 0.25);
            prop_assert!((0.25..=0.9995).contains(&rl_h));
            prop_assert!(rl_h + dr_h <= 0.9995 + 1e-6);
            prop_assert!((1e-4..=core::f32::consts::PI - 1e-4).contains(&th_h));
            prop_assert!(bg_h > 0.0 && bg_h.is_finite());

            let (tau_h, dap_h) = zoh.rescale_core2_delays(64.0, 19.0);
            prop_assert!(tau_h >= 4.0);
            prop_assert!((4..=255).contains(&dap_h));
        }
    }
}
