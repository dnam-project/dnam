//! Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability ($44.1\text{--}96\text{ kHz}$).

use crate::schema::{DnamError, cold_pow_f64, cold_sqrt_f64};

/// Rescaled discrete-time parameters for a host sample rate $f_{s,\text{host}} \in [44100, 192000]\text{ Hz}$
/// per [`docs/dsp/router_and_stages_topology.md` §9](../../../../docs/dsp/router_and_stages_topology.md#9-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--cl-20-4-bounce)
/// and [`docs/containers/dnam_binary_format.md` §4](../../../../docs/containers/dnam_binary_format.md#4-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--cl-20-4-bounce).
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
    /// Precomputed linear micro-ramp step $\alpha_{\text{ramp}} = 1.0 / K_{\text{ctrl}}(f_{s,\text{host}})$.
    /// Preserves strict 0 FDIV in the real-time audio loop.
    pub alpha_ramp_host: f32,
    /// Rescaled Stage II $5\text{ Hz}$ AC-coupling pole $R_{\text{ac}}(f_{s,\text{host}}) = R_{\text{ac}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_ac_host: f32,
    /// Rescaled Stage IV $5\text{ Hz}$ DC-blocker pole $R_{\text{dc}}(f_{s,\text{host}}) = R_{\text{dc}}(f_{s,\text{train}})^{\rho_{\text{sr}}}$.
    pub r_dc_host: f32,
    /// Rescaled Stage IV TZF reference tap $\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{host}}) = \tau_{\text{ref}}(f_{s,\text{train}}) / \rho_{\text{sr}}$.
    pub tau_ref_samples_host: f32,
    /// Rescaled Stage IV switch crossfade length $L_{\text{xfade}}(f_{s,\text{host}}) = \operatorname{round}(L_{\text{xfade}}(f_{s,\text{train}}) / \rho_{\text{sr}})$.
    pub switch_xfade_samples_host: u32,
}

impl ZohRescaledParams {
    /// Convenience constructor with default DNAM training anchors (`r_ac=0.9993`, `r_dc=0.995`, `tau_ref=64.0`, `xfade=128`).
    #[inline]
    pub fn new(fs_train_hz: u32, fs_host_hz: u32) -> Self {
        Self::compute(fs_train_hz, fs_host_hz, 0.9993, 0.995, 64.0, 128).unwrap_or(Self {
            sample_rate_host_hz: fs_train_hz,
            rho_sr: 1.0,
            control_decimation_host: 16,
            rho_ctrl: 1.0,
            alpha_ramp_host: 0.0625,
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
        r_ac_train: f32,
        r_dc_train: f32,
        tau_ref_train: f32,
        xfade_train: u32,
    ) -> Result<Self, DnamError> {
        Self::compute_oversampled(
            fs_train_hz,
            fs_host_hz,
            1,
            r_ac_train,
            r_dc_train,
            tau_ref_train,
            xfade_train,
        )
    }

    /// Computes the closed-form ZOH rescaled parameters for an internal oversampling rate
    /// $f_{s,\text{internal}} = \text{oversample\_factor} \times f_{s,\text{host}}$
    /// (`oversample_factor` $\in \{1, 2, 4\}$).
    #[allow(clippy::too_many_arguments)]
    pub fn compute_oversampled(
        fs_train_hz: u32,
        fs_host_hz: u32,
        oversample_factor: u32,
        r_ac_train: f32,
        r_dc_train: f32,
        tau_ref_train: f32,
        xfade_train: u32,
    ) -> Result<Self, DnamError> {
        if !(44_100..=192_000).contains(&fs_host_hz) || !matches!(oversample_factor, 1 | 2 | 4) {
            return Err(DnamError::InvalidSampleRate(fs_host_hz));
        }
        let fs_train = fs_train_hz as f64;
        let k_ctrl_train = (((fs_train / 3000.0) + 0.5) as usize).max(8);
        let fs_eff_hz = fs_host_hz * oversample_factor;
        if fs_eff_hz == fs_train_hz {
            let alpha_ramp_host = (1.0 / (k_ctrl_train as f64)) as f32;
            return Ok(Self {
                sample_rate_host_hz: fs_host_hz,
                rho_sr: 1.0,
                control_decimation_host: k_ctrl_train,
                rho_ctrl: 1.0,
                alpha_ramp_host,
                r_ac_host: r_ac_train,
                r_dc_host: r_dc_train,
                tau_ref_samples_host: tau_ref_train,
                switch_xfade_samples_host: xfade_train,
            });
        }

        let fs_eff = fs_eff_hz as f64;
        let rho_sr = fs_train / fs_eff;

        let k_host_raw = ((fs_eff / 3000.0) + 0.5) as usize;
        let k_host = k_host_raw.max(8);
        let rho_ctrl = (fs_train * (k_host as f64)) / (fs_eff * (k_ctrl_train as f64));
        let alpha_ramp_host = (1.0 / (k_host as f64)) as f32;

        let r_ac_host = cold_pow_f64(r_ac_train as f64, rho_sr) as f32;
        let r_dc_host = cold_pow_f64(r_dc_train as f64, rho_sr) as f32;
        let tau_ref_samples_host = ((tau_ref_train as f64) / rho_sr) as f32;
        let switch_xfade_samples_host = (((xfade_train as f64) / rho_sr) + 0.5) as u32;

        Ok(Self {
            sample_rate_host_hz: fs_host_hz,
            rho_sr,
            control_decimation_host: k_host,
            rho_ctrl,
            alpha_ramp_host,
            r_ac_host,
            r_dc_host,
            tau_ref_samples_host,
            switch_xfade_samples_host: switch_xfade_samples_host.max(16),
        })
    }

    /// Rescales an Op-M modal pole radius $r_c$, angle $\theta_{0,c}$, and $\mathcal{H}_2$-norm
    /// input normalizer $b_{\gamma,c}$ under ZOH (`CL-03`, `CL-04`).
    ///
    /// Per canonical TORA specification, pole radii are clamped to `[0.25, 0.9995]` *prior* to
    /// exponentiation by $\rho_{\text{sr}}$ so that physical decay time $T_{60}$ is 100% invariant
    /// across $44.1 / 48 / 88.2 / 96 / 192\text{ kHz}$.
    #[inline]
    pub fn rescale_op_m_channel(
        &self,
        r_low_train: f32,
        dr_span_train: f32,
        theta0_train: f32,
        b_gamma_train: f32,
    ) -> (f32, f32, f32, f32) {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return (r_low_train, dr_span_train, theta0_train, b_gamma_train);
        }
        let r_low_clamped = r_low_train.clamp(0.25, 0.9995);
        let r_high_clamped = (r_low_train + dr_span_train).clamp(0.25, 0.9995);
        let r_low_host = cold_pow_f64(r_low_clamped as f64, self.rho_sr) as f32;
        let r_high_host = cold_pow_f64(r_high_clamped as f64, self.rho_sr) as f32;
        let dr_span_host = (r_high_host - r_low_host).max(0.0);

        let theta_host =
            ((theta0_train as f64) * self.rho_sr).clamp(1e-4, core::f64::consts::PI - 1e-4) as f32;

        let r_bar_train = 0.5 * ((r_low_clamped + r_high_clamped) as f64);
        let r_bar_host = 0.5 * ((r_low_host + r_high_host) as f64);
        let norm_train = cold_sqrt_f64((1.0 - r_bar_train * r_bar_train).max(1e-7));
        let norm_host = cold_sqrt_f64((1.0 - r_bar_host * r_bar_host).max(1e-7));
        let b_gamma_host = ((b_gamma_train as f64) * (norm_host / norm_train)) as f32;

        (r_low_host, dr_span_host, theta_host, b_gamma_host)
    }

    /// Rescales a real 1-pole filter anchor (e.g. Bézout crossover, tone stack, or sag/bias pole)
    /// under continuous-time ZOH via $p_{\text{host}} = p_{\text{train}}^{\rho_{\text{sr}}}$.
    #[inline]
    pub fn rescale_real_pole(&self, p_train: f32) -> f32 {
        if (self.rho_sr - 1.0).abs() < 1e-12 {
            return p_train;
        }
        let p_clamped = p_train.clamp(1e-4, 0.99995);
        cold_pow_f64(p_clamped as f64, self.rho_sr) as f32
    }

    /// Rescales Stage 0 router sub-sample alignment latency:
    /// $\delta_{\text{host}} = \delta_{\text{train}} / \rho_{\text{sr}}$.
    #[inline]
    pub fn rescale_alignment_delay(&self, delay_train: f32) -> f32 {
        ((delay_train as f64) / self.rho_sr) as f32
    }

    /// Returns the 2D quadrature hysteresis differential rate factor $\rho_{\text{sr}}^{-1}$.
    #[inline]
    pub fn hysteresis_diff_scale(&self) -> f32 {
        (1.0 / self.rho_sr) as f32
    }

    /// Returns the Family 5 ZOH bitcrusher phase-step scaling factor $\rho_{\text{sr}}$.
    #[inline]
    pub fn zoh_phase_step_scale(&self) -> f32 {
        self.rho_sr as f32
    }

    /// Rescales an Op-T base delay $\tau_{0,m}^{\text{samples}}$ and integer Schroeder all-pass delay $D_{\text{ap},m}$ (`CL-06`, `CL-08`).
    #[inline]
    pub fn rescale_op_t_delays(&self, tau0_samples_train: f32, d_ap_train: f32) -> (f32, usize) {
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

    /// Rescales FIR taps using 4-tap Catmull-Rom interpolation and normalizes
    /// their sums when both are sufficiently nonzero. The selected Stage III
    /// tier-local rate conversion has not yet been validated against this helper.
    pub fn rescale_fir_taps(&self, train_taps: &[f32], out_taps: &mut [f32]) {
        let in_len = train_taps.len();
        let out_len = out_taps.len();
        if in_len == 0 || out_len == 0 {
            return;
        }
        if (self.rho_sr - 1.0).abs() < 1e-12 && in_len == out_len {
            out_taps.copy_from_slice(train_taps);
            return;
        }
        let mut sum_in = 0.0_f64;
        for &t in train_taps {
            sum_in += t as f64;
        }

        let mut sum_out = 0.0_f64;
        for (n, out) in out_taps.iter_mut().enumerate() {
            let u = (n as f64) * self.rho_sr;
            let i = u.floor() as isize;
            let alpha = (u - (i as f64)) as f32;
            let a2 = alpha * alpha;
            let a3 = a2 * alpha;
            let w_m1 = -0.5 * a3 + a2 - 0.5 * alpha;
            let w_0 = 1.5 * a3 - 2.5 * a2 + 1.0;
            let w_p1 = -1.5 * a3 + 2.0 * a2 + 0.5 * alpha;
            let w_p2 = 0.5 * a3 - 0.5 * a2;

            let get_tap = |idx: isize| -> f32 {
                let clamped = idx.clamp(0, (in_len - 1) as isize) as usize;
                train_taps[clamped]
            };
            let val = w_m1 * get_tap(i - 1)
                + w_0 * get_tap(i)
                + w_p1 * get_tap(i + 1)
                + w_p2 * get_tap(i + 2);
            let val_scaled = (val as f64) * self.rho_sr;
            *out = val_scaled as f32;
            sum_out += *out as f64;
        }

        // DC-preserving normalization
        if sum_out.abs() > 1e-9 && sum_in.abs() > 1e-9 {
            let gain = (sum_in / sum_out) as f32;
            for out in out_taps.iter_mut() {
                *out *= gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_zoh_sample_rate_rescaling_and_lfo_tick_invariance() {
        assert!(ZohRescaledParams::compute(48_000, 12_000, 0.9993, 0.995, 64.0, 128).is_err());

        let zoh48 = ZohRescaledParams::compute(48_000, 48_000, 0.9993, 0.995, 64.0, 128).unwrap();
        assert_eq!(
            zoh48.rescale_op_m_channel(0.9, 0.05, 0.5, 0.2),
            (0.9, 0.05, 0.5, 0.2)
        );
        assert_eq!(zoh48.rescale_op_t_delays(100.0, 20.0), (100.0, 20));
        assert_eq!(zoh48.alpha_ramp_host, 0.0625);

        let fs_train = 48_000_u32;
        let k_ctrl_train = 16_u8;
        let f_lfo_hz = 2.5_f64;
        let dphi_train =
            2.0 * core::f64::consts::PI * f_lfo_hz * (k_ctrl_train as f64) / (fs_train as f64);

        for &fs_host in &[44_100_u32, 48_000, 88_200, 96_000, 192_000] {
            let zoh =
                ZohRescaledParams::compute(fs_train, fs_host, 0.9993, 0.995, 64.0, 128).unwrap();
            let dphi_host = dphi_train * zoh.rho_ctrl;
            let ticks_per_sec = (fs_host as f64) / (zoh.control_decimation_host as f64);
            let f_lfo_host_hz = (dphi_host * ticks_per_sec) / (2.0 * core::f64::consts::PI);
            assert!((f_lfo_host_hz - f_lfo_hz).abs() < 1e-12);
            assert!(
                (zoh.alpha_ramp_host - 1.0 / (zoh.control_decimation_host as f32)).abs() < 1e-7
            );
        }
    }

    #[test]
    fn test_t60_invariance_across_sample_rates() {
        let fs_train = 48_000_f64;
        let r_train = 0.9995_f64;
        let ln10 = 10.0_f64.ln();
        let t60_train = (-3.0 * ln10) / (fs_train * r_train.ln());

        // Test across native sample rates and 4x oversampled bounce (192 kHz)
        for &(fs_host, oversample) in &[
            (44_100_u32, 1),
            (48_000, 1),
            (88_200, 1),
            (96_000, 1),
            (48_000, 4), // 192 kHz effective
            (192_000, 1),
        ] {
            let zoh = ZohRescaledParams::compute_oversampled(
                48_000, fs_host, oversample, 0.9993, 0.995, 64.0, 128,
            )
            .unwrap();
            let (r_host, _, _, _) = zoh.rescale_op_m_channel(r_train as f32, 0.0, 0.5, 1.0);
            let fs_eff = (fs_host * oversample) as f64;
            let t60_host = (-3.0 * ln10) / (fs_eff * (r_host as f64).ln());
            let error = (t60_host - t60_train).abs() / t60_train;
            assert!(
                error < 5e-4,
                "T60 at fs_eff={} Hz ({:.6}s) diverged from train ({:.6}s), error={:.6}",
                fs_eff,
                t60_host,
                t60_train,
                error
            );
        }
    }

    #[test]
    fn test_zoh_extended_helpers_and_fir_resampling() {
        let zoh96 = ZohRescaledParams::compute(48_000, 96_000, 0.9993, 0.995, 64.0, 128).unwrap();
        assert!((zoh96.rho_sr - 0.5).abs() < 1e-12);
        assert!((zoh96.hysteresis_diff_scale() - 2.0).abs() < 1e-6);
        assert!((zoh96.zoh_phase_step_scale() - 0.5).abs() < 1e-6);

        // Real pole rescaling: p_host = p_train^0.5
        let p_train = 0.81_f32;
        let p_host = zoh96.rescale_real_pole(p_train);
        assert!((p_host - 0.9).abs() < 1e-4);

        // Alignment delay: delay_host = delay_train / 0.5 = 2.0 * delay_train
        let d_train = 4.0_f32;
        let d_host = zoh96.rescale_alignment_delay(d_train);
        assert!((d_host - 8.0).abs() < 1e-4);

        // FIR tap resampling with DC preservation
        let train_taps = [0.1_f32, 0.2, 0.4, 0.2, 0.1];
        let mut out_taps = [0.0_f32; 10];
        zoh96.rescale_fir_taps(&train_taps, &mut out_taps);

        let sum_in: f32 = train_taps.iter().sum();
        let sum_out: f32 = out_taps.iter().sum();
        assert!((sum_in - sum_out).abs() < 1e-5);

        // Identity rate & fallback constructor edge cases
        let zoh48_new = ZohRescaledParams::new(48_000, 48_000);
        assert_eq!(zoh48_new.sample_rate_host_hz, 48_000);
        assert_eq!(zoh48_new.rescale_real_pole(0.85), 0.85);
        let mut id_taps = [0.0_f32; 5];
        zoh48_new.rescale_fir_taps(&train_taps, &mut id_taps);
        assert_eq!(id_taps, train_taps);
        zoh48_new.rescale_fir_taps(&[], &mut id_taps);

        let zoh_fallback = ZohRescaledParams::new(48_000, 0);
        assert_eq!(zoh_fallback.sample_rate_host_hz, 48_000);
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
            let zoh = ZohRescaledParams::compute(48_000, fs_host, 0.9993, 0.995, 64.0, 128).unwrap();
            let (rl_h, dr_h, th_h, bg_h) = zoh.rescale_op_m_channel(r_low, dr, theta, 0.25);
            prop_assert!(rl_h >= 0.25);
            prop_assert!(rl_h + dr_h <= 0.9999);
            prop_assert!((1e-4..=core::f32::consts::PI - 1e-4).contains(&th_h));
            prop_assert!(bg_h > 0.0 && bg_h.is_finite());

            let (tau_h, dap_h) = zoh.rescale_op_t_delays(64.0, 19.0);
            prop_assert!(tau_h >= 4.0);
            prop_assert!((4..=255).contains(&dap_h));
        }
    }
}
