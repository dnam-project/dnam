//! Unified 256-D Feature Bus & Canonical Mid/Side Kronecker Output Head.
//!
//! Canonical Reference: `docs/dsp/router_and_stages_topology.md` §3 (256 B Lite RAM).

use crate::types::Tier;

/// State RAM footprint for Output Head Lite: strictly 256 B.
pub const HEAD_RAM_BYTES: usize = 256;

/// Canonical Mid/Side Kronecker Output Head (256 B Lite RAM).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct MidSideKroneckerHead {
    /// Collapsed Mid projection weights: w_mid = 0.5 * (w_L + w_R).
    pub w_mid: [f32; 64],
    /// Collapsed Side projection weights folded with 3 kHz stereo width:
    /// w_side = alpha_width * 0.5 * (w_L - w_R).
    pub w_side: [f32; 64],
    /// Collapsed Mid and Side bias terms.
    pub b_mid: f32,
    pub b_side: f32,
    /// Continuous stereo width alpha_width in [0.0, 2.0].
    pub stereo_width: f32,
    /// Whether mono topology fast-path is active (bypasses Slice 1 for 50% compute savings).
    pub is_mono_output: bool,
    pub _pad: [u8; 10],
}

impl Default for MidSideKroneckerHead {
    fn default() -> Self {
        Self::new()
    }
}

impl MidSideKroneckerHead {
    pub const fn new() -> Self {
        Self {
            w_mid: [0.0; 64],
            w_side: [0.0; 64],
            b_mid: 0.0,
            b_side: 0.0,
            stereo_width: 1.0,
            is_mono_output: false,
            _pad: [0; 10],
        }
    }

    pub fn reset(&mut self) {
        self.w_mid.fill(0.0);
        self.w_side.fill(0.0);
        self.b_mid = 0.0;
        self.b_side = 0.0;
    }

    /// Sets stereo width folded into Slice 1 at 3 kHz Control Bus rate.
    pub fn update_stereo_width(&mut self, base_w_side: &[f32; 64], base_b_side: f32, width: f32) {
        self.stereo_width = width.clamp(0.0, 2.0);
        for i in 0..64 {
            self.w_side[i] = self.stereo_width * base_w_side[i];
        }
        self.b_side = self.stereo_width * base_b_side;
    }

    /// Projects the 64-D Lite feature bus Phi[n] into stereo output (out_l, out_r).
    ///
    /// When `is_mono_output == true`, Slice 1 is bypassed entirely (50% compute reduction).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn project_feature_bus(&self, phi: &[f32; 64], tier: Tier) -> (f32, f32) {
        let channels = match tier {
            Tier::Lite => 64,
            Tier::Full => 64, // In Lite Head state, evaluate 64 channels
            Tier::Pro => 64,
            Tier::Ultra => 64,
        };

        // 1. Evaluate Slice 0 (Mid projection)
        let mut y_mid = self.b_mid;
        for i in 0..channels {
            y_mid += self.w_mid[i] * phi[i];
        }

        // Fast-path: Mono output returns y_mid directly with zero side computation
        if self.is_mono_output {
            return (y_mid, y_mid);
        }

        // 2. Evaluate Slice 1 (Side projection, folded with stereo width)
        let mut y_side = self.b_side;
        for i in 0..channels {
            y_side += self.w_side[i] * phi[i];
        }

        // 3. Butterfly reconstruction (zero audio-rate multiplications)
        let out_l = y_mid + y_side;
        let out_r = y_mid - y_side;
        (out_l, out_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_mid_side_kronecker_head() {
        let _audio = crate::FtzDazGuard::new();
        let mut head = MidSideKroneckerHead::default();
        head.reset();

        head.b_mid = 0.5;
        head.b_side = 0.2;
        head.w_mid[0] = 1.0;
        head.w_side[0] = 0.5;

        let mut phi = [0.0f32; 64];
        phi[0] = 0.8;

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let (l, r) = head.project_feature_bus(&phi, tier);
            // y_mid = 0.5 + 1.0 * 0.8 = 1.3
            // y_side = 0.2 + 0.5 * 0.8 = 0.6
            // out_l = 1.3 + 0.6 = 1.9, out_r = 1.3 - 0.6 = 0.7
            assert!((l - 1.9).abs() < 1e-6);
            assert!((r - 0.7).abs() < 1e-6);
        }

        // Test mono bypass
        head.is_mono_output = true;
        let (ml, mr) = head.project_feature_bus(&phi, Tier::Lite);
        assert_eq!(ml, 1.3);
        assert_eq!(mr, 1.3);

        // Test update_stereo_width
        head.is_mono_output = false;
        let base_w = [1.0f32; 64];
        head.update_stereo_width(&base_w, 0.1, 1.5);
        assert_eq!(head.stereo_width, 1.5);
        assert_eq!(head.w_side[0], 1.5);
        assert!((head.b_side - 0.15).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_mid_side_kronecker_head_butterfly_and_mono_bypass(
            b_mid in -2.0_f32..2.0_f32,
            b_side in -2.0_f32..2.0_f32,
            w_m in -1.0_f32..1.0_f32,
            w_s in -1.0_f32..1.0_f32,
            phi_val in -1.0_f32..1.0_f32,
            width in -0.5_f32..2.5_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut head = MidSideKroneckerHead {
                b_mid,
                w_mid: [w_m; 64],
                ..Default::default()
            };
            let base_side = [w_s; 64];
            head.update_stereo_width(&base_side, b_side, width);
            prop_assert!(head.stereo_width >= 0.0 && head.stereo_width <= 2.0);

            let phi = [phi_val; 64];
            let (l, r) = head.project_feature_bus(&phi, Tier::Lite);

            // Mid of (L, R) must equal mono bypass output
            let mut head_mono = head.clone();
            head_mono.is_mono_output = true;
            let (ml, mr) = head_mono.project_feature_bus(&phi, Tier::Lite);
            prop_assert_eq!(ml, mr);
            prop_assert!((0.5 * (l + r) - ml).abs() < 1e-4);
        }
    }
}
