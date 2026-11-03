//! Zero-allocation 2-stage $4\times$ linear-phase polyphase oversampler (`Oversampler4xState`).

/// Exact linear-phase Plugin Delay Compensation (`PDC`) in host samples when `offline_bounce`
/// ($4\times$ 2-stage symmetric half-band FIR polyphase oversampling) is active.
pub const OFFLINE_BOUNCE_PDC_SAMPLES: usize = 12;

/// Stage 1 ($1\times \leftrightarrow 2\times$) 19-tap symmetric half-band polyphase odd-branch taps ($K_1 = 5$).
/// Exact dyadic rationals with denominator $2^{16} = 65536$, summing to $0.50000000$.
const HB_STAGE1_TAPS: [f32; 5] = [
    35.0 / 65536.0,
    -405.0 / 65536.0,
    2268.0 / 65536.0,
    -8820.0 / 65536.0,
    39690.0 / 65536.0,
];

/// Stage 2 ($2\times \leftrightarrow 4\times$) 11-tap symmetric half-band polyphase odd-branch taps ($K_2 = 3$).
/// Exact dyadic rationals with denominator $2^8 = 256$, summing to $0.50000000$.
const HB_STAGE2_TAPS: [f32; 3] = [3.0 / 256.0, -25.0 / 256.0, 150.0 / 256.0];

/// Zero-allocation 2-stage $4\times$ symmetric half-band linear-phase polyphase oversampler state.
///
/// Round-trip group delay is `(2 * K1 - 1) + K2 = 9 + 3 = 12` host samples ([`OFFLINE_BOUNCE_PDC_SAMPLES`]).
#[derive(Clone, Debug)]
pub struct Oversampler4xState {
    up1_hist: [[f32; 2]; 10],
    up2_hist: [[f32; 2]; 6],
    dn2_even_hist: [[f32; 2]; 6],
    dn2_odd_delay: [[f32; 2]; 4],
    stage2_align_z1: [f32; 2],
    dn1_even_hist: [[f32; 2]; 10],
    dn1_odd_delay: [[f32; 2]; 6],
}

impl Default for Oversampler4xState {
    fn default() -> Self {
        Self::new()
    }
}

impl Oversampler4xState {
    /// Creates a zeroed [`Oversampler4xState`].
    #[inline]
    pub const fn new() -> Self {
        Self {
            up1_hist: [[0.0; 2]; 10],
            up2_hist: [[0.0; 2]; 6],
            dn2_even_hist: [[0.0; 2]; 6],
            dn2_odd_delay: [[0.0; 2]; 4],
            stage2_align_z1: [0.0; 2],
            dn1_even_hist: [[0.0; 2]; 10],
            dn1_odd_delay: [[0.0; 2]; 6],
        }
    }

    /// Resets all internal FIR delay lines to zero.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Upsamples one $1\times$ stereo frame `[in_l, in_r]` into four chronological $4\times$ stereo frames.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn upsample_4x(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 4] {
        for k in (1..10).rev() {
            self.up1_hist[k] = self.up1_hist[k - 1];
        }
        self.up1_hist[0] = [in_l, in_r];

        let mut s2x_0 = [0.0_f32; 2];
        for (k, &h) in HB_STAGE1_TAPS.iter().enumerate() {
            let a = self.up1_hist[k];
            let b = self.up1_hist[9 - k];
            s2x_0[0] += h * (a[0] + b[0]);
            s2x_0[1] += h * (a[1] + b[1]);
        }
        let s2x_1 = self.up1_hist[4];
        let s2x = [s2x_0, s2x_1];

        let mut out_4x = [[0.0_f32; 2]; 4];
        for (m, &u2) in s2x.iter().enumerate() {
            for k in (1..6).rev() {
                self.up2_hist[k] = self.up2_hist[k - 1];
            }
            self.up2_hist[0] = u2;

            let mut s4x_0 = [0.0_f32; 2];
            for (k, &h) in HB_STAGE2_TAPS.iter().enumerate() {
                let a = self.up2_hist[k];
                let b = self.up2_hist[5 - k];
                s4x_0[0] += h * (a[0] + b[0]);
                s4x_0[1] += h * (a[1] + b[1]);
            }
            let s4x_1 = self.up2_hist[2];
            out_4x[m * 2] = s4x_0;
            out_4x[m * 2 + 1] = s4x_1;
        }
        out_4x
    }

    /// Downsamples four chronological $4\times$ stereo frames `w4x` into one $1\times$ stereo frame.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn downsample_4x(&mut self, w4x: [[f32; 2]; 4]) -> [f32; 2] {
        let mut v2x = [[0.0_f32; 2]; 2];
        for m in 0..2 {
            let w0 = w4x[m * 2];
            let w1 = w4x[m * 2 + 1];

            for k in (1..6).rev() {
                self.dn2_even_hist[k] = self.dn2_even_hist[k - 1];
            }
            self.dn2_even_hist[0] = w0;

            for k in (1..4).rev() {
                self.dn2_odd_delay[k] = self.dn2_odd_delay[k - 1];
            }
            self.dn2_odd_delay[0] = w1;

            let mut filt2 = [0.0_f32; 2];
            for (k, &h) in HB_STAGE2_TAPS.iter().enumerate() {
                let a = self.dn2_even_hist[k];
                let b = self.dn2_even_hist[5 - k];
                filt2[0] += h * (a[0] + b[0]);
                filt2[1] += h * (a[1] + b[1]);
            }
            let odd2 = self.dn2_odd_delay[3];
            let raw2 = [0.5 * (filt2[0] + odd2[0]), 0.5 * (filt2[1] + odd2[1])];
            v2x[m] = self.stage2_align_z1;
            self.stage2_align_z1 = raw2;
        }

        for k in (1..10).rev() {
            self.dn1_even_hist[k] = self.dn1_even_hist[k - 1];
        }
        self.dn1_even_hist[0] = v2x[0];

        for k in (1..6).rev() {
            self.dn1_odd_delay[k] = self.dn1_odd_delay[k - 1];
        }
        self.dn1_odd_delay[0] = v2x[1];

        let mut filt1 = [0.0_f32; 2];
        for (k, &h) in HB_STAGE1_TAPS.iter().enumerate() {
            let a = self.dn1_even_hist[k];
            let b = self.dn1_even_hist[9 - k];
            filt1[0] += h * (a[0] + b[0]);
            filt1[1] += h * (a[1] + b[1]);
        }
        let odd1 = self.dn1_odd_delay[5];
        [0.5 * (filt1[0] + odd1[0]), 0.5 * (filt1[1] + odd1[1])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oversampler_linear_phase() {
        let _audio = crate::FtzDazGuard::new();
        let mut oversampler = Oversampler4xState::default();
        let mut imp = [0.0_f32; 32];
        imp[0] = 1.0;
        let mut out = [0.0_f32; 32];
        for (i, &s) in imp.iter().enumerate() {
            let up = oversampler.upsample_4x(s, s);
            let dn = oversampler.downsample_4x(up);
            out[i] = dn[0];
        }
        // Group delay is 12 samples
        assert_eq!(
            out.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0,
            12
        );
    }
}
