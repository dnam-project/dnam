//! 12-Allpass (`6` I-branch + `6` Q-branch) `H1` Hilbert Single-Sideband Lattice, 15-Rate Anchor
//! Table, Exact `u128` Ties-to-Even Interpolation, and History-Consistent Rate Remap
//! (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.h1-anchors]`,
//! `ref[dsp.op-m.hilbert.ties-to-even]`, `ref[dsp.op-m.hilbert.rate-remap]`).

use crate::math::flush_recursive_state_f32;

/// 12-Allpass (`6` I-branch + `6` Q-branch) `H1` Hilbert Single-Sideband Lattice (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.h1-anchors]`).
///
/// Dual 6-stage all-pass lattice generating 90-degree quadrature pair `(I[n], Q[n])`:
/// `H_I(z) = prod_{k=0..5} (alpha_{2k} + z^-2) / (1 + alpha_{2k} z^-2)`
/// `H_Q(z) = z^-1 prod_{k=0..5} (alpha_{2k+1} + z^-2) / (1 + alpha_{2k+1} z^-2)`
#[derive(Clone, Debug)]
pub struct HilbertSsbLattice10th {
    pub i_stages: [f32; 6],
    pub i_stages_z2: [f32; 6],
    pub q_stages: [f32; 6],
    pub q_stages_z2: [f32; 6],
    pub q_z1: f32,
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
}

impl Default for HilbertSsbLattice10th {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertSsbLattice10th {
    /// Canonical `48 kHz` I-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`).
    pub const ALPHA_I: [f32; 6] = [
        f32::from_bits(0x3d91_bee6),
        f32::from_bits(0x3eec_b8e1),
        f32::from_bits(0x3f48_de0d),
        f32::from_bits(0x3f6d_91c9),
        f32::from_bits(0x3f7a_5939),
        f32::from_bits(0x3f7e_8e9d),
    ];
    /// Canonical `48 kHz` Q-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`).
    pub const ALPHA_Q: [f32; 6] = [
        f32::from_bits(0x3e80_025f),
        f32::from_bits(0x3f25_f6b0),
        f32::from_bits(0x3f5f_ada1),
        f32::from_bits(0x3f75_b332),
        f32::from_bits(0x3f7c_ffa1),
        f32::from_bits(0x3f7f_92d2),
    ];

    pub const fn new() -> Self {
        Self {
            i_stages: [0.0; 6],
            i_stages_z2: [0.0; 6],
            q_stages: [0.0; 6],
            q_stages_z2: [0.0; 6],
            q_z1: 0.0,
            alpha_i: Self::ALPHA_I,
            alpha_q: Self::ALPHA_Q,
        }
    }

    pub fn reset(&mut self) {
        self.i_stages.fill(0.0);
        self.i_stages_z2.fill(0.0);
        self.q_stages.fill(0.0);
        self.q_stages_z2.fill(0.0);
        self.q_z1 = 0.0;
    }

    /// Recomputes the 12-allpass `H1` coefficients for `sample_rate_hz` (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: f64) {
        if let Some(bits) = interpolate_hilbert_h1_profile_bits(sample_rate_hz) {
            for k in 0..6 {
                self.alpha_i[k] = f32::from_bits(bits[k]);
                self.alpha_q[k] = f32::from_bits(bits[6 + k]);
            }
        }
    }

    /// Steps the Hilbert lattice, producing analytic signal `(I[n], Q[n])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        // I-branch: 6 second-order real-pole all-pass sections in z^-2: (a + z^-2) / (1 + a * z^-2)
        let mut curr_i = x;
        for k in 0..6 {
            let a = self.alpha_i[k];
            let s1 = self.i_stages_z2[k];
            let s0 = self.i_stages[k];
            let y = flush_recursive_state_f32(a * curr_i + s1);
            self.i_stages_z2[k] = s0;
            self.i_stages[k] = flush_recursive_state_f32(a * y - curr_i);
            curr_i = y;
        }

        // Q-branch: z^-1 delay followed by 6 second-order real-pole all-pass sections in z^-2
        let mut curr_q = self.q_z1;
        self.q_z1 = flush_recursive_state_f32(x);
        for k in 0..6 {
            let a = self.alpha_q[k];
            let s1 = self.q_stages_z2[k];
            let s0 = self.q_stages[k];
            let y = flush_recursive_state_f32(a * curr_q + s1);
            self.q_stages_z2[k] = s0;
            self.q_stages[k] = flush_recursive_state_f32(a * y - curr_q);
            curr_q = y;
        }

        (curr_i, curr_q)
    }

    /// Single-sideband frequency shifting by `(cos(w*n), sin(w*n))`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_ssb(&mut self, x: f32, cos_w: f32, sin_w: f32, upper_sideband: bool) -> f32 {
        let (i_val, q_val) = self.step(x);
        if upper_sideband {
            i_val * cos_w - q_val * sin_w
        } else {
            i_val * cos_w + q_val * sin_w
        }
    }
}

/// Canonical 15-rate `H1` Hilbert anchor table (`rate_hz`, 6 `I`-branch + 6 `Q`-branch `f32` bit
/// patterns; `780 B` little-endian binary SHA-256 `bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`).
pub const HILBERT_H1_ANCHORS_15: [(u32, [u32; 12]); 15] = [
    (
        44_100,
        [
            0x3D8EC459, 0x3EE935E1, 0x3F47659D, 0x3F6CCEAA, 0x3F7A0188, 0x3F7E7297, 0x3E7B1AD6,
            0x3F243369, 0x3F5E97F2, 0x3F752D7D, 0x3F7CCA31, 0x3F7F8A32,
        ],
    ),
    (
        48_000,
        [
            0x3D91BEE6, 0x3EECB8E1, 0x3F48DE0D, 0x3F6D91C9, 0x3F7A5939, 0x3F7E8E9D, 0x3E80025F,
            0x3F25F6B0, 0x3F5FADA1, 0x3F75B332, 0x3F7CFFA1, 0x3F7F92D2,
        ],
    ),
    (
        64_000,
        [
            0x3DA2AF27, 0x3EFE4625, 0x3F4FBDA5, 0x3F70B815, 0x3F7B8289, 0x3F7EE2C7, 0x3E8C7CF3,
            0x3F2E9DD3, 0x3F647F75, 0x3F77A752, 0x3F7DA785, 0x3F7FAC3E,
        ],
    ),
    (
        80_000,
        [
            0x3DA88500, 0x3F0212AF, 0x3F523110, 0x3F72069B, 0x3F7C1032, 0x3F7F0F58, 0x3E90CC38,
            0x3F318514, 0x3F665AC6, 0x3F788657, 0x3F7DFC4C, 0x3F7FBA14,
        ],
    ),
    (
        96_000,
        [
            0x3DB073C5, 0x3F05A9C4, 0x3F54E508, 0x3F7343B2, 0x3F7C852B, 0x3F7F30C7, 0x3E963AF7,
            0x3F34F3ED, 0x3F683FD9, 0x3F794AFF, 0x3F7E3E99, 0x3F7FC438,
        ],
    ),
    (
        176_400,
        [
            0x3E0D917D, 0x3F216E28, 0x3F624034, 0x3F77D413, 0x3F7DE528, 0x3F7F8887, 0x3ECDF9DC,
            0x3F49A494, 0x3F703BD8, 0x3F7BD39E, 0x3F7EF730, 0x3F7FDDFB,
        ],
    ),
    (
        192_000,
        [
            0x3E1A79AD, 0x3F2691E2, 0x3F64674D, 0x3F787EBC, 0x3F7E12E1, 0x3F7F9298, 0x3ED99328,
            0x3F4D2EB7, 0x3F717613, 0x3F7C2D21, 0x3F7F0DA9, 0x3F7FE0D4,
        ],
    ),
    (
        208_000,
        [
            0x3E2646E1, 0x3F2A6E87, 0x3F65C7A3, 0x3F78E266, 0x3F7E2EFD, 0x3F7F99AA, 0x3EE33E4A,
            0x3F4F9D5A, 0x3F723264, 0x3F7C621C, 0x3F7F1C5F, 0x3F7FE2ED,
        ],
    ),
    (
        224_000,
        [
            0x3E34BA6C, 0x3F2F8A17, 0x3F67DE51, 0x3F798480, 0x3F7E584A, 0x3F7FA231, 0x3EEF3412,
            0x3F53121B, 0x3F736170, 0x3F7CB521, 0x3F7F3010, 0x3F7FE547,
        ],
    ),
    (
        240_000,
        [
            0x3E4151D9, 0x3F33437E, 0x3F693973, 0x3F79E8DF, 0x3F7E72F8, 0x3F7FA845, 0x3EF8BC96,
            0x3F556DD1, 0x3F741F05, 0x3F7CE93F, 0x3F7F3D5F, 0x3F7FE706,
        ],
    ),
    (
        256_000,
        [
            0x3E4CBD88, 0x3F363B81, 0x3F6A3C44, 0x3F7A3322, 0x3F7E877C, 0x3F7FAD32, 0x3F006617,
            0x3F573C54, 0x3F74AA8C, 0x3F7D1079, 0x3F7F47DE, 0x3F7FE876,
        ],
    ),
    (
        288_000,
        [
            0x3E64086E, 0x3F3BDB10, 0x3F6C20CB, 0x3F7ABA23, 0x3F7EABAF, 0x3F7FB5AD, 0x3F083609,
            0x3F5AA15E, 0x3F75ACD3, 0x3F7D5697, 0x3F7F5A2A, 0x3F7FEAEB,
        ],
    ),
    (
        320_000,
        [
            0x3E7D6662, 0x3F41DCD7, 0x3F6E3EDD, 0x3F7B5343, 0x3F7ED216, 0x3F7FBDBA, 0x3F1068CD,
            0x3F5E589B, 0x3F76D2C7, 0x3F7DA41A, 0x3F7F6C8B, 0x3F7FED2A,
        ],
    ),
    (
        352_000,
        [
            0x3E88E622, 0x3F45A7EB, 0x3F6F6903, 0x3F7BA4C3, 0x3F7EE847, 0x3F7FC310, 0x3F163553,
            0x3F6082FF, 0x3F776EE6, 0x3F7DCEB5, 0x3F7F77E5, 0x3F7FEEB9,
        ],
    ),
    (
        384_000,
        [
            0x3E942913, 0x3F49F3A3, 0x3F70D45F, 0x3F7C094D, 0x3F7F0173, 0x3F7FC869, 0x3F1C84D4,
            0x3F631031, 0x3F783122, 0x3F7E017B, 0x3F7F8401, 0x3F7FF039,
        ],
    ),
];

/// Encodes `HILBERT_H1_ANCHORS_15` into its canonical `780 B` little-endian binary table
/// (`15 × 13 × 4 B`, `ref[dsp.op-m.hilbert.h1-anchors]`).
pub fn encode_hilbert_h1_anchor_table_780b() -> [u8; 780] {
    let mut out = [0u8; 780];
    let mut pos = 0;
    for &(rate, ref coeffs) in &HILBERT_H1_ANCHORS_15 {
        out[pos..pos + 4].copy_from_slice(&rate.to_le_bytes());
        pos += 4;
        for &c in coeffs {
            out[pos..pos + 4].copy_from_slice(&c.to_le_bytes());
            pos += 4;
        }
    }
    out
}

fn f32_anchor_scaled_27(bits: u32) -> u128 {
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let sig = (1u128 << 23) | ((bits & 0x007f_ffff) as u128);
    let shift = 4 + exp;
    sig << (shift as u32)
}

fn round_positive_u128_ratio_to_f32_bits(numerator: u128, denominator: u128) -> Option<u32> {
    if numerator == 0 || numerator >= denominator {
        return None;
    }
    let num_bits = (128 - numerator.leading_zeros()) as i32;
    let den_bits = (128 - denominator.leading_zeros()) as i32;
    let mut exponent = num_bits - den_bits;
    if exponent >= 0 {
        if numerator < (denominator << (exponent as u32)) {
            exponent -= 1;
        }
    } else if (numerator << ((-exponent) as u32)) < denominator {
        exponent -= 1;
    }
    if exponent < -126 {
        return None;
    }
    let shifted_num = numerator << ((23 - exponent) as u32);
    let mut significand = shifted_num / denominator;
    let remainder = shifted_num % denominator;
    if 2 * remainder > denominator || (2 * remainder == denominator && (significand & 1) != 0) {
        significand += 1;
    }
    if significand == (1 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    Some((((exponent + 127) as u32) << 23) | ((significand as u32) - (1 << 23)))
}

/// Interpolates the 12 `H1` Hilbert coefficients (`[alpha_I[0..6], alpha_Q[0..6]]` as `f32` bit
/// patterns) between `low_rate` and `high_rate` at `rate_hz` (`low_rate <= rate_hz <= high_rate`)
/// using exact `u128` ties-to-even arithmetic (`ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_between_anchors_bits(
    rate_hz: f64,
    low_rate: u32,
    high_rate: u32,
) -> Option<[u32; 12]> {
    if !rate_hz.is_finite()
        || low_rate >= high_rate
        || rate_hz < (low_rate as f64)
        || rate_hz > (high_rate as f64)
    {
        return None;
    }
    let low_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == low_rate)?
        .1;
    let high_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == high_rate)?
        .1;

    // Decompose binary64 rate_hz = R / 2^k (0 <= k <= 37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u128 << 52) | ((bits & ((1u64 << 52) - 1)) as u128);
    let (r_int, k_shift): (u128, u32) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 0)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, raw_k - tz)
    };
    let left_weight = ((high_rate as u128) << k_shift) - r_int;
    let right_weight = r_int - ((low_rate as u128) << k_shift);
    let denominator = ((high_rate - low_rate) as u128) << (k_shift + 27);

    let mut out = [0u32; 12];
    for i in 0..12 {
        let a_scaled = f32_anchor_scaled_27(low_coeffs[i]);
        let b_scaled = f32_anchor_scaled_27(high_coeffs[i]);
        let numerator = left_weight * a_scaled + right_weight * b_scaled;
        out[i] = round_positive_u128_ratio_to_f32_bits(numerator, denominator)?;
    }
    Some(out)
}

/// Interpolates the 12 `H1` Hilbert coefficients for any supported `rate_hz` in either the `1×`
/// live band (`44,100..=96,000 Hz`) or the `4×` bounce band (`176,400..=384,000 Hz`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_profile_bits(rate_hz: f64) -> Option<[u32; 12]> {
    if !rate_hz.is_finite() {
        return None;
    }
    const LIVE_BAND: [u32; 5] = [44_100, 48_000, 64_000, 80_000, 96_000];
    const BOUNCE_BAND: [u32; 10] = [
        176_400, 192_000, 208_000, 224_000, 240_000, 256_000, 288_000, 320_000, 352_000, 384_000,
    ];
    for band in [&LIVE_BAND[..], &BOUNCE_BAND[..]] {
        for pair in band.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if rate_hz >= (lo as f64) && rate_hz <= (hi as f64) {
                return interpolate_hilbert_h1_between_anchors_bits(rate_hz, lo, hi);
            }
        }
    }
    None
}

/// Normative 6-section-per-branch `H1` Dual All-Pass Hilbert Lattice (`96 B` raw state + retained
/// prior outputs for history-consistent rate transitions, `ref[dsp.op-m.hilbert.lattice]`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.rate-remap]`).
#[derive(Clone, Debug)]
pub struct HilbertH1Lattice6Sec {
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
    /// Transposed all-pass state `[s[n-1], s[n-2]]` per section.
    pub i_state: [[f32; 2]; 6],
    pub q_state: [[f32; 2]; 6],
    /// Retained section outputs `[y[n-1], y[n-2]]` for history-consistent rate remap (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub i_prev_y: [[f32; 2]; 6],
    pub q_prev_y: [[f32; 2]; 6],
    pub q_z1: f32,
}

impl Default for HilbertH1Lattice6Sec {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl HilbertH1Lattice6Sec {
    pub const fn new_48khz() -> Self {
        Self {
            alpha_i: HilbertSsbLattice10th::ALPHA_I,
            alpha_q: HilbertSsbLattice10th::ALPHA_Q,
            i_state: [[0.0; 2]; 6],
            q_state: [[0.0; 2]; 6],
            i_prev_y: [[0.0; 2]; 6],
            q_prev_y: [[0.0; 2]; 6],
            q_z1: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.i_state = [[0.0; 2]; 6];
        self.q_state = [[0.0; 2]; 6];
        self.i_prev_y = [[0.0; 2]; 6];
        self.q_prev_y = [[0.0; 2]; 6];
        self.q_z1 = 0.0;
    }

    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let bits = interpolate_hilbert_h1_profile_bits(rate_hz)?;
        let alpha_i = core::array::from_fn(|k| f32::from_bits(bits[k]));
        let alpha_q = core::array::from_fn(|k| f32::from_bits(bits[6 + k]));
        Some(Self {
            alpha_i,
            alpha_q,
            i_state: [[0.0; 2]; 6],
            q_state: [[0.0; 2]; 6],
            i_prev_y: [[0.0; 2]; 6],
            q_prev_y: [[0.0; 2]; 6],
            q_z1: 0.0,
        })
    }

    /// Initializes all 12 section states and retained prior outputs to the resting DC equilibrium
    /// for constant input `x_dc` (`y_k = (-1)^(k+1) * x_dc`, `s_k = (-1)^(k+1) * (1 + alpha_k) * x_dc`,
    /// `ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn init_dc_equilibrium(&mut self, x_dc: f32) {
        let mut vi = x_dc;
        let mut vq = x_dc;
        self.q_z1 = x_dc;
        for k in 0..6 {
            let yi = -vi;
            let si = flush_recursive_state_f32(self.alpha_i[k] * yi - vi);
            self.i_state[k] = [si, si];
            self.i_prev_y[k] = [yi, yi];
            vi = yi;

            let yq = -vq;
            let sq = flush_recursive_state_f32(self.alpha_q[k] * yq - vq);
            self.q_state[k] = [sq, sq];
            self.q_prev_y[k] = [yq, yq];
            vq = yq;
        }
    }

    /// Updates the sample rate to `new_rate_hz` and remaps all 12 section states from the retained
    /// prior outputs $s'[n-i] = s[n-i] + (\beta - \alpha) y[n-i]$ (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn retune_sample_rate(&mut self, new_rate_hz: f64) -> Option<()> {
        let bits = interpolate_hilbert_h1_profile_bits(new_rate_hz)?;
        for k in 0..6 {
            let new_ai = f32::from_bits(bits[k]);
            let da_i = new_ai - self.alpha_i[k];
            self.i_state[k][0] =
                flush_recursive_state_f32(self.i_state[k][0] + da_i * self.i_prev_y[k][0]);
            self.i_state[k][1] =
                flush_recursive_state_f32(self.i_state[k][1] + da_i * self.i_prev_y[k][1]);
            self.alpha_i[k] = new_ai;

            let new_aq = f32::from_bits(bits[6 + k]);
            let da_q = new_aq - self.alpha_q[k];
            self.q_state[k][0] =
                flush_recursive_state_f32(self.q_state[k][0] + da_q * self.q_prev_y[k][0]);
            self.q_state[k][1] =
                flush_recursive_state_f32(self.q_state[k][1] + da_q * self.q_prev_y[k][1]);
            self.alpha_q[k] = new_aq;
        }
        Some(())
    }

    /// Steps one sample through the 6-section `H1` analytic pair (`y[n] = alpha * v[n] + s[n-2]`,
    /// `s[n] = alpha * y[n] - v[n]`, `ref[dsp.op-m.hilbert.lattice]`).
    #[inline]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let mut vi = x;
        for k in 0..6 {
            let a = self.alpha_i[k];
            let y = a * vi + self.i_state[k][1];
            self.i_state[k][1] = self.i_state[k][0];
            self.i_state[k][0] = flush_recursive_state_f32(a * y - vi);
            self.i_prev_y[k][1] = self.i_prev_y[k][0];
            self.i_prev_y[k][0] = y;
            vi = y;
        }

        let mut vq = self.q_z1;
        self.q_z1 = x;
        for k in 0..6 {
            let a = self.alpha_q[k];
            let y = a * vq + self.q_state[k][1];
            self.q_state[k][1] = self.q_state[k][0];
            self.q_state[k][0] = flush_recursive_state_f32(a * y - vq);
            self.q_prev_y[k][1] = self.q_prev_y[k][0];
            self.q_prev_y[k][0] = y;
            vq = y;
        }
        (vi, vq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::operators::{
        normalize_tracker_phasor_exponent_split, validate_and_eval_stage3_radial_gate,
    };
    use crate::provenance::sha256_digest;
    use alloc::{format, string::String};
    use proptest::prelude::*;

    #[test]
    fn test_hilbert_ssb_lattice_10th() {
        let _audio = crate::FtzDazGuard::new();
        let mut hilbert = HilbertSsbLattice10th::default();
        hilbert.reset();

        // Impulse energy conservation: I and Q are allpass cascades so sum(I^2) == 1 and sum(Q^2) == 1
        let mut energy_i = 0.0_f64;
        let mut energy_q = 0.0_f64;
        for n in 0..4096 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (i_out, q_out) = hilbert.step(x);
            energy_i += (i_out as f64) * (i_out as f64);
            energy_q += (q_out as f64) * (q_out as f64);
        }
        assert!((energy_i - 1.0).abs() < 1e-3);
        assert!((energy_q - 1.0).abs() < 1e-3);

        // Verify upper vs lower sideband relationship at cos_shift = 0, sin_shift = 1:
        // ssb_up = -Q, ssb_down = +Q (opposite signs for identical state)
        let mut h_up = HilbertSsbLattice10th::default();
        let mut h_dn = HilbertSsbLattice10th::default();
        for n in 0..16 {
            let x = ((n as f32) * 0.3 + 0.1).sin();
            let y_up = h_up.step_ssb(x, 0.0, 1.0, true);
            let y_dn = h_dn.step_ssb(x, 0.0, 1.0, false);
            assert!((y_up + y_dn).abs() < 1e-6);
        }

        // Verify 90-degree quadrature phase shift on a passband sine wave (e.g. f0 = 3 kHz at 48 kHz):
        // After lattice settling, I[n] and Q[n] are orthogonal (sum(I*Q) ~ 0) with constant analytic envelope I^2 + Q^2 ~ 1.0
        let mut h_quad = HilbertSsbLattice10th::default();
        let omega = core::f32::consts::TAU * 3_000.0 / 48_000.0;
        let mut dot_iq = 0.0_f64;
        let mut min_env = f64::INFINITY;
        let mut max_env = 0.0_f64;
        for n in 0..1024 {
            let x = ((n as f32) * omega).sin();
            let (i_v, q_v) = h_quad.step(x);
            if n >= 512 {
                let env2 = (i_v as f64) * (i_v as f64) + (q_v as f64) * (q_v as f64);
                min_env = min_env.min(env2);
                max_env = max_env.max(env2);
                dot_iq += (i_v as f64) * (q_v as f64);
            }
        }
        assert!(
            (max_env - min_env) < 0.10 && (min_env - 1.0).abs() < 0.06,
            "Hilbert quadrature envelope not constant: min_env={min_env}, max_env={max_env}"
        );
        assert!(
            (dot_iq / 512.0).abs() < 0.03,
            "Hilbert I/Q not orthogonal: avg(I*Q)={}",
            dot_iq / 512.0
        );

        hilbert.reset();
        assert_eq!(hilbert.q_z1, 0.0);
    }

    /// verify[dsp.op-m.hilbert.lattice]
    /// verify[dsp.op-m.hilbert.h1-anchors]
    /// verify[dsp.op-m.hilbert.ties-to-even]
    /// verify[dsp.op-m.hilbert.rate-remap]
    #[test]
    fn test_hilbert_h1_anchors_sha256_ties_to_even_quadrature_and_rate_remap() {
        // 1. 780 B H1 anchor table SHA-256
        let table_780b = encode_hilbert_h1_anchor_table_780b();
        let sha_hex: String = sha256_digest(&table_780b)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            sha_hex,
            "bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352"
        );

        // 2. Anchor preservation and exact ties-to-even test cases from specification
        for &(rate, ref expected_coeffs) in &HILBERT_H1_ANCHORS_15 {
            assert_eq!(
                interpolate_hilbert_h1_profile_bits(rate as f64).unwrap(),
                *expected_coeffs
            );
        }
        for &(rate, lo, hi, idx, exp_bits) in &[
            (49_200.0_f64, 48_000u32, 64_000u32, 8usize, 0x3F60_0A2Au32),
            (54_800.0, 48_000, 64_000, 8, 0x3F61_BA02),
            (48_400.0, 48_000, 64_000, 10, 0x3F7D_03D4),
            (80_960.0, 80_000, 96_000, 3, 0x3F72_19A2),
            (81_600.0, 80_000, 96_000, 4, 0x3F7C_1BE4),
            (181_080.0, 176_400, 192_000, 4, 0x3F7D_F2E0),
        ] {
            let coeffs = interpolate_hilbert_h1_between_anchors_bits(rate, lo, hi).unwrap();
            assert_eq!(coeffs[idx], exp_bits);
        }

        // Out-of-band rates are rejected
        for bad_rate in [44_099.0, 100_000.0, 176_399.0, 384_001.0, f64::NAN] {
            assert!(interpolate_hilbert_h1_profile_bits(bad_rate).is_none());
        }

        // 3. History-consistent rate transition remap preserves constant-input equilibrium
        let _guard = FtzDazGuard::new();
        let mut h1 = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        h1.init_dc_equilibrium(1.0);
        let (i_before, q_before) = h1.step(1.0);
        assert!((i_before - 1.0).abs() < 1e-6 && (q_before - 1.0).abs() < 1e-6);

        h1.retune_sample_rate(96_000.0).unwrap();
        let (i_after, q_after) = h1.step(1.0);
        assert!((i_after - 1.0).abs() < 1e-6 && (q_after - 1.0).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.op-m.hilbert.ties-to-even]
        /// verify[dsp.op-m.tracker.envelope-norm]
        /// verify[dsp.op-m.stage3.radial-gate]
        #[test]
        fn prop_op_m_hilbert_interpolation_t1_envelope_and_stage3_radial_gate(
            rate_live in 44_100.0_f64..=96_000.0_f64,
            i_val in -50.0_f32..50.0_f32,
            q_val in -50.0_f32..50.0_f32,
            r_low in 0.85_f32..0.95_f32,
            dr in 0.0_f32..0.049_f32,
            theta_10 in -2.0_f32..2.0_f32,
        ) {
            let coeffs = interpolate_hilbert_h1_profile_bits(rate_live).unwrap();
            for &b in &coeffs {
                let c = f32::from_bits(b);
                prop_assert!(c > 0.0 && c < 1.0);
            }

            if let Some((u_re, u_im, env)) = normalize_tracker_phasor_exponent_split(i_val, q_val) {
                prop_assert!(env >= 0.0 && env.is_finite());
                prop_assert!(u_re * u_re + u_im * u_im <= 1.001);
            }

            let (g_k, r_k, b_eff) =
                validate_and_eval_stage3_radial_gate(r_low, dr, 0.3, 1.0, theta_10).unwrap();
            prop_assert!((0.0..=1.0).contains(&g_k) && b_eff.is_finite());
            prop_assert!(r_k >= r_low && r_k <= r_low + dr && r_k < 1.0);
        }
    }
}
