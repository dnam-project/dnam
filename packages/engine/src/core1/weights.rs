//! Weight structures for **Core 1** (`PolyAdaaLayerWeights` and `RgLruLayerWeights`).

/// Pre-unpacked ZOH-rescaled `PolyADAA` layer parameters (`[b_conv, alpha, beta, delta]` across 4 tier groups of 4 channels).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct PolyAdaaLayerWeights {
    /// Per-channel convolution/grid operating-point bias $b_{\text{conv},c}$.
    pub b_conv: [[f32; 4]; 4],
    /// Un-delayed linear bypass gain $\alpha_c$.
    pub alpha: [[f32; 4]; 4],
    /// Odd cubic soft-clipper weight $\beta_c$.
    pub beta: [[f32; 4]; 4],
    /// Even quartic triode asymmetry weight $\delta_c$.
    pub delta: [[f32; 4]; 4],
}

impl Default for PolyAdaaLayerWeights {
    fn default() -> Self {
        Self {
            b_conv: [[0.0; 4]; 4],
            alpha: [[1.0; 4]; 4],
            beta: [[0.0; 4]; 4],
            delta: [[0.0; 4]; 4],
        }
    }
}

/// Pre-unpacked ZOH-rescaled `SelectiveRgLru` layer parameters across 4 tier groups of 4 channels.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct RgLruLayerWeights {
    /// Minimum pole radius $r_{\text{low},c} \in [0.25, 0.9995]$.
    pub r_low: [[f32; 4]; 4],
    /// Pole radius span $\Delta r_c = r_{\text{high},c} - r_{\text{low},c} \ge 0$.
    pub dr_span: [[f32; 4]; 4],
    /// Base pole angle $\theta_{0,c} \in [10^{-4}, \pi - 10^{-4}]$.
    pub theta0_rad: [[f32; 4]; 4],
    /// $\mathcal{H}_2$-normalized input projection $b_{\gamma,c} = b_c \sqrt{1 - \bar{r}_c^2}$.
    pub b_gamma_norm: [[f32; 4]; 4],
    /// Selectivity gate weight $w_{\text{sel},c}$.
    pub w_sel: [[f32; 4]; 4],
    /// Real output projection $c_{\text{real},c}$.
    pub c_real: [[f32; 4]; 4],
    /// Imaginary output projection $c_{\text{imag},c}$.
    pub c_imag: [[f32; 4]; 4],
    /// Pre-baked unit phasor $(\cos\theta_{0,c}, \sin\theta_{0,c})$ at current host sample rate.
    pub u_rot_cos: [[f32; 4]; 4],
    /// Pre-baked unit phasor $\sin\theta_{0,c}$ at current host sample rate.
    pub u_rot_sin: [[f32; 4]; 4],
}
