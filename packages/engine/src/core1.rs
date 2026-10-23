//! Core 1: 12-Layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + `SelectiveRgLru` Stack (`core1.rs`).
//!
//! Implements Subsystem 1 from [`docs/core1_hybrid_ssm_and_polyadaa.md`](../../../docs/core1_hybrid_ssm_and_polyadaa.md)
//! and [`docs/dnam_container_schema.md` §2](../../../docs/dnam_container_schema.md#2-five-hardware-loading-optimizations--50-ns-zero-allocation-load):
//! - **4 Execution-Order Interleaved Blocks (`b = 0..3`):** Each block executes two causal dilated
//!   convolutions ($K=6$, dilations $d \in [1, 3, 7, 19]$) with division-free algebraic `PolyADAA`
//!   (`CL-01`, `CL-02`) followed by a Radial-Polar `SelectiveRgLru` complex resonator layer (`CL-03`..`CL-05`).
//! - **Exact 3-State `PolyADAA` DC Resting Equilibrium:** Initializes $(u_c[-1], \mathcal{A}_c[-1], m_{\text{dc},c}[-1])$
//!   to their analytical fixed point on `reset_to_dc_equilibrium()`, guaranteeing `0.000000` Sample-0 transient.
//! - **16-Tap Collapsed SVD Output Head:** Cached in L1D memory when controls are static (`0` extra MACs/sample)
//!   and evaluated via `B::dot_product_16taps_4ch` (`vDSP_dotpr` on `AppleAmxBackend`).

use core::marker::PhantomData;

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4};

/// Number of interleaved Core 1 blocks (`4`).
pub const CORE1_NUM_BLOCKS: usize = 4;

/// Number of dilated conv + `PolyADAA` layers (`8`).
pub const CORE1_NUM_CONV_LAYERS: usize = 8;

/// Causal dilated convolution kernel width ($K = 6$).
pub const CORE1_CONV_KERNEL: usize = 6;

/// Power-of-two ring buffer length per dilated conv layer (`128` samples covers max delay $(6-1)\times 19 = 95 < 128$).
pub const CORE1_CONV_HIST_LEN: usize = 128;
const CORE1_CONV_HIST_MASK: usize = CORE1_CONV_HIST_LEN - 1;

/// Unfolded output head tap count ($K_{\text{head}} = 16$).
pub const CORE1_HEAD_TAPS: usize = 16;
const CORE1_HEAD_TAP_MASK: usize = CORE1_HEAD_TAPS - 1;

/// Canonical block dilation schedule `[1, 3, 7, 19]`.
pub const CORE1_DILATIONS: [usize; CORE1_NUM_BLOCKS] = [1, 3, 7, 19];

/// Pre-unpacked ZOH-rescaled `PolyADAA` layer parameters (`[b_conv, alpha, beta, delta]` across 3 tier groups of 4 channels).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct PolyAdaaLayerWeights {
    /// Per-channel convolution/grid operating-point bias $b_{\text{conv},c}$.
    pub b_conv: [[f32; 4]; 3],
    /// Un-delayed linear bypass gain $\alpha_c$.
    pub alpha: [[f32; 4]; 3],
    /// Odd cubic soft-clipper weight $\beta_c$.
    pub beta: [[f32; 4]; 3],
    /// Even quartic triode asymmetry weight $\delta_c$.
    pub delta: [[f32; 4]; 3],
}

impl Default for PolyAdaaLayerWeights {
    fn default() -> Self {
        Self {
            b_conv: [[0.0; 4]; 3],
            alpha: [[1.0; 4]; 3],
            beta: [[0.0; 4]; 3],
            delta: [[0.0; 4]; 3],
        }
    }
}

/// Pre-unpacked ZOH-rescaled `SelectiveRgLru` layer parameters across 3 tier groups of 4 channels.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct RgLruLayerWeights {
    /// Minimum pole radius $r_{\text{low},c} \in [0.25, 0.9995]$.
    pub r_low: [[f32; 4]; 3],
    /// Pole radius span $\Delta r_c = r_{\text{high},c} - r_{\text{low},c} \ge 0$.
    pub dr_span: [[f32; 4]; 3],
    /// Base pole angle $\theta_{0,c} \in [10^{-4}, \pi - 10^{-4}]$.
    pub theta0_rad: [[f32; 4]; 3],
    /// $\mathcal{H}_2$-normalized input projection $b_{\gamma,c} = b_c \sqrt{1 - \bar{r}_c^2}$.
    pub b_gamma_norm: [[f32; 4]; 3],
    /// Selectivity gate weight $w_{\text{sel},c}$.
    pub w_sel: [[f32; 4]; 3],
    /// Real output projection $c_{\text{real},c}$.
    pub c_real: [[f32; 4]; 3],
    /// Imaginary output projection $c_{\text{imag},c}$.
    pub c_imag: [[f32; 4]; 3],
    /// Pre-baked unit phasor $(\cos\theta_{0,c}, \sin\theta_{0,c})$ at current host sample rate.
    pub u_rot_cos: [[f32; 4]; 3],
    /// Pre-baked unit phasor $\sin\theta_{0,c}$ at current host sample rate.
    pub u_rot_sin: [[f32; 4]; 3],
}

/// Real-time state and collapsed L1D weight cache for **Core 1** (`#[repr(C, align(64))]`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Core1Stack<B: MathBackend> {
    /// Stereo-to-12ch input projection (`3 tier groups × 4 in × 4 out`).
    pub input_proj: [[[f32; 4]; 4]; 3],
    /// Base + 4 SVD mode biases for the stereo output head (`[5 modes][2 stereo ch]`).
    pub b_head_modes: [[f32; 2]; 5],
    /// Pre-computed linear loudness trims `10^(output_trim_db / 20)` for `[Lite, Full, Ultra]`.
    pub tier_gain_trim_lin: [f32; 3],
    /// Dry bypass mix gain.
    pub mix_dry_gain: f32,
    /// Wet Core 1/2 mix gain.
    pub mix_wet_gain: f32,
    /// Native training $5\text{ Hz}$ AC-coupling pole $R_{\text{ac}}(f_{s,\text{train}})$.
    pub r_ac_train: f32,
    /// Host ZOH-rescaled $5\text{ Hz}$ AC-coupling pole $R_{\text{ac}}(f_{s,\text{host}})$.
    pub r_ac_host: f32,

    /// Unpacked `PolyADAA` weights for all 8 conv layers.
    pub adaa_weights: [PolyAdaaLayerWeights; CORE1_NUM_CONV_LAYERS],
    /// Unpacked ZOH-rescaled `SelectiveRgLru` weights for all 4 blocks.
    pub rglru_weights: [RgLruLayerWeights; CORE1_NUM_BLOCKS],

    /// Collapsed stereo output head weights $\mathbf{W}_{\text{head}}(\mathbf{z})$ (`[2 stereo ch][3 tier groups][16 taps × 4 ch = 64]`).
    pub collapsed_head_w: [[[f32; 64]; 3]; 2],
    /// Collapsed stereo output head bias $\mathbf{b}_{\text{head}}(\mathbf{z})$ (`[L, R]`).
    pub collapsed_head_b: [f32; 2],

    // ---- Real-Time State Buffers (Zero Allocation at 48 kHz) ----
    /// Causal dilated convolution ring buffers (`[8 layers][128 history][3 tier groups][4 ch]`).
    pub conv_hist: [[[[f32; 4]; 3]; CORE1_CONV_HIST_LEN]; CORE1_NUM_CONV_LAYERS],
    /// Write cursor into `conv_hist` (`0..127`).
    pub conv_write_pos: usize,

    /// 3-State `PolyADAA` memory: previous scaled input $u_c[n-1]$.
    pub adaa_u_prev: [[[f32; 4]; 3]; CORE1_NUM_CONV_LAYERS],
    /// 3-State `PolyADAA` memory: previous algebraic segment average $\mathcal{A}_c[n-1]$.
    pub adaa_a_prev: [[[f32; 4]; 3]; CORE1_NUM_CONV_LAYERS],
    /// 3-State `PolyADAA` memory: $5\text{ Hz}$ AC-coupling running mean $m_{\text{dc},c}[n-1]$.
    pub adaa_m_dc: [[[f32; 4]; 3]; CORE1_NUM_CONV_LAYERS],

    /// Complex `SelectiveRgLru` state $\operatorname{Re}(z_{b,c}[n-1])$.
    pub rglru_z_re: [[[f32; 4]; 3]; CORE1_NUM_BLOCKS],
    /// Complex `SelectiveRgLru` state $\operatorname{Im}(z_{b,c}[n-1])$.
    pub rglru_z_im: [[[f32; 4]; 3]; CORE1_NUM_BLOCKS],

    /// 16-tap output head feature ring buffer (`[3 tier groups][16 taps × 4 ch = 64]`).
    pub head_tap_hist: [[f32; 64]; 3],
    /// Write cursor into `head_tap_hist` (`0..15`).
    pub head_write_pos: usize,

    /// Latest 4-channel Core 1 tap output (`ch 0..3` and `ch 4..7`) fed into Core 2 `M-UFDN`.
    pub latest_core1_tap: [[f32; 4]; 2],
    /// Peak absolute `PolyADAA` drive $|u_c|$ observed in the current block (for 60 fps telemetry).
    pub max_adaa_drive: f32,
    /// Peak instantaneous `SelectiveRgLru` pole radius $r_c[n]$ observed in the current block.
    pub max_pole_radius: f32,

    _backend: PhantomData<B>,
}

impl<B: MathBackend> Core1Stack<B> {
    /// Creates a new zero-initialized [`Core1Stack`].
    pub fn new() -> Self {
        Self {
            input_proj: [[[0.0; 4]; 4]; 3],
            b_head_modes: [[0.0; 2]; 5],
            tier_gain_trim_lin: [1.0, 1.0, 1.0],
            mix_dry_gain: 0.0,
            mix_wet_gain: 1.0,
            r_ac_train: 0.9993,
            r_ac_host: 0.9993,
            adaa_weights: [PolyAdaaLayerWeights::default(); CORE1_NUM_CONV_LAYERS],
            rglru_weights: [RgLruLayerWeights::default(); CORE1_NUM_BLOCKS],
            collapsed_head_w: [[[0.0; 64]; 3]; 2],
            collapsed_head_b: [0.0; 2],
            conv_hist: [[[[0.0; 4]; 3]; CORE1_CONV_HIST_LEN]; CORE1_NUM_CONV_LAYERS],
            conv_write_pos: 0,
            adaa_u_prev: [[[0.0; 4]; 3]; CORE1_NUM_CONV_LAYERS],
            adaa_a_prev: [[[0.0; 4]; 3]; CORE1_NUM_CONV_LAYERS],
            adaa_m_dc: [[[0.0; 4]; 3]; CORE1_NUM_CONV_LAYERS],
            rglru_z_re: [[[0.0; 4]; 3]; CORE1_NUM_BLOCKS],
            rglru_z_im: [[[0.0; 4]; 3]; CORE1_NUM_BLOCKS],
            head_tap_hist: [[0.0; 64]; 3],
            head_write_pos: 0,
            latest_core1_tap: [[0.0; 4]; 2],
            max_adaa_drive: 0.0,
            max_pole_radius: 0.0,
            _backend: PhantomData,
        }
    }

    /// Loads `Core1Stack` parameters from `Arena 0..2`, applies ZOH sample-rate rescaling,
    /// collapses the default stereo output head ($\mathbf{z}_{\text{head}} = \mathbf{0}$),
    /// and initializes the 3-state `PolyADAA` DC resting equilibrium.
    pub fn load_from_arena(&mut self, arena: &[u8], zoh: &ZohRescaledParams) {
        if arena.len() < 10_752 {
            return;
        }

        // 1. Arena 0 CL0 (0..64): 4x4 input projection for Lite (ch 0..3)
        for col in 0..4 {
            self.input_proj[0][col] = read_f32x4_le(arena, col * 16);
        }
        // Arena 0 CL1 (64..128): b_head (10 floats), tier_gain_trim_lin (3 floats), [mix_dry, mix_wet, r_ac_train]
        for mode in 0..5 {
            self.b_head_modes[mode][0] = read_f32_le(arena, 64 + (mode * 2) * 4);
            self.b_head_modes[mode][1] = read_f32_le(arena, 64 + (mode * 2 + 1) * 4);
        }
        for t in 0..3 {
            let trim = read_f32_le(arena, 104 + t * 4);
            self.tier_gain_trim_lin[t] = if trim > 1e-4 { trim } else { 1.0 };
        }
        self.mix_dry_gain = read_f32_le(arena, 116);
        let wet = read_f32_le(arena, 120);
        self.mix_wet_gain = if wet.abs() > 1e-6 || self.mix_dry_gain.abs() > 1e-6 {
            wet
        } else {
            1.0
        };
        let r_ac_tr = read_f32_le(arena, 124);
        self.r_ac_train = if r_ac_tr > 0.9 && r_ac_tr < 1.0 {
            r_ac_tr
        } else {
            0.9993
        };
        self.r_ac_host = if zoh.r_ac_host > 0.9 && zoh.r_ac_host < 1.0 {
            zoh.r_ac_host
        } else {
            self.r_ac_train
        };

        // Arena 1 input_proj (10752..10816) and Arena 2 input_proj (22336..22400)
        if arena.len() >= 22_336 {
            for col in 0..4 {
                self.input_proj[1][col] = read_f32x4_le(arena, 10_752 + col * 16);
            }
        }
        if arena.len() >= 36_224 {
            for col in 0..4 {
                self.input_proj[2][col] = read_f32x4_le(arena, 22_336 + col * 16);
            }
        }

        // 2. Unpack PolyADAA & SelectiveRgLru SoA blocks across tiers 0, 1, 2
        let max_tiers = if arena.len() >= 36_224 {
            3
        } else if arena.len() >= 22_336 {
            2
        } else {
            1
        };

        for tier in 0..max_tiers {
            for blk in 0..CORE1_NUM_BLOCKS {
                let (blk_base, adaa0_off, adaa1_off, rglru_off) = match tier {
                    0 => (128 + blk * 1024, 384, 832, 896),
                    1 => (10_816 + blk * 1792, 768, 1600, 1664),
                    _ => (22_400 + blk * 2560, 1152, 2368, 2432),
                };

                for (sub_layer, adaa_rel) in [(0, adaa0_off), (1, adaa1_off)] {
                    let layer_idx = blk * 2 + sub_layer;
                    let base = blk_base + adaa_rel;
                    self.adaa_weights[layer_idx].b_conv[tier] = read_f32x4_le(arena, base);
                    let alpha = read_f32x4_le(arena, base + 16);
                    let beta = read_f32x4_le(arena, base + 32);
                    let delta = read_f32x4_le(arena, base + 48);
                    self.adaa_weights[layer_idx].alpha[tier] = alpha;
                    self.adaa_weights[layer_idx].beta[tier] = beta;
                    self.adaa_weights[layer_idx].delta[tier] = delta;
                }

                // SelectiveRgLru SoA (128 B = 8 x [f32; 4])
                let rg_base = blk_base + rglru_off;
                let r_low_tr = read_f32x4_le(arena, rg_base);
                let dr_tr = read_f32x4_le(arena, rg_base + 16);
                let theta_tr = read_f32x4_le(arena, rg_base + 32);
                let b_gamma_tr = read_f32x4_le(arena, rg_base + 48);
                let w_sel = read_f32x4_le(arena, rg_base + 64);
                let c_real = read_f32x4_le(arena, rg_base + 80);
                let c_imag = read_f32x4_le(arena, rg_base + 96);

                let rg = &mut self.rglru_weights[blk];
                rg.w_sel[tier] = w_sel;
                rg.c_real[tier] = c_real;
                rg.c_imag[tier] = c_imag;

                for lane in 0..4 {
                    let r_l = if r_low_tr[lane] >= 0.25 && r_low_tr[lane] <= 0.9995 {
                        r_low_tr[lane]
                    } else {
                        0.90
                    };
                    let dr = dr_tr[lane].clamp(0.0, 0.9995 - r_l);
                    let th =
                        if theta_tr[lane] > 1e-4 && theta_tr[lane] < core::f32::consts::PI - 1e-4 {
                            theta_tr[lane]
                        } else {
                            0.1
                        };
                    let (r_low_h, dr_h, theta_h, bg_h) =
                        zoh.rescale_rglru_channel(r_l, dr, th, b_gamma_tr[lane]);
                    rg.r_low[tier][lane] = r_low_h;
                    rg.dr_span[tier][lane] = dr_h;
                    rg.theta0_rad[tier][lane] = theta_h;
                    rg.b_gamma_norm[tier][lane] = bg_h;
                    let (c_rot, s_rot) = B::fast_unit_phasor(theta_h);
                    rg.u_rot_cos[tier][lane] = c_rot;
                    rg.u_rot_sin[tier][lane] = s_rot;
                }
            }
        }

        // 3. Collapse stereo output head at default z_head = [0; 4] and reset to DC equilibrium
        self.collapse_stereo_head_3khz(arena, &[0.0; 4], max_tiers);
        self.reset_to_dc_equilibrium();
    }

    /// Resets all convolution, `SelectiveRgLru`, and 3-state `PolyADAA` states to their exact
    /// analytical DC resting equilibrium:
    /// $$u_{\text{rest},c} = \operatorname{clamp}(\tfrac{2}{3} b_{\text{conv},c}, -1, 1), \quad u_c[-1] = u_{\text{rest},c}, \quad \mathcal{A}_c[-1] = m_{\text{dc},c}[-1] = g_{\text{nl},c}(u_{\text{rest},c})$$
    /// guaranteeing **exact `0.000000` Sample-0 and Sample-1 transient on silence**.
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.conv_hist = [[[[0.0; 4]; 3]; CORE1_CONV_HIST_LEN]; CORE1_NUM_CONV_LAYERS];
        self.conv_write_pos = 0;
        self.rglru_z_re = [[[0.0; 4]; 3]; CORE1_NUM_BLOCKS];
        self.rglru_z_im = [[[0.0; 4]; 3]; CORE1_NUM_BLOCKS];
        self.head_tap_hist = [[0.0; 64]; 3];
        self.head_write_pos = 0;
        self.latest_core1_tap = [[0.0; 4]; 2];
        self.max_adaa_drive = 0.0;
        self.max_pole_radius = 0.0;

        for layer in 0..CORE1_NUM_CONV_LAYERS {
            let w = &self.adaa_weights[layer];
            for tier in 0..3 {
                for lane in 0..4 {
                    let b_conv = w.b_conv[tier][lane];
                    let beta = w.beta[tier][lane];
                    let delta = w.delta[tier][lane];
                    let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
                    let u2 = u_rest * u_rest;
                    let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
                    self.adaa_u_prev[layer][tier][lane] = u_rest;
                    self.adaa_a_prev[layer][tier][lane] = g_rest;
                    self.adaa_m_dc[layer][tier][lane] = g_rest;
                }
            }
        }
    }

    /// Collapses the 16-tap SVD stereo output head $\mathbf{W}_{\text{head}}(\mathbf{z}) = \mathbf{W}_{\text{head},0} + \sum_{k=1}^4 z_k \mathbf{W}_{\text{head},k}$
    /// at the $3\text{ kHz}$ control rate (`0` extra MACs/sample at $48\text{ kHz}$ when controls are static).
    #[inline]
    pub fn collapse_stereo_head_3khz(
        &mut self,
        arena: &[u8],
        z_head: &[f32; 4],
        active_tier_groups: usize,
    ) {
        let mut b_l = self.b_head_modes[0][0];
        let mut b_r = self.b_head_modes[0][1];
        for k in 0..4 {
            b_l += z_head[k] * self.b_head_modes[k + 1][0];
            b_r += z_head[k] * self.b_head_modes[k + 1][1];
        }
        self.collapsed_head_b = [b_l, b_r];

        let head_bases = [4224_usize, 17_984_usize, 32_640_usize];
        let z_vecs = [
            B::Vec4::splat(z_head[0]),
            B::Vec4::splat(z_head[1]),
            B::Vec4::splat(z_head[2]),
            B::Vec4::splat(z_head[3]),
        ];

        for tier in 0..active_tier_groups.min(3) {
            let base = head_bases[tier];
            if arena.len() < base + 2560 {
                continue;
            }
            // Mode 0: base + 0..512 (lr=0 is 0..256, lr=1 is 256..512)
            for lr in 0..2 {
                for vec_idx in 0..16 {
                    let off0 = base + lr * 256 + vec_idx * 16;
                    let mut acc = B::Vec4::from_array(read_f32x4_le(arena, off0));
                    for k in 0..4 {
                        if z_head[k].abs() > 1e-12 {
                            let offk = base + (k + 1) * 512 + lr * 256 + vec_idx * 16;
                            let wk = B::Vec4::from_array(read_f32x4_le(arena, offk));
                            acc = acc.fma(z_vecs[k], wk);
                        }
                    }
                    let arr = acc.to_array();
                    let dst = vec_idx * 4;
                    self.collapsed_head_w[lr][tier][dst..dst + 4].copy_from_slice(&arr);
                }
            }
        }
    }

    /// Executes one $48\text{ kHz}$ sample through the 4 interleaved Core 1 blocks and collapsed
    /// 16-tap stereo/mono output head (`0 FDIV`, `0 FSQRT`, `0` heap allocations).
    ///
    /// - `active_tier_groups`: `1` for `Lite` ($C=4$), `2` for `Full` ($C=8$), `3` for `Ultra` ($C=12$).
    /// - `mono_only`: when `true`, skips evaluating the Right output head channel (saving 50% of head MACs).
    #[inline(always)]
    pub fn process_sample(
        &mut self,
        x_left: f32,
        x_right: f32,
        arena: &[u8],
        active_tier_groups: usize,
        mono_only: bool,
    ) -> [f32; 2] {
        let groups = active_tier_groups.clamp(1, 3);
        let x_mid = 0.5 * (x_left + x_right);
        let x_side = 0.5 * (x_left - x_right);
        let in_ms = [x_mid, x_side, 0.0, 0.0];

        // 1. Stereo Mid/Side -> C-channel projection
        let mut h_curr = [[0.0_f32; 4]; 3];
        for tier in 0..groups {
            let mut acc = B::Vec4::zero();
            for col in 0..2 {
                let s = B::Vec4::splat(in_ms[col]);
                let p = B::Vec4::from_array(self.input_proj[tier][col]);
                acc = acc.fma(s, p);
            }
            h_curr[tier] = acc.to_array();
        }

        let w_pos = self.conv_write_pos;
        let two_thirds = B::Vec4::splat(2.0 / 3.0);
        let neg_one = B::Vec4::splat(-1.0);
        let pos_one = B::Vec4::splat(1.0);

        // 2. Walk 4 interleaved Core 1 blocks in execution order
        for blk in 0..CORE1_NUM_BLOCKS {
            let dilation = CORE1_DILATIONS[blk];

            for sub in 0..2 {
                let layer_idx = blk * 2 + sub;
                // Write current activations into causal history ring buffer
                self.conv_hist[layer_idx][w_pos][..groups].copy_from_slice(&h_curr[..groups]);

                let mut y_layer = [[0.0_f32; 4]; 3];
                for tier in 0..groups {
                    let conv_base = match (tier, sub) {
                        (0, 0) => 128 + blk * 1024,
                        (0, _) => 128 + blk * 1024 + 448,
                        (1, 0) => 10_816 + blk * 1792,
                        (1, _) => 10_816 + blk * 1792 + 832,
                        (_, 0) => 22_400 + blk * 2560,
                        (_, _) => 22_400 + blk * 2560 + 1216,
                    };

                    // Initialize conv accumulator with b_conv_c (0 extra instructions!)
                    let mut conv_acc =
                        B::Vec4::from_array(self.adaa_weights[layer_idx].b_conv[tier]);
                    let in_groups = tier + 1;
                    let mut weight_ptr = conv_base;

                    for tap in 0..CORE1_CONV_KERNEL {
                        let hist_idx = w_pos.wrapping_sub(tap * dilation) & CORE1_CONV_HIST_MASK;
                        for ig in 0..in_groups {
                            let h_tap = self.conv_hist[layer_idx][hist_idx][ig];
                            for ic in 0..4 {
                                let x_val = h_tap[ic];
                                if x_val.abs() > 0.0 {
                                    let w_vec =
                                        B::Vec4::from_array(read_f32x4_le(arena, weight_ptr));
                                    conv_acc = conv_acc.fma(B::Vec4::splat(x_val), w_vec);
                                }
                                weight_ptr += 16;
                            }
                        }
                    }

                    let x_conv = conv_acc.to_array();
                    // Track max ADAA drive |u_c| for telemetry
                    let u_curr_arr = conv_acc.mul(two_thirds).clamp(neg_one, pos_one).to_array();
                    for &u_val in &u_curr_arr {
                        let abs_u = u_val.abs();
                        if abs_u > self.max_adaa_drive {
                            self.max_adaa_drive = abs_u;
                        }
                    }

                    let mut y_adaa = [0.0_f32; 4];
                    B::poly_adaa_layer_4ch(
                        &x_conv,
                        &self.adaa_weights[layer_idx].alpha[tier],
                        &self.adaa_weights[layer_idx].beta[tier],
                        &self.adaa_weights[layer_idx].delta[tier],
                        self.r_ac_host,
                        &mut self.adaa_u_prev[layer_idx][tier],
                        &mut self.adaa_a_prev[layer_idx][tier],
                        &mut self.adaa_m_dc[layer_idx][tier],
                        &mut y_adaa,
                    );
                    // Remove static linear bias offset alpha * b_conv so silence equilibrium is exact 0.0
                    let b_vec = B::Vec4::from_array(self.adaa_weights[layer_idx].b_conv[tier]);
                    let a_vec = B::Vec4::from_array(self.adaa_weights[layer_idx].alpha[tier]);
                    let y_zero_dc = B::Vec4::from_array(y_adaa).fms(a_vec, b_vec);

                    // Residual connection: h_next = h_curr + y_zero_dc
                    let h_prev = B::Vec4::from_array(h_curr[tier]);
                    y_layer[tier] = h_prev.add(y_zero_dc).to_array();
                }
                h_curr = y_layer;
            }

            // 3. Radial-Polar SelectiveRgLru layer b
            let rg_w = &self.rglru_weights[blk];
            for tier in 0..groups {
                let mut y_rg = [0.0_f32; 4];
                B::selective_rglru_layer_4ch(
                    &h_curr[tier],
                    &rg_w.r_low[tier],
                    &rg_w.dr_span[tier],
                    &rg_w.w_sel[tier],
                    &rg_w.b_gamma_norm[tier],
                    &rg_w.c_real[tier],
                    &rg_w.c_imag[tier],
                    &rg_w.u_rot_cos[tier],
                    &rg_w.u_rot_sin[tier],
                    &mut self.rglru_z_re[blk][tier],
                    &mut self.rglru_z_im[blk][tier],
                    &mut y_rg,
                );
                for lane in 0..4 {
                    let r_max = rg_w.r_low[tier][lane] + rg_w.dr_span[tier][lane];
                    if r_max > self.max_pole_radius {
                        self.max_pole_radius = r_max;
                    }
                }
                let h_in = B::Vec4::from_array(h_curr[tier]);
                h_curr[tier] = h_in.add(B::Vec4::from_array(y_rg)).to_array();
            }
        }

        self.conv_write_pos = (w_pos + 1) & CORE1_CONV_HIST_MASK;

        // Save latest 4-channel / 8-channel Core 1 tap for Core 2 M-UFDN
        self.latest_core1_tap[0] = h_curr[0];
        self.latest_core1_tap[1] = if groups >= 2 { h_curr[1] } else { h_curr[0] };

        // 4. Write final Core 1 activations into 16-tap unfolded output head ring buffer
        let h_pos = self.head_write_pos;
        let dst_off = h_pos * 4;
        for tier in 0..groups {
            self.head_tap_hist[tier][dst_off..dst_off + 4].copy_from_slice(&h_curr[tier]);
        }
        self.head_write_pos = (h_pos + 1) & CORE1_HEAD_TAP_MASK;

        // 5. Evaluate collapsed 16-tap stereo/mono output head via B::dot_product_16taps_4ch
        let mut out_l = self.collapsed_head_b[0];
        let mut out_r = if mono_only {
            0.0
        } else {
            self.collapsed_head_b[1]
        };

        for tier in 0..groups {
            out_l += B::dot_product_16taps_4ch(
                &self.head_tap_hist[tier],
                &self.collapsed_head_w[0][tier],
            );
            if !mono_only {
                out_r += B::dot_product_16taps_4ch(
                    &self.head_tap_hist[tier],
                    &self.collapsed_head_w[1][tier],
                );
            }
        }

        if mono_only {
            out_r = out_l;
        }
        [out_l, out_r]
    }
}

impl<B: MathBackend> Default for Core1Stack<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn read_f32_le(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

#[inline(always)]
fn read_f32x4_le(bytes: &[u8], off: usize) -> [f32; 4] {
    [
        read_f32_le(bytes, off),
        read_f32_le(bytes, off + 4),
        read_f32_le(bytes, off + 8),
        read_f32_le(bytes, off + 12),
    ]
}

// ============================================================================
// Standalone `f32` / `f64` Reference DSP Kernels (for `packages/python` & Tests)
// ============================================================================

/// Reference `f64` division-free algebraic `PolyADAA` kernel (`CL-01`, `Test 1`):
/// $$\mathcal{A}_c[n] = \beta m_n\big(1.5 - 0.5(m_n^2 + d_n^2)\big) + \delta\Big((m_n^2 + \tfrac{1}{3}d_n^2) - (m_n^4 + 2m_n^2 d_n^2 + \tfrac{1}{5}d_n^4)\Big)$$
#[inline]
pub fn poly_adaa_algebraic_f64(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd = beta * m * (1.5 - 0.5 * (m2 + d2));
    let even = delta * ((m2 + d2 / 3.0) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2));
    odd + even
}

/// Reference `f64` single-sample `PolyADAA` step with $1\times$ passive B-spline weights $(0.875, 0.125)$
/// and $5\text{ Hz}$ AC coupling ($R_{\text{ac}}$, `CL-01`, `CL-02`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_step_f64(
    x: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: &mut f64,
    a_prev: &mut f64,
    m_dc: &mut f64,
) -> f64 {
    let u_curr = (x * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(*u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * (*a_prev);
    let m_dc_curr = r_ac * (*m_dc) + (1.0 - r_ac) * y_nl;
    *u_prev = u_curr;
    *a_prev = a_curr;
    *m_dc = m_dc_curr;
    alpha * x + (y_nl - m_dc_curr)
}

/// Processes a contiguous `f64` audio slice through the reference `PolyADAA` kernel.
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_slice_f64(
    input: &[f64],
    output: &mut [f64],
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
) {
    let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let u2 = u_rest * u_rest;
    let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
    let mut u_prev = u_rest;
    let mut a_prev = g_rest;
    let mut m_dc = g_rest;

    for (x, y) in input.iter().zip(output.iter_mut()) {
        let u_curr = ((*x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
        let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
        let y_nl = 0.875 * a_curr + 0.125 * a_prev;
        let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
        u_prev = u_curr;
        a_prev = a_curr;
        m_dc = m_dc_curr;
        *y = alpha * (*x) + (y_nl - m_dc_curr);
    }
}

/// Reference `f64` Radial-Polar `SelectiveRgLru` single-sample step (`CL-03`..`CL-05`, `Test 2`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_step_f64(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: &mut f64,
    z_im: &mut f64,
) -> f64 {
    let q = (w_sel * x.abs()).min(1.0);
    let g_sel = q * q * (3.0 - 2.0 * q);
    let r = (r_low + dr_span * (1.0 - g_sel)).clamp(0.25, 0.9995);
    let a_re = r * u_rot_cos;
    let a_im = r * u_rot_sin;
    let b_in = b_gamma_norm * x;
    let next_re = a_re * (*z_re) - a_im * (*z_im) + b_in;
    let next_im = a_im * (*z_re) + a_re * (*z_im);
    *z_re = next_re;
    *z_im = next_im;
    c_real * next_re - c_imag * next_im
}

/// Processes a contiguous `f64` slice through the reference Radial-Polar `SelectiveRgLru` kernel.
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_slice_f64(
    input: &[f64],
    output: &mut [f64],
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
) {
    let mut z_re = 0.0;
    let mut z_im = 0.0;
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = selective_rglru_step_f64(
            *x,
            r_low,
            dr_span,
            w_sel,
            b_gamma_norm,
            c_real,
            c_imag,
            u_rot_cos,
            u_rot_sin,
            &mut z_re,
            &mut z_im,
        );
    }
}
