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
    /// Pre-computed static peak pole radius per tier (`[Lite, Full, Ultra]`).
    pub static_max_pole_radius: [f32; 3],

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
    /// SIMD 4-lane accumulator for peak `PolyADAA` drive $|u_c|$ in the current block.
    pub max_adaa_drive_vec: [f32; 4],
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
            static_max_pole_radius: [0.0; 3],
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
            max_adaa_drive_vec: [0.0; 4],
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

        self.static_max_pole_radius = [0.0; 3];

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

                    let r_max = r_low_h + dr_h;
                    for t_idx in tier..3 {
                        if r_max > self.static_max_pole_radius[t_idx] {
                            self.static_max_pole_radius[t_idx] = r_max;
                        }
                    }
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
        self.max_adaa_drive_vec = [0.0; 4];
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

        let head_tile_bases = [66_usize, 281_usize, 510_usize];
        let tiles = arena_as_mat4x4_tiles(arena);
        let z_vecs = [
            B::Vec4::splat(z_head[0]),
            B::Vec4::splat(z_head[1]),
            B::Vec4::splat(z_head[2]),
            B::Vec4::splat(z_head[3]),
        ];

        for tier in 0..active_tier_groups.min(3) {
            let base_tile = head_tile_bases[tier];
            if tiles.len() < base_tile + 40 {
                continue;
            }
            let head_tiles = &tiles[base_tile..base_tile + 40];
            // Each mode is 8 tiles (lr=0 is 0..4, lr=1 is 4..8), each tile has 4 x [f32; 4]
            for lr in 0..2 {
                let (out_chunks, _) = self.collapsed_head_w[lr][tier].as_chunks_mut::<4>();
                for tile_idx in 0..4 {
                    for row in 0..4 {
                        let vec_idx = tile_idx * 4 + row;
                        let mut acc = B::Vec4::load(&head_tiles[lr * 4 + tile_idx][row]);
                        for k in 0..4 {
                            if z_head[k].abs() > 1e-12 {
                                let wk = B::Vec4::load(
                                    &head_tiles[(k + 1) * 8 + lr * 4 + tile_idx][row],
                                );
                                acc = acc.fma(z_vecs[k], wk);
                            }
                        }
                        acc.store(&mut out_chunks[vec_idx]);
                    }
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
        match active_tier_groups {
            1 => self.process_sample_const::<1>(x_left, x_right, arena, mono_only),
            2 => self.process_sample_const::<2>(x_left, x_right, arena, mono_only),
            _ => self.process_sample_const::<3>(x_left, x_right, arena, mono_only),
        }
    }

    #[inline(always)]
    fn process_sample_const<const GROUPS: usize>(
        &mut self,
        x_left: f32,
        x_right: f32,
        arena: &[u8],
        mono_only: bool,
    ) -> [f32; 2] {
        let tiles = arena_as_mat4x4_tiles(arena);
        let min_tiles = match GROUPS {
            1 => 168,
            2 => 349,
            _ => 566,
        };
        if tiles.len() < min_tiles {
            return [0.0; 2];
        }

        let x_mid = 0.5 * (x_left + x_right);
        let x_side = 0.5 * (x_left - x_right);
        let s_mid = B::Vec4::splat(x_mid);
        let s_side = B::Vec4::splat(x_side);

        // 1. Stereo Mid/Side -> C-channel projection (kept in explicit SIMD registers h0, h1, h2)
        let mut h0 = B::Vec4::zero()
            .fma(s_mid, B::Vec4::load(&self.input_proj[0][0]))
            .fma(s_side, B::Vec4::load(&self.input_proj[0][1]));
        let mut h1 = if GROUPS >= 2 {
            B::Vec4::zero()
                .fma(s_mid, B::Vec4::load(&self.input_proj[1][0]))
                .fma(s_side, B::Vec4::load(&self.input_proj[1][1]))
        } else {
            B::Vec4::zero()
        };
        let mut h2 = if GROUPS >= 3 {
            B::Vec4::zero()
                .fma(s_mid, B::Vec4::load(&self.input_proj[2][0]))
                .fma(s_side, B::Vec4::load(&self.input_proj[2][1]))
        } else {
            B::Vec4::zero()
        };

        let w_pos = self.conv_write_pos & CORE1_CONV_HIST_MASK;
        let r_ac = self.r_ac_host;
        let mut max_drive_v = B::Vec4::load(&self.max_adaa_drive_vec);

        // 2. Walk 4 interleaved Core 1 blocks in execution order
        for blk in 0..CORE1_NUM_BLOCKS {
            let dilation = CORE1_DILATIONS[blk];
            // Compute the 5 delayed ring-buffer indices ONCE per block (shared by sub=0 and sub=1)
            let idx1 = w_pos.wrapping_sub(dilation) & CORE1_CONV_HIST_MASK;
            let idx2 = w_pos.wrapping_sub(2 * dilation) & CORE1_CONV_HIST_MASK;
            let idx3 = w_pos.wrapping_sub(3 * dilation) & CORE1_CONV_HIST_MASK;
            let idx4 = w_pos.wrapping_sub(4 * dilation) & CORE1_CONV_HIST_MASK;
            let idx5 = w_pos.wrapping_sub(5 * dilation) & CORE1_CONV_HIST_MASK;

            for sub in 0..2 {
                let layer_idx = blk * 2 + sub;
                h0.store(&mut self.conv_hist[layer_idx][w_pos][0]);
                if GROUPS >= 2 {
                    h1.store(&mut self.conv_hist[layer_idx][w_pos][1]);
                }
                if GROUPS >= 3 {
                    h2.store(&mut self.conv_hist[layer_idx][w_pos][2]);
                }

                let h_layer = &self.conv_hist[layer_idx];
                let h_taps: [&[[f32; 4]; 3]; CORE1_CONV_KERNEL] = [
                    &h_layer[w_pos],
                    &h_layer[idx1],
                    &h_layer[idx2],
                    &h_layer[idx3],
                    &h_layer[idx4],
                    &h_layer[idx5],
                ];
                let adaa_w = &self.adaa_weights[layer_idx];

                // Tier 0 (Lite, 4x4)
                let tile_base0 = 2 + blk * 16 + sub * 7;
                let w_tiles0: &[[[f32; 4]; 4]; 6] =
                    (&tiles[tile_base0..tile_base0 + 6]).try_into().unwrap();
                let conv0 =
                    B::conv6_taps_1group(B::Vec4::load(&adaa_w.b_conv[0]), w_tiles0, &h_taps);
                let (y_dc0, u_abs0) = B::poly_adaa_step_vec4(
                    conv0,
                    &adaa_w.b_conv[0],
                    &adaa_w.alpha[0],
                    &adaa_w.beta[0],
                    &adaa_w.delta[0],
                    r_ac,
                    &mut self.adaa_u_prev[layer_idx][0],
                    &mut self.adaa_a_prev[layer_idx][0],
                    &mut self.adaa_m_dc[layer_idx][0],
                );
                max_drive_v = max_drive_v.max(u_abs0);
                h0 = h0.add(y_dc0);

                // Tier 1 (Full Delta, 4x8)
                if GROUPS >= 2 {
                    let tile_base1 = 169 + blk * 28 + sub * 13;
                    let w_tiles1: &[[[f32; 4]; 4]; 12] =
                        (&tiles[tile_base1..tile_base1 + 12]).try_into().unwrap();
                    let conv1 =
                        B::conv6_taps_2groups(B::Vec4::load(&adaa_w.b_conv[1]), w_tiles1, &h_taps);
                    let (y_dc1, u_abs1) = B::poly_adaa_step_vec4(
                        conv1,
                        &adaa_w.b_conv[1],
                        &adaa_w.alpha[1],
                        &adaa_w.beta[1],
                        &adaa_w.delta[1],
                        r_ac,
                        &mut self.adaa_u_prev[layer_idx][1],
                        &mut self.adaa_a_prev[layer_idx][1],
                        &mut self.adaa_m_dc[layer_idx][1],
                    );
                    max_drive_v = max_drive_v.max(u_abs1);
                    h1 = h1.add(y_dc1);
                }

                // Tier 2 (Ultra Delta, 4x12)
                if GROUPS >= 3 {
                    let tile_base2 = 350 + blk * 40 + sub * 19;
                    let w_tiles2: &[[[f32; 4]; 4]; 18] =
                        (&tiles[tile_base2..tile_base2 + 18]).try_into().unwrap();
                    let conv2 =
                        B::conv6_taps_3groups(B::Vec4::load(&adaa_w.b_conv[2]), w_tiles2, &h_taps);
                    let (y_dc2, u_abs2) = B::poly_adaa_step_vec4(
                        conv2,
                        &adaa_w.b_conv[2],
                        &adaa_w.alpha[2],
                        &adaa_w.beta[2],
                        &adaa_w.delta[2],
                        r_ac,
                        &mut self.adaa_u_prev[layer_idx][2],
                        &mut self.adaa_a_prev[layer_idx][2],
                        &mut self.adaa_m_dc[layer_idx][2],
                    );
                    max_drive_v = max_drive_v.max(u_abs2);
                    h2 = h2.add(y_dc2);
                }
            }

            // 3. Radial-Polar SelectiveRgLru layer b
            let rg_w = &self.rglru_weights[blk];
            let y_rg0 = B::selective_rglru_step_vec4(
                h0,
                &rg_w.r_low[0],
                &rg_w.dr_span[0],
                &rg_w.w_sel[0],
                &rg_w.b_gamma_norm[0],
                &rg_w.c_real[0],
                &rg_w.c_imag[0],
                &rg_w.u_rot_cos[0],
                &rg_w.u_rot_sin[0],
                &mut self.rglru_z_re[blk][0],
                &mut self.rglru_z_im[blk][0],
            );
            h0 = h0.add(y_rg0);

            if GROUPS >= 2 {
                let y_rg1 = B::selective_rglru_step_vec4(
                    h1,
                    &rg_w.r_low[1],
                    &rg_w.dr_span[1],
                    &rg_w.w_sel[1],
                    &rg_w.b_gamma_norm[1],
                    &rg_w.c_real[1],
                    &rg_w.c_imag[1],
                    &rg_w.u_rot_cos[1],
                    &rg_w.u_rot_sin[1],
                    &mut self.rglru_z_re[blk][1],
                    &mut self.rglru_z_im[blk][1],
                );
                h1 = h1.add(y_rg1);
            }

            if GROUPS >= 3 {
                let y_rg2 = B::selective_rglru_step_vec4(
                    h2,
                    &rg_w.r_low[2],
                    &rg_w.dr_span[2],
                    &rg_w.w_sel[2],
                    &rg_w.b_gamma_norm[2],
                    &rg_w.c_real[2],
                    &rg_w.c_imag[2],
                    &rg_w.u_rot_cos[2],
                    &rg_w.u_rot_sin[2],
                    &mut self.rglru_z_re[blk][2],
                    &mut self.rglru_z_im[blk][2],
                );
                h2 = h2.add(y_rg2);
            }
        }

        max_drive_v.store(&mut self.max_adaa_drive_vec);
        let md = self.max_adaa_drive_vec;
        let sample_max_drive = md[0].max(md[1]).max(md[2].max(md[3]));
        if sample_max_drive > self.max_adaa_drive {
            self.max_adaa_drive = sample_max_drive;
        }
        let tier_r_max = self.static_max_pole_radius[GROUPS - 1];
        if tier_r_max > self.max_pole_radius {
            self.max_pole_radius = tier_r_max;
        }

        self.conv_write_pos = (w_pos + 1) & CORE1_CONV_HIST_MASK;

        // Save latest 4-channel / 8-channel Core 1 tap for Core 2 M-UFDN
        h0.store(&mut self.latest_core1_tap[0]);
        if GROUPS >= 2 {
            h1.store(&mut self.latest_core1_tap[1]);
        } else {
            self.latest_core1_tap[1] = self.latest_core1_tap[0];
        }

        // 4. Write final Core 1 activations into 16-tap unfolded output head ring buffer
        let h_pos = self.head_write_pos & CORE1_HEAD_TAP_MASK;
        {
            let (chunks0, _) = self.head_tap_hist[0].as_chunks_mut::<4>();
            h0.store(&mut chunks0[h_pos]);
        }
        if GROUPS >= 2 {
            let (chunks1, _) = self.head_tap_hist[1].as_chunks_mut::<4>();
            h1.store(&mut chunks1[h_pos]);
        }
        if GROUPS >= 3 {
            let (chunks2, _) = self.head_tap_hist[2].as_chunks_mut::<4>();
            h2.store(&mut chunks2[h_pos]);
        }
        self.head_write_pos = (h_pos + 1) & CORE1_HEAD_TAP_MASK;

        // 5. Evaluate collapsed 16-tap stereo/mono output head
        let mut out_l = self.collapsed_head_b[0];
        if mono_only {
            out_l +=
                B::dot_product_16taps_4ch(&self.head_tap_hist[0], &self.collapsed_head_w[0][0]);
            if GROUPS >= 2 {
                out_l +=
                    B::dot_product_16taps_4ch(&self.head_tap_hist[1], &self.collapsed_head_w[0][1]);
            }
            if GROUPS >= 3 {
                out_l +=
                    B::dot_product_16taps_4ch(&self.head_tap_hist[2], &self.collapsed_head_w[0][2]);
            }
            [out_l, out_l]
        } else {
            let mut out_r = self.collapsed_head_b[1];
            let [dl0, dr0] = B::dot_product_16taps_4ch_stereo(
                &self.head_tap_hist[0],
                &self.collapsed_head_w[0][0],
                &self.collapsed_head_w[1][0],
            );
            out_l += dl0;
            out_r += dr0;
            if GROUPS >= 2 {
                let [dl1, dr1] = B::dot_product_16taps_4ch_stereo(
                    &self.head_tap_hist[1],
                    &self.collapsed_head_w[0][1],
                    &self.collapsed_head_w[1][1],
                );
                out_l += dl1;
                out_r += dr1;
            }
            if GROUPS >= 3 {
                let [dl2, dr2] = B::dot_product_16taps_4ch_stereo(
                    &self.head_tap_hist[2],
                    &self.collapsed_head_w[0][2],
                    &self.collapsed_head_w[1][2],
                );
                out_l += dl2;
                out_r += dr2;
            }
            [out_l, out_r]
        }
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

/// Zero-copy view of a 64-byte aligned arena (`CacheLine64`) as 64-byte $4 \times 4$ `f32` matrix tiles.
#[inline(always)]
fn arena_as_mat4x4_tiles(arena: &[u8]) -> &[[[f32; 4]; 4]] {
    const {
        assert!(cfg!(target_endian = "little"));
        assert!(core::mem::size_of::<[[f32; 4]; 4]>() == 64);
    }
    let aligned_len = arena.len() & !63;
    <[[[f32; 4]; 4]] as zerocopy::FromBytes>::ref_from_bytes(&arena[..aligned_len])
        .expect("ArenaStorage byte slice is 4-byte aligned")
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
