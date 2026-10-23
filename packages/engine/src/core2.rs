//! Core 2: Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`) (`core2.rs`).
//!
//! Implements Subsystem 2 from [`docs/core2_modulated_ufdn.md`](../../../docs/core2_modulated_ufdn.md):
//! - **Dynamic 64-Byte Aligned `fp16` Delay Buffers ([`DelayScratchStorage`]):** Allocates `0 KB` when
//!   `CORE2_ACTIVE == 0` (`ring_buffer_pow2_log2 == 0`), or $2^{\text{ring\_log2}}$ `fp16` samples per line
//!   when active, with cold-init volatile page pre-faulting (`0` page faults on the real-time thread).
//! - **64-Sample TZF Reference Delay Tap (`CL-06`):** Enables causal Through-Zero Flanging (`-54.99 dB` null).
//! - **4-Tap Cubic Lagrange Fractional Delay Read (`CL-06`) & 128-Sample Switch Crossfade (`CL-07`):**
//!   Division-free polynomial weights with zero-transient click-free delay switching.
//! - **Coprime Schroeder All-Pass Diffuser & Givens+Householder Orthogonal Mixing (`CL-08`):**
//!   Preserves exact unitary energy ($\|\mathbf{U}(\psi)\|_2 = 1.00000000$).
//! - **Gain-Normalized $5\text{ Hz}$ DC Blocker (`CL-09`) & Coercive-Saturating Lyapunov Loop (`CL-10`, `CL-11`):**
//!   Guarantees $S'(0) \equiv 1.0000$ and bounded self-oscillation even when pushed into Dub Echo regeneration.

use core::marker::PhantomData;

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, f32_to_fp16_bits, fp16_bits_to_f32};
use crate::memory::{AlignedFp16Block, DelayScratchStorage};

/// Maximum number of FDN delay lines (`4` in `Lite`, `8` in `Full`/`Ultra`).
pub const CORE2_MAX_LINES: usize = 8;

/// Length of the per-line Schroeder all-pass ring buffer (`256` samples covers max coprime delay `223`).
pub const CORE2_ALLPASS_RING_LEN: usize = 256;
const CORE2_ALLPASS_RING_MASK: usize = CORE2_ALLPASS_RING_LEN - 1;

/// Length of the stereo Through-Zero Flanging (TZF) dry reference ring buffer (`256` samples).
pub const CORE2_TZF_RING_LEN: usize = 256;
const CORE2_TZF_RING_MASK: usize = CORE2_TZF_RING_LEN - 1;

/// Canonical coprime Schroeder all-pass delay lengths in samples at $48\text{ kHz}$.
pub const CORE2_DEFAULT_ALLPASS_DELAYS: [u16; CORE2_MAX_LINES] =
    [19, 31, 47, 73, 103, 139, 179, 223];

/// Unpacked ZOH-rescaled Core 2 `M-UFDN` parameters for a 4-line SIMD group (`Lite = group 0`, `Full/Ultra = groups 0..1`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct MufdnGroupWeights {
    /// Target base delay in samples $\tau_{0,m}^{\text{samples}}(f_{s,\text{host}}) \ge 4.0$ (`CL-06`).
    pub tau0_target_samples: [f32; 4],
    /// Quiet-state linear feedback gain $g_{\text{lin},m} \in [0.0, 0.9990]$ (`CL-10`).
    pub g_lin: [f32; 4],
    /// Pre-inverted coercive knee reciprocal $1 / \sigma_m$ (`0 FDIV` at $48\text{ kHz}$, `CL-10`).
    pub inv_sigma_knee: [f32; 4],
    /// Pre-inverted soft-ceiling scale $\frac{2}{3 V_{\text{ceil},m}}$ (`0 FDIV` at $48\text{ kHz}$, `CL-11`).
    pub two_over_3v_ceil: [f32; 4],
    /// Soft-ceiling amplitude $V_{\text{ceil},m} \in [0.25, 2.0]$ (`CL-11`).
    pub v_ceil: [f32; 4],
    /// Schroeder all-pass diffusion coefficient $a_{\text{diff},m} \in [-0.75, +0.75]$ (`CL-08`).
    pub a_diff: [f32; 4],
    /// Integer Schroeder all-pass delay lengths $D_{\text{ap},m}(f_{s,\text{host}}) \in [4, 255]$.
    pub d_ap_samples: [usize; 4],
    /// Input injection gain from Core 1 tap into FDN line $m$.
    pub b_fdn_in: [f32; 4],
    /// Stereo Left wet output projection weight from FDN line $m$.
    pub c_fdn_left: [f32; 4],
    /// Stereo Right wet output projection weight from FDN line $m$.
    pub c_fdn_right: [f32; 4],
    /// Early-reflection tap delay in samples $\tau_{\text{early},m} \ge 4.0$.
    pub early_tap_samples: [f32; 4],
    /// Early-reflection stereo projection weight.
    pub early_gain: [f32; 4],
}

impl Default for MufdnGroupWeights {
    fn default() -> Self {
        Self {
            tau0_target_samples: [64.0; 4],
            g_lin: [0.0; 4],
            inv_sigma_knee: [1.0; 4],
            two_over_3v_ceil: [2.0 / 3.0; 4],
            v_ceil: [1.0; 4],
            a_diff: [0.0; 4],
            d_ap_samples: [19, 31, 47, 73],
            b_fdn_in: [0.0; 4],
            c_fdn_left: [0.0; 4],
            c_fdn_right: [0.0; 4],
            early_tap_samples: [16.0; 4],
            early_gain: [0.0; 4],
        }
    }
}

/// Real-time state and dynamic `fp16` ring-buffer manager for **Core 2 (`M-UFDN`)** (`#[repr(C, align(64))]`).
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ModulatedUfdn<'a, B: MathBackend> {
    /// Whether Core 2 is active for the currently loaded model (`CORE2_ACTIVE` bit 6).
    pub active: bool,
    /// Current power-of-two ring buffer $\log_2(N_{\text{ring}})$ (`0` when inactive, `8..=15` when active).
    pub ring_log2: u8,
    /// Maximum pre-allocated $\log_2(N_{\text{ring}})$ capacity in `delay_scratch`.
    pub max_capacity_log2: u8,
    /// Bitmask $(2^{\text{ring\_log2}} - 1)$ for branchless power-of-two delay wrapping.
    pub ring_mask: usize,
    /// Write cursor into the `fp16` delay ring buffer (`0..ring_mask`).
    pub write_pos: usize,

    /// Unpacked weights for Group 0 (`FDN 0..3`, `Lite`) and Group 1 (`FDN 4..7`, `Full`/`Ultra`).
    pub groups: [MufdnGroupWeights; 2],
    /// Stereo Givens rotation angle $\psi_{\text{givens}}$ and pre-computed $(\cos\psi, \sin\psi)$.
    pub givens_cos: f32,
    /// Pre-computed $\sin\psi_{\text{givens}}$.
    pub givens_sin: f32,
    /// Native training $5\text{ Hz}$ DC blocker pole $R_{\text{dc}}(f_{s,\text{train}})$.
    pub r_dc_train: f32,
    /// Host ZOH-rescaled $5\text{ Hz}$ DC blocker pole $R_{\text{dc}}(f_{s,\text{host}})$.
    pub r_dc_host: f32,
    /// Native training TZF dry reference delay tap (`64.0` samples at $48\text{ kHz}$).
    pub tau_ref_train: f32,
    /// Host ZOH-rescaled TZF dry reference delay tap $\tau_{\text{ref}}^{\text{samples}}(f_{s,\text{host}})$.
    pub tau_ref_host: f32,
    /// Native training switch crossfade window (`128` samples at $48\text{ kHz}$).
    pub xfade_samples_train: u32,
    /// Host ZOH-rescaled switch crossfade window $L_{\text{xfade}}(f_{s,\text{host}})$.
    pub xfade_samples_host: u32,
    /// Pre-computed reciprocal $1 / L_{\text{xfade}}(f_{s,\text{host}})$ for division-free crossfade (`0 FDIV`).
    pub inv_xfade_samples_host: f32,
    /// 1-pole continuous delay slew coefficient ($\tau \approx 15\text{ ms}$ at $48\text{ kHz}$).
    pub delay_slew_pole: f32,

    // ---- Real-Time State Arrays ----
    /// Smoothed current base delay $\bar{\tau}_{0,m}[n]$ for each of the 8 lines.
    pub tau0_smoothed: [[f32; 4]; 2],
    /// Dual-tap switch crossfade old delay anchor $\tau_{\text{old},m}$ (`CL-07`).
    pub xfade_old_tau: [[f32; 4]; 2],
    /// Remaining crossfade samples per line (`0` = no active switch crossfade).
    pub xfade_remaining: [[u32; 4]; 2],

    /// Previous input $u_{\text{mix},m}[n-1]$ to the $5\text{ Hz}$ in-loop DC blocker (`CL-09`).
    pub dc_x_prev: [[f32; 4]; 2],
    /// Previous output $y_{\text{dc},m}[n-1]$ of the $5\text{ Hz}$ in-loop DC blocker (`CL-09`).
    pub dc_y_prev: [[f32; 4]; 2],

    /// Schroeder all-pass ring buffers (`[2 groups][4 lines][256 samples]`).
    pub allpass_ring: [[[f32; CORE2_ALLPASS_RING_LEN]; 4]; 2],
    /// Write cursor into `allpass_ring` (`0..255`).
    pub allpass_write_pos: usize,

    /// Stereo TZF dry reference ring buffer (`[256 samples][L, R]`).
    pub tzf_dry_ring: [[f32; 2]; CORE2_TZF_RING_LEN],
    /// Write cursor into `tzf_dry_ring` (`0..255`).
    pub tzf_write_pos: usize,

    /// Peak effective feedback loop gain $\max_m g_{\text{eff},m}[n]$ observed in the current block (for telemetry).
    pub max_effective_loop_gain: f32,

    /// Dynamic 64-byte aligned `fp16` delay storage (`0 KB` when inactive; up to `8 × 32,768 × 2 B = 512 KB` or `256 KB`).
    /// Packed as `[ring_len][2 groups * 4 lines = 8 u16]`, where 4 consecutive sample frames = 1 `AlignedFp16Block` (64 B).
    pub delay_scratch: DelayScratchStorage<'a>,

    _backend: PhantomData<B>,
}

impl<'a, B: MathBackend> ModulatedUfdn<'a, B> {
    /// Creates an inactive [`ModulatedUfdn`] with `0` bytes of heap allocation (`0 KB` delay RAM).
    pub fn new_inactive() -> Self {
        Self {
            active: false,
            ring_log2: 0,
            max_capacity_log2: 0,
            ring_mask: 0,
            write_pos: 0,
            groups: [MufdnGroupWeights::default(); 2],
            givens_cos: 1.0,
            givens_sin: 0.0,
            r_dc_train: 0.995,
            r_dc_host: 0.995,
            tau_ref_train: 64.0,
            tau_ref_host: 64.0,
            xfade_samples_train: 128,
            xfade_samples_host: 128,
            inv_xfade_samples_host: 1.0 / 128.0,
            delay_slew_pole: 0.9986,
            tau0_smoothed: [[64.0; 4]; 2],
            xfade_old_tau: [[64.0; 4]; 2],
            xfade_remaining: [[0; 4]; 2],
            dc_x_prev: [[0.0; 4]; 2],
            dc_y_prev: [[0.0; 4]; 2],
            allpass_ring: [[[0.0; CORE2_ALLPASS_RING_LEN]; 4]; 2],
            allpass_write_pos: 0,
            tzf_dry_ring: [[0.0; 2]; CORE2_TZF_RING_LEN],
            tzf_write_pos: 0,
            max_effective_loop_gain: 0.0,
            delay_scratch: DelayScratchStorage::empty(),
            _backend: PhantomData,
        }
    }

    /// Creates a [`ModulatedUfdn`] pre-allocated with `max_ring_log2` capacity (`0` or `8..=15`)
    /// so subsequent `reload_from_arena` calls never allocate heap memory.
    pub fn with_max_capacity(max_ring_log2: u8) -> Self {
        let mut s = Self::new_inactive();
        if max_ring_log2 >= 8 {
            let clamped_log2 = max_ring_log2.min(15);
            let ring_len = 1_usize << clamped_log2;
            // Each sample frame across 8 FDN lines is 8 x u16 = 16 bytes; 4 frames = 64 bytes (1 AlignedFp16Block)
            let num_blocks = (ring_len * 8) / 32;
            s.delay_scratch = DelayScratchStorage::new_owned(num_blocks);
            s.max_capacity_log2 = clamped_log2;
        }
        s
    }

    /// Creates a [`ModulatedUfdn`] backed by a caller-supplied 64-byte aligned `AlignedFp16Block` slice
    /// (zero heap allocation even during cold init).
    pub fn with_borrowed_scratch(scratch: &'a mut [AlignedFp16Block], ring_log2: u8) -> Self {
        let mut s = Self::new_inactive();
        s.delay_scratch = DelayScratchStorage::Borrowed(scratch);
        s.max_capacity_log2 = ring_log2.min(15);
        s
    }

    /// Loads `core2_mufdn_soa` weights from `Arena 0` (`6784..7040`) and `Arena 1` (`20544..20800`),
    /// resizes or reuses `delay_scratch` (`0 KB` if `!core2_active`), and applies ZOH rescaling.
    pub fn load_from_arena(
        &mut self,
        arena: &[u8],
        core2_active: bool,
        ring_log2: u8,
        zoh: &ZohRescaledParams,
    ) {
        self.active = core2_active && ring_log2 >= 8;
        if !self.active {
            self.ring_log2 = 0;
            self.ring_mask = 0;
            self.max_effective_loop_gain = 0.0;
            return;
        }

        let target_log2 = ring_log2.clamp(8, 15);
        if target_log2 > self.max_capacity_log2 {
            let ring_len = 1_usize << target_log2;
            let num_blocks = (ring_len * 8) / 32;
            self.delay_scratch = DelayScratchStorage::new_owned(num_blocks);
            self.max_capacity_log2 = target_log2;
        }
        self.ring_log2 = target_log2;
        self.ring_mask = (1_usize << target_log2) - 1;

        // Unpack Group 0 (Arena 0 at 6784..7040 = 64 floats = 16 x [f32; 4])
        // and Group 1 (Arena 1 at 20544..20800 = 64 floats = 16 x [f32; 4])
        let group_bases = [6784_usize, 20_544_usize];
        for (g, &base) in group_bases.iter().enumerate() {
            if arena.len() < base + 256 {
                continue;
            }
            let tau0_tr = read_f32x4_le(arena, base);
            let g_lin = read_f32x4_le(arena, base + 16);
            let inv_sigma = read_f32x4_le(arena, base + 32);
            let two_over_3v = read_f32x4_le(arena, base + 48);
            let v_ceil = read_f32x4_le(arena, base + 64);
            let a_diff = read_f32x4_le(arena, base + 80);
            let d_ap_tr = read_f32x4_le(arena, base + 96);
            let b_in = read_f32x4_le(arena, base + 112);
            let c_l = read_f32x4_le(arena, base + 128);
            let c_r = read_f32x4_le(arena, base + 144);
            let early_tau_tr = read_f32x4_le(arena, base + 160);
            let early_g = read_f32x4_le(arena, base + 176);
            let globals = read_f32x4_le(arena, base + 192);

            let gw = &mut self.groups[g];
            for lane in 0..4 {
                let raw_tau = if tau0_tr[lane] >= 4.0 {
                    tau0_tr[lane]
                } else {
                    64.0 + (g * 4 + lane) as f32 * 37.0
                };
                let raw_dap = if d_ap_tr[lane] >= 4.0 {
                    d_ap_tr[lane]
                } else {
                    CORE2_DEFAULT_ALLPASS_DELAYS[g * 4 + lane] as f32
                };
                let (tau_h, dap_h) = zoh.rescale_core2_delays(raw_tau, raw_dap);
                gw.tau0_target_samples[lane] = tau_h.min((self.ring_mask.saturating_sub(8)) as f32);
                gw.d_ap_samples[lane] = dap_h;
                gw.g_lin[lane] = g_lin[lane].clamp(0.0, 0.9990);
                gw.inv_sigma_knee[lane] = if inv_sigma[lane] > 1e-4 {
                    inv_sigma[lane]
                } else {
                    1.0
                };
                let vc = if v_ceil[lane] >= 0.25 && v_ceil[lane] <= 2.0 {
                    v_ceil[lane]
                } else {
                    0.985
                };
                gw.v_ceil[lane] = vc;
                gw.two_over_3v_ceil[lane] = if two_over_3v[lane] > 1e-4 {
                    two_over_3v[lane]
                } else {
                    2.0 / (3.0 * vc)
                };
                gw.a_diff[lane] = a_diff[lane].clamp(-0.75, 0.75);
                gw.b_fdn_in[lane] = b_in[lane];
                gw.c_fdn_left[lane] = c_l[lane];
                gw.c_fdn_right[lane] = c_r[lane];
                let (etau_h, _) = zoh.rescale_core2_delays(early_tau_tr[lane].max(4.0), 19.0);
                gw.early_tap_samples[lane] = etau_h;
                gw.early_gain[lane] = early_g[lane];
            }

            if g == 0 {
                let psi = globals[0];
                let (c_psi, s_psi) = B::fast_unit_phasor(
                    psi.clamp(-core::f32::consts::PI + 1e-4, core::f32::consts::PI - 1e-4),
                );
                self.givens_cos = c_psi;
                self.givens_sin = s_psi;
                if globals[1] > 0.9 && globals[1] < 1.0 {
                    self.r_dc_train = globals[1];
                }
                if globals[2] >= 4.0 {
                    self.tau_ref_train = globals[2];
                }
                if globals[3] >= 16.0 {
                    self.xfade_samples_train = globals[3] as u32;
                }
            }
        }

        self.r_dc_host = zoh.r_dc_host;
        self.tau_ref_host = zoh.tau_ref_samples_host;
        self.xfade_samples_host = zoh.switch_xfade_samples_host.max(16);
        self.inv_xfade_samples_host = 1.0 / (self.xfade_samples_host as f32);

        self.reset_state();
    }

    /// Clears all delay lines, Schroeder all-pass buffers, and DC blocker states to silence (`0` allocations).
    pub fn reset_state(&mut self) {
        self.write_pos = 0;
        self.allpass_write_pos = 0;
        self.tzf_write_pos = 0;
        self.dc_x_prev = [[0.0; 4]; 2];
        self.dc_y_prev = [[0.0; 4]; 2];
        self.allpass_ring = [[[0.0; CORE2_ALLPASS_RING_LEN]; 4]; 2];
        self.tzf_dry_ring = [[0.0; 2]; CORE2_TZF_RING_LEN];
        self.xfade_remaining = [[0; 4]; 2];
        self.max_effective_loop_gain = 0.0;

        for g in 0..2 {
            self.tau0_smoothed[g] = self.groups[g].tau0_target_samples;
            self.xfade_old_tau[g] = self.groups[g].tau0_target_samples;
        }
        for blk in self.delay_scratch.as_slice_mut() {
            blk.0.fill(0);
        }
    }

    /// Triggers a click-free 128-sample raised-cosine dual-tap crossfade (`CL-07`) when a switch
    /// changes base delay $\tau_{0,m}$ by $> 64$ samples.
    #[inline]
    pub fn set_target_delay_samples(&mut self, line_idx: usize, new_tau_samples: f32) {
        if line_idx >= CORE2_MAX_LINES {
            return;
        }
        let g = line_idx / 4;
        let lane = line_idx & 3;
        let clamped_tau = new_tau_samples.clamp(4.0, (self.ring_mask.saturating_sub(8)) as f32);
        let old_tau = self.tau0_smoothed[g][lane];
        if (clamped_tau - old_tau).abs() > 64.0 {
            self.xfade_old_tau[g][lane] = old_tau;
            self.xfade_remaining[g][lane] = self.xfade_samples_host;
            self.tau0_smoothed[g][lane] = clamped_tau;
        }
        self.groups[g].tau0_target_samples[lane] = clamped_tau;
    }

    /// Reads 4 `fp16` samples at integer delay `int_delay` for group `g` (`0` or `1`) and converts to `[f32; 4]`.
    #[inline(always)]
    fn read_fp16_group(&self, g: usize, lane: usize, read_idx: usize) -> f32 {
        let blocks = self.delay_scratch.as_slice();
        if blocks.is_empty() {
            return 0.0;
        }
        // Frame `read_idx` occupies 8 u16 values; 4 frames per 32-u16 AlignedFp16Block
        let blk_idx = read_idx >> 2;
        let sub_frame = read_idx & 3;
        let word_idx = sub_frame * 8 + g * 4 + lane;
        let bits = blocks[blk_idx & (blocks.len() - 1)].0[word_idx];
        fp16_bits_to_f32(bits)
    }

    /// Writes 4 `f32` samples for group `g` into the `fp16` ring buffer at `write_idx`.
    #[inline(always)]
    fn write_fp16_group(&mut self, g: usize, write_idx: usize, vals: &[f32; 4]) {
        let blocks = self.delay_scratch.as_slice_mut();
        if blocks.is_empty() {
            return;
        }
        let blk_idx = write_idx >> 2;
        let sub_frame = write_idx & 3;
        let base_word = sub_frame * 8 + g * 4;
        let blk = &mut blocks[blk_idx & (blocks.len() - 1)].0;
        for lane in 0..4 {
            blk[base_word + lane] = f32_to_fp16_bits(vals[lane]);
        }
    }

    /// Evaluates a 4-tap Cubic Lagrange fractional delay read (`CL-06`) across 4 FDN lines in group `g`.
    #[inline(always)]
    fn read_cubic_lagrange_group(&self, g: usize, tau_samples: &[f32; 4]) -> [f32; 4] {
        let max_delay = (self.ring_mask.saturating_sub(4)) as f32;
        let mut d_int = [0_usize; 4];
        let mut mu_arr = [0.0_f32; 4];
        for lane in 0..4 {
            let t = tau_samples[lane].clamp(4.0, max_delay);
            let di = t as usize;
            d_int[lane] = di;
            mu_arr[lane] = t - (di as f32);
        }

        let mu_vec = B::Vec4::from_array(mu_arr);
        let (w_m1, w_0, w_p1, w_p2) = B::cubic_lagrange_weights_4ch(mu_vec);

        let mut s_m1 = [0.0_f32; 4];
        let mut s_0 = [0.0_f32; 4];
        let mut s_p1 = [0.0_f32; 4];
        let mut s_p2 = [0.0_f32; 4];

        let w_pos = self.write_pos;
        let mask = self.ring_mask;
        for lane in 0..4 {
            let di = d_int[lane];
            s_m1[lane] = self.read_fp16_group(g, lane, w_pos.wrapping_sub(di - 1) & mask);
            s_0[lane] = self.read_fp16_group(g, lane, w_pos.wrapping_sub(di) & mask);
            s_p1[lane] = self.read_fp16_group(g, lane, w_pos.wrapping_sub(di + 1) & mask);
            s_p2[lane] = self.read_fp16_group(g, lane, w_pos.wrapping_sub(di + 2) & mask);
        }

        let mut acc = w_m1.mul(B::Vec4::from_array(s_m1));
        acc = acc.fma(w_0, B::Vec4::from_array(s_0));
        acc = acc.fma(w_p1, B::Vec4::from_array(s_p1));
        acc = acc.fma(w_p2, B::Vec4::from_array(s_p2));
        acc.to_array()
    }

    /// Executes one $48\text{ kHz}$ sample through the Unified Stereo In-Loop Diffusive `M-UFDN`
    /// (`0 FDIV`, `0 FSQRT`, `0` heap allocations).
    ///
    /// Returns `[wet_left, wet_right, tzf_dry_left, tzf_dry_right]`.
    #[inline(always)]
    pub fn process_sample(
        &mut self,
        dry_left: f32,
        dry_right: f32,
        core1_tap: &[[f32; 4]; 2],
        delay_mod_samples: &[f32; 8],
        active_fdn_groups: usize,
    ) -> [f32; 4] {
        // Always record dry stereo into TZF reference ring buffer
        let tzf_w = self.tzf_write_pos;
        self.tzf_dry_ring[tzf_w] = [dry_left, dry_right];
        let tzf_delay = (self.tau_ref_host as usize).clamp(1, CORE2_TZF_RING_LEN - 1);
        let tzf_r = tzf_w.wrapping_sub(tzf_delay) & CORE2_TZF_RING_MASK;
        let tzf_dry = self.tzf_dry_ring[tzf_r];
        self.tzf_write_pos = (tzf_w + 1) & CORE2_TZF_RING_MASK;

        if !self.active || active_fdn_groups == 0 {
            return [0.0, 0.0, dry_left, dry_right];
        }

        let groups = active_fdn_groups.clamp(1, 2);
        let slew = self.delay_slew_pole;
        let one_minus_slew = 1.0 - slew;
        let ap_w = self.allpass_write_pos;

        let mut u_ap_out = [[0.0_f32; 4]; 2];
        let mut wet_l = 0.0_f32;
        let mut wet_r = 0.0_f32;

        // 1. Fractional delay read + 128-sample switch crossfade (CL-06, CL-07) + Schroeder all-pass (CL-08)
        for g in 0..groups {
            let mut tau_eff = [0.0_f32; 4];
            for lane in 0..4 {
                let tgt = self.groups[g].tau0_target_samples[lane];
                let sm = slew * self.tau0_smoothed[g][lane] + one_minus_slew * tgt;
                self.tau0_smoothed[g][lane] = sm;
                tau_eff[lane] = (sm + delay_mod_samples[g * 4 + lane]).max(4.0);
            }

            let mut d_read = self.read_cubic_lagrange_group(g, &tau_eff);

            // Check if any line has an active 128-sample raised-cosine switch crossfade (CL-07)
            let mut any_xfade = false;
            for lane in 0..4 {
                if self.xfade_remaining[g][lane] > 0 {
                    any_xfade = true;
                }
            }
            if any_xfade {
                let d_old = self.read_cubic_lagrange_group(g, &self.xfade_old_tau[g]);
                for lane in 0..4 {
                    let rem = self.xfade_remaining[g][lane];
                    if rem > 0 {
                        let prog = 1.0_f32 - (rem as f32) * self.inv_xfade_samples_host;
                        let p_clamped = prog.clamp(0.0, 1.0);
                        // Smooth C^1 Hermite/raised-cosine blend w(p) = p^2 (3 - 2p)
                        let w_new = p_clamped * p_clamped * (3.0 - 2.0 * p_clamped);
                        d_read[lane] = (1.0 - w_new) * d_old[lane] + w_new * d_read[lane];
                        self.xfade_remaining[g][lane] = rem - 1;
                    }
                }
            }

            // Early reflection tap + FDN stereo output projection
            let d_early = self.read_cubic_lagrange_group(g, &self.groups[g].early_tap_samples);
            let gw = &self.groups[g];
            for lane in 0..4 {
                let early = gw.early_gain[lane] * d_early[lane];
                wet_l += gw.c_fdn_left[lane] * d_read[lane] + early;
                wet_r += gw.c_fdn_right[lane] * d_read[lane] + early;

                // In-loop Schroeder all-pass diffuser (CL-08):
                // v_m[n] = -a_diff * d_m[n] + u_ap[n - D_ap]
                // u_ap[n] = d_m[n] + a_diff * v_m[n]
                let d_ap = gw.d_ap_samples[lane] & CORE2_ALLPASS_RING_MASK;
                let ap_r = ap_w.wrapping_sub(d_ap) & CORE2_ALLPASS_RING_MASK;
                let u_delayed = self.allpass_ring[g][lane][ap_r];
                let a = gw.a_diff[lane];
                let d_m = d_read[lane];
                let v_m = -a * d_m + u_delayed;
                let u_ap = d_m + a * v_m;
                self.allpass_ring[g][lane][ap_w] = u_ap;
                u_ap_out[g][lane] = v_m;
            }
        }
        self.allpass_write_pos = (ap_w + 1) & CORE2_ALLPASS_RING_MASK;

        // 2. Orthogonal Givens + Householder Mixing Matrix U(psi) (CL-08, Test 5)
        let mut u_mixed = [[0.0_f32; 4]; 2];
        let c_psi = self.givens_cos;
        let s_psi = self.givens_sin;
        for g in 0..groups {
            // Pairwise 2x2 Givens rotation on (0,1) and (2,3)
            let x0 = u_ap_out[g][0];
            let x1 = u_ap_out[g][1];
            let x2 = u_ap_out[g][2];
            let x3 = u_ap_out[g][3];
            u_mixed[g][0] = c_psi * x0 - s_psi * x1;
            u_mixed[g][1] = s_psi * x0 + c_psi * x1;
            u_mixed[g][2] = c_psi * x2 - s_psi * x3;
            u_mixed[g][3] = s_psi * x2 + c_psi * x3;
        }

        // Symmetric Householder reflection H_M = I_M - (2/M) 1 1^T (M=4 for Lite, M=8 for Full/Ultra)
        if groups == 1 {
            let v = B::Vec4::from_array(u_mixed[0]);
            let mean2 = B::Vec4::splat(0.5 * v.horizontal_sum());
            u_mixed[0] = v.sub(mean2).to_array();
        } else {
            let v0 = B::Vec4::from_array(u_mixed[0]);
            let v1 = B::Vec4::from_array(u_mixed[1]);
            let sum8 = v0.horizontal_sum() + v1.horizontal_sum();
            let scale = B::Vec4::splat(0.25 * sum8); // (2 / 8) * sum8
            u_mixed[0] = v0.sub(scale).to_array();
            u_mixed[1] = v1.sub(scale).to_array();
        }

        // 3. Exact Gain-Normalized 5 Hz In-Loop DC Blocker (CL-09) + Coercive Lyapunov Loop (CL-10, CL-11)
        let w_pos = self.write_pos;
        for g in 0..groups {
            let gw = &self.groups[g];
            // Inject Core 1 tap into mixed FDN state
            let tap_vec = B::Vec4::from_array(core1_tap[g]);
            let b_in_vec = B::Vec4::from_array(gw.b_fdn_in);
            let u_in = B::Vec4::from_array(u_mixed[g])
                .fma(tap_vec, b_in_vec)
                .to_array();

            let mut s_next = [0.0_f32; 4];
            B::mufdn_coercive_loop_4ch(
                &u_in,
                &gw.g_lin,
                &gw.inv_sigma_knee,
                &gw.two_over_3v_ceil,
                &gw.v_ceil,
                self.r_dc_host,
                &mut self.dc_x_prev[g],
                &mut self.dc_y_prev[g],
                &mut s_next,
            );
            for lane in 0..4 {
                if gw.g_lin[lane] > self.max_effective_loop_gain {
                    self.max_effective_loop_gain = gw.g_lin[lane];
                }
            }
            self.write_fp16_group(g, w_pos, &s_next);
        }

        self.write_pos = (w_pos + 1) & self.ring_mask;
        [wet_l, wet_r, tzf_dry[0], tzf_dry[1]]
    }
}

#[inline(always)]
fn read_f32x4_le(bytes: &[u8], off: usize) -> [f32; 4] {
    [
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
        f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]),
        f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]),
        f32::from_le_bytes([
            bytes[off + 12],
            bytes[off + 13],
            bytes[off + 14],
            bytes[off + 15],
        ]),
    ]
}

// ============================================================================
// Standalone `f32` / `f64` Reference DSP Kernels (for `packages/python` & Tests)
// ============================================================================

/// Reference `f64` 4-tap Cubic Lagrange fractional delay weights (`CL-06`, `Test 3`):
/// - $w_{-1}(\mu) = -\frac{1}{6}\mu(\mu - 1)(\mu - 2)$
/// - $w_{0}(\mu) = \frac{1}{2}(\mu + 1)(\mu - 1)(\mu - 2)$
/// - $w_{+1}(\mu) = -\frac{1}{2}\mu(\mu + 1)(\mu - 2)$
/// - $w_{+2}(\mu) = \frac{1}{6}\mu(\mu + 1)(\mu - 1)$
#[inline]
pub fn cubic_lagrange_weights_f64(mu: f64) -> [f64; 4] {
    let mu_p1 = mu + 1.0;
    let mu_m1 = mu - 1.0;
    let mu_m2 = mu - 2.0;
    let prod_m1_m2 = mu_m1 * mu_m2;
    let prod_mu_p1 = mu * mu_p1;
    [
        (-1.0 / 6.0) * mu * prod_m1_m2,
        0.5 * mu_p1 * prod_m1_m2,
        -0.5 * prod_mu_p1 * mu_m2,
        (1.0 / 6.0) * prod_mu_p1 * mu_m1,
    ]
}

/// Reference `f64` 8-channel Orthogonal Givens + Householder mixing matrix $\mathbf{U}(\psi)$ (`CL-08`, `Test 5`):
/// Applies $2 \times 2$ Givens rotations $(\cos\psi, \sin\psi)$ on pairs `(0,1), (2,3), (4,5), (6,7)`,
/// followed by the $8 \times 8$ Householder reflection $\mathbf{H}_8 = \mathbf{I}_8 - \frac{1}{4}\mathbf{1}\mathbf{1}^\top$.
pub fn givens_householder_mix_8ch_f64(x: &[f64; 8], cos_psi: f64, sin_psi: f64) -> [f64; 8] {
    let mut g_rot = [0.0_f64; 8];
    for p in 0..4 {
        let i0 = p * 2;
        let i1 = i0 + 1;
        g_rot[i0] = cos_psi * x[i0] - sin_psi * x[i1];
        g_rot[i1] = sin_psi * x[i0] + cos_psi * x[i1];
    }
    let mut sum = 0.0_f64;
    for &v in &g_rot {
        sum += v;
    }
    let sub = 0.25 * sum;
    let mut out = [0.0_f64; 8];
    for i in 0..8 {
        out[i] = g_rot[i] - sub;
    }
    out
}

/// Reference `f64` single-sample Coercive-Saturating Lyapunov loop step with exact gain-normalized
/// $5\text{ Hz}$ DC blocker (`CL-09`, `CL-10`, `CL-11`, `Test 4`).
#[inline]
pub fn mufdn_coercive_step_f64(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: &mut f64,
    dc_y_prev: &mut f64,
) -> f64 {
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let y_dc = alpha_dc * (u_in - *dc_x_prev) + r_dc * (*dc_y_prev);
    *dc_x_prev = u_in;
    *dc_y_prev = y_dc;

    let ratio = y_dc.abs() / sigma_knee;
    let g_eff = g_lin / (1.0 + ratio * ratio);
    let s_pre = g_eff * y_dc;
    let v_norm = (s_pre * (2.0 / (3.0 * v_ceil))).clamp(-1.0, 1.0);
    v_ceil * v_norm * (1.5 - 0.5 * v_norm * v_norm)
}

/// Processes a contiguous `f64` slice through a single-line reference Coercive-Saturating `M-UFDN` loop.
#[allow(clippy::too_many_arguments)]
pub fn mufdn_slice_f64(
    input: &[f64],
    output: &mut [f64],
    delay_samples: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) {
    let ring_len = (delay_samples + 16).next_power_of_two().max(64);
    let mask = ring_len - 1;
    let mut ring = alloc::vec![0.0_f64; ring_len];
    let mut w_pos = 0_usize;
    let mut dc_x_prev = 0.0_f64;
    let mut dc_y_prev = 0.0_f64;

    for (x, y) in input.iter().zip(output.iter_mut()) {
        let r_pos = w_pos.wrapping_sub(delay_samples) & mask;
        let delayed = ring[r_pos];
        let s_next = mufdn_coercive_step_f64(
            delayed + *x,
            g_lin,
            sigma_knee,
            v_ceil,
            r_dc,
            &mut dc_x_prev,
            &mut dc_y_prev,
        );
        ring[w_pos] = s_next;
        *y = s_next;
        w_pos = (w_pos + 1) & mask;
    }
}
