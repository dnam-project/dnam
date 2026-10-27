//! Real-time state and dynamic `fp16` ring-buffer manager for **Core 2 (`M-UFDN`)**.

use core::marker::PhantomData;

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4};
use crate::memory::{AlignedFp16Block, DelayScratchStorage};

use super::{
    CORE2_ALLPASS_RING_LEN, CORE2_ALLPASS_RING_MASK, CORE2_DEFAULT_ALLPASS_DELAYS, CORE2_MAX_LINES,
    CORE2_TZF_RING_LEN, CORE2_TZF_RING_MASK,
};

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
    /// Base grain window length in samples $L_{0,m}(f_{s,\text{host}}) \ge 16.0$.
    pub grain_len_samples: [f32; 4],
    /// Effective $3\text{ kHz}$ modulated grain window length $L_m[k] \ge 16.0$.
    pub eff_grain_len_samples: [f32; 4],
    /// Pre-inverted grain window length $1 / L_m[k]$ (`0 FDIV` at $48\text{ kHz}$).
    pub inv_grain_len_samples: [f32; 4],
    /// Base time-warp / pitch-shift rate $\nu_{0,m} = 1 - 2^{\Delta p_m / 12} \in [-2.0, +2.0]$.
    pub grain_warp_rate: [f32; 4],
    /// Effective $3\text{ kHz}$ modulated time-warp rate $\nu_m[k] \in [-2.0, +2.0]$.
    pub eff_grain_warp_rate: [f32; 4],
    /// Self-activating per-line complementary Hermite-Hann window depth $\kappa_{\text{win},m}[k] \in [0.0, 1.0]$.
    pub eff_grain_window_mix: [f32; 4],
    /// Effective $3\text{ kHz}$ modulated quiet-state feedback gain $g_{\text{lin},m}[k] \in [0.0, 0.9990]$.
    pub eff_g_lin: [f32; 4],
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
            grain_len_samples: [2048.0; 4],
            eff_grain_len_samples: [2048.0; 4],
            inv_grain_len_samples: [1.0 / 2048.0; 4],
            grain_warp_rate: [0.0; 4],
            eff_grain_warp_rate: [0.0; 4],
            eff_grain_window_mix: [0.0; 4],
            eff_g_lin: [0.0; 4],
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
    /// Pre-computed maximum `g_lin` for 1 group (`Lite`) and 2 groups (`Full`/`Ultra`).
    cached_max_loop_gain: [f32; 2],

    // ---- WT-UFDN Windowed Time-Warp & Unitary Spectral Freeze State ----
    /// Base complementary Hermite-Hann grain window mix $\kappa_{\text{win},0} \in [0.0, 1.0]$.
    pub grain_window_mix_base: f32,
    /// Effective $3\text{ kHz}$ modulated grain window mix $\kappa_{\text{win}}[k] \in [0.0, 1.0]$.
    pub eff_grain_window_mix_scalar: f32,
    /// Base in-loop grain feedback mix $\kappa_{\text{fb},0} \in [0.0, 1.0]$ (`0.0` = clean tank / scrubbable freeze, `> 0.0` = recursive Shimmer).
    pub grain_feedback_mix_base: f32,
    /// Effective $3\text{ kHz}$ modulated in-loop grain feedback mix $\kappa_{\text{fb}}[k] \in [0.0, 1.0]$.
    pub eff_grain_feedback_mix: f32,
    /// Native training grain spray depth in samples $\sigma_{\text{spray},0}$.
    pub grain_spray_samples_train: f32,
    /// Host ZOH-rescaled base grain spray depth in samples.
    pub grain_spray_samples_host: f32,
    /// Effective $3\text{ kHz}$ modulated grain spray depth in samples $\sigma_{\text{spray}}[k] \ge 0.0$.
    pub eff_grain_spray_samples: f32,
    /// Pre-Head Kronecker tap injection gain $\gamma_{\text{tap}} \in [-2.0, 2.0]$ (`0.1` default when active).
    pub head_tap_inject_gain: f32,
    /// Base spectral freeze gate $\alpha_{\text{freeze},0} \in [0.0, 1.0]$.
    pub freeze_alpha_base: f32,
    /// Explicit host / `FreezeLatch` override for spectral freeze (`Some(alpha)` when active).
    pub freeze_user_override: Option<f32>,
    /// Target spectral freeze gate $\alpha_{\text{freeze,target}}[k] \in [0.0, 1.0]$.
    pub freeze_target: f32,
    /// Smoothly slewed real-time spectral freeze gate $\bar{\alpha}_{\text{freeze}}[n] \in [0.0, 1.0]$.
    pub freeze_alpha_smoothed: f32,
    /// Causal attack onset threshold for automatic grain phase re-sync & auto-freeze ducking (`0.0` = disabled).
    pub onset_sync_threshold: f32,
    /// Normalized grain phasor $\phi_m[n] \in [0.0, 1.0)$ per line (`[2 groups][4 lines]`).
    pub grain_phasor: [[f32; 4]; 2],
    /// Current random spray offset in samples for Tap A (`[2 groups][4 lines]`).
    pub grain_spray_offset_a: [[f32; 4]; 2],
    /// Current random spray offset in samples for Tap B (`[2 groups][4 lines]`).
    pub grain_spray_offset_b: [[f32; 4]; 2],
    /// Deterministic LCG PRNG state (`0x9E37_79B9`) for click-free grain spray at $\phi = 0$.
    pub prng_state: u32,
    /// Latest 12-channel (`[Lite 4ch, Full 4ch, Ultra 4ch]`) tank/grain output fed into `Core1Stack::head_tap_hist`.
    pub latest_ap_out: [[f32; 4]; 3],

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
            cached_max_loop_gain: [0.0; 2],
            grain_window_mix_base: 0.0,
            eff_grain_window_mix_scalar: 0.0,
            grain_feedback_mix_base: 0.0,
            eff_grain_feedback_mix: 0.0,
            grain_spray_samples_train: 0.0,
            grain_spray_samples_host: 0.0,
            eff_grain_spray_samples: 0.0,
            head_tap_inject_gain: 0.1,
            freeze_alpha_base: 0.0,
            freeze_user_override: None,
            freeze_target: 0.0,
            freeze_alpha_smoothed: 0.0,
            onset_sync_threshold: 0.0,
            grain_phasor: [[0.0; 4]; 2],
            grain_spray_offset_a: [[0.0; 4]; 2],
            grain_spray_offset_b: [[0.0; 4]; 2],
            prng_state: 0x9E37_79B9,
            latest_ap_out: [[0.0; 4]; 3],
            delay_scratch: DelayScratchStorage::empty(),
            _backend: PhantomData,
        }
    }

    /// Creates a [`ModulatedUfdn`] pre-allocated with `max_ring_log2` capacity (`0` or `8..=18`)
    /// so subsequent `reload_from_arena` calls never allocate heap memory.
    pub fn with_max_capacity(max_ring_log2: u8) -> Self {
        let mut s = Self::new_inactive();
        if max_ring_log2 >= 8 {
            let clamped_log2 = max_ring_log2.min(18);
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
        s.max_capacity_log2 = ring_log2.min(18);
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

        let target_log2 = ring_log2.clamp(8, 18);
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
            let grain_len_tr = read_f32x4_le(arena, base + 208);
            let grain_warp = read_f32x4_le(arena, base + 224);
            let grain_meta = read_f32x4_le(arena, base + 240);

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
                gw.eff_g_lin[lane] = gw.g_lin[lane];
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

                let raw_glen = if grain_len_tr[lane] >= 16.0 {
                    grain_len_tr[lane]
                } else {
                    2048.0
                };
                let (glen_h, _) = zoh.rescale_core2_delays(raw_glen, 19.0);
                let max_glen = (self.ring_mask.saturating_sub(16).max(16)) as f32;
                let glen_clamped = glen_h.clamp(16.0, max_glen);
                gw.grain_len_samples[lane] = glen_clamped;
                gw.eff_grain_len_samples[lane] = glen_clamped;
                gw.inv_grain_len_samples[lane] = 1.0 / glen_clamped;

                let warp_clamped = grain_warp[lane].clamp(-2.0, 2.0);
                gw.grain_warp_rate[lane] = warp_clamped;
                gw.eff_grain_warp_rate[lane] = warp_clamped;
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

                self.grain_window_mix_base = grain_meta[0].clamp(0.0, 1.0);
                self.eff_grain_window_mix_scalar = self.grain_window_mix_base;
                self.grain_spray_samples_train = grain_meta[1].max(0.0);
                self.grain_spray_samples_host = if self.grain_spray_samples_train > 1e-4 {
                    let (sp_h, _) =
                        zoh.rescale_core2_delays(self.grain_spray_samples_train.max(4.0), 19.0);
                    sp_h
                } else {
                    0.0
                };
                self.eff_grain_spray_samples = self.grain_spray_samples_host;
                self.grain_feedback_mix_base = grain_meta[2].clamp(0.0, 1.0);
                self.eff_grain_feedback_mix = self.grain_feedback_mix_base;
                self.head_tap_inject_gain = if grain_meta[3].abs() > 1e-5 {
                    grain_meta[3].clamp(-2.0, 2.0)
                } else {
                    0.1
                };
            } else {
                self.freeze_alpha_base = grain_meta[0].clamp(0.0, 1.0);
                self.onset_sync_threshold = grain_meta[2].max(0.0);
            }
        }

        self.freeze_target = self
            .freeze_user_override
            .unwrap_or(self.freeze_alpha_base)
            .clamp(0.0, 1.0);
        self.recompute_self_activating_windows();

        self.r_dc_host = zoh.r_dc_host;
        self.tau_ref_host = zoh.tau_ref_samples_host;
        self.xfade_samples_host = zoh.switch_xfade_samples_host.max(16);
        self.inv_xfade_samples_host = 1.0 / (self.xfade_samples_host as f32);

        let max_g0 = self.groups[0].g_lin.iter().copied().fold(0.0_f32, f32::max);
        let max_g1 = self.groups[1].g_lin.iter().copied().fold(max_g0, f32::max);
        self.cached_max_loop_gain = [max_g0, max_g1];

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
        self.freeze_alpha_smoothed = self.freeze_target;
        self.grain_phasor = [[0.0, 0.25, 0.50, 0.75], [0.125, 0.375, 0.625, 0.875]];
        self.grain_spray_offset_a = [[0.0; 4]; 2];
        self.grain_spray_offset_b = [[0.0; 4]; 2];
        self.prng_state = 0x9E37_79B9;
        self.latest_ap_out = [[0.0; 4]; 3];

        for g in 0..2 {
            self.tau0_smoothed[g] = self.groups[g].tau0_target_samples;
            self.xfade_old_tau[g] = self.groups[g].tau0_target_samples;
        }
        for blk in self.delay_scratch.as_slice_mut() {
            blk.0.fill(0);
        }
    }

    /// Recomputes each line's self-activating complementary Hermite-Hann window depth
    /// $\kappa_{\text{win},m}[k] = \operatorname{clamp}(\max(\kappa_{\text{win,base}}, 64|\nu_m|, 0.25\sigma_{\text{spray}}), 0, 1)$.
    #[inline]
    pub fn recompute_self_activating_windows(&mut self) {
        let base_w = self.eff_grain_window_mix_scalar;
        let spray_w = (0.25 * self.eff_grain_spray_samples).clamp(0.0, 1.0);
        let floor_w = base_w.max(spray_w);
        for g in 0..2 {
            for lane in 0..4 {
                let warp_w =
                    (64.0 * self.groups[g].eff_grain_warp_rate[lane].abs()).clamp(0.0, 1.0);
                self.groups[g].eff_grain_window_mix[lane] = floor_w.max(warp_w);
            }
        }
    }

    /// Engages or releases Unitary Spectral Freeze (`true` $\to \alpha_{\text{freeze}} = 1.0$, `false` $\to 0.0$).
    #[inline]
    pub fn set_freeze(&mut self, freeze_active: bool) {
        self.set_freeze_amount(if freeze_active { 1.0 } else { 0.0 });
    }

    /// Sets the continuous Unitary Spectral Freeze amount $\alpha_{\text{freeze}} \in [0.0, 1.0]$.
    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        let clamped = alpha.clamp(0.0, 1.0);
        self.freeze_user_override = Some(clamped);
        self.freeze_target = clamped;
    }

    /// Returns `true` if Unitary Spectral Freeze is currently active ($\alpha_{\text{freeze,target}} \ge 0.5$).
    #[inline]
    pub const fn is_frozen(&self) -> bool {
        self.freeze_target >= 0.5
    }

    /// Configures grain window length (ms), pitch-shift interval (semitones, $\nu = 1 - 2^{\Delta p / 12}$),
    /// and random spray depth (`0.0..1.0` normalized or samples if `> 1.0`) at `sample_rate_hz`.
    #[inline]
    pub fn set_grain_params(
        &mut self,
        size_ms: Option<f32>,
        pitch_semitones: Option<f32>,
        spray: Option<f32>,
        sample_rate_hz: f32,
    ) {
        let sr = sample_rate_hz.max(8_000.0);
        let max_glen = (self.ring_mask.saturating_sub(16).max(16)) as f32;
        if let Some(ms) = size_ms {
            let glen = (ms.max(1.0) * 0.001 * sr).clamp(16.0, max_glen);
            let inv_glen = 1.0 / glen;
            for g in 0..2 {
                for lane in 0..4 {
                    self.groups[g].grain_len_samples[lane] = glen;
                    self.groups[g].eff_grain_len_samples[lane] = glen;
                    self.groups[g].inv_grain_len_samples[lane] = inv_glen;
                }
            }
        }
        if let Some(st) = pitch_semitones {
            let ratio =
                crate::schema::cold_pow_f64(2.0, (st.clamp(-24.0, 24.0) as f64) / 12.0) as f32;
            let warp_c = (1.0 - ratio).clamp(-2.0, 2.0);
            for g in 0..2 {
                for lane in 0..4 {
                    self.groups[g].grain_warp_rate[lane] = warp_c;
                    self.groups[g].eff_grain_warp_rate[lane] = warp_c;
                }
            }
        }
        if let Some(sp) = spray {
            let spray_samp = if sp <= 1.0 {
                sp.max(0.0) * 0.025 * sr
            } else {
                sp
            };
            self.grain_spray_samples_host = spray_samp.max(0.0);
            self.eff_grain_spray_samples = self.grain_spray_samples_host;
        }
        self.recompute_self_activating_windows();
    }

    /// Re-synchronizes all 8 grain phasors to a staggered 8-phase constellation on note attack onset.
    #[inline]
    pub fn trigger_onset_phase_sync(&mut self) {
        self.grain_phasor = [[0.0, 0.25, 0.50, 0.75], [0.125, 0.375, 0.625, 0.875]];
    }

    /// Applies $3\text{ kHz}$ Control-Bus state-space modulation (`W_ctrl_core2`) to Core 2's
    /// feedback gains, time-warp rates, grain lengths, spray, feedback mix, and freeze gate.
    #[inline]
    pub fn apply_control_bus_3khz(
        &mut self,
        gain_mod: &[f32; 8],
        warp_mod: &[f32; 8],
        len_scale: &[f32; 4],
        state_mod: &[f32; 4],
        onset_env: f32,
    ) {
        let max_glen = (self.ring_mask.saturating_sub(16).max(16)) as f32;
        for g in 0..2 {
            let l_scale = (1.0 + len_scale[g]).clamp(0.25, 4.0);
            let gw = &mut self.groups[g];
            for lane in 0..4 {
                let idx = g * 4 + lane;
                gw.eff_g_lin[lane] = (gw.g_lin[lane] + gain_mod[idx]).clamp(0.0, 0.9990);
                gw.eff_grain_warp_rate[lane] =
                    (gw.grain_warp_rate[lane] + warp_mod[idx]).clamp(-2.0, 2.0);
                let glen = (gw.grain_len_samples[lane] * l_scale).clamp(16.0, max_glen);
                gw.eff_grain_len_samples[lane] = glen;
                gw.inv_grain_len_samples[lane] = 1.0 / glen;
            }
        }
        let max_g0 = self.groups[0]
            .eff_g_lin
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        let max_g1 = self.groups[1]
            .eff_g_lin
            .iter()
            .copied()
            .fold(max_g0, f32::max);
        self.cached_max_loop_gain = [max_g0, max_g1];

        let base_freeze = self.freeze_user_override.unwrap_or(self.freeze_alpha_base);
        self.freeze_target = (base_freeze + state_mod[0]).clamp(0.0, 1.0);
        self.eff_grain_window_mix_scalar =
            (self.grain_window_mix_base + state_mod[1]).clamp(0.0, 1.0);
        self.eff_grain_feedback_mix = (self.grain_feedback_mix_base + state_mod[2]).clamp(0.0, 1.0);
        self.eff_grain_spray_samples = (self.grain_spray_samples_host + state_mod[3]).max(0.0);

        if self.onset_sync_threshold > 0.0 && onset_env > self.onset_sync_threshold {
            self.trigger_onset_phase_sync();
        }
        self.recompute_self_activating_windows();
    }

    #[inline(always)]
    fn next_spray_offset_samples(&mut self) -> f32 {
        if self.eff_grain_spray_samples <= 1e-4 {
            return 0.0;
        }
        self.prng_state = self
            .prng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        // Map upper 24 bits to [0.0, 1.0) without division
        let unit = ((self.prng_state >> 8) as f32) * (1.0 / 16_777_216.0);
        unit * self.eff_grain_spray_samples
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

    /// Writes 4 `f32` samples for group `g` into the `fp16` ring buffer at `write_idx` via
    /// hardware `B::f32x4_to_fp16x4` (`FCVTN` on `aarch64`).
    #[inline(always)]
    fn write_fp16_group(&mut self, g: usize, write_idx: usize, vals: &[f32; 4]) {
        let blocks = self.delay_scratch.as_slice_mut();
        if blocks.is_empty() {
            return;
        }
        let blk_idx = (write_idx >> 2) & (blocks.len() - 1);
        let sub_frame = write_idx & 3;
        let base_word = (sub_frame << 3) | ((g & 1) << 2);
        let half_words = B::f32x4_to_fp16x4(vals);
        blocks[blk_idx].0[base_word..base_word + 4].copy_from_slice(&half_words);
    }

    /// Evaluates a 4-tap Cubic Lagrange fractional delay read (`CL-06`) across 4 FDN lines in group `g`
    /// using hardware 4-lane `B::fp16x4_to_f32x4` (`FCVTL` on `aarch64`).
    #[inline(always)]
    fn read_cubic_lagrange_group(&self, g: usize, tau_samples: &[f32; 4]) -> [f32; 4] {
        let blocks = self.delay_scratch.as_slice();
        if blocks.is_empty() {
            return [0.0; 4];
        }
        let blk_mask = blocks.len() - 1;
        let g_bits = (g & 1) << 2;

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

        let mut bits_m1 = [0_u16; 4];
        let mut bits_0 = [0_u16; 4];
        let mut bits_p1 = [0_u16; 4];
        let mut bits_p2 = [0_u16; 4];

        let w_pos = self.write_pos;
        let mask = self.ring_mask;
        for lane in 0..4 {
            let di = d_int[lane];
            let lane_bits = g_bits | lane;

            let r_m1 = w_pos.wrapping_sub(di - 1) & mask;
            bits_m1[lane] = blocks[(r_m1 >> 2) & blk_mask].0[((r_m1 & 3) << 3) | lane_bits];

            let r_0 = w_pos.wrapping_sub(di) & mask;
            bits_0[lane] = blocks[(r_0 >> 2) & blk_mask].0[((r_0 & 3) << 3) | lane_bits];

            let r_p1 = w_pos.wrapping_sub(di + 1) & mask;
            bits_p1[lane] = blocks[(r_p1 >> 2) & blk_mask].0[((r_p1 & 3) << 3) | lane_bits];

            let r_p2 = w_pos.wrapping_sub(di + 2) & mask;
            bits_p2[lane] = blocks[(r_p2 >> 2) & blk_mask].0[((r_p2 & 3) << 3) | lane_bits];
        }

        let s_m1 = B::fp16x4_to_f32x4(&bits_m1);
        let s_0 = B::fp16x4_to_f32x4(&bits_0);
        let s_p1 = B::fp16x4_to_f32x4(&bits_p1);
        let s_p2 = B::fp16x4_to_f32x4(&bits_p2);

        let mut acc = w_m1.mul(B::Vec4::load(&s_m1));
        acc = acc.fma(w_0, B::Vec4::load(&s_0));
        acc = acc.fma(w_p1, B::Vec4::load(&s_p1));
        acc = acc.fma(w_p2, B::Vec4::load(&s_p2));
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
        let tzf_w = self.tzf_write_pos & CORE2_TZF_RING_MASK;
        self.tzf_dry_ring[tzf_w] = [dry_left, dry_right];
        let tzf_delay = (self.tau_ref_host as usize).clamp(1, CORE2_TZF_RING_LEN - 1);
        let tzf_r = tzf_w.wrapping_sub(tzf_delay) & CORE2_TZF_RING_MASK;
        let tzf_dry = self.tzf_dry_ring[tzf_r];
        self.tzf_write_pos = (tzf_w + 1) & CORE2_TZF_RING_MASK;

        if !self.active || active_fdn_groups == 0 {
            self.latest_ap_out = [[0.0; 4]; 3];
            return [0.0, 0.0, dry_left, dry_right];
        }

        // Smoothly slew Unitary Spectral Freeze gate alpha_freeze[n] in [0.0, 1.0]
        self.freeze_alpha_smoothed = self.delay_slew_pole * self.freeze_alpha_smoothed
            + (1.0 - self.delay_slew_pole) * self.freeze_target;
        let alpha_freeze = self.freeze_alpha_smoothed;
        let fb_mix = self.eff_grain_feedback_mix * (1.0 - alpha_freeze);

        let groups = active_fdn_groups.clamp(1, 2);
        let slew_v = B::Vec4::splat(self.delay_slew_pole);
        let one_minus_slew_v = B::Vec4::splat(1.0 - self.delay_slew_pole);
        let four_v = B::Vec4::splat(4.0);
        let ap_w = self.allpass_write_pos & CORE2_ALLPASS_RING_MASK;

        let mut u_ap_out = [[0.0_f32; 4]; 2];
        let mut d_early_all = [[0.0_f32; 4]; 2];
        let mut wet_l = 0.0_f32;
        let mut wet_r = 0.0_f32;
        let (mod_chunks, _) = delay_mod_samples.as_chunks::<4>();

        // 1. Fractional delay read + 128-sample switch crossfade (CL-06, CL-07) + Dual-Tap Hermite-Hann Grain Reader + Schroeder all-pass (CL-08)
        for g in 0..groups {
            let tgt_v = B::Vec4::load(&self.groups[g].tau0_target_samples);
            let sm_prev_v = B::Vec4::load(&self.tau0_smoothed[g]);
            let sm_v = sm_prev_v.mul(slew_v).fma(one_minus_slew_v, tgt_v);
            sm_v.store(&mut self.tau0_smoothed[g]);
            let mut tau_eff = [0.0_f32; 4];
            sm_v.add(B::Vec4::load(&mod_chunks[g]))
                .max(four_v)
                .store(&mut tau_eff);

            let mut d_base = self.read_cubic_lagrange_group(g, &tau_eff);

            // Single bitwise OR check for active 128-sample raised-cosine switch crossfade (CL-07)
            let xr = &self.xfade_remaining[g];
            if (xr[0] | xr[1] | xr[2] | xr[3]) != 0 {
                let d_old = self.read_cubic_lagrange_group(g, &self.xfade_old_tau[g]);
                for lane in 0..4 {
                    let rem = self.xfade_remaining[g][lane];
                    if rem > 0 {
                        let prog = 1.0_f32 - (rem as f32) * self.inv_xfade_samples_host;
                        let p_clamped = prog.clamp(0.0, 1.0);
                        let w_new = p_clamped * p_clamped * (3.0 - 2.0 * p_clamped);
                        d_base[lane] = (1.0 - w_new) * d_old[lane] + w_new * d_base[lane];
                        self.xfade_remaining[g][lane] = rem - 1;
                    }
                }
            }

            // Self-Activating Dual-Tap Complementary Hermite-Hann Grain Reader (0 FDIV, 0 FSQRT)
            let k_win = self.groups[g].eff_grain_window_mix;
            let grain_active =
                k_win[0] > 1e-5 || k_win[1] > 1e-5 || k_win[2] > 1e-5 || k_win[3] > 1e-5;
            let d_grain = if grain_active {
                let mut tau_a = [0.0_f32; 4];
                let mut tau_b = [0.0_f32; 4];
                let mut w_a_arr = [1.0_f32; 4];
                let mut w_b_arr = [0.0_f32; 4];
                for lane in 0..4 {
                    let kw = k_win[lane];
                    let nu = self.groups[g].eff_grain_warp_rate[lane];
                    let inv_len = self.groups[g].inv_grain_len_samples[lane];
                    let g_len = self.groups[g].eff_grain_len_samples[lane];
                    let step = (nu.abs() * inv_len).max(inv_len * 0.25).min(0.25);

                    let prev_phi = self.grain_phasor[g][lane];
                    let mut phi_a = prev_phi + step;
                    if phi_a >= 1.0 {
                        phi_a -= 1.0;
                        // Tap A is at phi_a ~ 0.0 where Hermite-Hann weight H(0) == 0.0 -> click-free spray jump
                        self.grain_spray_offset_a[g][lane] = self.next_spray_offset_samples();
                    }
                    if prev_phi < 0.5 && phi_a >= 0.5 {
                        // Tap B is at phi_b ~ 0.0 where Hermite-Hann weight H(0) == 0.0 -> click-free spray jump
                        self.grain_spray_offset_b[g][lane] = self.next_spray_offset_samples();
                    }
                    self.grain_phasor[g][lane] = phi_a;
                    let phi_b = if phi_a >= 0.5 {
                        phi_a - 0.5
                    } else {
                        phi_a + 0.5
                    };

                    let (warp_a, warp_b) = if nu < -1e-5 {
                        ((1.0 - phi_a) * g_len, (1.0 - phi_b) * g_len)
                    } else if nu > 1e-5 {
                        (phi_a * g_len, phi_b * g_len)
                    } else {
                        (0.0, 0.0)
                    };
                    tau_a[lane] = tau_eff[lane] + self.grain_spray_offset_a[g][lane] + warp_a;
                    tau_b[lane] = tau_eff[lane] + self.grain_spray_offset_b[g][lane] + warp_b;

                    let u = 1.0 - (2.0 * phi_a - 1.0).abs();
                    let h = u * u * (3.0 - 2.0 * u);
                    let wa = 1.0 - kw * (1.0 - h);
                    w_a_arr[lane] = wa;
                    w_b_arr[lane] = 1.0 - wa;
                }
                let tap_a = self.read_cubic_lagrange_group(g, &tau_a);
                let tap_b = self.read_cubic_lagrange_group(g, &tau_b);
                let v_wa = B::Vec4::load(&w_a_arr);
                let v_wb = B::Vec4::load(&w_b_arr);
                v_wa.mul(B::Vec4::load(&tap_a))
                    .fma(v_wb, B::Vec4::load(&tap_b))
                    .to_array()
            } else {
                d_base
            };

            // Early reflection tap + FDN stereo output projection + In-loop Schroeder all-pass diffuser (CL-08)
            let d_early = self.read_cubic_lagrange_group(g, &self.groups[g].early_tap_samples);
            d_early_all[g] = d_early;
            let gw = &self.groups[g];
            let vd_base = B::Vec4::load(&d_base);
            let vd_grain = B::Vec4::load(&d_grain);
            let vd_early = B::Vec4::load(&d_early);
            let vearly = B::Vec4::load(&gw.early_gain).mul(vd_early);
            let vwet_l = vearly.fma(B::Vec4::load(&gw.c_fdn_left), vd_grain);
            let vwet_r = vearly.fma(B::Vec4::load(&gw.c_fdn_right), vd_grain);
            let wl_arr = vwet_l.to_array();
            let wr_arr = vwet_r.to_array();
            wet_l += wl_arr[0] + wl_arr[1] + wl_arr[2] + wl_arr[3];
            wet_r += wr_arr[0] + wr_arr[1] + wr_arr[2] + wr_arr[3];

            // Blend in-loop feedback tap between un-warped d_base and warped d_grain via grain_feedback_mix
            let vd_loop = if grain_active && fb_mix > 1e-5 {
                vd_base.fma(B::Vec4::splat(fb_mix), vd_grain.sub(vd_base))
            } else {
                vd_base
            };

            let mut u_del = [0.0_f32; 4];
            for lane in 0..4 {
                let d_ap = gw.d_ap_samples[lane] & CORE2_ALLPASS_RING_MASK;
                let ap_r = ap_w.wrapping_sub(d_ap) & CORE2_ALLPASS_RING_MASK;
                u_del[lane] = self.allpass_ring[g][lane][ap_r];
            }
            let va = B::Vec4::load(&gw.a_diff);
            let vu_del = B::Vec4::load(&u_del);
            // v_m[n] = u_ap[n - D_ap] - a_diff * d_loop[n]
            let vv_m = vu_del.fms(va, vd_loop);
            // u_ap[n] = d_loop[n] + a_diff * v_m[n]
            let vu_ap = vd_loop.fma(va, vv_m);
            let u_ap_arr = vu_ap.to_array();
            for lane in 0..4 {
                self.allpass_ring[g][lane][ap_w] = u_ap_arr[lane];
            }
            vv_m.store(&mut u_ap_out[g]);
            let v_tap_out = if grain_active && fb_mix < 0.9999 {
                vv_m.fma(B::Vec4::splat(1.0 - fb_mix), vd_grain.sub(vd_base))
            } else {
                vv_m
            };
            v_tap_out.store(&mut self.latest_ap_out[g]);
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

        // Populate remaining 12-channel 3-tier taps (`[Lite 4ch, Full 4ch, Ultra 4ch]`) for Pre-Head Kronecker injection
        if groups < 2 {
            self.latest_ap_out[1] = self.latest_ap_out[0];
        }
        let g1_idx = if groups >= 2 { 1 } else { 0 };
        self.latest_ap_out[2] = [
            0.5 * (u_mixed[0][0] + d_early_all[0][0]),
            0.5 * (u_mixed[0][1] + d_early_all[0][1]),
            0.5 * (u_mixed[g1_idx][2] + d_early_all[g1_idx][2]),
            0.5 * (u_mixed[g1_idx][3] + d_early_all[g1_idx][3]),
        ];

        // 3. Exact Gain-Normalized 5 Hz In-Loop DC Blocker (CL-09) + Coercive Lyapunov Loop (CL-10, CL-11)
        let w_pos = self.write_pos;
        let in_gate = B::Vec4::splat(1.0 - alpha_freeze);
        let mut peak_g = self.cached_max_loop_gain[groups - 1];
        for g in 0..groups {
            let gw = &self.groups[g];
            // Inject Core 1 tap into mixed FDN state (scaled by 1 - alpha_freeze)
            let tap_vec = B::Vec4::from_array(core1_tap[g]).mul(in_gate);
            let b_in_vec = B::Vec4::from_array(gw.b_fdn_in);
            let u_in = B::Vec4::from_array(u_mixed[g])
                .fma(tap_vec, b_in_vec)
                .to_array();

            let mut g_loop = gw.eff_g_lin;
            if alpha_freeze > 1e-5 {
                for lane in 0..4 {
                    g_loop[lane] =
                        gw.eff_g_lin[lane] + alpha_freeze * (1.0035 - gw.eff_g_lin[lane]);
                    if g_loop[lane] > peak_g {
                        peak_g = g_loop[lane].min(1.0);
                    }
                }
            }

            let mut s_next = [0.0_f32; 4];
            B::mufdn_coercive_loop_4ch(
                &u_in,
                &g_loop,
                &gw.inv_sigma_knee,
                &gw.two_over_3v_ceil,
                &gw.v_ceil,
                self.r_dc_host,
                &mut self.dc_x_prev[g],
                &mut self.dc_y_prev[g],
                &mut s_next,
            );
            self.write_fp16_group(g, w_pos, &s_next);
        }
        if peak_g > self.max_effective_loop_gain {
            self.max_effective_loop_gain = peak_g;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;

    #[test]
    fn test_modulated_ufdn_capacities_and_crossfade() {
        let mut ufdn_inact = ModulatedUfdn::<ScalarBackend>::new_inactive();
        assert!(!ufdn_inact.active);
        assert_eq!(ufdn_inact.delay_scratch.byte_len(), 0);
        assert_eq!(
            ufdn_inact.process_sample(0.25, -0.25, &[[0.0; 4]; 2], &[0.0; 8], 2),
            [0.0, 0.0, 0.25, -0.25]
        );

        let mut ufdn = ModulatedUfdn::<ScalarBackend>::with_max_capacity(8);
        assert_eq!(ufdn.max_capacity_log2, 8);
        assert_eq!(ufdn.delay_scratch.byte_len(), (256 * 8) * 2);

        let zoh = ZohRescaledParams::new(48_000, 48_000);
        let arena = alloc::vec![0u8; 22_000];
        ufdn.load_from_arena(&arena, true, 8, &zoh);
        assert!(ufdn.active);

        // Trigger switch crossfade (> 64 sample jump) and out-of-bounds line_idx guard
        ufdn.set_target_delay_samples(99, 120.0);
        ufdn.set_target_delay_samples(0, 180.0);
        assert_eq!(ufdn.xfade_remaining[0][0], 128);

        for _ in 0..130 {
            let out = ufdn.process_sample(0.1, -0.1, &[[0.2; 4]; 2], &[0.5; 8], 2);
            assert!(out[0].is_finite() && out[1].is_finite());
        }
        assert_eq!(ufdn.xfade_remaining[0][0], 0);
    }

    #[test]
    fn test_wt_ufdn_self_activating_granular_shimmer_and_unitary_freeze() {
        let mut ufdn = ModulatedUfdn::<ScalarBackend>::with_max_capacity(11);
        let zoh = ZohRescaledParams::new(48_000, 48_000);
        let mut arena = alloc::vec![0u8; 22_000];

        // Configure Group 0 (6784..7040) and Group 1 (20544..20800):
        // Group 0: un-warped reverb lines (tau = [64, 96, 128, 160], g_lin = 0.96, warp_rate = 0.0)
        // Group 1: shimmer octave-up lines (tau = [192, 224, 256, 288], g_lin = 0.96, warp_rate = -1.0)
        let write_f32x4 = |buf: &mut [u8], off: usize, vals: [f32; 4]| {
            for (i, v) in vals.iter().enumerate() {
                buf[off + i * 4..off + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        };
        for (g, &base) in [6784usize, 20_544usize].iter().enumerate() {
            let taus = if g == 0 {
                [64.0, 96.0, 128.0, 160.0]
            } else {
                [192.0, 224.0, 256.0, 288.0]
            };
            write_f32x4(&mut arena, base, taus);
            write_f32x4(&mut arena, base + 16, [0.96; 4]); // g_lin
            write_f32x4(&mut arena, base + 32, [1.0; 4]); // inv_sigma
            write_f32x4(&mut arena, base + 48, [0.0; 4]); // two_over_3v (auto-derived from v_ceil)
            write_f32x4(&mut arena, base + 64, [0.95; 4]); // v_ceil
            write_f32x4(&mut arena, base + 80, [0.5; 4]); // a_diff
            write_f32x4(&mut arena, base + 96, [8.0, 12.0, 16.0, 20.0]); // d_ap_tr
            write_f32x4(&mut arena, base + 112, [0.5; 4]); // b_in
            write_f32x4(&mut arena, base + 128, [0.18; 4]); // c_l
            write_f32x4(&mut arena, base + 144, [0.18; 4]); // c_r
            write_f32x4(&mut arena, base + 160, [16.0, 24.0, 32.0, 40.0]); // early_tau_tr
            write_f32x4(&mut arena, base + 176, [0.05; 4]); // early_gain
            write_f32x4(&mut arena, base + 192, [0.25, 0.995, 64.0, 128.0]); // [psi, r_dc, tau_ref, xfade]
            write_f32x4(&mut arena, base + 208, [512.0; 4]); // grain_len_tr
            let warps = if g == 0 { [0.0; 4] } else { [-1.0; 4] };
            write_f32x4(&mut arena, base + 224, warps); // grain_warp_rate
            // grain_meta: [window_mix=0.0, spray=0.0, feedback_mix=0.5, head_tap_inject_gain=0.8] in G0
            let meta = if g == 0 {
                [0.0, 0.0, 0.5, 0.8]
            } else {
                [0.0, 0.25, 0.0, 0.0]
            };
            write_f32x4(&mut arena, base + 240, meta);
        }

        ufdn.load_from_arena(&arena, true, 11, &zoh);

        // Verify self-activating windows: Group 0 has warp=0 -> window_mix=0; Group 1 has warp=-1 -> window_mix=1
        assert_eq!(ufdn.groups[0].eff_grain_window_mix, [0.0; 4]);
        assert_eq!(ufdn.groups[1].eff_grain_window_mix, [1.0; 4]);
        assert!((ufdn.head_tap_inject_gain - 0.8).abs() < 1e-6);

        // 1. Feed a 1,024-sample chord burst
        let core1_tap = [[0.15_f32; 4]; 2];
        for n in 0..1024 {
            let t0 = ((n % 64) as f32) * (1.0 / 32.0) - 1.0;
            let t1 = ((n % 43) as f32) * (1.0 / 21.5) - 1.0;
            let sig = 0.4 * t0 + 0.25 * t1;
            let out = ufdn.process_sample(sig, -sig, &core1_tap, &[0.0; 8], 2);
            assert!(out[0].is_finite() && out[1].is_finite());
        }

        // Verify 12-channel 3-tier latest_ap_out is populated
        let ap_energy: f32 = ufdn.latest_ap_out.iter().flatten().map(|x| x * x).sum();
        assert!(ap_energy > 1e-5);

        // 2. Engage Unitary Spectral Freeze and run 100,000 samples with ZERO input
        ufdn.set_freeze(true);
        assert!(ufdn.is_frozen());
        let mut late_energy = 0.0_f64;
        for i in 0..100_000 {
            // Halfway through freeze, scrub grain size, pitch, and spray in real time
            if i == 50_000 {
                ufdn.set_grain_params(Some(45.0), Some(7.0), Some(0.35), 48_000.0);
            }
            let out = ufdn.process_sample(0.0, 0.0, &[[0.0; 4]; 2], &[0.0; 8], 2);
            assert!(out[0].is_finite() && out[1].is_finite());
            assert!(out[0].abs() <= 2.0 && out[1].abs() <= 2.0);
            if i >= 90_000 {
                late_energy +=
                    (out[0] as f64) * (out[0] as f64) + (out[1] as f64) * (out[1] as f64);
            }
        }
        let late_rms = crate::schema::cold_sqrt_f64(late_energy / 20_000.0);
        assert!(
            late_rms > 0.005 && late_rms < 1.0,
            "Frozen tank must sustain indefinitely on coercive limit cycle without decaying or exploding (got late_rms={late_rms})"
        );

        // 3. Test 3 kHz control bus modulation and onset phase sync
        ufdn.apply_control_bus_3khz(
            &[0.02; 8],
            &[0.1; 8],
            &[0.05; 4],
            &[-1.0, 0.2, 0.1, 4.0],
            0.5,
        );
        assert!(!ufdn.is_frozen());
    }
}
