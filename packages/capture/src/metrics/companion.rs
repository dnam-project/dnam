//! Companion Perceptual Metrics (`MR-IIR`, `LTAS` 33-Band 1/3-Octave Tilt, and `SegESR`).
//!
//! Canonical Reference: `docs/specs/architecture/psychoacoustic_quality_and_tiers.md` §3 (`r[qual.companion.*]`).

#![forbid(unsafe_code)]

use std::f64::consts::PI;

/// `+1.0 dB` linear power factor $10^{0.1} \approx 1.2589254117941673$ for noise-limited guards.
pub const ONE_DB_POWER_FACTOR: f64 = 1.258_925_411_794_167_3;

/// `+6.0 dB` linear power factor $10^{0.6} \approx 3.9810717055349722$ for active frame gating.
pub const SIX_DB_POWER_FACTOR: f64 = 3.981_071_705_534_972_2;
const MAX_METRIC_SAMPLE_RATE_HZ: f64 = 1_536_000.0;

// ============================================================================
// 1. 5-Scale Multi-Resolution Log-IIR (`MR-IIR`, r[qual.companion.mr-iir])
// ============================================================================

/// Computes the 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank Loss (`MR-IIR`,
/// $M \in \{256, 512, 1024, 2048, 4096\}$, pole radius $r_M = \exp(-\pi / M)$, hop $H = M / 4$,
/// $K = 32$ log-spaced probes, `ref[qual.companion.mr-iir]`, `ref[qual.besr.finite-timeline]`).
///
/// Returns `f64::INFINITY` on empty, length-mismatched, non-finite, or invalid-rate inputs so
/// invalid evaluations never pass tier gates.
#[must_use]
pub fn compute_mr_iir_loss(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> f64 {
    const SCALES: [usize; 5] = [256, 512, 1024, 2048, 4096];
    const NUM_PROBES: usize = 32;

    if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || warmup_samples >= target.len()
    {
        return f64::INFINITY;
    }
    let n_total = target.len();
    if !target.iter().all(|v| v.is_finite()) || !model.iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    if let Some(ns) = noise
        && (ns.len() < n_total || !ns[..n_total].iter().all(|v| v.is_finite()))
    {
        return f64::INFINITY;
    }

    let f_max = (0.45 * sample_rate_hz).min(20_000.0);
    if f_max <= 20.0 {
        return f64::INFINITY;
    }
    let t_sel = n_total - warmup_samples;
    const SUB_CHUNK: usize = 64;
    let num_full_chunks = t_sel / SUB_CHUNK;
    let mut chunk_t = vec![0.0_f64; num_full_chunks];
    let mut chunk_m = vec![0.0_f64; num_full_chunks];
    let mut chunk_n = vec![0.0_f64; num_full_chunks];
    let mut nz_t = vec![false; num_full_chunks];
    let mut nz_m = vec![false; num_full_chunks];
    let mut nz_n = vec![false; num_full_chunks];
    let mut tail_t = [0.0_f64; 4096];
    let mut tail_m = [0.0_f64; 4096];
    let mut tail_n = [0.0_f64; 4096];

    let mut total_loss = 0.0_f64;

    for &m_scale in &SCALES {
        let hop = m_scale / 4;
        let r_m = (-PI / (m_scale as f64)).exp();
        let r_m_sq = r_m * r_m;
        let norm = 0.5 * (1.0 - r_m_sq);
        let tail_start = t_sel.saturating_sub(m_scale);

        let mut sc_num_sq = 0.0_f64;
        let mut sc_den_sq = 0.0_f64;
        let mut mag_diff_sum = 0.0_f64;
        let mut cell_count = 0usize;
        let mut scale_nz = false;
        let mut scale_nz_n = false;
        let mut scale_nse_sq = 0.0_f64;

        for p in 0..NUM_PROBES {
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            let omega = 2.0 * PI * f_k / sample_rate_hz;
            let two_r_cos = 2.0 * r_m * omega.cos();

            // Step zero-initialized unity-peak-gain bandpass resonators across 0..n_total
            let (mut x1_t, mut x2_t, mut y1_t, mut y2_t) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_m, mut x2_m, mut y1_m, mut y2_m) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_n, mut x2_n, mut y1_n, mut y2_n) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

            chunk_t.fill(0.0);
            chunk_m.fill(0.0);
            chunk_n.fill(0.0);
            nz_t.fill(false);
            nz_m.fill(false);
            nz_n.fill(false);

            for i in 0..n_total {
                let xt = target[i];
                let drive_t = xt - x2_t;
                let mut bt = norm * drive_t + two_r_cos * y1_t - r_m_sq * y2_t;
                if drive_t == 0.0
                    && (xt == 0.0 || xt.abs() >= 1.0e-140)
                    && bt.abs() < 1.0e-140
                    && y1_t.abs() < 1.0e-140
                {
                    bt = 0.0;
                }
                x2_t = x1_t;
                x1_t = xt;
                y2_t = y1_t;
                y1_t = bt;

                let xm = model[i];
                let drive_m = xm - x2_m;
                let mut bm = norm * drive_m + two_r_cos * y1_m - r_m_sq * y2_m;
                if drive_m == 0.0
                    && (xm == 0.0 || xm.abs() >= 1.0e-140)
                    && bm.abs() < 1.0e-140
                    && y1_m.abs() < 1.0e-140
                {
                    bm = 0.0;
                }
                x2_m = x1_m;
                x1_m = xm;
                y2_m = y1_m;
                y1_m = bm;

                let xn = noise.map_or(0.0, |ns| ns[i]);
                let drive_n = xn - x2_n;
                let mut bn = norm * drive_n + two_r_cos * y1_n - r_m_sq * y2_n;
                if drive_n == 0.0
                    && (xn == 0.0 || xn.abs() >= 1.0e-140)
                    && bn.abs() < 1.0e-140
                    && y1_n.abs() < 1.0e-140
                {
                    bn = 0.0;
                }
                x2_n = x1_n;
                x1_n = xn;
                y2_n = y1_n;
                y1_n = bn;

                if i >= warmup_samples {
                    let sub_t = xt != 0.0 && xt.abs() < 1.0e-140;
                    let sub_m = xm != 0.0 && xm.abs() < 1.0e-140;
                    let sub_n = xn != 0.0 && xn.abs() < 1.0e-140;
                    let u = i - warmup_samples;
                    let c = u / SUB_CHUNK;
                    if c < num_full_chunks {
                        chunk_t[c] += bt * bt;
                        chunk_m[c] += bm * bm;
                        chunk_n[c] += bn * bn;
                        nz_t[c] |= bt != 0.0 || sub_t;
                        nz_m[c] |= bm != 0.0 || sub_m;
                        nz_n[c] |= bn != 0.0 || sub_n;
                    }
                    if u >= tail_start {
                        let ti = u - tail_start;
                        tail_t[ti] = if bt == 0.0 && sub_t { xt } else { bt };
                        tail_m[ti] = if bm == 0.0 && sub_m { xm } else { bm };
                        tail_n[ti] = if bn == 0.0 && sub_n { xn } else { bn };
                    }
                }
            }

            let mut eval_energies = |e_t: f64,
                                     e_m: f64,
                                     e_n: f64,
                                     ht: bool,
                                     hm: bool,
                                     hn: bool,
                                     w: usize|
             -> Result<(), ()> {
                if !e_t.is_finite() || !e_m.is_finite() || !e_n.is_finite() {
                    return Err(());
                }
                scale_nz |= ht || hm || hn;
                scale_nz_n |= hn;
                scale_nse_sq += e_n;
                let inv_sqrt_len = 1.0 / (w as f64).sqrt();
                let rms_t = e_t.sqrt() * inv_sqrt_len;
                let rms_m = e_m.sqrt() * inv_sqrt_len;
                let sigma = e_n.sqrt() * inv_sqrt_len;
                if (hn && (e_n == 0.0 || sigma == 0.0))
                    || (((ht && (e_t == 0.0 || rms_t == 0.0))
                        || (hm && (e_m == 0.0 || rms_m == 0.0)))
                        && sigma < 1.6e-162)
                {
                    return Err(());
                }
                let (a_t, a_m) = (rms_t.max(sigma), rms_m.max(sigma));
                if a_t == 0.0 && a_m == 0.0 {
                    cell_count += 1;
                    return Ok(());
                }
                if a_t == 0.0 || a_m == 0.0 || !a_t.is_finite() || !a_m.is_finite() {
                    return Err(());
                }
                let diff = a_t - a_m;
                sc_num_sq += diff * diff;
                sc_den_sq += a_t * a_t;
                mag_diff_sum += (a_t.ln() - a_m.ln()).abs();
                cell_count += 1;
                Ok(())
            };

            let sum_tail = |len: usize,
                            tt: &[f64],
                            tm: &[f64],
                            tn: &[f64]|
             -> (f64, f64, f64, bool, bool, bool) {
                let (mut et, mut em, mut en) = (0.0_f64, 0.0_f64, 0.0_f64);
                let (mut ht, mut hm, mut hn) = (false, false, false);
                for k in 0..len {
                    let (vt, vm, vn) = (tt[k], tm[k], tn[k]);
                    et += vt * vt;
                    em += vm * vm;
                    en += vn * vn;
                    ht |= vt != 0.0;
                    hm |= vm != 0.0;
                    hn |= vn != 0.0;
                }
                (et, em, en, ht, hm, hn)
            };

            if t_sel < m_scale {
                let (et, em, en, ht, hm, hn) = sum_tail(t_sel, &tail_t, &tail_m, &tail_n);
                if eval_energies(et, em, en, ht, hm, hn, t_sel).is_err() {
                    return f64::INFINITY;
                }
            } else {
                let chunks_per_win = m_scale / SUB_CHUNK;
                let chunks_per_hop = hop / SUB_CHUNK;
                let mut c0 = 0usize;
                while (c0 + chunks_per_win) * SUB_CHUNK <= t_sel {
                    let c_end = c0 + chunks_per_win;
                    let et: f64 = chunk_t[c0..c_end].iter().sum();
                    let em: f64 = chunk_m[c0..c_end].iter().sum();
                    let en: f64 = chunk_n[c0..c_end].iter().sum();
                    let ht = nz_t[c0..c_end].iter().any(|&b| b);
                    let hm = nz_m[c0..c_end].iter().any(|&b| b);
                    let hn = nz_n[c0..c_end].iter().any(|&b| b);
                    if eval_energies(et, em, en, ht, hm, hn, m_scale).is_err() {
                        return f64::INFINITY;
                    }
                    c0 += chunks_per_hop;
                }
                let start = c0 * SUB_CHUNK;
                let last_start = t_sel - m_scale;
                if last_start + hop != start {
                    let (et, em, en, ht, hm, hn) = sum_tail(m_scale, &tail_t, &tail_m, &tail_n);
                    if eval_energies(et, em, en, ht, hm, hn, m_scale).is_err() {
                        return f64::INFINITY;
                    }
                }
            }
        }

        if !sc_num_sq.is_finite()
            || !sc_den_sq.is_finite()
            || !mag_diff_sum.is_finite()
            || (scale_nz_n && scale_nse_sq == 0.0)
        {
            return f64::INFINITY;
        }
        let l_sc = if sc_den_sq == 0.0 {
            if sc_num_sq == 0.0 && !scale_nz {
                0.0
            } else {
                return f64::INFINITY;
            }
        } else {
            sc_num_sq.sqrt() / sc_den_sq.sqrt()
        };
        let l_mag = if cell_count == 0 {
            0.0
        } else {
            mag_diff_sum / (cell_count as f64)
        };
        total_loss += l_sc + l_mag;
    }

    let out = total_loss / (SCALES.len() as f64);
    if out.is_finite() { out } else { f64::INFINITY }
}

// ============================================================================
// 2. 33-Band Base-10 1/3-Octave Noise-Adjusted LTAS (`r[qual.companion.ltas]`)
// ============================================================================

/// Single interval in the canonical 33-band base-10 1/3-octave LTAS partition (`ref[qual.companion.ltas]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtasInterval {
    /// Lower edge in Hz (`binary64` rounded once).
    pub low_hz: f64,
    /// Upper edge in Hz (`binary64` rounded once).
    pub high_hz: f64,
    /// Whether the lower boundary is inclusive (`true` for `[low, ...)`, `false` for `(18000, 20000]`).
    pub include_low: bool,
    /// Whether the upper boundary is inclusive (`true` only at `18,000 Hz` and `20,000 Hz`).
    pub include_high: bool,
}

impl LtasInterval {
    /// Geometric center frequency $\sqrt{f_{\text{low}} f_{\text{high}}}$ in Hz.
    #[inline]
    #[must_use]
    pub fn center_hz(self) -> f64 {
        (self.low_hz * self.high_hz).sqrt()
    }

    /// Tests whether frequency `f_hz` belongs to this interval under `r[qual.companion.ltas]` bullet 1.
    #[inline]
    #[must_use]
    pub fn contains(self, f_hz: f64) -> bool {
        let lo_ok = if self.include_low {
            f_hz >= self.low_hz
        } else {
            f_hz > self.low_hz
        };
        let hi_ok = if self.include_high {
            f_hz <= self.high_hz
        } else {
            f_hz < self.high_hz
        };
        lo_ok && hi_ok
    }
}

/// Returns the exact 33-band base-10 1/3-octave partition over `[20 Hz, 20,000 Hz]` anchored at `1,000 Hz`:
/// - `21` Sub/Low-Mid intervals over `[20 Hz, 2,000 Hz)` ($j = -17 \dots 3$, clipped at `20 Hz`, split at `2,000 Hz`)
/// - `11` HF Diagnostic intervals over `[2,000 Hz, 18,000 Hz]` ($j = 3 \dots 13$, `18,000 Hz` inclusive)
/// - `1` Upper Air interval over `(18,000 Hz, 20,000 Hz]` ($j = 13$ above `18 kHz`, `20,000 Hz` inclusive)
///   (`ref[qual.companion.ltas]`).
#[must_use]
pub fn canonical_ltas_33_intervals_hz() -> [LtasInterval; 33] {
    let edge = |j2_plus: i32| -> f64 { 1000.0 * 10.0_f64.powf(f64::from(j2_plus) / 20.0) };
    let mut out = [LtasInterval {
        low_hz: 20.0,
        high_hz: 20.0,
        include_low: true,
        include_high: false,
    }; 33];
    let mut idx = 0usize;

    // j = -17 ..= +2 -> 20 intervals (j = -17 clipped at 20.0 Hz)
    for j in -17..=2 {
        let lo = if j == -17 { 20.0 } else { edge(2 * j - 1) };
        let hi = edge(2 * j + 1);
        out[idx] = LtasInterval {
            low_hz: lo,
            high_hz: hi,
            include_low: true,
            include_high: false,
        };
        idx += 1;
    }

    // j = +3 ([1778.279.., 2238.721..]) split at 2,000.0 Hz -> 21st Sub/Low-Mid & 1st HF Diagnostic
    let e5 = edge(5);
    let e7 = edge(7);
    out[idx] = LtasInterval {
        low_hz: e5,
        high_hz: 2_000.0,
        include_low: true,
        include_high: false,
    };
    idx += 1;
    out[idx] = LtasInterval {
        low_hz: 2_000.0,
        high_hz: e7,
        include_low: true,
        include_high: false,
    };
    idx += 1;

    // j = +4 ..= +12 -> 9 intervals
    for j in 4..=12 {
        out[idx] = LtasInterval {
            low_hz: edge(2 * j - 1),
            high_hz: edge(2 * j + 1),
            include_low: true,
            include_high: false,
        };
        idx += 1;
    }

    // j = +13 ([17782.794.., 22387.211..]) split at 18,000.0 Hz & clipped at 20,000.0 Hz
    let e25 = edge(25);
    out[idx] = LtasInterval {
        low_hz: e25,
        high_hz: 18_000.0,
        include_low: true,
        include_high: true,
    };
    idx += 1;
    out[idx] = LtasInterval {
        low_hz: 18_000.0,
        high_hz: 20_000.0,
        include_low: false,
        include_high: true,
    };
    debug_assert_eq!(idx + 1, 33);
    out
}

/// Returns the 33 geometric center frequencies (in Hz) of [`canonical_ltas_33_intervals_hz`].
#[must_use]
pub fn canonical_ltas_33_centers_hz() -> [f64; 33] {
    let intervals = canonical_ltas_33_intervals_hz();
    let mut centers = [0.0_f64; 33];
    for (dst, iv) in centers.iter_mut().zip(&intervals) {
        *dst = iv.center_hz();
    }
    centers
}

/// Complete 33-band LTAS evaluation report (`ref[qual.companion.ltas]`).
#[derive(Clone, Debug)]
pub struct LtasEvaluationReport {
    /// Whether the evaluation met preconditions (`Nyquist >= 20 kHz`, non-empty finite signals, valid guard).
    pub available: bool,
    /// Whether any excluded noise-limited band violated the `+1.0 dB` residual guard ($R^{\text{LTAS}} \le 10^{0.1} F^{\text{LTAS}}$).
    pub excluded_band_guard_failed: bool,
    /// Maximum absolute noise-adjusted band deviation $D^{\text{adj}}_{\max}$ in dB (`+inf` on failure).
    pub max_adj_deviation_db: f64,
    /// Unadjusted maximum band deviation in dB across active bands (reported separately per §3).
    pub max_unadjusted_deviation_db: f64,
    /// Number of active (non-excluded) bands evaluated.
    pub active_band_count: usize,
    /// Which of the 33 bands were excluded under the noise-limited band exception.
    pub excluded_bands: [bool; 33],
    /// Raw residual band power $R^{\text{LTAS}}_{b}$ for all 33 bands (used for adjacent-tier excluded-band non-regression).
    pub residual_band_power: [f64; 33],
}

fn fft_radix2_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && n == im.len());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let ang = -2.0 * PI / (len as f64);
        let (wlen_sin, wlen_cos) = ang.sin_cos();
        let mut start = 0usize;
        while start < n {
            let (mut wr, mut wi) = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + half] * wr - im[start + k + half] * wi;
                let v_im = re[start + k + half] * wi + im[start + k + half] * wr;
                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + half] = u_re - v_re;
                im[start + k + half] = u_im - v_im;
                let next_wr = wr * wlen_cos - wi * wlen_sin;
                wi = wr * wlen_sin + wi * wlen_cos;
                wr = next_wr;
            }
            start += len;
        }
        len <<= 1;
    }
}

fn accumulate_welch_33band_powers(
    sig: &[f64],
    sample_rate_hz: f64,
    max_freq_hz: f64,
    intervals: &[LtasInterval; 33],
) -> [f64; 33] {
    // Scale physical Welch window duration (>= 341.3 ms) and FFT size proportionally with sample_rate_hz
    // so bin spacing (<= 1.465 Hz) and Hann main-lobe width resolve Band 0 [20.0, 22.39 Hz) at all rates
    let rate_scale = (sample_rate_hz / 48_000.0).ceil().max(1.0) as usize;
    let win_cap = 16_384usize.saturating_mul(rate_scale.next_power_of_two());
    let n_fft = (win_cap * 2).max((sample_rate_hz.ceil() as usize).next_power_of_two());
    let seg_len = sig.len().min(win_cap);
    let hop = (seg_len / 2).max(1);

    let mut re = vec![0.0_f64; n_fft];
    let mut im = vec![0.0_f64; n_fft];
    let mut spec_acc = vec![0.0_f64; n_fft / 2 + 1];
    let mut n_wins = 0usize;

    let mut win = vec![1.0_f64; seg_len];
    let mut win_norm = seg_len as f64;
    if seg_len >= 8 {
        win_norm = 0.0;
        for (i, w) in win.iter_mut().enumerate() {
            let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((seg_len - 1) as f64)).cos());
            *w = v;
            win_norm += v * v;
        }
    }
    let inv_win_norm = 1.0 / win_norm.max(1.0e-12);

    let mut accum_full_seg = |seg_start: usize| {
        re.fill(0.0);
        im.fill(0.0);
        for i in 0..seg_len {
            re[i] = sig[seg_start + i] * win[i];
        }
        fft_radix2_in_place(&mut re, &mut im);
        for (k, slot) in spec_acc.iter_mut().enumerate() {
            let mag_sq = (re[k] * re[k] + im[k] * im[k]) * inv_win_norm;
            let one_sided = if k == 0 || k * 2 == n_fft {
                mag_sq
            } else {
                2.0 * mag_sq
            };
            *slot += one_sided;
        }
        n_wins += 1;
    };

    let mut start = 0usize;
    while start + seg_len <= sig.len() {
        accum_full_seg(start);
        start += hop;
    }
    let last_start = sig.len() - seg_len;
    if last_start + hop != start {
        accum_full_seg(last_start);
    }

    let inv_wins = 1.0 / (n_wins as f64);
    let df = sample_rate_hz / (n_fft as f64);
    let mut band_p = [0.0_f64; 33];
    for (k, &p_sum) in spec_acc.iter().enumerate() {
        let f_hz = (k as f64) * df;
        if f_hz < 20.0 {
            continue;
        }
        if f_hz > max_freq_hz {
            break;
        }
        let p_bin = p_sum * inv_wins;
        for (b, iv) in intervals.iter().enumerate() {
            if iv.high_hz <= max_freq_hz + 1.0e-9 && iv.contains(f_hz) {
                band_p[b] += p_bin;
                break;
            }
        }
    }
    band_p
}

/// Evaluates the full 33-Band Base-10 1/3-Octave Noise-Adjusted LTAS report (`ref[qual.companion.ltas]`).
///
/// Enforces `Nyquist >= 20,000 Hz` (`sample_rate_hz >= 40,000 Hz`) when `require_full_20khz` is `true`;
/// when `require_full_20khz` is `false`, evaluates all canonical intervals below `0.48 * sample_rate_hz`
/// so sub-`40 kHz` vintage digital devices can still inspect passband spectral tilt.
#[must_use]
pub fn evaluate_ltas_33band_report(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    require_full_20khz: bool,
) -> LtasEvaluationReport {
    let fail_unavailable = LtasEvaluationReport {
        available: false,
        excluded_band_guard_failed: false,
        max_adj_deviation_db: f64::INFINITY,
        max_unadjusted_deviation_db: f64::INFINITY,
        active_band_count: 0,
        excluded_bands: [false; 33],
        residual_band_power: [0.0; 33],
    };

    let min_sr = if require_full_20khz {
        40_000.0
    } else {
        8_000.0
    };
    if !(min_sr..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || !target.iter().all(|v| v.is_finite())
        || !model.iter().all(|v| v.is_finite())
    {
        return fail_unavailable;
    }
    let n = target.len();
    if let Some(ns) = noise
        && (ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()))
    {
        return fail_unavailable;
    }

    let max_freq_hz = if require_full_20khz {
        20_000.0
    } else {
        (0.48 * sample_rate_hz).min(20_000.0)
    };
    let intervals = canonical_ltas_33_intervals_hz();
    let residual: Vec<f64> = model
        .iter()
        .zip(target.iter())
        .map(|(m, t)| m - t)
        .collect();

    let p_y = accumulate_welch_33band_powers(target, sample_rate_hz, max_freq_hz, &intervals);
    let p_m = accumulate_welch_33band_powers(model, sample_rate_hz, max_freq_hz, &intervals);
    let p_r = accumulate_welch_33band_powers(&residual, sample_rate_hz, max_freq_hz, &intervals);
    let p_f = if let Some(ns) = noise {
        let pf = accumulate_welch_33band_powers(&ns[..n], sample_rate_hz, max_freq_hz, &intervals);
        if ns[..n].iter().any(|&v| v != 0.0) && pf.iter().all(|&v| v == 0.0) {
            return fail_unavailable;
        }
        pf
    } else {
        [0.0_f64; 33]
    };

    let mut max_adj_db = 0.0_f64;
    let mut max_raw_db = 0.0_f64;
    let mut active_count = 0usize;
    let mut excluded_bands = [false; 33];
    let mut guard_failed = false;

    // Dynamic range floor for negligible numerical FFT leakage (-180 dB relative to peak band power)
    let peak_band_p = p_y.iter().copied().fold(0.0_f64, f64::max);
    let fft_leak_floor = peak_band_p * 1.0e-18;
    let any_nz_res = residual.iter().any(|&r| r != 0.0);

    for b in 0..33 {
        if intervals[b].high_hz > max_freq_hz + 1.0e-9 {
            excluded_bands[b] = true;
            continue;
        }
        let (py, pm, pr, pf) = (p_y[b], p_m[b], p_r[b], p_f[b]);
        if !py.is_finite() || !pm.is_finite() || !pr.is_finite() || !pf.is_finite() {
            guard_failed = true;
            max_adj_db = f64::INFINITY;
            max_raw_db = f64::INFINITY;
            continue;
        }

        // Exclude noise-limited band when P_y - F_LTAS <= F_LTAS (`py <= 2.0 * pf`), enforcing R_LTAS <= 10^{0.1} F_LTAS
        if pf > 0.0 && py <= 2.0 * pf {
            excluded_bands[b] = true;
            if pr > ONE_DB_POWER_FACTOR * pf {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
            }
            continue;
        }
        if pf == 0.0 && py <= fft_leak_floor {
            excluded_bands[b] = true;
            let leak_guard = ONE_DB_POWER_FACTOR * py.max(fft_leak_floor);
            if (peak_band_p == 0.0 && (pr > 0.0 || any_nz_res))
                || (peak_band_p > 0.0 && pr > leak_guard)
            {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
            }
            continue;
        }

        // Active band: P_y - F_LTAS > F_LTAS >= 0
        active_count += 1;
        let denom_adj = py - pf;
        if pm == 0.0 {
            max_adj_db = f64::INFINITY;
            max_raw_db = f64::INFINITY;
        } else {
            let adj_db = (10.0 * (pm / denom_adj).log10()).abs();
            let raw_db = (10.0 * (pm / py).log10()).abs();
            if !adj_db.is_finite() || !raw_db.is_finite() {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
                max_raw_db = f64::INFINITY;
            } else {
                max_adj_db = max_adj_db.max(adj_db);
                max_raw_db = max_raw_db.max(raw_db);
            }
        }
    }

    let available = active_count > 0 && !guard_failed;
    if !available {
        max_adj_db = f64::INFINITY;
        max_raw_db = f64::INFINITY;
    }
    LtasEvaluationReport {
        available,
        excluded_band_guard_failed: guard_failed,
        max_adj_deviation_db: max_adj_db,
        max_unadjusted_deviation_db: max_raw_db,
        active_band_count: active_count,
        excluded_bands,
        residual_band_power: p_r,
    }
}

/// Computes `(max_adj_deviation_db, active_band_count)` for the canonical 33-band LTAS (`Nyquist >= 20 kHz`,
/// `ref[qual.companion.ltas]`).
#[must_use]
pub fn compute_ltas_33band_max_deviation_db(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
) -> (f64, usize) {
    let rep = evaluate_ltas_33band_report(target, model, noise, sample_rate_hz, true);
    (rep.max_adj_deviation_db, rep.active_band_count)
}

// ============================================================================
// 3. Segmental Frame-Wise ESR (`SegESR`, r[qual.companion.seg-esr])
// ============================================================================

/// Contiguous constant-sample-rate timeline segment for multi-rate `SegESR` aggregation (`ref[qual.companion.seg-esr]`).
#[derive(Clone, Copy, Debug)]
pub struct MultiRateSegEsrSlice<'a> {
    pub sample_rate_hz: f64,
    pub target: &'a [f64],
    pub model: &'a [f64],
    pub noise: Option<&'a [f64]>,
}

/// Evaluates duration-weighted `20 ms` (`L = round(fs / 50)`) `SegESR` across one or more constant-rate
/// segments without spanning rate transitions, retaining exact-duration-weighted partial tail frames
/// and enforcing the `+1.0 dB` excluded-frame residual guard (`ref[qual.companion.seg-esr]`).
#[must_use]
pub fn compute_multirate_seg_esr_db(segments: &[MultiRateSegEsrSlice<'_>]) -> (f64, usize, bool) {
    if segments.is_empty() {
        return (f64::INFINITY, 0, false);
    }
    let mut weighted_lin_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    let mut active_segs = 0usize;

    for seg in segments {
        let fs = seg.sample_rate_hz;
        if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&fs)
            || seg.target.is_empty()
            || seg.target.len() != seg.model.len()
            || !seg.target.iter().all(|v| v.is_finite())
            || !seg.model.iter().all(|v| v.is_finite())
        {
            return (f64::INFINITY, 0, false);
        }
        let n = seg.target.len();
        if let Some(ns) = seg.noise
            && (ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()))
        {
            return (f64::INFINITY, 0, false);
        }

        let full_len = (fs / 50.0).round().max(1.0) as usize;
        let mut start = 0usize;
        while start < n {
            let end = (start + full_len).min(n);
            let n_f = end - start;
            let mut e_sig = 0.0_f64;
            let mut e_err = 0.0_f64;
            let mut e_nse = 0.0_f64;
            let mut nz_sig = false;
            let mut nz_err = false;
            let mut nz_nse = false;
            for i in start..end {
                let yt = seg.target[i];
                let ym = seg.model[i];
                let yn = seg.noise.map_or(0.0, |ns| ns[i]);
                let diff = ym - yt;
                e_sig += yt * yt;
                e_err += diff * diff;
                e_nse += yn * yn;
                nz_sig |= yt != 0.0;
                nz_err |= diff != 0.0;
                nz_nse |= yn != 0.0;
            }
            if (nz_sig && e_sig == 0.0)
                || (nz_err && e_err == 0.0)
                || (nz_nse && e_nse == 0.0)
                || !e_sig.is_finite()
                || !e_err.is_finite()
                || !e_nse.is_finite()
            {
                return (f64::INFINITY, active_segs, false);
            }
            let is_active = if e_nse > 0.0 {
                e_sig > SIX_DB_POWER_FACTOR * e_nse
            } else {
                e_sig > 0.0
            };

            if is_active {
                let denom = e_sig + e_nse;
                let r_f = if denom.is_finite() {
                    (e_err - e_nse).max(0.0) / denom
                } else {
                    (0.5 * e_err - 0.5 * e_nse).max(0.0) / (0.5 * e_sig + 0.5 * e_nse)
                };
                let w_f = (n_f as f64) / fs;
                weighted_lin_sum += w_f * r_f;
                weight_sum += w_f;
                active_segs += 1;
            } else if e_err > ONE_DB_POWER_FACTOR * e_nse {
                // Excluded noise-limited / silent frame guard: R_{f,c} <= 10^{0.1} F_{f,c}
                return (f64::INFINITY, active_segs, false);
            }
            start = end;
        }
    }

    if active_segs == 0
        || weight_sum <= 0.0
        || !weighted_lin_sum.is_finite()
        || !weight_sum.is_finite()
    {
        return (f64::INFINITY, 0, false);
    }
    let r_bar = weighted_lin_sum / weight_sum;
    if !r_bar.is_finite() {
        return (f64::INFINITY, active_segs, false);
    }
    let db = if r_bar == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * r_bar.log10()
    };
    (db, active_segs, true)
}

/// Computes `20 ms` (`L = round(fs / 50)`) duration-weighted Segmented ESR (`SegESR`) in dB
/// over active signal frames, including partial trailing frames (`ref[qual.companion.seg-esr]`).
#[must_use]
pub fn compute_seg_esr_20ms_db(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
) -> (f64, usize) {
    let (db, count, _valid) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
        sample_rate_hz,
        target,
        model,
        noise,
    }]);
    (db, count)
}
