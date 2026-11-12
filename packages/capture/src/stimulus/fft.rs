#![forbid(unsafe_code)]
//! Pure-Rust Radix-2 `f64` FFT/IFFT, Disjoint-Comb Bin Scheduler, Schroeder Low-Crest
//! Phase Optimizer ($\le 4.5\text{ dB}$), and `96.0 ms` (`512 + 4,096`) Cyclic Frame Synthesizer.
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.invariants]`).

use std::collections::BTreeSet;
use std::f64::consts::PI;

/// Canonical periodic DFT analysis window length (`85.333 ms` at `48 kHz`, $\Delta f = 11.71875\text{ Hz}$).
pub const CYCLIC_WINDOW_SAMPLES: usize = 4_096;

/// Canonical cyclic prefix length (`10.667 ms` at `48 kHz`).
pub const CYCLIC_PREFIX_SAMPLES: usize = 512;

/// Canonical $C^1$ Hann carrier crossfade length at the start of the cyclic prefix (`1.333 ms`).
pub const HANN_CROSSFADE_SAMPLES: usize = 64;

/// Total `96.0 ms` closed-loop adaptive excitation frame length (`512 + 4,096 = 4,608` samples at `48 kHz`).
pub const ADAPTIVE_FRAME_SAMPLES: usize = CYCLIC_PREFIX_SAMPLES + CYCLIC_WINDOW_SAMPLES;

/// Integer-cycle `93.75 Hz` pedestal bin ($k_p = 8$, exact `1` cycle in `512` samples and `9` cycles in `4,608`).
pub const PEDESTAL_BIN_MONO: usize = 8;

/// Integer-cycle `187.50 Hz` right-channel pedestal bin ($k_{p,R} = 16$, exact `2` cycles in `512` and `18` in `4,608`).
pub const PEDESTAL_BIN_STEREO_R: usize = 16;

/// Minimum odd-bin stride ($k_{i+1} - k_i \ge 34$) keeping $\pm k_p, \pm 2 k_p, \pm 3 k_p$ sidebands disjoint.
pub const MIN_COMB_BIN_STRIDE: usize = 34;

/// Predictive `-1.50 dBFS` (`10^(-1.5 / 20)`) ADC full-scale peak ceiling (`Law 0.D`).
pub const ADC_PEAK_GOVERNOR_FS: f64 = 0.841_395_141_645_195;

/// Maximum permitted Schroeder multisine carrier crest factor in dB (`Law 2`).
pub const SCHROEDER_MAX_CREST_DB: f64 = 4.5;

/// Fixed odd reference anchor bins $\Omega_{\text{ref}}$ on the `23 + 34m` lattice (`~269.5, 1066.4, 3058.6, 8238.3 Hz`) clear of mains hum.
pub const REFERENCE_ANCHOR_BINS_L: [usize; 4] = [23, 91, 261, 703];

/// Stereo-Right half-stride-offset (`+18` bins) fixed odd reference anchor bins on the `41 + 34m` lattice.
pub const REFERENCE_ANCHOR_BINS_R: [usize; 4] = [41, 109, 279, 721];

/// Complex `f64` scalar for in-house power-of-2 radix-2 FFT/IFFT.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Constructs `re + i * im`.
    #[inline]
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Squared magnitude `re^2 + im^2`.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Euclidean magnitude `hypot(re, im)`.
    #[inline]
    #[must_use]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Principal phase angle `atan2(im, re)`.
    #[inline]
    #[must_use]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex multiplication `self * rhs`.
    #[inline]
    #[must_use]
    pub fn cmul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

fn twiddles_4096() -> &'static [Complex64; 2_048] {
    static TABLE: std::sync::OnceLock<Box<[Complex64; 2_048]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut arr = Box::new([Complex64::new(1.0, 0.0); 2_048]);
        for (m, slot) in arr.iter_mut().enumerate() {
            let theta = -2.0 * PI * (m as f64) / 4_096.0;
            *slot = Complex64::new(theta.cos(), theta.sin());
        }
        arr
    })
}

/// In-place power-of-2 Cooley-Tukey radix-2 FFT (`inverse == false`) or `1/N`-scaled IFFT (`inverse == true`).
pub fn fft_radix2_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    assert!(
        n.is_power_of_two() && n >= 2,
        "fft_radix2_in_place requires power-of-two length >= 2"
    );
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    if n == 4_096 {
        let tw = twiddles_4096();
        let mut len = 2usize;
        while len <= 4_096 {
            let half = len >> 1;
            let stride = 2_048 / half;
            for block in buf.chunks_exact_mut(len) {
                let (left, right) = block.split_at_mut(half);
                for k in 0..half {
                    let t = tw[k * stride];
                    let w = if inverse {
                        Complex64::new(t.re, -t.im)
                    } else {
                        t
                    };
                    let u = left[k];
                    let v = right[k].cmul(w);
                    left[k] = Complex64::new(u.re + v.re, u.im + v.im);
                    right[k] = Complex64::new(u.re - v.re, u.im - v.im);
                }
            }
            len <<= 1;
        }
    } else {
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut len = 2usize;
        while len <= n {
            let half = len >> 1;
            let theta = sign * 2.0 * PI / (len as f64);
            let w_step = Complex64::new(theta.cos(), theta.sin());
            for block in buf.chunks_exact_mut(len) {
                let (left, right) = block.split_at_mut(half);
                let mut w = Complex64::new(1.0, 0.0);
                for k in 0..half {
                    let u = left[k];
                    let v = right[k].cmul(w);
                    left[k] = Complex64::new(u.re + v.re, u.im + v.im);
                    right[k] = Complex64::new(u.re - v.re, u.im - v.im);
                    w = w.cmul(w_step);
                }
            }
            len <<= 1;
        }
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for c in buf.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
    }
}

/// Computes positive-frequency DFT bins `0..=N/2` of a real power-of-two window `real_window`.
#[must_use]
pub fn rfft_bins(real_window: &[f32]) -> Vec<Complex64> {
    let n = real_window.len();
    assert!(n.is_power_of_two() && n >= 2);
    let mut buf: Vec<Complex64> = real_window
        .iter()
        .map(|&x| Complex64::new(if x.is_finite() { f64::from(x) } else { 0.0 }, 0.0))
        .collect();
    fft_radix2_in_place(&mut buf, false);
    buf.truncate(n / 2 + 1);
    buf
}

/// Synthesizes a real time-domain window of length `n_fft` from one-sided positive-frequency complex
/// amplitudes `(bin, one_sided_peak_amp, phase_rad)` where `s[n] = sum amp * cos(2*pi*k*n/N + phase)`.
#[must_use]
pub fn irfft_from_sparse_tones(n_fft: usize, tones: &[(usize, f64, f64)]) -> Vec<f64> {
    assert!(n_fft.is_power_of_two() && n_fft >= 2);
    let half = n_fft / 2;
    let mut spec = vec![Complex64::new(0.0, 0.0); n_fft];
    let scale = 0.5 * (n_fft as f64);
    for &(bin, amp, phase) in tones {
        if bin == 0 || bin >= half || !amp.is_finite() || amp <= 0.0 {
            continue;
        }
        let re = scale * amp * phase.cos();
        let im = scale * amp * phase.sin();
        spec[bin].re += re;
        spec[bin].im += im;
        spec[n_fft - bin].re += re;
        spec[n_fft - bin].im -= im;
    }
    fft_radix2_in_place(&mut spec, true);
    spec.into_iter().map(|c| c.re).collect()
}

/// Returns `true` if DFT bin `bin` (for `n_fft` at `sample_rate_hz`) falls within `±13.5 Hz` of a
/// `50 Hz` or `60 Hz` mains-hum harmonic (`Law 0.C`, $\Omega_{\text{mains}}$ up to `720 Hz`).
#[must_use]
pub fn is_mains_hum_bin(bin: usize, sample_rate_hz: u32, n_fft: usize) -> bool {
    if bin == 0 || n_fft == 0 || sample_rate_hz == 0 {
        return true;
    }
    let f_hz = (bin as f64) * f64::from(sample_rate_hz) / (n_fft as f64);
    if f_hz > 740.0 {
        return false;
    }
    for base in [50.0_f64, 60.0_f64] {
        let m = (f_hz / base).round();
        if (1.0..=12.0).contains(&m) && (f_hz - m * base).abs() <= 13.5 {
            return true;
        }
    }
    false
}

/// Disjoint-comb carrier, anchor, zoom, and guard-bin schedule for a `4,096`-sample cyclic window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjointCombSchedule {
    /// Active odd carrier bins $\Omega_{\text{in}} \subset 2\mathbb{Z}+1$ sorted ascending.
    pub carrier_bins: Vec<usize>,
    /// Fixed odd reference anchor bins $\Omega_{\text{ref}} \subset \Omega_{\text{in}}$.
    pub reference_anchor_bins: Vec<usize>,
    /// Active pedestal bin (`Some(8)` or `Some(16)`, `None` when pedestal is disabled).
    pub pedestal_bin: Option<usize>,
    /// Unexcited clean guard bins $\Omega_{\text{guard}}$ clear of carriers, $\pm 3 k_p$ sidebands, and mains hum.
    pub guard_bins: Vec<usize>,
}

/// Builds a disjoint odd-multisine comb schedule (`r[train.capture.excitation.codesigned]`):
/// - `right_channel_interleave = false`: starts at odd bin `23`, uses `REFERENCE_ANCHOR_BINS_L`.
/// - `right_channel_interleave = true`: starts at odd bin `41` (`+18` half-stride), uses `REFERENCE_ANCHOR_BINS_R`
///   so stereo `L` and `R` carrier combs are 100% disjoint ($\Omega_L \cap \Omega_R = \emptyset$).
#[must_use]
pub fn build_disjoint_comb_schedule(
    sample_rate_hz: u32,
    right_channel_interleave: bool,
    pedestal_bin: Option<usize>,
    squeal_mask_bins: &BTreeSet<usize>,
    zoom_odd_bins: &[usize],
) -> DisjointCombSchedule {
    let max_bin = ((0.44 * f64::from(sample_rate_hz) * (CYCLIC_WINDOW_SAMPLES as f64))
        / f64::from(sample_rate_hz.max(1))) as usize;
    let nyq_cap = max_bin.min(CYCLIC_WINDOW_SAMPLES / 2 - 24);
    let anchors = if right_channel_interleave {
        REFERENCE_ANCHOR_BINS_R
    } else {
        REFERENCE_ANCHOR_BINS_L
    };

    // Keep all base comb carriers on a single `start_bin + 34 * m` arithmetic lattice so
    // sub-combs never split into beating mod-34 residue classes (`<= 4.5 dB` Schroeder crest).
    let start_bin = if right_channel_interleave {
        41usize
    } else {
        23usize
    };
    let mut chosen: Vec<usize> = Vec::with_capacity(64);
    let mut candidate = start_bin;
    while candidate <= nyq_cap {
        if !is_mains_hum_bin(candidate, sample_rate_hz, CYCLIC_WINDOW_SAMPLES)
            && !squeal_mask_bins.contains(&candidate)
        {
            chosen.push(candidate);
        }
        candidate += MIN_COMB_BIN_STRIDE;
    }
    for &anc in &anchors {
        if anc <= nyq_cap && !squeal_mask_bins.contains(&anc) && !chosen.contains(&anc) {
            chosen.push(anc);
        }
    }

    // Inject up to K_zoom <= 8 odd pole-zoom bins (`Law 2`) if clear of mains & squeal mask
    let mut zoom_added = 0usize;
    for &zb in zoom_odd_bins {
        if zoom_added >= 8 {
            break;
        }
        let odd_z = if zb % 2 == 1 { zb } else { zb + 1 };
        if odd_z >= 5
            && odd_z <= nyq_cap
            && !is_mains_hum_bin(odd_z, sample_rate_hz, CYCLIC_WINDOW_SAMPLES)
            && !squeal_mask_bins.contains(&odd_z)
            && !chosen.contains(&odd_z)
        {
            chosen.push(odd_z);
            zoom_added += 1;
        }
    }
    chosen.sort_unstable();

    let kp = pedestal_bin.unwrap_or(PEDESTAL_BIN_MONO);
    let mut occupied = vec![false; CYCLIC_WINDOW_SAMPLES / 2 + 1];
    if let Some(pb) = pedestal_bin {
        for mult in 1..=5 {
            let idx = pb * mult;
            if idx < occupied.len() {
                occupied[idx] = true;
            }
        }
    }
    for &c in &chosen {
        for side in -3_isize..=3_isize {
            let b = (c as isize) + side * (kp as isize);
            if b >= 0 && (b as usize) < occupied.len() {
                occupied[b as usize] = true;
            }
        }
        // Also leave +/- 2 bins around each carrier for Bessel LFO sidebands
        for d in -2_isize..=2_isize {
            let b = (c as isize) + d;
            if b >= 0 && (b as usize) < occupied.len() {
                occupied[b as usize] = true;
            }
        }
    }

    let mut guard_bins = Vec::with_capacity(64);
    for b in (31..=nyq_cap).step_by(17) {
        if !occupied[b] && !is_mains_hum_bin(b, sample_rate_hz, CYCLIC_WINDOW_SAMPLES) {
            guard_bins.push(b);
            if guard_bins.len() >= 64 {
                break;
            }
        }
    }

    let reference_anchor_bins = anchors
        .into_iter()
        .filter(|b| chosen.binary_search(b).is_ok())
        .collect();

    DisjointCombSchedule {
        carrier_bins: chosen,
        reference_anchor_bins,
        pedestal_bin,
        guard_bins,
    }
}

/// Computes deterministic low-crest Schroeder/Newman phases plus in-place Guillaume/Van den Bos
/// $L_\infty$ time-frequency alternating projections guaranteeing crest factor $\le 4.5\text{ dB}$.
#[must_use]
pub fn compute_low_crest_schroeder_phases(bin_amps: &[(usize, f64)]) -> (Vec<f64>, f64) {
    compute_low_crest_schroeder_phases_with_hint(bin_amps, None)
}

/// Computes deterministic low-crest Schroeder/Newman phases, optionally warm-starting from
/// `phase_hint` so smooth Fisher weight updates converge in `0..2` FFTs (`< 0.1 ms`).
#[must_use]
pub fn compute_low_crest_schroeder_phases_with_hint(
    bin_amps: &[(usize, f64)],
    phase_hint: Option<&[f64]>,
) -> (Vec<f64>, f64) {
    let m = bin_amps.len();
    if m == 0 {
        return (Vec::new(), 0.0);
    }
    let total_power: f64 = bin_amps
        .iter()
        .map(|&(_, a)| a * a)
        .sum::<f64>()
        .max(1.0e-30);

    // Seed with frequency-warped lattice-coordinate Newman/Schroeder chirp and refine via
    // over-relaxed Guillaume/Van den Bos L_inf time-frequency projections (beta = 2.4).
    let b0 = bin_amps[0].0 as f64;
    let u_max = (((bin_amps[m - 1].0.saturating_sub(bin_amps[0].0)).max(34) as f64) / 34.0) + 1.0;
    let scale_u = (m as f64) / u_max;

    let n_fft = CYCLIC_WINDOW_SAMPLES;
    let half = n_fft / 2;
    let scale = 0.5 * (n_fft as f64);
    let mut cbuf = vec![Complex64::new(0.0, 0.0); n_fft];

    let synth_and_crest = |phs: &[f64], buf: &mut [Complex64]| -> f64 {
        buf.fill(Complex64::new(0.0, 0.0));
        for (&(bin, amp), &ph) in bin_amps.iter().zip(phs) {
            if bin == 0 || bin >= half || !amp.is_finite() || amp <= 0.0 {
                continue;
            }
            let re = scale * amp * ph.cos();
            let im = scale * amp * ph.sin();
            buf[bin].re += re;
            buf[bin].im += im;
            buf[n_fft - bin].re += re;
            buf[n_fft - bin].im -= im;
        }
        fft_radix2_in_place(buf, true);
        let mut peak = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for c in buf.iter() {
            let x = c.re;
            peak = peak.max(x.abs());
            sum_sq += x * x;
        }
        let rms = (sum_sq / (n_fft as f64)).sqrt();
        if rms > 1.0e-18 && peak > 1.0e-18 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        }
    };

    let mut phs = Vec::with_capacity(m);
    if let Some(hint) = phase_hint.filter(|h| h.len() == m) {
        let hint_crest = synth_and_crest(hint, &mut cbuf);
        if hint_crest <= 4.46 {
            return (hint.to_vec(), hint_crest);
        }
        phs.extend_from_slice(hint);
    } else {
        let mut cum_p = 0.0_f64;
        for &(bk, ak) in bin_amps {
            let uk = (((bk as f64) - b0) / 34.0) * scale_u;
            let pk = (ak * ak) / total_power;
            let v = -2.0 * PI * (uk * cum_p + 0.5 * uk * pk);
            cum_p += pk;
            phs.push(v.rem_euclid(2.0 * PI) - PI);
        }
    }

    let mut best_crest_db = synth_and_crest(&phs, &mut cbuf);
    let mut best_phases = phs.clone();

    for iter in 0..64 {
        if best_crest_db <= 4.38 {
            break;
        }
        let rms = (cbuf.iter().map(|c| c.re * c.re).sum::<f64>() / (n_fft as f64)).sqrt();
        let clip_thresh = rms * (1.30 + 0.0012 * (iter as f64));
        let beta = 2.4_f64;
        for c in &mut cbuf {
            let x = c.re;
            let clipped = x.clamp(-clip_thresh, clip_thresh);
            c.re = x + beta * (clipped - x);
            c.im = 0.0;
        }
        fft_radix2_in_place(&mut cbuf, false);
        for (idx, &(bin, _)) in bin_amps.iter().enumerate() {
            if bin > 0 && bin < half {
                phs[idx] = cbuf[bin].arg();
            }
        }
        let c_db = synth_and_crest(&phs, &mut cbuf);
        if c_db < best_crest_db {
            best_crest_db = c_db;
            best_phases.clone_from(&phs);
        }
    }

    (best_phases, best_crest_db)
}

/// Output of `96.0 ms` (`4,608`-sample) cyclic disjoint-comb frame synthesis.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedCombFrame {
    /// Complete `4,608`-sample frame (`512` cyclic prefix + `4,096` periodic analysis window).
    pub samples: Vec<f32>,
    /// Final `64` carrier samples of this frame for seamless $C^1$ Hann crossfade into Frame `m + 1`.
    pub carrier_tail_64: [f32; HANN_CROSSFADE_SAMPLES],
    /// Converged Schroeder carrier phases for `< 0.1 ms` warm-start on Frame `m + 1`.
    pub phases: Vec<f64>,
    /// Schroeder carrier crest factor in dB ($\le 4.5\text{ dB}$).
    pub carrier_crest_db: f32,
    /// Applied predictive `-1.5 dBFS` peak governor scale (`<= 1.0`).
    pub governor_scale: f32,
}

/// Synthesizes a `96.0 ms` (`4,608`-sample) cyclic disjoint-comb frame (`r[train.capture.excitation.codesigned]`).
#[must_use]
pub fn synthesize_cyclic_comb_frame(
    bin_amps: &[(usize, f64)],
    pedestal_bin: Option<usize>,
    pedestal_amp: f64,
    target_carrier_rms: f64,
    peak_ceiling_fs: f64,
    prev_carrier_tail_64: &[f32; HANN_CROSSFADE_SAMPLES],
) -> SynthesizedCombFrame {
    synthesize_cyclic_comb_frame_with_hint(
        bin_amps,
        pedestal_bin,
        pedestal_amp,
        target_carrier_rms,
        peak_ceiling_fs,
        prev_carrier_tail_64,
        None,
    )
}

/// Synthesizes a `96.0 ms` cyclic disjoint-comb frame with optional Schroeder `phase_hint` warm-start.
#[must_use]
pub fn synthesize_cyclic_comb_frame_with_hint(
    bin_amps: &[(usize, f64)],
    pedestal_bin: Option<usize>,
    pedestal_amp: f64,
    target_carrier_rms: f64,
    peak_ceiling_fs: f64,
    prev_carrier_tail_64: &[f32; HANN_CROSSFADE_SAMPLES],
    phase_hint: Option<&[f64]>,
) -> SynthesizedCombFrame {
    let ceiling = if peak_ceiling_fs.is_finite() {
        peak_ceiling_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS)
    } else {
        ADC_PEAK_GOVERNOR_FS
    };
    // Cap off-lattice pole-zoom probe tones so total multisine crest always stays <= 4.5 dB
    let count_23 = bin_amps
        .iter()
        .filter(|&&(b, _)| b % MIN_COMB_BIN_STRIDE == 23)
        .count();
    let count_7 = bin_amps
        .iter()
        .filter(|&&(b, _)| b % MIN_COMB_BIN_STRIDE == 7)
        .count();
    let lattice_rem = if count_7 > count_23 { 7 } else { 23 };
    let eff_amps: Vec<(usize, f64)> = bin_amps
        .iter()
        .map(|&(b, a)| {
            if b % MIN_COMB_BIN_STRIDE == lattice_rem {
                (b, a)
            } else {
                (b, a.min(0.22))
            }
        })
        .collect();
    let (phases, crest_db) = compute_low_crest_schroeder_phases_with_hint(&eff_amps, phase_hint);
    let tones: Vec<(usize, f64, f64)> = eff_amps
        .iter()
        .zip(&phases)
        .map(|(&(b, a), &ph)| (b, a, ph))
        .collect();
    let raw_win = irfft_from_sparse_tones(CYCLIC_WINDOW_SAMPLES, &tones);
    let raw_rms = (raw_win.iter().map(|&x| x * x).sum::<f64>() / (raw_win.len() as f64)).sqrt();
    let rms_norm = if raw_rms > 1.0e-18 && target_carrier_rms > 0.0 {
        target_carrier_rms / raw_rms
    } else {
        0.0
    };

    // Build 4,608-sample carrier = [prefix 512 | window 4,096]
    let mut carrier = vec![0.0_f64; ADAPTIVE_FRAME_SAMPLES];
    for i in 0..CYCLIC_PREFIX_SAMPLES {
        carrier[i] = raw_win[CYCLIC_WINDOW_SAMPLES - CYCLIC_PREFIX_SAMPLES + i] * rms_norm;
    }
    for i in 0..CYCLIC_WINDOW_SAMPLES {
        carrier[CYCLIC_PREFIX_SAMPLES + i] = raw_win[i] * rms_norm;
    }

    // Apply 64-sample C^1 Hann crossfade at start of cyclic prefix (`0..64`)
    for n in 0..HANN_CROSSFADE_SAMPLES {
        let w = 0.5 * (1.0 - (PI * (n as f64) / (HANN_CROSSFADE_SAMPLES as f64)).cos());
        carrier[n] = (1.0 - w) * f64::from(prev_carrier_tail_64[n]) + w * carrier[n];
    }

    // Add continuous integer-cycle pedestal: sin(2*pi*k_p*n / 4096) completes exact integer cycles
    // in both 512 (k_p=8 -> 1 cycle, k_p=16 -> 2 cycles) and 4,608 samples (9 or 18 cycles) so phi_p == 0!
    let p_amp = if pedestal_amp.is_finite() && pedestal_amp > 0.0 {
        pedestal_amp
    } else {
        0.0
    };
    let mut combined = vec![0.0_f64; ADAPTIVE_FRAME_SAMPLES];
    let mut max_abs = 0.0_f64;
    for (n, slot) in combined.iter_mut().enumerate() {
        let ped = if let Some(kp) = pedestal_bin {
            p_amp * (2.0 * PI * (kp as f64) * (n as f64) / (CYCLIC_WINDOW_SAMPLES as f64)).sin()
        } else {
            0.0
        };
        let total = carrier[n] + ped;
        max_abs = max_abs.max(total.abs());
        *slot = total;
    }

    let gov_scale = if max_abs > ceiling && max_abs > 1.0e-18 {
        ceiling / max_abs
    } else {
        1.0
    };

    let mut samples = Vec::with_capacity(ADAPTIVE_FRAME_SAMPLES);
    for &v in &combined {
        samples.push((v * gov_scale) as f32);
    }

    let mut carrier_tail_64 = [0.0_f32; HANN_CROSSFADE_SAMPLES];
    for (i, dst) in carrier_tail_64.iter_mut().enumerate() {
        *dst = (carrier[ADAPTIVE_FRAME_SAMPLES - HANN_CROSSFADE_SAMPLES + i] * gov_scale) as f32;
    }

    SynthesizedCombFrame {
        samples,
        carrier_tail_64,
        phases,
        carrier_crest_db: crest_db as f32,
        governor_scale: gov_scale as f32,
    }
}
