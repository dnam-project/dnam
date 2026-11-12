#![forbid(unsafe_code)]
//! 192-Bit Wide Fixed/Expansion Gram Contraction (`Wide192Accum`) & `[4 × 4]` Register-Tiled
//! `uplo='U'` Sufficient-Statistic Accumulator (`solver::wide_gram`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.gram.wide192]`, `ref[train.gram.accuracy-contract]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.stat-uplo-u]`)

use super::{uplo_u_index, uplo_u_packed_len};

/// Veltkamp-Dekker splitting constant $2^{27} + 1$ for exact `f64 * f64 -> (hi, lo)` decomposition
/// without hardware FMA dependence (strictly bit-identical across `ARM64`, `x86_64`, and `wasm32`).
const DEKKER_SPLIT_F64: f64 = 134_217_729.0;

/// Number of active frames contracted in each L1-resident `[8 × 4]` / `[4 × 4]` register-tiled
/// micro-batch before merging into the `192`-bit (`3 × f64`) Priest/Dekker expansion accumulators.
const MICRO_BATCH_FRAMES: usize = 256;

/// Minimum magnitude $2^{-450}$ where `a_lo * b_lo >= 2^{-1004}` stays strictly normal (`> 2^{-1022}`).
const DEKKER_SAFE_MIN: f64 = f64::from_bits(0x23D0_0000_0000_0000);
/// Maximum magnitude $2^{+450}$ where `DEKKER_SPLIT_F64 * a < 2^{478}` cannot overflow `f64::MAX`.
const DEKKER_SAFE_MAX: f64 = f64::from_bits(0x5C10_0000_0000_0000);

#[inline(always)]
fn two_sum_exact(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline(always)]
fn two_prod_dekker_core(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let ca = DEKKER_SPLIT_F64 * a;
    let cb = DEKKER_SPLIT_F64 * b;
    let a_hi = ca - (ca - a);
    let a_lo = a - a_hi;
    let b_hi = cb - (cb - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
fn normalize_f64_mant_exp(x: f64) -> (f64, i32) {
    let ax = x.abs();
    let (bits, bias_adj) = if ax < f64::MIN_POSITIVE {
        // Scale subnormal up by 2^64 (`0x43F0_0000_0000_0000`) with zero bit loss.
        ((x * f64::from_bits(0x43F0_0000_0000_0000)).to_bits(), 1087)
    } else {
        (x.to_bits(), 1023)
    };
    let be = ((bits >> 52) & 0x7FF) as i32;
    let mant = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
    (mant, be - bias_adj)
}

#[inline]
fn ldexp_exact_f64(mut v: f64, mut exp_shift: i32) -> f64 {
    if v == 0.0 || exp_shift == 0 {
        return v;
    }
    while exp_shift > 512 {
        v *= f64::from_bits(0x5FF0_0000_0000_0000); // 2^512
        if !v.is_finite() {
            return v;
        }
        exp_shift -= 512;
    }
    while exp_shift < -512 {
        v *= f64::from_bits(0x1FF0_0000_0000_0000); // 2^-512
        if v == 0.0 {
            return 0.0;
        }
        exp_shift += 512;
    }
    let scale = f64::from_bits(((1023 + exp_shift) as u64) << 52);
    v * scale
}

#[cold]
#[inline(never)]
fn two_prod_dekker_extreme(a: f64, b: f64) -> (f64, f64) {
    if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
        return (a * b, 0.0);
    }
    let (am, ea) = normalize_f64_mant_exp(a);
    let (bm, eb) = normalize_f64_mant_exp(b);
    let (pm, em) = two_prod_dekker_core(am, bm);
    let e_tot = ea + eb;
    let hi = ldexp_exact_f64(pm, e_tot);
    if !hi.is_finite() {
        return (hi, 0.0);
    }
    (hi, ldexp_exact_f64(em, e_tot))
}

#[inline(always)]
fn two_prod_dekker(a: f64, b: f64) -> (f64, f64) {
    let aa = a.abs();
    let ab = b.abs();
    if (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&aa)
        && (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&ab)
    {
        two_prod_dekker_core(a, b)
    } else {
        two_prod_dekker_extreme(a, b)
    }
}

/// Deterministic `192`-bit (`3 × f64`, `~159`-bit significand) non-overlapping Priest/Dekker
/// expansion accumulator (`r[train.gram.wide192]`, `r[train.gram.accuracy-contract]`).
///
/// Uses strictly IEEE-754 binary64 `+`, `-`, and `*` operations (`0` `libm` calls, `0` hardware FMA
/// assumptions), guaranteeing exact cancellation immunity (e.g. `[1.0, 2^-30, -1.0] -> 2^-30`) and
/// bit-identical results across `ARM64`, `x86_64`, and `wasm32-unknown-unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Wide192Accum {
    limbs: [f64; 3],
}

impl Wide192Accum {
    /// Zero-initialized `192`-bit accumulator.
    #[must_use]
    pub const fn zero() -> Self {
        Self { limbs: [0.0; 3] }
    }

    /// Adds a finite `f64` scalar with full 3-limb Priest renormalization.
    #[inline(always)]
    pub fn add_f64(&mut self, x: f64) {
        if !x.is_finite() || x == 0.0 {
            return;
        }
        if self.limbs[0] == 0.0 {
            self.limbs[0] = x;
            return;
        }
        let (t0, e0) = two_sum_exact(self.limbs[0], x);
        let (t1, e1) = two_sum_exact(self.limbs[1], e0);
        if e1 == 0.0 && self.limbs[2] == 0.0 {
            let (r0, r1) = two_sum_exact(t0, t1);
            self.limbs[0] = r0;
            self.limbs[1] = r1;
            return;
        }
        let t2 = self.limbs[2] + e1;

        let (r0, u0) = two_sum_exact(t0, t1);
        let (r1, r2) = two_sum_exact(u0, t2);
        let (s0, v0) = two_sum_exact(r0, r1);
        let (s1, s2) = two_sum_exact(v0, r2);
        self.limbs = [s0, s1, s2];
    }

    /// Adds the exact `106`-bit product `a * b` (via Dekker Veltkamp splitting) into the `192`-bit expansion.
    #[inline(always)]
    pub fn add_product(&mut self, a: f64, b: f64) {
        if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
            return;
        }
        let (hi, lo) = two_prod_dekker(a, b);
        self.add_f64(hi);
        self.add_f64(lo);
    }

    #[cold]
    #[inline(never)]
    fn add_three_way_extreme(&mut self, w: f64, a: f64, b: f64) {
        let ab = a * b;
        if (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&ab.abs()) {
            self.add_product(w, ab);
            return;
        }
        let (wm, ew) = normalize_f64_mant_exp(w);
        let (am, ea) = normalize_f64_mant_exp(a);
        let (bm, eb) = normalize_f64_mant_exp(b);
        let e_tot = ew + ea + eb;
        let (wab_hi, wab_lo) = two_prod_dekker_core(wm * am, bm);
        let hi = ldexp_exact_f64(wab_hi, e_tot);
        if hi.is_finite() {
            self.add_f64(hi);
            self.add_f64(ldexp_exact_f64(wab_lo, e_tot));
        }
    }

    /// Adds a 3-way product `w * a * b` without premature intermediate overflow or underflow when `w * a`
    /// falls outside the normal Dekker envelope `[2^-450, 2^+450]`.
    #[inline(always)]
    pub(crate) fn add_three_way_f64(&mut self, w: f64, a: f64, b: f64) {
        if !w.is_finite() || !a.is_finite() || !b.is_finite() || w == 0.0 || a == 0.0 || b == 0.0 {
            return;
        }
        let wa = w * a;
        if (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&wa.abs()) {
            self.add_product(wa, b);
        } else {
            self.add_three_way_extreme(w, a, b);
        }
    }

    /// Merges another `Wide192Accum` into `self` from coarsest to finest limb.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.add_f64(other.limbs[0]);
        self.add_f64(other.limbs[1]);
        self.add_f64(other.limbs[2]);
    }

    /// Collapses the `192`-bit expansion to the nearest IEEE-754 `f64` (`0.0` canonicalized to `+0.0`)
    /// using Boldo-Melquiond round-to-odd on `limbs[1] + limbs[2]` to prevent double-rounding.
    #[must_use]
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        let (mut low, rem) = two_sum_exact(self.limbs[1], self.limbs[2]);
        if rem != 0.0 && low != 0.0 && (low.to_bits() & 1) == 0 {
            let same_sign = (low > 0.0) == (rem > 0.0);
            let adj_bits = if same_sign {
                low.to_bits() + 1
            } else {
                low.to_bits() - 1
            };
            low = f64::from_bits(adj_bits);
        }
        let total = self.limbs[0] + low;
        if total == 0.0 { 0.0 } else { total }
    }

    /// Exposes the raw ordered `[hi, mid, lo]` expansion limbs for diagnostic inspection.
    #[must_use]
    #[inline(always)]
    pub const fn limbs(self) -> [f64; 3] {
        self.limbs
    }
}

/// Accumulates a single weighted outer product $\mathbf{G} \mathrel{+}= w \boldsymbol{\phi}\boldsymbol{\phi}^\top$
/// and $\mathbf{C} \mathrel{+}= w \boldsymbol{\phi}\mathbf{y}^\top$ directly into `Wide192Accum` slices
/// (`r[train.gram.wide192]`).
pub fn accumulate_uplo_u_wide192(
    packed_gram: &mut [Wide192Accum],
    cross_corr: &mut [[Wide192Accum; 2]],
    features: &[f64],
    target_lr: [f64; 2],
    weight: f64,
) {
    let n = features.len();
    assert_eq!(packed_gram.len(), uplo_u_packed_len(n));
    assert_eq!(cross_corr.len(), n);
    if !weight.is_finite()
        || weight <= 0.0
        || !target_lr[0].is_finite()
        || !target_lr[1].is_finite()
        || !features.iter().all(|v| v.is_finite())
    {
        return;
    }
    for j in 0..n {
        let fj = features[j];
        if fj == 0.0 {
            continue;
        }
        let w_fj = weight * fj;
        let base = uplo_u_index(0, j);
        if (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&w_fj.abs()) {
            for i in 0..=j {
                packed_gram[base + i].add_product(w_fj, features[i]);
            }
            cross_corr[j][0].add_product(w_fj, target_lr[0]);
            cross_corr[j][1].add_product(w_fj, target_lr[1]);
        } else {
            for i in 0..=j {
                packed_gram[base + i].add_three_way_f64(weight, fj, features[i]);
            }
            cross_corr[j][0].add_three_way_f64(weight, fj, target_lr[0]);
            cross_corr[j][1].add_three_way_f64(weight, fj, target_lr[1]);
        }
    }
}

#[inline(always)]
fn fma4(acc: &mut [f64; 4], x: [f64; 4], s: f64) {
    acc[0] += x[0] * s;
    acc[1] += x[1] * s;
    acc[2] += x[2] * s;
    acc[3] += x[3] * s;
}

#[inline(always)]
fn mask256_overlaps(a: [u64; 4], b: [u64; 4]) -> bool {
    ((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3])) != 0
}

#[inline(always)]
fn mask256_or(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

#[inline(always)]
fn contract_offdiag_tile_4x4(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
    let (fi_pairs, fi_rem) = f_i.as_chunks::<2>();
    let (wf_pairs, wf_rem) = wf_j.as_chunks::<2>();
    let mut a0 = [0.0_f64; 4];
    let mut a1 = [0.0_f64; 4];
    let mut a2 = [0.0_f64; 4];
    let mut a3 = [0.0_f64; 4];
    let mut b0 = [0.0_f64; 4];
    let mut b1 = [0.0_f64; 4];
    let mut b2 = [0.0_f64; 4];
    let mut b3 = [0.0_f64; 4];
    for (&[fa, fb], &[wa, wb]) in fi_pairs.iter().zip(wf_pairs) {
        fma4(&mut a0, fa, wa[0]);
        fma4(&mut a1, fa, wa[1]);
        fma4(&mut a2, fa, wa[2]);
        fma4(&mut a3, fa, wa[3]);
        fma4(&mut b0, fb, wb[0]);
        fma4(&mut b1, fb, wb[1]);
        fma4(&mut b2, fb, wb[2]);
        fma4(&mut b3, fb, wb[3]);
    }
    for (&fa, &wa) in fi_rem.iter().zip(wf_rem) {
        fma4(&mut a0, fa, wa[0]);
        fma4(&mut a1, fa, wa[1]);
        fma4(&mut a2, fa, wa[2]);
        fma4(&mut a3, fa, wa[3]);
    }
    [
        [a0[0] + b0[0], a0[1] + b0[1], a0[2] + b0[2], a0[3] + b0[3]],
        [a1[0] + b1[0], a1[1] + b1[1], a1[2] + b1[2], a1[3] + b1[3]],
        [a2[0] + b2[0], a2[1] + b2[1], a2[2] + b2[2], a2[3] + b2[3]],
        [a3[0] + b3[0], a3[1] + b3[1], a3[2] + b3[2], a3[3] + b3[3]],
    ]
}

#[inline(always)]
fn contract_offdiag_tile_8x4(
    f_i0: &[[f64; 4]],
    f_i1: &[[f64; 4]],
    wf_j: &[[f64; 4]],
) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut a0 = [0.0_f64; 4];
    let mut a1 = [0.0_f64; 4];
    let mut a2 = [0.0_f64; 4];
    let mut a3 = [0.0_f64; 4];
    let mut b0 = [0.0_f64; 4];
    let mut b1 = [0.0_f64; 4];
    let mut b2 = [0.0_f64; 4];
    let mut b3 = [0.0_f64; 4];
    for ((&fa, &fb), &w) in f_i0.iter().zip(f_i1).zip(wf_j) {
        fma4(&mut a0, fa, w[0]);
        fma4(&mut b0, fb, w[0]);
        fma4(&mut a1, fa, w[1]);
        fma4(&mut b1, fb, w[1]);
        fma4(&mut a2, fa, w[2]);
        fma4(&mut b2, fb, w[2]);
        fma4(&mut a3, fa, w[3]);
        fma4(&mut b3, fb, w[3]);
    }
    ([a0, a1, a2, a3], [b0, b1, b2, b3])
}

/// Contracts `256-D` feature frames and stereo targets into a canonical `uplo='U'` Gram matrix
/// (`32,896 × f64`) and `[256][2]` cross-correlation matrix using `[8 × 4]` / `[4 × 4]`
/// register-tiled micro-batches merged into `Wide192Accum` (`r[train.gram.wide192]`,
/// `r[train.pipeline.stage-b-harvest]`).
pub fn contract_256d_gram_tiled_wide192(
    features: &[[f32; 256]],
    targets_lr: &[[f64; 2]],
    weights: &[f64],
    frame_stride: usize,
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let n_frames = features.len().min(targets_lr.len()).min(weights.len());
    let stride = frame_stride.max(1);
    let n_pack = uplo_u_packed_len(256);

    let mut valid_indices = Vec::with_capacity(n_frames.div_ceil(stride));
    let mut scan_idx = 0usize;
    while scan_idx < n_frames {
        let w = weights[scan_idx];
        let y = targets_lr[scan_idx];
        let fb = &features[scan_idx];
        if w.is_finite()
            && w > 0.0
            && y[0].is_finite()
            && y[1].is_finite()
            && fb.iter().all(|v| v.is_finite())
        {
            valid_indices.push(scan_idx);
        }
        scan_idx += stride;
    }
    let sampled_frames = valid_indices.len();
    if sampled_frames == 0 {
        return (vec![0.0_f64; n_pack], vec![[0.0_f64; 2]; 256]);
    }

    let mut wide_g = vec![Wide192Accum::zero(); n_pack];
    let mut wide_c = vec![[Wide192Accum::zero(); 2]; 256];

    // L1D-contiguous block layout `[64 blocks][256 frames][4 lanes]` (`8 KB` sequential per block).
    let b_cap = sampled_frames.min(MICRO_BATCH_FRAMES);
    let mut blk_f = vec![[[0.0_f64; 4]; MICRO_BATCH_FRAMES]; 64];
    let mut blk_wf = vec![[[0.0_f64; 4]; MICRO_BATCH_FRAMES]; 64];
    let mut is_lo = vec![[false; 256]; b_cap];
    let mut batch_y = [[0.0_f64; 2]; MICRO_BATCH_FRAMES];
    let mut batch_w = [0.0_f64; MICRO_BATCH_FRAMES];

    let scale_w = (n_frames as f64) / (sampled_frames as f64);
    for chunk_indices in valid_indices.chunks(MICRO_BATCH_FRAMES) {
        let b_len = chunk_indices.len();
        let mut w_max = 0.0_f64;
        let mut w_min = f64::INFINITY;
        let mut y_nz_mask = [[0_u64; 4]; 2];
        let mut max_y = [0.0_f64; 2];
        let mut min_nz_y = [f64::INFINITY; 2];
        let mut y_batch_mono = true;
        let mut col_max_f32 = [0.0_f32; 256];
        let mut col_min_nz_f32 = [f32::INFINITY; 256];

        for (k, &idx) in chunk_indices.iter().enumerate() {
            let w = weights[idx] * scale_w;
            batch_w[k] = w;
            w_max = w_max.max(w);
            w_min = w_min.min(w);

            let y = targets_lr[idx];
            batch_y[k] = y;
            if y[0].to_bits() != y[1].to_bits() {
                y_batch_mono = false;
            }
            let w_word = k >> 6;
            let w_bit = 1_u64 << (k & 63);
            let ay0 = y[0].abs();
            if ay0 > 0.0 {
                y_nz_mask[0][w_word] |= w_bit;
                max_y[0] = max_y[0].max(ay0);
                min_nz_y[0] = min_nz_y[0].min(ay0);
            }
            let ay1 = y[1].abs();
            if ay1 > 0.0 {
                y_nz_mask[1][w_word] |= w_bit;
                max_y[1] = max_y[1].max(ay1);
                min_nz_y[1] = min_nz_y[1].min(ay1);
            }

            // Branchless NEON-vectorizable f32 abs/max/min scan across the 256 feature columns.
            let fb = &features[idx];
            for d in 0..256 {
                let af = fb[d].abs();
                col_max_f32[d] = col_max_f32[d].max(af);
                let nz = if af > 0.0 { af } else { f32::INFINITY };
                col_min_nz_f32[d] = col_min_nz_f32[d].min(nz);
            }
        }

        let w_extreme = w_max > 1.0e90 || w_min < 1.0e-90;
        let mut col_max_f = [0.0_f64; 256];
        let mut col_min_nz_f = [f64::INFINITY; 256];
        let mut col_max_wf = [0.0_f64; 256];
        let mut col_min_nz_wf = [f64::INFINITY; 256];
        let mut col_wide_exp = [false; 256];
        let mut col_extreme = [false; 256];

        for d in 0..256 {
            let mf = f64::from(col_max_f32[d]);
            if mf == 0.0 {
                continue;
            }
            let nf = f64::from(col_min_nz_f32[d]);
            col_max_f[d] = mf;
            col_min_nz_f[d] = nf;
            // Exact upper/lower envelope: col_max_wf <= mf * w_max and col_min_nz_wf >= nf * w_min.
            if !w_extreme
                && mf * w_max <= nf * w_min * 65_536.0
                && mf * w_max <= 1.0e130
                && nf * w_min >= 1.0e-130
            {
                col_max_wf[d] = mf * w_max;
                col_min_nz_wf[d] = nf * w_min;
            } else {
                let mut mwf = 0.0_f64;
                let mut nwf = f64::INFINITY;
                for (k, &idx) in chunk_indices.iter().enumerate() {
                    let fd = f64::from(features[idx][d]);
                    if fd != 0.0 {
                        let awf = (batch_w[k] * fd).abs();
                        if awf > 0.0 && awf.is_finite() {
                            mwf = mwf.max(awf);
                            nwf = nwf.min(awf);
                        }
                    }
                }
                col_max_wf[d] = mwf;
                col_min_nz_wf[d] = nwf;
                let extreme = w_extreme
                    || mf > 1.0e130
                    || mwf > 1.0e130
                    || nf < 1.0e-130
                    || nwf < 1.0e-130
                    || mwf == 0.0;
                col_extreme[d] = extreme;
                col_wide_exp[d] = extreme || mf > nf * 65_536.0 || mwf > nwf * 65_536.0;
            }
        }

        let mut block_nz_mask = [[0_u64; 4]; 64];
        let mut col_nz_mask = [[0_u64; 4]; 256];
        let mut hi_col_mask = [[0_u64; 4]; 256];
        let mut frame_has_lo = [false; MICRO_BATCH_FRAMES];
        let mut any_lo = false;

        for blk in 0..64 {
            let b0 = blk * 4;
            if col_max_f[b0] == 0.0
                && col_max_f[b0 + 1] == 0.0
                && col_max_f[b0 + 2] == 0.0
                && col_max_f[b0 + 3] == 0.0
            {
                continue;
            }
            let blk_has_wide = col_wide_exp[b0]
                || col_wide_exp[b0 + 1]
                || col_wide_exp[b0 + 2]
                || col_wide_exp[b0 + 3];
            let dst_f = &mut blk_f[blk][..b_len];
            let dst_wf = &mut blk_wf[blk][..b_len];

            if !blk_has_wide {
                let mut m0 = [0_u64; 4];
                let mut m1 = [0_u64; 4];
                let mut m2 = [0_u64; 4];
                let mut m3 = [0_u64; 4];
                for (k, &idx) in chunk_indices.iter().enumerate() {
                    let w = batch_w[k];
                    let fb = &features[idx];
                    let f0 = f64::from(fb[b0]);
                    let f1 = f64::from(fb[b0 + 1]);
                    let f2 = f64::from(fb[b0 + 2]);
                    let f3 = f64::from(fb[b0 + 3]);
                    dst_f[k] = [f0, f1, f2, f3];
                    dst_wf[k] = [w * f0, w * f1, w * f2, w * f3];
                    let w_word = k >> 6;
                    let w_bit = 1_u64 << (k & 63);
                    if f0 != 0.0 {
                        m0[w_word] |= w_bit;
                    }
                    if f1 != 0.0 {
                        m1[w_word] |= w_bit;
                    }
                    if f2 != 0.0 {
                        m2[w_word] |= w_bit;
                    }
                    if f3 != 0.0 {
                        m3[w_word] |= w_bit;
                    }
                }
                col_nz_mask[b0] = m0;
                col_nz_mask[b0 + 1] = m1;
                col_nz_mask[b0 + 2] = m2;
                col_nz_mask[b0 + 3] = m3;
                hi_col_mask[b0] = m0;
                hi_col_mask[b0 + 1] = m1;
                hi_col_mask[b0 + 2] = m2;
                hi_col_mask[b0 + 3] = m3;
                block_nz_mask[blk] = mask256_or(mask256_or(m0, m1), mask256_or(m2, m3));
            } else {
                let mut hi_mask = [0_u64; 4];
                let tf = [
                    col_max_f[b0] * (1.0 / 65_536.0),
                    col_max_f[b0 + 1] * (1.0 / 65_536.0),
                    col_max_f[b0 + 2] * (1.0 / 65_536.0),
                    col_max_f[b0 + 3] * (1.0 / 65_536.0),
                ];
                let twf = [
                    col_max_wf[b0] * (1.0 / 65_536.0),
                    col_max_wf[b0 + 1] * (1.0 / 65_536.0),
                    col_max_wf[b0 + 2] * (1.0 / 65_536.0),
                    col_max_wf[b0 + 3] * (1.0 / 65_536.0),
                ];
                for (k, &idx) in chunk_indices.iter().enumerate() {
                    let w = batch_w[k];
                    let fb = &features[idx];
                    let w_word = k >> 6;
                    let w_bit = 1_u64 << (k & 63);
                    let mut out_f = [0.0_f64; 4];
                    let mut out_wf = [0.0_f64; 4];
                    for lane in 0..4 {
                        let d = b0 + lane;
                        let fd = f64::from(fb[d]);
                        if fd == 0.0 {
                            continue;
                        }
                        let wfd = w * fd;
                        col_nz_mask[d][w_word] |= w_bit;
                        if col_wide_exp[d]
                            && (col_extreme[d] || fd.abs() < tf[lane] || wfd.abs() < twf[lane])
                        {
                            is_lo[k][d] = true;
                            frame_has_lo[k] = true;
                            any_lo = true;
                        } else {
                            out_f[lane] = fd;
                            out_wf[lane] = wfd;
                            hi_col_mask[d][w_word] |= w_bit;
                            hi_mask[w_word] |= w_bit;
                        }
                    }
                    dst_f[k] = out_f;
                    dst_wf[k] = out_wf;
                }
                block_nz_mask[blk] = hi_mask;
            }
        }

        // Cross-correlation accumulation across active `(d, ch)` columns (`512` slots), using L1D-packed
        // `blk_wf` when span <= 2^16 AND no >= 2^32 cancellation occurred, else exact `Wide192Accum`.
        let y_either_nz = mask256_or(y_nz_mask[0], y_nz_mask[1]);
        if y_either_nz != [0; 4] {
            let y_wide = [
                max_y[0] > 1.0e130 || min_nz_y[0] < 1.0e-130 || max_y[0] > min_nz_y[0] * 65_536.0,
                max_y[1] > 1.0e130 || min_nz_y[1] < 1.0e-130 || max_y[1] > min_nz_y[1] * 65_536.0,
            ];
            for d in 0..256 {
                if !mask256_overlaps(col_nz_mask[d], y_either_nz) {
                    continue;
                }
                let blk_wf_d = &blk_wf[d >> 2][..b_len];
                let ld = d & 3;
                let ch_limit = if y_batch_mono { 1 } else { 2 };
                for ch in 0..ch_limit {
                    if !mask256_overlaps(col_nz_mask[d], y_nz_mask[ch]) {
                        continue;
                    }
                    let fast_ok = !col_wide_exp[d] && !y_wide[ch];
                    let mut used_fast = false;
                    if fast_ok {
                        let mut s = 0.0_f64;
                        for (k, wf4) in blk_wf_d.iter().enumerate() {
                            s += wf4[ld] * batch_y[k][ch];
                        }
                        if s.abs() > col_max_wf[d] * max_y[ch] * (1.0 / 4_294_967_296.0) {
                            wide_c[d][ch].add_f64(s);
                            if y_batch_mono {
                                wide_c[d][1].add_f64(s);
                            }
                            used_fast = true;
                        }
                    }
                    if !used_fast {
                        let mut acc = Wide192Accum::zero();
                        for (k, &idx) in chunk_indices.iter().enumerate() {
                            acc.add_three_way_f64(
                                batch_w[k],
                                f64::from(features[idx][d]),
                                batch_y[k][ch],
                            );
                        }
                        wide_c[d][ch].merge(&acc);
                        if y_batch_mono {
                            wide_c[d][1].merge(&acc);
                        }
                    }
                }
            }
        }

        // [8 x 4] / [4 x 4] L1D-sequential block upper-triangular contraction & fused Wide192Accum merge,
        // recomputing any pair with `>= 2^32` (`~193 dB`) cancellation via exact `w * (f_i * f_j)` products.
        let mut active_bi = [0usize; 64];
        for bj in 0..64 {
            let mask_j = block_nz_mask[bj];
            if mask_j == [0; 4] {
                continue;
            }
            let j0 = bj * 4;
            let cols_j = [
                uplo_u_index(0, j0),
                uplo_u_index(0, j0 + 1),
                uplo_u_index(0, j0 + 2),
                uplo_u_index(0, j0 + 3),
            ];
            let max_wf_j4 = [
                col_max_wf[j0] * (1.0 / 4_294_967_296.0),
                col_max_wf[j0 + 1] * (1.0 / 4_294_967_296.0),
                col_max_wf[j0 + 2] * (1.0 / 4_294_967_296.0),
                col_max_wf[j0 + 3] * (1.0 / 4_294_967_296.0),
            ];
            let f_j = &blk_f[bj][..b_len];
            let wf_j = &blk_wf[bj][..b_len];

            let d_tile = contract_offdiag_tile_4x4(f_j, wf_j);
            for lj in 0..4 {
                let j = j0 + lj;
                let mask_hj = hi_col_mask[j];
                if mask_hj == [0; 4] {
                    continue;
                }
                let col_j = cols_j[lj];
                for li in 0..lj {
                    let i = j0 + li;
                    let v = d_tile[lj][li];
                    if v.abs() > col_max_f[i] * max_wf_j4[lj] {
                        wide_g[col_j + i].add_f64(v);
                    } else if col_max_f[i] > 0.0 && mask256_overlaps(hi_col_mask[i], mask_hj) {
                        for k in 0..b_len {
                            let prod_ff = f_j[k][li] * f_j[k][lj];
                            if prod_ff != 0.0 {
                                wide_g[col_j + i].add_product(batch_w[k], prod_ff);
                            }
                        }
                    }
                }
                let dv = d_tile[lj][lj];
                if w_min >= 0.0 {
                    if dv != 0.0 {
                        wide_g[col_j + j].add_f64(dv);
                    }
                } else if dv.abs() > col_max_f[j] * max_wf_j4[lj] {
                    wide_g[col_j + j].add_f64(dv);
                } else {
                    for k in 0..b_len {
                        let prod_ff = f_j[k][lj] * f_j[k][lj];
                        if prod_ff != 0.0 {
                            wide_g[col_j + j].add_product(batch_w[k], prod_ff);
                        }
                    }
                }
            }

            let mut n_bi = 0usize;
            for bi in 0..bj {
                if mask256_overlaps(block_nz_mask[bi], mask_j) {
                    active_bi[n_bi] = bi;
                    n_bi += 1;
                }
            }
            let mut merge_tile = |bi: usize, tile: &[[f64; 4]; 4]| {
                let i0 = bi * 4;
                let f_i = &blk_f[bi][..b_len];
                for lj in 0..4 {
                    let j = j0 + lj;
                    let mask_hj = hi_col_mask[j];
                    if mask_hj == [0; 4] {
                        continue;
                    }
                    let col_j = cols_j[lj];
                    let mwf = max_wf_j4[lj];
                    for li in 0..4 {
                        let i = i0 + li;
                        let v = tile[lj][li];
                        if v.abs() > col_max_f[i] * mwf {
                            wide_g[col_j + i].add_f64(v);
                        } else if col_max_f[i] > 0.0 && mask256_overlaps(hi_col_mask[i], mask_hj) {
                            for k in 0..b_len {
                                let prod_ff = f_i[k][li] * f_j[k][lj];
                                if prod_ff != 0.0 {
                                    wide_g[col_j + i].add_product(batch_w[k], prod_ff);
                                }
                            }
                        }
                    }
                }
            };
            let (bi_pairs, bi_rem) = active_bi[..n_bi].as_chunks::<2>();
            for &[bi0, bi1] in bi_pairs {
                let (t0, t1) =
                    contract_offdiag_tile_8x4(&blk_f[bi0][..b_len], &blk_f[bi1][..b_len], wf_j);
                merge_tile(bi0, &t0);
                merge_tile(bi1, &t1);
            }
            for &bi in bi_rem {
                let t = contract_offdiag_tile_4x4(&blk_f[bi][..b_len], wf_j);
                merge_tile(bi, &t);
            }
        }

        // Exact 192-bit Wide192Accum accumulation for wide-exponent `lo` outliers (`48`-bit exact
        // `f32 * f32` multiplied by `w`, immune to intermediate `w * fd` rounding or underflow):
        // w * f_i * f_j == w * f_{i,hi} * f_{j,hi} + w * f_{i,lo} * f_j + w * f_{i,hi} * f_{j,lo}.
        if any_lo {
            for (k, &idx) in chunk_indices.iter().enumerate() {
                if !frame_has_lo[k] {
                    continue;
                }
                let w = batch_w[k];
                let fb = &features[idx];
                let lo_row = &mut is_lo[k];
                for d in 0..256 {
                    if !lo_row[d] {
                        continue;
                    }
                    let fd = f64::from(fb[d]);
                    if fd != 0.0 {
                        let col_d = uplo_u_index(0, d);
                        for (i, &fi_f32) in fb[..=d].iter().enumerate() {
                            if fi_f32 != 0.0 {
                                wide_g[col_d + i].add_product(w, f64::from(fi_f32) * fd);
                            }
                        }
                        for j in (d + 1)..256 {
                            if !lo_row[j] {
                                let fj_f32 = fb[j];
                                if fj_f32 != 0.0 {
                                    wide_g[uplo_u_index(d, j)]
                                        .add_product(w, fd * f64::from(fj_f32));
                                }
                            }
                        }
                    }
                }
                lo_row.fill(false);
            }
        }
    }

    let out_g: Vec<f64> = wide_g.into_iter().map(Wide192Accum::to_f64).collect();
    let out_c: Vec<[f64; 2]> = wide_c
        .into_iter()
        .map(|[l, r]| [l.to_f64(), r.to_f64()])
        .collect();
    (out_g, out_c)
}

pub(crate) use super::extract_rank4_svd_modes;

#[cfg(test)]
mod tests {
    use super::*;

    /// `verify[train.gram.wide192]`, `verify[train.gram.accuracy-contract]`
    #[test]
    fn test_wide192_cancellation_immunity_and_tiled_equivalence() {
        let mut acc = Wide192Accum::zero();
        let tiny = 2.0_f64.powi(-30);
        let ultra_tiny = 2.0_f64.powi(-90);
        acc.add_f64(1.0e8);
        acc.add_f64(tiny);
        acc.add_f64(ultra_tiny);
        acc.add_f64(-1.0e8);
        acc.add_f64(-tiny);
        assert_eq!(acc.to_f64().to_bits(), ultra_tiny.to_bits());
        assert_eq!(acc.limbs()[0], ultra_tiny);

        let mut other = Wide192Accum::zero();
        other.add_product(1.0 + 2.0_f64.powi(-28), 1.0 - 2.0_f64.powi(-28));
        other.add_f64(-1.0);
        assert_eq!(other.to_f64().to_bits(), (-2.0_f64.powi(-56)).to_bits());

        other.add_f64(f64::NAN);
        other.add_product(f64::MAX, f64::MAX);
        let mut merged = Wide192Accum::zero();
        merged.merge(&acc);
        assert_eq!(merged.to_f64().to_bits(), ultra_tiny.to_bits());
        assert_eq!(Wide192Accum::zero().to_f64().to_bits(), 0.0_f64.to_bits());

        // Verify 256-D [4 x 4] tiled contraction matches direct per-sample Wide192Accum to < 1e-14
        let n_frames = 140usize;
        let mut feats = vec![[0.0_f32; 256]; n_frames];
        let mut targets = vec![[0.0_f64; 2]; n_frames];
        let mut weights = vec![1.0_f64; n_frames];
        for (t, (fb, y)) in feats.iter_mut().zip(&mut targets).enumerate() {
            weights[t] = 0.5 + 0.01 * ((t % 17) as f64);
            for (d, slot) in fb.iter_mut().enumerate() {
                *slot = (((t * 13 + d * 7) as f32) * 0.037).sin() * 0.5;
            }
            y[0] = ((t as f64) * 0.11).cos();
            y[1] = ((t as f64) * 0.07).sin();
        }
        let (tiled_g, tiled_c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);

        let mut ref_g = vec![Wide192Accum::zero(); uplo_u_packed_len(256)];
        let mut ref_c = vec![[Wide192Accum::zero(); 2]; 256];
        for ((fb, &y), &w) in feats.iter().zip(&targets).zip(&weights) {
            let row: Vec<f64> = fb.iter().map(|&v| f64::from(v)).collect();
            accumulate_uplo_u_wide192(&mut ref_g, &mut ref_c, &row, y, w);
        }
        for (a, b) in tiled_g.iter().zip(&ref_g) {
            let r = b.to_f64();
            assert!((a - r).abs() <= 1.0e-13 * r.abs().max(1.0));
        }
        for (a, b) in tiled_c.iter().zip(&ref_c) {
            assert!((a[0] - b[0].to_f64()).abs() <= 1.0e-13);
            assert!((a[1] - b[1].to_f64()).abs() <= 1.0e-13);
        }
    }
}
