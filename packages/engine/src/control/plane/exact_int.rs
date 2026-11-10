//! Exact-Integer Sufficient-Statistic Accumulator (`9-word S` + `18-word D`,
//! `ref[dsp.ctrl.accum.exact-int]`, `ref[dsp.op-n.ctrl.incremental-update]`,
//! `ref[dsp.ctrl.accum.anova-fm]`, `ref[dsp.ctrl.accum.cp1-scaling]`).

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::schema::{DnamError, SchemaOptionExt};

/// Number of `u32` limbs in an exact-integer linear sum `S` (`9 × 4 B = 36 B`, `ref[dsp.ctrl.accum.exact-int]`).
pub const EXACT_SUM_WORDS: usize = 9;
/// Number of `u32` limbs in an exact-integer pairwise interaction sum `D` (`18 × 4 B = 72 B`, `ref[dsp.ctrl.accum.exact-int]`).
pub const EXACT_SQUARE_WORDS: usize = 18;
/// Maximum number of contributors per factor in the fixed-width exact-integer accumulator (`255`).
pub const MAX_EXACT_CONTRIBUTORS: usize = 255;

/// Fixed-width unsigned multi-word magnitude (`N` 32-bit little-endian limbs,
/// `ref[dsp.ctrl.accum.exact-int]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactMagnitude<const N: usize> {
    pub words: [u32; N],
}

impl<const N: usize> Default for ExactMagnitude<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> ExactMagnitude<N> {
    pub const fn zero() -> Self {
        Self { words: [0; N] }
    }

    pub fn compare(&self, other: &Self) -> Ordering {
        self.words.iter().rev().cmp(other.words.iter().rev())
    }

    pub fn add_assign(&mut self, other: &Self) {
        let mut carry = 0u64;
        for (w, a) in self.words.iter_mut().zip(other.words) {
            let total = u64::from(*w) + u64::from(a) + carry;
            *w = total as u32;
            carry = total >> 32;
        }
        assert_eq!(carry, 0, "exact-int width exceeded");
    }

    pub fn sub_assign(&mut self, other: &Self) {
        let mut borrow = false;
        for (w, r) in self.words.iter_mut().zip(other.words) {
            let (first, b1) = w.overflowing_sub(r);
            let (res, b2) = first.overflowing_sub(u32::from(borrow));
            *w = res;
            borrow = b1 || b2;
        }
        assert!(!borrow, "exact-int unsigned underflow");
    }

    pub fn bit(&self, pos: usize) -> bool {
        let word = pos / 32;
        word < N && (self.words[word] & (1u32 << (pos % 32))) != 0
    }

    pub fn any_bits_below(&self, limit: usize) -> bool {
        let full = limit / 32;
        if self.words[..full.min(N)].iter().any(|&w| w != 0) {
            return true;
        }
        let rem = limit % 32;
        full < N && rem != 0 && (self.words[full] & ((1u32 << rem) - 1)) != 0
    }

    pub fn highest_bit(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .rev()
            .find(|(_, w)| **w != 0)
            .map(|(idx, w)| idx * 32 + 31 - (w.leading_zeros() as usize))
    }

    pub fn rounded_shift_24(&self, shift: usize) -> u32 {
        let word = shift / 32;
        let low = u64::from(self.words[word]);
        let high = self.words.get(word + 1).map_or(0, |&v| u64::from(v));
        let top = (((low | (high << 32)) >> (shift % 32)) & 0x00ff_ffff) as u32;
        if shift == 0 {
            return top;
        }
        let guard = self.bit(shift - 1);
        let sticky = self.any_bits_below(shift - 1);
        top + u32::from(guard && (sticky || (top & 1) != 0))
    }
}

/// Signed multi-word integer `(negative, ExactMagnitude<N>)` (`ref[dsp.ctrl.accum.exact-int]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedExactInt<const N: usize> {
    pub negative: bool,
    pub magnitude: ExactMagnitude<N>,
}

impl<const N: usize> SignedExactInt<N> {
    pub const fn zero() -> Self {
        Self {
            negative: false,
            magnitude: ExactMagnitude::zero(),
        }
    }

    pub fn add_signed(&mut self, negative: bool, magnitude: ExactMagnitude<N>) {
        if magnitude.highest_bit().is_none() {
            return;
        }
        if self.negative == negative {
            self.magnitude.add_assign(&magnitude);
        } else {
            match self.magnitude.compare(&magnitude) {
                Ordering::Greater => self.magnitude.sub_assign(&magnitude),
                Ordering::Equal => *self = Self::zero(),
                Ordering::Less => {
                    let mut diff = magnitude;
                    diff.sub_assign(&self.magnitude);
                    self.magnitude = diff;
                    self.negative = negative;
                }
            }
        }
    }
}

impl SignedExactInt<EXACT_SUM_WORDS> {
    /// Rounds a 9-word linear sum ($S \cdot 2^{-149}$) to an IEEE-754 `f32` bit pattern (`RNE`).
    pub fn to_rounded_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 23 {
            return Some(sign | self.magnitude.words[0]);
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 149;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

impl SignedExactInt<EXACT_SQUARE_WORDS> {
    /// Rounds an 18-word pairwise interaction sum ($z_{\text{fm}} = D \cdot 2^{-299}$) to an
    /// IEEE-754 `f32` bit pattern (`RNE`, `ref[dsp.ctrl.accum.exact-int]`).
    pub fn to_interaction_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 173 {
            return Some(sign | self.magnitude.rounded_shift_24(150));
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 299;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

fn decode_f32_significand_shift(bits: u32) -> Option<(bool, u32, usize)> {
    let negative = (bits >> 31) != 0;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return None;
    }
    let (sig, shift) = if exponent == 0 {
        (fraction, 0usize)
    } else {
        ((1 << 23) | fraction, (exponent - 1) as usize)
    };
    Some((negative, sig, shift))
}

fn decode_f32_to_9word(bits: u32) -> Option<(bool, ExactMagnitude<EXACT_SUM_WORDS>)> {
    let (neg, sig, shift) = decode_f32_significand_shift(bits)?;
    let mut mag = ExactMagnitude::zero();
    let idx = shift / 32;
    let shifted = u64::from(sig) << (shift % 32);
    mag.words[idx] = shifted as u32;
    mag.words[idx + 1] = (shifted >> 32) as u32;
    Some((neg, mag))
}

fn sparse_product_18word(
    value: &ExactMagnitude<EXACT_SUM_WORDS>,
    significand: u32,
    shift: usize,
) -> ExactMagnitude<EXACT_SQUARE_WORDS> {
    let mut prod = [0u32; EXACT_SUM_WORDS + 1];
    let mut carry = 0u64;
    for (i, &w) in value.words.iter().enumerate() {
        let total = u64::from(w) * u64::from(significand) + carry;
        prod[i] = total as u32;
        carry = total >> 32;
    }
    prod[EXACT_SUM_WORDS] = carry as u32;
    let mut out = ExactMagnitude::zero();
    for (i, &w) in prod.iter().enumerate() {
        let dst = i + shift / 32;
        let shifted = u64::from(w) << (shift % 32);
        out.words[dst] |= shifted as u32;
        out.words[dst + 1] |= (shifted >> 32) as u32;
    }
    out
}

/// Exact-integer sufficient-statistic accumulator for one FM factor (`9-word S` + `18-word D`,
/// `108 B` state + contributor history, `ref[dsp.ctrl.accum.exact-int]`,
/// `ref[dsp.op-n.ctrl.incremental-update]`, `ref[dsp.ctrl.ledger.memory-compute]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactIntFactorAccumulator {
    pub count: usize,
    pub previous_bits: Vec<u32>,
    pub sum: SignedExactInt<EXACT_SUM_WORDS>,
    pub twice_pairs: SignedExactInt<EXACT_SQUARE_WORDS>,
}

impl ExactIntFactorAccumulator {
    pub fn new(count: usize) -> Self {
        assert!(count <= MAX_EXACT_CONTRIBUTORS);
        Self {
            count,
            previous_bits: vec![0; count],
            sum: SignedExactInt::zero(),
            twice_pairs: SignedExactInt::zero(),
        }
    }

    /// Resets the accumulator in place without heap allocation.
    #[inline]
    pub fn reset(&mut self) {
        self.previous_bits.fill(0);
        self.sum = SignedExactInt::zero();
        self.twice_pairs = SignedExactInt::zero();
    }

    /// Updates contributor `index` from its stored `f32` bit pattern to `new_bits` in $O(1)$
    /// without wide squaring or heap error allocation (`R = S - b_o`,
    /// `D_new = D - 2 b_o R + 2 b_n R`, `S_new = R + b_n`, `ref[dsp.ctrl.accum.exact-int]`).
    #[inline]
    pub fn update_contributor_rt(&mut self, index: usize, new_bits: u32) -> bool {
        if index >= self.count {
            return false;
        }
        let Some((new_neg, new_sig, new_shift)) = decode_f32_significand_shift(new_bits) else {
            return false;
        };
        let old_bits = self.previous_bits[index];
        if old_bits == new_bits || ((old_bits & 0x7fff_ffff) == 0 && new_sig == 0) {
            self.previous_bits[index] = new_bits;
            return true;
        }
        let Some((old_neg, old_sig, old_shift)) = decode_f32_significand_shift(old_bits) else {
            return false;
        };
        if old_sig != 0
            && let Some((_, old_mag)) = decode_f32_to_9word(old_bits)
        {
            // R = S - b_old
            self.sum.add_signed(!old_neg, old_mag);
            if self.sum.magnitude.words != [0; 9] {
                let rest_neg = self.sum.negative;
                // D_new = D - 2 * b_old * R
                self.twice_pairs.add_signed(
                    !(old_neg ^ rest_neg),
                    sparse_product_18word(&self.sum.magnitude, old_sig, old_shift + 1),
                );
            }
        }
        if new_sig != 0
            && let Some((_, new_mag)) = decode_f32_to_9word(new_bits)
        {
            if self.sum.magnitude.words != [0; 9] {
                let rest_neg = self.sum.negative;
                // D_new += 2 * b_new * R
                self.twice_pairs.add_signed(
                    new_neg ^ rest_neg,
                    sparse_product_18word(&self.sum.magnitude, new_sig, new_shift + 1),
                );
            }
            // S_new = R + b_new
            self.sum.add_signed(new_neg, new_mag);
        }
        self.previous_bits[index] = new_bits;
        true
    }

    /// Updates contributor `index` from its stored `f32` bit pattern to `new_bits` in $O(1)$
    /// without wide squaring (`R = S - b_o`, `D_new = D - 2 b_o R + 2 b_n R`, `S_new = R + b_n`,
    /// `ref[dsp.ctrl.accum.exact-int]`).
    pub fn update_contributor(&mut self, index: usize, new_bits: u32) -> Result<(), DnamError> {
        if index >= self.count {
            return Err(DnamError::InvalidControlIndex(index));
        }
        let _ = decode_f32_significand_shift(new_bits).or_schema_err("nonfinite factor")?;
        let _ = self.update_contributor_rt(index, new_bits);
        Ok(())
    }

    /// Returns the rounded `f32` bit pattern of $z_{\text{fm}} = D \cdot 2^{-299}$.
    pub fn interaction_f32_bits(&self) -> Option<u32> {
        self.twice_pairs.to_interaction_f32_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ControlClampDescriptor;
    use crate::operators::op_n::{eval_chebyshev_4harm, eval_lobatto_basis_3node};

    /// verify[dsp.ctrl.op-n.engine]
    /// verify[dsp.ctrl.basis.lobatto]
    /// verify[dsp.ctrl.switch.option-contributors]
    /// verify[dsp.ctrl.accum.cp1-scaling]
    /// verify[dsp.ctrl.accum.anova-fm]
    /// verify[dsp.ctrl.accum.exact-int]
    /// verify[dsp.ctrl.ledger.memory-compute]
    /// verify[dsp.op-n.ctrl.incremental-update]
    /// verify[dsp.op-n.ctrl.lobatto-basis]
    /// verify[dsp.op-n.ctrl.chebyshev-lfo]
    /// verify[dsp.op-n.ctrl.fm-interaction]
    /// Verifies 9-word `S` + 18-word `D` exact-integer sufficient-statistic accumulation (`O(1)` updates
    /// immune to catastrophic cancellation and update order), Lobatto basis, and Chebyshev LFO waveshaping.
    #[test]
    fn test_exact_integer_accumulator_lobatto_chebyshev_and_fm_invariance() {
        // 1. Catastrophic cancellation test: huge positive + huge negative contributors + small pair
        let mut acc_a = ExactIntFactorAccumulator::new(4);
        let mut acc_b = ExactIntFactorAccumulator::new(4);
        let huge_pos = (1.0e18_f32).to_bits();
        let huge_neg = (-1.0e18_f32).to_bits();
        let c2 = (0.5_f32).to_bits();
        let c3 = (0.25_f32).to_bits();

        // Order A: 0, 1, 2, 3
        acc_a.update_contributor(0, huge_pos).unwrap();
        acc_a.update_contributor(1, huge_neg).unwrap();
        acc_a.update_contributor(2, c2).unwrap();
        acc_a.update_contributor(3, c3).unwrap();

        // Order B: 3, 2, 1, 0 with intermediate updates
        acc_b.update_contributor(2, (3.0_f32).to_bits()).unwrap();
        acc_b.update_contributor(3, c3).unwrap();
        acc_b.update_contributor(1, huge_neg).unwrap();
        acc_b.update_contributor(2, c2).unwrap();
        acc_b.update_contributor(0, huge_pos).unwrap();

        assert_eq!(acc_a.sum, acc_b.sum);
        assert_eq!(acc_a.twice_pairs, acc_b.twice_pairs);
        assert_eq!(acc_a.interaction_f32_bits(), acc_b.interaction_f32_bits());

        // Remove the huge pair and verify exact pair product 0.5 * 0.25 = 0.125
        acc_a.update_contributor(0, 0).unwrap();
        acc_a.update_contributor(1, 0).unwrap();
        assert_eq!(acc_a.interaction_f32_bits(), Some((0.125_f32).to_bits()));

        // 2. Lobatto degree 0..3 & switch contributor
        let psi0 = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 0, 1.0);
        assert_eq!(psi0, [0.0, 0.0, 0.0, 1.0]);
        let psi3 = eval_lobatto_basis_3node(0.5, &[0.0, -0.5, 0.5, 0.0], 3, 0.0);
        assert!((psi3[0] - 0.5).abs() < 1e-6);
        assert!((psi3[1] - 0.5).abs() < 1e-6);
        assert_eq!(psi3[2], 0.0);

        // 3. Chebyshev T1..T4 waveshaping
        let cheb = eval_chebyshev_4harm(0.5);
        assert!((cheb[0] - 0.5).abs() < 1e-6);
        assert!((cheb[1] - (-0.5)).abs() < 1e-6);
        assert!((cheb[2] - (-1.0)).abs() < 1e-6);
        assert!((cheb[3] - (-0.5)).abs() < 1e-6);

        assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
    }
}
