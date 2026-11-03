//! Fixed-width control-interaction reference, not a production control bus.

use core::cmp::Ordering;

use dnam_engine::FtzDazGuard;
use proptest::prelude::*;

const SUM_WORDS: usize = 9;
const SQUARE_WORDS: usize = 18;
const MAX_CONTRIBUTORS: usize = 255;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Magnitude<const N: usize> {
    words: [u32; N],
}

impl<const N: usize> Magnitude<N> {
    const fn zero() -> Self {
        Self { words: [0; N] }
    }

    fn compare(&self, other: &Self) -> Ordering {
        self.words.iter().rev().cmp(other.words.iter().rev())
    }

    fn add(&mut self, other: &Self) {
        let mut carry = 0_u64;
        for (word, added) in self.words.iter_mut().zip(other.words) {
            let total = u64::from(*word) + u64::from(added) + carry;
            *word = total as u32;
            carry = total >> 32;
        }
        assert_eq!(carry, 0, "integer width exceeded");
    }

    fn subtract(&mut self, other: &Self) {
        let mut borrow = false;
        for (word, removed) in self.words.iter_mut().zip(other.words) {
            let (first, first_borrow) = word.overflowing_sub(removed);
            let (result, second_borrow) = first.overflowing_sub(u32::from(borrow));
            *word = result;
            borrow = first_borrow || second_borrow;
        }
        assert!(!borrow, "unsigned subtraction underflow");
    }

    fn shift_right_one_exact(&mut self) {
        let mut carry = 0_u32;
        for word in self.words.iter_mut().rev() {
            let next_carry = *word & 1;
            *word = (*word >> 1) | (carry << 31);
            carry = next_carry;
        }
        assert_eq!(carry, 0, "integer is not even");
    }

    fn bit(&self, position: usize) -> bool {
        let word = position / 32;
        word < N && (self.words[word] & (1_u32 << (position % 32))) != 0
    }

    fn any_bits_below(&self, limit: usize) -> bool {
        let full_words = limit / 32;
        if self.words[..full_words.min(N)]
            .iter()
            .any(|word| *word != 0)
        {
            return true;
        }
        let remaining_bits = limit % 32;
        full_words < N
            && remaining_bits != 0
            && (self.words[full_words] & ((1_u32 << remaining_bits) - 1)) != 0
    }

    fn highest_bit(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .rev()
            .find(|(_, word)| **word != 0)
            .map(|(index, word)| index * 32 + 31 - word.leading_zeros() as usize)
    }

    fn rounded_shift(&self, shift: usize) -> u32 {
        let word = shift / 32;
        let low = u64::from(self.words[word]);
        let high = self
            .words
            .get(word + 1)
            .map_or(0, |value| u64::from(*value));
        let top = (((low | (high << 32)) >> (shift % 32)) & 0x00ff_ffff) as u32;
        if shift == 0 {
            return top;
        }
        let guard = self.bit(shift - 1);
        let sticky = self.any_bits_below(shift - 1);
        top + u32::from(guard && (sticky || top & 1 != 0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SignedMagnitude<const N: usize> {
    negative: bool,
    magnitude: Magnitude<N>,
}

type SignedSum = SignedMagnitude<SUM_WORDS>;
type SignedInteraction = SignedMagnitude<SQUARE_WORDS>;

impl<const N: usize> SignedMagnitude<N> {
    const fn zero() -> Self {
        Self {
            negative: false,
            magnitude: Magnitude::zero(),
        }
    }

    fn add(&mut self, negative: bool, magnitude: Magnitude<N>) {
        if magnitude.highest_bit().is_none() {
            return;
        }
        if self.negative == negative {
            self.magnitude.add(&magnitude);
        } else {
            match self.magnitude.compare(&magnitude) {
                Ordering::Greater => self.magnitude.subtract(&magnitude),
                Ordering::Equal => *self = Self::zero(),
                Ordering::Less => {
                    let mut difference = magnitude;
                    difference.subtract(&self.magnitude);
                    self.magnitude = difference;
                    self.negative = negative;
                }
            }
        }
    }
}

impl SignedMagnitude<SUM_WORDS> {
    fn rounded_f32_bits(&self) -> Result<u32, ReferenceError> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Ok(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 23 {
            return Ok(sign | self.magnitude.words[0]);
        }
        let mut rounded = self.magnitude.rounded_shift(highest - 23);
        if rounded == 1 << 24 {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = highest as i32 - 149;
        if exponent > 127 {
            return Err(ReferenceError::Overflow);
        }
        Ok(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

fn decode_f32_parts(bits: u32) -> Option<(bool, u32, usize)> {
    let negative = bits >> 31 != 0;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return None;
    }
    let (significand, shift) = if exponent == 0 {
        (fraction, 0)
    } else {
        ((1 << 23) | fraction, exponent - 1)
    };
    Some((negative, significand, shift as usize))
}

fn decode_f32(bits: u32) -> Option<(bool, Magnitude<SUM_WORDS>)> {
    let (negative, significand, shift) = decode_f32_parts(bits)?;
    let mut magnitude = Magnitude::zero();
    let index = shift / 32;
    let shifted = u64::from(significand) << (shift % 32);
    magnitude.words[index] = shifted as u32;
    magnitude.words[index + 1] = (shifted >> 32) as u32;
    Some((negative, magnitude))
}

fn square(value: &Magnitude<SUM_WORDS>) -> Magnitude<SQUARE_WORDS> {
    let mut result = Magnitude::zero();
    for (left_index, left) in value.words.iter().enumerate() {
        let mut carry = 0_u64;
        for (right_index, right) in value.words.iter().enumerate() {
            let index = left_index + right_index;
            let total =
                u64::from(*left) * u64::from(*right) + u64::from(result.words[index]) + carry;
            result.words[index] = total as u32;
            carry = total >> 32;
        }
        assert_eq!(result.words[left_index + SUM_WORDS], 0);
        result.words[left_index + SUM_WORDS] = carry as u32;
    }
    result
}

fn sparse_product(
    value: &Magnitude<SUM_WORDS>,
    significand: u32,
    shift: usize,
) -> Magnitude<SQUARE_WORDS> {
    let mut product = [0_u32; SUM_WORDS + 1];
    let mut carry = 0_u64;
    for (index, word) in value.words.iter().enumerate() {
        let total = u64::from(*word) * u64::from(significand) + carry;
        product[index] = total as u32;
        carry = total >> 32;
    }
    product[SUM_WORDS] = carry as u32;
    let mut result = Magnitude::zero();
    for (index, word) in product.iter().enumerate() {
        let destination = index + shift / 32;
        let shifted = u64::from(*word) << (shift % 32);
        result.words[destination] |= shifted as u32;
        result.words[destination + 1] |= (shifted >> 32) as u32;
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReferenceError {
    InvalidIndex,
    NonFinite,
    Overflow,
}

fn round_interaction_bits(
    negative: bool,
    magnitude: &Magnitude<SQUARE_WORDS>,
) -> Result<u32, ReferenceError> {
    let Some(mut highest) = magnitude.highest_bit() else {
        return Ok(0);
    };
    let sign = u32::from(negative) << 31;
    if highest < 173 {
        return Ok(sign | magnitude.rounded_shift(150));
    }
    let mut rounded = magnitude.rounded_shift(highest - 23);
    if rounded == 1 << 24 {
        rounded >>= 1;
        highest += 1;
    }
    let exponent = highest as i32 - 299;
    if exponent > 127 {
        return Err(ReferenceError::Overflow);
    }
    Ok(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IntegerStats {
    count: usize,
    previous: [u32; MAX_CONTRIBUTORS],
    sum: SignedSum,
    squared_sum: Magnitude<SQUARE_WORDS>,
}

impl IntegerStats {
    fn new(count: usize) -> Self {
        assert!(count <= MAX_CONTRIBUTORS);
        Self {
            count,
            previous: [0; MAX_CONTRIBUTORS],
            sum: SignedSum::zero(),
            squared_sum: Magnitude::zero(),
        }
    }

    fn update(&mut self, index: usize, bits: u32) -> Result<(), ReferenceError> {
        if index >= self.count {
            return Err(ReferenceError::InvalidIndex);
        }
        let (new_negative, new_magnitude) = decode_f32(bits).ok_or(ReferenceError::NonFinite)?;
        let (old_negative, old_magnitude) =
            decode_f32(self.previous[index]).expect("stored contributor is finite");
        self.sum.add(!old_negative, old_magnitude);
        self.squared_sum.subtract(&square(&old_magnitude));
        self.sum.add(new_negative, new_magnitude);
        self.squared_sum.add(&square(&new_magnitude));
        self.previous[index] = bits;
        Ok(())
    }

    fn twice_interaction(&self) -> SignedInteraction {
        let squared_linear = square(&self.sum.magnitude);
        let (negative, mut magnitude) = match squared_linear.compare(&self.squared_sum) {
            Ordering::Less => {
                let mut difference = self.squared_sum;
                difference.subtract(&squared_linear);
                (true, difference)
            }
            Ordering::Equal => return SignedInteraction::zero(),
            Ordering::Greater => (false, squared_linear),
        };
        if !negative {
            magnitude.subtract(&self.squared_sum);
        }
        SignedInteraction {
            negative,
            magnitude,
        }
    }

    fn interaction_bits(&self) -> Result<u32, ReferenceError> {
        let interaction = self.twice_interaction();
        round_interaction_bits(interaction.negative, &interaction.magnitude)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SparsePairStats {
    count: usize,
    previous: [u32; MAX_CONTRIBUTORS],
    sum: SignedSum,
    twice_pairs: SignedInteraction,
}

impl SparsePairStats {
    fn new(count: usize) -> Self {
        assert!(count <= MAX_CONTRIBUTORS);
        Self {
            count,
            previous: [0; MAX_CONTRIBUTORS],
            sum: SignedSum::zero(),
            twice_pairs: SignedInteraction::zero(),
        }
    }

    fn update(&mut self, index: usize, bits: u32) -> Result<(), ReferenceError> {
        if index >= self.count {
            return Err(ReferenceError::InvalidIndex);
        }
        let (new_negative, new_significand, new_shift) =
            decode_f32_parts(bits).ok_or(ReferenceError::NonFinite)?;
        let (old_negative, old_significand, old_shift) =
            decode_f32_parts(self.previous[index]).expect("stored contributor is finite");
        let (_, old_magnitude) = decode_f32(self.previous[index]).expect("stored value is finite");
        let (_, new_magnitude) = decode_f32(bits).expect("validated contributor is finite");
        self.sum.add(!old_negative, old_magnitude);
        let rest_negative = self.sum.negative;
        self.twice_pairs.add(
            !(old_negative ^ rest_negative),
            sparse_product(&self.sum.magnitude, old_significand, old_shift + 1),
        );
        self.twice_pairs.add(
            new_negative ^ rest_negative,
            sparse_product(&self.sum.magnitude, new_significand, new_shift + 1),
        );
        self.sum.add(new_negative, new_magnitude);
        self.previous[index] = bits;
        Ok(())
    }

    fn interaction_bits(&self) -> Result<u32, ReferenceError> {
        round_interaction_bits(self.twice_pairs.negative, &self.twice_pairs.magnitude)
    }
}

fn shifted_units(value: &Magnitude<SUM_WORDS>, shift: usize) -> Magnitude<SQUARE_WORDS> {
    let mut shifted = Magnitude::zero();
    for (index, word) in value.words.iter().enumerate() {
        let destination = index + shift / 32;
        let bits = u64::from(*word) << (shift % 32);
        shifted.words[destination] |= bits as u32;
        shifted.words[destination + 1] |= (bits >> 32) as u32;
    }
    shifted
}

fn round_u64_right_even(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift > 64 {
        return 0;
    }
    if shift == 64 {
        return u64::from(value > 1_u64 << 63);
    }
    let truncated = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    truncated + u64::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

fn exact_product_bits(left: u32, right: u32) -> Result<u32, ReferenceError> {
    fn parts(bits: u32) -> Result<(u64, i32), ReferenceError> {
        let exponent = (bits >> 23) & 0xff;
        let fraction = bits & 0x007f_ffff;
        match exponent {
            0xff => Err(ReferenceError::NonFinite),
            0 => Ok((u64::from(fraction), -149)),
            _ => Ok((u64::from((1 << 23) | fraction), exponent as i32 - 150)),
        }
    }
    let (left_mantissa, left_power) = parts(left)?;
    let (right_mantissa, right_power) = parts(right)?;
    let product = left_mantissa * right_mantissa;
    if product == 0 {
        return Ok(0);
    }
    let sign = (left ^ right) & 0x8000_0000;
    let power = left_power + right_power;
    let highest = 63 - product.leading_zeros() as i32;
    let mut exponent = highest + power;
    if exponent < -126 {
        let shift = power + 149;
        let rounded = if shift >= 0 {
            product << shift as u32
        } else {
            round_u64_right_even(product, (-shift) as u32)
        };
        return Ok(sign | rounded as u32);
    }
    if exponent > 127 {
        return Err(ReferenceError::Overflow);
    }
    let shift = highest - 23;
    let mut rounded = if shift >= 0 {
        round_u64_right_even(product, shift as u32)
    } else {
        product << (-shift) as u32
    };
    if rounded == 1 << 24 {
        rounded >>= 1;
        exponent += 1;
    }
    if exponent > 127 {
        return Err(ReferenceError::Overflow);
    }
    Ok(sign | (((exponent + 127) as u32) << 23) | (rounded as u32 & 0x007f_ffff))
}

fn next_finite_encoding(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let bits = (*state >> 32) as u32;
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        (bits & !0x7f80_0000) | 0x7f00_0000
    } else {
        bits
    }
}

#[test]
fn guarded_integer_reference_preserves_small_interactions_and_ties() {
    let _guard = FtzDazGuard::new();
    let cases = [
        (0x3f80_0000, 0x3300_0000, 0x3300_0000),
        (0x3f80_0000, 0x0000_0001, 0x0000_0001),
        (0xbf80_0000, 0x0000_0001, 0x8000_0001),
        (0x3f00_0000, 0x0000_0001, 0x0000_0000),
        (0xbf00_0000, 0x0000_0001, 0x8000_0000),
        (0x3f00_0000, 0x0000_0003, 0x0000_0002),
        (0x3f80_0001, 0x3fc0_0000, 0x3fc0_0002),
        (0x3f80_0000, 0x7f7f_ffff, 0x7f7f_ffff),
    ];
    for (first, second, expected) in cases {
        let mut stats = IntegerStats::new(2);
        stats.update(0, first).expect("finite contributor");
        stats.update(1, second).expect("finite contributor");
        assert_eq!(stats.interaction_bits(), Ok(expected));
    }
}

#[test]
fn guarded_integer_reference_rounds_main_sums_and_cancellation() {
    let _guard = FtzDazGuard::new();
    let cases: &[(&[u32], u32)] = &[
        (&[0x0000_0001, 0x0000_0001], 0x0000_0002),
        (&[0x007f_ffff, 0x0000_0001], 0x0080_0000),
        (&[0x0080_0000, 0x8000_0001], 0x007f_ffff),
        (&[0x3f80_0000, 0x3380_0000], 0x3f80_0000),
        (&[0x3f80_0000, 0x3440_0000], 0x3f80_0002),
        (&[0x7f7f_ffff, 0xff7f_ffff, 0x0000_0001], 0x0000_0001),
        (&[0xff7f_ffff, 0x7f7f_ffff, 0x8000_0001], 0x8000_0001),
    ];
    for (values, expected) in cases {
        let mut stats = IntegerStats::new(values.len());
        for (index, bits) in values.iter().enumerate() {
            stats.update(index, *bits).expect("finite contributor");
        }
        assert_eq!(stats.sum.rounded_f32_bits(), Ok(*expected));
    }
    let mut stats = IntegerStats::new(2);
    stats.update(0, f32::MAX.to_bits()).expect("finite");
    stats.update(1, f32::MAX.to_bits()).expect("finite");
    assert_eq!(stats.sum.rounded_f32_bits(), Err(ReferenceError::Overflow));
}

#[test]
fn guarded_main_sum_rounding_covers_every_subnormal_value() {
    let _guard = FtzDazGuard::new();
    let mut sum = SignedSum::zero();
    for fraction in 0..0x0080_0000_u32 {
        sum.magnitude.words[0] = fraction;
        assert_eq!(sum.rounded_f32_bits(), Ok(fraction));
        if fraction != 0 && fraction % 127 == 0 {
            sum.negative = true;
            assert_eq!(sum.rounded_f32_bits(), Ok(fraction | 0x8000_0000));
            sum.negative = false;
        }
    }
}

#[test]
fn guarded_main_sum_rounding_covers_normal_boundaries_and_overflow() {
    let _guard = FtzDazGuard::new();
    let mut unit = Magnitude::<SUM_WORDS>::zero();
    unit.words[0] = 1;
    for exponent in 2..=254 {
        for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe] {
            let lower = (exponent << 23) | fraction;
            let (_, mut units) = decode_f32(lower).expect("finite");
            let (_, upper_units) = decode_f32(lower + 1).expect("finite");
            units.add(&upper_units);
            units.shift_right_one_exact();
            let expected = lower + (lower & 1);
            let mut sum = SignedSum {
                negative: false,
                magnitude: units,
            };
            assert_eq!(sum.rounded_f32_bits(), Ok(expected));
            sum.negative = true;
            assert_eq!(sum.rounded_f32_bits(), Ok(expected | 0x8000_0000));
            sum.negative = false;
            sum.magnitude.subtract(&unit);
            assert_eq!(sum.rounded_f32_bits(), Ok(lower));
            sum.magnitude = units;
            sum.magnitude.add(&unit);
            assert_eq!(sum.rounded_f32_bits(), Ok(lower + 1));
        }
    }
    let (_, mut maximum) = decode_f32(f32::MAX.to_bits()).expect("finite");
    let mut next_power = Magnitude::<SUM_WORDS>::zero();
    next_power.words[8] = 1 << 21;
    maximum.add(&next_power);
    maximum.shift_right_one_exact();
    let mut sum = SignedSum {
        negative: false,
        magnitude: maximum,
    };
    assert_eq!(sum.rounded_f32_bits(), Err(ReferenceError::Overflow));
    sum.magnitude.subtract(&unit);
    assert_eq!(sum.rounded_f32_bits(), Ok(f32::MAX.to_bits()));
}

#[test]
fn guarded_integer_rounding_covers_every_subnormal_midpoint() {
    let _guard = FtzDazGuard::new();
    for lower in 0..0x0080_0000_u32 {
        let mut midpoint = Magnitude::<SQUARE_WORDS>::zero();
        let shifted = u64::from(2 * lower + 1) << 21;
        midpoint.words[4] = shifted as u32;
        midpoint.words[5] = (shifted >> 32) as u32;
        let expected = lower + (lower & 1);
        assert_eq!(round_interaction_bits(false, &midpoint), Ok(expected));
        if lower % 127 == 0 {
            assert_eq!(
                round_interaction_bits(true, &midpoint),
                Ok(expected | 0x8000_0000)
            );
            let mut below = midpoint;
            let mut above = midpoint;
            let mut unit = Magnitude::<SQUARE_WORDS>::zero();
            unit.words[0] = 1;
            below.subtract(&unit);
            above.add(&unit);
            assert_eq!(round_interaction_bits(false, &below), Ok(lower));
            assert_eq!(round_interaction_bits(false, &above), Ok(lower + 1));
        }
    }
}

#[test]
fn guarded_integer_rounding_covers_normal_boundaries_and_overflow() {
    let _guard = FtzDazGuard::new();
    let mut unit = Magnitude::<SQUARE_WORDS>::zero();
    unit.words[0] = 1;
    for exponent in 1..=254 {
        for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe] {
            let lower = (exponent << 23) | fraction;
            let (_, mut units) = decode_f32(lower).expect("finite");
            let (_, upper_units) = decode_f32(lower + 1).expect("finite");
            units.add(&upper_units);
            let midpoint = shifted_units(&units, 149);
            let expected = lower + (lower & 1);
            assert_eq!(round_interaction_bits(false, &midpoint), Ok(expected));
            assert_eq!(
                round_interaction_bits(true, &midpoint),
                Ok(expected | 0x8000_0000)
            );
            let mut below = midpoint;
            let mut above = midpoint;
            below.subtract(&unit);
            above.add(&unit);
            assert_eq!(round_interaction_bits(false, &below), Ok(lower));
            assert_eq!(round_interaction_bits(false, &above), Ok(lower + 1));
        }
    }
    let (_, mut maximum) = decode_f32(f32::MAX.to_bits()).expect("finite");
    let mut next_power = Magnitude::<SUM_WORDS>::zero();
    next_power.words[8] = 1 << 21;
    maximum.add(&next_power);
    let overflow_midpoint = shifted_units(&maximum, 149);
    let mut below = overflow_midpoint;
    below.subtract(&unit);
    assert_eq!(
        round_interaction_bits(false, &below),
        Ok(f32::MAX.to_bits())
    );
    assert_eq!(
        round_interaction_bits(false, &overflow_midpoint),
        Err(ReferenceError::Overflow)
    );
    assert_eq!(
        round_interaction_bits(true, &overflow_midpoint),
        Err(ReferenceError::Overflow)
    );
}

#[test]
fn guarded_integer_reference_restores_state_and_rejects_bad_inputs() {
    let _guard = FtzDazGuard::new();
    let mut stats = IntegerStats::new(2);
    let mut sparse = SparsePairStats::new(2);
    stats.update(0, 1.0_f32.to_bits()).expect("finite");
    sparse.update(0, 1.0_f32.to_bits()).expect("finite");
    for _ in 0..10_000 {
        for value in [
            5.0_f32 * 2.0_f32.powi(-26),
            5.0_f32 * 2.0_f32.powi(-25),
            0.0,
        ] {
            stats.update(1, value.to_bits()).expect("finite");
            sparse.update(1, value.to_bits()).expect("finite");
        }
    }
    assert_eq!(stats.interaction_bits(), Ok(0));
    assert_eq!(sparse.interaction_bits(), Ok(0));
    assert_eq!(sparse.twice_pairs, stats.twice_interaction());
    let unchanged = stats.clone();
    assert_eq!(
        stats.update(1, f32::INFINITY.to_bits()),
        Err(ReferenceError::NonFinite)
    );
    assert_eq!(
        stats.update(1, f32::NAN.to_bits()),
        Err(ReferenceError::NonFinite)
    );
    assert_eq!(stats.update(2, 0), Err(ReferenceError::InvalidIndex));
    assert_eq!(stats, unchanged);

    stats.update(0, f32::MAX.to_bits()).expect("finite");
    stats.update(1, 2.0_f32.to_bits()).expect("finite");
    assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
}

#[test]
fn maximum_contributor_count_stays_inside_fixed_widths() {
    let _guard = FtzDazGuard::new();
    let mut stats = IntegerStats::new(MAX_CONTRIBUTORS);
    let mut sparse = SparsePairStats::new(MAX_CONTRIBUTORS);
    for index in 0..MAX_CONTRIBUTORS {
        stats.update(index, f32::MAX.to_bits()).expect("finite");
        sparse.update(index, f32::MAX.to_bits()).expect("finite");
    }
    assert!(stats.sum.magnitude.highest_bit().expect("nonzero") < 285);
    assert!(stats.squared_sum.highest_bit().expect("nonzero") < 562);
    assert!(square(&stats.sum.magnitude).highest_bit().expect("nonzero") < 570);
    assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
    assert_eq!(sparse.sum, stats.sum);
    assert_eq!(sparse.twice_pairs, stats.twice_interaction());

    for index in (1..MAX_CONTRIBUTORS).step_by(2) {
        stats.update(index, (-f32::MAX).to_bits()).expect("finite");
        sparse.update(index, (-f32::MAX).to_bits()).expect("finite");
    }
    assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
    assert_eq!(sparse.sum, stats.sum);
    assert_eq!(sparse.twice_pairs, stats.twice_interaction());
    assert_eq!(sparse.interaction_bits(), stats.interaction_bits());
}

#[test]
fn guarded_integer_reference_preserves_cancellation_of_extreme_pairs() {
    let _guard = FtzDazGuard::new();
    let mut stats = IntegerStats::new(3);
    stats.update(0, 0x7f7f_ffff).expect("finite");
    stats.update(1, 0x7f7f_ffff).expect("finite");
    stats.update(2, 0xfeff_ffff).expect("finite");
    assert_eq!(stats.interaction_bits(), Ok(0));
    stats.update(2, 0xfeff_fffe).expect("finite");
    assert_eq!(stats.interaction_bits(), Err(ReferenceError::Overflow));
}

#[test]
fn guarded_integer_reference_matches_independent_full_exponent_products() {
    let _guard = FtzDazGuard::new();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..20_000 {
        let left = next_finite_encoding(&mut state);
        let right = next_finite_encoding(&mut state);
        let mut stats = IntegerStats::new(2);
        let mut sparse = SparsePairStats::new(2);
        stats.update(0, left).expect("finite contributor");
        stats.update(1, right).expect("finite contributor");
        sparse.update(0, left).expect("finite contributor");
        sparse.update(1, right).expect("finite contributor");
        assert_eq!(stats.interaction_bits(), exact_product_bits(left, right));
        assert_eq!(sparse.twice_pairs, stats.twice_interaction());
        assert_eq!(sparse.interaction_bits(), exact_product_bits(left, right));
    }
}

#[test]
fn guarded_sparse_updates_match_wide_statistics_after_full_exponent_histories() {
    let _guard = FtzDazGuard::new();
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let mut stats = IntegerStats::new(MAX_CONTRIBUTORS);
    let mut sparse = SparsePairStats::new(MAX_CONTRIBUTORS);
    for step in 0..20_000 {
        let index = next_finite_encoding(&mut state) as usize % MAX_CONTRIBUTORS;
        let bits = if step % 17 == 0 {
            0
        } else {
            next_finite_encoding(&mut state)
        };
        stats.update(index, bits).expect("finite contributor");
        sparse.update(index, bits).expect("finite contributor");
        assert_eq!(sparse.sum, stats.sum);
        assert_eq!(sparse.twice_pairs, stats.twice_interaction());
        assert_eq!(sparse.interaction_bits(), stats.interaction_bits());
    }
    let unchanged = sparse.clone();
    assert_eq!(
        sparse.update(0, f32::INFINITY.to_bits()),
        Err(ReferenceError::NonFinite)
    );
    assert_eq!(
        sparse.update(0, f32::NAN.to_bits()),
        Err(ReferenceError::NonFinite)
    );
    assert_eq!(
        sparse.update(MAX_CONTRIBUTORS, 0),
        Err(ReferenceError::InvalidIndex)
    );
    assert_eq!(sparse, unchanged);
}

#[test]
fn guarded_sparse_updates_cover_every_finite_exponent_and_sign() {
    let _guard = FtzDazGuard::new();
    for left in [
        0x0000_0001,
        0x0080_0001,
        0x3f80_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x8080_0000,
    ] {
        let mut sparse = SparsePairStats::new(2);
        sparse.update(0, left).expect("finite contributor");
        for exponent in 0..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_ffff] {
                for sign in [0, 0x8000_0000] {
                    let right = sign | (exponent << 23) | fraction;
                    sparse.update(1, right).expect("finite contributor");
                    assert_eq!(sparse.interaction_bits(), exact_product_bits(left, right));
                }
            }
        }
    }
}

proptest! {
    #[test]
    fn integer_reference_matches_exact_small_dyadic_pairs(
        values in prop::collection::vec(-1000_i32..=1000, 0..=16),
        shift in 0_u32..=20,
    ) {
        let _guard = FtzDazGuard::new();
        let mut stats = IntegerStats::new(values.len());
        let mut sparse = SparsePairStats::new(values.len());
        for (index, value) in values.iter().enumerate() {
            let input = (*value as f64 * 2.0_f64.powi(-(shift as i32))) as f32;
            stats.update(index, input.to_bits()).expect("finite contributor");
            sparse.update(index, input.to_bits()).expect("finite contributor");
        }
        let mut exact_pairs = 0_i64;
        let mut exact_sum = 0_i64;
        for (index, left) in values.iter().enumerate() {
            exact_sum += i64::from(*left);
            for right in &values[index + 1..] {
                exact_pairs += i64::from(*left) * i64::from(*right);
            }
        }
        let expected = (exact_pairs as f64 * 2.0_f64.powi(-2 * shift as i32)) as f32;
        let expected_sum = (exact_sum as f64 * 2.0_f64.powi(-(shift as i32))) as f32;
        prop_assert_eq!(stats.interaction_bits(), Ok(expected.to_bits()));
        prop_assert_eq!(stats.sum.rounded_f32_bits(), Ok(expected_sum.to_bits()));
        prop_assert_eq!(sparse.twice_pairs, stats.twice_interaction());
        prop_assert_eq!(sparse.interaction_bits(), Ok(expected.to_bits()));
    }

    #[test]
    fn integer_reference_matches_mixed_exponent_pairs(
        values in prop::collection::vec((-1000_i32..=1000, 0_u32..=10), 0..=16),
    ) {
        let _guard = FtzDazGuard::new();
        let mut stats = IntegerStats::new(values.len());
        let mut integer_units = Vec::with_capacity(values.len());
        for (index, (coefficient, shift)) in values.iter().enumerate() {
            let input = (*coefficient as f64 * 2.0_f64.powi(-(*shift as i32))) as f32;
            stats.update(index, input.to_bits()).expect("finite contributor");
            integer_units.push(i64::from(*coefficient) << (10 - shift));
        }
        let mut exact_pairs = 0_i64;
        let mut exact_sum = 0_i64;
        for (index, left) in integer_units.iter().enumerate() {
            exact_sum += left;
            for right in &integer_units[index + 1..] {
                exact_pairs += left * right;
            }
        }
        let expected = (exact_pairs as f64 * 2.0_f64.powi(-20)) as f32;
        let expected_sum = (exact_sum as f64 * 2.0_f64.powi(-10)) as f32;
        prop_assert_eq!(stats.interaction_bits(), Ok(expected.to_bits()));
        prop_assert_eq!(stats.sum.rounded_f32_bits(), Ok(expected_sum.to_bits()));
    }
}
