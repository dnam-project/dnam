"""Independent Gram-precision examples; these are not accelerator backend tests.

verify[train.gram.wide192]
verify[train.gram.radix256]
verify[train.gram.exponent-aware]
verify[train.gram.group-merge]
verify[train.gram.accuracy-contract]
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import numpy as np
import pytest

_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")
_U32 = struct.Struct("<I")
_U32_PAIR = struct.Struct("<II")
_MAX_DIGIT = 255
_CHUNK_ROWS = 128
_WORD_BASE = 1 << 16
_WORD_COUNT = 56
_BINARY64_WORD_COUNT = 398


def _f32_from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _f32_bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def _exponent_aware_digits(values: list[float]) -> tuple[int, list[list[int]]]:
    parts: list[tuple[int, int, int]] = []
    for value in values:
        bits = _f32_bits(value)
        exponent = (bits >> 23) & 0xFF
        if exponent == 0xFF:
            raise ValueError("non-finite input")
        mantissa = bits & 0x7F_FFFF
        if exponent:
            mantissa |= 1 << 23
        sign = -1 if bits >> 31 else 1
        parts.append((sign, max(exponent, 1), mantissa))

    nonzero_exponents = [exponent for _, exponent, mantissa in parts if mantissa]
    if not nonzero_exponents:
        return 0, [[] for _ in values]
    largest = max(nonzero_exponents)
    smallest = min(nonzero_exponents)
    count = (24 + largest - smallest + 7) // 8
    result: list[list[int]] = []
    for sign, exponent, mantissa in parts:
        row = []
        for position in range(1, count + 1):
            shift = 8 * position - 24 - (largest - exponent)
            if shift >= 8 or shift <= -24:
                digit = 0
            elif shift >= 0:
                digit = (mantissa << shift) & _MAX_DIGIT
            else:
                digit = (mantissa >> -shift) & _MAX_DIGIT
            row.append(sign * digit)
        result.append(row)
    return largest - 126, result


def _candidate_dot(left: list[float], right: list[float]) -> Fraction:
    if len(left) != len(right):
        raise ValueError("different lengths")
    left_scale, left_rows = _exponent_aware_digits(left)
    right_scale, right_rows = _exponent_aware_digits(right)
    left_count = len(left_rows[0]) if left_rows else 0
    right_count = len(right_rows[0]) if right_rows else 0
    result = Fraction()
    for p in range(left_count):
        for q in range(right_count):
            coefficient = 0
            for start in range(0, len(left), _CHUNK_ROWS):
                left_chunk = [row[p] for row in left_rows[start : start + _CHUNK_ROWS]]
                right_chunk = [row[q] for row in right_rows[start : start + _CHUNK_ROWS]]
                exact = sum(a * b for a, b in zip(left_chunk, right_chunk, strict=True))
                product = np.dot(
                    np.asarray(left_chunk, dtype=np.float32),
                    np.asarray(right_chunk, dtype=np.float32),
                )
                assert int(product) == exact
                coefficient += exact
            exponent = left_scale + right_scale - 8 * (p + q + 2)
            result += coefficient * _power_of_two(exponent)
    return result


def _exact_dot(left: list[float], right: list[float]) -> Fraction:
    return sum(
        (Fraction.from_float(a) * Fraction.from_float(b) for a, b in zip(left, right, strict=True)),
        Fraction(),
    )


def _gamma(n: int) -> Fraction:
    return Fraction(n, (1 << 53) - n)


def _floor_binary_exponent(value: Fraction) -> int:
    if not value:
        raise ValueError("zero has no exponent")
    numerator = abs(value.numerator)
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    return exponent


def _round_unbounded_binary(value: Fraction, precision: int = 192) -> Fraction:
    if not value:
        return Fraction()
    sign = -1 if value < 0 else 1
    numerator = abs(value.numerator)
    denominator = value.denominator
    exponent = _floor_binary_exponent(value)
    shift = precision - 1 - exponent
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    significand, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and significand & 1):
        significand += 1
    return sign * significand * _power_of_two(exponent - precision + 1)


def _wide_sum(terms: list[Fraction]) -> Fraction:
    result = Fraction()
    for term in terms:
        result = _round_unbounded_binary(result + term)
    return result


def _wide_range_bounds(
    left: list[float], right: list[float], weights: list[float]
) -> tuple[Fraction, Fraction, Fraction]:
    if len(left) != len(right) or len(left) != len(weights):
        raise ValueError("different lengths")
    if not all(math.isfinite(value) for values in (left, right, weights) for value in values):
        raise ValueError("non-finite input")
    if any(weight < 0 for weight in weights):
        raise ValueError("negative weight")
    n = len(left)
    if n >= 2**53:
        raise ValueError("too many rows")
    maxima = [
        max((abs(Fraction.from_float(value)) for value in values), default=Fraction())
        for values in (left, right, weights)
    ]
    if not n or not all(maxima):
        return Fraction(), Fraction(), Fraction()
    exponent = sum(_floor_binary_exponent(value) + 1 for value in maxima)
    error_bound = 2 * n * (n - 1) * _power_of_two(exponent - 192)
    terms = [
        Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
        for a, b, weight in zip(left, right, weights, strict=True)
    ]
    approximate = _wide_sum(terms)
    return approximate - error_bound, approximate + error_bound, approximate


def _support_refined_wide_bounds(
    left: list[float], right: list[float], weights: list[float]
) -> tuple[Fraction, Fraction, Fraction]:
    lower, upper, approximate = _wide_range_bounds(left, right, weights)
    has_nonzero_term = any(
        a != 0 and b != 0 and weight != 0 for a, b, weight in zip(left, right, weights, strict=True)
    )
    if not has_nonzero_term:
        return Fraction(), Fraction(), Fraction()
    return lower, upper, approximate


def _wide_overflow_status(lower: Fraction, upper: Fraction) -> str:
    if lower > upper:
        raise ValueError("invalid interval")
    threshold = _power_of_two(1024) - _power_of_two(970)
    if -threshold < lower and upper < threshold:
        return "finite"
    if lower >= threshold or upper <= -threshold:
        return "overflow"
    return "ambiguous"


def _possible_underflow(lower: Fraction, upper: Fraction) -> bool:
    if lower > upper:
        raise ValueError("invalid interval")
    threshold = _power_of_two(-1022) - _power_of_two(-1075)
    return (lower != 0 or upper != 0) and upper > -threshold and lower < threshold


def _f32_parts(value: float) -> tuple[int, int, int]:
    if not math.isfinite(value):
        raise ValueError("non-finite input")
    if abs(value) > _f32_from_bits(0x7F7F_FFFF):
        raise ValueError("input is not binary32")
    bits = _f32_bits(value)
    if _f32_from_bits(bits) != value:
        raise ValueError("input is not binary32")
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7F_FFFF
    if exponent:
        mantissa |= 1 << 23
    return (-1 if bits >> 31 else 1), max(exponent, 1), mantissa


def _mantissa_bytes(mantissa: int) -> list[int]:
    return [(mantissa >> (8 * index)) & _MAX_DIGIT for index in range(3)]


def _multiply_mantissas(left: int, right: int) -> list[int]:
    a, b = _mantissa_bytes(left), _mantissa_bytes(right)
    result = []
    carry = 0
    for position in range(5):
        total = carry + sum(
            a[index] * b[position - index] for index in range(3) if 0 <= position - index < 3
        )
        assert total < 2**18
        result.append(total & _MAX_DIGIT)
        carry = total >> 8
    assert carry <= _MAX_DIGIT
    result.append(carry)
    return result


def _shift_bytes(digits: list[int], shift: int, count: int, sign: int) -> list[int]:
    result = [0] * count
    offset, remainder = divmod(shift, 8)
    carry = 0
    for index, digit in enumerate(digits):
        shifted = (digit << remainder) | carry
        result[offset + index] = sign * (shifted & _MAX_DIGIT)
        carry = shifted >> 8
    if carry:
        result[offset + len(digits)] = sign * carry
    return result


def _fixed_binary32_digits(value: float) -> list[int]:
    sign, exponent, mantissa = _f32_parts(value)
    return _shift_bytes(_mantissa_bytes(mantissa), exponent + 2, 35, sign)


def _fixed_weighted_digits(value: float, weight: float) -> list[int]:
    sign, exponent, mantissa = _f32_parts(value)
    weight_sign, weight_exponent, weight_mantissa = _f32_parts(weight)
    if weight_sign < 0 and weight_mantissa:
        raise ValueError("negative weight")
    product = _multiply_mantissas(mantissa, weight_mantissa)
    return _shift_bytes(product, exponent + weight_exponent + 4, 70, sign)


def _add_signed_coefficient(words: list[int], coefficient: int, digit_offset: int) -> None:
    if coefficient == 0:
        return
    sign_before = words[-1] >= _WORD_BASE // 2
    offset, remainder = divmod(8 * digit_offset, 16)
    shifted = abs(coefficient) << remainder
    assert shifted < 2**31
    parts = (shifted & (_WORD_BASE - 1), shifted >> 16)
    carry = 0
    for index in range(offset, len(words)):
        part = parts[index - offset] if index - offset < 2 else 0
        if coefficient < 0:
            total = words[index] - part - carry
            carry = int(total < 0)
        else:
            total = words[index] + part + carry
            carry = total >> 16
        assert -_WORD_BASE <= total < 2 * _WORD_BASE
        words[index] = total & (_WORD_BASE - 1)
        if index > offset and not carry:
            break
    sign_after = words[-1] >= _WORD_BASE // 2
    if (coefficient > 0 and not sign_before and sign_after) or (
        coefficient < 0 and sign_before and not sign_after
    ):
        raise OverflowError("signed accumulator overflow")


def _merge_accumulator_words(left: list[int], right: list[int]) -> list[int]:
    carry = 0
    result = []
    for a, b in zip(left, right, strict=True):
        total = a + b + carry
        result.append(total & (_WORD_BASE - 1))
        carry = total >> 16
    left_sign = left[-1] >= _WORD_BASE // 2
    right_sign = right[-1] >= _WORD_BASE // 2
    result_sign = result[-1] >= _WORD_BASE // 2
    if left_sign == right_sign and result_sign != left_sign:
        raise OverflowError("signed accumulator overflow")
    return result


def _accumulator_value(words: list[int], scale: int = 456) -> Fraction:
    unsigned = sum(word << (16 * index) for index, word in enumerate(words))
    signed = unsigned - (1 << (16 * len(words))) if words[-1] >= _WORD_BASE // 2 else unsigned
    return Fraction(signed, 1 << scale)


def _weighted_digit_accumulator(
    left: list[float], right: list[float], weights: list[float]
) -> list[int]:
    if len(left) != len(right) or len(left) != len(weights):
        raise ValueError("different lengths")
    weighted_rows = [_fixed_weighted_digits(a, w) for a, w in zip(left, weights, strict=True)]
    right_rows = [_fixed_binary32_digits(b) for b in right]
    active_left = [index for index in range(70) if any(row[index] for row in weighted_rows)]
    active_right = [index for index in range(35) if any(row[index] for row in right_rows)]
    words = [0] * _WORD_COUNT
    for p in active_left:
        for q in active_right:
            for start in range(0, len(left), _CHUNK_ROWS):
                a = [row[p] for row in weighted_rows[start : start + _CHUNK_ROWS]]
                b = [row[q] for row in right_rows[start : start + _CHUNK_ROWS]]
                exact = sum(x * y for x, y in zip(a, b, strict=True))
                product = np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
                assert int(product) == exact
                _add_signed_coefficient(words, exact, p + q)
    return words


def _f64_parts(value: float) -> tuple[int, int, list[int]]:
    low, high = _U32_PAIR.unpack(_F64.pack(value))
    exponent = (high >> 20) & 0x7FF
    if exponent == 0x7FF:
        raise ValueError("non-finite input")
    digits = [
        low & _MAX_DIGIT,
        (low >> 8) & _MAX_DIGIT,
        (low >> 16) & _MAX_DIGIT,
        (low >> 24) & _MAX_DIGIT,
        high & _MAX_DIGIT,
        (high >> 8) & _MAX_DIGIT,
        ((high >> 16) & 0x0F) | (0x10 if exponent else 0),
    ]
    return (-1 if high >> 31 else 1), max(exponent, 1), digits


def _multiply_binary64_mantissas(left: list[int], right: list[int]) -> list[int]:
    result = []
    carry = 0
    for position in range(13):
        total = carry + sum(
            left[index] * right[position - index] for index in range(7) if 0 <= position - index < 7
        )
        assert total < 2**19
        result.append(total & _MAX_DIGIT)
        carry = total >> 8
    assert carry <= _MAX_DIGIT
    result.append(carry)
    return result


def _fixed_binary64_digits(value: float) -> list[int]:
    sign, exponent, mantissa = _f64_parts(value)
    return _shift_bytes(mantissa, exponent + 5, 263, sign)


def _fixed_weighted_binary64_digits(value: float, weight: float) -> list[int]:
    sign, exponent, mantissa = _f64_parts(value)
    weight_sign, weight_exponent, weight_mantissa = _f64_parts(weight)
    if weight_sign < 0 and any(weight_mantissa):
        raise ValueError("negative weight")
    product = _multiply_binary64_mantissas(mantissa, weight_mantissa)
    return _shift_bytes(product, exponent + weight_exponent + 2, 525, sign)


def _weighted_binary64_accumulator(
    left: list[float], right: list[float], weights: list[float]
) -> list[int]:
    if len(left) != len(right) or len(left) != len(weights):
        raise ValueError("different lengths")
    weighted_rows = [
        _fixed_weighted_binary64_digits(a, w) for a, w in zip(left, weights, strict=True)
    ]
    right_rows = [_fixed_binary64_digits(b) for b in right]
    active_left = [index for index in range(525) if any(row[index] for row in weighted_rows)]
    active_right = [index for index in range(263) if any(row[index] for row in right_rows)]
    words = [0] * _BINARY64_WORD_COUNT
    for p in active_left:
        for q in active_right:
            for start in range(0, len(left), _CHUNK_ROWS):
                a = [row[p] for row in weighted_rows[start : start + _CHUNK_ROWS]]
                b = [row[q] for row in right_rows[start : start + _CHUNK_ROWS]]
                exact = sum(x * y for x, y in zip(a, b, strict=True))
                product = np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
                assert int(product) == exact
                _add_signed_coefficient(words, exact, p + q)
    return words


def test_three_shared_scale_slices_can_erase_a_cross_correlation() -> None:
    remainder = np.float32(2**-25) / np.float32(2)
    digits = []
    for _ in range(3):
        remainder = np.float32(remainder * np.float32(256))
        digit = np.rint(remainder)
        digits.append(int(digit))
        remainder = np.float32(remainder - digit)
    assert digits == [0, 0, 0]
    exact = _exact_dot([1.0, 2**-25], [0.0, 1.0])
    assert exact == _power_of_two(-25)
    assert abs(exact) > _gamma(2) * abs(exact)


def test_exponent_aware_digits_reconstruct_binary32_values() -> None:
    rng = np.random.default_rng(122)
    patterns = [
        0,
        0x8000_0000,
        1,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0x3F80_0001,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        *[int(bits) for bits in rng.integers(0, 0xFFFF_FFFF, size=1_000, dtype=np.uint32)],
    ]
    values = [_f32_from_bits(bits) for bits in patterns if ((bits >> 23) & 0xFF) != 0xFF]
    scale, digits = _exponent_aware_digits(values)
    assert len(digits[0]) == 35
    for value, row in zip(values, digits, strict=True):
        assert all(abs(digit) <= _MAX_DIGIT for digit in row)
        reconstructed = sum(
            (digit * _power_of_two(scale - 8 * (p + 1)) for p, digit in enumerate(row)),
            Fraction(),
        )
        assert reconstructed == Fraction.from_float(value)

    assert _exponent_aware_digits([0.0, -0.0]) == (0, [[], []])
    with pytest.raises(ValueError, match="non-finite"):
        _exponent_aware_digits([math.inf])


def test_exponent_aware_integer_dot_matches_exact_references() -> None:
    f32_max = _f32_from_bits(0x7F7F_FFFF)
    f32_min = _f32_from_bits(1)
    cases = [
        ([1.0, 2**-25], [0.0, 1.0]),
        ([1.0, 1.0, 2**-25], [1.0, -1.0, 1.0]),
        ([f32_max, f32_min, -f32_max], [f32_min, f32_max, f32_min]),
        ([1.0, 2**-25] * 129, [0.0, 1.0] * 129),
        ([0.0, -0.0], [1.0, -1.0]),
    ]
    for left, right in cases:
        expected = _exact_dot(left, right)
        assert _candidate_dot(left, right) == expected


def test_weighted_fixed_digits_preserve_binary32_products() -> None:
    f32_max = _f32_from_bits(0x7F7F_FFFF)
    f32_min = _f32_from_bits(1)
    values = [0.0, -0.0, f32_min, -f32_min, 1.0, -1.0, f32_max, -f32_max]
    weights = [0.0, -0.0, f32_min, _f32_from_bits(_f32_bits(0.1)), 1.0, f32_max]
    for value in values:
        digits = _fixed_binary32_digits(value)
        actual = sum(digit << (8 * index) for index, digit in enumerate(digits))
        assert Fraction(actual, 1 << 152) == Fraction.from_float(value)
        for weight in weights:
            digits = _fixed_weighted_digits(value, weight)
            actual = sum(digit << (8 * index) for index, digit in enumerate(digits))
            assert Fraction(actual, 1 << 304) == Fraction.from_float(value) * Fraction.from_float(
                weight
            )
    with pytest.raises(ValueError, match="negative weight"):
        _fixed_weighted_digits(1.0, -1.0)
    with pytest.raises(ValueError, match="non-finite"):
        _fixed_weighted_digits(1.0, float("nan"))
    with pytest.raises(ValueError, match="not binary32"):
        _fixed_weighted_digits(1.0, 0.1)


def test_weighted_digit_accumulator_and_exact_window_merge() -> None:
    f32_max = _f32_from_bits(0x7F7F_FFFF)
    f32_min = _f32_from_bits(1)
    tenth = _f32_from_bits(_f32_bits(0.1))
    cases = [
        ([1.0, 2**-25], [0.0, 1.0], [1.0, 1.0]),
        ([1.0, 1.0, 2**-25], [1.0, -1.0, 1.0], [tenth, tenth, tenth]),
        ([f32_max, f32_min, -f32_max], [f32_min, f32_max, f32_min], [tenth, 1.0, tenth]),
        ([f32_min, f32_max], [f32_min, -f32_max], [f32_min, f32_max]),
        ([1.0, 2**-25] * 129, [0.0, -1.0] * 129, [1.0, tenth] * 129),
        ([], [], []),
    ]
    for left, right, weights in cases:
        terms = [
            Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(w)
            for a, b, w in zip(left, right, weights, strict=True)
        ]
        exact = sum(terms, Fraction())
        whole = _weighted_digit_accumulator(left, right, weights)
        first = _weighted_digit_accumulator(left[:129], right[:129], weights[:129])
        second = _weighted_digit_accumulator(left[129:], right[129:], weights[129:])
        assert _accumulator_value(whole) == exact
        assert _accumulator_value(_merge_accumulator_words(first, second)) == exact
        stored = Fraction.from_float(float(exact))
        if terms:
            assert abs(stored - exact) <= _gamma(len(terms)) * sum(map(abs, terms), Fraction())
        else:
            assert stored == 0

    assert 384 + 53 + 456 == 893
    assert 16 * _WORD_COUNT - 1 > 893


def test_weighted_digit_accumulator_on_random_binary32_encodings() -> None:
    rng = np.random.default_rng(128)
    left_bits = rng.integers(0, 0x7F80_0000, 260, dtype=np.uint32)
    right_bits = rng.integers(0, 0x7F80_0000, 260, dtype=np.uint32)
    weight_bits = rng.integers(0, 0x7F80_0000, 260, dtype=np.uint32)
    left_bits[::2] |= np.uint32(0x8000_0000)
    right_bits[::3] |= np.uint32(0x8000_0000)
    left = [float(value) for value in left_bits.view(np.float32)]
    right = [float(value) for value in right_bits.view(np.float32)]
    weights = [float(value) for value in weight_bits.view(np.float32)]
    terms = [
        Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(w)
        for a, b, w in zip(left, right, weights, strict=True)
    ]
    exact = sum(terms, Fraction())
    assert _accumulator_value(_weighted_digit_accumulator(left, right, weights)) == exact
    assert abs(Fraction.from_float(float(exact)) - exact) <= _gamma(len(terms)) * sum(
        map(abs, terms), Fraction()
    )


def test_single_rounded_group_merge_meets_binary32_contraction_bound() -> None:
    tenth = _f32_from_bits(_f32_bits(0.1))
    terms = [
        Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(w)
        for a, b, w in zip(
            [1.0, 1.0, 2**-25, -(2**-70)] * 33,
            [1.0, -1.0, 1.0, 2**-70] * 33,
            [tenth, tenth, 1.0, 1.0] * 33,
            strict=True,
        )
    ]
    expected = sum(terms, Fraction())
    absolute_sum = sum(map(abs, terms), Fraction())
    for group_size in (1, 3, 17, 128, len(terms)):
        grouped = [
            sum(terms[start : start + group_size], Fraction())
            for start in range(0, len(terms), group_size)
        ]
        merged = sum(float(group) for group in grouped)
        assert abs(Fraction.from_float(merged) - expected) <= _gamma(len(terms)) * absolute_sum


def test_posthoc_weighting_introduces_an_additional_rounding() -> None:
    a, b, weight, multiplier = (
        Fraction.from_float(_f32_from_bits(bits))
        for bits in (0x3F85_DE03, 0x3F86_959B, 0x3F80_D0C6, 0x3F8C_F21B)
    )
    exact_group = a * b * weight
    expected = exact_group * multiplier
    actual = Fraction.from_float(float(exact_group) * float(multiplier))
    error = abs(actual - expected)
    assert error > _gamma(1) * abs(expected)
    assert error <= _gamma(2) * abs(expected)


def test_bounded_posthoc_assembly_of_preweighted_groups() -> None:
    tenth = _f32_from_bits(_f32_bits(0.1))
    inputs = [
        [(1.0, 1.0, tenth), (2**-25, -1.0, 1.0)],
        [(1.0, -1.0, tenth), (-(2**-70), 2**-70, 1.0)],
        [(1.0, 1.0, 1.0)],
    ]
    multipliers = [0.125, -0.75, _f32_from_bits(0x3F8C_F21B)]
    groups = [
        [
            Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
            for a, b, weight in rows
        ]
        for rows in inputs
    ]
    expected = sum(
        (
            Fraction.from_float(factor) * sum(group, Fraction())
            for factor, group in zip(multipliers, groups, strict=True)
        ),
        Fraction(),
    )
    absolute_sum = sum(
        (
            abs(Fraction.from_float(factor)) * sum(map(abs, group), Fraction())
            for factor, group in zip(multipliers, groups, strict=True)
        ),
        Fraction(),
    )
    assembled = sum(
        float(sum(group, Fraction())) * factor
        for factor, group in zip(multipliers, groups, strict=True)
    )
    assert abs(Fraction.from_float(assembled) - expected) <= _gamma(len(groups) + 1) * absolute_sum


def test_accumulator_word_overflow_is_reported() -> None:
    largest = [_WORD_BASE - 1] * (_WORD_COUNT - 1) + [_WORD_BASE // 2 - 1]
    with pytest.raises(OverflowError, match="overflow"):
        _add_signed_coefficient(largest.copy(), 1, 0)
    with pytest.raises(OverflowError, match="overflow"):
        _merge_accumulator_words(largest, [1] + [0] * (_WORD_COUNT - 1))


def test_f64_rounded_exact_gram_can_still_lose_rank() -> None:
    epsilon = Fraction(1, 1 << 27)
    exact_diagonal = Fraction(1) + epsilon * epsilon
    assert exact_diagonal - 1 > 0
    assert float(exact_diagonal) == 1.0
    rounded_determinant = float(exact_diagonal) * 1.0 - 1.0 * 1.0
    assert rounded_determinant == 0.0
    assert exact_diagonal - 1 <= _gamma(2) * exact_diagonal


def test_binary64_digits_preserve_full_range_without_narrowing() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    values = [0.0, -0.0, minimum, -minimum, 0.1, -0.1, 1.0, maximum, -maximum]
    weights = [0.0, -0.0, minimum, 0.1, 1.0, maximum]
    for value in values:
        digits = _fixed_binary64_digits(value)
        integer = sum(digit << (8 * index) for index, digit in enumerate(digits))
        assert Fraction(integer, 1 << 1080) == Fraction.from_float(value)
        for weight in weights:
            digits = _fixed_weighted_binary64_digits(value, weight)
            integer = sum(digit << (8 * index) for index, digit in enumerate(digits))
            assert Fraction(integer, 1 << 2152) == (
                Fraction.from_float(value) * Fraction.from_float(weight)
            )
    with pytest.raises(ValueError, match="negative weight"):
        _fixed_weighted_binary64_digits(1.0, -1.0)
    with pytest.raises(ValueError, match="non-finite"):
        _fixed_weighted_binary64_digits(1.0, math.nan)


def test_binary64_weighted_accumulator_and_underflow_contract() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    cases = [
        ([1.0], [1.0], [0.1]),
        ([minimum], [0.5], [1.0]),
        ([minimum, minimum, minimum], [minimum] * 3, [minimum] * 3),
        ([maximum, maximum], [minimum, -minimum], [1.0, 1.0]),
        ([maximum, minimum], [minimum, maximum], [0.1, 0.1]),
        ([1.0, -1.0, 2**-53], [1.0, 1.0, 1.0], [0.1, 0.1, 1.0]),
    ]
    half_subnormal = Fraction(1, 1 << 1075)
    for left, right, weights in cases:
        terms = [
            Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
            for a, b, weight in zip(left, right, weights, strict=True)
        ]
        exact = sum(terms, Fraction())
        actual = _accumulator_value(_weighted_binary64_accumulator(left, right, weights), 3232)
        assert actual == exact
        error = abs(Fraction.from_float(float(exact)) - exact)
        assert error <= _gamma(len(terms)) * sum(map(abs, terms), Fraction()) + half_subnormal

    exact_tie = Fraction.from_float(minimum) * Fraction(1, 2)
    assert float(exact_tie) == 0
    assert abs(exact_tie) > _gamma(1) * abs(exact_tie)
    assert 3072 + 53 + 3232 == 6357
    assert 16 * _BINARY64_WORD_COUNT - 1 > 6357


def test_binary64_rounded_groups_can_lose_representable_subnormal_sum() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    group = _weighted_binary64_accumulator([minimum], [0.5], [1.0])
    exact_group = _accumulator_value(group, 3232)
    exact = 2 * exact_group
    rounded_groups = float(exact_group) + float(exact_group)
    exact_merged = _accumulator_value(_merge_accumulator_words(group, group), 3232)
    half_subnormal = _power_of_two(-1075)
    assert exact_group == half_subnormal
    assert rounded_groups == 0.0
    assert float(exact_merged) == minimum
    assert abs(Fraction.from_float(rounded_groups) - exact) > _gamma(2) * exact + half_subnormal


def test_binary64_rounded_group_merge_obeys_its_weaker_bound() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    half_subnormal = _power_of_two(-1075)
    groups = [
        Fraction.from_float(minimum) / 2,
        Fraction.from_float(minimum) / 2,
        Fraction.from_float(0.1) * Fraction.from_float(1.0 + 2**-52),
        -Fraction.from_float(0.1),
    ]
    for count in range(1, len(groups) + 1):
        selected = groups[:count]
        result = sum(float(group) for group in selected)
        exact = sum(selected, Fraction())
        absolute_sum = sum(map(abs, selected), Fraction())
        bound = _gamma(count) * absolute_sum + count * (1 + _gamma(count - 1)) * half_subnormal
        assert abs(Fraction.from_float(result) - exact) <= bound

    largest_subnormal = float.fromhex("0x0.fffffffffffffp-1022")
    for left in [minimum, -minimum, largest_subnormal, -largest_subnormal]:
        for right in [minimum, -minimum, largest_subnormal, -largest_subnormal]:
            exact = Fraction.from_float(left) + Fraction.from_float(right)
            if abs(exact) < _power_of_two(-1022):
                assert Fraction.from_float(left + right) == exact


def test_binary64_internal_group_overflow_can_cancel_before_final_rounding() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    positive = _weighted_binary64_accumulator([maximum], [2.0], [1.0])
    negative = _weighted_binary64_accumulator([maximum], [-2.0], [1.0])
    with pytest.raises(OverflowError):
        float(_accumulator_value(positive, 3232))
    assert _accumulator_value(_merge_accumulator_words(positive, negative), 3232) == 0


def test_wide_significand_rounds_to_nearest_even_without_exponent_clipping() -> None:
    assert _round_unbounded_binary(Fraction(17, 16), 4) == 1
    assert _round_unbounded_binary(Fraction(19, 16), 4) == Fraction(5, 4)
    assert _round_unbounded_binary(Fraction(-19, 16), 4) == Fraction(-5, 4)
    assert _round_unbounded_binary(_power_of_two(-3222)) == _power_of_two(-3222)
    assert _round_unbounded_binary(_power_of_two(3071)) == _power_of_two(3071)
    with pytest.raises(ValueError, match="zero"):
        _floor_binary_exponent(Fraction())


def test_wide_binary64_accumulator_satisfies_componentwise_bound() -> None:
    minimum = float.fromhex("0x0.0000000000001p-1022")
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    cases = [
        ([1.0], [1.0 + 2**-52], [0.1]),
        ([minimum, minimum], [0.5, 0.5], [1.0, 1.0]),
        ([maximum, maximum, 1.0], [minimum, -minimum, 1.0], [0.1, 0.1, 0.1]),
        ([1.0, -1.0, 2**-53], [1.0, 1.0, 1.0], [0.1, 0.1, 1.0]),
        ([maximum, minimum], [minimum, maximum], [maximum, maximum]),
    ]
    half_subnormal = _power_of_two(-1075)
    for left, right, weights in cases:
        terms = [
            Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
            for a, b, weight in zip(left, right, weights, strict=True)
        ]
        assert all(_round_unbounded_binary(term) == term for term in terms)
        exact = sum(terms, Fraction())
        absolute_sum = sum(map(abs, terms), Fraction())
        approximate = _wide_sum(terms)
        steps = len(terms) - 1
        wide_gamma = Fraction(steps, (1 << 192) - steps)
        assert abs(approximate - exact) <= wide_gamma * absolute_sum
        rounded = Fraction.from_float(float(approximate))
        assert abs(rounded - exact) <= _gamma(len(terms)) * absolute_sum + half_subnormal

        split = len(terms) // 2
        groups = [_wide_sum(terms[:split]), _wide_sum(terms[split:])]
        merged = _round_unbounded_binary(groups[0] + groups[1])
        assert abs(merged - exact) <= wide_gamma * absolute_sum


def test_wide_accumulator_requires_independent_overflow_classification() -> None:
    huge = Fraction.from_float(float.fromhex("0x1.0p+1023"))
    terms = [huge * huge, huge * 4, -(huge * huge)]
    exact = sum(terms, Fraction())
    approximate = _wide_sum(terms)
    assert approximate == 0
    assert exact == _power_of_two(1025)
    with pytest.raises(OverflowError):
        float(exact)
    absolute_sum = sum(map(abs, terms), Fraction())
    wide_gamma = Fraction(2, (1 << 192) - 2)
    assert abs(approximate - exact) <= wide_gamma * absolute_sum


def test_binary32_weighted_contraction_has_binary64_range_headroom() -> None:
    minimum = Fraction.from_float(_f32_from_bits(1))
    maximum = Fraction.from_float(_f32_from_bits(0x7F7F_FFFF))
    assert minimum == _power_of_two(-149)
    assert minimum**3 == _power_of_two(-447)
    assert float(minimum**3).hex() == "0x1.0000000000000p-447"
    assert maximum < _power_of_two(128)
    assert ((1 << 53) - 1) * maximum**3 < _power_of_two(437)
    assert math.isfinite(float(((1 << 53) - 1) * maximum**3))


def test_binary32_cancellation_can_lose_a_normal_binary64_result() -> None:
    minimum = Fraction.from_float(_f32_from_bits(1))
    maximum = Fraction.from_float(_f32_from_bits(0x7F7F_FFFF))
    terms = [maximum**3, minimum**3, -(maximum**3)]
    exact = sum(terms, Fraction())
    approximate = _wide_sum(terms)
    assert exact == _power_of_two(-447)
    assert approximate == 0
    assert abs(exact - approximate) <= _gamma(3) * sum(map(abs, terms), Fraction())


def test_binary64_rounding_range_midpoints() -> None:
    maximum = Fraction.from_float(float.fromhex("0x1.fffffffffffffp+1023"))
    overflow_threshold = _power_of_two(1024) - _power_of_two(970)
    assert maximum == _power_of_two(1024) - _power_of_two(971)
    assert float(overflow_threshold - 1) == float(maximum)
    with pytest.raises(OverflowError):
        float(overflow_threshold)
    with pytest.raises(OverflowError):
        float(-overflow_threshold)

    underflow_threshold = _power_of_two(-1022) - _power_of_two(-1075)
    assert float(underflow_threshold) == float.fromhex("0x1p-1022")
    assert float(underflow_threshold - _power_of_two(-1200)) == float.fromhex(
        "0x0.fffffffffffffp-1022"
    )
    assert float(_power_of_two(-1075)) == 0.0
    assert float(_power_of_two(-1075) + _power_of_two(-1200)) == float.fromhex(
        "0x0.0000000000001p-1022"
    )


def test_wide_interval_classifies_or_defers_range_cases() -> None:
    huge = float.fromhex("0x1p+1023")
    low, high, approximate = _wide_range_bounds(
        [huge, huge, -huge], [huge] * 3, [1.0, math.ldexp(1.0, -1021), 1.0]
    )
    assert approximate == 0
    assert low < 0 < high
    assert low <= _power_of_two(1025) <= high
    assert _wide_overflow_status(low, high) == "ambiguous"
    assert _possible_underflow(low, high)

    low, high, _ = _wide_range_bounds([huge], [huge], [math.ldexp(1.0, -1021)])
    assert low == high == _power_of_two(1025)
    assert _wide_overflow_status(low, high) == "overflow"
    assert not _possible_underflow(low, high)

    low, high, _ = _wide_range_bounds([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    assert low < 2 < high
    assert _wide_overflow_status(low, high) == "finite"
    assert not _possible_underflow(low, high)

    low, high, approximate = _wide_range_bounds([huge, 0.0], [0.0, huge], [huge, huge])
    assert approximate == 0
    assert _wide_overflow_status(low, high) == "ambiguous"
    assert _possible_underflow(low, high)

    threshold = _power_of_two(1024) - _power_of_two(970)
    assert _wide_overflow_status(threshold - 1, threshold - 1) == "finite"
    assert _wide_overflow_status(threshold, threshold) == "overflow"
    assert _wide_overflow_status(-threshold, -threshold) == "overflow"
    assert _wide_overflow_status(threshold - 1, threshold) == "ambiguous"
    assert _wide_overflow_status(-threshold, -threshold + 1) == "ambiguous"


def test_support_flags_prove_disjoint_terms_are_exact_zero() -> None:
    huge = float.fromhex("0x1p+1023")
    left, right, weights = [huge, 0.0], [0.0, huge], [huge, huge]
    coarse_low, coarse_high, _ = _wide_range_bounds(left, right, weights)
    assert _wide_overflow_status(coarse_low, coarse_high) == "ambiguous"
    assert _possible_underflow(coarse_low, coarse_high)
    lower, upper, result = _support_refined_wide_bounds(left, right, weights)
    assert lower == upper == result == 0
    assert _wide_overflow_status(lower, upper) == "finite"
    assert not _possible_underflow(lower, upper)
    assert _support_refined_wide_bounds([huge], [huge], [-0.0]) == (0, 0, 0)


def test_support_flags_do_not_prove_cancellation_is_zero() -> None:
    left, right, weights = [1.0, 1.0], [1.0, -1.0], [1.0, 1.0]
    lower, upper, result = _support_refined_wide_bounds(left, right, weights)
    assert result == 0
    assert lower < 0 < upper
    assert _possible_underflow(lower, upper)


def test_wide_interval_underflow_uses_exact_midpoint_and_zero_semantics() -> None:
    threshold = _power_of_two(-1022) - _power_of_two(-1075)
    tiny = _power_of_two(-1200)
    zero = Fraction()
    assert not _possible_underflow(zero, zero)
    assert not _possible_underflow(threshold, threshold)
    assert not _possible_underflow(-threshold, -threshold)
    assert _possible_underflow(threshold - tiny, threshold - tiny)
    assert _possible_underflow(-threshold + tiny, -threshold + tiny)
    assert _possible_underflow(-tiny, tiny)
    assert _possible_underflow(zero, tiny)
    assert _possible_underflow(-threshold, zero)
    with pytest.raises(ValueError, match="invalid interval"):
        _possible_underflow(Fraction(1), Fraction(0))
    with pytest.raises(ValueError, match="invalid interval"):
        _wide_overflow_status(Fraction(1), Fraction(0))


def test_wide_interval_contains_random_exact_binary64_contractions() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(160):
        bits = rng.integers(0, 1 << 64, 18, dtype=np.uint64)
        values = [_F64.unpack(struct.pack("<Q", int(value)))[0] for value in bits]
        if not all(math.isfinite(value) for value in values):
            continue
        left, right, weights = values[:6], values[6:12], list(map(abs, values[12:]))
        low, high, _ = _wide_range_bounds(left, right, weights)
        exact = sum(
            (
                Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
                for a, b, weight in zip(left, right, weights, strict=True)
            ),
            Fraction(),
        )
        assert low <= exact <= high
        status = _wide_overflow_status(low, high)
        overflow_threshold = _power_of_two(1024) - _power_of_two(970)
        if status == "finite":
            assert abs(exact) < overflow_threshold
        if status == "overflow":
            assert abs(exact) >= overflow_threshold
        underflow_threshold = _power_of_two(-1022) - _power_of_two(-1075)
        if exact and abs(exact) < underflow_threshold:
            assert _possible_underflow(low, high)

    largest_count = 2**53 - 1
    assert (2 * largest_count * (largest_count - 1)).bit_length() == 107
    assert _wide_range_bounds([], [], []) == (0, 0, 0)
    assert _wide_range_bounds([1.0], [0.0], [1.0]) == (0, 0, 0)
    with pytest.raises(ValueError, match="negative weight"):
        _wide_range_bounds([1.0], [1.0], [-1.0])
    with pytest.raises(ValueError, match="non-finite"):
        _wide_range_bounds([1.0], [float("inf")], [1.0])


def test_full_range_overflow_classification_distinguishes_prefixes() -> None:
    huge = Fraction.from_float(float.fromhex("0x1.0p+1023"))
    exponents = range(1025, 1037)
    terms = []
    for exponent in exponents:
        weight = Fraction.from_float(math.ldexp(1.0, exponent - 2046))
        product = huge * huge * weight
        assert product == _power_of_two(exponent)
        terms.append(product)

    prefixes = [
        sum((term for index, term in enumerate(terms) if mask & (1 << index)), Fraction())
        for mask in range(1 << len(terms))
    ]
    assert len(set(prefixes)) == 1 << len(terms)
    for first, second in [(0, 0), (0, 1), (0xAAA, 0x555), (0xFFF, 0xFFE)]:
        difference = prefixes[first] - prefixes[second]
        if first == second:
            assert difference == 0
        else:
            assert abs(difference) >= _power_of_two(1025)
            with pytest.raises(OverflowError):
                float(difference)

    maximum_weight = Fraction.from_float(float.fromhex("0x1.0p+1023"))
    assert huge * huge * maximum_weight == _power_of_two(3069)
    assert 3069 - 1025 + 1 == 2045


def test_wide_accumulator_storage_and_binary64_triple_products() -> None:
    packed_entries = 2313 * 2314 // 2
    bytes_per_entry = (8 + 1) * 4
    assert packed_entries * bytes_per_entry == 96_341_076
    assert 6 * packed_entries * bytes_per_entry == 578_046_456
    rng = np.random.default_rng(192)
    pairs = rng.integers(0, 1 << 32, (300, 2), dtype=np.uint32)
    values = [
        _F64.unpack(_U32_PAIR.pack(int(low), int(high)))[0]
        for low, high in pairs
        if ((int(high) >> 20) & 0x7FF) != 0x7FF
    ]
    values = values[: len(values) - len(values) % 3]
    for a, b, weight in zip(values[::3], values[1::3], values[2::3], strict=True):
        product = Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(abs(weight))
        assert _round_unbounded_binary(product) == product
        if product:
            assert -3222 <= _floor_binary_exponent(product) <= 3071


def test_binary64_digits_on_random_finite_encodings() -> None:
    rng = np.random.default_rng(53)
    pairs = rng.integers(0, 1 << 32, (300, 2), dtype=np.uint32)
    values = [
        _F64.unpack(_U32_PAIR.pack(int(low), int(high)))[0]
        for low, high in pairs
        if ((int(high) >> 20) & 0x7FF) != 0x7FF
    ]
    for value in values:
        digits = _fixed_binary64_digits(value)
        integer = sum(digit << (8 * index) for index, digit in enumerate(digits))
        assert Fraction(integer, 1 << 1080) == Fraction.from_float(value)

    left = values[:8]
    right = values[8:16]
    weights = [abs(value) for value in values[16:24]]
    terms = [
        Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(weight)
        for a, b, weight in zip(left, right, weights, strict=True)
    ]
    assert _accumulator_value(_weighted_binary64_accumulator(left, right, weights), 3232) == sum(
        terms, Fraction()
    )


def test_binary64_final_overflow_is_reported() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    words = _weighted_binary64_accumulator([maximum], [maximum], [maximum])
    exact = _accumulator_value(words, 3232)
    assert exact == Fraction.from_float(maximum) ** 3
    with pytest.raises(OverflowError):
        float(exact)


def test_integer_chunk_and_subblock_limits() -> None:
    assert _CHUNK_ROWS * _MAX_DIGIT**2 < 2**23
    assert 16_384 * _MAX_DIGIT**2 < 2**30
    assert 2 * 16_384 * _MAX_DIGIT**2 < 2**31

    individual_cross_term = 65_536 * 128**2
    symmetric_sum = 2 * individual_cross_term
    assert individual_cross_term == 2**30
    assert symmetric_sum == 2**31
    assert (symmetric_sum % 2**32) - 2**32 == -(2**31)


def test_f32_scale_products_can_underflow_or_overflow() -> None:
    small_scale = np.float32(2**-69)
    large_scale = np.float32(2**65)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        small_factor = small_scale * small_scale * np.float32(2**-16)
        large_factor = large_scale * large_scale
        zero_times_large = np.float32(0) * large_factor
    assert small_factor == 0
    assert Fraction.from_float(2**-70) ** 2 == _power_of_two(-140)
    assert np.isinf(large_factor)
    assert np.isnan(zero_times_large)
    assert math.isfinite(float(2**128))


def test_binary64_weight_and_three_limbs_are_not_lossless_f32() -> None:
    weight = 0.1
    narrowed = _f32_from_bits(_f32_bits(weight))
    expected = Fraction.from_float(weight)
    assert abs(Fraction.from_float(narrowed) - expected) > _gamma(1) * abs(expected)

    three_limbs = Fraction(1) + _power_of_two(-24) + _power_of_two(-60)
    as_f64 = float(three_limbs)
    assert Fraction.from_float(as_f64) != three_limbs


def test_exponent_aware_digits_reject_nonfinite_binary32() -> None:
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="non-finite"):
            _exponent_aware_digits([value])
