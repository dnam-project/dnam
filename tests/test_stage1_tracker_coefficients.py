"""Independent arithmetic references for the Stage I T1 tracker coefficients."""

from __future__ import annotations

import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

_REFERENCE_NUMERATOR = 0xFFD51F
_REFERENCE_DENOMINATOR = 1 << 24
_F32_HALF_BITS = 0x3F000000
_HALF_SIGNIFICAND = 1 << 23
_BANDS = (
    (44_100.0, 96_000.0, 0xFFD000, 0xFFEBFF),
    (176_400.0, 384_000.0, 0xFFF400, 0xFFFBFF),
)


def _atanh_bounds(numerator: int, denominator: int, precision: int, terms: int) -> tuple[int, int]:
    assert 0 < numerator < denominator
    z_num = denominator - numerator
    z_den = denominator + numerator
    power_num = z_num
    power_den = z_den
    lower = 0
    for index in range(terms):
        lower += (power_num << precision) // ((2 * index + 1) * power_den)
        power_num *= z_num * z_num
        power_den *= z_den * z_den

    tail_num = (power_num << precision) * z_den * z_den
    tail_den = (2 * terms + 1) * power_den * (z_den * z_den - z_num * z_num)
    tail_upper = (tail_num + tail_den - 1) // tail_den
    return lower, lower + terms + tail_upper


def _compare_power_to_ratio(rate: float, numerator: int, denominator: int) -> int:
    rate_num, rate_den = rate.as_integer_ratio()
    exponent_num = 48_000 * rate_den
    common = math.gcd(exponent_num, rate_num)
    exponent_num //= common
    exponent_den = rate_num // common
    precision = 160
    terms = 8
    for _ in range(6):
        ref_lower, ref_upper = _atanh_bounds(
            _REFERENCE_NUMERATOR, _REFERENCE_DENOMINATOR, precision, terms
        )
        cmp_lower, cmp_upper = _atanh_bounds(numerator, denominator, precision, terms)
        if exponent_den * cmp_lower > exponent_num * ref_upper:
            return 1
        if exponent_den * cmp_upper < exponent_num * ref_lower:
            return -1
        precision *= 2
        terms *= 2
    raise AssertionError("independent logarithm bounds did not separate the values")


def _compare_power_exactly(rate: int, numerator: int, denominator: int) -> int:
    common = math.gcd(48_000, rate)
    exponent_num = 48_000 // common
    exponent_den = rate // common
    left = _REFERENCE_NUMERATOR**exponent_num * denominator**exponent_den
    right = numerator**exponent_den * _REFERENCE_DENOMINATOR**exponent_num
    return (left > right) - (left < right)


def _fixed_midpoint_comparison(
    rate: float, ref_bounds: tuple[int, int], midpoint_bounds: tuple[int, int]
) -> tuple[int, int, int]:
    rate_num, rate_den = rate.as_integer_ratio()
    exponent_num = 48_000 * rate_den
    low = rate_num * midpoint_bounds[0] - exponent_num * ref_bounds[1]
    high = rate_num * midpoint_bounds[1] - exponent_num * ref_bounds[0]
    if low > 0:
        return 1, low, rate_num
    if high < 0:
        return -1, -high, rate_num
    raise AssertionError("fixed integer intervals did not separate the values")


def _bounded_pole_bits(rate: float) -> int:
    if not math.isfinite(rate):
        raise ValueError("unsupported effective rate")
    band = next((band for band in _BANDS if band[0] <= rate <= band[1]), None)
    if band is None:
        raise ValueError("unsupported effective rate")
    _, _, first, last = band
    ref_bounds = _atanh_bounds(_REFERENCE_NUMERATOR, _REFERENCE_DENOMINATOR, 160, 8)
    lower, upper = first + 1, last
    while lower < upper:
        candidate = (lower + upper) // 2
        midpoint_bounds = _atanh_bounds(2 * candidate + 1, 1 << 25, 160, 8)
        direction, _, _ = _fixed_midpoint_comparison(rate, ref_bounds, midpoint_bounds)
        if direction > 0:
            lower = candidate + 1
        else:
            upper = candidate
    return _F32_HALF_BITS + lower - _HALF_SIGNIFICAND


def _correct_pole_bits(rate: float) -> int:
    lower = _HALF_SIGNIFICAND
    upper = 2 * _HALF_SIGNIFICAND
    while lower < upper:
        candidate = (lower + upper) // 2
        midpoint_num = 2 * candidate + 1
        if _compare_power_to_ratio(rate, midpoint_num, 1 << 25) > 0:
            lower = candidate + 1
        else:
            upper = candidate
    return _F32_HALF_BITS + lower - _HALF_SIGNIFICAND


def _gain_bits(pole_bits: int) -> int:
    numerator = (1 << 24) + pole_bits - _F32_HALF_BITS + _HALF_SIGNIFICAND
    quotient, remainder = divmod(numerator, 2)
    rounded = quotient + int(remainder == 1 and quotient % 2 == 1)
    return _F32_HALF_BITS + rounded - _HALF_SIGNIFICAND


def _f64_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _f64_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


@pytest.mark.parametrize(
    "rate",
    [44_100, 48_000, 48_001, 88_200, 96_000, 176_400, 192_000, 384_000],
)
def test_logarithm_enclosures_match_exact_integer_power_comparisons(rate: int) -> None:
    approximate = (_REFERENCE_NUMERATOR / _REFERENCE_DENOMINATOR) ** (48_000 / rate)
    candidate = _f32_bits(approximate) - _F32_HALF_BITS + _HALF_SIGNIFICAND
    for offset in (-2, -1, 0, 1):
        midpoint = 2 * (candidate + offset) + 1
        enclosed = _compare_power_to_ratio(float(rate), midpoint, 1 << 25)
        exact = _compare_power_exactly(rate, midpoint, 1 << 25)
        assert exact != 0
        assert enclosed == exact


_COEFFICIENT_VECTORS = (
    (44_100, 0x3F7FD155, 0x3F7FE8AA),
    (48_000, 0x3F7FD51F, 0x3F7FEA90),
    (48_001, 0x3F7FD51F, 0x3F7FEA90),
    (88_200, 0x3F7FE8A9, 0x3F7FF454),
    (96_000, 0x3F7FEA8F, 0x3F7FF548),
    (176_400, 0x3F7FF454, 0x3F7FFA2A),
    (192_000, 0x3F7FF547, 0x3F7FFAA4),
    (384_000, 0x3F7FFAA3, 0x3F7FFD52),
)


@pytest.mark.parametrize(("rate", "expected_pole", "expected_gain"), _COEFFICIENT_VECTORS)
def test_rounded_pole_and_gain_follow_exact_boundaries(
    rate: int, expected_pole: int, expected_gain: int
) -> None:
    bits = _correct_pole_bits(float(rate))
    significand = bits - _F32_HALF_BITS + _HALF_SIGNIFICAND
    assert _compare_power_exactly(rate, 2 * significand - 1, 1 << 25) > 0
    assert _compare_power_exactly(rate, 2 * significand + 1, 1 << 25) < 0
    gain = Fraction(1, 2) + Fraction(significand, 1 << 25)
    assert bits == expected_pole
    assert _bounded_pole_bits(float(rate)) == bits
    assert _gain_bits(bits) == expected_gain == _f32_bits(float(gain))
    assert 0x3F000000 < bits < 0x3F800000


@pytest.mark.parametrize("rate", [0.0, math.nan, math.inf, -math.inf, 100_000.0, 400_000.0])
def test_bounded_profile_rejects_unsupported_effective_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="unsupported effective rate"):
        _bounded_pole_bits(rate)


def test_fixed_intervals_decide_all_supported_binary64_rates() -> None:
    ref_bounds = _atanh_bounds(_REFERENCE_NUMERATOR, _REFERENCE_DENOMINATOR, 160, 8)
    assert ref_bounds[1] - ref_bounds[0] <= 9
    crossings = []
    for rate_min, rate_max, first, last in _BANDS:
        first_bits, last_bits = _f64_bits(rate_min), _f64_bits(rate_max)
        assert 2 * rate_min > 48_000
        count = 0
        for significand in range(first, last + 1):
            midpoint_bounds = _atanh_bounds(2 * significand + 1, 1 << 25, 160, 8)
            assert midpoint_bounds[1] - midpoint_bounds[0] <= 9

            def bound(bits: int, bounds: tuple[int, int] = midpoint_bounds) -> tuple[int, int, int]:
                return _fixed_midpoint_comparison(_f64_from_bits(bits), ref_bounds, bounds)

            first_sign, _, _ = bound(first_bits)
            last_sign, _, _ = bound(last_bits)
            assert first_sign <= last_sign
            if significand == first:
                assert first_sign == 1
            if significand == last:
                assert last_sign == -1
            if first_sign > 0:
                nearest = (first_bits,)
            elif last_sign < 0:
                nearest = (last_bits,)
            else:
                low, high = first_bits, last_bits
                while high - low > 1:
                    middle = (low + high) // 2
                    if bound(middle)[0] < 0:
                        low = middle
                    else:
                        high = middle
                assert bound(low)[0] == -1
                assert bound(high)[0] == 1
                nearest = (low, high)
                count += 1
                if count % 512 == 0:
                    pole = _F32_HALF_BITS + significand - _HALF_SIGNIFICAND
                    assert _bounded_pole_bits(_f64_from_bits(low)) == pole
                    assert _bounded_pole_bits(_f64_from_bits(high)) == pole + 1
            for bits in nearest:
                _, margin, rate_num = bound(bits)
                assert margin > rate_num << 72
        crossings.append(count)
    assert crossings == [6458, 1615]


def test_bounded_profile_matches_adaptive_reference_at_binary64_samples() -> None:
    for rate_min, rate_max, _, _ in _BANDS:
        first_bits, last_bits = _f64_bits(rate_min), _f64_bits(rate_max)
        for index in range(129):
            bits = first_bits + (last_bits - first_bits) * index // 128
            rate = _f64_from_bits(bits)
            assert _bounded_pole_bits(rate) == _correct_pole_bits(rate)


def test_gain_rounding_for_every_candidate_significand() -> None:
    for _, _, first, last in _BANDS:
        for significand in range(first + 1, last + 1):
            pole_bits = _F32_HALF_BITS + significand - _HALF_SIGNIFICAND
            exact = Fraction(1, 2) + Fraction(significand, 1 << 25)
            assert _gain_bits(pole_bits) == _f32_bits(float(exact))


def test_documented_t1_vectors_match_the_reference() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/dsp/op_m_modal_kautz.md"
    text = path.read_text(encoding="utf-8")
    table = text.split("```text\nT1-COEFFICIENT-VECTORS\n", 1)[1].split("\n```", 1)[0]
    rows = []
    for line in table.splitlines():
        rate, pole, gain = (part.strip() for part in line.split("|"))
        rows.append((int(rate), int(pole, 16), int(gain, 16)))
    assert tuple(rows) == _COEFFICIENT_VECTORS


def test_documented_midpoint_search_ranges_match_the_reference() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/dsp/op_m_modal_kautz.md"
    text = path.read_text(encoding="utf-8")
    table = text.split("```text\nT1-MIDPOINT-SEARCH\n", 1)[1].split("\n```", 1)[0]
    rows = []
    for line in table.splitlines()[1:]:
        rate, first, last, comparisons = (part.strip() for part in line.split("|"))
        low_rate, high_rate = (int(value) for value in rate.split(".."))
        rows.append((low_rate, high_rate, int(first, 16), int(last, 16), int(comparisons)))
    assert rows == [
        (int(low), int(high), first, last, (last - first - 1).bit_length())
        for low, high, first, last in _BANDS
    ]


def test_bounded_integer_preparation_intermediates_fit_the_documented_widths() -> None:
    z_num_max = (1 << 25) - 1
    z_den_max = (1 << 26) - 1
    term_num_bound = z_num_max**15 << 160
    term_den_bound = 15 * z_den_max**15
    tail_num_bound = (z_num_max**17 << 160) * z_den_max**2
    tail_den_bound = 17 * z_den_max**19
    assert max(term_num_bound, term_den_bound, tail_num_bound, tail_den_bound) < 1 << 640

    rate_num_bound = (1 << 53) - 1
    exponent_num_bound = 48_000 * (1 << 37)
    sum_bound = 8 * (1 << 160) + 9
    assert max(rate_num_bound, exponent_num_bound) < 1 << 53
    assert max(rate_num_bound, exponent_num_bound) * sum_bound < 1 << 256


def test_reference_rate_returns_published_coefficient_bits() -> None:
    pole_bits = _correct_pole_bits(48_000.0)
    assert pole_bits == 0x3F7FD51F
    assert _gain_bits(pole_bits) == 0x3F7FEA90


def test_correct_f64_intermediate_can_double_round_the_wrong_f32_pole() -> None:
    midpoint_num = 2 * _REFERENCE_NUMERATOR + 1
    lower_bits = _f64_bits(48_000.0)
    upper_bits = _f64_bits(48_010.0)
    assert _compare_power_to_ratio(_f64_from_bits(lower_bits), midpoint_num, 1 << 25) < 0
    assert _compare_power_to_ratio(_f64_from_bits(upper_bits), midpoint_num, 1 << 25) > 0
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        comparison = _compare_power_to_ratio(_f64_from_bits(middle_bits), midpoint_num, 1 << 25)
        if comparison < 0:
            lower_bits = middle_bits
        else:
            upper_bits = middle_bits

    below = _f64_from_bits(lower_bits)
    above = _f64_from_bits(upper_bits)
    assert lower_bits == 0x40E77045FD955545
    assert upper_bits == 0x40E77045FD955546
    assert _compare_power_to_ratio(below, midpoint_num, 1 << 25) < 0
    assert _compare_power_to_ratio(above, midpoint_num, 1 << 25) > 0
    midpoint_54 = midpoint_num << 29
    for rate in (below, above):
        assert _compare_power_to_ratio(rate, midpoint_54 - 1, 1 << 54) > 0
        assert _compare_power_to_ratio(rate, midpoint_54 + 1, 1 << 54) < 0
    assert _correct_pole_bits(below) == 0x3F7FD51F
    assert _correct_pole_bits(above) == 0x3F7FD520
    assert _bounded_pole_bits(below) == 0x3F7FD51F
    assert _bounded_pole_bits(above) == 0x3F7FD520
    assert _f32_bits(midpoint_num / (1 << 25)) == 0x3F7FD520

    with localcontext() as context:
        context.prec = 100
        reference = Decimal(_REFERENCE_NUMERATOR) / Decimal(_REFERENCE_DENOMINATOR)
        midpoint = Decimal(midpoint_num) / Decimal(1 << 25)
        low_value = (reference.ln() * Decimal(48_000) / Decimal.from_float(below)).exp()
        high_value = (reference.ln() * Decimal(48_000) / Decimal.from_float(above)).exp()
        assert low_value < midpoint < high_value
