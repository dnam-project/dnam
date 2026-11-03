"""Independent trust-region arithmetic examples, not real-time engine tests."""

from __future__ import annotations

import math
import random
import struct
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")
_F64 = struct.Struct("<d")
_U64 = struct.Struct("<Q")


def _f32_from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _f32_bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _widen_f32_bits(bits: int) -> float:
    sign = (bits & 0x8000_0000) << 32
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7F_FFFF
    if exponent == 0xFF:
        raise ValueError("invalid nonfinite f32 value")
    if exponent:
        widened = sign | ((exponent + 896) << 52) | (fraction << 29)
    elif fraction:
        leading = fraction.bit_length() - 1
        widened = sign | ((leading + 874) << 52) | ((fraction - (1 << leading)) << (52 - leading))
    else:
        widened = sign
    return _F64.unpack(_U64.pack(widened))[0]


def _round_right_even(mantissa: int, shift: int) -> int:
    if shift > 53:
        return 0
    truncated = mantissa >> shift
    remainder = mantissa & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return truncated + int(remainder > halfway or (remainder == halfway and truncated & 1 != 0))


def _round_f64_to_f32_bits(value: float) -> int:
    bits = _U64.unpack(_F64.pack(value))[0]
    sign = (bits >> 32) & 0x8000_0000
    exponent_bits = (bits >> 52) & 0x7FF
    if exponent_bits == 0:
        if bits & 0xF_FFFF_FFFF_FFFF:
            raise ValueError("f64 subnormal is outside the candidate domain")
        return sign
    if exponent_bits == 0x7FF:
        raise ValueError("nonfinite f64 value")
    exponent = exponent_bits - 1023
    mantissa = (1 << 52) | (bits & 0xF_FFFF_FFFF_FFFF)
    if exponent < -126:
        return sign | _round_right_even(mantissa, -exponent - 97)
    if exponent > 127:
        raise ValueError("f32 overflow")
    rounded = _round_right_even(mantissa, 29)
    if rounded == 1 << 24:
        exponent += 1
        rounded >>= 1
    if exponent > 127:
        raise ValueError("f32 overflow")
    return sign | ((exponent + 127) << 23) | (rounded & 0x7F_FFFF)


def _exponent(bits: int) -> int:
    magnitude = bits & 0x7FFF_FFFF
    exponent = magnitude >> 23
    if magnitude == 0 or exponent == 0xFF:
        raise ValueError("nonzero finite value required")
    if exponent:
        return exponent - 127
    return (magnitude & 0x7F_FFFF).bit_length() - 150


def _next_up(value: float) -> float:
    assert 0 < value < math.inf
    bits = _U64.unpack(_F64.pack(value))[0]
    return _F64.unpack(_U64.pack(bits + 1))[0]


def _upper_sum_squares(values: tuple[float, ...], initial: float = 0.0) -> float:
    total = initial
    for value in values:
        square = value * value
        if square:
            total = square if total == 0 else _next_up(total + square)
    return total


def _upper_product(left: float, right: float) -> float:
    if left == 0 or right == 0:
        return 0.0
    return _next_up(left * right)


def _candidate(
    x: tuple[float, ...], radius_squared: float, energy_upper: float, scale: float
) -> tuple[bool, tuple[int, ...]]:
    output = tuple(_round_f64_to_f32_bits(value * scale) for value in x)
    rounded = tuple(_widen_f32_bits(bits) for bits in output)
    scale_squared_upper = _upper_product(scale, scale)
    ideal_upper = _upper_product(scale_squared_upper, energy_upper)
    rounded_upper = _upper_sum_squares(rounded)
    return ideal_upper <= radius_squared and rounded_upper <= radius_squared, output


def _inverse_seed(energy_upper: float) -> float:
    bits = _U64.unpack(_F64.pack(energy_upper))[0]
    exponent = ((bits >> 52) & 0x7FF) - 1023
    inverse_bits = (1023 - ((exponent + 2) // 2)) << 52
    return _F64.unpack(_U64.pack(inverse_bits))[0]


def _fast_candidate(
    x: tuple[float, ...], radius: float, epsilon: float
) -> tuple[bool, tuple[int, ...], float]:
    energy_upper = _upper_sum_squares(x, epsilon)
    inverse = _inverse_seed(energy_upper)
    for _ in range(6):
        square = (energy_upper * inverse) * inverse
        inverse = inverse * (1.5 - 0.5 * square)
    guard = 1 - 2**-20
    scale = min(guard, (radius * inverse) * guard)
    accepted, output = _candidate(x, radius * radius, energy_upper, scale)
    return accepted, output, scale


def _clip_reference(
    values: tuple[int, ...], radius_bits: int, epsilon_bits: int
) -> tuple[tuple[int, ...], float, float]:
    if len(values) not in (4, 8):
        raise ValueError("unsupported latent dimension")
    x = tuple(_widen_f32_bits(bits) for bits in values)
    radius = _widen_f32_bits(radius_bits)
    epsilon = _widen_f32_bits(epsilon_bits)
    if (
        not all(math.isfinite(value) for value in x)
        or not math.isfinite(radius)
        or not math.isfinite(epsilon)
        or radius <= 0
        or epsilon <= 0
    ):
        raise ValueError("invalid latent, radius or epsilon")
    ideal_energy_upper = _upper_sum_squares(x, epsilon)
    radius_squared = radius * radius

    accepted, original = _candidate(x, radius_squared, ideal_energy_upper, 1.0)
    if accepted:
        return original, 1.0, 1.0

    nonzero = [bits for bits in values if bits & 0x7FFF_FFFF]
    norm_exponent = max((_exponent(bits) for bits in nonzero), default=_exponent(epsilon_bits) // 2)
    norm_exponent = max(norm_exponent, _exponent(epsilon_bits) // 2)
    high_exponent = _exponent(radius_bits) + 1 - norm_exponent
    high = 1.0 if high_exponent >= 0 else _F64.unpack(_U64.pack((high_exponent + 1023) << 52))[0]
    initial_high = high
    low = 0.0
    result = tuple(_round_f64_to_f32_bits(0.0 * value) for value in x)
    for _ in range(32):
        middle = (low + high) * 0.5
        accepted, output = _candidate(x, radius_squared, ideal_energy_upper, middle)
        if accepted:
            low, result = middle, output
        else:
            high = middle
    return result, low, initial_high


def _fraction(bits: int) -> Fraction:
    return Fraction.from_float(_f32_from_bits(bits))


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _ideal_decimal_clip(
    values: tuple[Fraction, ...], radius: Fraction, epsilon: Fraction
) -> tuple[Decimal, ...]:
    inputs = tuple(_decimal_fraction(value) for value in values)
    denominator = (
        sum((value * value for value in inputs), Decimal()) + _decimal_fraction(epsilon)
    ).sqrt()
    scale = min(Decimal(1), _decimal_fraction(radius) / denominator)
    return tuple(value * scale for value in inputs)


def _decimal_distance(left: tuple[Decimal, ...], right: tuple[Decimal, ...]) -> Decimal:
    return sum(
        ((x - y) ** 2 for x, y in zip(left, right, strict=True)),
        Decimal(),
    ).sqrt()


def _check_exact_normal_output_error(
    values: tuple[int, ...], output: tuple[int, ...], radius_bits: int, epsilon_bits: int
) -> None:
    inputs = tuple(_fraction(bits) for bits in values)
    rounded = tuple(_fraction(bits) for bits in output)
    radius = _fraction(radius_bits)
    input_energy = sum((value * value for value in inputs), Fraction())
    ideal_energy = input_energy + _fraction(epsilon_bits)
    error_limit_squared = (Fraction(1185, 10**9) * radius) ** 2
    if ideal_energy <= radius * radius:
        error_squared = sum(
            ((actual - ideal) ** 2 for actual, ideal in zip(rounded, inputs, strict=True)),
            Fraction(),
        )
        assert error_squared < error_limit_squared
        return

    scale_squared = radius * radius / ideal_energy
    rounded_energy = sum((value * value for value in rounded), Fraction())
    dot_product = sum(
        (value * actual for value, actual in zip(inputs, rounded, strict=True)), Fraction()
    )
    assert dot_product >= 0
    remainder = rounded_energy + scale_squared * input_energy - error_limit_squared
    assert remainder < 0 or remainder**2 < 4 * scale_squared * dot_product**2


def _check_exact_bounds(
    values: tuple[int, ...], radius_bits: int, epsilon_bits: int
) -> tuple[tuple[int, ...], float, float]:
    result, scale, high = _clip_reference(values, radius_bits, epsilon_bits)
    radius_squared = _fraction(radius_bits) ** 2
    ideal_energy = sum((_fraction(bits) ** 2 for bits in values), _fraction(epsilon_bits))
    rounded_energy = sum((_fraction(bits) ** 2 for bits in result), Fraction())
    assert Fraction.from_float(scale) ** 2 * ideal_energy <= radius_squared
    assert rounded_energy <= radius_squared
    bracket = Fraction.from_float(high)
    assert (bracket / 12) ** 2 * ideal_energy <= radius_squared
    if high < 1:
        assert bracket**2 * ideal_energy > radius_squared
    if radius_bits & 0x7F80_0000:
        retained = Fraction(1) - Fraction(1, 2**20) - Fraction(12, 2**32)
        if ideal_energy <= radius_squared:
            assert Fraction.from_float(scale) >= retained
        else:
            assert Fraction.from_float(scale) ** 2 * ideal_energy >= retained**2 * radius_squared
        _check_exact_normal_output_error(values, result, radius_bits, epsilon_bits)
    x = tuple(_f32_from_bits(bits) for bits in values)
    fast_accepted, fast_output, fast_scale = _fast_candidate(
        x, _f32_from_bits(radius_bits), _f32_from_bits(epsilon_bits)
    )
    if fast_accepted:
        assert Fraction.from_float(fast_scale) ** 2 * ideal_energy <= radius_squared
        assert sum((_fraction(bits) ** 2 for bits in fast_output), Fraction()) <= radius_squared
    if radius_bits & 0x7F80_0000:
        assert fast_accepted
        _check_exact_normal_output_error(values, fast_output, radius_bits, epsilon_bits)
        retained = Fraction(1) - Fraction(1, 2**20) - Fraction(1, 2**30)
        if ideal_energy <= radius_squared:
            assert Fraction.from_float(fast_scale) >= retained
        else:
            assert (
                Fraction.from_float(fast_scale) ** 2 * ideal_energy >= retained**2 * radius_squared
            )
    return result, scale, high


def test_ideal_clip_is_nonexpansive_across_sampled_boundaries() -> None:
    rng = random.Random(3_418_511)
    small = Fraction(1, 10**20)
    cases: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction, Fraction]] = [
        (
            (Fraction(4) - small, Fraction(), Fraction(), Fraction()),
            (Fraction(4) + small, Fraction(), Fraction(), Fraction()),
            Fraction(5),
            Fraction(9),
        ),
        (
            (Fraction(), Fraction(1), Fraction(), Fraction()),
            (Fraction(2), Fraction(-1), Fraction(), Fraction()),
            Fraction(1),
            Fraction(4),
        ),
    ]
    for dimension in (4, 8):
        for _ in range(100):
            left = tuple(Fraction(rng.randint(-1024, 1024), 256) for _ in range(dimension))
            right = tuple(Fraction(rng.randint(-1024, 1024), 256) for _ in range(dimension))
            radius = Fraction(rng.randint(1, 8), 2)
            epsilon = Fraction(rng.randint(1, 32), 16)
            cases.append((left, right, radius, epsilon))

    with localcontext() as context:
        context.prec = 100
        for left, right, radius, epsilon in cases:
            left_decimal = tuple(_decimal_fraction(value) for value in left)
            right_decimal = tuple(_decimal_fraction(value) for value in right)
            input_distance = _decimal_distance(left_decimal, right_decimal)
            output_distance = _decimal_distance(
                _ideal_decimal_clip(left, radius, epsilon),
                _ideal_decimal_clip(right, radius, epsilon),
            )
            assert output_distance <= input_distance + Decimal("1e-70")


def test_ideal_clip_bound_handles_radius_and_epsilon_changes() -> None:
    rng = random.Random(8_220_014)
    with localcontext() as context:
        context.prec = 100
        for dimension in (4, 8):
            for _ in range(100):
                left = tuple(Fraction(rng.randint(-1024, 1024), 256) for _ in range(dimension))
                right = tuple(Fraction(rng.randint(-1024, 1024), 256) for _ in range(dimension))
                radius_left = Fraction(rng.randint(1, 16), 4)
                radius_right = Fraction(rng.randint(1, 16), 4)
                epsilon_left = Fraction(rng.randint(1, 32), 16)
                epsilon_right = Fraction(rng.randint(1, 32), 16)
                input_distance = _decimal_distance(
                    tuple(_decimal_fraction(value) for value in left),
                    tuple(_decimal_fraction(value) for value in right),
                )
                epsilon_distance = (
                    _decimal_fraction(epsilon_left).sqrt() - _decimal_fraction(epsilon_right).sqrt()
                )
                bound = (input_distance**2 + epsilon_distance**2).sqrt() + abs(
                    _decimal_fraction(radius_left - radius_right)
                )
                output_distance = _decimal_distance(
                    _ideal_decimal_clip(left, radius_left, epsilon_left),
                    _ideal_decimal_clip(right, radius_right, epsilon_right),
                )
                assert output_distance <= bound + Decimal("1e-70")


def test_normal_clip_candidate_composes_with_raw_input_error() -> None:
    rng = random.Random(14_709)
    radius_bits = _f32_bits(1.0)
    epsilon_bits = _f32_bits(2.0**-20)
    radius = _fraction(radius_bits)
    epsilon = _fraction(epsilon_bits)
    with localcontext() as context:
        context.prec = 100
        for dimension in (4, 8):
            for _ in range(40):
                desired = tuple(
                    Fraction(rng.randint(-512, 512), 128) + Fraction(rng.choice([-1, 1]), 2**30)
                    for _ in range(dimension)
                )
                input_bits = tuple(_f32_bits(float(value)) for value in desired)
                output_bits, _, _ = _clip_reference(input_bits, radius_bits, epsilon_bits)
                actual_input = tuple(_fraction(bits) for bits in input_bits)
                actual_output = tuple(_decimal_fraction(_fraction(bits)) for bits in output_bits)
                raw_error = _decimal_distance(
                    tuple(_decimal_fraction(value) for value in desired),
                    tuple(_decimal_fraction(value) for value in actual_input),
                )
                output_error = _decimal_distance(
                    _ideal_decimal_clip(desired, radius, epsilon), actual_output
                )
                assert output_error <= raw_error + Decimal("1.185e-6") * _decimal_fraction(radius)


def test_bitwise_widening_matches_f32_values_across_exponents() -> None:
    rng = random.Random(20260607)
    bits_to_check = [
        (exponent << 23) | fraction | sign
        for exponent in range(0xFF)
        for fraction in (0, 1, 0x3F_FFFF, 0x7F_FFFF)
        for sign in (0, 0x8000_0000)
    ]
    bits_to_check.extend(
        rng.randrange(0x7F80_0000) | (rng.randrange(2) << 31) for _ in range(5_000)
    )
    for bits in bits_to_check:
        actual = _U64.unpack(_F64.pack(_widen_f32_bits(bits)))[0]
        expected = _U64.unpack(_F64.pack(_f32_from_bits(bits)))[0]
        assert actual == expected


def test_bitwise_f32_rounding_matches_ieee_halfway_neighbors() -> None:
    rng = random.Random(20260608)
    lower_bits = [0, 1, 0x7F_FFFE, 0x7F_FFFF, 0x80_0000, 0x3F7F_FFFF, 0x7F7F_FFFE]
    lower_bits.extend(rng.randrange(0x7F7F_FFFF) for _ in range(4_000))
    for bits in lower_bits:
        low, high = _widen_f32_bits(bits), _widen_f32_bits(bits + 1)
        halfway = (low + high) * 0.5
        for value in (math.nextafter(halfway, 0.0), halfway, math.nextafter(halfway, math.inf)):
            for signed in (value, -value):
                assert _round_f64_to_f32_bits(signed) == _f32_bits(signed)
    assert _round_f64_to_f32_bits(-0.0) == 0x8000_0000
    with pytest.raises(ValueError, match="subnormal"):
        _round_f64_to_f32_bits(math.ulp(0.0))
    with pytest.raises(ValueError, match="nonfinite"):
        _round_f64_to_f32_bits(math.inf)
    with pytest.raises(ValueError, match="overflow"):
        _round_f64_to_f32_bits(2.0**128)


def test_subnormal_rounding_is_not_equivalent_to_flushing() -> None:
    values = (1, 0x8000_0001, 0, 0x8000_0000)
    result, scale, _ = _check_exact_bounds(values, _f32_bits(1), 1)
    assert scale == 1
    assert result == values
    flushed = tuple(bits & 0x8000_0000 if bits & 0x7F80_0000 == 0 else bits for bits in result)
    assert flushed != result
    assert sum((_fraction(bits) ** 2 for bits in flushed), Fraction()) <= 1


def test_rounded_output_can_violate_the_exact_ideal_ball() -> None:
    values = tuple(
        _f32_bits(value)
        for value in (
            -0.07378888875246048,
            0.1867510974407196,
            1.6696151494979858,
            1.5949534177780151,
        )
    )
    x = tuple(_f32_from_bits(bits) for bits in values)
    ideal_scale = 1 / math.sqrt(sum(value * value for value in x) + 2**-149)
    naive = tuple(_f32_bits(value * ideal_scale) for value in x)
    with localcontext() as context:
        context.prec = 100
        energy = sum((Decimal.from_float(value) ** 2 for value in x), Decimal.from_float(2**-149))
        high_precision_scale = Decimal(1) / energy.sqrt()
        for value, bits in zip(x, naive, strict=True):
            target = Decimal.from_float(value) * high_precision_scale
            sign, magnitude = bits & 0x8000_0000, bits & 0x7FFF_FFFF
            distance = abs(Decimal.from_float(_f32_from_bits(bits)) - target)
            for neighbor in (sign | (magnitude - 1), sign | (magnitude + 1)):
                gap = abs(Decimal.from_float(_f32_from_bits(neighbor)) - target) - distance
                assert gap > Decimal("1e-40")
    assert sum((_fraction(bits) ** 2 for bits in naive), Fraction()) > 1
    result, _, _ = _check_exact_bounds(values, _f32_bits(1.0), 1)
    assert result != naive


def test_float32_norm_can_round_away_a_positive_component() -> None:
    values = tuple(_f32_bits(value) for value in (1.0, 2**-12, 0.0, 0.0))
    rounded_norm_squared = _f32_from_bits(_f32_bits(1 + 2**-24))
    assert rounded_norm_squared == 1
    assert sum((_fraction(bits) ** 2 for bits in values), Fraction()) > 1
    _check_exact_bounds(values, _f32_bits(1.0), 1)


def test_zero_vector_preserves_signed_zero_with_an_epsilon_dominated_scale() -> None:
    values = (0, 0x8000_0000, 0, 0x8000_0000)
    result, scale, _ = _check_exact_bounds(values, _f32_bits(1.0), _f32_bits(2.0))
    assert result == values
    assert 0.707106 < scale < 0.707107


def test_candidate_acceptance_is_monotone_on_sampled_brackets() -> None:
    for values, radius_bits, epsilon_bits in (
        (tuple(_f32_bits(value) for value in (1.0, -2.0, 0.5, 0.1)), _f32_bits(1), 1),
        ((0x7F7F_FFFF, 0x8080_0001, 0x1, 0), 1, 1),
    ):
        _, _, high = _check_exact_bounds(values, radius_bits, epsilon_bits)
        x = tuple(_f32_from_bits(bits) for bits in values)
        radius_squared = _f32_from_bits(radius_bits) ** 2
        energy_upper = _upper_sum_squares(x, _f32_from_bits(epsilon_bits))
        accepted = [
            _candidate(x, radius_squared, energy_upper, high * index / 64)[0] for index in range(65)
        ]
        assert accepted[0]
        assert not accepted[-1]
        assert accepted == sorted(accepted, reverse=True)


def test_fast_candidate_handles_the_lowest_newton_seed() -> None:
    x = (0.0, -0.0, 0.0, -0.0)
    accepted, output, scale = _fast_candidate(x, 0.5, 1.0)
    assert accepted
    assert output == (0, 0x8000_0000, 0, 0x8000_0000)
    assert Fraction.from_float(scale) >= Fraction(1, 2) * (
        1 - Fraction(1, 2**20) - Fraction(1, 2**30)
    )
    assert scale < 0.5


def test_fast_newton_seed_and_iteration_bounds() -> None:
    for exponent in range(-149, 260):
        for mantissa in (1.0, 1.5, math.nextafter(2.0, 0.0)):
            energy_upper = math.ldexp(mantissa, exponent)
            seed = _inverse_seed(energy_upper)
            normalized_squared = Fraction.from_float(energy_upper) * Fraction.from_float(seed) ** 2
            assert Fraction(1, 4) <= normalized_squared < 1

    error = Fraction(1, 2)
    for _ in range(6):
        error = Fraction(3, 2) * error**2 - Fraction(1, 2) * error**3
    assert error < Fraction(1, 50_000_000_000)
    rounding_multiplier = 16 * sum((Fraction(6, 5) ** index for index in range(6)), Fraction())
    assert rounding_multiplier < 160


def test_fast_candidate_uses_safe_fallback_for_subnormal_radius() -> None:
    values = tuple(_f32_bits(value) for value in (1.0, 1.0, 0.0, 0.0))
    x = tuple(_f32_from_bits(bits) for bits in values)
    accepted, output, _ = _fast_candidate(x, _f32_from_bits(1), _f32_from_bits(1))
    assert not accepted
    assert sum((_fraction(bits) ** 2 for bits in output), Fraction()) > _fraction(1) ** 2
    fallback, _, _ = _check_exact_bounds(values, 1, 1)
    assert all(not bits & 0x7FFF_FFFF for bits in fallback)
    ideal_squared_over_radius_squared = Fraction(2) / (Fraction(2) + _fraction(1))
    assert ideal_squared_over_radius_squared > (1 - Fraction(1, 2**100)) ** 2


@pytest.mark.parametrize("dimension", [4, 8])
def test_adversarial_float32_extremes_remain_inside_both_exact_bounds(dimension: int) -> None:
    magnitudes = (1, 0x7F_FFFF, 0x80_0000, _f32_bits(1.0), 0x7F7F_FFFF)
    for radius_bits in magnitudes:
        for epsilon_bits in magnitudes:
            values = tuple(
                magnitudes[(index * 2 + 1) % len(magnitudes)] | ((index & 1) << 31)
                for index in range(dimension)
            )
            _check_exact_bounds(values, radius_bits, epsilon_bits)

    rng = random.Random(20260603 + dimension)
    for _ in range(500):
        values = tuple(
            rng.randrange(0x7F80_0000) | (rng.randrange(2) << 31) for _ in range(dimension)
        )
        radius_bits = rng.randrange(1, 0x7F80_0000)
        epsilon_bits = rng.randrange(1, 0x7F80_0000)
        _check_exact_bounds(values, radius_bits, epsilon_bits)


def test_sampled_normal_outputs_stay_close_to_the_ideal_scale() -> None:
    rng = random.Random(20260604)
    for _ in range(300):
        values = tuple(_f32_bits(rng.uniform(-10, 10)) for _ in range(8))
        radius_bits = _f32_bits(rng.uniform(0.1, 5))
        epsilon_bits = _f32_bits(rng.uniform(1e-8, 0.1))
        _, scale, high = _check_exact_bounds(values, radius_bits, epsilon_bits)
        radius = _f32_from_bits(radius_bits)
        epsilon = _f32_from_bits(epsilon_bits)
        norm = sum(_f32_from_bits(bits) ** 2 for bits in values)
        ideal = min(1.0, radius / math.sqrt(norm + epsilon))
        assert scale <= ideal
        assert ideal - scale <= 1e-6 * ideal + high * 2**-32


def test_invalid_trust_clip_inputs_are_rejected() -> None:
    valid = (_f32_bits(0.5), 0, 0, 0)
    for values, radius, epsilon in (
        (valid[:3], _f32_bits(1), _f32_bits(0.1)),
        ((0x7F80_0000, *valid[1:]), _f32_bits(1), _f32_bits(0.1)),
        (valid, 0, _f32_bits(0.1)),
        (valid, _f32_bits(1), 0),
        (valid, _f32_bits(-1), _f32_bits(0.1)),
        (valid, _f32_bits(1), 0x7FC0_0000),
    ):
        with pytest.raises(ValueError, match=r"invalid|unsupported"):
            _clip_reference(values, radius, epsilon)
