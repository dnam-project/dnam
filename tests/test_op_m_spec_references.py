"""Independent numerical examples for the provisional Stage I Op-M specification.

These are mathematical feasibility checks, not tests of the Rust engine.
"""

from __future__ import annotations

import cmath
import hashlib
import itertools
import math
import struct
from fractions import Fraction

import pytest

_F32 = struct.Struct(">f")
_U32 = struct.Struct(">I")

# Initial H1 reference coefficients; these are not an engine implementation.
# Each row stores six I-branch and then six Q-branch IEEE-754 f32 bit patterns.
_HILBERT_PROFILE_BITS = {
    44_100: (
        0x3D8EC459,
        0x3EE935E1,
        0x3F47659D,
        0x3F6CCEAA,
        0x3F7A0188,
        0x3F7E7297,
        0x3E7B1AD6,
        0x3F243369,
        0x3F5E97F2,
        0x3F752D7D,
        0x3F7CCA31,
        0x3F7F8A32,
    ),
    48_000: (
        0x3D91BEE6,
        0x3EECB8E1,
        0x3F48DE0D,
        0x3F6D91C9,
        0x3F7A5939,
        0x3F7E8E9D,
        0x3E80025F,
        0x3F25F6B0,
        0x3F5FADA1,
        0x3F75B332,
        0x3F7CFFA1,
        0x3F7F92D2,
    ),
    64_000: (
        0x3DA2AF27,
        0x3EFE4625,
        0x3F4FBDA5,
        0x3F70B815,
        0x3F7B8289,
        0x3F7EE2C7,
        0x3E8C7CF3,
        0x3F2E9DD3,
        0x3F647F75,
        0x3F77A752,
        0x3F7DA785,
        0x3F7FAC3E,
    ),
    80_000: (
        0x3DA88500,
        0x3F0212AF,
        0x3F523110,
        0x3F72069B,
        0x3F7C1032,
        0x3F7F0F58,
        0x3E90CC38,
        0x3F318514,
        0x3F665AC6,
        0x3F788657,
        0x3F7DFC4C,
        0x3F7FBA14,
    ),
    96_000: (
        0x3DB073C5,
        0x3F05A9C4,
        0x3F54E508,
        0x3F7343B2,
        0x3F7C852B,
        0x3F7F30C7,
        0x3E963AF7,
        0x3F34F3ED,
        0x3F683FD9,
        0x3F794AFF,
        0x3F7E3E99,
        0x3F7FC438,
    ),
    176_400: (
        0x3E0D917D,
        0x3F216E28,
        0x3F624034,
        0x3F77D413,
        0x3F7DE528,
        0x3F7F8887,
        0x3ECDF9DC,
        0x3F49A494,
        0x3F703BD8,
        0x3F7BD39E,
        0x3F7EF730,
        0x3F7FDDFB,
    ),
    192_000: (
        0x3E1A79AD,
        0x3F2691E2,
        0x3F64674D,
        0x3F787EBC,
        0x3F7E12E1,
        0x3F7F9298,
        0x3ED99328,
        0x3F4D2EB7,
        0x3F717613,
        0x3F7C2D21,
        0x3F7F0DA9,
        0x3F7FE0D4,
    ),
    208_000: (
        0x3E2646E1,
        0x3F2A6E87,
        0x3F65C7A3,
        0x3F78E266,
        0x3F7E2EFD,
        0x3F7F99AA,
        0x3EE33E4A,
        0x3F4F9D5A,
        0x3F723264,
        0x3F7C621C,
        0x3F7F1C5F,
        0x3F7FE2ED,
    ),
    224_000: (
        0x3E34BA6C,
        0x3F2F8A17,
        0x3F67DE51,
        0x3F798480,
        0x3F7E584A,
        0x3F7FA231,
        0x3EEF3412,
        0x3F53121B,
        0x3F736170,
        0x3F7CB521,
        0x3F7F3010,
        0x3F7FE547,
    ),
    240_000: (
        0x3E4151D9,
        0x3F33437E,
        0x3F693973,
        0x3F79E8DF,
        0x3F7E72F8,
        0x3F7FA845,
        0x3EF8BC96,
        0x3F556DD1,
        0x3F741F05,
        0x3F7CE93F,
        0x3F7F3D5F,
        0x3F7FE706,
    ),
    256_000: (
        0x3E4CBD88,
        0x3F363B81,
        0x3F6A3C44,
        0x3F7A3322,
        0x3F7E877C,
        0x3F7FAD32,
        0x3F006617,
        0x3F573C54,
        0x3F74AA8C,
        0x3F7D1079,
        0x3F7F47DE,
        0x3F7FE876,
    ),
    288_000: (
        0x3E64086E,
        0x3F3BDB10,
        0x3F6C20CB,
        0x3F7ABA23,
        0x3F7EABAF,
        0x3F7FB5AD,
        0x3F083609,
        0x3F5AA15E,
        0x3F75ACD3,
        0x3F7D5697,
        0x3F7F5A2A,
        0x3F7FEAEB,
    ),
    320_000: (
        0x3E7D6662,
        0x3F41DCD7,
        0x3F6E3EDD,
        0x3F7B5343,
        0x3F7ED216,
        0x3F7FBDBA,
        0x3F1068CD,
        0x3F5E589B,
        0x3F76D2C7,
        0x3F7DA41A,
        0x3F7F6C8B,
        0x3F7FED2A,
    ),
    352_000: (
        0x3E88E622,
        0x3F45A7EB,
        0x3F6F6903,
        0x3F7BA4C3,
        0x3F7EE847,
        0x3F7FC310,
        0x3F163553,
        0x3F6082FF,
        0x3F776EE6,
        0x3F7DCEB5,
        0x3F7F77E5,
        0x3F7FEEB9,
    ),
    384_000: (
        0x3E942913,
        0x3F49F3A3,
        0x3F70D45F,
        0x3F7C094D,
        0x3F7F0173,
        0x3F7FC869,
        0x3F1C84D4,
        0x3F631031,
        0x3F783122,
        0x3F7E017B,
        0x3F7F8401,
        0x3F7FF039,
    ),
}

_HILBERT_RATE_BANDS = (
    (44_100, 48_000, 64_000, 80_000, 96_000),
    (176_400, 192_000, 208_000, 224_000, 240_000, 256_000, 288_000, 320_000, 352_000, 384_000),
)


def _f32(value: float) -> float:
    return _F32.unpack(_F32.pack(value))[0]


def _from_bits(bits: int) -> float:
    return _F32.unpack(_U32.pack(bits))[0]


def _to_bits(value: float) -> int:
    return _U32.unpack(_F32.pack(value))[0]


def _rsqrt_nr2(value: float) -> float:
    estimate = _from_bits(0x5F3759DF - (_to_bits(value) >> 1))
    for _ in range(2):
        correction = _f32(1.5 - _f32(_f32(_f32(0.5 * value) * estimate) * estimate))
        estimate = _f32(estimate * correction)
    return estimate


def _normalized_phasor(x: float, y: float) -> tuple[float, float, float]:
    x, y = _f32(x), _f32(y)
    maximum = max(abs(x), abs(y))
    if maximum == 0:
        raise ValueError("the reference requires a nonzero phasor")
    exponent = (_to_bits(maximum) >> 23) & 0xFF
    subnormal_restore = 1.0
    if exponent == 0:
        x, y = _f32(x * 2**24), _f32(y * 2**24)
        exponent = (_to_bits(max(abs(x), abs(y))) >> 23) & 0xFF
        subnormal_restore = 2**-24
    power = min(exponent - 127, 126)
    sx, sy = _f32(x * 2.0**-power), _f32(y * 2.0**-power)
    squared = _f32(_f32(sx * sx) + _f32(sy * sy))
    inverse = _rsqrt_nr2(squared)
    magnitude = _f32(_f32(squared * inverse) * 2.0**power)
    return _f32(sx * inverse), _f32(sy * inverse), _f32(magnitude * subnormal_restore)


def _half_angle(real: float, imag: float) -> float:
    real = min(1.0, max(-1.0, real))
    if real >= 0:
        h = _f32(0.5 * _f32(1.0 + real))
        return min(1.0, max(0.0, _f32(h * _rsqrt_nr2(h))))
    denominator = _f32(2.0 * _f32(1.0 - real))
    return min(1.0, max(0.0, _f32(abs(imag) * _rsqrt_nr2(denominator))))


def _half_angle_pair(real: float, imag: float) -> tuple[float, float]:
    real = min(1.0, max(-1.0, real))
    imag = min(1.0, max(-1.0, imag))
    h = _f32(0.5 * _f32(1.0 + abs(real)))
    inverse = _rsqrt_nr2(h)
    major = min(1.0, max(0.0, _f32(h * inverse)))
    minor = min(1.0, max(0.0, _f32(abs(imag) * _f32(0.5 * inverse))))
    return (major, minor) if real >= 0 else (minor, major)


def _crosses_cut(previous: tuple[float, float], current: tuple[float, float]) -> bool:
    px, py = previous
    x, y = current
    determinant = px * y - py * x
    if px * x + py * y < 0 and abs(determinant) <= 2**-18:
        return False
    return (py >= 0 > y and determinant > 0) or (py < 0 <= y and determinant < 0)


def _round_positive_fraction_to_f32(value: Fraction) -> float:
    if not 0 < value < 1:
        raise ValueError("unsupported coefficient")
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    if exponent < -126:
        raise ValueError("subnormal coefficient")
    significand, remainder = divmod(numerator << (23 - exponent), denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and significand & 1):
        significand += 1
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    bits = ((exponent + 127) << 23) | (significand - (1 << 23))
    return _from_bits(bits)


def _coefficients(rate: float, low: int, high: int) -> tuple[float, ...]:
    if not math.isfinite(rate) or not low <= rate <= high or low >= high:
        raise ValueError("invalid interpolation rate")
    exact_rate = Fraction.from_float(rate)
    left = Fraction(high) - exact_rate
    right = exact_rate - Fraction(low)
    return tuple(
        _round_positive_fraction_to_f32(
            (left * Fraction.from_float(_from_bits(a)) + right * Fraction.from_float(_from_bits(b)))
            / (high - low)
        )
        for a, b in zip(_HILBERT_PROFILE_BITS[low], _HILBERT_PROFILE_BITS[high], strict=True)
    )


def _profile_coefficients(rate: float) -> tuple[float, ...]:
    if not math.isfinite(rate):
        raise ValueError("unsupported effective rate")
    for anchors in _HILBERT_RATE_BANDS:
        for low, high in itertools.pairwise(anchors):
            if low <= rate <= high:
                return _coefficients(rate, low, high)
    raise ValueError("unsupported effective rate")


def _hilbert_response(
    rate: float, frequency: float, alpha: tuple[float, ...]
) -> tuple[complex, complex]:
    z1 = cmath.exp(-2j * math.pi * frequency / rate)
    z2 = z1 * z1
    i_response, q_response = 1 + 0j, z1
    for index in range(6):
        ai, aq = alpha[index], alpha[index + 6]
        i_response *= (ai - z2) / (1 - ai * z2)
        q_response *= (aq - z2) / (1 - aq * z2)
    return i_response, q_response


def _quadrature_error_degrees(rate: float, frequency: float, alpha: tuple[float, ...]) -> float:
    i_response, q_response = _hilbert_response(rate, frequency, alpha)
    return abs(math.degrees(cmath.phase(1j * q_response / i_response)))


def test_experimental_hilbert_profile_on_sampled_rates_and_frequencies() -> None:
    frequencies = [
        *[20.0 * (1000.0 ** (index / 255)) for index in range(256)],
        *[20.0 + 19_980.0 * index / 255 for index in range(256)],
    ]
    for anchors in _HILBERT_RATE_BANDS:
        for low, high in itertools.pairwise(anchors):
            for step in range(17):
                rate = low + (high - low) * step / 16
                alpha = _coefficients(rate, low, high)
                assert all(0.0 < value < 1.0 for value in alpha)
                assert (
                    max(
                        _quadrature_error_degrees(rate, frequency, alpha)
                        for frequency in frequencies
                    )
                    < 0.35
                )


def test_experimental_hilbert_reference_band_at_48_khz() -> None:
    alpha = tuple(_from_bits(bits) for bits in _HILBERT_PROFILE_BITS[48_000])
    assert (
        max(
            _quadrature_error_degrees(48_000, 35 * (630 ** (index / 255)), alpha)
            for index in range(256)
        )
        < 0.35
    )


def _step_transposed_section(
    coefficient: Fraction, state: tuple[Fraction, Fraction], sample: Fraction
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    newest, older = state
    output = coefficient * sample + older
    return output, (coefficient * output - sample, newest)


@pytest.mark.parametrize("index", range(12))
def test_hilbert_retune_copy_changes_constant_input_equilibrium(index: int) -> None:
    old = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[48_000][index]))
    new = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[96_000][index]))
    sample = Fraction(1)
    state = (-1 - old, -1 - old)
    copied_output, _ = _step_transposed_section(new, state, sample)
    assert copied_output + 1 == new - old != 0

    remapped = tuple(cell + (new - old) * -1 for cell in state)
    remapped_output, next_state = _step_transposed_section(new, (remapped[0], remapped[1]), sample)
    assert remapped_output == -1
    assert next_state == remapped


@pytest.mark.parametrize("index", range(12))
def test_hilbert_equal_cells_do_not_determine_history_consistent_remap(index: int) -> None:
    old = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[48_000][index]))
    new = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[96_000][index]))
    zero = Fraction()

    def run(samples: tuple[Fraction, ...]) -> tuple[tuple[Fraction, Fraction], list[Fraction]]:
        state = (zero, zero)
        outputs = []
        for sample in samples:
            output, state = _step_transposed_section(old, state, sample)
            outputs.append(output)
        return state, outputs

    empty_state, empty_outputs = run((zero, zero, zero))
    driven_samples = (Fraction(1), zero, -old)
    driven_state, driven_outputs = run(driven_samples)
    assert empty_state == driven_state == (zero, zero)
    assert empty_outputs[-1] == empty_outputs[-2] == driven_outputs[-2] == 0
    assert driven_outputs[-1] == -1

    remapped = (
        driven_state[0] + (new - old) * driven_outputs[-1],
        driven_state[1] + (new - old) * driven_outputs[-2],
    )
    assert remapped == (
        new * driven_outputs[-1] - driven_samples[-1],
        new * driven_outputs[-2] - driven_samples[-2],
    )
    first, state = _step_transposed_section(new, remapped, zero)
    second, _ = _step_transposed_section(new, state, zero)
    assert first == 0
    assert second == old - new != 0


@pytest.mark.parametrize(
    ("rate", "low", "high", "index", "expected"),
    [
        (49_200.0, 48_000, 64_000, 8, 0x3F600A2A),
        (54_800.0, 48_000, 64_000, 8, 0x3F61BA02),
        (48_400.0, 48_000, 64_000, 10, 0x3F7D03D4),
        (80_960.0, 80_000, 96_000, 3, 0x3F7219A2),
        (81_600.0, 80_000, 96_000, 4, 0x3F7C1BE4),
        (181_080.0, 176_400, 192_000, 4, 0x3F7DF2E0),
    ],
)
def test_hilbert_rate_interpolation_rounds_exact_ties_to_even(
    rate: float, low: int, high: int, index: int, expected: int
) -> None:
    actual = _coefficients(rate, low, high)[index]
    assert _to_bits(actual) == expected

    fraction = (rate - low) / (high - low)
    a = _from_bits(_HILBERT_PROFILE_BITS[low][index])
    b = _from_bits(_HILBERT_PROFILE_BITS[high][index])
    naive = _f32((1 - fraction) * a + fraction * b)
    assert _to_bits(naive) != expected


def test_hilbert_rate_interpolation_preserves_anchor_bits() -> None:
    for anchors in _HILBERT_RATE_BANDS:
        for low, high in itertools.pairwise(anchors):
            low_bits = tuple(map(_to_bits, _coefficients(float(low), low, high)))
            high_bits = tuple(map(_to_bits, _coefficients(float(high), low, high)))
            assert low_bits == _HILBERT_PROFILE_BITS[low]
            assert high_bits == _HILBERT_PROFILE_BITS[high]
            assert tuple(map(_to_bits, _profile_coefficients(float(low)))) == low_bits
            assert tuple(map(_to_bits, _profile_coefficients(float(high)))) == high_bits


def test_hilbert_exact_tie_and_adjacent_binary64_rates() -> None:
    rate = 49_200.0
    low = Fraction.from_float(_from_bits(0x3F600A2A))
    high = Fraction.from_float(_from_bits(0x3F600A2B))
    left = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[48_000][8]))
    right = Fraction.from_float(_from_bits(_HILBERT_PROFILE_BITS[64_000][8]))
    exact = (Fraction(14_800) * left + Fraction(1_200) * right) / 16_000
    assert exact == (low + high) / 2
    assert _to_bits(_profile_coefficients(math.nextafter(rate, -math.inf))[8]) == 0x3F600A2A
    assert _to_bits(_profile_coefficients(rate)[8]) == 0x3F600A2A
    assert _to_bits(_profile_coefficients(math.nextafter(rate, math.inf))[8]) == 0x3F600A2B


def test_hilbert_profile_anchor_table_binary_digest() -> None:
    """Verifies the 780 B H1 anchor table digest (`verify[dsp.op-m.hilbert.h1-anchors]`)."""
    rates = list(_HILBERT_PROFILE_BITS)
    assert rates == sorted(set(rates))
    for values in _HILBERT_PROFILE_BITS.values():
        assert len(values) == 12
    binary = b"".join(
        struct.pack("<13I", rate, *values) for rate, values in _HILBERT_PROFILE_BITS.items()
    )
    assert len(binary) == 780
    assert hashlib.sha256(binary).hexdigest() == (
        "bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352"
    )


def test_hilbert_exact_interpolation_has_bounded_integer_intermediates() -> None:
    for anchors in _HILBERT_RATE_BANDS:
        for low, high in itertools.pairwise(anchors):
            assert high - low <= 32_000
            for rate in (
                float(low),
                math.nextafter(float(low), math.inf),
                (low + high) / 2,
                math.nextafter(float(high), -math.inf),
                float(high),
            ):
                fraction = Fraction.from_float(rate)
                k = fraction.denominator.bit_length() - 1
                assert fraction.denominator == 1 << k
                assert k <= 37
                actual = _coefficients(rate, low, high)
                for index, (a, b) in enumerate(
                    zip(_HILBERT_PROFILE_BITS[low], _HILBERT_PROFILE_BITS[high], strict=True)
                ):
                    a_scaled = Fraction.from_float(_from_bits(a)) * (1 << 27)
                    b_scaled = Fraction.from_float(_from_bits(b)) * (1 << 27)
                    assert a_scaled.denominator == b_scaled.denominator == 1
                    assert (1 << 23) <= a_scaled.numerator < (1 << 27)
                    assert (1 << 23) <= b_scaled.numerator < (1 << 27)
                    left_weight = high * fraction.denominator - fraction.numerator
                    right_weight = fraction.numerator - low * fraction.denominator
                    numerator = left_weight * a_scaled.numerator + right_weight * b_scaled.numerator
                    denominator = (high - low) * fraction.denominator * (1 << 27)
                    assert 0 < numerator < denominator < (1 << 79)
                    assert numerator << 27 < 1 << 106
                    assert actual[index] == _round_positive_fraction_to_f32(
                        Fraction(numerator, denominator)
                    )


@pytest.mark.parametrize("rate", [math.nan, math.inf, -math.inf, 47_999.0, 64_001.0])
def test_hilbert_rate_interpolation_rejects_invalid_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="invalid interpolation rate"):
        _coefficients(rate, 48_000, 64_000)


@pytest.mark.parametrize(
    "rate",
    [math.nan, math.inf, 44_099.0, math.nextafter(96_000.0, math.inf), 176_399.0, 384_001.0],
)
def test_hilbert_profile_rejects_unsupported_effective_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="unsupported effective rate"):
        _profile_coefficients(rate)


def test_experimental_hilbert_group_delay_is_frequency_dependent() -> None:
    alpha = tuple(_from_bits(bits) for bits in _HILBERT_PROFILE_BITS[48_000])

    def delay_ms(frequency: float, branch: tuple[float, ...]) -> float:
        angle = 2 * math.pi * frequency / 48_000
        samples = sum(
            2 * (1 - coefficient**2) / (1 + coefficient**2 - 2 * coefficient * math.cos(2 * angle))
            for coefficient in branch
        )
        return samples * 1000 / 48_000

    assert 13.08 < delay_ms(20, alpha[:6]) < 13.10
    assert 0.428 < delay_ms(1000, alpha[:6]) < 0.430


@pytest.mark.parametrize("amplitude", [2**-125, 1.0e-30, 1.0e-12, 1.0e-7, 0.1, 100.0])
def test_scaled_octave_feature_preserves_quiet_and_loud_amplitudes(amplitude: float) -> None:
    for index in range(129):
        phase = -math.pi + 2 * math.pi * index / 128
        ux, uy, envelope = _normalized_phasor(
            amplitude * math.cos(phase), amplitude * math.sin(phase)
        )
        actual = envelope * _half_angle(ux, uy)
        expected = amplitude * abs(math.cos(phase / 2))
        assert abs(actual - expected) <= amplitude * 2.0e-5
        cosine, sine = _half_angle_pair(ux, uy)
        assert abs(cosine - _half_angle(ux, uy)) < 2.0e-7
        assert abs(envelope * cosine - expected) <= amplitude * 2.0e-5
        assert abs(envelope * sine - amplitude * abs(math.sin(phase / 2))) <= amplitude * 2.0e-5


def test_cut_crossing_uses_the_chord_not_the_last_real_component() -> None:
    previous = (math.cos(math.radians(150)), math.sin(math.radians(150)))
    current = (math.cos(math.radians(300)), math.sin(math.radians(300)))
    assert _crosses_cut(previous, current)
    assert not _crosses_cut((1.0, 0.0), (-0.5, 0.2))
    assert not _crosses_cut((1.0, 0.0), (-1.0, 1.0e-7))


def test_exact_axis_is_counted_once_in_either_direction() -> None:
    points = [(-1.0, -0.2), (-1.0, 0.0), (-1.0, 0.2)]
    for sequence in (points, list(reversed(points))):
        assert sum(_crosses_cut(a, b) for a, b in itertools.pairwise(sequence)) == 1


@pytest.mark.parametrize(
    ("sample_rate", "frequency"),
    [(48_000, 20), (48_000, 1000), (44_100, 20_000), (384_000, 20), (384_000, 20_000)],
)
def test_sheet_tracks_single_tone_over_multiple_periods(sample_rate: int, frequency: int) -> None:
    sheet = 1.0
    previous: tuple[float, float] | None = None
    for sample in range(math.ceil(2.5 * sample_rate / frequency)):
        phase = 0.13 + sample * 2 * math.pi * frequency / sample_rate
        current = (_f32(math.cos(phase)), _f32(math.sin(phase)))
        if previous is not None and _crosses_cut(previous, current):
            sheet = -sheet
        cosine, sine = _half_angle_pair(*current)
        sign = 1.0 if current[1] >= 0 else -1.0
        assert abs(sheet * cosine - math.cos(phase / 2)) < 2.0e-5
        assert abs(sheet * sign * sine - math.sin(phase / 2)) < 2.0e-5
        previous = current


def test_half_angle_pair_is_continuous_at_the_negative_real_axis() -> None:
    epsilon = 0.001
    for direction in (1, -1):
        sheet = 1.0
        points = [
            (math.cos(direction * (math.pi - epsilon)), math.sin(direction * (math.pi - epsilon))),
            (-1.0, 0.0),
            (math.cos(direction * (math.pi + epsilon)), math.sin(direction * (math.pi + epsilon))),
        ]
        values = []
        previous = None
        for point in points:
            if previous is not None and _crosses_cut(previous, point):
                sheet = -sheet
            cosine, sine = _half_angle_pair(*point)
            sign = 1.0 if point[1] >= 0 else -1.0
            values.append((sheet * cosine, sheet * sign * sine))
            previous = point
        assert all(abs(values[index + 1][0] - values[index][0]) < epsilon for index in range(2))
        assert all(abs(values[index + 1][1] - values[index][1]) < epsilon for index in range(2))


def test_single_half_angle_and_square_cannot_fit_every_sinusoidal_phase() -> None:
    samples = 65_536
    phase = [4 * math.pi * sample / samples for sample in range(samples)]
    cosine = [math.cos(value) for value in phase]
    target = [math.sin(value) for value in phase]
    half_square = [1.0 if math.cos(value - math.pi / 4) >= 0 else -1.0 for value in phase]
    quarter_square = [
        1.0 if math.cos(value / 2 - 3 * math.pi / 8) >= 0 else -1.0 for value in phase
    ]

    def dot(left: list[float], right: list[float]) -> float:
        return math.fsum(a * b for a, b in zip(left, right, strict=True)) / samples

    cc, cq, qq = dot(cosine, cosine), dot(cosine, half_square), dot(half_square, half_square)
    cy, qy = dot(cosine, target), dot(half_square, target)
    determinant = cc * qq - cq * cq
    a = (cy * qq - qy * cq) / determinant
    b = (qy * cc - cy * cq) / determinant
    residual = [y - a * c - b * q for y, c, q in zip(target, cosine, half_square, strict=True)]
    ratio = dot(residual, residual) / dot(target, target)
    exact_ratio = 1 - 4 / (math.pi**2 - 4)
    assert abs(ratio - exact_ratio) < 1.0e-4
    assert -5.1 < 10 * math.log10(ratio) < -4.9
    assert abs(dot(quarter_square, residual)) < 1.0e-4


def test_tracker_quadratures_and_square_features_have_the_intended_single_tone_rates() -> None:
    rate, frequency, count = 48_000, 1200, 6400
    q1 = q2 = 1.0
    sheet = 1.0
    previous = None
    features: list[tuple[float, float, float, float]] = []
    for sample in range(count):
        phase = 0.13 + 2 * math.pi * frequency * sample / rate
        point = (_f32(math.cos(phase)), _f32(math.sin(phase)))
        if previous is not None:
            if previous[0] <= 0 < point[0]:
                q1 = -q1
                if q1 == 1:
                    q2 = -q2
            if _crosses_cut(previous, point):
                sheet = -sheet
        cosine, sine = _half_angle_pair(*point)
        sign = 1.0 if point[1] >= 0 else -1.0
        features.append((sheet * cosine, sheet * sign * sine, q1, q2))
        previous = point

    def magnitude(column: int, output_frequency: float) -> float:
        projection = sum(
            row[column] * cmath.exp(-2j * math.pi * output_frequency * sample / rate)
            for sample, row in enumerate(features)
        )
        return 2 * abs(projection) / count

    assert 0.999 < magnitude(0, frequency / 2) < 1.001
    assert 0.999 < magnitude(1, frequency / 2) < 1.001
    assert 1.26 < magnitude(2, frequency / 2) < 1.29
    assert 1.26 < magnitude(3, frequency / 4) < 1.29
    assert magnitude(3, frequency / 2) < 1.0e-10


def _tracker_crossing_counts(points: list[tuple[float, float]]) -> tuple[int, int, int]:
    q1 = 1
    half_count = 0
    quarter_count = 0
    sheet_count = 0
    for previous, current in itertools.pairwise(points):
        if previous[0] <= 0 < current[0]:
            q1 = -q1
            half_count += 1
            if q1 == 1:
                quarter_count += 1
        if _crosses_cut(previous, current):
            sheet_count += 1
    return half_count, quarter_count, sheet_count


def test_tracker_mute_does_not_make_its_crossing_history_optional() -> None:
    points = [
        (math.cos(2 * math.pi * (index + 0.125) / 16), math.sin(2 * math.pi * (index + 0.125) / 16))
        for index in range(80)
    ]

    def run(*, pause_muted: bool) -> tuple[float, float, float]:
        previous: tuple[float, float] | None = None
        half, quarter, sheet = 1.0, 1.0, 1.0
        for index, point in enumerate(points):
            if pause_muted and 32 <= index < 56:
                continue
            if previous is not None:
                if previous[0] <= 0 < point[0]:
                    half = -half
                    if half > 0:
                        quarter = -quarter
                if _crosses_cut(previous, point):
                    sheet = -sheet
            previous = point
            if index == 56:
                return half, quarter, sheet
        raise AssertionError("reactivation sample was not processed")

    assert run(pause_muted=False) != run(pause_muted=True)


def _tracker_dc_step(value: float, previous: float, state: float) -> tuple[float, float]:
    pole = _from_bits(0x3F7FD51F)
    gain = _from_bits(0x3F7FEA90)
    output = _f32(_f32(gain * _f32(value - previous)) + _f32(pole * state))
    return value, output


def test_pausing_a_muted_tracker_preserves_a_stale_dc_blocker_state() -> None:
    gain = _from_bits(0x3F7FEA90)
    previous, state = 0.0, 0.0
    for _ in range(24_000):
        previous, state = _tracker_dc_step(0.5, previous, state)
    _, continuing = _tracker_dc_step(0.5, previous, state)
    stale = _f32(gain * 0.5)
    assert abs(continuing) < 1e-6
    assert stale > 0.49


def test_tracker_blocker_state_alias_can_reintroduce_return_to_send_feedback() -> None:
    instrument = [_f32(0.1)] * 8
    no_return = [0.0] * len(instrument)
    driven_return = [_f32(0.5)] * len(instrument)

    def run(returns: list[float], *, shared: bool) -> list[float]:
        send_state = (0.0, 0.0)
        return_state = (0.0, 0.0)
        sends = []
        for sample, return_sample in zip(instrument, returns, strict=True):
            send_state = _tracker_dc_step(sample, *send_state)
            sends.append(send_state[1])
            if shared:
                send_state = _tracker_dc_step(return_sample, *send_state)
            else:
                return_state = _tracker_dc_step(return_sample, *return_state)
        return sends

    assert run(no_return, shared=False) == run(driven_return, shared=False)
    assert run(no_return, shared=True) != run(driven_return, shared=True)


def test_shared_recursive_filter_is_advanced_once_before_fanout() -> None:
    first = _tracker_dc_step(_f32(0.1), 0.0, 0.0)
    duplicate_step = _tracker_dc_step(_f32(0.1), *first)
    assert first[1] != duplicate_step[1]

    left = _tracker_dc_step(_f32(0.2), *first)
    right = _tracker_dc_step(_f32(-0.2), *first)
    independent_left = _tracker_dc_step(_f32(0.2), *_tracker_dc_step(_f32(0.1), 0.0, 0.0))
    independent_right = _tracker_dc_step(_f32(-0.2), *_tracker_dc_step(_f32(0.1), 0.0, 0.0))
    assert left == independent_left
    assert right == independent_right
    assert left != right


def test_tracker_sign_history_changes_cross_terms_despite_equal_feature_energy() -> None:
    cycle = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    future = cycle + cycle[:3]

    def render(q1: float, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        previous = (1.0, 0.0)
        rows = []
        for current in points:
            if previous[0] <= 0 < current[0]:
                q1 = -q1
            rows.append((1.0, q1))
            previous = current
        return rows

    earlier = render(1.0, cycle)
    assert earlier[-1][1] == -1.0
    old_features = render(1.0, future)
    new_features = render(earlier[-1][1], future)
    target = (1.0,) * len(future)

    def gram(rows: list[tuple[float, float]]) -> tuple[float, float, float]:
        return (
            sum(left * left for left, _ in rows),
            sum(left * right for left, right in rows),
            sum(right * right for _, right in rows),
        )

    def correlation(rows: list[tuple[float, float]]) -> tuple[float, float]:
        return (
            sum(row[0] * y for row, y in zip(rows, target, strict=True)),
            sum(row[1] * y for row, y in zip(rows, target, strict=True)),
        )

    assert gram(old_features) == (7.0, -1.0, 7.0)
    assert gram(new_features) == (7.0, 1.0, 7.0)
    assert correlation(old_features) == (7.0, -1.0)
    assert correlation(new_features) == (7.0, 1.0)


def test_tracker_finite_replay_cannot_recover_discrete_winding_history() -> None:
    diagonal = math.sqrt(0.5)
    cycle = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    two_turns = [(1.0, 0.0)] + cycle * 2
    one_turn = [
        (1.0, 0.0),
        (diagonal, diagonal),
        (0.0, 1.0),
        (-diagonal, diagonal),
        (-1.0, 0.0),
        (-diagonal, -diagonal),
        (0.0, -1.0),
        (diagonal, -diagonal),
        (1.0, 0.0),
    ]
    suffix = cycle * 128

    def signs(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
        q1 = sheet = 1
        states = []
        for previous, current in itertools.pairwise(points):
            if previous[0] <= 0 < current[0]:
                q1 = -q1
            if _crosses_cut(previous, current):
                sheet = -sheet
            states.append((q1, sheet))
        return states

    assert len(two_turns) == len(one_turn)
    assert two_turns[-1] == one_turn[-1]
    fast = signs(two_turns + suffix)
    slow = signs(one_turn + suffix)
    assert fast[len(two_turns) - 2] == (1, 1)
    assert slow[len(one_turn) - 2] == (-1, -1)
    for (q1_a, sheet_a), (q1_b, sheet_b) in zip(
        fast[len(two_turns) - 1 :], slow[len(one_turn) - 1 :], strict=True
    ):
        assert q1_a == -q1_b
        assert sheet_a == -sheet_b


def test_tracker_crossing_uses_raw_i_when_normalization_is_flushed() -> None:
    smallest_normal = _from_bits(0x00800000)
    normalized_i, _, _ = _normalized_phasor(smallest_normal, 1.0)
    normal_positive_i, _, _ = _normalized_phasor(1.0, 1.0)
    normal_negative_i, _, _ = _normalized_phasor(-1.0, 1.0)
    assert 0.0 < normalized_i < smallest_normal

    flushed_i = (
        math.copysign(0.0, normalized_i)
        if _to_bits(normalized_i) & 0x7F800000 == 0
        else normalized_i
    )
    assert flushed_i == 0.0
    assert not (smallest_normal <= 0.0 < 1.0)
    assert flushed_i <= 0.0 < normal_positive_i
    assert -1.0 <= 0.0 < smallest_normal
    assert not (normal_negative_i <= 0.0 < flushed_i)


@pytest.mark.parametrize("signed_zero", [0.0, -0.0])
def test_tracker_detector_handles_axis_zero_and_silence(signed_zero: float) -> None:
    def step(
        state: tuple[int, bool, bool], i_value: float, q_value: float
    ) -> tuple[int, bool, bool]:
        sign, valid, was_nonpositive = state
        if i_value == 0.0 and q_value == 0.0:
            return sign, False, was_nonpositive
        if valid and was_nonpositive and i_value > 0.0:
            sign = -sign
        return sign, True, i_value <= 0.0

    start = (1, False, False)
    on_axis = step(start, signed_zero, 1.0)
    assert on_axis == (1, True, True)
    rising = step(on_axis, 1.0, 1.0)
    assert rising == (-1, True, False)
    assert step(rising, 1.0, 1.0) == rising

    negative = step(start, -1.0, 1.0)
    silence = step(negative, signed_zero, signed_zero)
    assert silence == (1, False, True)
    assert step(silence, 1.0, 1.0) == (1, True, False)


def test_tracker_direct_only_local_mix_does_not_bypass_the_output_head() -> None:
    source = 0.5
    tracker_feature = 0.25
    local_mix = source + 0.0 * tracker_feature
    head_output = local_mix + 0.5 * tracker_feature
    assert local_mix == source
    assert head_output == 0.625
    assert head_output != source


def _dc_blocker_response(rate: float, frequency: float) -> complex:
    angle_ref = 2 * math.pi * 5 / 48_000
    pole_ref = _f32(math.cos(angle_ref) / (1 + math.sin(angle_ref)))
    pole = _f32(pole_ref ** (48_000 / rate))
    gain = _f32((1 + pole) / 2)
    z1 = cmath.exp(-2j * math.pi * frequency / rate)
    return gain * (1 - z1) / (1 - pole * z1)


def test_tracker_dc_profile_reference_coefficient_bits() -> None:
    angle = 2 * math.pi * 5 / 48_000
    pole = _f32(math.cos(angle) / (1 + math.sin(angle)))
    gain = _f32((1 + pole) / 2)
    assert _to_bits(pole) == 0x3F7FD51F
    assert _to_bits(gain) == 0x3F7FEA90


def test_tracker_feature_slots_fit_within_lite_prefix() -> None:
    """Verifies T1 feature slots 4..11 (`verify[dsp.op-m.tracker.toggle-sheet]`,
    `verify[dsp.head.feature-bus-slots]`).
    """
    tracker_slots = set(range(4, 12))
    stage3_lite_modal_slots = set(range(0, 4))
    stage3_lite_fir_slots = {62, 63}
    stage3_lite = stage3_lite_modal_slots | stage3_lite_fir_slots
    lite_other_stage_slots = set(range(4, 62))
    assert tracker_slots.isdisjoint(stage3_lite)
    assert tracker_slots <= lite_other_stage_slots
    assert all(tracker_slots < set(range(width)) for width in (64, 128, 192, 256))


def test_tracker_dc_offset_can_suppress_both_octave_dividers() -> None:
    rate, frequency = 48_000, 1000
    alpha = tuple(_from_bits(bits) for bits in _HILBERT_PROFILE_BITS[rate])
    i_response, q_response = _hilbert_response(rate, frequency, alpha)
    bias, amplitude = 0.25, 0.1
    count = 40 * rate // frequency
    points = []
    for sample in range(count + 1):
        carrier = cmath.exp(2j * math.pi * frequency * sample / rate)
        points.append(
            (
                bias + amplitude * (i_response * carrier).real,
                bias + amplitude * (q_response * carrier).real,
            )
        )
    assert min(point[0] for point in points) > 0.14
    assert _tracker_crossing_counts(points) == (0, 0, 0)

    blocker = _dc_blocker_response(rate, frequency)
    conditioned = []
    for sample in range(count + 1):
        carrier = cmath.exp(2j * math.pi * frequency * sample / rate)
        conditioned.append(
            (
                amplitude * (i_response * blocker * carrier).real,
                amplitude * (q_response * blocker * carrier).real,
            )
        )
    half, quarter, sheet = _tracker_crossing_counts(conditioned)
    assert 39 <= half <= 41
    assert 19 <= quarter <= 20
    assert 39 <= sheet <= 41


@pytest.mark.parametrize("rate", [44_100, 48_000, 96_000, 176_400, 384_000])
def test_tracker_dc_candidate_keeps_the_ideal_20_hz_response_near_unity(rate: int) -> None:
    assert _dc_blocker_response(rate, 0) == 0
    gain_db = 20 * math.log10(abs(_dc_blocker_response(rate, 20)))
    assert -0.27 < gain_db < -0.25
