"""Independent references for an unselected energy-preserving delay."""

from __future__ import annotations

import math
from fractions import Fraction


def _projection_step(
    state: tuple[Fraction, Fraction],
    direction: tuple[Fraction, Fraction],
    value: Fraction,
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    output = sum((x * r for x, r in zip(state, direction, strict=True)), Fraction())
    updated = tuple(x + r * (value - output) for x, r in zip(state, direction, strict=True))
    return output, (updated[0], updated[1])


def test_projected_feedback_has_exact_unforced_energy_certificate() -> None:
    directions = (
        (Fraction(1), Fraction(0)),
        (Fraction(3, 5), Fraction(4, 5)),
        (Fraction(4, 5), Fraction(3, 5)),
        (Fraction(0), Fraction(1)),
    )
    states = [(Fraction(index + 1), Fraction(2 - index)) for index in range(4)]
    gain = Fraction(999, 1000)
    total_output_energy = Fraction()
    initial_energy = sum((x * x for state in states for x in state), Fraction())
    for frame in range(24):
        chosen = [directions[(frame + index) % len(directions)] for index in range(4)]
        outputs = [
            sum((x * r for x, r in zip(state, direction, strict=True)), Fraction())
            for state, direction in zip(states, chosen, strict=True)
        ]
        mixed = [output - sum(outputs, Fraction()) / 2 for output in outputs]
        before = sum((x * x for state in states for x in state), Fraction())
        states = [
            _projection_step(state, direction, gain * value)[1]
            for state, direction, value in zip(states, chosen, mixed, strict=True)
        ]
        after = sum((x * x for state in states for x in state), Fraction())
        output_energy = sum((x * x for x in outputs), Fraction())
        assert after == before - (1 - gain**2) * output_energy
        total_output_energy += output_energy
    assert total_output_energy <= initial_energy / (1 - gain**2)


def _bounded_drive_projection_pulse(frames: int) -> float:
    first, second = 1.0, -1.0
    injection = 0.1
    smallest_dot = math.cos(0.1)
    previous_direction: tuple[float, float] | None = None
    for _ in range(frames):
        positive, negative = first, -second
        radius = math.hypot(positive, negative)
        direction = (negative / radius, positive / radius)
        if previous_direction is not None:
            dot = sum(x * y for x, y in zip(direction, previous_direction, strict=True))
            assert dot >= smallest_dot
        previous_direction = direction
        output = first * direction[0] + second * direction[1]
        value = -injection if positive >= negative else injection
        first += direction[0] * (value - output)
        second += direction[1] * (value - output)
        assert first > 0
        assert second < 0
    radius = math.hypot(first, second)
    assert math.isclose(radius * radius, 2 + frames * injection**2, rel_tol=1e-10)
    direction = (-second / radius, first / radius)
    rotation = 0.1
    rotated = (
        math.cos(rotation) * direction[0] - math.sin(rotation) * direction[1],
        math.sin(rotation) * direction[0] + math.cos(rotation) * direction[1],
    )
    assert min(rotated) >= 0
    output = first * rotated[0] + second * rotated[1]
    assert math.isclose(abs(output), radius * math.sin(rotation), rel_tol=1e-10)
    return abs(output)


def test_passive_variable_projection_does_not_imply_uniform_bibo() -> None:
    short = _bounded_drive_projection_pulse(10_000)
    long = _bounded_drive_projection_pulse(100_000)
    assert short > 1
    assert long > 3 * short
