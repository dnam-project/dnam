"""Packed control-projection examples, not production loader tests."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import pytest


@dataclass(frozen=True)
class _Range:
    control_id: str
    degree: int
    nodes: tuple[float, ...]


@dataclass(frozen=True)
class _Switch:
    control_id: str
    options: tuple[str, ...]
    reference: str
    active: tuple[str, ...]


@dataclass(frozen=True)
class _Contributor:
    control_id: str
    option_id: str | None
    lanes: int


@dataclass(frozen=True)
class _Record:
    base_main: tuple[Fraction, ...]
    embedding: tuple[Fraction, ...]
    base_factors: tuple[Fraction, ...]
    extra_main: tuple[Fraction, ...]
    extra_factors: tuple[Fraction, ...]


_Control = _Range | _Switch
_Pair = tuple[str, str]
_MAX_U64 = (1 << 64) - 1


def _contributors(
    controls: tuple[_Control, ...], represented: tuple[_Pair, ...]
) -> tuple[_Contributor, ...]:
    ids = tuple(control.control_id for control in controls)
    if not all(ids) or len(set(ids)) != len(ids) or len(set(represented)) != len(represented):
        raise ValueError("duplicate or empty identity")
    selected = set(represented)
    ordered: list[_Pair] = []
    contributors: list[_Contributor] = []
    for control in controls:
        if isinstance(control, _Range):
            if (
                type(control.degree) is not int
                or not 0 <= control.degree <= 3
                or len(control.nodes) < control.degree
            ):
                raise ValueError("invalid active degree")
            active_nodes = control.nodes[: control.degree]
            if any(not math.isfinite(node) or abs(node) > 1 for node in active_nodes):
                raise ValueError("invalid anchor")
            if len(set(active_nodes)) != len(active_nodes):
                raise ValueError("duplicate anchor")
            if control.degree:
                contributors.append(_Contributor(control.control_id, None, control.degree))
            continue
        if (
            not control.options
            or not all(control.options)
            or len(set(control.options)) != len(control.options)
            or control.reference not in control.options
            or not control.active
            or len(set(control.active)) != len(control.active)
            or not set(control.active) <= set(control.options)
        ):
            raise ValueError("invalid switch")
        for option in control.options:
            pair = (control.control_id, option)
            if option != control.reference and pair in selected:
                ordered.append(pair)
                contributors.append(_Contributor(control.control_id, option, 1))
            if option != control.reference and option in control.active and pair not in selected:
                raise ValueError("missing active option")
    if tuple(ordered) != represented:
        raise ValueError("invalid or unordered option mapping")
    return tuple(contributors)


def _size(lanes: int, *, ultra: bool) -> int:
    if not 1 <= lanes <= 3:
        raise ValueError("invalid contributor lane count")
    return 4 * ((8 if ultra else 9) * lanes + 4)


def _encode(lanes: tuple[int, ...], *, ultra: bool) -> bytes:
    values: list[float] = []
    for index, count in enumerate(lanes):
        for component in range(_size(count, ultra=ultra) // 4):
            values.append((1000 * (index + 1) + component + (500 if ultra else 0)) / 32)
    return struct.pack(f"<{len(values)}f", *values)


def _decode_record(base: bytes, extra: bytes | None, lanes: int) -> _Record:
    if len(base) != _size(lanes, ultra=False) or (
        extra is not None and len(extra) != _size(lanes, ultra=True)
    ):
        raise ValueError("incorrect logical record length")
    base_values = struct.unpack(f"<{len(base) // 4}f", base)
    extra_values = () if extra is None else struct.unpack(f"<{len(extra) // 4}f", extra)
    if not all(math.isfinite(value) for value in (*base_values, *extra_values)):
        raise ValueError("nonfinite coefficient")
    base_exact = tuple(map(Fraction.from_float, base_values))
    extra_exact = tuple(map(Fraction.from_float, extra_values))
    main_end = 8 * lanes
    return _Record(
        base_exact[:main_end],
        base_exact[main_end : main_end + lanes],
        base_exact[main_end + lanes :],
        extra_exact[:main_end],
        extra_exact[main_end:],
    )


def _decode_stream(base: bytes, extra: bytes | None, lanes: tuple[int, ...]) -> tuple[_Record, ...]:
    if len(base) != sum(_size(count, ultra=False) for count in lanes):
        raise ValueError("incorrect base tensor length")
    if extra is not None and len(extra) != sum(_size(count, ultra=True) for count in lanes):
        raise ValueError("incorrect delta tensor length")
    records: list[_Record] = []
    base_offset = 0
    extra_offset = 0
    for count in lanes:
        base_end = base_offset + _size(count, ultra=False)
        extra_end = extra_offset + _size(count, ultra=True)
        delta = None if extra is None else extra[extra_offset:extra_end]
        records.append(_decode_record(base[base_offset:base_end], delta, count))
        base_offset, extra_offset = base_end, extra_end
    return tuple(records)


def _packed_raw(
    records: tuple[_Record, ...], bases: tuple[tuple[Fraction, ...], ...], *, ultra: bool
) -> tuple[Fraction, ...]:
    dimensions, rank = (16, 8) if ultra else (8, 4)
    if ultra and any(not record.extra_factors for record in records):
        raise ValueError("Ultra projection is absent")
    main = [Fraction()] * dimensions
    factors = [[] for _ in range(rank)]
    for record, basis in zip(records, bases, strict=True):
        main_blocks = (record.base_main, record.extra_main) if ultra else (record.base_main,)
        for block_index, block in enumerate(main_blocks):
            for lane, value in enumerate(basis):
                for coordinate in range(8):
                    main[8 * block_index + coordinate] += value * block[8 * lane + coordinate]
        embedding = sum(
            (value * weight for value, weight in zip(basis, record.embedding, strict=True)),
            Fraction(),
        )
        weights = record.base_factors + record.extra_factors if ultra else record.base_factors
        for factor, weight in enumerate(weights):
            factors[factor].append(embedding * weight)
    interactions = [
        (sum(values, Fraction()) ** 2 - sum((value * value for value in values), Fraction())) / 2
        for values in factors
    ]
    return tuple(
        main[coordinate]
        + sum(
            (
                Fraction((coordinate + factor) % 3 - 1, 4) * interactions[factor]
                for factor in range(rank)
            ),
            Fraction(),
        )
        for coordinate in range(dimensions)
    )


def _dense_raw(
    records: tuple[_Record, ...],
    contributors: tuple[_Contributor, ...],
    bases: tuple[tuple[Fraction, ...], ...],
    *,
    ultra: bool,
) -> tuple[Fraction, ...]:
    dimensions, rank = (16, 8) if ultra else (8, 4)
    total = [Fraction()] * dimensions
    factors = [[] for _ in range(rank)]
    for record, contributor, basis in zip(records, contributors, bases, strict=True):
        positions = (3,) if contributor.option_id is not None else tuple(range(contributor.lanes))
        dense_basis = [Fraction()] * 4
        dense_embedding = [Fraction()] * 4
        dense_main = [[Fraction()] * 4 for _ in range(dimensions)]
        for lane, position in enumerate(positions):
            dense_basis[position] = basis[lane]
            dense_embedding[position] = record.embedding[lane]
            for coordinate in range(dimensions):
                block = record.base_main if coordinate < 8 else record.extra_main
                dense_main[coordinate][position] = block[8 * lane + coordinate % 8]
        embedding = sum(
            (x * w for x, w in zip(dense_basis, dense_embedding, strict=True)), Fraction()
        )
        for coordinate in range(dimensions):
            total[coordinate] += sum(
                (x * w for x, w in zip(dense_basis, dense_main[coordinate], strict=True)),
                Fraction(),
            )
        weights = record.base_factors + record.extra_factors if ultra else record.base_factors
        for factor, weight in enumerate(weights):
            factors[factor].append(embedding * weight)
    for coordinate in range(dimensions):
        for factor, values in enumerate(factors):
            coefficient = Fraction((coordinate + factor) % 3 - 1, 4)
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    total[coordinate] += coefficient * left * right
    return tuple(total)


def _check_extent(start: int, length: int, tier_start: int, tier_end: int) -> None:
    if any(value < 0 or value > _MAX_U64 for value in (start, length, tier_start, tier_end)):
        raise ValueError("invalid extent")
    if length > _MAX_U64 - start:
        raise ValueError("overflow")
    if start % 64 or start < tier_start or start + length > tier_end:
        raise ValueError("misaligned or outside tier")


def _check_nonoverlapping(extents: tuple[tuple[int, int], ...]) -> None:
    ordered = sorted(extents)
    if any(start + length > next_start for (start, length), (next_start, _) in pairwise(ordered)):
        raise ValueError("overlapping tensors")


def _check_padding(data: bytes, start: int, end: int) -> None:
    if any(data[start:end]):
        raise ValueError("nonzero alignment padding")


def _read_matrix(data: bytes, count: int) -> tuple[Fraction, ...]:
    if len(data) != 4 * count:
        raise ValueError("incorrect matrix length")
    values = struct.unpack(f"<{count}f", data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite matrix coefficient")
    return tuple(map(Fraction.from_float, values))


def _interaction_required(contributors: tuple[_Contributor, ...]) -> bool:
    return len({contributor.control_id for contributor in contributors}) >= 2


def _check_presence(
    count: int,
    creator_max: int,
    retained: int,
    *,
    base_entry: bool,
    extra_entry: bool,
    base_bytes: bool,
    extra_bytes: bool,
) -> None:
    if count < 0 or not 0 <= retained <= creator_max <= 3:
        raise ValueError("invalid tier or count")
    if base_entry != (count > 0) or extra_entry != (count > 0 and creator_max == 3):
        raise ValueError("invalid creator directory")
    if base_bytes != (count > 0) or extra_bytes != (count > 0 and retained == 3):
        raise ValueError("invalid retained payload")


def _controls() -> tuple[_Control, ...]:
    return (
        _Range("gain", 3, (0.0, -1.0, 1.0)),
        _Range("tone", 2, (0.0, 1.0)),
        _Range("fixed", 0, ()),
        _Switch("mode", ("off", "on", "bright"), "off", ("off", "on", "bright")),
    )


def test_packed_records_have_expected_order_sizes_and_unaligned_offsets() -> None:
    contributors = _contributors(_controls(), (("mode", "on"), ("mode", "bright")))
    assert tuple((item.control_id, item.option_id, item.lanes) for item in contributors) == (
        ("gain", None, 3),
        ("tone", None, 2),
        ("mode", "on", 1),
        ("mode", "bright", 1),
    )
    lanes = tuple(item.lanes for item in contributors)
    assert tuple(_size(count, ultra=False) for count in lanes) == (124, 88, 52, 52)
    assert tuple(_size(count, ultra=True) for count in lanes) == (112, 80, 48, 48)
    base, extra = _encode(lanes, ultra=False), _encode(lanes, ultra=True)
    assert (len(base), len(extra)) == (316, 288)
    assert (124 % 16, 212 % 16, 264 % 16) == (12, 4, 8)
    records = _decode_stream(base, extra, lanes)
    assert records[0].base_main[0] == Fraction(1000, 32)
    assert records[1].base_main[0] == Fraction(2000, 32)
    assert records[2].embedding[0] == Fraction(3008, 32)
    assert records[3].extra_factors[-1] == Fraction(4511, 32)


def test_packed_profile_matches_dense_logical_projection_in_exact_arithmetic() -> None:
    contributors = _contributors(_controls(), (("mode", "on"), ("mode", "bright")))
    lanes = tuple(item.lanes for item in contributors)
    records = _decode_stream(_encode(lanes, ultra=False), _encode(lanes, ultra=True), lanes)
    bases = (
        (Fraction(1, 2), Fraction(-1, 4), Fraction(1, 8)),
        (Fraction(2, 3), Fraction(1, 6)),
        (Fraction(1),),
        (Fraction(0),),
    )
    for ultra in (False, True):
        assert _packed_raw(records, bases, ultra=ultra) == _dense_raw(
            records, contributors, bases, ultra=ultra
        )
    assert _packed_raw(records, bases, ultra=False) != _packed_raw(records, bases, ultra=True)[:8]


def test_reference_mask_and_degree_zero_do_not_remove_physical_descriptors() -> None:
    controls: tuple[_Control, ...] = (
        _Range("fixed", 0, ()),
        _Switch("mode", ("off", "on"), "off", ("on",)),
        _Switch("only", ("off",), "off", ("off",)),
    )
    assert _contributors(controls, (("mode", "on"),)) == (_Contributor("mode", "on", 1),)
    assert len(controls) == 3
    assert _contributors((), ()) == ()


def test_masked_options_may_be_represented_or_omitted() -> None:
    switch = _Switch("mode", ("off", "on", "bright"), "off", ("on",))
    assert _contributors((switch,), (("mode", "on"),)) == (_Contributor("mode", "on", 1),)
    assert _contributors((switch,), (("mode", "on"), ("mode", "bright"))) == (
        _Contributor("mode", "on", 1),
        _Contributor("mode", "bright", 1),
    )


def test_alternate_options_of_one_switch_do_not_interact() -> None:
    switch = _Switch("mode", ("off", "on", "bright"), "off", ("off", "on", "bright"))
    contributors = _contributors((switch,), (("mode", "on"), ("mode", "bright")))
    assert len(contributors) == 2
    lanes = (1, 1)
    records = _decode_stream(_encode(lanes, ultra=False), _encode(lanes, ultra=True), lanes)
    for selected in (0, 1):
        bases = tuple((Fraction(int(index == selected)),) for index in range(2))
        for ultra in (False, True):
            output = _packed_raw(records, bases, ultra=ultra)
            active = records[selected]
            expected = active.base_main + active.extra_main if ultra else active.base_main
            assert output == expected
    assert not _interaction_required(contributors)
    assert _interaction_required((*contributors, _Contributor("gain", None, 1)))


@pytest.mark.parametrize(
    ("represented", "message"),
    [
        ((("mode", "off"), ("mode", "on"), ("mode", "bright")), "invalid or unordered"),
        ((("mode", "on"), ("mode", "on"), ("mode", "bright")), "duplicate or empty"),
        ((("mode", "bright"), ("mode", "on")), "invalid or unordered"),
        ((("mode", "on"),), "missing active option"),
        ((("gain", "on"), ("mode", "on"), ("mode", "bright")), "invalid or unordered"),
        ((("other", "on"), ("mode", "on"), ("mode", "bright")), "invalid or unordered"),
        ((("mode", "on"), ("mode", "bright"), ("mode", "missing")), "invalid or unordered"),
    ],
)
def test_malformed_option_mappings_are_rejected(
    represented: tuple[_Pair, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _contributors(_controls(), represented)


@pytest.mark.parametrize(
    ("controls", "message"),
    [
        ((_Range("gain", 4, (0.0, 1.0, -1.0, 0.5)),), "invalid active degree"),
        ((_Range("gain", 2, (0.0, 0.0)),), "duplicate anchor"),
        ((_Range("gain", 1, (math.nan,)),), "invalid anchor"),
        ((_Range("gain", 1, (0.0,)), _Range("gain", 0, ())), "duplicate or empty"),
        ((_Switch("mode", ("off", "on"), "missing", ("on",)),), "invalid switch"),
        ((_Switch("mode", ("off", "on"), "off", ()),), "invalid switch"),
    ],
)
def test_malformed_descriptors_are_rejected(controls: tuple[_Control, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _contributors(controls, ())


def test_truncated_and_nonfinite_projection_records_are_rejected() -> None:
    lanes = (3, 2, 1, 1)
    base, extra = _encode(lanes, ultra=False), _encode(lanes, ultra=True)
    with pytest.raises(ValueError, match="base tensor length"):
        _decode_stream(base[:-1], extra, lanes)
    with pytest.raises(ValueError, match="delta tensor length"):
        _decode_stream(base, extra[:-4], lanes)
    invalid = struct.pack("<I", 0x7F80_0000) + base[4:]
    with pytest.raises(ValueError, match="nonfinite"):
        _decode_stream(invalid, extra, lanes)


def test_mixer_wire_blocks_map_factor_and_coordinate_without_transposition() -> None:
    base = _read_matrix(struct.pack("<32f", *(float(i) for i in range(1, 33))), 32)
    extra = _read_matrix(struct.pack("<96f", *(float(i) for i in range(101, 197))), 96)

    def coefficient(coordinate: int, factor: int) -> Fraction:
        if coordinate < 8 and factor < 4:
            return base[factor * 8 + coordinate]
        if coordinate < 8:
            return extra[(factor - 4) * 8 + coordinate]
        block = 1 if factor < 4 else 2
        return extra[(block * 4 + factor % 4) * 8 + coordinate - 8]

    assert coefficient(0, 0) == 1
    assert coefficient(7, 3) == 32
    assert coefficient(0, 4) == 101
    assert coefficient(7, 7) == 132
    assert coefficient(8, 0) == 133
    assert coefficient(15, 3) == 164
    assert coefficient(8, 4) == 165
    assert coefficient(15, 7) == 196
    factors = tuple(Fraction(index + 1) for index in range(8))
    for coordinate in range(16):
        from_logical_matrix = sum(
            (coefficient(coordinate, factor) * factors[factor] for factor in range(8)), Fraction()
        )
        if coordinate < 8:
            from_blocks = sum(
                (
                    base[factor * 8 + coordinate] * factors[factor]
                    + extra[factor * 8 + coordinate] * factors[factor + 4]
                    for factor in range(4)
                ),
                Fraction(),
            )
        else:
            from_blocks = sum(
                (
                    extra[(4 + factor) * 8 + coordinate - 8] * factors[factor]
                    + extra[(8 + factor) * 8 + coordinate - 8] * factors[factor + 4]
                    for factor in range(4)
                ),
                Fraction(),
            )
        assert from_logical_matrix == from_blocks

    with pytest.raises(ValueError, match="matrix length"):
        _read_matrix(bytes(127), 32)
    with pytest.raises(ValueError, match="nonfinite matrix"):
        _read_matrix(struct.pack("<I", 0x7FC0_0000) + bytes(124), 32)


def test_dispatch_wire_order_keeps_return_aware_dynamics_out_of_send_rows() -> None:
    base = _read_matrix(struct.pack("<256f", *(float(i) for i in range(1, 257))), 256)
    extra = _read_matrix(struct.pack("<64f", *(float(i) for i in range(1001, 1065))), 64)
    static = tuple(Fraction(index + 1) for index in range(8))
    send = static + tuple(Fraction(index + 1) for index in range(8))
    downstream = static + tuple(Fraction(index + 11) for index in range(8))
    stage_delta = tuple(Fraction(index + 2) for index in range(4))

    def dispatch(
        q_send: tuple[Fraction, ...], q_down: tuple[Fraction, ...]
    ) -> tuple[Fraction, ...]:
        return tuple(
            sum(
                (base[j * 16 + i] * (q_send if i < 5 else q_down)[j] for j in range(16)),
                Fraction(),
            )
            + sum((extra[j * 16 + i] * stage_delta[j] for j in range(4)), Fraction())
            for i in range(16)
        )

    output = dispatch(send, downstream)
    changed_downstream = dispatch(send, static + (Fraction(100),) * 8)
    changed_send = dispatch(static + (Fraction(100),) * 8, downstream)
    assert output[:5] == changed_downstream[:5]
    assert output[5:] == changed_send[5:]
    assert output[5:] != changed_downstream[5:]
    assert output[:5] != changed_send[:5]
    assert (base[0], base[15], base[16]) == (1, 16, 17)
    assert (extra[0], extra[15], extra[16]) == (1001, 1016, 1017)


def test_creator_directory_survives_stripping_without_absent_tier_bytes() -> None:
    _check_presence(4, 3, 3, base_entry=True, extra_entry=True, base_bytes=True, extra_bytes=True)
    _check_presence(4, 3, 2, base_entry=True, extra_entry=True, base_bytes=True, extra_bytes=False)
    _check_presence(4, 2, 0, base_entry=True, extra_entry=False, base_bytes=True, extra_bytes=False)
    _check_presence(
        0, 3, 3, base_entry=False, extra_entry=False, base_bytes=False, extra_bytes=False
    )
    with pytest.raises(ValueError, match="creator directory"):
        _check_presence(
            4, 3, 2, base_entry=True, extra_entry=False, base_bytes=True, extra_bytes=False
        )
    with pytest.raises(ValueError, match="retained payload"):
        _check_presence(
            4, 3, 2, base_entry=True, extra_entry=True, base_bytes=True, extra_bytes=True
        )
    with pytest.raises(ValueError, match="retained payload"):
        _check_presence(
            4, 3, 3, base_entry=True, extra_entry=True, base_bytes=True, extra_bytes=False
        )


def test_aligned_slices_have_logical_lengths_and_preserve_the_stripped_prefix() -> None:
    lanes = (3, 2, 1, 1)
    base, extra = _encode(lanes, ultra=False), _encode(lanes, ultra=True)
    base_start = 64
    base_end = base_start + len(base)
    lite_end = (base_end + 63) & ~63
    ultra_start = lite_end + 128
    ultra_end = (ultra_start + len(extra) + 63) & ~63
    _check_extent(base_start, len(base), 64, lite_end)
    _check_extent(ultra_start, len(extra), ultra_start, ultra_end)
    _check_nonoverlapping(((base_start, len(base)), (ultra_start, len(extra))))
    original_prefix = bytes(64) + base + bytes(lite_end - base_end) + b"F" * 64 + b"P" * 64
    suffix = b"mutable"
    original = original_prefix + extra + bytes(ultra_end - ultra_start - len(extra)) + suffix
    stripped = original[:ultra_start] + suffix
    assert stripped[:lite_end] == original[:lite_end]
    assert stripped == original_prefix + suffix
    full_records = _decode_stream(
        original[base_start:base_end], original[ultra_start : ultra_start + len(extra)], lanes
    )
    stripped_records = _decode_stream(stripped[base_start:base_end], None, lanes)
    bases = tuple((Fraction(1),) * count for count in lanes)
    assert _packed_raw(full_records, bases, ultra=False) == _packed_raw(
        stripped_records, bases, ultra=False
    )
    with pytest.raises(ValueError, match="Ultra projection is absent"):
        _packed_raw(stripped_records, bases, ultra=True)
    _check_padding(original, base_end, lite_end)
    _check_padding(original, ultra_start + len(extra), ultra_end)
    corrupt_padding = original[:base_end] + b"\x01" + original[base_end + 1 :]
    with pytest.raises(ValueError, match="nonzero alignment padding"):
        _check_padding(corrupt_padding, base_end, lite_end)
    with pytest.raises(ValueError, match="overlapping tensors"):
        _check_nonoverlapping(((base_start, len(base)), (320, 64)))
    with pytest.raises(ValueError, match="outside tier"):
        _check_extent(65, len(base), 64, lite_end)
    with pytest.raises(ValueError, match="outside tier"):
        _check_extent(base_start, len(base) + 5, 64, lite_end)
    with pytest.raises(ValueError, match="overflow"):
        _check_extent(_MAX_U64 - 63, 128, 0, _MAX_U64)
