"""Open checkpoint framing examples with a placeholder, not a signature verifier."""

from __future__ import annotations

import hashlib
import struct

import pytest

from test_open_layout_spec_references import (
    _align64,
    _append_mutable,
    _example_file,
    _field,
    _frame_for_test,
    _read_mutable_frame,
    _retained_prefix,
    _validate_layout,
)

_HEADER = b'{"version":1}'
_BUNDLES = tuple(
    bytes([0x41 + tier]) * 64 + bytes([0x61 + tier]) * 64 + bytes([0x31 + tier]) * 64
    for tier in range(4)
)
_RUNTIME = _example_file(
    _HEADER,
    b"S" * 64,
    b"E" * 16,
    _frame_for_test(b"{}", 18),
    tuple(bundle[:64] for bundle in _BUNDLES),
)
_RUNTIME_LAYOUT, _ = _validate_layout(_RUNTIME)
_TEMPLATE = _RUNTIME[: _RUNTIME_LAYOUT.tier_start]
_SHARED = _TEMPLATE + b"C" * 64
_LABELS = ("bundle_lite", "bundle_full", "bundle_pro", "bundle_ultra")
_PREFIX_HEX = (
    "71979cd5235f0d914d99f79adcd3a3184e382b8019ba86e39891fa4ad9531675",
    "3d9055890214c04d532c434a058c93f6d09a94e8f428aad9b1299232df3a7e68",
    "46d550042a1a5b45e2f359b7d732b7ce5bb83055eb330ecd4a35a21309bf4013",
    "d44970efb24b4f5a79ffb4e9b987c6527e79c61de5784032f9628b0e24b636c0",
)


def _bundle_prefixes(
    preamble: bytes, directory: bytes, header: bytes, shared: bytes, bundles: tuple[bytes, ...]
) -> list[bytes]:
    initial = (
        _field("domain", b"DNCK-CHECKPOINT-PREFIX-V1")
        + _field("checkpoint_header_sha256", hashlib.sha256(header).digest())
        + _field("preamble", preamble)
        + _field("directory", directory)
        + _field("shared_checkpoint_binary", shared)
    )
    result = []
    for index in range(len(bundles)):
        encoded = initial + b"".join(
            _field(_LABELS[tier], bundles[tier]) for tier in range(index + 1)
        )
        result.append(hashlib.sha256(encoded).digest())
    return result


def _example(maximum: int = 3) -> tuple[bytes, tuple[int, ...]]:
    count = maximum + 1
    bundles = _BUNDLES[:count]
    shared_offset = _align64(192 + len(_HEADER))
    auth_offset = _align64(shared_offset + len(_SHARED))
    auth_length = 106 + 33 * count
    bundle_start = _align64(auth_offset + auth_length)
    ends = tuple(bundle_start + 192 * (tier + 1) for tier in range(count))
    preamble = struct.pack("<4sHHBBBBI48s", b"DNCK", 2, 0, 0, 0, count - 1, 0, 48_000, bytes(48))
    directory = struct.pack(
        "<13Q24s",
        192,
        len(_HEADER),
        shared_offset,
        len(_SHARED),
        auth_offset,
        auth_length,
        0,
        0,
        bundle_start,
        *(ends + (0,) * (4 - count)),
        bytes(24),
    )
    digests = _bundle_prefixes(preamble, directory, _HEADER, _SHARED, bundles)
    prefix_list = bytes([count]) + b"".join(
        bytes([tier]) + digest for tier, digest in enumerate(digests)
    )
    envelope = b"DCA1" + bytes(4) + b"K" * 32 + bytes([count - 1]) + prefix_list + b"X" * 64
    assert len(envelope) == auth_length
    output = bytearray(ends[-1])
    output[:64] = preamble
    output[64:192] = directory
    output[192 : 192 + len(_HEADER)] = _HEADER
    output[shared_offset : shared_offset + len(_SHARED)] = _SHARED
    output[auth_offset : auth_offset + auth_length] = envelope
    offset = bundle_start
    for bundle in bundles:
        output[offset : offset + len(bundle)] = bundle
        offset += len(bundle)
    frame = _frame_for_test(b'{"user_metadata":{"tag":"amp"}}', 47)
    return _append_mutable(bytes(output), frame), ends


def _retained(data: bytes) -> tuple[int, tuple[int, ...], int]:
    if len(data) < 208:
        raise ValueError("short checkpoint")
    magic, major, minor, profile, auth_kind, maximum, reserved, rate, pad = struct.unpack(
        "<4sHHBBBBI48s", data[:64]
    )
    if (
        magic != b"DNCK"
        or (major, minor, profile, auth_kind, reserved, rate) != (2, 0, 0, 0, 0, 48_000)
        or maximum > 3
        or any(pad)
    ):
        raise ValueError("invalid preamble")
    fields = struct.unpack("<13Q24s", data[64:192])
    (
        header_offset,
        header_length,
        shared_offset,
        shared_length,
        auth_offset,
        auth_length,
    ) = fields[:6]
    _, _, bundle_start, *tail = fields[6:13]
    ends = tuple(tail)
    issued_ends = ends[: maximum + 1]
    if (
        fields[6] != 0
        or fields[7] != 0
        or any(fields[13])
        or header_offset != 192
        or not header_length
        or shared_offset != _align64(header_offset + header_length)
        or auth_offset != _align64(shared_offset + shared_length)
        or auth_length != 106 + 33 * (maximum + 1)
        or bundle_start != _align64(auth_offset + auth_length)
        or len(ends) != 4
        or any(ends[maximum + 1 :])
        or any(end % 64 for end in issued_ends)
        or not all(
            left < right
            for left, right in zip((bundle_start, *issued_ends), issued_ends, strict=False)
        )
    ):
        raise ValueError("invalid directory")
    if any(
        any(data[start:end])
        for start, end in (
            (header_offset + header_length, shared_offset),
            (shared_offset + shared_length, auth_offset),
            (auth_offset + auth_length, bundle_start),
        )
    ):
        raise ValueError("nonzero alignment padding")
    magic, frame_length, reserved = struct.unpack("<4sQI", data[-16:])
    if magic != b"DMUE" or reserved or frame_length > len(data) - 16:
        raise ValueError("invalid footer")
    boundary = len(data) - 16 - frame_length
    if boundary not in issued_ends:
        raise ValueError("invalid retained boundary")
    tier = issued_ends.index(boundary)
    _read_mutable_frame(data[boundary:-16])
    envelope = data[auth_offset : auth_offset + auth_length]
    if (
        len(envelope) != auth_length
        or envelope[:8] != b"DCA1" + bytes(4)
        or envelope[40] != maximum
        or envelope[41] != maximum + 1
        or any(envelope[42 + 33 * index] != index for index in range(maximum + 1))
    ):
        raise ValueError("invalid envelope")
    starts = (bundle_start, *ends[:-1])
    bundles = tuple(data[start:end] for start, end in zip(starts[: tier + 1], ends, strict=False))
    computed = _bundle_prefixes(
        data[:64],
        data[64:192],
        data[header_offset : header_offset + header_length],
        data[shared_offset : shared_offset + shared_length],
        bundles,
    )[-1]
    if computed != envelope[43 + 33 * tier : 75 + 33 * tier]:
        raise ValueError("checkpoint prefix mismatch")
    return tier, issued_ends, boundary


def _export_runtime(data: bytes) -> bytes:
    tier, ends, boundary = _retained(data)
    shared_offset, shared_length = struct.unpack_from("<2Q", data, 64 + 16)
    bundle_start = struct.unpack_from("<Q", data, 64 + 64)[0]
    if shared_length < 192:
        raise ValueError("short runtime template")
    template = data[shared_offset : shared_offset + shared_length]
    runtime_start = struct.unpack_from("<Q", template, 64 + 64)[0]
    runtime_ends = struct.unpack_from("<4Q", template, 64 + 72)
    if runtime_start > len(template):
        raise ValueError("invalid runtime template")
    output = bytearray(template[:runtime_start])
    checkpoint_starts = (bundle_start, *ends[:-1])
    previous_runtime = runtime_start
    for index in range(tier + 1):
        runtime_length = runtime_ends[index] - previous_runtime
        checkpoint_length = ends[index] - checkpoint_starts[index]
        if runtime_length <= 0 or runtime_length % 64 or runtime_length > checkpoint_length:
            raise ValueError("invalid runtime slice")
        start = checkpoint_starts[index]
        output.extend(data[start : start + runtime_length])
        previous_runtime = runtime_ends[index]
    output.extend(data[boundary:])
    _, exported_tier = _validate_layout(bytes(output))
    if exported_tier != tier:
        raise ValueError("runtime tier mismatch")
    return bytes(output)


@pytest.mark.parametrize("tier", range(4))
def test_original_directory_and_complete_bundle_prefix_survive_stripping(tier: int) -> None:
    original, ends = _example()
    suffix = original[ends[-1] :]
    stripped = original[: ends[tier]] + suffix
    assert _retained(stripped) == (tier, ends, ends[tier])
    assert stripped[ends[tier] :] == suffix


def test_complete_checkpoint_prefix_vectors() -> None:
    original, _ = _example()
    auth_offset = struct.unpack_from("<Q", original, 64 + 32)[0]
    envelope = original[auth_offset:]
    assert tuple(envelope[43 + 33 * i : 75 + 33 * i].hex() for i in range(4)) == _PREFIX_HEX


@pytest.mark.parametrize("maximum", range(4))
def test_each_original_issuance_maximum_keeps_its_signed_boundaries(maximum: int) -> None:
    original, ends = _example(maximum)
    suffix = original[ends[-1] :]
    for tier, end in enumerate(ends):
        assert _retained(original[:end] + suffix) == (tier, ends, end)


@pytest.mark.parametrize("tier", range(4))
def test_export_reconstructs_original_runtime_prefix_and_keeps_suffix(tier: int) -> None:
    original, ends = _example()
    suffix = original[ends[-1] :]
    stripped = original[: ends[tier]] + suffix
    exported = _export_runtime(stripped)
    expected = _RUNTIME[: _RUNTIME_LAYOUT.tier_ends[tier]] + suffix
    assert exported == expected
    pristine = _RUNTIME[: _RUNTIME_LAYOUT.tier_ends[tier]] + _RUNTIME[_RUNTIME_LAYOUT.mutable[0] :]
    assert _retained_prefix(exported, b"DNAM-MASTER-PREFIX-V1") == _retained_prefix(
        pristine, b"DNAM-MASTER-PREFIX-V1"
    )


@pytest.mark.parametrize("component", range(3))
def test_tampering_with_each_bundle_component_breaks_its_prefix(component: int) -> None:
    original, ends = _example()
    altered = bytearray(original)
    altered[ends[0] - 192 + component * 64] ^= 1
    with pytest.raises(ValueError, match="checkpoint prefix mismatch"):
        _retained(bytes(altered))


def test_partial_bundle_and_untrusted_footer_are_rejected() -> None:
    original, ends = _example()
    suffix = original[ends[-1] :]
    with pytest.raises(ValueError, match="invalid retained boundary"):
        _retained(original[: ends[1] - 64] + suffix)
    changed = bytearray(original)
    changed[-1] = 1
    with pytest.raises(ValueError, match="invalid footer"):
        _retained(bytes(changed))
