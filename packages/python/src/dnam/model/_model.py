"""Deployable `.dnam` `Model` container class (`ref[sdk.python.api.core-types]`,
`ref[sdk.python.model.containers-and-checkpoints]`).
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from dnam._native import (
    native_build_container,
    native_build_dmu1_frame,
    native_build_reference_container,
    native_parse_dmu1_frame,
    native_strip_container_to_tier,
    native_update_header_in_place_fast,
    native_validate_container,
    native_validate_v2_open_layout,
)
from dnam.controls._space import ControlSpace
from dnam.errors import ContainerError, ValidationError
from dnam.model._preamble import Preamble
from dnam.schema._base import _TIER_TUPLE, Tier, VerificationStatus
from dnam.schema.controls import ControlSchemaDef, PresetDef
from dnam.schema.header import DnamJsonHeader
from dnam.schema.signatures import DnamProtectionProfile

if TYPE_CHECKING:
    from dnam._native import ValidatedContainerDict


def _extract_preamble(raw: ValidatedContainerDict) -> Preamble:
    return Preamble._construct_trusted_dict(
        {
            "magic": raw["magic"],
            "version_major": raw["version_major"],
            "version_minor": raw["version_minor"],
            "flags": raw["flags"],
            "num_controls": raw["num_controls"],
            "sample_rate_train_hz": raw["sample_rate_train_hz"],
            "json_header_byte_len": raw["json_header_byte_len"],
            "tensor_arena_offset": raw["tensor_arena_offset"],
            "tensor_arena_byte_len": raw["tensor_arena_byte_len"],
            "arena_lite_end_cl": raw["arena_lite_end_cl"],
            "arena_full_end_cl": raw["arena_full_end_cl"],
            "arena_pro_end_cl": raw["arena_pro_end_cl"],
            "arena_ultra_end_cl": raw["arena_ultra_end_cl"],
            "arena_offset_lite_end": raw["arena_offset_lite_end"],
            "arena_offset_full_end": raw["arena_offset_full_end"],
            "arena_offset_pro_end": raw["arena_offset_pro_end"],
            "arena_offset_ultra_end": raw["arena_offset_ultra_end"],
            "arena_crc32c_lite": raw["arena_crc32c_lite"],
            "arena_crc32c_full": raw["arena_crc32c_full"],
            "arena_crc32c_pro": raw["arena_crc32c_pro"],
            "arena_crc32c_ultra": raw["arena_crc32c_ultra"],
            "num_ckpt_stems": raw["num_ckpt_stems"],
            "tail_time_ms": raw["tail_time_ms"],
            "header_crc32c": raw["header_crc32c"],
            "highest_tier": _TIER_TUPLE[raw["highest_tier"]],
            "is_stereo_out": raw["is_stereo_out"],
            "has_signatures": raw["has_signatures"],
            "is_encrypted_arena": raw["is_encrypted_arena"],
            "has_calib_stats": raw["has_calib_stats"],
        }
    )


def _resolve_arena_bytes(raw_bytes: bytes, raw: ValidatedContainerDict) -> bytes:
    maybe_arena = raw["arena_bytes"]
    if maybe_arena is not None:
        return maybe_arena
    off = raw["tensor_arena_offset"]
    length = raw["tensor_arena_byte_len"]
    return raw_bytes[off : off + length]


class Model:
    """
    Immutable, validated `.dnam` deployment model container (`ref[sdk.python.api.core-types]`).
    """

    __slots__ = (
        "_arena_bytes",
        "_container_bytes",
        "_control_space",
        "_header",
        "_header_json",
        "_preamble",
        "_protection_profile",
        "_verification",
    )

    def __init__(
        self,
        container_bytes: bytes,
        *,
        preamble: Preamble,
        header: DnamJsonHeader | None = None,
        header_json: str,
        arena_bytes: bytes | None = None,
        verification: VerificationStatus,
        protection_profile: DnamProtectionProfile,
    ) -> None:
        self._container_bytes = container_bytes
        self._preamble = preamble
        self._header = header
        self._header_json = header_json
        self._arena_bytes = arena_bytes
        self._verification = verification
        self._protection_profile = protection_profile
        self._control_space: ControlSpace | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        device_id_hash: str | None = None,
        device_secret: bytes | None = None,
        trusted_marketplace_pubkey_hex: str | None = None,
    ) -> Model:
        """Validates and loads a `.dnam` or `.dnam-seal` (`DNSL`) container from bytes."""
        raw_bytes = data if type(data) is bytes else bytes(data)
        raw = native_validate_container(
            raw_bytes,
            device_id_hash,
            device_secret,
            trusted_marketplace_pubkey_hex,
        )
        preamble = _extract_preamble(raw)
        header_json = raw["header_json"]
        if preamble.magic == "DNCK" or '"container_kind":"dnam-ckpt"' in header_json:
            raise ContainerError(
                "Buffer is a .dnam-ckpt training checkpoint; "
                "load it with dnam.Checkpoint.from_bytes(...)"
            )
        verification = VerificationStatus.from_rust_debug(raw["verification_status"])
        profile = DnamProtectionProfile(raw["protection_profile"])
        return cls(
            raw_bytes,
            preamble=preamble,
            header=None,
            header_json=header_json,
            arena_bytes=raw["arena_bytes"],
            verification=verification,
            protection_profile=profile,
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        device_id_hash: str | None = None,
        device_secret: bytes | None = None,
        trusted_marketplace_pubkey_hex: str | None = None,
    ) -> Model:
        """Loads and validates a `.dnam` or `.dnam-seal` file from disk."""
        file_bytes = Path(path).read_bytes()
        return cls.from_bytes(
            file_bytes,
            device_id_hash=device_id_hash,
            device_secret=device_secret,
            trusted_marketplace_pubkey_hex=trusted_marketplace_pubkey_hex,
        )

    @classmethod
    def reference(cls, *, supports_freeze: bool = True) -> Model:
        """Synthesizes a canonical reference `.dnam` `Model`."""
        raw_bytes = native_build_reference_container(
            supports_freeze,
            False,
        )
        return cls.from_bytes(raw_bytes)

    @classmethod
    def build(
        cls,
        header: DnamJsonHeader,
        arena_bytes: bytes | bytearray | memoryview,
        *,
        plaintext_arena_crc_source: bytes | None = None,
        encrypted_arena: bool = False,
    ) -> Model:
        """Builds a 4 KB page-aligned `.dnam` `Model` from `header` and `arena_bytes`."""
        if header.container_kind != "dnam":
            raise ValidationError(
                "Model.build requires header.container_kind == 'dnam', "
                f"got '{header.container_kind}'"
            )
        raw_arena = arena_bytes if type(arena_bytes) is bytes else bytes(arena_bytes)
        built = native_build_container(
            header.to_json(),
            raw_arena,
            plaintext_arena_crc_source,
            encrypted_arena,
        )
        if encrypted_arena:
            raw = native_validate_container(
                built[:4096] + (plaintext_arena_crc_source or raw_arena)
            )
            return cls(
                built,
                preamble=_extract_preamble(raw),
                header=header,
                header_json=header.to_json(),
                arena_bytes=raw_arena,
                verification=VerificationStatus.from_rust_debug(raw["verification_status"]),
                protection_profile=DnamProtectionProfile.DEVICE_ALLOWLISTED,
            )
        return cls.from_bytes(built)

    @property
    def header(self) -> DnamJsonHeader:
        """Parsed and validated `DnamJsonHeader`."""
        hdr = self._header
        if hdr is None:
            hdr = DnamJsonHeader.from_json(self._header_json)
            self._header = hdr
        return hdr

    @property
    def name(self) -> str:
        """Human-readable model name (`header.metadata.name`)."""
        return self.header.metadata.name

    @property
    def preamble(self) -> Preamble:
        """Parsed 64-byte binary preamble."""
        return self._preamble

    @property
    def control_space(self) -> ControlSpace:
        """Lazily compiled `ControlSpace` for this model's `controls_schema`."""
        if self._control_space is None:
            self._control_space = ControlSpace(self.header, header_json=self._header_json)
        return self._control_space

    @property
    def controls(self) -> Mapping[str, ControlSchemaDef]:
        """Read-only mapping of control ID to `ControlSchemaDef`."""
        return self.control_space.schema

    @property
    def presets(self) -> Mapping[str, PresetDef]:
        """Read-only mapping of preset ID to `PresetDef`."""
        return self.control_space.presets

    @property
    def verification(self) -> VerificationStatus:
        """Cryptographic provenance verification status."""
        return self._verification

    @property
    def protection_profile(self) -> DnamProtectionProfile:
        """Active container protection profile (`open`, `device_allowlisted`, `enclave_sealed`)."""
        return self._protection_profile

    @property
    def max_tier(self) -> Tier:
        """Highest slimmable tier available in this container."""
        return self._preamble.highest_tier

    @property
    def arena_bytes(self) -> bytes:
        """Decrypted runtime tensor arena byte slice (`Arena 0..max_tier`)."""
        arena = self._arena_bytes
        if arena is None:
            off = self._preamble.tensor_arena_offset
            length = self._preamble.tensor_arena_byte_len
            arena = self._container_bytes[off : off + length]
            self._arena_bytes = arena
        return arena

    def tier_arena(self, tier: Tier | str | int) -> memoryview:
        """Returns a read-only `memoryview` over the cumulative tensor arena prefix up to `tier`."""
        t = Tier.parse(tier)
        if t > self._preamble.highest_tier:
            raise ValidationError(
                f"Requested tier {t.label} exceeds model's highest tier "
                f"({self._preamble.highest_tier.label})"
            )
        ends = (
            self._preamble.arena_offset_lite_end,
            self._preamble.arena_offset_full_end,
            self._preamble.arena_offset_pro_end,
            self._preamble.arena_offset_ultra_end,
        )
        arena = self.arena_bytes
        end = min(ends[int(t)], len(arena))
        return memoryview(arena)[:end]

    def with_header(self, updated_header: DnamJsonHeader) -> Model:
        """Updates the JSON header in-place within the 4 KB page boundary without touching the
        arena.
        """
        fast_json = updated_header.to_json()
        new_bytes, json_len, hdr_crc = native_update_header_in_place_fast(
            self._container_bytes,
            fast_json,
        )
        canonical_json = new_bytes[64 : 64 + json_len].decode("utf-8")
        new_preamble = self._preamble.replace(
            json_header_byte_len=json_len,
            header_crc32c=hdr_crc,
        )
        return Model(
            new_bytes,
            preamble=new_preamble,
            header=updated_header,
            header_json=canonical_json,
            arena_bytes=self._arena_bytes,
            verification=self._verification,
            protection_profile=self._protection_profile,
        )

    def with_tier_limit(self, max_tier: Tier | str | int) -> Model:
        """Strips the container down to `max_tier` without modifying the immutable JSON header."""
        target = Tier.parse(max_tier)
        stripped = native_strip_container_to_tier(
            self._container_bytes,
            int(target),
        )
        return Model.from_bytes(stripped)

    def to_bytes(self) -> bytes:
        """Returns the immutable binary `.dnam` container bytes."""
        return self._container_bytes

    def save(self, path: str | PathLike[str]) -> Path:
        """Writes the `.dnam` container bytes to `path` and returns the resolved `Path`."""
        out_path = Path(path)
        out_path.write_bytes(self._container_bytes)
        return out_path

    @staticmethod
    def validate_v2_open_layout(data: bytes | bytearray | memoryview) -> dict[str, object]:
        """
        Validates a v2.0 Open `.dnam` container wire layout and returns its directory metadata.
        """
        return dict(native_validate_v2_open_layout(bytes(data)))

    @staticmethod
    def build_dmu1_frame(mutable_json: str | bytes, capacity: int = 64) -> bytes:
        """Builds a 64B-aligned `DMU1` mutable suffix frame."""
        raw = mutable_json.encode("utf-8") if isinstance(mutable_json, str) else mutable_json
        return native_build_dmu1_frame(raw, capacity)

    @staticmethod
    def parse_dmu1_frame(frame_bytes: bytes | bytearray | memoryview) -> str:
        """Validates and parses a 64B-aligned `DMU1` mutable suffix frame into canonical JSON."""
        return native_parse_dmu1_frame(bytes(frame_bytes))

    def __bytes__(self) -> bytes:
        return self._container_bytes

    def __len__(self) -> int:
        return len(self._container_bytes)

    def __repr__(self) -> str:
        tier_range = (
            "Lite" if self.max_tier == Tier.LITE else f"Lite..{self.max_tier.name.capitalize()}"
        )
        return (
            f"<Model {self.name!r} tiers={tier_range} "
            f"controls={len(self.header.controls_schema)} "
            f"status={self._verification.value}>"
        )
