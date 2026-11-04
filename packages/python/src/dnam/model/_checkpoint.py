"""Training `.dnam-ckpt` (`DNCK`) `Checkpoint` container class (`ref[sdk.python.api.core-types]`,
`ref[sdk.python.model.containers-and-checkpoints]`).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from dnam._native import (
    native_build_container,
    native_build_reference_checkpoint,
    native_export_runtime_dnam_from_ckpt,
    native_validate_container,
    native_validate_v2_dnck_checkpoint,
    native_verify_arena5_calib_stats,
    native_verify_stem_arena4_crc32c,
)
from dnam.errors import ContainerError, ValidationError
from dnam.model._model import Model, _extract_preamble, _resolve_arena_bytes
from dnam.model._preamble import Preamble
from dnam.schema.checkpoint import AuditReport, StemRegistryEntry
from dnam.schema.header import DnamJsonHeader


class Checkpoint:
    """Immutable, validated `.dnam-ckpt` (`DNCK`) iterative training checkpoint container."""

    __slots__ = ("_arena_bytes", "_ckpt_bytes", "_header", "_preamble")

    def __init__(
        self,
        ckpt_bytes: bytes,
        *,
        preamble: Preamble,
        header: DnamJsonHeader,
        arena_bytes: bytes,
    ) -> None:
        self._ckpt_bytes = ckpt_bytes
        self._preamble = preamble
        self._header = header
        self._arena_bytes = arena_bytes

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Checkpoint:
        """Validates and loads a `.dnam-ckpt` (`DNCK`) checkpoint container from bytes."""
        raw_bytes = data if type(data) is bytes else bytes(data)
        raw = native_validate_container(raw_bytes)
        preamble = _extract_preamble(raw)
        header = DnamJsonHeader.from_json(raw["header_json"])
        if preamble.magic != "DNCK" and header.container_kind != "dnam-ckpt":
            raise ContainerError(
                "Buffer is a deployment .dnam model; load it with dnam.Model.from_bytes(...)"
            )
        return cls(
            raw_bytes,
            preamble=preamble,
            header=header,
            arena_bytes=_resolve_arena_bytes(raw_bytes, raw),
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Checkpoint:
        """Loads and validates a `.dnam-ckpt` file from disk."""
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def reference(cls, *, include_calib_stats: bool = True) -> Checkpoint:
        """Synthesizes a canonical reference `.dnam-ckpt` `Checkpoint`."""
        raw_bytes = native_build_reference_checkpoint(include_calib_stats)
        return cls.from_bytes(raw_bytes)

    @classmethod
    def build(
        cls,
        header: DnamJsonHeader,
        arena_bytes: bytes | bytearray | memoryview,
    ) -> Checkpoint:
        """Builds a `.dnam-ckpt` `Checkpoint` from `header` and `arena_bytes`."""
        if header.container_kind != "dnam-ckpt":
            raise ValidationError(
                "Checkpoint.build requires header.container_kind == 'dnam-ckpt', "
                f"got '{header.container_kind}'"
            )
        built = native_build_container(
            header.to_json(),
            bytes(arena_bytes),
            None,
            False,
        )
        return cls.from_bytes(built)

    @property
    def name(self) -> str:
        """Human-readable model name (`header.metadata.name`)."""
        return self._header.metadata.name

    @property
    def preamble(self) -> Preamble:
        """Parsed 64-byte binary preamble."""
        return self._preamble

    @property
    def header(self) -> DnamJsonHeader:
        """Parsed and validated `DnamJsonHeader`."""
        return self._header

    @property
    def stems(self) -> tuple[StemRegistryEntry, ...]:
        """Tuple of `StemRegistryEntry` records registered in this checkpoint."""
        return tuple(self._header.stems_registry or ())

    @property
    def audit_report(self) -> AuditReport | None:
        """Optional 3-phase control clamp `AuditReport`."""
        return self._header.audit_report

    @property
    def arena_bytes(self) -> bytes:
        """Full checkpoint arena byte slice (`Arena 0..3` + `Arena 4` + optional `Arena 5`)."""
        return self._arena_bytes

    @property
    def arena4_bytes(self) -> bytes | None:
        """Returns the `Arena 4..5` suffix (`arena[64000:]`) if present, else `None`."""
        ultra_end = self._preamble.arena_offset_ultra_end
        if len(self._arena_bytes) <= ultra_end:
            return None
        return self._arena_bytes[ultra_end:]

    def verify_stems(self) -> int:
        """Verifies every registered stem's `stem_arena4_crc32c` over its `Arena 4` slice."""
        stems = self.stems
        for stem in stems:
            off = stem.g_audio_uplo_u_f64.byte_offset
            length = stem.g_audio_uplo_u_f64.byte_length + stem.c_audio_tier_f64.byte_length
            native_verify_stem_arena4_crc32c(
                self._arena_bytes,
                off,
                length,
                stem.stem_arena4_crc32c,
            )
        return len(stems)

    def verify_calib_stats(self, arena5_offset: int | None = None) -> int:
        """
        Verifies `Arena 5` (`DNCA` calibration statistics) and returns its verified byte length.
        """
        if arena5_offset is None:
            end_a4 = self._preamble.arena_offset_ultra_end
            for stem in self.stems:
                end_a4 = max(
                    end_a4,
                    stem.c_audio_tier_f64.byte_offset + stem.c_audio_tier_f64.byte_length,
                )
            arena5_offset = end_a4
        return native_verify_arena5_calib_stats(
            self._arena_bytes,
            arena5_offset,
        )

    def export_model(self) -> Model:
        """Exports the runtime `Arena 0..3` weights and header into a deployable `.dnam` `Model`."""
        runtime_bytes = native_export_runtime_dnam_from_ckpt(self._ckpt_bytes)
        return Model.from_bytes(runtime_bytes)

    @staticmethod
    def validate_v2_dnck(data: bytes | bytearray | memoryview) -> dict[str, object]:
        """Validates a v2.0 `.dnam-ckpt` (`DNCK`) bundle layout and `DCA1` prefix commitment."""
        return dict(native_validate_v2_dnck_checkpoint(bytes(data)))

    @staticmethod
    def export_v2_runtime_bytes(data: bytes | bytearray | memoryview) -> bytes:
        """Exports v2.0 `.dnam` runtime container bytes from a v2.0 `.dnam-ckpt` (`DNCK`)."""
        return native_export_runtime_dnam_from_ckpt(bytes(data))

    def to_bytes(self) -> bytes:
        """Returns the immutable binary `.dnam-ckpt` container bytes."""
        return self._ckpt_bytes

    def save(self, path: str | PathLike[str]) -> Path:
        """Writes the `.dnam-ckpt` container bytes to `path` and returns the resolved `Path`."""
        out_path = Path(path)
        out_path.write_bytes(self._ckpt_bytes)
        return out_path

    def __bytes__(self) -> bytes:
        return self._ckpt_bytes

    def __len__(self) -> int:
        return len(self._ckpt_bytes)

    def __repr__(self) -> str:
        return (
            f"<Checkpoint {self.name!r} stems={len(self.stems)} "
            f"calib_stats={self._preamble.has_calib_stats}>"
        )
