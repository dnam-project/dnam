"""First-class `.dnam` and `.dnam-ckpt` container model (`DnamContainer`),
`Arena 3` `uplo='U'` Kronecker matrix packer (`GramMatrix`), and hardware `crc32c`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_engine import _native, provenance
from dnam_engine.control import ControlSpace
from dnam_engine.exceptions import DnamValidationError, call_native
from dnam_engine.schema.base import VerificationStatus
from dnam_engine.schema.controls import ClampedControlsResult, ControlSchemaDef, PresetDef
from dnam_engine.schema.header import DnamJsonHeader
from dnam_engine.schema.runtime import DnamPreambleInfo, ZohRescaledParams
from dnam_engine.schema.signatures import WatermarkParams


class DnamContainer:
    """Immutable, copy-on-write representation of a validated `.dnam` or `.dnam-ckpt` container."""

    def __init__(
        self,
        raw_bytes: bytes,
        *,
        _prevalidated: tuple[
            DnamPreambleInfo,
            DnamJsonHeader,
            VerificationStatus,
            bytes,
            ControlSpace,
        ]
        | None = None,
    ) -> None:
        self._raw_bytes = raw_bytes
        if _prevalidated is not None:
            (
                self._preamble,
                self._header,
                self._verification,
                self._arena_bytes,
                self._control_space,
            ) = _prevalidated
            return

        info = call_native(_native.native_validate_container, self._raw_bytes)
        self._preamble = DnamPreambleInfo._construct_trusted(
            magic=str(info["magic"]),
            version_major=int(info["version_major"]),
            version_minor=int(info["version_minor"]),
            flags=int(info["flags"]),
            num_controls=int(info["num_controls"]),
            ring_buffer_pow2_log2=int(info["ring_buffer_pow2_log2"]),
            sample_rate_train_hz=int(info["sample_rate_train_hz"]),
            json_header_byte_len=int(info["json_header_byte_len"]),
            tensor_arena_offset=int(info["tensor_arena_offset"]),
            tensor_arena_byte_len=int(info["tensor_arena_byte_len"]),
            arena_offset_lite_end=int(info["arena_offset_lite_end"]),
            arena_offset_full_end=int(info["arena_offset_full_end"]),
            arena_offset_ultra_end=int(info["arena_offset_ultra_end"]),
            arena_crc32c_lite=int(info["arena_crc32c_lite"]),
            arena_crc32c_full=int(info["arena_crc32c_full"]),
            arena_crc32c_ultra=int(info["arena_crc32c_ultra"]),
            num_ckpt_stems=int(info["num_ckpt_stems"]),
            control_decimation=int(info["control_decimation"]),
            head_svd_rank=int(info["head_svd_rank"]),
            tail_time_ms=float(info["tail_time_ms"]),
            header_crc32c=int(info["header_crc32c"]),
        )
        self._header = DnamJsonHeader.model_validate_json(str(info["header_json"]))
        self._verification = VerificationStatus.from_rust_debug(str(info["verification_status"]))
        self._arena_bytes: bytes = bytes(info["arena_bytes"])
        self._control_space = ControlSpace(self._header)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> DnamContainer:
        """Parses and validates a `.dnam` or `.dnam-ckpt` container from raw bytes."""
        return cls(bytes(data))

    @classmethod
    def from_file(cls, path: str | Path) -> DnamContainer:
        """Loads and validates a `.dnam` or `.dnam-ckpt` container from `path`."""
        return cls(Path(path).read_bytes())

    @classmethod
    def reference(
        cls,
        *,
        core2_active: bool = False,
        is_checkpoint: bool = False,
    ) -> DnamContainer:
        """Synthesizes a canonical reference `.dnam` or `.dnam-ckpt` container."""
        raw = call_native(
            _native.native_build_reference_container,
            core2_active,
            is_checkpoint,
        )
        return cls(raw)

    @classmethod
    def build(
        cls,
        header: DnamJsonHeader,
        arena: bytes | bytearray | memoryview | None = None,
        *,
        plaintext_arena_crc_source: bytes | None = None,
        encrypted_arena: bool = False,
    ) -> DnamContainer:
        """Builds a 4 KB page-aligned `.dnam` or `.dnam-ckpt` container from `header` & `arena`."""
        arena_bytes = bytes(arena) if arena is not None else b"\x00" * 36_224
        raw = call_native(
            _native.native_build_container,
            header.model_dump_json(by_alias=True),
            arena_bytes,
            plaintext_arena_crc_source,
            encrypted_arena,
        )
        return cls(raw)

    @property
    def preamble(self) -> DnamPreambleInfo:
        """Returns the 64-byte binary preamble metadata."""
        return self._preamble

    @property
    def header(self) -> DnamJsonHeader:
        """Returns the validated `DnamJsonHeader` Pydantic model."""
        return self._header

    @property
    def control_space(self) -> ControlSpace:
        """Returns the `ControlSpace` bound to this container's `controls_schema`."""
        return self._control_space

    @property
    def controls(self) -> Mapping[str, ControlSchemaDef]:
        """Returns the control schema definitions keyed by control `id`."""
        return self._control_space.schema

    @property
    def presets(self) -> Mapping[str, PresetDef]:
        """Returns the built-in presets keyed by preset `name`."""
        return self._control_space.presets

    @property
    def verification(self) -> VerificationStatus:
        """Returns the cryptographic provenance verification status."""
        return self._verification

    @property
    def is_checkpoint(self) -> bool:
        """Returns `True` if this container is a `.dnam-ckpt` training checkpoint (`DNCK`)."""
        return self._preamble.magic == "DNCK"

    @property
    def arena(self) -> memoryview:
        """Returns a read-only zero-copy `memoryview` of the 64-byte aligned tensor arena."""
        return memoryview(self._arena_bytes)

    def to_bytes(self) -> bytes:
        """Returns the complete binary container bytes."""
        return self._raw_bytes

    def __bytes__(self) -> bytes:
        return self._raw_bytes

    def save(self, path: str | Path) -> Path:
        """Writes the container bytes to `path` and returns the resolved `Path`."""
        out_path = Path(path)
        out_path.write_bytes(self._raw_bytes)
        return out_path

    def with_header(self, updated_header: DnamJsonHeader) -> DnamContainer:
        """Returns a new `DnamContainer` with `updated_header`, using the zero-weight-copy
        4 KB in-place fast path when serialized header fits within `header_capacity_bytes`."""
        header_json = updated_header.model_dump_json(by_alias=True)
        raw = call_native(
            _native.native_update_header_in_place,
            self._raw_bytes,
            header_json,
        )
        # Parse updated 64-byte preamble fields directly from `raw[0..64]`
        json_len = int.from_bytes(raw[12:16], "little")
        arena_off = int.from_bytes(raw[16:20], "little")
        hdr_crc = int.from_bytes(raw[60:64], "little")
        preamble_fields = dict(self._preamble.__dict__)
        preamble_fields["num_controls"] = len(updated_header.controls_schema)
        preamble_fields["json_header_byte_len"] = json_len
        preamble_fields["tensor_arena_offset"] = arena_off
        preamble_fields["header_crc32c"] = hdr_crc
        new_preamble = DnamPreambleInfo._construct_trusted(**preamble_fields)
        reuse_native = (
            self._control_space._native_space
            if updated_header.controls_schema == self._header.controls_schema
            else None
        )
        new_space = ControlSpace(updated_header, _native_space=reuse_native)
        return DnamContainer(
            raw,
            _prevalidated=(
                new_preamble,
                updated_header,
                self._verification,
                self._arena_bytes,
                new_space,
            ),
        )

    def clamp_controls(
        self,
        raw_controls: Mapping[str, float | str] | None = None,
    ) -> ClampedControlsResult:
        """Evaluates 3-Phase control clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, etc.)."""
        return self._control_space.clamp(raw_controls)

    def zoh_params(self, sample_rate_hz: float) -> ZohRescaledParams:
        """Computes Continuous-Time Zero-Order Hold (ZOH) parameters for `sample_rate_hz`."""
        d = call_native(
            _native.native_compute_zoh,
            self._preamble.sample_rate_train_hz,
            round(float(sample_rate_hz)),
            max(8, self._preamble.control_decimation),
            self._header.core1_hybrid_ssm.ac_coupling_pole,
            self._header.core2_mufdn.dc_blocker_pole,
            self._header.core2_mufdn.tzf_reference_tap_samples,
            self._header.core2_mufdn.switch_xfade_samples,
        )
        return ZohRescaledParams.model_validate(d)

    def immutable_header_sha256(self) -> str:
        """Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest."""
        return provenance.immutable_header_sha256(self._header)

    def structural_hash(self) -> NDArray[np.float64]:
        """Extracts the 160-D Gauge-Invariant Structural Hash vector `f_struct in R^160`."""
        return provenance.extract_structural_hash(self._arena_bytes)

    def with_watermark(self, params: WatermarkParams) -> DnamContainer:
        """Returns a new `DnamContainer` with Dual-Mechanism Buyer Watermarking (`Test 8H`)."""
        watermarked_arena = provenance.apply_dual_watermark(self._arena_bytes, params)
        return DnamContainer.build(self._header, watermarked_arena)


class GramMatrix:
    """`Arena 3` Column-Major Upper-Triangular (`LAPACK uplo='U'`) symmetric Gram matrix packer."""

    @staticmethod
    def packed_len(dim: int) -> int:
        """Returns the packed element count `dim * (dim + 1) // 2`."""
        if dim < 1:
            raise DnamValidationError(f"GramMatrix dimension must be >= 1, got {dim}")
        return (dim * (dim + 1)) // 2

    @staticmethod
    def pack(matrix: ArrayLike) -> NDArray[np.float64]:
        """Packs a symmetric `(N, N)` matrix into `LAPACK uplo='U'` upper-triangular format."""
        arr = np.ascontiguousarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
            raise DnamValidationError(
                f"GramMatrix.pack requires a non-empty square 2D matrix, got shape {arr.shape}"
            )
        n_dim = int(arr.shape[0])
        return call_native(_native.native_pack_uplo_u_f64, arr.ravel(), n_dim)

    @staticmethod
    def unpack(packed: ArrayLike, *, dim: int) -> NDArray[np.float64]:
        """Unpacks a `LAPACK uplo='U'` buffer into a symmetric `(dim, dim)` matrix.

        Supports the Nested Principal Submatrix Prefix Theorem: passing `dim=64` (`Lite`) or
        `dim=128` (`Full`) on an `Ultra` (`N=192`) packed buffer extracts the exact leading
        principal submatrix from its contiguous prefix.
        """
        if dim < 1:
            raise DnamValidationError(f"GramMatrix.unpack dim must be >= 1, got {dim}")
        p_arr = (
            packed
            if isinstance(packed, np.ndarray)
            and packed.dtype == np.float64
            and packed.flags.c_contiguous
            else np.ascontiguousarray(packed, dtype=np.float64)
        )
        if p_arr.ndim != 1:
            raise DnamValidationError(
                f"GramMatrix.unpack requires a 1D packed buffer, got shape {p_arr.shape}"
            )
        flat = call_native(_native.native_unpack_uplo_u_f64, p_arr, dim)
        return flat.reshape((dim, dim))


def crc32c(data: bytes | bytearray | memoryview) -> int:
    """Computes the Castagnoli `CRC32C` checksum over `data`."""
    return call_native(_native.native_crc32c, bytes(data))
