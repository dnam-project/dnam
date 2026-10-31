"""Two-tier calibration and intermediate cache:
1. **Portable `.dnam-ckpt` Arena 4 (`HAS_CALIB_STATS` bit 11):**
   Compact (`~3.6 MB` at 30 stems x 30 s) per-input and per-stem reusable calibration statistics
   (segmentation window Grams/RHS, Stage-1 orthonormal projector, USB jitter lag, measured
   per-band noise floors, silent flag, band energies, and 8,192-tap causal Wiener LTI impulse
   responses). Each entry is protected by a per-entry Castagnoli CRC32C and overall body CRC32C,
   padded to 64-byte alignment, and stripped on `.dnam` runtime export.
2. **Local Workspace Sidecar (`models/<id>/.dnam-cache/*.dnce`):**
   Content-addressed 64-byte-aligned memory-mappable `.dnce` tensor cache with a measured
   read-vs-regenerate cost model, stage-level crash resume markers, disk-floor fail-fast check,
   LRU quota enforcement (`16 GB` default), and automatic stage/run lifecycle cleanup.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import mmap
import os
import shutil
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from dnam_engine.container import crc32c
from dnam_trainer.events import CacheEvent, TrainingObserver

ARENA4_MAGIC: bytes = b"DNCA"
ARENA4_VERSION: int = 1
ARENA4_HEADER_BYTES: int = 64
ARENA4_HEADER_STRUCT = struct.Struct(
    "<4sHHHHII44s"
)  # 64 bytes: magic, ver, flags, n_in, n_st, total_len, body_crc, pad44
ARENA4_INPUT_RECORD_MAGIC: bytes = b"INPC"
ARENA4_INPUT_ENTRY_HEADER = struct.Struct(
    "<4s32sIHHHH8sII"
)  # 64 bytes: magic, sha32, num_samples, w_count, p_dim, s1_in, s1_out, pad8, rec_len, rec_crc
ARENA4_STEM_RECORD_MAGIC: bytes = b"STMC"
# 128B: magic, sha32, flags, w_count, p_dim, lti_len, 10x f32, pad32, rec_len, rec_crc, pad4
ARENA4_STEM_ENTRY_HEADER = struct.Struct("<4s32sHHHHffffffffff32sII4s")

DNCE_MAGIC: bytes = b"DNCE"
DNCE_VERSION: int = 1
# 64-byte preamble: magic(4s), version(H), dtype_code(B), entry_kind(B), dim0(Q), dim1(Q),
# payload_bytes(I), crc32c(I), key_sha256(32s) -> 4+2+1+1+8+8+4+4+32 = 64 bytes
DNCE_PREAMBLE_STRUCT = struct.Struct("<4sHBBQQII32s")

CACHE_CODE_VERSION: str = "dnam-calib-v1"

DTYPE_FLOAT32: int = 1
DTYPE_BFLOAT16: int = 2
DTYPE_INT32: int = 3
DTYPE_FLOAT64: int = 4
DTYPE_JSON_UTF8: int = 5

KIND_INPUT_ONLY: int = 1
KIND_STEM_SET: int = 2
KIND_STAGE_BASIS: int = 3
KIND_STAGE_MARKER: int = 4

STEM_FLAG_IS_SILENT: int = 1 << 0
STEM_FLAG_HAS_LTI_IR: int = 1 << 1
STEM_FLAG_HAS_SEG_RHS: int = 1 << 2


def _sha256_bytes32(val: str | bytes) -> bytes:
    """Normalizes a hex SHA-256 string or arbitrary key into a 32-byte digest."""
    if isinstance(val, bytes):
        if len(val) == 32:
            return val
        return hashlib.sha256(val).digest()
    s = val.strip()
    if len(s) == 64:
        try:
            return bytes.fromhex(s)
        except ValueError:
            pass
    return hashlib.sha256(s.encode("utf-8")).digest()


def _pad64(data: bytes) -> bytes:
    rem = len(data) % 64
    if rem == 0:
        return data
    return data + b"\x00" * (64 - rem)


@dataclass(frozen=True)
class InputCalibStats:
    """Per-input reusable calibration statistics stored in `.dnam-ckpt` Arena 4."""

    input_pcm_sha256: str
    num_samples: int
    win_bounds: tuple[tuple[int, int], ...]
    win_grams_f64: NDArray[np.float64]  # (W, P, P)
    s1_proj_f32: NDArray[np.float32] | None = None  # (C_in, C_out)


@dataclass(frozen=True)
class StemCalibStats:
    """Per-stem reusable calibration statistics stored in `.dnam-ckpt` Arena 4."""

    stem_id: str
    input_pcm_sha256: str
    is_silent: bool
    jitter_lag_samples: float
    jitter_corr: float
    noise_floor_dbfs_broadband: float
    noise_floor_dbfs_lf: float
    noise_floor_dbfs_mf: float
    noise_floor_dbfs_hf: float
    band_energy_broadband: float
    band_energy_lf: float
    band_energy_mf: float
    band_energy_hf: float
    seg_rhs_f64: NDArray[np.float64] | None = None  # (W, P)
    seg_y2_f64: NDArray[np.float64] | None = None  # (W,)
    lti_ir_f32: NDArray[np.float32] | None = None  # (K_lti,)


@dataclass
class Arena4CalibrationData:
    """Container for all per-input and per-stem calibration records in `.dnam-ckpt` Arena 4."""

    inputs: dict[str, InputCalibStats] = field(default_factory=dict)
    stems: dict[str, StemCalibStats] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serializes Arena 4 into a 64-byte-aligned binary buffer with CRC32C."""
        body_chunks: list[bytes] = []

        sorted_inputs = sorted(self.inputs.values(), key=lambda it: it.input_pcm_sha256)
        sorted_stems = sorted(self.stems.values(), key=lambda st: st.stem_id)

        for inp in sorted_inputs:
            sha32 = _sha256_bytes32(inp.input_pcm_sha256)
            w_count = len(inp.win_bounds)
            p_dim = int(inp.win_grams_f64.shape[1]) if inp.win_grams_f64.ndim == 3 else 0
            bounds_arr = np.asarray(inp.win_bounds, dtype="<i4").reshape(w_count, 2)
            grams_arr = np.ascontiguousarray(inp.win_grams_f64, dtype="<f8")
            if inp.s1_proj_f32 is not None and inp.s1_proj_f32.ndim == 2:
                s1_in, s1_out = int(inp.s1_proj_f32.shape[0]), int(inp.s1_proj_f32.shape[1])
                s1_bytes = np.ascontiguousarray(inp.s1_proj_f32, dtype="<f4").tobytes()
            else:
                s1_in, s1_out = 0, 0
                s1_bytes = b""

            sha_utf8 = inp.input_pcm_sha256.encode("utf-8")
            sha_hdr = struct.pack("<H", len(sha_utf8)) + sha_utf8
            raw_payload = sha_hdr + bounds_arr.tobytes() + grams_arr.tobytes() + s1_bytes
            padded_payload = _pad64(raw_payload)
            rec_len = 64 + len(padded_payload)
            rec_crc = crc32c(padded_payload)
            hdr = ARENA4_INPUT_ENTRY_HEADER.pack(
                ARENA4_INPUT_RECORD_MAGIC,
                sha32,
                inp.num_samples,
                w_count,
                p_dim,
                s1_in,
                s1_out,
                b"\x00" * 8,
                rec_len,
                rec_crc,
            )
            body_chunks.append(hdr)
            body_chunks.append(padded_payload)

        for st in sorted_stems:
            sha32 = _sha256_bytes32(st.input_pcm_sha256)
            flags = STEM_FLAG_IS_SILENT if st.is_silent else 0
            w_count = 0
            p_dim = 0
            seg_bytes = b""
            if (
                st.seg_rhs_f64 is not None
                and st.seg_y2_f64 is not None
                and st.seg_rhs_f64.ndim == 2
            ):
                flags |= STEM_FLAG_HAS_SEG_RHS
                w_count, p_dim = int(st.seg_rhs_f64.shape[0]), int(st.seg_rhs_f64.shape[1])
                seg_bytes = (
                    np.ascontiguousarray(st.seg_rhs_f64, dtype="<f8").tobytes()
                    + np.ascontiguousarray(st.seg_y2_f64, dtype="<f8").tobytes()
                )
            lti_len = 0
            lti_bytes = b""
            if st.lti_ir_f32 is not None and st.lti_ir_f32.ndim == 1 and st.lti_ir_f32.size > 0:
                flags |= STEM_FLAG_HAS_LTI_IR
                lti_len = int(st.lti_ir_f32.shape[0])
                lti_bytes = np.ascontiguousarray(st.lti_ir_f32, dtype="<f4").tobytes()

            stem_id_utf8 = st.stem_id.encode("utf-8")
            in_sha_utf8 = st.input_pcm_sha256.encode("utf-8")
            id_hdr = (
                struct.pack("<HH", len(stem_id_utf8), len(in_sha_utf8)) + stem_id_utf8 + in_sha_utf8
            )
            raw_payload = id_hdr + seg_bytes + lti_bytes
            padded_payload = _pad64(raw_payload)
            rec_len = 128 + len(padded_payload)
            rec_crc = crc32c(padded_payload)
            hdr = ARENA4_STEM_ENTRY_HEADER.pack(
                ARENA4_STEM_RECORD_MAGIC,
                sha32,
                flags,
                w_count,
                p_dim,
                lti_len,
                float(st.jitter_lag_samples),
                float(st.jitter_corr),
                float(st.noise_floor_dbfs_broadband),
                float(st.noise_floor_dbfs_lf),
                float(st.noise_floor_dbfs_mf),
                float(st.noise_floor_dbfs_hf),
                float(st.band_energy_broadband),
                float(st.band_energy_lf),
                float(st.band_energy_mf),
                float(st.band_energy_hf),
                b"\x00" * 32,
                rec_len,
                rec_crc,
                b"\x00" * 4,
            )
            body_chunks.append(hdr)
            body_chunks.append(padded_payload)

        padded_body = b"".join(body_chunks)
        total_byte_len = ARENA4_HEADER_BYTES + len(padded_body)
        body_crc = crc32c(padded_body)
        top_hdr = ARENA4_HEADER_STRUCT.pack(
            ARENA4_MAGIC,
            ARENA4_VERSION,
            0,
            len(sorted_inputs),
            len(sorted_stems),
            total_byte_len,
            body_crc,
            b"\x00" * 44,
        )
        return top_hdr + padded_body

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> Arena4CalibrationData:
        """Parses and CRC32C-verifies an Arena 4 binary buffer."""
        raw = bytes(data)
        if len(raw) < ARENA4_HEADER_BYTES or len(raw) % 64 != 0:
            raise ValueError(
                f"Invalid Arena 4 buffer length {len(raw)} (must be >= 64 and 64B-aligned)"
            )
        (
            magic,
            version,
            _flags,
            n_inputs,
            n_stems,
            total_byte_len,
            expected_body_crc,
            _pad44,
        ) = ARENA4_HEADER_STRUCT.unpack(raw[:ARENA4_HEADER_BYTES])
        if magic != ARENA4_MAGIC:
            raise ValueError(f"Invalid Arena 4 magic {magic!r}, expected {ARENA4_MAGIC!r}")
        if version != ARENA4_VERSION:
            raise ValueError(f"Unsupported Arena 4 version {version}")
        if total_byte_len != len(raw):
            raise ValueError(
                f"Arena 4 total_byte_len ({total_byte_len}) != buffer length ({len(raw)})"
            )

        body = raw[ARENA4_HEADER_BYTES:]
        actual_body_crc = crc32c(body)
        if actual_body_crc != expected_body_crc:
            raise ValueError(
                f"Arena 4 body CRC32C mismatch: expected {expected_body_crc:#010x}, "
                f"got {actual_body_crc:#010x}"
            )

        cursor = 0
        inputs: dict[str, InputCalibStats] = {}
        stems: dict[str, StemCalibStats] = {}

        for _ in range(n_inputs):
            if cursor + 64 > len(body):
                raise ValueError("Truncated Arena 4 input entry header")
            (
                rec_magic,
                _sha32,
                num_samples,
                w_count,
                p_dim,
                s1_in,
                s1_out,
                _pad12,
                rec_len,
                rec_crc,
            ) = ARENA4_INPUT_ENTRY_HEADER.unpack(body[cursor : cursor + 64])
            if rec_magic != ARENA4_INPUT_RECORD_MAGIC:
                raise ValueError(f"Invalid Arena 4 input record magic {rec_magic!r}")
            if rec_len < 64 or rec_len % 64 != 0 or cursor + rec_len > len(body):
                raise ValueError("Truncated or unaligned Arena 4 input entry payload")
            payload = body[cursor + 64 : cursor + rec_len]
            cursor += rec_len
            if crc32c(payload) != rec_crc:
                raise ValueError("Arena 4 input entry CRC32C mismatch")

            p_cur = 0
            (sha_len,) = struct.unpack("<H", payload[p_cur : p_cur + 2])
            p_cur += 2
            in_sha_str = payload[p_cur : p_cur + sha_len].decode("utf-8")
            p_cur += sha_len

            bounds_bytes = w_count * 2 * 4
            bounds_np = np.frombuffer(payload[p_cur : p_cur + bounds_bytes], dtype="<i4").reshape(
                w_count, 2
            )
            win_bounds = tuple((int(a), int(b)) for a, b in bounds_np)
            p_cur += bounds_bytes

            grams_bytes = w_count * p_dim * p_dim * 8
            grams_np = (
                np.frombuffer(payload[p_cur : p_cur + grams_bytes], dtype="<f8")
                .reshape(w_count, p_dim, p_dim)
                .copy()
            )
            p_cur += grams_bytes

            if s1_in > 0 and s1_out > 0:
                s1_bytes = s1_in * s1_out * 4
                s1_proj = (
                    np.frombuffer(payload[p_cur : p_cur + s1_bytes], dtype="<f4")
                    .reshape(s1_in, s1_out)
                    .copy()
                )
            else:
                s1_proj = None

            inputs[in_sha_str] = InputCalibStats(
                input_pcm_sha256=in_sha_str,
                num_samples=int(num_samples),
                win_bounds=win_bounds,
                win_grams_f64=grams_np,
                s1_proj_f32=s1_proj,
            )

        for _ in range(n_stems):
            if cursor + 128 > len(body):
                raise ValueError("Truncated Arena 4 stem entry header")
            (
                rec_magic,
                _sha32,
                flags,
                w_count,
                p_dim,
                lti_len,
                jitter_lag,
                jitter_corr,
                nf_bb,
                nf_lf,
                nf_mf,
                nf_hf,
                be_bb,
                be_lf,
                be_mf,
                be_hf,
                _pad36,
                rec_len,
                rec_crc,
                _pad4,
            ) = ARENA4_STEM_ENTRY_HEADER.unpack(body[cursor : cursor + 128])
            if rec_magic != ARENA4_STEM_RECORD_MAGIC:
                raise ValueError(f"Invalid Arena 4 stem record magic {rec_magic!r}")
            if rec_len < 128 or rec_len % 64 != 0 or cursor + rec_len > len(body):
                raise ValueError("Truncated or unaligned Arena 4 stem entry payload")
            payload = body[cursor + 128 : cursor + rec_len]
            cursor += rec_len
            if crc32c(payload) != rec_crc:
                raise ValueError("Arena 4 stem entry CRC32C mismatch")

            p_cur = 0
            id_len, sha_len = struct.unpack("<HH", payload[p_cur : p_cur + 4])
            p_cur += 4
            stem_id_str = payload[p_cur : p_cur + id_len].decode("utf-8")
            p_cur += id_len
            in_sha_str = payload[p_cur : p_cur + sha_len].decode("utf-8")
            p_cur += sha_len

            if flags & STEM_FLAG_HAS_SEG_RHS:
                rhs_bytes = w_count * p_dim * 8
                y2_bytes = w_count * 8
                seg_rhs = (
                    np.frombuffer(payload[p_cur : p_cur + rhs_bytes], dtype="<f8")
                    .reshape(w_count, p_dim)
                    .copy()
                )
                p_cur += rhs_bytes
                seg_y2 = (
                    np.frombuffer(payload[p_cur : p_cur + y2_bytes], dtype="<f8")
                    .reshape(w_count)
                    .copy()
                )
                p_cur += y2_bytes
            else:
                seg_rhs = None
                seg_y2 = None

            if flags & STEM_FLAG_HAS_LTI_IR:
                ir_bytes = lti_len * 4
                lti_ir = (
                    np.frombuffer(payload[p_cur : p_cur + ir_bytes], dtype="<f4")
                    .reshape(lti_len)
                    .copy()
                )
                p_cur += ir_bytes
            else:
                lti_ir = None

            stems[stem_id_str] = StemCalibStats(
                stem_id=stem_id_str,
                input_pcm_sha256=in_sha_str,
                is_silent=bool(flags & STEM_FLAG_IS_SILENT),
                jitter_lag_samples=float(jitter_lag),
                jitter_corr=float(jitter_corr),
                noise_floor_dbfs_broadband=float(nf_bb),
                noise_floor_dbfs_lf=float(nf_lf),
                noise_floor_dbfs_mf=float(nf_mf),
                noise_floor_dbfs_hf=float(nf_hf),
                band_energy_broadband=float(be_bb),
                band_energy_lf=float(be_lf),
                band_energy_mf=float(be_mf),
                band_energy_hf=float(be_hf),
                seg_rhs_f64=seg_rhs,
                seg_y2_f64=seg_y2,
                lti_ir_f32=lti_ir,
            )

        return cls(inputs=inputs, stems=stems)


class SidecarCacheStore:
    """Content-addressed `.dnam-cache/*.dnce` local workspace sidecar store with cost model,
    quota enforcement, and stage-level crash resume.
    """

    def __init__(
        self,
        cache_dir: Path | str | None,
        *,
        num_stems: int = 1,
        max_gb: float | None = None,
        device_kind: str = "cpu",
        observer: TrainingObserver | None = None,
    ) -> None:
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.num_stems = max(1, num_stems)
        self.device_kind = device_kind
        self.observer = observer

        self.floor_bytes: int = 100_000_000 + 90_000_000 * self.num_stems
        self.quota_bytes: int = self._resolve_quota_bytes(max_gb)
        self.measured_read_bw_bytes_per_s: float = 2.5e9  # 2.5 GB/s initial SSD estimate
        self.measured_write_bw_bytes_per_s: float = 1.5e9
        self.hits: int = 0
        self.misses: int = 0
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self._pinned_files: set[str] = set()

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._check_disk_floor()
            self.startup_maintenance()

    def _resolve_quota_bytes(self, max_gb: float | None) -> int:
        env_gb = os.environ.get("DNAM_CACHE_MAX_GB")
        if env_gb is not None:
            try:
                val = float(env_gb)
                if val > 0:
                    return int(val * (1024**3))
            except ValueError:
                pass
        if max_gb is not None and max_gb > 0:
            return int(float(max_gb) * (1024**3))

        free_bytes = self._free_disk_bytes()
        twenty_pct = int(0.20 * free_bytes)
        sixteen_gb = 16 * (1024**3)
        return max(self.floor_bytes, min(sixteen_gb, twenty_pct))

    def _free_disk_bytes(self) -> int:
        target = self.cache_dir if self.cache_dir is not None else Path.cwd()
        check_path = target if target.exists() else target.parent
        try:
            usage = shutil.disk_usage(check_path)
            return usage.free
        except OSError:
            return 64 * (1024**3)

    def _existing_cache_bytes(self) -> int:
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        total = 0
        for p in self.cache_dir.glob("*.dnce"):
            with contextlib.suppress(OSError):
                total += p.stat().st_size
        return total

    def _check_disk_floor(self) -> None:
        free_plus_reclaimable = self._free_disk_bytes() + self._existing_cache_bytes()
        if free_plus_reclaimable < self.floor_bytes:
            raise RuntimeError(
                f"Insufficient disk space for .dnam-cache: available "
                f"{free_plus_reclaimable / 1e9:.2f} GB is below required floor "
                f"{self.floor_bytes / 1e9:.2f} GB for {self.num_stems} stems."
            )

    def make_key(self, namespace: str, *parts: str | int | float) -> tuple[str, bytes]:
        """Builds a content-addressed key `(filename_stem, raw_sha256_32)`."""
        h = hashlib.sha256()
        h.update(CACHE_CODE_VERSION.encode("utf-8"))
        h.update(b"|")
        h.update(self.device_kind.encode("utf-8"))
        h.update(b"|")
        h.update(namespace.encode("utf-8"))
        for pt in parts:
            h.update(b"|")
            h.update(str(pt).encode("utf-8"))
        digest = h.digest()
        safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)[:24]
        fname = f"{safe_prefix}_{digest.hex()[:24]}.dnce"
        return fname, digest

    def should_materialize(
        self,
        payload_bytes: int,
        expected_passes: int,
        est_regen_ms: float,
    ) -> bool:
        """Cost model comparing disk write + `(expected_passes - 1)` reads against regeneration."""
        if self.cache_dir is None or expected_passes <= 1 or payload_bytes <= 0:
            return False
        if payload_bytes > self.quota_bytes:
            return False
        read_ms = (float(payload_bytes) / max(self.measured_read_bw_bytes_per_s, 1.0e7)) * 1000.0
        write_ms = (float(payload_bytes) / max(self.measured_write_bw_bytes_per_s, 1.0e7)) * 1000.0
        saved_ms = float(expected_passes - 1) * float(est_regen_ms)
        cost_ms = write_ms + float(expected_passes - 1) * read_ms
        return saved_ms > cost_ms

    def startup_maintenance(self) -> int:
        """Removes orphan `.tmp` files, corrupt headers, expired entries, and evicts LRU."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        now = time.time()
        evicted_bytes = 0

        for tmp_p in self.cache_dir.glob("*.tmp"):
            try:
                evicted_bytes += tmp_p.stat().st_size
                tmp_p.unlink(missing_ok=True)
            except OSError:
                pass

        valid_entries: list[tuple[float, int, Path]] = []
        for p in self.cache_dir.glob("*.dnce"):
            try:
                st = p.stat()
                if st.st_size < 64 or st.st_size % 64 != 0:
                    evicted_bytes += st.st_size
                    p.unlink(missing_ok=True)
                    continue
                with p.open("rb") as f:
                    hdr = f.read(64)
                magic, ver, _dt, kind, _d0, _d1, pay_len, _crc, _k32 = DNCE_PREAMBLE_STRUCT.unpack(
                    hdr
                )
                if magic != DNCE_MAGIC or ver != DNCE_VERSION or 64 + pay_len > st.st_size:
                    evicted_bytes += st.st_size
                    p.unlink(missing_ok=True)
                    continue
                age_days = (now - st.st_mtime) / 86_400.0
                max_days = 14.0 if kind == KIND_INPUT_ONLY else 7.0
                if age_days > max_days and p.name not in self._pinned_files:
                    evicted_bytes += st.st_size
                    p.unlink(missing_ok=True)
                    continue
                valid_entries.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass

        # LRU eviction down to quota_bytes
        total_bytes = sum(sz for _, sz, _ in valid_entries)
        if total_bytes > self.quota_bytes:
            valid_entries.sort(key=lambda item: item[0])  # oldest first
            for _mtime, sz, p in valid_entries:
                if total_bytes <= self.quota_bytes:
                    break
                if p.name in self._pinned_files:
                    continue
                try:
                    p.unlink(missing_ok=True)
                    total_bytes -= sz
                    evicted_bytes += sz
                except OSError:
                    pass

        if evicted_bytes > 0:
            self._emit_event("evict", "startup_maintenance", evicted_bytes)
        return evicted_bytes

    def _evict_for_space(self, needed_bytes: int) -> None:
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in self.cache_dir.glob("*.dnce"):
            try:
                st = p.stat()
                total += st.st_size
                if p.name not in self._pinned_files:
                    entries.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass
        if total + needed_bytes <= self.quota_bytes:
            return
        entries.sort(key=lambda item: item[0])
        for _mtime, sz, p in entries:
            if total + needed_bytes <= self.quota_bytes:
                break
            try:
                p.unlink(missing_ok=True)
                total -= sz
            except OSError:
                pass

    def put_tensor(
        self,
        namespace: str,
        key_parts: Sequence[str | int | float],
        tensor: Tensor,
        *,
        kind: int = KIND_STAGE_BASIS,
        pin: bool = True,
    ) -> Path | None:
        """Writes a 1D or 2D tensor into a 64-byte-aligned `.dnce` sidecar file."""
        if self.cache_dir is None:
            return None
        t0 = time.perf_counter()
        fname, key32 = self.make_key(namespace, *key_parts)
        target_path = self.cache_dir / fname
        tmp_path = self.cache_dir / f"{fname}.tmp"

        t_cpu = tensor.detach().to(device="cpu").contiguous()
        if t_cpu.ndim == 1:
            d0, d1 = t_cpu.shape[0], 1
        elif t_cpu.ndim == 2:
            d0, d1 = t_cpu.shape[0], t_cpu.shape[1]
        else:
            raise ValueError(f"SidecarCacheStore supports 1D or 2D tensors, got {t_cpu.ndim}D")

        if t_cpu.dtype == torch.float32:
            dtype_code = DTYPE_FLOAT32
            raw_bytes = t_cpu.numpy().tobytes()
        elif t_cpu.dtype == torch.bfloat16:
            dtype_code = DTYPE_BFLOAT16
            raw_bytes = t_cpu.view(torch.int16).numpy().tobytes()
        elif t_cpu.dtype == torch.int32:
            dtype_code = DTYPE_INT32
            raw_bytes = t_cpu.numpy().tobytes()
        else:
            t_f32 = t_cpu.to(torch.float32)
            dtype_code = DTYPE_FLOAT32
            raw_bytes = t_f32.numpy().tobytes()

        payload_len = len(raw_bytes)
        self._evict_for_space(64 + payload_len + 64)
        payload_crc = crc32c(raw_bytes)
        preamble = DNCE_PREAMBLE_STRUCT.pack(
            DNCE_MAGIC,
            DNCE_VERSION,
            dtype_code,
            kind,
            d0,
            d1,
            payload_len,
            payload_crc,
            key32,
        )
        padded_payload = _pad64(raw_bytes)

        with tmp_path.open("wb") as f:
            f.write(preamble)
            f.write(padded_payload)
        tmp_path.replace(target_path)

        elapsed = max(1.0e-6, time.perf_counter() - t0)
        total_written = 64 + len(padded_payload)
        self.bytes_written += total_written
        self.measured_write_bw_bytes_per_s = 0.7 * self.measured_write_bw_bytes_per_s + 0.3 * (
            float(total_written) / elapsed
        )
        if pin:
            self._pinned_files.add(fname)
        self._emit_event("write", namespace, total_written)
        return target_path

    def get_tensor(
        self,
        namespace: str,
        key_parts: Sequence[str | int | float],
        *,
        device: torch.device | str = "cpu",
        row_slice: tuple[int, int] | None = None,
        verify_crc: bool = True,
    ) -> Tensor | None:
        """Loads a tensor (or `[r0:r1]` row slice via zero-copy `mmap`) from `.dnam-cache`."""
        if self.cache_dir is None:
            return None
        fname, expected_key32 = self.make_key(namespace, *key_parts)
        path = self.cache_dir / fname
        if not path.exists():
            self.misses += 1
            self._emit_event("miss", namespace, 0)
            return None

        t0 = time.perf_counter()
        try:
            with path.open("rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    if len(mm) < 64 or len(mm) % 64 != 0:
                        raise ValueError("Invalid .dnce file size")
                    magic, ver, dt_code, _kind, d0, d1, pay_len, exp_crc, key32 = (
                        DNCE_PREAMBLE_STRUCT.unpack(mm[:64])
                    )
                    if (
                        magic != DNCE_MAGIC
                        or ver != DNCE_VERSION
                        or key32 != expected_key32
                        or 64 + pay_len > len(mm)
                    ):
                        raise ValueError("Header mismatch in .dnce file")

                    if (
                        verify_crc
                        and row_slice is None
                        and crc32c(mm[64 : 64 + pay_len]) != exp_crc
                    ):
                        raise ValueError("CRC32C mismatch in .dnce payload")

                    elem_size = 2 if dt_code == DTYPE_BFLOAT16 else 4
                    r0 = 0 if row_slice is None else max(0, min(int(d0), row_slice[0]))
                    r1 = int(d0) if row_slice is None else max(r0, min(int(d0), row_slice[1]))
                    byte_start = 64 + r0 * int(d1) * elem_size
                    byte_end = 64 + r1 * int(d1) * elem_size
                    raw_slice = mm[byte_start:byte_end]

                    if dt_code == DTYPE_FLOAT32:
                        arr = (
                            np.frombuffer(raw_slice, dtype=np.float32)
                            .reshape(r1 - r0, int(d1))
                            .copy()
                        )
                        out_t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                    elif dt_code == DTYPE_BFLOAT16:
                        arr_i16 = (
                            np.frombuffer(raw_slice, dtype=np.int16)
                            .reshape(r1 - r0, int(d1))
                            .copy()
                        )
                        out_t = torch.from_numpy(arr_i16).view(torch.bfloat16).to(device=device)
                    elif dt_code == DTYPE_INT32:
                        arr_i32 = (
                            np.frombuffer(raw_slice, dtype=np.int32)
                            .reshape(r1 - r0, int(d1))
                            .copy()
                        )
                        out_t = torch.from_numpy(arr_i32).to(device=device, dtype=torch.int32)
                    else:
                        raise ValueError(f"Unsupported tensor dtype_code {dt_code}")
                finally:
                    mm.close()

            elapsed = max(1.0e-6, time.perf_counter() - t0)
            read_len = (r1 - r0) * int(d1) * elem_size
            self.hits += 1
            self.bytes_read += read_len
            if read_len >= 65_536:
                self.measured_read_bw_bytes_per_s = (
                    0.7 * self.measured_read_bw_bytes_per_s + 0.3 * (float(read_len) / elapsed)
                )
            os.utime(path, None)
            self._emit_event("hit", namespace, read_len)
            return out_t
        except Exception as exc:
            # Corrupt or mismatched entry: ignore, delete, emit warning, and regenerate
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            self.misses += 1
            self._emit_event(
                "corrupt",
                namespace,
                0,
                warning=f"Ignored corrupt sidecar entry '{fname}': {exc}",
            )
            return None

    def write_stage_marker(
        self,
        stage_name: str,
        key_parts: Sequence[str | int | float],
        metadata: Mapping[str, Any],
    ) -> Path | None:
        """Writes a stage completion marker `.dnce` entry for crash resume."""
        if self.cache_dir is None:
            return None
        fname, key32 = self.make_key(f"marker_{stage_name}", *key_parts)
        target_path = self.cache_dir / fname
        raw_bytes = json.dumps(dict(metadata), sort_keys=True).encode("utf-8")
        payload_crc = crc32c(raw_bytes)
        preamble = DNCE_PREAMBLE_STRUCT.pack(
            DNCE_MAGIC,
            DNCE_VERSION,
            DTYPE_JSON_UTF8,
            KIND_STAGE_MARKER,
            len(raw_bytes),
            1,
            len(raw_bytes),
            payload_crc,
            key32,
        )
        padded = _pad64(raw_bytes)
        with target_path.open("wb") as f:
            f.write(preamble)
            f.write(padded)
        self._pinned_files.add(fname)
        return target_path

    def read_stage_marker(
        self,
        stage_name: str,
        key_parts: Sequence[str | int | float],
    ) -> dict[str, Any] | None:
        """Reads and CRC32C-verifies a stage completion marker `.dnce` entry."""
        if self.cache_dir is None:
            return None
        fname, expected_key32 = self.make_key(f"marker_{stage_name}", *key_parts)
        path = self.cache_dir / fname
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            if len(raw) < 64 or len(raw) % 64 != 0:
                raise ValueError("Invalid stage marker length")
            magic, ver, dt_code, kind, _d0, _d1, pay_len, exp_crc, key32 = (
                DNCE_PREAMBLE_STRUCT.unpack(raw[:64])
            )
            if (
                magic != DNCE_MAGIC
                or ver != DNCE_VERSION
                or dt_code != DTYPE_JSON_UTF8
                or kind != KIND_STAGE_MARKER
                or key32 != expected_key32
                or 64 + pay_len > len(raw)
            ):
                raise ValueError("Invalid stage marker header")
            payload = raw[64 : 64 + pay_len]
            if crc32c(payload) != exp_crc:
                raise ValueError("Stage marker CRC32C mismatch")
            result: dict[str, Any] = json.loads(payload.decode("utf-8"))
            return result
        except Exception:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            return None

    def cleanup_stage_bases(self, stage_prefix: str | None = None) -> int:
        """Deletes `KIND_STAGE_BASIS` entries at the end of a stage."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        removed = 0
        for p in self.cache_dir.glob("*.dnce"):
            if stage_prefix is not None and not p.name.startswith(stage_prefix):
                continue
            try:
                with p.open("rb") as f:
                    hdr = f.read(64)
                if len(hdr) == 64:
                    _m, _v, _dt, kind, *_ = DNCE_PREAMBLE_STRUCT.unpack(hdr)
                    if kind == KIND_STAGE_BASIS:
                        removed += p.stat().st_size
                        self._pinned_files.discard(p.name)
                        p.unlink(missing_ok=True)
            except OSError:
                pass
        return removed

    def cleanup_on_success(self) -> int:
        """Deletes stem-set, stage-basis, and marker entries on training success; retains
        input-only entries for 14 days.
        """
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        removed = 0
        for p in self.cache_dir.glob("*.dnce"):
            try:
                with p.open("rb") as f:
                    hdr = f.read(64)
                if len(hdr) == 64:
                    _m, _v, _dt, kind, *_ = DNCE_PREAMBLE_STRUCT.unpack(hdr)
                    if kind == KIND_INPUT_ONLY:
                        os.utime(p, None)
                    else:
                        removed += p.stat().st_size
                        self._pinned_files.discard(p.name)
                        p.unlink(missing_ok=True)
            except OSError:
                pass
        self._pinned_files.clear()
        return removed

    def cleanup_on_failure(self) -> None:
        """Unpins entries on failure so they remain available for crash resume for up to 7 days."""
        self._pinned_files.clear()

    def _emit_event(
        self,
        action: Literal["hit", "miss", "write", "evict", "corrupt"],
        key: str,
        bytes_delta: int,
        *,
        warning: str | None = None,
    ) -> None:
        if self.observer is None:
            return
        mapped_action: Literal["hit", "miss", "write", "evict", "corrupt_ignored", "cleanup"] = (
            "corrupt_ignored" if action == "corrupt" else action
        )
        self.observer.on_training_event(
            CacheEvent(
                tier="sidecar",
                action=mapped_action,
                key=key,
                bytes_count=max(0, bytes_delta),
                elapsed_ms=0.0,
                message=warning or "",
            )
        )


__all__ = [
    "ARENA4_MAGIC",
    "ARENA4_VERSION",
    "CACHE_CODE_VERSION",
    "DNCE_MAGIC",
    "DNCE_VERSION",
    "KIND_INPUT_ONLY",
    "KIND_STAGE_BASIS",
    "KIND_STAGE_MARKER",
    "KIND_STEM_SET",
    "Arena4CalibrationData",
    "InputCalibStats",
    "SidecarCacheStore",
    "StemCalibStats",
]
