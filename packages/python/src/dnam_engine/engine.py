"""Callable `DnamEngine` (`pedalboard`-compatible duck-typing, live `ControlRack`,
SLIMMABLE tiers, Continuous-Time ZOH rescaling, explicit SIMD/Scalar backend selection,
and zero-allocation model hot-reloading)."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam_engine import _native
from dnam_engine.container import DnamContainer
from dnam_engine.control import ControlRack
from dnam_engine.exceptions import DnamValidationError, call_native
from dnam_engine.schema.base import BackendKind, Tier
from dnam_engine.schema.runtime import DnamEngineSettings, DnamTelemetrySnapshot


def _coerce_f32_1d(arr: ArrayLike) -> NDArray[np.float32]:
    """Zero-copy fast path for 1D C-contiguous `float32` arrays; coerces other `ArrayLike`."""
    if isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _validate_f32_out(out: NDArray[np.float32], expected_shape: tuple[int, ...]) -> None:
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.float32
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise DnamValidationError("out= buffer must be a writeable C-contiguous float32 ndarray")
    if out.shape != expected_shape:
        raise DnamValidationError(
            f"out= buffer shape {out.shape} does not match expected shape {expected_shape}"
        )


def _parse_tier(tier: Tier | int) -> Tier:
    try:
        return Tier(int(tier))
    except (ValueError, TypeError) as exc:
        raise DnamValidationError(f"Invalid tier: {tier!r}") from exc


def _parse_backend(backend: str | BackendKind) -> BackendKind:
    try:
        return BackendKind(str(backend))
    except ValueError as exc:
        raise DnamValidationError(f"Invalid backend: {backend!r}") from exc


class DnamEngine:
    """Real-time and offline stereo neural audio modeler (`Tri-Physics`) processor."""

    def __init__(
        self,
        container: DnamContainer | bytes | bytearray | memoryview,
        settings: DnamEngineSettings | None = None,
        *,
        tier: Tier | int = Tier.FULL,
        sample_rate: float = 48_000.0,
        backend: BackendKind
        | Literal[
            "auto", "scalar", "arm_neon", "arm_sme", "x86_simd", "wasm_simd128"
        ] = BackendKind.AUTO,
        buffer_size: int | None = None,
    ) -> None:
        self._container = (
            container
            if isinstance(container, DnamContainer)
            else DnamContainer.from_bytes(container)
        )
        if settings is not None:
            if not isinstance(settings, DnamEngineSettings):
                raise DnamValidationError("settings must be a DnamEngineSettings instance")
            self._settings = settings
        else:
            self._settings = DnamEngineSettings(
                tier=_parse_tier(tier),
                sample_rate_hz=float(sample_rate),
                backend=_parse_backend(backend),
                buffer_size=buffer_size,
            )

        self._sample_rate_hz = round(self._settings.sample_rate_hz)
        self._native_engine = call_native(
            _native.NativeDnamEngine,
            self._container.to_bytes(),
            int(self._settings.tier),
            self._sample_rate_hz,
            str(self._settings.backend),
        )
        self._backend = BackendKind(self._native_engine.backend_kind())
        self._controls = self._bind_control_rack()

    def _bind_control_rack(self) -> ControlRack:
        return ControlRack(
            self._container.control_space,
            set_slew_fn=self._native_engine.set_control_value,
            set_snap_fn=self._native_engine.set_control_immediate,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        settings: DnamEngineSettings | None = None,
        *,
        tier: Tier | int = Tier.FULL,
        sample_rate: float = 48_000.0,
        backend: BackendKind
        | Literal[
            "auto", "scalar", "arm_neon", "arm_sme", "x86_simd", "wasm_simd128"
        ] = BackendKind.AUTO,
        buffer_size: int | None = None,
    ) -> DnamEngine:
        """Constructs a `DnamEngine` from `.dnam` or `.dnam-ckpt` bytes."""
        return cls(
            DnamContainer.from_bytes(data),
            settings,
            tier=tier,
            sample_rate=sample_rate,
            backend=backend,
            buffer_size=buffer_size,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        settings: DnamEngineSettings | None = None,
        *,
        tier: Tier | int = Tier.FULL,
        sample_rate: float = 48_000.0,
        backend: BackendKind
        | Literal[
            "auto", "scalar", "arm_neon", "arm_sme", "x86_simd", "wasm_simd128"
        ] = BackendKind.AUTO,
        buffer_size: int | None = None,
    ) -> DnamEngine:
        """Loads a `DnamEngine` from a `.dnam` or `.dnam-ckpt` file on disk."""
        return cls(
            DnamContainer.from_file(path),
            settings,
            tier=tier,
            sample_rate=sample_rate,
            backend=backend,
            buffer_size=buffer_size,
        )

    @classmethod
    def reference(
        cls,
        *,
        core2_active: bool = False,
        is_checkpoint: bool = False,
        tier: Tier | int = Tier.FULL,
        sample_rate: float = 48_000.0,
        backend: BackendKind
        | Literal[
            "auto", "scalar", "arm_neon", "arm_sme", "x86_simd", "wasm_simd128"
        ] = BackendKind.AUTO,
        buffer_size: int | None = None,
    ) -> DnamEngine:
        """Constructs a `DnamEngine` loaded with a canonical reference container."""
        return cls(
            DnamContainer.reference(core2_active=core2_active, is_checkpoint=is_checkpoint),
            tier=tier,
            sample_rate=sample_rate,
            backend=backend,
            buffer_size=buffer_size,
        )

    def reload(
        self,
        container: DnamContainer | bytes | bytearray | memoryview,
        *,
        tier: Tier | int | None = None,
        sample_rate: float | None = None,
    ) -> None:
        """Hot-swaps a new `.dnam` or `.dnam-ckpt` model in-place without reallocating buffers."""
        new_container = (
            container
            if isinstance(container, DnamContainer)
            else DnamContainer.from_bytes(container)
        )
        tier_int = int(_parse_tier(tier)) if tier is not None else None
        sr_int = round(float(sample_rate)) if sample_rate is not None else None
        call_native(
            self._native_engine.reload,
            new_container.to_bytes(),
            tier_int,
            sr_int,
        )
        if sr_int is not None:
            self._sample_rate_hz = sr_int
        self._container = new_container
        self._controls = self._bind_control_rack()

    @property
    def container(self) -> DnamContainer:
        """Returns the currently loaded `DnamContainer`."""
        return self._container

    @property
    def controls(self) -> ControlRack:
        """Returns the live `ControlRack` `MutableMapping` proxy for reading/mutating controls."""
        return self._controls

    @property
    def tier(self) -> Tier:
        """Returns or switches the active SLIMMABLE execution tier (`Lite`, `Full`, `Ultra`)."""
        return Tier(self._native_engine.active_tier())

    @tier.setter
    def tier(self, new_tier: Tier | int) -> None:
        t = _parse_tier(new_tier)
        self._native_engine.set_tier(int(t))

    @property
    def sample_rate(self) -> float:
        """Returns or updates the host sample rate (`44100..=96000` Hz) via Continuous-Time ZOH."""
        return float(self._sample_rate_hz)

    @sample_rate.setter
    def sample_rate(self, new_sample_rate_hz: float) -> None:
        sr_int = round(float(new_sample_rate_hz))
        call_native(self._native_engine.set_sample_rate, sr_int)
        self._sample_rate_hz = sr_int

    @property
    def backend(self) -> BackendKind:
        """Returns the active hardware SIMD or scalar backend (`BackendKind`)."""
        return self._backend

    @property
    def telemetry(self) -> DnamTelemetrySnapshot:
        """Returns a lock-free `DnamTelemetrySnapshot` of the engine's current DSP state."""
        raw = call_native(self._native_engine.telemetry)
        return DnamTelemetrySnapshot._construct_trusted(
            active_tier=Tier(int(raw["active_tier"])),
            backend=self._backend,
            sample_rate_hz=int(raw["sample_rate_hz"]),
            core2_active=bool(raw["core2_active"]),
            lfo_env_active=bool(raw["lfo_env_active"]),
            stereo_out=bool(raw["stereo_out"]),
            active_clamp_mask=int(raw["active_clamp_mask"]),
            control_decimation_host=int(raw["control_decimation_host"]),
            max_adaa_drive=float(raw["max_adaa_drive"]),
            max_pole_radius=float(raw["max_pole_radius"]),
            max_effective_loop_gain=float(raw["max_effective_loop_gain"]),
            tail_time_ms=float(raw["tail_time_ms"]),
            z_head=raw["z_head"],
            env_states=raw["env_states"],
            lfo_harmonics=raw["lfo_harmonics"],
            clamped_controls=raw["clamped_controls"],
            rho_sr=float(raw["rho_sr"]),
            rho_ctrl=float(raw["rho_ctrl"]),
            r_ac_host=float(raw["r_ac_host"]),
            r_dc_host=float(raw["r_dc_host"]),
            tau_ref_samples_host=float(raw["tau_ref_samples_host"]),
            switch_xfade_samples_host=int(raw["switch_xfade_samples_host"]),
        )

    def reset(self) -> None:
        """Resets all Core 1 (`PolyADAA`, `SelectiveRgLru`) and Core 2 (`M-UFDN`) states."""
        self._native_engine.reset()

    def __enter__(self) -> DnamEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.reset()

    def process_planar(
        self,
        left_in: ArrayLike,
        right_in: ArrayLike,
        *,
        left_out: NDArray[np.float32] | None = None,
        right_out: NDArray[np.float32] | None = None,
        buffer_size: int | None = None,
        reset: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Processes separate planar 1D `left_in` and `right_in` buffers."""
        if reset:
            self.reset()
        l_in = _coerce_f32_1d(left_in)
        r_in = _coerce_f32_1d(right_in)
        if l_in.ndim != 1 or r_in.ndim != 1 or l_in.shape != r_in.shape:
            raise DnamValidationError(
                f"process_planar requires matching 1D arrays; got {l_in.shape} and {r_in.shape}"
            )
        if left_out is not None:
            _validate_f32_out(left_out, l_in.shape)
            l_out = left_out
        else:
            l_out = np.empty_like(l_in, dtype=np.float32)
        if right_out is not None:
            _validate_f32_out(right_out, r_in.shape)
            r_out = right_out
        else:
            r_out = np.empty_like(r_in, dtype=np.float32)

        eff_chunk = buffer_size if buffer_size is not None else self._settings.buffer_size
        call_native(
            self._native_engine.process_stereo,
            l_in,
            r_in,
            l_out,
            r_out,
            eff_chunk,
        )
        return l_out, r_out

    def process(
        self,
        input_array: ArrayLike,
        sample_rate: float | None = None,
        *,
        buffer_size: int | None = None,
        reset: bool = False,
        mono_downmix: bool = False,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Processes 1D mono `(N,)`, 2D mono `(1, N)`, or 2D stereo `(2, N)` audio arrays.

        - Compatible with `pedalboard` (`engine(audio, sample_rate, buffer_size=..., reset=...)`).
        - Defaults mono `(N,)` or `(1, N)` input to true-stereo `(2, N)` output unless
          `mono_downmix=True` is passed (which returns `(N,)` or `(1, N)` via `process_mono`).
        """
        if sample_rate is not None and round(float(sample_rate)) != self._sample_rate_hz:
            self.sample_rate = float(sample_rate)
        if reset:
            self.reset()

        arr = (
            input_array
            if isinstance(input_array, np.ndarray)
            and input_array.dtype == np.float32
            and input_array.flags.c_contiguous
            else np.ascontiguousarray(input_array, dtype=np.float32)
        )
        eff_chunk = buffer_size if buffer_size is not None else self._settings.buffer_size

        if arr.ndim == 2 and arr.shape[0] == 2:
            n = int(arr.shape[1])
            if mono_downmix:
                raise DnamValidationError(
                    "mono_downmix=True is only supported for mono (N,) or (1, N) input arrays"
                )
            if out is not None:
                _validate_f32_out(out, (2, n))
                target_2d = out
            else:
                target_2d = np.empty((2, n), dtype=np.float32)
            call_native(
                self._native_engine.process_stereo_2d,
                arr,
                target_2d,
                eff_chunk,
            )
            return target_2d

        if arr.ndim == 1:
            n = int(arr.shape[0])
            if mono_downmix:
                if out is not None:
                    _validate_f32_out(out, (n,))
                    target_1d = out
                else:
                    target_1d = np.empty((n,), dtype=np.float32)
                call_native(self._native_engine.process_mono, arr, target_1d, eff_chunk)
                return target_1d

            if out is not None:
                _validate_f32_out(out, (2, n))
                target_2d = out
            else:
                target_2d = np.empty((2, n), dtype=np.float32)
            call_native(
                self._native_engine.process_mono_to_stereo_2d,
                arr,
                target_2d,
                eff_chunk,
            )
            return target_2d

        if arr.ndim == 2 and arr.shape[0] == 1:
            n = int(arr.shape[1])
            mono_1d = arr[0]
            if mono_downmix:
                if out is not None:
                    _validate_f32_out(out, (1, n))
                    target_2d_mono = out
                else:
                    target_2d_mono = np.empty((1, n), dtype=np.float32)
                call_native(
                    self._native_engine.process_mono,
                    mono_1d,
                    target_2d_mono[0],
                    eff_chunk,
                )
                return target_2d_mono

            if out is not None:
                _validate_f32_out(out, (2, n))
                target_2d = out
            else:
                target_2d = np.empty((2, n), dtype=np.float32)
            call_native(
                self._native_engine.process_mono_to_stereo_2d,
                mono_1d,
                target_2d,
                eff_chunk,
            )
            return target_2d

        raise DnamValidationError(
            f"DnamEngine expected (N,), (1, N), or (2, N) input; got shape {arr.shape}"
        )

    def __call__(
        self,
        input_array: ArrayLike,
        sample_rate: float | None = None,
        *,
        buffer_size: int | None = None,
        reset: bool = False,
        mono_downmix: bool = False,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        return self.process(
            input_array,
            sample_rate,
            buffer_size=buffer_size,
            reset=reset,
            mono_downmix=mono_downmix,
            out=out,
        )
