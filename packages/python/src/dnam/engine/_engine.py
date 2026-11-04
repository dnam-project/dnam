"""Stateful real-time `Engine` processor (`ref[sdk.python.api.core-types]`,
`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.engine.pedalboard-call]`).
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from threading import Lock

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dnam._arrays import (
    _F32_DTYPE,
    coerce_audio_f32,
    coerce_f32_1d,
    validate_f32_out,
)
from dnam._native import PyEngine
from dnam.engine._controls import Controls
from dnam.engine._presets import Presets
from dnam.engine._telemetry import Telemetry
from dnam.engine._tempo import Tempo
from dnam.errors import ControlError, ValidationError
from dnam.model._checkpoint import Checkpoint
from dnam.model._model import Model
from dnam.schema._base import _BACKEND_BY_STR, _TIER_TUPLE, Backend, Tier


def _arrays_overlap(a: NDArray[np.float32], b: NDArray[np.float32]) -> bool:
    """Fast O(1) memory-overlap check avoiding `np.shares_memory` on independent arrays."""
    if a is b:
        return True
    if a.base is None and b.base is None:
        return False
    return np.shares_memory(a, b)


class Engine:
    """
    Real-time SIMD DNAM processor with `pedalboard`-compatible `__call__` and structured facets.
    """

    __slots__ = (
        "_backend",
        "_controls",
        "_is_stereo_out",
        "_lock",
        "_model",
        "_native",
        "_presets",
        "_sample_rate",
        "_tempo",
    )

    def __init__(
        self,
        model: Model | Checkpoint | str | PathLike[str] | bytes,
        *,
        sample_rate: int | float = 48_000,
        tier: Tier | str | int = Tier.FULL,
        backend: Backend | str = Backend.AUTO,
        max_supported_tier: Tier | str | int | None = None,
        offline_bounce: bool = False,
        maximize_tier_on_bounce: bool = True,
    ) -> None:
        resolved_model = self._coerce_model(model)
        parsed_tier = Tier.parse(tier)
        parsed_backend = Backend.parse(backend)
        parsed_max_tier = Tier.parse(max_supported_tier) if max_supported_tier is not None else None
        sr_int = sample_rate if type(sample_rate) is int else round(float(sample_rate))

        self._native = PyEngine(
            resolved_model.to_bytes(),
            int(parsed_tier),
            sr_int,
            parsed_backend.value,
            int(parsed_max_tier) if parsed_max_tier is not None else None,
            offline_bounce,
            maximize_tier_on_bounce,
        )
        self._model = resolved_model
        self._is_stereo_out = resolved_model.preamble.is_stereo_out
        self._sample_rate = sr_int
        self._backend = _BACKEND_BY_STR[self._native.backend_kind()]
        self._controls = Controls(self._native, resolved_model.control_space)
        self._presets = Presets(self._native, self._controls, resolved_model.presets)
        self._tempo = Tempo(self._native, self._controls)
        self._lock = Lock()

    @staticmethod
    def _coerce_model(
        source: Model | Checkpoint | str | PathLike[str] | bytes,
    ) -> Model:
        if isinstance(source, Model):
            return source
        if isinstance(source, Checkpoint):
            return source.export_model()
        if isinstance(source, (bytes, bytearray, memoryview)):
            return Model.from_bytes(source)
        if isinstance(source, (str, PathLike)):
            return Model.load(source)
        raise ValidationError(
            f"Expected Model, Checkpoint, path, or bytes, got {type(source).__name__}"
        )

    @classmethod
    def from_file(
        cls,
        path: str | PathLike[str],
        *,
        sample_rate: int | float = 48_000,
        tier: Tier | str | int = Tier.FULL,
        backend: Backend | str = Backend.AUTO,
    ) -> Engine:
        """Loads a `.dnam` file from `path` and constructs a ready-to-run `Engine`."""
        return cls(
            Model.load(path),
            sample_rate=sample_rate,
            tier=tier,
            backend=backend,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        sample_rate: int | float = 48_000,
        tier: Tier | str | int = Tier.FULL,
        backend: Backend | str = Backend.AUTO,
    ) -> Engine:
        """Loads a `.dnam` buffer from `data` and constructs a ready-to-run `Engine`."""
        return cls(
            Model.from_bytes(data),
            sample_rate=sample_rate,
            tier=tier,
            backend=backend,
        )

    @classmethod
    def reference(
        cls,
        *,
        sample_rate: int | float = 48_000,
        tier: Tier | str | int = Tier.FULL,
        backend: Backend | str = Backend.AUTO,
        supports_freeze: bool = True,
    ) -> Engine:
        """Constructs an `Engine` loaded with the canonical reference `.dnam` `Model`."""
        return cls(
            Model.reference(supports_freeze=supports_freeze),
            sample_rate=sample_rate,
            tier=tier,
            backend=backend,
        )

    @property
    def model(self) -> Model:
        """Currently loaded `.dnam` `Model`."""
        return self._model

    @property
    def controls(self) -> Controls:
        """Live `Controls` (`MutableMapping[str, float | str]`) facet."""
        return self._controls

    @property
    def presets(self) -> Presets:
        """Live `Presets` (`Mapping[str, PresetDef]`) facet."""
        return self._presets

    @property
    def tempo(self) -> Tempo:
        """Live `Tempo` synchronization facet."""
        return self._tempo

    @property
    def telemetry(self) -> Telemetry:
        """Lock-free `Telemetry` snapshot of current engine, control-bus, and ZOH state."""
        raw = self._native.telemetry()
        raw["active_tier"] = _TIER_TUPLE[self._native.effective_tier()]
        raw["backend"] = self._backend
        return Telemetry._construct_trusted_dict(raw)

    @property
    def tier(self) -> Tier:
        """Requested slimmable `Tier` (`LITE`, `FULL`, `PRO`, `ULTRA`)."""
        return _TIER_TUPLE[self._native.active_tier()]

    @tier.setter
    def tier(self, value: Tier | str | int) -> None:
        self._native.set_tier(int(Tier.parse(value)))

    @property
    def max_available_tier(self) -> Tier:
        """Highest `Tier` present in the loaded `.dnam` container."""
        return _TIER_TUPLE[self._native.max_available_tier()]

    @property
    def max_supported_tier(self) -> Tier:
        """Host ceiling `Tier` (`max_supported_tier`)."""
        return _TIER_TUPLE[self._native.max_supported_tier()]

    @max_supported_tier.setter
    def max_supported_tier(self, value: Tier | str | int) -> None:
        self._native.set_max_supported_tier(int(Tier.parse(value)))

    @property
    def effective_tier(self) -> Tier:
        """
        Effective active `Tier` (incorporating `offline_bounce` and `maximize_tier_on_bounce`).
        """
        return _TIER_TUPLE[self._native.effective_tier()]

    @property
    def backend(self) -> Backend:
        """Active hardware SIMD or scalar `Backend`."""
        return self._backend

    @property
    def sample_rate(self) -> int:
        """Active host sample rate in Hz (`[44100, 192000]`)."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        sr_int = value if type(value) is int else round(float(value))
        self._native.set_sample_rate(sr_int)
        self._sample_rate = sr_int

    @property
    def latency_samples(self) -> int:
        """Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`)."""
        return self._native.latency_samples()

    @property
    def offline_bounce(self) -> bool:
        """`True` if $4\\times$ polyphase offline bounce mode is enabled."""
        return self._native.offline_bounce()

    @offline_bounce.setter
    def offline_bounce(self, enabled: bool) -> None:
        self._native.set_offline_bounce(enabled)

    @property
    def maximize_tier_on_bounce(self) -> bool:
        """`True` if tier is promoted to `min(max_available_tier, max_supported_tier)` on bounce."""
        return self._native.maximize_tier_on_bounce()

    @maximize_tier_on_bounce.setter
    def maximize_tier_on_bounce(self, enabled: bool) -> None:
        self._native.set_maximize_tier_on_bounce(enabled)

    @property
    def supports_freeze(self) -> bool:
        """`True` if the loaded model supports Stage IV captured hardware freeze."""
        return self._native.supports_freeze()

    @property
    def is_frozen(self) -> bool:
        """`True` if Stage IV unitary spectral freeze is currently engaged."""
        return self._native.is_frozen()

    def set_freeze(self, freeze: bool | float) -> None:
        """Engages or disengages Stage IV unitary spectral freeze."""
        if not self.supports_freeze and bool(freeze):
            raise ControlError(
                f"Model {self._model.name!r} does not support captured hardware freeze"
            )
        if isinstance(freeze, bool):
            self._native.set_freeze(freeze)
        else:
            self._native.set_freeze_amount(float(freeze))

    @contextmanager
    def frozen(self, amount: float = 1.0) -> Generator[Engine]:
        """
        Context manager that engages captured hardware freeze and restores prior state on exit.
        """
        if not self.supports_freeze:
            raise ControlError(
                f"Model {self._model.name!r} does not support captured hardware freeze"
            )
        prev = self.is_frozen
        self._native.set_freeze_amount(float(amount))
        try:
            yield self
        finally:
            self._native.set_freeze(prev)

    @property
    def is_bypassed(self) -> bool:
        """`True` if the engine is currently bypassed."""
        return self._native.bypass_state()[0]

    @property
    def trails_enabled(self) -> bool:
        """`True` if spillover trails (`SpilloverTrails`) are enabled when bypassed."""
        return self._native.bypass_state()[1]

    def set_bypass(self, bypassed: bool, *, trails: bool = True) -> None:
        """Sets the engine bypass state (`SpilloverTrails` or `HardBypass`)."""
        self._native.set_bypass(bypassed, trails)

    @contextmanager
    def bypassed(self, *, trails: bool = True) -> Generator[Engine]:
        """Context manager that enters bypass and restores prior `(bypassed, trails)` state on
        exit.
        """
        prev_byp, prev_trails = self._native.bypass_state()
        self._native.set_bypass(True, trails)
        try:
            yield self
        finally:
            self._native.set_bypass(prev_byp, prev_trails)

    @property
    def width(self) -> float:
        """Output Mid/Side stereo width in `[0.0, 2.0]` (`1.0` = nominal stereo, `0.0` = mono)."""
        return float(self._native.stereo_width())

    @width.setter
    def width(self, value: float) -> None:
        self._native.set_stereo_width(float(value))

    def configure_host_mix(
        self,
        *,
        blend_wet: float = 1.0,
        output_gain_db: float = 0.0,
        apply_unity_leveling: bool = True,
    ) -> None:
        """
        Configures host parallel dry/wet blend, output gain trim (dB), and auto-unity leveling.
        """
        self._native.configure_host_mix_and_leveling(
            float(blend_wet),
            float(output_gain_db),
            apply_unity_leveling,
        )

    def reload(
        self,
        model: Model | Checkpoint | str | PathLike[str] | bytes,
        *,
        tier: Tier | str | int | None = None,
        sample_rate: int | float | None = None,
    ) -> None:
        """Hot-swaps a new `.dnam` `Model` in-place without reallocating engine scratch memory."""
        resolved = self._coerce_model(model)
        t_int = int(Tier.parse(tier)) if tier is not None else None
        sr_int = (
            (sample_rate if type(sample_rate) is int else round(float(sample_rate)))
            if sample_rate is not None
            else None
        )
        with self._lock:
            self._native.reload(
                resolved.to_bytes(),
                t_int,
                sr_int,
            )
            self._model = resolved
            self._is_stereo_out = resolved.preamble.is_stereo_out
            if sr_int is not None:
                self._sample_rate = sr_int
            self._controls = Controls(self._native, resolved.control_space)
            self._presets = Presets(self._native, self._controls, resolved.presets)
            self._tempo = Tempo(self._native, self._controls)

    def reset(self) -> None:
        """Resets all delay, modal, cascade, and modulator states to analytical DC resting
        equilibrium.
        """
        with self._lock:
            self._native.reset()

    def process_planar(
        self,
        left: ArrayLike,
        right: ArrayLike | None = None,
        *,
        out_left: NDArray[np.float32] | None = None,
        out_right: NDArray[np.float32] | None = None,
        buffer_size: int | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Processes 1D planar `left` (and optional `right`) buffers into stereo `(out_left,
        out_right)`.
        """
        l_in = coerce_f32_1d(left, name="left")
        n = l_in.shape[0]
        if out_left is not None:
            validate_f32_out(out_left, (n,), name="out_left")
            l_out = out_left
        else:
            l_out = np.empty(n, dtype=np.float32)

        if out_right is not None:
            validate_f32_out(out_right, (n,), name="out_right")
            r_out = out_right
        else:
            r_out = np.empty(n, dtype=np.float32)

        with self._lock:
            if right is None:
                self._native.process_mono_to_stereo(
                    l_in,
                    l_out,
                    r_out,
                    buffer_size,
                )
            else:
                r_in = coerce_f32_1d(right, name="right")
                if r_in.shape[0] != n:
                    raise ValidationError(
                        f"left and right must have identical length, got {n} and {r_in.shape[0]}"
                    )
                if _arrays_overlap(l_in, l_out) and _arrays_overlap(r_in, r_out):
                    self._native.process_stereo_in_place(
                        l_out,
                        r_out,
                        buffer_size,
                    )
                else:
                    self._native.process_stereo(
                        l_in,
                        r_in,
                        l_out,
                        r_out,
                        buffer_size,
                    )
        return l_out, r_out

    def process_into(
        self,
        audio: ArrayLike,
        out: NDArray[np.float32],
        *,
        buffer_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Zero-allocation block processing writing directly into caller-supplied `out`."""
        if (
            buffer_size is None
            and self._is_stereo_out
            and type(audio) is np.ndarray
            and type(out) is np.ndarray
            and audio.dtype == _F32_DTYPE
            and out.dtype == _F32_DTYPE
            and audio.ndim == 2
            and audio.shape[0] == 2
            and out.shape == audio.shape
            and audio.flags.c_contiguous
            and out.flags.c_contiguous
            and out.flags.writeable
        ):
            with self._lock:
                if _arrays_overlap(audio, out):
                    self._native.process_stereo_2d_in_place(out, None)
                else:
                    self._native.process_stereo_2d(audio, out, None)
            return out
        return self.process(audio, out=out, buffer_size=buffer_size)

    def process(
        self,
        audio: ArrayLike,
        *,
        out: NDArray[np.float32] | None = None,
        buffer_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Processes 1D `(N,)` or 2D `(1, N)` / `(2, N)` audio through the engine."""
        if buffer_size is not None and buffer_size <= 0:
            raise ValidationError(f"buffer_size must be positive, got {buffer_size}")
        arr, in_ch, was_1d = coerce_audio_f32(audio)
        n = arr.shape[-1]
        is_stereo_out = self._is_stereo_out

        with self._lock:
            if in_ch == 2:
                if is_stereo_out:
                    expected_shape = (2, n)
                    if out is not None:
                        validate_f32_out(out, expected_shape)
                        stereo_out = out
                        if _arrays_overlap(arr, stereo_out):
                            self._native.process_stereo_2d_in_place(
                                stereo_out,
                                buffer_size,
                            )
                        else:
                            self._native.process_stereo_2d(
                                arr,
                                stereo_out,
                                buffer_size,
                            )
                    else:
                        stereo_out = np.empty(expected_shape, dtype=np.float32)
                        self._native.process_stereo_2d(
                            arr,
                            stereo_out,
                            buffer_size,
                        )
                    return stereo_out

                mono_fold = np.ascontiguousarray(0.5 * (arr[0] + arr[1]), dtype=np.float32)
                expected_shape = (1, n)
                if out is not None:
                    validate_f32_out(out, expected_shape)
                    mono_out_2d = out
                else:
                    mono_out_2d = np.empty(expected_shape, dtype=np.float32)
                self._native.process_mono(
                    mono_fold,
                    mono_out_2d[0],
                    buffer_size,
                )
                return mono_out_2d

            mono_1d = arr if was_1d else arr[0]
            if is_stereo_out:
                expected_shape = (2, n)
                if out is not None:
                    validate_f32_out(out, expected_shape)
                    stereo_out = out
                else:
                    stereo_out = np.empty(expected_shape, dtype=np.float32)
                self._native.process_mono_to_stereo_2d(
                    mono_1d,
                    stereo_out,
                    buffer_size,
                )
                return stereo_out

            expected_shape = (n,) if was_1d else (1, n)
            if out is not None:
                validate_f32_out(out, expected_shape)
                target = out
            else:
                target = np.empty(expected_shape, dtype=np.float32)
            target_1d = target if was_1d else target[0]
            self._native.process_mono(
                mono_1d,
                target_1d,
                buffer_size,
            )
            return target

    def bounce(
        self,
        audio: ArrayLike,
        *,
        maximize_tier: bool = True,
    ) -> NDArray[np.float32]:
        """Renders `audio` in $4\\times$ linear-phase polyphase Offline Bounce mode with PDC
        stripped.
        """
        arr, in_ch, _ = coerce_audio_f32(audio)
        n = arr.shape[-1]
        if in_ch == 1:
            l_in = arr if arr.ndim == 1 else arr[0]
            r_in = l_in
        else:
            l_in = arr[0]
            r_in = arr[1]

        out = np.empty((2, n), dtype=np.float32)
        with self._lock:
            prev_max = self.maximize_tier_on_bounce
            self.maximize_tier_on_bounce = maximize_tier
            try:
                self._native.render_offline_bounce(
                    l_in,
                    r_in,
                    out[0],
                    out[1],
                )
            finally:
                self.maximize_tier_on_bounce = prev_max
        return out

    def __call__(
        self,
        audio: ArrayLike,
        sample_rate: int | float | None = None,
        *,
        buffer_size: int = 8192,
        reset: bool = True,
    ) -> NDArray[np.float32]:
        """`Spotify pedalboard`-compatible callable (`engine(audio, sample_rate, buffer_size=8192,
        reset=True)`).
        """
        if sample_rate is not None:
            sr_int = sample_rate if type(sample_rate) is int else round(float(sample_rate))
            if sr_int != self._sample_rate:
                self.sample_rate = sr_int
        if reset:
            self._native.reset()
        return self.process(audio, buffer_size=buffer_size)

    def __repr__(self) -> str:
        return (
            f"<Engine model={self._model.name!r} tier={self.tier.label} "
            f"backend={self.backend.value} sample_rate={self._sample_rate}Hz>"
        )
