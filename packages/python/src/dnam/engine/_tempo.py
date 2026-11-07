"""Live `Tempo` synchronization facet on `Engine` (`ref[sdk.python.api.engine-facets]`,
`ref[sdk.python.engine.free-threading]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dnam.schema.tempo import TempoSubdivision

if TYPE_CHECKING:
    from dnam.engine._engine import Engine


class Tempo:
    """Host and tap-tempo synchronization controller on `Engine`
    (`ref[dsp.tempo.physical-inversion]`).
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def bpm(self) -> float:
        """Current host tempo in Beats Per Minute (`[20.0, 400.0]` BPM)."""
        return float(self._engine._state._native.current_bpm())

    @bpm.setter
    def bpm(self, value: float) -> None:
        self.sync(value)

    def sync(
        self,
        bpm: float,
        subdivision: TempoSubdivision | str | None = None,
        *,
        fold_octave: bool = False,
    ) -> int:
        """Synchronizes tempo-calibrated controls to `bpm` and optional `subdivision`."""
        sub_str = TempoSubdivision.parse(subdivision).value if subdivision is not None else None
        with self._engine._lock:
            state = self._engine._state
            state._clamped_cache = None
            state._current_cache = None
            count = state._native.sync_tempo(float(bpm), sub_str, fold_octave)
            state._bpm = float(state._native.current_bpm())
            return count

    def tap(
        self,
        subdivision: TempoSubdivision | str | None = None,
        *,
        fold_octave: bool = False,
    ) -> tuple[float, int]:
        """Registers a tap-tempo event at the current audio frame counter."""
        sub_str = TempoSubdivision.parse(subdivision).value if subdivision is not None else None
        with self._engine._lock:
            state = self._engine._state
            state._clamped_cache = None
            state._current_cache = None
            bpm, count = state._native.register_tap_tempo(sub_str, fold_octave)
            state._bpm = float(bpm)
            return float(bpm), count

    def __repr__(self) -> str:
        return f"<Tempo bpm={self.bpm:.1f}>"
