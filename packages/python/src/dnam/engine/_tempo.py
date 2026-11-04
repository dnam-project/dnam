"""Live `Tempo` synchronization facet on `Engine` (`ref[sdk.python.api.engine-facets]`)."""

from __future__ import annotations

from dnam._native import PyEngine
from dnam.engine._controls import Controls
from dnam.schema.tempo import TempoSubdivision


class Tempo:
    """Host and tap-tempo synchronization controller on `Engine`
    (`ref[dsp.tempo.physical-inversion]`).
    """

    __slots__ = ("_controls", "_native")

    def __init__(self, native: PyEngine, controls: Controls) -> None:
        self._native = native
        self._controls = controls

    @property
    def bpm(self) -> float:
        """Current host tempo in Beats Per Minute (`[20.0, 400.0]` BPM)."""
        return float(self._native.current_bpm())

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
        count = self._native.sync_tempo(float(bpm), sub_str, fold_octave)
        if count > 0:
            self._controls._invalidate()
        return count

    def tap(
        self,
        subdivision: TempoSubdivision | str | None = None,
        *,
        fold_octave: bool = False,
    ) -> tuple[float, int]:
        """Registers a tap-tempo event at the current audio frame counter."""
        sub_str = TempoSubdivision.parse(subdivision).value if subdivision is not None else None
        bpm, count = self._native.register_tap_tempo(sub_str, fold_octave)
        if count > 0:
            self._controls._invalidate()
        return float(bpm), count

    def __repr__(self) -> str:
        return f"<Tempo bpm={self.bpm:.1f}>"
