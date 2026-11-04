"""Example 2: Live `Controls`, Atomic `Presets`, `Tempo` Sync & Context Managers
(`examples/controls_presets_and_tempo.py`).

Run with:
  uv run --no-sync python packages/python/examples/controls_presets_and_tempo.py
"""

from __future__ import annotations

import numpy as np

import dnam


def main() -> None:
    """Demonstrates `engine.controls`, `engine.presets`, `engine.tempo`, and scoped state."""
    engine = dnam.Engine.reference(sample_rate=48_000, tier="full")
    silence = np.zeros((2, 128), dtype=np.float32)

    # 1. MutableMapping control rack + 3-Phase clamp diagnostics
    engine.controls["gain"] = 12.0  # Clamped to 9.5 dB when boost=off
    clamp_report = engine.controls.clamped
    print(
        f"Slewed gain=12.0 -> clamped={engine.controls['gain']} "
        f"| Clamped IDs: {clamp_report.clamped_ids}"
    )

    # 2. Conditional range clamp when switch `boost="on"` (max gain clamped to 8.0 dB)
    engine.controls.snap(gain=9.0, boost="on")
    print(
        f"Snapped (gain=9.0, boost='on') -> clamped gain={engine.controls['gain']} "
        f"| Warning: {engine.controls.clamped['gain'].clamp_warning}"
    )

    # 3. Atomic preset batching & 12-subdivision tempo synchronization
    engine.presets.apply("warm_slapback")
    updated = engine.tempo.sync(bpm=120.0, subdivision="1/8.")
    print(
        f"Applied preset 'warm_slapback' & synced tempo (120 BPM, 1/8.) -> "
        f"updated {updated} control(s), delay_ms={engine.controls['delay_ms']:.1f} ms"
    )

    # 4. Scoped bypass (`SpilloverTrails`) and captured hardware freeze (`frozen`)
    with engine.bypassed(trails=True):
        assert engine.is_bypassed
        assert engine.trails_enabled
        _ = engine.process(silence)
    with engine.frozen(amount=1.0):
        assert engine.is_frozen
        _ = engine.process(silence)

    tel = engine.telemetry
    print(
        f"Telemetry | tier={tel.active_tier.label} sr={tel.sample_rate_hz}Hz "
        f"bpm={tel.current_bpm:.1f} pole_max={tel.max_pole_radius:.4f}"
    )


if __name__ == "__main__":
    main()
