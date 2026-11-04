"""Example 1: Real-Time Stereo Block Processing, `pedalboard` Call & 4x Offline Bounce
(`examples/realtime_audio_and_bounce.py`).

Run with:
  uv run --no-sync python packages/python/examples/realtime_audio_and_bounce.py
"""

from __future__ import annotations

import numpy as np

import dnam


def main() -> None:
    """Demonstrates zero-copy `process_into`, `process_planar`, `__call__`, and `bounce`."""
    model = dnam.Model.reference()
    engine = dnam.Engine(model, sample_rate=48_000, tier="full")

    # 1. Zero-allocation 128-frame stereo block processing across all 4 slimmable tiers
    t = np.linspace(0.0, 128.0 / 48_000.0, 128, endpoint=False, dtype=np.float32)
    audio_in = np.stack(
        [
            0.25 * np.sin(2.0 * np.pi * 220.0 * t),
            0.20 * np.cos(2.0 * np.pi * 330.0 * t),
        ],
        axis=0,
        dtype=np.float32,
    )
    audio_out = np.empty_like(audio_in)

    for tier in dnam.Tier:
        engine.tier = tier
        engine.reset()
        engine.process_into(audio_in, audio_out)
        peak = float(np.max(np.abs(audio_out)))
        print(
            f"Tier {tier.label:<5} (C={tier.channels:>2}, bus={tier.feature_bus_dim:>3}-D) "
            f"| Peak output: {peak:.6f}"
        )

    # 2. 1D planar processing (`process_planar`) and pedalboard-compatible callable (`__call__`)
    l_out, r_out = engine.process_planar(audio_in[0], audio_in[1])
    wet = engine(audio_in, sample_rate=96_000, buffer_size=64, reset=True)
    assert l_out.shape == (128,)
    assert r_out.shape == (128,)
    assert wet.shape == (2, 128)

    # 3. 4x linear-phase polyphase Offline Bounce (12-sample PDC lead-in stripped)
    engine.sample_rate = 48_000
    bounced = engine.bounce(audio_in, maximize_tier=True)
    print(f"4x Offline Bounce shape: {bounced.shape} | Peak: {float(np.max(np.abs(bounced))):.6f}")


if __name__ == "__main__":
    main()
