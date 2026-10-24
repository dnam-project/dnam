"""Unit tests for `dnam_trainer.project.runner` (`inspect_dnam_container`,
`render_with_dnam_engine`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnam_engine import DnamContainer
from dnam_trainer.capture.excitation import read_wav, write_wav_24bit
from dnam_trainer.project.runner import inspect_dnam_container, render_with_dnam_engine
from dnam_trainer.schema import Tier


def test_inspect_and_render_with_dnam_engine(tmp_path: Path) -> None:
    ref_dnam = DnamContainer.reference(core2_active=True, is_checkpoint=False)
    dnam_path = ref_dnam.save(tmp_path / "ref.dnam")

    summary = inspect_dnam_container(dnam_path)
    assert summary.magic == "DNAM"
    assert not summary.is_checkpoint
    assert summary.core2_active
    assert summary.sample_rate_train_hz == 48_000

    dry = np.sin(np.linspace(0.0, 20.0, 2048, dtype=np.float32)) * 0.25
    in_wav = write_wav_24bit(tmp_path / "in.wav", dry, sample_rate_hz=48_000)
    out_wav = tmp_path / "out.wav"

    rendered_path = render_with_dnam_engine(
        dnam_path,
        in_wav,
        out_wav,
        tier="full",
        preset=ref_dnam.header.presets[0].id if ref_dnam.header.presets else None,
        controls={ref_dnam.header.controls_schema[0].id: 5.0}
        if ref_dnam.header.controls_schema
        else None,
    )
    assert rendered_path.is_file()
    _ = render_with_dnam_engine(dnam_path, in_wav, out_wav, tier=Tier.ULTRA)
    wet_pcm, sr = read_wav(rendered_path)
    assert sr == 48_000
    assert wet_pcm.shape == (2, 2048)
    assert np.all(np.isfinite(wet_pcm))
