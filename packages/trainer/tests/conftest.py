"""Shared pytest fixtures for `dnam_trainer` unit, CLI, and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dnam_trainer.capture.excitation import generate_dry_excitation
from dnam_trainer.project.workspace import (
    DnamProject,
    parse_compact_range_spec,
    parse_compact_switch_spec,
)
from dnam_trainer.schema import PresetDef, TrainingStem


@pytest.fixture
def short_dry_signal() -> NDArray[np.float32]:
    """Generates a deterministic 1.5-second 48 kHz dry excitation signal (`72,000` samples)."""
    return generate_dry_excitation(duration_sec=1.5, sample_rate_hz=48_000, seed=42)


@pytest.fixture
def synthetic_training_stems(short_dry_signal: NDArray[np.float32]) -> list[TrainingStem]:
    """Builds 3 deterministic stereo training stems spanning `drive` (`2.0, 5.0, 8.0`) and
    `attack`."""
    dry = short_dry_signal[:8192]
    stems: list[TrainingStem] = []
    settings: list[tuple[str, float, str]] = [
        ("stem_00_ref", 5.0, "flat"),
        ("stem_01_low", 2.0, "cut"),
        ("stem_02_high", 8.0, "boost"),
    ]
    for s_id, drive_v, atk_v in settings:
        gain = 1.0 + 0.15 * drive_v * (1.2 if atk_v == "boost" else 0.9)
        u = np.clip((2.0 / 3.0) * (dry.astype(np.float64) * gain), -1.0, 1.0)
        wet_l = (0.75 * dry + 0.25 * (1.5 * u - 0.5 * (u**3))).astype(np.float32)
        wet_r = np.roll(wet_l, 2)
        wet_r[:2] = 0.0
        target = np.stack([wet_l, (0.95 * wet_r + 0.05 * wet_l).astype(np.float32)], axis=0)
        stems.append(
            TrainingStem(
                stem_id=s_id,
                controls_meta={"drive": drive_v, "attack": atk_v},
                input_audio=dry,
                target_audio=target,
                sample_rate_hz=48_000,
                noise_floor_dbfs=-92.0,
            )
        )
    return stems


@pytest.fixture
def initialized_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DnamProject:
    """Creates an isolated `DnamProject` in `tmp_path` with a range and switch control."""
    monkeypatch.setenv("DNAM_CONFIG_DIR", str(tmp_path / "user_config"))
    proj_dir = tmp_path / "studio_proj"
    proj = DnamProject.init(
        proj_dir,
        project_name="Test Pedal Pack",
        author="Pytest Creator",
        sample_rate_hz=48_000,
        dry_duration_sec=1.5,
        initial_model_id="b7k_drive",
    )
    proj.add_control("b7k_drive", parse_compact_range_spec("drive=0..10:5"))
    proj.add_control("b7k_drive", parse_compact_switch_spec("attack=cut,flat,boost:flat"))
    proj.add_preset(
        "b7k_drive",
        PresetDef(
            id="default_noon",
            name="Default Noon",
            controls={"drive": 5.0, "attack": "flat"},
        ),
    )
    return proj
