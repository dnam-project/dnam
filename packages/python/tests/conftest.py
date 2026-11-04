"""Shared pytest fixtures for `dnam` test suites."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import dnam


@pytest.fixture(scope="session")
def reference_model() -> dnam.Model:
    """Session-scoped canonical reference `.dnam` `Model` (`supports_freeze=True`)."""
    return dnam.Model.reference(supports_freeze=True)


@pytest.fixture(scope="session")
def reference_model_no_freeze() -> dnam.Model:
    """Session-scoped canonical reference `.dnam` `Model` with `supports_freeze=False`."""
    return dnam.Model.reference(supports_freeze=False)


@pytest.fixture(scope="session")
def reference_checkpoint() -> dnam.Checkpoint:
    """Session-scoped canonical reference `.dnam-ckpt` `Checkpoint`."""
    return dnam.Checkpoint.reference(include_calib_stats=True)


@pytest.fixture
def engine_full(reference_model: dnam.Model) -> dnam.Engine:
    """Fresh `Engine` instance at `Tier.FULL` and `48,000 Hz`."""
    return dnam.Engine(reference_model, sample_rate=48_000, tier="full")


@pytest.fixture(scope="session")
def test_stereo_audio() -> NDArray[np.float32]:
    """Deterministic `(2, 256)` stereo test signal."""
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    left = 0.25 * np.sin(2.0 * np.pi * 5.0 * t)
    right = 0.20 * np.cos(2.0 * np.pi * 7.0 * t)
    return np.ascontiguousarray(np.stack([left, right], axis=0), dtype=np.float32)
