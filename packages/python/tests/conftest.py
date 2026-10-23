"""Shared pytest fixtures and Hypothesis profiles for `packages/python/tests`."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, Verbosity, settings

from dnam_engine import DnamContainer, DnamEngine, Tier

settings.register_profile(
    "dev",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
)
settings.register_profile(
    "smt",
    max_examples=15,
    backend="crosshair",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.load_profile("dev")


@pytest.fixture(scope="session")
def ref_container_core1() -> DnamContainer:
    """Session-scoped reference `.dnam` container with Core 2 bypassed."""
    return DnamContainer.reference(core2_active=False, is_checkpoint=False)


@pytest.fixture(scope="session")
def ref_container_core2() -> DnamContainer:
    """Session-scoped reference `.dnam` container with Core 2 (`M-UFDN`) active."""
    return DnamContainer.reference(core2_active=True, is_checkpoint=False)


@pytest.fixture(scope="session")
def ref_checkpoint_container() -> DnamContainer:
    """Session-scoped reference `.dnam-ckpt` checkpoint container."""
    return DnamContainer.reference(core2_active=True, is_checkpoint=True)


@pytest.fixture
def engine_core2(ref_container_core2: DnamContainer) -> DnamEngine:
    """Fresh `DnamEngine` instance loaded with `ref_container_core2` (`Tier.FULL`, 48 kHz)."""
    return DnamEngine(ref_container_core2, tier=Tier.FULL, sample_rate=48_000.0)
