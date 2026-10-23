"""Unit and property-based tests for `DnamEngine`: `pedalboard`-compatible calling conventions,
1D `(N,)` / 2D `(1, N)` / 2D `(2, N)` processing, `mono_downmix`, `process_planar`,
SIMD vs. `Scalar` parity, hot-reloading, SLIMMABLE tiers, and ZOH rescaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.typing import NDArray

from dnam_engine import (
    BackendKind,
    DnamContainer,
    DnamEngine,
    DnamEngineSettings,
    DnamSampleRateError,
    DnamValidationError,
    Tier,
)
from strategies import st_audio_1d_f32, st_audio_2d_stereo_f32


@given(stereo_in=st_audio_2d_stereo_f32(min_len=32, max_len=128))
def test_engine_stereo_and_simd_vs_scalar_parity(
    ref_container_core2: DnamContainer,
    stereo_in: NDArray[np.float32],
) -> None:
    eng_auto = DnamEngine(ref_container_core2, tier=Tier.FULL, backend="auto")
    eng_scalar = DnamEngine(ref_container_core2, tier=Tier.FULL, backend=BackendKind.SCALAR)
    assert eng_scalar.backend == BackendKind.SCALAR

    y_auto = eng_auto(stereo_in, reset=True)
    y_scalar = eng_scalar(stereo_in, reset=True)
    assert y_auto.shape == stereo_in.shape
    assert np.all(np.isfinite(y_auto))
    np.testing.assert_allclose(y_auto, y_scalar, atol=1e-5)

    # Chunked buffer_size execution parity
    y_chunked = eng_auto(stereo_in, buffer_size=16, reset=True)
    np.testing.assert_allclose(y_chunked, y_auto, atol=1e-6)

    # Pre-allocated out= buffer
    out_buf = np.empty_like(stereo_in)
    returned = eng_auto.process(stereo_in, reset=True, out=out_buf)
    assert returned is out_buf
    np.testing.assert_allclose(out_buf, y_auto, atol=1e-6)


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(mono_in=st_audio_1d_f32(min_len=32, max_len=128))
def test_engine_mono_to_stereo_and_mono_downmix(
    engine_core2: DnamEngine,
    mono_in: NDArray[np.float32],
) -> None:
    n = mono_in.shape[0]
    # 1. Default 1D (N,) -> true-stereo (2, N)
    y_stereo = engine_core2(mono_in, reset=True)
    assert y_stereo.shape == (2, n)

    out_2d = np.empty((2, n), dtype=np.float32)
    engine_core2(mono_in, reset=True, out=out_2d)
    np.testing.assert_allclose(out_2d, y_stereo, atol=1e-6)

    # 2. 1D (N,) with mono_downmix=True -> (N,)
    y_mono_1d = engine_core2(mono_in, reset=True, mono_downmix=True)
    assert y_mono_1d.shape == (n,)
    out_1d = np.empty((n,), dtype=np.float32)
    engine_core2(mono_in, reset=True, mono_downmix=True, out=out_1d)
    np.testing.assert_allclose(out_1d, y_mono_1d, atol=1e-6)

    # 3. pedalboard-style 2D (1, N) mono -> true-stereo (2, N) by default
    mono_2d = mono_in.reshape((1, n))
    y_from_2d = engine_core2(mono_2d, reset=True)
    np.testing.assert_allclose(y_from_2d, y_stereo, atol=1e-6)

    out_2d_b = np.empty((2, n), dtype=np.float32)
    engine_core2(mono_2d, reset=True, out=out_2d_b)
    np.testing.assert_allclose(out_2d_b, y_stereo, atol=1e-6)

    # 4. 2D (1, N) with mono_downmix=True -> (1, N)
    y_2d_down = engine_core2(mono_2d, reset=True, mono_downmix=True)
    assert y_2d_down.shape == (1, n)
    out_2d_mono = np.empty((1, n), dtype=np.float32)
    engine_core2(mono_2d, reset=True, mono_downmix=True, out=out_2d_mono)
    np.testing.assert_allclose(out_2d_mono, y_2d_down, atol=1e-6)


def test_engine_constructors_properties_planar_and_reload(
    tmp_path: Path,
    ref_container_core1: DnamContainer,
    ref_container_core2: DnamContainer,
) -> None:
    path = ref_container_core1.save(tmp_path / "ref1.dnam")
    settings = DnamEngineSettings(tier=Tier.LITE, sample_rate_hz=44_100.0)

    with DnamEngine.from_file(path, settings) as eng:
        assert eng.tier == Tier.LITE
        assert eng.sample_rate == 44_100.0
        assert eng.container.header == ref_container_core1.header

        eng.tier = Tier.ULTRA
        assert eng.tier == Tier.ULTRA
        eng.sample_rate = 96_000.0
        assert eng.sample_rate == 96_000.0

        # pedalboard sample_rate positional argument auto-updates engine.sample_rate
        x_f64 = np.linspace(-0.5, 0.5, 64, dtype=np.float64)
        y = eng(x_f64, 48_000.0, reset=True)
        assert eng.sample_rate == 48_000.0
        assert y.shape == (2, 64)

        # Planar processing with and without pre-allocated buffers
        l_in = np.ones(32, dtype=np.float32) * 0.25
        r_in = np.ones(32, dtype=np.float32) * -0.25
        l_out, r_out = eng.process_planar(l_in, r_in, reset=True)
        l_pre = np.empty_like(l_in)
        r_pre = np.empty_like(r_in)
        eng.process_planar(
            l_in.astype(np.float64),
            r_in.astype(np.float64),
            left_out=l_pre,
            right_out=r_pre,
            reset=True,
        )
        np.testing.assert_allclose(l_pre, l_out, atol=1e-6)
        np.testing.assert_allclose(r_pre, r_out, atol=1e-6)

        # Telemetry snapshot
        tel = eng.telemetry
        assert tel.active_tier == Tier.ULTRA
        assert tel.sample_rate_hz == 48_000

        # Hot-reload in-place
        eng.reload(ref_container_core2, tier=Tier.FULL, sample_rate=48_000.0)
        assert eng.telemetry.core2_active
        eng.reload(ref_container_core1.to_bytes())
        assert not eng.telemetry.core2_active

    eng_bytes = DnamEngine.from_bytes(ref_container_core1.to_bytes())
    assert eng_bytes.tier == Tier.FULL
    eng_ref = DnamEngine.reference(core2_active=True)
    assert eng_ref.telemetry.core2_active


def test_engine_validation_errors(engine_core2: DnamEngine) -> None:
    with pytest.raises(DnamValidationError):
        DnamEngine(engine_core2.container, cast(Any, "not_settings"))
    with pytest.raises(DnamValidationError):
        DnamEngine(engine_core2.container, tier=99)
    with pytest.raises(DnamValidationError):
        DnamEngine(engine_core2.container, backend=cast(Any, "invalid_backend"))
    with pytest.raises(DnamSampleRateError):
        engine_core2.sample_rate = 12_000.0
    with pytest.raises(DnamValidationError):
        engine_core2(np.zeros((3, 16), dtype=np.float32))
    with pytest.raises(DnamValidationError):
        engine_core2(np.zeros((2, 16), dtype=np.float32), mono_downmix=True)
    with pytest.raises(DnamValidationError):
        engine_core2.process_planar(np.zeros(16, dtype=np.float32), np.zeros(8, dtype=np.float32))
    with pytest.raises(DnamValidationError):
        engine_core2(
            np.zeros((2, 16), dtype=np.float32),
            out=np.zeros((2, 8), dtype=np.float32),
        )
    with pytest.raises(DnamValidationError):
        engine_core2(
            np.zeros((2, 16), dtype=np.float32),
            out=cast(Any, np.zeros((2, 16), dtype=np.float64)),
        )
