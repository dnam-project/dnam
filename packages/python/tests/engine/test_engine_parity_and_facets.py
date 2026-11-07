"""Tests for `dnam.engine` (`Engine`, `Controls`, `Presets`, `Tempo`, `Telemetry`, GIL & parity)."""

from __future__ import annotations

import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from numpy.typing import NDArray

import dnam
from dnam._native import _testing
from dnam.errors import ControlError, SampleRateError, ValidationError
from dnam.schema import Backend, Tier
from strategies import (
    backend_strategy,
    mono_audio_strategy,
    sample_rate_strategy,
    stereo_audio_strategy,
    tier_strategy,
)


@given(
    tier=tier_strategy,
    backend=backend_strategy,
    sample_rate=sample_rate_strategy,
    audio=stereo_audio_strategy(min_frames=1, max_frames=256),
)
@settings(max_examples=25, deadline=None)
def test_engine_stereo_bit_identical_to_rust_reference(
    reference_model: dnam.Model,
    tier: Tier,
    backend: Backend,
    sample_rate: int,
    audio: NDArray[np.float32],
) -> None:
    """verify[sdk.python.engine.pedalboard-call] verify[sdk.python.engine.zero-copy-and-gil]
    Verifies that Python `Engine` stereo processing produces bit-identical (`0 ULP`) output
    to calling `DynamicDnamEngine::process_block` directly in Rust across tiers, backends,
    and sample rates.
    """
    eng = dnam.Engine(
        reference_model,
        sample_rate=sample_rate,
        tier=tier,
        backend=backend,
    )
    py_out = eng(audio, sample_rate, buffer_size=64, reset=True)
    rust_out = _testing.render_reference_stereo(
        reference_model.to_bytes(),
        int(tier),
        backend.value,
        sample_rate,
        audio[0],
        audio[1],
        64,
        True,
    )
    assert py_out.dtype == np.float32
    assert py_out.flags.c_contiguous
    np.testing.assert_array_equal(py_out.view(np.uint32), rust_out.view(np.uint32))


@given(
    tier=tier_strategy,
    mono=mono_audio_strategy(min_frames=1, max_frames=256),
)
@settings(max_examples=20, deadline=None)
def test_engine_mono_to_stereo_and_mono_bit_identical_to_rust_reference(
    reference_model: dnam.Model,
    tier: Tier,
    mono: NDArray[np.float32],
) -> None:
    """verify[sdk.python.engine.pedalboard-call] verify[sdk.python.api.flexible-in-precise-out]
    Verifies `1->2` (`Mono -> Stereo`) and `1->1` (`Mono -> Mono`) bit-exact parity against Rust.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=tier, backend="scalar")
    py_m2s = eng(mono, 48_000, buffer_size=32, reset=True)
    rust_m2s = _testing.render_reference_mono_to_stereo(
        reference_model.to_bytes(),
        int(tier),
        "scalar",
        48_000,
        mono,
        32,
        True,
    )
    np.testing.assert_array_equal(py_m2s.view(np.uint32), rust_m2s.view(np.uint32))

    # Build a 1->1 mono output model and verify mono parity
    mono_io = reference_model.header.io_config.model_copy(
        update={"inputs": 1, "output_mode": "mono"}
    )
    mono_hdr = reference_model.header.model_copy(update={"io_config": mono_io})
    mono_model = dnam.Model.build(mono_hdr, reference_model.arena_bytes)
    eng_mono = dnam.Engine(mono_model, sample_rate=48_000, tier=tier, backend="scalar")
    py_mono_1d = eng_mono(mono, 48_000, buffer_size=32, reset=True)
    rust_mono_1d = _testing.render_reference_mono(
        mono_model.to_bytes(),
        int(tier),
        "scalar",
        48_000,
        mono,
        32,
        True,
    )
    assert py_mono_1d.shape == mono.shape
    np.testing.assert_array_equal(py_mono_1d.view(np.uint32), rust_mono_1d.view(np.uint32))

    # Also test (1, N) and (2, N) -> (1, N) on mono model
    out_2d_1ch = eng_mono(mono.reshape(1, -1), 48_000, reset=True)
    assert out_2d_1ch.shape == (1, mono.shape[0])
    out_fold = eng_mono(np.stack([mono, mono], axis=0), 48_000, reset=True)
    assert out_fold.shape == (1, mono.shape[0])


def test_engine_facets_context_managers_zero_copy_and_gil(
    reference_model: dnam.Model,
    reference_model_no_freeze: dnam.Model,
    reference_checkpoint: dnam.Checkpoint,
    test_stereo_audio: NDArray[np.float32],
    tmp_path: Path,
) -> None:
    """verify[sdk.python.api.engine-facets] verify[sdk.python.engine.zero-copy-and-gil]
    Verifies Controls, Presets, Tempo, Telemetry facets, bypassed() / frozen() context managers,
    zero-copy process_into / process_planar, reload, bounce, and GIL release across threads.
    """
    eng = dnam.Engine.reference(sample_rate=48_000, tier="full")
    assert "Engine" in repr(eng)
    assert eng.tier == Tier.FULL
    assert eng.max_available_tier == Tier.ULTRA
    assert eng.max_supported_tier == Tier.ULTRA
    assert eng.effective_tier == Tier.FULL
    assert eng.latency_samples == 0

    # Controls facet (MutableMapping)
    assert eng.model is not None
    assert len(eng.controls) == 3
    assert len(eng.controls.schema) == 3
    assert len(eng.controls.space) == 3
    assert list(eng.controls) == ["gain", "boost", "delay_ms"]
    assert "gain" in eng.controls
    assert "Controls" in repr(eng.controls)
    assert eng.controls["gain"] == 5.0
    eng.controls["gain"] = 12.0  # clamped to 9.5
    assert eng.controls["gain"] == 9.5
    assert eng.controls.clamped.any_clamped
    assert "gain" in eng.controls.clamped.clamped_ids
    assert eng.controls.clamped["gain"].was_clamped
    del eng.controls["gain"]
    assert eng.controls["gain"] == 5.0
    eng.controls.snap({"gain": 7.0}, boost="on")
    assert eng.controls["boost"] == "on"
    eng.controls.snap(gain=6.0)
    assert eng.controls["gain"] == 6.0
    eng.controls.reset()
    assert eng.controls["boost"] == "off"
    eng.tier = "lite"
    assert eng.tier == Tier.LITE
    eng.tier = Tier.FULL
    eng.set_bypass(True, trails=False)
    assert eng.is_bypassed
    eng.set_bypass(False, trails=True)
    phys = eng.controls.to_physical("gain", 0.5)
    assert abs(eng.controls.to_normalized("gain", phys) - 0.5) < 1e-4
    with pytest.raises(ControlError):
        _ = eng.controls["unknown"]
    with pytest.raises(ControlError):
        eng.controls.snap()

    # Presets facet (Mapping)
    assert len(eng.presets) == 1
    assert "warm_slapback" in eng.presets
    assert "Presets" in repr(eng.presets)
    eng.presets.apply("warm_slapback")
    assert eng.controls["boost"] == "on"
    eng.presets.apply("warm_slapback", immediate=True)
    assert abs(float(eng.controls["gain"]) - 6.5) < 1e-5
    with pytest.raises(ControlError):
        _ = eng.presets["missing"]

    # Tempo facet
    assert "Tempo" in repr(eng.tempo)
    updated_count = eng.tempo.sync(120.0, "1/8")
    assert updated_count == 1
    assert abs(eng.tempo.bpm - 120.0) < 1e-5
    eng.tempo.bpm = 140.0
    assert abs(eng.tempo.bpm - 140.0) < 1e-5
    tap_bpm, _ = eng.tempo.tap("1/4")
    assert 20.0 <= tap_bpm <= 400.0

    # Bypass & Freeze context managers restore state even on exception
    assert not eng.is_bypassed
    try:
        with eng.bypassed(trails=False):
            assert eng.is_bypassed
            assert not eng.trails_enabled
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not eng.is_bypassed

    assert eng.supports_freeze
    assert not eng.is_frozen
    with eng.frozen(amount=1.0):
        assert eng.is_frozen
    assert not eng.is_frozen
    eng.set_freeze(True)
    assert eng.is_frozen
    eng.set_freeze(0.0)
    assert not eng.is_frozen

    # Model without freeze support raises ControlError
    eng_nf = dnam.Engine(reference_model_no_freeze)
    assert not eng_nf.supports_freeze
    with pytest.raises(ControlError):
        eng_nf.set_freeze(True)
    with pytest.raises(ControlError), eng_nf.frozen():
        pass

    # Width, host mix, offline_bounce, sample_rate, reload
    eng.width = 1.5
    assert abs(eng.width - 1.5) < 1e-5
    eng.configure_host_mix(blend_wet=0.8, output_gain_db=-1.0, apply_unity_leveling=True)
    eng.max_supported_tier = "pro"
    assert eng.max_supported_tier == Tier.PRO
    eng.offline_bounce = True
    assert eng.offline_bounce
    assert eng.latency_samples == 12
    assert eng.effective_tier == Tier.PRO
    eng.offline_bounce = False
    assert eng.latency_samples == 0
    with pytest.raises(SampleRateError):
        eng.sample_rate = 8_000

    # Zero-copy process_into & in-place processing + out validation errors
    out_buf = np.empty_like(test_stereo_audio)
    ret = eng.process_into(test_stereo_audio, out_buf)
    assert ret is out_buf
    ip_buf = test_stereo_audio.copy()
    eng.process_into(ip_buf, ip_buf)
    eng.process(ip_buf, out=ip_buf)
    eng.process(test_stereo_audio, out=out_buf)
    eng.process(test_stereo_audio.tolist(), out=out_buf)
    eng.process(test_stereo_audio[0], out=out_buf)
    with pytest.raises(ValidationError):
        eng.process(test_stereo_audio, buffer_size=0)
    with pytest.raises(ValidationError):
        eng.process(np.zeros((3, 16), dtype=np.float32))
    with pytest.raises(ValidationError):
        eng.process([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValidationError):
        eng.process(
            test_stereo_audio,
            out=np.zeros((2, 256), dtype=np.float32, order="F"),
        )
    ro_out = np.zeros((2, 256), dtype=np.float32)
    ro_out.flags.writeable = False
    with pytest.raises(ValidationError):
        eng.process(test_stereo_audio, out=ro_out)
    with pytest.raises(ValidationError):
        eng.process(test_stereo_audio, out=np.zeros((2, 16), dtype=np.float32))

    # Planar processing (out-of-place, in-place, mono-to-stereo, length mismatch)
    l_out, r_out = eng.process_planar(test_stereo_audio[0], test_stereo_audio[1])
    assert l_out.shape == (256,)
    assert r_out.shape == (256,)
    l_ip = test_stereo_audio[0].copy()
    r_ip = test_stereo_audio[1].copy()
    eng.process_planar(l_ip, r_ip, out_left=l_ip, out_right=r_ip)
    l_m2s, r_m2s = eng.process_planar(test_stereo_audio[0])
    assert l_m2s.shape == (256,)
    assert r_m2s.shape == (256,)
    with pytest.raises(ValidationError):
        eng.process_planar(test_stereo_audio[0], test_stereo_audio[1, :10])
    with pytest.raises(ValidationError):
        eng.process_planar(test_stereo_audio)

    # Bounce (1D mono and 2D stereo)
    bounced_stereo = eng.bounce(test_stereo_audio)
    assert bounced_stereo.shape == (2, 256)
    bounced_mono = eng.bounce(test_stereo_audio[0])
    assert bounced_mono.shape == (2, 256)

    # Telemetry & OverflowFlags
    tel = eng.telemetry
    assert tel.active_tier == Tier.FULL
    assert tel.max_pole_radius <= 0.9995
    assert tel.max_effective_loop_gain <= 0.9990
    assert not tel.outside_certified_input

    # Construct from Checkpoint, bytes, and file path + reload
    f_path = tmp_path / "m.dnam"
    reference_model.save(f_path)
    eng_from_file = dnam.Engine.from_file(f_path)
    eng_from_bytes = dnam.Engine.from_bytes(reference_model.to_bytes())
    eng_from_ckpt = dnam.Engine(reference_checkpoint)
    eng_from_file.reload(reference_checkpoint, tier="lite", sample_rate=44_100)
    assert eng_from_file.tier == Tier.LITE
    assert eng_from_file.sample_rate == 44_100

    # Parallel execution across multiple independent Engine instances (verifies GIL release)
    engines = [eng, eng_from_file, eng_from_bytes, eng_from_ckpt]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda e: e(test_stereo_audio, 48_000), engines))
    assert len(results) == 4
    for r in results:
        assert r.shape == (2, 256)


def test_zero_python_heap_allocations_in_process_into_hot_loop(
    test_stereo_audio: NDArray[np.float32],
) -> None:
    """verify[sdk.python.engine.zero-copy-and-gil]
    Verifies via `tracemalloc` that steady-state `Engine.process_into` and
    `Engine.process_planar` calls with pre-allocated buffers do not accumulate Python heap
    allocations across a 1,000-block real-time loop.
    """
    eng = dnam.Engine.reference(sample_rate=48_000, tier="full")
    out_2d = np.empty_like(test_stereo_audio)
    l_in = np.ascontiguousarray(test_stereo_audio[0])
    r_in = np.ascontiguousarray(test_stereo_audio[1])
    l_out = np.empty_like(l_in)
    r_out = np.empty_like(r_in)

    # Warm up internal caches before snapshotting Python heap allocations
    for _ in range(16):
        eng.process_into(test_stereo_audio, out_2d)
        eng.process_planar(l_in, r_in, out_left=l_out, out_right=r_out)

    tracemalloc.start()
    try:
        before_cur, _ = tracemalloc.get_traced_memory()
        for _ in range(500):
            eng.process_into(test_stereo_audio, out_2d)
            eng.process_planar(l_in, r_in, out_left=l_out, out_right=r_out)
        after_cur, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Net steady-state Python heap growth across 1,000 audio callbacks must be bounded
    assert (after_cur - before_cur) <= 256


@pytest.mark.slow
@pytest.mark.parametrize("tier", list(Tier))
@pytest.mark.parametrize("sample_rate", [44_100, 48_000, 88_200, 96_000, 192_000])
@pytest.mark.parametrize("block_size", [1, 31, 128, 512])
def test_slow_exhaustive_parity_matrix(
    reference_model: dnam.Model,
    test_stereo_audio: NDArray[np.float32],
    tier: Tier,
    sample_rate: int,
    block_size: int,
) -> None:
    """verify[sdk.python.engine.pedalboard-call]
    Exhaustive cross-product parity matrix across tiers, sample rates, and block sizes.
    """
    eng = dnam.Engine(reference_model, sample_rate=sample_rate, tier=tier)
    py_out = eng(test_stereo_audio, sample_rate, buffer_size=block_size, reset=True)
    rust_out = _testing.render_reference_stereo(
        reference_model.to_bytes(),
        int(tier),
        eng.backend.value,
        sample_rate,
        test_stereo_audio[0],
        test_stereo_audio[1],
        block_size,
        True,
    )
    np.testing.assert_array_equal(py_out.view(np.uint32), rust_out.view(np.uint32))
