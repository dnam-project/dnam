"""Free-threading concurrency, shared-`Engine` contention, atomic `Engine.reload`, and cyclic GC
stress tests (`verify[sdk.python.engine.free-threading]`).
"""

from __future__ import annotations

import gc
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import dnam
from dnam.schema import PresetDef, Tier, WatermarkParams


def test_shared_engine_concurrent_threads_no_borrow_mut_error(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Verifies that concurrent threads sharing a single `Engine` instance (executing audio blocks,
    mutating controls, applying presets, syncing tempo, reading telemetry, and atomically reloading
    models) never raise `PyBorrowMutError` or observe torn `_EngineState`.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)
    extra_preset = PresetDef(
        id="crunch_lead",
        name="Crunch Lead",
        controls={"gain": 8.0, "boost": "on"},
    )
    alt_header = reference_model.header.model_copy(
        update={"presets": [*reference_model.header.presets, extra_preset]}
    )
    alt_model = reference_model.with_header(alt_header)

    barrier = threading.Barrier(6)
    iters = 80

    def _audio_worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        audio = rng.uniform(-0.25, 0.25, size=(2, 128)).astype(np.float32)
        out = np.empty_like(audio)
        l_in = audio[0].copy()
        r_in = audio[1].copy()
        l_out = np.empty_like(l_in)
        r_out = np.empty_like(r_in)
        barrier.wait()
        for i in range(iters):
            if i % 2 == 0:
                eng.process_into(audio, out)
                assert np.all(np.isfinite(out))
            else:
                eng.process_planar(l_in, r_in, out_left=l_out, out_right=r_out)
                assert np.all(np.isfinite(l_out))
                assert np.all(np.isfinite(r_out))

    def _controls_worker() -> None:
        barrier.wait()
        for i in range(iters):
            eng.controls["gain"] = 3.0 + float(i % 6)
            eng.controls.update({"gain": 4.5, "boost": "on" if i % 2 == 0 else "off"})
            eng.controls.snap(gain=5.5, delay_ms=200.0 + float(i % 50))
            clamped = eng.controls.clamped
            assert "gain" in clamped.controls
            val = eng.controls["gain"]
            assert isinstance(val, float)
            if i % 10 == 0:
                del eng.controls["gain"]
            if i % 25 == 0:
                eng.controls.reset()

    def _presets_and_tempo_worker() -> None:
        barrier.wait()
        for i in range(iters):
            assert "warm_slapback" in eng.presets
            eng.presets.apply("warm_slapback")
            _ = list(eng.presets.keys())
            matched = eng.tempo.sync(110.0 + float(i % 30), subdivision="1/8", fold_octave=True)
            assert matched >= 1
            if i % 8 == 0:
                bpm, tap_matched = eng.tempo.tap(subdivision="1/4")
                assert bpm > 0.0
                assert tap_matched >= 0

    def _telemetry_and_reload_worker() -> None:
        barrier.wait()
        for i in range(iters):
            tel = eng.telemetry
            assert tel.sample_rate_hz in (48_000, 96_000)
            assert len(tel.clamped_controls) == len(eng.controls)
            if i % 6 == 0:
                target_model = alt_model if (i // 6) % 2 == 0 else reference_model
                eng.reload(target_model, tier=Tier.FULL if i % 12 == 0 else Tier.LITE)
                assert len(eng.model.header.presets) in (1, 2)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(_audio_worker, 101),
            pool.submit(_audio_worker, 202),
            pool.submit(_audio_worker, 303),
            pool.submit(_controls_worker),
            pool.submit(_presets_and_tempo_worker),
            pool.submit(_telemetry_and_reload_worker),
        ]
        for fut in futures:
            fut.result()


def test_gc_collect_stress_during_detached_native_operations(
    reference_model: dnam.Model,
    reference_checkpoint: dnam.Checkpoint,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Runs `gc.collect()` in a tight background loop while worker threads concurrently execute
    `py.detach`-wrapped Rust operations across `Engine`, `Model`, `Checkpoint`, `GramMatrix`,
    `ControlSpace`, `dnam.provenance`, and `dnam.dsp`.
    """
    stop_gc = threading.Event()
    raw_model_bytes = reference_model.to_bytes()
    arena_bytes = reference_model.arena_bytes
    wm_params = WatermarkParams()
    dense_mat = np.eye(32, dtype=np.float64)

    def _gc_loop() -> None:
        while not stop_gc.is_set():
            cycle_a: list[object] = []
            cycle_b: list[object] = [cycle_a]
            cycle_a.append(cycle_b)
            gc.collect(0)
            gc.collect()

    def _worker(worker_id: int) -> None:
        eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)
        audio = np.full((2, 64), 0.1, dtype=np.float32)
        out = np.empty_like(audio)
        u_grid = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
        sig = np.linspace(-0.5, 0.5, 256, dtype=np.float64)
        sig_out = np.empty_like(sig)

        for i in range(25):
            eng.process_into(audio, out)
            assert np.all(np.isfinite(out))

            loaded = dnam.Model.from_bytes(raw_model_bytes)
            slimmed = loaded.with_tier_limit(Tier.LITE)
            assert slimmed.preamble.highest_tier == Tier.LITE

            if (i + worker_id) % 5 == 0:
                exported = reference_checkpoint.export_model()
                assert exported.preamble.magic == "DNAM"

            gram = dnam.GramMatrix.from_dense(dense_mat)
            unpacked = gram.to_dense()
            assert unpacked.shape == (32, 32)

            clamped = loaded.control_space.clamp({"gain": 9.9, "boost": "on"})
            assert clamped.any_clamped

            ver = dnam.provenance.verify_model(loaded)
            assert ver.status == "unsigned"
            gauged = dnam.provenance.apply_gauge_watermark(arena_bytes, wm_params)
            shash = dnam.provenance.extract_structural_hash(gauged)
            assert shash.shape == (224,)
            assert dnam.crc32c(gauged) > 0
            assert len(dnam.provenance.immutable_header_sha256(loaded.header)) == 64

            _ = dnam.dsp.lobatto_basis(u_grid)
            _ = dnam.dsp.chebyshev_lfo(u_grid)
            _ = dnam.dsp.unit_rotor(u_grid)
            y_adaa, _, _, _ = dnam.dsp.poly_adaa_slice(sig, out=sig_out)
            assert np.all(np.isfinite(y_adaa))

    gc_thread = threading.Thread(target=_gc_loop, daemon=True)
    gc_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker, w) for w in range(4)]
            for fut in futures:
                fut.result()
    finally:
        stop_gc.set()
        gc_thread.join(timeout=5.0)


def test_parallel_independent_engines_free_threaded_runtime(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Verifies that free-threaded builds run with the GIL disabled (`sys._is_gil_enabled() is False`)
    and that N independent `Engine` instances render bit-identical outputs in parallel.
    """
    if sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
        assert callable(is_gil_enabled)
        assert is_gil_enabled() is False

    num_threads = 4
    audio = np.linspace(-0.4, 0.4, 2 * 1024, dtype=np.float32).reshape(2, 1024)
    expected = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL).process(audio)

    barrier = threading.Barrier(num_threads)

    def _render_worker() -> np.typing.NDArray[np.float32]:
        eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)
        out = np.empty_like(audio)
        barrier.wait()
        for _ in range(16):
            eng.reset()
            eng.process_into(audio, out)
        return out

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = [f.result() for f in [pool.submit(_render_worker) for _ in range(num_threads)]]

    for res in results:
        np.testing.assert_array_equal(res.view(np.uint32), expected.view(np.uint32))


def test_partial_and_cross_channel_overlap_and_mono_in_place(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Verifies that exact in-place buffers (`2->2`, `1->1`, and planar stereo) route to in-place
    kernels while partial or cross-channel memory overlaps raise `ValidationError` cleanly.
    """
    import pytest

    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)

    # Partial 2D overlap must raise ValidationError instead of corrupting or ignoring input
    slab = np.linspace(-0.3, 0.3, 256, dtype=np.float32)
    a_2d = slab[:128].reshape(2, 64)
    b_2d = slab[64:192].reshape(2, 64)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process_into(a_2d, b_2d)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process(a_2d, out=b_2d)

    # 1->2 overlap between 1D mono input and 2D stereo output must raise ValidationError
    stereo_buf = np.zeros((2, 64), dtype=np.float32)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process(stereo_buf[0], out=stereo_buf)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process_planar(stereo_buf[0], out_left=stereo_buf[0], out_right=stereo_buf[1])

    # Cross-channel or single-channel planar overlaps must raise ValidationError
    l_buf = np.ones(64, dtype=np.float32) * 0.1
    r_buf = np.ones(64, dtype=np.float32) * -0.1
    with pytest.raises(dnam.ValidationError, match="share memory"):
        eng.process_planar(l_buf, r_buf, out_left=l_buf, out_right=l_buf)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process_planar(l_buf, r_buf, out_left=r_buf, out_right=l_buf)
    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng.process_planar(l_buf, l_buf, out_left=l_buf, out_right=r_buf)

    # 1->1 mono output model: exact 1D and (1, N) in-place works, partial overlap raises
    mono_io = reference_model.header.io_config.model_copy(
        update={"inputs": 1, "output_mode": "mono"}
    )
    mono_hdr = reference_model.header.model_copy(update={"io_config": mono_io})
    mono_model = dnam.Model.build(mono_hdr, reference_model.arena_bytes)
    eng_mono = dnam.Engine(mono_model, sample_rate=48_000, tier=Tier.FULL)

    x_1d = np.linspace(-0.25, 0.25, 300, dtype=np.float32)
    expected_1d = eng_mono.process(x_1d)
    eng_mono.reset()
    in_place_1d = x_1d.copy()
    eng_mono.process_into(in_place_1d, in_place_1d, buffer_size=64)
    np.testing.assert_array_equal(in_place_1d.view(np.uint32), expected_1d.view(np.uint32))

    eng_mono.reset()
    in_place_2d = x_1d.copy().reshape(1, 300)
    eng_mono.process(in_place_2d, out=in_place_2d)
    np.testing.assert_array_equal(in_place_2d[0].view(np.uint32), expected_1d.view(np.uint32))

    with pytest.raises(dnam.ValidationError, match="overlap"):
        eng_mono.process(slab[:64], out=slab[32:96])


def test_reload_trails_and_concurrent_atomic_bounce_and_dsp_in_place(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Verifies that `Engine.reload` preserves `trails_enabled=True` when `is_bypassed=False`,
    `Engine.bounce` executes atomically under concurrent access, and `dnam.dsp` slice helpers
    support in-place `out=signal` buffers.
    """
    eng = dnam.Engine(
        reference_model,
        sample_rate=48_000,
        tier=Tier.LITE,
        maximize_tier_on_bounce=True,
    )
    eng.set_bypass(False, trails=True)
    assert not eng.is_bypassed
    assert eng.trails_enabled
    eng.reload(reference_model)
    assert not eng.is_bypassed
    assert eng.trails_enabled

    audio_1d = np.linspace(-0.2, 0.2, 256, dtype=np.float32)
    audio_2d = np.vstack([audio_1d, -audio_1d])
    barrier = threading.Barrier(3)

    def _bounce_worker() -> None:
        barrier.wait()
        for i in range(15):
            rendered = eng.bounce(
                audio_1d if i % 2 == 0 else audio_2d,
                maximize_tier=(i % 2 == 0),
            )
            assert rendered.shape == (2, 256)
            assert np.all(np.isfinite(rendered))

    def _call_worker() -> None:
        barrier.wait()
        for i in range(25):
            sr = 48_000 if i % 2 == 0 else 96_000
            out = eng(audio_2d, sample_rate=sr, buffer_size=64, reset=True)
            assert out.shape == (2, 256)
            assert np.all(np.isfinite(out))

    def _observer_worker() -> None:
        barrier.wait()
        for _ in range(40):
            assert eng.maximize_tier_on_bounce
            assert not eng.offline_bounce
            assert eng.effective_tier == Tier.LITE

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_bounce_worker),
            pool.submit(_call_worker),
            pool.submit(_observer_worker),
        ]
        for fut in futs:
            fut.result()

    # Verify in-place out=signal in dnam.dsp slice functions
    sig = np.linspace(-0.4, 0.4, 64, dtype=np.float64)
    s_tank = sig.copy()
    dnam.dsp.tank_slice(s_tank, out=s_tank)
    np.testing.assert_allclose(s_tank, dnam.dsp.tank_slice(sig))

    s_modal = sig.copy()
    dnam.dsp.modal_slice(s_modal, out=s_modal)
    np.testing.assert_allclose(s_modal, dnam.dsp.modal_slice(sig)[0])

    s_adaa = sig.copy()
    dnam.dsp.poly_adaa_slice(s_adaa, out=s_adaa)
    np.testing.assert_allclose(s_adaa, dnam.dsp.poly_adaa_slice(sig)[0])
