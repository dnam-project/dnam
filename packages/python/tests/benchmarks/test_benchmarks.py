"""Calibrated `pytest-benchmark` performance suite and FFI overhead budget checks
(`tests/benchmarks/test_benchmarks.py`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

import dnam
from dnam.schema import Backend, PresetDef, Tier, WatermarkParams


class BenchmarkFixture(Protocol):
    def __call__[**P, R](self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...


# ---------------------------------------------------------------------------
# Group 1: Engine Real-Time Blocks, Stem Rendering & 1.10x FFI Overhead Budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier",
    [Tier.LITE, Tier.FULL, Tier.PRO, Tier.ULTRA],
    ids=["lite", "full", "pro", "ultra"],
)
@pytest.mark.parametrize("block_size", [64, 128, 256, 1024], ids=["n64", "n128", "n256", "n1024"])
def test_bench_engine_process_into_stereo(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
    tier: Tier,
    block_size: int,
) -> None:
    """verify[sdk.python.engine.zero-copy-and-gil]
    Benchmarks zero-allocation stereo `(2, N)` processing per tier and block size.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=tier)
    audio: NDArray[np.float32] = np.zeros((2, block_size), dtype=np.float32)
    out: NDArray[np.float32] = np.empty((2, block_size), dtype=np.float32)
    res = benchmark(eng.process_into, audio, out)
    assert res.shape == (2, block_size)


@pytest.mark.parametrize("backend", [Backend.AUTO, Backend.SCALAR], ids=["simd", "scalar"])
def test_bench_engine_planar_1s_stem(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
    backend: Backend,
) -> None:
    """verify[sdk.python.engine.zero-copy-and-gil]
    Benchmarks 1.0-second (`48,000` samples @ 48 kHz) stereo planar stem rendering.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL, backend=backend)
    l_in = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
    r_in = np.linspace(0.5, -0.5, 48_000, dtype=np.float32)
    l_out = np.empty_like(l_in)
    r_out = np.empty_like(r_in)
    l_res, _ = benchmark(eng.process_planar, l_in, r_in, out_left=l_out, out_right=r_out)
    assert l_res.shape == (48_000,)


def test_bench_engine_pedalboard_call(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.pedalboard-call]
    Benchmarks `pedalboard`-compatible `engine(audio, 48000, reset=False)` invocation.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)
    audio = np.zeros((2, 256), dtype=np.float32)
    res = benchmark(eng, audio, 48_000, reset=False)
    assert res.shape == (2, 256)


def test_engine_process_into_ffi_overhead_within_1_10x_budget(
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.zero-copy-and-gil]
    Verifies that `Engine.process_into` adds <= 10% (`<= 1.10x`) overhead over direct native
    `PyEngine.process_stereo_2d` for a 128-sample stereo block.
    """
    eng = dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL)
    native = eng._native
    audio = np.zeros((2, 128), dtype=np.float32)
    out = np.empty_like(audio)

    for _ in range(500):
        native.process_stereo_2d(audio, out, None)
        eng.process_into(audio, out)

    iters = 2_000
    native_rounds: list[int] = []
    sdk_rounds: list[int] = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            native.process_stereo_2d(audio, out, None)
        native_rounds.append(time.perf_counter_ns() - t0)

        t1 = time.perf_counter_ns()
        for _ in range(iters):
            eng.process_into(audio, out)
        sdk_rounds.append(time.perf_counter_ns() - t1)

    t_native = min(native_rounds)
    t_sdk = min(sdk_rounds)
    ratio = t_sdk / max(t_native, 1)
    assert ratio <= 1.10, (
        f"SDK process_into overhead ratio {ratio:.4f}x exceeds 1.10x budget "
        f"(sdk={t_sdk / iters:.1f} ns vs native={t_native / iters:.1f} ns)"
    )


def test_bench_parallel_multi_thread_engine_scaling(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.engine.free-threading]
    Benchmarks 4 independent `Engine` instances rendering stereo blocks in parallel across 4 OS
    threads via `py.detach` and verifies `sys._is_gil_enabled() is False` + multi-thread scaling.
    """
    import sys
    import sysconfig
    from concurrent.futures import ThreadPoolExecutor

    if sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
        assert callable(is_gil_enabled)
        assert is_gil_enabled() is False

    num_threads = 4
    blocks_per_task = 16
    engines = [
        dnam.Engine(reference_model, sample_rate=48_000, tier=Tier.FULL) for _ in range(num_threads)
    ]
    inputs = [np.zeros((2, 1024), dtype=np.float32) for _ in range(num_threads)]
    outputs = [np.empty((2, 1024), dtype=np.float32) for _ in range(num_threads)]

    def _render_one(idx: int) -> None:
        eng = engines[idx]
        inp = inputs[idx]
        out = outputs[idx]
        for _ in range(blocks_per_task):
            eng.process_into(inp, out)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for _ in range(3):
            list(pool.map(_render_one, range(num_threads)))

        seq_times: list[int] = []
        par_times: list[int] = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            for idx in range(num_threads):
                _render_one(idx)
            seq_times.append(time.perf_counter_ns() - t0)

            t1 = time.perf_counter_ns()
            list(pool.map(_render_one, range(num_threads)))
            par_times.append(time.perf_counter_ns() - t1)

        def _parallel_batch() -> None:
            list(pool.map(_render_one, range(num_threads)))

        benchmark(_parallel_batch)

    speedup = min(seq_times) / max(min(par_times), 1)
    assert speedup >= 1.25, (
        f"Expected parallel 4-thread speedup >= 1.25x, got {speedup:.2f}x "
        f"(seq={min(seq_times) / 1e6:.2f} ms vs par={min(par_times) / 1e6:.2f} ms)"
    )
    assert outputs[0].shape == (2, 1024)


# ---------------------------------------------------------------------------
# Group 2: Controls, Telemetry & ControlSpace Clamping
# ---------------------------------------------------------------------------


def test_bench_controls_slew_get_and_clamped(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.api.engine-facets]
    Benchmarks live `engine.controls` set, get, and cached `.clamped` readout.
    """
    eng = dnam.Engine(reference_model)
    ctrls = eng.controls

    def _op() -> float | str:
        ctrls["gain"] = 7.5
        _ = ctrls.clamped
        return ctrls["gain"]

    val = benchmark(_op)
    assert val == 7.5


def test_bench_control_space_clamp(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.controls.space-and-tempo]
    Benchmarks pre-compiled ControlSpace 3-phase clamping.
    """
    space = reference_model.control_space
    res = benchmark(space.clamp, {"gain": 7.5, "boost": "on"})
    assert not res.any_clamped


def test_bench_engine_telemetry(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.api.engine-facets]
    Benchmarks lock-free `engine.telemetry` snapshot construction.
    """
    eng = dnam.Engine(reference_model)
    tel = benchmark(lambda: eng.telemetry)
    assert tel.sample_rate_hz == 48_000


# ---------------------------------------------------------------------------
# Group 3: Model, GramMatrix, Provenance & Reference DSP Kernels
# ---------------------------------------------------------------------------


def test_bench_model_from_bytes_and_with_header(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.model.containers-and-checkpoints]
    Benchmarks `.dnam` container validation (`Model.from_bytes`) and 4 KB in-place header update.
    """
    raw = reference_model.to_bytes()
    new_preset = PresetDef(id="lead", name="Lead", controls={"gain": 7.0, "boost": "on"})
    updated_hdr = reference_model.header.replace(
        presets=[*reference_model.header.presets, new_preset]
    )

    def _load_and_edit() -> dnam.Model:
        m = dnam.Model.from_bytes(raw)
        return m.with_header(updated_hdr)

    edited = benchmark(_load_and_edit)
    assert "lead" in edited.presets


def test_bench_gram_matrix_pack_unpack_ultra(
    benchmark: BenchmarkFixture,
) -> None:
    """verify[sdk.python.model.containers-and-checkpoints]
    Benchmarks `GramMatrix.from_dense` and `to_dense` for `Ultra` (`256 x 256`).
    """
    mat = np.eye(256, dtype=np.float64)

    def _roundtrip() -> NDArray[np.float64]:
        g = dnam.GramMatrix.from_dense(mat)
        return g.to_dense()

    dense = benchmark(_roundtrip)
    assert dense.shape == (256, 256)


def test_bench_provenance_watermark_and_hash(
    benchmark: BenchmarkFixture,
    reference_model: dnam.Model,
) -> None:
    """verify[sdk.python.provenance.v2-security]
    Benchmarks Isometric Gauge Watermark + 224-D Structural Hash + CRC32C.
    """
    arena = reference_model.arena_bytes
    wm = WatermarkParams()

    def _wm_op() -> int:
        gauged = dnam.provenance.apply_gauge_watermark(arena, wm)
        _ = dnam.provenance.extract_structural_hash(gauged)
        return dnam.crc32c(gauged)

    crc = benchmark(_wm_op)
    assert crc > 0


def test_bench_dsp_reference_kernels(
    benchmark: BenchmarkFixture,
) -> None:
    """verify[sdk.python.dsp.reference-kernels]
    Benchmarks vectorized TORA reference kernels (`PolyADAA`, `Lobatto`, `Chebyshev`, `Rotor`).
    """
    u = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)
    sig = np.linspace(-0.8, 0.8, 4096, dtype=np.float64)
    out = np.empty_like(sig)

    def _kernels() -> NDArray[np.float64]:
        _ = dnam.dsp.lobatto_basis(u)
        _ = dnam.dsp.chebyshev_lfo(u)
        _ = dnam.dsp.trust_region_softclip(u)
        _ = dnam.dsp.unit_rotor(u)
        res, _, _, _ = dnam.dsp.poly_adaa_slice(sig, out=out)
        return res

    res = benchmark(_kernels)
    assert res.shape == (4096,)
