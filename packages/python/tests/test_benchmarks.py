"""`pytest-benchmark` performance benchmarks across `dnam_engine` subsystems:
1. `DnamEngine` real-time blocks (`64`, `256`, `1024`) & 1-second offline stem rendering (`48000`)
2. `ControlRack`, `ControlSpace`, `LobattoBasis`, `Rank4Fm`, `chebyshev_lfo`, `trust_region`
3. Reference DSP (`PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `unit_rotor`, `lagrange_weights`)
4. `DnamContainer`, `GramMatrix` (`uplo='U'`), `crc32c`, and `provenance` crypto/watermark ops
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
import pytest

from dnam_engine import (
    BackendKind,
    DnamContainer,
    DnamEngine,
    GramMatrix,
    Tier,
    control,
    crc32c,
    dsp,
    provenance,
)
from dnam_engine.schema import (
    LobattoParams,
    MufdnParams,
    PolyAdaaParams,
    PresetDef,
    SelectiveRgLruParams,
    WatermarkParams,
)

_R = TypeVar("_R")


class BenchmarkCallable(Protocol):
    """Protocol for `pytest-benchmark`'s `benchmark` fixture."""

    def __call__(
        self,
        func: Callable[..., _R],
        *args: object,
        **kwargs: object,
    ) -> _R: ...


# ---------------------------------------------------------------------------
# Group 1: DnamEngine Real-Time & Offline Audio Processing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", [Tier.LITE, Tier.FULL, Tier.ULTRA], ids=["lite", "full", "ultra"])
@pytest.mark.parametrize("block_size", [64, 256, 1024], ids=["n64", "n256", "n1024"])
def test_bench_engine_stereo_zero_alloc(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
    tier: Tier,
    block_size: int,
) -> None:
    """Benchmarks zero-allocation stereo `(2, N)` processing with pre-allocated `out=` buffer."""
    eng = DnamEngine(ref_container_core2, tier=tier, backend=BackendKind.AUTO)
    audio_in = np.linspace(-0.5, 0.5, 2 * block_size, dtype=np.float32).reshape((2, block_size))
    audio_out = np.empty_like(audio_in)

    res = benchmark(eng.process, audio_in, out=audio_out)
    assert res.shape == (2, block_size)


@pytest.mark.parametrize("backend", [BackendKind.AUTO, BackendKind.SCALAR], ids=["simd", "scalar"])
def test_bench_engine_planar_vs_backend_1s_stem(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
    backend: BackendKind,
) -> None:
    """Benchmarks 1.0-second (`48,000` samples @ 48 kHz) stereo stem rendering."""
    eng = DnamEngine(ref_container_core2, tier=Tier.FULL, backend=backend)
    left_in = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
    right_in = np.linspace(0.5, -0.5, 48_000, dtype=np.float32)
    left_out = np.empty_like(left_in)
    right_out = np.empty_like(right_in)

    l_res, _ = benchmark(
        eng.process_planar,
        left_in,
        right_in,
        left_out=left_out,
        right_out=right_out,
    )
    assert l_res.shape == (48_000,)


def test_bench_engine_mono_to_stereo_and_downmix(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks 1D mono `(256,)` -> true-stereo `(2, 256)` processing."""
    eng = DnamEngine(ref_container_core2, tier=Tier.FULL)
    mono_in = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    stereo_out = np.empty((2, 256), dtype=np.float32)

    res = benchmark(eng.process, mono_in, out=stereo_out)
    assert res.shape == (2, 256)


def test_bench_engine_pedalboard_call_with_sample_rate(
    benchmark: BenchmarkCallable,
    ref_container_core1: DnamContainer,
) -> None:
    """Benchmarks `pedalboard`-style `engine(audio, sample_rate=48000.0)` invocation."""
    eng = DnamEngine(ref_container_core1, tier=Tier.FULL, sample_rate=48_000.0)
    stereo_in = np.zeros((2, 256), dtype=np.float32)
    stereo_out = np.empty_like(stereo_in)

    res = benchmark(eng, stereo_in, 48_000.0, out=stereo_out)
    assert res.shape == (2, 256)


# ---------------------------------------------------------------------------
# Group 2: ControlRack, ControlSpace & Control Bus Kernels
# ---------------------------------------------------------------------------


def test_bench_control_rack_set_slew(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks `engine.controls['gain'] = 7.5` (3 kHz anti-zipper slew update)."""
    eng = DnamEngine(ref_container_core2)
    rack = eng.controls

    def _update() -> None:
        rack["gain"] = 7.5

    benchmark(_update)
    assert rack["gain"] == 7.5


def test_bench_control_rack_get_and_clamped(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks reading `engine.controls['gain']` and `engine.controls.clamped`."""
    eng = DnamEngine(ref_container_core2)
    rack = eng.controls
    rack.snap(gain=7.5, boost="on")

    def _read() -> float | str:
        _ = rack.clamped
        return rack["gain"]

    val = benchmark(_read)
    assert val == 7.5


def test_bench_control_space_clamp(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks 3-Phase control clamping (`ControlSpace.clamp`)."""
    space = ref_container_core2.control_space
    raw = {"gain": 9.2, "boost": "on"}

    res = benchmark(space.clamp, raw)
    assert res.any_clamped


def test_bench_lobatto_basis_vectorized(benchmark: BenchmarkCallable) -> None:
    """Benchmarks vectorized `LobattoBasis` evaluation over `N = 1024` points."""
    basis = control.LobattoBasis(LobattoParams(anchors=(0.0, -1.0, 1.0), degree=3))
    u = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)

    res = benchmark(basis, u)
    assert res.shape == (1024, 3)


def test_bench_rank4_fm(benchmark: BenchmarkCallable) -> None:
    """Benchmarks `Rank4Fm` evaluation for `D = 16` control features."""
    rng = np.random.default_rng(42)
    v_factors = rng.standard_normal((16, 4))
    features = rng.standard_normal(16)
    fm = control.Rank4Fm(v_factors)

    res = benchmark(fm, features)
    assert res.shape == (4,)


def test_bench_chebyshev_lfo_vectorized(benchmark: BenchmarkCallable) -> None:
    """Benchmarks vectorized `chebyshev_lfo` over `N = 1024` phase samples."""
    cos_phi = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)

    res = benchmark(control.chebyshev_lfo, cos_phi)
    assert res.shape == (1024, 4)


def test_bench_trust_region_softclip_vectorized(benchmark: BenchmarkCallable) -> None:
    """Benchmarks vectorized `trust_region_softclip` over `N = 1024` samples."""
    z = np.linspace(-5.0, 5.0, 1024, dtype=np.float64)

    res = benchmark(control.trust_region_softclip, z, mh=2.5)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1024,)


# ---------------------------------------------------------------------------
# Group 3: Reference DSP Kernels (PolyAdaa, SelectiveRgLru, ModulatedUfdn)
# ---------------------------------------------------------------------------


def test_bench_dsp_poly_adaa_scan(benchmark: BenchmarkCallable) -> None:
    """Benchmarks `dsp.PolyAdaa` scan over `N = 4096` samples."""
    adaa = dsp.PolyAdaa(PolyAdaaParams(b_conv=0.1, alpha=1.2, beta=0.65, delta=0.25))
    x = np.linspace(-0.8, 0.8, 4096, dtype=np.float64)
    out = np.empty_like(x)

    res = benchmark(adaa.process, x, out=out)
    assert res.shape == (4096,)


def test_bench_dsp_selective_rglru_scan(benchmark: BenchmarkCallable) -> None:
    """Benchmarks `dsp.SelectiveRgLru` scan over `N = 4096` samples."""
    rglru = dsp.SelectiveRgLru(SelectiveRgLruParams.from_polar(r_low=0.92, dr_span=0.04))
    x = np.linspace(-0.8, 0.8, 4096, dtype=np.float64)
    out = np.empty_like(x)

    res = benchmark(rglru.process, x, out=out)
    assert res.shape == (4096,)


def test_bench_dsp_modulated_ufdn_scan(benchmark: BenchmarkCallable) -> None:
    """Benchmarks `dsp.ModulatedUfdn` coercive-saturating delay scan over `N = 4096` samples."""
    ufdn = dsp.ModulatedUfdn(MufdnParams(delay_samples=64, g_lin=0.88))
    x = np.linspace(-0.8, 0.8, 4096, dtype=np.float64)
    out = np.empty_like(x)

    res = benchmark(ufdn.process, x, out=out)
    assert res.shape == (4096,)


def test_bench_dsp_unit_rotor_vectorized(benchmark: BenchmarkCallable) -> None:
    """Benchmarks vectorized `dsp.unit_rotor` over `N = 1024` angles."""
    phi = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)

    res = benchmark(dsp.unit_rotor, phi)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 1024)


# ---------------------------------------------------------------------------
# Group 4: DnamContainer, GramMatrix & Provenance
# ---------------------------------------------------------------------------


def test_bench_container_from_bytes_and_in_place_header_update(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks 4 KB in-place header update (`DnamContainer.with_header`)."""
    new_preset = PresetDef(id="lead", name="Lead", controls={"gain": 7.0, "boost": "on"})
    updated_hdr = ref_container_core2.header.model_copy(
        update={"presets": [*ref_container_core2.header.presets, new_preset]}
    )

    res = benchmark(ref_container_core2.with_header, updated_hdr)
    assert "lead" in res.presets


def test_bench_engine_hot_reload_and_telemetry(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks zero-allocation `DnamEngine.reload` and `DnamEngine.telemetry` readout."""
    eng = DnamEngine(ref_container_core2)

    def _reload_and_telemetry() -> int:
        eng.reload(ref_container_core2)
        return eng.telemetry.sample_rate_hz

    sr = benchmark(_reload_and_telemetry)
    assert sr == 48_000


def test_bench_gram_matrix_pack_unpack_ultra(benchmark: BenchmarkCallable) -> None:
    """Benchmarks `GramMatrix.pack` and `GramMatrix.unpack` for `Ultra` (`192 x 192`)."""
    mat = np.eye(192, dtype=np.float64)

    def _roundtrip() -> np.ndarray:
        packed = GramMatrix.pack(mat)
        return GramMatrix.unpack(packed, dim=192)

    res = benchmark(_roundtrip)
    assert res.shape == (192, 192)


def test_bench_provenance_watermark_and_structural_hash(
    benchmark: BenchmarkCallable,
    ref_container_core2: DnamContainer,
) -> None:
    """Benchmarks Dual-Mechanism Buyer Watermarking (`Test 8H`) + 160-D structural hash."""
    arena_bytes = bytes(ref_container_core2.arena)
    code = provenance.generate_tardos_code(b"\x11" * 32, model_id="m1", license_id="l1")
    wm = WatermarkParams(
        kappa_48=tuple(1.0 + 0.005 * ((i % 3) - 1) for i in range(48)),
        tardos_code_4608=tuple(float(v) for v in code),
    )

    def _watermark_and_score() -> float:
        watermarked = provenance.apply_dual_watermark(arena_bytes, wm)
        _ = provenance.extract_structural_hash(watermarked)
        _ = crc32c(watermarked)
        return provenance.score_tardos_watermark(arena_bytes, watermarked, code_4608=code)

    score = benchmark(_watermark_and_score)
    assert score > 50.0
