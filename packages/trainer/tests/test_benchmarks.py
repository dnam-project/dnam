"""`pytest-benchmark` throughput gates for `dnam_trainer` hot paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dnam_trainer.control.audit import run_phase1_preflight_audit
from dnam_trainer.dsp.selective_rglru import DifferentiableSelectiveRgLru
from dnam_trainer.schema import parse_compact_range_spec, parse_compact_switch_spec
from dnam_trainer.training.kronecker_head import (
    compute_stem_sufficient_statistics,
    solve_all_tiers_kronecker_head_float64,
)


def test_bench_phase1_preflight_audit(benchmark: Any) -> None:
    """Benchmarks Phase-1 Pre-Flight Control-Space Audit (`< 200 ms` target)."""
    ctrls = [
        parse_compact_range_spec("drive=0..10:5"),
        parse_compact_range_spec("tone=0..10:5"),
        parse_compact_switch_spec("voice=clean,crunch,lead:clean"),
    ]
    stems = [
        {"controls": {"drive": 2.0, "tone": 5.0, "voice": "clean"}},
        {"controls": {"drive": 5.0, "tone": 5.0, "voice": "crunch"}},
        {"controls": {"drive": 8.0, "tone": 7.0, "voice": "lead"}},
    ]
    res = benchmark(run_phase1_preflight_audit, ctrls, stems)
    assert len(res.updated_controls) == 3


def test_bench_kronecker_multi_tier_float64_solve(benchmark: Any) -> None:
    """Benchmarks the 3-tier (`Lite`, `Full`, `Ultra`) `float64` Kronecker Cholesky solver (`< 50
    ms`)."""
    rng = np.random.default_rng(123)
    phi = rng.standard_normal((1024, 192)).astype(np.float32) * 0.2
    y = rng.standard_normal((1024, 2)).astype(np.float32) * 0.1
    stat1 = compute_stem_sufficient_statistics(
        phi,
        y,
        stem_id="s1",
        controls_meta={"drive": 2.0},
        input_pcm_sha256="00" * 32,
        target_pcm_sha256="11" * 32,
    )
    stat2 = compute_stem_sufficient_statistics(
        phi * 1.1,
        y * 1.2,
        stem_id="s2",
        controls_meta={"drive": 8.0},
        input_pcm_sha256="00" * 32,
        target_pcm_sha256="22" * 32,
    )
    z_stems = np.array([[-0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]], dtype=np.float64)
    sol = benchmark(solve_all_tiers_kronecker_head_float64, [stat1, stat2], z_stems)
    assert sol.w_ultra_5x2x192.shape == (5, 2, 192)


def test_bench_selective_rglru_chunked_scan(benchmark: Any) -> None:
    """Benchmarks the Two-Level Chunked Associative Scan (`L_chunk = 256`) on 4,096 frames."""
    layer = DifferentiableSelectiveRgLru(num_channels=4, dtype=torch.float32)
    x = torch.randn((1, 4, 4096), dtype=torch.float32)
    with torch.no_grad():
        out, _ = benchmark(layer, x)
    assert out.shape == (1, 4, 4096)
