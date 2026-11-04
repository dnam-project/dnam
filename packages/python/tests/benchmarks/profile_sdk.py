"""Standalone Scalene CPU/memory/copy profiling driver for the `dnam` Python SDK.

Run via:
  moon run python:profile
  uv run --no-sync scalene run --cli packages/python/tests/benchmarks/profile_sdk.py
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

import dnam
from dnam.schema import WatermarkParams


def run_audio_hotpaths(eng: dnam.Engine) -> None:
    """Profiles zero-copy `process_into`, `process`, `process_planar`, and `__call__`."""
    audio_2d = np.zeros((2, 128), dtype=np.float32)
    out_2d = np.empty_like(audio_2d)
    l_in = np.zeros(128, dtype=np.float32)
    r_in = np.zeros(128, dtype=np.float32)
    l_out = np.empty_like(l_in)
    r_out = np.empty_like(r_in)

    for _ in range(15_000):
        eng.process_into(audio_2d, out_2d)
        eng.process(audio_2d, out=out_2d)
        eng.process_planar(l_in, r_in, out_left=l_out, out_right=r_out)
        eng(audio_2d, 48_000, reset=False)


def run_controls_and_telemetry_hotpaths(eng: dnam.Engine, space: dnam.ControlSpace) -> None:
    """Profiles `Engine.controls`, `ControlSpace.clamp`, and `Engine.telemetry`."""
    ctrls = eng.controls
    raw_query: dict[str, float | int | str] = {"gain": 7.5, "boost": "on"}
    for _ in range(25_000):
        ctrls["gain"] = 7.5
        _ = ctrls["gain"]
        ctrls.snap(gain=7.5, boost="on")
        _ = ctrls.clamped
        _ = space.clamp(raw_query)
        _ = eng.telemetry


def run_model_and_gram_hotpaths(model: dnam.Model) -> None:
    """Profiles `Model.from_bytes`, `Model.with_header`, and `GramMatrix` `uplo='U'` packing."""
    raw = model.to_bytes()
    updated_hdr = model.header.model_copy(update={"user_metadata": {"tag": "fast"}})
    mat = np.eye(256, dtype=np.float64)
    gram = dnam.GramMatrix.from_dense(mat)

    for _ in range(2_000):
        _ = dnam.Model.from_bytes(raw)
        _ = model.with_header(updated_hdr)
        g = dnam.GramMatrix.from_dense(mat)
        _ = g.to_dense()
        _ = gram.to_dense(submatrix_dim=64)


def run_provenance_and_dsp_hotpaths(model: dnam.Model) -> None:
    """Profiles buyer watermarking, 224-D structural hashing, and TORA reference DSP kernels."""
    arena = model.arena_bytes
    code = dnam.provenance.generate_tardos_code(b"\x42" * 32, "model_01", "buyer_01")
    wm_params = WatermarkParams()
    u = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)
    sig = np.linspace(-0.8, 0.8, 4096, dtype=np.float64)
    out = np.empty_like(sig)

    for _ in range(2_000):
        _ = dnam.provenance.apply_gauge_watermark(arena, wm_params)
        _ = dnam.provenance.extract_structural_hash(arena)
        _ = dnam.provenance.score_tardos_watermark(arena, arena, code)
        _ = dnam.dsp.unit_rotor(u)
        _ = dnam.dsp.lobatto_basis(u)
        _ = dnam.dsp.chebyshev_lfo(u)
        _ = dnam.dsp.trust_region_softclip(u)
        _ = dnam.dsp.poly_adaa_slice(sig, out=out)


def main() -> None:
    """Executes all SDK profiling workloads."""
    sys.modules["torch"] = ModuleType("torch")
    model = dnam.Model.reference()
    eng = dnam.Engine(model, sample_rate=48_000, tier="full")
    run_audio_hotpaths(eng)
    run_controls_and_telemetry_hotpaths(eng, model.control_space)
    run_model_and_gram_hotpaths(model)
    run_provenance_and_dsp_hotpaths(model)


if __name__ == "__main__":
    main()
