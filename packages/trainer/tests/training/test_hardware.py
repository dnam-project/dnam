"""Unit tests for `dnam_trainer.training.hardware` (TPU, ROCm, CUDA, MPS, CPU runtime profiles)."""

from __future__ import annotations

import os

import pytest
import torch

from dnam_trainer.training.hardware import (
    move_to_host_float64,
    resolve_hardware_profile,
    setup_accelerator_environment,
    torch_dtype_from_profile_name,
)


def test_resolve_hardware_profiles_all_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    env_rocm = setup_accelerator_environment("rocm")
    assert env_rocm["MIOPEN_FIND_MODE"] == "FAST"
    assert os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"

    p_tpu = resolve_hardware_profile("tpu", enable_compile=True)
    assert p_tpu.device_type == "tpu"
    assert p_tpu.contraction_dtype == "bfloat16"
    assert p_tpu.recurrence_dtype == "float32"
    assert p_tpu.kronecker_solver_dtype == "float64"
    assert p_tpu.compile_backend == "tpu"

    p_rocm = resolve_hardware_profile("rocm", enable_compile=True)
    assert p_rocm.device_type == "rocm"
    assert p_rocm.rocm_miopen_fast_find
    assert not p_rocm.cudnn_benchmark

    p_cuda = resolve_hardware_profile("cuda", precision="fp32")
    assert p_cuda.device_type == "cuda"
    assert p_cuda.contraction_dtype == "float32"

    p_mps = resolve_hardware_profile("mps")
    assert p_mps.device_type == "mps"
    assert p_mps.contraction_dtype == "float32"

    p_cpu = resolve_hardware_profile("cpu")
    assert p_cpu.device_type == "cpu"
    assert p_cpu.torch_device == "cpu"

    p_auto = resolve_hardware_profile("auto")
    assert p_auto.device_type in ("tpu", "cuda", "rocm", "mps", "cpu")

    assert torch_dtype_from_profile_name("bfloat16") == torch.bfloat16
    assert torch_dtype_from_profile_name("float64") == torch.float64
    assert torch_dtype_from_profile_name("float32") == torch.float32

    t_in = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    arr64 = move_to_host_float64(t_in)
    assert arr64.dtype.name == "float64"
    assert arr64.shape == (2,)

    setup_accelerator_environment("cuda")
    setup_accelerator_environment("mps")

    # Exercise auto-detection branches (TPU, CUDA, CPU fallback)
    class _FakeTpuMod:
        @staticmethod
        def is_available() -> bool:
            return True

    monkeypatch.setattr(torch, "tpu", _FakeTpuMod(), raising=False)
    assert resolve_hardware_profile("auto").device_type == "tpu"
    monkeypatch.delattr(torch, "tpu", raising=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_hardware_profile("auto", precision="fp32").device_type in ("cuda", "rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_hardware_profile("auto").device_type == "cpu"
