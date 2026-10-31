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
    assert env_rocm["FLOAT32_MATMUL_PRECISION"] == "highest"

    # Explicit unavailable devices fail fast with ValueError
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="not available"):
            resolve_hardware_profile("cuda")
        with pytest.raises(ValueError, match="not available"):
            resolve_hardware_profile("rocm")
    with pytest.raises(ValueError, match="not available"):
        resolve_hardware_profile("tpu")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        p_mps = resolve_hardware_profile("mps")
        assert p_mps.device_type == "mps"
        assert p_mps.contraction_dtype in ("float32", "bfloat16")
        assert not p_mps.mps_fast_math
        assert p_mps.fast_math_off
        assert p_mps.fp32_matmul_exact

    p_cpu = resolve_hardware_profile("cpu")
    assert p_cpu.device_type == "cpu"
    assert p_cpu.torch_device == "cpu"
    assert p_cpu.cpu_fallback_warning is None

    p_auto = resolve_hardware_profile("auto")
    assert p_auto.device_type in ("tpu", "cuda", "rocm", "mps", "cpu")

    assert torch_dtype_from_profile_name("bfloat16") == torch.bfloat16
    assert torch_dtype_from_profile_name("float32") == torch.float32
    with pytest.raises(ValueError, match="Unsupported device dtype"):
        torch_dtype_from_profile_name("float64")

    t_in = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    arr64 = move_to_host_float64(t_in)
    assert arr64.dtype.name == "float64"
    assert arr64.shape == (2,)

    setup_accelerator_environment("tpu")
    setup_accelerator_environment("cuda")
    setup_accelerator_environment("mps")
    setup_accelerator_environment(None)

    from dnam_trainer.training.accelerator import AcceleratorRuntime, ProbeResult
    from dnam_trainer.training.hardware import _is_tpu_available, runtime_to_hardware_profile

    assert isinstance(_is_tpu_available(), bool)
    fake_tpu_rt = AcceleratorRuntime(
        ProbeResult(
            device_kind="tpu",
            torch_device="cpu",
            contraction_dtype="bfloat16",
            budget_bytes=8 * 1024**3,
            max_tensor_bytes=1 * 1024**3,
            bf16_native=True,
            bf16_speedup_vs_fp32=1.5,
            fp32_matmul_exact=True,
            fast_math_off=True,
        )
    )
    assert runtime_to_hardware_profile(fake_tpu_rt, enable_compile=True).compile_backend == "tpu"
    fake_cuda_rt = AcceleratorRuntime(
        ProbeResult(
            device_kind="cuda",
            torch_device="cpu",
            contraction_dtype="float32",
            budget_bytes=8 * 1024**3,
            max_tensor_bytes=1 * 1024**3,
            bf16_native=True,
            bf16_speedup_vs_fp32=1.0,
            fp32_matmul_exact=True,
            fast_math_off=True,
        )
    )
    assert (
        runtime_to_hardware_profile(fake_cuda_rt, enable_compile=True).compile_backend == "inductor"
    )

    # Exercise auto-detection branches (CPU fallback emits warning)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    p_fallback = resolve_hardware_profile("auto")
    assert p_fallback.device_type == "cpu"
    assert p_fallback.cpu_fallback_warning is not None
