"""Unit and property tests for `dnam_trainer.training.accelerator`."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from dnam_trainer.training.accelerator import (
    AcceleratorRuntime,
    ProbeResult,
    assert_32bit_device_tensor,
    clear_probe_cache,
    configure_precision_backends,
    resolve_accelerator_runtime,
)


def test_resolve_accelerator_auto_and_explicit_cpu() -> None:
    clear_probe_cache()
    rt_cpu = resolve_accelerator_runtime("cpu", budget_gb_override=8.0, force_reprobe=True)
    assert rt_cpu.kind == "cpu"
    assert rt_cpu.device == torch.device("cpu")
    assert rt_cpu.contraction_dtype in (torch.float32, torch.bfloat16)
    assert rt_cpu.budget_bytes == 8 * (1024**3)
    assert rt_cpu.cpu_fallback_warning is None
    assert rt_cpu.probe.fp32_matmul_exact is True
    assert rt_cpu.probe.fast_math_off is True

    rt_auto = resolve_accelerator_runtime("auto")
    assert rt_auto.kind in ("tpu", "cuda", "rocm", "mps", "cpu")


def test_resolve_accelerator_fail_fast_on_unavailable_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_probe_cache()
    monkeypatch.setattr("dnam_trainer.training.accelerator.is_tpu_available", lambda: False)
    with pytest.raises(ValueError, match="Requested accelerator device 'tpu' is not available"):
        resolve_accelerator_runtime("tpu", force_reprobe=True)

    monkeypatch.setattr("dnam_trainer.training.accelerator.is_cuda_available", lambda: False)
    with pytest.raises(ValueError, match="Requested accelerator device 'cuda' is not available"):
        resolve_accelerator_runtime("cuda", force_reprobe=True)

    monkeypatch.setattr("dnam_trainer.training.accelerator.is_rocm_available", lambda: False)
    with pytest.raises(ValueError, match="Requested accelerator device 'rocm' is not available"):
        resolve_accelerator_runtime("rocm", force_reprobe=True)

    monkeypatch.setattr("dnam_trainer.training.accelerator.is_mps_available", lambda: False)
    with pytest.raises(ValueError, match="Requested accelerator device 'mps' is not available"):
        resolve_accelerator_runtime("mps", force_reprobe=True)

    bad_dev: Any = "quantum_asic"
    with pytest.raises(ValueError, match="Unsupported accelerator device"):
        resolve_accelerator_runtime(bad_dev)


def test_resolve_accelerator_auto_cpu_fallback_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_probe_cache()
    monkeypatch.setattr("dnam_trainer.training.accelerator.is_tpu_available", lambda: False)
    monkeypatch.setattr("dnam_trainer.training.accelerator.is_rocm_available", lambda: False)
    monkeypatch.setattr("dnam_trainer.training.accelerator.is_cuda_available", lambda: False)
    monkeypatch.setattr("dnam_trainer.training.accelerator.is_mps_available", lambda: False)

    rt = resolve_accelerator_runtime("auto", force_reprobe=True)
    assert rt.kind == "cpu"
    assert rt.cpu_fallback_warning is not None
    assert "falling back to CPU" in rt.cpu_fallback_warning


def test_resolve_memory_budget_env_and_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_probe_cache()
    monkeypatch.setenv("DNAM_ACCEL_MEMORY_BUDGET_GB", "12.0")
    rt_env = resolve_accelerator_runtime("cpu", force_reprobe=True)
    assert rt_env.budget_bytes == 12 * (1024**3)


def test_plan_2d_scales_down_and_up() -> None:
    # 1. 8 GiB budget on M1 Pro with 30s (1,440,000 samples) and 30 stems -> STREAMED
    rt_8gb = AcceleratorRuntime(
        ProbeResult(
            device_kind="mps",
            torch_device="cpu",
            contraction_dtype="float32",
            bf16_native=False,
            bf16_speedup_vs_fp32=0.0,
            fp32_matmul_exact=True,
            fast_math_off=True,
            budget_bytes=8 * (1024**3),
            max_tensor_bytes=1 * (1024**3),
            unverified_hardware=False,
        )
    )
    plan_s = rt_8gb.plan_2d(
        num_samples=1_440_000,
        num_stems=30,
        max_basis_cols=480,
    )
    assert plan_s.mode == "STREAMED"
    assert plan_s.estimated_peak_bytes <= rt_8gb.budget_bytes
    assert rt_8gb.plan_rows(480) >= 128
    assert rt_8gb.adjust_block_rows_if_over_budget(96_000) == 96_000

    # 2. 64 GiB budget with moderate length/stems -> RESIDENT or RESIDENT_BATCHED
    rt_64gb = AcceleratorRuntime(
        ProbeResult(
            device_kind="cuda",
            torch_device="cpu",
            contraction_dtype="float32",
            bf16_native=True,
            bf16_speedup_vs_fp32=1.4,
            fp32_matmul_exact=True,
            fast_math_off=True,
            budget_bytes=64 * (1024**3),
            max_tensor_bytes=4 * (1024**3),
            unverified_hardware=True,
        )
    )
    plan_r = rt_64gb.plan_2d(
        num_samples=240_000,
        num_stems=8,
        max_basis_cols=240,
    )
    assert plan_r.mode in ("RESIDENT", "RESIDENT_BATCHED")

    # 3. 192 GiB TPU7x Ironwood budget -> RESIDENT_BATCHED
    rt_tpu7x = AcceleratorRuntime(
        ProbeResult(
            device_kind="tpu",
            torch_device="cpu",
            contraction_dtype="bfloat16",
            bf16_native=True,
            bf16_speedup_vs_fp32=1.8,
            fp32_matmul_exact=True,
            fast_math_off=True,
            budget_bytes=int(0.85 * 192 * (1024**3)),
            max_tensor_bytes=4 * (1024**3),
            unverified_hardware=True,
        )
    )
    plan_rb = rt_tpu7x.plan_2d(
        num_samples=480_000,
        num_stems=16,
        max_basis_cols=320,
    )
    assert plan_rb.mode == "RESIDENT_BATCHED"
    assert plan_rb.stem_group_size == 16


def test_assert_32bit_device_tensor_and_precision_config() -> None:
    cfg = configure_precision_backends("cpu")
    assert cfg["FLOAT32_MATMUL_PRECISION"] == "highest"

    assert_32bit_device_tensor(torch.zeros(4, dtype=torch.float32))
    assert_32bit_device_tensor(torch.zeros(4, dtype=torch.int32))
    assert_32bit_device_tensor(torch.zeros(4, dtype=torch.bfloat16))

    with pytest.raises(TypeError, match="Forbidden 64-bit tensor dtype"):
        assert_32bit_device_tensor(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(TypeError, match="Forbidden 64-bit tensor dtype"):
        assert_32bit_device_tensor(torch.zeros(4, dtype=torch.int64))


def test_accelerator_runtime_methods_and_mocked_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnam_trainer.training.accelerator import (
        _detect_budget_bytes,
        _probe_bf16_contraction,
        _probe_fp32_matmul_integer_exact,
        _probe_two_sum_exact,
        is_mps_available,
    )

    dev_cpu = torch.device("cpu")
    assert _probe_two_sum_exact(dev_cpu) is True
    assert _probe_fp32_matmul_integer_exact(dev_cpu) is True
    bf16_ok, bf16_spd = _probe_bf16_contraction(dev_cpu)
    assert isinstance(bf16_ok, bool)
    assert bf16_spd >= 0.0

    if is_mps_available():
        rt_mps = resolve_accelerator_runtime("mps", force_reprobe=True)
        rt_mps.synchronize()
        rt_mps.reclaim()
        assert rt_mps.current_allocated_bytes() >= 0

    # Test _detect_budget_bytes across device kinds
    assert _detect_budget_bytes("tpu", dev_cpu) == int(192 * (1024**3) * 0.85)
    assert _detect_budget_bytes("mps", dev_cpu) >= 1 * (1024**3)
    assert _detect_budget_bytes("cpu", dev_cpu) >= 1 * (1024**3)

    # Test runtime methods: synchronize, reclaim, current_allocated_bytes, adjust_block_rows
    rt = resolve_accelerator_runtime("cpu", budget_gb_override=1.0, force_reprobe=True)
    rt.synchronize()
    rt.reclaim()
    assert rt.current_allocated_bytes() == 0

    monkeypatch.setattr(rt, "current_allocated_bytes", lambda: 4 * (1024**3))
    assert rt.adjust_block_rows_if_over_budget(96_000, min_rows=24_000) == 48_000
    monkeypatch.setattr(rt, "current_allocated_bytes", lambda: 100)
    assert rt.adjust_block_rows_if_over_budget(96_000, min_rows=24_000) == 96_000

    # Test env var budget override & invalid env var
    monkeypatch.setenv("DNAM_ACCEL_MEMORY_BUDGET_GB", "4.0")
    assert _detect_budget_bytes("cpu", dev_cpu) == 4 * (1024**3)
    monkeypatch.setenv("DNAM_ACCEL_MEMORY_BUDGET_GB", "not_a_float")
    assert _detect_budget_bytes("cpu", dev_cpu) >= 1 * (1024**3)
    monkeypatch.delenv("DNAM_ACCEL_MEMORY_BUDGET_GB", raising=False)

    # Test mocked TPU / ROCm / CUDA branches of resolve_accelerator_runtime & precision policies
    import dnam_trainer.training.accelerator as accel_mod

    orig_torch_device = torch.device
    monkeypatch.setattr(accel_mod.torch, "device", lambda _s: orig_torch_device("cpu"))
    monkeypatch.setattr(accel_mod, "_probe_bf16_contraction", lambda _dev: (True, 1.5))

    class _FakeTpuMod:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return (64 * (1024**3), 192 * (1024**3))

    monkeypatch.setattr(accel_mod.torch, "tpu", _FakeTpuMod, raising=False)
    monkeypatch.setattr(accel_mod, "is_tpu_available", lambda: True)
    rt_tpu_auto = resolve_accelerator_runtime("auto", precision="bf16-mixed", force_reprobe=True)
    assert rt_tpu_auto.kind == "tpu"
    assert rt_tpu_auto.contraction_dtype == torch.bfloat16
    rt_tpu_exp = resolve_accelerator_runtime("tpu", precision="fp32", force_reprobe=True)
    assert rt_tpu_exp.kind == "tpu"
    assert rt_tpu_exp.contraction_dtype == torch.float32

    monkeypatch.setattr(accel_mod, "is_tpu_available", lambda: False)
    monkeypatch.setattr(accel_mod, "is_rocm_available", lambda: True)
    assert resolve_accelerator_runtime("auto", force_reprobe=True).kind == "rocm"
    assert resolve_accelerator_runtime("rocm", force_reprobe=True).kind == "rocm"

    monkeypatch.setattr(accel_mod, "is_rocm_available", lambda: False)
    monkeypatch.setattr(accel_mod, "is_cuda_available", lambda: True)
    monkeypatch.setattr(accel_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        accel_mod.torch.cuda, "mem_get_info", lambda _d=None: (8 * (1024**3), 16 * (1024**3))
    )
    monkeypatch.setattr(accel_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(accel_mod.torch.cuda, "memory_allocated", lambda _d=None: 4096)
    assert resolve_accelerator_runtime("auto", force_reprobe=True).kind == "cuda"
    rt_cuda = resolve_accelerator_runtime("cuda", force_reprobe=True)
    assert rt_cuda.kind == "cuda"
    rt_cuda.reclaim()
    assert rt_cuda.current_allocated_bytes() == 4096

    # Test RESIDENT mode in plan_2d (budget >= resident_est and < 3 * resident_est)
    rt_res = resolve_accelerator_runtime("cpu", budget_gb_override=1.2, force_reprobe=True)
    assert rt_res.plan_2d(48_000, 4).mode == "RESIDENT"

    # Test MPS > 16 GiB recommended_max_memory and RuntimeError fallbacks
    if hasattr(accel_mod.torch, "mps"):
        monkeypatch.setattr(accel_mod.torch.mps, "recommended_max_memory", lambda: 64 * (1024**3))
        assert _detect_budget_bytes("mps", dev_cpu) == int(64 * (1024**3) * 0.80)

        def _raise_rt() -> int:
            raise RuntimeError("fail")

        monkeypatch.setattr(accel_mod.torch.mps, "recommended_max_memory", _raise_rt)
        assert _detect_budget_bytes("mps", dev_cpu) == 8 * (1024**3)
        monkeypatch.setattr(accel_mod.torch.mps, "driver_allocated_memory", _raise_rt)
        monkeypatch.setattr(accel_mod.torch.mps, "current_allocated_memory", lambda: 2048)
        rt_mps_fb = resolve_accelerator_runtime("cpu", force_reprobe=True)
        rt_mps_fb.kind = "mps"
        assert rt_mps_fb.current_allocated_bytes() == 2048
