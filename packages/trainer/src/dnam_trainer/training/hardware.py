"""Hardware runtime profile resolution, accelerator environment setup, and 32-bit device
precision configuration.

Supports:
1. **Google Cloud TPU (`torch-tpu` / `PrivateUse1`):** capability-probed `bfloat16` contractions
   (`MXU`), `float32` IIR recurrences (`VPU`), `highest` matmul precision, and batched host-CPU
   `float64` dense solvers.
2. **AMD ROCm (`torch.version.hip` / `gfx1201`):** `MIOPEN_FIND_MODE="FAST"`,
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"`, `cudnn.benchmark = False`, `allow_tf32 = False`,
   capability-probed `bfloat16` contractions + `float32` recurrences.
3. **NVIDIA CUDA:** `allow_tf32 = False` (`highest` matmul precision), `cudnn.benchmark = True`,
   capability-probed `bfloat16` contractions + `float32` recurrences.
4. **Apple Silicon (`MPS`):** `mps_fast_math = False` (required for IEEE-754 `two_sum` exactness),
   capability-probed `float32`/`bfloat16` contractions, `1 GiB` per-tensor cap, and batched
   host-CPU `float64` dense solvers.
5. **CPU:** 32-bit device path (`float32` contractions and recurrences) + batched host-CPU
   `float64` dense solvers.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from dnam_trainer.schema import PrecisionPolicy, StrictModel
from dnam_trainer.training.accelerator import (
    AcceleratorRuntime,
    ContractionDtypeName,
    DeviceKind,
    configure_precision_backends,
    is_tpu_available,
    resolve_accelerator_runtime,
)


class HardwareRuntimeProfile(StrictModel):
    """Resolved accelerator runtime profile for the 4-Stage DNAM trainer."""

    device_type: DeviceKind
    torch_device: str
    contraction_dtype: ContractionDtypeName = "float32"
    recurrence_dtype: Literal["float32"] = "float32"
    kronecker_solver_dtype: Literal["float64"] = "float64"
    compile_backend: str | None = None
    rocm_miopen_fast_find: bool = False
    cudnn_benchmark: bool = False
    mps_fast_math: bool = False
    pinned_memory: bool = False
    budget_bytes: int = 8 * 1024 * 1024 * 1024
    max_tensor_bytes: int = 1024 * 1024 * 1024
    bf16_native: bool = False
    fp32_matmul_exact: bool = True
    fast_math_off: bool = True
    unverified_hardware: bool = False
    cpu_fallback_warning: str | None = None


def _is_tpu_available() -> bool:
    """Checks whether `torch.tpu` (`torch-tpu` `PrivateUse1` backend) is available."""
    return is_tpu_available()


def setup_accelerator_environment(device_type: str | None = None) -> dict[str, str]:
    """Configures environment variables and PyTorch backend flags before tensor allocation.

    Forces `allow_tf32 = False` and `float32_matmul_precision = "highest"`. Never enables MPS
    fast-math (`two_sum` error-free transformations require strict IEEE-754 rounding).
    """
    if device_type == "tpu":
        kind: DeviceKind = "tpu"
    elif device_type == "cuda":
        kind = "cuda"
    elif device_type == "rocm":
        kind = "rocm"
    elif device_type == "mps":
        kind = "mps"
    else:
        kind = "cpu"
    return configure_precision_backends(kind)


def resolve_hardware_profile(
    requested_device: Literal["auto", "tpu", "cuda", "rocm", "mps", "cpu"] = "auto",
    *,
    precision: PrecisionPolicy = "auto",
    enable_compile: bool = False,
    budget_gb_override: float | None = None,
) -> HardwareRuntimeProfile:
    """Resolves the [`HardwareRuntimeProfile`] for the current or requested accelerator.

    Raises `ValueError` if an explicitly requested accelerator (`tpu`, `cuda`, `rocm`, `mps`)
    is unavailable on the current host.
    """
    runtime = resolve_accelerator_runtime(
        requested_device,
        precision=precision,
        budget_gb_override=budget_gb_override,
    )
    return runtime_to_hardware_profile(runtime, enable_compile=enable_compile)


def runtime_to_hardware_profile(
    runtime: AcceleratorRuntime,
    *,
    enable_compile: bool = False,
) -> HardwareRuntimeProfile:
    """Converts an [`AcceleratorRuntime`] instance into a [`HardwareRuntimeProfile`]."""
    p = runtime.probe
    compile_backend: str | None = None
    if enable_compile:
        if p.device_kind == "tpu":
            compile_backend = "tpu"
        elif p.device_kind in ("cuda", "rocm"):
            compile_backend = "inductor"

    return HardwareRuntimeProfile(
        device_type=p.device_kind,
        torch_device=p.torch_device,
        contraction_dtype=p.contraction_dtype,
        recurrence_dtype="float32",
        kronecker_solver_dtype="float64",
        compile_backend=compile_backend,
        rocm_miopen_fast_find=(p.device_kind == "rocm"),
        cudnn_benchmark=(p.device_kind == "cuda"),
        mps_fast_math=False,
        pinned_memory=(p.device_kind in ("cuda", "rocm")),
        budget_bytes=p.budget_bytes,
        max_tensor_bytes=p.max_tensor_bytes,
        bf16_native=p.bf16_native,
        fp32_matmul_exact=p.fp32_matmul_exact,
        fast_math_off=p.fast_math_off,
        unverified_hardware=p.unverified_hardware,
        cpu_fallback_warning=p.cpu_fallback_warning,
    )


def torch_dtype_from_profile_name(name: str) -> torch.dtype:
    """Maps a 32-bit/16-bit profile dtype name (`'float32'`, `'bfloat16'`) to `torch.dtype`."""
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(
        f"Unsupported device dtype name '{name}'; only 'float32' and 'bfloat16' are permitted."
    )


def move_to_host_float64(tensor: torch.Tensor) -> NDArray[np.float64]:
    """Detaches `tensor`, moves it to CPU in `float32`, and converts to a contiguous `float64`
    NumPy array on the host.
    """
    arr: Any = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return np.ascontiguousarray(arr, dtype=np.float64)
