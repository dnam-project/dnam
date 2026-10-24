"""Hardware runtime profile resolution, accelerator environment setup, and split-precision
configuration.

Supports:
1. **Google Cloud TPU (`torch-tpu` / `PrivateUse1`):** `bfloat16` contractions (`MXU`),
   `float32` IIR recurrences (`VPU`), host-CPU `float64` Kronecker Cholesky solver, and
   `torch.compile(backend="tpu")`.
2. **AMD ROCm (`torch.version.hip` / `gfx1201`):** `MIOPEN_FIND_MODE="FAST"`,
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"`, `cudnn.benchmark = False` (avoids MIOpen
   exhaustive-search hangs on variable-length audio), `bfloat16` contractions + `float32`
   recurrences.
3. **NVIDIA CUDA:** TF32 matmul/cuDNN enabled, `cudnn.benchmark = True`, `bfloat16`
   contractions (or `float32` if `bf16` unsupported) + `float32` recurrences.
4. **Apple Silicon (`MPS`):** `torch._C._mps_setFastMathEnabled(True)`, `float32` contractions and
recurrences (avoids non-contiguous MPS `bf16` fallback copies), host-CPU `float64` Kronecker
solver.
5. **CPU:** `float32` contractions and recurrences, native `float64` Kronecker solver.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from dnam_trainer.schema import PrecisionPolicy, StrictModel

ContractionDtypeName = Literal["float32", "bfloat16"]


class HardwareRuntimeProfile(StrictModel):
    """Resolved accelerator runtime profile for the 4-Stage DNAM trainer."""

    device_type: Literal["tpu", "cuda", "rocm", "mps", "cpu"]
    torch_device: str
    contraction_dtype: ContractionDtypeName = "float32"
    recurrence_dtype: Literal["float32"] = "float32"
    kronecker_solver_dtype: Literal["float64"] = "float64"
    compile_backend: str | None = None
    rocm_miopen_fast_find: bool = False
    cudnn_benchmark: bool = False
    mps_fast_math: bool = False
    pinned_memory: bool = False


def _is_tpu_available() -> bool:
    """Checks whether `torch.tpu` (`torch-tpu` `PrivateUse1` backend) is available."""
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is None:
        return False
    is_avail = getattr(tpu_mod, "is_available", None)
    return bool(callable(is_avail) and is_avail())


def setup_accelerator_environment(device_type: str | None = None) -> dict[str, str]:
    """Configures environment variables and PyTorch backend flags before tensor allocation.

    Returns a dictionary of environment variables set or verified during initialization.
    """
    applied_env: dict[str, str] = {}
    is_rocm = getattr(torch.version, "hip", None) is not None

    if is_rocm or device_type == "rocm":
        for key, val in (
            ("MIOPEN_FIND_MODE", "FAST"),
            ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
        ):
            if key not in os.environ:
                os.environ[key] = val
            applied_env[key] = os.environ[key]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
    elif torch.cuda.is_available() or device_type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) or (
        device_type == "mps"
    ):
        mps_fast_math = getattr(torch._C, "_mps_setFastMathEnabled", None)
        if callable(mps_fast_math):
            try:
                mps_fast_math(True)
                applied_env["MPS_FAST_MATH"] = "1"
            except RuntimeError:
                pass

    return applied_env


def resolve_hardware_profile(
    requested_device: Literal["auto", "tpu", "cuda", "rocm", "mps", "cpu"] = "auto",
    *,
    precision: PrecisionPolicy = "auto",
    enable_compile: bool = False,
) -> HardwareRuntimeProfile:
    """Resolves the optimal [`HardwareRuntimeProfile`] for the current or requested accelerator."""
    setup_accelerator_environment(None if requested_device == "auto" else requested_device)
    is_rocm = getattr(torch.version, "hip", None) is not None

    resolved: Literal["tpu", "cuda", "rocm", "mps", "cpu"]
    if requested_device == "auto":
        if _is_tpu_available():
            resolved = "tpu"
        elif torch.cuda.is_available():
            resolved = "rocm" if is_rocm else "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = requested_device

    force_fp32 = precision == "fp32"

    if resolved == "tpu":
        return HardwareRuntimeProfile(
            device_type="tpu",
            torch_device="tpu:0",
            contraction_dtype="float32" if force_fp32 else "bfloat16",
            recurrence_dtype="float32",
            kronecker_solver_dtype="float64",
            compile_backend="tpu" if enable_compile else None,
            rocm_miopen_fast_find=False,
            cudnn_benchmark=False,
            mps_fast_math=False,
            pinned_memory=False,
        )

    if resolved == "rocm":
        return HardwareRuntimeProfile(
            device_type="rocm",
            torch_device="cuda:0" if torch.cuda.is_available() else "cpu",
            contraction_dtype="float32" if force_fp32 else "bfloat16",
            recurrence_dtype="float32",
            kronecker_solver_dtype="float64",
            compile_backend="inductor" if enable_compile else None,
            rocm_miopen_fast_find=True,
            cudnn_benchmark=False,
            mps_fast_math=False,
            pinned_memory=torch.cuda.is_available(),
        )

    if resolved == "cuda":
        bf16_ok = (
            not force_fp32
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        contraction: ContractionDtypeName = "bfloat16" if bf16_ok else "float32"
        return HardwareRuntimeProfile(
            device_type="cuda",
            torch_device="cuda:0" if torch.cuda.is_available() else "cpu",
            contraction_dtype=contraction,
            recurrence_dtype="float32",
            kronecker_solver_dtype="float64",
            compile_backend="inductor" if enable_compile else None,
            rocm_miopen_fast_find=False,
            cudnn_benchmark=True,
            mps_fast_math=False,
            pinned_memory=torch.cuda.is_available(),
        )

    if resolved == "mps":
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return HardwareRuntimeProfile(
            device_type="mps",
            torch_device="mps" if mps_avail else "cpu",
            contraction_dtype="float32",
            recurrence_dtype="float32",
            kronecker_solver_dtype="float64",
            compile_backend=None,
            rocm_miopen_fast_find=False,
            cudnn_benchmark=False,
            mps_fast_math=True,
            pinned_memory=False,
        )

    return HardwareRuntimeProfile(
        device_type="cpu",
        torch_device="cpu",
        contraction_dtype="float32",
        recurrence_dtype="float32",
        kronecker_solver_dtype="float64",
        compile_backend=None,
        rocm_miopen_fast_find=False,
        cudnn_benchmark=False,
        mps_fast_math=False,
        pinned_memory=False,
    )


def torch_dtype_from_profile_name(name: str) -> torch.dtype:
    """Maps a profile dtype name (`'float32'`, `'bfloat16'`, `'float64'`) to `torch.dtype`."""
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    return torch.float32


def move_to_host_float64(tensor: torch.Tensor) -> NDArray[np.float64]:
    """Detaches `tensor`, moves it to CPU, and converts to a contiguous `float64` NumPy array.

    Safe across CPU, CUDA, ROCm, MPS, and TPU (`PrivateUse1`) where `float64` on-device ops
    may be unsupported or emulated.
    """
    arr: Any = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return np.ascontiguousarray(arr, dtype=np.float64)
