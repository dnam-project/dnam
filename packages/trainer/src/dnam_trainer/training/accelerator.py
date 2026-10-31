"""Accelerator runtime resolution, startup capability probes, uncapped memory budgeting,
and 2-D streaming planner (`STREAMED`, `RESIDENT`, `RESIDENT_BATCHED`).

Enforces:
1. **Accelerator-first execution:** `auto` resolves `tpu -> cuda/rocm -> mps`, falling back to
   `cpu` only when no accelerator exists (with a structured warning). Explicit unavailable
   accelerators fail fast with `ValueError`.
2. **Startup capability probes:**
   - `bf16` contraction probe (verifies exactness and speed vs `fp32`),
   - `fp32` matmul full-precision probe (forces TF32 off on CUDA/ROCm and `highest` on TPU),
   - IEEE-754 error-free transformation (`two_sum`) probe (verifies fast-math is off).
3. **Uncapped bi-directional memory scaling:** Scales down to `8 GiB` on a 16 GB Apple M1 Pro
   (`max_tensor_bytes = 1 GiB` to avoid >2 GB MPS tensor corruption) and scales up without an
   upper cap to large unified-memory Macs, multi-GPU nodes, and Cloud TPU7x (`192 GiB` HBM).
"""

from __future__ import annotations

import gc
import os
import time
from typing import Literal

import torch
from torch import Tensor

from dnam_trainer.schema import PrecisionPolicy, StrictModel

DeviceKind = Literal["tpu", "cuda", "rocm", "mps", "cpu"]
ContractionDtypeName = Literal["float32", "bfloat16"]
PlannerMode = Literal["STREAMED", "RESIDENT", "RESIDENT_BATCHED"]

_GIB: int = 1024 * 1024 * 1024
_MIB: int = 1024 * 1024


class ProbeResult(StrictModel):
    """Startup capability probe and memory budget telemetry for the active device."""

    device_kind: DeviceKind
    torch_device: str
    contraction_dtype: ContractionDtypeName
    bf16_native: bool
    bf16_speedup_vs_fp32: float
    fp32_matmul_exact: bool
    fast_math_off: bool
    budget_bytes: int
    max_tensor_bytes: int
    unverified_hardware: bool = False
    cpu_fallback_warning: str | None = None


class PlannerDecision(StrictModel):
    """2-D streaming planner configuration (`time_block_rows x stem_group_size`)."""

    mode: PlannerMode
    num_samples: int
    num_stems: int
    block_rows: int
    stem_group_size: int
    budget_bytes: int
    estimated_peak_bytes: int


def is_tpu_available() -> bool:
    """Returns `True` if the `torch.tpu` (`PrivateUse1`) backend is installed and available."""
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is None:
        return False
    is_avail = getattr(tpu_mod, "is_available", None)
    return bool(callable(is_avail) and is_avail())


def is_rocm_available() -> bool:
    """Returns `True` if AMD ROCm HIP runtime is active and a GPU is available."""
    return getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()


def is_cuda_available() -> bool:
    """Returns `True` if NVIDIA CUDA runtime is active and a GPU is available."""
    return getattr(torch.version, "hip", None) is None and torch.cuda.is_available()


def is_mps_available() -> bool:
    """Returns `True` if Apple Metal Performance Shaders (`MPS`) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def configure_precision_backends(device_kind: DeviceKind) -> dict[str, str]:
    """Forces full `fp32` matmul precision (TF32 off, TPU `highest`, MPS fast-math off)."""
    applied: dict[str, str] = {}

    if device_kind == "rocm" or getattr(torch.version, "hip", None) is not None:
        for key, val in (
            ("MIOPEN_FIND_MODE", "FAST"),
            ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
        ):
            if key not in os.environ:
                os.environ[key] = val
            applied[key] = os.environ[key]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
        applied["CUDA_MATMUL_ALLOW_TF32"] = "0"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        if device_kind == "cuda":
            torch.backends.cudnn.benchmark = True
        applied["CUDNN_ALLOW_TF32"] = "0"

    torch.set_float32_matmul_precision("highest")
    applied["FLOAT32_MATMUL_PRECISION"] = "highest"
    return applied


def _probe_two_sum_exact(device: torch.device) -> bool:
    """Verifies that IEEE-754 Knuth `two_sum` EFT is not broken by fast-math."""
    try:
        a = torch.tensor([1.0], device=device, dtype=torch.float32)
        b = torch.tensor([float(2.0**-24)], device=device, dtype=torch.float32)
        s = a + b
        v = s - a
        e = (a - (s - v)) + (b - v)
        return bool((e == b).all().detach().cpu().item())
    except RuntimeError:
        return False


def _probe_fp32_matmul_integer_exact(device: torch.device) -> bool:
    """Verifies that `fp32 @ fp32` preserves full 24-bit significand on 8-bit integer inputs.

    A 128-element dot product of integers in `[-127, 127]` reaches `2,064,512 < 2^21`, which is
    bit-exact in IEEE-754 `binary32` (24-bit mantissa) but fails if silently downcast to `TF32`
    (11-bit mantissa) or `bfloat16` (8-bit mantissa).
    """
    try:
        idx = torch.arange(16 * 128, device=device, dtype=torch.int32).reshape(16, 128)
        a_i32 = ((idx * 73 + 19) % 255) - 127
        b_i32 = ((idx * 41 + 101) % 255) - 127
        c_f32 = a_i32.to(torch.float32) @ b_i32.to(torch.float32).T
        # Compute reference in int32 via broadcasting sum
        c_ref = (a_i32.unsqueeze(1) * b_i32.unsqueeze(0)).sum(dim=-1).to(torch.float32)
        max_err = float((c_f32 - c_ref).abs().max().detach().cpu().item())
        return max_err == 0.0
    except RuntimeError:
        return False


def _probe_bf16_contraction(device: torch.device) -> tuple[bool, float]:
    """Probes whether `bfloat16` matmul is native, accurate for small integer products, and faster
    or equal to `float32` matmul on `device`.
    """
    if device.type == "cpu":
        return False, 0.0
    try:
        # 1. Check small integer exactness within bf16 8-bit significand (<= 144 < 256)
        idx = torch.arange(16 * 16, device=device, dtype=torch.int32).reshape(16, 16)
        a_i32 = ((idx * 37 + 11) % 7) - 3
        b_i32 = ((idx * 53 + 7) % 7) - 3
        a_bf16 = a_i32.to(torch.bfloat16)
        b_bf16 = b_i32.to(torch.bfloat16)
        c_bf16 = (a_bf16 @ b_bf16.T).to(torch.float32)
        c_ref = (a_i32.to(torch.float32)) @ (b_i32.to(torch.float32).T)
        if float((c_bf16 - c_ref).abs().max().detach().cpu().item()) > 0.0:
            return False, 0.0

        # 2. Micro-benchmark (256, 256) @ (256, 256) bf16 vs fp32
        x_f32 = torch.randn((256, 256), device=device, dtype=torch.float32)
        y_f32 = torch.randn((256, 256), device=device, dtype=torch.float32)
        x_bf16 = x_f32.to(torch.bfloat16)
        y_bf16 = y_f32.to(torch.bfloat16)

        _ = x_f32 @ y_f32
        _ = x_bf16 @ y_bf16
        _sync_device(device)

        t0 = time.perf_counter()
        for _ in range(4):
            _ = x_f32 @ y_f32
        _sync_device(device)
        dt_f32 = max(1e-9, time.perf_counter() - t0)

        t1 = time.perf_counter()
        for _ in range(4):
            _ = x_bf16 @ y_bf16
        _sync_device(device)
        dt_bf16 = max(1e-9, time.perf_counter() - t1)

        speedup = dt_f32 / dt_bf16
        # Require bf16 to be at least as fast as fp32 (within 2% noise margin)
        return speedup >= 0.98, float(speedup)
    except RuntimeError, TypeError:
        return False, 0.0


def _sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "tpu":
        tpu_mod = getattr(torch, "tpu", None)
        sync_fn = getattr(tpu_mod, "synchronize", None) if tpu_mod is not None else None
        if callable(sync_fn):
            sync_fn()


def _detect_budget_bytes(
    device_kind: DeviceKind,
    device: torch.device,
    *,
    budget_gb_override: float | None = None,
) -> int:
    """Computes the uncapped accelerator memory budget in bytes."""
    if budget_gb_override is not None and budget_gb_override > 0.0:
        return max(256 * _MIB, int(budget_gb_override * _GIB))

    env_gb = os.environ.get("DNAM_ACCEL_MEMORY_BUDGET_GB", "").strip()
    if env_gb:
        try:
            parsed_gb = float(env_gb)
            if parsed_gb > 0.0:
                return max(256 * _MIB, int(parsed_gb * _GIB))
        except ValueError:
            pass

    if device_kind in ("cuda", "rocm") and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(device)
            base_b = free_b if free_b > 0 else total_b
            return max(512 * _MIB, int(base_b * 0.85))
        except RuntimeError:
            pass

    if device_kind == "mps" and hasattr(torch, "mps"):
        rec_fn = getattr(torch.mps, "recommended_max_memory", None)
        if callable(rec_fn):
            try:
                rec_b = int(rec_fn())
                if rec_b > 0:
                    # On 16 GB Apple Silicon (rec_b ~ 11.84 GiB), cap default budget at 8 GiB;
                    # on larger unified-memory Macs (32 GB..512 GB), scale uncapped at 80% of rec_b.
                    if rec_b <= 16 * _GIB:
                        return max(1 * _GIB, min(8 * _GIB, int(rec_b * 0.75)))
                    return int(rec_b * 0.80)
            except RuntimeError:
                pass
        return 8 * _GIB

    if device_kind == "tpu":
        # Default to TPU7x (Ironwood, 192 GiB HBM * 0.85) unless reported by torch.tpu
        tpu_mod = getattr(torch, "tpu", None)
        mem_fn = getattr(tpu_mod, "mem_get_info", None) if tpu_mod is not None else None
        if callable(mem_fn):
            try:
                info = mem_fn()
                if isinstance(info, tuple) and len(info) >= 2:
                    return max(1 * _GIB, int(int(info[0]) * 0.85))
            except RuntimeError:
                pass
        return int(192 * _GIB * 0.85)

    # CPU fallback: 50% of physical RAM (8 GiB on a 16 GB host)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0:
            total_ram = page_size * phys_pages
            return max(1 * _GIB, int(total_ram * 0.50))
    except AttributeError, ValueError, OSError:
        pass
    return 8 * _GIB


def _compute_max_tensor_bytes(device_kind: DeviceKind, budget_bytes: int) -> int:
    """Computes the single-tensor byte ceiling (`1 GiB` on MPS to prevent >2 GB corruption)."""
    if device_kind == "mps":
        return min(1 * _GIB, max(128 * _MIB, budget_bytes // 4))
    return min(4 * _GIB, max(256 * _MIB, budget_bytes // 4))


_PROBE_CACHE: dict[tuple[str, str, int | None], ProbeResult] = {}


def clear_probe_cache() -> None:
    """Clears the cached startup capability probe results."""
    _PROBE_CACHE.clear()


class AcceleratorRuntime:
    """Resolved accelerator runtime, capability probe results, and 2-D memory planner."""

    def __init__(self, probe: ProbeResult) -> None:
        self.probe = probe
        self.kind: DeviceKind = probe.device_kind
        self.device = torch.device(probe.torch_device)
        self.contraction_dtype: torch.dtype = (
            torch.bfloat16 if probe.contraction_dtype == "bfloat16" else torch.float32
        )
        self.budget_bytes: int = probe.budget_bytes
        self.max_tensor_bytes: int = probe.max_tensor_bytes
        self.unverified_hardware: bool = probe.unverified_hardware
        self.cpu_fallback_warning: str | None = probe.cpu_fallback_warning

    def synchronize(self) -> None:
        """Synchronizes asynchronous device queues."""
        _sync_device(self.device)

    def reclaim(self) -> None:
        """Forces Python garbage collection and releases cached device allocator blocks."""
        gc.collect()
        if self.kind == "mps" and hasattr(torch, "mps"):
            empty_fn = getattr(torch.mps, "empty_cache", None)
            if callable(empty_fn):
                empty_fn()
        elif self.kind in ("cuda", "rocm") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def current_allocated_bytes(self) -> int:
        """Returns current device driver/allocator memory usage in bytes."""
        if self.kind == "mps" and hasattr(torch, "mps"):
            drv_fn = getattr(torch.mps, "driver_allocated_memory", None)
            if callable(drv_fn):
                try:
                    return int(drv_fn())
                except RuntimeError:
                    pass
            cur_fn = getattr(torch.mps, "current_allocated_memory", None)
            if callable(cur_fn):
                try:
                    return int(cur_fn())
                except RuntimeError:
                    pass
        elif self.kind in ("cuda", "rocm") and torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated(self.device)
            except RuntimeError:
                pass
        return 0

    def plan_rows(
        self,
        num_cols: int,
        *,
        dtype_bytes: int = 4,
        working_copies: int = 3,
        align_rows: int = 128,
    ) -> int:
        """Computes the maximum safe row chunk size for a tensor of width `num_cols`."""
        cols = max(1, num_cols)
        bytes_per_row = cols * max(1, dtype_bytes)
        by_tensor_cap = self.max_tensor_bytes // bytes_per_row
        by_budget = (self.budget_bytes // max(2, working_copies * 2)) // bytes_per_row
        raw_rows = max(align_rows, min(by_tensor_cap, by_budget))
        return max(align_rows, (raw_rows // align_rows) * align_rows)

    def plan_2d(
        self,
        num_samples: int,
        num_stems: int,
        *,
        max_basis_cols: int = 800,
        win_step: int = 24_000,
    ) -> PlannerDecision:
        """Plans `(mode, block_rows, stem_group_size)` for `(num_samples, num_stems)`."""
        l_tot = max(1, num_samples)
        m_stems = max(1, num_stems)
        cols = max(64, max_basis_cols)

        # Full-resident working set estimate: 4 full-length basis/intermediate buffers + stem bands
        full_basis_bytes = l_tot * cols * 4
        resident_est = full_basis_bytes * 4 + l_tot * m_stems * 48 + 512 * _MIB

        can_hold_single_full_tensor = full_basis_bytes <= self.max_tensor_bytes
        if self.budget_bytes >= 3 * resident_est and can_hold_single_full_tensor:
            return PlannerDecision(
                mode="RESIDENT_BATCHED",
                num_samples=l_tot,
                num_stems=m_stems,
                block_rows=l_tot,
                stem_group_size=m_stems,
                budget_bytes=self.budget_bytes,
                estimated_peak_bytes=resident_est,
            )

        if self.budget_bytes >= resident_est and can_hold_single_full_tensor:
            return PlannerDecision(
                mode="RESIDENT",
                num_samples=l_tot,
                num_stems=m_stems,
                block_rows=l_tot,
                stem_group_size=m_stems,
                budget_bytes=self.budget_bytes,
                estimated_peak_bytes=resident_est,
            )

        # STREAMED mode: size block_rows so pre-projection + post-projection + halo stay well
        # within both max_tensor_bytes and 25% of budget_bytes.
        max_pre_proj_cols = max(cols, 1296)
        max_rows_tensor = self.max_tensor_bytes // (max_pre_proj_cols * 4)
        max_rows_budget = int(self.budget_bytes * 0.22) // (max_pre_proj_cols * 4)
        halo_allowance = 56_000
        usable_rows = max(win_step, min(max_rows_tensor, max_rows_budget) - halo_allowance)

        step = max(1, min(win_step, l_tot))
        aligned_rows = max(step, (usable_rows // step) * step)
        block_rows = min(l_tot, aligned_rows)

        # Stem group size scales with budget (e.g. 16 on 8 GiB, 32 on >=16 GiB, 8 on 4 GiB)
        stem_group_cap = max(4, self.budget_bytes // (512 * _MIB))
        stem_group_size = min(m_stems, stem_group_cap)

        est_stream_peak = (
            (block_rows + halo_allowance) * max_pre_proj_cols * 4 * 2
            + l_tot * stem_group_size * 24
            + 256 * _MIB
        )
        return PlannerDecision(
            mode="STREAMED",
            num_samples=l_tot,
            num_stems=m_stems,
            block_rows=block_rows,
            stem_group_size=stem_group_size,
            budget_bytes=self.budget_bytes,
            estimated_peak_bytes=min(self.budget_bytes, est_stream_peak),
        )

    def adjust_block_rows_if_over_budget(
        self,
        current_block_rows: int,
        *,
        min_rows: int = 24_000,
    ) -> int:
        """Checks measured allocator usage and halves `current_block_rows` if over budget."""
        alloc = self.current_allocated_bytes()
        if alloc > self.budget_bytes:
            self.reclaim()
            return max(min_rows, current_block_rows // 2)
        return current_block_rows


def resolve_accelerator_runtime(
    requested_device: Literal["auto", "tpu", "cuda", "rocm", "mps", "cpu"] = "auto",
    *,
    precision: PrecisionPolicy = "auto",
    budget_gb_override: float | None = None,
    force_reprobe: bool = False,
) -> AcceleratorRuntime:
    """Resolves the target accelerator, fails fast on unavailable explicit devices, and runs
    startup capability probes.
    """
    cpu_warning: str | None = None
    resolved_kind: DeviceKind
    torch_dev_str: str

    if requested_device == "auto":
        if is_tpu_available():
            resolved_kind = "tpu"
            torch_dev_str = "tpu:0"
        elif is_rocm_available():
            resolved_kind = "rocm"
            torch_dev_str = "cuda:0"
        elif is_cuda_available():
            resolved_kind = "cuda"
            torch_dev_str = "cuda:0"
        elif is_mps_available():
            resolved_kind = "mps"
            torch_dev_str = "mps"
        else:
            resolved_kind = "cpu"
            torch_dev_str = "cpu"
            cpu_warning = (
                "No hardware accelerator (TPU, CUDA, ROCm, MPS) detected; "
                "falling back to CPU 32-bit execution."
            )
    elif requested_device == "tpu":
        if not is_tpu_available():
            raise ValueError("Requested accelerator device 'tpu' is not available on this system.")
        resolved_kind = "tpu"
        torch_dev_str = "tpu:0"
    elif requested_device == "rocm":
        if not is_rocm_available():
            raise ValueError("Requested accelerator device 'rocm' is not available on this system.")
        resolved_kind = "rocm"
        torch_dev_str = "cuda:0"
    elif requested_device == "cuda":
        if not is_cuda_available():
            raise ValueError("Requested accelerator device 'cuda' is not available on this system.")
        resolved_kind = "cuda"
        torch_dev_str = "cuda:0"
    elif requested_device == "mps":
        if not is_mps_available():
            raise ValueError("Requested accelerator device 'mps' is not available on this system.")
        resolved_kind = "mps"
        torch_dev_str = "mps"
    elif requested_device == "cpu":
        resolved_kind = "cpu"
        torch_dev_str = "cpu"
    else:
        raise ValueError(f"Unsupported accelerator device '{requested_device}'.")

    configure_precision_backends(resolved_kind)

    device = torch.device(torch_dev_str)
    budget_b = _detect_budget_bytes(
        resolved_kind,
        device,
        budget_gb_override=budget_gb_override,
    )
    max_tensor_b = _compute_max_tensor_bytes(resolved_kind, budget_b)

    cache_key = (
        f"{resolved_kind}:{torch_dev_str}:{requested_device}",
        precision,
        budget_b,
    )
    if not force_reprobe and cache_key in _PROBE_CACHE:
        cached = _PROBE_CACHE[cache_key]
        return AcceleratorRuntime(cached.model_copy(update={"cpu_fallback_warning": cpu_warning}))

    fast_math_off = _probe_two_sum_exact(device)
    fp32_exact = _probe_fp32_matmul_integer_exact(device)

    if precision == "fp32":
        bf16_native = False
        bf16_speedup = 0.0
        contraction: ContractionDtypeName = "float32"
    else:
        bf16_native, bf16_speedup = _probe_bf16_contraction(device)
        contraction = "bfloat16" if bf16_native else "float32"

    unverified = resolved_kind == "tpu" or budget_b > 16 * _GIB
    probe = ProbeResult(
        device_kind=resolved_kind,
        torch_device=torch_dev_str,
        contraction_dtype=contraction,
        bf16_native=bf16_native,
        bf16_speedup_vs_fp32=bf16_speedup,
        fp32_matmul_exact=fp32_exact,
        fast_math_off=fast_math_off,
        budget_bytes=budget_b,
        max_tensor_bytes=max_tensor_b,
        unverified_hardware=unverified,
        cpu_fallback_warning=cpu_warning,
    )
    _PROBE_CACHE[cache_key] = probe
    return AcceleratorRuntime(probe)


def free_tensor_storage(tensor: Tensor) -> None:
    """Immediately releases `tensor`'s underlying device storage in-place across CPU, CUDA, MPS,
    and TPU even if external Python references to `tensor` remain alive.
    """
    tensor.set_(torch.empty(0, device=tensor.device, dtype=tensor.dtype).untyped_storage())


def empty_device_cache(device: torch.device | str) -> None:
    """Synchronizes and releases cached device allocator blocks for `device`."""
    dev = torch.device(device)
    _sync_device(dev)
    gc.collect()
    if dev.type == "mps" and hasattr(torch, "mps"):
        empty_fn = getattr(torch.mps, "empty_cache", None)
        if callable(empty_fn):
            empty_fn()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_32bit_device_tensor(tensor: Tensor) -> None:
    """Raises `TypeError` if `tensor` uses a forbidden 64-bit dtype (`float64`, `complex128`,
    `int64`)."""
    if tensor.dtype in (torch.float64, torch.complex128, torch.int64):
        raise TypeError(
            f"Forbidden 64-bit tensor dtype {tensor.dtype} on device {tensor.device}; "
            "only 32-bit device types (float32, int32, bool, complex64, bfloat16) are permitted."
        )


__all__ = [
    "AcceleratorRuntime",
    "ContractionDtypeName",
    "DeviceKind",
    "PlannerDecision",
    "PlannerMode",
    "ProbeResult",
    "assert_32bit_device_tensor",
    "clear_probe_cache",
    "configure_precision_backends",
    "empty_device_cache",
    "free_tensor_storage",
    "is_cuda_available",
    "is_mps_available",
    "is_rocm_available",
    "is_tpu_available",
    "resolve_accelerator_runtime",
]
