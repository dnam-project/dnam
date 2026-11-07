---
id: ADR-0014
title: "Python 3.15 Free-Threaded Dev Runtime, PEP 803 Dual-ABI (abi3.abi3t) Wheels & PyO3 Concurrency"
status: accepted
date: "2026-11-06"
supersedes: []
superseded_by: null
specs:
  - "../specs/sdks/python_sdk.md"
requirements:
  - REQ-4.1
  - REQ-6.1
  - REQ-6.3
---

# ADR-0014: Python 3.15 Free-Threaded Dev Runtime, PEP 803 Dual-ABI (`abi3.abi3t`) Wheels & PyO3 Concurrency

This record documents the architectural decision to establish **Python `>=3.15`** (`3.15t` free-threaded development interpreter) as the workspace floor, compile a single **PEP 803 dual-ABI (`cp315-abi3.abi3t`)** native wheel per platform via `PyO3` (`abi3-py315` + `abi3t-py315`) and `maturin >= 1.16`, declare `#[pymodule(gil_used = false)]` on `dnam._native`, and align dependency and profiling toolchains for free-threaded execution, specified normatively in [`../specs/sdks/python_sdk.md`](../specs/sdks/python_sdk.md).

---

## 1. Context & Problem Statement

Following the initial `dnam` Python SDK release on Python 3.14 (`abi3-py314`, [`ADR-0012`](0012-python-sdk-architecture-and-idiomatic-api.md)), four runtime and packaging constraints motivated migrating to Python 3.15 (`3.15t`) and PEP 803 (`abi3t`):

| Previous Formulation | Failure Mechanism | Runtime or Packaging Consequence |
| :--- | :--- | :--- |
| **`abi3-py314` Stable ABI Without `abi3t`** | Python 3.14's legacy `abi3` stable ABI does not support free-threaded (`Py_GIL_DISABLED`) interpreters; PEP 803 introduces `abi3t` starting at Python `3.15` (`Py_ABI_VERSION` `2026.0`). | Loading an `abi3`-only extension on a free-threaded interpreter either fails at import time or forces a per-minor-version (`cp314t`, `cp315t`) wheel matrix. |
| **Implicit GIL Re-Enablement on Import** | Any C/Rust extension that omits `Py_mod_gil = Py_MOD_GIL_NOT_USED` (`#[pymodule(gil_used = false)]`) causes the free-threaded interpreter to re-enable the Global Interpreter Lock globally on `import`. | Importing `dnam` inside a multi-threaded audio batch renderer or `dnam_trainer` worker silently degrades the entire Python process back to serialized GIL execution (`sys._is_gil_enabled() == True`). |
| **Non-`cp315t` Dependencies (`polars`, `scalene`, `crosshair-tool`, AMD `whl-next`)** | `polars-runtime-32` ships `cp311-abi3` wheels only, `scalene` and `crosshair-tool` lack 3.15t wheels, and AMD's `stable.repo.amd.com/rocm/whl-next` index publishes neither `cp315` nor `cp315t` wheels. | Blocks installing the workspace on `3.15t` or re-enables the GIL when importing transitive dependencies. |
| **Unchecked Slice Borrows Under Concurrent Python Threads** | Extracting raw `as_slice()` / `as_slice_mut()` pointers without `rust-numpy`'s borrow tracker allows a second Python thread on `3.15t` to mutate or resize the same `ndarray` concurrently during native DSP execution. | Creates data races (`UB`) reachable purely from safe Python callers under free-threading. |

---

## 2. Decision & Counterexample Matrix

| Mechanism & Spec Rule | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Python `>=3.15` Floor & `3.15t` Free-Threaded Dev Interpreter (`ref[sdk.python.abi.abi3-abi3t]`)** | Pin `.python-version` to `3.15t` managed by `uv >= 0.13.7` (removing `"python"` from `mise.toml` `idiomatic_version_file_enable_tools` because `mise` filters `t`-suffixed versions) and require `requires-python = ">=3.15"` across the workspace. | Keeping `<3.15` compatibility precludes PEP 803 `abi3t`, which has a hard minimum floor of Python `3.15`. |
| **Single PEP 803 Dual-ABI Wheel (`cp315-abi3.abi3t`, `ref[sdk.python.abi.abi3-abi3t]`)** | Enable both `abi3-py315` and `abi3t-py315` features on `pyo3` with `maturin >= 1.16` so one compiled binary (`_native.abi3t.so` / `_native.abi3t.pyd`) loads natively on both GIL-enabled `3.15+` and free-threaded `3.15t+`. | Building separate `cp315-abi3` and `cp315t-abi3t` wheels doubles build artifacts for an identical `>=3.15` floor. |
| **Explicit `#[pymodule(gil_used = false)]` & `py.detach` (`ref[sdk.python.abi.abi3-abi3t]`, `ref[sdk.python.engine.zero-copy-and-gil]`)** | Annotate `dnam._native` with `#[pymodule(gil_used = false)]`, verify `sys._is_gil_enabled() is False` after importing `dnam` and `dnam_trainer` on `3.15t`, and detach heavy native work via `py.detach(|| ...)` so attached threads never stall stop-the-world cyclic GC on `3.15t` or block threads on `3.15`. | Running long container verification, Gram packing, or audio rendering while attached to the Python runtime blocks `3.15t` stop-the-world garbage collection across all threads. |
| **Checked `rust-numpy` Borrow Tracking (`ref[sdk.python.engine.zero-copy-and-gil]`)** | Retain `rust-numpy` 0.30 (`PyReadonlyArray` / `PyReadwriteArray`) checked borrows across all native array entry points. | Bypassing `rust-numpy`'s process-wide borrow table in favor of raw buffer protocol pointers permits concurrent Python thread mutation (`UB`) on `3.15t`. |
| **Free-Threaded Dependency & ROCm 10.0 Alignment** | Replace `polars` telemetry persistence with Pydantic JSONL (`telemetry.jsonl`) + Altair comprehensions; replace `scalene` with Python 3.15 stdlib PEP 799 `profiling.sampling` (`Tachyon`) and `tracemalloc`; switch ROCm to `https://download.pytorch.org/whl/rocm10.0` (`torch 2.15.0+rocm10.0` `cp315t`) with per-GPU extras restricted to trainer-qualified RDNA 3/3.5/4 and CDNA 2/3/4 architectures (`gfx1100`, `gfx1101`, `gfx1102`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx1250`, `gfx90a`, `gfx942`, `gfx950`). | Pulling `rocm[device-all]` downloads 25 device archives (`~9 GB`), including RDNA 1 (`gfx101*`), RDNA 2 (`gfx103*`), CDNA 1 (`gfx908`), and mobile iGPUs that lack native `bfloat16` WMMA/MFMA matrix cores, strict IEEE-754 `fp32`, AOTriton support, or `>= 12 GB` VRAM. |

---

## 3. Consequences & Phased Roadmap

1. **Verified Dual-Runtime Wheel Gate (`moon run python:wheel-check`):**
   - Builds the release wheel under `3.15t`, asserts the `cp315-abi3.abi3t` wheel filename and `_native.abi3t` shared object, installs the wheel into isolated `3.15` (GIL) and `3.15t` (free-threaded) virtual environments, and verifies `sys._is_gil_enabled() is False` on `3.15t`.
2. **Follow-On Concurrency Hardening (`Phase 2` & `Phase 3`):**
   - Phase 2 moves per-engine synchronization into Rust (`#[pyclass(frozen)] PyEngine`) with an atomic Python `_EngineState` reference swap for `reload`.
   - Phase 3 introduces lock-free engine-core SPSC control ingress (`ref[dsp.ctrl.op-m.spsc-ingress]`), wait-free seqlock telemetry, and `dnam_engine::host` shared across Python, Node, and real-time plugin hosts.
