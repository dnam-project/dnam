# Contributing to `dnam` Python SDK (`packages/python/`)

This guide covers architecture, layout conventions, PEP 803 dual-ABI (`cp315-abi3.abi3t`) free-threading rules, FFI zero-copy rules, static typing (`pyrefly`), parity-first testing, and PEP 799 (`profiling.sampling` / `tracemalloc`) performance profiling standards for **`dnam`** ([`README.md`](README.md)), specified in [`docs/specs/sdks/python_sdk.md`](../../docs/specs/sdks/python_sdk.md), [`ADR-0012`](../../docs/adr/0012-python-sdk-architecture-and-idiomatic-api.md), and [`ADR-0014`](../../docs/adr/0014-python-3-15-free-threading-and-stable-abi-abi3t.md).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Mirrored Package Architecture (`src/dnam/`, `rust/`, `tests/`)

Every domain subpackage in `src/dnam/` is mirrored 1:1 by a native module in `rust/` (using the modern Rust 2018+ `foo.rs` + `foo/bar.rs` layout with zero `mod.rs` files) and a test directory in `tests/`:

```text
packages/python/
├── src/dnam/
│   ├── __init__.py          # Curated top-level exports (Model, Engine, Checkpoint, Tier, Backend)
│   ├── _native.pyi          # PEP 561 static type stub kept in lockstep with rust/
│   ├── errors.py            # Typed DnamError hierarchy raised directly by rust/errors.rs
│   ├── _arrays.py           # Zero-copy NumPy float32/float64 coercion helpers
│   ├── model/               # Model, Checkpoint, Preamble, GramMatrix, crc32c
│   ├── engine/              # Engine + Controls, Presets, Tempo, Telemetry facets
│   ├── controls/            # ControlSpace, ClampResult, ZohParams, tempo inversion
│   ├── schema/              # File-for-file Pydantic v2 StrictModel mirror of engine/src/schema/
│   ├── provenance/          # Verify, Sign, Seal (DNSL), Marketplace (DAU1/DME1), Watermark
│   └── dsp/                 # Vectorized TORA reference kernels (op_t, op_m, op_n, control_bus, mixing)
├── rust/
│   ├── lib.rs, errors.rs, arrays.rs, testing.rs
│   ├── engine.rs & engine/{process.rs, runtime.rs, telemetry.rs}
│   ├── model.rs & model/{checkpoint.rs, gram.rs}
│   ├── controls.rs
│   ├── provenance.rs & provenance/{verify.rs, signing.rs, sealing.rs, marketplace.rs, watermark.rs}
│   └── dsp.rs
└── tests/
    ├── conftest.py, test_public_api.py
    ├── strategies/          # Hypothesis strategies split by domain
    ├── model/, engine/, controls/, schema/, provenance/, dsp/
    └── benchmarks/          # pytest-benchmark suite (test_benchmarks.py) & PEP 799 workload (profile_sdk.py)
└── examples/                # 5 runnable SDK demonstrations (realtime_audio_and_bounce.py,
                             # controls_presets_and_tempo.py, model_slimming_and_checkpoint.py,
                             # marketplace_provenance_and_watermark.py, reference_dsp_operators.py)
```

### Test, Benchmark & Example Placement

- **Unit, Property & Parity Tests (`tests/<subpackage>/`):** Place subpackage tests in [`tests/model/`](tests/model/), [`tests/engine/`](tests/engine/), [`tests/controls/`](tests/controls/), [`tests/schema/`](tests/schema/), [`tests/provenance/`](tests/provenance/), and [`tests/dsp/`](tests/dsp/), using Hypothesis strategies from [`tests/strategies/`](tests/strategies/).
- **Benchmarks & Profiling (`tests/benchmarks/`):** Place `pytest-benchmark` suites in [`tests/benchmarks/test_benchmarks.py`](tests/benchmarks/test_benchmarks.py) and the standalone PEP 799 (`profiling.sampling` / `profiling.tracing`) workload in [`tests/benchmarks/profile_sdk.py`](tests/benchmarks/profile_sdk.py).
- **Runnable Examples (`examples/`):** Place self-contained runnable API demonstrations in [`examples/realtime_audio_and_bounce.py`](examples/realtime_audio_and_bounce.py), [`examples/controls_presets_and_tempo.py`](examples/controls_presets_and_tempo.py), [`examples/model_slimming_and_checkpoint.py`](examples/model_slimming_and_checkpoint.py), [`examples/marketplace_provenance_and_watermark.py`](examples/marketplace_provenance_and_watermark.py), and [`examples/reference_dsp_operators.py`](examples/reference_dsp_operators.py) (smoke-tested automatically by [`tests/test_public_api.py`](tests/test_public_api.py)).

---

## 2. Core Engineering & Hot-Path Performance Rules

1. **Private `_`-Prefixed Implementation Modules & Curated `__all__` (`ref[sdk.python.layout.package-namespace]`, `ref[sdk.python.layout.curated-exports]`):**
   - Put implementation details inside `_`-prefixed modules (`_model.py`, `_engine.py`) and re-export the curated public surface in `__init__.py` via an explicit `__all__`.
   - Never re-export `dnam._native` or `native_*` symbols in any public `__all__`.
2. **Thin Native Shim & Direct Exception Instantiation (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.errors.typed-hierarchy]`):**
   - `rust/engine.rs` wraps `dnam_engine::DynamicDnamEngine<'static>` and delegates 100% of DSP, ZOH sample-rate rescaling, preset batching, and tempo synchronization to `packages/engine`.
   - `rust/errors.rs` instantiates `dnam.errors.{ContainerError, ValidationError, SampleRateError, ControlError, ProvenanceError}` directly on the cold error path so Python hot paths invoke `dnam._native` functions directly without Python wrapper frames or `*args` tuple allocations.
   - Always release the Python GIL (`py.detach(|| ...)`) inside block audio processing and offline bounce methods, and intern recurring `PyDict` keys in Rust with `pyo3::intern!`.
3. **Zero-Copy NumPy Coercion & Fast Overlap Checks (`ref[sdk.python.api.flexible-in-precise-out]`):**
   - Check `type(arr) is np.ndarray and arr.dtype == _F32_DTYPE and arr.flags.c_contiguous` before calling `np.asarray`.
   - Avoid calling `np.shares_memory(a, b)` when `a is not b and a.base is None and b.base is None` (two distinct base arrays cannot overlap).
4. **Strict Static Typing (`pyrefly`, Zero Suppressions) & Fast `StrictModel` Construction:**
   - Update [`src/dnam/_native.pyi`](src/dnam/_native.pyi) in lockstep with any change in `rust/`.
   - Never use `# type: ignore` or `# pyrefly: ignore` (`ref[repo.code.type-safety-and-imports]`).
   - Keep `StrictModel.__pydantic_custom_init__ = False` so `DnamJsonHeader.from_json` validates nested models directly inside Rust `pydantic_core`, and use `StrictModel._construct_trusted_dict(...)` (reusing the per-class `_trusted_fields_set`) when constructing models from pre-validated Rust dictionaries on hot paths (`Engine.telemetry`, `ControlSpace.clamp`, `Model.from_bytes`).
5. **Parity-First Verification Without Golden Binaries:**
   - Verify audio rendering against `dnam._native._testing.render_reference_*`, schema serialization against `dnam._native._testing.reference_header_json`, and exception mapping against `dnam._native._testing.error_variant_names`.
   - Cite the verified `r[sdk.python.*]` rules in test docstrings via `verify[sdk.python.*]`.

---

## 3. Testing, Coverage (≥ 95%), Benchmarking & PEP 799 Profiling

Run all commands from the monorepo root via `moon`:

```bash
# 1. Run all python quality gates (format-check, lint, typecheck, build, test) & dual-ABI wheel check
moon check python
moon run python:wheel-check

# 2. Run the exhaustive cross-product parity matrix (@pytest.mark.slow)
moon run python:test-full

# 3. Check the >= 95% line + branch coverage floor
moon run python:coverage

# 4. Run calibrated pytest-benchmark suite (enforces <= 1.10x native FFI budget)
moon run python:bench

# 5. Profile Python vs. native Rust CPU & call stacks with stdlib PEP 799 (profiling.sampling / tracing)
moon run python:profile
uv run --no-sync python -m profiling.sampling run --native --mode cpu --flamegraph \
  -o target/profile_sdk.html packages/python/tests/benchmarks/profile_sdk.py
```

> **Note on `profiling.sampling` (`Tachyon`) & `tracemalloc`:** On macOS and Windows, `python -m profiling.sampling` reads target process memory and may require elevated privileges or developer entitlements (`moon run python:profile` automatically falls back to `python -m profiling.tracing`). Steady-state Python heap zero-allocation checks run inside `pytest` via stdlib `tracemalloc`, while native Rust zero-allocation checks run via `hotpath-alloc`.

---

## 4. Pre-Commit Checklist for `packages/python/`

- [ ] `moon check python` passes all Ruff, Clippy (`-D warnings`), Pyrefly (`0` diagnostics with `src/dnam/_native.pyi` in lockstep), build, and unit/Hypothesis parity tests.
- [ ] `moon run python:coverage` confirms ≥ 95% line and branch coverage.
- [ ] `moon run python:bench` and `moon run python:profile` confirm `Engine.process_into` stays within `≤ 1.10×` of native `PyEngine.process_stereo_2d` with zero unexpected Python heap allocations on hot paths.
