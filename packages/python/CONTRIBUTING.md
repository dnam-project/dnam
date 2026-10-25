# Contributing to `dnam-engine` Python Package (`packages/python/`)

This guide covers engineering, FFI, typing, testing, coverage, and performance benchmarking standards for **`dnam-engine`** ([`README.md`](README.md)), the idiomatic Python 3.14 interface and `PyO3` (`abi3-py314`) + `rust-numpy` extension wrapping `packages/engine`.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Dual-Layer Package Architecture

Consult [`AGENTS.md`](AGENTS.md) for the module-to-spec discovery table:

1. **Native PyO3 / `rust-numpy` FFI Layer (`rust/` -> `dnam_engine._native`):**
   - [`rust/lib.rs`](rust/lib.rs): Module registration for `dnam_engine._native`.
   - [`rust/engine.rs`](rust/engine.rs): `NativeDnamEngine` wrapping `DynamicDnamEngine<'static>` with zero-copy 1D planar and 2D `(2, N)` C-contiguous slice methods (`process_stereo_2d`, `process_mono_to_stereo_2d`, `process_stereo`, `process_stereo_in_place`, `process_mono`) that release the Python GIL via `py.detach(...)`.
   - [`rust/container.rs`](rust/container.rs): `NativeControlSpace` pre-compiled clamper, `.dnam` / `.dnam-ckpt` container builders/validators, and `uplo='U'` Gram matrix packing.
   - [`rust/dsp.rs`](rust/dsp.rs): Scalar and vectorized slice bindings for `PolyADAA`, `SelectiveRgLru`, `ModulatedUfdn`, `LobattoBasis`, `Rank4Fm`, `chebyshev_lfo`, `trust_region_softclip`, `unit_rotor`, and `lagrange_weights`.
   - [`rust/provenance.rs`](rust/provenance.rs): JCS SHA-256, `Ed25519` signing messages, `ChaCha20` arena encryption, and Buyer Watermarking (`Test 8H`).
2. **Idiomatic Python 3.14 Interface (`src/dnam_engine/`):**
   - [`src/dnam_engine/engine.py`](src/dnam_engine/engine.py): `DnamEngine` (with `Spotify pedalboard` `__call__(audio, sample_rate, buffer_size=..., reset=...)` compatibility, zero-allocation `out=` buffer support, and `process_planar`).
   - [`src/dnam_engine/container.py`](src/dnam_engine/container.py): `DnamContainer`, `GramMatrix`, and `crc32c`.
   - [`src/dnam_engine/control.py`](src/dnam_engine/control.py): `ControlSpace`, `ControlRack` (`MutableMapping[str, float | str]`), `LobattoBasis`, `Rank4Fm`, `chebyshev_lfo`, and `trust_region_softclip`.
   - [`src/dnam_engine/dsp.py`](src/dnam_engine/dsp.py): Stateful callable `PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `lagrange_weights`, `unit_rotor`, and `givens_householder_4x4`.
   - [`src/dnam_engine/provenance.py`](src/dnam_engine/provenance.py): Cryptographic provenance and watermarking helpers.
   - [`src/dnam_engine/schema/`](src/dnam_engine/schema/__init__.py) & [`src/dnam_engine/exceptions.py`](src/dnam_engine/exceptions.py): Pydantic v2 `StrictModel` hierarchy and typed `DnamError` exceptions.
   - [`src/dnam_engine/_native.pyi`](src/dnam_engine/_native.pyi) & [`src/dnam_engine/py.typed`](src/dnam_engine/py.typed): PEP 561 stub file for strict static type checking across the FFI boundary.

---

## 2. Development Setup & Editable Release Builds

`packages/python/pyproject.toml` sets `editable-profile = "release"` under `[tool.maturin]`, so `uv sync` and `uv run` automatically compile `dnam_engine._native` with `--release` optimizations:

```bash
# Sync dependencies and build the optimized editable wheel
uv sync

# Or explicitly recompile the PyO3 extension via moon
pnpm moon run python:build
```

---

## 3. Coding, Typing (`pyrefly`), Pydantic & FFI Performance Rules

1. **Strict Static Typing (`pyrefly`, `preset = "strict"`):**
   - Target Python 3.14 (`py314`). Every function, method, property, pytest fixture, and test helper must have explicit type annotations.
   - Never disable rules, downscale presets, or use `# type: ignore`.
   - Whenever you add or modify a `#[pyfunction]` or `#[pymethods]` signature in `rust/*.rs`, update [`src/dnam_engine/_native.pyi`](src/dnam_engine/_native.pyi) in lockstep.
2. **Code Formatting & Linting (`ruff` + `clippy`):**
   - Enforce `line-length = 100` and pass `pnpm moon run python:format-check` and `pnpm moon run python:lint` with zero warnings.
3. **Zero Per-Element Python Loops Over Audio or Control Arrays:**
   - Never loop over 1D or 2D NumPy arrays in Python to invoke scalar FFI functions. Always pass C-contiguous NumPy arrays into a slice-based `#[pyfunction]` in `rust/*.rs`.
   - Release the GIL (`py.detach(|| ...)`) in `rust/engine.rs` during block audio processing.
4. **Fast Pydantic v2 Construction on Hot Paths:**
   - Use `StrictModel` (`frozen=True, extra="forbid"`) for all user-supplied configuration and header validation.
   - When constructing `StrictModel` instances from pre-validated Rust return values inside hot paths (`ControlSpace.clamp`, `DnamEngine.telemetry`, `PolyAdaa.process`, `SelectiveRgLru.process`, `DnamContainer.with_header`), use `StrictModel._construct_trusted(...)` and cache converted NumPy arrays in `model_post_init` (as in `WatermarkParams.cached_f32_arrays()`).

---

## 4. Testing (`pytest` + `hypothesis`), Coverage ($\ge 95\%$) & Benchmarks (`pytest-benchmark`)

All test files reside in [`tests/`](tests/) (`test_schema.py`, `test_container.py`, `test_control.py`, `test_dsp.py`, `test_provenance.py`, `test_engine.py`, and `test_benchmarks.py`):

```bash
# 1. Run all python quality gates (format-check, lint, typecheck, build, test)
pnpm moon check python

# 2. Run unit, property-based (Hypothesis), and smoke benchmark tests with >= 95% coverage
pnpm moon run python:coverage

# 3. Run full calibrated pytest-benchmark suite
pnpm moon run python:bench
```

---

## 5. Pre-Commit Checklist for `packages/python/`

- [ ] `pnpm moon check python` passes all Ruff, Clippy, Pyrefly (`0` diagnostics with `src/dnam_engine/_native.pyi` up to date), build, and test checks.
- [ ] `pnpm moon run python:coverage` passes with $\ge 95\%$ coverage.
- [ ] `pnpm moon run python:bench` confirms no performance regressions.
