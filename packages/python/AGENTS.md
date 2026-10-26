# `packages/python` — Python 3.14 & `PyO3` Engine Routing Guide

**`dnam-engine`** wraps `packages/engine` in a zero-copy `PyO3` (`abi3-py314`) + `rust-numpy` extension (`rust/`) and an idiomatic Python 3.14 interface (`src/dnam_engine/`).

## 1. Key Modules

- [`rust/`](rust/lib.rs) & [`_native.pyi`](src/dnam_engine/_native.pyi): Native `dnam_engine._native` FFI bindings ([`engine.rs`](rust/engine.rs), [`container.rs`](rust/container.rs), [`dsp.rs`](rust/dsp.rs), [`provenance.rs`](rust/provenance.rs)) and PEP 561 type stub.
- [`engine.py`](src/dnam_engine/engine.py): `DnamEngine` real-time processor (`pedalboard`-compatible `__call__`, `process`, `process_planar`, `reload`, `telemetry`).
- [`container.py`](src/dnam_engine/container.py): `DnamContainer`, `GramMatrix` (`uplo='U'`), and `crc32c`.
- [`control.py`](src/dnam_engine/control.py): `ControlSpace`, `ControlRack`, `LobattoBasis`, `Rank4Fm`, `chebyshev_lfo`, and `trust_region_softclip`.
- [`dsp.py`](src/dnam_engine/dsp.py): Stateful Reference DSP (`PolyAdaa`, `SelectiveRgLru`, `ModulatedUfdn`, `lagrange_weights`, `unit_rotor`, `givens_householder_4x4`).
- [`provenance.py`](src/dnam_engine/provenance.py), [`schema/`](src/dnam_engine/schema/__init__.py) & [`exceptions.py`](src/dnam_engine/exceptions.py): Cryptographic provenance/watermarking, `StrictModel` Pydantic v2 schemas, and `DnamError` hierarchy.
- [`tests/`](tests/): Unit, Hypothesis property, and `pytest-benchmark` suites (`test_*.py`, [`test_benchmarks.py`](tests/test_benchmarks.py)).

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Dual-layer FFI architecture, strict `pyrefly` typing, Pydantic `_construct_trusted` hot-path rules, and `moon check python` verification workflow.
2. [`README.md`](README.md) & [`monorepo_implementation_guide.md`](../../docs/monorepo_implementation_guide.md) — Python API reference and specification.
