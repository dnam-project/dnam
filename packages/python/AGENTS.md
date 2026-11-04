# `packages/python` — `dnam` Python 3.14 SDK & `PyO3` Routing Guide

**`dnam`** wraps `packages/engine` in a zero-copy `PyO3` (`abi3-py314`) + `rust-numpy` shim (`rust/`) and an idiomatic Python 3.14 SDK (`src/dnam/`).

## 1. Submodule Discovery Map

| Subpackage | Python (`src/dnam/`) & Native (`rust/`) | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Errors** | [`__init__.py`](src/dnam/__init__.py), [`errors.py`](src/dnam/errors.py), [`_native.pyi`](src/dnam/_native.pyi), [`rust/lib.rs`](rust/lib.rs) | [`python_sdk.md`](../../docs/specs/sdks/python_sdk.md) |
| **`dnam.model`** | [`model/`](src/dnam/model/__init__.py) (`Model`, `Checkpoint`, `Preamble`, `GramMatrix`), [`rust/model.rs`](rust/model.rs) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md), [`checkpoint_and_cache_format.md`](../../docs/specs/containers/checkpoint_and_cache_format.md) |
| **`dnam.engine`** | [`engine/`](src/dnam/engine/__init__.py) (`Engine`, `Controls`, `Presets`, `Tempo`, `Telemetry`), [`rust/engine.rs`](rust/engine.rs) | [`tora_architecture.md`](../../docs/specs/architecture/tora_architecture.md), [`router_and_stages_topology.md`](../../docs/specs/dsp/router_and_stages_topology.md) |
| **`dnam.controls`** | [`controls/`](src/dnam/controls/__init__.py) (`ControlSpace`, `ClampResult`, `TempoSubdivision`), [`rust/controls.rs`](rust/controls.rs) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **`dnam.schema`** | [`schema/`](src/dnam/schema/__init__.py) (`DnamJsonHeader`, `RouterConfig`, `Stage1..4*Config`) | [`python_sdk.md`](../../docs/specs/sdks/python_sdk.md) |
| **`dnam.provenance`** | [`provenance/`](src/dnam/provenance/__init__.py) (`verify_model`, `seal_model`, `Dau1Manifest`, Watermark), [`rust/provenance.rs`](rust/provenance.rs) | [`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../docs/specs/security/watermarking_and_marketplace.md) |
| **`dnam.dsp`** | [`dsp/`](src/dnam/dsp/__init__.py) (`op_t`, `op_m`, `op_n`, `control_bus`, `mixing`), [`rust/dsp.rs`](rust/dsp.rs) | [`python_sdk.md`](../../docs/specs/sdks/python_sdk.md) |

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) & [`README.md`](README.md) — Architecture, typing, and `moon check python`.
2. [`ADR-0012`](../../docs/adr/0012-python-sdk-architecture-and-idiomatic-api.md) — SDK design rationale.
