---
id: ADR-0012
title: "Idiomatic Python SDK Architecture, Domain-Free Naming, Thin Native Shim & Schema Parity"
status: accepted
date: "2026-11-04"
supersedes: null
superseded_by: null
specs:
  - "../specs/sdks/python_sdk.md"
  - "../specs/architecture/tora_architecture.md"
  - "../specs/containers/dnam_binary_format.md"
requirements:
  - REQ-4.1
  - REQ-5.2
  - REQ-6.1
  - REQ-7.1
  - REQ-8.8
---

# ADR-0012: Idiomatic Python SDK Architecture, Domain-Free Naming, Thin Native Shim & Schema Parity

This record documents the architectural and ergonomic design decisions for **`dnam`** (`packages/python/`), the idiomatic Python SDK and `PyO3` + `rust-numpy` extension wrapping `packages/engine/`, specified normatively in [`../specs/sdks/python_sdk.md`](../specs/sdks/python_sdk.md) (and extended to Python 3.15 `3.15t` free-threading and PEP 803 `cp315-abi3.abi3t` dual-ABI wheels in [`ADR-0014`](0014-python-3-15-free-threading-and-stable-abi-abi3t.md)).

---

## 1. Context & Problem Statement

Following the TORA v2.0 engine implementation ([`ADR-0002`](0002-tri-operator-recurrent-architecture.md), [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md)), the initial Python package (`dnam_engine`) exhibited five structural and maintenance liabilities:

1. **Redundant Namespace & Prefixing (`dnam_engine.DnamEngine`, `dnam_engine.DnamContainer`):**
   Repeating `dnam` in both the module import path (`import dnam_engine`) and every class identifier (`DnamEngine`, `DnamContainer`, `DnamTelemetrySnapshot`, `DnamPreambleInfo`) obscured the domain separation between a deployable `.dnam` artifact (`Model`), a stateful real-time processor (`Engine`), and an iterative `.dnam-ckpt` training artifact (`Checkpoint`).
2. **Mixed-Responsibility Flat Modules:**
   Placing live engine control racks, offline control-space clamping, and low-level Lobatto/FM kernels inside a single flat `control.py` module—and duplicating control clamping inside `DnamContainer`—blurred the boundary between container inspection, real-time state mutation, and reference DSP evaluation.
3. **Duplicated DSP / ZOH Logic in the PyO3 FFI Layer (`ref[sdk.python.engine.zero-copy-and-gil]`):**
   Recomputing `ZohRescaledParams` and control-decimation strides inside `packages/python/rust/` duplicated invariants already owned by `DnamEngine` (`ref[dsp.zoh.ctrl-stride]`), causing immediate compile-time and behavioral drift whenever the Rust engine's internal rescaling contract evolved.
4. **Silent Schema Drift Between Rust `serde` and Python `Pydantic` (`ref[sdk.python.schema.rust-parity]`):**
   Maintaining hand-written Pydantic models without automated Rust-to-Python round-trip verification allowed the Python schema to retain superseded Tri-Physics fields (`Core1Config`, `Core2Config`, `SelectiveRgLruParams`, `ModulatedUfdn`) while missing TORA Stage 0–IV configurations (`RouterConfig`, `Stage1PreFxConfig`, `Stage2CascadeConfig`, `Stage3CabinetKautzConfig`, `Stage4SpatialTankConfig`).
5. **Incomplete Runtime Dispatch on `DynamicDnamEngine`:**
   Because `DynamicDnamEngine` lacked forwarding methods for `supports_freeze`, `set_bypass`, `set_stereo_width`, `configure_host_mix_and_leveling`, `sync_tempo`, `register_tap_tempo`, `apply_preset`, and `process_block_in_place`, the Python layer could not expose the full TORA runtime UX without reimplementing engine state transitions in Python.

---

## 2. Considered Options & Rejected Candidates

| Candidate Approach | Why It Fails | Quantified Failure / Maintenance Penalty |
| :--- | :--- | :--- |
| **Thin C-Style Binding Mirror (`import dnam_engine` with Flat 1:1 Function Exports)** | Exposes raw FFI functions and index-addressed control setters (`set_control_value(idx, val)`) directly to Python callers instead of idiomatic Python protocols (`MutableMapping`, context managers, `os.PathLike`). | Forces every caller (`packages/trainer`, notebooks, automation scripts) to manually map control names, restore bypass/freeze states on exceptions, and coerce NumPy memory layouts. |
| **Duplicating ZOH, Preset, or Clamping Math Inside `packages/python/rust/` (`ref[sdk.python.engine.zero-copy-and-gil]`)** | Maintains a second copy of sample-rate rescaling, preset default restoration, or switch crossfade logic in the PyO3 wrapper crate. | Guarantees behavioral divergence from `packages/engine` and breaks bit-identical parity between Python `engine(audio, sample_rate)` and native Rust `DynamicDnamEngine::process_block`. |
| **Untested or Dict-Only JSON Header Wrappers (`ref[sdk.python.schema.rust-parity]`)** | Returning raw `dict[str, Any]` from `model.header` sacrifices static typing (`ref[repo.code.type-safety-and-imports]`), whereas unverified Pydantic models drift from Rust `DnamJsonHeader`. | Violates the zero-`any` quality gate or causes runtime deserialization failures when loading valid `.dnam` v2 containers. |
| **Retaining Legacy Tri-Physics Classes (`SelectiveRgLru`, `ModulatedUfdn`) in `dnam.dsp`** | Preserves dead wrapper code for an architecture superseded by [`ADR-0002`](0002-tri-operator-recurrent-architecture.md). | Violates the clean TORA operator taxonomy (`Op-T`, `Op-M`, `Op-N`) and misleads downstream consumers (`packages/trainer` is scheduled for a dedicated TORA migration). |

---

## 3. Decision Outcome

We establish the **`dnam` v2 Python SDK architecture** ([`../specs/sdks/python_sdk.md`](../specs/sdks/python_sdk.md)):

1. **Task-Oriented `import dnam` Namespace & Domain-Free Core Types (`ref[sdk.python.layout.package-namespace]`, `ref[sdk.python.api.core-types]`):**
   - Rename the Python package and import root to `dnam`, exposing `dnam.Model` (`.dnam` artifact), `dnam.Engine` (stateful real-time processor), and `dnam.Checkpoint` (`.dnam-ckpt` training artifact) at the top level alongside `Tier`, `Backend`, and `errors`.
   - Organize the package into six domain subpackages (`dnam.model`, `dnam.engine`, `dnam.controls`, `dnam.schema`, `dnam.provenance`, `dnam.dsp`) mirrored 1:1 by `packages/python/rust/` (using the modern Rust `foo.rs` + `foo/bar.rs` module layout) and `packages/python/tests/`.
2. **Facet-Oriented `Engine` UX & Context Managers (`ref[sdk.python.api.engine-facets]`):**
   - Expose controls (`engine.controls` implementing `MutableMapping[str, float | str]` with `.snap(...)`, `.reset()`, and `.clamped`), presets (`engine.presets` implementing `Mapping[str, PresetDef]` with `.apply(preset_id)`), tempo (`engine.tempo.sync(...)` and `engine.tempo.tap(...)`), and telemetry (`engine.telemetry`) as structured facets on `Engine`.
   - Provide exception-safe context managers `with engine.bypassed(trails=True):` and `with engine.frozen():` that restore the prior bypass or freeze state on block exit.
3. **Complete `DynamicDnamEngine` Forwarding & Thin Zero-Copy Native Shim (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.python.engine.pedalboard-call]`):**
   - Forward all runtime UX and in-place processing methods on `dnam_engine::DynamicDnamEngine` so `packages/python/rust/engine.rs` delegates 100% of DSP, ZOH rescaling, preset batching, and tempo synchronization to `packages/engine`.
   - Release the Python GIL via `py.detach(...)` across all block audio processing (`process`, `process_planar`, `process_into`, `bounce`, `__call__`) and guard concurrent access safely.
4. **File-for-File `StrictModel` Schema Mirror Verified by JCS Round-Trip Parity (`ref[sdk.python.schema.rust-parity]`):**
   - Mirror `packages/engine/src/schema/` in `dnam.schema` using frozen slotted `@dataclass(frozen=True, slots=True)` `StrictModel` classes (`.replace(...)`, `.to_dict()`, `.to_json()`, `.from_dict()`, `.from_json()`, and `_construct_trusted` hot-path helpers) delegating header validation, calibration math, and RFC 8785 JCS normalization to `dnam-engine` (`serde_json`) with zero runtime `pydantic` dependency.
   - Verify exact schema parity without golden binary files by having a test-gated native helper emit both `DnamJsonHeader::default_reference()` and a fully populated `DnamJsonHeader` (exercising every optional field and enum variant) and asserting byte-identical RFC 8785 JCS serialization after Python round-tripping.
5. **Removal of Legacy Tri-Physics Surface (`ref[sdk.python.api.core-types]`, `ref[sdk.python.dsp.reference-kernels]`):**
   - Replace `SelectiveRgLru`, `ModulatedUfdn`, `Core1Config`, and `Core2Config` with vectorized slice bindings for the canonical TORA reference kernels (`dnam.dsp.op_t`, `dnam.dsp.op_m`, `dnam.dsp.op_n`, `dnam.dsp.control_bus`, `dnam.dsp.mixing`). `packages/trainer` is updated only at the workspace dependency-name level in `pyproject.toml` and will migrate its DSP imports in a dedicated trainer task.
