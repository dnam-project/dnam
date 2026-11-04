# `dnam` — Idiomatic Python 3.14 SDK for Dynamic Neural Audio Modeler (DNAM)

**`dnam`** provides an idiomatic, strictly typed Python 3.14 SDK and a zero-copy `PyO3` (`abi3-py314`) + `rust-numpy` SIMD runtime wrapping the **Dynamic Neural Audio Modeler (DNAM / TORA)** Rust engine ([`packages/engine/`](../engine/README.md), specified in [`docs/specs/sdks/python_sdk.md`](../../docs/specs/sdks/python_sdk.md) and [`ADR-0012`](../../docs/adr/0012-python-sdk-architecture-and-idiomatic-api.md)).

---

## Quickstart

```python
import numpy as np
import dnam

# 1. Load or synthesize a validated .dnam Model
model = dnam.Model.reference()
assert model.max_tier == dnam.Tier.ULTRA

# 2. Create a stateful real-time Engine and drive its facets
engine = dnam.Engine(model, sample_rate=48_000, tier="full")
engine.controls["gain"] = 7.5
engine.controls.snap(gain=7.5, boost="on")
engine.presets.apply("warm_slapback")
engine.tempo.sync(bpm=120, subdivision="1/8.")

# 3. Render audio (pedalboard-compatible __call__, GIL released)
audio = np.zeros((2, 256), dtype=np.float32)
wet = engine(audio, 48_000)
assert wet.shape == (2, 256) and wet.dtype == np.float32

# 4. Scoped bypass & hardware freeze context managers
with engine.bypassed(trails=True):
    tail = engine(audio, 48_000, reset=False)
with engine.frozen():
    pad = engine(audio, 48_000, reset=False)

# 5. Offline 4x polyphase bounce & .dnam-ckpt Checkpoint export
bounced = engine.bounce(audio)
ckpt = dnam.Checkpoint.reference()
lite_model = ckpt.export_model().with_tier_limit("lite")
assert lite_model.max_tier == dnam.Tier.LITE
```

---

## Package Tour by Domain Subpackage

| Subpackage | Public Surface | Purpose |
| :--- | :--- | :--- |
| **`dnam`** | `Model`, `Engine`, `Checkpoint`, `Tier`, `Backend`, `ControlSpace`, `ClampResult`, `errors` | Task-oriented top-level entry points for 90% of workflows. |
| **`dnam.model`** | `Model`, `Checkpoint`, `Preamble`, `GramMatrix`, `crc32c` | Declarative `.dnam` and `.dnam-ckpt` (`DNCK`) inspection, 4 KB in-place header editing (`with_header`), tier stripping (`with_tier_limit`), runtime export (`export_model`), and `uplo='U'` Gram packing. |
| **`dnam.engine`** | `Engine`, `Controls`, `Presets`, `Tempo`, `Telemetry`, `OverflowFlags`, `Tier`, `Backend` | Real-time SIMD audio processing (`__call__`, `process`, `process_into`, `process_planar`, `bounce`, `reload`) with GIL release (`py.detach`) and live facets. |
| **`dnam.controls`** | `ControlSpace`, `ClampResult`, `ClampedControl`, `TempoSubdivision`, `ZohParams`, `compute_zoh`, `invert_tempo_calibration` | Pre-compiled 3-Phase control clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`), ZOH rescaling, and 32-step binary-search tempo calibration inversion. |
| **`dnam.schema`** | `DnamJsonHeader`, `RouterConfig`, `Stage1..4*Config`, `IoConfig`, `ControlSchemaDef`, `PresetDef`, `ModelSignatures`, ... | File-for-file frozen Pydantic v2 `StrictModel` mirror of `dnam-engine::schema` verified for byte-identical RFC 8785 JCS round-trip parity. |
| **`dnam.provenance`** | `verify_model`, `sign_creator_header`, `sign_marketplace_receipt`, `seal_model`, `unseal_model`, `Dau1Manifest`, `apply_gauge_watermark`, `apply_tardos_watermark`, `score_tardos_watermark`, `extract_structural_hash` | End-to-end v2 provenance, `OemDeviceCertificate` attestation, `.dnam-seal` (`DNSL`) HPKE hardware sealing, `DAU1`/`DME1` framing, preview containment, and dual-mechanism buyer watermarking. |
| **`dnam.dsp`** | `op_t`, `op_m`, `op_n`, `control_bus`, `mixing` | Vectorized slice-based `f64` TORA reference kernels (`catmull_rom_hermite`, `coercive_step`, `tank_slice`, `modal_slice`, `unit_rotor`, `poly_adaa_slice`, `lobatto_basis`, `rank4_fm`, `chebyshev_lfo`, `trust_region_softclip`, `givens_householder_8ch/16ch`). |

---

## Measured Python SDK & FFI Benchmark Summary (`48 kHz` Stereo)

Measured via `moon run python:bench` ([`tests/benchmarks/test_benchmarks.py`](tests/benchmarks/test_benchmarks.py)) and profiled with **Scalene** ([`tests/benchmarks/profile_sdk.py`](tests/benchmarks/profile_sdk.py)):

| Operation & Tier Profile | Backend | Median Latency | Throughput (`ops/s`) | FFI / Native Overhead |
| :--- | :--- | ---: | ---: | ---: |
| **`Engine.process_into` (`Lite`, `128` stereo frames)** | `ARM64 NEON` | **`19.92 µs`** | **`48.9 Kops/s`** | **`1.026x` (`558 ns`)** |
| **`Engine.process_into` (`Full`, `128` stereo frames)** | `ARM64 NEON` | **`22.00 µs`** | **`44.3 Kops/s`** | **`1.026x` (`<= 1.10x` budget)** |
| **`Engine.process_into` (`Pro`, `128` stereo frames)** | `ARM64 NEON` | **`23.08 µs`** | **`42.2 Kops/s`** | **`1.026x`** |
| **`Engine.process_into` (`Ultra`, `128` stereo frames)** | `ARM64 NEON` | **`24.50 µs`** | **`38.5 Kops/s`** | **`1.026x`** |
| **`Engine.process_planar` (`Full`, `1.0 s` / `48,000` frames)** | `ARM64 NEON` | **`8.30 ms`** | **`120.4x` RT** | **`1.054x`** |
| **`Engine.controls['gain']` (`get` / `set` slew / `.clamped`)** | `PyO3` | **`174 ns` / `280 ns` / `43 ns`** | **`5.76 M` / `3.58 M` / `23.4 M`** | `0` dict/list allocs |
| **`ControlSpace.clamp({'gain': 7.5, 'boost': 'on'})`** | `PyO3` | **`3.42 µs`** | **`282.9 Kops/s`** | Pre-compiled `ControlClampDescriptor` |
| **`Engine.telemetry` Snapshot Readout** | `PyO3` | **`1.54 µs`** | **`625.7 Kops/s`** | Interned `PyDict` + zero-copy `__dict__` |
| **`Model.from_bytes` (`68 KB`) / `Model.with_header` (`4 KB`)** | `PyO3` | **`44.8 µs` / `42.0 µs`** | **`22.3 K` / `23.8 Kops/s`** | Lazy header parse & fast in-place CRC |
| **`GramMatrix` `256 x 256` `uplo='U'` Pack + Unpack** | `PyO3` | **`62.75 µs`** | **`15.3 Kops/s`** | Zero-copy `into` slices |

---

## Running Tests, Coverage, Benchmarks & Scalene Profiling

```bash
# 1. Quality gates, unit/property tests, exhaustive matrix & >= 95% coverage
moon check python
moon run python:test
moon run python:test-full
moon run python:coverage

# 2. Calibrated pytest-benchmark suite (enforces <= 1.10x native FFI budget)
moon run python:bench

# 3. CPU, memory & copy-volume profiling with Scalene
moon run python:profile
uv run --no-sync scalene run --profile-all -o scalene-profile.json \
  packages/python/tests/benchmarks/profile_sdk.py
uv run --no-sync scalene view --cli scalene-profile.json

# 4. Runnable API Examples (packages/python/examples/)
uv run --no-sync python packages/python/examples/realtime_audio_and_bounce.py
uv run --no-sync python packages/python/examples/controls_presets_and_tempo.py
uv run --no-sync python packages/python/examples/model_slimming_and_checkpoint.py
uv run --no-sync python packages/python/examples/marketplace_provenance_and_watermark.py
uv run --no-sync python packages/python/examples/reference_dsp_operators.py
```

API demonstrations live in [`examples/`](examples/). See the [contributing guide](CONTRIBUTING.md#3-testing-coverage--95-benchmarking--scalene-profiling) for detailed FFI optimization and profiling rules.
