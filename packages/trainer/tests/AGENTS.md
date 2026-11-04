# DNAM Trainer Test Suite Guide (`packages/trainer/tests/`)

1:1 mirrored unit, property-based (`hypothesis`), worker CLI, integration (`Tests 6–8H`), guardrail, and benchmark test suite for `dnam_trainer`.

## 1. Test Directory Map

| Path | Scope |
| :--- | :--- |
| [`test_guardrails.py`](test_guardrails.py) | Trainer-package AST & device-precision guardrails (`ref[repo.code.type-safety-and-imports]`, `ref[train.gram.device-32bit]`, `ref[train.gram.window-streaming]`; see [`tests/AGENTS.md`](../../../tests/AGENTS.md) for monorepo guardrails). |
| [`test_schema_and_events.py`](test_schema_and_events.py) | Pydantic v2 validation (`TrainingStem`, `StemAudioRef`, `TrainingJobRequest`, `CaptureEvent`, `TrainingEvent`). |
| [`dsp/`](dsp/) | `test_poly_adaa.py`, `test_selective_rglru.py`, `test_modulated_ufdn.py`. |
| [`control/`](control/) | `test_basis.py` (Lobatto HDMR & Rank-4 FM), `test_audit.py` (3-Phase audit). |
| [`training/`](training/) | `test_accelerator.py`, `test_hardware.py`, `test_numerics.py`, `test_no_64bit.py`, `test_windowing.py`, `test_calib_cache.py`, `test_quiet_stems.py`, `test_streaming.py`, `test_metrics_and_cascade.py`, `test_surrogate_delay.py` (`Test 6`), `test_kronecker_head.py`, `test_arena.py`, `test_telemetry.py`, `test_pipeline.py`, `test_backends.py`. |
| [`cli/`](cli/) | `test_train_worker_cli.py` (`dnam-train` headless worker across `jsonl`, `sse`, `rich`, `--status-json`, and `.dnam-ckpt` resume). |
| [`integration/test_dnam_verification.py`](integration/test_dnam_verification.py) | End-to-end verification of `Tests 6–8I` from [`four_stage_pipeline.md`](../../../docs/specs/training/four_stage_pipeline.md). |
| [`test_benchmarks.py`](test_benchmarks.py) | `pytest-benchmark` throughput gates for Phase-1 audit, Kronecker `float64` solve, and associative scans. |
