# DNAM Trainer Test Suite Guide (`packages/trainer/tests/`)

1:1 mirrored unit, property-based (`hypothesis`), worker CLI, integration (`Tests 6–8H`), guardrail, and benchmark test suite for `dnam_trainer`.

## 1. Test Directory Map

| Path | Scope |
| :--- | :--- |
| [`test_guardrails.py`](test_guardrails.py) | AST guardrails: forbidden imports, zero `# type: ignore`, `AGENTS.md` size limits, and `CL-01`–`CL-18` registry checks. |
| [`test_schema_and_events.py`](test_schema_and_events.py) | Pydantic v2 validation (`TrainingStem`, `StemAudioRef`, `TrainingJobRequest`, `CaptureEvent`, `TrainingEvent`). |
| [`dsp/`](dsp/) | `test_poly_adaa.py` (`CL-01`..`CL-02`), `test_selective_rglru.py` (`CL-03`..`CL-05`), `test_modulated_ufdn.py` (`CL-06`..`CL-11`). |
| [`control/`](control/) | `test_basis.py` (`CL-12`..`CL-14`, Lobatto HDMR & Rank-4 FM), `test_audit.py` (`CL-15`..`CL-17`, 3-Phase audit). |
| [`training/`](training/) | `test_hardware.py`, `test_metrics_and_cascade.py` (`bESR`, Bezout crossover, LTI IRs & 3-stage cascade), `test_surrogate_delay.py` (`Test 6`), `test_kronecker_head.py` (`CL-18`), `test_arena.py`, `test_telemetry.py`, `test_pipeline.py`, `test_backends.py`. |
| [`cli/`](cli/) | `test_train_worker_cli.py` (`dnam-train` headless worker across `jsonl`, `sse`, `rich`, `--status-json`, and `.dnam-ckpt` resume). |
| [`integration/test_dnam_verification.py`](integration/test_dnam_verification.py) | End-to-end verification of `Tests 6–8I` from [`verification_and_guardrails.md`](../../../../docs/verification_and_guardrails.md). |
| [`test_benchmarks.py`](test_benchmarks.py) | `pytest-benchmark` throughput gates for Phase-1 audit, Kronecker `float64` solve, and associative scans. |
