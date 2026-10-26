# `dnam_trainer/training/` — 4-Stage Pipeline, Cascade Solver & `.dnam-ckpt` Resume

## 1. Module Routing Map

- [`hardware.py`](hardware.py): Accelerator runtime profile & environment setup (`ROCm`/`CUDA`/`MPS`/`TPU`/`CPU`).
- [`metrics.py`](metrics.py): Psychoacoustic evaluation suite (`bESR`, 4th-order Bezout 3-band `ESR`, `MR-STFT`, `LTAS` tilt, `SegESR`).
- [`selective_cascade.py`](selective_cascade.py): Causal Wiener LTI (`IR`/space) + 3-stage selective state-space & 3-band cascade + $48\text{ kHz}$ HF polish.
- [`surrogate_delay.py`](surrogate_delay.py): Multi-scale Gaussian derivative envelope surrogate delay readout (`Test 6`).
- [`kronecker_head.py`](kronecker_head.py): Partitioned SVD control projector, 3-band `bESR` Gram stats, Dual-Ridge `float64` Kronecker solver (`CL-18`), and `.dnam-ckpt` resume.
- [`arena.py`](arena.py): 64B-aligned `Arena 0..3` weight layout (`36,224 B` + `151,296 B/stem`) & `Ed25519` container builder.
- [`telemetry.py`](telemetry.py): Polars + Altair `bESR` convergence Parquet & HTML reports.
- [`pipeline.py`](pipeline.py): `DnamTriPhysicsModel` and 4-Stage `DnamTrainer` orchestrator.
- [`backends.py`](backends.py): `TrainingBackend`, `LocalTrainingBackend`, and `RemoteTrainingBackend`.

## 2. Canonical Specifications & Guides

1. [`training_and_iterative_resume.md`](../../../../../docs/training_and_iterative_resume.md) — 4-Stage pipeline, psychoacoustic `bESR` tiers, and `.dnam-ckpt` resume.
2. [`clamp_and_audit_registry.md`](../../../../../docs/clamp_and_audit_registry.md) & [`dnam_container_schema.md`](../../../../../docs/dnam_container_schema.md) — `CL-18` Dual-Ridge & `Arena 0..3` `uplo='U'` packing.
3. [`packages/trainer/CONTRIBUTING.md`](../../../CONTRIBUTING.md) — Mixed-precision, hardware accelerator, and runtime invariants.
