# `dnam_trainer/training/` — 4-Stage Pipeline, Cascade Solver & `.dnam-ckpt` Resume

## 1. Module Routing Map

- [`accelerator.py`](accelerator.py) & [`hardware.py`](hardware.py): Accelerator runtime (`tpu -> cuda/rocm -> mps -> cpu`), capability probes, memory budgets, and 2-D streaming planner (`STREAMED`/`RESIDENT`/`RESIDENT_BATCHED`).
- [`numerics.py`](numerics.py): 32-bit device exact Gram accumulator (`ExactGramAccumulator`) and batched host-`fp64` solver (`HostDenseSolver`).
- [`windowing.py`](windowing.py): On-demand causal B-spline windows (`CausalBsplineWindowSchedule`) and DP regime segmentation (`segment_regimes_dp`).
- [`calib_cache.py`](calib_cache.py): `.dnam-ckpt` `Arena 4` (`DNCA`) calibration stats & `.dnam-cache` sidecar store (`SidecarCacheStore`).
- [`metrics.py`](metrics.py): Psychoacoustic metrics (`bESR`, Bezout 3-band `ESR`, `MR-STFT`, `LTAS` tilt, `SegESR`, per-stem noise floors).
- [`selective_cascade.py`](selective_cascade.py), [`cascade_primitives.py`](cascade_primitives.py), [`cascade_orthogonal_fft.py`](cascade_orthogonal_fft.py), [`cascade_selective_stages.py`](cascade_selective_stages.py): Causal Wiener LTI + 3-stage selective tube & 4-stage orthogonal FFT cascade solvers.
- [`surrogate_delay.py`](surrogate_delay.py): Multi-scale Gaussian derivative envelope surrogate delay readout.
- [`kronecker_head.py`](kronecker_head.py): Partitioned SVD control projector, 3-band `bESR` Gram stats, Dual-Ridge `float64` Kronecker solver (`CL-18`), and `.dnam-ckpt` resume.
- [`wav_io.py`](wav_io.py): 16/24/32-bit PCM & 32-bit IEEE-754 float WAV I/O.
- [`arena.py`](arena.py): 64B-aligned `Arena 0..4` weight/checkpoint layout & `Ed25519` container builder.
- [`telemetry.py`](telemetry.py): Polars + Altair `bESR` convergence Parquet & HTML reports.
- [`pipeline.py`](pipeline.py) & [`backends.py`](backends.py): `DnamTriPhysicsModel`, 4-Stage `DnamTrainer`, and local/remote backends.

## 2. Canonical Specifications & Guides

1. [`training_and_iterative_resume.md`](../../../../../docs/training_and_iterative_resume.md)
2. [`clamp_and_audit_registry.md`](../../../../../docs/clamp_and_audit_registry.md) & [`dnam_container_schema.md`](../../../../../docs/dnam_container_schema.md)
3. [`packages/trainer/CONTRIBUTING.md`](../../../CONTRIBUTING.md)
