# DNAM 4-Stage Training Pipeline, Hardware Runtime & `.dnam-ckpt` Resume (`dnam_trainer/training/`)

This directory implements the 4-stage PyTorch optimization pipeline, hardware runtime profiler, multi-scale Gaussian derivative delay surrogate (`Test 6`), `float64` Dual-Ridge Kronecker head solver (`CL-18`), 64B-aligned `Arena 0..3` packer, Polars + Altair telemetry reporter, and local/remote training backends.

## 1. Module Discovery Map

| Module | Responsibilities & Canonical Docs |
| :--- | :--- |
| [`hardware.py`](hardware.py) | `HardwareRuntimeProfile`, `setup_accelerator_environment` (ROCm MIOpen `MIOPEN_FIND_MODE="FAST"`, `cudnn.benchmark = False` on HIP, MPS fast math, CUDA TF32), and `resolve_hardware_profile` (`tpu` $\to$ `cuda`/`rocm` $\to$ `mps` $\to$ `cpu`). |
| [`surrogate_delay.py`](surrogate_delay.py) | `SurrogateDelayReadout` (`torch.autograd.Function`) and `optimize_delay_with_surrogate` (`Test 6` Multi-Scale Gaussian Derivative Envelope Surrogate $\sigma_{\text{grad}}: 35.0 \to 0.0\text{ samples}$). See [`training_and_iterative_resume.md`](../../../../../docs/training_and_iterative_resume.md). |
| [`kronecker_head.py`](kronecker_head.py) | `compute_svd_control_projector`, `compute_stem_sufficient_statistics`, `solve_kronecker_head_float64` (`CL-18` Dual-Ridge $\lambda_{\text{init}} = 10^{-3}\bar{\tau}$, $\lambda_{\text{polish}} = 10^{-6}\bar{\tau}$), and `resume_from_checkpoint` (`< 1e-8` full-retrain parity). See [`clamp_and_audit_registry.md`](../../../../../docs/clamp_and_audit_registry.md). |
| [`arena.py`](arena.py) | `ArenaWeightLayout` (`Arena 0..2` `36,224 B` + `Arena 3` `151,296 B/stem`) and `build_signed_dnam_containers`. See [`dnam_container_schema.md`](../../../../../docs/dnam_container_schema.md). |
| [`telemetry.py`](telemetry.py) | Polars + Altair convergence and ESR telemetry (`build_training_telemetry_df`, `save_telemetry_parquet`, `save_telemetry_html_report`). |
| [`pipeline.py`](pipeline.py) | `DnamTriPhysicsModel` and 4-Stage `DnamTrainer` (Step-0 tail detection for `topology="auto"|"core1_only"|"full"`, Per-Stem State-Carry TBPTT, `torch._foreach_mul_` 3-tier block-diagonal masking, zero-sync inner loop). |
| [`backends.py`](backends.py) | `TrainingBackend`, `LocalTrainingBackend`, and `RemoteTrainingBackend`. |
