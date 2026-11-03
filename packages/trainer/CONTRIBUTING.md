# Contributing to `dnam-trainer` (`packages/trainer/`)

This guide covers development, numerical precision, hardware acceleration, and verification rules for the `dnam-trainer` Python package (`packages/trainer/`). Read the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) before modifying this package.

## 1. Core Architectural & Numerical Invariants

1. **Enforce Registered Clamps (`CL-01`–`CL-20`):**
   - Every differentiable DSP layer (`dsp/`), control basis (`control/`), and Kronecker solver (`training/kronecker_head.py`) must enforce its assigned clamps from [`docs/training/clamp_and_audit_registry.md`](../../docs/training/clamp_and_audit_registry.md) and reference the `CL-XX` identifier in module docstrings/comments so [`tests/test_guardrails.py`](../../tests/test_guardrails.py) (`G-03`) verifies 100% registry integrity.
2. **32-Bit Device Tensors + Batched Host-CPU `float64` Normal Equations (`accelerator.py` & `numerics.py`):**
   - Device tensors across `dsp/` and `training/` use only 32-bit types natively shared by Apple Silicon MPS and Google Cloud TPU: `float32`, `int32`, `bool`, `complex64`, plus `int8`/`int16` storage-only and capability-probed `bfloat16` contractions. **Never allocate `float64`, `complex128`, or `int64` (`torch.long`) tensors on any accelerator device.**
   - Stateful IIR recurrences and fractional delay interpolators (`PolyADAA` $R_{\text{ac}} = 0.9993$, `SelectiveRgLru` $r \le 0.9995$, `ModulatedUfdn` Lagrange delay & Lyapunov loop) must execute in IEEE-754 `float32` (`torch.autocast(enabled=False)`) so 7-bit `bf16` mantissas never round $0.9995 \to 1.0000$.
   - Bulk Gram and cross-correlation accumulation must run on the accelerator and meet the [contraction-accuracy contract](../../docs/training/four_stage_pipeline.md#31-weighted-gram-accuracy-contract). The current `ExactGramAccumulator` does not satisfy it. Small $N \times N$ Cholesky, Kronecker, and `eigh` subspace solves execute in batched **Host-CPU `float64`** via `HostDenseSolver` (`numerics.py`).
3. **Device-Agnostic Psychoacoustic `bESR` Quality Tiers & Noise-Floor-Aware Stems (`metrics.py` & `selective_cascade.py` / `cascade_*.py`):**
   - Evaluate models using Floor-Regularized, Noise-Adjusted Band-Balanced ESR (`bESR`), Band-Split `ESR_LF`/`ESR_Mid`/`ESR_HF` via the 4th-order-flat Bezout complementary crossover, `MR-STFT`, 1/3-octave `LTAS` spectral tilt, and `SegESR` with a single device-agnostic threshold per tier (`Lite <= -18 dB` Live Grade, `Full <= -23 dB` Studio Quality with `ESR_HF <= -20 dB`, `Pro <= -30 dB` Mastering / Reference Grade, `Ultra <= -34 dB` Unconstrained Workstation Max with `D_h=8` SVD).
   - Measure per-stem per-band noise floors from silent-input frames (`measure_stem_band_noise_floor`): stems within $6\text{ dB}$ of their floor are `silent` (must stay at or below floor); quieter tonal stems are `active` and join cascade fitting and tier evaluation, and any band whose measured signal-to-noise ratio $\text{SNR}_b = \text{Signal}_{\text{dBFS},b} - \text{NoiseFloor}_{\text{dBFS},b}$ is smaller than $-\text{Goal}_{\text{dB},b}$ is evaluated via the objective noise-limited rule `is_band_goal_satisfied` ($\text{Residual}_{\text{dBFS},b} \le \text{NoiseFloor}_{\text{dBFS},b} + 1.0\text{ dB}$).
4. **Control Bus & 3-Phase Audit Invariants (`control/basis.py` & `control/audit.py`):**
   - **Zero Old-Stem Perturbation (`CL-13`):** Unlocking higher Lobatto polynomial degrees ($K_p \in \lbrace 0, 1, 2, 3 \rbrace$) must evaluate to exact `0.000000` at all previously anchored stem coordinates $u_{p,1}, \dots, u_{p,K_p}$.
   - **Stop-Gradient Corner Barrier (`CL-16`):** Always call `.detach()` (`sg`) on 1st-order main effects $\boldsymbol{\theta}_{\text{main}}$ inside $\mathcal{L}_{\text{barrier}}$ so virtual corner penalties only steer 2nd-order Rank-4 FM weights $\Delta\boldsymbol{\theta}_{\text{FM}}$.
5. **Hardware Accelerator & Pipeline Runtime Rules (`training/accelerator.py`, `training/hardware.py` & `training/pipeline.py`):**
   - `resolve_accelerator_runtime` and `setup_accelerator_environment` resolve accelerators in priority order `tpu` $\to$ `cuda`/`rocm` $\to$ `mps` $\to$ `cpu` (failing fast if an explicitly requested accelerator is unavailable, and emitting a `RuntimeEvent` warning when `"auto"` falls back to CPU), disable CUDA TF32 and MPS fast-math for exact `fp32` integer slice dot products, set TPU `XLA_MATMUL_PRECISION="highest"`, and plan uncapped 2-D memory execution (`STREAMED`, `RESIDENT`, `RESIDENT_BATCHED`).
   - `DnamTrainer` performs Step-0 reverb/delay tail detection (topology auto-detection or dry-only vs. full stage activation), routes Stages I–III dry rigs through the analytical LTI + multi-stage selective/orthogonal-FFT cascade (`selective_cascade.py`, `cascade_primitives.py`, `cascade_orthogonal_fft.py`, `cascade_selective_stages.py`), and uses Per-Stem State-Carry TBPTT with `torch._foreach_mul_` 3-tier block-diagonal masking when Stage IV spatial tank is active.
6. **Reference Calibration Integrity:**
   - `dnam capture calibrate` must never back off `send_gain_db` when a hardware device compresses or clips in Step 2; DNAM captures 100% of device gain, compression, and clipping relative to the Step-1 bypass `unity_ref_dbfs`.

## 2. Local Development & Verification Commands

Run from the monorepo root:

```bash
# Run the `dnam-train` headless 4-stage trainer CLI
moon run trainer:cli -- --help

# Run all trainer quality gates (format-check, lint, typecheck, build, test)
moon check trainer

# Run trainer coverage (>= 95%) and benchmarks
moon run trainer:coverage
moon run trainer:bench
```
