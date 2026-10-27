# Contributing to `dnam-trainer` (`packages/trainer/`)

This guide covers development, numerical precision, hardware acceleration, and verification rules for the `dnam-trainer` Python package (`packages/trainer/`). Read the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) before modifying this package.

## 1. Core Architectural & Numerical Invariants

1. **Enforce Registered Clamps (`CL-01`–`CL-18`):**
   - Every differentiable DSP layer (`dsp/`), control basis (`control/`), and Kronecker solver (`training/kronecker_head.py`) must enforce its assigned clamps from [`docs/clamp_and_audit_registry.md`](../../docs/clamp_and_audit_registry.md) and reference the `CL-XX` identifier in module docstrings/comments so `test_guardrails.py` verifies 100% registry integrity.
2. **Split Mixed-Precision Policy (`bf16` Contractions + `float32` IIR + Host-CPU `float64` Normal Equations):**
   - Stateless matrix contractions (`Conv1d`, `1x1` mixers, Control Hyper-Bus projections) may use `bfloat16` (`torch.autocast`) on supported accelerators (`TorchTPU`, `ROCm`, `CUDA`, `MPS`).
   - Stateful IIR recurrences and fractional delay interpolators (`PolyADAA` $R_{\text{ac}} = 0.9993$, `SelectiveRgLru` $r \le 0.9995$, `ModulatedUfdn` Lagrange delay & Lyapunov loop) must execute in IEEE-754 `float32` (`torch.autocast(enabled=False)`) so 7-bit `bf16` mantissas never round $0.9995 \to 1.0000$.
   - Stage-2 Wiener-Hopf and Stage-4 Cholesky Kronecker normal equation solves (`192x192`) must always execute in **Host-CPU `float64`** (`device="cpu", dtype=torch.float64`) for compatibility with TorchTPU and Apple Silicon MPS.
3. **Psychoacoustic `bESR` Quality Tiers (`metrics.py` & `selective_cascade.py` / `cascade_*.py`):**
   - Evaluate models using Floor-Regularized Band-Balanced ESR (`bESR`), Band-Split `ESR_LF`/`ESR_Mid`/`ESR_HF` via the 4th-order-flat Bezout complementary crossover, `MR-STFT`, 1/3-octave `LTAS` spectral tilt, and `SegESR` (`Lite <= -18 dB` Live Grade, `Full <= -23 dB` Studio Quality with `ESR_HF <= -20 dB`, `Ultra <= -30 dB` clean/LTI or `<= -25 dB` high-gain tube Reference Grade).
4. **Control Bus & 3-Phase Audit Invariants (`control/basis.py` & `control/audit.py`):**
   - **Zero Old-Stem Perturbation (`CL-13`):** Unlocking higher Lobatto polynomial degrees ($K_p \in \lbrace 0, 1, 2, 3 \rbrace$) must evaluate to exact `0.000000` at all previously anchored stem coordinates $u_{p,1}, \dots, u_{p,K_p}$.
   - **Stop-Gradient Corner Barrier (`CL-16`):** Always call `.detach()` (`sg`) on 1st-order main effects $\boldsymbol{\theta}_{\text{main}}$ inside $\mathcal{L}_{\text{barrier}}$ so virtual corner penalties only steer 2nd-order Rank-4 FM weights $\Delta\boldsymbol{\theta}_{\text{FM}}$.
5. **Hardware Accelerator & Pipeline Runtime Rules (`training/hardware.py` & `training/pipeline.py`):**
   - `setup_accelerator_environment` configures ROCm MIOpen (`MIOPEN_FIND_MODE="FAST"`, `cudnn.benchmark = False` on HIP), MPS fast math, and CUDA TF32, resolving accelerators in priority order `tpu` $\to$ `cuda`/`rocm` $\to$ `mps` $\to$ `cpu`.
   - `DnamTrainer` performs Step-0 reverb/delay tail detection (`topology="auto"|"core1_only"|"full"`), routes `Core 1`-only rigs through the analytical LTI + multi-stage selective/orthogonal-FFT cascade (`selective_cascade.py`, `cascade_primitives.py`, `cascade_orthogonal_fft.py`, `cascade_selective_stages.py`), and uses Per-Stem State-Carry TBPTT with `torch._foreach_mul_` 3-tier block-diagonal masking when `Core 2` (`M-UFDN`) is active.
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
