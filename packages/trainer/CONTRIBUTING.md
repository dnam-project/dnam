# Contributing to `dnam-trainer` (`packages/trainer/`)

This guide covers development, numerical precision, hardware acceleration, and verification rules for the `dnam-trainer` Python package (`packages/trainer/`). Read the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) before modifying this package.

## 1. Core Architectural & Numerical Invariants

1. **Enforce Registered Clamps (`CL-01`–`CL-18`):**
   - Every differentiable DSP layer (`dsp/`), control basis (`control/`), and Kronecker solver (`training/kronecker_head.py`) must enforce its assigned clamps from [`docs/clamp_and_audit_registry.md`](../../docs/clamp_and_audit_registry.md) and reference the `CL-XX` identifier in module docstrings/comments so `test_guardrails.py` verifies 100% registry integrity.
2. **Split Mixed-Precision Policy (`bf16` Contractions + `float32` IIR + Host-CPU `float64` Normal Equations):**
   - Stateless matrix contractions (`Conv1d`, `1x1` mixers, Control Hyper-Bus projections) may use `bfloat16` (`torch.autocast`) on supported accelerators (`TorchTPU`, `ROCm`, `CUDA`, `MPS`).
   - Stateful IIR recurrences and fractional delay interpolators (`PolyADAA` $R_{\text{ac}} = 0.9993$, `SelectiveRgLru` $r \le 0.9995$, `ModulatedUfdn` Lagrange delay & Lyapunov loop) must execute in IEEE-754 `float32` (`torch.autocast(enabled=False)`) so 7-bit `bf16` mantissas never round $0.9995 \to 1.0000$.
   - Stage-2 Wiener-Hopf and Stage-4 Cholesky Kronecker normal equation solves (`192x192`) must always execute in **Host-CPU `float64`** (`device="cpu", dtype=torch.float64`) for compatibility with TorchTPU and Apple Silicon MPS.
3. **Zero `torch` Imports in `project/` and `capture/`:**
   - Keep `dnam_trainer.project` and `dnam_trainer.capture` free of top-level `torch` imports so workspace and capture CLI commands start in `< 20 ms`.
4. **Reference Calibration Integrity:**
   - `dnam capture calibrate` must never back off `send_gain_db` when a hardware device compresses or clips in Step 2; DNAM captures 100% of device gain, compression, and clipping relative to the Step-1 bypass `unity_ref_dbfs`.

## 2. Local Development & Verification Commands

Run from the monorepo root:

```bash
# Run the `dnam-train` headless 4-stage trainer CLI
pnpm dnam-train --help  # or: pnpm moon run trainer:cli -- --help

# Run all trainer quality gates (format-check, lint, typecheck, build, test)
pnpm moon check trainer

# Run trainer coverage (>= 95%) and benchmarks
pnpm moon run trainer:coverage
pnpm moon run trainer:bench
```
