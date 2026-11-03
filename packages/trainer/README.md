# `dnam-trainer` — PyTorch 4-Stage Differentiable Trainer & Headless `dnam-train` Worker

`dnam-trainer` (`dnam-train`) is the Two-Sweep Block-Coordinate Cholesky differentiable PyTorch trainer (Stage 0 Router, Stage I Pre-FX, Stage II Commuted Cascade, Stage III Kautz Modals, Stage IV Spatial Tank, and 256-D Mid/Side Kronecker Head), `.dnam-ckpt` SVD-Kronecker iterative solver, and 3-Phase Control-Space Audit worker for **Dynamic Neural Audio Modeler (DNAM)**.

It is installed and invoked automatically by the native **`dnam` CLI & Tauri Desktop Studio (`packages/studio`)**:

```bash
# 1. Install or inspect the managed Python trainer environment from the `dnam` CLI
dnam trainer install --accelerator auto
dnam trainer status

# 2. Run 4-Stage training or Two-Tier .dnam-ckpt resume via `dnam`
dnam train --model b7k_ultra

# 3. Direct headless worker invocation (used by `packages/studio` & cloud GPU workers)
dnam-train --status-json
dnam-train \
  --request-json job_request.json \
  --output-dnam dist/b7k_ultra.dnam \
  --output-ckpt models/b7k_ultra/checkpoint.dnam-ckpt \
  --reports-dir models/b7k_ultra/reports \
  --event-format jsonl
```

## Architecture Documentation

- [`docs/training/four_stage_pipeline.md`](../../docs/training/four_stage_pipeline.md)
- [`docs/training/iterative_resume_and_caching.md`](../../docs/training/iterative_resume_and_caching.md)
- [`docs/training/clamp_and_audit_registry.md`](../../docs/training/clamp_and_audit_registry.md)
- [`docs/verification/verification_and_guardrails.md`](../../docs/verification/verification_and_guardrails.md)
