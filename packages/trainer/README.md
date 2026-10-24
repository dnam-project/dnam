# `dnam-trainer` — PyTorch 4-Stage Differentiable Trainer, Project Manager & Capture Pipeline

`dnam-trainer` (`dnam` / `dnam-train`) is the project workspace manager, hardware/software capture pipeline, and 4-Stage differentiable PyTorch trainer for **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")**.

## CLI Overview (`dnam`)

```bash
# 1. Initialize a multi-model DNAM project (generates dnam.toml, audio/input.wav & creator Ed25519 key)
dnam init my-rig --name "Studio Rig" --author "Tone Lab"

# 2. Define a model with range and switch controls
dnam model add b7k_ultra --name "Darkglass B7K Ultra" \
  --range drive=0..10:5 \
  --range blend=0..10:7 \
  --switch attack=cut,flat,boost:flat

# 3. Calibrate hardware (Step 1: Bypass Unity Reference; Step 2: Engaged Hot-Corner Verification)
dnam capture calibrate --model b7k_ultra --backend anagram

# 4. Inspect the Star + Switch + Corner capture plan & record stems interactively
dnam capture plan --model b7k_ultra
dnam capture record --model b7k_ultra --all-planned

# 5. Run <200ms pre-flight audit or full 4-Stage training (auto-resumes from checkpoint.dnam-ckpt)
dnam train --model b7k_ultra --audit-only
dnam train --model b7k_ultra

# 6. Inspect the trained .dnam container or audition a WAV through the Rust SIMD DnamEngine
dnam inspect b7k_ultra
dnam render b7k_ultra --input audio/input.wav --output renders/audition.wav --tier ultra
```

## Architecture Documentation

- [`docs/training_and_iterative_resume.md`](../../docs/training_and_iterative_resume.md)
- [`docs/clamp_and_audit_registry.md`](../../docs/clamp_and_audit_registry.md)
- [`docs/verification_and_guardrails.md`](../../docs/verification_and_guardrails.md)
