# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

**Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")** unifies:
1. **Core 1:** 12-layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + Radial-Polar Selective Complex RG-LRU stack.
2. **Core 2:** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`).
3. **Control Bus:** $3\text{ kHz}$ Envelopes/LFOs + Lobatto HDMR & Rank-4 FM (`range` / `switch`), **Non-Stop 3-Phase Clamping**, and `.dnam-ckpt` Kronecker Stem Expansion.

## 1. Global Tech Stack & Tooling Rules

- **Monorepo Build (`moon`):** `pnpm check` (`moon check --all`) or `pnpm moon run :<task>` (`[`.moon/workspace.yml`](.moon/workspace.yml)`).
- **Rust:** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test`.
- **Python (`uv`, `py314`):** `uv run ruff check`, `uv run ruff format --check`, `uv run pyrefly check` (zero `# type: ignore`).
- **JS/TS (`pnpm`, `node 24/26`):** `pnpm format:check`, `pnpm lint` (`@biomejs/biome`), `pnpm typecheck` (`tsc --noEmit`).
- **Testing & Benchmarks:** $\ge 95\%$ coverage (`pnpm coverage`) and zero regressions (`pnpm bench`).
- **Git & CLA:** Sign off commits via `git commit -s` per [`CLA.md`](CLA.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero `FSQRT` / `FDIV` in Real-Time Loops:** Never use division, square-root, or `libm` transcendentals in $48\text{ kHz}$ or $3\text{ kHz}$ loops; use the algebraic factorizations in [`core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce all registered stability bounds (`CL-01`–`CL-20`) in [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md).
3. **Non-Stop 3-Phase Control Clamping:** Automatically clamp unobserved ranges, switches, and corners per [`parametric_control_bus.md`](docs/parametric_control_bus.md) without pausing training.
4. **Fail-Fast & 64B-Aligned Containers:** Validate all `.dnam` and `.dnam-ckpt` containers, `Ed25519` signatures, and `uplo='U'` stem statistics per [`dnam_container_schema.md`](docs/dnam_container_schema.md).

## 3. Progressive Disclosure & Codebase Discovery Map

| Package / Domain | Package `AGENTS.md` & Contributing | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Monorepo & Guardrails** | [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) | [`index.md`](docs/index.md), [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md), [`verification_and_guardrails.md`](docs/verification_and_guardrails.md) |
| **Rust Engine (`packages/engine/`)** | [`engine/AGENTS.md`](packages/engine/AGENTS.md), [`engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) | [`architecture_design.md`](docs/architecture_design.md), [`core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md), [`core2_modulated_ufdn.md`](docs/core2_modulated_ufdn.md) |
| **Python Engine (`packages/python/`)** | [`python/AGENTS.md`](packages/python/AGENTS.md), [`python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) | [`parametric_control_bus.md`](docs/parametric_control_bus.md), [`dnam_container_schema.md`](docs/dnam_container_schema.md), [`digital_marketplace_and_provenance.md`](docs/digital_marketplace_and_provenance.md) |
| **JS / WebAudio (`packages/js/`)** | [`js/AGENTS.md`](packages/js/AGENTS.md), [`js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) | [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md), [`dnam_container_schema.md`](docs/dnam_container_schema.md) |
| **Design System (`packages/ui/`)** | [`ui/AGENTS.md`](packages/ui/AGENTS.md), [`ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) | [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md), [`digital_marketplace_and_provenance.md`](docs/digital_marketplace_and_provenance.md) |
| **CLI & Studio (`packages/studio/`)** | [`studio/AGENTS.md`](packages/studio/AGENTS.md), [`studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) | [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md), [`training_and_iterative_resume.md`](docs/training_and_iterative_resume.md) |
| **Trainer (`packages/trainer/`)** | [`trainer/AGENTS.md`](packages/trainer/AGENTS.md), [`trainer/CONTRIBUTING.md`](packages/trainer/CONTRIBUTING.md) | [`training_and_iterative_resume.md`](docs/training_and_iterative_resume.md), [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Zero Duplication:** Keep `AGENTS.md` files strictly as routing maps (`< 5,000` B root, `< 2,500` B scoped).
- **Single Source of Truth:** Document every equation, clamp (`CL-XX`), or workflow rule in its canonical `docs/` file or [`CONTRIBUTING.md`](CONTRIBUTING.md).
