# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

**Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")** is a real-time stereo neural audio modeling, differentiable DSP, and parametric system-identification monorepo unifying:
1. **Core 1:** 12-layer Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) + Radial-Polar Selective Complex RG-LRU stack.
2. **Core 2:** Unified Stereo In-Loop Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`).
3. **Control Bus:** $3\text{ kHz}$ Envelopes/LFOs + Node-Anchored Lobatto HDMR & Rank-4 Factorization Machine for `range` and `switch` controls, backed by **Non-Stop 3-Phase Control Clamping** and **Single-Checkpoint (`.dnam-ckpt`) Iterative Stem Expansion**.

## 1. Global Tech Stack & Tooling Rules

- **Rust Workspace:** Build, lint, and test native crates via `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test`.
- **Python Workspace:** Always use **`uv`** (`uv sync`, `uv run`) targeting Python 3.14 (`py314`).
- **Lint, Format & Strict Typing:** Pass `uv run ruff check`, `uv run ruff format --check`, and `uv run pyrefly check` (`preset = "strict"` in [`pyproject.toml`](pyproject.toml); never use `# type: ignore`).
- **Testing & Benchmarks:** Maintain $\ge 95\%$ test coverage (`cargo llvm-cov`, `uv run pytest --cov`) and verify zero performance regressions (`cargo bench`, `uv run pytest --benchmark-enable`).
- **Git & CLA:** Sign off all commits via `git commit -s` per [`CLA.md`](CLA.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero `FSQRT` / `FDIV` in Real-Time Loops:** Never use division, square-root, or `libm` transcendentals in $48\text{ kHz}$ or $3\text{ kHz}$ loops; use the algebraic factorizations in [`core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce all registered stability bounds (`CL-01`–`CL-20`) in [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md).
3. **Non-Stop 3-Phase Control Clamping:** Automatically clamp unobserved ranges, switches, and corners per [`parametric_control_bus.md`](docs/parametric_control_bus.md) without pausing training.
4. **Fail-Fast & 64B-Aligned Containers:** Validate all `.dnam` and `.dnam-ckpt` containers, `Ed25519` signatures, and `uplo='U'` stem statistics per [`dnam_container_schema.md`](docs/dnam_container_schema.md).

## 3. Progressive Disclosure & Codebase Discovery Map

Consult the package-scoped `AGENTS.md`, `CONTRIBUTING.md`, and Single-Source-of-Truth (SSOT) specifications before modifying a subsystem:

| Package / Domain | Package `AGENTS.md` & Contributing | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Monorepo & Guardrails** | [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) | [`index.md`](docs/index.md), [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md), [`verification_and_guardrails.md`](docs/verification_and_guardrails.md) |
| **Rust SIMD Engine (`packages/engine/`)** | [`engine/AGENTS.md`](packages/engine/AGENTS.md), [`engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) | [`architecture_design.md`](docs/architecture_design.md), [`core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md), [`core2_modulated_ufdn.md`](docs/core2_modulated_ufdn.md) |
| **Python Engine (`packages/python/`)** | [`python/AGENTS.md`](packages/python/AGENTS.md), [`python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) | [`parametric_control_bus.md`](docs/parametric_control_bus.md), [`dnam_container_schema.md`](docs/dnam_container_schema.md), [`digital_marketplace_and_provenance.md`](docs/digital_marketplace_and_provenance.md) |
| **PyTorch Trainer (`packages/trainer/`)** | *(Phase 3)* | [`training_and_iterative_resume.md`](docs/training_and_iterative_resume.md), [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) |
| **Node.js, Plugin & Web Studio** | *(Phases 2B, 4, 5)* | [`architecture_overview.md`](docs/architecture_overview.md), [`monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Zero Duplication:** Keep `AGENTS.md` files strictly as routing maps (`< 5,000` B root, `< 2,500` B scoped). Never copy formulas, numeric clamp tables, or JSON schemas into `AGENTS.md`.
- **Single Source of Truth:** Document every equation, clamp (`CL-XX`), or workflow rule in its canonical `docs/` file or [`CONTRIBUTING.md` §5](CONTRIBUTING.md#5-maintaining-agentsmd--progressive-disclosure-documentation) and link to it.
