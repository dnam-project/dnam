# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

**Dynamic Neural Audio Modeler (DNAM / TORA)** unifies:
1. **Operators:** Op-T (Transport/Elastic Buffer), Op-M (Modal Bank/Kautz), Op-N (Commuted Cascade/PolyADAA).
2. **Topology:** 8-Port Matrix Router + Stages I–IV + 256-D Mid/Side Kronecker Head.
3. **Control Bus:** $3\text{ kHz}$ Envelopes/LFOs + Lobatto HDMR & Rank-4 FM, Non-Stop 3-Phase Clamping, and `.dnam-ckpt` Stem Expansion.

## 1. Global Tech Stack & Tooling Rules

- **Monorepo Orchestration (`moon` + `mise`):** Run all builds, quality gates, tests, CLIs, and dev servers via `moon` ([`.moon/workspace.yml`](.moon/workspace.yml) & [`mise.toml`](mise.toml)): `moon check --all`, `moon check <project>`, or `moon run <project>:<task>`.
- **CLIs & Dev Servers:** `moon run studio:cli -- <args>`, `moon run trainer:cli -- <args>`, `moon run studio:dev`, `moon run studio:tauri-dev`, and `moon run ui:dev`.
- **Quality Gates:** `moon run :format-check`, `moon run :lint`, `moon run :typecheck` (zero `# type: ignore` or `any`), `moon run :coverage` (≥ 95%), and `moon run :bench`.
- **Git & CLA:** Sign off commits via `git commit -s` per [`CLA.md`](CLA.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero `FSQRT` / `FDIV` in Real-Time Loops:** Never use division, square-root, or `libm` transcendentals in $48\text{ kHz}$ or $3\text{ kHz}$ loops; use the algebraic factorizations in [`op_n_commuted_cascade.md`](docs/specs/dsp/op_n_commuted_cascade.md) and [`op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce all registered stability bounds in [`control_space_audit.md`](docs/specs/training/control_space_audit.md).
3. **Non-Stop 3-Phase Control Clamping:** Automatically clamp unobserved ranges, switches, and corners per [`parametric_control_bus.md`](docs/specs/dsp/parametric_control_bus.md) without pausing training.
4. **Fail-Fast & 64B-Aligned Containers:** Validate all `.dnam` and `.dnam-ckpt` containers, `Ed25519` signatures, and `uplo='U'` stem statistics per [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md).

## 3. Progressive Disclosure & Codebase Discovery Map

| Package / Domain | Package `AGENTS.md` & Contributing | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Monorepo & Guardrails (`tests/`)** | [`AGENTS.md`](AGENTS.md), [`tests/AGENTS.md`](tests/AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | [`index.md`](docs/index.md), [`tora_architecture.md`](docs/specs/architecture/tora_architecture.md), [`project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md) |
| **Rust Engine (`packages/engine/`)** | [`engine/AGENTS.md`](packages/engine/AGENTS.md), [`engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) | [`tora_architecture.md`](docs/specs/architecture/tora_architecture.md), [`router_and_stages_topology.md`](docs/specs/dsp/router_and_stages_topology.md) |
| **Python SDK (`packages/python/`)** | [`python/AGENTS.md`](packages/python/AGENTS.md), [`python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) | [`python_sdk.md`](docs/specs/sdks/python_sdk.md), [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) |
| **JS / WebAudio (`packages/js/`)** | [`js/AGENTS.md`](packages/js/AGENTS.md), [`js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) | [`typescript_sdk.md`](docs/specs/sdks/typescript_sdk.md), [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) |
| **Design System (`packages/ui/`)** | [`ui/AGENTS.md`](packages/ui/AGENTS.md), [`ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) | [`control_space_audit.md`](docs/specs/training/control_space_audit.md), [`watermarking_and_marketplace.md`](docs/specs/security/watermarking_and_marketplace.md) |
| **CLI & Studio (`packages/studio/`)** | [`studio/AGENTS.md`](packages/studio/AGENTS.md), [`studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) | [`capture_and_alignment.md`](docs/specs/training/capture_and_alignment.md), [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) |
| **Trainer (`packages/trainer/`)** | [`trainer/AGENTS.md`](packages/trainer/AGENTS.md), [`trainer/CONTRIBUTING.md`](packages/trainer/CONTRIBUTING.md) | [`four_stage_pipeline.md`](docs/specs/training/four_stage_pipeline.md), [`control_space_audit.md`](docs/specs/training/control_space_audit.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Zero Duplication:** Keep `AGENTS.md` files strictly as routing maps (`< 5,000` B root, `< 2,500` B scoped).
- **Single Source of Truth:** Document every equation, stability rule (`r[...]`), or workflow rule in its canonical `docs/` file per [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
