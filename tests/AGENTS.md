# `tests/` — Monorepo Guardrail Suite Routing Guide

Top-level guardrail suite (`moon check tests`) enforcing [`project_and_documentation_quality.md`](../docs/specs/architecture/project_and_documentation_quality.md) (`r[repo.*]`) and [`system_goals_and_requirements.md`](../docs/requirements/system_goals_and_requirements.md) (`REQ-1.1`–`REQ-8.8`).

| File / Domain | Spec Rules Enforced | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`test_guardrails.test.ts`](test_guardrails.test.ts) (`r[repo.docs.*]`) | `AGENTS.md` byte budgets (`< 5,000 B` / `< 2,500 B`), 40 `REQ-*` rows, unique `r[...]` / `ref[...]` / `verify[...]`, [`spec_scope.toml`](../packages/engine/spec_scope.toml), relative Markdown links/anchors, table-free inline math, GitHub MathJax LaTeX, and Mermaid syntax. | [`project_and_documentation_quality.md`](../docs/specs/architecture/project_and_documentation_quality.md), [`docs/CONTRIBUTING.md`](../docs/CONTRIBUTING.md) |
| [`test_guardrails.test.ts`](test_guardrails.test.ts) (`r[repo.code.*]`) | Zero TS `any`/`@ts-ignore`, `#![forbid(unsafe_code)]`, `< 900`-line Rust file ceiling, no `mod.rs`, zero `.py` files, leaf dev-only `dnam-fixtures` boundaries, zero `libm`/`rem_euclid`/`coverage(off)`/`macro_rules!` in `dnam-engine`, and [`LICENSE`](../LICENSE) / [`CLA.md`](../CLA.md) parity. | [`project_and_documentation_quality.md`](../docs/specs/architecture/project_and_documentation_quality.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
