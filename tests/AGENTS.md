# DNAM Monorepo Guardrail Test Suite (`tests/`)

Top-level guardrail suite (`moon check tests`) enforcing [`project_and_documentation_quality.md`](../docs/specs/architecture/project_and_documentation_quality.md) (`r[repo.*]`) and [`system_goals_and_requirements.md`](../docs/requirements/system_goals_and_requirements.md) (`REQ-1.1`–`REQ-8.8`).

## 1. Guardrail Test Routing Map ([`test_guardrails.test.ts`](test_guardrails.test.ts))

| Test Function | Spec Rule Referenced | Enforced Invariant |
| :--- | :--- | :--- |
| `test_progressive_disclosure_agents_md_byte_limits` | `ref[repo.docs.agents-md-budgets]` | Root [`AGENTS.md`](../AGENTS.md) `< 5,000 B`; scoped `AGENTS.md` `< 2,500 B`. |
| `test_requirement_registry_and_agnostic_prose` | `ref[repo.docs.requirements-layer]` | All 40 `REQ-1.1`–`REQ-8.8` rows with implementation-agnostic prose. |
| `test_no_inline_latex_math_in_markdown_tables` | `ref[repo.docs.markdown-table-math]` | Zero raw `$` inline math outside backticks in Markdown table rows. |
| `test_valid_github_latex_math_across_docs` | `ref[repo.docs.github-latex-and-mermaid]` | Valid GitHub MathJax LaTeX math and semicolon-free Mermaid `sequenceDiagram` blocks. |
| `test_relative_markdown_links_resolve` | `ref[repo.docs.relative-links-and-anchors]` | Every relative Markdown link and `#section-anchor` fragment resolves on disk. |
| `test_spec_rule_identifiers_adr_structure_and_no_spec_text_parsing` | `ref[repo.docs.spec-rules-and-adrs]` | Unique `r[...]` rules, resolved `ref[...]`/`verify[...]`, MADR `docs/adr/NNNN-*.md`, `docs/templates/`, no `mod.rs`, and no spec-text parsing in tests. |
| `test_engine_spec_scope_manifest` | `ref[repo.docs.spec-rules-and-adrs]` | Validates [`packages/engine/spec_scope.toml`](../packages/engine/spec_scope.toml) and 100% `verify[...]` coverage of all engine-scoped rules in `packages/engine/`. |
| `test_spec_rules_verified_in_tests` | `ref[repo.docs.spec-rules-and-adrs]` | Reports all `r[...]` rules in `docs/specs/` not yet cited via `verify[...]` in test files. |
| `test_engine_realtime_no_libm_or_float_rem_euclid` | `ref[repo.code.realtime-disassembly-audit]` | Zero `rem_euclid` (`bl _fmodf`) or float transcendental `libm` calls in `packages/engine/src/`. |
| `test_engine_no_coverage_off_or_simd_macros` | `ref[repo.code.explicit-simd-and-coverage]` | Zero `coverage(off)` in `packages/engine/src/` and zero `macro_rules!` in `src/math/`. |
