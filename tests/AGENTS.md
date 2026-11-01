# DNAM Monorepo Guardrail Test Suite (`tests/`)

Top-level project-scoped guardrail suite (`moon check tests`) enforcing progressive disclosure (`AGENTS.md`), specification registries (`CL-01`–`CL-20` & `REQ-1.1`–`REQ-8.8`), GitHub MathJax/Mermaid syntax, and Markdown link integrity (`G-02`–`G-05`, `G-08`).

## 1. Guardrail Test Routing Map

| File & Test Function | Guardrail ID | Enforced Invariant & Canonical Spec |
| :--- | :---: | :--- |
| [`test_guardrails.py`](test_guardrails.py) :: `test_progressive_disclosure_agents_md_byte_limits` | **`G-02`** | Root [`AGENTS.md`](../AGENTS.md) `< 5,000 B` and scoped `packages/**/AGENTS.md` + `tests/AGENTS.md` `< 2,500 B` ([`CONTRIBUTING.md` §5.1](../CONTRIBUTING.md#51-two-tier-agentsmd-hierarchy--byte-budgets)). |
| [`test_guardrails.py`](test_guardrails.py) :: `test_clamp_and_requirement_registries_documented` | **`G-03`** | `CL-01`–`CL-20` in [`clamp_and_audit_registry.md`](../docs/training/clamp_and_audit_registry.md), `REQ-1.1`–`REQ-8.8` (38 requirements) in [`system_goals_and_requirements.md`](../docs/overview/system_goals_and_requirements.md), and [`docs/CONTRIBUTING.md`](../docs/CONTRIBUTING.md). |
| [`test_guardrails.py`](test_guardrails.py) :: `test_no_inline_latex_math_in_markdown_tables` | **`G-04`** | Zero raw `$` inline math delimiters outside backticks inside Markdown table rows ([`docs/CONTRIBUTING.md` §4.2](../docs/CONTRIBUTING.md#42-zero-inline-latex-math-in-markdown-tables-g-04)). |
| [`test_guardrails.py`](test_guardrails.py) :: `test_valid_github_latex_math_across_docs` | **`G-05`** | Valid GitHub CommonMark + MathJax LaTeX math and semicolon-free Mermaid `sequenceDiagram` blocks ([`docs/CONTRIBUTING.md` §4.1–§4.3](../docs/CONTRIBUTING.md#41-mermaid-diagrams-g-05)). |
| [`test_guardrails.py`](test_guardrails.py) :: `test_relative_markdown_links_resolve` | **`G-08`** | Every relative Markdown file link and `#section-anchor` fragment resolves on disk ([`docs/CONTRIBUTING.md` §4.4](../docs/CONTRIBUTING.md#44-relative-markdown-links--heading-anchors-g-08)). |
