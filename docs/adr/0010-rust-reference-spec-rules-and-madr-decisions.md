---
id: ADR-0010
title: "Three-Layer Requirements/Specs/ADR Architecture, Rust Reference Rule IDs & Documentation Quality Guardrails"
status: accepted
date: "2026-03-30"
supersedes: null
superseded_by: null
specs:
  - "../specs/architecture/project_and_documentation_quality.md"
requirements:
  - REQ-4.1
  - REQ-6.1
  - REQ-8.8
---

# ADR-0010: Three-Layer Requirements/Specs/ADR Architecture, Rust Reference Rule IDs & Documentation Quality Guardrails

This record justifies the three-layer documentation hierarchy (`docs/requirements/`, `docs/specs/`, `docs/adr/`), the Rust Reference-style `r[...]` / `ref[...]` / `verify[...]` rule identifier syntax, MADR decision records, and automated Markdown/MathJax/AGENTS.md guardrails specified in [`../specs/architecture/project_and_documentation_quality.md`](../specs/architecture/project_and_documentation_quality.md).

---

## 1. Context & Problem Statement

As DNAM scales across Rust (`packages/engine/`, `packages/trainer/`), WebAssembly (`packages/js/`), and Tauri/React (`packages/studio/`, `packages/ui/`), mixing high-level system requirements ("the what"), normative mathematical/wire specifications ("the how"), historical R&D derivations ("the why"), and ad-hoc prefix registries (`CL-XX`, `G-XX`, `Test X`) created four maintenance hazards:

1. **Conflated Requirements and Implementation Specs:** Embedding wire field names or algorithm choices in system requirements prevented evaluating alternative implementations against implementation-agnostic goals.
2. **Verbose Rationale in Normative Specs:** Including lengthy "Why Naive Approach X Fails" derivations inside normative specs obscured the exact equations and bit-level contracts needed by engine and trainer implementers.
3. **Brittle Spec-Text Parsing in Tests:** Tests that read Markdown files from `docs/` and split on fenced code blocks broke whenever documentation formatting or prose was refined, and coupled specs to temporary test filenames.
4. **Ambiguous Rule Definitions vs. References:** Using a single token syntax (or separate `CL-XX` / `G-XX` registries) made it impossible to mechanically distinguish where a rule is *defined* from where it is *referenced* or *verified*.

---

## 2. Considered Options & Rejected Candidates

| Candidate Approach | Why It Fails | Quantified Failure / Maintenance Penalty |
| :--- | :--- | :--- |
| **Single Monolithic Spec + Separate `CL-XX` / `G-XX` / `verification_and_guardrails.md` Registries** | Splits normative rules across subsystem specs, a separate clamp registry, and a separate verification manual (`ref[repo.docs.spec-rules-and-adrs]`). | Forces every stability or guardrail change to touch 3+ files and duplicates thresholds across `docs/specs/` and `docs/verification/`. |
| **Untagged Prose Specs + Markdown Text-Block Parsing in Tests** | Unit tests open `docs/**/*.md` and slice ```` ```text ```` blocks to extract tables or constants (`ref[repo.docs.spec-rules-and-adrs]`). | Any editorial improvement or section reordering in `docs/` breaks unit tests unrelated to the edit. |
| **Identical Syntax for Rule Definitions and Cross-References** | Using `r[...]` for both defining a rule and citing it in ADRs, code, or tests (`ref[repo.docs.spec-rules-and-adrs]`). | Cannot enforce single-definition uniqueness (`1:1` canonical rule location) or distinguish doc cross-references from test verifications. |

---

## 3. Decision Outcome

We adopt the three-layer documentation architecture and rule-tracking system defined in [`../specs/architecture/project_and_documentation_quality.md`](../specs/architecture/project_and_documentation_quality.md):

1. **Three Distinct Documentation Layers (`ref[repo.docs.requirements-layer]`, `ref[repo.docs.spec-rules-and-adrs]`):**
   - `docs/requirements/`: Implementation-agnostic system goals and requirements (`REQ-1.1`–`REQ-8.8`, **the what**).
   - `docs/specs/{architecture,dsp,training,containers,security}/`: Concise normative specifications (**the how**), free of long R&D narratives or temporary test-file links.
   - `docs/adr/NNNN-kebab-case.md`: Atomic MADR records (**the why**), linked via YAML frontmatter (`adrs:` / `specs:`) and guarded against superseded ADR references.
2. **Tri-Partite Rule Syntax (`r[...]`, `ref[...]`, `verify[...]`, `ref[repo.docs.spec-rules-and-adrs]`):**
   - `` `r[<domain>.<subdomain>.<rule>]` `` defines a rule at its single authoritative location in `docs/specs/**/*.md`.
   - `ref[<domain>.<subdomain>.<rule>]` references that rule in ADRs, docs, or implementation comments.
   - `verify[<domain>.<subdomain>.<rule>]` cites that rule in test suites (`tests/`, `packages/**/tests/`), allowing `test_spec_rules_verified_in_tests` to report unverified specification rules as engine and trainer implementations progress.
3. **Progressive-Disclosure & Rendering Guardrails (`ref[repo.docs.agents-md-budgets]`, `ref[repo.docs.markdown-table-math]`, `ref[repo.docs.github-latex-and-mermaid]`, `ref[repo.docs.relative-links-and-anchors]`):**
   - Automated guardrails enforce `< 5,000 B` root / `< 2,500 B` scoped `AGENTS.md` routing maps, CommonMark + MathJax LaTeX compatibility, zero raw `$` in Markdown table rows, and relative link/anchor resolution.
