---
id: ADR-NNNN
title: "<Short Noun Phrase Stating the Architectural Decision>"
status: proposed
supersedes: []
superseded_by: null
specs:
  - docs/specs/architecture/tora_architecture.md
requirements:
  - REQ-1.1
---

# ADR-NNNN: `<Short Noun Phrase Stating the Architectural Decision>`

> [!NOTE]
> Use this MADR (`docs/adr/NNNN-kebab-case-title.md`) template to record **why** a mathematical formulation, wire layout, or architectural trade-off was chosen over naive or exploratory R&D candidates (see [`../specs/architecture/tora_architecture.md`](../specs/architecture/tora_architecture.md) and [`REQ-1.1`](../requirements/system_goals_and_requirements.md)). Keep normative **"how"** contracts in `docs/specs/`.

---

## 1. Context & Problem Statement

Describe the physical, numerical, or hardware constraint (e.g., `ARM Cortex-A76` `64 KB` L1D cache, `FTZ`/`DAZ` subnormal flushing, `f32` catastrophic cancellation, or `1/f²` spectral weighting) that makes naive approaches fail.

| Naive Formulation | Failure Mechanism | Audible or Numerical Consequence |
| :--- | :--- | :--- |
| **`<Naive Approach A>`** | Explain the mathematical or hardware flaw (no raw dollar signs in table cells per `ref[repo.docs.markdown-table-math]`). | Quantify the error, drift, aliasing, or CPU budget violation. |

---

## 2. Decision & Counterexample Matrix

| Mechanism & Spec Rule | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **`<Mechanism Name>` (`ref[example.subsystem.recurrence]`)** | Summarize the chosen division-free algorithm, representation, or protocol. | Give a concrete numerical counterexample demonstrating why the rejected alternative fails. |

---

## 3. Consequences & Evaluated R&D Candidates

1. **Positive Consequences:** State the verified bound, invariant, or complexity improvement established by the decision.
2. **Evaluated R&D Alternatives:** Record any secondary R&D candidate formulations evaluated and archived during design exploration.
