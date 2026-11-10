---
id: spec.arch.project-quality
title: "Repository, Code & Documentation Quality Specification"
status: canonical
version: "1.0.0"
domain: architecture
rule_prefixes:
  - "r[repo."
requirements:
  - REQ-4.1
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
  - REQ-8.8
adrs:
  - ADR-0010
  - ADR-0019
---

# Repository, Code & Documentation Quality Specification

This specification defines the normative repository structure, static type-safety and import rules, progressive-disclosure `AGENTS.md` byte budgets, 3-layer documentation separation (`requirements/`, `specs/`, `adr/`), Rust Reference `r[...]` / `ref[...]` / `verify[...]` specification rule identifiers, Markdown YAML frontmatter schemas, and GitHub CommonMark + MathJax / Mermaid rendering contracts across **Dynamic Neural Audio Modeler (DNAM)** ([`CONTRIBUTING.md`](../../../CONTRIBUTING.md), [`docs/CONTRIBUTING.md`](../../CONTRIBUTING.md)). For the architectural rationale behind these rules, see [`../../adr/0010-rust-reference-spec-rules-and-madr-decisions.md`](../../adr/0010-rust-reference-spec-rules-and-madr-decisions.md).

---

## 1. Static Type Safety & Package Import Boundaries

`r[repo.code.type-safety-and-imports]`

All packages across the monorepo must enforce strict static typing and lightweight runtime dependency boundaries:

1. **Zero Type Suppressions:** Zero `any` or `@ts-ignore` / `@ts-expect-error` directives in TypeScript (`packages/js/`, `packages/ui/`, `packages/studio/`, `tests/`), and strict `#![forbid(unsafe_code)]` / `#![deny(unsafe_code)]` across all Rust crates (`packages/engine/`, `packages/fixtures/`, `packages/trainer/`, `packages/js/rust/`, `packages/studio/rust/`).
2. **Zero Forbidden Heavy Runtime Imports:** `packages/trainer/` must remain 100% pure Rust (`dnam-trainer`) without external Python/PyTorch/CUDA runtime dependencies; all linear-algebra, FFT, and container operations execute natively in safe Rust.
3. **Safe-Rust-First & Strict `#[allow(unsafe_code)]` Discipline:** All Rust crates (`packages/engine/`, `packages/fixtures/`, `packages/trainer/`, `packages/js/rust/`, `packages/studio/rust/`) must enforce `#![forbid(unsafe_code)]` or `#![deny(unsafe_code)]`. Contributors and AI coding agents must **exhaustively search for safe Rust formulations first** (e.g., `core::sync::atomic::AtomicU64` word arrays for lock-free rings and seqlocks, atomic index ping-pong slots, disjoint struct field borrows, `zerocopy`, and checked `napi` slice borrows) and must never add `#[allow(unsafe_code)]` or `unsafe` blocks out of convenience. `#[allow(unsafe_code)]` is permitted **only** when:
   - **Strictly required by a hardware or C-ABI boundary** that has no safe Rust abstraction (CPU SIMD intrinsics in `packages/engine/src/math/`, hardware `FTZ`/`DAZ` CSR registers in `packages/engine/src/types.rs`, aligned heap allocation in `packages/engine/src/memory.rs`, or raw C-ABI entrypoints in `packages/engine/src/ffi.rs`); or
   - **Proven by benchmark to eliminate a high-impact real-time performance bottleneck** where the safe borrow-/bounds-checked formulation measurably violates real-time throughput/WCET budgets **and** every `unsafe` block is accompanied by a complete `// SAFETY:` proof covering aliasing, alignment, lifetimes, and thread-safety invariants.
4. `r[repo.code.realtime-disassembly-audit]` **Automated Release-Binary Disassembly Audit & Real-Time Source Guardrails (`REQ-6.2`, `REQ-6.3`, [`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):**
   - **Release-Binary Disassembly & Call-Graph Gate (`packages/engine/tests/realtime_disassembly_audit.rs`):** Wired into `moon run engine:test`, this integration test compiles `libdnam_engine.rlib` in `--release` mode, disassembles all object code via `objdump -dr` and `objdump -dr --demangle`, computes the transitive call-graph closure from all real-time entrypoints (`DnamEngine<*>::process_*`, `step_frame*`, `step_control_tick_3khz`, `refresh_collapsed_head_weights`, `DynamicDnamEngine::process_*`, `ControlBus<*>::tick_3khz*`, `StagePipeline<*>::retune_kautz_formant_3khz`, `Stage1PreDynamics<*>::retune_op_m_pre_3khz`, `_dnam_engine_process_*`), and fails the build if any reachable real-time symbol contains floating-point division (`fdiv`, `vdiv*`, `divss`, `divps`, `divsd`, `divpd`), floating-point square root (`fsqrt`, `vsqrt*`, `sqrtss`, `sqrtps`, `sqrtsd`, `sqrtpd`), integer division/remainder (`udiv`, `sdiv`, `div`, `divl`, `divq`, `idiv`, `idivl`, `idivq`, `__udivti3`, `__divti3`, `__umodti3`, `__modti3`), heap allocation (`__rust_alloc`, `__rust_dealloc`, `__rust_realloc`, `__rust_alloc_zeroed`, including Rust v0 mangled shims), or uncached OS syscalls (`sysctlbyname`, `getauxval`), or if any symbol in the crate contains an external `libm` relocation (`fmodf`, `fmod`, `sin*`, `cos*`, `tan*`, `exp*`, `log*`, `pow*`, `sqrt*`, `cbrt*`, `hypot*`, `atan*`, `asin*`, `acos*`, `sinh*`, `cosh*`, `tanh*`).
   - **Static Source Guardrail (`tests/test_guardrails.test.ts`):** Verifies that non-test `packages/engine/src/**/*.rs` files contain zero `rem_euclid` calls (which LLVM lowers to `bl _fmodf`) and zero direct `f32`/`f64` transcendental method calls.
5. `r[repo.code.explicit-simd-and-coverage]` **Explicit SIMD Trait Implementations, Zero `coverage(off)` & Multi-Target Verification (`REQ-6.1`, [`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):**
   - **Zero SIMD-Hiding Macros:** All SIMD backend modules (`packages/engine/src/math/aarch64/{neon,sve2,sme}`, `packages/engine/src/math/x86/{sse2,sse42,avx2}`, `packages/engine/src/math/wasm.rs`, `packages/engine/src/math/scalar.rs`) must be written as explicit, directly inspectable Rust trait implementations without `macro_rules!` wrapper generation that obscures instruction lowering or line-coverage attribution.
   - **Zero `#[cfg_attr(coverage, coverage(off))]`:** Never suppress coverage instrumentation to pass the `>= 95%` coverage gate (`moon run engine:coverage`). Cross-ISA backends must be verified via `moon run engine:test-sve2` (`linux/arm64` QEMU `-cpu max` for `ArmSve2Backend`) and `moon run engine:test-x86` (`x86_64-apple-darwin` under Rosetta 2 with `ROSETTA_ADVERTISE_AVX=1` for `X86Sse2Backend`, `X86Sse42Backend`, and `X86Avx2FmaBackend`).

---

## 2. Two-Tier `AGENTS.md` Progressive-Disclosure Hierarchy

`r[repo.docs.agents-md-budgets]`

`AGENTS.md` files act strictly as concise routing maps pointing to canonical specifications in `docs/specs/` and `CONTRIBUTING.md` guides, never duplicating mathematical derivations or equations:

| `AGENTS.md` Tier | Target Files | Strict Byte Ceiling | Required Content |
| :--- | :--- | :---: | :--- |
| **Tier 1 (Monorepo Root)** | [`AGENTS.md`](../../../AGENTS.md) | `< 5,000 B` | Global `moon` + `mise` commands, core architectural invariants, and the package/domain discovery table. |
| **Tier 2 (Scoped Package / Suite)** | `packages/**/AGENTS.md`, [`tests/AGENTS.md`](../../../tests/AGENTS.md) | `< 2,500 B` | Module-to-file routing table, local `moon` task commands, and links to the canonical `docs/specs/` files. |

---

## 3. Three-Layer Documentation Architecture & YAML Frontmatter

### 3.1 Implementation-Agnostic Requirements Layer (`docs/requirements/`)

`r[repo.docs.requirements-layer]`

[`system_goals_and_requirements.md`](../../requirements/system_goals_and_requirements.md) defines **what** the system must achieve (`REQ-1.1` through `REQ-8.8`, 40 requirements across 8 top-level goals):

1. Every requirement ID `REQ-1.1` through `REQ-8.8` must appear in a canonical Markdown table row (`| **\`REQ-X.Y\`** | ... |`).
2. Outside Markdown link targets (`](...)`), the prose of `docs/requirements/` must remain strictly **implementation-agnostic**, containing zero container file extensions (`.dnam`, `.dnam-ckpt`, `.dnam-cache`, `.dnam-seal`, `.dnce`) and zero internal wire-schema field names (`max_supported_tier`, `user_metadata`, `royalty_splits`, `allowed_protection_profiles`, `allowed_oem_vendors`).

### 3.2 Specification Rules (`r[...]`, `ref[...]`, `verify[...]`), MADR Records & Frontmatter

`r[repo.docs.spec-rules-and-adrs]`

Every Markdown document under `docs/requirements/`, `docs/specs/`, `docs/adr/`, and `docs/templates/` must begin at line 1 with a YAML frontmatter block (`--- ... ---`) and obey the following traceability rules:

| Construct | Syntax & Location | Normative Invariant |
| :--- | :--- | :--- |
| **Rule Definition** | `` `r[<domain>.<subdomain>.<rule>]` `` on its own line or at the start of a numbered list item in `docs/specs/**/*.md` | Globally unique across the repository; matches one of the file's declared frontmatter `rule_prefixes`; every leaf normative section (`##` / `###` / `####`) in `docs/specs/**/*.md` must define at least one `r[...]` rule. Never used outside `docs/specs/` and `docs/templates/spec_template.md`. |
| **Rule Cross-Reference** | `ref[<domain>.<subdomain>.<rule>]` in `docs/adr/`, `docs/`, or source code comments | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. |
| **Test Verification Citation** | `verify[<domain>.<subdomain>.<rule>]` in test files (`tests/`, `packages/**/tests/`) | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. Every `r[...]` rule in `docs/specs/` is tracked for test verification coverage. |
| **MADR Decision Files** | `docs/adr/NNNN-kebab-case-title.md` (`id: ADR-NNNN`, `status: proposed \| accepted \| superseded`) | Numbered sequentially starting at `0001`. Superseded ADRs must declare `superseded_by: ADR-NNNN` pointing to an active ADR, and normative documents in `docs/specs/` and `docs/requirements/` must never reference a superseded ADR. |
| **Authoring Templates** | [`requirement_template.md`](../../templates/requirement_template.md), [`spec_template.md`](../../templates/spec_template.md), [`adr_template.md`](../../templates/adr_template.md) | Canonical templates must exist in `docs/templates/` with valid YAML frontmatter. |
| **Zero Spec-Text Parsing in Tests** | All test files outside [`tests/test_guardrails.test.ts`](../../../tests/test_guardrails.test.ts) | Unit, integration, and reference tests must never read or parse Markdown files from `docs/`; they reference specifications via `verify[...]` citations. |

---

## 4. Markdown, Mermaid & GitHub MathJax Rendering Rules

### 4.1 Zero Inline LaTeX Math in Markdown Tables

`r[repo.docs.markdown-table-math]`

No Markdown table row (any line starting with `|`) in any `.md` file across the repository may contain a raw `$` character outside backtick code spans (`` `...` ``). Inside Markdown table cells, format symbolic expressions using backtick code spans with Unicode mathematical symbols or move multi-line derivations into `$$...$$` display math blocks above or below the table.

### 4.2 GitHub CommonMark + MathJax LaTeX & Mermaid Syntax

`r[repo.docs.github-latex-and-mermaid]`

All `$...$` and `$$...$$` math blocks and `mermaid` fenced blocks across `.md` files must satisfy GitHub's two-stage CommonMark + MathJax parser and Mermaid renderer:

1. **Balanced Braces & Delimiters:** Every `$...$` and `$$...$$` block must have balanced `{` and `}` braces and must not place spaces immediately inside inline `$ ... $` delimiters.
2. **Zero CommonMark Backslash-Escape Collisions in Math:** Never use `\_`, `\;`, `\,`, `\{`, `\}`, `\|`, `\\`, `\operatorname{...}`, `\text{...}` containing underscores, or `|<expr>|_1` / `|<expr>|_2` inside `$...$` or `$$...$$` (use `\lbrace`, `\rbrace`, `\Vert`, `\mathrm{...}`, and `\text{...}` with hyphens or spaces instead).
3. **Semicolon-Free Mermaid Sequence Diagrams:** Never use semicolons (`;`) inside a Mermaid `sequenceDiagram` block.

### 4.3 Relative Markdown Link & Heading Anchor Integrity

`r[repo.docs.relative-links-and-anchors]`

Every relative Markdown link (`[label](relative/path.md#anchor)` or `[label](#anchor)`) outside fenced code blocks across `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `CLA.md`, `docs/**/*.md`, `tests/**/*.md`, and `packages/**/*.md` must resolve to an existing file or directory on disk, and any `#section-anchor` fragment must match a GitHub-slugified heading in the target Markdown file.
