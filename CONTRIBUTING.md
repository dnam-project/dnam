# Contributing to Dynamic Neural Audio Modeler (DNAM)

Thank you for contributing to the **Dynamic Neural Audio Modeler (DNAM)** monorepo! All contributors—both human engineers and AI coding agents—must follow the licensing, architectural, numerical, testing, and documentation standards defined in this guide.

---

## 1. License, Contributor License Agreement (CLA) & Git Sign-Off

Dynamic Neural Audio Modeler (DNAM) is distributed under the [`DNAM Source-Available & Branding License 1.0.0`](LICENSE). To ensure contributions can be integrated and distributed across community, hardware DSP, and commercial deployments, **all contributors must agree to the [DNAM Contributor License Agreement (`CLA.md`)](CLA.md)**.

### How to Sign Off Commits

Include a standard Git `Signed-off-by` trailer on every commit using `-s`:

```bash
git commit -s -m "feat(engine): optimize Stage II SoA block execution"
```

Or include the trailer explicitly in your commit message:

```text
Signed-off-by: Your Legal Name <your.email@example.com>
```

---

## 2. Monorepo Package & Specification Contributing Guides

DNAM is organized as a multi-language workspace across `docs/`, `packages/*`, and `tests/`. Before modifying a package or specification, read its scoped `AGENTS.md` routing map and `CONTRIBUTING.md` guide:

| Scope / Package | Directory | Scoped `AGENTS.md` / Router | Contributing Guide |
| :--- | :--- | :--- | :--- |
| **Specification Manual (`docs/`)** | `docs/` | [`docs/index.md`](docs/index.md) | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |
| **Rust SIMD Engine (`dnam-engine`)** | `packages/engine/` | [`packages/engine/AGENTS.md`](packages/engine/AGENTS.md) | [`packages/engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) |
| **Python SDK (`dnam`)** | `packages/python/` | [`packages/python/AGENTS.md`](packages/python/AGENTS.md) | [`packages/python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) |
| **TypeScript / WebAudio SDK (`@dnam-project/dnam`)** | `packages/js/` | [`packages/js/AGENTS.md`](packages/js/AGENTS.md) | [`packages/js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) |
| **PyTorch Two-Sweep Trainer (`dnam_trainer`)** | `packages/trainer/` | [`packages/trainer/AGENTS.md`](packages/trainer/AGENTS.md) | [`packages/trainer/CONTRIBUTING.md`](packages/trainer/CONTRIBUTING.md) |
| **Design System & Visualizers (`@dnam-project/ui`)** | `packages/ui/` | [`packages/ui/AGENTS.md`](packages/ui/AGENTS.md) | [`packages/ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) |
| **CLI & Desktop Studio (`@dnam-project/studio`)** | `packages/studio/` | [`packages/studio/AGENTS.md`](packages/studio/AGENTS.md) | [`packages/studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) |
| **Project Guardrail Suite (`dnam-tests`)** | `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |

---

## 3. Workspace Setup, Local Development & Verification Commands

DNAM uses **`mise`** ([`mise.toml`](mise.toml)) to provision toolchains and **`moon`** ([`.moon/workspace.yml`](.moon/workspace.yml)) to orchestrate and cache tasks across **`cargo`** (Rust 2024 edition, `1.99+`), **`uv`** (Python `3.15t` free-threaded dev runtime, `uv >= 0.13.7`), and **`pnpm`** (Node.js `24` / `26` LTS):

### 3.1 Setup & Local Development (CLI, Desktop Studio UI & React Cosmos)

```bash
# Install pinned toolchains (moon, uv, node, pnpm, python, rust)
mise install

# Run the unified `dnam` CLI (`init`, `model`, `capture`, `train`, `inspect`, `studio`)
moon run studio:cli -- --help

# Run the Python Two-Sweep Cholesky trainer worker CLI (`dnam-train`)
moon run trainer:cli -- --help

# Launch the Desktop Studio UI in browser dev mode (Vite HMR at http://localhost:1420)
moon run studio:dev

# Launch the native Tauri v2 Desktop Studio app (Rust host + WebView)
moon run studio:tauri-dev

# Launch the React Cosmos component & visualizer workbench (http://localhost:5000)
moon run ui:dev
```

### 3.2 Quality Gates, Tests & Benchmarks

```bash
# Run the unified moon DAG across all Rust, Python, JS/TS packages, and project guardrails
moon check --all        # format-check, lint, typecheck, build, test
moon run :build
moon run :test
moon run :lint
moon run :format        # auto-format all Rust, Python, and TS/CSS files
moon run :format-check
moon run :typecheck
moon run :coverage      # >= 95% coverage across all packages
moon run :bench
moon run studio:test-e2e

# Check a single project via moon
moon check engine
moon check python
moon check trainer
moon check js
moon check ui
moon check studio
moon check tests
```

---

## 4. Core Cross-Cutting Engineering & Numerical Invariants

1. **Zero `FSQRT` / `FDIV` / `libm` in Real-Time Loops:**
   - Never introduce floating-point division (`/`), hardware square-root (`FSQRT`), or `libm` transcendental calls (`sin`, `cos`, `exp`, `tanh`) into the $48\text{ kHz}$ per-sample audio loop or the $3\text{ kHz}$ control-bus update.
   - Always use the division-free algebraic `PolyADAA` ($m_n, d_n$ factorization) with $1\times$ passive B-spline weights $(0.875, 0.125)$ and the $3\text{ kHz}$ quadrant-reduced contractive rotor specified in [`docs/specs/dsp/op_n_commuted_cascade.md`](docs/specs/dsp/op_n_commuted_cascade.md) and [`docs/specs/dsp/op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md).
2. **128-Bit / 64-Byte SIMD Alignment:**
   - Feature bus widths ($D$), channel counts ($C$), Kautz modal pole counts ($K$), and FDN delay line counts ($M$) must remain exact multiples of 4 (`D=64, C=4, K=4, M=4` for `Lite`; `D=128, C=8, K=8, M=8` for `Full`; `D=192, C=12, K=16, M=8` for `Pro`; `D=256, C=16, K=32, M=16` for `Ultra`), and `.dnam` / `.dnam-ckpt` tensor arenas must remain 64-byte cache-line aligned (`4,096 B` page offset).
3. **Unconditional Stability & Registered Clamps:**
   - Every pole radius ($r_c \le 0.9995$), FDN loop gain ($g_{\text{lin},m} \le 0.9990$), all-pass diffusion coefficient ($|a_{\text{diff},m}| \le 0.75$), fractional delay ($\tau_m \ge 4.0$), and control range must obey the bounds in [`docs/specs/training/control_space_audit.md`](docs/specs/training/control_space_audit.md).
4. **Fail-Fast Validation:**
   - Never silently substitute default controls, sample rates, or tensor shapes when loading `.dnam` or `.dnam-ckpt` containers; return or raise explicit, typed diagnostic errors (`DnamError`).
5. **Zero Duplicated Engine Logic in Language SDKs (`packages/python/`, `packages/js/`):**
   - Language SDKs (`dnam` and `@dnam-project/dnam`) and their FFI shims (`packages/python/rust/`, `packages/js/rust/`) must **never** reimplement DSP math, ZOH rescaling, 3-Phase control clamping, Lobatto calibration inversion, schema validation, RFC 8785 JCS canonicalization, or provenance signature assembly in Python or TypeScript (`ref[sdk.python.engine.zero-copy-and-gil]`, `ref[sdk.ts.runtime.dual-target-drivers]`).
   - Delegate 100% of domain logic to `dnam-engine` in Rust; keep SDK host-language layers strictly focused on idiomatic ergonomics, casing/type coercion, and zero-copy buffer/handle lifecycles.
6. **Property-Based Testing (`proptest!` / `hypothesis` / `fast-check`), TDD, Coverage (≥ 95%) & Benchmarks:**
   - **Mandatory `proptest!` / `hypothesis` / `fast-check` Usage:** Use `proptest!` in Rust unit (`#[cfg(test)]` inside `src/**`) and integration (`tests/*`) test suites, `hypothesis` in Python test suites, and `fast-check` (`fc.assert(fc.property(...))`) in TypeScript test suites wherever possible to verify algebraic identities, ULP/rounding bounds, state stability, wire-format roundtrips, and cross-tier/cross-driver equivalence over arbitrary input domains rather than relying solely on fixed hand-picked examples.
   - Write a failing unit or property-based regression test before fixing a bug.
   - Unit, property, and integration tests must cite canonical `r[...]` spec rules via `verify[...]` and must **never** read or parse Markdown text blocks from `docs/` (only [`tests/test_guardrails.py`](tests/test_guardrails.py) reads `docs/` to verify references and registries).
   - Maintain ≥ 95% test coverage in Rust (`cargo llvm-cov`), Python (`pytest --cov`), and TypeScript (`vitest run --coverage`), and verify zero performance regressions with `cargo bench`, `pytest-benchmark`, and `vitest bench`.

---

## 5. Maintaining `AGENTS.md` & Progressive-Disclosure Documentation

DNAM enforces a **Progressive-Disclosure Documentation Architecture** and a **Single Source of Truth (SSOT)** policy so AI agents and developers never encounter stale or duplicated specifications. For detailed specification authoring, `r[...]` rule identifiers, MADR records, templates (`docs/templates/`), Mermaid diagrams, and GitHub MathJax LaTeX rules (`ref[repo.docs.spec-rules-and-adrs]`, `ref[repo.docs.markdown-table-math]`, `ref[repo.docs.github-latex-and-mermaid]`, `ref[repo.docs.relative-links-and-anchors]`), see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

### 5.1 Two-Tier `AGENTS.md` Hierarchy & Byte Budgets

1. **Tier 1 — Root [`AGENTS.md`](AGENTS.md) (`< 5,000` bytes):**
   - Loaded into the agent context window on every turn.
   - Contains **only** the project summary, global `moon` + `mise` tooling rules, core cross-cutting invariants, and the top-level Progressive Disclosure & Codebase Discovery Map.
2. **Tier 2 — Scoped Package & Subsystem `AGENTS.md` Files (`< 2,500` bytes each):**
   - Located in package/test roots (e.g., [`packages/engine/AGENTS.md`](packages/engine/AGENTS.md), [`packages/trainer/AGENTS.md`](packages/trainer/AGENTS.md), [`packages/studio/AGENTS.md`](packages/studio/AGENTS.md), [`tests/AGENTS.md`](tests/AGENTS.md)) and multi-module subsystem directories (e.g., [`packages/engine/src/operators/AGENTS.md`](packages/engine/src/operators/AGENTS.md), [`packages/trainer/src/dnam_trainer/training/AGENTS.md`](packages/trainer/src/dnam_trainer/training/AGENTS.md)).
   - Act **strictly as routing tables** (`Module / File` $\to$ `Responsibility` $\to$ `Canonical Spec / Contributing Guide`).
3. **Offloading Rule (Never Wordsmith `AGENTS.md` to Fit Byte Budgets):**
   - When an `AGENTS.md` file approaches its byte budget as new modules are added, **do not** try to compress or reword implementation details inside `AGENTS.md`. Instead, offload non-routing information to its canonical destination:
     - **System requirements (`REQ-*`), normative specifications (`r[...]`), psychoacoustic metrics (`bESR`), stability clamps, and MADR rationales (`ADR-NNNN`)** $\to$ Offload to [`docs/`](docs/index.md) using the templates in [`docs/templates/`](docs/templates/spec_template.md) (see [§5.2](#52-single-source-of-truth-ssot-documentation-map) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)).
     - **Package architecture invariants, compiler/runtime flags (`#[forbid(unsafe_code)]`, `zerocopy`, `torch.autocast`, `pyrefly`), workbench setup, and verification commands (`moon check <pkg>`)** $\to$ Offload to `packages/<pkg>/CONTRIBUTING.md`.
     - **Public API examples and CLI command/flag references** $\to$ Offload to `packages/<pkg>/README.md`.
     - **Root landing page ([`README.md`](README.md))** $\to$ Keep tailored for DNAM's target audience (musicians, tone creators, and producers first, with a concise overview for hardware OEMs and plugin/SDK integrators); developer setup, `moon` commands, and internal DSP/spec details belong in [`CONTRIBUTING.md`](CONTRIBUTING.md), `packages/<pkg>/README.md`, and [`docs/`](docs/index.md).
     - **Directory-level module growth** $\to$ Delegate file-level routing rows to a subsystem `AGENTS.md` (e.g., `src/<subsystem>/AGENTS.md`) and link the subsystem from the parent `packages/<pkg>/AGENTS.md`.
4. **Zero Inline LaTeX Math Inside Markdown Tables (`ref[repo.docs.markdown-table-math]`) & ASCII Layouts for Dense Maps:**
   - GitHub's MathJax renderer wraps inline math (`$...$`) inside Markdown table cells (`<th>` / `<td>`) in scrollable containers that display horizontal scrollbars. Never use `$...$` inside Markdown table rows; use plain text, Unicode symbols (`≤`, `≥`, `×`, `→`, `↔`, `Δ`, `σ`, `∈`), or inline code spans (`` `...` ``).
   - Prefer monospaced ASCII box-drawing diagrams (`text` code blocks) over Markdown tables or prose whenever fixed-width spatial alignment displays information more clearly (such as multi-arena parameter budgets, `64B` cache-line field layouts, hierarchical MAC ledgers, and SIMD lane grids); see [`docs/CONTRIBUTING.md` §4.5](docs/CONTRIBUTING.md#45-preferring-monospaced-ascii-diagrams-over-markdown-tables--prose).

### 5.2 Single-Source-of-Truth (SSOT) Documentation Map

Never copy mathematical equations, stability tables, or JSON schemas into `AGENTS.md` or `CONTRIBUTING.md`. Update the canonical specification in `docs/` (following [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`docs/templates/`](docs/templates/spec_template.md)) and link to it:

| When You Modify... | Update This Canonical Document (SSOT) |
| :--- | :--- |
| **System goals, complex full-rig requirements (`REQ-1.5`), hardware budgets, `bESR` tiers, or TORA topology** | [`docs/requirements/system_goals_and_requirements.md`](docs/requirements/system_goals_and_requirements.md), [`docs/specs/architecture/psychoacoustic_quality_and_tiers.md`](docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`docs/specs/architecture/tora_architecture.md`](docs/specs/architecture/tora_architecture.md) |
| **Any 3-phase control-space audit or automatic clamping rule (`r[train.audit.*]`)** | [`docs/specs/training/control_space_audit.md`](docs/specs/training/control_space_audit.md) |
| **Op-N Commuted Cascade, `1x B-Spline PolyADAA`, or DC resting equilibrium (`r[dsp.op-n.*]`)** | [`docs/specs/dsp/op_n_commuted_cascade.md`](docs/specs/dsp/op_n_commuted_cascade.md) |
| **Op-M Modal Bank, symplectic recurrence, Kautz filter, or contractive rotor (`r[dsp.op-m.*]`)** | [`docs/specs/dsp/op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md) |
| **Op-T Transport, Catmull-Rom fractional delay, diffusers, or dynamic ducking (`r[dsp.op-t.*]`)** | [`docs/specs/dsp/op_t_transport.md`](docs/specs/dsp/op_t_transport.md) |
| **8-Port Matrix Router, dual return aligners, or multi-stage rig topology (`r[dsp.router.*]`, `r[dsp.zoh.*]`)** | [`docs/specs/dsp/router_and_stages_topology.md`](docs/specs/dsp/router_and_stages_topology.md) |
| **Parametric Control Bus, Lobatto HDMR, Rank-4/8 FM, Chebyshev LFO, or tempo sync (`r[dsp.ctrl.*]`, `r[dsp.tempo.*]`)** | [`docs/specs/dsp/parametric_control_bus.md`](docs/specs/dsp/parametric_control_bus.md) |
| **`.dnam` / `.dnam-ckpt` 64B preamble, JSON header schema, SoA arena offsets, or `DNCA` (`r[container.*]`)** | [`docs/specs/containers/dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md), [`docs/specs/containers/checkpoint_and_cache_format.md`](docs/specs/containers/checkpoint_and_cache_format.md) |
| **JCS SHA-256, `Ed25519` signatures, `.dnam-seal` (`DNSL`), or Buyer Watermarking (`r[sec.*]`)** | [`docs/specs/security/provenance_and_enclave_drm.md`](docs/specs/security/provenance_and_enclave_drm.md), [`docs/specs/security/watermarking_and_marketplace.md`](docs/specs/security/watermarking_and_marketplace.md) |
| **Capture alignment, Two-Sweep Cholesky pipeline, `uplo='U'` Gram matrices, or stem expansion (`r[train.*]`)** | [`docs/specs/training/capture_and_alignment.md`](docs/specs/training/capture_and_alignment.md), [`docs/specs/training/four_stage_pipeline.md`](docs/specs/training/four_stage_pipeline.md), [`docs/specs/training/iterative_resume_and_caching.md`](docs/specs/training/iterative_resume_and_caching.md) |
| **Repository, code, or documentation quality rules (`r[repo.*]`)** | [`docs/specs/architecture/project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md) |
| **Architectural trade-offs, mathematical counterexamples, or rejected R&D candidates (`ADR-NNNN`)** | [`docs/adr/`](docs/adr/0002-tri-operator-recurrent-architecture.md) (using [`docs/templates/adr_template.md`](docs/templates/adr_template.md)) |

---

## 6. Submitting a Pull Request

1. **Create a topic branch** from `main` (`git checkout -b feat/your-feature`).
2. **Run all package and workspace quality gates** (see [§3](#3-workspace-setup-local-development--verification-commands) and the relevant package `CONTRIBUTING.md`).
3. **Verify `AGENTS.md` byte budgets, registries, and Markdown link integrity** via `moon check tests`.
4. **Sign off your commit** (`git commit -s -m "feat(scope): summary"`).
5. **Open a Pull Request** against `main` describing the motivation, architectural invariants preserved, and benchmark/test results.
