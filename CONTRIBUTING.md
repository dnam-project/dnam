# Contributing to Dynamic Neural Audio Modeler (DNAM)

Thank you for contributing to the **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")** monorepo! All contributors—both human engineers and AI coding agents—must follow the licensing, architectural, numerical, testing, and documentation standards defined in this guide.

---

## 1. License, Contributor License Agreement (CLA) & Git Sign-Off

Dynamic Neural Audio Modeler (DNAM) is distributed under the [`DNAM Source-Available & Branding License 1.0.0`](LICENSE). To ensure contributions can be integrated and distributed across community, hardware DSP, and commercial deployments, **all contributors must agree to the [DNAM Contributor License Agreement (`CLA.md`)](CLA.md)**.

### How to Sign Off Commits

Include a standard Git `Signed-off-by` trailer on every commit using `-s`:

```bash
git commit -s -m "feat(engine): optimize Core 1 SoA block execution"
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
| **Python Engine (`dnam_engine`)** | `packages/python/` | [`packages/python/AGENTS.md`](packages/python/AGENTS.md) | [`packages/python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) |
| **JS / WebAudio SDK (`@dnam-project/engine`)** | `packages/js/` | [`packages/js/AGENTS.md`](packages/js/AGENTS.md) | [`packages/js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) |
| **PyTorch 4-Stage Trainer (`dnam_trainer`)** | `packages/trainer/` | [`packages/trainer/AGENTS.md`](packages/trainer/AGENTS.md) | [`packages/trainer/CONTRIBUTING.md`](packages/trainer/CONTRIBUTING.md) |
| **Design System & Visualizers (`@dnam-project/ui`)** | `packages/ui/` | [`packages/ui/AGENTS.md`](packages/ui/AGENTS.md) | [`packages/ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) |
| **CLI & Desktop Studio (`@dnam-project/studio`)** | `packages/studio/` | [`packages/studio/AGENTS.md`](packages/studio/AGENTS.md) | [`packages/studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) |
| **Project Guardrail Suite (`dnam-tests`)** | `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |

---

## 3. Workspace Setup, Local Development & Verification Commands

DNAM uses **`mise`** ([`mise.toml`](mise.toml)) to provision toolchains and **`moon`** ([`.moon/workspace.yml`](.moon/workspace.yml)) to orchestrate and cache tasks across **`cargo`** (Rust 2024 edition, `1.99+`), **`uv`** (Python `3.14`), and **`pnpm`** (Node.js `24` / `26` LTS):

### 3.1 Setup & Local Development (CLI, Desktop Studio UI & React Cosmos)

```bash
# Install pinned toolchains (moon, uv, node, pnpm, python, rust)
mise install

# Run the unified `dnam` CLI (`init`, `model`, `capture`, `train`, `inspect`, `studio`)
moon run studio:cli -- --help

# Run the Python 4-stage trainer worker CLI (`dnam-train`)
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
   - Always use the division-free algebraic `PolyADAA` ($m_n, d_n$ factorization) with $1\times$ passive B-spline weights $(0.875, 0.125)$ and the $3\text{ kHz}$ 4-FMA + Newton rotor ($y_1 = 1.5 - 0.5(\hat{c}^2 + \hat{s}^2)$) specified in [`docs/dsp/core1_polyadaa_and_rglru.md`](docs/dsp/core1_polyadaa_and_rglru.md).
2. **128-Bit / 64-Byte SIMD Alignment:**
   - Channel counts ($C$) and FDN delay line counts ($M$) must remain exact multiples of 4 (`C=4, M=4` for `Lite`; `C=8, M=8` for `Full`; `C=12, M=8` for `Pro`; `C=16, M=16` for `Ultra`), and `.dnam` / `.dnam-ckpt` tensor arenas must remain 64-byte cache-line aligned (`4,096 B` page offset).
3. **Unconditional Stability & Registered Clamps (`CL-01`–`CL-20`):**
   - Every pole radius ($r_c \le 0.9995$), FDN loop gain ($g_{\text{lin},m} \le 0.9990$), all-pass diffusion coefficient ($|a_{\text{diff},m}| \le 0.75$), fractional delay ($\tau_m \ge 4.0$), and control range must obey the bounds in [`docs/training/clamp_and_audit_registry.md`](docs/training/clamp_and_audit_registry.md).
4. **Fail-Fast Validation:**
   - Never silently substitute default controls, sample rates, or tensor shapes when loading `.dnam` or `.dnam-ckpt` containers; return or raise explicit, typed diagnostic errors (`DnamError`).
5. **Test-Driven Bug Fixes (TDD), Coverage (≥ 95%) & Performance Benchmarks:**
   - Write a failing unit or property-based regression test before fixing a bug.
   - Maintain ≥ 95% test coverage in both Rust (`cargo llvm-cov`) and Python (`pytest --cov`) and verify zero performance regressions with `cargo bench` and `pytest-benchmark`.

---

## 5. Maintaining `AGENTS.md` & Progressive-Disclosure Documentation

DNAM enforces a **Progressive-Disclosure Documentation Architecture** and a **Single Source of Truth (SSOT)** policy so AI agents and developers never encounter stale or duplicated specifications. For detailed specification authoring, Mermaid diagram, and GitHub MathJax LaTeX rules (`G-03`–`G-05`, `G-08`), see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

### 5.1 Two-Tier `AGENTS.md` Hierarchy & Byte Budgets

1. **Tier 1 — Root [`AGENTS.md`](AGENTS.md) (`< 5,000` bytes):**
   - Loaded into the agent context window on every turn.
   - Contains **only** the project summary, global `moon` + `mise` tooling rules, core cross-cutting invariants, and the top-level Progressive Disclosure & Codebase Discovery Map.
2. **Tier 2 — Scoped Package & Subsystem `AGENTS.md` Files (`< 2,500` bytes each):**
   - Located in package/test roots (e.g., [`packages/engine/AGENTS.md`](packages/engine/AGENTS.md), [`packages/trainer/AGENTS.md`](packages/trainer/AGENTS.md), [`packages/studio/AGENTS.md`](packages/studio/AGENTS.md), [`tests/AGENTS.md`](tests/AGENTS.md)) and multi-module subsystem directories (e.g., [`packages/engine/src/core1/AGENTS.md`](packages/engine/src/core1/AGENTS.md), [`packages/trainer/src/dnam_trainer/training/AGENTS.md`](packages/trainer/src/dnam_trainer/training/AGENTS.md)).
   - Act **strictly as routing tables** (`Module / File` $\to$ `Responsibility` $\to$ `Canonical Spec / Contributing Guide`).
3. **Offloading Rule (Never Wordsmith `AGENTS.md` to Fit Byte Budgets):**
   - When an `AGENTS.md` file approaches its byte budget as new modules are added, **do not** try to compress or reword implementation details inside `AGENTS.md`. Instead, offload non-routing information to its canonical destination:
     - **Equations, DSP derivations, psychoacoustic metrics (`bESR`), and stability clamps (`CL-01`–`CL-20`)** $\to$ Offload to [`docs/`](docs/index.md) (see [§5.2](#52-single-source-of-truth-ssot-documentation-map) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)).
     - **Package architecture invariants, compiler/runtime flags (`#[forbid(unsafe_code)]`, `zerocopy`, `torch.autocast`, `pyrefly`), workbench setup, and verification commands (`moon check <pkg>`)** $\to$ Offload to `packages/<pkg>/CONTRIBUTING.md`.
     - **Public API examples and CLI command/flag references** $\to$ Offload to `packages/<pkg>/README.md`.
     - **Directory-level module growth** $\to$ Delegate file-level routing rows to a subsystem `AGENTS.md` (e.g., `src/<subsystem>/AGENTS.md`) and link the subsystem from the parent `packages/<pkg>/AGENTS.md`.
4. **Zero Inline LaTeX Math Inside Markdown Tables (`G-04`):**
   - GitHub's MathJax renderer wraps inline math (`$...$`) inside Markdown table cells (`<th>` / `<td>`) in scrollable containers that display horizontal scrollbars. Never use `$...$` inside Markdown table rows; use plain text, Unicode symbols (`≤`, `≥`, `×`, `→`, `↔`, `Δ`, `σ`, `∈`), or inline code spans (`` `...` ``).

### 5.2 Single-Source-of-Truth (SSOT) Documentation Map

Never copy mathematical equations, clamp tables (`CL-XX`), or JSON schemas into `AGENTS.md` or `CONTRIBUTING.md`. Update the canonical specification in `docs/` (following [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)) and link to it:

| When You Modify... | Update This Canonical Document (SSOT) |
| :--- | :--- |
| **System goals, complex full-rig requirements (`REQ-1.5`), hardware budgets, `bESR` tiers, or Tri-Physics topology** | [`docs/overview/system_goals_and_requirements.md`](docs/overview/system_goals_and_requirements.md), [`docs/overview/psychoacoustic_quality_and_tiers.md`](docs/overview/psychoacoustic_quality_and_tiers.md), [`docs/overview/tri_physics_architecture.md`](docs/overview/tri_physics_architecture.md) |
| **Any clamp, stability bound, trust region, or 3-phase audit rule (`CL-01`–`CL-20`)** | [`docs/training/clamp_and_audit_registry.md`](docs/training/clamp_and_audit_registry.md) |
| **Core 1 `1x B-Spline PolyADAA`, DC resting equilibrium, or Selective Complex RG-LRU** | [`docs/dsp/core1_polyadaa_and_rglru.md`](docs/dsp/core1_polyadaa_and_rglru.md) |
| **Core 2 Modulated Unitary FDN (`M-UFDN`), Givens/Householder, or Lyapunov saturation** | [`docs/dsp/core2_modulated_ufdn.md`](docs/dsp/core2_modulated_ufdn.md) |
| **Parametric Control Bus, Lobatto HDMR, Rank-4 FM, Chebyshev LFO, or 3 kHz envelopes** | [`docs/dsp/parametric_control_bus.md`](docs/dsp/parametric_control_bus.md) |
| **`.dnam` / `.dnam-ckpt` 64B preamble, JSON header schema, SoA arena offsets, or ZOH** | [`docs/containers/dnam_binary_format.md`](docs/containers/dnam_binary_format.md), [`docs/containers/checkpoint_and_cache_format.md`](docs/containers/checkpoint_and_cache_format.md) |
| **JCS SHA-256, `Ed25519` signatures, `.dnam-seal` (`DNSL`), or Buyer Watermarking** | [`docs/security/provenance_and_enclave_drm.md`](docs/security/provenance_and_enclave_drm.md), [`docs/security/watermarking_and_marketplace.md`](docs/security/watermarking_and_marketplace.md) |
| **Capture alignment, 4-Stage curriculum, `uplo='U'` Gram matrices, or stem expansion** | [`docs/training/capture_and_alignment.md`](docs/training/capture_and_alignment.md), [`docs/training/four_stage_pipeline.md`](docs/training/four_stage_pipeline.md), [`docs/training/iterative_resume_and_caching.md`](docs/training/iterative_resume_and_caching.md) |
| **Verification tests (`Test 1`–`Test 8I`) or repository guardrails (`G-01`–`G-08`)** | [`docs/verification/verification_and_guardrails.md`](docs/verification/verification_and_guardrails.md) |

---

## 6. Submitting a Pull Request

1. **Create a topic branch** from `main` (`git checkout -b feat/your-feature`).
2. **Run all package and workspace quality gates** (see [§3](#3-workspace-setup-local-development--verification-commands) and the relevant package `CONTRIBUTING.md`).
3. **Verify `AGENTS.md` byte budgets, registries, and Markdown link integrity** via `moon check tests`.
4. **Sign off your commit** (`git commit -s -m "feat(scope): summary"`).
5. **Open a Pull Request** against `main` describing the motivation, architectural invariants preserved, and benchmark/test results.
