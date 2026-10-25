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

## 2. Monorepo Package Contributing Guides

DNAM is organized as a multi-language workspace under `packages/*`. Before modifying a package, read both its scoped `AGENTS.md` routing map and its package-specific `CONTRIBUTING.md` guide:

| Package | Directory | Scoped `AGENTS.md` | Package Contributing Guide |
| :--- | :--- | :--- | :--- |
| **Rust SIMD Engine (`dnam-engine`)** | `packages/engine/` | [`packages/engine/AGENTS.md`](packages/engine/AGENTS.md) | [`packages/engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) |
| **Python Engine (`dnam_engine`)** | `packages/python/` | [`packages/python/AGENTS.md`](packages/python/AGENTS.md) | [`packages/python/CONTRIBUTING.md`](packages/python/CONTRIBUTING.md) |
| **JS / WebAudio SDK (`@dnam-project/engine`)** | `packages/js/` | [`packages/js/AGENTS.md`](packages/js/AGENTS.md) | [`packages/js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) |
| **PyTorch 4-Stage Trainer (`dnam_trainer`)** | `packages/trainer/` | [`packages/trainer/AGENTS.md`](packages/trainer/AGENTS.md) | [`packages/trainer/CONTRIBUTING.md`](packages/trainer/CONTRIBUTING.md) |
| **Design System & Visualizers (`@dnam-project/ui`)** | `packages/ui/` | [`packages/ui/AGENTS.md`](packages/ui/AGENTS.md) | [`packages/ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) |
| **CLI & Desktop Studio (`@dnam-project/studio`)** | `packages/studio/` | [`packages/studio/AGENTS.md`](packages/studio/AGENTS.md) | [`packages/studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) |
| **VST3 / CLAP / Standalone Plugin** | `packages/plugin/` | *(Phase 4)* | [`docs/monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md) |
| **Web Docs & Interactive Studio** | `packages/web/` | *(Phase 5)* | [`docs/monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md) |

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
# Run the unified moon DAG across all Rust, Python, and JS/TS packages
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

# Check a single package via moon
moon check engine
moon check python
moon check trainer
moon check js
moon check ui
moon check studio
```

---

## 4. Core Cross-Cutting Engineering & Numerical Invariants

1. **Zero `FSQRT` / `FDIV` / `libm` in Real-Time Loops:**
   - Never introduce floating-point division (`/`), hardware square-root (`FSQRT`), or `libm` transcendental calls (`sin`, `cos`, `exp`, `tanh`) into the $48\text{ kHz}$ per-sample audio loop or the $3\text{ kHz}$ control-bus update.
   - Always use the division-free algebraic `PolyADAA` ($m_n, d_n$ factorization) with $1\times$ passive B-spline weights $(0.875, 0.125)$ and the $3\text{ kHz}$ 4-FMA + Newton rotor ($y_1 = 1.5 - 0.5(\hat{c}^2 + \hat{s}^2)$) specified in [`docs/core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md).
2. **128-Bit / 64-Byte SIMD Alignment:**
   - Channel counts ($C$) and FDN delay line counts ($M$) must remain exact multiples of 4 (`C=4, M=4` for `Lite`; `C=8, M=8` for `Full`; `C=12, M=8` for `Ultra`), and `.dnam` / `.dnam-ckpt` tensor arenas must remain 64-byte cache-line aligned (`4,096 B` page offset).
3. **Unconditional Stability & Registered Clamps (`CL-01`–`CL-20`):**
   - Every pole radius ($r_c \le 0.9995$), FDN loop gain ($g_{\text{lin},m} \le 0.9990$), all-pass diffusion coefficient ($|a_{\text{diff},m}| \le 0.75$), fractional delay ($\tau_m \ge 4.0$), and control range must obey the bounds in [`docs/clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md).
4. **Fail-Fast Validation:**
   - Never silently substitute default controls, sample rates, or tensor shapes when loading `.dnam` or `.dnam-ckpt` containers; return or raise explicit, typed diagnostic errors (`DnamError`).
5. **Test-Driven Bug Fixes (TDD), Coverage ($\ge 95\%$) & Performance Benchmarks:**
   - Write a failing unit or property-based regression test before fixing a bug.
   - Maintain $\ge 95\%$ test coverage in both Rust (`cargo llvm-cov`) and Python (`pytest --cov`) and verify zero performance regressions with `cargo bench` and `pytest-benchmark`.

---

## 5. Maintaining `AGENTS.md` & Progressive-Disclosure Documentation

DNAM enforces a **Progressive-Disclosure Documentation Architecture** and a **Single Source of Truth (SSOT)** policy so AI agents and developers never encounter stale or duplicated specifications.

### 5.1 Two-Tier `AGENTS.md` Hierarchy & Byte Budgets

1. **Root [`AGENTS.md`](AGENTS.md) (`< 5,000` bytes):**
   - Loaded into the agent context window on every turn.
   - Contains **only** the project summary, monorepo deliverable rules, core cross-cutting invariants, and the Progressive Disclosure & Documentation Discovery Map.
2. **Scoped `AGENTS.md` Files (`< 2,500` bytes each):**
   - Located in package and subsystem directories (e.g., [`packages/engine/AGENTS.md`](packages/engine/AGENTS.md), [`packages/engine/src/*/AGENTS.md`](packages/engine/AGENTS.md), [`packages/python/AGENTS.md`](packages/python/AGENTS.md)).
   - Act strictly as routing tables linking source files and tests to canonical `docs/*.md` specifications and package `CONTRIBUTING.md` guides.

### 5.2 Single-Source-of-Truth (SSOT) Documentation Map

Never copy mathematical equations, clamp tables (`CL-XX`), or JSON schemas into `AGENTS.md` or `CONTRIBUTING.md`. Update the canonical specification in `docs/` and link to it:

| When You Modify... | Update This Canonical Document (SSOT) |
| :--- | :--- |
| **Any clamp, stability bound, trust region, or 3-phase audit rule (`CL-01`–`CL-20`)** | [`docs/clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) |
| **Core 1 `1x B-Spline PolyADAA`, DC resting equilibrium, or Selective Complex RG-LRU** | [`docs/core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md) |
| **Core 2 Modulated Unitary FDN (`M-UFDN`), Givens/Householder, or Lyapunov saturation** | [`docs/core2_modulated_ufdn.md`](docs/core2_modulated_ufdn.md) |
| **Parametric Control Bus, Lobatto HDMR, Rank-4 FM, Chebyshev LFO, or 3 kHz envelopes** | [`docs/parametric_control_bus.md`](docs/parametric_control_bus.md) |
| **`.dnam` / `.dnam-ckpt` 64B preamble, JSON header schema, SoA arena offsets, or ZOH** | [`docs/dnam_container_schema.md`](docs/dnam_container_schema.md) |
| **JCS SHA-256, `Ed25519` signatures, `ChaCha20` encryption, or Buyer Watermarking** | [`docs/digital_marketplace_and_provenance.md`](docs/digital_marketplace_and_provenance.md) |
| **4-Stage PyTorch curriculum, `uplo='U'` Gram matrices, or iterative stem expansion** | [`docs/training_and_iterative_resume.md`](docs/training_and_iterative_resume.md) |
| **Package deliverables, FFI contracts, or implementation phases** | [`docs/monorepo_implementation_guide.md`](docs/monorepo_implementation_guide.md) |
| **Verification tests (`Test 1`–`Test 10`) or repository guardrails** | [`docs/verification_and_guardrails.md`](docs/verification_and_guardrails.md) |

---

## 6. Submitting a Pull Request

1. **Create a topic branch** from `main` (`git checkout -b feat/your-feature`).
2. **Run all package and workspace quality gates** (see [§3](#3-workspace-setup-local-development--verification-commands) and the relevant package `CONTRIBUTING.md`).
3. **Verify `AGENTS.md` byte budgets and Markdown link integrity** via `moon run trainer:test`.
4. **Sign off your commit** (`git commit -s -m "feat(scope): summary"`).
5. **Open a Pull Request** against `main` describing the motivation, architectural invariants preserved, and benchmark/test results.
