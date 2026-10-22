# Contributing to Neural Audio Modeler (NAM A3)

Welcome to the **Neural Audio Modeler (Architecture 3)** monorepo. All contributors (human and AI agents) must follow these engineering, numerical, and documentation standards.

---

## 1. Contributor License Agreement & Git Sign-Off

All commits must include a `Signed-off-by` trailer (`git commit -s`) certifying agreement with [`CLA.md`](CLA.md).

---

## 2. Core Engineering & Numerical Rules

1. **Zero `FSQRT` / `FDIV` in Real-Time Paths:** Never introduce floating-point division (`/`), square-root (`sqrt`), or `libm` transcendental calls (`sin`, `cos`, `exp`, `tanh`) into the $48\text{ kHz}$ per-sample loop or the $3\text{ kHz}$ control-bus rotor update. Use the exact algebraic factorizations specified in [`core1_hybrid_ssm_and_polyadaa.md`](docs/core1_hybrid_ssm_and_polyadaa.md).
2. **128-Bit SIMD Lane Alignment:** Channel counts ($C$) and FDN delay line counts ($M$) must be exact multiples of 4 (`C=4, M=4` for Lite; `C=8, M=8` for Full) so every layer maps cleanly onto 4-lane `float32x4_t` (ARM NEON / WASM SIMD128) and `f32x4` / `f32x8` (AVX2) registers.
3. **Unconditional Stability & Registered Clamps:** Every pole radius, FDN loop gain, all-pass coefficient, and user control range must be bounded by a registered clamp in [`clamp_and_audit_registry.md`](docs/clamp_and_audit_registry.md) and verified by the automated test suite in [`verification_and_guardrails.md`](docs/verification_and_guardrails.md).
4. **Fail-Fast Validation:** Never silently substitute default controls, sample rates, or tensor shapes when loading `.nam` v3 or `.nam-ckpt` files; raise explicit diagnostic errors.

---

## 3. Maintaining `AGENTS.md` & Progressive-Disclosure Documentation

1. **Byte Budgets:**
   - Root [`AGENTS.md`](AGENTS.md) must remain strictly `< 5,000` bytes.
   - Every directory-scoped `AGENTS.md` file created during monorepo implementation must remain strictly `< 2,500` bytes.
2. **Single Source of Truth (SSOT):**
   - `AGENTS.md` files are routing maps only. Place all mathematical derivations, clamp tables (`CL-XX`), and JSON schemas in [`docs/`](docs/index.md) and link to them.
   - All relative Markdown links across `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, and `docs/**/*.md` must resolve to existing files on disk.
