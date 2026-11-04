# `src/dynamic` — Block-Level Dynamic Backend Engine Dispatcher

Implements runtime hardware backend selection (`DynamicDnamEngine<'a>`):
- Block-level dynamic dispatch with **zero per-sample branching**.
- Supports Scalar, ARM64 NEON, ARM SME, x86_64 AVX2/SSE, and WASM SIMD128 backends.
- Automatic CPU feature detection and model hot-swapping.

## 1. Submodule Map

| Component / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| `dispatch` | Block-level engine forwarding and tier / bounce property getters/setters | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for embedded memory budgets and real-time execution safety.
