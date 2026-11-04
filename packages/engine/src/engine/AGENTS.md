# `src/engine` — TORA Audio Engine Dispatcher & Runtime

Implements the top-level audio engine runtime dispatcher (`#[forbid(unsafe_code)]`):
- Dual-rate execution lifecycle ($48\text{ kHz}$ audio / $3\text{ kHz}$ control bus).
- 4-tier slimmable SIMD dispatch (`Lite` ⊂ `Full` ⊂ `Pro` ⊂ `Ultra`).
- Native Mono (`1 → 1`, `2 → 1`) and True Stereo (`1 → 2`, `2 → 2`) processing.
- Zero audio-loop allocations and L1D-resident state buffer management.

## 1. Submodule Map

| Component / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`process.rs`](process.rs) | Dual-rate audio block callback, channel topology dispatch, and `step_frame_tora` | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`runtime.rs`](runtime.rs) | Model initialization, sample-rate reconfiguration, hot-swap reload, and offline bounce | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`oversampler.rs`](oversampler.rs) | `4x` linear-phase polyphase FIR oversampler (`OFFLINE_BOUNCE_PDC_SAMPLES = 12`) | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`tempo.rs`](tempo.rs) | Host BPM tempo-sync subdivision resolver and octave folding | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for embedded memory budgets and real-time execution safety.
