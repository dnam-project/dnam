# `src/host` — Real-Time Control SPSC Ingress & Wait-Free Telemetry

Lock-free real-time host concurrency primitives (`#[forbid(unsafe_code)]`): 3-Lane SPSC control ingress (`SpscControlRing`, `AcceptedStateShadow`), wait-free Seqlock telemetry (`TelemetryCell`), ping-pong model swap (`RealtimeEngineSwap`), and unified `DnamHostHandle`.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`spsc.rs`](spsc.rs) | 3-Lane SPSC re-exports | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`spsc/ring.rs`](spsc/ring.rs) | 64B-aligned Lane 1/2/3 rings & packing | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`spsc/shadow.rs`](spsc/shadow.rs) | `AcceptedStateShadow` clamping & roles | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`spsc/producer.rs`](spsc/producer.rs) | Non-RT `ControlProducer` & presets | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`spsc/consumer.rs`](spsc/consumer.rs) | Real-time `ControlConsumer` drain | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`telemetry.rs`](telemetry.rs) | Wait-free Seqlock `TelemetryCell` | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`swap.rs`](swap.rs) | Off-thread model swap & reclamation | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`handle.rs`](handle.rs) | `DnamHostHandle` root & mutex | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`handle/lifecycle.rs`](handle/lifecycle.rs) | Init, hot-reload, rate, telemetry | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`handle/controls.rs`](handle/controls.rs) | Non-blocking control & tempo API | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`handle/process.rs`](handle/process.rs) | Sub-block audio & offline bounce | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for real-time execution safety invariants.


