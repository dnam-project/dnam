# `packages/trainer/` — PyTorch Trainer & Capture CLI (`dnam_trainer`)

**`dnam-trainer`** implements the 4-Stage PyTorch Trainer, `.dnam-ckpt` SVD-Kronecker Solver, 3-Phase Control Audit, Multi-Model Project Manager, and Capture Pipeline (`dnam` / `dnam-train`).

## 1. Module Routing Map

| Module | Responsibility | Scoped `AGENTS.md` / Canonical Spec |
| :--- | :--- | :--- |
| [`schema.py`](src/dnam_trainer/schema.py), [`events.py`](src/dnam_trainer/events.py) | Pydantic v2 contracts (`StemAudioRef`) & SSE events | [`dnam_container_schema.md`](../../docs/dnam_container_schema.md) |
| [`project/`](src/dnam_trainer/project/) | `DnamProject`, capture planner & `DnamEngine` runner | [`project/AGENTS.md`](src/dnam_trainer/project/AGENTS.md) |
| [`capture/`](src/dnam_trainer/capture/) | 60s excitation, latency alignment & `CaptureBackend` | [`capture/AGENTS.md`](src/dnam_trainer/capture/AGENTS.md) |
| [`dsp/`](src/dnam_trainer/dsp/) | Differentiable `PolyADAA`, `SelectiveRgLru` & `ModulatedUfdn` (`CL-01`–`CL-11`) | [`dsp/AGENTS.md`](src/dnam_trainer/dsp/AGENTS.md) |
| [`control/`](src/dnam_trainer/control/) | Lobatto HDMR, Rank-4 FM & 3-Phase Audit (`CL-12`–`CL-17`) | [`control/AGENTS.md`](src/dnam_trainer/control/AGENTS.md) |
| [`training/`](src/dnam_trainer/training/) | Runtime, Surrogate Delay, `float64` Kronecker solver & `TrainingBackend` (`CL-18`) | [`training/AGENTS.md`](src/dnam_trainer/training/AGENTS.md) |
| [`cli/`](src/dnam_trainer/cli/) | `cyclopts` + `rich` CLI (`init`, `status`, `inspect`, `render`, `model`, `control`, `preset`, `stem`, `capture`, `train`) | [`README.md`](README.md) |
| [`tests/`](tests/) | 1:1 mirrored unit, Hypothesis, guardrail, verification & benchmark tests | [`tests/AGENTS.md`](tests/AGENTS.md) |

## 2. Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for strict typing (`pyrefly`), coverage ($\ge 95\%$), and precision rules.
