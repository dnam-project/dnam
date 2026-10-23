# Dynamic Neural Audio Modeler (DNAM) — Architecture 3 ("Tri-Physics" Monorepo)

**Dynamic Neural Audio Modeler (DNAM Architecture 3 / A3)** is a real-time true-stereo neural audio modeling, differentiable DSP, and parametric system-identification platform co-designed for embedded ARMv8.2-A NEON processors (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins, and browser/Node.js **WebAssembly (`AudioWorklet`)** runtimes.

It unifies **nonlinear amplifier/preamp drive**, **selective state-space filters (Wah, Phaser, Uni-Vibe, power-amp sag)**, **modulated delay-line effects (Through-Zero Flanger, Stereo Chorus, Tape Echo, Spring Reverb Drip, Lush Hall Reverb, Leslie Rotary)**, and **user-definable continuous (`range`) and discrete (`switch`) controls** trained from multi-stem captures—running **$2.6\times$ faster than NAM A2** ($\sim 4,625\text{ MACs/sample}$ vs. $12,146\text{ MACs/sample}$) with **$0\text{ extra MACs/sample}$ at $48\text{ kHz}$ when user knobs are static**.

---

## Target Monorepo Blueprint

| Component | Target Path | Description |
| :--- | :--- | :--- |
| **Canonical Specifications** | [`docs/`](docs/index.md) | Pure Single-Source-of-Truth (SSOT) mathematical, DSP, training, clamp registry (`CL-01..CL-18`), and `.dnam` v3 container specifications. |
| **Rust SIMD Engine Core** | `packages/engine/` | Zero-allocation, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full), `FTZ`/`DAZ`-enabled pure Rust engine (`dnam-engine`) with zero `FSQRT` or `FDIV` instructions at $48\text{ kHz}$ and $3\text{ kHz}$. |
| **Packaged Python Engine Library** | `packages/python/` | Standalone `PyO3` / `maturin` Python wheel wrapping `packages/engine` for zero-copy NumPy audio inference, `.dnam` v3 validation, 3-Phase control clamping, and canonical Reference DSP primitives. |
| **Packaged Node.js / WebAudio Library** | `packages/nodejs/` | Standalone TypeScript / npm package wrapping `packages/engine` via WebAssembly SIMD128 & N-API bindings + WebAudio `DnamAudioWorkletProcessor`. |
| **PyTorch Differentiable Trainer** | `packages/trainer/` | PyTorch 4-stage deterministic trainer, `.dnam-ckpt` factorized Kronecker iterative stem expansion, Non-Stop 3-Phase Control-Space Clamping Audit, and 8-suite verification tests (imports `packages/python` as Reference DSP). |
| **Real-Time Audio Plugin** | `packages/plugin/` | VST3/CLAP/Standalone audio plugin wrapping `packages/engine` with lock-free atomic parameter binding for dynamic `.dnam` v3 `controls_schema` (`range` & `switch`) and ZOH sample-rate rescaling ($44.1\text{--}96\text{ kHz}$). |
| **Web Docs & React WASM Studio** | `packages/web/` | Astro + Starlight documentation website mounting `docs/*.md` as a comprehensive Specification section with interactive React/Canvas DSP visualizers + live React WASM (`AudioWorklet` SIMD128) Modeler Plugin. |

---

## Hardware & DSP Benchmark (NAM A2 vs. DNAM A3)

| Hardware / DSP Metric | NAM A2 Lite (Mono) | **DNAM A3 Lite (Stereo)** | NAM A2 Full (Mono) | **DNAM A3 Full (Stereo)** |
| :--- | :---: | :---: | :---: | :---: |
| **Channels ($C$) / FDN Lines ($M$)** | $C = 3$ (Unaligned) | **$C = 4$ (1 NEON reg), $M = 4$** | $C = 8$, No FDN | **$C = 8$ (2 NEON regs), $M = 8$** |
| **Core 1 Architecture** | 23 Dilated Conv | **8 Conv + 4 Selective RG-LRU** | 23 Dilated Conv | **8 Conv + 4 Selective RG-LRU** |
| **User Controls (`range` & `switch`)** | None (Static Snapshot) | **Unlimited ($0\text{ MACs}$ @ $48\text{ kHz}$ static)** | None (Static Snapshot) | **Unlimited ($0\text{ MACs}$ @ $48\text{ kHz}$ static)** |
| **$48\text{ kHz}$ `FSQRT` / `FDIV` Instructions** | $0$ | **$0$ (100% `FMAX`/`FMIN`/`FMLA`)** | $0$ | **$0$ (100% `FMAX`/`FMIN`/`FMLA`)** |
| **NEON SIMD Lane Utilization** | $75\%$ ($3/4$ lanes) | **$100\%$ ($4/4$ lanes)** | $100\%$ ($8/8$ lanes) | **$100\%$ ($8/8$ lanes)** |
| **Core 1 L1 Cache State RAM** | $76.2\text{ KB}$ | **$3.7\text{ KB}$ ($-95\%$)** | $203.1\text{ KB}$ | **$7.5\text{ KB}$ ($-96\%$)** |
| **Core 2 Delay Ring Buffer RAM** | $0\text{ KB}$ | **$96\text{ KB}$ (`fp16`)** | $0\text{ KB}$ | **$208\text{ KB}$ (`fp16`)** |
| **Effective Receptive Field** | $132.2\text{ ms}$ (Hard cutoff) | **$\infty$ (Core 1 SSM + Core 2 FDN)** | $132.2\text{ ms}$ (Hard cutoff) | **$\infty$ (Core 1 SSM + Core 2 FDN)** |
| **Total Compute (MACs / Sample)** | $1,871\text{ MACs}$ | **$\sim 1,245\text{ MACs}$ ($1.5\times$ faster + $33\%$ wider)** | $12,146\text{ MACs}$ | **$\sim 4,625\text{ MACs}$ ($2.6\times$ faster)** |
| **Sample-Rate Support** | Fixed $48\text{ kHz}$ | **Native $44.1\text{--}96\text{ kHz}$ (ZOH)** | Fixed $48\text{ kHz}$ | **Native $44.1\text{--}96\text{ kHz}$ (ZOH)** |
| **Audio I/O Topology** | Mono ($1 \to 1$) | **True Stereo ($1 \to 2$ or $2 \to 2$)** | Mono ($1 \to 1$) | **True Stereo ($1 \to 2$ or $2 \to 2$)** |

---

## Comparison with GuitarML PROTEUS & NAM ParametricOD

See [`docs/architecture_overview.md` §3](docs/architecture_overview.md#3-comparison-with-prior-art-nam-a2-guitarml-proteus--nam-parametricod) for the full technical comparison:

| Capability | **GuitarML PROTEUS** | **NAM ParametricOD** | **DNAM Architecture 3 ("Tri-Physics")** |
| :--- | :---: | :---: | :---: |
| **Backbone** | 1-layer LSTM (`hidden=40`) | Conditional WaveNet / LSTM | **Core 1 (`PolyADAA` + Selective RG-LRU) + Core 2 (`M-UFDN`)** |
| **Control Schema** | 1–2 continuous knobs | Hardcoded pedal layout | **Unlimited user-defined `range` knobs & `switch` selectors** |
| **Control Rate & Static Overhead** | $48\text{ kHz}$ ($\propto P$ MACs/sample) | $48\text{ kHz}$ ($\propto P$ MACs/sample) | **$3\text{ kHz}$ Lobatto HDMR-FM ($0\text{ MACs/sample}$ @ $48\text{ kHz}$ when static)** |
| **Anti-Aliasing** | None (`tanh` at $1\times$ rate) | None (`tanh`/`ReLU` at $1\times$ rate) | **$1\times$ B-Spline Division-Free Algebraic `PolyADAA` ($38.87\text{ dB}$ reduction)** |
| **Time-Varying Delay / Reverb / LFOs** | No (double-echo ghosting) | No (FIR $< 30\text{ ms}$) | **Yes (Autonomous Chebyshev LFO + Order-Statistic Transport on `M-UFDN`)** |
| **Control-Space Safety & Clamping** | None | Hardcoded plugin bounds | **Non-Stop 3-Phase Clamping (`RANGE`, `SWITCH`, `STABILITY`, `DYNAMIC`)** |
| **Iterative Stem Addition** | Retrain from scratch | Retrain from scratch | **Single `.dnam-ckpt` Factorized-Kronecker Resume ($< 5\text{ s}$, $0.00$ old-stem drift)** |

---

## Canonical Knowledge Base (`docs/`)

Start with [`AGENTS.md`](AGENTS.md) and [`docs/index.md`](docs/index.md):

1. **[Master Index & Agent Implementation Roadmap](docs/index.md)**
2. **[Complete Architecture 3 Design Artifact (`architecture_3_design.md`)](docs/architecture_3_design.md)**
3. **[Executable 8-Suite Empirical Verification Script (`verify_a3_architecture.py`)](scripts/verify_a3_architecture.py)**
4. **[Architecture 3 Overview & 18 Verified Breakthroughs](docs/architecture_overview.md)**
5. **[Core 1: Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) & Selective Complex RG-LRU](docs/core1_hybrid_ssm_and_polyadaa.md)**
6. **[Core 2: Unified Stereo In-Loop Diffusive Modulated Unitary FDN (`M-UFDN`)](docs/core2_modulated_ufdn.md)**
7. **[Parametric Control Bus (`Range` & `Switch` Controls, Lobatto HDMR-FM & Order-Statistic Transport)](docs/parametric_control_bus.md)**
8. **[4-Stage Training Pipeline & Single-Checkpoint (`.dnam-ckpt`) Iterative Stem Expansion](docs/training_and_iterative_resume.md)**
9. **[Non-Stop 3-Phase Control-Space Audit & Clamp Registry (`CL-01`..`CL-18`)](docs/clamp_and_audit_registry.md)**
10. **[`.dnam` v3 Container Schema & Continuous-Time ZOH Sample-Rate Rescaling](docs/dnam_v3_container_schema.md)**
11. **[Monorepo Implementation Blueprint (Web Docs, Python, Native DSP, Plugin & Node.js WASM)](docs/monorepo_implementation_guide.md)**
12. **[Empirical Verification Suite (8 Test Suites) & Automated Guardrails](docs/verification_and_guardrails.md)**
