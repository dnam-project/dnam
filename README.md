# Dynamic Neural Audio Modeler (DNAM)

**Dynamic Neural Audio Modeler (DNAM)** is a real-time true-stereo neural audio modeling, differentiable DSP, and parametric system-identification platform co-designed for embedded ARMv8.2-A NEON processors (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins, and browser/Node.js **WebAssembly (`AudioWorklet`)** runtimes.

It unifies **nonlinear amplifier/preamp drive**, **selective state-space filters (Wah, Phaser, Uni-Vibe, power-amp sag)**, **modulated delay-line effects (Through-Zero Flanger, Stereo Chorus, Tape Echo, Spring Reverb Drip, Lush Hall Reverb, Leslie Rotary)**, and **user-definable continuous (`range`) and discrete (`switch`) controls** trained from multi-stem captures—running **2.6× faster than NAM A2** in `Full` (`C=8`, ~4,625 MACs/sample vs. 12,146 MACs/sample) and **1.22× faster with 50% more channels** in `Ultra` (`C=12`, ~9,950 MACs/sample), with **0 extra MACs/sample at 48 kHz when user knobs are static**.

---

## Target Monorepo Blueprint

| Component | Target Path | Description |
| :--- | :--- | :--- |
| **Canonical Specifications** | [`docs/`](docs/index.md) | Pure Single-Source-of-Truth (SSOT) mathematical, DSP, training, clamp registry (`CL-01..CL-18`), and `.dnam` container specifications. |
| **Rust SIMD Engine Core** | `packages/engine/` | Zero-allocation, 16-byte SIMD-aligned (`C=4` Lite / `C=8` Full / `C=12` Ultra), `FTZ`/`DAZ`-enabled pure Rust engine (`dnam-engine`) with zero `FSQRT` or `FDIV` instructions at 48 kHz and 3 kHz. |
| **Packaged Python Engine Library** | `packages/python/` | Standalone `PyO3` / `maturin` Python wheel wrapping `packages/engine` for zero-copy NumPy audio inference, `.dnam` validation, 3-Phase control clamping, and canonical Reference DSP primitives. |
| **Packaged Node.js / WebAudio Library** | `packages/js/` | Standalone TypeScript / npm package (`@dnam-project/engine`) wrapping `packages/engine` via WebAssembly SIMD128 & N-API bindings + WebAudio `DnamAudioWorkletProcessor`. |
| **PyTorch Differentiable Trainer** | `packages/trainer/` | PyTorch 4-stage deterministic trainer, `.dnam-ckpt` factorized Kronecker iterative stem expansion, Non-Stop 3-Phase Control-Space Clamping Audit, and 8-suite verification tests (imports `packages/python` as Reference DSP). |
| **Real-Time Audio Plugin** | `packages/plugin/` | VST3/CLAP/Standalone audio plugin wrapping `packages/engine` with lock-free atomic parameter binding for dynamic `.dnam` `controls_schema` (`range` & `switch`) and ZOH sample-rate rescaling (44.1–96 kHz). |
| **Web Docs & React WASM Studio** | `packages/web/` | Astro + Starlight documentation website mounting `docs/*.md` as a comprehensive Specification section with interactive React/Canvas DSP visualizers + live React WASM (`AudioWorklet` SIMD128) Modeler Plugin. |

---

## Hardware & DSP Benchmark (NAM A2 vs. DNAM)

| Hardware / DSP Metric | NAM A2 Lite (Mono) | **DNAM Lite (Stereo)** | NAM A2 Full (Mono) | **DNAM Full (Stereo)** | **DNAM Ultra (Stereo)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Channels (`C`) / FDN Lines (`M`)** | `C = 3` (Unaligned) | **`C = 4` (1 NEON reg), `M = 4`** | `C = 8`, No FDN | **`C = 8` (2 NEON regs), `M = 8`** | **`C = 12` (3 NEON regs), `M = 8`** |
| **Core 1 Architecture** | 23 Dilated Conv | **8 Conv + 4 Selective RG-LRU** | 23 Dilated Conv | **8 Conv + 4 Selective RG-LRU** | **8 Conv + 4 Selective RG-LRU** |
| **User Controls (`range` & `switch`)** | None (Static Snapshot) | **Unlimited (0 MACs @ 48 kHz static)** | None (Static Snapshot) | **Unlimited (0 MACs @ 48 kHz static)** | **Unlimited (0 MACs @ 48 kHz static)** |
| **48 kHz `FSQRT` / `FDIV` Instructions** | 0 | **0 (100% `FMAX`/`FMIN`/`FMLA`)** | 0 | **0 (100% `FMAX`/`FMIN`/`FMLA`)** | **0 (100% `FMAX`/`FMIN`/`FMLA`)** |
| **NEON SIMD Lane Utilization** | 75% (3/4 lanes) | **100% (4/4 lanes)** | 100% (8/8 lanes) | **100% (8/8 lanes)** | **100% (12/12 lanes, `8+4` tiled)** |
| **Core 1 L1 Cache State RAM** | 76.2 KB | **3.7 KB (-95%)** | 203.1 KB | **7.5 KB (-96%)** | **11.2 KB (-94%)** |
| **Core 2 Delay Ring Buffer RAM** | 0 KB | **96 KB (`fp16`)** | 0 KB | **208 KB (`fp16`)** | **208 KB (`fp16`)** |
| **Effective Receptive Field** | 132.2 ms (Hard cutoff) | **∞ (Core 1 SSM + Core 2 FDN)** | 132.2 ms (Hard cutoff) | **∞ (Core 1 SSM + Core 2 FDN)** | **∞ (Core 1 SSM + Core 2 FDN)** |
| **Total Compute (MACs / Sample)** | 1,871 MACs | **~1,245 MACs (1.5× faster + 33% wider)** | 12,146 MACs | **~4,625 MACs (2.6× faster)** | **~9,950 MACs (1.22× faster + 50% wider)** |
| **Sample-Rate Support** | Fixed 48 kHz | **Native 44.1–96 kHz (ZOH)** | Fixed 48 kHz | **Native 44.1–96 kHz (ZOH)** | **Native 44.1–96 kHz (ZOH)** |
| **Audio I/O Topology** | Mono (1 → 1) | **True Stereo (1 → 2 or 2 → 2)** | Mono (1 → 1) | **True Stereo (1 → 2 or 2 → 2)** | **True Stereo (1 → 2 or 2 → 2)** |

---

## Comparison with GuitarML PROTEUS & NAM ParametricOD

See [`docs/architecture_overview.md` §3](docs/architecture_overview.md#3-comparison-with-prior-art-nam-a2-guitarml-proteus--nam-parametricod) for the full technical comparison:

| Capability | **GuitarML PROTEUS** | **NAM ParametricOD** | **DNAM ("Tri-Physics")** |
| :--- | :---: | :---: | :---: |
| **Backbone** | 1-layer LSTM (`hidden=40`) | Conditional WaveNet / LSTM | **Core 1 (`PolyADAA` + Selective RG-LRU) + Core 2 (`M-UFDN`)** |
| **Control Schema** | 1–2 continuous knobs | Hardcoded pedal layout | **Unlimited user-defined `range` knobs & `switch` selectors** |
| **Control Rate & Static Overhead** | 48 kHz (∝ `P` MACs/sample) | 48 kHz (∝ `P` MACs/sample) | **3 kHz Lobatto HDMR-FM (0 MACs/sample @ 48 kHz when static)** |
| **Anti-Aliasing** | None (`tanh` at 1× rate) | None (`tanh`/`ReLU` at 1× rate) | **1× B-Spline Division-Free Algebraic `PolyADAA` (38.87 dB reduction)** |
| **Time-Varying Delay / Reverb / LFOs** | No (double-echo ghosting) | No (FIR < 30 ms) | **Yes (Autonomous Chebyshev LFO + Order-Statistic Transport on `M-UFDN`)** |
| **Control-Space Safety & Clamping** | None | Hardcoded plugin bounds | **Non-Stop 3-Phase Clamping (`RANGE`, `SWITCH`, `STABILITY`, `DYNAMIC`)** |
| **Iterative Stem Addition** | Retrain from scratch | Retrain from scratch | **Single `.dnam-ckpt` Factorized-Kronecker Resume (< 5 s, 0.00 old-stem drift)** |

---

## Canonical Knowledge Base (`docs/`)

Start with [`AGENTS.md`](AGENTS.md) and [`docs/index.md`](docs/index.md):

1. **[Master Index & Agent Implementation Roadmap](docs/index.md)**
2. **[Complete DNAM Architecture Design Specification (`architecture_design.md`)](docs/architecture_design.md)**
3. **[Executable 8-Suite Empirical Verification Script (`verify_dnam_architecture.py`)](scripts/verify_dnam_architecture.py)**
4. **[DNAM Architecture Overview & 18 Verified Breakthroughs](docs/architecture_overview.md)**
5. **[Core 1: Hybrid Dilated-Conv (`1x B-Spline PolyADAA`) & Selective Complex RG-LRU](docs/core1_hybrid_ssm_and_polyadaa.md)**
6. **[Core 2: Unified Stereo In-Loop Diffusive Modulated Unitary FDN (`M-UFDN`)](docs/core2_modulated_ufdn.md)**
7. **[Parametric Control Bus (`Range` & `Switch` Controls, Lobatto HDMR-FM & Order-Statistic Transport)](docs/parametric_control_bus.md)**
8. **[4-Stage Training Pipeline & Single-Checkpoint (`.dnam-ckpt`) Iterative Stem Expansion](docs/training_and_iterative_resume.md)**
9. **[Non-Stop 3-Phase Control-Space Audit & Clamp Registry (`CL-01`..`CL-18`)](docs/clamp_and_audit_registry.md)**
10. **[`.dnam` Container Schema & Continuous-Time ZOH Sample-Rate Rescaling](docs/dnam_container_schema.md)**
11. **[Digital Marketplace Architecture, Split-Payload `Ed25519` Signatures & Traitor-Tracing Watermarks](docs/digital_marketplace_and_provenance.md)**
12. **[Monorepo Implementation Blueprint (Web Docs, Python, Native DSP, Plugin & Node.js WASM)](docs/monorepo_implementation_guide.md)**
13. **[Empirical Verification Suite (8 Test Suites) & Automated Guardrails](docs/verification_and_guardrails.md)**

---

## Contributing & License

We welcome contributions! All commits must be signed off (`git commit -s`) under our [Contributor License Agreement (`CLA.md`)](CLA.md). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for engineering, numerical, and documentation standards, and [`AGENTS.md`](AGENTS.md) for AI agent discovery guidelines.

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`LICENSE`)](LICENSE):
- **Hardware & Closed-Source Plugin Linking Allowed:** Hardware manufacturers and software developers may link `dnam-engine` (and `packages/python`, `packages/js`) into commercial hardware firmware and closed-source audio plugins without open-sourcing their own host or plugin code.
- **Individual Creator Publishing & Direct Sales Allowed:** Creators retain full ownership of their trained `.dnam` models (trained locally or on their own rented cloud GPUs) and may freely publish or commercially sell their own first-party models individually (e.g., via their own website, Shopify/Gumroad/Bandcamp/Patreon storefront, or bundled with compliant hardware/plugins).
- **Mandatory DNAM Branding, Format & Provenance Integrity:** Products linking `dnam-engine` or loading `.dnam` models must retain user-facing **"DNAM"** ("Dynamic Neural Audio Modeler" / **"Powered by DNAM"**) branding, unmodified `.dnam` container compatibility, and intact `Ed25519` signatures, device authorization, and buyer watermarks (no white-labeling or DRM/watermark circumvention).
- **No Unlicensed Marketplaces, Hosted Cloud Training, or Bulk Distillation:** Repackaging the standalone reference plugin (`packages/plugin`), trainer (`packages/trainer`), or web studio (`packages/web`), offering third-party hosted cloud training-as-a-service, operating an unlicensed multi-vendor `.dnam` model marketplace or aggregation hub, or bulk-distilling third-party `.dnam` models into competing formats is prohibited without a commercial license.

For bespoke OEM, marketplace, cloud-training, or commercial licensing inquiries, contact **Peter Nguyen** (<peter@phn.dev>).
