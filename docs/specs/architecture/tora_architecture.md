---
id: spec.arch.tora
title: "Tri-Operator Recurrent Architecture (TORA) System Overview"
status: canonical
version: "1.0.0"
domain: architecture
rule_prefixes:
  - "r[arch."
requirements:
  - REQ-1.1
  - REQ-1.2
  - REQ-1.3
  - REQ-1.4
  - REQ-1.5
  - REQ-5.1
  - REQ-5.4
  - REQ-5.5
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
  - REQ-7.1
  - REQ-7.2
  - REQ-7.4
adrs:
  - ADR-0002
  - ADR-0015
---

# Tri-Operator Recurrent Architecture (`TORA`) System Overview

This specification defines the **Tri-Operator Recurrent Architecture (`TORA`)** for **Dynamic Neural Audio Modeler (DNAM)** across [`packages/engine/`](../../../packages/engine/README.md) and [`packages/trainer/`](../../../packages/trainer/README.md). TORA unifies the modeling of all analog and digital audio gear across **three universal zero-`FDIV` SIMD operator kernels** (`Op-T`, `Op-M`, `Op-N`), four canonical physical stages, an **8-Port Matrix Router** (`Stage 0`), a **Self-Similar 3 kHz Dual-Rate Control Plane**, and a **256-D Mid/Side Kronecker Output Head**.

For implementation-agnostic requirements, see [`../../requirements/system_goals_and_requirements.md`](../../requirements/system_goals_and_requirements.md). For psychoacoustic quality tiers, see [`psychoacoustic_quality_and_tiers.md`](psychoacoustic_quality_and_tiers.md). For architectural decision records, see [`../../adr/0002-tri-operator-recurrent-architecture.md`](../../adr/0002-tri-operator-recurrent-architecture.md).

---

## 1. Executive Summary & Core Architectural Pillars

### 1.1 Three Foundational Pillars of `TORA`

The **Tri-Operator Recurrent Architecture (`TORA`)** unifies all physical audio hardware into **three universal zero-`FDIV` SIMD operator kernels** (`Op-T`, `Op-M`, `Op-N`) orchestrated across **four canonical physical stages**, an **8-Port Matrix Router**, and a **Unified Dual-Rate Self-Similar Control Plane**:

1. `r[arch.pillars.dual-space-control]` **Pillar 1: Dual-Space Factored Control Architecture:**  
   Factors kinematic controls and nonlinear drive to support progressive stem refinement ([`../training/iterative_resume_and_caching.md` §1.2](../training/iterative_resume_and_caching.md#12-conditions-for-old-stem-trajectory--arena-4-gram-reuse-on-stem-resume-req-3)).
2. `r[arch.pillars.dual-rate-plane]` **Pillar 2: Unified Self-Similar Dual-Rate Control Plane (`Op-M[Ctrl]` & `Op-N[Ctrl]`):**  
   Executes the control plane using the `Op-M` and `Op-N` kernels at a 16-sample stride at the `48 kHz` reference rate (`3 kHz`), and the rate-dependent stride $K_{\text{ctrl}}(f_{s,\text{eff}})$ at other effective rates ([router §9](../dsp/router_and_stages_topology.md#9-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--4-bounce)).
3. `r[arch.pillars.cp1-projections]` **Pillar 3: Shared Control Descriptors & Tier-Local `CP1` Projections:**  
   Supports $P \ge 0$ physical controls using shared descriptors and packed tier-local `f32` projection slices ([`../containers/dnam_binary_format.md` §2.4](../containers/dnam_binary_format.md#packed-control-projection-profile)). `Lite`, `Full`, and `Pro` use `8-D` static latents (`4` head + `4` stage, Rank-4 FM); `Ultra` adds an `8-D` delta (`+4` head + `+4` stage, Rank-8 FM) and a `16 × 4` stage-dispatch delta in `Arena 3` ([`../dsp/parametric_control_bus.md` §2.3](../dsp/parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)).

### 1.2 The TORA Architectural Breakthrough

```mermaid
flowchart TD
    subgraph CtrlPlane ["Nominal 3 kHz Control Plane (16 Samples at 48 kHz Reference)"]
        direction TB
        CtrlTable["Shared Control Descriptors +<br/>Packed Tier-Local CP1 f32 Projections"]
        OpNCtrl["Op-N[Ctrl]: Exact-Integer Sufficient-Statistic Engine<br/>• Lite/Full/Pro: 8-D (4 head + 4 stage, Rank-4 FM)<br/>• Ultra: 16-D (8 head + 8 stage, Rank-8 FM)<br/>• Outward-Checked Trust-Region Clip"]
        SendMod["Send-Safe Instrument / Independent-Key Dynamics"]
        DownMod["Downstream Instrument / Return / Key Dynamics"]
        SendBus["Send Dispatch Rows 0..4<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
        DownBus["Downstream Dispatch Rows 5..15<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
        CtrlTable --> OpNCtrl
        OpNCtrl & SendMod --> SendBus
        OpNCtrl & DownMod --> DownBus
    end

    subgraph Router ["Stage 0: 8-Port Matrix Router (return-to-send isolation)"]
        direction TB
        RIn["Inputs & Loop Alignment:<br/>• Port In 0: Inst In L<br/>• Port In 1: Inst In R<br/>• Port In 2: FX Return L<br/>• Port In 3: FX Return R / Aux Key"]
        ROut["Mode-dependent outputs:<br/>• Port Out 0: Send L<br/>• Port Out 1: Send R / WDW dry"]
    end

    subgraph Stages ["Four Canonical Physical Stages (48 kHz Audio Rate)"]
        direction LR
        S1["Stage I: Pre-FX / Synth / Mod<br/>(Op-T[Pre], Op-M[Pre], Op-N[Pre])"]
        S2["Stage II: 3-Stage LTI-Commuted<br/>Nonlinear Amp Cascade (Op-N)"]
        S3["Stage III: 64-Tap Early SIMD FIR<br/>+ Shared Kautz Modal Bank (Op-M)"]
        S4["Stage IV: Unitary Spatial &<br/>Barberpole Tank (Op-T[Tank])"]
        S1 -->|"Downstream LR4 high when split"| S2 --> S3 --> S4
    end

    RIn -->|"Instrument inputs only"| S1
    S1 -->|"Send L/R when enabled"| ROut
    DryTap["WDW dry selector<br/>PreSpatialCore / DirectDry"] -->|"WDW"| ROut
    RIn -->|"Aligned return: LR4 high when split"| S2
    RIn -->|"LR4 low of selected downstream source"| S3
    RIn -->|"Instrument / independent key"| SendMod
    RIn -->|"Downstream sources"| DownMod

    DownBus --> S2 & S3 & S4
    SendBus --> Router & S1

    subgraph FeatBus ["Unified 256-D Feature Bus Phi[n]"]
        direction LR
        FB["Lite: ch [0..63] (64 D) | Full: ch [0..127] (128 D) | Pro: ch [0..191] (192 D) | Ultra: ch [0..255] (256 D)"]
    end

    S1 & S2 & S3 & S4 --> FB

    subgraph MixHead ["Output Projection & Stereo Mixer"]
        direction LR
        KHead["256-D 4-Tier SVD Kronecker Head<br/>W_head(z) @ Phi[n] (2x256 FMAs, Canonical Mid/Side Slice 0 & Slice 1)"]
        TankMix["Direct Tank Stereo<br/>Wet/Dry Mixer"]
    end

    FB --> KHead
    S4 --> TankMix
    DownBus --> KHead
    KHead & TankMix --> MainOut["Output Ports:<br/>• Port Out 2: Main L / WDW wet L<br/>• Port Out 3: Main R / WDW wet R"]
```

#### Dual-Rate (`48 kHz` Audio vs. `3 kHz` Control) TORA Execution Lifecycle

`r[arch.lifecycle.dual-rate]`

Host parameter updates apply at the control boundary in timestamp order, with DAW automation taking precedence over live Studio/MIDI on exact-timestamp ties ([control bus §1.1](../dsp/parametric_control_bus.md#host-automation-timing)):

```mermaid
sequenceDiagram
    autonumber
    participant Host as Audio Host / 8-Port Router
    participant CtrlM as Op-M[Ctrl] (3 kHz Symplectic & Ballistic)
    participant CtrlN as Op-N[Ctrl] (3 kHz Polynomial & FM)
    participant Bus as Tier-aware modulation routing
    participant Audio as Four Stages I-IV & Kronecker Head (48 kHz)

    Host->>CtrlM: Complete previous detector window at control boundary
    CtrlM->>CtrlM: Advance Q0.192 LFOs with previous-window parameters or hold zero increment
    CtrlM->>CtrlM: Update 4-Band block-peak envelopes from the completed window
    alt tempo_dirty != 0x00 (Host BPM / MIDI 0xF8 / Tap)
        CtrlM->>CtrlM: Update Q64 timeline, Q32 subdivision targets, and phasor recovery
    end
    CtrlM->>Bus: Emit 8-D Dynamic Modulators m_dyn = [LFO_I, LFO_Q, LFO_sh0, LFO_sh1, Env_L, Env_M, Env_H, Env_Aux]^T

    alt is_dirty != 0x00 (Controls Moved in Current Frame)
        CtrlN->>CtrlN: Evaluate CP1 contributor deltas into exact-integer sufficient statistics
        CtrlN->>CtrlN: Mix Rank-4/8 FM interactions and apply outward-checked trust clip
    end
    CtrlN->>Bus: Emit tier-specific head (4 or 8) and stage-control (4 or 8) latents

    Bus->>Bus: Route static latents and 8 dynamic values
    Bus->>Audio: Dispatch 16x16 base + Ultra 16x4 stage parameters and collapse head modes
    Bus->>Audio: Compute division-free ramp delta_theta = (theta_next - theta_curr) * alpha_ramp

    loop For each sample n in 16-sample sub-block (48 kHz)
        Audio->>Audio: theta_stage = theta_stage + delta_theta (Micro-Ramping FMA)
        Host->>Audio: Route 8-Port Matrix S_router (S_{ret->send} == 0) & configured parallel alignment
        Audio->>Audio: Stage I Pre-FX (Op-T/M/N) -> Loop Send L/R -> Stage II Multi-Drive Cascade
        Audio->>Audio: Stage III SIMD Early FIR + Kautz Modals -> Stage IV Unitary Tank
        Audio->>Host: Stream Phi[n] into 256-D Mid/Side Kronecker Head -> Port Out 2 & 3
    end
```

#### Floating-Point Environment at the Audio Callback

`r[arch.fp.callback-env]`

All block-processing entry points enforce round-to-nearest, ties-to-even (`RNE`) with non-trapping floating-point exceptions and subnormal state protection via `FtzDazGuard`:

| Target Architecture | Hardware Control Register Policy (`FtzDazGuard`) |
| :--- | :--- |
| **`aarch64`** | Save `FPCR`; clear `RMode` bits `23:22` and exception trap-enable bits `8:12, 15`; set `FZ` bit `24`; if `AH` bit `1` is set (`FEAT_AFP`), set `FIZ` bit `0`; restore saved `FPCR` on exit. |
| **`x86_64`** | Save `MXCSR`; clear `RC` bits `14:13` and sticky flags `0:5`; set `FTZ` (`15`), `DAZ` (`6`), and exception masks `7:12`; restore saved `MXCSR` on exit. |
| **WebAssembly / Portable** | Standard WASM enforces `RNE` with gradual underflow; apply branchless 4-lane `SimdVec4::flush_subnormal` (`v128` exponent-mask zeroing, `exp == 0 -> ±0.0f32`) to persistent recursive-state writebacks while preserving sign, normals, infinities, and `NaN` payloads. |

#### Finite-Precision Range and Saturation Policy

`r[arch.fp.range-saturation]`

Every model and configuration guarantees an unclipped operating envelope of at least `±100.0` (`+40 dBFS` peak) simultaneously on all active audio inputs (`REQ-7.1`), and finite output/state safety for all finite `f32` inputs via the `±1e9` (`S = 1,000,000,000.0`) and `fp16` (`±65,504.0`) safety guards (`REQ-7.2`):

| Bit | Event Flag (`overflow_or_saturation_flags`) | Condition |
| :---: | :--- | :--- |
| `0` | `INPUT_SATURATED` | A processed audio input exceeded the `±1e9` safety ceiling. |
| `1` | `FP16_SATURATED` | A finite delay write exceeded `±65,504.0` (`0x7BFF` / `0xFBFF`). |
| `2` | `INTERNAL_SATURATED` | An intermediate or recursive-state safety guard clamped a value to `±1e9`. |
| `3` | `OUTPUT_SATURATED` | The processed output guard clamped a value to `±1e9`. |
| `4` | `OUTSIDE_CERTIFIED_INPUT` | An input exceeded the model's certified input envelope (`≥ ±100.0`). |
| `5` | `INVALID_NONFINITE` | A non-finite external sample, control, or parameter was detected and rejected. |

#### Wait-Free Telemetry Seqlock & Real-Time Host Concurrency

`r[arch.telemetry.seqlock-publication]`

Real-time host wrappers (`dnam_engine::host::DnamHostHandle`, [`../../adr/0015-realtime-host-concurrency-spsc-and-telemetry-seqlock.md`](../../adr/0015-realtime-host-concurrency-spsc-and-telemetry-seqlock.md)) decouple audio rendering callbacks from concurrent UI, MIDI, Python (`3.15t`), and TypeScript (`N-API` / `WASM`) control and telemetry threads in 100% safe `#![no_std]` + `alloc` Rust (`#[forbid(unsafe_code)]`):
1. **Wait-Free `AtomicU64` Seqlock (`TelemetryCell`):** `TelemetryWriter` packs `DnamTelemetrySnapshot` (`RawTelemetryRecord`) into a cache-line-aligned (`#[repr(align(64))]`) array of `29` `AtomicU64` words guarded by a monotonically advancing sequence counter `seq: AtomicU64` (`seq` odd before relaxed word stores, `Release` store `seq + 1` on completion). `TelemetryReader::read()` loads `seq1` (`Acquire`), copies the `29` atomic words, issues an `Acquire` fence, and verifies `seq1 == seq2` (with bounded retry and last-good snapshot fallback) without locking `DynamicDnamEngine`.
2. **Sub-Block SPSC Drain & Telemetry Publication:** `DnamHostHandle` drains the 3-Lane `SpscControlRing` (`ref[dsp.ctrl.op-m.spsc-ingress]`) at the start of every `64`-frame sub-block and publishes the updated `DnamTelemetrySnapshot` to `TelemetryCell` at sub-block exit, while `RealtimeEngineSwap` performs non-blocking ping-pong hot-swaps so retired `DynamicDnamEngine` instances are dropped exclusively on the non-real-time controller thread.

---

### 1.3 The 3 Universal Zero-FDIV SIMD Operator Kernels

`r[arch.operators.tri-kernel]`

Every analog, digital, and acoustic circuit in the audio production domain is spanned by three orthogonal mathematical operator kernels:

```mermaid
flowchart LR
    subgraph OpT ["Op-T: Transport, Pitch & All-Pass Kernel"]
        direction TB
        T1["• Model-Declared 64B-Aligned Histories with Branchless Wrap<br/>• Stateless 4-Tap Catmull-Rom Hermite Cubic Delay Reader (C¹-Smooth, Horner FMA, 0 FDIV)<br/>• Centered M-Phase Partition-of-Unity Windowing (M=1 Delay, M=2 Pitch Shifter, M=3 Barberpole)<br/>• Partitioned In-Buffer Schroeder All-Pass Diffusers & Grouped Bounded-History Guard"]
    end

    subgraph OpM ["Op-M: Complex Modal, Hilbert, Kautz & Bistable Dynamics Kernel"]
        direction TB
        M1["• 4-Lane SIMD Symplectic Complex Recurrence: z_c[n] = (r_c[n] u_rot,c[k]) z_c[n-1] + b_c x_c[n]<br/>• Radial-Polar Split (Quadrant-Reduced Rotor + H2 Norm + Radial Gate)<br/>• 15-Anchor 6-Section Dual All-Pass Hilbert Lattice (H1) for Quadrature (I, Q)<br/>• Bounded Half-Angle & Discrete Toggle Sub-Harmonic Trackers (T1)<br/>• Shared-Pole Kautz Modal Bank with Independent L/R Readouts"]
    end

    subgraph OpN ["Op-N: LTI-Commuted Nonlinearity, Hysteresis & Quantizer Kernel"]
        direction TB
        N1["• Multi-Drive 1x B-Spline PolyADAA + 6-Step Division-Free Rail Crossing<br/>• C² Cubic-Envelope Wavefolder with Division-Free Degree-7/8 PolyADAA<br/>• 2D Quadrature Phase-Space Hysteresis: (s_re Δs_re, s_re s_im) + Multi-Tau Envelope Sag<br/>• Fixed-Hertz Precomputed-Reciprocal ZOH Reducer & Bitcrusher (0 FDIV Audio Loop)<br/>• LTI-Commuted SIMD Projection: v_k[n] = u_nl[n] W_k"]
    end

    T1 <--> M1
    M1 <--> N1
```

#### 2.0 Self-Similar Dual-Rate Kernel Architecture

`r[arch.operators.self-similar]`

| Universal Kernel | Audio-Rate Execution (`48 kHz`) | Control-Plane Execution (`3 kHz`, `f_s / K_ctrl`) | Shared SIMD Mechanism |
| :--- | :--- | :--- | :--- |
| **`Op-T` (Transport)** | Fractional delay ring buffer, pitch grains, all-pass diffusers | Integer-plus-`Q0.32` delay targets and inward `f32` reader bounds | Branchless wrap and Catmull-Rom Horner FMA |
| **`Op-M` (Symplectic Complex & Ballistics)** | Complex resonant filters, `H1` Hilbert SSB, Kautz modal bank | `Q0.192` quadrature LFO rotors, `Q64`/`Q32` tempo phase, 4-Band block-peak envelopes | Quadrant-reduced contractive rotor and 1-pole ballistics |
| **`Op-N` (Polynomials & Nonlinearities)** | Multi-Drive `1×` B-Spline `PolyADAA`, `C²` wavefolder, bitcrushers | Packed `CP1` Lobatto/FM projections, exact-integer accumulator, outward trust clip | Branchless polynomial FMA and straight-through bounding |

#### 2.4.1 SIMD Lane Packing (AVX2, ARM NEON & WebAssembly SIMD128)

`r[arch.simd.lane-packing]`

`TORA` organizes all multichannel state, `fp16` delay conversions (`f32x4_to_fp16x4` / `fp16x4_to_f32x4`), and parameter matrices in **4-lane blocks** (`float32x4_t` in NEON, `v128` `f32x4` in WebAssembly `SIMD128`, half of `__m256` in AVX2) matching the nested tier hierarchy:

```text
NEON 128-Bit Vector Layout:
+-------------------+-------------------+-------------------+-------------------+
|  Lane 0: Ch 0     |  Lane 1: Ch 1     |  Lane 2: Ch 2     |  Lane 3: Ch 3     |  -> Tier 0: Lite
+-------------------+-------------------+-------------------+-------------------+
|  Lane 4: Ch 4     |  Lane 5: Ch 5     |  Lane 6: Ch 6     |  Lane 7: Ch 7     |  -> Tier 1: Full
+-------------------+-------------------+-------------------+-------------------+
|  Lane 8: Ch 8     |  Lane 9: Ch 9     |  Lane 10: Ch 10   |  Lane 11: Ch 11   |  -> Tier 2: Pro
+-------------------+-------------------+-------------------+-------------------+
|  Lane 12: Ch 12   |  Lane 13: Ch 13   |  Lane 14: Ch 14   |  Lane 15: Ch 15   |  -> Tier 3: Ultra
+-------------------+-------------------+-------------------+-------------------+
```

| Tier | Tier Index | Active Channels (`C_tier`) | NEON 128-Bit Vectors (`float32x4_t`) | AVX2 256-Bit Vectors (`__m256`) |
| :--- | :---: | :---: | :---: | :---: |
| **`Lite`** | Tier `0` | `4` channels (`Ch 0..3`) | `1×` SIMD vector | `0.5×` SIMD vector (`__m128`) |
| **`Full`** | Tier `1` | `8` channels (`Ch 0..7`) | `2×` SIMD vectors | `1×` SIMD vector |
| **`Pro`** | Tier `2` | `12` channels (`Ch 0..11`) | `3×` SIMD vectors | `1.5×` SIMD vectors (`8 + 4`) |
| **`Ultra`** | Tier `3` | `16` channels (`Ch 0..15`) | `4×` SIMD vectors | `2×` SIMD vectors |

#### 2.4.2 Differentiability and Gradient Dynamics

`r[arch.operators.differentiability]`

| Operator | Forward & Backward Mathematical Formulation |
| :--- | :--- |
| **1. `Op-T` (Continuous Fractional Delay)** | Horner Catmull-Rom derivative `∂d(τ)/∂μ = 3 c_3' μ² + 2 c_2' μ + c_1'` with coarse-to-fine Gaussian envelope surrogate gradient during delay identification. |
| **2. `Op-M` (Parallel Scan Recurrence)** | First-order complex recurrence admits an `O(log₂ L)` associative scan `(a₂, b̃₂) ∘ (a₁, b̃₁) = (a₂ a₁, a₂ b̃₁ + b̃₂)`. |
| **3. `Op-N` (Non-Smooth Surrogates & STE)** | Hard runtime forward `q̂(x) = sg[Q_B(x)] + (x - sg[x])` with grouped stop-gradient and optional 4-term Fourier backward surrogate ([Op-N §2.3](../dsp/op_n_commuted_cascade.md#23-family-5-precomputed-reciprocal-zoh-sample-rate-reducer--bitcrusher)). |

$$\frac{\partial d(\tau)}{\partial \mu} = 3 c_3' \mu^2 + 2 c_2' \mu + c_1', \qquad (a_2, \tilde{b}_2) \circ (a_1, \tilde{b}_1) = (a_2 a_1, a_2 \tilde{b}_1 + \tilde{b}_2), \qquad \hat q(x) = \mathrm{sg}[Q_B(x)] + \big(x - \mathrm{sg}[x]\big)$$

---

## 2. 4-Tier Slimmable SIMD Hierarchy (`Lite` / `Full` / `Pro` / `Ultra`)

### 2.1 Lower-Block-Triangular Arena Hierarchy

`r[arch.tiers.arena-hierarchy]`

Weights use a four-tier lower-block-triangular delta layout (`Arena 0..3`) where lower-tier payloads are contiguous authenticated prefixes of higher-tier payloads:

| Parameter | `Lite` (`Arena 0`) | `Full` (`Arena 0..1`) | `Pro` (`Arena 0..2`) | `Ultra` (`Arena 0..3`) |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Bus Width (`Phi[n]`)** | `64` channels | `128` channels | `192` channels | `256` channels |
| **Stage I Pre Delay Buffer** | Model-declared 64B-aligned | Model-declared 64B-aligned | Model-declared 64B-aligned | Model-declared 64B-aligned |
| **Stage II Nonlinear Basis** | `8L + 8R` multi-drive + fold/sag | Tier delta extension | Tier delta extension | Tier delta extension |
| **Stage III Early FIR Taps** | `16` per channel (`128 B` L/R) | `32` per channel (`256 B` L/R) | `48` per channel (`384 B` L/R) | `64` per channel (`512 B` L/R) |
| **Stage III Shared Kautz Pole Pairs** | `4` shared (`128 B` slice) | `8` shared (`256 B` cumulative) | `16` shared (`512 B` cumulative) | `32` shared (`1,024 B` cumulative) |
| **Stage IV Tank Delay Lines (`M`)** | `M = 4` lines (`fp16`) | `M = 8` lines (`fp16`) | `M = 8` lines (`fp16`) | `M = 16` lines (`fp16`) |
| **Control Bus Latent Dim (`z_ctrl`)**| `8-D` (`4` head + `4` stage, `R=4`) | `8-D` (`4` + `4`, `R=4`) | `8-D` (`4` + `4`, `R=4`) | `16-D` (`8` head + `8` stage, `R=8`) |

### 2.2 Hardware & DSP Benchmark Summary

`r[arch.tiers.hardware-budget]`

Reference embedded instance budgets on `1×` ARM Cortex-A76 core (`2.4 GHz`, `20,833 ns/sample` at `48 kHz`, see [`../../requirements/system_goals_and_requirements.md` §7](../../requirements/system_goals_and_requirements.md#7-goal-6--real-time-commodity-hardware-budget--execution-constraints)):

| Requirement | Reference Budget |
| :--- | :--- |
| **`Lite` instances per embedded core** | `≥ 10×` at `48 kHz` (`≤ 2,083 ns/sample`) |
| **`Full` instances per embedded core** | `≥ 6×` at `48 kHz` (`≤ 3,472 ns/sample`) |
| **`Pro` instances per embedded core** | `≥ 2×` at `48 kHz` (`≤ 10,416 ns/sample`) |
| **Real-time loop instruction invariant** | Zero hardware `FDIV`, `FSQRT`, or `libm` transcendental calls (`REQ-6.2`) |

**Rockchip RK3582 (`Cortex-A76 @ 2.4 GHz`) & LV2 Plugin Overhead Estimation Model:**

To translate standalone desktop `ARM64 NEON` microbenchmark frame times $t_{\text{bench}}(t)$ (`143.23 ns` `Lite`, `160.45 ns` `Full`, `172.21 ns` `Pro`, `179.34 ns` `Ultra` at $f_s = 48\text{ kHz}$, $T_{\text{sample}} = 1 / f_s \approx 20,833.33\text{ ns}$) to practical embedded multi-plugin chains on the **Rockchip RK3582** (`2× Cortex-A76 @ 2.4 GHz` with `64 KB` L1D / `512 KB` L2 per core + `4× Cortex-A55 @ 2.0 GHz`, `3 MB` shared L3), the per-instance frame latency $t_{\text{A76+LV2}}(t, B)$, single-core utilization $u_{\text{core}}(t, B)$, and safe simultaneous instance count $N_{\text{safe}}(t, B)$ for block size $B \in [64, 128]$ are modeled as:

$$t_{\text{A76+LV2}}(t, B) = \kappa_{\text{cache}} \left(\alpha_{\text{uarch}}(t) \cdot t_{\text{bench}}(t) + \frac{T_{\text{LV2-block}}}{B}\right), \qquad u_{\text{core}}(t, B) = \frac{t_{\text{A76+LV2}}(t, B)}{T_{\text{sample}}}, \qquad N_{\text{safe}}(t, B) = \left\lfloor \frac{\eta_{\text{RT}}}{u_{\text{core}}(t, B)} \right\rfloor$$

1. **Microarchitectural & Tier Scaling Factor ($\alpha_{\text{uarch}}(t)$):** Scales desktop wide-issue Apple Silicon NEON timings (`4× 128-bit` SIMD pipes, `~3.8 GHz`, large L1/L2) to a single in-order-constrained **Cortex-A76 @ `2.4 GHz`** (`2× 128-bit` NEON FMA pipes = `8` `f32` FLOPs/cycle peak or `19.2 GFLOP/s` per core), accounting for clock ratio (`~1.6×`), half the SIMD issue width (`2×`), and tier-dependent working-set growth from `Lite` (`64-D`, `K=4`, `M=4`) to `Ultra` (`256-D`, `K=32`, `M=16`).
2. **Amortized LV2 Plugin & Host Graph Dispatch ($T_{\text{LV2-block}} / B \approx 50\text{–}120\text{ ns/frame}$):** Captures per-block `LV2_Descriptor::run()` C ABI entry/exit, `FPCR` control-register save/restore (`FtzDazGuard`), `LV2_Atom_Sequence` parameter automation and `LV2_Time_Position` host BPM/transport parsing, and host audio-port buffer routing over `64–128` sample blocks.
3. **Multi-Instance Cache Spill Multiplier ($\kappa_{\text{cache}} \approx 1.15\text{–}1.30$):** Accounts for L1D (`64 KB`) $\to$ L2 (`512 KB`) cache refills when chaining multiple plugin instances whose combined parameter arenas (`Arena 0..3`), `fp16` delay ring buffers, and host port buffers exceed L1D residency.
4. **Real-Time Scheduling Headroom Ceiling ($\eta_{\text{RT}} = 0.80$):** Caps usable audio DSP allocation at `80%` of the `20,833.33 ns/sample` period (`2,666.67 µs` per `128`-sample block), reserving `20%` margin for ALSA/DMA interrupts, inter-core graph synchronization, and OS scheduling jitter.

| Tier Profile | Desktop NEON (`t_bench`) | Est. RK3582 `1× A76` + LV2 (`t_A76+LV2`) | Est. `1× Cortex-A76` CPU / Instance | Est. Safe Instances (`1× A76` @ 80% RT Cap) | Est. Safe Instances (`2× A76` Audio Cores) | Normative Floor (`REQ-6.1` on `1× A76`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Lite` (`64-D, C=4, K=4, M=4`)** | `143.23 ns` (`0.69%`) | `~520–730 ns` | **`2.5% – 3.5%`** | **`22 – 30×`** | `44 – 60×` | **`≥ 10×`** (`≤ 2,083 ns`) |
| **`Full` (`128-D, C=8, K=8, M=8`)** | `160.45 ns` (`0.77%`) | `~830–1,150 ns` | **`4.0% – 5.5%`** | **`14 – 20×`** | `28 – 40×` | **`≥ 6×`** (`≤ 3,472 ns`) |
| **`Pro` (`192-D, C=12, K=16, M=8`)** | `172.21 ns` (`0.83%`) | `~1,350–1,875 ns` | **`6.5% – 9.0%`** | **`8 – 12×`** | `16 – 24×` | **`≥ 2×`** (`≤ 10,416 ns`) |
| **`Ultra` (`256-D, C=16, K=32, M=16`)** | `179.34 ns` (`0.86%`) | `~2,080–2,900 ns` | **`10.0% – 14.0%`** | **`5 – 8×`** | `10 – 16×` | Workstation / `4×` Bounce |

**Prior-Art Equivalent-Quality LV2 Instance Comparison on `1× RK3582 Cortex-A76` (`2.4 GHz`, `48 kHz` + LV2 @ `80%` RT Cap):**

When comparing plugin instance budgets on an embedded Linux LV2 host (such as Darkglass Anagram / MOD / Elk on Rockchip RK3582), **LV2 plugin and host-graph overhead** compounds across three distinct mechanisms:

1. **Per-Instance LV2 Wrapper & Port Dispatch (`T_LV2-block / B ≈ 50–150 ns/sample`):** Every active LV2 plugin instance executes `LV2_Descriptor::run()`, parses `LV2_Atom_Sequence` / control ports, applies input/output gain-trim smoothing, and routes host audio port buffers on every `64–128` sample block.
2. **Dual-Instance Penalty for Stereo (`2×` LV2 Instances):** Because `NAM A2`, `GuitarML Proteus`, and `NAM ParametricOD` are mono (`1 → 1`) architectures, processing a stereo bus or dual-channel path requires instantiating **two separate LV2 plugins**—doubling LV2 host dispatch, port buffer copies, and per-instance L1D/L2 state RAM (`2 × 203.1 KB = 406.2 KB` for stereo `NAM A2 Standard`, consuming `> 79%` of the core's `512 KB` L2 cache for a single stereo pair). By contrast, `DNAM` executes true stereo (`2 → 2`) inside **one LV2 instance** with a shared `3 kHz` control plane and shared Kautz cabinet poles.
3. **Multi-Plugin Chain Penalty for Full Rigs (`4–6×` LV2 Instances vs. `1×` Unified `.dnam` LV2):** Modeling a complete stereo rig (`Drive → Amp → Cab IR → Stereo Delay/Reverb`) with mono amp-only plugins or discrete factory "grey-box" DSP blocks requires chaining **4 to 7 separate plugin blocks** (`4–7×` `run()` calls, per-block oversampling filter stages, intermediate host buffer reads/writes, and L1I/L1D instruction/data cache eviction across distinct `.so` binaries), whereas `DNAM` models the entire multi-stage hardware rig—or distills an offline SPICE/WDF solver and multi-plugin studio chain—in **a single LV2 plugin instance** (`1×` LV2 dispatch; see [`../../adr/0002-tri-operator-recurrent-architecture.md` §1.1](../../adr/0002-tri-operator-recurrent-architecture.md#11-why-tora-outperforms-grey-box-multi-fx-approximations--enables-offline-spice-distillation)).

| Architecture & Equivalent Quality Tier | Native LV2 Topology | Per-Instance Compute, Cache & LV2 Overhead on `Cortex-A76` (`64 KB` L1D / `512 KB` L2) | Est. `1× A76` CPU / LV2 Instance | Est. Safe Mono LV2 Instances (`1× A76` @ 80% Cap) | Est. Safe Stereo LV2 Chains (`1× A76` @ 80% Cap) | Est. Complete Stereo Rigs (`Drive+Amp+Cab+FX` on `1× A76`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Factory "Grey-Box" Multi-FX Blocks** (C++ LUTs + `2×–8×` OS Newton–Raphson + FFT Cab + FDN) | Per-block mono/stereo (`4–7×` blocks per rig) | Decoupled sub-circuits (`1D`/`2D` LUTs, `1`-step NR, per-block oversampling filters, FFT/FIR cab, FDN tank) + `4–7×` host block dispatch & L1I/L1D eviction | `2.0% – 6.0%` per single block (`14.0% – 26.0%` per `5–7` block stereo rig) | `15 – 35×` single blocks | `12 – 25×` single stereo blocks | `3 – 5×` full rigs (`5–7` chained blocks each) |
| **`NAM A2 Standard / Full` (Studio Grade)** | Mono LV2 (`2×` LV2s for stereo) | `~12,146` MACs (23 dilated conv layers) + `203.1 KB` state (`> 3.1×` L1D; `2` instances spill `512 KB` L2 to L3/DRAM) + C++ `Eigen` & LV2 port/trim overhead | `16.0% – 22.0%` (mono)<br/>`32.0% – 44.0%` (stereo pair) | `3 – 5×` mono LV2s | `1 – 2×` stereo pairs (`2–4` LV2s) | `1 – 2×` full rigs (`4–6` chained LV2s each) |
| **`NAM A2 Lite / Feather` (Live Grade)** | Mono LV2 (`2×` LV2s for stereo) | `~1,871` MACs at `75%` NEON utilization (`C = 3` unaligned loops) + `76.2 KB` state (`> 64 KB` L1D) + LV2 port/trim wrapper overhead | `6.5% – 9.0%` (mono)<br/>`13.0% – 18.0%` (stereo pair) | `8 – 12×` mono LV2s | `4 – 6×` stereo pairs (`8–12` LV2s) | `3 – 4×` full rigs (`4–6` chained LV2s each) |
| **`GuitarML Proteus` (`LSTM-40`, 1–2 Knobs)** | Mono LV2 (`2×` LV2s for stereo) | `~6,560–7,040` serial recurrent MACs + `160×` `sigmoid`/`tanh` `libm` calls/sample + `48 kHz` LV2 knob port smoothing (`+40·P` MACs static) + LV2 dispatch | `12.0% – 17.0%` (mono)<br/>`24.0% – 34.0%` (stereo pair) | `4 – 6×` mono LV2s | `2 – 3×` stereo pairs (`4–6` LV2s) | `2 – 3×` full rigs (`4–6` chained LV2s each) |
| **`NAM ParametricOD` (3-Knob WaveNet/LSTM)** | Mono LV2 (`2×` LV2s for stereo) | Dilated conv / LSTM stack + `48 kHz` per-sample `FiLM` knob projections (`+2·C·P` MACs/layer when static) + `tanh` + LV2 control port smoothing | `18.0% – 25.0%` (mono)<br/>`36.0% – 50.0%` (stereo pair) | `3 – 4×` mono LV2s | `1 – 2×` stereo pairs (`2–4` LV2s) | `1 – 2×` full rigs (`4–6` chained LV2s each) |
| **`DNAM Full` (True Studio Quality)** | **True Stereo LV2 (`1×` unified LV2)** | **Commuted 4-lane SIMD (`100%` NEON), L1D-resident core state, `0` static control MACs (`3 kHz` bus), `0` `FDIV`/`FSQRT`/`libm`, `1×` LV2 dispatch for full rig** | **`4.0% – 5.5%` (true stereo)** | **`14 – 20×` LV2s** | **`14 – 20×` true-stereo LV2s** | **`14 – 20×` complete stereo rigs (`1` LV2 each)** |
| **`DNAM Lite` (Live / Pedalboard Grade)** | **True Stereo LV2 (`1×` unified LV2)** | **`64-D` commuted 4-lane SIMD (`100%` NEON), `192 B` Stage III state, `0` static control MACs, `0` `FDIV`/`FSQRT`/`libm`, `1×` LV2 dispatch for full rig** | **`2.5% – 3.5%` (true stereo)** | **`22 – 30×` LV2s** | **`22 – 30×` true-stereo LV2s** | **`22 – 30×` complete stereo rigs (`1` LV2 each)** |

### 2.3 State RAM Audit: Lite Budget

`r[arch.tiers.state-ram]`

| Raw State Component | `Lite` | `Full` | `Pro` | `Ultra` |
| :--- | :---: | :---: | :---: | :---: |
| **Stage III L/R FIR Histories (`f32`)** | `128 B` | `256 B` | `384 B` | `512 B` |
| **Stage III L/R Complex Modal States (`f32`)** | `64 B` | `128 B` | `256 B` | `512 B` |
| **Stage III Array Subtotal** | `192 B` | `384 B` | `640 B` | `1,024 B` |
| **Static-Control Sufficient-Statistic Accumulator** | `64 B` | `64 B` | `64 B` | `128 B` |
| **Stage I / IV Delay Ring Histories (`fp16`)** | Model-declared (admitted against host ceiling on cold path) | Model-declared | Model-declared | Model-declared |

### 2.4 Computational Efficiency & Verification

`r[arch.tiers.efficiency]`

Both Stage III channel histories and modal states advance on every sample across all I/O topologies. Mono (`1 → 1`, `2 → 1`) output topologies evaluate only Slice 0 (`Mid`) of the Kronecker output head while retaining stereo-derived internal stage features. All inline and cold-allocated state is admitted against the host memory ceiling before activation with zero real-time heap allocation.

---

## 3. Canonical Signal Flow & 4-Stage Topology Overview

### 3.1 Detailed Signal Flow & Stage Boundaries

`r[arch.topology.signal-flow]`

TORA organizes all real-time processing into four canonical physical stages coordinated by an 8-Port Matrix Router front-end, mirroring the physical topology of complete professional rig setups:

```mermaid
flowchart TD
    subgraph ControlBusGroup ["3 kHz Unified Control Plane"]
        CtrlSrc["Shared descriptors and packed tier-local f32 projections<br/>(remaining control encoding pending)"] --> OpN["Op-N[Ctrl]: Control & Polynomial Engine<br/>(incremental update design target) -> tier-specific static latents"]
        SendDetector["Instrument / independent aux detector"] --> SendMod["Send-safe dynamic vector"]
        DownDetector["Downstream instrument / return / aux detector"] --> DownMod["Return-aware dynamic vector"]
        OpN & SendMod --> SendBus["Send dispatch rows 0..4<br/>Ultra delta + micro-ramp"]
        OpN & DownMod --> DownBus["Downstream dispatch rows 5..15<br/>Ultra delta + micro-ramp"]
    end

    In0["Port In 0: Inst In L"] & In1["Port In 1: Inst In R"] --> PreRouter["Stage 0: instrument and send routing<br/>• Mechanical Normaling<br/>• Voltage-Norm Mid/Side"]
    In2["Port In 2: FX Return L"] & In3["Port In 3: FX Return R / Aux Key"] --> DownRouter["Stage 0: return and downstream routing<br/>• Parallel-path alignment modes<br/>• Direct return-to-send block zero"]
    PreRouter -.-> SendDetector
    PreRouter & DownRouter -.-> DownDetector

    PreRouter --> S1["Stage I: Pre-FX / Synth / Mod<br/>(Op-T / Op-M / Op-N)"]
    S1 --> Out0["Port Out 0: Send L<br/>(External Amp 1)"]
    S1 -->|"Modes with second send"| Out1["Port Out 1: Send R / WDW dry"]
    DryTap["WDW dry selector<br/>PreSpatialCore / DirectDry"] -->|"WDW"| Out1
    S1 --> DownRouter

    DownRouter -->|"LR4 low: clean bypass"| S3
    DownRouter -->|"LR4 high: drive path"| S2["Stage II: 3-Stage Amp Cascade<br/>(V1A Preamp -> V1B -> Power Amp)"]
    S2 --> S3["Stage III: Cabinet & Modals<br/>(64-Tap SIMD FIR + Kautz Bank)"]
    S3 --> S4["Stage IV: Unitary Spatial Tank<br/>(In-Buffer AP + Barberpole FDN)"]

    DownBus --> DownRouter & S2 & S3 & S4
    SendBus --> PreRouter & S1

    S1 & S2 & S3 & S4 --> Phi["Unified 256-D Feature Bus Phi[n]"]
    Phi --> Head["256-D SVD Kronecker Head<br/>(Canonical Mid/Side Slices 0 & 1)"]
    DownBus --> Head
    Head --> Out2["Port Out 2: Main L / WDW wet L"]
    Head --> Out3["Port Out 3: Main R / WDW wet R"]
```

---

### 3.2 Stage 0: 8-Port Matrix Router & Voltage-Normalized Mid/Side

`r[arch.topology.stage0-router]`

Stage 0 provides 4 input ports and 4 output ports:
- **Port In 0..1:** Instrument In Left / Right.
- **Port In 2..3:** FX Loop Return Left / Right (or Aux Key).
- **Port Out 0:** FX Loop Send Left / optional mono WDW send.
- **Port Out 1:** FX Loop Send Right, or center dry in WDW.
- **Port Out 2..3:** Main Out Left / Right; wet-only L/R in WDW.

In WDW mode (`PreSpatialCore` or `DirectDry`), Port Out 1 carries the center dry tap while Ports Out 2/3 carry the stereo wet output and Port Out 0 provides an optional mono send. A zero $\mathbf{S}_{\text{ret}\to\text{send}}$ matrix block and partitioned detector/dispatch routing isolate external sends from return-derived feedback (see [router §1](../dsp/router_and_stages_topology.md#1-stage-0-8-port-matrix-router-req-74)). Mid/Side channels are voltage-normalized ($x_{\text{mid}} = 0.5(L+R), x_{\text{side}} = 0.5(L-R)$) for unity-gain reconstruction.

### 3.3 Gated RMS Auto-Unity Leveling & Per-Tier Loudness Parity

`r[arch.topology.auto-unity-leveling]`

Auto-unity leveling (`io_config.leveling`) anchors at the declared default control settings. Pre-stored linear tier gain trims (`tier_gain_trim_lin`) maintain calibrated per-tier loudness parity across `Lite`, `Full`, `Pro`, and `Ultra` without runtime division or transcendental calls.

#### On-Demand Tier Transition Contract

`r[arch.tiers.transition-contract]`

Runtime tier changes use **on-demand state transfer and a bounded parallel crossfade** rather than continuously stepping inactive tiers:

```mermaid
stateDiagram-v2
    [*] --> Active: Tier T_old Running
    Active --> Pending: Request Tier T_new (T_eff = min(T_req, T_ceiling, T_avail))
    Pending --> Transitioning: Cold-Path Admission & State Snapshot Ready
    Pending --> Rejected: Insufficient Memory / WCET Headroom or Active Frozen Tail
    Transitioning --> Active: Bounded Crossfade Complete -> Retire T_old & Commit T_new
    Transitioning --> Active: Invalidating Event -> Rollback to Advancing T_old
    Rejected --> Active: Retain T_old & Report Reason
```

1. **Request & Admission:** Track the requested tier, effective target $\min(t_{\text{req}}, t_{\text{ceiling}}, t_{\text{avail}})$, active tier, and transition status (`pending`, `transitioning`, `completed`, `rejected`). Admit a transition only after verifying state compatibility, stable latency, prepared memory, and worst-case callback headroom.
2. **State Preparation:** Prepare target weights, transition buffers, and crossfade ramps off the audio thread, transferring compatible recursive, delay, control, and tempo state from a sample-aligned snapshot.
3. **Active Tail Preservation:** Never silently reset or drop an active delay, modal, autonomous, or frozen tail; defer or reject a down-tier request if the target tier cannot retain an active contribution.
4. **Parallel Render & Handoff:** Advance both old and target paths with time-aligned inputs and control events across a bounded precomputed crossfade, retiring the old path and committing the target tier only after the final fade sample.

### 3.4 Spillover Trails State Machine

`r[arch.topology.spillover-trails]`

When bypassing an active rig with `trails = true`, Stage IV Spatial Tank continues ringing out its tail while input injection is muted:

```mermaid
stateDiagram-v2
    [*] --> Active: Engine Initialized (bypassed = false)
    Active --> HardBypass: set_bypass(true, trails = false)
    Active --> SpilloverTrails: set_bypass(true, trails = true) & stage4_active == true
    Active --> HardBypass: set_bypass(true, trails = true) & stage4_active == false

    SpilloverTrails --> Active: set_bypass(false, _)
    HardBypass --> Active: set_bypass(false, _)

    state Active {
        [*] --> FullDSP: Process Stages 0 through IV & Mid/Side Head
    }
    state SpilloverTrails {
        [*] --> RingingOut: Dry Input Passthrough + Mute Tank Input + Decay Tank State
    }
    state HardBypass {
        [*] --> PassThrough: Zero-Copy Input -> Output (0 DSP MACs)
    }
```

### 3.5 $4\times$ ZOH Polyphase Oversampling on Offline Bounce

`r[arch.bounce.oversampling]`

When a DAW host renders offline (`is_offline_bounce == true`):
1. **Tier Promotion:** When `maximize_tier_on_bounce` is enabled, promote to $t_{\text{bounce}} = \min(t_{\text{avail}}, t_{\text{max-sup}})$.
2. **Polyphase Oversampling:** Step the ZOH-rescaled engine at $4 f_s$ through a 2-stage linear-phase half-band polyphase upsampler/downsampler ($>95\text{ dB}$ stopband attenuation, `12` host samples oversampler PDC).
3. **Alignment Promotion:** Promote active parallel-alignment groups to `FullBand64Tap` (`31` host samples centering offset, `43` host samples combined nominal PDC with the oversampler; see [router §1](../dsp/router_and_stages_topology.md#parallel-path-sub-sample-alignment)).
4. **Live State Isolation:** Execute offline bounce on an isolated state or restore live monitoring state and tier upon completion without disturbing active histories.

---

## 4. Coverage of the 7 Goal 1 Hardware Families

`r[arch.families.coverage]`

```mermaid
flowchart TD
    subgraph Stage0Router ["Stage 0: 8-Port Matrix Router"]
        F7["Family 7: Multi-Input Sidechain / 4CM / 7CM / WDW<br/>• Return-to-send isolation<br/>• 64-tap full-band / live alignment<br/>• 200 Hz LR4 crossover + WDW dry tap"]
    end

    subgraph StageIPre ["Stage I: Pre-FX / Synth / Pitch / Mod Bus"]
        F1["Family 1: Bistables & PLL Synths<br/>• Op-M[Pre] + Op-N[Pre]<br/>• T1 DC Blocker + Half-Angle & Toggle Trackers"]
        F2["Family 2: Carrier Mod & Hilbert SSB<br/>• Op-M[Pre] 6-Section H1 All-Pass Pair<br/>• Quadrant-Reduced Carrier Rotor"]
        F3A["Family 3 (Pre): Polyphonic Pitch Shifter<br/>• Op-T[Pre] Mode M=2 Centered Hermite-Hann<br/>• Softmax Diatonic Scale Quantizer"]
        F4A["Family 4 (Pre): Hysteretic Gate / Comp<br/>• 3 kHz Pitchfork Hysteresis -> Op-N[Pre]"]
        F5["Family 5: Bitcrusher & ZOH Reducer<br/>• Op-N[Pre] & Stage II Op-N<br/>• Fixed-Hertz Phase + Precomputed 1/Delta (0 FDIV)"]
    end

    subgraph StageIIICab ["Stage III: Modal & Formant Filterbank"]
        F6A["Family 6 (Formants): Vocoder & Vowels<br/>• Op-M[Modal] Shared Kautz Resonators<br/>• Dynamic Sweep Contraction (Refinement 10)"]
    end

    subgraph StageIVTank ["Stage IV: Unitary Spatial & Barberpole Tank"]
        F3B["Family 3 (Tank): Shimmer Pitch Grains<br/>• Op-T[Tank] Mode M=2 (H(phi_A)+H(phi_B)==1)"]
        F4B["Family 4 (Tank): Sidechain Ducking Delay<br/>• 1-Pole Ballistic Slew Limiter (Refinement 9)"]
        F6B["Family 6 (Tank): Barberpole Sweep<br/>• Op-T[Tank] Mode M=3 (3-Phase Raised Cosine)"]
    end

    F7 --> F1 & F2 & F3A & F4A & F5
    F1 & F2 & F3A & F4A & F5 --> F6A
    F6A --> F3B & F4B & F6B
```

| Family | Hardware Archetypes | TORA Stage & Operator Mapping | Core Zero-`FDIV` Audio-Rate Formulation | Differentiable Training Surrogate | Acceptance Criterion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Family 1: Bistables & PLL Synths** | Boss OC-2, Mu-Tron Octave Divider, Schumann PLL | **Stage I:** `Op-M[Pre]` + `Op-N[Pre]` | `T1` `5 Hz` DC blocker, scaled `rsqrt`, half-angle (`cos`, `sin`) and toggle (`f/2`, `f/4`) trackers | Hard-forward tracker with surrogate gradient | Sub-octave frequency/amplitude tracking and zero DC offset |
| **Family 2: Carrier Mod & Hilbert SSB** | Moog MF-102 Ring Mod, Bode 735 Frequency Shifter | **Stage I:** `Op-M[Pre]` | 15-anchor 6-section `H1` all-pass pair + quadrant-reduced carrier rotor | Phase-insensitive demodulated STFT loss `L_carrier(ω_c)` | SSB rejection `≥ 42.0 dB` across `100 Hz–10 kHz` (`≤ 0.34°` quadrature error) |
| **Family 3: Polyphonic & Scale Pitch Shifters** | Eventide H3000, DigiTech Whammy, EHX POG | **Stage I** (`Op-T[Pre]`) & **Stage IV** (`Op-T[Tank]`) | Centered `M = 2` Hermite-Hann grains, 4-tap Catmull-Rom, `±24` st pitch/warp profile | Continuous Softmax Diatonic Scale Quantizer + Catmull-Rom `∂d/∂μ` | Pitched chord THD+N `≤ -68 dB` and click-free sweep transitions |
| **Family 4: Hysteretic Dynamics & Ducking** | Boss NS-2, SG-1 Slow Gear, UREI 1176LN, SPL Transient, TC 2290 | **`3 kHz` Control Bus** → `Op-N[Pre]` & `Op-T[Tank]` VCA | Supercritical pitchfork hysteresis map `h[k]`, monotone `f32` `log₂`/`exp₂` lookup, 1-pole ballistic slew (`τ_att ≥ 0.5 ms`, `τ_rel ≥ 20 ms`) | Differentiable pitchfork derivative `∂h/∂e = (1/τ_h)σ'(u)` + `ln(x + ε)` | Silence noise floor reduction `≥ 72 dB`; zero gate chatter |
| **Family 5: Bitcrushers & ZOH Reducers** | E-mu SP-1200, WMD Geiger Counter | **Stage I** (`Op-N[Pre]`) & **Stage II** (`Op-N`) | Nearest-step quantizer (halfway ties to `+∞`) + fixed-hertz integer ZOH phase (`0 FDIV`) | Hard runtime forward + grouped STE / 4-term Fourier backward surrogate | Exact dyadic boundary rounding and full-rate/fractional latch parity |
| **Family 6: Vocoders, Formants & Barberpole** | EMS Vocoder 2000, EHX Talking Pedal, Maestro Stage Phaser | **Stage III** (`Op-M[Modal]`) & **Stage IV** (`Op-T[Tank]`) | Shared Kautz modal poles, dynamic sweep contraction, `M = 3` barberpole and Givens/Householder mixing | Log-spectral formant envelope loss `L_formant` + continuous Catmull-Rom | Vowel formant peaks (`F₁, F₂, F₃`) within `±0.4 dB` |
| **Family 7: Multi-Input Sidechain / 4CM / 7CM / WDW** | Dual-amp 4CM/7CM loops, keyed gates/ducks, WDW center dry rigs | **Stage 0: 8-Port Matrix Router** + Stage I/IV Partition | Return-to-send isolation, `200 Hz` TPT LR4 split, 64-tap sub-sample alignment, and WDW dry tap | Decoupled dual-target multi-task loss `L_4CM = λ_pre L_pre + λ_post L_post` | Crossover passband ripple `≤ 0.05 dB` and isolated send paths |

### 4.1 Family 1: Bistables & PLL Guitar Synthesizers

`r[arch.families.f1-bistables]`

- **Analog Archetypes:** Boss OC-2, Mu-Tron Octave Divider, Schumann PLL (CMOS 4013 dual D-flip-flops generating $f_0/2, f_0/4$ sub-octaves and 4046 VCO PLL tracking).
- **TORA Realization (`Stage I` `Op-M[Pre]` + `Op-N[Pre]`):** Applies the tracker-only `5 Hz` `T1` DC blocker, scaled exponent-split envelope normalization, independent `prev_i_nonpositive` axis-crossing detection, smooth half-angle quadratures (`cos`, `sin`), and divider square features (`f/2`, `f/4`) routed into `Lite` feature bus slots `4..11` ([`op_m_modal_kautz.md` §3](../dsp/op_m_modal_kautz.md#3-family-1-sub-harmonic-synthesizers-bistable-dividers--pll-tracking), [router §7](../dsp/router_and_stages_topology.md#7-unified-256-d-4-tier-kronecker-output-head)).

### 4.2 Family 2: Audio-Rate Carrier Modulation & Hilbert SSB

`r[arch.families.f2-hilbert-ssb]`

- **Analog Archetypes:** Moog MF-102 Ring Modulator, Bode 735 Frequency Shifter (4-quadrant multiplier and Weaver dual $90^\circ$ dome filter).
- **TORA Realization (`Stage I` `Op-M[Pre]`):** Uses the 6-section all-pass pair with the 15-anchor `H1` rate table (`≤ 0.34°` quadrature phase error across `100 Hz–10 kHz`) and the quadrant-reduced contractive carrier rotor (see [`op_m_modal_kautz.md` §2](../dsp/op_m_modal_kautz.md#2-family-2-hilbert-ssb-lattice--carrier-modulation)).

### 4.3 Family 3: Polyphonic Phase-Locked & Intelligent Scale Pitch Shifters

`r[arch.families.f3-pitch]`

- **Analog Archetypes:** Eventide H3000, DigiTech Whammy, EHX POG.
- **TORA Realization (`Stage I` `Op-T[Pre]` & `Stage IV` `Op-T[Tank]`):** Uses the centered grain-duration construction with partition-of-unity cubic Hermite-Hann windows ($u = 1 - |2\phi - 1|$, $H(\phi) = u^2(3 - 2u)$, $H(\phi_A) + H(\phi_B) \equiv 1$), 4-tap Catmull-Rom delay reads with per-head causal floor $\tau_m \ge 4.0$, and the `-24..24` semitone control-rate pitch/warp profile ([control bus §5](../dsp/parametric_control_bus.md#pitch-ratio-and-time-warp-profile)).

### 4.4 Family 4: Hysteretic Dynamics, Noise Gates, Limiters & Ducking

`r[arch.families.f4-dynamics]`

- **Analog Archetypes:** Boss NS-2, Boss SG-1 Slow Gear, UREI 1176LN, SPL Transient Designer, TC 2290.
- **TORA Realization (`3 kHz` Control Bus → `Op-N[Pre]` & `Op-T[Tank]` VCA):** Evaluates the supercritical pitchfork bifurcation hysteresis map $h[k] = \sigma\big((e[k] - \theta_{\mathrm{mid}})/\tau_h + \kappa(2 h[k-1] - 1)\big)$, monotone `f32` $\log_2$ lookup ([control bus §5](../dsp/parametric_control_bus.md#monotone-f32-log-envelope-lookup-profile)), and 1-pole ballistic slew limiting ($\tau_{\mathrm{attack}} \ge 0.5\text{ ms}, \tau_{\mathrm{release}} \ge 20.0\text{ ms}$) with audio-rate linear gain micro-ramping.

### 4.5 Family 5: Sample-Rate Reducers, Bitcrushers & Non-Smooth Digital Quantizers

`r[arch.families.f5-bitcrusher]`

- **Analog Archetypes:** E-mu SP-1200, WMD Geiger Counter.
- **TORA Realization (`Stage I` `Op-N[Pre]` & `Stage II` `Op-N`):** Evaluates the uniform nearest-step quantizer $Q_B(x) = \lfloor x 2^{B-1} + 0.5 \rfloor 2^{-(B-1)}$ (halfway ties toward $+\infty$) and the fixed-hertz integer ZOH phase accumulator with pre-reduction carry detection and division-free `u128` reciprocal rate transfer ([Op-N §2.3](../dsp/op_n_commuted_cascade.md#23-family-5-precomputed-reciprocal-zoh-sample-rate-reducer--bitcrusher)).

### 4.6 Family 6: Multi-Band Vocoders, Formant Vowels & Barberpole Tank

`r[arch.families.f6-formants]`

- **Analog Archetypes:** EMS Vocoder 2000, EHX Talking Pedal, Maestro Stage Phaser.
- **TORA Realization (`Stage III` `Op-M[Modal]` & `Stage IV` `Op-T[Tank]`):** Combines shared-pole Kautz resonators with dynamic sweep contraction ([`op_m_modal_kautz.md` §4.3](../dsp/op_m_modal_kautz.md#43-dynamic-sweep-contraction-rule-refinement-10)), 3-phase raised-cosine barberpole phase wrapping ($M=3$), and Givens/Householder unitary tank mixing.

### 4.7 Family 7: Multi-Input Sidechain / 4CM / 7CM / WDW Units

`r[arch.families.f7-multi-input]`

- **Analog Archetypes:** External amplifier loop insertion (4CM/7CM dual amps), auxiliary key sidechain detection, and Wet-Dry-Wet (WDW) rigs.
- **TORA Realization (`Stage 0: 8-Port Matrix Router`):** Enforces return-to-send isolation, `FullBand64Tap` / `Live16Tap` / `IntegerDelayOnly` parallel path alignment ([router §1](../dsp/router_and_stages_topology.md#parallel-path-sub-sample-alignment)), the `200 Hz` TPT LR4 clean/drive crossover ([router §1](../dsp/router_and_stages_topology.md#stage-0-lr4-cleandrive-crossover)), and `PreSpatialCore` / `DirectDry` WDW routing ([router §1](../dsp/router_and_stages_topology.md#wdw-dry-tap-and-wet-output-contract)).

---

## 5. Five End-to-End Trainer–Engine Isomorphism & Pipeline Guarantees

`r[arch.isomorphism.guarantees]`

TORA establishes five end-to-end trainer–engine isomorphism and pipeline guarantees across [`packages/trainer/`](../../../packages/trainer/README.md) and [`packages/engine/`](../../../packages/engine/README.md) (for architectural rationale, see [`../../adr/0002-tri-operator-recurrent-architecture.md` §1](../../adr/0002-tri-operator-recurrent-architecture.md#1-context-five-structural-bottlenecks-in-conventional-neural-audio)):

| Guarantee ID | Subsystem Scope | TORA Architectural Realization |
| :--- | :--- | :--- |
| **1. Direct Commuted Stage Serialization (§5.1)** | Stage II (`stage2_cascade.py` / `op_n.rs`) | Directly serializes fitted `W_stage^(2.1..2.3)` matrices into `Arena 0..3` using identical project-then-filter ordering in training and runtime. |
| **2. Joint Full-Rig Training (§5.2)** | Two-Sweep Pipeline (`sweep1_cascade.py`, `sweep2_tank.py`, `polish_head.py`) | Freezes upstream stage/control behavior before accumulating tier-specific sufficient statistics, solving Stages I–IV sequentially in Sweep 1 and polishing the joint head in Sweep 2. |
| **3. End-to-End 3-Phase Audit Wiring (§5.3)** | Step 0 & Audit Engine | Wires Step-0 order-statistic calibration, physical variance gates, non-stop 3-phase clamping, and export certification into the training pipeline. |
| **4. True-Stereo Calibration & 8-Port Routing (§5.4)** | Stem Extractor & Router | Preserves 2-channel stem arrays `X ∈ R^(N×2)` (`[Left, Right]` or `[Main, Aux]`) with causal parallel alignment and return-to-send isolation. |
| **5. Nested 4-Tier Physical Hierarchy (§5.5)** | Feature Bus & Arenas (`Arena 0..3`, `head_kronecker.py`) | Extracts four nested physical feature definitions (`64 / 128 / 192 / 256 D`) and stores tier-bundled per-stem sufficient statistics in `.dnam-ckpt`. |

### 5.1 Direct Stage II Commuted Cascade Serialization & Runtime Parity

`r[arch.isomorphism.stage2-serialization]`

Stage II serializes the fitted commuted projection matrices $\mathbf{W}_{\mathrm{stage}}^{(2.1..2.3)}$ directly into `Arena 0..3` and models the runtime's project-then-filter ordering under control automation ([`../training/four_stage_pipeline.md` §4](../training/four_stage_pipeline.md#4-trainerengine-isomorphism-protocol--closed-form-normal-equations)).

### 5.2 Joint Stage I–IV Full-Rig Training via Two-Sweep Block-Coordinate Cholesky

`r[arch.isomorphism.two-sweep]`

The **Two-Sweep Block-Coordinate Cholesky Pipeline** ([`../training/four_stage_pipeline.md`](../training/four_stage_pipeline.md)) solves Stage III (Cabinet), Stage II (Amp Cascade), and Stage IV (`Op-T[Tank]`) sequentially on forward residuals in Sweep 1, and solves the joint 256-D Mid/Side Kronecker head across all 4 stages in Sweep 2, enabling single-run full-rig training (`REQ-1.5`).

### 5.3 Step-0 System-ID & Non-Stop 3-Phase Audit Wiring in `fit_job`

`r[arch.isomorphism.audit-wiring]`

The pipeline combines Step-0 calibration, variance gates, non-stop clamp phases (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`, `[CLAMP-RAIL]`, `[CLAMP-DYNAMIC]`), and head gain certification ([`../training/control_space_audit.md`](../training/control_space_audit.md)).

### 5.4 True-Stereo Input Preservation & 8-Port Matrix Router Execution

`r[arch.isomorphism.stereo-preservation]`

Stem extraction preserves 2-channel audio arrays $\mathbf{X} \in \mathbb{R}^{N \times 2}$ (`[Left, Right]` or `[Main, Aux]`) and enforces the physical port assignments, causal alignment, and send isolation of the 8-Port Matrix Router.

### 5.5 Nested 4-Tier Feature Bus Extraction (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`)

`r[arch.isomorphism.nested-bus]`

TORA builds a nested 4-tier physical hierarchy matching `REQ-5`: `Lite` ($64\text{ D}$), `Full` ($128\text{ D}$), `Pro` ($192\text{ D}$), and `Ultra` ($256\text{ D}$), with tier-bundled runtime weights and sufficient statistics in `.dnam-ckpt` ([`../containers/checkpoint_and_cache_format.md` §2](../containers/checkpoint_and_cache_format.md#2-tier-ordered-column-major-upper-triangular-packing-uplou-for-mathbfg_textaudios)).

---

## 6. Summary of the 19 Architectural Refinements

`r[arch.refinements.summary]`

| # | Architectural Refinement | Target Subsystem | Mathematical / Structural Specification |
| :- | :--- | :--- | :--- |
| **1** | **Partitioned Schroeder All-Pass Buffer Segmentation** | Stage IV `Op-T[Tank]` | Non-overlapping `[0..L_main-1 ∣ L_main..L_main+L_ap-1]` segments with independent logical wrap ([Op-T §1 & §3](../dsp/op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)). |
| **2** | **Strictly Causal Stage II Delays & Early FIR Fitting** | Stage II & Stage III | Causal offsets (`d ≥ 0`) in Stage II and Stage III early FIR taps ([Op-N §3.2](../dsp/op_n_commuted_cascade.md#32-causality-audit--elimination-of-acausal-lookahead)). |
| **3** | **4-Filter Tone Stack Basis (`Direct + 2 LPF + 1 HPF`)** | Sub-Stage 2.1 | Adds matched-pole `HPF_1200` alongside `Direct`, `LPF_80`, and `LPF_400` ([Op-N §4.1](../dsp/op_n_commuted_cascade.md#41-four-filter-tone-basis-refinement-3)). |
| **4** | **Two-Step Warped Laguerre/Kautz Modal Pole Extraction** | Stage III Trainer | Log-spaced initial eigenmodes (`40–4500 Hz`) with shared poles and independent L/R readouts ([`op_m_modal_kautz.md` §4.4](../dsp/op_m_modal_kautz.md#44-two-step-warped-laguerrekautz-modal-pole-extraction-refinement-4)). |
| **5** | **Slew-Rate Bounded Givens Rotation in Stage IV** | Stage IV `U(ψ, v)` | Bounds stereo rotation velocity to `∣Δψ∣ ≤ 0.05 rad/control tick` (`150 rad/s` at `3 kHz`). |
| **6** | **Topological Order Descriptors (`stage1_topology`)** | Stage I `Arena 0` | 4-bit bitfield selecting among the 4 canonical `Op-T`/`Op-M`/`Op-N` pre-chain permutations without buffer contention. |
| **7** | **LR4 Parallel Crossover & WDW Routing** | Stage 0 Router | `200 Hz` TPT LR4 clean/drive split and `PreSpatialCore`/`DirectDry` WDW output mapping ([router §1](../dsp/router_and_stages_topology.md#stage-0-lr4-cleandrive-crossover)). |
| **8** | **Stage I Sub-Harmonic DC Conditioning (`T1`)** | Family 1 → Stage I | Dedicated `5 Hz` TPT DC blocker before the analytic tracker pair ([`op_m_modal_kautz.md` §3](../dsp/op_m_modal_kautz.md#3-family-1-sub-harmonic-synthesizers-bistable-dividers--pll-tracking)). |
| **9** | **1-Pole Ballistic Slew Limiter on Sidechain Ducking Gains** | Family 4 Sidechain | `τ_attack ≥ 0.5 ms`, `τ_release ≥ 20.0 ms` eliminates zipper noise and transient chatter. |
| **10** | **Radial-Polar Rotor Contraction during Formant Sweeps** | Stage III Kautz | Dynamic sweep contraction `r_eff,m = min(r_m, max(0.25, 0.9950 - 0.15|Δθ_m|))` ([`op_m_modal_kautz.md` §4.3](../dsp/op_m_modal_kautz.md#43-dynamic-sweep-contraction-rule-refinement-10)). |
| **11** | **Block-Diagonal Lower-Triangular Kronecker Head Staging** | Output Head | Nested tier blocks with precomputed `tier_gain_trim_lin` loudness matching. |
| **12** | **Projected LTI Filtering `L(u[n] W[k])`** | Stage II Cascade | Projects the nonlinear basis by `W_k[n]` before stepping the tone-stack filters. |
| **13** | **Normalized Coupled-Form State-Space for Kautz Modals** | Stage III `Op-M[Modal]` | Coupled-form recurrence with forced-state bound `‖x[n]‖ ≤ r_max ‖x[n-1]‖ + ‖b_m‖ |u[n]|` ([`op_m_modal_kautz.md` §4.2](../dsp/op_m_modal_kautz.md#42-normalized-coupled-form-state-space-refinement-13)). |
| **14** | **Input Headroom Normalization & Denormal-Tail Handling** | Stage IV `Op-T[Tank]` | `-12 dB` tank input attenuation for `fp16` headroom plus `FtzDazGuard` denormal immunity ([audio-block policy](#floating-point-environment-at-the-audio-callback)). |
| **15** | **Per-Head Causal Floor Margin `τ_m ≥ 4.0` Samples** | `Op-T` Multi-Tap Reader | Enforces `τ_m[n] = max(4.0, τ_raw,m[n])` independently per granular/barberpole read head. |
| **16** | **Return-to-Send Router Isolation** | Stage 0 8-Port Router | Enforces `S_{ret->send} == 0` and isolates send-path detectors/controls from returns. |
| **17** | **Parallel-Path Sub-Sample Alignment** | Stage 0 Router | `FullBand64Tap` (default/bounce), `Live16Tap`, and `IntegerDelayOnly` alignment modes ([router §1](../dsp/router_and_stages_topology.md#parallel-path-sub-sample-alignment)). |
| **18** | **Unified Self-Similar Dual-Rate Control Plane (`Op-M[Ctrl]` / `Op-N[Ctrl]`)** | Control Plane | Executes `3 kHz` control via 4-lane SIMD instances of `Op-M` and `Op-N`. |
| **19** | **Dual-Space Factored Control & `CP1` Additive Latent Scaling** | Parameter Bus | Shared descriptors + packed tier-local `CP1` `f32` projections (`8-D` Rank-4 base + `16-D` Rank-8 `Ultra`) with exact-integer sufficient statistics ([`../dsp/parametric_control_bus.md` §2.3](../dsp/parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)). |

---

## 7. Canonical TORA Requirement Realization Matrix (`REQ-1`–`REQ-8.8`)

`r[arch.verification.matrix]`

| Goal ID | Requirement Domain | TORA Architectural Realization | Canonical Specifications |
| :--- | :--- | :--- | :--- |
| **`REQ-1`** | **Model Classes & Accuracy** | `Op-T`, `Op-M`, and `Op-N` across Stages 0–IV and the 256-D Mid/Side Kronecker head. | [`op_t_transport.md`](../dsp/op_t_transport.md), [`op_m_modal_kautz.md`](../dsp/op_m_modal_kautz.md), [`op_n_commuted_cascade.md`](../dsp/op_n_commuted_cascade.md) |
| **`REQ-1.5`** | **Full-Rig Multi-Control Modeling** | Packed tier-local `CP1` projections (`8-D` Rank-4 in `Lite..Pro`, `16-D` Rank-8 + `16×4` stage-dispatch delta in `Ultra`). | [`parametric_control_bus.md`](../dsp/parametric_control_bus.md), [`four_stage_pipeline.md`](../training/four_stage_pipeline.md) |
| **`REQ-2`** | **Quality, Control Bus & Tempo Sync** | Per-output noise-adjusted `bESR`, `MR-STFT`, 33-band `LTAS`, and duration-weighted `SegESR` with non-stop 3-phase clamping. | [`psychoacoustic_quality_and_tiers.md`](psychoacoustic_quality_and_tiers.md), [`control_space_audit.md`](../training/control_space_audit.md) |
| **`REQ-2.5`** | **MIDI / Host Tempo Sync** | `20–400 BPM` `Q64`/`Q32` tempo tracking across 12 subdivisions, reachable `f32` physical inversion, and `SmoothRelock` / `ImmediatePhaseLock`. | [`parametric_control_bus.md` §7](../dsp/parametric_control_bus.md#7-host-midi-and-tap-tempo-synchronization-req-25) |
| **`REQ-3`** | **Continuous Checkpointing & Stem Resume** | Tier-bundled `.dnam-ckpt` (`DCA1`), `uplo='U'` `Arena 4` slices, SHA-256 Gram-input fingerprint, and windowed `< 32 MiB` refresh. | [`iterative_resume_and_caching.md`](../training/iterative_resume_and_caching.md), [`checkpoint_and_cache_format.md`](../containers/checkpoint_and_cache_format.md) |
| **`REQ-4`** | **Accelerator-First, Accurate & Memory-Scaled Training** | 32-bit device streaming, weighted Gram contraction bound (`∣ĝ - g∣ ≤ γ_N S`), batched host `float64` solver, and `< 8 GiB` host budget. | [`four_stage_pipeline.md` §3](../training/four_stage_pipeline.md#3-accelerator-first-32-bit-execution-gram-accuracy--streaming) |
| **`REQ-5`** | **Multi-Tier Slimmable Scaling** | Four nested tiers (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`), single-run joint training, on-demand bounded tier crossfade, and keyless prefix stripping. | [`tora_architecture.md` §2–§3](#2-4-tier-slimmable-simd-hierarchy-lite--full--pro--ultra), [`dnam_binary_format.md`](../containers/dnam_binary_format.md) |
| **`REQ-6`** | **Embedded Real-Time Execution** | `≥ 10× Lite`, `≥ 6× Full`, `≥ 2× Pro` @ `48 kHz` on `1×` Cortex-A76, zero real-time `FDIV`/`FSQRT`/`libm`, and zero callback allocations. | [`tora_architecture.md` §1–§2](#1-executive-summary--core-architectural-pillars) |
| **`REQ-7`** | **Unclipped `f32` Signal Chain, Topologies & Stability** | `±100.0` (`+40 dBFS`) certified unclipped envelope, `±1e9` all-finite safety guards, `1→1`/`2→1`/`1→2`/`2→2` topologies, and 3-phase control-space clamping (`ref[train.audit.phase1.clamp-stability]`). | [`router_and_stages_topology.md`](../dsp/router_and_stages_topology.md), [`control_space_audit.md`](../training/control_space_audit.md) |
| **`REQ-8.1`–`REQ-8.8`** | **Provenance, Enclave DRM, Watermarking & Declarative Containers** | RFC 8785 JCS header digest, `DAU1`/`DAI1`/`DSA1` `Ed25519` prefix signatures, RFC 9180 HPKE `EnclaveSealed`, dual-mechanism Tardos/gauge watermark, `.dnam-prev` containment, and 64B-aligned code-execution-free containers. | [`provenance_and_enclave_drm.md`](../security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../security/watermarking_and_marketplace.md), [`dnam_binary_format.md`](../containers/dnam_binary_format.md) |

---

## 8. Subsystem Specification Index

`r[arch.index.subsystems]`

- **Architecture & Quality Specifications (`docs/specs/architecture/`)**
  - [`psychoacoustic_quality_and_tiers.md`](psychoacoustic_quality_and_tiers.md): Noise-Adjusted Band-Split ESR (`bESR`), Companion Metrics & 4-Tier Quality Matrix.
  - [`project_and_documentation_quality.md`](project_and_documentation_quality.md): Repository, Code & Documentation Quality Specification (`r[repo.*]`).
- **Real-Time DSP Subsystems (`docs/specs/dsp/`)**
  - [`op_t_transport.md`](../dsp/op_t_transport.md): Universal Transport, Windowing & Schroeder Diffusion Operator (`Op-T`).
  - [`op_m_modal_kautz.md`](../dsp/op_m_modal_kautz.md): Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator (`Op-M`).
  - [`op_n_commuted_cascade.md`](../dsp/op_n_commuted_cascade.md): Universal Nonlinear, Hysteretic & Commuted Cascade Operator (`Op-N`).
  - [`router_and_stages_topology.md`](../dsp/router_and_stages_topology.md): Stage 0 Router, Canonical Stages I–IV & 256-D Output Head.
  - [`parametric_control_bus.md`](../dsp/parametric_control_bus.md): Self-Similar 3 kHz Dual-Rate Control Plane & Modulation Bus.
- **Training Pipeline & Control Audit (`docs/specs/training/`)**
  - [`four_stage_pipeline.md`](../training/four_stage_pipeline.md): Two-Sweep Block-Coordinate Cholesky Training Pipeline.
  - [`iterative_resume_and_caching.md`](../training/iterative_resume_and_caching.md): Dual-Space Factored Control Stem Resume.
  - [`control_space_audit.md`](../training/control_space_audit.md): Non-Stop 3-Phase Control-Space Audit Specification.
  - [`capture_and_alignment.md`](../training/capture_and_alignment.md): Hardware Excitation, Multi-Mic Calibration & Router Alignment.
- **Container Binary Formats (`docs/specs/containers/`)**
  - [`dnam_binary_format.md`](../containers/dnam_binary_format.md): 64B-Aligned Hybrid Binary Wire Format (`DNAM`).
  - [`checkpoint_and_cache_format.md`](../containers/checkpoint_and_cache_format.md): Checkpoint (`.dnam-ckpt`) & Sidecar Cache (`.dnam-cache`).
- **Security, Provenance & Marketplace (`docs/specs/security/`)**
  - [`provenance_and_enclave_drm.md`](../security/provenance_and_enclave_drm.md): Ed25519 Signatures, Enclave DRM & Protection Profiles.
  - [`watermarking_and_marketplace.md`](../security/watermarking_and_marketplace.md): Buyer Watermarking, Structural Fingerprint & Storefront Previews.
- **Architectural Decision Records (`docs/adr/`)**
  - [`0002-tri-operator-recurrent-architecture.md`](../../adr/0002-tri-operator-recurrent-architecture.md) through [`0015-realtime-host-concurrency-spsc-and-telemetry-seqlock.md`](../../adr/0015-realtime-host-concurrency-spsc-and-telemetry-seqlock.md).
