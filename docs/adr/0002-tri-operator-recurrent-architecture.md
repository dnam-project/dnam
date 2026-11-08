---
id: ADR-0002
title: "Tri-Operator Recurrent Architecture (TORA) & 4-Tier Slimmable Hierarchy"
status: accepted
supersedes:
  - ADR-0001
superseded_by: null
specs:
  - docs/specs/architecture/tora_architecture.md
  - docs/specs/dsp/router_and_stages_topology.md
requirements:
  - REQ-1.2
  - REQ-1.3
  - REQ-1.4
  - REQ-1.5
  - REQ-5.1
  - REQ-5.6
---

# ADR-0002: Tri-Operator Recurrent Architecture (`TORA`) & 4-Tier Slimmable Hierarchy

This record (superseding [`ADR-0001`](0001-superseded-tri-physics-prototype.md)) justifies the canonical [`tora_architecture.md`](../specs/architecture/tora_architecture.md) and [`router_and_stages_topology.md`](../specs/dsp/router_and_stages_topology.md) specifications.

## 1. Context: Five Structural Bottlenecks in Conventional Neural Audio

Conventional neural audio architectures that pair deep dilated-convolution stacks with unsegmented feedback delay networks encounter five structural bottlenecks when deployed on embedded hardware (`ARM Cortex-A76`, `64 KB` L1D cache) and full-rig studio captures ([`ADR-0001`](0001-superseded-tri-physics-prototype.md)):

```mermaid
flowchart LR
    subgraph Challenges ["Conventional Neural Audio Bottlenecks"]
        direction TB
        G1["1. Trainer-Engine Discrepancy<br/>Non-convex BPTT over dilated conv stacks (30,720 B)<br/>vs. closed-form Wiener-Hammerstein fits (> 12 dB mismatch)"]
        G2["2. L1D Cache Thrashing<br/>201.7-229.2 KB State RAM (> 3.5x 64 KB Cortex-A76 L1D)<br/>(12-18 ns L2 miss jitter)"]
        G3["3. Amp vs. Spatial Tank Exclusivity<br/>Separate amp-only or delay-only training branches<br/>(Blocks REQ-1.5 full-rig capture)"]
        G4["4. Truncated Cabinet IR Tails<br/>8,192-tap (170.7 ms) cabinet FIRs cannot run at 0 latency<br/>within embedded cycle budgets (truncated to 6.27 ms)"]
        G5["5. Rigid 1D Ordering & Port Conflicts<br/>Fixed 1D pipelines & 4-port routers block Pre-FX before Amp,<br/>Cab between Amp & Post-FX, and 8-Port Stereo 4CM/Aux/WDW"]
    end

    subgraph TORA ["TORA Unified Isomorphic Resolution"]
        direction TB
        R1["Stage II Op-N Commuted Cascade<br/>Direct serialization of fitted W_stage^(2.1..2.3)"]
        R2["Commuted stages + partitioned Op-T buffers<br/>Compact fixed state + model-declared delay rings"]
        R3["Analytical VarPro + Block-Coordinate Cholesky<br/>Joint Stage I-IV full-rig training + tier-aware stem resume"]
        R4["Stage III 64-Tap Early SIMD FIR + Kautz Bank<br/>Native 170 ms cabinet diffraction & room tail @ 0 latency"]
        R5["8-Port Matrix Router + 4 Canonical Stages<br/>Pre-FX -> Amp -> Cab -> Spatial Tank + Stereo 4CM/7CM/WDW"]
    end

    G1 --> R1
    G2 --> R2
    G3 --> R3
    G4 --> R4
    G5 --> R5
```

| Challenge | Traditional Bottleneck Mechanism | TORA Architectural Resolution |
| :--- | :--- | :--- |
| **1. Trainer–Engine Isomorphism** | Multi-layer dilated convolutions (`K=6`, `d ∈ {1, 3, 7, 19}`) consume `30,720 B` (`48.00%` of a `64,000 B` container) and `65,536 B` (`64.0 KB`) of convolution history RAM, yet are non-convex under BPTT on `1/f²` audio (`κ(H) > 10¹⁰`) with only a `301`-sample (`6.27 ms`) sparse receptive field. | **Stage II Commuted Nonlinear Cascade (`Op-N`):** Serializes fitted stage matrices directly and enforces identical project-then-filter ordering in training and runtime. |
| **2. Embedded L1D Cache Residency** | ARM Cortex-A76 cores (Rockchip RK3582 on Darkglass Anagram) have a **`64 KB` L1 Data (L1D) Cache**. Uncommuted convolution histories (`65,536 B`), multi-tap head histories (`16,384 B`), and unsegmented diffuser rings (`131,072 B`) reach **`229,184 B` (`> 3.5×` L1D)**. | **Commuted Stage II + Partitioned Op-T Histories:** Eliminates `conv_hist` and `head_tap_hist` while cold-allocating exact 64B-aligned model-declared delay histories. |
| **3. Joint Full-Rig Capture (`REQ-1.5`)** | Treating nonlinear amp modeling and time-warped spatial delay/reverb tanks as mutually exclusive pipelines prevents single-pass capture of complete rigs (Pre-FX + tube amp + speaker cabinet + stereo post-FX). | **Analytical `VarPro` + Block-Coordinate Cholesky Pipeline ([`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md)):** Solves Stage I–IV sequentially on forward residuals and re-balances the joint `256-D` head across all four tiers. |
| **4. Zero-Latency `170 ms` Speaker Cabinets (`REQ-1.2`)** | An `8,192`-tap (`170.7 ms` at `48 kHz`) direct FIR filter for speaker cabinet diffraction and room decay cannot execute at zero latency within embedded cycle budgets. | **Stage III Hybrid Cabinet (`Op-M`):** Combines tier-specific early FIR taps (`16/32/48/64` taps/ch) with a shared Kautz modal bank (`4/8/16/32` pairs) beginning at sample 16 (`0.33 ms` at `48 kHz`). |
| **5. Canonical Signal Routing & 8-Port I/O (`REQ-1.3`, `REQ-1.5`)** | A rigid 1D pipeline cannot place Pre-FX before amp clipping, cabinets between amp and stereo post-FX, or represent 4CM/7CM/WDW modes. | **Stage 0 8-Port Matrix Router + 4 Canonical Stages:** Provides four physical inputs and four outputs with return-to-send isolation, `200 Hz` LR4 split, and parallel sub-sample alignment. |

### 1.1 Why TORA Outperforms Grey-Box Multi-FX Approximations & Enables Offline SPICE Distillation

Full white-box circuit simulators (such as SPICE or coupled multi-nonlinear Wave Digital Filters solving stiff differential-algebraic equations via iterative Newton–Raphson at `8×–16×` oversampling) have unbounded worst-case execution time (WCET) and are far too computationally expensive to run in real time on embedded pedalboards. Consequently, commercial digital multi-effects pedals (such as Line 6 Helix/HX, Fractal Axe-Fx, Neural DSP Quad Cortex, and Darkglass factory blocks) rely on five real-time **"grey-box" engineering approximations**:

1. **Manual Stage Decoupling (Wiener–Hammerstein Cascades):** Engineers slice schematics at coupling capacitors into isolated `LTI Filter → Static Nonlinearity → LTI Filter` stages, discarding bidirectional impedance loading and multi-stage reactive coupling.
2. **Low-Dimensional Lookup Tables (`1D`/`2D` LUTs):** Local implicit diode/triode feedback loops are solved offline (via the DK-method or single-root WDFs) and baked into `1D` or `2D` LUTs. Because LUT memory explodes exponentially beyond two dimensions, circuits with `3+` simultaneous reactive states and panel controls must be simplified.
3. **Truncated `1`-Step Newton–Raphson + High Oversampling:** Analytical nonlinearities run at `2×–8×` oversampling with only `1–2` fixed Newton–Raphson iterations seeded from `y[n-1]`, trading transient slew accuracy and per-block oversampling filter latency for bounded CPU time.
4. **Symbolic Biquads & TPT State-Variable Filters (SVFs):** Passive tone stacks, wahs, and phasers are reduced to hand-derived rational transfer functions `H(s)` mapped to biquads or TPT SVFs.
5. **Phenomenological Delay & FDN Stand-Ins:** Analog BBD chips (`4,096` MOSFET stages) and magnetic tape loops are replaced by fractional delay buffers, LFOs, and feedback delay networks (FDNs) with low-pass filters and soft clippers in the loop.

```mermaid
flowchart LR
    subgraph Sources ["Hardware & Offline White-Box Sources"]
        direction TB
        HW["Physical Analog / Digital Gear<br/>(Amps, Cabs, Pedals, Studio Racks)"]
        SPICE["Uncompromised Offline SPICE / Faust<br/>(Iterative DAE Solvers @ 8x-16x OS)"]
        Chain["Multi-Plugin Studio Chains<br/>(N Chained White-Box / DAW Blocks)"]
    end

    subgraph Trainer ["Pure-Rust Analytical VarPro & Cholesky Trainer (ADR-0017)"]
        direction TB
        Fit["Joint Stage I-IV & CP1 Control-Space Fit<br/>(Captures Cross-Stage Loading & Hysteresis)"]
    end

    subgraph Runtime ["TORA Real-Time Engine (O(1) Constant WCET)"]
        direction TB
        Kernels["3 Universal Zero-FDIV SIMD Kernels<br/>• Op-N: Multi-Drive PolyADAA, C2 Wavefolder, 2D Hysteresis & ZOH<br/>• Op-M: Symplectic Rotors, H1 Hilbert, T1 Dividers & Kautz Modals<br/>• Op-T: 4-Tap Catmull-Rom Delay, Granular Pitch & Unitary FDN Tank"]
    end

    HW & SPICE & Chain -->|"Closed-Loop Adaptive Excitation<br/>& White-Box Distillation (ADR-0016, REQ-1.7/1.8)"| Fit -->|"Declarative 64B-Aligned .dnam<br/>(Zero Executable Code, REQ-8.8)"| Kernels
```

`TORA` (`ref[arch.operators.tri-kernel]`, `ref[arch.families.coverage]`) unifies the mathematical superset of those real-time DSP building blocks into three zero-`FDIV` SIMD operator kernels (`Op-N`, `Op-M`, `Op-T`) and solves their cross-stage projections jointly from closed-loop adaptive captures or white-box distillations ([`ADR-0016`](0016-codesigned-disjoint-comb-stimulus-and-null-homing-capture.md), [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), [`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)):

| Dimension | Commercial Factory Multi-FX ("Grey-Box" C++ Blocks) | Black-Box Neural Captures (`NAM A2`, `Proteus`, QC Capture) | **DNAM (`TORA` Unified Physical & Digital Basis)** |
| :--- | :--- | :--- | :--- |
| **Nonlinear & Reactive Coupling** | Hand-decoupled sub-circuits; `1D`/`2D` LUTs (`≤ 2` coupled states) or `1`-step Newton–Raphson at `2×–8×` oversampling. | Generic mono `WaveNet` / `LSTM` layers; no long-tail modal or spatial delay memory. | **Jointly fitted across all 4 stages (`ref[arch.isomorphism.two-sweep]`):** `Op-N` `1×` B-Spline `PolyADAA`, 6-step division-free rail crossing, degree-7/8 `C²` wavefolder, 2D phase-space hysteresis, and B+ sag. |
| **Time, Modulation, Pitch & Synth** | Bespoke per-pedal C/C++ algorithms; users cannot capture their own modulation, pitch, or reverb gear. | Cannot capture chorus, flangers, auto-wahs, delays, reverbs, shimmer, synths, or freeze. | **Captures & distills all 7 Hardware Families (`ref[arch.families.coverage]`):** `Op-M` symplectic rotors, `H1` Hilbert SSB, `T1` sub-octave dividers, Kautz modals, and `Op-T` Hermite-Hann pitch/barberpole/FDN tanks. |
| **Offline SPICE / White-Box Distillation** | Requires manual schematic reduction and custom C/C++ coding per circuit. | Fails on time-varying, modulated, pitch-shifted, or long-decay white-box algorithms. | **Direct offline-to-realtime distillation ([`ADR-0018`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)):** Any offline SPICE/Faust/WDF solver or multi-plugin studio patch can be rendered and distilled into a constant-time `.dnam` model. |
| **Multi-Block Chain Cost (`N` Effects)** | Scales linearly `O(N)` in CPU, intermediate buffer copies, per-block oversampling filters, and L1I/L1D cache pressure. | Requires `2×` plugins for stereo and `4–6×` chained plugins for a full rig (`> 400 KB` state spilling L2 cache). | **`O(1)` single-instance execution (`ref[arch.tiers.hardware-budget]`):** Collapses `Pre-FX → Amp → Zero-Latency Cab → Spatial Tank` into one pass (`14–20× Full` or `22–30× Lite` stereo rigs per `Cortex-A76` core). |
| **Shipping New Effects (`REQ-8.8`)** | Requires compiling and shipping new firmware or native `.so` binaries with variable WCET. | Static mono JSON/weights for amps/drives only. | **Pure declarative data (`ref[container.dnam.formats]`):** Zero executable code surface, guaranteed 3-phase stability (`ref[train.audit.phase1.clamp-stability]`), and 4 nested tiers (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`). |

---

## 2. Decision: Core Product & System Topology Choices

| Domain | Canonical Decision | Superseded / Rejected R&D Alternative | Normative Spec & Test |
| :--- | :--- | :--- | :--- |
| **Slimmable Tier Hierarchy (`REQ-5.1`)** | Four nested tiers (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`). `Lite..Pro` use `8-D` static control (`4` head + `4` stage, Rank-4 FM); `Ultra` adds an `8-D` delta (`+4` head + `+4` stage, Rank-8 FM) and a `16 × 4` stage-dispatch delta in `Arena 3`. | Capping `Ultra` at `4` stage coordinates (creates a rank-4 bottleneck across the 8 physical stage mechanisms) or baking `16-D` into `Lite..Pro`. | [`parametric_control_bus.md` §2.3](../specs/dsp/parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling) |
| **Stage III Cabinet Layout (`REQ-1.2`, `REQ-5.6`)** | Separate 64B-aligned per-tier slices for independent L/R early FIR taps (`16 / 32 / 48 / 64` taps/ch) and shared-pole Kautz modal pairs (`4 / 8 / 16 / 32` pairs) with independent L/R readouts. Fixed 16-sample reference modal onset (`x[n-16]`). Both channels advance in all I/O topologies. | Single unpartitioned `384 B` Stage III slot per tier (which collided `stage3_early_fir` with `stage2_filter_params` and assumed 16 poles/tier). | [`op_m_modal_kautz.md` §4.1](../specs/dsp/op_m_modal_kautz.md#41-stage-iii-architecture--tier-allocation), [`dnam_binary_format.md` §2.4](../specs/containers/dnam_binary_format.md#24-tier-local-projection--stage-iii-slice-layout) |
| **Hardware Freeze (`REQ-1.4`)** | Freeze is modeled only when present and captured on the target hardware; unobserved options remain clamped. Stage IV freeze stops tank input injection while Stages I–III continue processing live play-over input. | Post-head synthetic native freeze with a separate `4,096 B` snapshot buffer and artificial visibility/blending rules. | [`router_and_stages_topology.md` §6.1](../specs/dsp/router_and_stages_topology.md#61-captured-freeze-and-modeled-controls) |

## 3. Companion Subsystem ADRs

- [`ADR-0003: Floor-Regularized Band-Balanced bESR & Rectangular Noise Qualification`](0003-band-balanced-besr-and-noise-qualification.md)
- [`ADR-0004: Unclamped 6-Step PolyADAA Rail Crossing, C² Wavefolder & Fixed-Hertz ZOH Bitcrusher`](0004-division-free-polyadaa-and-c2-wavefolder.md)
- [`ADR-0005: Quadrant-Reduced Contractive Rotor, H1 Hilbert Lattice, T1 Tracker & Stage III ZOH`](0005-contractive-rotor-hilbert-and-t1-tracker.md)
- [`ADR-0006: Model-Declared fp16 Delay Histories, Centered Granular Pitch & Stage IV Feedback Guard`](0006-model-declared-delays-and-tank-feedback-guard.md)
- [`ADR-0007: Centered ANOVA + Rank-R FM Control Bus, Exact-Integer Accumulator & Physical Tempo Inversion`](0007-exact-integer-control-bus-and-tempo-inversion.md)
- [`ADR-0008: Per-Stem uplo='U' Kronecker Statistics & 192-Bit Wide-Significand Gram Contraction`](0008-per-stem-kronecker-statistics-and-wide-gram.md)
- [`ADR-0009: RFC 8785 JCS Header Digest, Prefix-Composable Signatures, RFC 9180 HPKE & Rank-r Preview Containment`](0009-jcs-prefix-signing-hpke-and-preview-containment.md)
- [`ADR-0016: Closed-Loop Feedback-Adaptive Disjoint-Comb Stimulus, Mode A/B Control Sweeps & Acoustic Null Homing`](0016-codesigned-disjoint-comb-stimulus-and-null-homing-capture.md)
- [`ADR-0017: Pure-Rust Analytical VarPro Solver, 256-D Schur Re-Balancing & EMR-Weighted MM-Cholesky KKT`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md)
- [`ADR-0018: Pure-Rust Signal-Chain Fixtures Crate (dnam-fixtures), Dual-Mode White-Box Distillation & Python Retirement`](0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)

