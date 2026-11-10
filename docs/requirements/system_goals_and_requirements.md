---
id: req.system-goals
title: "System Goals, Functional Requirements & Hardware Budgets"
status: canonical
version: "1.0.0"
layer: requirements
requirement_prefix: REQ
---

# System Goals, Functional Requirements & Hardware Budgets

This specification defines the **eight implementation-agnostic system goals, functional requirements (`REQ-1`–`REQ-8`), real-time commodity hardware budgets, and model/marketplace security guarantees** that govern **Dynamic Neural Audio Modeler (DNAM)**.

All requirements in this document are stated independently of internal model topology, cryptographic primitive, or solver implementation:
- For the perceptual quality metric (`bESR`) and 4-tier acceptance thresholds (`Lite`, `Full`, `Pro`, `Ultra`), see [`psychoacoustic_quality_and_tiers.md`](../specs/architecture/psychoacoustic_quality_and_tiers.md).
- For the concrete "TORA" DSP & training architecture that fulfills these goals (and how it compares to `NAM A2`, `GuitarML PROTEUS`, and `NAM ParametricOD`), see [`tora_architecture.md`](../specs/architecture/tora_architecture.md).
- For the concrete cryptographic provenance, hardware enclave DRM, forensic watermarking, and anti-cloning architecture that fulfills Goal 8, see [`../specs/security/provenance_and_enclave_drm.md`](../specs/security/provenance_and_enclave_drm.md) and [`../specs/security/watermarking_and_marketplace.md`](../specs/security/watermarking_and_marketplace.md).

---

## 1. End-to-End Requirement Architecture

```mermaid
flowchart LR
    subgraph Goal12 ["Goals 1 & 2: System Coverage & Perceptual Quality"]
        G1A["Goal 1 (REQ-1): Nonlinear, LTI, Resonant, Delay & Complex Full-Rig Systems"]
        G1B["Goal 1 (REQ-1): Closed-Loop Adaptive Excitation, Null Homing & White-Box Distillation"]
        G2A["Goal 2 (REQ-2): Device-Agnostic 3-Band Noise-Adjusted bESR Tiers"]
        G2B["Goal 2 (REQ-2): Quiet Stems, Parametric Control, Tempo Sync & Non-Stop Clamping"]
    end

    subgraph Goal34 ["Goals 3 & 4: Iterative Resume & Deterministic Training"]
        G3A["Goal 3 (REQ-3): Single-Checkpoint Incremental Stem Refinement"]
        G3B["Goal 3 (REQ-3): Zero Old-Stem Drift & Bounded Workspace Cache"]
        G4A["Goal 4 (REQ-4): Deterministic Zero-GPU Native & Browser Execution (< 100 ms) + float64 Accuracy"]
        G4B["Goal 4 (REQ-4): Bounded Low-Memory Footprint (< 256 MiB Host / < 64 MiB Browser Worker)"]
    end

    subgraph Goal56 ["Goals 5 & 6: Single-Run Slimmable Tiers & Real-Time Budgets"]
        G5A["Goal 5 (REQ-5): Nested Slimmable Hierarchy (Lite < Full < Pro < Ultra)"]
        G5B["Goal 5 (REQ-5): Single-Run Joint Multi-Tier Training (No Per-Tier Retraining)"]
        G5C["Goal 5 (REQ-5): Dynamic Tier Switching, Bounce Promotion & Tier Stripping"]
        G6A["Goal 6 (REQ-6): Embedded Budget (>= 10x Lite, >= 6x Full, >= 2x Pro @ 48 kHz)"]
    end

    subgraph Goal78 ["Goals 7 & 8: Unclipped f32 I/O, Channel Topologies & Marketplace Security"]
        G7A["Goal 7 (REQ-7): Unclipped f32 I/O, Mono/True-Stereo (1->1, 2->1, 1->2, 2->2) & Stability"]
        G8A["Goal 8 (REQ-8): Creator Provenance, Distribution Policy & 3 DRM Profiles"]
        G8B["Goal 8 (REQ-8): Zero-Degradation Buyer Watermarking & Collusion Resistance"]
        G8C["Goal 8 (REQ-8): Multi-Stage Anti-Cloning Gate & Automated Device Revocation"]
    end

    Goal12 --> Goal34 --> Goal56 --> Goal78
```

---

## 2. Goal 1 — Universal Instrument, System & Flexible Input Coverage (Nonlinear + LTI + Full Rigs)

A conforming system must accurately model four fundamental physical classes of audio equipment and acoustic spaces—both as isolated hardware units and combined into **complex, end-to-end full rigs**—across the full audible bandwidth ($20\text{ Hz}\text{–}20\text{ kHz}$) for any musical source (electric/acoustic guitar, bass, synthesizers, drums, vocals, acoustic instruments, and mastering gear):

| Requirement ID | Physical System Class | Representative Hardware & Acoustic Targets | Required Physical Phenomena |
| :--- | :--- | :--- | :--- |
| **`REQ-1.1`** | **Nonlinear Preamps, Amplifiers & Drives** | Tube heads (`Preamp → Tone Stack → Driver → Push-Pull Power/OT`), solid-state/CMOS/JFET drives, comparator/octave wavefolder fuzzes, optical/FET compressors, analog tape | Multi-stage soft/hard clipping, asymmetric conduction, dual-polarity interstage bias shift, power-supply sag, and folded phase-space hysteresis |
| **`REQ-1.2`** | **Linear, Time-Invariant (LTI) Systems** | Speaker cabinets (`1x15` ported flip-top cabs, `4x12` guitar cabs, multi-mic setups), passive/active EQs, studio rooms, early reflections, chambers, plates, halls | Multi-resonance impulse responses: bass-reflex port resonances (`40–110 Hz`), speaker cone breakup modes (`1.5–6 kHz`), mic phase/comb notches (`3–16 kHz`), and reverberant decay |
| **`REQ-1.3`** | **Selective State-Space & Resonant Filters** | Envelope filters / auto-wahs, analog resonant `LP`/`BP`/`HP` filters (including self-oscillating high-`Q` peaks), dynamic EQs, phasers, Uni-Vibe, harmonic tremolo | Signal-envelope, free-running LFO, and tempo-synced (`REQ-2.5`) resonant frequency and damping trajectories with preserved resonance `Q` across wide sweeps |
| **`REQ-1.4`** | **Modulated & Time-Warped Delay Networks** | Through-zero flangers, stereo chorus, rotary Doppler, BBD/tape echoes (with self-oscillation and rhythmic subdivisions), spring reverb drip, hall reverbs, granular clouds, pitch shimmer, spectral freeze | Click-free fractional delay modulation (both manual `ms`/`Hz` and host/MIDI tempo-synced subdivisions per `REQ-2.5`), dispersive all-pass chirp, stereo feedback diffusion, windowed time/pitch warping, and modeled freeze responses, including lossless infinite hold when present in the captured source |
| **`REQ-1.5`** | **Complex Full-Rig & Multi-Block Signal Chains** | End-to-end multi-unit studio/stage rigs spanning pre-effects (compressors, envelope filters, fuzzes/drives, pitch/octave dividers), nonlinear preamps/amplifiers, LTI speaker cabinets (`IR`s) and microphones, and post-amp time-varying modulation, delay, granular/shimmer, and reverberant acoustic spaces (e.g., `Compressor → Envelope Filter → Drive/Fuzz → Tube Amp → Speaker Cab + Mic → Stereo Chorus/Delay → Hall Reverb`, or multi-path bass rigs combining sub-bass compression with high-pass distortion and cabinet simulation) | Single unified model capturing arbitrary series, parallel (wet/dry or frequency-split), and mono/stereo-routed (`1 → 1`, `2 → 1`, `1 → 2`, `2 → 2`, and 4-Cable-Method `4CM` per `REQ-7.4`) signal chains—supporting both fixed/knobless full-rig captures (`0` exposed controls, treating the entire multi-unit rig as a single black-box signal chain) and parametric/tempo-synced full-rig models (`range` & `switch` controls across the rig)—while jointly resolving compound cross-block interactions (`REQ-1.1`–`REQ-1.4`) within the same `bESR` quality tiers (`REQ-2.1`–`REQ-2.2`), single-run slimmable tier hierarchy (`REQ-5.1`–`REQ-5.2`), and real-time embedded hardware budgets (`REQ-6.1`) |

### Complex End-to-End Full-Rig & Multi-Block Signal Chain Coverage (`REQ-1.5`)

The system must not be limited to isolated single pedals or standalone amplifier heads; it must capture and model **complete, end-to-end multi-unit rigs as a single unified model**:

```mermaid
flowchart LR
    subgraph Pre ["1. Pre-Effects & Routing"]
        P1["Series, Parallel (Wet/Dry) or<br/>Frequency-Split Crossover Paths"]
        P2["Compressors, Envelope Filters,<br/>Fuzzes/Drives & Pitch/Octave Dividers"]
        P1 --> P2
    end

    subgraph AmpCab ["2. Nonlinear Amp & LTI Acoustic Core"]
        A1["Nonlinear Preamps, Tone Stacks<br/>& Push-Pull Power Amps (REQ-1.1)"]
        A2["LTI Speaker Cabinets (IRs),<br/>Multi-Mic Phase & Room Modes (REQ-1.2)"]
        A1 --> A2
    end

    subgraph Post ["3. Post-Amp Stereo Time-Varying Space"]
        S1["Stereo Chorus, Flanger, Phaser<br/>& Resonant Sweeps (REQ-1.3, REQ-1.4)"]
        S2["Modulated Delay, Granular/Shimmer<br/>& Reverberant Spaces (REQ-1.4)"]
        S1 --> S2
    end

    Pre --> AmpCab --> Post --> Unified["Single Unified Full-Rig Model:<br/>• Fixed/Knobless (0 Controls, Black-Box Chain)<br/>• Parametric (Arbitrary Range & Switch Controls)<br/>• Same bESR Tiers (REQ-2), Slimmable Hierarchy (REQ-5)<br/>  & Real-Time Embedded Budgets (REQ-6.1)"]
```

| Full-Rig Requirement Aspect | Normative Specification (`REQ-1.5`) |
| :--- | :--- |
| **Arbitrary Multi-Block Topologies** | Must model arbitrary series, parallel (wet/dry blend or frequency-split crossover, such as sub-bass compression combined with high-pass distortion and cabinet simulation), and mono (`1 → 1`, `2 → 1`) or true-stereo (`1 → 2`, `2 → 2`, `REQ-7.4`) chains spanning pre-effects, amplifiers, speaker cabinets (`IR`s) + microphones, and post-amp modulation/delay/granular/reverb units as one unified model. |
| **Fixed/Knobless vs. Parametric Full-Rig Modes** | Must support both **fixed/knobless full-rig captures** (`0` exposed controls, treating the entire multi-unit signal chain as a single black-box system) and **parametric full-rig models** exposing arbitrary user-selected continuous `range` knobs and discrete `switch` selectors across one or more units in the rig. |
| **Compound Cross-Block Physical Interactions** | Must jointly resolve inter-stage physical coupling—including nonlinear preamp/power-amp harmonics exciting speaker-cabinet and room resonances (`REQ-1.1` + `REQ-1.2`), envelope/LFO filter sweeps feeding or following nonlinear clipping (`REQ-1.3`), and stereo diffusive delay/reverb tails (`REQ-1.4`)—while meeting the same device-agnostic `bESR` quality tiers (`REQ-2.1`–`REQ-2.2`), single-run slimmable tier hierarchy (`REQ-5.1`–`REQ-5.2`), and real-time embedded hardware budgets (`REQ-6.1`). |

### Flexible Dry/Wet Stem & Excitation Requirements (`REQ-1.6`–`REQ-1.8`)

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-1.6`** | **Zero Control-Name or Manifest Heuristics** | Capture excitation synthesis and model training must never rely on control names or manifest heuristics to infer circuit behavior, ensuring fixed/knobless rigs and multi-effect + amp + cab chains (`REQ-1.5`) are supported identically ([`../specs/training/capture_and_alignment.md` §1.4](../specs/training/capture_and_alignment.md#14-heuristic-free-300-second-5-section-universal-excitation--live-tail-extension)). |
| **`REQ-1.7`** | **Closed-Loop Feedback-Adaptive & Co-Designed Full-Bandwidth Excitation, Musical Transient Coverage & Dynamic Tail Extension** | Must provide a deterministic, full-bandwidth (`20 Hz–20 kHz`) universal excitation signal for any musical source that excites linear frequency response, harmonic and intermodulation nonlinearities, dynamic envelope ballistics, and musical transient trajectories without manual per-device tuning, with real-time feedback adaptation of spectral pre-emphasis, operating knee/gate levels, high-resonance frequency zoom, and early convergence termination (plus automatic live recording extension whenever a stem's wet decay tail continues ringing above the measured noise floor and in-window guard-band bleed tracking). |
| **`REQ-1.8`** | **Adaptive Continuous/Discrete Control Capture, Acoustic Null Homing & White-Box Distillation** | Must support adaptive continuous-sweep and stationary-stop hardware control capture with live acoustic null homing (`< 0.05 dB`) alongside programmatic white-box and grey-box target distillation from circuit simulators, DSP languages, scientific computing runtimes, and audio plugins. |

---

## 3. Goal 2 — Perceptual Studio Quality (`bESR`), Parametric Control & Quiet Stem Handling

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-2.1`** | **Perceptual Band-Balanced Metric (`bESR`)** | Plain broadband `ESR` must not be used as a quality criterion because `1/f²` low-frequency dominance, time-domain MSE high-frequency phase-mismatch bias, and loud-passage masking hide upper-harmonic errors. Report **Floor-Regularized, Noise-Adjusted Band-Balanced `bESR`** separately for each applicable output channel. Its raw score weights **Low (`20–250 Hz`)**, **Mid (`250 Hz–2.5 kHz`)**, and **High (`2.5–18 kHz`)** equally and can never be more than `4.77 dB` (`10 log₁₀ 3`) better than that channel's worst single band. Apply the separately reported noise-qualified composite acceptance rule ([`psychoacoustic_quality_and_tiers.md`](../specs/architecture/psychoacoustic_quality_and_tiers.md)). |
| **`REQ-2.2`** | **Device-Agnostic Quality Standard** | One identical, signal-measured quality standard per tier (`Lite`, `Full`, `Pro`, `Ultra`) applies across all device types—with zero subjective `"clean vs. high-gain"` carve-outs—by accounting for sub-sample clock jitter and applicable measured per-channel, per-band noise floors. Each applicable output must pass independently; qualifying noise-limited bands follow the specified band and composite acceptance rules, and silent inputs alone do not establish noise-only output ([`psychoacoustic_quality_and_tiers.md` §2](../specs/architecture/psychoacoustic_quality_and_tiers.md#2-device-agnostic-quality-evaluation-zero-subjective-carve-outs)). |
| **`REQ-2.3`** | **Quiet Tonal Stems vs. Muted Stems** | Quiet outputs whose active signal exceeds their measured noise floor by more than `6.0 dB` (e.g., a minimum-volume stem that reduces level without fully muting) must be fitted and evaluated as active rather than discarded by a fixed level cutoff. A genuinely muted output follows the measured-noise-floor residual rule independently of other channels. |
| **`REQ-2.4`** | **Parametric Conditioning & Non-Stop Clamping** | Models must interpolate smoothly across intermediate continuous `range` and discrete `switch` control states while automatically clamping unobserved ranges, unrecorded switch options, and unstable multi-knob corners without pausing training for user confirmation ([`../specs/training/control_space_audit.md`](../specs/training/control_space_audit.md)). |
| **`REQ-2.5`** | **MIDI / Host Tempo Sync & Rhythmic Subdivision** | Time-based model parameters—including delay and echo repeat times, modulation rates (chorus, flanger, phaser, tremolo, rotary, auto-wah, and cyclic sweeps), granular/slicer periods, and rhythmic step trajectories—must natively support both **free-running manual mode (`Hz` / `ms`)** and **tempo-synchronized rhythmic mode (`BPM`)** locked to host transport tempo, external MIDI clock (`24 PPQN` beat clock), or tap tempo across all **12 canonical musical subdivisions** (`1/2.`, `1/2`, `1/2T`, `1/4.`, `1/4`, `1/4T`, `1/8.`, `1/8`, `1/8T`, `1/16.`, `1/16`, `1/16T`, spanning dotted `1.5×`, straight `1.0×`, and triplet `2/3×` note feels), including independent Left/Right polyrhythmic subdivisions (e.g., `1/4` + `1/8.` ping-pong). Tempo synchronization must preserve the modeled hardware's time-dependent tonal and spatial diffusion characteristics (including automatic range selection on multi-range units), reject external clock/tap jitter without pitch warble, and transition across continuous tempo ramps or discrete subdivision changes without clicks or pitch discontinuities while meeting all real-time execution constraints (`REQ-6.1`–`REQ-6.3`) ([`../specs/dsp/parametric_control_bus.md` §7](../specs/dsp/parametric_control_bus.md#7-host-midi-and-tap-tempo-synchronization-req-25), [`tora_architecture.md` §8](../specs/architecture/tora_architecture.md#8-subsystem-specification-index)). |

### 3.1 MIDI / Host Tempo Sync & 12-Subdivision Rhythmic Grid (`REQ-2.5`)

To support studio DAW automation and live pedalboard performance without manual millisecond/Hertz conversion, any time-varying delay, modulation, granular, or sequenced model must support seamless switching between manual parameter control and musical tempo synchronization:

| Tempo Sync Aspect | Normative Requirement (`REQ-2.5`) |
| :--- | :--- |
| **Clock Sources & Jitter Rejection** | Must lock to (1) **Host DAW / Plugin Transport Tempo (`BPM` & musical beat/bar phase `PPQ`)**, (2) **External MIDI Beat Clock (`0xF8` at `24 PPQN` with `Start` `0xFA` / `Continue` `0xFB` / `Stop` `0xFC`)**, and (3) **Hardware Footswitch / UI Tap Tempo**, rejecting external clock arrival jitter and human tap variance so sustained repeats never exhibit pitch warble, with instant fallback to **Free-Running Manual Mode (`ms` / `Hz`)** when tempo sync is disabled. |
| **Supported Tempo Range** | Support finite tempos from **20 through 400 BPM**, inclusive, for all three clock sources. Report unsupported or invalid values; never present a silently clamped tempo as synchronized to the requested source. |
| **Transport Start & Position Jumps** | Initialize synchronized phase to the supplied position on the first frame of playback start and offline bounce. For in-play jumps, provide smooth shortest-arc relocking by default and an immediate phase-lock option with parameter micro-ramping; retain delay history and validate audible transitions. |
| **12 Canonical Subdivisions, Dual Polyrhythms & Spatial Preservation** | Must support all 12 rhythmic subdivisions relative to a quarter-note beat (`T_quarter = 60 / BPM` seconds): **`1/2.`** (`3.0×`), **`1/2`** (`2.0×`), **`1/2T`** (`4/3×`), **`1/4.`** (`1.5×`), **`1/4`** (`1.0×`), **`1/4T`** (`2/3×`), **`1/8.`** (`0.75×`), **`1/8`** (`0.5×`), **`1/8T`** (`1/3×`), **`1/16.`** (`0.375×`), **`1/16`** (`0.25×`), and **`1/16T`** (`1/6×`), supporting independent Left/Right dual subdivisions (e.g., `1/4` + `1/8.` ping-pong) while preserving the natural spatial diffusion of multi-tap and reverberant tails. |
| **Coupled Timbral Fidelity & Multi-Range Continuity** | Synchronizing a time-based control must preserve the hardware's time-dependent tonal behavior—such as analog clock/tape high-frequency rolloff, feedback damping, and automatic range selection (or octave folding) when a target subdivision spans multiple hardware time ranges—across both desktop and zero-allocation embedded runtimes. |
| **Musical Phase Alignment & Artifact-Free Transitions** | Musical bar/beat phase (`PPQ`) must align periodic modulation cycles and rhythmic/granular windows to downbeats without causing discontinuities in active delay repeats; continuous tempo ramps must glide smoothly, whereas large tempo jumps or discrete subdivision changes must transition without clicks or unnatural pitch jumps while satisfying `REQ-6.1`–`REQ-6.3`. |

---

## 4. Goal 3 — Progressive Model Refinement & Low-Memory Caching

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-3.1`** | **Progressive Single-Checkpoint Refinement** | Models must support progressive refinement with additional or re-recorded control-state stems from a single training checkpoint without retraining unchanged stems from scratch whenever incremental refinement is faster than full retraining ([`../specs/training/iterative_resume_and_caching.md` §1](../specs/training/iterative_resume_and_caching.md#1-single-checkpoint-dnam-ckpt-two-tier-iterative-stem-expansion)). |
| **`REQ-3.2`** | **Zero Old-Stem Perturbation on Expansion** | Adding stems to expand a control's observed range or polynomial degree must preserve previously converged stem responses without degradation. |
| **`REQ-3.3`** | **Reusable Calibration & Bounded Workspace Cache** | Training checkpoints and a gitignored local workspace cache (with configurable size limits, a read-vs-regenerate cost model, and automatic cleanup) must persist reusable calibration and stem state across runs to avoid expensive recalculations on low-memory systems, while keeping exported runtime model containers stripped of training-only data ([`../specs/training/iterative_resume_and_caching.md` §2](../specs/training/iterative_resume_and_caching.md#2-two-tier-calibration-persistence-arena-5-dnca--workspace-dnam-cache)). |

---

## 5. Goal 4 — Deterministic Zero-GPU Training & Low-Memory Scaling

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-4.1`** | **Deterministic Zero-GPU Native & Browser Execution (`< 100 ms` Interactive Solve)** | Training must execute deterministically on any commodity host CPU and inside a browser background worker in `< 100 ms` (`< 25 ms` typical) with zero external GPU, TPU, or deep-learning framework runtime dependencies. |
| **`REQ-4.2`** | **Cross-Platform `float64` & Wide-Significand Solver Accuracy** | All sufficient-statistic contractions, rational pole identification, spectral-balance projections, and multi-tier linear solves must achieve `IEEE-754 float64` accuracy (using compensated or 192-bit wide-significand accumulation) for bit-reproducible results across desktop, server, and web runtimes ([`../specs/training/four_stage_pipeline.md` §3](../specs/training/four_stage_pipeline.md#3-accelerator-first-32-bit-execution-gram-accuracy--streaming)). |
| **`REQ-4.3`** | **Bounded Low-Memory Footprint (`< 256 MiB` Host / `< 64 MiB` Browser Worker) & Multi-Core Batch Scaling** | Training must operate within a `< 256 MiB` memory ceiling on desktop/server hosts and `< 64 MiB` inside browser worker sandboxes while scaling linearly across available CPU cores during multi-stem batch rendering. |

---

## 6. Goal 5 — Single-Run Slimmable Multi-Tier Architecture (`Lite` / `Full` / `Pro` / `Ultra`)

Rather than training, storing, and distributing separate disconnected models for different compute budgets (e.g., separate "Nano", "Feather", "Standard", and "Heavy" models that each require an independent training run), a conforming system must use a **nested slimmable multi-tier hierarchy (`Lite` $\subset$ `Full` $\subset$ `Pro` $\subset$ `Ultra`) where training at the highest target tier simultaneously trains all lower tiers in a single training run**:

```mermaid
flowchart LR
    subgraph Train ["1. Single-Run Joint Multi-Tier Training (REQ-5.1, REQ-5.2)"]
        Stems["Dry/Wet Capture Stems + Control Schema"] --> Joint["Single Training Run at Target Max Tier (e.g. Ultra)<br/>(Simultaneously Optimizes Lite, Full, Pro & Ultra;<br/>Zero Independent Per-Tier Training Runs Required)"]
    end

    subgraph Container ["2. Nested Prefix Model & Checkpoint Layout (REQ-5.1, REQ-5.3)"]
        direction TB
        T0["Tier 0: Lite Base Prefix"]
        T1["Tier 1: + Full Delta (Lite + Delta 1 = Full)"]
        T2["Tier 2: + Pro Delta (Full + Delta 2 = Pro)"]
        T3["Tier 3: + Ultra Delta (Pro + Delta 3 = Ultra)"]
        T0 --> T1 --> T2 --> T3
    end

    subgraph Deploy ["3. Flexible Runtime & Distribution Scaling (REQ-5.4..REQ-5.6)"]
        D1["Dynamic Runtime Tier Selection<br/>Calibrated Level Matching & Safe Switch Admission"]
        D2["Host Tier Ceiling Cap & Automatic Offline Bounce Promotion"]
        D3["Signature-Preserving Container Tier Stripping (Zero Retraining / Zero Re-Signing)"]
    end

    Joint --> Container --> Deploy
```

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-5.1`** | **Nested Slimmable Multi-Tier Hierarchy (`Lite` ⊂ `Full` ⊂ `Pro` ⊂ `Ultra`)** | Models and containers must organize capacity into exactly four nested tiers—**`Lite`** (Mix-Ready / Live Grade), **`Full`** (True Studio Quality), **`Pro`** (Mastering / Reference Grade), and **`Ultra`** (Unconstrained Workstation Max)—where each lower tier is an exact contiguous prefix/subset of every higher tier with zero duplicated parameters and zero structural padding zeros. Tier capacities may be adjusted within explicit supported bounds without adding tiers, while retaining their quality and deployment requirements ([`tora_architecture.md` §2.1](../specs/architecture/tora_architecture.md#21-lower-block-triangular-arena-hierarchy), [`../specs/containers/dnam_binary_format.md` §2](../specs/containers/dnam_binary_format.md#2-tier-prefix-hardware-loading)). |
| **`REQ-5.2`** | **Single-Run Joint Multi-Tier Training (Zero Independent Per-Tier Training)** | **Training a model at the highest tier (or any selected maximum target tier `t_max`) must simultaneously train and calibrate all lower tiers (`t ≤ t_max`) within that single training run.** Each tier must **not** need to be independently trained, captured, or distilled; a single training execution produces one unified container in which `Lite`, `Full`, `Pro`, and `Ultra` each independently meet their respective quality standards ([`psychoacoustic_quality_and_tiers.md` §4](../specs/architecture/psychoacoustic_quality_and_tiers.md#4-canonical-4-tier-quality-benchmark-matrix), [`../specs/training/four_stage_pipeline.md` §2.2](../specs/training/four_stage_pipeline.md#22-sweep-2--joint-full-rig-kronecker-polish--export-certification)). |
| **`REQ-5.3`** | **Multi-Tier Incremental Checkpoint Refinement** | A single incremental refinement invocation must update all active tiers (`Lite` through `t_max`) together from one checkpoint, without separate per-tier training jobs. Reuse unchanged stem/tier statistics without re-evaluation; changes to a target, weights, or other statistics dependencies may require replay even when the upstream trajectory is unchanged. The incremental pass must account for each tier's actual trajectory and preserve per-stem state in a nested tier-prefix representation ([`../specs/containers/checkpoint_and_cache_format.md` §2](../specs/containers/checkpoint_and_cache_format.md#2-tier-ordered-column-major-upper-triangular-packing-uplou-for-mathbfg_textaudios)). |
| **`REQ-5.4`** | **Loudness-Matched Dynamic Runtime Tier Switching** | Runtime engines and hosts must be able to request a different tier (`Lite`, `Full`, `Pro`, `Ultra`) from the same loaded container without file reloads. Admitted changes must avoid audible level jumps and target `0.00 dB` calibrated per-tier loudness parity at default controls. Defer or explicitly reject a change when its state or resources cannot be preserved safely; immediate completion is not guaranteed ([tier-transition contract](../specs/architecture/tora_architecture.md#on-demand-tier-transition-contract)). |
| **`REQ-5.5`** | **Host Maximum Tier Ceiling & Automatic Offline Bounce Promotion** | Plugins and embedded hosts must be able to cap the maximum loaded tier to bound memory and compute on resource-constrained hardware. A tier request targets `min(t_requested, t_host_ceiling, t_container_max)` under the applied ceiling; until an admitted change completes, the active tier may remain different from that target. Ceiling changes must not be reported as applied prematurely. Desktop DAWs must support optional automatic promotion to the highest available tier during offline bounce/rendering ([`tora_architecture.md` §3.5](../specs/architecture/tora_architecture.md#35-4times-zoh-polyphase-oversampling-on-offline-bounce)). |
| **`REQ-5.6`** | **Signature-Preserving Container Tier Stripping** | Any multi-tier container, including an encrypted container, must support lossless stripping of higher-tier weights and control projections down to a target lower tier (`Pro`, `Full`, or `Lite`) **without retraining and without invalidating creator or marketplace cryptographic signatures**. Verification must authenticate the retained control descriptors, tier boundaries, and lower-tier payload ([`../specs/containers/dnam_binary_format.md` §2.4](../specs/containers/dnam_binary_format.md#24-tier-local-projection--stage-iii-slice-layout), [`../specs/security/provenance_and_enclave_drm.md` §2.3](../specs/security/provenance_and_enclave_drm.md#23-two-certificate-ed25519-delegation-chain--4-tier-cumulative-arena-digests)). |

> [!NOTE]
> `REQ-5.3`'s single statistics pass is over the checkpoint and new or statistics-invalidated stems; it does not require distinct tiers to share one feature trajectory. Target or weight updates may require bounded replay of the affected stems. A separate validation-only render may be necessary to establish `REQ-3.2`; it does not refresh unchanged statistics and its cost must be accounted for. All tiers remain part of the one joint training run required by `REQ-5.2`.

---

## 7. Goal 6 — Real-Time Commodity Hardware Budget & Execution Constraints

### 7.1 Target Embedded Platform & Per-Tier Instance Budgets (`REQ-6.1`)

The reference commodity embedded target is the **Rockchip RK3582** (as deployed on the **Darkglass Anagram**: $2\times$ Cortex-A76 @ $2.4\text{ GHz}$ + $4\times$ Cortex-A55 @ $1.8\text{ GHz}$ with ARM64 NEON SIMD, allocating $1\times$ Cortex-A76 core = **$20,833\text{ ns/sample}$ budget at $48\text{ kHz}$** for a series signal chain). A conforming runtime must sustain the following simultaneous real-time ($48\text{ kHz}$) instance counts ([`tora_architecture.md` §2](../specs/architecture/tora_architecture.md#2-4-tier-slimmable-simd-hierarchy-lite--full--pro--ultra)):

| DNAM Tier | Target Grade | Minimum Simultaneous `48 kHz` Instances (`1×` Cortex-A76) | Maximum Per-Instance Sample Budget | Max Single-Core CPU Share | Primary Deployment Target |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Lite`** | **Mix-Ready / Live Grade** | **`≥ 10×` instances** | `≤ 2,083 ns/sample` | `≤ 10.0%` | High-density embedded multi-block pedalboard chains & web previews |
| **`Full`** | **True Studio Quality** | **`≥ 6×` instances** | `≤ 3,472 ns/sample` | `≤ 16.7%` | Default studio & stage flagship block |
| **`Pro`** | **Mastering / Reference Grade** | **`≥ 2×` instances** | `≤ 10,416 ns/sample` | `≤ 50.0%` | Mastering / reference solo rig on embedded & desktop hosts |
| **`Ultra`** | **Unconstrained Workstation Max** | **Desktop Real-Time & `4×` Bounce** | Workstation / Offline | Workstation / Offline | Desktop DAWs & `4×` oversampled offline rendering (with optional automatic tier promotion on bounce) |

### 7.2 Real-Time Hardware Budget & Loop Execution Requirements (`REQ-6.1`–`REQ-6.3`)

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-6.1`** | **Simultaneous Real-Time Tier Instance Budgets & Single-Binary Runtime SIMD Dispatch** | On a single reference embedded core (`1×` Cortex-A76 @ `2.4 GHz`, `20,833 ns/sample` at `48 kHz`), a conforming runtime must sustain `≥ 10×` simultaneous `Lite` instances (`≤ 2,083 ns/sample`), `≥ 6×` simultaneous `Full` instances (`≤ 3,472 ns/sample`), and `≥ 2×` simultaneous `Pro` instances (`≤ 10,416 ns/sample`), with `0` extra `48 kHz` MACs/sample when user controls are static. Each target architecture (`ARM64`, `x86-64`, `WebAssembly`) must ship as a single library binary that automatically selects the widest supported hardware SIMD instruction set at startup while maintaining bit-accurate parity across backends. Validate representative supported control counts and active automation as well as static settings ([`tora_architecture.md` §2.2](../specs/architecture/tora_architecture.md#22-hardware--dsp-benchmark-summary)). |
| **`REQ-6.2`** | **Zero Pipeline-Stalling Instructions, `libm` Calls or Unbounded Loops** | Real-time audio (`48 kHz`) and control (`3 kHz`) loops must execute zero pipeline-stalling variable-latency division or square-root instructions—including floating-point division (`FDIV`), floating-point square root (`FSQRT`), and integer division or remainder (`IDIV`, `DIV`, `SDIV`, `UDIV`, or software `128`-bit division helpers)—zero `libm` transcendental or floating-point remainder calls (`fmod`, `fmodf`, `sin`, `cos`, `tan`, `exp`, `log`, `pow`), and zero unbounded data-dependent loops, enforced by automated release-binary disassembly verification in continuous integration ([`../specs/dsp/op_n_commuted_cascade.md`](../specs/dsp/op_n_commuted_cascade.md) and [`../specs/dsp/op_m_modal_kautz.md`](../specs/dsp/op_m_modal_kautz.md)). |
| **`REQ-6.3`** | **Zero Real-Time Allocations, Nondeterministic Calls, RNE & Denormal-Tail Resilience** | Audio callbacks must execute with zero heap allocations, zero redundant intermediate feature-bus or cross-boundary audio buffer copies, zero blocking locks or OS syscalls (including uncached runtime feature queries or mutex acquisitions on the audio path), bounded constant-time per-frame work and admitted state, IEEE-754 round-to-nearest, ties-to-even, and non-trapping floating-point exceptions. Long captured histories may exceed L1 capacity; the instance and timing budgets still apply. Prevent sustained recirculation of nonzero subnormal `f32` values in recursive states using platform-appropriate handling (eliding software bit-masking when hardware flush-to-zero is active at block entry). Gradual underflow of intermediate values is permitted where hardware flushing is unavailable; implementations must still meet applicable real-time budgets, including during decaying tails ([floating-point policy](../specs/architecture/tora_architecture.md#floating-point-environment-at-the-audio-callback)). |

Before loading or changing a model, tier or effective sample rate, admit its
declared memory and any simultaneously live transition state against the
host's memory ceiling. An exceeded ceiling, unrepresentable size or
allocation failure must be reported explicitly without silently truncating
captured delay history or disturbing an already active configuration.

---

## 8. Goal 7 — Unclipped 32-Bit IEEE-754 Float (`f32`) Signal Chain Input & Output

Modern embedded DSP pedalboards (such as the Darkglass Anagram) and desktop DAWs maintain a 32-bit IEEE-754 floating-point (`f32`) signal chain throughout the internal processing graph prior to final hardware DAC quantization:

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-7.1`** | **Validated Unclipped `f32` Operating Range** | Every certified model and supported configuration must have a validated per-input operating envelope of at least `[-100.0, +100.0]` (`+40 dBFS` peak) simultaneously on its audio inputs, and support outputs through at least the same magnitude without safety clipping. With valid controls and certified initial or reachable state, inputs in that envelope must not trigger representational saturation in the signal path. Larger ranges may be advertised only when validated. Never apply an artificial `0 dBFS` clip; intended modeled nonlinearities and ordinary finite-precision rounding remain. |
| **`REQ-7.2`** | **All-Finite-Input Safety & Stability** | Accept every finite `f32` audio input. Outside the certified range, defined representational saturation may alter the signal, but outputs and persistent states must remain finite and saturation must be reported. Recurrent systems must meet their BIBO and applicable Lyapunov guarantees for permitted controls; finite-value saturation alone does not establish Lyapunov stability ([`../specs/training/control_space_audit.md`](../specs/training/control_space_audit.md)). |
| **`REQ-7.3`** | **32-Bit IEEE-754 Float WAV I/O** | Capture and training WAV I/O must support unclipped 32-bit IEEE-754 float (`WAVE_FORMAT_IEEE_FLOAT`, `format_tag = 3`) alongside 16/24/32-bit integer PCM ([`../specs/training/capture_and_alignment.md` §1.2](../specs/training/capture_and_alignment.md#12-2-step-hardware-calibration--clean-32-bit-float-pad-staging)). |
| **`REQ-7.4`** | **Mono (`1 → 1`, `2 → 1`) & True Stereo (`1 → 2`, `2 → 2`) Channel Topologies** | Hardware capture, model training, and real-time inference must natively support all four audio channel I/O routing topologies with `0.00 dB` nonlinear drive and threshold parity between `1 → 2` and centered `2 → 2` inputs, zero unrequested mono-collapsing of stereo inputs, and zero right-channel loss when stereo models are deployed in mono outputs: (1) **Mono-In / Mono-Out (`1 → 1`)** for single-channel instrument/line chains (producing a phase-coherent mono mix without redundant right-channel output computation), (2) **Stereo-In / Mono-Out (`2 → 1`)** for dual-input, stereo-summed, or auxiliary sidechain-keyed mono processors (preserving dual-channel input interactions and sidechain keying without premature input summing), (3) **Mono-In / Stereo-Out (`1 → 2`)** for mono sources expanding into stereo modulation, ping-pong delay, rotary Doppler, dual-mic cabinets, or reverberant spaces, and (4) **True Stereo (`2 → 2`)** for independent stereo left/right (or 4-Cable-Method `Inst`/`Return` dual-path) input through true-stereo nonlinear, tonal, stereo-linked dynamics, and spatial processing to true stereo output ([`tora_architecture.md` §3.2](../specs/architecture/tora_architecture.md#32-stage-0-8-port-matrix-router--voltage-normalized-midside), [`../specs/dsp/router_and_stages_topology.md` §1](../specs/dsp/router_and_stages_topology.md#1-stage-0-8-port-matrix-router-req-74)). |

> [!IMPORTANT]
> A finite maximum-`f32` input passed through a gain of two has no finite `f32` representation. The certified unclipped envelope (`REQ-7.1`) and all-finite-input safety (`REQ-7.2`) are therefore distinct; see the [numerical range and saturation specification](../specs/architecture/tora_architecture.md#finite-precision-range-and-saturation-policy).

### 8.1 Canonical Channel I/O Topologies (`1 → 1`, `2 → 1`, `1 → 2`, `2 → 2`, `REQ-7.4`)

A conforming system must treat channel topology as a first-class property across capture, training, container metadata, and runtime execution:

| I/O Topology | Channel Routing | Representative Hardware & Rig Targets | Capture, Training & Runtime Execution Requirement (`REQ-7.4`) |
| :--- | :---: | :--- | :--- |
| **Mono (`1 → 1`)** | `1 In → 1 Out` | Single-channel overdrive/fuzz pedals, mono compressors, envelope filters, standalone amp heads, single-mic speaker cabinets | Captures and trains 1-channel dry to 1-channel wet stems; at runtime, avoids redundant stereo output computation in mono signal chains while producing a phase-coherent mono output (preserving both channels of any stereo-captured ping-pong delay or dual-mic model rather than dropping the right channel). |
| **Stereo-to-Mono (`2 → 1`)** | `2 In → 1 Out` | Dual-input vintage tube amps (`Normal` + `Bright` jumpered inputs), stereo-in mono-summed stages, sidechain-keyed mono compressors/gates (`Main In` + `Aux Key In` → `Mono Out`) | Preserves both input channels (`[x_L, x_R]` or `[x_main, x_aux]` sidechain key) during capture, training, and internal nonlinear, tonal, and dynamics processing without prematurely collapsing the input to mono, then produces a single phase-coherent mono output without redundant second-channel output computation. |
| **Mono-to-Stereo (`1 → 2`)** | `1 In → 2 Out` | Mono guitar/synth into stereo chorus, through-zero flanger, rotary speaker, ping-pong delay, dual-mic speaker cabinet (`SM57` + `R121`), stereo hall/shimmer reverb, or mono-in stereo-out full rig | Captures 1-channel dry against 2-channel (`[y_L, y_R]`) wet stems; expands mono input across stereo tonal and spatial stages to synthesize phase-accurate stereo imaging with continuous stereo width control down to mono compatibility. |
| **True Stereo (`2 → 2`)** | `2 In → 2 Out` | True-stereo rack/pedal processors (stereo chorus/phaser/delay/reverb fed by a stereo source), dual-amp stereo rigs, stereo mastering bus gear, and 4-Cable-Method (`4CM`) `[Inst In, Loop Return] → [Loop Send, Main Out]` rigs | Captures and trains full 2-channel dry (`[x_L, x_R]` or `[x_inst, x_return]`) to 2-channel wet (`[y_L, y_R]`) stems without collapsing input channels to mono; preserves independent Left/Right nonlinear, tonal, and spatial separation (or isolated 4CM loop routing), fully linked dynamics imaging (`0.00 deg` to two decimal places in the [normal finite domain](../specs/dsp/router_and_stages_topology.md#24-programmable-stereo-linked--aux-key-sidechain-envelope-detector)), and `0.00 dB` drive-threshold parity with `1 → 2` mode when fed a centered mono signal. Direction is undefined for silence or zero gain; underflow follows the finite-safety and denormal policies rather than an angle guarantee. |

---

## 9. Goal 8 — Model Provenance, Hardware DRM, Forensic Watermarking & Marketplace Security

A conforming ecosystem must protect creator intellectual property, hardware OEM partnerships, and marketplace integrity across open community sharing, desktop/mobile DAWs, browser runtimes, and offline hardware pedalboards—**without requiring internet connectivity during live performance and without adding any overhead (`0 flops`) to real-time audio processing**:

```mermaid
flowchart LR
    subgraph Prov ["1. Provenance & Creator Policy (REQ-8.1, REQ-8.2)"]
        P1["Cryptographic Creator & Marketplace Signatures<br/>(Immutable Model/Tier Digests vs. Mutable Presets/Tags)"]
        P2["Creator-Signed Distribution Policy<br/>(Allowed Protection Profiles & Manufacturer/Model Exclusivity)"]
    end

    subgraph DRM ["2. Graduated DRM & Hardware Attestation (REQ-8.3, REQ-8.4)"]
        D1["3 Profiles: Open, DeviceAllowlisted & EnclaveSealed"]
        D2["Flexible OEM Onboarding: Self-Registered, OTA Companion Enrollment, or Factory HSM"]
        D3["100% Offline Stage Operation & 0 Real-Time Audio Flops"]
    end

    subgraph Trace ["3. Forensic Watermarking & Anti-Cloning (REQ-8.5, REQ-8.6)"]
        W1["Audible-Zero Buyer Watermark Across All Tiers<br/>(Survives Header Stripping, Collusion & Lite Truncation)"]
        W2["Multi-Stage Anti-Cloning Ingestion Gate<br/>(Signatures + Prefix-Composable Structural Fingerprint + Active Audio Probe)"]
        W3["Evidence-Based Attribution & Revocation from Future Licensing"]
    end

    subgraph Market ["4. Storefront & Container Safety (REQ-8.7, REQ-8.8)"]
        M1["Interactive Browser Auditioning, Measured Preview Defenses & Multi-Creator Royalties"]
        M2["Declarative, Code-Execution-Free Binary Containers (Zero Arbitrary Code on Load)"]
    end

    Prov --> DRM --> Trace --> Market
```

| Requirement ID | Requirement | Specification |
| :--- | :--- | :--- |
| **`REQ-8.1`** | **Cryptographic Provenance & Preset-Safe Metadata Partitioning** | Containers must carry verifiable cryptographic signatures binding the creator's identity, training dataset provenance, model topology, control descriptors and projection data, tier boundaries, and all applicable creator and buyer cumulative tier digests (`REQ-5.6`), plus a marketplace licensing counter-signature when commercially distributed. The container format must partition **immutable signed fields** from **mutable user and storefront fields** (user presets and platform catalog tags) so users and marketplaces can edit and grow that metadata in the same file without invalidating signatures, subject to explicit host resource limits ([`../specs/security/provenance_and_enclave_drm.md` §2.1](../specs/security/provenance_and_enclave_drm.md#21-preset-safe-canonical-header-digest-rfc-8785-jcs)). |
| **`REQ-8.2`** | **Creator-Signed Distribution Policy & OEM Hardware Exclusivity** | Creators must be able to cryptographically restrict which protection profiles (`Open`, `DeviceAllowlisted`, `EnclaveSealed`) and which hardware OEM manufacturers—either manufacturer-wide or restricted to specific hardware device models—may license, seal, or execute a model. Neither a buyer nor a marketplace may strip or widen a creator's distribution policy without invalidating the creator's signature ([`../specs/security/provenance_and_enclave_drm.md` §2.2](../specs/security/provenance_and_enclave_drm.md#22-creator-signed-distributionpolicy--protection-profile-exclusivity)). |
| **`REQ-8.3`** | **Graduated Protection Profiles & Flexible OEM Hardware Integration** | The system must support three graduated container protection profiles: (1) **`Open`** (unencrypted weights for community and unrestricted cross-platform use), (2) **`DeviceAllowlisted`** (plaintext header for instant DAW browser indexing + encrypted weights bound to a user's allowlisted device keys, requiring zero factory attestation infrastructure), and (3) **`EnclaveSealed`** (opaque hardware-sealed container decrypted only inside secure hardware memory). OEM hardware must support self-registered keys, post-factory OTA companion-app certificate enrollment (for already-shipped units), and factory-line HSM provisioning ([`../specs/security/provenance_and_enclave_drm.md` §3–§4](../specs/security/provenance_and_enclave_drm.md#3-pillar-2-hardware-enclave--oem-drm-oemdevicecertificate-attestation--dnam-seal-bdnsl-wrapper)). |
| **`REQ-8.4`** | **Offline Stage Reliability & Zero Real-Time Audio Overhead (`0` DSP Flops)** | After a protected model is authorized and issued to a device, it must remain loadable and usable on that device offline indefinitely without periodic renewal, provided its authorization and required local keys remain intact. Signature verification, local policy enforcement, key unwrapping, and weight decryption must complete at container load time with **zero runtime internet dependency** and **zero operations (`0 flops`)** added to the real-time `48 kHz` audio or `3 kHz` control loops ([offline authorization](../specs/security/provenance_and_enclave_drm.md#offline-authorization-and-future-revocation)). |
| **`REQ-8.5`** | **Perceptually Transparent, Collusion- & Truncation-Resistant Buyer Watermarking** | Every licensed commercial container must embed a deterministic, buyer-specific forensic watermark directly inside the model weights across all tiers (`Lite` through `Ultra`) that introduces zero audible degradation (`0.00 dB`), cannot be removed by stripping headers or applying weight-space equivalence/rescaling transforms, and survives weight quantization (`f32 → fp16`), multi-buyer collusion averaging, and tier stripping down to `Lite`-only (`REQ-5.6`) ([`../specs/security/watermarking_and_marketplace.md` §1](../specs/security/watermarking_and_marketplace.md#1-pillar-3-forensic-traitor-tracing--02-ms-dual-mechanism-buyer-watermarking)). |
| **`REQ-8.6`** | **Multi-Stage Anti-Cloning Ingestion Gate & Evidence-Based Device Revocation** | Marketplace uploads must be screened by a fast multi-stage anti-cloning pipeline that checks (A) cryptographic signature and dataset provenance, (B) tier-prefix-composable structural fingerprints, and (C) active synthetic-probe responses. Evaluate detection and attribution against defined leak, collusion, stripping and recapture workloads. A confirmed leak may trigger automatic revocation of a device from **future licensing only when independent evidence identifies that device as the source**. A license watermark or device-specific copy alone is not such evidence; retain inconclusive cases for review without revoking an unidentified device ([`../specs/security/watermarking_and_marketplace.md` §2](../specs/security/watermarking_and_marketplace.md#2-pillar-4-3-stage-anti-cloning-ingestion-gate--50-ms-clone--re-capture-prevention)). |
| **`REQ-8.7`** | **Interactive Auditioning, Measured Preview Protection, Stateless Fulfillment & Royalty Splits** | Commercial storefronts must enable live, zero-latency browser auditioning across curated presets via dedicated, on-demand-generated (`< 1 ms`) preview containers that: (1) physically omit all higher-tier (`Full`, `Pro`, `Ultra`) weights, (2) anchor each preview around a target preset, with a bounded responsive control for eligible single-control models, (3) target perceptually transparent stage-fused phase dispersion, (4) target defenses against defined black-box capture stimuli while preserving natural playing, and (5) target session-bound white-box and acoustic watermarks. Measure protection, false alarms, audition fidelity and attribution against declared workloads; do not promise prevention of all reconstruction or replay-based copying. Storefronts must also store only a single canonical master per model with stateless on-demand per-buyer watermarking/sealing at fulfillment time (zero storage multiplication) and automated multi-creator revenue splits ([`../specs/security/watermarking_and_marketplace.md` §3–§4](../specs/security/watermarking_and_marketplace.md#3-commercial-storefront-platform-auditioning-royalty-splits--storefront-metadata)). |
| **`REQ-8.8`** | **Declarative, Code-Execution-Free Binary Containers** | All runtime model containers, preview containers, training checkpoints, calibration caches, and hardware-sealed envelopes must use strictly validated declarative metadata and non-executable binary layouts that never execute arbitrary code during deserialization (prohibiting executable serialization formats such as Python `pickle`) and fail fast on malformed headers, unaligned offsets, or checksum mismatches ([`../specs/containers/checkpoint_and_cache_format.md` §1](../specs/containers/checkpoint_and_cache_format.md#1-zero-weight-duplication--zero-pickle-vulnerability)). |

### 9.1 Interactive Storefront Auditioning & Anti-Harvesting Security Requirements (`REQ-8.7`)

Prospective buyers must be able to play their own instruments and feel live parameter responsiveness in the browser while protecting commercial models against weight extraction and automated black-box capture:

| Protection Layer | Implementation-Agnostic Requirement (`REQ-8.7`) | Threat Mitigated |
| :--- | :--- | :--- |
| **1. Dedicated Preview Container Type & High-Tier Omission** | Interactive previews must use a distinct container identifier that is rejected as a full production model by DAWs/hardware hosts and physically omits all higher-tier (`Full`, `Pro`, `Ultra`) parameter arenas, transmitting only the base `Lite` tier. | Prevents direct extraction of studio/mastering tier weights from browser memory or network caches and prevents confusion between preview and production files. |
| **2. Bounded Preview Controls & Rank Deficiency** | Anchor each preview at a preset and restrict its controllable behavior: single-control models expose a bounded responsive window or safe switch pair, whereas multi-control models with a nonzero continuous-control space enforce strict aggregate control-rank deficiency across published previews. | Preserves single-control auditioning while preventing multi-control previews from jointly spanning the full continuous control manifold. |
| **3. Sub-Millisecond Stateless On-Demand Generation (`< 1 ms`)** | Preview containers must be synthesized on demand directly from the master container's base-tier prefix in `< 1 ms` of CPU time without executing audio rendering or neural network forward passes, requiring zero pre-generated per-session file storage. | Enables instant per-click preset switching and unique per-visitor session personalization at the CDN edge with zero storage multiplication. |
| **4. Stage-Fused Static Phase Dispersion (Rebuilt-Engine Resistance)** | Preview generation applies a perceptually transparent, flat-magnitude all-pass phase and state-selectivity perturbation fused into intrinsic stage and output projection weights rather than a removable external filter or runtime flag. | Prevents trivial recovery of production weights by stripping an outer filter or recompiling the open-source runtime. |
| **5. Dynamic Synthetic-Sweep & Impulse Input Guard** | The control loop monitors input excitation statistics with zero hardware division or square-root calls and smoothly engages non-stationary phase/spectral dispersion when synthetic probe stimuli are detected, while preserving natural musical playing. | Disrupts automated swept-sine, impulse, and white-noise black-box harvesting stimuli during browser auditioning. |
| **6. Black-Box Acoustic Watermarking & Ephemeral Session Binding** | Preview generation embeds keyed, psychoacoustically masked session-specific transfer-function watermarks alongside session expiration and origin bindings. | Enables forensic attribution of unauthorized browser captures and binds previews to ephemeral storefront sessions. |

> [!NOTE]
> **Control-Rank Containment Rules:**
> - **Single adjustable control ($P = 1$):** Expose a bounded continuous window or two observed safe switch options around the preset anchor; strict rank deficiency does not apply to a 1-D or discrete space.
> - **No adjustable controls ($P = 0$):** Zero-dimensional control space; no control surface is exposed.
> - **Multiple adjustable controls ($P \ge 2$):** Across all published preset previews for a model, the combined continuous control directions must remain strictly rank-deficient ($r < D_{\text{cont}}$) inside a shared model-scoped subspace ([`../specs/security/watermarking_and_marketplace.md` §3](../specs/security/watermarking_and_marketplace.md#3-commercial-storefront-platform-auditioning-royalty-splits--storefront-metadata)). Discrete switches remain locked to their preset options when $D_{\text{cont}} \le 1$.
