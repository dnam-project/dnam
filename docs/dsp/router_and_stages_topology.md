# Stage 0 Router, Canonical Stages Topology & 256-D Mid/Side Head

**Document Version:** `1.0.0`  
**Status:** Canonical DSP Specification (SSOT)  
**Parent Architecture:** [`docs/overview/tora_architecture.md`](../overview/tora_architecture.md)  
**Target Engine:** [`packages/engine/src/engine/`](../../packages/engine/README.md)  
**Target Trainer:** [`packages/trainer/src/dnam_trainer/training/`](../../packages/trainer/README.md)  
**Canonical Compliance:** [`CL-09` (DC Blocker)](../training/clamp_and_audit_registry.md), [`CL-10` (DAG Isolation & Quiet Gain)](../training/clamp_and_audit_registry.md), [`CL-11` (Loop Saturator)](../training/clamp_and_audit_registry.md), [`REQ-7.4` (Channel Topologies)](../overview/system_goals_and_requirements.md)

This specification defines the **Stage 0 8-Port Matrix Router**, the **Four Canonical Physical Stages (Stages I–IV)**, the **Unified 256-D Feature Bus & Canonical Mid/Side Kronecker Output Head**, and **Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability (`44.1–96 kHz` & `CL-20` `4×` Bounce)** across **Dynamic Neural Audio Modeler (DNAM)**. For R&D trade-offs and counterexamples, see [`../rnd/tora_phase1_decisions.md`](../rnd/tora_phase1_decisions.md).

---

## 0. Detailed Signal Flow & Stage Boundaries

```mermaid
flowchart TD
    subgraph ControlBusGroup ["3 kHz Unified Control Plane"]
        CtrlSrc["Shared Descriptors & Packed Tier-Local CP1 Projections"] --> OpN["Op-N[Ctrl]: Exact-Integer Sufficient-Statistic Engine"]
        SendDetector["Instrument / Independent Aux Detector"] --> SendMod["Send-Safe Dynamic Vector m_send"]
        DownDetector["Downstream Instrument / Return / Aux Detector"] --> DownMod["Downstream Dynamic Vector m_down"]
        OpN & SendMod --> SendBus["Send Dispatch Rows 0..4<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
        OpN & DownMod --> DownBus["Downstream Dispatch Rows 5..15<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
    end

    In0["Port In 0: Inst In L"] & In1["Port In 1: Inst In R"] --> PreRouter["Stage 0: Instrument & Send Routing<br/>• Mechanical Normaling<br/>• Voltage-Normalized Mid/Side"]
    In2["Port In 2: FX Return L"] & In3["Port In 3: FX Return R / Aux Key"] --> DownRouter["Stage 0: Return & Downstream Routing<br/>• Parallel-Path Sub-Sample Alignment<br/>• Direct Return-to-Send Block Zero (CL-10)"]
    PreRouter -.-> SendDetector
    PreRouter & DownRouter -.-> DownDetector

    PreRouter --> S1["Stage I: Pre-FX / Synth / Mod<br/>(Op-T / Op-M / Op-N)"]
    S1 --> Out0["Port Out 0: Send L<br/>(External Amp 1)"]
    S1 -->|"Dual-Amp Modes"| Out1["Port Out 1: Send R / WDW Dry"]
    DryTap["WDW Dry Selector<br/>PreSpatialCore / DirectDry"] -->|"WDW"| Out1
    S1 --> DownRouter

    DownRouter -->|"200 Hz TPT LR4 Low: Clean Bypass"| S3
    DownRouter -->|"200 Hz TPT LR4 High: Drive Path"| S2["Stage II: 3-Stage Amp Cascade<br/>(V1A Preamp -> V1B -> Power Amp)"]
    S2 --> S3["Stage III: Cabinet & Modals<br/>(64-Tap SIMD FIR + Shared Kautz Bank)"]
    S3 --> S4["Stage IV: Unitary Spatial Tank<br/>(In-Buffer AP + Barberpole FDN)"]

    DownBus --> DownRouter & S2 & S3 & S4
    SendBus --> PreRouter & S1

    S1 & S2 & S3 & S4 --> Phi["Unified 256-D Feature Bus Phi[n]"]
    Phi --> Head["256-D SVD Kronecker Head<br/>(Canonical Mid/Side Slices 0 & 1)"]
    DownBus --> Head
    Head --> Out2["Port Out 2: Main L / WDW Wet L"]
    Head --> Out3["Port Out 3: Main R / WDW Wet R"]
```

---

## 1. Stage 0: 8-Port Matrix Router (`CL-10`, `REQ-7.4`)

| Port Identifier | Signal Variable | Physical & Logical Routing Role |
| :--- | :--- | :--- |
| **`Port In 0 (Inst In L)`** | `x_{inst,L}[n]` | Primary mono or Left instrument input signal. |
| **`Port In 1 (Inst In R)`** | `x_{inst,R}[n]` | Secondary Right instrument input signal; mechanically normaled to `Port In 0` when unplugged. |
| **`Port In 2 (FX Return L)`** | `x_{ret,L}[n]` | External amplifier effects loop return for Amp 1 / Left channel. |
| **`Port In 3 (FX Return R / Aux Key)`** | `x_{ret,R}[n]` or `x_{aux}[n]` | External amplifier effects loop return for Amp 2 / Right channel OR dedicated auxiliary sidechain detector key; mechanically normaled to `Port In 2` when unplugged in stereo return mode. |
| **`Port Out 0 (Send L)`** | `y_{send,L}[n]` | Pre-loop preamp send driving Amp 1 / Left external amplifier front-end (or optional mono send in WDW). |
| **`Port Out 1 (Send R / WDW Dry)`** | `y_{send,R}[n]` or `y_{dry}[n]` | Right external send in dual-amp modes; dedicated center dry output in WDW (`PreSpatialCore` or `DirectDry`). |
| **`Port Out 2 (Main Out L)`** | `y_{out,L}[n]` or `y_{wet,L}[n]` | Main output Left; wet-only Left in WDW. |
| **`Port Out 3 (Main Out R)`** | `y_{out,R}[n]` or `y_{wet,R}[n]` | Main output Right; wet-only Right in WDW. |

#### WDW Dry-Tap and Wet-Output Contract

| `WdwDryTap` Mode | Center Dry on `Port Out 1` | Wet-Only L/R on `Ports Out 2/3` | Default Applicability |
| :--- | :--- | :--- | :--- |
| **`PreSpatialCore`** | Mid contribution `m_c` of Stages I–III core (including allocated head terms) | Stage IV spatial contribution `m_w ± α s_w` | Default for WDW-capable full-rig models |
| **`DirectDry`** | Voltage-normalized Mid `0.5(x_L + x_R)` of instrument input (or active aligned return) | Effect-only model response relative to the selected direct source | Default for standalone wet-effect or external-center-amp rigs |

For `PreSpatialCore`, partition the head's Stages I–III core $(m_c, s_c)$ from its Stage IV spatial contribution $(m_w, s_w)$ with width $\alpha$:
$$y_L = m_c + m_w + \alpha(s_c + s_w), \qquad y_R = m_c + m_w - \alpha(s_c + s_w)$$
$$y_{\text{dry}} = m_c, \qquad y_{\text{wet},L} = m_w + \alpha s_w, \qquad y_{\text{wet},R} = m_w - \alpha s_w$$

#### $4 \times 4$ Scattering Matrix $\mathbf{S}_{\mathrm{router}}$ & Return-to-Send Isolation (`CL-10`)

```text
                  [ S_00  S_01 |   0     0   ]   [ S_{inst->send}    0_{2x2}    ]
S_router =        [ S_10  S_11 |   0     0   ] = [                              ]
                  [------------+-------------]   [                              ]
                  [ S_20  S_21 | S_22  S_23  ]   [ S_{inst->down}  S_{ret->down}]
                  [ S_30  S_31 | S_32  S_33  ]
```

$$\mathbf{y}_{\mathrm{send}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{send}} \mathbf{x}_{\mathrm{inst}}[n], \qquad \mathbf{u}_{\mathrm{down}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{down}} \mathbf{x}_{\mathrm{inst}}[n] + \mathbf{S}_{\mathrm{ret} \to \mathrm{down}} \mathbf{x}_{\mathrm{ret, aligned}}[n]$$

1. **Direct-Path Isolation (`CL-10`):** Enforcing $\mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \equiv \mathbf{0}_{2 \times 2}$ guarantees open-loop return ratio $\mathbf{T}_{\mathrm{loop}}(z) = \mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \mathbf{H}_{\mathrm{amp}}(z) \equiv \mathbf{0}_{2 \times 2}$ and $\det(\mathbf{I} - \mathbf{T}_{\mathrm{loop}}(z)) \equiv 1$.
2. **Send-Safe Control & Detector Partition:** Stage I and router send gains (`θ_0..θ_4`) consume only the send-safe dynamic vector $\mathbf{m}_{\text{send}}$ (derived from instrument inputs and any independent auxiliary key), never return-derived samples or detector histories; downstream stages (`θ_5..θ_15`) consume $\mathbf{m}_{\text{down}}$ ([`parametric_control_bus.md` §3.1](parametric_control_bus.md#31-stage-parameter-dispatch-matrix-mathbfw_textctrl)).

#### Parallel-Path Sub-Sample Alignment

| `AlignmentMode` | Interpolation Kernel | Host-Rate Centering Offset (`G`) | Selection & Offline Bounce Rule |
| :--- | :--- | :---: | :--- |
| **`FullBand64Tap`** | 64-tap Kaiser-derived windowed sinc (`β = 8.6`) | `31` host samples (`0.65 ms` @ `48 kHz`) | Default for full-band 4CM/WDW, training, certification, and `4×` offline bounce |
| **`LowLatency16Tap`** | 16-tap Kaiser-derived windowed sinc (`β = 6.5`) | `7` host samples (`0.15 ms` @ `48 kHz`) | Selectable low-latency live monitoring mode |
| **`MinLatency4Tap`** | Centered 4-tap Catmull-Rom cubic Hermite | `1` host sample (`0.021 ms` @ `48 kHz`) | Low-band alignment or minimum-latency live monitoring mode |

For $K \in \lbrace 16, 64 \rbrace$ with $G = (K - 2)/2$, $R = K/2$, and $s_k = k - G - \mu$, prepare the normalized zero-ended Kaiser-derived sinc weights on the cold path:
$$w_{\beta}(s) = \frac{I_0\big(\beta\sqrt{1 - (s/R)^2}\big) - 1}{I_0(\beta) - 1} \quad (|s| \le R), \qquad h_k(\mu) = \frac{\mathrm{sinc}(s_k) w_{\beta}(s_k)}{\sum_{j=0}^{K-1}\mathrm{sinc}(s_j) w_{\beta}(s_j)}$$
For oversampling factor $M \in \lbrace 1, 4 \rbrace$, measured effective-rate arrivals $a_i$, $A = \max_i a_i$, and optional common host-sample pad $p \ge 0$, set $\Delta_i = A - a_i + M p$, $J_i = \lfloor \Delta_i \rfloor$, $\mu_i = \Delta_i - J_i$, and $B = (M - 1)G$:
$$y_i[n] = \sum_{k=0}^{K-1} h_k(\mu_i) x_i[n - B - J_i - k], \qquad d_{\mathrm{eff},i} = M G + \Delta_i$$
At `4×` bounce, `FullBand64Tap` applies $B = 93$ plus its `31`-effective-sample center to preserve a `31`-host-sample alignment offset (`43` host samples combined nominal PDC with the `12`-host-sample oversampler; minimum `157` effective-rate samples of ring history per channel). `MinLatency4Tap` ($G = 1$) uses the 4-tap Catmull-Rom basis ($\sum_{k=-1}^2 c_k(\mu) \equiv 1.0$, `0.00 dB` exact DC gain):
$$c_{-1}(\mu) = -0.5\mu(1 - \mu)^2, \quad c_0(\mu) = 1 - 0.5\mu^2(5 - 3\mu), \quad c_1(\mu) = 0.5\mu(1 + 4\mu - 3\mu^2), \quad c_2(\mu) = 0.5\mu^2(\mu - 1)$$

#### Stage 0 LR4 Clean/Drive Crossover

When parallel sub-bass bypass is active, split the selected downstream source using a 4th-order Linkwitz–Riley (`LR4`, default $f_c = 200\text{ Hz}$) crossover realized as two cascaded 2nd-order Butterworth **trapezoidal state-variable filter (TPT SVF)** sections ($Q = 1/\sqrt{2}$). Prepare cold-path coefficients in `f64` and store as `f32`:
$$g = \tan\left(\frac{\pi f_c}{f_{s,\text{eff}}}\right), \qquad k = \sqrt{2}, \qquad a_1 = \frac{1}{1 + g(g + k)}, \qquad a_2 = g a_1, \qquad a_3 = g a_2$$
Each SVF section updates integrator states $(s_1, s_2)$ without real-time division:
$$v_3 = x - s_2, \qquad v_1 = a_1 s_1 + a_2 v_3, \qquad v_2 = s_2 + a_2 s_1 + a_3 v_3$$
$$\ell = v_2, \qquad h = x - k v_1 - v_2, \qquad s_1' = 2 v_1 - s_1, \qquad s_2' = 2 v_2 - s_2$$
Route $L_4 = L_2^2$ to the Stage III clean bypass and $H_4 = H_2^2$ to Stage II (`|L₄(e^{jω}) + H₄(e^{jω})| ≡ 1`, `-6.02 dB` and equal phase at $f_c$).

---

## 2. Mid/Side Encoding, Output Head & Stereo-Linked Dynamics

### 2.1 Drive-Invariant Voltage-Normalized Mid/Side Input Transformation
To preserve `0.00 dB` drive-threshold parity between mono (`1 → 2`) and centered stereo (`2 → 2`, $x_L = x_R = x$) inputs (`REQ-7.4`), Stage 0 uses **Voltage-Normalized Mid/Side Encoding** rather than orthonormal $1/\sqrt{2}$ power scaling:
$$x_{\mathrm{mid}}[n] = \frac{1}{2}\big(x_L[n] + x_R[n]\big), \qquad x_{\mathrm{side}}[n] = \frac{1}{2}\big(x_L[n] - x_R[n]\big), \qquad x_L[n] = x_{\mathrm{mid}}[n] + x_{\mathrm{side}}[n], \qquad x_R[n] = x_{\mathrm{mid}}[n] - x_{\mathrm{side}}[n]$$

### 2.2 Canonical Mid/Side Output Head Factorization
At each `3 kHz` control tick, the $D_h \in \lbrace 4, 8 \rbrace$ SVD head modes collapse into active Mid ($a = \mathrm{mid}$, Slice 0) and Side ($a = \mathrm{side}$, Slice 1) weight vectors over the unified feature bus $\boldsymbol{\Phi}[n] \in \mathbb{R}^D$ ($D \in \lbrace 64, 128, 192, 256 \rbrace$):
$$\mathbf{w}_a[n] = \mathbf{w}_{a,0} + \sum_{d=1}^{D_h} z_{\mathrm{head},d}[n] \mathbf{w}_{a,d}, \qquad b_a[n] = b_{a,0} + \sum_{d=1}^{D_h} z_{\mathrm{head},d}[n] b_{a,d}$$
$$\mathbf{w}_{\mathrm{mid}} \triangleq \frac{1}{2}\big(\mathbf{w}_L + \mathbf{w}_R\big), \qquad \mathbf{w}_{\mathrm{side}} \triangleq \frac{1}{2}\big(\mathbf{w}_L - \mathbf{w}_R\big), \qquad b_{\mathrm{mid}} \triangleq \frac{1}{2}\big(b_L + b_R\big), \qquad b_{\mathrm{side}} \triangleq \frac{1}{2}\big(b_L - b_R\big)$$
$$y_{\mathrm{mid}}[n] = \mathbf{w}_{\mathrm{mid}}^\top \boldsymbol{\Phi}[n] + b_{\mathrm{mid}}, \qquad y_{\mathrm{side}}'[n] = \alpha_{\mathrm{width}}[n] \big(\mathbf{w}_{\mathrm{side}}^\top \boldsymbol{\Phi}[n] + b_{\mathrm{side}}\big)$$
$$y_L[n] = y_{\mathrm{mid}}[n] + y_{\mathrm{side}}'[n], \qquad y_R[n] = y_{\mathrm{mid}}[n] - y_{\mathrm{side}}'[n]$$

1. **Mono (`1 → 1`, `2 → 1`) Execution:** Evaluates only Slice 0 ($y_{\mathrm{mid}}[n]$), preserving dual-mic (`SM57 + R121`) and ping-pong delay sums without dropping the right channel while skipping Slice 1 dot-products.
2. **Mid Reconstruction Bound:** For matching state and features, $|m - \hat{m}| \le 2 \mathrm{ulp}(M) + 2 F_{\min}$ where $M = \max(|y_{\mathrm{mid}}|, |y_{\mathrm{side}}'|)$.
3. **Stereo Width Folding ($\alpha_{\mathrm{width}} \in [0.0, 2.0]$):** When static over a control block, fold $\alpha_{\mathrm{width}}$ into $\tilde{\mathbf{w}}_{\mathrm{side}} = \alpha_{\mathrm{width}}\mathbf{w}_{\mathrm{side}}$ and $\tilde{b}_{\mathrm{side}} = \alpha_{\mathrm{width}}b_{\mathrm{side}}$ at the `3 kHz` boundary (`0` extra audio-rate multiplies); when automated, micro-ramp $\alpha_{\mathrm{width}}[n]$ per sample.

### 2.3 Topology-Adaptive Even/Odd SIMD Lane Partitioning

| Parameter / Feature | Mono (`1 → 1`) | Stereo-to-Mono (`2 → 1`) | Mono-to-Stereo (`1 → 2`) | True Stereo (`2 → 2`) |
| :--- | :---: | :---: | :---: | :---: |
| **Engine Entrypoint** | `process_mono` | `process_stereo_to_mono` | `process_mono_to_stereo` | `process_stereo` |
| **Stage I & II Execution** | Normaled L/R paths & cross-coupling retained | Full stereo L/R paths & coupling | Normaled L/R paths & cross-coupling retained | Full stereo L/R paths & coupling |
| **Stage III Early FIR & Modals** | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance |
| **Output Head Slices** | Slice 0 (`w_mid`) only | Slice 0 (`w_mid`) only | Slices 0 & 1 (`w_mid, w_side'`) | Slices 0 & 1 (`w_mid, w_side'`) |

### 2.4 Programmable Stereo-Linked & Aux-Key Sidechain Envelope Detector
The sidechain detector blends peak-linked stereo detection ($\alpha_{\mathrm{link}} \in [0.0, 1.0]$) and external auxiliary keying ($\beta_{\mathrm{key}} \in [0.0, 1.0]$) with zero `-6.02 dB` hard-panned peak loss:
$$e_{\mathrm{link}}[n] = \max\big(|x_L[n]|, |x_R[n]|\big), \qquad e_{\mathrm{key}}[n] = |x_{\mathrm{aux}}[n]|$$
$$e_{\mathrm{det}, c}[n] = (1 - \beta_{\mathrm{key}})\Big((1 - \alpha_{\mathrm{link}})|x_c[n]| + \alpha_{\mathrm{link}} e_{\mathrm{link}}[n]\Big) + \beta_{\mathrm{key}} e_{\mathrm{key}}[n], \qquad c \in \lbrace L, R \rbrace$$
At full link ($\alpha_{\mathrm{link}} = 1.0$) with a shared positive normal gain $G[n] > 0$ and normal finite inputs/products ($u = 2^{-24}$), the stereo image angle error satisfies ([`test_stereo_link_spec_references.py`](../../tests/test_stereo_link_spec_references.py)):
$$\Vert\widehat{\mathbf{y}} - G\mathbf{x}\Vert_2 \le u\Vert G\mathbf{x}\Vert_2, \qquad |\sin(\Delta\theta)| \le \frac{u}{1 - u}, \qquad |\Delta\theta| < 0.00000342^\circ$$
Audio-rate ballistic envelope smoothing prepares $\alpha_{\mathrm{att}} = 1 - \exp(-1/(f_s\tau_{\mathrm{att}}))$ and $\alpha_{\mathrm{rel}} = 1 - \exp(-1/(f_s\tau_{\mathrm{rel}}))$ on the cold path and updates $v_{\mathrm{env},c}[n] = v_{\mathrm{env},c}[n-1] + \mathrm{select}(\Delta e_c[n] > 0, \alpha_{\mathrm{att}}, \alpha_{\mathrm{rel}})\Delta e_c[n]$ branchlessly.

### 2.5 4-Band Asymmetric Ballistic Envelope Bank & Aux Key Detector

| Band ID | Frequency Range | Ballistic Attack (`τ_att`) | Ballistic Release (`τ_rel`) | Physical Circuit Role |
| :--- | :--- | :---: | :---: | :--- |
| **Band 0 (Low)** | `< 250 Hz` | `10.0 ms` | `150.0 ms` | Power supply B+ rail sag, cone excursion pumping, transformer saturation. |
| **Band 1 (Mid)** | `250–2000 Hz` | `2.0 ms` | `40.0 ms` | Dynamic auto-wah envelope, speech formant modulation, optical compression. |
| **Band 2 (High)** | `> 2000 Hz` | `0.2 ms` | `15.0 ms` | Pick transient attack detection, noise gate onset, sizzle ducking. |
| **Band 3 (Aux Key)** | Wideband / Sidechain | `0.5 ms` | `50.0 ms` | Driven from `Port In 3 (Aux Key)` or stereo-linked instrument sum. |

---

## 3. Stage I: Pre-FX / Synth / Pitch / Mod (Pre-Bus)

| Stage I Kernel | State Allocation | Operational Modes & Physical Roles |
| :--- | :--- | :--- |
| **`Op-T[Pre]` (Transport & Pitch)** | Model-declared 64B-aligned delay histories with 4-tap Catmull-Rom (`CL-06`, [`op_t_transport.md` §1](op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)) | • **`M = 1`:** Fractional micro-delay and through-zero flanging.<br/>• **`M = 2`:** Centered dual-tap Hermite-Hann pitch shifter (`±24 st`, Family 3).<br/>• **`M = 3`:** Endless barberpole phaser/flanger sweep (Family 6). |
| **`Op-M[Pre]` (Modal, Hilbert & Bistable)** | `4` complex pole channels/tier + `H1` analytic pairs + `T1` DC blocker/tracker states ([`op_m_modal_kautz.md` §2–§3](op_m_modal_kautz.md#2-family-2-hilbert-ssb-lattice--carrier-modulation)) | • **`M-1`:** 6-section `H1` all-pass pair + carrier rotor for SSB/ring mod (Family 2).<br/>• **`M-2`:** `T1` sub-harmonic tracker routed to `Lite` slots `4..11` (Family 1).<br/>• **`M-3`:** High-`Q` auto-wah and resonant phaser sweeps (`CL-03`–`CL-05`). |
| **`Op-N[Pre]` (Waveshaping, Dynamics & ZOH)** | `C_tier` ballistic envelope & fixed-hertz ZOH quantizer states ([`op_n_commuted_cascade.md` §2.3](op_n_commuted_cascade.md#23-family-5-precomputed-reciprocal-zoh-sample-rate-reducer--bitcrusher)) | • Hysteretic gate / auto-swell / compressor VCA (Family 4).<br/>• Fixed-hertz ZOH reducer + nearest-step bitcrusher (`0 FDIV`, Family 5).<br/>• Pre-drive boost and soft-clipping `PolyADAA`. |

---

## 4. Stage II: 3-Stage LTI-Commuted Nonlinear Cascade
Stage II models the 3-stage amplifier cascade (`V1A Preamp + Tone Stack` $\to$ `V1B Driver + C² Wavefolder` $\to$ `Push-Pull Power Amp + Sag + Hysteresis`) via the Exact LTI Commutativity Theorem (`Theorem 1`), evaluating the multi-drive $d_{\text{bank}} = [0.25\times, 1.0\times, 4.0\times, 16.0\times]$ `PolyADAA` bank and folding $\mathbf{z}_{\text{stage}}$ into $\mathbf{W}_k^{(s)}(\mathbf{z}_{\text{stage}})$ at `3 kHz` ([`op_n_commuted_cascade.md` §3–§4](op_n_commuted_cascade.md#3-exact-lti-basis-commutativity-theorem-theorem-1)).

---

## 5. Stage III: 64-Tap Early SIMD FIR + Kautz Modal Cabinet Bank
Stage III executes between Stage II and Stage IV, combining independent L/R early FIR segments (`16 / 32 / 48 / 64` taps/ch) with shared-pole Kautz modal pairs (`4 / 8 / 16 / 32` pairs, driven at $x_a[n-16]$ at `48 kHz`) and independent L/R complex readouts ([`op_m_modal_kautz.md` §4](op_m_modal_kautz.md#4-stage-iii--family-6-shared-kautz-modal-bank)).

---

## 6. Stage IV: Unitary Spatial / Barberpole Tank (`Op-T[Tank]`)

| Stage IV Component | Architectural & Mathematical Specification |
| :--- | :--- |
| **Physical Structure** | `M` delay lines (`M = 4` Lite, `M = 8` Full/Pro, `M = 16` Ultra) with model-declared 64B-aligned `fp16` histories and grouped bounded-history feedback guard ([Op-T §1](op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)). |
| **Partitioned Schroeder All-Pass Diffusers (`CL-08`)** | Non-overlapping in-buffer segments with rate-converted integer lengths and `∣a_diff,m∣ ≤ 0.75` ([Op-T §3](op_t_transport.md#31-memory-partitioning--rate-aware-capacity)). |
| **`O(M)` Givens + Householder Mixing `U(ψ, v)`** | Block-diagonal `2×2` Givens rotation `R(ψ)` (`∣Δψ∣ ≤ 0.05 rad/tick`) paired with unit-norm Householder reflection `H(v) = I_M - 2vv^T`. |
| **Input & Readout DC Conditioning (`CL-09`)** | Passband-normalized `5 Hz` DC blocker (`0x3F7FD51F` @ `48 kHz`) applied at tank input injection and non-recirculating tank readouts ([`clamp_and_audit_registry.md` §3.1](../training/clamp_and_audit_registry.md#31-stage-iv-dc-blocker-cl-09)). |
| **Blended Feedback Knee & Loop Saturator (`CL-10`, `CL-11`)** | Per-line blended knee `g_eff(u) = g_lin (u² + α ε_knee) / (u² + ε_knee)` (`g_lin ≤ 0.9990`) followed by unity-slope cubic saturator (`V_ceil,m ≤ 0.9850`). |

### 6.1 Captured Freeze and Modeled Controls
Hardware freeze is modeled when present and captured in the training stems ([`capture_and_alignment.md` §1.3](../training/capture_and_alignment.md#freeze-transition-capture)); unobserved freeze options remain clamped under `CL-15`. When a captured Stage IV freeze stops new tank input injection, Stages 0–III continue processing live instrument input for play-over performance, while the `CL-09` DC blockers condition the tank input and non-recirculating readouts without damping the held tank state ([`test_captured_freeze_spec_references.py`](../../tests/test_captured_freeze_spec_references.py)).

---

## 7. Unified 256-D 4-Tier Kronecker Output Head

For each 4-mode Kautz block in Stage III, pair-grouped L/R modal readouts expose four features `[y_0L + y_2L, y_1L + y_3L, y_0R + y_2R, y_1R + y_3R]`, and each tier's 16-tap L/R FIR segment exposes two features:

| Tier | Cumulative Bus Width | New Modal Features | New FIR Features | Stage III Delta | Slots Available to Other Stages |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Lite`** | `64` | `4` | `2` | `6` | `58` |
| **`Full`** | `128` | `4` | `2` | `6` | `58` |
| **`Pro`** | `192` | `8` | `2` | `10` | `54` |
| **`Ultra`** | `256` | `16` | `2` | `18` | `46` |
| **Total** | `256` | `32` | `8` | `40` | `216` |

```text
STAGE-III-TIER-SLOTS
Tier    New modal groups          Other-stage slots    FIR L, FIR R
Lite      0..3   (block 0)             4..61              62,  63
Full     64..67  (block 1)            68..125            126, 127
Pro     128..135 (blocks 2..3)       136..189            190, 191
Ultra   192..207 (blocks 4..7)       208..253            254, 255
```

Slots `4..11` in the `Lite` block carry the L/R Stage I `T1` sub-harmonic tracker features ([`op_m_modal_kautz.md` §3.2](op_m_modal_kautz.md#32-discrete-toggle--half-angle-sheet)):

```text
STAGE-I-T1-FEATURES
  4 | L-cos
  5 | L-sin
  6 | L-half-square
  7 | L-quarter-square
  8 | R-cos
  9 | R-sin
 10 | R-half-square
 11 | R-quarter-square
```

**Effective-Tier Equivalence:** For identical external inputs, control values, initial active-tier state, and backend, a container stripped to tier `T` produces bit-identical Stage II feeds, L/R FIR histories, and retained FIR-segment features to the unstripped container running at effective tier `T` (with all tier `> T` weights and `Ultra` control extensions gated off).

#### Block-Diagonal Lower-Triangular Staging (Refinement 11)
```text
W_head = [ W_lite           0             0             0           ]
         [ W_full,lite     W_full,delta  0             0           ]
         [ W_pro,lite      W_pro,full    W_pro,delta   0           ]
         [ W_ultra,lite    W_ultra,full  W_ultra,pro   W_ultra,delta ]
```

---

## 8. Tier-Aware Static & Dynamic Modulation Routing

### 8.1 Bus Architecture
$$\mathbf{q}_{16}[k] = \big[\mathbf{z}_{\text{ctrl}}^{(8)}[k], \ \mathbf{m}_{\text{dyn}}^{(8)}[k]\big]^\top \in \mathbb{R}^{16}$$

1. **Static Control Latents:** `Lite/Full/Pro` expose `4` head + `4` stage coordinates (`Rank-4` FM); `Ultra` adds `+4` head (`modes 5..8`) + `+4` stage coordinates (`Rank-8` FM) and a `16 × 4` stage-dispatch delta $\mathbf{W}_{\text{ctrl},\Delta}^{\text{Ultra}}$ in `Arena 3` ([`parametric_control_bus.md` §3.1](parametric_control_bus.md#31-stage-parameter-dispatch-matrix-mathbfw_textctrl)).
2. **Autonomous Dynamic Modulators $\mathbf{m}_{\text{dyn}} \in \mathbb{R}^8$ (`float32x4_t[2]`):**
   - $m_0 = z_{\mathrm{re}, \text{lfo0}}[k]$, $m_1 = z_{\mathrm{im}, \text{lfo0}}[k]$ (Quadrature LFO 0 In-Phase & $90^\circ$ rotors).
   - $m_2 = T_{\text{shape}}(z_{\mathrm{re}, \text{lfo0}}[k])$, $m_3 = T_{\text{shape}}(z_{\mathrm{re}, \text{lfo1}}[k])$ (Chebyshev shaped LFO 0 & 1).
   - $m_4 = e_{\text{low}}[k]$ ($< 250\text{ Hz}$), $m_5 = e_{\text{mid}}[k]$ ($250\text{–}2000\text{ Hz}$), $m_6 = e_{\text{high}}[k]$ ($> 2000\text{ Hz}$), $m_7 = e_{\text{aux}}[k]$ (`Port In 3` Aux Key).

---

## 9. Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability (44.1–96 kHz & `CL-20` 4× Bounce)

Containers store discrete-time parameters anchored at $f_{s,\text{train}} = 48,000\text{ Hz}$. For host rate $f_{s,\text{host}} \in [44,100, 96,000]\text{ Hz}$ and oversampling factor $M_{\text{os}} \in \lbrace 1, 4 \rbrace$, let $f_{s,\text{eff}} = M_{\text{os}} f_{s,\text{host}}$ and $\rho_{\text{sr}} = 48000 / f_{s,\text{eff}}$:

1. **Stage I (`Op-M[Pre]`) & Stage III (`Op-M[Modal]`) Pole Radii & Angles (`CL-03`, `CL-04`):**
   $$r_c(f_{s,\text{eff}}) = \big(r_{\text{ref},c}\big)^{\rho_{\text{sr}}} \in \big[0.25^{\rho_{\text{sr}}},\ 0.9995^{\rho_{\text{sr}}}\big], \qquad \theta_c(f_{s,\text{eff}}) = \mathrm{clamp}\Big(\rho_{\text{sr}} \theta_c(f_{s,\text{train}}),\ 10^{-4},\ \pi - 10^{-4}\Big)$$
   At $f_{s,\text{eff}} = 48,000\text{ Hz}$, use the validated serialized coefficients directly.
2. **Stage 0 Relative Alignment Latency & `Op-M[Ctrl]` Detector Poles:**
   $$\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{eff}}) = \frac{\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}, \qquad p_{\text{det},b}(f_{s,\text{eff}}) = \big(p_{\text{det},b}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
3. **Stage II (`Op-N`) Tone-Stack Poles, Sag/Bias Poles, 2D Hysteresis & Family 5 ZOH Phase:**
   $$p_{\text{tone},k}(f_{s,\text{eff}}) = \big(p_{\text{tone},k}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}, \qquad R_{\text{sag},k}(f_{s,\text{eff}}) = \big(R_{\text{sag},k}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}, \qquad R_{\text{ac}}(f_{s,\text{eff}}) = \big(R_{\text{ac}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
   $$\Delta s_{\text{re}}[n] = \rho_{\text{sr}}^{-1}\big(s_{\text{re}}[n] - s_{\text{re}}[n-1]\big)$$
4. **Stage I & IV (`Op-T`) Delays, All-Pass Diffusers, Crossfades & DC Blocker (`CL-06`–`CL-09`):**
   $$\tau_{0,m}^{\text{samples}}(f_{s,\text{eff}}) = \max\left(4.0,\ \frac{\tau_{0,m}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right), \qquad D_{\text{ap},m}(f_{s,\text{eff}}) = \max\left(1,\ \mathrm{round}\left(\frac{D_{\text{ap},m}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)\right)$$
   $$L_{\text{xfade}}(f_{s,\text{eff}}) = \max\left(16,\ \mathrm{round}\left(\frac{L_{\text{xfade}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)\right), \qquad R_{\text{dc}}(f_{s,\text{eff}}) = \big(R_{\text{dc}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
5. **Control-Bus Decimation Stride $K_{\text{ctrl}}$, Tick Ratio $\rho_{\text{ctrl}}$ & Micro-Ramp Step $\alpha_{\text{ramp}}$:**
   $$K_{\text{ctrl}}(f_{s,\text{eff}}) = \max\left(8,\ \mathrm{round}\left(\frac{f_{s,\text{eff}}}{3000.0}\right)\right), \qquad \rho_{\text{ctrl}} = \frac{f_{s,\text{train}} \cdot K_{\text{ctrl}}(f_{s,\text{eff}})}{f_{s,\text{eff}} \cdot K_{\text{ctrl}}(f_{s,\text{train}})}, \qquad \alpha_{\text{ramp}}(f_{s,\text{eff}}) = \frac{1.0}{K_{\text{ctrl}}(f_{s,\text{eff}})}$$
   $$\Delta\phi_{\text{lfo},j}(f_{s,\text{eff}}) = \rho_{\text{ctrl}} \Delta\phi_{\text{lfo},j}(f_{s,\text{train}}), \qquad R_{\text{env},b}(f_{s,\text{eff}}) = \big(R_{\text{env},b}(f_{s,\text{train}})\big)^{\rho_{\text{ctrl}}}$$
6. **Stage III FIR & Modal Onset Conversion (`Up to 8 Host Samples` Budget):**
   - **Integer Multiples ($f_{s,\text{eff}} = q \cdot 48000, q \in \lbrace 1, 2, 4, 8 \rbrace$):** Evaluate exact sparse FIR delays $x_a[n - q(16t + j)]$ with integer modal onset $16q$ (`0` added latency; [`op_m_modal_kautz.md` §4.1](op_m_modal_kautz.md#41-stage-iii-architecture--tier-allocation)).
   - **Live `44.1–48 kHz` First-Tap Continuous Difference Family:** With $\lambda = (48000 - f)/3900$ and $\widetilde p_j = \mathrm{RN}_{32}(\lambda p_j)$, evaluate $x[n-1] + \sum_{j=0}^{62} \widetilde p_j (x[n-j] - x[n-j-1])$ (`< 0.0177` ideal complex error through `20 kHz`, exact unit DC gain on constant normal histories; [`stage3_first_tap_spec_references.rs`](../../packages/engine/tests/stage3_first_tap_spec_references.rs)).
   - **`4×` Bounce (`176.4–384 kHz`) Six-Point Lagrange Difference Mapping:** For tap $k \ge 1$ with exact integer decomposition $d = k f / 48000 = m + \mu$ ($m = \lfloor d \rfloor \ge 3$, $\mu = r/D$), anchor at $a = 2$ ($\mu \le 1/2$) or $a = 3$ ($\mu > 1/2$) with residual coefficients $p_j = \mathrm{RN}_{32}(\sum_{i=0}^j h_i(\mu) - \mathbf{1}_{j \ge a})$:
     $$y_{\text{tap},k}[n] = x[n - m - a + 2] + \sum_{j=0}^4 p_j \big(x[n - m + 2 - j] - x[n - m + 1 - j]\big)$$
     With integer scales $c_i = (-1, 5, -10, 10, -5, 1)$ at nodes $v_i = (-2, -1, 0, 1, 2, 3)$ and exact integer rate ratio $f = M 2^{e-52}$, $D = 48000 \cdot 2^{52-e}$, $k M = m D + r$:
     $$Q_j = \sum_{i=0}^j c_i \prod_{\ell \ne i}(r - v_\ell D) - 120 D^5 \mathbf{1}_{j \ge a}, \qquad p_j = \mathrm{RN}_{32}\left(\frac{Q_j}{120 D^5}\right)$$
     This causal 6-point kernel has ideal complex error $< 0.00064$ through `20 kHz` ($(5/1024)(2\pi \cdot 20000/176400)^6 + 2^{-23} + 10 \cdot 2^{-150} < 0.00064$), full-band gain $< 1.391$, and max ring history `131 / 251 / 379 / 507` samples/ch across `Lite..Ultra` ([`test_cabinet_modal_spec_references.py`](../../tests/test_cabinet_modal_spec_references.py)). Other non-integer effective rates fail fast.
