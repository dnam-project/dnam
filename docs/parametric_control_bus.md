# Autonomous & Parametric Control Bus ($16\times$ Decimated @ $3\text{ kHz}$)

Every $K_{\text{ctrl}} = 16\text{ samples}$ ($3\text{ kHz}$ at $f_s = 48\text{ kHz}$), the Control Bus updates both the **autonomous signal-driven modulators** $\mathbf{m}[k] \in \mathbb{R}^7$ and the **user-defined parametric control offsets** $\Delta\boldsymbol{\theta}(\mathbf{u}[k], \mathbf{s}[k])$ trained from multi-stem captures.

---

## 1. Autonomous Signal-Driven Modulators ($\mathbf{m}[k] \in \mathbb{R}^7$)

1. **3-Band Asymmetric Envelope Follower ($\mathbf{e}[k] \in \mathbb{R}^3$):**
   Tracks RMS/peak envelopes across three musical sub-bands via one-pole asymmetric attack/release filters:
   - **Low Band ($< 250\text{ Hz}$):** Power-supply capacitor sag, transformer saturation, and speaker cone excursion.
   - **Mid Band ($250\text{--}2,000\text{ Hz}$):** Envelope-filter / Auto-Wah resonant cutoff modulation and optical compression.
   - **Full Band:** Instantaneous grid-current blocking distortion bias shift and dynamic FDN ducking.
2. **Autonomous 4-Harmonic Chebyshev LFO Bank ($\boldsymbol{\phi}[k] \in \mathbb{R}^4$):**
   Two quadrature oscillator pairs advance on the unit circle at $3\text{ kHz}$ via complex multiplication $z_{\text{lfo},j}[k] = e^{i \cdot 16\omega_{\text{lfo},j}} z_{\text{lfo},j}[k-1]$ followed by the algebraic Newton unit-circle stabilizer $y_1 = 1.5 - 0.5|z_{\text{lfo},j}|^2$ (`CL-03`). A 4-harmonic Chebyshev polynomial shaper ($T_1(x) = x$, $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$, $T_4(x) = 8x^4 - 8x^2 + 1$) transforms the pure sine wave into **triangle, sawtooth, asymmetric incandescent Uni-Vibe lamp, or rotary horn/drum LFO trajectories**.
3. **Blind Phase-Locked Teacher Extractor (Training Only, Discarded at Export):**
   Because physical analog RC/op-amp oscillators exhibit thermal frequency drift over a 3-minute capture, a lightweight 1D-CNN Teacher extracts instantaneous LFO phase $\hat{\phi}_{\text{teacher}}[k]$ from the target wet stem during training and fades out via cosine annealing so the exported `.dnam` model runs 100% autonomously.

---

## 2. User-Definable Parametric Controls (`Range` & `Switch`)

Users can declare an arbitrary list of continuous knobs (`type: "range"`) and discrete selectors (`type: "switch"`) in `controls_schema` and provide multiple training stems tagged with control metadata.

### 2.1 Why Naive Conditioning Architectures Fail on Multi-Stem Captures
1. **Why $48\text{ kHz}$ FiLM/Concatenation Wastes Silicon:** Concatenating static knob values to the $48\text{ kHz}$ audio tensor forces the CPU to recompute constant matrix products $48,000\text{ times/sec}$.
2. **Why Dense MLPs Hallucinate on Sparse Captures:** Real users record **Star ("1- knob-at-a-time") sweeps** ($1 + 2P$ stems) rather than exhaustive $5^P$ grids. A dense ReLU/GELU MLP entangles all inputs and hallucinates wild gain/pole explosions in unrecorded multi-knob corners.
3. **Why Channel Crossfading Causes "Double-Echo" Ghosts:** Interpolating between two stems with different delay times ($\tau = 200\text{ ms}$ and $\tau = 400\text{ ms}$) by crossfading output channels produces **two simultaneous echoes at $200\text{ ms}$ and $400\text{ ms}$** instead of a single echo at $300\text{ ms}$!

### 2.2 Fixed Declared-Range Coordinate System + Safe-Interval Clamping (`CL-12`)
Each continuous `range` control $p \in \{1, \dots, P_{\text{range}}\}$ has an **immutable `declared_range`** $[u_{p,\text{decl\_min}}, u_{p,\text{decl\_max}}]$ and a verified **`safe_clamped_range`** $[u_{p,\text{safe\_min}}, u_{p,\text{safe\_max}}] \subseteq [u_{p,\text{decl\_min}}, u_{p,\text{decl\_max}}]$. At runtime (`CL-12`):
$$x_{p,\text{clamped}} = \operatorname{clamp}\!\big(x_p,\; u_{p,\text{safe\_min}},\; u_{p,\text{safe\_max}}\big), \qquad u_p = 2 \cdot \frac{x_{p,\text{clamped}} - u_{p,\text{decl\_min}}}{u_{p,\text{decl\_max}} - u_{p,\text{decl\_min}}} - 1 \in [-1.0, +1.0]$$

> [!IMPORTANT]
> **Why Normalization Always Uses `declared_range` Instead of `safe_clamped_range`:** If $u_p$ were normalized by $[u_{p,\text{safe\_min}}, u_{p,\text{safe\_max}}]$, then adding a new stem at `gain = 10.0` during iterative resume (expanding `safe_clamped_range` from `[2.0, 8.0]` to `[2.0, 10.0]`) would shift the normalized coordinates $u_p^{(s)}$ of all previously recorded stems! Normalizing by the immutable `declared_range` $[0.0, 10.0]$ guarantees that previously recorded stem coordinates $u_p^{(s)}$ remain **100% invariant across all future stem expansions**.

### 2.3 Cardinality-Bounded Node-Anchored Newton/Lobatto Basis (`CL-13`)
Let $U_p$ be the number of distinct values of knob $p$ present in the training stems, ordered with the reference/noon stem $u_{p,1}$ first: $\{u_{p,1}, u_{p,2}, u_{p,3}, \dots\}$. To prevent polynomial Runge oscillation when a knob is only sampled at 1 or 2 positions, the polynomial degree is hard-bounded by (`CL-13`):
$$K_p = \min\!\big(U_p - 1,\; 3\big) \in \{0, 1, 2, 3\}$$
Each knob $u_p$ expands into $K_p$ hierarchical **Node-Anchored Newton Basis** terms:
$$\tilde{\boldsymbol{\psi}}_p(u_p) = \Big[\underbrace{u_p - u_{p,1}}_{k=1},\; \underbrace{(u_p - u_{p,1})(u_p - u_{p,2})}_{k=2},\; \underbrace{(u_p - u_{p,1})(u_p - u_{p,2})(u_p - u_{p,3})}_{k=3}\Big]^\top \in \mathbb{R}^{K_p}$$
- **Special Case (Standard Noon + Endpoints $\{0, -1, +1\}$):** When $u_{p,1} = 0, u_{p,2} = -1, u_{p,3} = +1$, this reduces to the **Noon-Anchored Lobatto Basis** $[u_p,\; u_p(u_p + 1),\; u_p(u_p^2 - 1)]^\top$ (or centered $[u_p,\; u_p^2,\; u_p(u_p^2 - 1)]^\top$).
- **Zero Old-Stem Perturbation on Iterative Stem Addition (`Test 8G`):** Because the $k$-th Newton basis term $\psi_{p,k}(u_p) = \prod_{j=1}^k (u_p - u_{p,j})$ vanishes identically (`== 0.0`) at **all $k$ previously recorded stem coordinates** $\{u_{p,1}, \dots, u_{p,k}\}$, unlocking $\psi_{p,k}$ when a new stem $u_{p,k+1}$ is added fits the new stem with **exact $0.00 \times 10^0$ perturbation ($< 10^{-15}$)** to previously converged stems!

### 2.4 Centered ANOVA / HDMR + Rank-4 Factorization Machine (`FM`) Hyper-Bus
To combine 1st-order `range` main effects, discrete `switch` embeddings $\mathbf{E}_q[s_q]$, and pairwise knob-knob / knob-switch interactions (e.g., `Gain` $\times$ `Boost`, `Feedback` $\times$ `Mode`, or Fender TMB tone-stack interdependence) in linear $O(P \cdot R)$ time with **zero unrecorded corner hallucination**:
$$\Delta\boldsymbol{\theta}(\mathbf{u}, \mathbf{s}) = \sum_{p=1}^{P_{\text{range}}} \mathbf{G}_p \odot \big(\mathbf{A}_p \, \tilde{\boldsymbol{\psi}}_p(u_p)\big) + \sum_{q=1}^{P_{\text{switch}}} \mathbf{E}_q[s_q] + \mathbf{U}_{\text{int}} \cdot \frac{1}{2}\!\left[\left(\sum_{i=1}^{P} \mathbf{v}_i z_i\right)^{\odot 2} - \sum_{i=1}^{P} \mathbf{v}_i^{\odot 2} z_i^2\right]$$
where $z_i$ is the primary coordinate of control $i$ ($u_p - u_{p,1}$ for ranges, or indicator for switches) and $\mathbf{v}_i \in \mathbb{R}^R$ ($R = 4$) are low-rank interaction factors penalized by $L_2$ regularization so pairwise cross-terms remain zero unless supported by multi-knob stems.

### 2.5 Step-0 Order-Statistic Optimal Transport & Physical Variance Gating (`CL-14`)
1. **Step-0 Order-Statistic Optimal Transport (Preventing Permutation & Double-Echo Artifacts):**
   Before Epoch 1, the trainer extracts the physical delay times $\tau_{0,1}^{(s)} < \tau_{0,2}^{(s)} < \dots < \tau_{0,M}^{(s)}$ (via autocorrelation/cepstral peak picking) and sorted complex pole frequencies $f_{0,1}^{(s)} < \dots < f_{0,C}^{(s)}$ for each stem $s$, matching the $m$-th order statistic of stem $s$ directly to delay line $m$ and fitting the log-delay trajectory $\log \tau_{0,m}(\mathbf{u}, \mathbf{s})$ in closed form (**$3.41 \times 10^{-13}\text{ ms}$ unseen delay error** in `Test 8A`).
2. **Step-0 Physical Variance Gating ($\mathbf{G}_p \in \{0, 1\}^{\dim(\boldsymbol{\theta})}$, `CL-14`):**
   If varying knob $p$ across the training stems produces $< 0.2\text{ samples}$ of variance in delay time $\tau_{0,m}$ or LFO frequency $f_{\text{lfo},j}$ (sensitivity below $\tau_{\text{var}} = 0.5$, e.g., a `Drive` or `Treble` knob), the binary gate $\mathbf{G}_p$ hard-locks the $u_p \to \tau_{0,m}$ and $u_p \to f_{\text{lfo},j}$ projection weights to exact `0.0` (`Test 8B`), guaranteeing that turning `Drive` can never cause parasitic delay pitch shifts.

### 2.6 Zero-MAC Runtime Caching at $48\text{ kHz}$
When user controls are static ($\mathbf{u}[k] == \mathbf{u}[k-1]$ and $\mathbf{s}[k] == \mathbf{s}[k-1]$), a dirty-bit check at $3\text{ kHz}$ skips the entire parametric polynomial evaluation and keeps the collapsed $2 \times 16C$ stereo output head $\mathbf{W}_{\text{head}}(\mathbf{u}, \mathbf{s}) = \mathbf{W}_{\text{head},0} + \sum_{k=1}^{D_h} z_{\text{head},k}(\mathbf{u}, \mathbf{s})\,\mathbf{W}_{\text{head},k}$ cached in L1 memory—costing **$0\text{ extra MACs/sample}$ at $48\text{ kHz}$**!
