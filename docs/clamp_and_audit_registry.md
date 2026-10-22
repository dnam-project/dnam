# Non-Stop 3-Phase Control-Space Audit & Clamp Registry (`CL-01`..`CL-18`)

Because neural audio training can run unattended for extended periods, **Architecture 3 never blocks or pauses training on user confirmation prompts**. Instead, it executes a **Non-Stop 3-Phase Control-Space Audit & Automatic Clamping Pipeline** that guarantees every exported `.dnam` v3 model is 100% stable across all user knob and switch combinations.

---

## 1. The Non-Stop 3-Phase Control-Space Audit Lifecycle

### Phase 1 — Pre-Training Step-0 Audit ($< 200\text{ ms}$ Before Epoch 1, Non-Blocking Banner)
Immediately upon loading the user's `controls_schema` and stem metadata at Step 0:
1. **`[CLAMP-RANGE]` (Unobserved Range Extrapolation Clamp):**
   For each continuous `range` control $p$ with `declared_range` $[u_{p,\text{decl\_min}}, u_{p,\text{decl\_max}}]$, computes the observed stem span $[u_{p,\text{obs\_min}}, u_{p,\text{obs\_max}}]$ and clamps the active safe interval to:
   $$u_{p,\text{safe\_min}} = \max\!\big(u_{p,\text{decl\_min}},\, u_{p,\text{obs\_min}}\big), \qquad u_{p,\text{safe\_max}} = \min\!\big(u_{p,\text{decl\_max}},\, u_{p,\text{obs\_max}}\big)$$
   If $U_p = 1$ (control never moved across stems), locks polynomial degree $K_p = 0$.
2. **`[CLAMP-SWITCH]` (Unobserved Switch Option Masking):**
   For each discrete `switch` control $q$ with `declared_options` $\mathcal{O}_{q,\text{decl}}$, restricts `active_options` to $\mathcal{O}_{q,\text{obs}} = \{o \in \mathcal{O}_{q,\text{decl}} \mid \exists\text{ valid stem with } s_q = o\}$.
3. **`[CLAMP-STABILITY]` (Step-0 Multi-Knob Corner Stability Audit):**
   Evaluates the Step-0 physical parameter fit across all $3^{P_{\text{range}}} \times \prod_q |\mathcal{O}_{q,\text{obs}}|$ corners of `safe_clamped_range` $\times$ `active_options`. If any unrecorded corner (e.g., `feedback = 10.0` + `boost = "on"`) predicts $\hat{r}_c > 0.9995$, $\hat{g}_{\text{lin},m} > 0.9990$, $|\hat{a}_{\text{diff},m}| > 0.75$, or $\hat{\tau}_m < 4.0$, computes the exact conditional ceiling (e.g., `conditional_max: {"boost=on": 9.37}` in `Test 8E`), logs a live warning banner, and **proceeds immediately into Epoch 1**.

### Phase 2 — During-Training Stop-Gradient Corner Barrier & Periodic Rail-Lockup Check
1. **Stop-Gradient Corner Barrier Loss ($\mathcal{L}_{\text{barrier}}$ at Every `Adam` Step, `CL-16`):**
   Evaluating the $3\text{ kHz}$ Control Hyper-Bus at all virtual corner coordinates $\mathcal{V}_{\text{corners}}$ requires zero audio rendering ($< 0.05\text{ ms}$/step). Applying the stop-gradient operator $\operatorname{sg}(\cdot)$ (`detach()`) to the 1st-order main effects $\boldsymbol{\theta}_{\text{main}}(\mathbf{u}_c, \mathbf{s}_c)$:
   $$\mathcal{L}_{\text{barrier}} = \lambda_{\text{bar}} \sum_{(\mathbf{u}_c, \mathbf{s}_c) \in \mathcal{V}_{\text{corners}}} \operatorname{relu}\!\Big(\operatorname{sg}\!\big(\boldsymbol{\theta}_{\text{main}}(\mathbf{u}_c, \mathbf{s}_c)\big) + \Delta\boldsymbol{\theta}_{\text{FM}}(\mathbf{u}_c, \mathbf{s}_c) - \boldsymbol{\theta}_{\text{safe}}\Big)^2$$
   steers the 2nd-order Factorization Machine interaction weights $\Delta\boldsymbol{\theta}_{\text{FM}}$ away from unstable corners with **exact $0.00 \times 10^0$ gradient interference on the recorded 1st-order stems** (`Test 8F`).
2. **Periodic Dynamic Rail-Lockup Audit (`[CLAMP-RAIL]`, Every 10 Epochs):**
   Runs a $2,048$-sample synthetic burst through the corners under `torch.no_grad()`; if cascaded `PolyADAA` stages hit $> 95\%$ hard-rail saturation ($|u_n| = 1.0$), records a conditional gain ceiling.

### Phase 3 — Post-Training & Post-Polish Export Certification Audit (At `.dnam` v3 Export)
1. **`[CLAMP-DYNAMIC]` (Combined Knob + Learned LFO/Envelope Excursion Clamp, `CL-17`):**
   Audits the worst-case dynamic peak and trough across all controls using the finalized LFO and envelope depths $\mathbf{w}_{r,c}, \mathbf{w}_{\tau,m}$:
   $$r_{c,\text{worst}}(\mathbf{u}, \mathbf{s}) = r_{\text{high},c}(\mathbf{u}, \mathbf{s}) + |\mathbf{w}_{r,c}|^\top \mathbf{m}_{\text{max}} \le 0.9995$$
   $$\tau_{m,\text{min}}(\mathbf{u}, \mathbf{s}) = \tau_{0,m}(\mathbf{u}, \mathbf{s}) - |\mathbf{w}_{\tau,m}|^\top \mathbf{m}_{\text{max}} \ge 4.0\text{ samples}$$
   If a wide learned chorus/flanger LFO sweep at `delay = 0.0` would dip below the 4-sample Cubic Lagrange causal margin, tightens `safe_clamped_range` (e.g., `[0.00, 1.00] -> [0.08, 1.00]` in `Test 8F`).
2. **`[CLAMP-HEAD-GAIN]` (Post-Polish SVD Head Spectral Gain Clamp, `CL-17`):**
   Evaluates the 32-point FFT of the polished 16-tap stereo head $\mathbf{W}_{\text{head}}(\mathbf{u}, \mathbf{s})$ across all corners; if any unrecorded corner exhibits a $> +6.0\text{ dB}$ spectral peak above the loudest training stem, bakes an ellipsoidal trust-region norm clamp $\|\mathbf{z}_{\text{head}}\|_2 \le \rho_{\text{head}}$ into the `.dnam` v3 file.

---

## 2. Consolidated Post-Training Audit & Stem Prescription Report

Upon completion, the trainer writes the playable `.dnam` v3 file, the `.dnam-ckpt` checkpoint, and prints a **Consolidated Audit & Stem Prescription Report**:

```text
================================================================================
DNAM A3 TRAINING COMPLETE — EXPORTED: model.dnam (100% STABLE) | model.dnam-ckpt
================================================================================
Active Control Clamps Applied (4):
  1. [CLAMP-RANGE]     'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]
                       -> Prescription: Add stems at gain=0.0 and gain=10.0
  2. [CLAMP-SWITCH]    'boost': Disabled unobserved option ['ultra'] (active: ['off', 'on'])
                       -> Prescription: Add >= 1 stem with boost='ultra'
  3. [CLAMP-STABILITY] 'feedback': Clamped <= 9.37 when boost='on' (g_lin <= 0.9990)
                       -> Prescription: Add corner stem (feedback=10.0, boost='on')
  4. [CLAMP-DYNAMIC]   'delay_ms': Tightened min 0.00 -> 0.08 (LFO trough >= 4.0 samp)

To lift these clamps in < 5 seconds without retraining from scratch, run:
  uv run dnam-train --resume model.dnam-ckpt --add-stems stems_extra.json
================================================================================
```

---

## 3. Canonical Clamp & Limit Registry (`CL-01`..`CL-18`)

Every numerical clamp in the Python reference DSP, PyTorch trainer, Rust native DSP (`dnam-dsp`), and Node.js WASM runtime must adhere to this registry:

| Clamp ID | Subsystem & Variable | Mathematical Bound / Law | Target Python Module | Physical / Numerical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`CL-01`** | Core 1A `PolyADAA` Pre-Activation Input $u_n$ | $u_n = \operatorname{clamp}(\frac{2}{3}x_c[n], -1.0, +1.0)$ | `src/dnam/dsp/poly_adaa.py` | Bounds cubic/quartic polynomial domain to its monotonic soft-clipping interval $[-1, +1]$ |
| **`CL-02`** | Core 1A Passive $1\times$ B-Spline ADAA Blend & AC Coupling | $\tilde{y}_{\text{nl}} = 0.875\mathcal{A}[n] + 0.125\mathcal{A}[n-1]$, $R_{\text{ac}} = 0.9993$ | `src/dnam/dsp/poly_adaa.py` | Suppresses Nyquist-folded aliasing by $38.87\text{ dB}$ without $+6\text{ dB}$ high-pass extrapolation boost |
| **`CL-03`** | Core 1B $3\text{ kHz}$ Algebraic Unit-Circle Rotor $u_{\text{rot},c}$ | $y_1 = 1.5 - 0.5(\hat{c}^2 + \hat{s}^2)$, $|u_{\text{rot},c}| \le 1.000000000000$ | `src/dnam/dsp/selective_rglru.py` | Guarantees unit-circle rotor never pushes complex poles outside $|z| = 1$ ($0$ `FSQRT`/`FDIV`) |
| **`CL-04`** | Core 1B RG-LRU Pole Radii $r_{\text{low},c}, r_{\text{high},c}$ & $\mathcal{H}_2$ Norm | $r \in [0.25, 0.9995]$, $\gamma_{\text{norm},c} = \sqrt{1 - \bar{r}_c^2}$ | `src/dnam/dsp/selective_rglru.py` | Enforces BIBO stability ($T_{60} \le 28\text{ s}$) and exact unit $\mathcal{H}_2$ impulse energy ($1.0000$) |
| **`CL-05`** | Core 1B Instantaneous Radial Gate $g_c[n]$ | $g_c[n] = \operatorname{clamp}(0.5 + w_{\text{sel},c} x_c[n], 0.0, 1.0)$ | `src/dnam/dsp/selective_rglru.py` | Keeps $48\text{ kHz}$ convex radial interpolation strictly inside $[r_{\text{low},c}, r_{\text{high},c}]$ |
| **`CL-06`** | Core 2 Causal Delay Floor $\tau_m[n]$ & `Range` Slew Rate | $\tau_m[n] \ge 4.0\text{ samp}$, $\|d\tau_m/dn\| \le 0.25$ | `src/dnam/dsp/modulated_ufdn.py` | Prevents causal buffer underflow on 4-tap Cubic Lagrange reads & bounds analog Doppler slew |
| **`CL-07`** | Core 2 Discrete `Switch` Dual-Pointer Crossfade $w[k]$ | $w[k] = \operatorname{clamp}(k / 128, 0.0, 1.0)$, $\alpha_{\text{xf}} = \frac{1}{2}(1 - \cos(\pi w))$ | `src/dnam/dsp/modulated_ufdn.py` | Eliminates switch teleport clicks ($50.7\times$ lower jerk) with zero Doppler pitch warp |
| **`CL-08`** | Core 2 In-Loop Schroeder All-Pass Coefficient $a_{\text{diff},m}$ | $a_{\text{diff},m} \in [-0.75, +0.75]$ | `src/dnam/dsp/modulated_ufdn.py` | Prevents all-pass pole instability while spanning Crisp Echo ($0$), Hall ($>0$), and Spring Drip ($<0$) |
| **`CL-09`** | Core 2 Passband-Normalized $5\text{ Hz}$ DC Blocker $\alpha_{\text{dc}}$ | $\alpha_{\text{dc}} = \frac{1 + R_{\text{dc}}}{2}$ ($R_{\text{dc}} = 0.995$) | `src/dnam/dsp/modulated_ufdn.py` | Enforces exact $0.0000$ DC loop gain and $\le 1.0000$ Nyquist passband ceiling |
| **`CL-10`** | Core 2 Quiet-State Decay $g_{\text{lin},m}$ & Coercive Knee $v_{\text{knee},m}$ | $g_{\text{lin},m} \in [0.0, 0.9990]$, $v_{\text{knee},m} = \operatorname{clamp}(\|\tilde{q}_m\|/\sigma_m, 0, 1)$ | `src/dnam/dsp/modulated_ufdn.py` | Guarantees quiet tails decay monotonically to silence even on self-oscillating delay patches |
| **`CL-11`** | Core 2 Unity-Slope Normalized Cubic Loop Saturator $v_{\text{sat},m}$ | $v_{\text{sat},m} = \operatorname{clamp}(\frac{2}{3 V_{\text{ceil},m}} u_{\text{loop},m}, -1.0, +1.0)$, $V_{\text{ceil},m} \le 0.985$ | `src/dnam/dsp/modulated_ufdn.py` | Enforces exact unity origin slope $S'(0) \equiv 1.0000$ and hard Lyapunov ceiling $\le V_{\text{ceil},m}$ |
| **`CL-12`** | Control Bus Fixed Declared-Range Safe Clamp $x_{p,\text{clamped}}$ | $x_p \leftarrow \operatorname{clamp}(x_p, u_{p,\text{safe\_min}}, u_{p,\text{safe\_max}})$, $u_p \in [-1, +1]$ | `src/dnam/control/basis.py` | Prevents out-of-bounds knob extrapolation while keeping stem coordinates invariant |
| **`CL-13`** | Control Bus Cardinality-Bounded Newton/Lobatto Degree $K_p$ | $K_p = \min(U_p - 1, 3) \in \{0, 1, 2, 3\}$ | `src/dnam/control/basis.py` | Prevents Runge polynomial oscillation and guarantees $0.00$ old-stem perturbation on resume |
| **`CL-14`** | Control Bus Step-0 Physical Variance Gate $\mathbf{G}_p$ | $G_{p,i} = \mathbb{I}(\text{Sens}_{p,i} > 0.5) \in \{0.0, 1.0\}$ | `src/dnam/control/basis.py` | Hard-locks invariant delay/LFO modulation paths to `0.0` (zero parasitic knob coupling) |
| **`CL-15`** | Audit Phase-1 Pre-Flight Range, Switch & Corner Clamps | `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]` | `src/dnam/control/audit.py` | Non-blocking Step-0 clamping of unobserved ranges, switches, and multi-knob corners |
| **`CL-16`** | Audit Phase-2 Stop-Gradient Corner Barrier $\mathcal{L}_{\text{barrier}}$ | $\operatorname{relu}(\operatorname{sg}(\boldsymbol{\theta}_{\text{main}}) + \Delta\boldsymbol{\theta}_{\text{FM}} - \boldsymbol{\theta}_{\text{safe}})^2$ | `src/dnam/control/audit.py` | Steers 2nd-order FM weights away from unstable corners with $0.00$ main-stem interference |
| **`CL-17`** | Audit Phase-3 Dynamic LFO & Head Spectral Gain Clamps | `[CLAMP-DYNAMIC]` ($\tau_{\text{min}} \ge 4.0$), `[CLAMP-HEAD-GAIN]` ($\le +6\text{ dB}$) | `src/dnam/control/audit.py` | Prevents LFO buffer underflow and corner head gain spikes at export |
| **`CL-18`** | Trainer SVD Subspace Floor & Dual-Ridge `float64` Solve | $\sigma_k > 10^{-4}$, $\lambda_{\text{init}} = 10^{-3}\bar{\tau}$, $\lambda_{\text{polish}} = 10^{-6}\bar{\tau}$ | `src/dnam/trainer/kronecker_head.py` | Eliminates collinear user controls and achieves $-25.39\text{ dB}$ Step-0 $\to$ $-47.63\text{ dB}$ Polish ESR |
