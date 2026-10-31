# Non-Stop 3-Phase Control-Space Audit & Clamp Registry (`CL-01`..`CL-18`)

Because neural audio training can run unattended for extended periods, **DNAM never blocks or pauses training on user confirmation prompts**. Instead, it executes a **Non-Stop 3-Phase Control-Space Audit & Automatic Clamping Pipeline** that guarantees every exported `.dnam` model is 100% stable across all user knob and switch combinations.

---

## 1. The Non-Stop 3-Phase Control-Space Audit Lifecycle

### Phase 1 — Pre-Training Step-0 Audit ($< 200\text{ ms}$ Before Epoch 1, Non-Blocking Banner)
Immediately upon loading the user's `controls_schema` and stem metadata at Step 0:
1. **`[CLAMP-RANGE]` (Unobserved Range Extrapolation Clamp):**
   For each continuous `range` control $p$ with `declared_range` $[u_{p,\text{decl-min}}, u_{p,\text{decl-max}}]$, computes the observed stem span $[u_{p,\text{obs-min}}, u_{p,\text{obs-max}}]$ and clamps the active safe interval to:
   $$u_{p,\text{safe-min}} = \max\big(u_{p,\text{decl-min}}, u_{p,\text{obs-min}}\big), \qquad u_{p,\text{safe-max}} = \min\big(u_{p,\text{decl-max}}, u_{p,\text{obs-max}}\big)$$
   If $U_p = 1$ (control never moved across stems), locks polynomial degree $K_p = 0$.
2. **`[CLAMP-SWITCH]` (Unobserved Switch Option Masking):**
   For each discrete `switch` control $q$ with `declared_options` $\mathcal{O}_{q,\text{decl}}$, restricts `active_options` to $\mathcal{O}_{q,\text{obs}} = \lbrace o \in \mathcal{O}_{q,\text{decl}} \mid \exists\text{ valid stem with } s_q = o \rbrace$.
3. **`[CLAMP-STABILITY]` (Step-0 Multi-Knob Corner Stability Audit):**
   Evaluates the Step-0 physical parameter fit across all $3^{P_{\text{range}}} \times \prod_q |\mathcal{O}_{q,\text{obs}}|$ corners of `safe_clamped_range` $\times$ `active_options`. If any unrecorded corner (e.g., `feedback = 10.0` + `boost = "on"`) predicts $\hat{r}_c > 0.9995$, $\hat{g}_{\text{lin},m} > 0.9990$, $|\hat{a}_{\text{diff},m}| > 0.75$, or $\hat{\tau}_m < 4.0$, computes the exact conditional ceiling (e.g., `conditional_max: {"boost=on": 9.37}` in `Test 8E`), logs a live warning banner, and **proceeds immediately into Epoch 1**.

### Phase 2 — During-Training Stop-Gradient Corner Barrier & Periodic Rail-Lockup Check
1. **Stop-Gradient Corner Barrier Loss ($\mathcal{L}_{\text{barrier}}$ at Every `Adam` Step, `CL-16`):**
   Evaluating the $3\text{ kHz}$ Control Hyper-Bus at all virtual corner coordinates $\mathcal{V}_{\text{corners}}$ requires zero audio rendering ($< 0.05\text{ ms}$/step). Applying the stop-gradient operator $\mathrm{sg}(\cdot)$ (`detach()`) to the 1st-order main effects $\boldsymbol{\theta}_{\text{main}}(\mathbf{u}_c, \mathbf{s}_c)$:
   $$\mathcal{L}_{\text{barrier}} = \lambda_{\text{bar}} \sum_{(\mathbf{u}_c, \mathbf{s}_c) \in \mathcal{V}_{\text{corners}}} \mathrm{relu}\Big(\mathrm{sg}\big(\boldsymbol{\theta}_{\text{main}}(\mathbf{u}_c, \mathbf{s}_c)\big) + \Delta\boldsymbol{\theta}_{\text{FM}}(\mathbf{u}_c, \mathbf{s}_c) - \boldsymbol{\theta}_{\text{safe}}\Big)^2$$
   steers the 2nd-order Factorization Machine interaction weights $\Delta\boldsymbol{\theta}_{\text{FM}}$ away from unstable corners with **exact $0.00 \times 10^0$ gradient interference on the recorded 1st-order stems** (`Test 8F`).
2. **Periodic Dynamic Rail-Lockup Audit (`[CLAMP-RAIL]`, Every 10 Epochs):**
   Runs a $2,048$-sample synthetic burst through the corners under `torch.no_grad()`; if cascaded `PolyADAA` stages hit > 95% hard-rail saturation ($|u_n| = 1.0$), records a conditional gain ceiling.

### Phase 3 — Post-Training & Post-Polish Export Certification Audit (At `.dnam` Export)
1. **`[CLAMP-DYNAMIC]` (Combined Knob + Learned LFO/Envelope Excursion Clamp, `CL-17`):**
   Audits the worst-case dynamic peak and trough across all controls using the finalized LFO and envelope depths $\mathbf{w}_{r,c}, \mathbf{w}_{\tau,m}$:
   $$r_{c,\text{worst}}(\mathbf{u}, \mathbf{s}) = r_{\text{high},c}(\mathbf{u}, \mathbf{s}) + |\mathbf{w}_{r,c}|^\top \mathbf{m}_{\text{max}} \le 0.9995$$
   $$\tau_{m,\text{min}}(\mathbf{u}, \mathbf{s}) = \tau_{0,m}(\mathbf{u}, \mathbf{s}) - |\mathbf{w}_{\tau,m}|^\top \mathbf{m}_{\text{max}} \ge 4.0\text{ samples}$$
   If a wide learned chorus/flanger LFO sweep at `delay = 0.0` would dip below the 4-sample Cubic Lagrange causal margin, tightens `safe_clamped_range` (e.g., `[0.00, 1.00] -> [0.08, 1.00]` in `Test 8F`).
2. **`[CLAMP-HEAD-GAIN]` (Post-Polish SVD Head Spectral Gain Clamp, `CL-17`):**
   Evaluates the 32-point FFT of the polished 16-tap stereo head $\mathbf{W}_{\text{head}}(\mathbf{u}, \mathbf{s})$ across all corners; if any unrecorded corner exhibits a $> +6.0\text{ dB}$ spectral peak above the loudest training stem, bakes an ellipsoidal trust-region norm clamp $\Vert \mathbf{z}_{\text{head}}\Vert _2 \le \rho_{\text{head}}$ into the `.dnam` file.

---

## 2. Consolidated Post-Training Audit & Stem Prescription Report

Upon completion, the trainer writes the playable `.dnam` file, the `.dnam-ckpt` checkpoint, and prints a **Consolidated Audit & Stem Prescription Report**:

```text
================================================================================
DNAM TRAINING COMPLETE — EXPORTED: model.dnam (100% STABLE) | model.dnam-ckpt
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
| **`CL-01`** | Core 1A `PolyADAA` Pre-Activation Input `u_n` | `u_n = clamp((2/3) x_c[n], -1.0, +1.0)` | `packages/trainer/src/dnam_trainer/dsp/poly_adaa.py` | Bounds cubic/quartic polynomial domain to its monotonic soft-clipping interval `[-1, +1]` |
| **`CL-02`** | Core 1A Passive 1× B-Spline ADAA Blend & AC Coupling | `ỹ_nl = 0.875 A[n] + 0.125 A[n-1]`, `R_ac = 0.9993` | `packages/trainer/src/dnam_trainer/dsp/poly_adaa.py` | Suppresses Nyquist-folded aliasing by 38.87 dB without +6 dB high-pass extrapolation boost |
| **`CL-03`** | Core 1B 3 kHz Algebraic Unit-Circle Rotor `u_{rot,c}` | `y₁ = 1.5 - 0.5(ĉ² + ŝ²)`, `|u_{rot,c}| ≤ 1.000000000000` | `packages/trainer/src/dnam_trainer/dsp/selective_rglru.py` | Guarantees unit-circle rotor never pushes complex poles outside `|z| = 1` (0 `FSQRT`/`FDIV`) |
| **`CL-04`** | Core 1B RG-LRU Pole Radii `r_{low,c}, r_{high,c}` & H₂ Norm | `r ∈ [0.25, 0.9995]`, `γ_{norm,c} = √(1 - r̄_c²)` | `packages/trainer/src/dnam_trainer/dsp/selective_rglru.py` | Enforces BIBO stability (`T₆₀ ≤ 28 s`) and exact unit H₂ impulse energy (1.0000) |
| **`CL-05`** | Core 1B Instantaneous Radial Gate `g_c[n]` | `g_c[n] = clamp(0.5 + w_{sel,c} x_c[n], 0.0, 1.0)` | `packages/trainer/src/dnam_trainer/dsp/selective_rglru.py` | Keeps 48 kHz convex radial interpolation strictly inside `[r_{low,c}, r_{high,c}]` |
| **`CL-06`** | Core 2 Causal Delay Floor `τ_m[n]` & `Range` Slew Rate | `τ_m[n] ≥ 4.0 samp`, `|dτ_m/dn| ≤ 0.25` | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Prevents causal buffer underflow on 4-tap Cubic Lagrange reads & bounds analog Doppler slew |
| **`CL-07`** | Core 2 Discrete `Switch` Dual-Pointer Crossfade `w[k]` | `w[k] = clamp(k / 128, 0.0, 1.0)`, `α_xf = 0.5(1 - cos(π w))` | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Eliminates switch teleport clicks (50.7× lower jerk) with zero Doppler pitch warp |
| **`CL-08`** | Core 2 In-Loop Schroeder All-Pass Coefficient `a_{diff,m}` | `a_{diff,m} ∈ [-0.75, +0.75]` | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Prevents all-pass pole instability while spanning Crisp Echo (`0`), Hall (`>0`), and Spring Drip (`<0`) |
| **`CL-09`** | Core 2 Passband-Normalized 5 Hz DC Blocker `α_dc` | `α_dc = (1 + R_dc) / 2` (`R_dc = 0.995`) | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Enforces exact 0.0000 DC loop gain and `≤ 1.0000` Nyquist passband ceiling |
| **`CL-10`** | Core 2 Quiet-State Decay `g_{lin,m}` & Coercive Knee `v_{knee,m}` | `g_{lin,m} ∈ [0.0, 0.9990]`, `v_{knee,m} = clamp(|q̃_m|/σ_m, 0, 1)` | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Guarantees quiet tails decay monotonically to silence even on self-oscillating delay patches |
| **`CL-11`** | Core 2 Unity-Slope Normalized Cubic Loop Saturator `v_{sat,m}` | `v_{sat,m} = clamp((2 / (3 V_{ceil,m})) u_{loop,m}, -1.0, +1.0)`, `V_{ceil,m} ≤ 0.985` | `packages/trainer/src/dnam_trainer/dsp/modulated_ufdn.py` | Enforces exact unity origin slope `S'(0) ≡ 1.0000` and hard Lyapunov ceiling `≤ V_{ceil,m}` |
| **`CL-12`** | Control Bus Fixed Declared-Range Safe Clamp `x_{p,clamped}` | `x_p ← clamp(x_p, u_{p,safe_min}, u_{p,safe_max})`, `u_p ∈ [-1, +1]` | `packages/trainer/src/dnam_trainer/control/basis.py` | Prevents out-of-bounds knob extrapolation while keeping stem coordinates invariant |
| **`CL-13`** | Control Bus Cardinality-Bounded Newton/Lobatto Degree `K_p` | `K_p = min(U_p - 1, 3) ∈ {0, 1, 2, 3}` | `packages/trainer/src/dnam_trainer/control/basis.py` | Prevents Runge polynomial oscillation and guarantees 0.00 old-stem perturbation on resume |
| **`CL-14`** | Control Bus Step-0 Physical Variance Gate `G_p` | `G_{p,i} = I(Sens_{p,i} > 0.5) ∈ {0.0, 1.0}` | `packages/trainer/src/dnam_trainer/control/basis.py` | Hard-locks invariant delay/LFO modulation paths to `0.0` (zero parasitic knob coupling) |
| **`CL-15`** | Audit Phase-1 Pre-Flight Range, Switch & Corner Clamps | `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]` | `packages/trainer/src/dnam_trainer/control/audit.py` | Non-blocking Step-0 clamping of unobserved ranges, switches, and multi-knob corners |
| **`CL-16`** | Audit Phase-2 Stop-Gradient Corner Barrier `L_barrier` | `relu(sg(θ_main) + Δθ_FM - θ_safe)²` | `packages/trainer/src/dnam_trainer/control/audit.py` | Steers 2nd-order FM weights away from unstable corners with 0.00 main-stem interference |
| **`CL-17`** | Audit Phase-3 Dynamic LFO & Head Spectral Gain Clamps | `[CLAMP-DYNAMIC]` (`τ_min ≥ 4.0`), `[CLAMP-HEAD-GAIN]` (`≤ +6 dB`) | `packages/trainer/src/dnam_trainer/control/audit.py` | Prevents LFO buffer underflow and corner head gain spikes at export |
| **`CL-18`** | Trainer Exact Device Gram & Batched Host `float64` Solve | `σ_k > 10⁻⁴`, `λ_init = 10⁻³ τ̄`, `λ_polish = 10⁻⁶ τ̄` | `packages/trainer/src/dnam_trainer/training/numerics.py` | Accumulates exact 32-bit device Grams (`ExactGramAccumulator`) and solves batched host `float64` normal equations (`HostDenseSolver`) |
