# `.dnam` Container Schema, `.dnam-ckpt` Checkpoint & Continuous-Time ZOH Rescaling

DNAM defines two companion file formats:
1. **`.dnam` (Version `1.0.0`):** The lightweight, single-file real-time deployment container loaded by the native DSP engine (`crates/dnam-dsp`), audio plugin (`crates/dnam-plugin`), and Node.js/WASM runtime (`packages/nodejs`).
2. **`.dnam-ckpt` (Version `1.0.0`):** The single-file iterative training checkpoint containing the model weights, control metadata, and per-stem factorized `float64` audio sufficient statistics $\{\mathbf{G}_{\text{audio}}^{(s)}, \mathbf{C}_{\text{audio}}^{(s)}\}$ ($130\text{ KB/stem}$).

---

## 1. Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability ($44.1\text{--}96\text{ kHz}$)

Whereas NAM A2 models are hard-locked to a single fixed training sample rate ($48\text{ kHz}$), `.dnam` stores all pole decay rates $\gamma_c\text{ [s}^{-1}]$, pole frequencies $f_{0,c}\text{ [Hz]}$, LFO frequencies $f_{\text{lfo},j}\text{ [Hz]}$, and FDN delay times $\tau_{0,m}\text{ [ms]}$ in **continuous physical time**.

When a `.dnam` file trained at $f_{s,\text{train}} = 48,000\text{ Hz}$ is loaded by a DAW or host running at an arbitrary sample rate $f_{s,\text{host}} \in [44,100,\; 96,000]\text{ Hz}$, define the sample-rate ratio $\rho_{\text{sr}} = \frac{f_{s,\text{train}}}{f_{s,\text{host}}}$:
1. **Core 1B Complex Pole Radii & Angles:**
   $$r_c(f_{s,\text{host}}) = \big(r_c(f_{s,\text{train}})\big)^{\rho_{\text{sr}}} = \exp\!\left(-\frac{\gamma_c}{f_{s,\text{host}}}\right), \qquad \theta_c(f_{s,\text{host}}) = \rho_{\text{sr}} \, \theta_c(f_{s,\text{train}}) = \frac{2\pi f_{0,c}}{f_{s,\text{host}}}$$
2. **Core 1B $\mathcal{H}_2$-Norm Input Normalizer:**
   $$\gamma_{\text{norm},c}(f_{s,\text{host}}) = \sqrt{1 - \bar{r}_c(f_{s,\text{host}})^2}$$
3. **Core 2 Fractional Delay Lines & All-Pass Diffusers:**
   $$\tau_{0,m}^{\text{samples}}(f_{s,\text{host}}) = \max\!\left(4.0,\; \frac{\tau_{0,m}^{\text{ms}} \cdot f_{s,\text{host}}}{1000.0}\right), \qquad D_{\text{ap},m}(f_{s,\text{host}}) = \operatorname{round}\!\left(\frac{D_{\text{ap},m}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)$$
4. **Control-Bus Decimation Stride & 1-Pole Envelopes:**
   $$K_{\text{ctrl}}(f_{s,\text{host}}) = \max\!\left(8,\; \operatorname{round}\!\left(\frac{f_{s,\text{host}}}{3000.0}\right)\right), \qquad R_{\text{dc}}(f_{s,\text{host}}) = \big(R_{\text{dc}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$

---

## 2. Slimmable Block-Diagonal Tiers (`Lite` $C=4, M=4$ vs. `Full` $C=8, M=8$)

Every `.dnam` file contains a single unified weight set structured with a **2-tier block-diagonal mask**:
- **`Lite` Tier ($C = 4\text{ channels}, M = 4\text{ FDN lines}$):** Occupies channels `0..3` (exact 128-bit `float32x4_t` single-register alignment). Convolutions and state-space recurrences on channels `0..3` never read from channels `4..7`.
- **`Full` Tier ($C = 8\text{ channels}, M = 8\text{ FDN lines}$):** Occupies channels `0..7` (exact two-register `float32x4_t` alignment). Channels `4..7` may read from `0..7`, refining harmonic detail and spatial density without altering the `Lite` sub-network.

---

## 3. Canonical `.dnam` JSON Schema

```json
{
  "version": "1.0.0",
  "architecture": "DNAM_TriPhysics",
  "sample_rate_train_hz": 48000,
  "io_config": {
    "inputs": 2,
    "outputs": 2,
    "stereo_mode": "true_stereo_mid_side"
  },
  "slimmable_tiers": {
    "lite": { "channels": 4, "fdn_lines": 4, "early_taps": 4, "head_svd_rank": 4 },
    "full": { "channels": 8, "fdn_lines": 8, "early_taps": 8, "head_svd_rank": 4 }
  },
  "controls_schema": [
    {
      "id": "gain",
      "name": "Drive",
      "type": "range",
      "unit": "",
      "lobatto_degree": 2,
      "anchor_nodes_norm": [0.0, -0.6, 0.6],
      "declared_range": [0.0, 10.0],
      "safe_clamped_range": [2.0, 8.0],
      "default": 5.0,
      "conditional_max": {},
      "clamp_warning": "CLAMP-RANGE: Clamped to observed stem coverage [2.0, 8.0]"
    },
    {
      "id": "feedback",
      "name": "Repeats",
      "type": "range",
      "unit": "%",
      "lobatto_degree": 2,
      "anchor_nodes_norm": [0.0, -1.0, 1.0],
      "declared_range": [0.0, 10.0],
      "safe_clamped_range": [0.0, 10.0],
      "default": 5.0,
      "conditional_max": { "boost=on": 9.37 },
      "clamp_warning": "CLAMP-STABILITY: Clamped <= 9.37 when boost='on' (g_lin <= 0.9990)"
    },
    {
      "id": "boost",
      "name": "Boost Mode",
      "type": "switch",
      "declared_options": ["off", "on", "ultra"],
      "active_options": ["off", "on"],
      "default": "off",
      "clamp_warning": "CLAMP-SWITCH: Disabled unobserved option ['ultra']"
    }
  ],
  "control_bus": {
    "decimation": 16,
    "envelopes": 3,
    "lfo_oscillators": 2,
    "lfo_harmonics": 4,
    "fm_interaction_rank": 4,
    "head_trust_region_radius": 2.5
  },
  "core1_hybrid_ssm": {
    "blocks": 4,
    "conv_kernel": 6,
    "dilations": [1, 3, 7, 19],
    "adaa_mode": "1x_bspline_algebraic",
    "ac_coupling_pole": 0.9993
  },
  "core2_mufdn": {
    "delay_storage": "fp16",
    "interp": "cubic_lagrange_4tap",
    "tzf_reference_tap_samples": 64.0,
    "switch_xfade_samples": 128,
    "allpass_delays_samples": [19, 31, 47, 73, 103, 139, 179, 223],
    "dc_blocker_pole": 0.995,
    "loop_ceiling": 0.985
  },
  "weights": {
    "control_bus_tensors": "...",
    "core1_blocks": "...",
    "core1_stereo_head_svd": "...",
    "core2_mufdn_tensors": "..."
  }
}
```

---

## 4. Canonical `.dnam-ckpt` Iterative Checkpoint Structure

The `.dnam-ckpt` checkpoint stores:
1. **`dnam_model`:** The complete exported `.dnam` dictionary above.
2. **`backbone_state_dict`:** Full `float32` PyTorch state dictionary of Core 1, Core 2, and the Control Bus.
3. **`stems_registry`:** List of recorded stem descriptors, each containing:
   - `stem_id`: Unique string identifier
   - `controls_meta`: Dictionary of raw control values `{"gain": 5.0, "feedback": 5.0, "boost": "off"}`
   - `extracted_physics`: Step-0 extracted delays $\tau_{0,m}^{(s)}$, pole frequencies $f_{0,c}^{(s)}$, LFO frequencies $f_{\text{lfo},j}^{(s)}$, and decay gains $g_{\text{lin},m}^{(s)}$
   - `g_audio_f64`: Factorized per-stem audio Gram matrix $\mathbf{G}_{\text{audio}}^{(s)} = \mathbf{\Phi}_s^\top \mathbf{\Phi}_s \in \mathbb{R}^{16C \times 16C}$ (`float64`, $131,072\text{ B}$ for $C=8$)
   - `c_audio_f64`: Factorized per-stem audio cross-correlation $\mathbf{C}_{\text{audio}}^{(s)} = \mathbf{\Phi}_s^\top \mathbf{Y}_s \in \mathbb{R}^{16C \times 2}$ (`float64`, $2,048\text{ B}$ for $C=8$)
4. **`audit_report`:** Structured Phase 1–3 clamp diagnostics and actionable stem prescriptions.
