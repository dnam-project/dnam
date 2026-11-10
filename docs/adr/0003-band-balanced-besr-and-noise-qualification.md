---
id: ADR-0003
title: "Floor-Regularized Band-Balanced bESR & Rectangular Noise Qualification"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/architecture/psychoacoustic_quality_and_tiers.md
requirements:
  - REQ-2.1
  - REQ-2.2
  - REQ-2.3
---

# ADR-0003: Floor-Regularized Band-Balanced `bESR` & Rectangular Noise Qualification

This record justifies the perceptual quality evaluation framework in [`../specs/architecture/psychoacoustic_quality_and_tiers.md`](../specs/architecture/psychoacoustic_quality_and_tiers.md) fulfilling [`REQ-2.1`–`REQ-2.3`](../requirements/system_goals_and_requirements.md#3-goal-2--perceptual-studio-quality-besr-parametric-control--quiet-stem-handling).

## 1. Context: Why Plain Broadband `ESR` and Fixed High-Pass Pre-Emphasis Fail

Evaluating neural audio models with plain broadband Error-to-Signal Ratio ($\text{ESR} = \Vert \hat{y} - y\Vert _2^2 / \Vert y\Vert _2^2$) or fixed first-order high-pass pre-emphasis ($H_{\text{pre}}(z) = 1 - 0.95 z^{-1}$) fails across universal instruments for five physical reasons:

| Failure Mode | Physical Cause | Audible Consequence |
| :--- | :--- | :--- |
| **1. Broadband `ESR` Low-Frequency Dominance** | Musical instruments roll off at `-6` to `-12 dB/oct` above `400 Hz` (`1/f²` spectrum), concentrating `95–99%` of total energy below `1 kHz` while upper harmonics (`2.5–18 kHz`) carry `< 0.5%` of total energy. | A model that drops `> 2.5 kHz` harmonics by `-6 dB` or blankets pick attack still reports a deceptive `-25 dB` broadband `ESR`. |
| **2. Time-Domain MSE Phase-Mismatch Bias & Loud-Passage Masking** | Unweighted time-domain MSE penalizes high-frequency phase error and integrates energy over full files, so loud passages can mask quiet-passage errors. For an equal-amplitude sinusoid, an absolute wrapped phase error greater than `60°` has more error energy than omitting it; a smaller phase error does not. | Phase mismatch can favor attenuating a partial; full-file averaging can hide crossover or decay distortion during quiet tails. |
| **3. Fixed Pre-Emphasis Sub-Bass Starvation** | A fixed `+6 dB/oct` high-pass filter (`1 - 0.95 z⁻¹`) attenuates `40 Hz` by `-24 dB` relative to `4 kHz`. | Sacrifices sub-bass fundamental weight and phase coherence on bass amplifiers, `808`s, and sub-synths (`30–120 Hz`). |
| **4. Bright-Source Midrange Starvation** | Cymbals, drums, and bright synth leads already contain high `5–16 kHz` energy. | Fixed high-pass pre-emphasis over-weights HF and starves midrange body (`150–800 Hz`). |
| **5. Analog Thermal Hiss Amplification** | Clean or low-passed analog sources (e.g., bass cabs, warm Rhodes, steep low-pass filter sweeps) have no signal above `2.5 kHz` other than `-60` to `-75 dBFS` thermal hiss. | Blindly boosting HF forces the model to fit uncorrelated random Johnson-Nyquist thermal noise. |

Similarly, a 1st-order (`-6 dB/oct`) high-pass filter at `2.5 kHz` attenuates `800 Hz` by only `-10.3 dB` and `110 Hz` by `-27 dB`, allowing strong low/mid instrument fundamentals to leak into the `HF` band (`74%` of `tgt_hf` energy on `1/f²` guitar and bass signals).

## 2. Decision & Counterexamples to Naive Formulations

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Noise-Adjusted `bESR`, `MR-IIR`, `LTAS` & `SegESR` (`ref[qual.besr.noise-subtracted]`, `ref[qual.besr.joint-intervals]`, `ref[qual.companion.mr-iir]`, `ref[qual.companion.ltas]`, `ref[qual.companion.seg-esr]`)** | Per-output evaluation (`γ_hiss = 1.0`, sum denominator), 5-scale 32-probe unity-peak bandpass resonator bank `MR-IIR` (`M ∈ {256..4096}`), 33-band base-10 1/3-octave `LTAS` (`20 Hz–20 kHz` preserving `2–18 kHz` HF), duration-weighted linear `SegESR` across rate runs, and rectangular joint energy-interval qualification bounds. | Pooled stereo `bESR` hides a missing quiet channel; arithmetic dB averaging of frame ESRs turns one `-180 dB` frame + one `0 dB` frame into `-90 dB`; target-only endpoint checks miss interior qualification jumps when `Y g` crosses `F`. |

## 3. Statistical Noise-Floor & Calibration Profiles

For uniform quantizer step $\Delta = R_{\text{FS}} / 2^{B-1}$, quantizer variance is $\sigma_q^2 = \Delta^2 / 12$; finite-sample variance intervals use Gaussian, Hoeffding, or Empirical-Bernstein concentration bounds.
