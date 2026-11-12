# `packages/capture/src/metrics/` — 4-Metric Perceptual Certification Guide

Implements the 4-metric psychoacoustic certification suite (`bESR`, 5-scale 32-probe `MR-IIR`, 33-band `1/3`-octave `LTAS`, `SegESR`) and 4-tier non-regression verification (`Lite`, `Full`, `Pro`, `Ultra`).

## Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../metrics.rs`](../metrics.rs) | Degree-7 Bézout 3-band sum-denominator `bESR`, per-tier noise-limited qualification (`15..38 dB` band SNR), and 5-row rectangular interval audit | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`companion.rs`](companion.rs) | 5-scale 32-probe `MR-IIR`, 33-band `1/3`-octave Welch `LTAS` (`0.75 / 0.35 / 0.15 / 0.10 dB` tier limits), and duration-weighted `SegESR` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`certification.rs`](certification.rs) | Quiet/muted stem classification, `PerceptualCertificationSummary`, `certify_4metric_with_bit_depth`, and `verify_monotone_4tier_summaries` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
