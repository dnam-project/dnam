# `packages/fixtures` (`dnam-fixtures`) — Agent Discovery & Module Map

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before modifying `packages/fixtures/`.

| Module / File | Responsibility | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | `#![forbid(unsafe_code)]` zero-dependency crate root (`BlackBoxModel`, `FixtureDeviceMetadata`, `FixtureBitDepth`, `FixtureControlSpec`, `FixtureSwitchSpec`) | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md) |
| [`src/chains.rs`](src/chains.rs) | `4×` `f64` anti-aliasing `Oversampler4xF64` and multi-rate/multi-bit-depth `Chain-01..12` submodules (`with_format`, `set_sample_rate_hz`, `set_bit_depth`): [`drives.rs`](src/chains/drives.rs) (`Chain-01..03`), [`amps_and_cabs.rs`](src/chains/amps_and_cabs.rs) (`Chain-04..05`), [`mod_and_spatial.rs`](src/chains/mod_and_spatial.rs) (`Chain-06..08`), [`dynamics_and_pitch.rs`](src/chains/dynamics_and_pitch.rs) (`Chain-09..10`), [`rigs.rs`](src/chains/rigs.rs) (`Chain-11..12`) | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md) |
| [`src/adversaries.rs`](src/adversaries.rs) | 14 hardware fault decorators across [`interface_and_control.rs`](src/adversaries/interface_and_control.rs), [`thermal_and_noise.rs`](src/adversaries/thermal_and_noise.rs), and [`mechanical_and_capsule.rs`](src/adversaries/mechanical_and_capsule.rs) | [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| [`src/test_utils.rs`](src/test_utils.rs) | Shared `#[cfg(test)]` unit-test helpers (`sine_wave`, `run_mono_block`, `run_stereo_block`) | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| [`tests/prop_fixtures_invariants.rs`](tests/prop_fixtures_invariants.rs) | Cross-module integration and `proptest!` verification suite | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
