# `dnam-fixtures` — Pure-Rust Zero-Dependency Black-Box Audio Signal-Chain Models & Hardware Impairment Decorators

`dnam-fixtures` (`packages/fixtures`) is a standalone, zero-dependency pure-Rust (`#![forbid(unsafe_code)]`) leaf crate providing black-box ground-truth physical audio signal-chain models (`Chain-01..12`), device-intrinsic format (`sample_rate_hz`, `FixtureBitDepth`: `Int12`/`Int16`/`Int24`/`Int32`/`Float32`) and control/switch metadata (`FixtureDeviceMetadata`), and composable hardware/interface fault decorators ([`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)).

---

## 1. Crate Architecture

| Module | Scope |
| :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | Defines the `BlackBoxModel` (`AudioFixture`) trait (`metadata`, `sample_rate_hz`, `bit_depth`, `set_sample_rate_hz`, `set_bit_depth`, `reset`, `num_inputs`, `num_outputs`, `num_controls`, `num_switches`, `process_frame`, `process_block`) and device-intrinsic metadata (`FixtureDeviceMetadata`, `FixtureBitDepth`, `FixtureControlSpec`, `FixtureSwitchSpec`) with zero external crate dependencies. |
| [`src/chains.rs`](src/chains.rs) | Multi-rate `Oversampler4xF64` (`4×` pure `f64` sinc-droop-compensated ZOH + Butterworth TPT SVF engine across `8 kHz`..`192 kHz` base device rates) and five domain submodules implementing all twelve physical signal chains (`with_format`): [`drives`](src/chains/drives.rs) (`Chain-01..03`), [`amps_and_cabs`](src/chains/amps_and_cabs.rs) (`Chain-04..05`), [`mod_and_spatial`](src/chains/mod_and_spatial.rs) (`Chain-06..08`), [`dynamics_and_pitch`](src/chains/dynamics_and_pitch.rs) (`Chain-09..10`), and [`rigs`](src/chains/rigs.rs) (`Chain-11..12`). |
| [`src/adversaries.rs`](src/adversaries.rs) | Fourteen composable black-box hardware/interface fault decorators organized into [`interface_and_control`](src/adversaries/interface_and_control.rs), [`thermal_and_noise`](src/adversaries/thermal_and_noise.rs), and [`mechanical_and_capsule`](src/adversaries/mechanical_and_capsule.rs), wrapping any `M: BlackBoxModel` and forwarding its `FixtureDeviceMetadata`. |
| [`src/test_utils.rs`](src/test_utils.rs) | Shared `#[cfg(test)]` helper module used by in-module unit tests across `src/chains/**` and `src/adversaries/**`. |
| [`tests/prop_fixtures_invariants.rs`](tests/prop_fixtures_invariants.rs) | Cross-module integration tests and `proptest!` property suites composing chains, metadata sweeps, and stacked impairment decorators. |

---

## 2. Quick Start & Verification Commands

```bash
# Run formatting, clippy lints, typecheck, build, and tests via moon
moon check fixtures

# Run >= 95% line coverage gate
moon run fixtures:coverage
```
