# Contributing to `dnam-fixtures` (`packages/fixtures`)

Before editing `packages/fixtures/`, review the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md), [`packages/fixtures/AGENTS.md`](AGENTS.md), and [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md).

---

## 1. Architectural & Safety Invariants

1. **100% Safe Rust (`#![forbid(unsafe_code)]`):**
   - `packages/fixtures/src/lib.rs` and every submodule must forbid `unsafe_code`. Never introduce `#[allow(unsafe_code)]` or raw pointer casts.
2. **Zero Runtime Dependencies (`[dependencies]` Empty):**
   - `dnam-fixtures` is a standalone leaf ground-truth physical audio modeling crate with zero runtime or build `[dependencies]` (no workspace crates, `serde`, or external DSP libraries may be added; only `[dev-dependencies]` for `proptest` property tests is permitted).
3. **Black-Box Ground-Truth Models + Device Format & Control Metadata Only:**
   - Every signal chain (`Chain01SymmetricOd` .. `Chain12FullEndToEndStereoRig`, each offering `with_format(sample_rate_hz, bit_depth)`) and hardware impairment decorator (`AdcClipAndPadDecorator`, `ConverterLoopbackDecorator::with_converter_format`, `MicrophoneCapsuleClipDecorator`, etc.) implements `BlackBoxModel` (`metadata`, `sample_rate_hz`, `bit_depth`, `set_sample_rate_hz`, `set_bit_depth`, `reset`, `num_inputs`, `num_outputs`, `num_controls`, `num_switches`, `process_frame`, `process_block`). Models expose only physical device-intrinsic sample-rate (`8 kHz..192 kHz`), ADC/DAC bit-depth (`FixtureBitDepth`: `Int12`, `Int16`, `Int24`, `Int32`, `Float32`), control/switch metadata (`FixtureDeviceMetadata`, `FixtureControlSpec`, `FixtureSwitchSpec`), and black-box audio I/O.
4. **Submodule Organization & Unit vs. Integration Tests:**
   - Organize signal chains under category submodules (`src/chains/{drives,amps_and_cabs,mod_and_spatial,dynamics_and_pitch,rigs}/<chain>.rs`) and hardware decorators under domain submodules (`src/adversaries/{interface_and_control,thermal_and_noise,mechanical_and_capsule}.rs`).
   - Keep module unit tests inside `#[cfg(test)] mod tests` within each submodule (sharing helpers via `src/test_utils.rs`), and reserve `tests/` strictly for cross-module integration and `proptest!` tests.
   - Never create `mod.rs` files (`ref[repo.code.type-safety-and-imports]`); keep every `.rs` source file strictly `< 900` lines after `cargo fmt`.

---

## 2. Quality Gates

```bash
moon check fixtures
moon run fixtures:coverage
moon check tests
```
