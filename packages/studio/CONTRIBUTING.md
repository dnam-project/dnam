# Contributing to `@dnam-project/studio` / `dnam-studio` (`packages/studio/`)

This guide covers the Rust 2024 CLI/Capture/Trainer-Bridge core (`rust/`), the Node.js binary wrapper (`bin/dnam.mjs`), and the Tauri v2 React 19 Desktop Studio (`src/`) using `@dnam-project/ui` (Tailwind CSS v4 + `shadcn/ui` Base UI + `lucide-react`).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Architecture Invariants

- **Rust 2024 Non-`mod.rs` Module Roots:** Use `rust/project.rs` + `rust/project/*.rs`, `rust/capture.rs` + `rust/capture/*.rs`, `rust/trainer.rs` + `rust/trainer/*.rs`, and `rust/cli.rs` + `rust/cli/*.rs` with `#![forbid(unsafe_code)]`.
- **Single Ownership of Workspace & Capture:** `packages/studio` owns `dnam.toml`, `models/<id>/model.toml`, and 16/24/32-bit PCM & unclipped 32-bit IEEE-754 float WAV capture/alignment; it invokes `packages/trainer` (`dnam-train`) via `--request-json` and consumes live JSONL `TrainingEvent` messages.
- **Zero Control-Name Heuristics in 30-Second Universal Excitation Synthesis (`rust/capture/excitation.rs`):** Never use control-name or manifest heuristics (such as checking whether a model has `cutoff` or `sensitivity` controls) or pre-generated length branching to tailor or switch the dry excitation signal. Heuristic branching fails whenever a user captures a fixed/knobless rig or a full multi-effect signal chain (e.g., compressor + delay + envelope filter + fuzz + amp + cab sim). Always use the single canonical 30.0-second (`1,440,000`-sample @ `48 kHz`, `DEFAULT_HARDWARE_EXCITATION_SEC = 30`) 5-section universal [`generate_excitation_signal_with_profile`](rust/capture/excitation.rs) generator (`27.0s` active prefix + `3.0s` baseline tail), while `cpal_play_and_record` (`should_extend_live_tail_window`) dynamically extends live hardware recording in `1.0s` silent-input increments (up to `+15.0s` extra via `MAX_DYNAMIC_TAIL_EXTRA_SEC`) only on stems whose wet return is still decaying above the pre-roll noise floor.
- **Native Hardware Acceleration in Tauri Studio:** [`src/host.ts`](src/host.ts) (`TauriNativeAudioHost`) implements `@dnam-project/ui`'s `DnamAudioHost` over native Rust SIMD (`ARM SME`, `ARM64 NEON`, `x86_64 AVX2+FMA`).

---

## 2. Local Development & Verification Commands

```bash
# Run the `dnam` CLI (`init`, `model`, `capture`, `train`, `cache`, `inspect`, `studio`)
moon run studio:cli -- --help

# Launch the Studio UI Vite dev server (http://localhost:1420)
moon run studio:dev

# Launch the native Tauri v2 Desktop Studio app
moon run studio:tauri-dev

# Run studio quality gates, unit/integration tests, and Playwright E2E tests
moon check studio
moon run studio:test-e2e
moon run studio:coverage
```
