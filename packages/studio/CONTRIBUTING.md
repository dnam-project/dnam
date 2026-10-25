# Contributing to `@dnam-project/studio` / `dnam-studio` (`packages/studio/`)

This guide covers the Rust 2024 CLI/Capture/Trainer-Bridge core (`rust/`), the Node.js binary wrapper (`bin/dnam.mjs`), and the Tauri v2 React 19 Desktop Studio (`src/`) using `@dnam-project/ui` (Tailwind CSS v4 + `shadcn/ui` Base UI + `lucide-react`).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Architecture Invariants

- **Rust 2024 Non-`mod.rs` Module Roots:** Use `rust/project.rs` + `rust/project/*.rs`, `rust/capture.rs` + `rust/capture/*.rs`, `rust/trainer.rs` + `rust/trainer/*.rs`, and `rust/cli.rs` + `rust/cli/*.rs` with `#![forbid(unsafe_code)]`.
- **Single Ownership of Workspace & Capture:** `packages/studio` owns `dnam.toml`, `models/<id>/model.toml`, and 24-bit WAV capture/alignment; it invokes `packages/trainer` (`dnam-train`) via `--request-json` and consumes live JSONL `TrainingEvent` messages.
- **Native Hardware Acceleration in Tauri Studio:** [`src/host.ts`](src/host.ts) (`TauriNativeAudioHost`) implements `@dnam-project/ui`'s `DnamAudioHost` over native Rust SIMD (`ARM SME`, `ARM64 NEON`, `x86_64 AVX2+FMA`).

---

## 2. Local Development & Verification Commands

```bash
# Run the `dnam` CLI (`init`, `model`, `capture`, `train`, `inspect`, `studio`)
pnpm dnam --help        # or: pnpm moon run studio:cli -- --help

# Launch the Studio UI Vite dev server (http://localhost:1420)
pnpm studio:dev         # or: pnpm moon run studio:dev

# Launch the native Tauri v2 Desktop Studio app
pnpm studio:tauri       # or: pnpm moon run studio:tauri-dev

# Run studio quality gates, unit/integration tests, and Playwright E2E tests
pnpm moon check studio
pnpm moon run studio:test-e2e
pnpm moon run studio:coverage
```
