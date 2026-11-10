# `packages/studio` — Unified Rust/Node `dnam` CLI & Tauri v2 Desktop Studio Routing Guide

**`dnam-studio` (`@dnam-project/studio`)** owns DNAM workspace management (`dnam.toml`, `models/<id>/model.toml`), 12/16/24/32-bit PCM & 32-bit IEEE-754 float WAV I/O, native SIMD `.dnam`/`.dnam-ckpt` inspection and rendering, and the Tauri v2 Desktop Studio GUI.

## 1. Key Modules

- [`rust/project.rs`](rust/project.rs) & [`rust/project/`](rust/project/manifest.rs): `DnamProject`, `DnamProjectManifest`, and `ModelManifest`.
- [`rust/capture.rs`](rust/capture.rs) & [`rust/capture/`](rust/capture/wav.rs): 12/16/24/32-bit PCM & 32-bit float WAV I/O, `AtomicF32RingBuffer`, and `cpal` audio device enumeration.
- [`rust/runner.rs`](rust/runner.rs): Native hardware SIMD (`dnam-engine`) `.dnam`/`.dnam-ckpt` container inspector, tier stripper, and 32-bit float WAV renderer.
- [`rust/ipc.rs`](rust/ipc.rs) & [`rust/cli.rs`](rust/cli.rs): Tauri v2 `StudioIpcHandler` and `clap` v4 `dnam` CLI router (`workspace_cmd`, `format`).
- [`src/`](src/index.ts) & [`tests/`](tests/): TypeScript SDK ([`binary.ts`](src/binary.ts), [`ipc.ts`](src/ipc.ts), [`host.ts`](src/host.ts) `TauriNativeAudioHost`), Tauri v2 React 19 UI ([`src/ui/StudioApp.tsx`](src/ui/StudioApp.tsx), [`src/ui/studio_store.ts`](src/ui/studio_store.ts)), co-located `src/**/*.test.ts(x)` unit tests, [`tests/integration/`](tests/integration/) Vitest Browser integration tests, and [`tests/e2e/studio_desktop.spec.ts`](tests/e2e/studio_desktop.spec.ts) (`@playwright/test`).

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) for Rust 2024 module layout, Tauri v2 IPC isolation rules, and `moon check studio` / `moon run studio:test-e2e` verification commands.
