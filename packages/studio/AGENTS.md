# `packages/studio` — Unified Rust/Node `dnam` CLI & Tauri v2 Desktop Studio Routing Guide

**`dnam-studio` (`@dnam-project/studio`)** is the single owner of DNAM workspace management (`dnam.toml`, `models/<id>/model.toml`), Star/Switch/Corner capture schedules, 24-bit WAV synthesis/alignment, `uv` Python trainer orchestration (`dnam-train`), and the Tauri v2 Desktop Studio GUI.

## 1. Key Modules

- [`rust/project.rs`](rust/project.rs) & [`rust/project/`](rust/project/manifest.rs): `DnamProject`, `DnamProjectManifest`, `ModelManifest`, and `build_capture_plan`.
- [`rust/capture.rs`](rust/capture.rs) & [`rust/capture/`](rust/capture/wav.rs): 24-bit WAV I/O, excitation probe synthesis, sub-sample lag/SNR alignment, and `software`/`usb`/`anagram` capture backends.
- [`rust/runner.rs`](rust/runner.rs): Native hardware SIMD (`dnam-engine`) `.dnam` container inspector and 24-bit WAV renderer.
- [`rust/trainer.rs`](rust/trainer.rs) & [`rust/trainer/`](rust/trainer/env_manager.rs): `TrainerEnvManager` (`uv` auto-discovery & `<config_dir>/trainer-venv`) and `TrainerBridge` (`--request-json` + JSONL stream).
- [`rust/ipc.rs`](rust/ipc.rs) & [`rust/cli.rs`](rust/cli.rs): Tauri v2 `StudioIpcHandler` and `clap` v4 `dnam` CLI router.
- [`src/`](src/index.ts) & [`tests/`](tests/): TypeScript SDK ([`binary.ts`](src/binary.ts), [`ipc.ts`](src/ipc.ts), [`host.ts`](src/host.ts) `TauriNativeAudioHost`), Tauri v2 React 19 UI ([`src/ui/StudioApp.tsx`](src/ui/StudioApp.tsx), [`src/ui/studio_store.ts`](src/ui/studio_store.ts)), co-located `src/**/*.test.ts(x)` unit tests, [`tests/integration/`](tests/integration/) Vitest Browser integration tests, and [`tests/e2e/studio_desktop.spec.ts`](tests/e2e/studio_desktop.spec.ts) (`@playwright/test`).

## 2. Verification

```bash
cargo test --package dnam-studio && pnpm --filter @dnam-project/studio lint && pnpm --filter @dnam-project/studio typecheck && pnpm --filter @dnam-project/studio coverage
```
