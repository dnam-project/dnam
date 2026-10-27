# `packages/studio` — Unified Rust/Node `dnam` CLI & Tauri v2 Desktop Studio Routing Guide

**`dnam-studio` (`@dnam-project/studio`)** is the single owner of DNAM workspace management (`dnam.toml`, `models/<id>/model.toml`), Star/Switch/Corner capture schedules, 16/24/32-bit PCM & 32-bit IEEE-754 float WAV synthesis/alignment, `uv` Python trainer orchestration (`dnam-train`), and the Tauri v2 Desktop Studio GUI.

## 1. Key Modules

- [`rust/project.rs`](rust/project.rs) & [`rust/project/`](rust/project/manifest.rs): `DnamProject`, `DnamProjectManifest`, `ModelManifest`, and `build_capture_plan` (Star, Switch $\times$ Gain, Dual-Engine Morph $2\times 2$, and Parametric EQ $2\times 2$ corners).
- [`rust/capture.rs`](rust/capture.rs) & [`rust/capture/`](rust/capture/wav.rs): 16/24/32-bit PCM & 32-bit float WAV I/O, excitation probe synthesis, sub-sample lag/SNR/USB-jitter alignment (with clipped fuzz proxy fallback), and `software`/`usb`/`anagram` capture backends.
- [`rust/runner.rs`](rust/runner.rs): Native hardware SIMD (`dnam-engine`) `.dnam` container inspector and 24-bit WAV renderer.
- [`rust/trainer.rs`](rust/trainer.rs) & [`rust/trainer/`](rust/trainer/env_manager.rs): `TrainerEnvManager` (`uv` auto-discovery & `<config_dir>/trainer-venv`) and `TrainerBridge` (`--request-json` + JSONL stream).
- [`rust/ipc.rs`](rust/ipc.rs) & [`rust/cli.rs`](rust/cli.rs): Tauri v2 `StudioIpcHandler` and `clap` v4 `dnam` CLI router.
- [`src/`](src/index.ts) & [`tests/`](tests/): TypeScript SDK ([`binary.ts`](src/binary.ts), [`ipc.ts`](src/ipc.ts), [`host.ts`](src/host.ts) `TauriNativeAudioHost`), Tauri v2 React 19 UI ([`src/ui/StudioApp.tsx`](src/ui/StudioApp.tsx), [`src/ui/studio_store.ts`](src/ui/studio_store.ts)), co-located `src/**/*.test.ts(x)` unit tests, [`tests/integration/`](tests/integration/) Vitest Browser integration tests, and [`tests/e2e/studio_desktop.spec.ts`](tests/e2e/studio_desktop.spec.ts) (`@playwright/test`).

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) for Rust 2024 module layout, Tauri v2 IPC isolation rules, and `moon check studio` / `moon run studio:test-e2e` verification commands.
