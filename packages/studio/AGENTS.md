# `packages/studio` — Standalone Tauri v2 Desktop Studio Routing Guide

**`dnam-studio` (`@dnam-project/studio`)** owns the standalone Tauri v2 Desktop Studio GUI, native OS audio device enumeration (`cpal`), wait-free SPSC ring-buffer audio I/O, and the isolated IPC bridge over `dnam-engine` (`.dnam`) and `dnam-capture` (`.dnam-proj`).

## 1. Key Modules

- [`rust/capture.rs`](rust/capture.rs) & [`rust/capture/backends.rs`](rust/capture/backends.rs): `AtomicF32RingBuffer` and `cpal` native audio device enumeration (`CoreAudio` / `ALSA` / `WASAPI`).
- [`rust/ipc.rs`](rust/ipc.rs) & [`rust/main.rs`](rust/main.rs): Tauri v2 `StudioIpcHandler` (`.dnam-proj` inspection, isolated IPC allowlisting, `dnam://` deep-link routing) and standalone `dnam-studio` desktop binary.
- [`src/`](src/index.ts): TypeScript IPC client ([`ipc.ts`](src/ipc.ts)), `TauriNativeAudioHost` ([`host.ts`](src/host.ts)), and Tauri v2 React 19 UI ([`src/ui/StudioApp.tsx`](src/ui/StudioApp.tsx), [`src/ui/studio_store.ts`](src/ui/studio_store.ts)).

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) for Rust 2024 module layout, Tauri v2 IPC isolation rules, and `moon check studio` / `moon run studio:test-e2e` verification commands.
