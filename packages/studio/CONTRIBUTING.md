# Contributing to `@dnam-project/studio` / `dnam-studio` (`packages/studio/`)

This guide covers the standalone Tauri v2 Desktop Studio Rust core (`rust/`) and React 19 UI (`src/`) built on `@dnam-project/ui` (Tailwind CSS v4 + `shadcn/ui` Base UI + `lucide-react`), `dnam-engine`, and `dnam-capture`.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Architecture Invariants

- **Rust 2024 Non-`mod.rs` Module Roots:** Use `rust/capture.rs` + `rust/capture/backends.rs`, `rust/ipc.rs`, and `rust/main.rs` with `#![forbid(unsafe_code)]`.
- **Single-File `.dnam-proj` & Native Audio Separation:** Capture math, WAV I/O, and `.dnam-proj` (`DNPJ`) container persistence live in `dnam-capture`; `packages/studio` owns only native OS audio device enumeration (`cpal`), `AtomicF32RingBuffer`, and the Tauri v2 Desktop Studio host.
- **Native Hardware Acceleration in Tauri Studio:** [`src/host.ts`](src/host.ts) (`TauriNativeAudioHost`) implements `@dnam-project/ui`'s `DnamAudioHost` over native Rust SIMD (`ARM SME`, `ARM64 NEON`, `x86_64 AVX2+FMA`).

---

## 2. Local Development & Verification Commands

```bash
# Launch the Studio UI Vite dev server (http://localhost:1420)
moon run studio:dev

# Launch the native Tauri v2 Desktop Studio app
moon run studio:tauri-dev

# Run studio quality gates, unit/integration tests, and Playwright E2E tests
moon check studio
moon run studio:test-e2e
moon run studio:coverage
```
