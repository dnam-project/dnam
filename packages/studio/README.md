# `@dnam-project/studio` (`dnam-studio`) — Tauri v2 Desktop Studio

Standalone Tauri v2 Desktop Studio for the **Dynamic Neural Audio Modeler (DNAM)**.

## Highlights

- **Tauri v2 Desktop Studio (`dnam-studio`):** Built on `@dnam-project/ui` (React 19, Tailwind CSS v4, `shadcn/ui` Base UI, `lucide-react`), `dnam-capture` (`.dnam-proj` single-file project container), and `TauriNativeAudioHost` backed by native Rust `dnam-engine` SIMD (`ARM SME`, `ARM64 NEON`, `x86_64 AVX2+FMA`).
- **Tier 1 & Tier 2 Official Tauri v2 Plugins:**
  - **Tier 1:** `dialog` (native `.dnam`/`.dnam-proj`/`.wav` file pickers), `opener` (reveal exported `.dnam`/`.wav` in Finder/Explorer & open HTML convergence reports), `window-state` (multi-monitor window geometry persistence).
  - **Tier 2:** `single-instance` (exclusive USB audio interface protection), `deep-link` (`dnam://` Marketplace $\to$ Studio model handoff), `notification` (OS toasts on capture/fit completion), and `updater` + `process` (`Ed25519`-signed OTA desktop updates).
- **Tauri v2 IPC Isolation Pattern (`isolation/index.html` + `tauriIsolationHook`):** Enforces command allowlisting, blocks `..` path traversal and `NaN`/`±Infinity` parameters, and encrypts IPC payloads with `AES-GCM` (`SubtleCrypto`) before forwarding to Rust.
