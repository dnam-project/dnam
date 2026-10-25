# `@dnam-project/studio` (`dnam-studio`) — DNAM CLI & Tauri v2 Desktop Studio

Unified Rust + Node.js/TypeScript CLI (`dnam`) and Tauri v2 Desktop Studio for the **Dynamic Neural Audio Modeler (DNAM / "Tri-Physics")**.

## Highlights

- **`dnam` CLI:** Workspace initialization (`dnam init`), model & control definition (`dnam model`, `dnam control`), hardware/software re-amp capture (`dnam capture`), and `uv`-managed Python trainer execution (`dnam train`, `dnam trainer`).
- **Tauri v2 Desktop Studio (`dnam studio`):** Built on `@dnam-project/ui` (React 19, Tailwind CSS v4, `shadcn/ui` Base UI, `lucide-react`) and `TauriNativeAudioHost` backed by native Rust `dnam-engine` SIMD (`ARM SME`, `ARM64 NEON`, `x86_64 AVX2+FMA`).
- **Tier 1 & Tier 2 Official Tauri v2 Plugins:**
  - **Tier 1:** `dialog` (native `.dnam`/`.dnam-ckpt`/`.wav` file & workspace folder pickers), `opener` (reveal exported `.dnam`/`.wav` in Finder/Explorer & open Altair HTML convergence reports), `window-state` (multi-monitor window geometry persistence).
  - **Tier 2:** `single-instance` (exclusive USB audio interface protection), `deep-link` (`dnam://` Marketplace $\to$ Studio model handoff), `notification` (OS toasts on capture/training completion), and `updater` + `process` (`Ed25519`-signed OTA desktop updates).
- **Tauri v2 IPC Isolation Pattern (`isolation/index.html` + `tauriIsolationHook`):** Enforces command allowlisting, blocks `..` path traversal and `NaN`/`±Infinity` parameters, and encrypts IPC payloads with `AES-GCM` (`SubtleCrypto`) before forwarding to Rust.
