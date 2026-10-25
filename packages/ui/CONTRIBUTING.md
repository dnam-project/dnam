# Contributing to `@dnam-project/ui` (`packages/ui/`)

This guide covers component architecture, Tailwind CSS v4 + `shadcn/ui` (Base UI / `@base-ui/react`) conventions, `DnamAudioHost` abstraction rules, strict TypeScript/Biome checks, and Vitest verification for **`@dnam-project/ui`**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Architecture & Design System Stack

- **Tailwind CSS v4 (`@tailwindcss/vite` + `@astrojs/starlight-tailwind`):** [`src/styles.css`](src/styles.css) defines `@import "tailwindcss";` and `@theme` tokens (`--color-dnam-surface`, `--color-dnam-core1`, `--color-dnam-core2`, `--color-dnam-control`, `--color-dnam-safe`, `--color-dnam-warn`, `--color-dnam-danger`, `--font-sans`, `--font-mono`).
- **`shadcn/ui` (Base UI / `@base-ui/react`) & `lucide-react`:** Base primitives in [`src/components/ui/`](src/components/ui/index.ts) follow [`components.json`](components.json) (`base-nova` style, `@base-ui/react` primitives with `render` composition, `class-variance-authority`, `clsx`, `tailwind-merge`).
- **Host-Agnostic Audio Bridge (`DnamAudioHost`):** Components in `src/components/rack/` and `src/components/controls/` communicate exclusively through `useDnamAudioHost()` so the same components drive WebAudio `AudioWorklet` in `packages/web` and native Rust SIMD (`ARM SME` / `NEON` / `AVX2+FMA`) in `packages/studio`.
- **React Cosmos Workbench:** Develop and preview components in isolation via [`cosmos.config.json`](cosmos.config.json), [`src/cosmos.decorator.tsx`](src/cosmos.decorator.tsx), and co-located domain fixtures (`src/components/**/*.fixture.tsx`) (`pnpm ui:dev` / `cosmos:export`).
- **Vitest Browser Mode + ARIA Snapshots + `axe-core`:** Co-located unit/component tests (`src/**/*.test.ts(x)`) and cross-feature integration tests (`tests/*.integration.test.tsx`) run in headless Chromium via `@vitest/browser-playwright` and `vitest-browser-react`, asserting semantic accessibility trees via `toMatchAriaInlineSnapshot` and WCAG 2.1 AA rules via `expectZeroAxeViolations` ([`tests/setup.ts`](tests/setup.ts)).

---

## 2. Verification Commands

```bash
pnpm --filter @dnam-project/ui dev
pnpm --filter @dnam-project/ui lint
pnpm --filter @dnam-project/ui format:check
pnpm --filter @dnam-project/ui typecheck
pnpm --filter @dnam-project/ui coverage
pnpm --filter @dnam-project/ui bench
```
