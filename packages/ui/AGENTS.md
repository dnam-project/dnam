# `packages/ui` — Unified Design System (`@dnam-project/ui`) Routing Guide

**`@dnam-project/ui`** provides the shared React 19, Tailwind CSS v4 (`@tailwindcss/vite`), `shadcn/ui` (Base UI / `@base-ui/react`), and `lucide-react` component library used across `packages/studio` and `packages/web`.

## 1. Key Modules

- [`src/styles.css`](src/styles.css) & [`src/tokens.ts`](src/tokens.ts): Tailwind CSS v4 `@theme` design tokens, Starlight variables, and typed palette constants.
- [`src/lib/utils.ts`](src/lib/utils.ts): `cn()` (`clsx` + `tailwind-merge`) and numerical formatting utilities.
- [`src/host/`](src/host/index.ts): `DnamAudioHost` interface ([`types.ts`](src/host/types.ts)), `<DnamAudioHostProvider>` / `useDnamAudioHost()` ([`context.tsx`](src/host/context.tsx)), and `WebAudioWorkletHost` ([`web_worklet_host.ts`](src/host/web_worklet_host.ts)).
- [`src/components/ui/`](src/components/ui/index.ts): `shadcn/ui` Base UI primitives (`Button`, `Badge`, `Card`, `Tabs`, `Progress`).
- [`src/components/controls/`](src/components/controls/index.ts): `RotaryKnob`, `DiscreteSwitch`, `AccelerationBadge`, `StabilityAuditBadge`, and [`geometry.ts`](src/components/controls/geometry.ts).
- [`src/components/rack/`](src/components/rack/index.ts) & [`src/components/workflows/`](src/components/workflows/index.ts): `DnamPluginRack`, `TriPhysicsTelemetry`, `CapturePlanPanel`, and `TrainingTimelinePanel`.
- [`src/components/visualizers/`](src/components/visualizers/index.ts): All 7 interactive Tri-Physics visualizers (`PolyAdaaVisualizer`, `RgLruPoleVisualizer`, `ModulatedUfdnVisualizer`, `ControlBusVisualizer`, `KroneckerResumeVisualizer`, `ClampAuditVisualizer`, `ZohRescalerVisualizer`).

## 2. Workbench & Verification

- **React Cosmos Workbench:** [`cosmos.config.json`](cosmos.config.json), [`src/cosmos.decorator.tsx`](src/cosmos.decorator.tsx), and co-located `src/components/**/*.fixture.tsx` (`pnpm ui:dev` on `http://localhost:5005`).
- **Vitest Browser Mode (`@vitest/browser-playwright` + `vitest-browser-react`):** Co-located unit/component tests (`src/**/*.test.ts(x)`), benchmarks (`src/**/*.bench.ts`), and integration tests (`tests/integration/*.test.tsx`) in headless Chromium with ARIA snapshots, `axe-core` ([`tests/setup.ts`](tests/setup.ts)), and V8 coverage.

```bash
pnpm --filter @dnam-project/ui lint && pnpm --filter @dnam-project/ui typecheck && pnpm --filter @dnam-project/ui coverage && pnpm --filter @dnam-project/ui bench
```
