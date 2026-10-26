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

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for Tailwind CSS v4 tokens, `shadcn/ui` Base UI conventions, React Cosmos workbench (`cosmos.config.json`), Vitest Browser Mode + `axe-core` accessibility verification, and `moon check ui` commands.
