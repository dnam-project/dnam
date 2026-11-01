/**
 * Pluggable real-time audio & telemetry bridge shared across:
 * 1. `packages/web` (`WebAudioWorkletHost` -> WebAssembly SIMD128 `AudioWorklet`)
 * 2. `packages/studio` (`TauriNativeAudioHost` -> Native Rust `dnam-engine` ARM SME / NEON / AVX2+FMA)
 */

export type DnamBackendAcceleration =
  | "arm-sme"
  | "arm64-neon"
  | "x86-avx2-fma"
  | "wasm-simd128"
  | "scalar-reference";

export type DnamModelTier = "lite" | "full" | "pro" | "ultra";

export const DNAM_MODEL_TIERS: readonly DnamModelTier[] = ["lite", "full", "pro", "ultra"] as const;

export const DNAM_MODEL_TIER_LABELS: Readonly<Record<DnamModelTier, string>> = {
  lite: "Lite",
  full: "Full",
  pro: "Pro",
  ultra: "Ultra",
} as const;

export function tierOrderIndex(tier: DnamModelTier): number {
  return DNAM_MODEL_TIERS.indexOf(tier);
}

export function clampTierToMaxSupported(
  tier: DnamModelTier,
  maxSupported: DnamModelTier,
): DnamModelTier {
  return tierOrderIndex(tier) <= tierOrderIndex(maxSupported) ? tier : maxSupported;
}

export interface DnamTelemetrySnapshot {
  readonly sampleRateHz: number;
  readonly controlRateHz: number;
  readonly backend: DnamBackendAcceleration;
  readonly dspLoadPercent: number;
  readonly inputRmsDbL: number;
  readonly inputRmsDbR: number;
  readonly outputRmsDbL: number;
  readonly outputRmsDbR: number;
  readonly maxRgLruRadius: number;
  readonly ufdnLyapunovMargin: number;
  readonly mahalanobisR2: number;
  readonly activeTrustRegionAlpha: number;
  readonly bypassActive: boolean;
  readonly activeTier?: DnamModelTier | undefined;
  readonly effectiveTier?: DnamModelTier | undefined;
  readonly maxSupportedTier?: DnamModelTier | undefined;
  readonly offlineBounce?: boolean | undefined;
  readonly maximizeTierOnBounce?: boolean | undefined;
}

export interface DnamAudioHost {
  readonly kind: "web-worklet" | "tauri-native" | "mock";
  readonly backend: DnamBackendAcceleration;
  getParameter(id: string): number;
  setParameter(id: string, value: number): void;
  getSwitchIndex(id: string): number;
  setSwitchIndex(id: string, index: number): void;
  getTier?(): DnamModelTier;
  setTier?(tier: DnamModelTier): void;
  getMaxSupportedTier?(): DnamModelTier;
  setMaxSupportedTier?(tier: DnamModelTier): void;
  getOfflineBounce?(): boolean;
  setOfflineBounce?(enabled: boolean): void;
  getMaximizeTierOnBounce?(): boolean;
  setMaximizeTierOnBounce?(enabled: boolean): void;
  getTelemetry(): DnamTelemetrySnapshot;
  subscribe(listener: (snapshot: DnamTelemetrySnapshot) => void): () => void;
}
