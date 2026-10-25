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
}

export interface DnamAudioHost {
  readonly kind: "web-worklet" | "tauri-native" | "mock";
  readonly backend: DnamBackendAcceleration;
  getParameter(id: string): number;
  setParameter(id: string, value: number): void;
  getSwitchIndex(id: string): number;
  setSwitchIndex(id: string, index: number): void;
  getTelemetry(): DnamTelemetrySnapshot;
  subscribe(listener: (snapshot: DnamTelemetrySnapshot) => void): () => void;
}
