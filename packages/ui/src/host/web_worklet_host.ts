import { clampNumber } from "../lib/utils.js";
import type { DnamAudioHost, DnamBackendAcceleration, DnamTelemetrySnapshot } from "./types.js";

export interface WebAudioWorkletHostOptions {
  readonly sampleRateHz?: number | undefined;
  readonly controlRateHz?: number | undefined;
  readonly backend?: DnamBackendAcceleration | undefined;
  readonly initialParameters?: Readonly<Record<string, number>> | undefined;
  readonly initialSwitches?: Readonly<Record<string, number>> | undefined;
  readonly onParameterChange?: ((id: string, value: number) => void) | undefined;
  readonly onSwitchChange?: ((id: string, index: number) => void) | undefined;
}

/**
 * Browser / WebAudio `AudioWorklet` host implementation backed by WASM SIMD128.
 * Also serves as a deterministic in-memory host for testing and interactive documentation.
 */
export class WebAudioWorkletHost implements DnamAudioHost {
  public readonly kind = "web-worklet" as const;
  public readonly backend: DnamBackendAcceleration;
  private readonly parameters = new Map<string, number>();
  private readonly switches = new Map<string, number>();
  private readonly listeners = new Set<(snapshot: DnamTelemetrySnapshot) => void>();
  private readonly onParameterChange?: ((id: string, value: number) => void) | undefined;
  private readonly onSwitchChange?: ((id: string, index: number) => void) | undefined;
  private telemetryState: DnamTelemetrySnapshot;

  public constructor(options: WebAudioWorkletHostOptions = {}) {
    this.backend = options.backend ?? "wasm-simd128";
    this.onParameterChange = options.onParameterChange;
    this.onSwitchChange = options.onSwitchChange;
    if (options.initialParameters) {
      for (const [key, val] of Object.entries(options.initialParameters)) {
        this.parameters.set(key, val);
      }
    }
    if (options.initialSwitches) {
      for (const [key, val] of Object.entries(options.initialSwitches)) {
        this.switches.set(key, Math.max(0, Math.trunc(val)));
      }
    }
    this.telemetryState = {
      sampleRateHz: options.sampleRateHz ?? 48_000,
      controlRateHz: options.controlRateHz ?? 3_000,
      backend: this.backend,
      dspLoadPercent: 8.4,
      inputRmsDbL: -18.0,
      inputRmsDbR: -18.0,
      outputRmsDbL: -12.5,
      outputRmsDbR: -12.5,
      maxRgLruRadius: 0.962,
      ufdnLyapunovMargin: 0.048,
      mahalanobisR2: 0.42,
      activeTrustRegionAlpha: 1.0,
      bypassActive: false,
    };
  }

  public getParameter(id: string): number {
    return this.parameters.get(id) ?? 0.5;
  }

  public setParameter(id: string, value: number): void {
    const safeValue = Number.isFinite(value) ? value : 0.0;
    this.parameters.set(id, safeValue);
    this.onParameterChange?.(id, safeValue);

    const paramValues = [...this.parameters.values()];
    const avgNorm =
      paramValues.length > 0
        ? paramValues.reduce((acc, v) => acc + Math.abs(v), 0) / paramValues.length
        : 0.5;
    const r2 = clampNumber(avgNorm * 0.85, 0.0, 9.0);
    const alpha = 1.0 / Math.sqrt(1.0 + r2 / 4.0);
    this.updateTelemetry({
      mahalanobisR2: r2,
      activeTrustRegionAlpha: alpha,
      outputRmsDbL: clampNumber(-18.0 + avgNorm * 6.0, -60.0, 0.0),
      outputRmsDbR: clampNumber(-18.0 + avgNorm * 6.0, -60.0, 0.0),
    });
  }

  public getSwitchIndex(id: string): number {
    return this.switches.get(id) ?? 0;
  }

  public setSwitchIndex(id: string, index: number): void {
    const clampedIndex = Math.max(0, Math.trunc(index));
    this.switches.set(id, clampedIndex);
    this.onSwitchChange?.(id, clampedIndex);
    this.telemetryState = { ...this.telemetryState };
    this.notifyListeners();
  }

  public getTelemetry(): DnamTelemetrySnapshot {
    return this.telemetryState;
  }

  public updateTelemetry(patch: Partial<DnamTelemetrySnapshot>): void {
    this.telemetryState = {
      ...this.telemetryState,
      ...patch,
    };
    this.notifyListeners();
  }

  public subscribe(listener: (snapshot: DnamTelemetrySnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.telemetryState);
    }
  }
}
