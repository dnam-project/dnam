/**
 * TORA Stage 0–IV, Control Bus, and Tensor Directory interfaces
 * (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Zero runtime JS dependencies; validated by native Rust `serde` (`dnam_engine::schema`).
 *
 * @module
 */

/**
 * Control-Bus configuration (`header.controlBus`).
 */
export interface ControlBusConfig {
  readonly envelopes: number;
  readonly lfoOscillators: number;
  readonly lfoHarmonics: number;
  readonly fmInteractionRank: number;
  readonly modulationBusDim: number;
  readonly headTrustRegionRadius: number;
}

/**
 * Stage 0 8-Port Matrix Router configuration (`header.stage0Router`).
 */
export interface RouterConfig {
  readonly inputPorts: number;
  readonly outputPorts: number;
  readonly subCrossoverHz: number;
}

/**
 * Stage I Pre-Dynamics and Tone Shaping configuration (`header.stage1Pre`).
 */
export interface Stage1PreFxConfig {
  readonly topologySelector: number;
  readonly noiseGateEnabled: boolean;
  readonly gateThresholdDb: number;
}

/**
 * Stage II Commuted Cascade configuration (`header.stage2Cascade`).
 */
export interface Stage2CascadeConfig {
  readonly numStages: number;
  readonly adaaMode: string;
  readonly acCouplingPole: number;
}

/**
 * Stage III Cabinet Early FIR + Kautz Modal Bank configuration (`header.stage3Cabinet`).
 */
export interface Stage3CabinetKautzConfig {
  readonly firTaps: number;
  readonly kautzModals: number;
}

/**
 * Default 16 coprime Schroeder all-pass delay lengths in samples.
 */
export const DEFAULT_ALLPASS_DELAYS: readonly number[] = Object.freeze([
  19, 31, 47, 73, 103, 139, 179, 223, 29, 53, 83, 113, 151, 191, 233, 251,
]);

/**
 * Stage IV Spatial Tank configuration (`header.stage4Tank`).
 */
export interface Stage4SpatialTankConfig {
  readonly delayStorage: string;
  readonly interp: string;
  readonly tzfReferenceTapSamples: number;
  readonly switchXfadeSamples: number;
  readonly allpassDelaysSamples: readonly number[];
  readonly dcBlockerPole: number;
  readonly loopCeiling: number;
}

/**
 * Tensor slice descriptor in `header.tensorDirectory` or `.dnam-ckpt` stem records.
 */
export interface TensorSliceEntry {
  readonly arena: number;
  readonly dtype: string;
  readonly packing: string | null;
  readonly featureDim: number | null;
  readonly packedLen: number | null;
  readonly shape: readonly number[] | null;
  readonly byteOffset: number;
  readonly byteLength: number;
}
