/**
 * Unified backend driver interface (`DnamDriver`) implemented by both:
 * 1. `NativeDriver` (Node-API v8 zero-copy host addon `dnam_js.<platform>-<arch>.node`)
 * 2. `WasmDriver` (WebAssembly SIMD128 `dnam_engine.wasm` for Browser, WebAudio Worklet & Node differential testing)
 */

import type { PreambleSnapshot, TelemetrySnapshot, ZohParamsSnapshot } from "../schema/runtime.js";
import type { DnamProtectionProfile } from "../schema/signatures.js";

export type BackendChoice =
  | "auto"
  | "native"
  | "wasm"
  | "wasm_simd128"
  | "scalar"
  | "arm_neon"
  | "arm_sme"
  | "x86_simd";

export interface DnamEngineHandle {
  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void;
  readonly backend: string;
  activeTier: number;
  sampleRateHz: number;
  setControlValue(ctrlIdx: number, value: number): void;
  setControlImmediate(ctrlIdx: number, value: number): void;
  reset(): void;
  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void;
  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void;
  processMono(monoIn: Float32Array, monoOut: Float32Array, bufferSize?: number): void;
  processAsync(
    leftIn: Float32Array,
    rightIn?: Float32Array,
    bufferSize?: number,
  ): Promise<[Float32Array, Float32Array]>;
  telemetry(): TelemetrySnapshot;
  dispose(): void;
}

export interface RawClampResult {
  clampedNumeric: Float32Array;
  normalizedU: Float32Array;
  wasClamped: boolean[];
}

export interface DnamControlSpaceHandle {
  clamp(rawNumericValues: Float32Array): RawClampResult;
  dispose(): void;
}

export interface ValidatedContainerRaw {
  preamble: PreambleSnapshot;
  verificationStatus: "Unsigned" | "CreatorVerified" | "MarketplaceVerified";
  protectionProfile: DnamProtectionProfile;
  headerJson: string;
  arenaBytes: Uint8Array;
}

export interface DnamDriver {
  readonly kind: "native" | "wasm";

  createEngine(
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string,
  ): DnamEngineHandle;

  createControlSpace(headerJson: string): DnamControlSpaceHandle;

  validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): ValidatedContainerRaw;

  buildContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    plaintextArenaCrcSource?: Uint8Array,
    encryptedArena?: boolean,
  ): Uint8Array;

  buildReferenceContainer(core2Active: boolean, isCheckpoint: boolean): Uint8Array;

  updateHeaderInPlace(containerBytes: Uint8Array, updatedHeaderJson: string): Uint8Array;

  computeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    controlDecimationTrain: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): ZohParamsSnapshot;

  packUploU(denseFlat: Float64Array, nDim: number): Float64Array;
  unpackUploU(packedU: Float64Array, nTier: number): Float64Array;
  crc32c(data: Uint8Array): number;

  polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number;
  polyAdaaStep(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
  ): [number, number, number, number];
  polyAdaaScan(
    inp: Float64Array,
    out: Float64Array,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    reset: boolean,
  ): [number, number, number];

  selectiveRgLruStep(
    x: number,
    rLow: number,
    drSpan: number,
    wSel: number,
    bGammaNorm: number,
    cReal: number,
    cImag: number,
    uRotCos: number,
    uRotSin: number,
    zRe: number,
    zIm: number,
  ): [number, number, number];
  selectiveRgLruScan(
    inp: Float64Array,
    out: Float64Array,
    rLow: number,
    drSpan: number,
    wSel: number,
    bGammaNorm: number,
    cReal: number,
    cImag: number,
    uRotCos: number,
    uRotSin: number,
    zRe: number,
    zIm: number,
    reset: boolean,
  ): [number, number];

  unitRotor(phi: number): [number, number];
  unitRotorSlice(phi: Float64Array, out: Float64Array): void;
  cubicLagrangeWeights(mu: number): [number, number, number, number];
  givensHouseholderMix8ch(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array;
  mufdnCoerciveStep(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): [number, number, number];
  mufdnSlice(
    inp: Float64Array,
    out: Float64Array,
    delaySamples: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void;

  evalLobattoBasis(u: number, anchors: Float64Array, degree: number): [number, number, number];
  evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void;
  evalRank4Fm(features: Float64Array, vFactorsFlat: Float64Array): [number, number, number, number];
  evalChebyshevLfo(cosPhi: number): [number, number, number, number];
  evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void;
  evalTrustRegionSoftclip(z: number, mh: number): number;
  evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void;

  immutableHeaderSha256(headerJson: string): string;
  sha256Hex(data: Uint8Array): string;
  buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array;
  buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array;
  chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array;
  applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa48: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array;
  generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array;
  applyTardosWatermark(arenaBytes: Uint8Array, code4608: Float32Array, epsilon: number): Uint8Array;
  scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code4608: Float32Array,
  ): number;
  extractStructuralHash(arenaBytes: Uint8Array): Float64Array;
  structuralCosineSimilarity(a: Float64Array, b: Float64Array): number;

  ed25519PubkeyHex(seed32: Uint8Array): string;
  ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string;
  x25519PubkeyHex(secret32: Uint8Array): string;
  wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string;
  signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX25519_32: Uint8Array,
  ): string;
  verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string;
  sealContainer(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array;
  unsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array;
}
