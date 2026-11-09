/**
 * Node.js 26 `N-API v10` driver implementation (`NapiDriver`) and test-only parity bindings
 * (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.engine.zero-copy-and-async]`).
 *
 * @module
 */

import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { DnamError, fromNativeError } from "../../errors.js";
import type { ActiveBackend } from "../../types.js";
import { coerceBackend } from "../coerce.js";
import type {
  CreateNativeEngineOptions,
  NativeControlSpaceHandle,
  NativeDriver,
  NativeEngineHandle,
  NativeValidatedContainerInfo,
} from "./driver.js";
import {
  configureWasmMemoryImport,
  instantiateWasmDriver,
  setCachedWasmBytes,
  type WasmDriver,
  type WasmMemoryOptions,
} from "./wasm.js";

interface RawNapiEngine {
  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void;
  readonly backend: string;
  readonly tier: number;
  setTier(tier: number): void;
  readonly maxSupportedTier: number;
  setMaxSupportedTier(tier: number): void;
  readonly maxAvailableTier: number;
  readonly effectiveTier: number;
  readonly sampleRate: number;
  setSampleRate(sampleRateHz: number): void;
  readonly latencySamples: number;
  readonly offlineBounce: boolean;
  setOfflineBounce(enabled: boolean): void;
  readonly maximizeTierOnBounce: boolean;
  setMaximizeTierOnBounce(enabled: boolean): void;
  readonly numControls: number;
  rawControlValues(): Float32Array;
  clampedControlValues(): Float32Array;
  clampedControlValue(ctrlIdx: number): number;
  setControlValue(ctrlIdx: number, value: number): void;
  setControlImmediate(ctrlIdx: number, value: number): void;
  setControlsImmediateBatch(indices: number[], values: number[]): void;
  resetControlsToDefaults(): void;
  reset(): void;
  readonly supportsFreeze: boolean;
  readonly isFrozen: boolean;
  setFreeze(freeze: boolean): void;
  setFreezeAmount(alpha: number): void;
  bypassState(): boolean[];
  setBypass(bypassed: boolean, trails: boolean): void;
  readonly stereoWidth: number;
  setStereoWidth(width: number): void;
  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void;
  readonly currentBpm: number;
  syncTempo(bpm: number, subdivision?: string, foldOctave?: boolean): number;
  registerTapTempo(subdivision?: string, foldOctave?: boolean): number[];
  applyPreset(presetId: string): void;
  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void;
  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void;
  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void;
  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void;
  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void;
  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<Float32Array[]>;
  telemetryInto(out56: Float64Array): void;
}

interface RawNapiControlSpace {
  readonly numControls: number;
  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number;
  toPhysical(ctrlIdx: number, normalizedU: number): number;
  toNormalized(ctrlIdx: number, physicalVal: number): number;
}

export interface NativeTestingBindings {
  testingRenderReferenceStereo(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array[];
  testingRenderReferenceMonoToStereo(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    monoIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array[];
  testingRenderReferenceMono(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    monoIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array;
  testingBenchmarkNativeStereoNs(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    frames: number,
    iterations: number,
  ): number;
  testingReferenceHeaderJson(kind?: string): string;
  testingErrorVariantNames(): string[];
  testingRaiseErrorVariant(name: string): void;
}

interface RawNapiAddon extends NativeTestingBindings {
  NativeEngine: new (
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string,
    maxSupportedTier?: number,
    offlineBounce?: boolean,
    maximizeTierOnBounce?: boolean,
  ) => RawNapiEngine;
  NativeControlSpace: new (headerJson: string) => RawNapiControlSpace;
  validateContainerWithKeyNative(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): NativeValidatedContainerInfo;
  buildReferenceContainerNative(
    supportsFreeze?: boolean,
    inputs?: number,
    outputMode?: string,
  ): Uint8Array;
  buildDnamContainerNative(
    headerJson: string,
    arenaBytes: Uint8Array,
    encryptedArenaBytes?: Uint8Array,
    hasSignaturesFlag?: boolean,
  ): Uint8Array;
  updateHeaderInPlaceNative(containerBytes: Uint8Array, newHeaderJson: string): Uint8Array;
  stripContainerToTierNative(containerBytes: Uint8Array, maxTier: number): Uint8Array;
  parseSchemaVersionNative(versionStr: string): number[];
  crc32CComputeNative(data: Uint8Array): number;
  validateAndNormalizeHeaderNative(headerJson: string): string;
  canonicalizeJcsNative(jsonStr: string, stripMutable?: boolean, rejectFloats?: boolean): string;
  buildDmu1FrameNative(mutableJsonBytes: Uint8Array, capacity: number): Uint8Array;
  parseDmu1FrameNative(frameBytes: Uint8Array): string;
  validateV2OpenDnamLayoutNative(containerBytes: Uint8Array): string;
  stripV2OpenDnamContainerNative(containerBytes: Uint8Array, targetTier: number): Uint8Array;
  buildReferenceCheckpointNative(includeCalibStats?: boolean): Uint8Array;
  verifyStemArena4Crc32CNative(
    arenaBytes: Uint8Array,
    stemByteOffset: number,
    stemByteLen: number,
    expectedCrc32c: number,
  ): void;
  verifyArena5CalibStatsNative(arenaBytes: Uint8Array, arena5Offset: number): number;
  validateV2DnckCheckpointNative(ckptBytes: Uint8Array): number[];
  exportRuntimeDnamFromCkptNative(ckptBytes: Uint8Array): Uint8Array;
  uploUPackedLenNative(nDim: number): number;
  packUploUF64IntoNative(denseFlat: Float64Array, nDim: number, outPacked: Float64Array): void;
  unpackUploUF64IntoNative(packedU: Float64Array, nTier: number, outDense: Float64Array): void;
  computeZohNative(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    rAcTrain?: number,
    rDcTrain?: number,
    tauRefTrain?: number,
    switchXfadeTrain?: number,
  ): Float64Array;
  controlStrideForRateNative(sampleRateHz: number): number;
  controlTickRatioNative(sampleRateHz: number): number;
  evalSubdivisionPhaseNative(
    beatMod12: number,
    withinBeatQ32: number,
    subdivision: string,
    offsetQ32?: number,
  ): number[];
  eval3NodeCalibrationNative(anchors: Float64Array, u: number): number;
  invertTempoCalibrationNative(
    declaredMin: number,
    declaredMax: number,
    safeMin: number,
    safeMax: number,
    anchors: Float64Array,
    target: number,
    flushSubnormals?: boolean,
  ): Float64Array;
  verifyContainerProvenanceNative(
    headerJson: string,
    plaintextArena: Uint8Array,
    hasSignaturesFlag?: boolean,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): string;
  enforceDistributionPolicyNative(headerJson: string, activeProfile: string): void;
  evaluateAnticlonePolicyNative(
    fingerprintMatched: boolean,
    buyerWatermarkIdentified: boolean,
    independentDeviceSourceEvidence: boolean,
  ): string;
  immutableHeaderSha256Native(headerJson: string): string;
  sha256HexNative(data: Uint8Array): string;
  hmacSha256HexNative(key32: Uint8Array, message: Uint8Array): string;
  ed25519PubkeyHexNative(seed32: Uint8Array): string;
  ed25519SignHexNative(seed32: Uint8Array, message: Uint8Array): string;
  x25519PubkeyHexNative(secret32: Uint8Array): string;
  buildCreatorSigningMessageNative(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array;
  buildMarketplaceSigningMessageNative(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array;
  signCreatorHeaderNative(
    headerJson: string,
    masterArena: Uint8Array,
    creatorSeed32: Uint8Array,
    signerId: string,
    includeTierPrefixes?: boolean,
  ): string;
  signMarketplaceReceiptNative(
    headerJson: string,
    buyerArena: Uint8Array,
    marketplaceSeed32: Uint8Array,
    specJson: string,
  ): string;
  signOemDeviceCertNative(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX2551932: Uint8Array,
  ): string;
  verifyOemDeviceCertNative(certJson: string, trustedOemPubkeyHex: string): string;
  buildOemCertV2SigningMessageNative(
    oemVendor: string,
    modelCode: string,
    deviceIdHash16: Uint8Array,
    pubkeyX2551932: Uint8Array,
    issuerPubkeyEd2551932: Uint8Array,
    assurance?: number,
    attestationPolicySha25632?: Uint8Array,
    attestationEvidenceSha25632?: Uint8Array,
  ): Uint8Array;
  chacha20XorArenaNative(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter?: number,
  ): Uint8Array;
  wrapDeviceCekHexNative(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string;
  sealContainerNative(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array;
  unsealContainerNative(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array;
  encodeDau1EnvelopeNative(specJson: string): Uint8Array;
  decodeDau1EnvelopeNative(wireBytes: Uint8Array): string;
  buildDme1HeaderNative(
    profile: number,
    recipientCount: number,
    ciphertextLength: number,
  ): Uint8Array;
  parseDme1PlaintextNative(plaintext: Uint8Array, recipientKeyDigestsHex: string[]): string;
  encodeRecipientKeyIdsNative(keyDigestsHex: string[]): Uint8Array;
  parseRecipientKeyIdsNative(data: Uint8Array, maxRecipients?: number): string[];
  encodeHpkeSlotsNative(slotsJson: string): Uint8Array;
  parseHpkeSlotsNative(data: Uint8Array, expectedCount: number, maxRecipients?: number): string;
  encodeRecordDirectoryNative(maximumTier: number, recordsJson: string): Uint8Array;
  parseRecordDirectoryNative(data: Uint8Array, maximumTier: number): string;
  computePreviewWindowNative(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
  ): Float64Array;
  validatePreviewSwitchOptionsNative(
    preset: string,
    alternative: string,
    observed: string[],
    active: string[],
    safe: string[],
  ): string[];
  validateRoyaltySplitsNative(licenseTier: string, royaltySplitsJson: string): void;
  applyIsometricGaugeWatermarkNative(
    arenaBytes: Uint8Array,
    kappa: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array;
  generateTardosCodeNative(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array;
  applyTardosWatermarkNative(
    arenaBytes: Uint8Array,
    code: Float32Array,
    epsilon: number,
  ): Uint8Array;
  scoreTardosWatermarkNative(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code: Float32Array,
  ): number;
  extractStructuralHashNative(arenaBytes: Uint8Array): Float64Array;
  structuralCosineSimilarityNative(a: Float64Array, b: Float64Array): number;
  deriveWatermarkSeedHexNative(
    key32: Uint8Array,
    marketplacePubkey32: Uint8Array,
    watermarkKeyId16: Uint8Array,
    creatorSignedManifestSha25632: Uint8Array,
    modelId16: Uint8Array,
    watermarkProfileSha25632: Uint8Array,
    owningTier: number,
    licenseId16?: Uint8Array,
    buyerId16?: Uint8Array,
  ): string;
  extractStage3FingerprintNative(
    earlyFir128b: Uint8Array,
    kautzModalBytes: Uint8Array,
    addedModes: number,
  ): Float64Array;
  catmullRomHermiteNative(sM1: number, s0: number, s1: number, s2: number, mu: number): number;
  catmullRomHermiteSliceNative(
    sM1: Float64Array,
    s0: Float64Array,
    s1: Float64Array,
    s2: Float64Array,
    mu: Float64Array,
    out: Float64Array,
  ): void;
  opTCoerciveStepNative(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): Float64Array;
  opTTankSliceNative(
    inp: Float64Array,
    out: Float64Array,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void;
  opMModalStepNative(
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
  ): Float64Array;
  opMModalSliceNative(
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
  ): Float64Array;
  unitRotorNative(phi: number): Float64Array;
  unitRotorSliceNative(phi: Float64Array, out: Float64Array): void;
  polyAdaaAlgebraicNative(uPrev: number, uCurr: number, beta: number, delta: number): number;
  polyAdaaStepNative(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
  ): Float64Array;
  polyAdaaSliceNative(
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
  ): Float64Array;
  evalLobattoBasisNative(u: number, anchors: Float64Array, degree: number): Float64Array;
  evalLobattoBasisSliceNative(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void;
  evalRank4FmNative(features: Float64Array, vFactorsFlat: Float64Array): Float64Array;
  evalRank4FmSliceNative(
    featuresFlat: Float64Array,
    nFrames: number,
    dDim: number,
    vFactorsFlat: Float64Array,
    out: Float64Array,
  ): void;
  evalChebyshevLfoNative(cosPhi: number): Float64Array;
  evalChebyshevLfoSliceNative(cosPhi: Float64Array, out: Float64Array): void;
  evalTrustRegionSoftclipNative(z: number, mh: number): number;
  evalTrustRegionSoftclipSliceNative(z: Float64Array, out: Float64Array, mh: number): void;
  cubicLagrangeWeightsNative(mu: number): Float64Array;
  cubicLagrangeWeightsSliceNative(mu: Float64Array, out: Float64Array): void;
  givensHouseholder8ChNative(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array;
  givensHouseholder8ChSliceNative(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
  ): void;
  givensHouseholder16ChNative(
    x: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos: Float64Array,
    crossSin: Float64Array,
  ): Float64Array;
  givensHouseholder16ChSliceNative(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos: Float64Array,
    crossSin: Float64Array,
  ): void;
}

function callNapi<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    throw fromNativeError(err);
  }
}

class NapiEngineHandle implements NativeEngineHandle {
  readonly #raw: RawNapiEngine;
  #disposed = false;
  #busy = false;

  constructor(raw: RawNapiEngine) {
    this.#raw = raw;
  }

  #ensureAlive(): RawNapiEngine {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    return this.#raw;
  }

  #ensureAudioIdle(): RawNapiEngine {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    if (this.#busy) {
      throw new DnamError(
        "ERR_DNAM_BUSY",
        "Engine is busy executing an in-flight processAsync task",
      );
    }
    return this.#raw;
  }

  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    callNapi(() => this.#ensureAlive().reload(containerBytes, tier, sampleRateHz));
  }

  backend(): ActiveBackend {
    return coerceBackend(callNapi(() => this.#ensureAlive().backend)) as ActiveBackend;
  }

  tier(): number {
    return callNapi(() => this.#ensureAlive().tier);
  }

  setTier(tier: number): void {
    callNapi(() => this.#ensureAlive().setTier(tier));
  }

  maxSupportedTier(): number {
    return callNapi(() => this.#ensureAlive().maxSupportedTier);
  }

  setMaxSupportedTier(tier: number): void {
    callNapi(() => this.#ensureAlive().setMaxSupportedTier(tier));
  }

  maxAvailableTier(): number {
    return callNapi(() => this.#ensureAlive().maxAvailableTier);
  }

  effectiveTier(): number {
    return callNapi(() => this.#ensureAlive().effectiveTier);
  }

  sampleRate(): number {
    return callNapi(() => this.#ensureAlive().sampleRate);
  }

  setSampleRate(sampleRateHz: number): void {
    callNapi(() => this.#ensureAudioIdle().setSampleRate(sampleRateHz));
  }

  latencySamples(): number {
    return callNapi(() => this.#ensureAlive().latencySamples);
  }

  offlineBounce(): boolean {
    return callNapi(() => this.#ensureAlive().offlineBounce);
  }

  setOfflineBounce(enabled: boolean): void {
    callNapi(() => this.#ensureAlive().setOfflineBounce(enabled));
  }

  maximizeTierOnBounce(): boolean {
    return callNapi(() => this.#ensureAlive().maximizeTierOnBounce);
  }

  setMaximizeTierOnBounce(enabled: boolean): void {
    callNapi(() => this.#ensureAlive().setMaximizeTierOnBounce(enabled));
  }

  numControls(): number {
    return callNapi(() => this.#ensureAlive().numControls);
  }

  rawControlValues(): Float32Array {
    return callNapi(() => this.#ensureAlive().rawControlValues());
  }

  clampedControlValue(ctrlIdx: number): number {
    return callNapi(() => this.#ensureAlive().clampedControlValue(ctrlIdx));
  }

  setControlValue(ctrlIdx: number, value: number): void {
    callNapi(() => this.#ensureAlive().setControlValue(ctrlIdx, value));
  }

  setControlImmediate(ctrlIdx: number, value: number): void {
    callNapi(() => this.#ensureAlive().setControlImmediate(ctrlIdx, value));
  }

  setControlsImmediateBatch(indices: readonly number[], values: readonly number[]): void {
    callNapi(() => this.#ensureAlive().setControlsImmediateBatch([...indices], [...values]));
  }

  resetControlsToDefaults(): void {
    callNapi(() => this.#ensureAlive().resetControlsToDefaults());
  }

  reset(): void {
    callNapi(() => this.#ensureAlive().reset());
  }

  supportsFreeze(): boolean {
    return callNapi(() => this.#ensureAlive().supportsFreeze);
  }

  isFrozen(): boolean {
    return callNapi(() => this.#ensureAlive().isFrozen);
  }

  setFreeze(freeze: boolean): void {
    callNapi(() => this.#ensureAlive().setFreeze(freeze));
  }

  setFreezeAmount(alpha: number): void {
    callNapi(() => this.#ensureAlive().setFreezeAmount(alpha));
  }

  bypassState(): readonly [boolean, boolean] {
    const res = callNapi(() => this.#ensureAlive().bypassState());
    return [Boolean(res[0]), Boolean(res[1])];
  }

  setBypass(bypassed: boolean, trails: boolean): void {
    callNapi(() => this.#ensureAlive().setBypass(bypassed, trails));
  }

  stereoWidth(): number {
    return callNapi(() => this.#ensureAlive().stereoWidth);
  }

  setStereoWidth(width: number): void {
    callNapi(() => this.#ensureAlive().setStereoWidth(width));
  }

  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void {
    callNapi(() =>
      this.#ensureAlive().configureHostMix(blendWet, outputGainDb, applyUnityLeveling),
    );
  }

  currentBpm(): number {
    return callNapi(() => this.#ensureAlive().currentBpm);
  }

  syncTempo(bpm: number, subdivision: string | undefined, foldOctave: boolean): number {
    return callNapi(() => this.#ensureAlive().syncTempo(bpm, subdivision, foldOctave));
  }

  registerTapTempo(
    subdivision: string | undefined,
    foldOctave: boolean,
  ): readonly [number, number] {
    const res = callNapi(() => this.#ensureAlive().registerTapTempo(subdivision, foldOctave));
    return [Number(res[0]), Number(res[1])];
  }

  applyPreset(presetId: string): void {
    callNapi(() => this.#ensureAlive().applyPreset(presetId));
  }

  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void {
    callNapi(() => this.#ensureAudioIdle().processStereoInPlace(leftIo, rightIo));
  }

  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    callNapi(() =>
      this.#ensureAudioIdle().processStereo(leftIn, rightIn, leftOut, rightOut, blockSize),
    );
  }

  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    callNapi(() =>
      this.#ensureAudioIdle().processMonoToStereo(monoIn, leftOut, rightOut, blockSize),
    );
  }

  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void {
    callNapi(() => this.#ensureAudioIdle().processMono(monoIn, monoOut, blockSize));
  }

  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void {
    callNapi(() => this.#ensureAudioIdle().renderOfflineBounce(leftIn, rightIn, leftOut, rightOut));
  }

  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<readonly [Float32Array, Float32Array]> {
    const raw = this.#ensureAudioIdle();
    this.#busy = true;
    let task: Promise<Float32Array[]>;
    try {
      task = raw.processStereoAsync(leftIn, rightIn, blockSize);
    } catch (err) {
      this.#busy = false;
      throw fromNativeError(err);
    }
    return task.then(
      (out) => {
        this.#busy = false;
        const l = out[0];
        const r = out[1];
        if (!l || !r) {
          throw new DnamError("ERR_DNAM_VALIDATION", "Invalid output from processStereoAsync");
        }
        return [l, r] as const;
      },
      (err: unknown) => {
        this.#busy = false;
        throw fromNativeError(err);
      },
    );
  }

  telemetryInto(out56: Float64Array): void {
    callNapi(() => this.#ensureAlive().telemetryInto(out56));
  }

  dispose(): void {
    this.#disposed = true;
  }
}

class NapiControlSpaceHandle implements NativeControlSpaceHandle {
  readonly #raw: RawNapiControlSpace;
  #disposed = false;

  constructor(raw: RawNapiControlSpace) {
    this.#raw = raw;
  }

  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.clampInto(rawNumeric, clampedOut, normOut));
  }

  toPhysical(ctrlIdx: number, normalizedU: number): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.toPhysical(ctrlIdx, normalizedU));
  }

  toNormalized(ctrlIdx: number, physicalVal: number): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.toNormalized(ctrlIdx, physicalVal));
  }

  dispose(): void {
    this.#disposed = true;
  }
}

export class NapiDriver implements NativeDriver {
  readonly kind = "napi" as const;
  readonly #addon: RawNapiAddon;

  constructor(addon: RawNapiAddon) {
    this.#addon = addon;
  }

  createEngine(options: CreateNativeEngineOptions): NativeEngineHandle {
    const raw = callNapi(
      () =>
        new this.#addon.NativeEngine(
          options.containerBytes,
          options.tier,
          options.sampleRateHz,
          options.backend,
          options.maxSupportedTier,
          options.offlineBounce,
          options.maximizeTierOnBounce,
        ),
    );
    return new NapiEngineHandle(raw);
  }

  createControlSpace(headerJson: string): NativeControlSpaceHandle {
    const raw = callNapi(() => new this.#addon.NativeControlSpace(headerJson));
    return new NapiControlSpaceHandle(raw);
  }

  validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): NativeValidatedContainerInfo {
    return callNapi(() =>
      this.#addon.validateContainerWithKeyNative(
        containerBytes,
        deviceIdHash,
        deviceSecret32,
        trustedMarketplacePubkeyHex,
      ),
    );
  }

  buildReferenceContainer(supportsFreeze: boolean, inputs: number, outputMode: string): Uint8Array {
    return callNapi(() =>
      this.#addon.buildReferenceContainerNative(supportsFreeze, inputs, outputMode),
    );
  }

  buildDnamContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    encryptedArenaBytes?: Uint8Array,
    hasSignaturesFlag?: boolean,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.buildDnamContainerNative(
        headerJson,
        arenaBytes,
        encryptedArenaBytes,
        hasSignaturesFlag,
      ),
    );
  }

  updateHeaderInPlace(containerBytes: Uint8Array, newHeaderJson: string): Uint8Array {
    return callNapi(() => this.#addon.updateHeaderInPlaceNative(containerBytes, newHeaderJson));
  }

  stripContainerToTier(containerBytes: Uint8Array, maxTier: number): Uint8Array {
    return callNapi(() => this.#addon.stripContainerToTierNative(containerBytes, maxTier));
  }

  parseSchemaVersion(versionStr: string): readonly [number, number, number, boolean] {
    const res = callNapi(() => this.#addon.parseSchemaVersionNative(versionStr));
    return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3]) !== 0];
  }

  crc32c(data: Uint8Array): number {
    return callNapi(() => this.#addon.crc32CComputeNative(data)) >>> 0;
  }

  validateAndNormalizeHeader(headerJson: string): string {
    return callNapi(() => this.#addon.validateAndNormalizeHeaderNative(headerJson));
  }

  canonicalizeJcs(jsonStr: string, stripMutable = false, rejectFloats = false): string {
    return callNapi(() => this.#addon.canonicalizeJcsNative(jsonStr, stripMutable, rejectFloats));
  }

  buildDmu1Frame(mutableJsonBytes: Uint8Array, capacity: number): Uint8Array {
    return callNapi(() => this.#addon.buildDmu1FrameNative(mutableJsonBytes, capacity));
  }

  parseDmu1Frame(frameBytes: Uint8Array): string {
    return callNapi(() => this.#addon.parseDmu1FrameNative(frameBytes));
  }

  validateV2OpenDnamLayout(containerBytes: Uint8Array): string {
    return callNapi(() => this.#addon.validateV2OpenDnamLayoutNative(containerBytes));
  }

  stripV2OpenDnamContainer(containerBytes: Uint8Array, targetTier: number): Uint8Array {
    return callNapi(() => this.#addon.stripV2OpenDnamContainerNative(containerBytes, targetTier));
  }

  buildReferenceCheckpoint(includeCalibStats: boolean): Uint8Array {
    return callNapi(() => this.#addon.buildReferenceCheckpointNative(includeCalibStats));
  }

  verifyStemArena4Crc32c(
    arenaBytes: Uint8Array,
    stemByteOffset: number,
    stemByteLen: number,
    expectedCrc32c: number,
  ): void {
    callNapi(() =>
      this.#addon.verifyStemArena4Crc32CNative(
        arenaBytes,
        stemByteOffset,
        stemByteLen,
        expectedCrc32c,
      ),
    );
  }

  verifyArena5CalibStats(arenaBytes: Uint8Array, arena5Offset: number): number {
    return callNapi(() => this.#addon.verifyArena5CalibStatsNative(arenaBytes, arena5Offset));
  }

  validateV2DnckCheckpoint(ckptBytes: Uint8Array): {
    readonly retainedTier: number;
    readonly bundleEnds: readonly number[];
    readonly mutableBoundary: number;
  } {
    const arr = callNapi(() => this.#addon.validateV2DnckCheckpointNative(ckptBytes));
    const retainedTier = arr[0] ?? 0;
    const mutableBoundary = arr[arr.length - 1] ?? 0;
    const bundleEnds = arr.slice(1, arr.length - 1);
    return { retainedTier, bundleEnds, mutableBoundary };
  }

  exportRuntimeDnamFromCkpt(ckptBytes: Uint8Array): Uint8Array {
    return callNapi(() => this.#addon.exportRuntimeDnamFromCkptNative(ckptBytes));
  }

  uploUPackedLen(nDim: number): number {
    return callNapi(() => this.#addon.uploUPackedLenNative(nDim));
  }

  packUploUF64Into(denseFlat: Float64Array, nDim: number, outPacked: Float64Array): void {
    callNapi(() => this.#addon.packUploUF64IntoNative(denseFlat, nDim, outPacked));
  }

  unpackUploUF64Into(packedU: Float64Array, nTier: number, outDense: Float64Array): void {
    callNapi(() => this.#addon.unpackUploUF64IntoNative(packedU, nTier, outDense));
  }

  computeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): Float64Array {
    return callNapi(() =>
      this.#addon.computeZohNative(
        sampleRateTrainHz,
        sampleRateHostHz,
        rAcTrain,
        rDcTrain,
        tauRefTrain,
        switchXfadeTrain,
      ),
    );
  }

  controlStrideForRate(sampleRateHz: number): number {
    return callNapi(() => this.#addon.controlStrideForRateNative(sampleRateHz));
  }

  controlTickRatio(sampleRateHz: number): number {
    return callNapi(() => this.#addon.controlTickRatioNative(sampleRateHz));
  }

  evalSubdivisionPhase(
    beatMod12: number,
    withinBeatQ32: number,
    subdivision: string,
    offsetQ32: number,
  ): readonly [number, number, number] {
    const res = callNapi(() =>
      this.#addon.evalSubdivisionPhaseNative(beatMod12, withinBeatQ32, subdivision, offsetQ32),
    );
    return [Number(res[0]), Number(res[1]), Number(res[2])];
  }

  eval3NodeCalibration(anchors3: Float64Array, u: number): number {
    return callNapi(() => this.#addon.eval3NodeCalibrationNative(anchors3, u));
  }

  invertTempoCalibration(
    declaredMin: number,
    declaredMax: number,
    safeMin: number,
    safeMax: number,
    anchors3: Float64Array,
    target: number,
    flushSubnormals: boolean,
  ): Float64Array {
    return callNapi(() =>
      this.#addon.invertTempoCalibrationNative(
        declaredMin,
        declaredMax,
        safeMin,
        safeMax,
        anchors3,
        target,
        flushSubnormals,
      ),
    );
  }

  verifyContainerProvenance(
    headerJson: string,
    plaintextArena: Uint8Array,
    hasSignaturesFlag: boolean,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): string {
    return callNapi(() =>
      this.#addon.verifyContainerProvenanceNative(
        headerJson,
        plaintextArena,
        hasSignaturesFlag,
        deviceIdHash,
        deviceSecret32,
        trustedMarketplacePubkeyHex,
      ),
    );
  }

  enforceDistributionPolicy(headerJson: string, activeProfile: string): void {
    callNapi(() => this.#addon.enforceDistributionPolicyNative(headerJson, activeProfile));
  }

  evaluateAnticlonePolicy(
    fingerprintMatched: boolean,
    buyerWatermarkIdentified: boolean,
    independentDeviceSourceEvidence: boolean,
  ): string {
    return callNapi(() =>
      this.#addon.evaluateAnticlonePolicyNative(
        fingerprintMatched,
        buyerWatermarkIdentified,
        independentDeviceSourceEvidence,
      ),
    );
  }

  immutableHeaderSha256(headerJson: string): string {
    return callNapi(() => this.#addon.immutableHeaderSha256Native(headerJson));
  }

  sha256Hex(data: Uint8Array): string {
    return callNapi(() => this.#addon.sha256HexNative(data));
  }

  hmacSha256Hex(key32: Uint8Array, message: Uint8Array): string {
    return callNapi(() => this.#addon.hmacSha256HexNative(key32, message));
  }

  ed25519PubkeyHex(seed32: Uint8Array): string {
    return callNapi(() => this.#addon.ed25519PubkeyHexNative(seed32));
  }

  ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string {
    return callNapi(() => this.#addon.ed25519SignHexNative(seed32, message));
  }

  x25519PubkeyHex(secret32: Uint8Array): string {
    return callNapi(() => this.#addon.x25519PubkeyHexNative(secret32));
  }

  buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.buildCreatorSigningMessageNative(immutableHeaderSha256Hex, masterArenaSha256Hex),
    );
  }

  buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.buildMarketplaceSigningMessageNative(
        immutableHeaderSha256Hex,
        masterArenaSha256Hex,
        receiptJson,
      ),
    );
  }

  signCreatorHeader(
    headerJson: string,
    masterArena: Uint8Array,
    creatorSeed32: Uint8Array,
    signerId: string,
    includeTierPrefixes: boolean,
  ): string {
    return callNapi(() =>
      this.#addon.signCreatorHeaderNative(
        headerJson,
        masterArena,
        creatorSeed32,
        signerId,
        includeTierPrefixes,
      ),
    );
  }

  signMarketplaceReceipt(
    headerJson: string,
    buyerArena: Uint8Array,
    marketplaceSeed32: Uint8Array,
    specJson: string,
  ): string {
    return callNapi(() =>
      this.#addon.signMarketplaceReceiptNative(headerJson, buyerArena, marketplaceSeed32, specJson),
    );
  }

  signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX2551932: Uint8Array,
  ): string {
    return callNapi(() =>
      this.#addon.signOemDeviceCertNative(
        oemSeed32,
        oemVendor,
        modelCode,
        deviceIdHash,
        pubkeyX2551932,
      ),
    );
  }

  verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string {
    return callNapi(() => this.#addon.verifyOemDeviceCertNative(certJson, trustedOemPubkeyHex));
  }

  buildOemCertV2SigningMessage(
    oemVendor: string,
    modelCode: string,
    deviceIdHash16: Uint8Array,
    pubkeyX2551932: Uint8Array,
    issuerPubkeyEd2551932: Uint8Array,
    assurance: number,
    attestationPolicySha25632?: Uint8Array,
    attestationEvidenceSha25632?: Uint8Array,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.buildOemCertV2SigningMessageNative(
        oemVendor,
        modelCode,
        deviceIdHash16,
        pubkeyX2551932,
        issuerPubkeyEd2551932,
        assurance,
        attestationPolicySha25632,
        attestationEvidenceSha25632,
      ),
    );
  }

  chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.chacha20XorArenaNative(arenaBytes, key32, nonce12, startBlockCounter),
    );
  }

  wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string {
    return callNapi(() =>
      this.#addon.wrapDeviceCekHexNative(ephemeralSecret32, devicePubkey32, deviceIdHash, cek32),
    );
  }

  sealContainer(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.sealContainerNative(
        innerDnamBytes,
        recipientCertsJson,
        trustedOemPubkeysJson,
        marketplaceSeed32,
        ephemeralSecret32,
        cek32,
        nonce12,
      ),
    );
  }

  unsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.unsealContainerNative(
        sealedBytes,
        deviceIdHash,
        deviceSecret32,
        trustedMarketplacePubkeyHex,
      ),
    );
  }

  encodeDau1Envelope(specJson: string): Uint8Array {
    return callNapi(() => this.#addon.encodeDau1EnvelopeNative(specJson));
  }

  decodeDau1Envelope(wireBytes: Uint8Array): string {
    return callNapi(() => this.#addon.decodeDau1EnvelopeNative(wireBytes));
  }

  buildDme1Header(profile: number, recipientCount: number, ciphertextLength: number): Uint8Array {
    return callNapi(() =>
      this.#addon.buildDme1HeaderNative(profile, recipientCount, ciphertextLength),
    );
  }

  parseDme1Plaintext(plaintext: Uint8Array, recipientKeyDigestsHex: readonly string[]): string {
    return callNapi(() =>
      this.#addon.parseDme1PlaintextNative(plaintext, [...recipientKeyDigestsHex]),
    );
  }

  encodeRecipientKeyIds(keyDigestsHex: readonly string[]): Uint8Array {
    return callNapi(() => this.#addon.encodeRecipientKeyIdsNative([...keyDigestsHex]));
  }

  parseRecipientKeyIds(data: Uint8Array, maxRecipients: number): readonly string[] {
    return callNapi(() => this.#addon.parseRecipientKeyIdsNative(data, maxRecipients));
  }

  encodeHpkeSlots(slotsJson: string): Uint8Array {
    return callNapi(() => this.#addon.encodeHpkeSlotsNative(slotsJson));
  }

  parseHpkeSlots(data: Uint8Array, expectedCount: number, maxRecipients: number): string {
    return callNapi(() => this.#addon.parseHpkeSlotsNative(data, expectedCount, maxRecipients));
  }

  encodeRecordDirectory(maximumTier: number, recordsJson: string): Uint8Array {
    return callNapi(() => this.#addon.encodeRecordDirectoryNative(maximumTier, recordsJson));
  }

  parseRecordDirectory(data: Uint8Array, maximumTier: number): string {
    return callNapi(() => this.#addon.parseRecordDirectoryNative(data, maximumTier));
  }

  computePreviewWindow(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
  ): readonly [number, number] {
    const res = callNapi(() =>
      this.#addon.computePreviewWindowNative(anchor, delta, safeMin, safeMax),
    );
    return [Number(res[0]), Number(res[1])];
  }

  validatePreviewSwitchOptions(
    preset: string,
    alternative: string,
    observed: readonly string[],
    active: readonly string[],
    safe: readonly string[],
  ): readonly [string, string] {
    const res = callNapi(() =>
      this.#addon.validatePreviewSwitchOptionsNative(
        preset,
        alternative,
        [...observed],
        [...active],
        [...safe],
      ),
    );
    return [String(res[0]), String(res[1])];
  }

  validateRoyaltySplits(licenseTier: string, royaltySplitsJson: string): void {
    callNapi(() => this.#addon.validateRoyaltySplitsNative(licenseTier, royaltySplitsJson));
  }

  applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa64: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array {
    return callNapi(() =>
      this.#addon.applyIsometricGaugeWatermarkNative(arenaBytes, kappa64, qBuyerFlat16),
    );
  }

  generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array {
    return callNapi(() =>
      this.#addon.generateTardosCodeNative(marketplaceSecret32, modelId, licenseId),
    );
  }

  applyTardosWatermark(
    arenaBytes: Uint8Array,
    code7680: Float32Array,
    epsilon: number,
  ): Uint8Array {
    return callNapi(() => this.#addon.applyTardosWatermarkNative(arenaBytes, code7680, epsilon));
  }

  scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code7680: Float32Array,
  ): number {
    return callNapi(() =>
      this.#addon.scoreTardosWatermarkNative(masterArena, suspectArena, code7680),
    );
  }

  extractStructuralHash(arenaBytes: Uint8Array): Float64Array {
    return callNapi(() => this.#addon.extractStructuralHashNative(arenaBytes));
  }

  structuralCosineSimilarity(a: Float64Array, b: Float64Array): number {
    return callNapi(() => this.#addon.structuralCosineSimilarityNative(a, b));
  }

  deriveWatermarkSeedHex(
    key32: Uint8Array,
    marketplacePubkey32: Uint8Array,
    watermarkKeyId16: Uint8Array,
    creatorSignedManifestSha25632: Uint8Array,
    modelId16: Uint8Array,
    watermarkProfileSha25632: Uint8Array,
    owningTier: number,
    licenseId16?: Uint8Array,
    buyerId16?: Uint8Array,
  ): string {
    return callNapi(() =>
      this.#addon.deriveWatermarkSeedHexNative(
        key32,
        marketplacePubkey32,
        watermarkKeyId16,
        creatorSignedManifestSha25632,
        modelId16,
        watermarkProfileSha25632,
        owningTier,
        licenseId16,
        buyerId16,
      ),
    );
  }

  extractStage3Fingerprint(
    earlyFir128b: Uint8Array,
    kautzModalBytes: Uint8Array,
    addedModes: number,
  ): Float64Array {
    return callNapi(() =>
      this.#addon.extractStage3FingerprintNative(earlyFir128b, kautzModalBytes, addedModes),
    );
  }

  catmullRomHermite(sM1: number, s0: number, s1: number, s2: number, mu: number): number {
    return callNapi(() => this.#addon.catmullRomHermiteNative(sM1, s0, s1, s2, mu));
  }

  catmullRomHermiteSlice(
    sM1: Float64Array,
    s0: Float64Array,
    s1: Float64Array,
    s2: Float64Array,
    mu: Float64Array,
    out: Float64Array,
  ): void {
    callNapi(() => this.#addon.catmullRomHermiteSliceNative(sM1, s0, s1, s2, mu, out));
  }

  opTCoerciveStep(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): readonly [number, number, number] {
    const res = callNapi(() =>
      this.#addon.opTCoerciveStepNative(uIn, gLin, sigmaKnee, vCeil, rDc, dcXPrev, dcYPrev),
    );
    return [Number(res[0]), Number(res[1]), Number(res[2])];
  }

  opTTankSlice(
    inp: Float64Array,
    out: Float64Array,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void {
    callNapi(() => this.#addon.opTTankSliceNative(inp, out, gLin, sigmaKnee, vCeil, rDc));
  }

  opMModalStep(
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
  ): readonly [number, number, number] {
    const res = callNapi(() =>
      this.#addon.opMModalStepNative(
        x,
        rLow,
        drSpan,
        wSel,
        bGammaNorm,
        cReal,
        cImag,
        uRotCos,
        uRotSin,
        zRe,
        zIm,
      ),
    );
    return [Number(res[0]), Number(res[1]), Number(res[2])];
  }

  opMModalSlice(
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
  ): readonly [number, number] {
    const res = callNapi(() =>
      this.#addon.opMModalSliceNative(
        inp,
        out,
        rLow,
        drSpan,
        wSel,
        bGammaNorm,
        cReal,
        cImag,
        uRotCos,
        uRotSin,
        zRe,
        zIm,
        reset,
      ),
    );
    return [Number(res[0]), Number(res[1])];
  }

  unitRotor(phi: number): readonly [number, number] {
    const res = callNapi(() => this.#addon.unitRotorNative(phi));
    return [Number(res[0]), Number(res[1])];
  }

  unitRotorSlice(phi: Float64Array, out: Float64Array): void {
    callNapi(() => this.#addon.unitRotorSliceNative(phi, out));
  }

  polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number {
    return callNapi(() => this.#addon.polyAdaaAlgebraicNative(uPrev, uCurr, beta, delta));
  }

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
  ): readonly [number, number, number, number] {
    const res = callNapi(() =>
      this.#addon.polyAdaaStepNative(x, bConv, alpha, beta, delta, rAc, uPrev, aPrev, mDc),
    );
    return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
  }

  polyAdaaSlice(
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
  ): readonly [number, number, number] {
    const res = callNapi(() =>
      this.#addon.polyAdaaSliceNative(
        inp,
        out,
        bConv,
        alpha,
        beta,
        delta,
        rAc,
        uPrev,
        aPrev,
        mDc,
        reset,
      ),
    );
    return [Number(res[0]), Number(res[1]), Number(res[2])];
  }

  evalLobattoBasis(
    u: number,
    anchors3: Float64Array,
    degree: number,
  ): readonly [number, number, number] {
    const res = callNapi(() => this.#addon.evalLobattoBasisNative(u, anchors3, degree));
    return [Number(res[0]), Number(res[1]), Number(res[2])];
  }

  evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors3: Float64Array,
    degree: number,
  ): void {
    callNapi(() => this.#addon.evalLobattoBasisSliceNative(u, out, anchors3, degree));
  }

  evalRank4Fm(
    features: Float64Array,
    vFactorsFlat: Float64Array,
  ): readonly [number, number, number, number] {
    const res = callNapi(() => this.#addon.evalRank4FmNative(features, vFactorsFlat));
    return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
  }

  evalRank4FmSlice(
    featuresFlat: Float64Array,
    nFrames: number,
    dDim: number,
    vFactorsFlat: Float64Array,
    out: Float64Array,
  ): void {
    callNapi(() =>
      this.#addon.evalRank4FmSliceNative(featuresFlat, nFrames, dDim, vFactorsFlat, out),
    );
  }

  evalChebyshevLfo(cosPhi: number): readonly [number, number, number, number] {
    const res = callNapi(() => this.#addon.evalChebyshevLfoNative(cosPhi));
    return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
  }

  evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void {
    callNapi(() => this.#addon.evalChebyshevLfoSliceNative(cosPhi, out));
  }

  evalTrustRegionSoftclip(z: number, mh: number): number {
    return callNapi(() => this.#addon.evalTrustRegionSoftclipNative(z, mh));
  }

  evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void {
    callNapi(() => this.#addon.evalTrustRegionSoftclipSliceNative(z, out, mh));
  }

  cubicLagrangeWeights(mu: number): readonly [number, number, number, number] {
    const res = callNapi(() => this.#addon.cubicLagrangeWeightsNative(mu));
    return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
  }

  cubicLagrangeWeightsSlice(mu: Float64Array, out: Float64Array): void {
    callNapi(() => this.#addon.cubicLagrangeWeightsSliceNative(mu, out));
  }

  givensHouseholder8ch(x8: Float64Array, cosPsi: number, sinPsi: number): Float64Array {
    return callNapi(() => this.#addon.givensHouseholder8ChNative(x8, cosPsi, sinPsi));
  }

  givensHouseholder8chSlice(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
  ): void {
    callNapi(() => this.#addon.givensHouseholder8ChSliceNative(xFlat, out, cosPsi, sinPsi));
  }

  givensHouseholder16ch(
    x16: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos8: Float64Array,
    crossSin8: Float64Array,
  ): Float64Array {
    return callNapi(() =>
      this.#addon.givensHouseholder16ChNative(x16, cosPsi, sinPsi, crossCos8, crossSin8),
    );
  }

  givensHouseholder16chSlice(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos8: Float64Array,
    crossSin8: Float64Array,
  ): void {
    callNapi(() =>
      this.#addon.givensHouseholder16ChSliceNative(
        xFlat,
        out,
        cosPsi,
        sinPsi,
        crossCos8,
        crossSin8,
      ),
    );
  }
}

function resolveDistDir(): string {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  const candidateDist = resolve(currentDir, "../../../dist");
  if (existsSync(join(candidateDist, "dnam.wasm"))) {
    return candidateDist;
  }
  const directDist = resolve(currentDir);
  if (existsSync(join(directDist, "dnam.wasm"))) {
    return directDist;
  }
  return resolve(currentDir, "../../dist");
}

let cachedAddon: RawNapiAddon | undefined;
let cachedNapiDriver: NapiDriver | undefined;
let cachedNodeWasmDriver: WasmDriver | undefined;

export function getNapiAddon(): RawNapiAddon {
  if (cachedAddon) {
    return cachedAddon;
  }
  const distDir = resolveDistDir();
  const addonPath = join(distDir, `dnam.${process.platform}-${process.arch}.node`);
  if (!existsSync(addonPath)) {
    throw new DnamError("ERR_DNAM_CONTAINER", `Native N-API addon not found at ${addonPath}`);
  }
  const req = createRequire(import.meta.url);
  cachedAddon = req(addonPath) as RawNapiAddon;
  return cachedAddon;
}

export function getTestingBindings(): NativeTestingBindings {
  return getNapiAddon();
}

export function readDistWasmBytes(): Uint8Array<ArrayBuffer> {
  const distDir = resolveDistDir();
  const wasmPath = join(distDir, "dnam.wasm");
  const bytes = new Uint8Array(readFileSync(wasmPath));
  setCachedWasmBytes(bytes);
  return bytes;
}

export function loadNodeWasmDriverSync(options?: WasmMemoryOptions): WasmDriver {
  if (!options && cachedNodeWasmDriver) {
    return cachedNodeWasmDriver;
  }
  const wasmBytes = readDistWasmBytes();
  const wantShared = Boolean(options?.shared);
  const mod = new WebAssembly.Module(configureWasmMemoryImport(wasmBytes, wantShared));
  const driver = instantiateWasmDriver(mod, options);
  if (!options) {
    cachedNodeWasmDriver = driver;
  }
  return driver;
}

export function loadNodeDriverSync(): NativeDriver {
  if (cachedNapiDriver) {
    return cachedNapiDriver;
  }
  try {
    cachedNapiDriver = new NapiDriver(getNapiAddon());
    return cachedNapiDriver;
  } catch {
    return loadNodeWasmDriverSync();
  }
}
