/**
 * Unified private `NativeDriver` interface implemented by both `NapiDriver` (Node.js 26 N-API v10)
 * and `WasmDriver` (WebAssembly SIMD128) (`ref[sdk.ts.runtime.dual-target-drivers]`).
 *
 * @module
 */

import type { ActiveBackend } from "../../types.js";

/**
 * Number of `f64` elements written by `NativeEngineHandle.telemetryInto`.
 */
export const TELEMETRY_F64_LEN = 56;

/**
 * Validated container metadata returned by `driver.validateContainer`.
 */
export interface NativeValidatedContainerInfo {
  readonly magic: string;
  readonly versionMajor: number;
  readonly versionMinor: number;
  readonly flags: number;
  readonly numControls: number;
  readonly sampleRateTrainHz: number;
  readonly jsonHeaderByteLen: number;
  readonly tensorArenaOffset: number;
  readonly tensorArenaByteLen: number;
  readonly arenaLiteEndCl: number;
  readonly arenaFullEndCl: number;
  readonly arenaProEndCl: number;
  readonly arenaUltraEndCl: number;
  readonly arenaOffsetLiteEnd: number;
  readonly arenaOffsetFullEnd: number;
  readonly arenaOffsetProEnd: number;
  readonly arenaOffsetUltraEnd: number;
  readonly arenaCrc32cLite: number;
  readonly arenaCrc32cFull: number;
  readonly arenaCrc32cPro: number;
  readonly arenaCrc32cUltra: number;
  readonly numCkptStems: number;
  readonly tailTimeMs: number;
  readonly headerCrc32c: number;
  readonly highestTier: number;
  readonly isStereoOut: boolean;
  readonly hasSignatures: boolean;
  readonly isEncryptedArena: boolean;
  readonly hasCalibStats: boolean;
  readonly verificationStatus: string;
  readonly protectionProfile: string;
  readonly headerJson: string;
  readonly arenaBytes: Uint8Array;
}

/**
 * Options for instantiating a native engine handle.
 */
export interface CreateNativeEngineOptions {
  readonly containerBytes: Uint8Array;
  readonly tier: number;
  readonly sampleRateHz: number;
  readonly backend: string;
  readonly maxSupportedTier?: number;
  readonly offlineBounce?: boolean;
  readonly maximizeTierOnBounce?: boolean;
}

/**
 * Stateful native engine handle (`DynamicDnamEngine<'static>`).
 */
export interface NativeEngineHandle {
  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void;
  backend(): ActiveBackend;
  tier(): number;
  setTier(tier: number): void;
  maxSupportedTier(): number;
  setMaxSupportedTier(tier: number): void;
  maxAvailableTier(): number;
  effectiveTier(): number;
  sampleRate(): number;
  setSampleRate(sampleRateHz: number): void;
  latencySamples(): number;
  offlineBounce(): boolean;
  setOfflineBounce(enabled: boolean): void;
  maximizeTierOnBounce(): boolean;
  setMaximizeTierOnBounce(enabled: boolean): void;
  numControls(): number;
  rawControlValues(): Float32Array;
  clampedControlValue(ctrlIdx: number): number;
  setControlValue(ctrlIdx: number, value: number): void;
  setControlImmediate(ctrlIdx: number, value: number): void;
  setControlsImmediateBatch(indices: readonly number[], values: readonly number[]): void;
  resetControlsToDefaults(): void;
  reset(): void;
  supportsFreeze(): boolean;
  isFrozen(): boolean;
  setFreeze(freeze: boolean): void;
  setFreezeAmount(alpha: number): void;
  bypassState(): readonly [boolean, boolean];
  setBypass(bypassed: boolean, trails: boolean): void;
  stereoWidth(): number;
  setStereoWidth(width: number): void;
  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void;
  currentBpm(): number;
  syncTempo(bpm: number, subdivision: string | undefined, foldOctave: boolean): number;
  registerTapTempo(subdivision: string | undefined, foldOctave: boolean): readonly [number, number];
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
  ): Promise<readonly [Float32Array, Float32Array]>;
  telemetryInto(out56: Float64Array): void;
  dispose(): void;
}

/**
 * Pre-compiled 3-Phase control space handle (`ControlClampDescriptor`).
 */
export interface NativeControlSpaceHandle {
  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number;
  toPhysical(ctrlIdx: number, normalizedU: number): number;
  toNormalized(ctrlIdx: number, physicalVal: number): number;
  dispose(): void;
}

/**
 * Target-agnostic driver contract implemented by `NapiDriver` and `WasmDriver`.
 */
export interface NativeDriver {
  readonly kind: "napi" | "wasm";

  // 1. Engine & Control Space handles
  createEngine(options: CreateNativeEngineOptions): NativeEngineHandle;
  createControlSpace(headerJson: string): NativeControlSpaceHandle;

  // 2. Containers & Checkpoints
  validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): NativeValidatedContainerInfo;
  buildReferenceContainer(supportsFreeze: boolean, inputs: number, outputMode: string): Uint8Array;
  buildDnamContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    encryptedArenaBytes?: Uint8Array,
    hasSignaturesFlag?: boolean,
  ): Uint8Array;
  updateHeaderInPlace(containerBytes: Uint8Array, newHeaderJson: string): Uint8Array;
  stripContainerToTier(containerBytes: Uint8Array, maxTier: number): Uint8Array;
  parseSchemaVersion(versionStr: string): readonly [number, number, number, boolean];
  crc32c(data: Uint8Array): number;
  validateAndNormalizeHeader(headerJson: string): string;
  canonicalizeJcs(jsonStr: string, stripMutable?: boolean, rejectFloats?: boolean): string;
  buildDmu1Frame(mutableJsonBytes: Uint8Array, capacity: number): Uint8Array;
  parseDmu1Frame(frameBytes: Uint8Array): string;
  validateV2OpenDnamLayout(containerBytes: Uint8Array): string;
  stripV2OpenDnamContainer(containerBytes: Uint8Array, targetTier: number): Uint8Array;
  buildReferenceCheckpoint(includeCalibStats: boolean): Uint8Array;
  verifyStemArena4Crc32c(
    arenaBytes: Uint8Array,
    stemByteOffset: number,
    stemByteLen: number,
    expectedCrc32c: number,
  ): void;
  verifyArena5CalibStats(arenaBytes: Uint8Array, arena5Offset: number): number;
  validateV2DnckCheckpoint(ckptBytes: Uint8Array): {
    readonly retainedTier: number;
    readonly bundleEnds: readonly number[];
    readonly mutableBoundary: number;
  };
  exportRuntimeDnamFromCkpt(ckptBytes: Uint8Array): Uint8Array;
  uploUPackedLen(nDim: number): number;
  packUploUF64Into(denseFlat: Float64Array, nDim: number, outPacked: Float64Array): void;
  unpackUploUF64Into(packedU: Float64Array, nTier: number, outDense: Float64Array): void;

  // 3. Tempo & ZOH
  computeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): Float64Array;
  controlStrideForRate(sampleRateHz: number): number;
  controlTickRatio(sampleRateHz: number): number;
  evalSubdivisionPhase(
    beatMod12: number,
    withinBeatQ32: number,
    subdivision: string,
    offsetQ32: number,
  ): readonly [number, number, number];
  eval3NodeCalibration(anchors3: Float64Array, u: number): number;
  invertTempoCalibration(
    declaredMin: number,
    declaredMax: number,
    safeMin: number,
    safeMax: number,
    anchors3: Float64Array,
    target: number,
    flushSubnormals: boolean,
  ): Float64Array;

  // 4. Provenance, Sealing & Watermarking
  verifyContainerProvenance(
    headerJson: string,
    plaintextArena: Uint8Array,
    hasSignaturesFlag: boolean,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): string;
  enforceDistributionPolicy(headerJson: string, activeProfile: string): void;
  evaluateAnticlonePolicy(
    fingerprintMatched: boolean,
    buyerWatermarkIdentified: boolean,
    independentDeviceSourceEvidence: boolean,
  ): string;
  immutableHeaderSha256(headerJson: string): string;
  sha256Hex(data: Uint8Array): string;
  hmacSha256Hex(key32: Uint8Array, message: Uint8Array): string;
  ed25519PubkeyHex(seed32: Uint8Array): string;
  ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string;
  x25519PubkeyHex(secret32: Uint8Array): string;
  buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array;
  buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array;
  signCreatorHeader(
    headerJson: string,
    masterArena: Uint8Array,
    creatorSeed32: Uint8Array,
    signerId: string,
    includeTierPrefixes: boolean,
  ): string;
  signMarketplaceReceipt(
    headerJson: string,
    buyerArena: Uint8Array,
    marketplaceSeed32: Uint8Array,
    specJson: string,
  ): string;
  signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX2551932: Uint8Array,
  ): string;
  verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string;
  buildOemCertV2SigningMessage(
    oemVendor: string,
    modelCode: string,
    deviceIdHash16: Uint8Array,
    pubkeyX2551932: Uint8Array,
    issuerPubkeyEd2551932: Uint8Array,
    assurance: number,
    attestationPolicySha25632?: Uint8Array,
    attestationEvidenceSha25632?: Uint8Array,
  ): Uint8Array;
  chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array;
  wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string;
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
  encodeDau1Envelope(specJson: string): Uint8Array;
  decodeDau1Envelope(wireBytes: Uint8Array): string;
  buildDme1Header(profile: number, recipientCount: number, ciphertextLength: number): Uint8Array;
  parseDme1Plaintext(plaintext: Uint8Array, recipientKeyDigestsHex: readonly string[]): string;
  encodeRecipientKeyIds(keyDigestsHex: readonly string[]): Uint8Array;
  parseRecipientKeyIds(data: Uint8Array, maxRecipients: number): readonly string[];
  encodeHpkeSlots(slotsJson: string): Uint8Array;
  parseHpkeSlots(data: Uint8Array, expectedCount: number, maxRecipients: number): string;
  encodeRecordDirectory(maximumTier: number, recordsJson: string): Uint8Array;
  parseRecordDirectory(data: Uint8Array, maximumTier: number): string;
  computePreviewWindow(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
  ): readonly [number, number];
  validatePreviewSwitchOptions(
    preset: string,
    alternative: string,
    observed: readonly string[],
    active: readonly string[],
    safe: readonly string[],
  ): readonly [string, string];
  validateRoyaltySplits(licenseTier: string, royaltySplitsJson: string): void;
  applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa64: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array;
  generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array;
  applyTardosWatermark(arenaBytes: Uint8Array, code7680: Float32Array, epsilon: number): Uint8Array;
  scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code7680: Float32Array,
  ): number;
  extractStructuralHash(arenaBytes: Uint8Array): Float64Array;
  structuralCosineSimilarity(a: Float64Array, b: Float64Array): number;
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
  ): string;
  extractStage3Fingerprint(
    earlyFir128b: Uint8Array,
    kautzModalBytes: Uint8Array,
    addedModes: number,
  ): Float64Array;

  // 5. Reference DSP Kernels
  catmullRomHermite(sM1: number, s0: number, s1: number, s2: number, mu: number): number;
  catmullRomHermiteSlice(
    sM1: Float64Array,
    s0: Float64Array,
    s1: Float64Array,
    s2: Float64Array,
    mu: Float64Array,
    out: Float64Array,
  ): void;
  opTCoerciveStep(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): readonly [number, number, number];
  opTTankSlice(
    inp: Float64Array,
    out: Float64Array,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void;
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
  ): readonly [number, number, number];
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
  ): readonly [number, number];
  unitRotor(phi: number): readonly [number, number];
  unitRotorSlice(phi: Float64Array, out: Float64Array): void;
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
  ): readonly [number, number, number, number];
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
  ): readonly [number, number, number];
  evalLobattoBasis(
    u: number,
    anchors3: Float64Array,
    degree: number,
  ): readonly [number, number, number];
  evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors3: Float64Array,
    degree: number,
  ): void;
  evalRank4Fm(
    features: Float64Array,
    vFactorsFlat: Float64Array,
  ): readonly [number, number, number, number];
  evalRank4FmSlice(
    featuresFlat: Float64Array,
    nFrames: number,
    dDim: number,
    vFactorsFlat: Float64Array,
    out: Float64Array,
  ): void;
  evalChebyshevLfo(cosPhi: number): readonly [number, number, number, number];
  evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void;
  evalTrustRegionSoftclip(z: number, mh: number): number;
  evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void;
  cubicLagrangeWeights(mu: number): readonly [number, number, number, number];
  cubicLagrangeWeightsSlice(mu: Float64Array, out: Float64Array): void;
  givensHouseholder8ch(x8: Float64Array, cosPsi: number, sinPsi: number): Float64Array;
  givensHouseholder8chSlice(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
  ): void;
  givensHouseholder16ch(
    x16: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos8: Float64Array,
    crossSin8: Float64Array,
  ): Float64Array;
  givensHouseholder16chSlice(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos8: Float64Array,
    crossSin8: Float64Array,
  ): void;
}
