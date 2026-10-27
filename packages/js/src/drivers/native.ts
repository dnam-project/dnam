/**
 * Node.js N-API v8 (`napi-rs` v3) native host driver (`NativeDriver`) and zero-setup
 * synchronous `dnam_engine.wasm` loader for Node.js.
 */

import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DnamError, rethrowDnamError } from "../errors.js";
import type { PreambleSnapshot, TelemetrySnapshot, ZohParamsSnapshot } from "../schema/runtime.js";
import type { DnamProtectionProfile } from "../schema/signatures.js";
import { registerNativeDriver, registerNodeWasmAutoLoader } from "./index.js";
import type {
  DnamControlSpaceHandle,
  DnamDriver,
  DnamEngineHandle,
  RawClampResult,
  ValidatedContainerRaw,
} from "./types.js";
import { getWasmDriver, hasWasmDriver, initDnamWasmSync, type WasmDriver } from "./wasm.js";

interface NativeTelemetryBinding {
  activeTier: number;
  backend: string;
  sampleRateHz: number;
  core2Active: boolean;
  lfoEnvActive: boolean;
  stereoOut: boolean;
  activeClampMask: number;
  controlDecimationHost: number;
  maxAdaaDrive: number;
  maxPoleRadius: number;
  maxEffectiveLoopGain: number;
  tailTimeMs: number;
  zHead: number[];
  envStates: number[];
  lfoHarmonics: number[];
  clampedControls: number[];
  rhoSr: number;
  rhoCtrl: number;
  rAcHost: number;
  rDcHost: number;
  tauRefSamplesHost: number;
  switchXfadeSamplesHost: number;
}

interface NativeEngineInstance {
  reload(containerBytes: Uint8Array, tier?: number | null, sampleRateHz?: number | null): void;
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
    bufferSize?: number | null,
  ): void;
  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number | null,
  ): void;
  processMono(monoIn: Float32Array, monoOut: Float32Array, bufferSize?: number | null): void;
  processAsync(
    leftIn: Float32Array,
    rightIn?: Float32Array | null,
    bufferSize?: number | null,
  ): Promise<Float32Array[]>;
  telemetry(): NativeTelemetryBinding;
  telemetryInto(out: Float64Array): void;
}

interface NativeControlSpaceInstance {
  clamp(rawNumericValues: Float32Array): {
    clampedNumeric: Float32Array;
    normalizedU: Float32Array;
    wasClamped: boolean[];
  };
  clampInto(
    rawNumericValues: Float32Array,
    clampedOut: Float32Array,
    normOut: Float32Array,
  ): number;
}

interface NativeValidatedContainerBinding {
  magic: string;
  versionMajor: number;
  versionMinor: number;
  flagsBits: number;
  numControls: number;
  ringBufferPow2Log2: number;
  sampleRateTrainHz: number;
  jsonHeaderByteLen: number;
  tensorArenaOffset: number;
  tensorArenaByteLen: number;
  arenaOffsetLiteEnd: number;
  arenaOffsetFullEnd: number;
  arenaOffsetUltraEnd: number;
  arenaCrc32cLite: number;
  arenaCrc32cFull: number;
  arenaCrc32cUltra: number;
  numCkptStems: number;
  controlDecimation: number;
  headSvdRank: number;
  tailTimeMs: number;
  headerCrc32c: number;
  core2Active: boolean;
  lfoEnvActive: boolean;
  stereoOut: boolean;
  verificationStatus: string;
  protectionProfile: string;
  headerJson: string;
  arenaBytes: Uint8Array;
}

interface NativeZohParamsBinding {
  sampleRateHostHz: number;
  rhoSr: number;
  controlDecimationHost: number;
  rhoCtrl: number;
  rAcHost: number;
  rDcHost: number;
  tauRefSamplesHost: number;
  switchXfadeSamplesHost: number;
}

interface NativeAddonExports {
  NativeDnamEngine: new (
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string | null,
  ) => NativeEngineInstance;
  NativeControlSpace: new (headerJson: string) => NativeControlSpaceInstance;
  nativeValidateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string | null,
    deviceSecret32?: Uint8Array | null,
    trustedMarketplacePubkeyHex?: string | null,
  ): NativeValidatedContainerBinding;
  nativeBuildContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    plaintextArenaCrcSource?: Uint8Array | null,
    encryptedArena?: boolean,
  ): Uint8Array;
  nativeBuildReferenceContainer(core2Active: boolean, isCheckpoint: boolean): Uint8Array;
  nativeUpdateHeaderInPlace(containerBytes: Uint8Array, updatedHeaderJson: string): Uint8Array;
  nativeComputeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    controlDecimationTrain: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): NativeZohParamsBinding;
  nativeComputeZohInto(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    controlDecimationTrain: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
    out: Float64Array,
  ): void;
  nativePackUploU(denseFlat: Float64Array, nDim: number): Float64Array;
  nativeUnpackUploU(packedU: Float64Array, nTier: number): Float64Array;
  nativeCrc32C(data: Uint8Array): number;

  nativePolyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number;
  nativePolyAdaaStep(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
  ): number[];
  nativePolyAdaaStepInto(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    out: Float64Array,
  ): void;
  nativePolyAdaaScan(
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
  ): number[];

  nativeSelectiveRglruStep(
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
  ): number[];
  nativeSelectiveRglruStepInto(
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
    out: Float64Array,
  ): void;
  nativeSelectiveRglruScan(
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
  ): number[];

  nativeUnitRotor(phi: number): number[];
  nativeUnitRotorInto(phi: number, out: Float64Array): void;
  nativeUnitRotorSlice(phi: Float64Array, out: Float64Array): void;
  nativeCubicLagrangeWeights(mu: number): number[];
  nativeCubicLagrangeWeightsInto(mu: number, out: Float64Array): void;
  nativeGivensHouseholderMix8Ch(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array;
  nativeMufdnCoerciveStep(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): number[];
  nativeMufdnCoerciveStepInto(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
    out: Float64Array,
  ): void;
  nativeMufdnSlice(
    inp: Float64Array,
    out: Float64Array,
    delaySamples: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void;

  nativeEvalLobattoBasis(u: number, anchors: Float64Array, degree: number): number[];
  nativeEvalLobattoBasisInto(
    u: number,
    anchors: Float64Array,
    degree: number,
    out: Float64Array,
  ): void;
  nativeEvalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void;
  nativeEvalRank4Fm(features: Float64Array, vFactorsFlat: Float64Array): number[];
  nativeEvalChebyshevLfo(cosPhi: number): number[];
  nativeEvalChebyshevLfoInto(cosPhi: number, out: Float64Array): void;
  nativeEvalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void;
  nativeEvalTrustRegionSoftclip(z: number, mh: number): number;
  nativeEvalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void;

  nativeImmutableHeaderSha256(headerJson: string): string;
  nativeSha256Hex(data: Uint8Array): string;
  nativeBuildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array;
  nativeBuildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array;
  nativeChacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array;
  nativeApplyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa48: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array;
  nativeGenerateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array;
  nativeApplyTardosWatermark(
    arenaBytes: Uint8Array,
    code4608: Float32Array,
    epsilon: number,
  ): Uint8Array;
  nativeScoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code4608: Float32Array,
  ): number;
  nativeExtractStructuralHash(arenaBytes: Uint8Array): Float64Array;
  nativeStructuralCosineSimilarity(a: Float64Array, b: Float64Array): number;
  nativeEd25519PubkeyHex(seed32: Uint8Array): string;
  nativeEd25519SignHex(seed32: Uint8Array, message: Uint8Array): string;
  nativeX25519PubkeyHex(secret32: Uint8Array): string;
  nativeWrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string;
  nativeSignOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX25519_32: Uint8Array,
  ): string;
  nativeVerifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string;
  nativeSealContainer(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array;
  nativeUnsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string | null,
    deviceSecret32?: Uint8Array | null,
    trustedMarketplacePubkeyHex?: string | null,
  ): Uint8Array;
}

function findDistFile(fileName: string): string {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    join(currentDir, fileName),
    resolve(currentDir, "..", "dist", fileName),
    resolve(currentDir, "..", "..", "dist", fileName),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new DnamError(
    `Could not locate native artifact '${fileName}' in ${candidates.join(", ")}. Run 'pnpm run build:rust' first.`,
  );
}

function callNative<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    rethrowDnamError(err);
  }
}

const NATIVE_BACKEND_NAMES: readonly string[] = [
  "scalar",
  "arm_neon",
  "arm_sme",
  "x86_simd",
  "wasm_simd128",
];

class NativeEngineHandleWrapper implements DnamEngineHandle {
  private inner: NativeEngineInstance | null;
  private readonly telemetryBuf = new Float64Array(50);

  public constructor(inner: NativeEngineInstance) {
    this.inner = inner;
  }

  private get engine(): NativeEngineInstance {
    if (this.inner === null) {
      throw new DnamError("DnamEngine instance has already been disposed");
    }
    return this.inner;
  }

  public reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    callNative(() => {
      this.engine.reload(containerBytes, tier ?? null, sampleRateHz ?? null);
    });
  }

  public get backend(): string {
    return callNative(() => this.engine.backend);
  }

  public get activeTier(): number {
    return callNative(() => this.engine.activeTier);
  }

  public set activeTier(tier: number) {
    callNative(() => {
      this.engine.activeTier = tier;
    });
  }

  public get sampleRateHz(): number {
    return callNative(() => this.engine.sampleRateHz);
  }

  public set sampleRateHz(sampleRateHz: number) {
    callNative(() => {
      this.engine.sampleRateHz = sampleRateHz;
    });
  }

  public setControlValue(ctrlIdx: number, value: number): void {
    this.engine.setControlValue(ctrlIdx, value);
  }

  public setControlImmediate(ctrlIdx: number, value: number): void {
    this.engine.setControlImmediate(ctrlIdx, value);
  }

  public reset(): void {
    this.engine.reset();
  }

  public processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void {
    callNative(() => {
      this.engine.processStereo(leftIn, rightIn, leftOut, rightOut, bufferSize ?? null);
    });
  }

  public processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void {
    callNative(() => {
      this.engine.processMonoToStereo(monoIn, leftOut, rightOut, bufferSize ?? null);
    });
  }

  public processMono(monoIn: Float32Array, monoOut: Float32Array, bufferSize?: number): void {
    callNative(() => {
      this.engine.processMono(monoIn, monoOut, bufferSize ?? null);
    });
  }

  public async processAsync(
    leftIn: Float32Array,
    rightIn?: Float32Array,
    bufferSize?: number,
  ): Promise<[Float32Array, Float32Array]> {
    const pair = await this.engine.processAsync(leftIn, rightIn ?? null, bufferSize ?? null);
    return [pair[0] ?? new Float32Array(0), pair[1] ?? new Float32Array(0)];
  }

  public telemetry(): TelemetrySnapshot {
    const v = this.telemetryBuf;
    this.engine.telemetryInto(v);
    const clampedControls = new Array<number>(16);
    for (let i = 0; i < 16; i++) {
      clampedControls[i] = v[28 + i] ?? 0;
    }
    return {
      activeTier: v[0] ?? 0,
      backend: NATIVE_BACKEND_NAMES[v[1] ?? 0] ?? "arm_neon",
      sampleRateHz: v[2] ?? 48_000,
      core2Active: (v[3] ?? 0) !== 0,
      lfoEnvActive: (v[4] ?? 0) !== 0,
      stereoOut: (v[5] ?? 0) !== 0,
      activeClampMask: v[6] ?? 0,
      controlDecimationHost: v[7] ?? 16,
      maxAdaaDrive: v[8] ?? 0,
      maxPoleRadius: v[9] ?? 0,
      maxEffectiveLoopGain: v[10] ?? 0,
      tailTimeMs: v[11] ?? 0,
      zHead: [v[12] ?? 0, v[13] ?? 0, v[14] ?? 0, v[15] ?? 0],
      envStates: [v[16] ?? 0, v[17] ?? 0, v[18] ?? 0, v[19] ?? 0],
      lfoHarmonics: [
        v[20] ?? 0,
        v[21] ?? 0,
        v[22] ?? 0,
        v[23] ?? 0,
        v[24] ?? 0,
        v[25] ?? 0,
        v[26] ?? 0,
        v[27] ?? 0,
      ],
      clampedControls,
      rhoSr: v[44] ?? 1,
      rhoCtrl: v[45] ?? 1,
      rAcHost: v[46] ?? 0.9993,
      rDcHost: v[47] ?? 0.995,
      tauRefSamplesHost: v[48] ?? 64,
      switchXfadeSamplesHost: v[49] ?? 128,
    };
  }

  public dispose(): void {
    this.inner = null;
  }
}

class NativeControlSpaceWrapper implements DnamControlSpaceHandle {
  private inner: NativeControlSpaceInstance | null;
  private readonly clampedBuf = new Float32Array(16);
  private readonly normBuf = new Float32Array(16);
  private readonly wasClampedBuf: boolean[] = new Array<boolean>(16).fill(false);

  public constructor(inner: NativeControlSpaceInstance) {
    this.inner = inner;
  }

  public clamp(rawNumericValues: Float32Array): RawClampResult {
    if (this.inner === null) {
      throw new DnamError("ControlSpace instance has already been disposed");
    }
    const n = Math.min(16, rawNumericValues.length);
    const mask = this.inner.clampInto(rawNumericValues, this.clampedBuf, this.normBuf);
    this.wasClampedBuf.length = n;
    for (let i = 0; i < n; i++) {
      this.wasClampedBuf[i] = (mask & (1 << i)) !== 0;
    }
    return {
      clampedNumeric: this.clampedBuf.subarray(0, n),
      normalizedU: this.normBuf.subarray(0, n),
      wasClamped: this.wasClampedBuf,
    };
  }

  public dispose(): void {
    this.inner = null;
  }
}

export class NativeDriver implements DnamDriver {
  public readonly kind = "native" as const;
  private readonly addon: NativeAddonExports;
  private readonly scalarBuf = new Float64Array(16);

  public constructor(addon: NativeAddonExports) {
    this.addon = addon;
  }

  public createEngine(
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string,
  ): DnamEngineHandle {
    return callNative(() => {
      const instance = new this.addon.NativeDnamEngine(
        containerBytes,
        tier,
        sampleRateHz,
        backend ?? null,
      );
      return new NativeEngineHandleWrapper(instance);
    });
  }

  public createControlSpace(headerJson: string): DnamControlSpaceHandle {
    return callNative(() => {
      const instance = new this.addon.NativeControlSpace(headerJson);
      return new NativeControlSpaceWrapper(instance);
    });
  }

  public validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): ValidatedContainerRaw {
    return callNative(() => {
      const v = this.addon.nativeValidateContainer(
        containerBytes,
        deviceIdHash ?? null,
        deviceSecret32 ?? null,
        trustedMarketplacePubkeyHex ?? null,
      );
      const preamble: PreambleSnapshot = {
        magic: v.magic,
        versionMajor: v.versionMajor,
        versionMinor: v.versionMinor,
        flagsBits: v.flagsBits,
        numControls: v.numControls,
        ringBufferPow2Log2: v.ringBufferPow2Log2,
        sampleRateTrainHz: v.sampleRateTrainHz,
        jsonHeaderByteLen: v.jsonHeaderByteLen,
        tensorArenaOffset: v.tensorArenaOffset,
        tensorArenaByteLen: v.tensorArenaByteLen,
        arenaOffsetLiteEnd: v.arenaOffsetLiteEnd,
        arenaOffsetFullEnd: v.arenaOffsetFullEnd,
        arenaOffsetUltraEnd: v.arenaOffsetUltraEnd,
        arenaCrc32cLite: v.arenaCrc32cLite >>> 0,
        arenaCrc32cFull: v.arenaCrc32cFull >>> 0,
        arenaCrc32cUltra: v.arenaCrc32cUltra >>> 0,
        numCkptStems: v.numCkptStems,
        controlDecimation: v.controlDecimation,
        headSvdRank: v.headSvdRank,
        tailTimeMs: v.tailTimeMs,
        headerCrc32c: v.headerCrc32c >>> 0,
        core2Active: v.core2Active,
        lfoEnvActive: v.lfoEnvActive,
        stereoOut: v.stereoOut,
      };
      const verificationStatus =
        v.verificationStatus === "CreatorVerified" || v.verificationStatus === "MarketplaceVerified"
          ? v.verificationStatus
          : "Unsigned";
      const protectionProfile: DnamProtectionProfile =
        v.protectionProfile === "device_allowlisted" || v.protectionProfile === "enclave_sealed"
          ? v.protectionProfile
          : "open";
      return {
        preamble,
        verificationStatus,
        protectionProfile,
        headerJson: v.headerJson,
        arenaBytes: v.arenaBytes,
      };
    });
  }

  public buildContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    plaintextArenaCrcSource?: Uint8Array,
    encryptedArena = false,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeBuildContainer(
        headerJson,
        arenaBytes,
        plaintextArenaCrcSource ?? null,
        encryptedArena,
      ),
    );
  }

  public buildReferenceContainer(core2Active: boolean, isCheckpoint: boolean): Uint8Array {
    return callNative(() => this.addon.nativeBuildReferenceContainer(core2Active, isCheckpoint));
  }

  public updateHeaderInPlace(containerBytes: Uint8Array, updatedHeaderJson: string): Uint8Array {
    return callNative(() =>
      this.addon.nativeUpdateHeaderInPlace(containerBytes, updatedHeaderJson),
    );
  }

  public computeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    controlDecimationTrain: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): ZohParamsSnapshot {
    const v = this.scalarBuf;
    callNative(() => {
      this.addon.nativeComputeZohInto(
        sampleRateTrainHz,
        sampleRateHostHz,
        controlDecimationTrain,
        rAcTrain,
        rDcTrain,
        tauRefTrain,
        switchXfadeTrain,
        v,
      );
    });
    return {
      sampleRateHostHz: v[0] ?? sampleRateHostHz,
      rhoSr: v[1] ?? 1.0,
      controlDecimationHost: v[2] ?? 16,
      rhoCtrl: v[3] ?? 1.0,
      rAcHost: v[4] ?? 0.9993,
      rDcHost: v[5] ?? 0.995,
      tauRefSamplesHost: v[6] ?? 64.0,
      switchXfadeSamplesHost: v[7] ?? 128,
    };
  }

  public packUploU(denseFlat: Float64Array, nDim: number): Float64Array {
    return callNative(() => this.addon.nativePackUploU(denseFlat, nDim));
  }

  public unpackUploU(packedU: Float64Array, nTier: number): Float64Array {
    return callNative(() => this.addon.nativeUnpackUploU(packedU, nTier));
  }

  public crc32c(data: Uint8Array): number {
    return this.addon.nativeCrc32C(data) >>> 0;
  }

  public polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number {
    return this.addon.nativePolyAdaaAlgebraic(uPrev, uCurr, beta, delta);
  }

  public polyAdaaStep(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
  ): [number, number, number, number] {
    const v = this.scalarBuf;
    this.addon.nativePolyAdaaStepInto(x, bConv, alpha, beta, delta, rAc, uPrev, aPrev, mDc, v);
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0];
  }

  public polyAdaaScan(
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
  ): [number, number, number] {
    return callNative(() => {
      const st = this.addon.nativePolyAdaaScan(
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
      );
      return [st[0] ?? 0, st[1] ?? 0, st[2] ?? 0];
    });
  }

  public selectiveRgLruStep(
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
  ): [number, number, number] {
    const v = this.scalarBuf;
    this.addon.nativeSelectiveRglruStepInto(
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
      v,
    );
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
  }

  public selectiveRgLruScan(
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
  ): [number, number] {
    return callNative(() => {
      const st = this.addon.nativeSelectiveRglruScan(
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
      );
      return [st[0] ?? 0, st[1] ?? 0];
    });
  }

  public unitRotor(phi: number): [number, number] {
    const v = this.scalarBuf;
    this.addon.nativeUnitRotorInto(phi, v);
    return [v[0] ?? 1, v[1] ?? 0];
  }

  public unitRotorSlice(phi: Float64Array, out: Float64Array): void {
    callNative(() => {
      this.addon.nativeUnitRotorSlice(phi, out);
    });
  }

  public cubicLagrangeWeights(mu: number): [number, number, number, number] {
    const v = this.scalarBuf;
    this.addon.nativeCubicLagrangeWeightsInto(mu, v);
    return [v[0] ?? 0, v[1] ?? 1, v[2] ?? 0, v[3] ?? 0];
  }

  public givensHouseholderMix8ch(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array {
    return callNative(() => this.addon.nativeGivensHouseholderMix8Ch(x, cosPsi, sinPsi));
  }

  public mufdnCoerciveStep(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): [number, number, number] {
    const v = this.scalarBuf;
    this.addon.nativeMufdnCoerciveStepInto(uIn, gLin, sigmaKnee, vCeil, rDc, dcXPrev, dcYPrev, v);
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
  }

  public mufdnSlice(
    inp: Float64Array,
    out: Float64Array,
    delaySamples: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void {
    callNative(() => {
      this.addon.nativeMufdnSlice(inp, out, delaySamples, gLin, sigmaKnee, vCeil, rDc);
    });
  }

  public evalLobattoBasis(
    u: number,
    anchors: Float64Array,
    degree: number,
  ): [number, number, number] {
    const v = this.scalarBuf;
    callNative(() => {
      this.addon.nativeEvalLobattoBasisInto(u, anchors, degree, v);
    });
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
  }

  public evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void {
    callNative(() => {
      this.addon.nativeEvalLobattoBasisSlice(u, out, anchors, degree);
    });
  }

  public evalRank4Fm(
    features: Float64Array,
    vFactorsFlat: Float64Array,
  ): [number, number, number, number] {
    return callNative(() => {
      const v = this.addon.nativeEvalRank4Fm(features, vFactorsFlat);
      return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0];
    });
  }

  public evalChebyshevLfo(cosPhi: number): [number, number, number, number] {
    const v = this.scalarBuf;
    this.addon.nativeEvalChebyshevLfoInto(cosPhi, v);
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0];
  }

  public evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void {
    callNative(() => {
      this.addon.nativeEvalChebyshevLfoSlice(cosPhi, out);
    });
  }

  public evalTrustRegionSoftclip(z: number, mh: number): number {
    return this.addon.nativeEvalTrustRegionSoftclip(z, mh);
  }

  public evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void {
    callNative(() => {
      this.addon.nativeEvalTrustRegionSoftclipSlice(z, out, mh);
    });
  }

  public immutableHeaderSha256(headerJson: string): string {
    return callNative(() => this.addon.nativeImmutableHeaderSha256(headerJson));
  }

  public sha256Hex(data: Uint8Array): string {
    return this.addon.nativeSha256Hex(data);
  }

  public buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeBuildCreatorSigningMessage(immutableHeaderSha256Hex, masterArenaSha256Hex),
    );
  }

  public buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeBuildMarketplaceSigningMessage(
        immutableHeaderSha256Hex,
        masterArenaSha256Hex,
        receiptJson,
      ),
    );
  }

  public chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeChacha20XorArena(arenaBytes, key32, nonce12, startBlockCounter),
    );
  }

  public applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa48: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeApplyIsometricGaugeWatermark(arenaBytes, kappa48, qBuyerFlat16),
    );
  }

  public generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array {
    return callNative(() =>
      this.addon.nativeGenerateTardosCode(marketplaceSecret32, modelId, licenseId),
    );
  }

  public applyTardosWatermark(
    arenaBytes: Uint8Array,
    code4608: Float32Array,
    epsilon: number,
  ): Uint8Array {
    return callNative(() => this.addon.nativeApplyTardosWatermark(arenaBytes, code4608, epsilon));
  }

  public scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code4608: Float32Array,
  ): number {
    return callNative(() =>
      this.addon.nativeScoreTardosWatermark(masterArena, suspectArena, code4608),
    );
  }

  public extractStructuralHash(arenaBytes: Uint8Array): Float64Array {
    return callNative(() => this.addon.nativeExtractStructuralHash(arenaBytes));
  }

  public structuralCosineSimilarity(a: Float64Array, b: Float64Array): number {
    return callNative(() => this.addon.nativeStructuralCosineSimilarity(a, b));
  }

  public ed25519PubkeyHex(seed32: Uint8Array): string {
    return callNative(() => this.addon.nativeEd25519PubkeyHex(seed32));
  }

  public ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string {
    return callNative(() => this.addon.nativeEd25519SignHex(seed32, message));
  }

  public x25519PubkeyHex(secret32: Uint8Array): string {
    return callNative(() => this.addon.nativeX25519PubkeyHex(secret32));
  }

  public wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string {
    return callNative(() =>
      this.addon.nativeWrapDeviceCekHex(ephemeralSecret32, devicePubkey32, deviceIdHash, cek32),
    );
  }

  public signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX25519_32: Uint8Array,
  ): string {
    return callNative(() =>
      this.addon.nativeSignOemDeviceCert(
        oemSeed32,
        oemVendor,
        modelCode,
        deviceIdHash,
        pubkeyX25519_32,
      ),
    );
  }

  public verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string {
    return callNative(() => this.addon.nativeVerifyOemDeviceCert(certJson, trustedOemPubkeyHex));
  }

  public sealContainer(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeSealContainer(
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

  public unsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array {
    return callNative(() =>
      this.addon.nativeUnsealContainer(
        sealedBytes,
        deviceIdHash ?? null,
        deviceSecret32 ?? null,
        trustedMarketplacePubkeyHex ?? null,
      ),
    );
  }
}

let cachedNativeDriver: NativeDriver | null = null;

export function getNativeDriver(): NativeDriver {
  if (cachedNativeDriver !== null) {
    return cachedNativeDriver;
  }
  const addonFileName = `dnam_js.${process.platform}-${process.arch}.node`;
  const addonPath = findDistFile(addonFileName);
  const req = createRequire(import.meta.url);
  const addon = req(addonPath) as NativeAddonExports;
  cachedNativeDriver = new NativeDriver(addon);
  registerNativeDriver(cachedNativeDriver);
  return cachedNativeDriver;
}

export function ensureNodeWasmDriver(): WasmDriver {
  if (hasWasmDriver()) {
    return getWasmDriver();
  }
  const wasmPath = findDistFile("dnam_engine.wasm");
  const bytes = readFileSync(wasmPath);
  return initDnamWasmSync(bytes);
}

// Auto-register both the native N-API driver and the Node synchronous WASM fallback on import
getNativeDriver();
registerNodeWasmAutoLoader(ensureNodeWasmDriver);
