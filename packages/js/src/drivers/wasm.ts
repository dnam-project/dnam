/**
 * Browser & AudioWorklet-safe WebAssembly SIMD128 driver (`WasmDriver`) wrapping `dnam_engine.wasm`.
 * Zero `node:*` imports so it bundles cleanly into Browser and `AudioWorkletGlobalScope`.
 */

import { DnamError, mapTaggedError } from "../errors.js";
import type { PreambleSnapshot, TelemetrySnapshot, ZohParamsSnapshot } from "../schema/runtime.js";
import type { DnamProtectionProfile } from "../schema/signatures.js";
import type {
  DnamControlSpaceHandle,
  DnamDriver,
  DnamEngineHandle,
  RawClampResult,
  ValidatedContainerRaw,
} from "./types.js";

export interface DnamWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  dnam_wasm_alloc(size: number, align: number): number;
  dnam_wasm_dealloc(ptr: number, size: number, align: number): void;
  dnam_wasm_last_error_ptr(): number;
  dnam_wasm_last_error_len(): number;
  dnam_wasm_result_ptr(): number;
  dnam_wasm_result_len(): number;
  dnam_wasm_arena_ptr(): number;
  dnam_wasm_arena_len(): number;

  dnam_wasm_engine_create(
    containerPtr: number,
    containerLen: number,
    tier: number,
    sampleRateHz: number,
    backendCode: number,
  ): number;
  dnam_wasm_engine_destroy(handle: number): void;
  dnam_wasm_engine_reload(
    handle: number,
    containerPtr: number,
    containerLen: number,
    tier: number,
    sampleRateHz: number,
  ): number;
  dnam_wasm_engine_backend_code(handle: number): number;
  dnam_wasm_engine_active_tier(handle: number): number;
  dnam_wasm_engine_set_tier(handle: number, tier: number): number;
  dnam_wasm_engine_sample_rate(handle: number): number;
  dnam_wasm_engine_set_sample_rate(handle: number, sampleRateHz: number): number;
  dnam_wasm_engine_set_control_value(handle: number, ctrlIdx: number, value: number): void;
  dnam_wasm_engine_set_control_immediate(handle: number, ctrlIdx: number, value: number): void;
  dnam_wasm_engine_reset(handle: number): void;
  dnam_wasm_engine_process_stereo(
    handle: number,
    leftInPtr: number,
    rightInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    bufferSize: number,
  ): number;
  dnam_wasm_engine_process_mono_to_stereo(
    handle: number,
    monoInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    bufferSize: number,
  ): number;
  dnam_wasm_engine_process_mono(
    handle: number,
    monoInPtr: number,
    monoOutPtr: number,
    numSamples: number,
    bufferSize: number,
  ): number;
  dnam_wasm_engine_telemetry(handle: number, outF64Ptr: number): number;

  dnam_wasm_control_space_create(headerJsonPtr: number, headerJsonLen: number): number;
  dnam_wasm_control_space_destroy(handle: number): void;
  dnam_wasm_control_space_clamp(
    handle: number,
    rawPtr: number,
    numControls: number,
    clampedOutPtr: number,
    normOutPtr: number,
    wasClampedOutPtr: number,
  ): number;

  dnam_wasm_validate_container(
    containerPtr: number,
    containerLen: number,
    outF64Ptr: number,
  ): number;
  dnam_wasm_validate_container_with_key(
    containerPtr: number,
    containerLen: number,
    deviceIdPtr: number,
    deviceIdLen: number,
    deviceSecretPtr: number,
    deviceSecretLen: number,
    mktPkPtr: number,
    mktPkLen: number,
    outF64Ptr: number,
  ): number;
  dnam_wasm_build_container(
    headerJsonPtr: number,
    headerJsonLen: number,
    arenaPtr: number,
    arenaLen: number,
    crcSrcPtr: number,
    crcSrcLen: number,
    encryptedArena: number,
  ): number;
  dnam_wasm_build_reference_container(core2Active: number, isCheckpoint: number): number;
  dnam_wasm_update_header_in_place(
    containerPtr: number,
    containerLen: number,
    headerJsonPtr: number,
    headerJsonLen: number,
  ): number;
  dnam_wasm_compute_zoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    controlDecimationTrain: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
    outF64Ptr: number,
  ): number;
  dnam_wasm_pack_uplo_u(
    densePtr: number,
    denseLen: number,
    nDim: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_unpack_uplo_u(
    packedPtr: number,
    packedLen: number,
    nTier: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_crc32c(dataPtr: number, dataLen: number): number;

  dnam_wasm_poly_adaa_algebraic(uPrev: number, uCurr: number, beta: number, delta: number): number;
  dnam_wasm_poly_adaa_step(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    out4Ptr: number,
  ): void;
  dnam_wasm_poly_adaa_scan(
    inPtr: number,
    outPtr: number,
    len: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    reset: number,
    stateOut3Ptr: number,
  ): number;
  dnam_wasm_selective_rglru_step(
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
    out3Ptr: number,
  ): void;
  dnam_wasm_selective_rglru_scan(
    inPtr: number,
    outPtr: number,
    len: number,
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
    reset: number,
    stateOut2Ptr: number,
  ): number;
  dnam_wasm_unit_rotor(phi: number, out2Ptr: number): void;
  dnam_wasm_unit_rotor_slice(phiPtr: number, len: number, out2nPtr: number): number;
  dnam_wasm_cubic_lagrange_weights(mu: number, out4Ptr: number): void;
  dnam_wasm_givens_householder_mix_8ch(
    xPtr: number,
    xLen: number,
    cosPsi: number,
    sinPsi: number,
    out8Ptr: number,
  ): number;
  dnam_wasm_mufdn_coercive_step(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
    out3Ptr: number,
  ): void;
  dnam_wasm_mufdn_slice(
    inPtr: number,
    outPtr: number,
    len: number,
    delaySamples: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): number;
  dnam_wasm_eval_lobatto_basis(
    u: number,
    anchorsPtr: number,
    anchorsLen: number,
    degree: number,
    out3Ptr: number,
  ): number;
  dnam_wasm_eval_lobatto_basis_slice(
    uPtr: number,
    len: number,
    out3nPtr: number,
    anchorsPtr: number,
    anchorsLen: number,
    degree: number,
  ): number;
  dnam_wasm_eval_rank4_fm(
    featuresPtr: number,
    featuresLen: number,
    vFactorsPtr: number,
    vFactorsLen: number,
    out4Ptr: number,
  ): number;
  dnam_wasm_eval_chebyshev_lfo(cosPhi: number, out4Ptr: number): void;
  dnam_wasm_eval_chebyshev_lfo_slice(cosPhiPtr: number, len: number, out4nPtr: number): number;
  dnam_wasm_eval_trust_region_softclip(z: number, mh: number): number;
  dnam_wasm_eval_trust_region_softclip_slice(
    zPtr: number,
    len: number,
    outPtr: number,
    mh: number,
  ): number;

  dnam_wasm_immutable_header_sha256(headerJsonPtr: number, headerJsonLen: number): number;
  dnam_wasm_sha256_hex(dataPtr: number, dataLen: number): number;
  dnam_wasm_build_creator_signing_message(
    hdrShaPtr: number,
    hdrShaLen: number,
    arenaShaPtr: number,
    arenaShaLen: number,
  ): number;
  dnam_wasm_build_marketplace_signing_message(
    hdrShaPtr: number,
    hdrShaLen: number,
    arenaShaPtr: number,
    arenaShaLen: number,
    receiptJsonPtr: number,
    receiptJsonLen: number,
  ): number;
  dnam_wasm_chacha20_xor_arena(
    arenaPtr: number,
    arenaLen: number,
    keyPtr: number,
    keyLen: number,
    noncePtr: number,
    nonceLen: number,
    startBlockCounter: number,
  ): number;
  dnam_wasm_apply_isometric_gauge_watermark(
    arenaPtr: number,
    arenaLen: number,
    kappaPtr: number,
    kappaLen: number,
    qPtr: number,
    qLen: number,
  ): number;
  dnam_wasm_generate_tardos_code(
    secretPtr: number,
    secretLen: number,
    modelIdPtr: number,
    modelIdLen: number,
    licenseIdPtr: number,
    licenseIdLen: number,
    out4608Ptr: number,
  ): number;
  dnam_wasm_apply_tardos_watermark(
    arenaPtr: number,
    arenaLen: number,
    codePtr: number,
    codeLen: number,
    epsilon: number,
  ): number;
  dnam_wasm_score_tardos_watermark(
    masterPtr: number,
    masterLen: number,
    suspectPtr: number,
    suspectLen: number,
    codePtr: number,
    codeLen: number,
    outScorePtr: number,
  ): number;
  dnam_wasm_extract_structural_hash(arenaPtr: number, arenaLen: number, out160Ptr: number): number;
  dnam_wasm_structural_cosine_similarity(
    aPtr: number,
    aLen: number,
    bPtr: number,
    bLen: number,
    outCosPtr: number,
  ): number;
  dnam_wasm_ed25519_pubkey_hex(seedPtr: number, seedLen: number): number;
  dnam_wasm_ed25519_sign_hex(
    seedPtr: number,
    seedLen: number,
    msgPtr: number,
    msgLen: number,
  ): number;
  dnam_wasm_x25519_pubkey_hex(secretPtr: number, secretLen: number): number;
  dnam_wasm_wrap_device_cek_hex(
    ephPtr: number,
    ephLen: number,
    devPkPtr: number,
    devPkLen: number,
    devIdPtr: number,
    devIdLen: number,
    cekPtr: number,
    cekLen: number,
  ): number;
  dnam_wasm_sign_oem_device_cert(
    oemSeedPtr: number,
    oemSeedLen: number,
    vendorPtr: number,
    vendorLen: number,
    modelPtr: number,
    modelLen: number,
    devIdPtr: number,
    devIdLen: number,
    pkX25519Ptr: number,
    pkX25519Len: number,
  ): number;
  dnam_wasm_verify_oem_device_cert(
    certJsonPtr: number,
    certJsonLen: number,
    oemPkPtr: number,
    oemPkLen: number,
  ): number;
  dnam_wasm_seal_container(
    innerPtr: number,
    innerLen: number,
    certsJsonPtr: number,
    certsJsonLen: number,
    oemPksJsonPtr: number,
    oemPksJsonLen: number,
    mktSeedPtr: number,
    mktSeedLen: number,
    ephSecPtr: number,
    ephSecLen: number,
    cekPtr: number,
    cekLen: number,
    noncePtr: number,
    nonceLen: number,
  ): number;
  dnam_wasm_unseal_container(
    sealedPtr: number,
    sealedLen: number,
    devIdPtr: number,
    devIdLen: number,
    devSecPtr: number,
    devSecLen: number,
    mktPkPtr: number,
    mktPkLen: number,
  ): number;
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const BACKEND_NAMES: readonly string[] = [
  "scalar",
  "arm_neon",
  "arm_sme",
  "x86_simd",
  "wasm_simd128",
];

const BACKEND_CODES: Record<string, number> = {
  auto: 255,
  wasm: 255,
  scalar: 0,
  neon: 1,
  arm_neon: 1,
  arm64_neon: 1,
  arm_sme: 2,
  armsme: 2,
  sme: 2,
  x86_simd: 3,
  x86_avx2_fma: 3,
  avx2: 3,
  x86_sse42: 3,
  wasm_simd128: 4,
};

function backendCodeToName(code: number): string {
  return BACKEND_NAMES[code] ?? "wasm_simd128";
}

function backendNameToCode(backend?: string): number {
  const b = (backend ?? "auto").toLowerCase();
  const code = BACKEND_CODES[b];
  if (code === undefined) {
    throw mapTaggedError(`[DNAM-VALIDATION] Unknown backend kind: '${backend ?? ""}'`);
  }
  return code;
}

interface EngineCleanupToken {
  exports: DnamWasmExports;
  handle: number;
  scratchPtr: number;
  scratchByteSize: number;
  telemetryPtr: number;
}

const engineFinalizer =
  typeof FinalizationRegistry !== "undefined"
    ? new FinalizationRegistry<EngineCleanupToken>((token) => {
        if (token.handle !== 0) {
          token.exports.dnam_wasm_engine_destroy(token.handle);
        }
        if (token.scratchPtr !== 0) {
          token.exports.dnam_wasm_dealloc(token.scratchPtr, token.scratchByteSize, 64);
        }
        if (token.telemetryPtr !== 0) {
          token.exports.dnam_wasm_dealloc(token.telemetryPtr, 50 * 8, 64);
        }
      })
    : null;

interface ControlSpaceCleanupToken {
  exports: DnamWasmExports;
  handle: number;
  scratchPtr: number;
}

const controlSpaceFinalizer =
  typeof FinalizationRegistry !== "undefined"
    ? new FinalizationRegistry<ControlSpaceCleanupToken>((token) => {
        if (token.handle !== 0) {
          token.exports.dnam_wasm_control_space_destroy(token.handle);
        }
        if (token.scratchPtr !== 0) {
          token.exports.dnam_wasm_dealloc(token.scratchPtr, 256, 64);
        }
      })
    : null;

class WasmEngineHandle implements DnamEngineHandle {
  private readonly exports: DnamWasmExports;
  private readonly throwLastError: () => never;
  private handle: number;
  private scratchPtr: number;
  private scratchSamplesCap: number;
  private readonly telemetryPtr: number;
  private readonly cleanupToken: EngineCleanupToken;
  private cachedMemBuf: ArrayBuffer | null = null;
  private cachedScratchF32: Float32Array | null = null;
  private cachedTelemetryF64: Float64Array | null = null;

  public constructor(
    exports: DnamWasmExports,
    throwLastError: () => never,
    handle: number,
    initialSamplesCap = 4096,
  ) {
    this.exports = exports;
    this.throwLastError = throwLastError;
    this.handle = handle;
    this.scratchSamplesCap = initialSamplesCap;
    const scratchByteSize = initialSamplesCap * 4 * 4;
    this.scratchPtr = exports.dnam_wasm_alloc(scratchByteSize, 64);
    this.telemetryPtr = exports.dnam_wasm_alloc(50 * 8, 64);
    this.cleanupToken = {
      exports,
      handle,
      scratchPtr: this.scratchPtr,
      scratchByteSize,
      telemetryPtr: this.telemetryPtr,
    };
    engineFinalizer?.register(this, this.cleanupToken, this);
  }

  private ensureDisposedNot(): void {
    if (this.handle === 0) {
      throw new DnamError("DnamEngine instance has already been disposed");
    }
  }

  private checkRc(rc: number): void {
    if (rc < 0) {
      this.throwLastError();
    }
  }

  private ensureScratchCapacity(numSamples: number): number {
    if (numSamples <= this.scratchSamplesCap) {
      return this.scratchPtr;
    }
    const nextCap = Math.max(numSamples, this.scratchSamplesCap * 2);
    const oldByteSize = this.scratchSamplesCap * 4 * 4;
    const newByteSize = nextCap * 4 * 4;
    const newPtr = this.exports.dnam_wasm_alloc(newByteSize, 64);
    if (newPtr === 0) {
      throw new DnamError("Out of WASM linear memory allocating audio scratch arena");
    }
    this.exports.dnam_wasm_dealloc(this.scratchPtr, oldByteSize, 64);
    this.scratchPtr = newPtr;
    this.scratchSamplesCap = nextCap;
    this.cachedScratchF32 = null;
    this.cleanupToken.scratchPtr = newPtr;
    this.cleanupToken.scratchByteSize = newByteSize;
    return newPtr;
  }

  private getScratchF32View(): Float32Array {
    const buf = this.exports.memory.buffer;
    if (this.cachedScratchF32 === null || this.cachedMemBuf !== buf) {
      this.cachedMemBuf = buf;
      this.cachedScratchF32 = new Float32Array(buf, this.scratchPtr, this.scratchSamplesCap * 4);
      this.cachedTelemetryF64 = new Float64Array(buf, this.telemetryPtr, 50);
    }
    return this.cachedScratchF32;
  }

  private getTelemetryF64View(): Float64Array {
    const buf = this.exports.memory.buffer;
    if (this.cachedTelemetryF64 === null || this.cachedMemBuf !== buf) {
      this.cachedMemBuf = buf;
      this.cachedScratchF32 = new Float32Array(buf, this.scratchPtr, this.scratchSamplesCap * 4);
      this.cachedTelemetryF64 = new Float64Array(buf, this.telemetryPtr, 50);
    }
    return this.cachedTelemetryF64;
  }

  public reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    this.ensureDisposedNot();
    const len = Math.max(1, containerBytes.byteLength);
    const ptr = this.exports.dnam_wasm_alloc(len, 64);
    try {
      new Uint8Array(this.exports.memory.buffer, ptr, containerBytes.byteLength).set(
        containerBytes,
      );
      this.checkRc(
        this.exports.dnam_wasm_engine_reload(
          this.handle,
          ptr,
          containerBytes.byteLength,
          tier ?? -1,
          sampleRateHz ?? 0,
        ),
      );
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, len, 64);
    }
  }

  public get backend(): string {
    this.ensureDisposedNot();
    return backendCodeToName(this.exports.dnam_wasm_engine_backend_code(this.handle));
  }

  public get activeTier(): number {
    this.ensureDisposedNot();
    return this.exports.dnam_wasm_engine_active_tier(this.handle);
  }

  public set activeTier(tier: number) {
    this.ensureDisposedNot();
    this.checkRc(this.exports.dnam_wasm_engine_set_tier(this.handle, tier));
  }

  public get sampleRateHz(): number {
    this.ensureDisposedNot();
    return this.exports.dnam_wasm_engine_sample_rate(this.handle);
  }

  public set sampleRateHz(sampleRateHz: number) {
    this.ensureDisposedNot();
    this.checkRc(this.exports.dnam_wasm_engine_set_sample_rate(this.handle, sampleRateHz));
  }

  public setControlValue(ctrlIdx: number, value: number): void {
    this.ensureDisposedNot();
    this.exports.dnam_wasm_engine_set_control_value(this.handle, ctrlIdx, value);
  }

  public setControlImmediate(ctrlIdx: number, value: number): void {
    this.ensureDisposedNot();
    this.exports.dnam_wasm_engine_set_control_immediate(this.handle, ctrlIdx, value);
  }

  public reset(): void {
    this.ensureDisposedNot();
    this.exports.dnam_wasm_engine_reset(this.handle);
  }

  public processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void {
    this.ensureDisposedNot();
    const n = leftIn.length;
    if (rightIn.length !== n || leftOut.length !== n || rightOut.length !== n) {
      throw mapTaggedError(
        "[DNAM-VALIDATION] leftIn, rightIn, leftOut, and rightOut must have identical length",
      );
    }
    if (n === 0) {
      return;
    }
    const basePtr = this.ensureScratchCapacity(n);
    const byteStride = n * 4;
    const liPtr = basePtr;
    const riPtr = basePtr + byteStride;
    const loPtr = basePtr + byteStride * 2;
    const roPtr = basePtr + byteStride * 3;

    const inView = this.getScratchF32View();
    inView.set(leftIn, 0);
    inView.set(rightIn, n);

    this.checkRc(
      this.exports.dnam_wasm_engine_process_stereo(
        this.handle,
        liPtr,
        riPtr,
        loPtr,
        roPtr,
        n,
        bufferSize ?? 0,
      ),
    );

    const outView = this.getScratchF32View();
    leftOut.set(outView.subarray(n * 2, n * 3));
    rightOut.set(outView.subarray(n * 3, n * 4));
  }

  public processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    bufferSize?: number,
  ): void {
    this.ensureDisposedNot();
    const n = monoIn.length;
    if (leftOut.length !== n || rightOut.length !== n) {
      throw mapTaggedError(
        "[DNAM-VALIDATION] monoIn, leftOut, and rightOut must have identical length",
      );
    }
    if (n === 0) {
      return;
    }
    const basePtr = this.ensureScratchCapacity(n);
    const byteStride = n * 4;
    const miPtr = basePtr;
    const loPtr = basePtr + byteStride;
    const roPtr = basePtr + byteStride * 2;

    this.getScratchF32View().set(monoIn, 0);
    this.checkRc(
      this.exports.dnam_wasm_engine_process_mono_to_stereo(
        this.handle,
        miPtr,
        loPtr,
        roPtr,
        n,
        bufferSize ?? 0,
      ),
    );
    const outView = this.getScratchF32View();
    leftOut.set(outView.subarray(n, n * 2));
    rightOut.set(outView.subarray(n * 2, n * 3));
  }

  public processMono(monoIn: Float32Array, monoOut: Float32Array, bufferSize?: number): void {
    this.ensureDisposedNot();
    const n = monoIn.length;
    if (monoOut.length !== n) {
      throw mapTaggedError("[DNAM-VALIDATION] monoIn and monoOut must have identical length");
    }
    if (n === 0) {
      return;
    }
    const basePtr = this.ensureScratchCapacity(n);
    const byteStride = n * 4;
    const miPtr = basePtr;
    const moPtr = basePtr + byteStride;

    this.getScratchF32View().set(monoIn, 0);
    this.checkRc(
      this.exports.dnam_wasm_engine_process_mono(this.handle, miPtr, moPtr, n, bufferSize ?? 0),
    );
    monoOut.set(this.getScratchF32View().subarray(n, n * 2));
  }

  public async processAsync(
    leftIn: Float32Array,
    rightIn?: Float32Array,
    bufferSize?: number,
  ): Promise<[Float32Array, Float32Array]> {
    const n = leftIn.length;
    const leftOut = new Float32Array(n);
    const rightOut = new Float32Array(n);
    if (rightIn !== undefined) {
      this.processStereo(leftIn, rightIn, leftOut, rightOut, bufferSize);
    } else {
      this.processMonoToStereo(leftIn, leftOut, rightOut, bufferSize);
    }
    return [leftOut, rightOut];
  }

  public telemetry(): TelemetrySnapshot {
    this.ensureDisposedNot();
    this.checkRc(this.exports.dnam_wasm_engine_telemetry(this.handle, this.telemetryPtr));
    const v = this.getTelemetryF64View();
    const clampedControls = new Array<number>(16);
    for (let i = 0; i < 16; i++) {
      clampedControls[i] = v[28 + i] ?? 0;
    }
    return {
      activeTier: v[0] ?? 0,
      backend: backendCodeToName(v[1] ?? 4),
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
    if (this.handle !== 0) {
      engineFinalizer?.unregister(this);
      this.exports.dnam_wasm_engine_destroy(this.handle);
      this.exports.dnam_wasm_dealloc(this.scratchPtr, this.scratchSamplesCap * 4 * 4, 64);
      this.exports.dnam_wasm_dealloc(this.telemetryPtr, 50 * 8, 64);
      this.handle = 0;
      this.cleanupToken.handle = 0;
      this.cleanupToken.scratchPtr = 0;
      this.cleanupToken.telemetryPtr = 0;
    }
  }
}

class WasmControlSpaceHandle implements DnamControlSpaceHandle {
  private readonly exports: DnamWasmExports;
  private readonly throwLastError: () => never;
  private handle: number;
  private readonly scratchPtr: number;
  private readonly cleanupToken: ControlSpaceCleanupToken;
  private cachedMemBuf: ArrayBuffer | null = null;
  private cachedRawView: Float32Array | null = null;
  private cachedClampedView: Float32Array | null = null;
  private cachedNormView: Float32Array | null = null;
  private cachedFlagsView: Uint8Array | null = null;
  private readonly wasClampedBuf: boolean[] = new Array<boolean>(16).fill(false);

  public constructor(exports: DnamWasmExports, throwLastError: () => never, handle: number) {
    this.exports = exports;
    this.throwLastError = throwLastError;
    this.handle = handle;
    this.scratchPtr = exports.dnam_wasm_alloc(256, 64);
    this.cleanupToken = { exports, handle, scratchPtr: this.scratchPtr };
    controlSpaceFinalizer?.register(this, this.cleanupToken, this);
  }

  private ensureViews(): void {
    const buf = this.exports.memory.buffer;
    if (this.cachedMemBuf !== buf || this.cachedRawView === null) {
      this.cachedMemBuf = buf;
      this.cachedRawView = new Float32Array(buf, this.scratchPtr, 16);
      this.cachedClampedView = new Float32Array(buf, this.scratchPtr + 64, 16);
      this.cachedNormView = new Float32Array(buf, this.scratchPtr + 128, 16);
      this.cachedFlagsView = new Uint8Array(buf, this.scratchPtr + 192, 16);
    }
  }

  public clamp(rawNumericValues: Float32Array): RawClampResult {
    if (this.handle === 0) {
      throw new DnamError("ControlSpace instance has already been disposed");
    }
    this.ensureViews();
    const nIn = Math.min(16, rawNumericValues.length);
    const rawPtr = this.scratchPtr;
    const clampedPtr = this.scratchPtr + 64;
    const normPtr = this.scratchPtr + 128;
    const flagsPtr = this.scratchPtr + 192;

    if (nIn > 0) {
      this.cachedRawView?.set(rawNumericValues.subarray(0, nIn));
    }
    const nOut = this.exports.dnam_wasm_control_space_clamp(
      this.handle,
      rawPtr,
      nIn,
      clampedPtr,
      normPtr,
      flagsPtr,
    );
    if (nOut < 0) {
      this.throwLastError();
    }
    this.ensureViews();
    const flagsView = this.cachedFlagsView as Uint8Array;
    this.wasClampedBuf.length = nOut;
    for (let i = 0; i < nOut; i++) {
      this.wasClampedBuf[i] = (flagsView[i] ?? 0) !== 0;
    }
    return {
      clampedNumeric: (this.cachedClampedView as Float32Array).subarray(0, nOut),
      normalizedU: (this.cachedNormView as Float32Array).subarray(0, nOut),
      wasClamped: this.wasClampedBuf,
    };
  }

  public dispose(): void {
    if (this.handle !== 0) {
      controlSpaceFinalizer?.unregister(this);
      this.exports.dnam_wasm_control_space_destroy(this.handle);
      this.exports.dnam_wasm_dealloc(this.scratchPtr, 256, 64);
      this.handle = 0;
      this.cleanupToken.handle = 0;
      this.cleanupToken.scratchPtr = 0;
    }
  }
}

const DRIVER_SCRATCH_CAP_BYTES = 262_144;

export class WasmDriver implements DnamDriver {
  public readonly kind = "wasm" as const;
  private readonly exports: DnamWasmExports;
  private readonly scalarScratchPtr: number;
  private readonly scratchBasePtr: number;
  private scratchOffset = 0;
  private cachedMemBuf: ArrayBuffer | null = null;
  private cachedScalarF64: Float64Array | null = null;

  public constructor(instance: WebAssembly.Instance) {
    this.exports = instance.exports as DnamWasmExports;
    this.scalarScratchPtr = this.exports.dnam_wasm_alloc(256, 64);
    this.scratchBasePtr = this.exports.dnam_wasm_alloc(DRIVER_SCRATCH_CAP_BYTES, 64);
  }

  private getScalarF64View(): Float64Array {
    const buf = this.exports.memory.buffer;
    if (this.cachedScalarF64 === null || this.cachedMemBuf !== buf) {
      this.cachedMemBuf = buf;
      this.cachedScalarF64 = new Float64Array(buf, this.scalarScratchPtr, 32);
    }
    return this.cachedScalarF64;
  }

  private withTempScratch<T>(byteLen: number, fn: (ptr: number) => T): T {
    const alignedLen = (Math.max(8, byteLen) + 63) & ~63;
    const savedOffset = this.scratchOffset;
    if (this.scratchBasePtr !== 0 && savedOffset + alignedLen <= DRIVER_SCRATCH_CAP_BYTES) {
      const ptr = this.scratchBasePtr + savedOffset;
      this.scratchOffset = savedOffset + alignedLen;
      try {
        return fn(ptr);
      } finally {
        this.scratchOffset = savedOffset;
      }
    }
    const ptr = this.exports.dnam_wasm_alloc(alignedLen, 64);
    try {
      return fn(ptr);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, alignedLen, 64);
    }
  }

  private throwLastError = (): never => {
    const ptr = this.exports.dnam_wasm_last_error_ptr();
    const len = this.exports.dnam_wasm_last_error_len();
    const msg =
      len > 0
        ? textDecoder.decode(new Uint8Array(this.exports.memory.buffer, ptr, len))
        : "[DNAM-VALIDATION] Unknown WASM FFI error";
    throw mapTaggedError(msg);
  };

  private checkRc(rc: number): number {
    if (rc < 0) {
      this.throwLastError();
    }
    return rc;
  }

  private checkHandle(handle: number): number {
    if (handle === 0) {
      this.throwLastError();
    }
    return handle;
  }

  private readLastResultBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_result_ptr();
    const len = this.exports.dnam_wasm_result_len();
    return new Uint8Array(this.exports.memory.buffer, ptr, len).slice();
  }

  private readLastArenaBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_arena_ptr();
    const len = this.exports.dnam_wasm_arena_len();
    return new Uint8Array(this.exports.memory.buffer, ptr, len).slice();
  }

  private withBytes<T>(bytes: Uint8Array, fn: (ptr: number, len: number) => T): T {
    return this.withTempScratch(bytes.byteLength, (ptr) => {
      if (bytes.byteLength > 0) {
        new Uint8Array(this.exports.memory.buffer, ptr, bytes.byteLength).set(bytes);
      }
      return fn(ptr, bytes.byteLength);
    });
  }

  private withString<T>(str: string, fn: (ptr: number, len: number) => T): T {
    return this.withBytes(textEncoder.encode(str), fn);
  }

  private withF64Slice<T>(arr: Float64Array, fn: (ptr: number, len: number) => T): T {
    return this.withTempScratch(arr.byteLength, (ptr) => {
      if (arr.length > 0) {
        new Float64Array(this.exports.memory.buffer, ptr, arr.length).set(arr);
      }
      return fn(ptr, arr.length);
    });
  }

  private withF32Slice<T>(arr: Float32Array, fn: (ptr: number, len: number) => T): T {
    return this.withTempScratch(arr.byteLength, (ptr) => {
      if (arr.length > 0) {
        new Float32Array(this.exports.memory.buffer, ptr, arr.length).set(arr);
      }
      return fn(ptr, arr.length);
    });
  }

  public createEngine(
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string,
  ): DnamEngineHandle {
    const backendCode = backendNameToCode(backend);
    return this.withBytes(containerBytes, (ptr, len) => {
      const handle = this.checkHandle(
        this.exports.dnam_wasm_engine_create(ptr, len, tier, sampleRateHz, backendCode),
      );
      return new WasmEngineHandle(this.exports, this.throwLastError, handle);
    });
  }

  public createControlSpace(headerJson: string): DnamControlSpaceHandle {
    return this.withString(headerJson, (ptr, len) => {
      const handle = this.checkHandle(this.exports.dnam_wasm_control_space_create(ptr, len));
      return new WasmControlSpaceHandle(this.exports, this.throwLastError, handle);
    });
  }

  public validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): ValidatedContainerRaw {
    return this.withBytes(containerBytes, (ptr, len) =>
      this.withString(deviceIdHash ?? "", (devIdPtr, devIdLen) =>
        this.withBytes(deviceSecret32 ?? new Uint8Array(0), (devSecPtr, devSecLen) =>
          this.withString(trustedMarketplacePubkeyHex ?? "", (mktPkPtr, mktPkLen) => {
            this.checkRc(
              this.exports.dnam_wasm_validate_container_with_key(
                ptr,
                len,
                devIdPtr,
                devIdLen,
                devSecPtr,
                devSecLen,
                mktPkPtr,
                mktPkLen,
                this.scalarScratchPtr,
              ),
            );
            const v = this.getScalarF64View();
            const headerJson = textDecoder.decode(this.readLastResultBytes());
            const arenaBytes = this.readLastArenaBytes();
            const preamble: PreambleSnapshot = {
              magic: (v[0] ?? 0) === 1 ? "DNCK" : "DNAM",
              versionMajor: v[1] ?? 1,
              versionMinor: v[2] ?? 0,
              flagsBits: v[3] ?? 0,
              numControls: v[4] ?? 0,
              ringBufferPow2Log2: v[5] ?? 0,
              sampleRateTrainHz: v[6] ?? 48_000,
              jsonHeaderByteLen: v[7] ?? 0,
              tensorArenaOffset: v[8] ?? 4096,
              tensorArenaByteLen: v[9] ?? 0,
              arenaOffsetLiteEnd: v[10] ?? 10_752,
              arenaOffsetFullEnd: v[11] ?? 22_336,
              arenaOffsetUltraEnd: v[12] ?? 36_224,
              arenaCrc32cLite: (v[13] ?? 0) >>> 0,
              arenaCrc32cFull: (v[14] ?? 0) >>> 0,
              arenaCrc32cUltra: (v[15] ?? 0) >>> 0,
              numCkptStems: v[16] ?? 0,
              controlDecimation: v[17] ?? 16,
              headSvdRank: v[18] ?? 4,
              tailTimeMs: v[19] ?? 0,
              headerCrc32c: (v[20] ?? 0) >>> 0,
              core2Active: (v[21] ?? 0) !== 0,
              lfoEnvActive: (v[22] ?? 0) !== 0,
              stereoOut: (v[23] ?? 0) !== 0,
            };
            const statusCode = v[24] ?? 0;
            const verificationStatus =
              statusCode === 1
                ? "CreatorVerified"
                : statusCode === 2
                  ? "MarketplaceVerified"
                  : "Unsigned";
            const profileCode = v[25] ?? 0;
            const protectionProfile: DnamProtectionProfile =
              profileCode === 1
                ? "device_allowlisted"
                : profileCode === 2
                  ? "enclave_sealed"
                  : "open";
            return {
              preamble,
              verificationStatus,
              protectionProfile,
              headerJson,
              arenaBytes,
            };
          }),
        ),
      ),
    );
  }

  public buildContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    plaintextArenaCrcSource?: Uint8Array,
    encryptedArena = false,
  ): Uint8Array {
    return this.withString(headerJson, (hdrPtr, hdrLen) =>
      this.withBytes(arenaBytes, (arenaPtr, arenaLen) => {
        if (plaintextArenaCrcSource !== undefined) {
          return this.withBytes(plaintextArenaCrcSource, (crcPtr, crcLen) => {
            this.checkRc(
              this.exports.dnam_wasm_build_container(
                hdrPtr,
                hdrLen,
                arenaPtr,
                arenaLen,
                crcPtr,
                crcLen,
                encryptedArena ? 1 : 0,
              ),
            );
            return this.readLastResultBytes();
          });
        }
        this.checkRc(
          this.exports.dnam_wasm_build_container(
            hdrPtr,
            hdrLen,
            arenaPtr,
            arenaLen,
            0,
            0,
            encryptedArena ? 1 : 0,
          ),
        );
        return this.readLastResultBytes();
      }),
    );
  }

  public buildReferenceContainer(core2Active: boolean, isCheckpoint: boolean): Uint8Array {
    this.checkRc(
      this.exports.dnam_wasm_build_reference_container(core2Active ? 1 : 0, isCheckpoint ? 1 : 0),
    );
    return this.readLastResultBytes();
  }

  public updateHeaderInPlace(containerBytes: Uint8Array, updatedHeaderJson: string): Uint8Array {
    return this.withBytes(containerBytes, (cPtr, cLen) =>
      this.withString(updatedHeaderJson, (hPtr, hLen) => {
        this.checkRc(this.exports.dnam_wasm_update_header_in_place(cPtr, cLen, hPtr, hLen));
        return this.readLastResultBytes();
      }),
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
    this.checkRc(
      this.exports.dnam_wasm_compute_zoh(
        sampleRateTrainHz,
        sampleRateHostHz,
        controlDecimationTrain,
        rAcTrain,
        rDcTrain,
        tauRefTrain,
        switchXfadeTrain,
        this.scalarScratchPtr,
      ),
    );
    const v = this.getScalarF64View();
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
    const packedLen = Math.floor((nDim * (nDim + 1)) / 2);
    return this.withTempScratch(packedLen * 8, (outPtr) =>
      this.withF64Slice(denseFlat, (inPtr, inLen) => {
        const rc = this.checkRc(
          this.exports.dnam_wasm_pack_uplo_u(inPtr, inLen, nDim, outPtr, packedLen),
        );
        return new Float64Array(this.exports.memory.buffer, outPtr, rc).slice();
      }),
    );
  }

  public unpackUploU(packedU: Float64Array, nTier: number): Float64Array {
    const denseLen = nTier * nTier;
    return this.withTempScratch(denseLen * 8, (outPtr) =>
      this.withF64Slice(packedU, (inPtr, inLen) => {
        const rc = this.checkRc(
          this.exports.dnam_wasm_unpack_uplo_u(inPtr, inLen, nTier, outPtr, denseLen),
        );
        return new Float64Array(this.exports.memory.buffer, outPtr, rc).slice();
      }),
    );
  }

  public crc32c(data: Uint8Array): number {
    return this.withBytes(data, (ptr, len) => this.exports.dnam_wasm_crc32c(ptr, len) >>> 0);
  }

  public polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number {
    return this.exports.dnam_wasm_poly_adaa_algebraic(uPrev, uCurr, beta, delta);
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
    this.exports.dnam_wasm_poly_adaa_step(
      x,
      bConv,
      alpha,
      beta,
      delta,
      rAc,
      uPrev,
      aPrev,
      mDc,
      this.scalarScratchPtr,
    );
    const v = this.getScalarF64View();
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
    if (inp.length !== out.length) {
      throw mapTaggedError("[DNAM-VALIDATION] input and output must have identical length");
    }
    const n = inp.length;
    return this.withTempScratch(n * 8, (outPtr) =>
      this.withF64Slice(inp, (inPtr, len) => {
        this.checkRc(
          this.exports.dnam_wasm_poly_adaa_scan(
            inPtr,
            outPtr,
            len,
            bConv,
            alpha,
            beta,
            delta,
            rAc,
            uPrev,
            aPrev,
            mDc,
            reset ? 1 : 0,
            this.scalarScratchPtr,
          ),
        );
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, n));
        }
        const st = this.getScalarF64View();
        return [st[0] ?? 0, st[1] ?? 0, st[2] ?? 0];
      }),
    );
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
    this.exports.dnam_wasm_selective_rglru_step(
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
      this.scalarScratchPtr,
    );
    const v = this.getScalarF64View();
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
    if (inp.length !== out.length) {
      throw mapTaggedError("[DNAM-VALIDATION] input and output must have identical length");
    }
    const n = inp.length;
    return this.withTempScratch(n * 8, (outPtr) =>
      this.withF64Slice(inp, (inPtr, len) => {
        this.checkRc(
          this.exports.dnam_wasm_selective_rglru_scan(
            inPtr,
            outPtr,
            len,
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
            reset ? 1 : 0,
            this.scalarScratchPtr,
          ),
        );
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, n));
        }
        const st = this.getScalarF64View();
        return [st[0] ?? 0, st[1] ?? 0];
      }),
    );
  }

  public unitRotor(phi: number): [number, number] {
    this.exports.dnam_wasm_unit_rotor(phi, this.scalarScratchPtr);
    const v = this.getScalarF64View();
    return [v[0] ?? 1, v[1] ?? 0];
  }

  public unitRotorSlice(phi: Float64Array, out: Float64Array): void {
    const n = phi.length;
    if (out.length !== 2 * n) {
      throw mapTaggedError("[DNAM-VALIDATION] unitRotor output length must be 2 * N");
    }
    this.withTempScratch(2 * n * 8, (outPtr) => {
      this.withF64Slice(phi, (inPtr, len) => {
        this.checkRc(this.exports.dnam_wasm_unit_rotor_slice(inPtr, len, outPtr));
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, 2 * n));
        }
      });
    });
  }

  public cubicLagrangeWeights(mu: number): [number, number, number, number] {
    this.exports.dnam_wasm_cubic_lagrange_weights(mu, this.scalarScratchPtr);
    const v = this.getScalarF64View();
    return [v[0] ?? 0, v[1] ?? 1, v[2] ?? 0, v[3] ?? 0];
  }

  public givensHouseholderMix8ch(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array {
    return this.withF64Slice(x, (xPtr, xLen) => {
      this.checkRc(
        this.exports.dnam_wasm_givens_householder_mix_8ch(
          xPtr,
          xLen,
          cosPsi,
          sinPsi,
          this.scalarScratchPtr,
        ),
      );
      return this.getScalarF64View().subarray(0, 8).slice();
    });
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
    this.exports.dnam_wasm_mufdn_coercive_step(
      uIn,
      gLin,
      sigmaKnee,
      vCeil,
      rDc,
      dcXPrev,
      dcYPrev,
      this.scalarScratchPtr,
    );
    const v = this.getScalarF64View();
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
    if (inp.length !== out.length) {
      throw mapTaggedError("[DNAM-VALIDATION] input and output must have identical length");
    }
    const n = inp.length;
    this.withTempScratch(n * 8, (outPtr) => {
      this.withF64Slice(inp, (inPtr, len) => {
        this.checkRc(
          this.exports.dnam_wasm_mufdn_slice(
            inPtr,
            outPtr,
            len,
            delaySamples,
            gLin,
            sigmaKnee,
            vCeil,
            rDc,
          ),
        );
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, n));
        }
      });
    });
  }

  public evalLobattoBasis(
    u: number,
    anchors: Float64Array,
    degree: number,
  ): [number, number, number] {
    if (anchors.length === 3) {
      const sv = this.getScalarF64View();
      sv[0] = anchors[0] ?? 0;
      sv[1] = anchors[1] ?? -1;
      sv[2] = anchors[2] ?? 1;
      const outPtr = this.scalarScratchPtr + 64;
      this.checkRc(
        this.exports.dnam_wasm_eval_lobatto_basis(u, this.scalarScratchPtr, 3, degree, outPtr),
      );
      const outV = this.getScalarF64View();
      return [outV[8] ?? 0, outV[9] ?? 0, outV[10] ?? 0];
    }
    return this.withF64Slice(anchors, (ancPtr, ancLen) => {
      this.checkRc(
        this.exports.dnam_wasm_eval_lobatto_basis(u, ancPtr, ancLen, degree, this.scalarScratchPtr),
      );
      const v = this.getScalarF64View();
      return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
    });
  }

  public evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void {
    const n = u.length;
    if (out.length !== 3 * n) {
      throw mapTaggedError("[DNAM-VALIDATION] LobattoBasis output length must be 3 * N");
    }
    this.withTempScratch(3 * n * 8, (outPtr) => {
      this.withF64Slice(u, (uPtr, len) =>
        this.withF64Slice(anchors, (ancPtr, ancLen) => {
          this.checkRc(
            this.exports.dnam_wasm_eval_lobatto_basis_slice(
              uPtr,
              len,
              outPtr,
              ancPtr,
              ancLen,
              degree,
            ),
          );
          if (n > 0) {
            out.set(new Float64Array(this.exports.memory.buffer, outPtr, 3 * n));
          }
        }),
      );
    });
  }

  public evalRank4Fm(
    features: Float64Array,
    vFactorsFlat: Float64Array,
  ): [number, number, number, number] {
    return this.withF64Slice(features, (fPtr, fLen) =>
      this.withF64Slice(vFactorsFlat, (vPtr, vLen) => {
        this.checkRc(
          this.exports.dnam_wasm_eval_rank4_fm(fPtr, fLen, vPtr, vLen, this.scalarScratchPtr),
        );
        const v = this.getScalarF64View();
        return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0];
      }),
    );
  }

  public evalChebyshevLfo(cosPhi: number): [number, number, number, number] {
    this.exports.dnam_wasm_eval_chebyshev_lfo(cosPhi, this.scalarScratchPtr);
    const v = this.getScalarF64View();
    return [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 0];
  }

  public evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void {
    const n = cosPhi.length;
    if (out.length !== 4 * n) {
      throw mapTaggedError("[DNAM-VALIDATION] chebyshevLfo output length must be 4 * N");
    }
    this.withTempScratch(4 * n * 8, (outPtr) => {
      this.withF64Slice(cosPhi, (inPtr, len) => {
        this.checkRc(this.exports.dnam_wasm_eval_chebyshev_lfo_slice(inPtr, len, outPtr));
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, 4 * n));
        }
      });
    });
  }

  public evalTrustRegionSoftclip(z: number, mh: number): number {
    return this.exports.dnam_wasm_eval_trust_region_softclip(z, mh);
  }

  public evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void {
    const n = z.length;
    if (out.length !== n) {
      throw mapTaggedError(
        "[DNAM-VALIDATION] trustRegionSoftclip input and output lengths must match",
      );
    }
    this.withTempScratch(n * 8, (outPtr) => {
      this.withF64Slice(z, (inPtr, len) => {
        this.checkRc(
          this.exports.dnam_wasm_eval_trust_region_softclip_slice(inPtr, len, outPtr, mh),
        );
        if (n > 0) {
          out.set(new Float64Array(this.exports.memory.buffer, outPtr, n));
        }
      });
    });
  }

  public immutableHeaderSha256(headerJson: string): string {
    return this.withString(headerJson, (ptr, len) => {
      this.checkRc(this.exports.dnam_wasm_immutable_header_sha256(ptr, len));
      return textDecoder.decode(this.readLastResultBytes());
    });
  }

  public sha256Hex(data: Uint8Array): string {
    return this.withBytes(data, (ptr, len) => {
      this.exports.dnam_wasm_sha256_hex(ptr, len);
      return textDecoder.decode(this.readLastResultBytes());
    });
  }

  public buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array {
    return this.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
      this.withString(masterArenaSha256Hex, (aPtr, aLen) => {
        this.checkRc(this.exports.dnam_wasm_build_creator_signing_message(hPtr, hLen, aPtr, aLen));
        return this.readLastResultBytes();
      }),
    );
  }

  public buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array {
    return this.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
      this.withString(masterArenaSha256Hex, (aPtr, aLen) =>
        this.withString(receiptJson, (rPtr, rLen) => {
          this.checkRc(
            this.exports.dnam_wasm_build_marketplace_signing_message(
              hPtr,
              hLen,
              aPtr,
              aLen,
              rPtr,
              rLen,
            ),
          );
          return this.readLastResultBytes();
        }),
      ),
    );
  }

  public chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array {
    return this.withBytes(arenaBytes, (aPtr, aLen) =>
      this.withBytes(key32, (kPtr, kLen) =>
        this.withBytes(nonce12, (nPtr, nLen) => {
          this.checkRc(
            this.exports.dnam_wasm_chacha20_xor_arena(
              aPtr,
              aLen,
              kPtr,
              kLen,
              nPtr,
              nLen,
              startBlockCounter,
            ),
          );
          return this.readLastResultBytes();
        }),
      ),
    );
  }

  public applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa48: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array {
    return this.withBytes(arenaBytes, (aPtr, aLen) =>
      this.withF32Slice(kappa48, (kPtr, kLen) =>
        this.withF32Slice(qBuyerFlat16, (qPtr, qLen) => {
          this.checkRc(
            this.exports.dnam_wasm_apply_isometric_gauge_watermark(
              aPtr,
              aLen,
              kPtr,
              kLen,
              qPtr,
              qLen,
            ),
          );
          return this.readLastResultBytes();
        }),
      ),
    );
  }

  public generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array {
    return this.withTempScratch(4608 * 4, (outPtr) =>
      this.withBytes(marketplaceSecret32, (sPtr, sLen) =>
        this.withString(modelId, (mPtr, mLen) =>
          this.withString(licenseId, (lPtr, lLen) => {
            this.checkRc(
              this.exports.dnam_wasm_generate_tardos_code(
                sPtr,
                sLen,
                mPtr,
                mLen,
                lPtr,
                lLen,
                outPtr,
              ),
            );
            return new Float32Array(this.exports.memory.buffer, outPtr, 4608).slice();
          }),
        ),
      ),
    );
  }

  public applyTardosWatermark(
    arenaBytes: Uint8Array,
    code4608: Float32Array,
    epsilon: number,
  ): Uint8Array {
    return this.withBytes(arenaBytes, (aPtr, aLen) =>
      this.withF32Slice(code4608, (cPtr, cLen) => {
        this.checkRc(
          this.exports.dnam_wasm_apply_tardos_watermark(aPtr, aLen, cPtr, cLen, epsilon),
        );
        return this.readLastResultBytes();
      }),
    );
  }

  public scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code4608: Float32Array,
  ): number {
    return this.withBytes(masterArena, (mPtr, mLen) =>
      this.withBytes(suspectArena, (sPtr, sLen) =>
        this.withF32Slice(code4608, (cPtr, cLen) => {
          this.checkRc(
            this.exports.dnam_wasm_score_tardos_watermark(
              mPtr,
              mLen,
              sPtr,
              sLen,
              cPtr,
              cLen,
              this.scalarScratchPtr,
            ),
          );
          return this.getScalarF64View()[0] ?? 0;
        }),
      ),
    );
  }

  public extractStructuralHash(arenaBytes: Uint8Array): Float64Array {
    return this.withTempScratch(160 * 8, (outPtr) =>
      this.withBytes(arenaBytes, (aPtr, aLen) => {
        this.checkRc(this.exports.dnam_wasm_extract_structural_hash(aPtr, aLen, outPtr));
        return new Float64Array(this.exports.memory.buffer, outPtr, 160).slice();
      }),
    );
  }

  public structuralCosineSimilarity(a: Float64Array, b: Float64Array): number {
    return this.withF64Slice(a, (aPtr, aLen) =>
      this.withF64Slice(b, (bPtr, bLen) => {
        this.checkRc(
          this.exports.dnam_wasm_structural_cosine_similarity(
            aPtr,
            aLen,
            bPtr,
            bLen,
            this.scalarScratchPtr,
          ),
        );
        return this.getScalarF64View()[0] ?? 0;
      }),
    );
  }

  public ed25519PubkeyHex(seed32: Uint8Array): string {
    return this.withBytes(seed32, (ptr, len) => {
      this.checkRc(this.exports.dnam_wasm_ed25519_pubkey_hex(ptr, len));
      return textDecoder.decode(this.readLastResultBytes());
    });
  }

  public ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string {
    return this.withBytes(seed32, (sPtr, sLen) =>
      this.withBytes(message, (mPtr, mLen) => {
        this.checkRc(this.exports.dnam_wasm_ed25519_sign_hex(sPtr, sLen, mPtr, mLen));
        return textDecoder.decode(this.readLastResultBytes());
      }),
    );
  }

  public x25519PubkeyHex(secret32: Uint8Array): string {
    return this.withBytes(secret32, (ptr, len) => {
      this.checkRc(this.exports.dnam_wasm_x25519_pubkey_hex(ptr, len));
      return textDecoder.decode(this.readLastResultBytes());
    });
  }

  public wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string {
    return this.withBytes(ephemeralSecret32, (ePtr, eLen) =>
      this.withBytes(devicePubkey32, (dPtr, dLen) =>
        this.withString(deviceIdHash, (idPtr, idLen) =>
          this.withBytes(cek32, (cPtr, cLen) => {
            this.checkRc(
              this.exports.dnam_wasm_wrap_device_cek_hex(
                ePtr,
                eLen,
                dPtr,
                dLen,
                idPtr,
                idLen,
                cPtr,
                cLen,
              ),
            );
            return textDecoder.decode(this.readLastResultBytes());
          }),
        ),
      ),
    );
  }

  public signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX25519_32: Uint8Array,
  ): string {
    return this.withBytes(oemSeed32, (sPtr, sLen) =>
      this.withString(oemVendor, (vPtr, vLen) =>
        this.withString(modelCode, (mPtr, mLen) =>
          this.withString(deviceIdHash, (idPtr, idLen) =>
            this.withBytes(pubkeyX25519_32, (pkPtr, pkLen) => {
              this.checkRc(
                this.exports.dnam_wasm_sign_oem_device_cert(
                  sPtr,
                  sLen,
                  vPtr,
                  vLen,
                  mPtr,
                  mLen,
                  idPtr,
                  idLen,
                  pkPtr,
                  pkLen,
                ),
              );
              return textDecoder.decode(this.readLastResultBytes());
            }),
          ),
        ),
      ),
    );
  }

  public verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string {
    return this.withString(certJson, (cPtr, cLen) =>
      this.withString(trustedOemPubkeyHex, (pPtr, pLen) => {
        this.checkRc(this.exports.dnam_wasm_verify_oem_device_cert(cPtr, cLen, pPtr, pLen));
        return textDecoder.decode(this.readLastResultBytes());
      }),
    );
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
    return this.withBytes(innerDnamBytes, (inPtr, inLen) =>
      this.withString(recipientCertsJson, (cPtr, cLen) =>
        this.withString(trustedOemPubkeysJson, (oPtr, oLen) =>
          this.withBytes(marketplaceSeed32, (mPtr, mLen) =>
            this.withBytes(ephemeralSecret32, (ePtr, eLen) =>
              this.withBytes(cek32, (kPtr, kLen) =>
                this.withBytes(nonce12, (nPtr, nLen) => {
                  this.checkRc(
                    this.exports.dnam_wasm_seal_container(
                      inPtr,
                      inLen,
                      cPtr,
                      cLen,
                      oPtr,
                      oLen,
                      mPtr,
                      mLen,
                      ePtr,
                      eLen,
                      kPtr,
                      kLen,
                      nPtr,
                      nLen,
                    ),
                  );
                  return this.readLastResultBytes();
                }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  public unsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array {
    return this.withBytes(sealedBytes, (sPtr, sLen) =>
      this.withString(deviceIdHash ?? "", (idPtr, idLen) =>
        this.withBytes(deviceSecret32 ?? new Uint8Array(0), (secPtr, secLen) =>
          this.withString(trustedMarketplacePubkeyHex ?? "", (mPtr, mLen) => {
            this.checkRc(
              this.exports.dnam_wasm_unseal_container(
                sPtr,
                sLen,
                idPtr,
                idLen,
                secPtr,
                secLen,
                mPtr,
                mLen,
              ),
            );
            return this.readLastResultBytes();
          }),
        ),
      ),
    );
  }
}

let cachedWasmDriver: WasmDriver | null = null;
let cachedWasmBinary: ArrayBuffer | null = null;

function toArrayBuffer(source: BufferSource): ArrayBuffer {
  if (source instanceof ArrayBuffer) {
    return source;
  }
  const view = new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  return view.slice().buffer;
}

/**
 * Synchronously instantiates `dnam_engine.wasm` from a `BufferSource` (`Uint8Array` / `ArrayBuffer`)
 * or pre-compiled `WebAssembly.Module`, caching the global `WasmDriver`.
 */
export function initDnamWasmSync(wasmSource: BufferSource | WebAssembly.Module): WasmDriver {
  if (wasmSource instanceof WebAssembly.Module) {
    const instance = new WebAssembly.Instance(wasmSource, {});
    cachedWasmDriver = new WasmDriver(instance);
    return cachedWasmDriver;
  }
  const buf = toArrayBuffer(wasmSource);
  cachedWasmBinary = buf;
  const mod = new WebAssembly.Module(buf);
  const instance = new WebAssembly.Instance(mod, {});
  cachedWasmDriver = new WasmDriver(instance);
  return cachedWasmDriver;
}

/**
 * Asynchronously initializes the `WasmDriver` in Browser or Node.js environments.
 */
export async function initDnamWasm(
  wasmSource?: BufferSource | WebAssembly.Module | Response | Promise<Response> | string | URL,
): Promise<WasmDriver> {
  if (wasmSource === undefined && cachedWasmDriver !== null) {
    return cachedWasmDriver;
  }
  if (wasmSource instanceof WebAssembly.Module) {
    return initDnamWasmSync(wasmSource);
  }
  if (
    wasmSource instanceof ArrayBuffer ||
    (wasmSource !== undefined && ArrayBuffer.isView(wasmSource))
  ) {
    return initDnamWasmSync(wasmSource);
  }

  let response: Response;
  if (wasmSource instanceof Response) {
    response = wasmSource;
  } else if (wasmSource instanceof Promise) {
    response = await wasmSource;
  } else {
    const url = wasmSource ?? new URL(/* @vite-ignore */ "./dnam_engine.wasm", import.meta.url);
    response = await fetch(url);
  }
  if (!response.ok) {
    throw new DnamError(
      `Failed to fetch dnam_engine.wasm: ${String(response.status)} ${response.statusText}`,
    );
  }
  const buf = await response.arrayBuffer();
  return initDnamWasmSync(buf);
}

/**
 * Returns the initialized `WasmDriver`, throwing a descriptive `DnamError` if `initDnamWasm`
 * has not yet been called in Browser environments.
 */
export function getWasmDriver(): WasmDriver {
  if (cachedWasmDriver === null) {
    throw new DnamError(
      "WASM driver is not initialized. Call `await initDnamWasm()` or `initDnamWasmSync(bytes)` before creating a WASM DnamEngine in the browser.",
    );
  }
  return cachedWasmDriver;
}

export function hasWasmDriver(): boolean {
  return cachedWasmDriver !== null;
}

export function getCachedWasmBinary(): ArrayBuffer | null {
  return cachedWasmBinary;
}
