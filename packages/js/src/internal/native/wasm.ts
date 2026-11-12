/**
 * High-performance WebAssembly `SIMD128` driver (`WasmDriver`) with persistent linear-memory
 * audio scratch buffers and cached typed-array views (`ref[sdk.ts.runtime.dual-target-drivers]`,
 * `ref[sdk.ts.engine.zero-copy-and-async]`).
 *
 * Zero `node:*` dependencies so it runs identically in browsers, Node.js, and
 * `AudioWorkletGlobalScope`.
 *
 * @module
 */

import { DnamError, fromNativeError } from "../../errors.js";
import type { ActiveBackend } from "../../types.js";
import { backendFromCode, backendToCode, coerceBytes } from "../coerce.js";
import {
  type CreateNativeEngineOptions,
  type NativeControlSpaceHandle,
  type NativeDriver,
  type NativeEngineHandle,
  type NativeValidatedContainerInfo,
  TELEMETRY_F64_LEN,
} from "./driver.js";

const sharedEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : undefined;
const sharedDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : undefined;

function encodeUtf8(str: string): Uint8Array {
  return sharedEncoder ? sharedEncoder.encode(str) : Uint8Array.from(str, (c) => c.charCodeAt(0));
}

function isSharedBuffer(buf: unknown): boolean {
  return typeof SharedArrayBuffer !== "undefined" && buf instanceof SharedArrayBuffer;
}

function decodeUtf8(bytes: Uint8Array): string {
  if (!sharedDecoder) {
    let out = "";
    for (let i = 0; i < bytes.length; i++) {
      out += String.fromCharCode(bytes[i] ?? 0);
    }
    return out;
  }
  const safeBytes = isSharedBuffer(bytes.buffer) ? bytes.slice() : bytes;
  return sharedDecoder.decode(safeBytes);
}

export interface RawWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
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
    maxSupportedTier: number,
    offlineBounce: number,
    maximizeTierOnBounce: number,
  ): number;
  dnam_wasm_engine_free(ptr: number): void;
  dnam_wasm_engine_reload(
    ptr: number,
    containerPtr: number,
    containerLen: number,
    tier: number,
    sampleRateHz: number,
  ): number;
  dnam_wasm_engine_backend_code(ptr: number): number;
  dnam_wasm_engine_tier(ptr: number): number;
  dnam_wasm_engine_set_tier(ptr: number, tier: number): number;
  dnam_wasm_engine_max_supported_tier(ptr: number): number;
  dnam_wasm_engine_set_max_supported_tier(ptr: number, tier: number): number;
  dnam_wasm_engine_max_available_tier(ptr: number): number;
  dnam_wasm_engine_effective_tier(ptr: number): number;
  dnam_wasm_engine_sample_rate(ptr: number): number;
  dnam_wasm_engine_set_sample_rate(ptr: number, sampleRateHz: number): number;
  dnam_wasm_engine_latency_samples(ptr: number): number;
  dnam_wasm_engine_offline_bounce(ptr: number): number;
  dnam_wasm_engine_set_offline_bounce(ptr: number, enabled: number): void;
  dnam_wasm_engine_maximize_tier_on_bounce(ptr: number): number;
  dnam_wasm_engine_set_maximize_tier_on_bounce(ptr: number, enabled: number): void;
  dnam_wasm_engine_num_controls(ptr: number): number;
  dnam_wasm_engine_raw_controls_into(ptr: number, outPtr: number, len: number): number;
  dnam_wasm_engine_clamped_controls_into(ptr: number, outPtr: number, len: number): number;
  dnam_wasm_engine_clamped_control_value(ptr: number, ctrlIdx: number, outPtr: number): number;
  dnam_wasm_engine_set_control_value(ptr: number, ctrlIdx: number, value: number): number;
  dnam_wasm_engine_set_control_immediate(ptr: number, ctrlIdx: number, value: number): number;
  dnam_wasm_engine_set_controls_immediate_batch(
    ptr: number,
    indicesPtr: number,
    valuesPtr: number,
    len: number,
  ): number;
  dnam_wasm_engine_reset_controls_to_defaults(ptr: number): void;
  dnam_wasm_engine_reset(ptr: number): void;
  dnam_wasm_engine_supports_freeze(ptr: number): number;
  dnam_wasm_engine_is_frozen(ptr: number): number;
  dnam_wasm_engine_set_freeze(ptr: number, freeze: number): void;
  dnam_wasm_engine_set_freeze_amount(ptr: number, alpha: number): void;
  dnam_wasm_engine_bypass_state(ptr: number): number;
  dnam_wasm_engine_set_bypass(ptr: number, bypassed: number, trails: number): void;
  dnam_wasm_engine_stereo_width(ptr: number): number;
  dnam_wasm_engine_set_stereo_width(ptr: number, width: number): void;
  dnam_wasm_engine_configure_host_mix(
    ptr: number,
    blendWet: number,
    outputGainDb: number,
    applyUnityLeveling: number,
  ): void;
  dnam_wasm_engine_current_bpm(ptr: number): number;
  dnam_wasm_engine_sync_tempo(
    ptr: number,
    bpm: number,
    subPtr: number,
    subLen: number,
    foldOctave: number,
  ): number;
  dnam_wasm_engine_register_tap_tempo(
    ptr: number,
    subPtr: number,
    subLen: number,
    foldOctave: number,
    outBpmPtr: number,
  ): number;
  dnam_wasm_engine_apply_preset(ptr: number, presetPtr: number, presetLen: number): number;
  dnam_wasm_engine_process_stereo_in_place(
    ptr: number,
    leftIoPtr: number,
    rightIoPtr: number,
    numSamples: number,
  ): number;
  dnam_wasm_engine_process_stereo(
    ptr: number,
    leftInPtr: number,
    rightInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_process_mono_to_stereo(
    ptr: number,
    monoInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_process_mono(
    ptr: number,
    monoInPtr: number,
    monoOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_render_offline_bounce(
    ptr: number,
    leftInPtr: number,
    rightInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
  ): number;
  dnam_wasm_engine_telemetry_into(ptr: number, outPtr: number, outLen: number): number;

  dnam_wasm_control_space_create(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_control_space_free(ptr: number): void;
  dnam_wasm_control_space_num_controls(ptr: number): number;
  dnam_wasm_control_space_clamp_into(
    ptr: number,
    rawPtr: number,
    clampedPtr: number,
    normPtr: number,
    len: number,
  ): number;
  dnam_wasm_control_space_to_physical(
    ptr: number,
    ctrlIdx: number,
    normalizedU: number,
    outPtr: number,
  ): number;
  dnam_wasm_control_space_to_normalized(
    ptr: number,
    ctrlIdx: number,
    physicalVal: number,
    outPtr: number,
  ): number;
  dnam_wasm_compute_zoh(
    srTrain: number,
    srHost: number,
    rAc: number,
    rDc: number,
    tauRef: number,
    switchXfade: number,
    out9Ptr: number,
  ): number;
  dnam_wasm_control_stride_for_rate(sampleRateHz: number): number;
  dnam_wasm_control_tick_ratio(sampleRateHz: number): number;
  dnam_wasm_eval_subdivision_phase(
    beatMod12: number,
    withinBeatQ32: number,
    subPtr: number,
    subLen: number,
    offsetQ32: number,
    out3Ptr: number,
  ): number;
  dnam_wasm_eval_3node_calibration(anchorsPtr: number, u: number, outPtr: number): number;
  dnam_wasm_invert_tempo_calibration(
    declMin: number,
    declMax: number,
    safeMin: number,
    safeMax: number,
    anchorsPtr: number,
    target: number,
    flushSubnormals: number,
    out5Ptr: number,
  ): number;

  dnam_wasm_validate_container_with_key(
    cPtr: number,
    cLen: number,
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_build_reference_container(
    supportsFreeze: number,
    inputs: number,
    modePtr: number,
    modeLen: number,
  ): number;
  dnam_wasm_build_dnam_container(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    encPtr: number,
    encLen: number,
    hasSig: number,
  ): number;
  dnam_wasm_update_header_in_place(
    cPtr: number,
    cLen: number,
    hdrPtr: number,
    hdrLen: number,
  ): number;
  dnam_wasm_strip_container_to_tier(cPtr: number, cLen: number, maxTier: number): number;
  dnam_wasm_parse_schema_version(verPtr: number, verLen: number, out4Ptr: number): number;
  dnam_wasm_crc32c(dataPtr: number, dataLen: number): number;
  dnam_wasm_validate_and_normalize_header(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_canonicalize_jcs(
    jsonPtr: number,
    jsonLen: number,
    stripMutable: number,
    rejectFloats: number,
  ): number;
  dnam_wasm_build_dmu1_frame(jsonPtr: number, jsonLen: number, capacity: number): number;
  dnam_wasm_parse_dmu1_frame(framePtr: number, frameLen: number): number;
  dnam_wasm_validate_v2_open_dnam_layout(cPtr: number, cLen: number): number;
  dnam_wasm_strip_v2_open_dnam_container(cPtr: number, cLen: number, targetTier: number): number;
  dnam_wasm_uplo_u_packed_len(nDim: number): number;
  dnam_wasm_pack_uplo_u_f64_into(
    densePtr: number,
    denseLen: number,
    nDim: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_unpack_uplo_u_f64_into(
    packedPtr: number,
    packedLen: number,
    nTier: number,
    outPtr: number,
    outLen: number,
  ): number;

  dnam_wasm_verify_container_provenance(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    hasSig: number,
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_enforce_distribution_policy(
    hdrPtr: number,
    hdrLen: number,
    profPtr: number,
    profLen: number,
  ): number;
  dnam_wasm_evaluate_anticlone_policy(
    fpMatched: number,
    wmIdentified: number,
    indepEvidence: number,
  ): number;
  dnam_wasm_immutable_header_sha256(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_sha256_hex(dataPtr: number, dataLen: number): number;
  dnam_wasm_hmac_sha256_hex(keyPtr: number, keyLen: number, msgPtr: number, msgLen: number): number;
  dnam_wasm_ed25519_pubkey_hex(seedPtr: number, seedLen: number): number;
  dnam_wasm_ed25519_sign_hex(
    seedPtr: number,
    seedLen: number,
    msgPtr: number,
    msgLen: number,
  ): number;
  dnam_wasm_x25519_pubkey_hex(secPtr: number, secLen: number): number;
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
    rcptPtr: number,
    rcptLen: number,
  ): number;
  dnam_wasm_sign_creator_header(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    seedPtr: number,
    seedLen: number,
    signerPtr: number,
    signerLen: number,
    includeTierPrefixes: number,
  ): number;
  dnam_wasm_sign_marketplace_receipt(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    seedPtr: number,
    seedLen: number,
    specPtr: number,
    specLen: number,
  ): number;
  dnam_wasm_sign_oem_device_cert(
    seedPtr: number,
    seedLen: number,
    vendorPtr: number,
    vendorLen: number,
    modelPtr: number,
    modelLen: number,
    devIdPtr: number,
    devIdLen: number,
    pkPtr: number,
    pkLen: number,
  ): number;
  dnam_wasm_verify_oem_device_cert(
    certPtr: number,
    certLen: number,
    pkHexPtr: number,
    pkHexLen: number,
  ): number;
  dnam_wasm_build_oem_cert_v2_signing_message(
    vendorPtr: number,
    vendorLen: number,
    modelPtr: number,
    modelLen: number,
    devIdPtr: number,
    devIdLen: number,
    pkPtr: number,
    pkLen: number,
    issPkPtr: number,
    issPkLen: number,
    assurance: number,
    polPtr: number,
    polLen: number,
    evPtr: number,
    evLen: number,
  ): number;
  dnam_wasm_chacha20_xor_arena(
    arenaPtr: number,
    arenaLen: number,
    keyPtr: number,
    keyLen: number,
    noncePtr: number,
    nonceLen: number,
    startCounter: number,
  ): number;
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
  dnam_wasm_seal_container(
    innerPtr: number,
    innerLen: number,
    certsPtr: number,
    certsLen: number,
    oemPksPtr: number,
    oemPksLen: number,
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
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_encode_dau1_envelope(specPtr: number, specLen: number): number;
  dnam_wasm_decode_dau1_envelope(wirePtr: number, wireLen: number): number;
  dnam_wasm_build_dme1_header(
    profile: number,
    recipientCount: number,
    ciphertextLength: bigint,
  ): number;
  dnam_wasm_parse_dme1_plaintext(
    ptPtr: number,
    ptLen: number,
    digestsPtr: number,
    digestsLen: number,
  ): number;
  dnam_wasm_encode_recipient_key_ids(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_recipient_key_ids(
    dataPtr: number,
    dataLen: number,
    maxRecipients: number,
  ): number;
  dnam_wasm_encode_hpke_slots(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_hpke_slots(
    dataPtr: number,
    dataLen: number,
    expectedCount: number,
    maxRecipients: number,
  ): number;
  dnam_wasm_encode_record_directory(maximumTier: number, jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_record_directory(dataPtr: number, dataLen: number, maximumTier: number): number;
  dnam_wasm_compute_preview_window(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
    out2Ptr: number,
  ): number;
  dnam_wasm_validate_preview_switch_options(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_validate_royalty_splits(
    tierPtr: number,
    tierLen: number,
    splitsPtr: number,
    splitsLen: number,
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
    secPtr: number,
    secLen: number,
    modelPtr: number,
    modelLen: number,
    licPtr: number,
    licLen: number,
    outPtr: number,
    outLen: number,
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
  dnam_wasm_extract_structural_hash(
    arenaPtr: number,
    arenaLen: number,
    outPtr: number,
    outCap: number,
  ): number;
  dnam_wasm_structural_cosine_similarity(
    aPtr: number,
    aLen: number,
    bPtr: number,
    bLen: number,
    outSimPtr: number,
  ): number;
  dnam_wasm_derive_watermark_seed_hex(
    keyPtr: number,
    keyLen: number,
    mktPkPtr: number,
    mktPkLen: number,
    wmKidPtr: number,
    wmKidLen: number,
    manShaPtr: number,
    manShaLen: number,
    modelIdPtr: number,
    modelIdLen: number,
    profShaPtr: number,
    profShaLen: number,
    owningTier: number,
    licPtr: number,
    licLen: number,
    buyerPtr: number,
    buyerLen: number,
  ): number;
  dnam_wasm_extract_stage3_fingerprint(
    firPtr: number,
    firLen: number,
    kautzPtr: number,
    kautzLen: number,
    addedModes: number,
    outPtr: number,
    outCap: number,
  ): number;

  dnam_wasm_catmull_rom_hermite(
    sM1: number,
    s0: number,
    s1: number,
    s2: number,
    mu: number,
  ): number;
  dnam_wasm_catmull_rom_hermite_slice(
    sM1Ptr: number,
    s0Ptr: number,
    s1Ptr: number,
    s2Ptr: number,
    muPtr: number,
    outPtr: number,
    len: number,
  ): number;
  dnam_wasm_op_t_coercive_step(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
    out3Ptr: number,
  ): void;
  dnam_wasm_op_t_tank_slice(
    inpPtr: number,
    outPtr: number,
    len: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): number;
  dnam_wasm_op_m_modal_step(
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
  dnam_wasm_op_m_modal_slice(
    inpPtr: number,
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
    state2Ptr: number,
  ): number;
  dnam_wasm_unit_rotor(phi: number, out2Ptr: number): void;
  dnam_wasm_unit_rotor_slice(phiPtr: number, len: number, outPtr: number, outLen: number): number;
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
  dnam_wasm_poly_adaa_slice(
    inpPtr: number,
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
    state3Ptr: number,
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
    uLen: number,
    outPtr: number,
    outLen: number,
    anchorsPtr: number,
    anchorsLen: number,
    degree: number,
  ): number;
  dnam_wasm_eval_rank4_fm(
    featPtr: number,
    featLen: number,
    vfPtr: number,
    vfLen: number,
    out4Ptr: number,
  ): number;
  dnam_wasm_eval_rank4_fm_slice(
    featPtr: number,
    featLen: number,
    nFrames: number,
    dDim: number,
    vfPtr: number,
    vfLen: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_eval_chebyshev_lfo(cosPhi: number, out4Ptr: number): void;
  dnam_wasm_eval_chebyshev_lfo_slice(
    cosPhiPtr: number,
    len: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_eval_trust_region_softclip(z: number, mh: number): number;
  dnam_wasm_eval_trust_region_softclip_slice(
    zPtr: number,
    outPtr: number,
    len: number,
    mh: number,
  ): number;
  dnam_wasm_cubic_lagrange_weights(mu: number, out4Ptr: number): void;
  dnam_wasm_cubic_lagrange_weights_slice(
    muPtr: number,
    len: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_givens_householder_8ch(
    xPtr: number,
    xLen: number,
    cosPsi: number,
    sinPsi: number,
    out8Ptr: number,
  ): number;
  dnam_wasm_givens_householder_8ch_slice(
    xPtr: number,
    outPtr: number,
    len: number,
    cosPsi: number,
    sinPsi: number,
  ): number;
  dnam_wasm_givens_householder_16ch(
    xPtr: number,
    xLen: number,
    cosPsi: number,
    sinPsi: number,
    ccPtr: number,
    ccLen: number,
    csPtr: number,
    csLen: number,
    out16Ptr: number,
  ): number;
  dnam_wasm_givens_householder_16ch_slice(
    xPtr: number,
    outPtr: number,
    len: number,
    cosPsi: number,
    sinPsi: number,
    ccPtr: number,
    ccLen: number,
    csPtr: number,
    csLen: number,
  ): number;
}

class WasmMemoryArena {
  readonly exports: RawWasmExports;
  readonly memory: WebAssembly.Memory;
  #cachedBuffer: ArrayBufferLike | undefined;
  #u8View: Uint8Array | undefined;
  #f32View: Float32Array | undefined;
  #f64View: Float64Array | undefined;
  #u32View: Uint32Array | undefined;

  // Persistent small scratch region for telemetry (56 f64 = 448 B) + control values (256 f32 = 1024 B)
  readonly smallF64Ptr: number;
  readonly smallF32Ptr: number;
  readonly smallU32Ptr: number;

  // Persistent dynamically grown audio scratch region (4 * maxFrames * 4 bytes)
  #audioPtr = 0;
  #audioCapacityFrames = 0;

  constructor(exports: RawWasmExports, memory?: WebAssembly.Memory) {
    this.exports = exports;
    this.memory = memory ?? exports.memory;
    this.smallF64Ptr = this.alloc(64 * 8);
    this.smallF32Ptr = this.alloc(256 * 4);
    this.smallU32Ptr = this.alloc(256 * 4);
    this.#ensureAudioScratch(4096);
  }

  alloc(size: number): number {
    const ptr = this.exports.dnam_wasm_alloc(size, 64);
    if (ptr === 0) {
      this.throwLastError();
    }
    return ptr;
  }

  #refreshViews(): void {
    const buf = this.memory.buffer;
    if (this.#cachedBuffer !== buf || this.#u8View?.byteLength !== buf.byteLength) {
      this.#cachedBuffer = buf;
      this.#u8View = new Uint8Array(buf);
      this.#f32View = new Float32Array(buf);
      this.#f64View = new Float64Array(buf);
      this.#u32View = new Uint32Array(buf);
    }
  }

  u8(): Uint8Array {
    this.#refreshViews();
    return this.#u8View as Uint8Array;
  }

  f32(): Float32Array {
    this.#refreshViews();
    return this.#f32View as Float32Array;
  }

  f64(): Float64Array {
    this.#refreshViews();
    return this.#f64View as Float64Array;
  }

  u32(): Uint32Array {
    this.#refreshViews();
    return this.#u32View as Uint32Array;
  }

  ensureAudioScratch(frames: number): {
    readonly ch0InPtr: number;
    readonly ch1InPtr: number;
    readonly ch0OutPtr: number;
    readonly ch1OutPtr: number;
  } {
    this.#ensureAudioScratch(frames);
    const strideBytes = this.#audioCapacityFrames * 4;
    return {
      ch0InPtr: this.#audioPtr,
      ch1InPtr: this.#audioPtr + strideBytes,
      ch0OutPtr: this.#audioPtr + strideBytes * 2,
      ch1OutPtr: this.#audioPtr + strideBytes * 3,
    };
  }

  #ensureAudioScratch(frames: number): void {
    if (frames <= this.#audioCapacityFrames && this.#audioPtr !== 0) {
      return;
    }
    const nextCap = Math.max(4096, 1 << Math.ceil(Math.log2(Math.max(1, frames))));
    const nextPtr = this.alloc(nextCap * 16);
    if (this.#audioPtr !== 0) {
      this.exports.dnam_wasm_dealloc(this.#audioPtr, this.#audioCapacityFrames * 16, 64);
    }
    this.#audioPtr = nextPtr;
    this.#audioCapacityFrames = nextCap;
  }

  throwLastError(): never {
    const ptr = this.exports.dnam_wasm_last_error_ptr();
    const len = this.exports.dnam_wasm_last_error_len();
    const msg =
      len > 0
        ? decodeUtf8(this.u8().subarray(ptr, ptr + len))
        : "[DNAM-VALIDATION] WASM call failed";
    throw fromNativeError(new Error(msg));
  }

  checkStatus(rc: number): number {
    if (rc < 0) {
      this.throwLastError();
    }
    return rc;
  }

  readResultBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_result_ptr();
    const len = this.exports.dnam_wasm_result_len();
    return this.u8().slice(ptr, ptr + len);
  }

  readResultString(): string {
    const ptr = this.exports.dnam_wasm_result_ptr();
    const len = this.exports.dnam_wasm_result_len();
    return decodeUtf8(this.u8().subarray(ptr, ptr + len));
  }

  readArenaBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_arena_ptr();
    const len = this.exports.dnam_wasm_arena_len();
    return this.u8().slice(ptr, ptr + len);
  }

  withBytes<T>(data: Uint8Array | undefined, fn: (ptr: number, len: number) => T): T {
    if (!data || data.length === 0) {
      return fn(0, 0);
    }
    const ptr = this.alloc(data.length);
    try {
      this.u8().set(data, ptr);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, data.length, 64);
    }
  }

  withString<T>(str: string | undefined, fn: (ptr: number, len: number) => T): T {
    if (!str || str.length === 0) {
      return fn(0, 0);
    }
    return this.withBytes(encodeUtf8(str), fn);
  }

  withF32Slice<T>(data: Float32Array, fn: (ptr: number, len: number) => T): T {
    const byteLen = Math.max(4, data.byteLength);
    const ptr = this.alloc(byteLen);
    try {
      this.f32().set(data, ptr >>> 2);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }

  withF64Slice<T>(data: Float64Array, fn: (ptr: number, len: number) => T): T {
    const byteLen = Math.max(8, data.byteLength);
    const ptr = this.alloc(byteLen);
    try {
      this.f64().set(data, ptr >>> 3);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }

  withF64Out<T>(
    len: number,
    fn: (ptr: number) => T,
  ): { readonly result: T; readonly out: Float64Array } {
    const byteLen = Math.max(8, len * 8);
    const ptr = this.alloc(byteLen);
    try {
      const result = fn(ptr);
      const start = ptr >>> 3;
      const out = this.f64().slice(start, start + len);
      return { result, out };
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }
}

class WasmEngineHandle implements NativeEngineHandle {
  readonly #mem: WasmMemoryArena;
  #ptr: number;
  #busy = false;

  constructor(mem: WasmMemoryArena, ptr: number) {
    this.#mem = mem;
    this.#ptr = ptr;
  }

  #ensurePtr(): number {
    if (this.#ptr === 0) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    return this.#ptr;
  }

  #ensureAudioIdlePtr(): number {
    const ptr = this.#ensurePtr();
    if (this.#busy) {
      throw new DnamError(
        "ERR_DNAM_BUSY",
        "Engine is busy executing an in-flight processAsync task",
      );
    }
    return ptr;
  }

  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    const ptr = this.#ensurePtr();
    this.#mem.withBytes(containerBytes, (cPtr, cLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_reload(ptr, cPtr, cLen, tier ?? -1, sampleRateHz ?? 0),
      );
    });
  }

  backend(): ActiveBackend {
    return backendFromCode(this.#mem.exports.dnam_wasm_engine_backend_code(this.#ensurePtr()));
  }

  tier(): number {
    return this.#mem.exports.dnam_wasm_engine_tier(this.#ensurePtr());
  }

  setTier(tier: number): void {
    this.#mem.checkStatus(this.#mem.exports.dnam_wasm_engine_set_tier(this.#ensurePtr(), tier));
  }

  maxSupportedTier(): number {
    return this.#mem.exports.dnam_wasm_engine_max_supported_tier(this.#ensurePtr());
  }

  setMaxSupportedTier(tier: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_max_supported_tier(this.#ensurePtr(), tier),
    );
  }

  maxAvailableTier(): number {
    return this.#mem.exports.dnam_wasm_engine_max_available_tier(this.#ensurePtr());
  }

  effectiveTier(): number {
    return this.#mem.exports.dnam_wasm_engine_effective_tier(this.#ensurePtr());
  }

  sampleRate(): number {
    return this.#mem.exports.dnam_wasm_engine_sample_rate(this.#ensurePtr());
  }

  setSampleRate(sampleRateHz: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_sample_rate(this.#ensureAudioIdlePtr(), sampleRateHz),
    );
  }

  latencySamples(): number {
    return this.#mem.exports.dnam_wasm_engine_latency_samples(this.#ensurePtr());
  }

  offlineBounce(): boolean {
    return this.#mem.exports.dnam_wasm_engine_offline_bounce(this.#ensurePtr()) !== 0;
  }

  setOfflineBounce(enabled: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_offline_bounce(this.#ensurePtr(), enabled ? 1 : 0);
  }

  maximizeTierOnBounce(): boolean {
    return this.#mem.exports.dnam_wasm_engine_maximize_tier_on_bounce(this.#ensurePtr()) !== 0;
  }

  setMaximizeTierOnBounce(enabled: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_maximize_tier_on_bounce(
      this.#ensurePtr(),
      enabled ? 1 : 0,
    );
  }

  numControls(): number {
    return this.#mem.exports.dnam_wasm_engine_num_controls(this.#ensurePtr());
  }

  rawControlValues(): Float32Array {
    const ptr = this.#ensurePtr();
    const n = this.#mem.exports.dnam_wasm_engine_num_controls(ptr);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_raw_controls_into(ptr, this.#mem.smallF32Ptr, n),
    );
    const start = this.#mem.smallF32Ptr >>> 2;
    return this.#mem.f32().slice(start, start + n);
  }

  clampedControlValue(ctrlIdx: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_clamped_control_value(ptr, ctrlIdx, this.#mem.smallF32Ptr),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  setControlValue(ctrlIdx: number, value: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_control_value(this.#ensurePtr(), ctrlIdx, value),
    );
  }

  setControlImmediate(ctrlIdx: number, value: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_control_immediate(this.#ensurePtr(), ctrlIdx, value),
    );
  }

  setControlsImmediateBatch(indices: readonly number[], values: readonly number[]): void {
    const ptr = this.#ensurePtr();
    const len = indices.length;
    if (len === 0) {
      return;
    }
    const u32View = this.#mem.u32();
    const f32View = this.#mem.f32();
    const u32Start = this.#mem.smallU32Ptr >>> 2;
    const f32Start = this.#mem.smallF32Ptr >>> 2;
    for (let i = 0; i < len; i++) {
      u32View[u32Start + i] = indices[i] ?? 0;
      f32View[f32Start + i] = values[i] ?? 0;
    }
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_controls_immediate_batch(
        ptr,
        this.#mem.smallU32Ptr,
        this.#mem.smallF32Ptr,
        len,
      ),
    );
  }

  resetControlsToDefaults(): void {
    this.#mem.exports.dnam_wasm_engine_reset_controls_to_defaults(this.#ensurePtr());
  }

  reset(): void {
    this.#mem.exports.dnam_wasm_engine_reset(this.#ensurePtr());
  }

  supportsFreeze(): boolean {
    return this.#mem.exports.dnam_wasm_engine_supports_freeze(this.#ensurePtr()) !== 0;
  }

  isFrozen(): boolean {
    return this.#mem.exports.dnam_wasm_engine_is_frozen(this.#ensurePtr()) !== 0;
  }

  setFreeze(freeze: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_freeze(this.#ensurePtr(), freeze ? 1 : 0);
  }

  setFreezeAmount(alpha: number): void {
    this.#mem.exports.dnam_wasm_engine_set_freeze_amount(this.#ensurePtr(), alpha);
  }

  bypassState(): readonly [boolean, boolean] {
    const bits = this.#mem.exports.dnam_wasm_engine_bypass_state(this.#ensurePtr());
    return [(bits & 1) !== 0, (bits & 2) !== 0];
  }

  setBypass(bypassed: boolean, trails: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_bypass(
      this.#ensurePtr(),
      bypassed ? 1 : 0,
      trails ? 1 : 0,
    );
  }

  stereoWidth(): number {
    return this.#mem.exports.dnam_wasm_engine_stereo_width(this.#ensurePtr());
  }

  setStereoWidth(width: number): void {
    this.#mem.exports.dnam_wasm_engine_set_stereo_width(this.#ensurePtr(), width);
  }

  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void {
    this.#mem.exports.dnam_wasm_engine_configure_host_mix(
      this.#ensurePtr(),
      blendWet,
      outputGainDb,
      applyUnityLeveling ? 1 : 0,
    );
  }

  currentBpm(): number {
    return this.#mem.exports.dnam_wasm_engine_current_bpm(this.#ensurePtr());
  }

  syncTempo(bpm: number, subdivision: string | undefined, foldOctave: boolean): number {
    const ptr = this.#ensurePtr();
    return this.#mem.withString(subdivision, (sPtr, sLen) =>
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_sync_tempo(ptr, bpm, sPtr, sLen, foldOctave ? 1 : 0),
      ),
    );
  }

  registerTapTempo(
    subdivision: string | undefined,
    foldOctave: boolean,
  ): readonly [number, number] {
    const ptr = this.#ensurePtr();
    return this.#mem.withString(subdivision, (sPtr, sLen) => {
      const count = this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_register_tap_tempo(
          ptr,
          sPtr,
          sLen,
          foldOctave ? 1 : 0,
          this.#mem.smallF32Ptr,
        ),
      );
      const bpm = this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 120;
      return [bpm, count] as const;
    });
  }

  applyPreset(presetId: string): void {
    const ptr = this.#ensurePtr();
    this.#mem.withString(presetId, (pPtr, pLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_engine_apply_preset(ptr, pPtr, pLen));
    });
  }

  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIo.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch1InPtr } = this.#mem.ensureAudioScratch(Math.max(n, rightIo.length));
    const f32 = this.#mem.f32();
    const lIdx = ch0InPtr >>> 2;
    const rIdx = ch1InPtr >>> 2;
    f32.set(leftIo, lIdx);
    f32.set(rightIo, rIdx);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_stereo_in_place(ptr, ch0InPtr, ch1InPtr, n),
    );
    const f32After = this.#mem.f32();
    leftIo.set(f32After.subarray(lIdx, lIdx + n));
    rightIo.set(f32After.subarray(rIdx, rIdx + n));
  }

  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIn.length;
    if (n === 0) {
      return;
    }
    if (!blockSize || blockSize >= n) {
      const { ch0InPtr, ch1InPtr } = this.#mem.ensureAudioScratch(n);
      const f32 = this.#mem.f32();
      const lIdx = ch0InPtr >>> 2;
      const rIdx = ch1InPtr >>> 2;
      f32.set(leftIn, lIdx);
      f32.set(rightIn, rIdx);
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_process_stereo_in_place(ptr, ch0InPtr, ch1InPtr, n),
      );
      const f32After = this.#mem.f32();
      leftOut.set(f32After.subarray(lIdx, lIdx + n));
      rightOut.set(f32After.subarray(rIdx, rIdx + n));
      return;
    }
    const { ch0InPtr, ch1InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    const f32 = this.#mem.f32();
    f32.set(leftIn, ch0InPtr >>> 2);
    f32.set(rightIn, ch1InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_stereo(
        ptr,
        ch0InPtr,
        ch1InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
        blockSize,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = monoIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    this.#mem.f32().set(monoIn, ch0InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_mono_to_stereo(
        ptr,
        ch0InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
        blockSize ?? 0,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = monoIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch0OutPtr } = this.#mem.ensureAudioScratch(n);
    this.#mem.f32().set(monoIn, ch0InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_mono(ptr, ch0InPtr, ch0OutPtr, n, blockSize ?? 0),
    );
    const moIdx = ch0OutPtr >>> 2;
    monoOut.set(this.#mem.f32().subarray(moIdx, moIdx + n));
  }

  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch1InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    const f32 = this.#mem.f32();
    f32.set(leftIn, ch0InPtr >>> 2);
    f32.set(rightIn, ch1InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_render_offline_bounce(
        ptr,
        ch0InPtr,
        ch1InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<readonly [Float32Array, Float32Array]> {
    this.#ensureAudioIdlePtr();
    if (rightIn.length !== leftIn.length) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `right_in length (${String(rightIn.length)}) must match left_in (${String(leftIn.length)})`,
      );
    }
    const leftSnapshot = leftIn.slice();
    const rightSnapshot = rightIn.slice();
    this.#busy = true;
    return Promise.resolve().then(() => {
      try {
        const leftOut = new Float32Array(leftSnapshot.length);
        const rightOut = new Float32Array(rightSnapshot.length);
        this.#busy = false;
        this.processStereo(leftSnapshot, rightSnapshot, leftOut, rightOut, blockSize);
        return [leftOut, rightOut] as const;
      } finally {
        this.#busy = false;
      }
    });
  }

  telemetryInto(out56: Float64Array): void {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_telemetry_into(
        ptr,
        this.#mem.smallF64Ptr,
        TELEMETRY_F64_LEN,
      ),
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    out56.set(this.#mem.f64().subarray(start, start + TELEMETRY_F64_LEN));
  }

  dispose(): void {
    if (this.#ptr !== 0) {
      this.#mem.exports.dnam_wasm_engine_free(this.#ptr);
      this.#ptr = 0;
    }
  }
}

class WasmControlSpaceHandle implements NativeControlSpaceHandle {
  readonly #mem: WasmMemoryArena;
  #ptr: number;

  constructor(mem: WasmMemoryArena, ptr: number) {
    this.#mem = mem;
    this.#ptr = ptr;
  }

  #ensurePtr(): number {
    if (this.#ptr === 0) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return this.#ptr;
  }

  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number {
    const ptr = this.#ensurePtr();
    const n = rawNumeric.length;
    if (n === 0) {
      return 0;
    }
    const baseIdx = this.#mem.smallF32Ptr >>> 2;
    const f32 = this.#mem.f32();
    f32.set(rawNumeric, baseIdx);
    const clampedPtr = this.#mem.smallF32Ptr + n * 4;
    const normPtr = clampedPtr + n * 4;
    const mask =
      this.#mem.exports.dnam_wasm_control_space_clamp_into(
        ptr,
        this.#mem.smallF32Ptr,
        clampedPtr,
        normPtr,
        n,
      ) >>> 0;
    const f32After = this.#mem.f32();
    const cIdx = clampedPtr >>> 2;
    const nIdx = normPtr >>> 2;
    clampedOut.set(f32After.subarray(cIdx, cIdx + n));
    normOut.set(f32After.subarray(nIdx, nIdx + n));
    return mask;
  }

  toPhysical(ctrlIdx: number, normalizedU: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_control_space_to_physical(
        ptr,
        ctrlIdx,
        normalizedU,
        this.#mem.smallF32Ptr,
      ),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  toNormalized(ctrlIdx: number, physicalVal: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_control_space_to_normalized(
        ptr,
        ctrlIdx,
        physicalVal,
        this.#mem.smallF32Ptr,
      ),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  dispose(): void {
    if (this.#ptr !== 0) {
      this.#mem.exports.dnam_wasm_control_space_free(this.#ptr);
      this.#ptr = 0;
    }
  }
}

export class WasmDriver implements NativeDriver {
  readonly kind = "wasm" as const;
  readonly #mem: WasmMemoryArena;

  constructor(exports: RawWasmExports, memory?: WebAssembly.Memory) {
    this.#mem = new WasmMemoryArena(exports, memory);
  }

  /**
   * Underlying linear `WebAssembly.Memory` imported into and used by this `WasmDriver` instance.
   */
  get memory(): WebAssembly.Memory {
    return this.#mem.memory;
  }

  /**
   * Returns `true` when `this.memory` is backed by a `SharedArrayBuffer` (`shared: true`).
   */
  get isSharedMemory(): boolean {
    return isSharedBuffer(this.#mem.memory.buffer);
  }

  createEngine(options: CreateNativeEngineOptions): NativeEngineHandle {
    const backendCode = backendToCode(options.backend);
    const ptr = this.#mem.withBytes(options.containerBytes, (cPtr, cLen) =>
      this.#mem.exports.dnam_wasm_engine_create(
        cPtr,
        cLen,
        options.tier,
        options.sampleRateHz,
        backendCode,
        options.maxSupportedTier ?? -1,
        options.offlineBounce ? 1 : 0,
        options.maximizeTierOnBounce === false ? 0 : 1,
      ),
    );
    if (ptr === 0) {
      this.#mem.throwLastError();
    }
    return new WasmEngineHandle(this.#mem, ptr);
  }

  createControlSpace(headerJson: string): NativeControlSpaceHandle {
    const ptr = this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.exports.dnam_wasm_control_space_create(hPtr, hLen),
    );
    if (ptr === 0) {
      this.#mem.throwLastError();
    }
    return new WasmControlSpaceHandle(this.#mem, ptr);
  }

  validateContainer(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): NativeValidatedContainerInfo {
    return this.#mem.withBytes(containerBytes, (cPtr, cLen) =>
      this.#mem.withString(deviceIdHash, (idPtr, idLen) =>
        this.#mem.withBytes(deviceSecret32, (secPtr, secLen) =>
          this.#mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_validate_container_with_key(
                cPtr,
                cLen,
                idPtr,
                idLen,
                secPtr,
                secLen,
                mktPtr,
                mktLen,
              ),
            );
            const metaJson = this.#mem.readResultString();
            const parsed = JSON.parse(metaJson) as Omit<NativeValidatedContainerInfo, "arenaBytes">;
            return {
              ...parsed,
              arenaBytes: this.#mem.readArenaBytes(),
            };
          }),
        ),
      ),
    );
  }

  buildReferenceContainer(supportsFreeze: boolean, inputs: number, outputMode: string): Uint8Array {
    return this.#mem.withString(outputMode, (mPtr, mLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_build_reference_container(
          supportsFreeze ? 1 : 0,
          inputs,
          mPtr,
          mLen,
        ),
      );
      return this.#mem.readResultBytes();
    });
  }

  buildDnamContainer(
    headerJson: string,
    arenaBytes: Uint8Array,
    encryptedArenaBytes?: Uint8Array,
    hasSignaturesFlag = false,
  ): Uint8Array {
    return this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.withBytes(arenaBytes, (aPtr, aLen) =>
        this.#mem.withBytes(encryptedArenaBytes, (ePtr, eLen) => {
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_build_dnam_container(
              hPtr,
              hLen,
              aPtr,
              aLen,
              ePtr,
              eLen,
              hasSignaturesFlag ? 1 : 0,
            ),
          );
          return this.#mem.readResultBytes();
        }),
      ),
    );
  }

  updateHeaderInPlace(containerBytes: Uint8Array, newHeaderJson: string): Uint8Array {
    return this.#mem.withBytes(containerBytes, (cPtr, cLen) =>
      this.#mem.withString(newHeaderJson, (hPtr, hLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_update_header_in_place(cPtr, cLen, hPtr, hLen),
        );
        return this.#mem.readResultBytes();
      }),
    );
  }

  stripContainerToTier(containerBytes: Uint8Array, maxTier: number): Uint8Array {
    return this.#mem.withBytes(containerBytes, (cPtr, cLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_strip_container_to_tier(cPtr, cLen, maxTier),
      );
      return this.#mem.readResultBytes();
    });
  }

  parseSchemaVersion(versionStr: string): readonly [number, number, number, boolean] {
    return this.#mem.withString(versionStr, (vPtr, vLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_parse_schema_version(vPtr, vLen, this.#mem.smallU32Ptr),
      );
      const start = this.#mem.smallU32Ptr >>> 2;
      const u32 = this.#mem.u32();
      return [
        Number(u32[start]),
        Number(u32[start + 1]),
        Number(u32[start + 2]),
        Number(u32[start + 3]) !== 0,
      ];
    });
  }

  crc32c(data: Uint8Array): number {
    return this.#mem.withBytes(
      data,
      (dPtr, dLen) => this.#mem.exports.dnam_wasm_crc32c(dPtr, dLen) >>> 0,
    );
  }

  validateAndNormalizeHeader(headerJson: string): string {
    return this.#mem.withString(headerJson, (hPtr, hLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_validate_and_normalize_header(hPtr, hLen));
      return this.#mem.readResultString();
    });
  }

  canonicalizeJcs(jsonStr: string, stripMutable = false, rejectFloats = false): string {
    return this.#mem.withString(jsonStr, (jPtr, jLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_canonicalize_jcs(
          jPtr,
          jLen,
          stripMutable ? 1 : 0,
          rejectFloats ? 1 : 0,
        ),
      );
      return this.#mem.readResultString();
    });
  }

  buildDmu1Frame(mutableJsonBytes: Uint8Array, capacity: number): Uint8Array {
    return this.#mem.withBytes(mutableJsonBytes, (jPtr, jLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_build_dmu1_frame(jPtr, jLen, capacity));
      return this.#mem.readResultBytes();
    });
  }

  parseDmu1Frame(frameBytes: Uint8Array): string {
    return this.#mem.withBytes(frameBytes, (fPtr, fLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_parse_dmu1_frame(fPtr, fLen));
      return this.#mem.readResultString();
    });
  }

  validateV2OpenDnamLayout(containerBytes: Uint8Array): string {
    return this.#mem.withBytes(containerBytes, (cPtr, cLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_validate_v2_open_dnam_layout(cPtr, cLen));
      return this.#mem.readResultString();
    });
  }

  stripV2OpenDnamContainer(containerBytes: Uint8Array, targetTier: number): Uint8Array {
    return this.#mem.withBytes(containerBytes, (cPtr, cLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_strip_v2_open_dnam_container(cPtr, cLen, targetTier),
      );
      return this.#mem.readResultBytes();
    });
  }

  uploUPackedLen(nDim: number): number {
    return this.#mem.exports.dnam_wasm_uplo_u_packed_len(nDim);
  }

  packUploUF64Into(denseFlat: Float64Array, nDim: number, outPacked: Float64Array): void {
    this.#mem.withF64Slice(denseFlat, (dPtr, dLen) => {
      const { out } = this.#mem.withF64Out(outPacked.length, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_pack_uplo_u_f64_into(
            dPtr,
            dLen,
            nDim,
            oPtr,
            outPacked.length,
          ),
        ),
      );
      outPacked.set(out);
    });
  }

  unpackUploUF64Into(packedU: Float64Array, nTier: number, outDense: Float64Array): void {
    this.#mem.withF64Slice(packedU, (pPtr, pLen) => {
      const { out } = this.#mem.withF64Out(outDense.length, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_unpack_uplo_u_f64_into(
            pPtr,
            pLen,
            nTier,
            oPtr,
            outDense.length,
          ),
        ),
      );
      outDense.set(out);
    });
  }

  computeZoh(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    rAcTrain: number,
    rDcTrain: number,
    tauRefTrain: number,
    switchXfadeTrain: number,
  ): Float64Array {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_compute_zoh(
        sampleRateTrainHz,
        sampleRateHostHz,
        rAcTrain,
        rDcTrain,
        tauRefTrain,
        switchXfadeTrain,
        this.#mem.smallF64Ptr,
      ),
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    return this.#mem.f64().slice(start, start + 9);
  }

  controlStrideForRate(sampleRateHz: number): number {
    return this.#mem.exports.dnam_wasm_control_stride_for_rate(sampleRateHz);
  }

  controlTickRatio(sampleRateHz: number): number {
    return this.#mem.exports.dnam_wasm_control_tick_ratio(sampleRateHz);
  }

  evalSubdivisionPhase(
    beatMod12: number,
    withinBeatQ32: number,
    subdivision: string,
    offsetQ32: number,
  ): readonly [number, number, number] {
    return this.#mem.withString(subdivision, (sPtr, sLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_eval_subdivision_phase(
          beatMod12,
          withinBeatQ32,
          sPtr,
          sLen,
          offsetQ32,
          this.#mem.smallF64Ptr,
        ),
      );
      const start = this.#mem.smallF64Ptr >>> 3;
      const f64 = this.#mem.f64();
      return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
    });
  }

  eval3NodeCalibration(anchors3: Float64Array, u: number): number {
    if (anchors3.length !== 3) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `anchors must have 3 elements, got ${String(anchors3.length)}`,
      );
    }
    const start = this.#mem.smallF64Ptr >>> 3;
    this.#mem.f64().set(anchors3, start);
    const outPtr = this.#mem.smallF64Ptr + 24;
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_eval_3node_calibration(this.#mem.smallF64Ptr, u, outPtr),
    );
    return this.#mem.f64()[outPtr >>> 3] ?? 0;
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
    if (anchors3.length !== 3) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `anchors must have 3 elements, got ${String(anchors3.length)}`,
      );
    }
    const start = this.#mem.smallF64Ptr >>> 3;
    this.#mem.f64().set(anchors3, start);
    const outPtr = this.#mem.smallF64Ptr + 24;
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_invert_tempo_calibration(
        declaredMin,
        declaredMax,
        safeMin,
        safeMax,
        this.#mem.smallF64Ptr,
        target,
        flushSubnormals ? 1 : 0,
        outPtr,
      ),
    );
    const outStart = outPtr >>> 3;
    return this.#mem.f64().slice(outStart, outStart + 5);
  }

  verifyContainerProvenance(
    headerJson: string,
    plaintextArena: Uint8Array,
    hasSignaturesFlag: boolean,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): string {
    return this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.withBytes(plaintextArena, (aPtr, aLen) =>
        this.#mem.withString(deviceIdHash, (idPtr, idLen) =>
          this.#mem.withBytes(deviceSecret32, (secPtr, secLen) =>
            this.#mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
              this.#mem.checkStatus(
                this.#mem.exports.dnam_wasm_verify_container_provenance(
                  hPtr,
                  hLen,
                  aPtr,
                  aLen,
                  hasSignaturesFlag ? 1 : 0,
                  idPtr,
                  idLen,
                  secPtr,
                  secLen,
                  mktPtr,
                  mktLen,
                ),
              );
              return this.#mem.readResultString();
            }),
          ),
        ),
      ),
    );
  }

  enforceDistributionPolicy(headerJson: string, activeProfile: string): void {
    this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.withString(activeProfile, (pPtr, pLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_enforce_distribution_policy(hPtr, hLen, pPtr, pLen),
        );
      }),
    );
  }

  evaluateAnticlonePolicy(
    fingerprintMatched: boolean,
    buyerWatermarkIdentified: boolean,
    independentDeviceSourceEvidence: boolean,
  ): string {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_evaluate_anticlone_policy(
        fingerprintMatched ? 1 : 0,
        buyerWatermarkIdentified ? 1 : 0,
        independentDeviceSourceEvidence ? 1 : 0,
      ),
    );
    return this.#mem.readResultString();
  }

  immutableHeaderSha256(headerJson: string): string {
    return this.#mem.withString(headerJson, (hPtr, hLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_immutable_header_sha256(hPtr, hLen));
      return this.#mem.readResultString();
    });
  }

  sha256Hex(data: Uint8Array): string {
    return this.#mem.withBytes(data, (dPtr, dLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_sha256_hex(dPtr, dLen));
      return this.#mem.readResultString();
    });
  }

  hmacSha256Hex(key32: Uint8Array, message: Uint8Array): string {
    return this.#mem.withBytes(key32, (kPtr, kLen) =>
      this.#mem.withBytes(message, (mPtr, mLen) => {
        this.#mem.checkStatus(this.#mem.exports.dnam_wasm_hmac_sha256_hex(kPtr, kLen, mPtr, mLen));
        return this.#mem.readResultString();
      }),
    );
  }

  ed25519PubkeyHex(seed32: Uint8Array): string {
    return this.#mem.withBytes(seed32, (sPtr, sLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_ed25519_pubkey_hex(sPtr, sLen));
      return this.#mem.readResultString();
    });
  }

  ed25519SignHex(seed32: Uint8Array, message: Uint8Array): string {
    return this.#mem.withBytes(seed32, (sPtr, sLen) =>
      this.#mem.withBytes(message, (mPtr, mLen) => {
        this.#mem.checkStatus(this.#mem.exports.dnam_wasm_ed25519_sign_hex(sPtr, sLen, mPtr, mLen));
        return this.#mem.readResultString();
      }),
    );
  }

  x25519PubkeyHex(secret32: Uint8Array): string {
    return this.#mem.withBytes(secret32, (sPtr, sLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_x25519_pubkey_hex(sPtr, sLen));
      return this.#mem.readResultString();
    });
  }

  buildCreatorSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array {
    return this.#mem.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
      this.#mem.withString(masterArenaSha256Hex, (aPtr, aLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_build_creator_signing_message(hPtr, hLen, aPtr, aLen),
        );
        return this.#mem.readResultBytes();
      }),
    );
  }

  buildMarketplaceSigningMessage(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array {
    return this.#mem.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
      this.#mem.withString(masterArenaSha256Hex, (aPtr, aLen) =>
        this.#mem.withString(receiptJson, (rPtr, rLen) => {
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_build_marketplace_signing_message(
              hPtr,
              hLen,
              aPtr,
              aLen,
              rPtr,
              rLen,
            ),
          );
          return this.#mem.readResultBytes();
        }),
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
    return this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.withBytes(masterArena, (aPtr, aLen) =>
        this.#mem.withBytes(creatorSeed32, (sPtr, sLen) =>
          this.#mem.withString(signerId, (idPtr, idLen) => {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_sign_creator_header(
                hPtr,
                hLen,
                aPtr,
                aLen,
                sPtr,
                sLen,
                idPtr,
                idLen,
                includeTierPrefixes ? 1 : 0,
              ),
            );
            return this.#mem.readResultString();
          }),
        ),
      ),
    );
  }

  signMarketplaceReceipt(
    headerJson: string,
    buyerArena: Uint8Array,
    marketplaceSeed32: Uint8Array,
    specJson: string,
  ): string {
    return this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.withBytes(buyerArena, (aPtr, aLen) =>
        this.#mem.withBytes(marketplaceSeed32, (sPtr, sLen) =>
          this.#mem.withString(specJson, (spPtr, spLen) => {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_sign_marketplace_receipt(
                hPtr,
                hLen,
                aPtr,
                aLen,
                sPtr,
                sLen,
                spPtr,
                spLen,
              ),
            );
            return this.#mem.readResultString();
          }),
        ),
      ),
    );
  }

  signOemDeviceCert(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX2551932: Uint8Array,
  ): string {
    return this.#mem.withBytes(oemSeed32, (sPtr, sLen) =>
      this.#mem.withString(oemVendor, (vPtr, vLen) =>
        this.#mem.withString(modelCode, (mPtr, mLen) =>
          this.#mem.withString(deviceIdHash, (dPtr, dLen) =>
            this.#mem.withBytes(pubkeyX2551932, (pPtr, pLen) => {
              this.#mem.checkStatus(
                this.#mem.exports.dnam_wasm_sign_oem_device_cert(
                  sPtr,
                  sLen,
                  vPtr,
                  vLen,
                  mPtr,
                  mLen,
                  dPtr,
                  dLen,
                  pPtr,
                  pLen,
                ),
              );
              return this.#mem.readResultString();
            }),
          ),
        ),
      ),
    );
  }

  verifyOemDeviceCert(certJson: string, trustedOemPubkeyHex: string): string {
    return this.#mem.withString(certJson, (cPtr, cLen) =>
      this.#mem.withString(trustedOemPubkeyHex, (pPtr, pLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_verify_oem_device_cert(cPtr, cLen, pPtr, pLen),
        );
        return this.#mem.readResultString();
      }),
    );
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
    return this.#mem.withString(oemVendor, (vPtr, vLen) =>
      this.#mem.withString(modelCode, (mPtr, mLen) =>
        this.#mem.withBytes(deviceIdHash16, (dPtr, dLen) =>
          this.#mem.withBytes(pubkeyX2551932, (pPtr, pLen) =>
            this.#mem.withBytes(issuerPubkeyEd2551932, (iPtr, iLen) =>
              this.#mem.withBytes(attestationPolicySha25632, (polPtr, polLen) =>
                this.#mem.withBytes(attestationEvidenceSha25632, (evPtr, evLen) => {
                  this.#mem.checkStatus(
                    this.#mem.exports.dnam_wasm_build_oem_cert_v2_signing_message(
                      vPtr,
                      vLen,
                      mPtr,
                      mLen,
                      dPtr,
                      dLen,
                      pPtr,
                      pLen,
                      iPtr,
                      iLen,
                      assurance,
                      polPtr,
                      polLen,
                      evPtr,
                      evLen,
                    ),
                  );
                  return this.#mem.readResultBytes();
                }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  chacha20XorArena(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter: number,
  ): Uint8Array {
    return this.#mem.withBytes(arenaBytes, (aPtr, aLen) =>
      this.#mem.withBytes(key32, (kPtr, kLen) =>
        this.#mem.withBytes(nonce12, (nPtr, nLen) => {
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_chacha20_xor_arena(
              aPtr,
              aLen,
              kPtr,
              kLen,
              nPtr,
              nLen,
              startBlockCounter,
            ),
          );
          return this.#mem.readResultBytes();
        }),
      ),
    );
  }

  wrapDeviceCekHex(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string {
    return this.#mem.withBytes(ephemeralSecret32, (ePtr, eLen) =>
      this.#mem.withBytes(devicePubkey32, (dPtr, dLen) =>
        this.#mem.withString(deviceIdHash, (idPtr, idLen) =>
          this.#mem.withBytes(cek32, (cPtr, cLen) => {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_wrap_device_cek_hex(
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
            return this.#mem.readResultString();
          }),
        ),
      ),
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
    return this.#mem.withBytes(innerDnamBytes, (iPtr, iLen) =>
      this.#mem.withString(recipientCertsJson, (cPtr, cLen) =>
        this.#mem.withString(trustedOemPubkeysJson, (oPtr, oLen) =>
          this.#mem.withBytes(marketplaceSeed32, (mPtr, mLen) =>
            this.#mem.withBytes(ephemeralSecret32, (ePtr, eLen) =>
              this.#mem.withBytes(cek32, (kPtr, kLen) =>
                this.#mem.withBytes(nonce12, (nPtr, nLen) => {
                  this.#mem.checkStatus(
                    this.#mem.exports.dnam_wasm_seal_container(
                      iPtr,
                      iLen,
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
                  return this.#mem.readResultBytes();
                }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  unsealContainer(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array {
    return this.#mem.withBytes(sealedBytes, (sPtr, sLen) =>
      this.#mem.withString(deviceIdHash, (idPtr, idLen) =>
        this.#mem.withBytes(deviceSecret32, (secPtr, secLen) =>
          this.#mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_unseal_container(
                sPtr,
                sLen,
                idPtr,
                idLen,
                secPtr,
                secLen,
                mktPtr,
                mktLen,
              ),
            );
            return this.#mem.readResultBytes();
          }),
        ),
      ),
    );
  }

  encodeDau1Envelope(specJson: string): Uint8Array {
    return this.#mem.withString(specJson, (sPtr, sLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_encode_dau1_envelope(sPtr, sLen));
      return this.#mem.readResultBytes();
    });
  }

  decodeDau1Envelope(wireBytes: Uint8Array): string {
    return this.#mem.withBytes(wireBytes, (wPtr, wLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_decode_dau1_envelope(wPtr, wLen));
      return this.#mem.readResultString();
    });
  }

  buildDme1Header(profile: number, recipientCount: number, ciphertextLength: number): Uint8Array {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_build_dme1_header(
        profile,
        recipientCount,
        BigInt(ciphertextLength),
      ),
    );
    return this.#mem.readResultBytes();
  }

  parseDme1Plaintext(plaintext: Uint8Array, recipientKeyDigestsHex: readonly string[]): string {
    return this.#mem.withBytes(plaintext, (pPtr, pLen) =>
      this.#mem.withString(JSON.stringify(recipientKeyDigestsHex), (dPtr, dLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_parse_dme1_plaintext(pPtr, pLen, dPtr, dLen),
        );
        return this.#mem.readResultString();
      }),
    );
  }

  encodeRecipientKeyIds(keyDigestsHex: readonly string[]): Uint8Array {
    return this.#mem.withString(JSON.stringify(keyDigestsHex), (jPtr, jLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_encode_recipient_key_ids(jPtr, jLen));
      return this.#mem.readResultBytes();
    });
  }

  parseRecipientKeyIds(data: Uint8Array, maxRecipients: number): readonly string[] {
    return this.#mem.withBytes(data, (dPtr, dLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_parse_recipient_key_ids(dPtr, dLen, maxRecipients),
      );
      return JSON.parse(this.#mem.readResultString()) as string[];
    });
  }

  encodeHpkeSlots(slotsJson: string): Uint8Array {
    return this.#mem.withString(slotsJson, (jPtr, jLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_encode_hpke_slots(jPtr, jLen));
      return this.#mem.readResultBytes();
    });
  }

  parseHpkeSlots(data: Uint8Array, expectedCount: number, maxRecipients: number): string {
    return this.#mem.withBytes(data, (dPtr, dLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_parse_hpke_slots(dPtr, dLen, expectedCount, maxRecipients),
      );
      return this.#mem.readResultString();
    });
  }

  encodeRecordDirectory(maximumTier: number, recordsJson: string): Uint8Array {
    return this.#mem.withString(recordsJson, (jPtr, jLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_encode_record_directory(maximumTier, jPtr, jLen),
      );
      return this.#mem.readResultBytes();
    });
  }

  parseRecordDirectory(data: Uint8Array, maximumTier: number): string {
    return this.#mem.withBytes(data, (dPtr, dLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_parse_record_directory(dPtr, dLen, maximumTier),
      );
      return this.#mem.readResultString();
    });
  }

  computePreviewWindow(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
  ): readonly [number, number] {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_compute_preview_window(
        anchor,
        delta,
        safeMin,
        safeMax,
        this.#mem.smallF64Ptr,
      ),
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [Number(f64[start]), Number(f64[start + 1])];
  }

  validatePreviewSwitchOptions(
    preset: string,
    alternative: string,
    observed: readonly string[],
    active: readonly string[],
    safe: readonly string[],
  ): readonly [string, string] {
    const specJson = JSON.stringify({ preset, alternative, observed, active, safe });
    return this.#mem.withString(specJson, (jPtr, jLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_validate_preview_switch_options(jPtr, jLen),
      );
      const parsed = JSON.parse(this.#mem.readResultString()) as [string, string];
      return [parsed[0], parsed[1]];
    });
  }

  validateRoyaltySplits(licenseTier: string, royaltySplitsJson: string): void {
    this.#mem.withString(licenseTier, (tPtr, tLen) =>
      this.#mem.withString(royaltySplitsJson, (sPtr, sLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_validate_royalty_splits(tPtr, tLen, sPtr, sLen),
        );
      }),
    );
  }

  applyIsometricGaugeWatermark(
    arenaBytes: Uint8Array,
    kappa64: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array {
    return this.#mem.withBytes(arenaBytes, (aPtr, aLen) =>
      this.#mem.withF32Slice(kappa64, (kPtr, kLen) =>
        this.#mem.withF32Slice(qBuyerFlat16, (qPtr, qLen) => {
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_apply_isometric_gauge_watermark(
              aPtr,
              aLen,
              kPtr,
              kLen,
              qPtr,
              qLen,
            ),
          );
          return this.#mem.readResultBytes();
        }),
      ),
    );
  }

  generateTardosCode(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array {
    return this.#mem.withBytes(marketplaceSecret32, (sPtr, sLen) =>
      this.#mem.withString(modelId, (mPtr, mLen) =>
        this.#mem.withString(licenseId, (lPtr, lLen) => {
          const outLen = 7680;
          const outPtr = this.#mem.alloc(outLen * 4);
          try {
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_generate_tardos_code(
                sPtr,
                sLen,
                mPtr,
                mLen,
                lPtr,
                lLen,
                outPtr,
                outLen,
              ),
            );
            const start = outPtr >>> 2;
            return this.#mem.f32().slice(start, start + outLen);
          } finally {
            this.#mem.exports.dnam_wasm_dealloc(outPtr, outLen * 4, 64);
          }
        }),
      ),
    );
  }

  applyTardosWatermark(
    arenaBytes: Uint8Array,
    code7680: Float32Array,
    epsilon: number,
  ): Uint8Array {
    return this.#mem.withBytes(arenaBytes, (aPtr, aLen) =>
      this.#mem.withF32Slice(code7680, (cPtr, cLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_apply_tardos_watermark(aPtr, aLen, cPtr, cLen, epsilon),
        );
        return this.#mem.readResultBytes();
      }),
    );
  }

  scoreTardosWatermark(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code7680: Float32Array,
  ): number {
    return this.#mem.withBytes(masterArena, (mPtr, mLen) =>
      this.#mem.withBytes(suspectArena, (sPtr, sLen) =>
        this.#mem.withF32Slice(code7680, (cPtr, cLen) => {
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_score_tardos_watermark(
              mPtr,
              mLen,
              sPtr,
              sLen,
              cPtr,
              cLen,
              this.#mem.smallF64Ptr,
            ),
          );
          return this.#mem.f64()[this.#mem.smallF64Ptr >>> 3] ?? 0;
        }),
      ),
    );
  }

  extractStructuralHash(arenaBytes: Uint8Array): Float64Array {
    return this.#mem.withBytes(arenaBytes, (aPtr, aLen) => {
      const cap = 224;
      const { result: len, out } = this.#mem.withF64Out(cap, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_extract_structural_hash(aPtr, aLen, oPtr, cap),
        ),
      );
      return out.slice(0, len);
    });
  }

  structuralCosineSimilarity(a: Float64Array, b: Float64Array): number {
    return this.#mem.withF64Slice(a, (aPtr, aLen) =>
      this.#mem.withF64Slice(b, (bPtr, bLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_structural_cosine_similarity(
            aPtr,
            aLen,
            bPtr,
            bLen,
            this.#mem.smallF64Ptr,
          ),
        );
        return this.#mem.f64()[this.#mem.smallF64Ptr >>> 3] ?? 0;
      }),
    );
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
    return this.#mem.withBytes(key32, (kPtr, kLen) =>
      this.#mem.withBytes(marketplacePubkey32, (mpPtr, mpLen) =>
        this.#mem.withBytes(watermarkKeyId16, (wkPtr, wkLen) =>
          this.#mem.withBytes(creatorSignedManifestSha25632, (cmPtr, cmLen) =>
            this.#mem.withBytes(modelId16, (miPtr, miLen) =>
              this.#mem.withBytes(watermarkProfileSha25632, (wpPtr, wpLen) =>
                this.#mem.withBytes(licenseId16, (liPtr, liLen) =>
                  this.#mem.withBytes(buyerId16, (biPtr, biLen) => {
                    this.#mem.checkStatus(
                      this.#mem.exports.dnam_wasm_derive_watermark_seed_hex(
                        kPtr,
                        kLen,
                        mpPtr,
                        mpLen,
                        wkPtr,
                        wkLen,
                        cmPtr,
                        cmLen,
                        miPtr,
                        miLen,
                        wpPtr,
                        wpLen,
                        owningTier,
                        liPtr,
                        liLen,
                        biPtr,
                        biLen,
                      ),
                    );
                    return this.#mem.readResultString();
                  }),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  extractStage3Fingerprint(
    earlyFir128b: Uint8Array,
    kautzModalBytes: Uint8Array,
    addedModes: number,
  ): Float64Array {
    return this.#mem.withBytes(earlyFir128b, (fPtr, fLen) =>
      this.#mem.withBytes(kautzModalBytes, (kPtr, kLen) => {
        const cap = 256;
        const { result: len, out } = this.#mem.withF64Out(cap, (oPtr) =>
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_extract_stage3_fingerprint(
              fPtr,
              fLen,
              kPtr,
              kLen,
              addedModes,
              oPtr,
              cap,
            ),
          ),
        );
        return out.slice(0, len);
      }),
    );
  }

  catmullRomHermite(sM1: number, s0: number, s1: number, s2: number, mu: number): number {
    return this.#mem.exports.dnam_wasm_catmull_rom_hermite(sM1, s0, s1, s2, mu);
  }

  catmullRomHermiteSlice(
    sM1: Float64Array,
    s0: Float64Array,
    s1: Float64Array,
    s2: Float64Array,
    mu: Float64Array,
    out: Float64Array,
  ): void {
    const n = s0.length;
    this.#mem.withF64Slice(sM1, (pM1) =>
      this.#mem.withF64Slice(s0, (p0) =>
        this.#mem.withF64Slice(s1, (p1) =>
          this.#mem.withF64Slice(s2, (p2) =>
            this.#mem.withF64Slice(mu, (pMu) => {
              const { out: res } = this.#mem.withF64Out(n, (pOut) =>
                this.#mem.checkStatus(
                  this.#mem.exports.dnam_wasm_catmull_rom_hermite_slice(
                    pM1,
                    p0,
                    p1,
                    p2,
                    pMu,
                    pOut,
                    n,
                  ),
                ),
              );
              out.set(res);
            }),
          ),
        ),
      ),
    );
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
    this.#mem.exports.dnam_wasm_op_t_coercive_step(
      uIn,
      gLin,
      sigmaKnee,
      vCeil,
      rDc,
      dcXPrev,
      dcYPrev,
      this.#mem.smallF64Ptr,
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])];
  }

  opTTankSlice(
    inp: Float64Array,
    out: Float64Array,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void {
    this.#mem.withF64Slice(inp, (iPtr, n) => {
      const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_op_t_tank_slice(iPtr, oPtr, n, gLin, sigmaKnee, vCeil, rDc),
        ),
      );
      out.set(res);
    });
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
    this.#mem.exports.dnam_wasm_op_m_modal_step(
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
      this.#mem.smallF64Ptr,
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])];
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
    return this.#mem.withF64Slice(inp, (iPtr, n) => {
      const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_op_m_modal_slice(
            iPtr,
            oPtr,
            n,
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
            this.#mem.smallF64Ptr,
          ),
        ),
      );
      out.set(res);
      const start = this.#mem.smallF64Ptr >>> 3;
      const f64 = this.#mem.f64();
      return [Number(f64[start]), Number(f64[start + 1])] as const;
    });
  }

  unitRotor(phi: number): readonly [number, number] {
    this.#mem.exports.dnam_wasm_unit_rotor(phi, this.#mem.smallF64Ptr);
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [Number(f64[start]), Number(f64[start + 1])];
  }

  unitRotorSlice(phi: Float64Array, out: Float64Array): void {
    this.#mem.withF64Slice(phi, (pPtr, n) => {
      const { out: res } = this.#mem.withF64Out(out.length, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_unit_rotor_slice(pPtr, n, oPtr, out.length),
        ),
      );
      out.set(res);
    });
  }

  polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number {
    return this.#mem.exports.dnam_wasm_poly_adaa_algebraic(uPrev, uCurr, beta, delta);
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
    this.#mem.exports.dnam_wasm_poly_adaa_step(
      x,
      bConv,
      alpha,
      beta,
      delta,
      rAc,
      uPrev,
      aPrev,
      mDc,
      this.#mem.smallF64Ptr,
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [
      Number(f64[start]),
      Number(f64[start + 1]),
      Number(f64[start + 2]),
      Number(f64[start + 3]),
    ];
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
    return this.#mem.withF64Slice(inp, (iPtr, n) => {
      const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_poly_adaa_slice(
            iPtr,
            oPtr,
            n,
            bConv,
            alpha,
            beta,
            delta,
            rAc,
            uPrev,
            aPrev,
            mDc,
            reset ? 1 : 0,
            this.#mem.smallF64Ptr,
          ),
        ),
      );
      out.set(res);
      const start = this.#mem.smallF64Ptr >>> 3;
      const f64 = this.#mem.f64();
      return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
    });
  }

  evalLobattoBasis(
    u: number,
    anchors3: Float64Array,
    degree: number,
  ): readonly [number, number, number] {
    return this.#mem.withF64Slice(anchors3, (aPtr, aLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_eval_lobatto_basis(
          u,
          aPtr,
          aLen,
          degree,
          this.#mem.smallF64Ptr,
        ),
      );
      const start = this.#mem.smallF64Ptr >>> 3;
      const f64 = this.#mem.f64();
      return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
    });
  }

  evalLobattoBasisSlice(
    u: Float64Array,
    out: Float64Array,
    anchors3: Float64Array,
    degree: number,
  ): void {
    this.#mem.withF64Slice(u, (uPtr, uLen) =>
      this.#mem.withF64Slice(anchors3, (aPtr, aLen) => {
        const { out: res } = this.#mem.withF64Out(out.length, (oPtr) =>
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_eval_lobatto_basis_slice(
              uPtr,
              uLen,
              oPtr,
              out.length,
              aPtr,
              aLen,
              degree,
            ),
          ),
        );
        out.set(res);
      }),
    );
  }

  evalRank4Fm(
    features: Float64Array,
    vFactorsFlat: Float64Array,
  ): readonly [number, number, number, number] {
    return this.#mem.withF64Slice(features, (fPtr, fLen) =>
      this.#mem.withF64Slice(vFactorsFlat, (vPtr, vLen) => {
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_eval_rank4_fm(fPtr, fLen, vPtr, vLen, this.#mem.smallF64Ptr),
        );
        const start = this.#mem.smallF64Ptr >>> 3;
        const f64 = this.#mem.f64();
        return [
          Number(f64[start]),
          Number(f64[start + 1]),
          Number(f64[start + 2]),
          Number(f64[start + 3]),
        ] as const;
      }),
    );
  }

  evalRank4FmSlice(
    featuresFlat: Float64Array,
    nFrames: number,
    dDim: number,
    vFactorsFlat: Float64Array,
    out: Float64Array,
  ): void {
    this.#mem.withF64Slice(featuresFlat, (fPtr, fLen) =>
      this.#mem.withF64Slice(vFactorsFlat, (vPtr, vLen) => {
        const { out: res } = this.#mem.withF64Out(out.length, (oPtr) =>
          this.#mem.checkStatus(
            this.#mem.exports.dnam_wasm_eval_rank4_fm_slice(
              fPtr,
              fLen,
              nFrames,
              dDim,
              vPtr,
              vLen,
              oPtr,
              out.length,
            ),
          ),
        );
        out.set(res);
      }),
    );
  }

  evalChebyshevLfo(cosPhi: number): readonly [number, number, number, number] {
    this.#mem.exports.dnam_wasm_eval_chebyshev_lfo(cosPhi, this.#mem.smallF64Ptr);
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [
      Number(f64[start]),
      Number(f64[start + 1]),
      Number(f64[start + 2]),
      Number(f64[start + 3]),
    ];
  }

  evalChebyshevLfoSlice(cosPhi: Float64Array, out: Float64Array): void {
    this.#mem.withF64Slice(cosPhi, (cPtr, cLen) => {
      const { out: res } = this.#mem.withF64Out(out.length, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_eval_chebyshev_lfo_slice(cPtr, cLen, oPtr, out.length),
        ),
      );
      out.set(res);
    });
  }

  evalTrustRegionSoftclip(z: number, mh: number): number {
    return this.#mem.exports.dnam_wasm_eval_trust_region_softclip(z, mh);
  }

  evalTrustRegionSoftclipSlice(z: Float64Array, out: Float64Array, mh: number): void {
    this.#mem.withF64Slice(z, (zPtr, n) => {
      const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_eval_trust_region_softclip_slice(zPtr, oPtr, n, mh),
        ),
      );
      out.set(res);
    });
  }

  cubicLagrangeWeights(mu: number): readonly [number, number, number, number] {
    this.#mem.exports.dnam_wasm_cubic_lagrange_weights(mu, this.#mem.smallF64Ptr);
    const start = this.#mem.smallF64Ptr >>> 3;
    const f64 = this.#mem.f64();
    return [
      Number(f64[start]),
      Number(f64[start + 1]),
      Number(f64[start + 2]),
      Number(f64[start + 3]),
    ];
  }

  cubicLagrangeWeightsSlice(mu: Float64Array, out: Float64Array): void {
    this.#mem.withF64Slice(mu, (mPtr, n) => {
      const { out: res } = this.#mem.withF64Out(out.length, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_cubic_lagrange_weights_slice(mPtr, n, oPtr, out.length),
        ),
      );
      out.set(res);
    });
  }

  givensHouseholder8ch(x8: Float64Array, cosPsi: number, sinPsi: number): Float64Array {
    return this.#mem.withF64Slice(x8, (xPtr, xLen) => {
      const { out } = this.#mem.withF64Out(8, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_givens_householder_8ch(xPtr, xLen, cosPsi, sinPsi, oPtr),
        ),
      );
      return out;
    });
  }

  givensHouseholder8chSlice(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
  ): void {
    this.#mem.withF64Slice(xFlat, (xPtr, n) => {
      const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
        this.#mem.checkStatus(
          this.#mem.exports.dnam_wasm_givens_householder_8ch_slice(xPtr, oPtr, n, cosPsi, sinPsi),
        ),
      );
      out.set(res);
    });
  }

  givensHouseholder16ch(
    x16: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos8: Float64Array,
    crossSin8: Float64Array,
  ): Float64Array {
    return this.#mem.withF64Slice(x16, (xPtr, xLen) =>
      this.#mem.withF64Slice(crossCos8, (ccPtr, ccLen) =>
        this.#mem.withF64Slice(crossSin8, (csPtr, csLen) => {
          const { out } = this.#mem.withF64Out(16, (oPtr) =>
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_givens_householder_16ch(
                xPtr,
                xLen,
                cosPsi,
                sinPsi,
                ccPtr,
                ccLen,
                csPtr,
                csLen,
                oPtr,
              ),
            ),
          );
          return out;
        }),
      ),
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
    this.#mem.withF64Slice(xFlat, (xPtr, n) =>
      this.#mem.withF64Slice(crossCos8, (ccPtr, ccLen) =>
        this.#mem.withF64Slice(crossSin8, (csPtr, csLen) => {
          const { out: res } = this.#mem.withF64Out(n, (oPtr) =>
            this.#mem.checkStatus(
              this.#mem.exports.dnam_wasm_givens_householder_16ch_slice(
                xPtr,
                oPtr,
                n,
                cosPsi,
                sinPsi,
                ccPtr,
                ccLen,
                csPtr,
                csLen,
              ),
            ),
          );
          out.set(res);
        }),
      ),
    );
  }
}

/**
 * Minimum initial 64 KiB WebAssembly linear-memory pages enforced by the SDK (`18` pages = `1.125 MiB`,
 * safely above the `15`-page module minimum from `-zstack-size=786432` + static data).
 */
export const WASM_MIN_INITIAL_PAGES = 18;

/**
 * Default initial 64 KiB WebAssembly linear-memory pages (`18` pages = `1.125 MiB`, providing
 * `15` pages for the `768 KiB` shadow stack + static data and `3` initial heap pages before `memory.grow`).
 */
export const WASM_DEFAULT_INITIAL_PAGES = 18;

/**
 * Default maximum 64 KiB WebAssembly linear-memory pages (`4,096` pages = `256 MiB`, so V8
 * reserves bounded virtual address space upfront without over-reserving multi-GiB cages).
 */
export const WASM_DEFAULT_MAXIMUM_PAGES = 4096;

/**
 * Architectural `wasm32` ceiling in 64 KiB pages (`65,536` pages = `4 GiB`).
 */
export const WASM_MAX_PAGES = 65536;

/**
 * Linear-memory configuration for `new WebAssembly.Memory({ initial, maximum, shared })`
 * (`ref[sdk.ts.api.sync-init]`, `ref[sdk.ts.engine.zero-copy-and-async]`).
 */
export interface WasmMemoryOptions {
  readonly memory?: WebAssembly.Memory;
  readonly initialPages?: number;
  readonly maximumPages?: number;
  readonly shared?: boolean;
}

/**
 * Options for `initWasm(...)` (`ref[sdk.ts.api.sync-init]`).
 */
export interface InitWasmOptions extends WasmMemoryOptions {
  readonly module?: WebAssembly.Module;
  readonly bytes?: BufferSource;
  readonly url?: string | URL;
}

/**
 * Returns `true` if `new WebAssembly.Memory({ initial, maximum, shared: true })` (`SharedArrayBuffer`)
 * is supported and permitted by the current execution environment (`crossOriginIsolated !== false`).
 */
export function isSharedWasmMemorySupported(): boolean {
  if (typeof SharedArrayBuffer === "undefined" || typeof Atomics === "undefined") {
    return false;
  }
  if (
    typeof globalThis !== "undefined" &&
    "crossOriginIsolated" in globalThis &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === false
  ) {
    return false;
  }
  try {
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true });
    return isSharedBuffer(mem.buffer);
  } catch {
    return false;
  }
}

/**
 * Returns `true` when `memory.buffer` is backed by a `SharedArrayBuffer` (`shared: true`).
 */
export function isSharedWasmMemory(memory: WebAssembly.Memory): boolean {
  return isSharedBuffer(memory.buffer);
}

/**
 * Creates a bounded `WebAssembly.Memory` suitable for `{ env: { memory } }` instantiation of
 * `dnam.wasm`, supporting optional `shared: true` when permitted by the environment.
 */
export function createWasmMemory(options?: WasmMemoryOptions): WebAssembly.Memory {
  if (options?.memory instanceof WebAssembly.Memory) {
    return options.memory;
  }
  const initial = Math.min(
    WASM_MAX_PAGES,
    Math.max(
      WASM_MIN_INITIAL_PAGES,
      Math.trunc(options?.initialPages ?? WASM_DEFAULT_INITIAL_PAGES),
    ),
  );
  const maximum = Math.min(
    WASM_MAX_PAGES,
    Math.max(initial, Math.trunc(options?.maximumPages ?? WASM_DEFAULT_MAXIMUM_PAGES)),
  );
  if (options?.shared && isSharedWasmMemorySupported()) {
    try {
      return new WebAssembly.Memory({ initial, maximum, shared: true });
    } catch {
      // Fall back cleanly to non-shared WebAssembly.Memory if disallowed by runtime policy.
    }
  }
  return new WebAssembly.Memory({ initial, maximum });
}

const ENV_MEMORY_IMPORT_PREFIX = [
  0x01, 0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02,
] as const;

function encodeLeb128U32(val: number): number[] {
  const out: number[] = [];
  let v = val >>> 0;
  do {
    let b = v & 0x7f;
    v >>>= 7;
    if (v !== 0) {
      b |= 0x80;
    }
    out.push(b);
  } while (v !== 0);
  return out;
}

/**
 * Returns a `Uint8Array` copy of `wasmBytes` whose Section 2 (`Import`) `"env"."memory"` descriptor
 * matches `shared` and `maxPages`:
 * - `maxPages !== null` (default `4,096` pages = `256 MiB`): `flags = shared ? 0x03 : 0x01`,
 *   `min = minPages`, `max = maxPages`, enabling V8 compile-time static guard-page bounds-check elision.
 * - `maxPages === null`: `flags = 0x00, min = minPages` when `shared === false` (accepts any unshared
 *   `WebAssembly.Memory` with `initial >= minPages`, with or without `maximum`), or
 *   `flags = 0x03, min = minPages, max = 65536` when `shared === true`.
 */
export function configureWasmMemoryImport(
  wasmBytes: Uint8Array,
  shared: boolean,
  maxPages: number | null = WASM_DEFAULT_MAXIMUM_PAGES,
): Uint8Array<ArrayBuffer> {
  if (
    wasmBytes.length < 8 ||
    wasmBytes[0] !== 0x00 ||
    wasmBytes[1] !== 0x61 ||
    wasmBytes[2] !== 0x73 ||
    wasmBytes[3] !== 0x6d
  ) {
    const copy = new Uint8Array(wasmBytes.length);
    copy.set(wasmBytes);
    return copy;
  }

  let pos = 8;
  const readLeb128U32 = (): number => {
    let res = 0;
    let shift = 0;
    while (pos < wasmBytes.length) {
      const b = wasmBytes[pos++] ?? 0;
      res |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) {
        return res >>> 0;
      }
      shift += 7;
    }
    return res >>> 0;
  };

  while (pos < wasmBytes.length) {
    const secStart = pos;
    const secId = wasmBytes[pos++] ?? 0;
    const secLen = readLeb128U32();
    const payloadStart = pos;
    const secEnd = payloadStart + secLen;
    if (secEnd > wasmBytes.length) {
      break;
    }
    if (secId === 2 && secLen >= ENV_MEMORY_IMPORT_PREFIX.length + 2) {
      let matched = true;
      for (let i = 0; i < ENV_MEMORY_IMPORT_PREFIX.length; i++) {
        if (wasmBytes[payloadStart + i] !== ENV_MEMORY_IMPORT_PREFIX[i]) {
          matched = false;
          break;
        }
      }
      if (matched) {
        pos = payloadStart + ENV_MEMORY_IMPORT_PREFIX.length;
        const oldFlags = wasmBytes[pos++] ?? 0;
        const minPages = readLeb128U32();
        if ((oldFlags & 0x01) !== 0) {
          readLeb128U32();
        }
        const restPayload = wasmBytes.subarray(pos, secEnd);
        const resolvedMax =
          maxPages !== null
            ? Math.min(WASM_MAX_PAGES, Math.max(minPages, Math.trunc(maxPages)))
            : shared
              ? WASM_MAX_PAGES
              : null;
        const newLimits =
          resolvedMax !== null
            ? [shared ? 0x03 : 0x01, ...encodeLeb128U32(minPages), ...encodeLeb128U32(resolvedMax)]
            : [0x00, ...encodeLeb128U32(minPages)];
        const newPayloadLen =
          ENV_MEMORY_IMPORT_PREFIX.length + newLimits.length + restPayload.length;
        const secLenBytes = encodeLeb128U32(newPayloadLen);
        const out = new Uint8Array(
          secStart + 1 + secLenBytes.length + newPayloadLen + (wasmBytes.length - secEnd),
        );
        let w = 0;
        out.set(wasmBytes.subarray(0, secStart), w);
        w += secStart;
        out[w++] = 2;
        out.set(secLenBytes, w);
        w += secLenBytes.length;
        out.set(
          wasmBytes.subarray(payloadStart, payloadStart + ENV_MEMORY_IMPORT_PREFIX.length),
          w,
        );
        w += ENV_MEMORY_IMPORT_PREFIX.length;
        out.set(newLimits, w);
        w += newLimits.length;
        out.set(restPayload, w);
        w += restPayload.length;
        out.set(wasmBytes.subarray(secEnd), w);
        return out;
      }
    }
    pos = secEnd;
  }

  const copy = new Uint8Array(wasmBytes.length);
  copy.set(wasmBytes);
  return copy;
}

let cachedWasmBytes: Uint8Array<ArrayBuffer> | undefined;
let cachedWasmModule: WebAssembly.Module | undefined;
let cachedSharedWasmModule: WebAssembly.Module | undefined;

/**
 * Records raw `dnam.wasm` bytes so both unshared and `shared: true` `WebAssembly.Module` variants
 * can be compiled on demand.
 */
export function setCachedWasmBytes(bytes: Uint8Array): void {
  cachedWasmBytes = configureWasmMemoryImport(bytes, false);
}

function shouldRequestSharedMemory(options?: WasmMemoryOptions): boolean {
  if (options?.memory instanceof WebAssembly.Memory) {
    return isSharedWasmMemory(options.memory);
  }
  return Boolean(options?.shared) && isSharedWasmMemorySupported();
}

function cacheCompiledModule(mod: WebAssembly.Module, shared: boolean): WebAssembly.Module {
  if (shared) {
    cachedSharedWasmModule = mod;
  } else {
    cachedWasmModule = mod;
  }
  return mod;
}

/**
 * Returns the cached compiled `WebAssembly.Module` matching `shared`, if initialized.
 */
export function getCompiledWasmModule(shared = false): WebAssembly.Module | undefined {
  if (shared) {
    if (!cachedSharedWasmModule && cachedWasmBytes) {
      try {
        cachedSharedWasmModule = new WebAssembly.Module(
          configureWasmMemoryImport(cachedWasmBytes, true),
        );
      } catch {
        // Browser main threads disallow synchronous compilation > 4 KiB; use await compileWasmModule({ shared: true }).
      }
    }
    return cachedSharedWasmModule;
  }
  if (!cachedWasmModule && cachedWasmBytes) {
    try {
      cachedWasmModule = new WebAssembly.Module(configureWasmMemoryImport(cachedWasmBytes, false));
    } catch {
      // Browser main threads disallow synchronous compilation > 4 KiB; use await compileWasmModule().
    }
  }
  return cachedWasmModule;
}

/**
 * Instantiates a synchronous `WasmDriver` from a compiled `WebAssembly.Module` and a
 * `WebAssembly.Memory` passed via `{ env: { memory } }`.
 */
export function instantiateWasmDriver(
  module: WebAssembly.Module,
  options?: WasmMemoryOptions,
): WasmDriver {
  const memory = createWasmMemory(options);
  const wantShared = isSharedWasmMemory(memory);

  try {
    const instance = new WebAssembly.Instance(module, { env: { memory } });
    cacheCompiledModule(module, wantShared);
    return new WasmDriver(instance.exports as RawWasmExports, memory);
  } catch (err) {
    const explicitSharedOrCustomMem =
      options?.shared !== undefined ||
      options?.memory !== undefined ||
      (options?.maximumPages !== undefined && options.maximumPages > WASM_DEFAULT_MAXIMUM_PAGES);
    if (explicitSharedOrCustomMem && cachedWasmBytes) {
      const useUnbounded =
        options?.memory instanceof WebAssembly.Memory ||
        (options?.maximumPages !== undefined && options.maximumPages > WASM_DEFAULT_MAXIMUM_PAGES);
      try {
        const compatModule = new WebAssembly.Module(
          configureWasmMemoryImport(
            cachedWasmBytes,
            wantShared,
            useUnbounded ? null : WASM_DEFAULT_MAXIMUM_PAGES,
          ),
        );
        const instance = new WebAssembly.Instance(compatModule, { env: { memory } });
        if (!useUnbounded) {
          cacheCompiledModule(compatModule, wantShared);
        }
        return new WasmDriver(instance.exports as RawWasmExports, memory);
      } catch {
        // Fall through to alternative memory recovery below.
      }
    }
    if (options?.memory === undefined) {
      try {
        const altMemory = createWasmMemory({
          ...(options?.initialPages !== undefined ? { initialPages: options.initialPages } : {}),
          ...(options?.maximumPages !== undefined ? { maximumPages: options.maximumPages } : {}),
          shared: !wantShared,
        });
        const instance = new WebAssembly.Instance(module, { env: { memory: altMemory } });
        cacheCompiledModule(module, isSharedWasmMemory(altMemory));
        return new WasmDriver(instance.exports as RawWasmExports, altMemory);
      } catch {
        // Re-throw original instantiation error below.
      }
    }
    throw err;
  }
}

function resolveDefaultWasmUrl(): URL {
  const metaUrl = import.meta.url;
  if (metaUrl.includes("/src/internal/native/")) {
    return new URL(/* @vite-ignore */ "../../../dist/wasm/debug/dnam.wasm", metaUrl);
  }
  return new URL(/* @vite-ignore */ "./wasm/release/dnam.wasm", metaUrl);
}

/**
 * Compiles and caches the WebAssembly `SIMD128` module from a `WebAssembly.Module`,
 * `BufferSource`, or URL (`ref[sdk.ts.api.sync-init]`).
 */
export async function compileWasmModule(options?: InitWasmOptions): Promise<WebAssembly.Module> {
  const wantShared = shouldRequestSharedMemory(options);
  const useUnbounded =
    options?.memory instanceof WebAssembly.Memory ||
    (options?.maximumPages !== undefined && options.maximumPages > WASM_DEFAULT_MAXIMUM_PAGES);
  const importMaxPages = useUnbounded ? null : WASM_DEFAULT_MAXIMUM_PAGES;

  if (options?.module instanceof WebAssembly.Module) {
    return cacheCompiledModule(options.module, wantShared);
  }
  if (options?.bytes !== undefined) {
    const rawBytes = coerceBytes(options.bytes as Uint8Array | ArrayBuffer);
    const mod = await WebAssembly.compile(
      configureWasmMemoryImport(rawBytes, wantShared, importMaxPages),
    );
    setCachedWasmBytes(rawBytes);
    return useUnbounded ? mod : cacheCompiledModule(mod, wantShared);
  }
  if (options?.url === undefined) {
    if (!useUnbounded) {
      const existing = wantShared ? cachedSharedWasmModule : cachedWasmModule;
      if (existing) {
        return existing;
      }
    }
    if (cachedWasmBytes) {
      const mod = await WebAssembly.compile(
        configureWasmMemoryImport(cachedWasmBytes, wantShared, importMaxPages),
      );
      return useUnbounded ? mod : cacheCompiledModule(mod, wantShared);
    }
  }
  const targetUrl = options?.url ?? resolveDefaultWasmUrl();
  if (typeof fetch === "function") {
    const resp = await fetch(targetUrl);
    if ("ok" in resp && resp.ok === false) {
      throw new DnamError(
        "ERR_DNAM_CONTAINER",
        `Failed to fetch dnam.wasm from ${String(targetUrl)} (HTTP ${String(resp.status)})`,
      );
    }
    const rawBytes = new Uint8Array(await resp.arrayBuffer());
    const mod = await WebAssembly.compile(
      configureWasmMemoryImport(rawBytes, wantShared, importMaxPages),
    );
    setCachedWasmBytes(rawBytes);
    return useUnbounded ? mod : cacheCompiledModule(mod, wantShared);
  }
  throw new DnamError(
    "ERR_DNAM_CONTAINER",
    "Unable to fetch dnam.wasm in this environment; pass { module } or { bytes } to initWasm()",
  );
}
