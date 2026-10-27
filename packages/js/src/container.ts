/**
 * `.dnam` and `.dnam-ckpt` Hybrid Binary Container (`DnamContainer`), `Arena 3` `uplo='U'`
 * Gram matrix packing (`GramMatrix`), Castagnoli `CRC32C` (`crc32c`), and ZOH rescaling.
 */

import { type BackendChoice, resolveDriver } from "./drivers/index.js";
import { DnamValidationError } from "./errors.js";
import {
  type SealDnamContainerOptions,
  sealDnamContainer,
  type UnsealDnamContainerOptions,
} from "./provenance.js";
import { validateStandardSchema } from "./schema/base.js";
import {
  type DnamJsonHeader,
  DnamJsonHeaderSchema,
  dnamJsonHeaderToWire,
} from "./schema/header.js";
import type { PreambleSnapshot, ZohParamsSnapshot } from "./schema/runtime.js";
import type { DnamProtectionProfile } from "./schema/signatures.js";

export function toUint8Array(input: Uint8Array | ArrayBuffer | ArrayBufferView): Uint8Array {
  if (input instanceof Uint8Array) {
    return input;
  }
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
}

export interface FromBytesOptions extends UnsealDnamContainerOptions {
  backend?: BackendChoice | undefined;
}

export type SealContainerOptions = Omit<SealDnamContainerOptions, "innerDnamBytes">;

export interface BuildContainerOptions {
  header: DnamJsonHeader | Record<string, unknown> | string;
  arenaBytes: Uint8Array | ArrayBuffer;
  plaintextArenaCrcSource?: Uint8Array | ArrayBuffer | undefined;
  encryptedArena?: boolean | undefined;
  backend?: BackendChoice | undefined;
}

export interface BuildReferenceOptions {
  core2Active?: boolean | undefined;
  isCheckpoint?: boolean | undefined;
  backend?: BackendChoice | undefined;
}

export interface WatermarkOptions {
  kappa48?: Float32Array | readonly number[] | undefined;
  qBuyer4x4?: Float32Array | readonly number[] | undefined;
  tardosCode4608?: Float32Array | readonly number[] | undefined;
  tardosEpsilon?: number | undefined;
  backend?: BackendChoice | undefined;
}

export interface ZohComputeOptions {
  sampleRateTrainHz?: number | undefined;
  sampleRateHostHz: number;
  controlDecimationTrain?: number | undefined;
  rAcTrain?: number | undefined;
  rDcTrain?: number | undefined;
  tauRefTrain?: number | undefined;
  switchXfadeTrain?: number | undefined;
  backend?: BackendChoice | undefined;
}

function serializeHeaderInput(header: DnamJsonHeader | Record<string, unknown> | string): string {
  if (typeof header === "string") {
    const parsed: unknown = JSON.parse(header);
    const validated = validateStandardSchema(DnamJsonHeaderSchema, parsed, "DnamJsonHeader");
    return JSON.stringify(dnamJsonHeaderToWire(validated));
  }
  const validated = validateStandardSchema(DnamJsonHeaderSchema, header, "DnamJsonHeader");
  return JSON.stringify(dnamJsonHeaderToWire(validated));
}

/**
 * Validated `.dnam` or `.dnam-ckpt` container holding the 64B preamble, typed `DnamJsonHeader`,
 * `Ed25519` verification status, 64B-aligned `arenaBytes`, and raw container `bytes`.
 */
export class DnamContainer {
  public readonly bytes: Uint8Array;
  public readonly preamble: PreambleSnapshot;
  public readonly header: DnamJsonHeader;
  public readonly headerJson: string;
  public readonly verificationStatus: "Unsigned" | "CreatorVerified" | "MarketplaceVerified";
  public readonly protectionProfile: DnamProtectionProfile;
  public readonly arenaBytes: Uint8Array;
  private readonly defaultBackend?: BackendChoice | undefined;

  private constructor(
    bytes: Uint8Array,
    preamble: PreambleSnapshot,
    header: DnamJsonHeader,
    headerJson: string,
    verificationStatus: "Unsigned" | "CreatorVerified" | "MarketplaceVerified",
    protectionProfile: DnamProtectionProfile,
    arenaBytes: Uint8Array,
    defaultBackend?: BackendChoice,
  ) {
    this.bytes = bytes;
    this.preamble = preamble;
    this.header = header;
    this.headerJson = headerJson;
    this.verificationStatus = verificationStatus;
    this.protectionProfile = protectionProfile;
    this.arenaBytes = arenaBytes;
    this.defaultBackend = defaultBackend;
  }

  /**
   * Alias for `arenaBytes` (read-only view of the 64B-aligned binary tensor arena).
   */
  public get arena(): Uint8Array {
    return this.arenaBytes;
  }

  /**
   * Parses and validates a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) container byte buffer,
   * verifying 64-byte alignment, Castagnoli `CRC32C` checksums, JSON schema, `Ed25519` signatures,
   * and `metadata.distribution_policy`.
   */
  public static fromBytes(
    bytes: Uint8Array | ArrayBuffer | ArrayBufferView,
    backendOrOptions?: BackendChoice | FromBytesOptions,
  ): DnamContainer {
    const u8 = toUint8Array(bytes);
    const opts: FromBytesOptions =
      typeof backendOrOptions === "string" || backendOrOptions === undefined
        ? { backend: backendOrOptions }
        : backendOrOptions;
    const driver = resolveDriver(opts.backend);
    const raw = driver.validateContainer(
      u8,
      opts.deviceIdHash,
      opts.deviceSecretX25519 !== undefined ? toUint8Array(opts.deviceSecretX25519) : undefined,
      opts.trustedMarketplacePubkeyHex,
    );
    const parsedHeader: unknown = JSON.parse(raw.headerJson);
    const header = validateStandardSchema(DnamJsonHeaderSchema, parsedHeader, "DnamJsonHeader");
    return new DnamContainer(
      u8,
      raw.preamble,
      header,
      raw.headerJson,
      raw.verificationStatus,
      raw.protectionProfile,
      raw.arenaBytes,
      opts.backend,
    );
  }

  /**
   * Seals this `.dnam` container into a `.dnam-seal` (`b"DNSL"`) 64-byte aligned outer binary envelope
   * with nested double encryption (`cek_outer` + `cek_inner`).
   */
  public seal(options: SealContainerOptions): Uint8Array {
    return sealDnamContainer({
      ...options,
      innerDnamBytes: this.bytes,
      backend: options.backend ?? this.defaultBackend,
    });
  }

  /**
   * Unseals a `.dnam-seal` (`b"DNSL"`) outer binary envelope and returns the validated `DnamContainer`.
   */
  public static unseal(
    sealedBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
    options: FromBytesOptions = {},
  ): DnamContainer {
    return DnamContainer.fromBytes(sealedBytes, options);
  }

  /**
   * Builds a 4 KB page-aligned `.dnam` or `.dnam-ckpt` container from `header` and `arenaBytes`.
   */
  public static build(options: BuildContainerOptions): DnamContainer {
    const builtBytes = buildDnamContainer(options);
    return DnamContainer.fromBytes(builtBytes, options.backend);
  }

  /**
   * Synthesizes a canonical reference `.dnam` (or `.dnam-ckpt`) container with built-in `"gain"` (`range`)
   * and `"boost"` (`switch`) controls, a `"warm_slapback"` preset, and calibrated pass-through weights.
   */
  public static buildReference(options: BuildReferenceOptions = {}): DnamContainer {
    const driver = resolveDriver(options.backend);
    const bytes = driver.buildReferenceContainer(
      options.core2Active ?? false,
      options.isCheckpoint ?? false,
    );
    return DnamContainer.fromBytes(bytes, options.backend);
  }

  /**
   * Convenience alias for `DnamContainer.buildReference`.
   */
  public static createReference(
    core2ActiveOrOptions: boolean | BuildReferenceOptions = false,
    backend?: BackendChoice,
  ): DnamContainer {
    if (typeof core2ActiveOrOptions === "boolean") {
      return DnamContainer.buildReference({ core2Active: core2ActiveOrOptions, backend });
    }
    return DnamContainer.buildReference(core2ActiveOrOptions);
  }

  /**
   * Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` hex digest for this container.
   */
  public immutableHeaderSha256(backend?: BackendChoice): string {
    const driver = resolveDriver(backend ?? this.defaultBackend);
    const wireJson = JSON.stringify(dnamJsonHeaderToWire(this.header));
    return driver.immutableHeaderSha256(wireJson);
  }

  /**
   * Extracts the 160-Dimensional Gauge-Invariant Structural Hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{160}$
   * from this container's `arenaBytes`.
   */
  public structuralHash(backend?: BackendChoice): Float64Array {
    const driver = resolveDriver(backend ?? this.defaultBackend);
    return driver.extractStructuralHash(this.arenaBytes);
  }

  /**
   * Returns a new `DnamContainer` with Dual-Mechanism Buyer Watermarking (`Test 8H`:
   * Mechanism A isometric gauge transform + optional Mechanism B Tardos spread-spectrum code).
   */
  public withWatermark(options: WatermarkOptions): DnamContainer {
    const backend = options.backend ?? this.defaultBackend;
    const driver = resolveDriver(backend);
    const kappa =
      options.kappa48 instanceof Float32Array
        ? options.kappa48
        : new Float32Array(options.kappa48 ?? new Array<number>(48).fill(1.0));
    const qBuyer =
      options.qBuyer4x4 instanceof Float32Array
        ? options.qBuyer4x4
        : new Float32Array(options.qBuyer4x4 ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    let watermarkedArena = driver.applyIsometricGaugeWatermark(this.arenaBytes, kappa, qBuyer);
    if (options.tardosCode4608 !== undefined) {
      const code =
        options.tardosCode4608 instanceof Float32Array
          ? options.tardosCode4608
          : new Float32Array(options.tardosCode4608);
      watermarkedArena = driver.applyTardosWatermark(
        watermarkedArena,
        code,
        options.tardosEpsilon ?? 5e-4,
      );
    }
    return DnamContainer.build({
      header: this.header,
      arenaBytes: watermarkedArena,
      backend,
    });
  }

  /**
   * Updates the JSON header in-place inside the 4 KB page-aligned region (`0x40..tensor_arena_offset`)
   * without touching or re-hashing the binary tensor arenas.
   */
  public updateHeaderInPlace(
    updatedHeader: DnamJsonHeader | Record<string, unknown> | string,
    backend?: BackendChoice,
  ): DnamContainer {
    const driver = resolveDriver(backend);
    const headerJson = serializeHeaderInput(updatedHeader);
    const updatedBytes = driver.updateHeaderInPlace(this.bytes, headerJson);
    return DnamContainer.fromBytes(updatedBytes, backend);
  }

  /**
   * Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters for `sampleRateHostHz`.
   */
  public computeZoh(sampleRateHostHz: number, backend?: BackendChoice): ZohParamsSnapshot {
    return computeZohRescaledParams({
      sampleRateTrainHz: this.preamble.sampleRateTrainHz,
      sampleRateHostHz,
      controlDecimationTrain: this.preamble.controlDecimation,
      rAcTrain: this.header.core1HybridSsm.acCouplingPole,
      rDcTrain: this.header.core2Mufdn.dcBlockerPole,
      tauRefTrain: this.header.core2Mufdn.tzfReferenceTapSamples,
      switchXfadeTrain: this.header.core2Mufdn.switchXfadeSamples,
      backend,
    });
  }
}

/**
 * Validates a `.dnam`, `.dnam-ckpt`, or `.dnam-seal` (`b"DNSL"`) byte buffer and returns a `DnamContainer`.
 */
export function validateDnamModel(
  bytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  backendOrOptions?: BackendChoice | FromBytesOptions,
): DnamContainer {
  return DnamContainer.fromBytes(bytes, backendOrOptions);
}

/**
 * Builds a 4 KB page-aligned `.dnam` or `.dnam-ckpt` container byte buffer.
 */
export function buildDnamContainer(options: BuildContainerOptions): Uint8Array {
  const driver = resolveDriver(options.backend);
  const headerJson = serializeHeaderInput(options.header);
  const arenaU8 = toUint8Array(options.arenaBytes);
  const crcSrc =
    options.plaintextArenaCrcSource !== undefined
      ? toUint8Array(options.plaintextArenaCrcSource)
      : undefined;
  return driver.buildContainer(headerJson, arenaU8, crcSrc, options.encryptedArena ?? false);
}

/**
 * Builds a canonical reference `.dnam` or `.dnam-ckpt` container byte buffer.
 */
export function buildReferenceContainer(
  core2Active = false,
  isCheckpoint = false,
  backend?: BackendChoice,
): Uint8Array {
  return DnamContainer.buildReference({ core2Active, isCheckpoint, backend }).bytes;
}

/**
 * Updates the JSON header of a `.dnam` or `.dnam-ckpt` container in-place within the 4 KB page boundary.
 */
export function updateHeaderInPlace(
  containerBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
  updatedHeader: DnamJsonHeader | Record<string, unknown> | string,
  backend?: BackendChoice,
): Uint8Array {
  const driver = resolveDriver(backend);
  const headerJson = serializeHeaderInput(updatedHeader);
  return driver.updateHeaderInPlace(toUint8Array(containerBytes), headerJson);
}

/**
 * Computes Continuous-Time Zero-Order Hold (ZOH) rescaled parameters for `sampleRateHostHz`.
 */
export function computeZohRescaledParams(options: ZohComputeOptions): ZohParamsSnapshot {
  const driver = resolveDriver(options.backend);
  return driver.computeZoh(
    options.sampleRateTrainHz ?? 48_000,
    options.sampleRateHostHz,
    options.controlDecimationTrain ?? 16,
    options.rAcTrain ?? 0.9993,
    options.rDcTrain ?? 0.995,
    options.tauRefTrain ?? 64.0,
    options.switchXfadeTrain ?? 128,
  );
}

/**
 * Computes the hardware-accelerated Castagnoli `CRC32C` (`0x1EDC6F41`) checksum of `data`.
 */
export function crc32c(
  data: Uint8Array | ArrayBuffer | ArrayBufferView,
  backend?: BackendChoice,
): number {
  const driver = resolveDriver(backend);
  return driver.crc32c(toUint8Array(data));
}

/**
 * `.dnam-ckpt` `Arena 3` `LAPACK uplo='U'` column-major upper-triangular Gram matrix utilities.
 */
export const GramMatrix = {
  /**
   * Returns the packed upper-triangular length $N(N+1)/2$ for dimension $N$.
   */
  packedLen(nDim: number): number {
    if (!Number.isInteger(nDim) || nDim < 1) {
      throw new DnamValidationError(`nDim must be a positive integer, got ${String(nDim)}`);
    }
    return Math.floor((nDim * (nDim + 1)) / 2);
  },

  /**
   * Packs a flattened row-major $N \times N$ symmetric `Float64Array` into `LAPACK uplo='U'`
   * column-major upper-triangular format.
   */
  packUploU(denseFlat: Float64Array, nDim: number, backend?: BackendChoice): Float64Array {
    const driver = resolveDriver(backend);
    return driver.packUploU(denseFlat, nDim);
  },

  /**
   * Unpacks a `LAPACK uplo='U'` column-major upper-triangular `Float64Array` into a flattened
   * $N_{\text{tier}} \times N_{\text{tier}}$ symmetric matrix, supporting zero-copy leading principal
   * minor extraction ($N_{\text{tier}} \le N_{\max}$).
   */
  unpackUploU(packedU: Float64Array, nTier: number, backend?: BackendChoice): Float64Array {
    const driver = resolveDriver(backend);
    return driver.unpackUploU(packedU, nTier);
  },
} as const;
