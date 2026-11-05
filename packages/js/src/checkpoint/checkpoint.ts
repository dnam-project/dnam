/**
 * Immutable, validated `.dnam-ckpt` (`DNCK`) training checkpoint container (`Checkpoint`)
 * supporting `Arena 4` stem `CRC32C` verification, `Arena 5` (`DNCA`) calibration inspection,
 * and runtime `Model` export (`ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @module
 */

import { DnamError } from "../errors.js";
import { coerceBytes } from "../internal/coerce.js";
import { registerDisposable } from "../internal/handle.js";
import { getActiveDriver } from "../internal/native/load.js";
import { type ModelSource, readModelSourceBytes } from "../model/load.js";
import { Model, type ModelLoadOptions } from "../model/model.js";
import { extractPreambleAndArenas, type PreambleInfo } from "../model/preamble.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import type { AuditReport, StemRegistryEntry } from "../schema/checkpoint.js";
import { decodeHeader, type ModelHeader } from "../schema/header.js";

/**
 * Options for `Checkpoint.reference(options)`.
 */
export interface ReferenceCheckpointOptions {
  readonly includeCalibStats?: boolean;
}

/**
 * Validated v2.0 `.dnam-ckpt` (`DNCK`) bundle directory metadata.
 */
export interface DnckLayoutInfo {
  readonly retainedTier: number;
  readonly bundleEnds: readonly number[];
  readonly mutableBoundary: number;
}

/**
 * Immutable, validated `.dnam-ckpt` (`DNCK`) training checkpoint artifact
 * (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[container.ckpt.bundle-framing]`).
 *
 * @example
 * ```ts
 * const ckpt = Checkpoint.reference();
 * const model = ckpt.toModel().withMaxTier("lite");
 * ```
 */
export class Checkpoint implements Disposable {
  readonly #bytes: Uint8Array;
  readonly #arenaBytes: Uint8Array;
  readonly #rawHeaderJson: string;
  readonly #preamble: PreambleInfo;
  #header: ModelHeader | undefined;
  readonly #release: () => void;
  #disposed = false;

  private constructor(bytes: Uint8Array, options?: ModelLoadOptions) {
    const driver = getActiveDriver();
    const info = driver.validateContainer(
      bytes,
      options?.deviceIdHash,
      options?.deviceSecret,
      options?.trustedMarketplacePubkeyHex,
    );
    if (info.magic !== "DNCK") {
      throw new DnamError(
        "ERR_DNAM_CONTAINER",
        `Expected .dnam-ckpt (DNCK) magic, got '${info.magic}'`,
      );
    }
    const { preamble } = extractPreambleAndArenas(info);
    this.#bytes = bytes.slice();
    this.#arenaBytes = info.arenaBytes;
    this.#rawHeaderJson = info.headerJson;
    this.#preamble = preamble;
    this.#release = registerDisposable(this, () => {
      this.#disposed = true;
    });
  }

  /**
   * Asynchronously loads and validates a `.dnam-ckpt` (`DNCK`) checkpoint from a file path, URL,
   * `Blob`, `Response`, `ReadableStream`, or byte buffer.
   */
  static async load(source: ModelSource, options?: ModelLoadOptions): Promise<Checkpoint> {
    const bytes = await readModelSourceBytes(source);
    return new Checkpoint(bytes, options);
  }

  /**
   * Synchronously parses and validates a `.dnam-ckpt` (`DNCK`) checkpoint from bytes.
   */
  static from(
    bytes: Uint8Array | ArrayBuffer | ArrayBufferView,
    options?: ModelLoadOptions,
  ): Checkpoint {
    return new Checkpoint(coerceBytes(bytes), options);
  }

  /**
   * Synthesizes a reference `.dnam-ckpt` (`DNCK`) checkpoint container with an `Arena 4` stem
   * record and optional `Arena 5` (`DNCA`) calibration statistics record.
   */
  static reference(options?: ReferenceCheckpointOptions): Checkpoint {
    const bytes = getActiveDriver().buildReferenceCheckpoint(options?.includeCalibStats ?? true);
    return new Checkpoint(bytes);
  }

  /**
   * Validates a v2.0 `.dnam-ckpt` (`DNCK`) bundle layout directly from raw bytes.
   */
  static validateV2Bundle(ckptBytes: Uint8Array | ArrayBuffer | ArrayBufferView): DnckLayoutInfo {
    return Object.freeze(getActiveDriver().validateV2DnckCheckpoint(coerceBytes(ckptBytes)));
  }

  #ensureAlive(): void {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Checkpoint has been disposed");
    }
  }

  /**
   * Validated 64-byte binary preamble metadata.
   */
  get preamble(): PreambleInfo {
    this.#ensureAlive();
    return this.#preamble;
  }

  /**
   * Lazily decoded, deeply frozen camelCase `ModelHeader`.
   */
  get header(): ModelHeader {
    this.#ensureAlive();
    if (!this.#header) {
      this.#header = decodeHeader(this.#rawHeaderJson);
    }
    return this.#header;
  }

  /**
   * Registered training stems (`header.stemsRegistry ?? []`).
   */
  get stems(): readonly StemRegistryEntry[] {
    return this.header.stemsRegistry ?? [];
  }

  /**
   * 3-Phase control-space clamp audit report (`header.auditReport`).
   */
  get auditReport(): AuditReport | null {
    return this.header.auditReport;
  }

  /**
   * Verifies `stemArena4Crc32c` for every stem in `this.stems` against `Arena 4`.
   */
  verifyStems(): void {
    this.#ensureAlive();
    const driver = getActiveDriver();
    for (const stem of this.stems) {
      const start = Math.min(stem.gAudioUploUF64.byteOffset, stem.cAudioTierF64.byteOffset);
      const end = Math.max(
        stem.gAudioUploUF64.byteOffset + stem.gAudioUploUF64.byteLength,
        stem.cAudioTierF64.byteOffset + stem.cAudioTierF64.byteLength,
      );
      driver.verifyStemArena4Crc32c(this.#arenaBytes, start, end - start, stem.stemArena4Crc32c);
    }
  }

  /**
   * Verifies the `Arena 5` (`DNCA` calibration statistics) record at `arena5Offset`,
   * returning its verified byte length.
   */
  verifyCalibrationStats(arena5Offset = 68_096): number {
    this.#ensureAlive();
    return getActiveDriver().verifyArena5CalibStats(this.#arenaBytes, arena5Offset);
  }

  /**
   * Exports a deployable runtime `.dnam` `Model` stripped of training-only `Arena 4..5`
   * stem/calibration records (`ref[sdk.ts.model.containers-and-checkpoints]`).
   */
  toModel<TParams extends ParameterValueMap = Record<string, number | string>>(): Model<TParams> {
    this.#ensureAlive();
    const runtimeBytes = getActiveDriver().exportRuntimeDnamFromCkpt(this.#bytes);
    return Model.from<TParams>(runtimeBytes);
  }

  /**
   * Returns a copy of the serialized `.dnam-ckpt` container bytes.
   */
  toBytes(): Uint8Array<ArrayBuffer> {
    this.#ensureAlive();
    return this.#bytes.slice();
  }

  /**
   * Releases the checkpoint's resources idempotently.
   */
  dispose(): void {
    this.#release();
  }

  /**
   * Disposes the checkpoint (`using` protocol).
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}
