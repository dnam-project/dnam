/**
 * Immutable, validated `.dnam` deployment artifact (`Model<TParams>`) supporting inspection,
 * 4 KB in-place header updates, tier prefix stripping, and `Engine` creation
 * (`ref[sdk.ts.model.containers-and-checkpoints]`, `ref[sdk.ts.api.resources-and-data]`).
 *
 * @module
 */

import type { Engine, EngineOptions } from "../engine/engine.js";
import { DnamError } from "../errors.js";
import { coerceBytes, coerceTierIndex } from "../internal/coerce.js";
import { registerDisposable } from "../internal/handle.js";
import { getActiveDriver } from "../internal/native/load.js";
import type { ParameterValueMap } from "../parameters/typed-model.js";
import { decodeHeader, encodeHeader, type ModelHeader } from "../schema/header.js";
import type { ParameterDescriptor, Preset } from "../schema/parameters.js";
import type { ProtectionProfile, Tier, TierInput } from "../types.js";
import { type ModelSource, readModelSourceBytes } from "./load.js";
import { extractPreambleAndArenas, type PreambleInfo, type TierArenas } from "./preamble.js";

let engineFactory:
  | (<TParams extends ParameterValueMap>(
      model: Model<TParams>,
      options?: EngineOptions,
    ) => Engine<TParams>)
  | undefined;

/**
 * Internal registration hook invoked by `src/engine/engine.ts` to keep `src/model/` acyclic.
 */
export function registerModelEngineFactory(
  factory: <TParams extends ParameterValueMap>(
    model: Model<TParams>,
    options?: EngineOptions,
  ) => Engine<TParams>,
): void {
  engineFactory = factory;
}

/**
 * Options for `Model.from(bytes, options)` and `Model.load(source, options)`.
 */
export interface ModelLoadOptions {
  readonly deviceIdHash?: string;
  readonly deviceSecret?: Uint8Array;
  readonly trustedMarketplacePubkeyHex?: string;
}

/**
 * Options for `Model.reference(options)`.
 */
export interface ReferenceModelOptions {
  readonly supportsFreeze?: boolean;
  readonly inputs?: 1 | 2;
  readonly outputMode?: "mono" | "stereo";
}

/**
 * Options for `Model.build(header, arenaBytes, options)`.
 */
export interface BuildModelOptions {
  readonly encryptedArenaBytes?: Uint8Array;
  readonly hasSignatures?: boolean;
}

/**
 * Validated `.dnam` v2.0 open container directory layout (`validateOpenLayout()`).
 */
export interface V2OpenLayoutInfo {
  readonly protectionProfile: number;
  readonly authKind: number;
  readonly maximumTier: number;
  readonly retainedTier: number;
  readonly sampleRateTrainHz: number;
  readonly headerExtent: readonly [number, number];
  readonly sharedExtent: readonly [number, number];
  readonly authExtent: readonly [number, number];
  readonly tierStart: number;
  readonly tierEnds: readonly number[];
  readonly mutableExtent: readonly [number, number];
}

/**
 * Computes the Castagnoli `CRC32C` checksum over `data` (`ref[sdk.ts.model.containers-and-checkpoints]`).
 */
export function crc32c(data: Uint8Array | ArrayBuffer | ArrayBufferView): number {
  return getActiveDriver().crc32c(coerceBytes(data));
}

/**
 * Immutable, validated `.dnam` deployment model container (`ref[sdk.ts.api.resources-and-data]`,
 * `ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @example
 * ```ts
 * const model = Model.reference();
 * console.log(model.name, model.maxTier);
 * ```
 */
export class Model<TParams extends ParameterValueMap = Record<string, number | string>>
  implements Disposable
{
  readonly #bytes: Uint8Array;
  readonly #arenaBytes: Uint8Array;
  readonly #rawHeaderJson: string;
  readonly #preamble: PreambleInfo;
  readonly #tierArenas: TierArenas;
  #header: ModelHeader | undefined;
  #presetsMap: ReadonlyMap<string, Preset> | undefined;
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
    const { preamble, tierArenas } = extractPreambleAndArenas(info);
    this.#bytes = bytes.slice();
    this.#arenaBytes = info.arenaBytes;
    this.#rawHeaderJson = info.headerJson;
    this.#preamble = preamble;
    this.#tierArenas = tierArenas;
    this.#release = registerDisposable(this, () => {
      this.#disposed = true;
    });
  }

  /**
   * Asynchronously loads and validates a `.dnam` container from a file path, URL, `Blob`,
   * `Response`, `ReadableStream`, or byte buffer.
   */
  static async load<TParams extends ParameterValueMap = Record<string, number | string>>(
    source: ModelSource,
    options?: ModelLoadOptions,
  ): Promise<Model<TParams>> {
    const bytes = await readModelSourceBytes(source);
    return new Model<TParams>(bytes, options);
  }

  /**
   * Synchronously parses and validates a `.dnam` container from a `Uint8Array` or `ArrayBuffer`.
   */
  static from<TParams extends ParameterValueMap = Record<string, number | string>>(
    bytes: Uint8Array | ArrayBuffer | ArrayBufferView,
    options?: ModelLoadOptions,
  ): Model<TParams> {
    return new Model<TParams>(coerceBytes(bytes), options);
  }

  /**
   * Synthesizes a calibrated reference `.dnam` model container (`"gain"`, `"boost"`, `"delay_ms"`
   * parameters and `"warm_slapback"` preset).
   */
  static reference<
    TParams extends ParameterValueMap = {
      readonly gain: number;
      readonly boost: "off" | "on" | "ultra";
      readonly delay_ms: number;
    },
  >(options?: ReferenceModelOptions): Model<TParams> {
    const bytes = getActiveDriver().buildReferenceContainer(
      options?.supportsFreeze ?? true,
      options?.inputs ?? 2,
      options?.outputMode ?? "stereo",
    );
    return new Model<TParams>(bytes);
  }

  /**
   * Builds a 64B-aligned `.dnam` container from a `ModelHeader` (or wire JSON string) and `arenaBytes`.
   */
  static build<TParams extends ParameterValueMap = Record<string, number | string>>(
    header: ModelHeader | string,
    arenaBytes: Uint8Array | ArrayBuffer | ArrayBufferView,
    options?: BuildModelOptions,
  ): Model<TParams> {
    const headerJson = typeof header === "string" ? header : encodeHeader(header);
    const bytes = getActiveDriver().buildDnamContainer(
      headerJson,
      coerceBytes(arenaBytes),
      options?.encryptedArenaBytes,
      options?.hasSignatures ?? false,
    );
    return new Model<TParams>(bytes);
  }

  #ensureAlive(): void {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Model has been disposed");
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
   * Raw snake_case UTF-8 JSON header string stored in the container.
   */
  get rawHeaderJson(): string {
    this.#ensureAlive();
    return this.#rawHeaderJson;
  }

  /**
   * Human-readable model name (`header.metadata.name`).
   */
  get name(): string {
    return this.header.metadata.name;
  }

  /**
   * Maximum available tier in this container (`"lite" | "full" | "pro" | "ultra"`).
   */
  get maxTier(): Tier {
    this.#ensureAlive();
    return this.#preamble.maxTier;
  }

  /**
   * Training sample rate in Hz (`44,100..192,000`).
   */
  get sampleRateTrainHz(): number {
    this.#ensureAlive();
    return this.#preamble.sampleRateTrainHz;
  }

  /**
   * Container protection profile (`"open" | "device_allowlisted" | "enclave_sealed"`).
   */
  get protectionProfile(): ProtectionProfile {
    this.#ensureAlive();
    return this.#preamble.protectionProfile;
  }

  /**
   * Exposed parameter descriptors (`header.controlsSchema`).
   */
  get parameters(): readonly ParameterDescriptor[] {
    return this.header.controlsSchema;
  }

  /**
   * Read-only map of factory presets keyed by preset ID (`ReadonlyMap<string, Preset>`).
   */
  get presets(): ReadonlyMap<string, Preset> {
    if (!this.#presetsMap) {
      const map = new Map<string, Preset>();
      for (const preset of this.header.presets) {
        map.set(preset.id, preset);
      }
      this.#presetsMap = map;
    }
    return this.#presetsMap;
  }

  /**
   * Cumulative tier arena byte slices (`lite`, `full`, `pro`, `ultra`).
   */
  get tierArenas(): TierArenas {
    this.#ensureAlive();
    return this.#tierArenas;
  }

  /**
   * Plaintext tensor arena bytes (`Arena 0..3`).
   */
  get arenaBytes(): Uint8Array<ArrayBuffer> {
    this.#ensureAlive();
    return this.#arenaBytes.slice();
  }

  /**
   * Returns a copy of the complete serialized `.dnam` container bytes.
   */
  toBytes(): Uint8Array<ArrayBuffer> {
    this.#ensureAlive();
    return this.#bytes.slice();
  }

  /**
   * Returns a new `Model` with its JSON header updated in-place within the 4 KB page budget
   * (`ref[sdk.ts.model.containers-and-checkpoints]`).
   *
   * Accepts either a replacement `ModelHeader` or an updater callback `(currentHeader) => nextHeader`.
   */
  withHeader(
    headerOrUpdater: ModelHeader | ((current: ModelHeader) => ModelHeader),
  ): Model<TParams> {
    this.#ensureAlive();
    const nextHeader =
      typeof headerOrUpdater === "function" ? headerOrUpdater(this.header) : headerOrUpdater;
    const nextJson = encodeHeader(nextHeader);
    const updatedBytes = getActiveDriver().updateHeaderInPlace(this.#bytes, nextJson);
    return new Model<TParams>(updatedBytes);
  }

  /**
   * Returns a new `Model` stripped down to `maxTier` (`"lite" | "full" | "pro" | "ultra"`)
   * without invalidating prefix-committed signatures (`ref[container.dnam.tier-prefix]`).
   */
  withMaxTier(maxTier: TierInput | string): Model<TParams> {
    this.#ensureAlive();
    const tierIdx = coerceTierIndex(maxTier);
    if (
      this.#bytes.length >= 8 &&
      this.#preamble.magic === "DNAM" &&
      this.#preamble.versionMajor === 2
    ) {
      const strippedV2 = getActiveDriver().stripV2OpenDnamContainer(this.#bytes, tierIdx);
      return new Model<TParams>(strippedV2);
    }
    const stripped = getActiveDriver().stripContainerToTier(this.#bytes, tierIdx);
    return new Model<TParams>(stripped);
  }

  /**
   * Validates the v2.0 Open `.dnam` container directory layout when `versionMajor === 2`.
   */
  validateOpenLayout(): V2OpenLayoutInfo {
    this.#ensureAlive();
    const jsonStr = getActiveDriver().validateV2OpenDnamLayout(this.#bytes);
    return Object.freeze(JSON.parse(jsonStr) as V2OpenLayoutInfo);
  }

  /**
   * Creates a stateful real-time `Engine<TParams>` bound to this model.
   */
  createEngine(options?: EngineOptions): Engine<TParams> {
    this.#ensureAlive();
    if (!engineFactory) {
      throw new DnamError(
        "ERR_DNAM_CONTAINER",
        "Engine factory is not initialized; import Engine or '@dnam-project/dnam'",
      );
    }
    return engineFactory(this, options);
  }

  /**
   * Releases the model's resources idempotently.
   */
  dispose(): void {
    this.#release();
  }

  /**
   * Disposes the model (`using` protocol).
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}
