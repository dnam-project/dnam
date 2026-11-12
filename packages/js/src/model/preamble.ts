/**
 * 64-byte `.dnam` preamble metadata (`PreambleInfo`) and cumulative tier arenas
 * (`TierArenas`) (`ref[sdk.ts.model.containers-and-checkpoints]`).
 *
 * @module
 */

import { tierFromIndex } from "../internal/coerce.js";
import type { NativeValidatedContainerInfo } from "../internal/native/driver.js";
import type { ProtectionProfile, Tier } from "../types.js";

/**
 * Validated 64-byte `.dnam` binary preamble metadata (`PreambleInfo`).
 */
export interface PreambleInfo {
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
  readonly tailTimeMs: number;
  readonly headerCrc32c: number;
  readonly maxTier: Tier;
  readonly isStereoOut: boolean;
  readonly hasSignatures: boolean;
  readonly isEncryptedArena: boolean;
  readonly protectionProfile: ProtectionProfile;
  readonly verificationStatus: string;
}

/**
 * Cumulative binary arena views for each available tier (`lite`, `full`, `pro`, `ultra`).
 */
export interface TierArenas {
  readonly lite: Uint8Array;
  readonly full?: Uint8Array;
  readonly pro?: Uint8Array;
  readonly ultra?: Uint8Array;
}

/**
 * Extracts a frozen `PreambleInfo` and `TierArenas` from `NativeValidatedContainerInfo`.
 */
export function extractPreambleAndArenas(info: NativeValidatedContainerInfo): {
  readonly preamble: PreambleInfo;
  readonly tierArenas: TierArenas;
} {
  const maxTier = tierFromIndex(Math.min(3, Math.max(0, info.highestTier)));
  const preamble: PreambleInfo = Object.freeze({
    magic: info.magic,
    versionMajor: info.versionMajor,
    versionMinor: info.versionMinor,
    flags: info.flags,
    numControls: info.numControls,
    sampleRateTrainHz: info.sampleRateTrainHz,
    jsonHeaderByteLen: info.jsonHeaderByteLen,
    tensorArenaOffset: info.tensorArenaOffset,
    tensorArenaByteLen: info.tensorArenaByteLen,
    arenaLiteEndCl: info.arenaLiteEndCl,
    arenaFullEndCl: info.arenaFullEndCl,
    arenaProEndCl: info.arenaProEndCl,
    arenaUltraEndCl: info.arenaUltraEndCl,
    arenaOffsetLiteEnd: info.arenaOffsetLiteEnd,
    arenaOffsetFullEnd: info.arenaOffsetFullEnd,
    arenaOffsetProEnd: info.arenaOffsetProEnd,
    arenaOffsetUltraEnd: info.arenaOffsetUltraEnd,
    arenaCrc32cLite: info.arenaCrc32cLite >>> 0,
    arenaCrc32cFull: info.arenaCrc32cFull >>> 0,
    arenaCrc32cPro: info.arenaCrc32cPro >>> 0,
    arenaCrc32cUltra: info.arenaCrc32cUltra >>> 0,
    tailTimeMs: info.tailTimeMs,
    headerCrc32c: info.headerCrc32c >>> 0,
    maxTier,
    isStereoOut: info.isStereoOut,
    hasSignatures: info.hasSignatures,
    isEncryptedArena: info.isEncryptedArena,
    protectionProfile: info.protectionProfile as ProtectionProfile,
    verificationStatus: info.verificationStatus,
  });

  const arena = info.arenaBytes;
  const liteEnd = Math.min(arena.length, info.arenaOffsetLiteEnd);
  const fullEnd = Math.min(arena.length, info.arenaOffsetFullEnd);
  const proEnd = Math.min(arena.length, info.arenaOffsetProEnd);
  const ultraEnd = Math.min(arena.length, info.arenaOffsetUltraEnd);

  const tierArenas: TierArenas = Object.freeze({
    lite: arena.subarray(0, liteEnd),
    ...(info.highestTier >= 1 ? { full: arena.subarray(0, fullEnd) } : {}),
    ...(info.highestTier >= 2 ? { pro: arena.subarray(0, proEnd) } : {}),
    ...(info.highestTier >= 3 ? { ultra: arena.subarray(0, ultraEnd) } : {}),
  });

  return { preamble, tierArenas };
}
