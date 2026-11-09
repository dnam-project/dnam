/**
 * `.dnam-ckpt` (`DNCK`) stem registry, extracted physics, and 3-phase clamp audit report
 * interfaces (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Zero runtime JS dependencies; validated by native Rust `serde` (`dnam_engine::schema`).
 *
 * @module
 */

import type { TensorSliceEntry } from "./stages.js";

/**
 * Platform-agnostic IEEE-754 PCM SHA-256 provenance for a training stem.
 */
export interface StemProvenance {
  readonly inputPcmSha256: string;
  readonly targetPcmSha256: string;
  readonly numFrames: number;
}

/**
 * Extracted Stage-1 physical system-identification anchors for a stem.
 */
export interface ExtractedPhysics {
  readonly delaysMs: readonly number[];
  readonly lfoFreqsHz: readonly number[];
  readonly decayGainsLin: readonly number[];
  readonly detectedGrainSizeMs: number;
  readonly detectedPitchShiftSemitones: number;
}

/**
 * `.dnam-ckpt` per-stem registry record (`header.stemsRegistry[*]`).
 */
export interface StemRegistryEntry {
  readonly stemId: string;
  readonly enabled: boolean;
  readonly controlsMeta: Readonly<Record<string, unknown>>;
  readonly provenance: StemProvenance;
  readonly stemArena4Crc32c: number;
  readonly userMetadata: Readonly<Record<string, unknown>>;
  readonly extractedPhysics: ExtractedPhysics | null;
  readonly gAudioUploUF64: TensorSliceEntry;
  readonly cAudioTierF64: TensorSliceEntry;
}

/**
 * `.dnam-ckpt` 3-phase control clamp audit report and actionable stem prescriptions.
 */
export interface AuditReport {
  readonly phase1Clamps: readonly string[];
  readonly phase2RailWarnings: readonly string[];
  readonly phase3Clamps: readonly string[];
  readonly prescriptions: readonly string[];
}
