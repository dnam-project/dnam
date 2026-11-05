/**
 * Audio I/O topology, capabilities, auto-unity leveling, and slimmable tier interfaces
 * (`ref[sdk.ts.schema.camel-case-codec-parity]`).
 *
 * Zero runtime JS dependencies; validated by native Rust `serde` (`dnam_engine::schema`).
 *
 * @module
 */

/**
 * Host dry/wet blend capability modes (`"none" | "host_parallel" | "modeled_control"`).
 */
export const blendModes = ["none", "host_parallel", "modeled_control"] as const;

/**
 * Host dry/wet blend capability literal union.
 */
export type BlendMode = (typeof blendModes)[number];

/**
 * Loudness measurement metrics (`"rms_dbfs" | "k_weighted_lufs"`).
 */
export const levelingMetrics = ["rms_dbfs", "k_weighted_lufs"] as const;

/**
 * Loudness measurement metric literal union.
 */
export type LevelingMetric = (typeof levelingMetrics)[number];

/**
 * Model-level physical capabilities (`ioConfig.capabilities`).
 */
export interface ModelCapabilities {
  readonly blendMode: BlendMode;
  readonly defaultBlend: number;
  readonly wetPolarityInverted: boolean;
  readonly supportsTrails: boolean;
  readonly tempoSyncCapable: boolean;
  readonly supportsFreeze: boolean;
  readonly freezeMutesDry: boolean;
}

/**
 * Auto-unity loudness leveling configuration (`ioConfig.leveling`).
 */
export interface LevelingConfig {
  readonly autoUnityEnabled: boolean;
  readonly levelingMetric: LevelingMetric;
  readonly referenceInputLevelDb: number;
  readonly measuredDefaultOutputLevelDb: number;
  readonly unityGainOffsetDb: number;
  readonly targetGainDb: number;
  readonly calibratedTrimDb: number;
}

/**
 * Audio I/O topology and host capabilities configuration (`header.ioConfig`).
 */
export interface IoConfig {
  readonly inputs: number;
  readonly outputMode: "mono" | "stereo";
  readonly latencySamples: number;
  readonly tailTimeMs: number;
  readonly capabilities: ModelCapabilities;
  readonly leveling: LevelingConfig;
}

/**
 * Metadata and prefix checksum for a slimmable tier (`lite`, `full`, `pro`, `ultra`).
 */
export interface TierProfile {
  readonly featureBusDim: number;
  readonly channels: number;
  readonly earlyFirTaps: number;
  readonly kautzModes: number;
  readonly tankLines: number;
  readonly headSvdRank: number;
  readonly loudnessLufs: number;
  readonly outputTrimDb: number;
  readonly validationEsrDb: number;
  readonly arenaEndByteOffset: number;
  readonly prefixCrc32c: number;
}

/**
 * Slimmable tier profiles (`lite`, `full`, `pro`, `ultra`).
 */
export interface SlimmableTiersConfig {
  readonly lite: TierProfile;
  readonly full: TierProfile;
  readonly pro: TierProfile;
  readonly ultra: TierProfile;
}
