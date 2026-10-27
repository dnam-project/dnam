/**
 * Canonical `.dnam` and `.dnam-ckpt` UTF-8 JSON Header (`DnamJsonHeader`), subsystem configurations,
 * tensor directory (`TensorSliceEntry`), stem registry (`StemRegistryEntry`), and audit report (`AuditReport`).
 */

import { z } from "zod";
import { type JsonValue, JsonValueSchema } from "./base.js";
import {
  type ControlSchemaDef,
  ControlSchemaDefSchema,
  controlSchemaDefToWire,
  type PresetDef,
  PresetDefSchema,
  presetDefToWire,
} from "./controls.js";
import {
  type DatasetProvenance,
  DatasetProvenanceSchema,
  type DistributionPolicy,
  DistributionPolicySchema,
  datasetProvenanceToWire,
  distributionPolicyToWire,
  type ModelSignatures,
  ModelSignaturesSchema,
  modelSignaturesToWire,
} from "./signatures.js";

export interface ModelMetadata {
  name: string;
  author?: string | undefined;
  description?: string | undefined;
  license?: string | undefined;
  createdAt: string;
  inputReferenceDbfs: number;
  provenance: DatasetProvenance;
  distributionPolicy?: DistributionPolicy | undefined;
  signatures?: ModelSignatures | undefined;
}

const RawModelMetadataSchema = z
  .object({
    name: z.string().min(1),
    author: z.string().nullable().optional(),
    description: z.string().nullable().optional(),
    license: z.string().nullable().optional(),
    created_at: z.string().optional(),
    createdAt: z.string().optional(),
    input_reference_dbfs: z.number().optional(),
    inputReferenceDbfs: z.number().optional(),
    provenance: DatasetProvenanceSchema.optional(),
    distribution_policy: DistributionPolicySchema.nullable().optional(),
    distributionPolicy: DistributionPolicySchema.nullable().optional(),
    signatures: ModelSignaturesSchema.nullable().optional(),
  })
  .strict();

export const ModelMetadataSchema: z.ZodType<ModelMetadata, unknown> =
  RawModelMetadataSchema.transform((raw): ModelMetadata => {
    return {
      name: raw.name,
      author: raw.author ?? undefined,
      description: raw.description ?? undefined,
      license: raw.license ?? undefined,
      createdAt: raw.createdAt ?? raw.created_at ?? "2026-10-23T00:00:00Z",
      inputReferenceDbfs: raw.inputReferenceDbfs ?? raw.input_reference_dbfs ?? -20.5,
      provenance: raw.provenance ?? {
        datasetManifestSha256: "",
        epochsTrained: 0,
      },
      distributionPolicy: raw.distributionPolicy ?? raw.distribution_policy ?? undefined,
      signatures: raw.signatures ?? undefined,
    };
  });

export type BlendMode = "none" | "host_parallel" | "modeled_control";

export interface ModelCapabilities {
  blendMode: BlendMode;
  defaultBlend: number;
  wetPolarityInverted: boolean;
  supportsTrails: boolean;
  tempoSyncCapable: boolean;
  supportsFreeze: boolean;
  freezeMutesDry: boolean;
}

const RawModelCapabilitiesSchema = z
  .object({
    blend_mode: z.enum(["none", "host_parallel", "modeled_control"]).optional(),
    blendMode: z.enum(["none", "host_parallel", "modeled_control"]).optional(),
    default_blend: z.number().min(0).max(1).optional(),
    defaultBlend: z.number().min(0).max(1).optional(),
    wet_polarity_inverted: z.boolean().optional(),
    wetPolarityInverted: z.boolean().optional(),
    supports_trails: z.boolean().optional(),
    supportsTrails: z.boolean().optional(),
    tempo_sync_capable: z.boolean().optional(),
    tempoSyncCapable: z.boolean().optional(),
    supports_freeze: z.boolean().optional(),
    supportsFreeze: z.boolean().optional(),
    freeze_mutes_dry: z.boolean().optional(),
    freezeMutesDry: z.boolean().optional(),
  })
  .strict();

export const ModelCapabilitiesSchema: z.ZodType<ModelCapabilities, unknown> =
  RawModelCapabilitiesSchema.transform(
    (raw): ModelCapabilities => ({
      blendMode: raw.blendMode ?? raw.blend_mode ?? "none",
      defaultBlend: raw.defaultBlend ?? raw.default_blend ?? 1.0,
      wetPolarityInverted: raw.wetPolarityInverted ?? raw.wet_polarity_inverted ?? false,
      supportsTrails: raw.supportsTrails ?? raw.supports_trails ?? false,
      tempoSyncCapable: raw.tempoSyncCapable ?? raw.tempo_sync_capable ?? false,
      supportsFreeze: raw.supportsFreeze ?? raw.supports_freeze ?? false,
      freezeMutesDry: raw.freezeMutesDry ?? raw.freeze_mutes_dry ?? false,
    }),
  );

export type LevelingMetric = "rms_dbfs" | "k_weighted_lufs";

export interface LevelingConfig {
  autoUnityEnabled: boolean;
  levelingMetric: LevelingMetric;
  referenceInputLevelDb: number;
  measuredDefaultOutputLevelDb: number;
  unityGainOffsetDb: number;
  targetGainDb: number;
  calibratedTrimDb: number;
}

const RawLevelingConfigSchema = z
  .object({
    auto_unity_enabled: z.boolean().optional(),
    autoUnityEnabled: z.boolean().optional(),
    leveling_metric: z.enum(["rms_dbfs", "k_weighted_lufs"]).optional(),
    levelingMetric: z.enum(["rms_dbfs", "k_weighted_lufs"]).optional(),
    reference_input_level_db: z.number().optional(),
    referenceInputLevelDb: z.number().optional(),
    measured_default_output_level_db: z.number().optional(),
    measuredDefaultOutputLevelDb: z.number().optional(),
    unity_gain_offset_db: z.number().optional(),
    unityGainOffsetDb: z.number().optional(),
    target_gain_db: z.number().optional(),
    targetGainDb: z.number().optional(),
    calibrated_trim_db: z.number().optional(),
    calibratedTrimDb: z.number().optional(),
  })
  .strict();

export const LevelingConfigSchema: z.ZodType<LevelingConfig, unknown> =
  RawLevelingConfigSchema.transform(
    (raw): LevelingConfig => ({
      autoUnityEnabled: raw.autoUnityEnabled ?? raw.auto_unity_enabled ?? true,
      levelingMetric: raw.levelingMetric ?? raw.leveling_metric ?? "rms_dbfs",
      referenceInputLevelDb: raw.referenceInputLevelDb ?? raw.reference_input_level_db ?? -20.5,
      measuredDefaultOutputLevelDb:
        raw.measuredDefaultOutputLevelDb ?? raw.measured_default_output_level_db ?? -20.5,
      unityGainOffsetDb: raw.unityGainOffsetDb ?? raw.unity_gain_offset_db ?? 0.0,
      targetGainDb: raw.targetGainDb ?? raw.target_gain_db ?? 0.0,
      calibratedTrimDb: raw.calibratedTrimDb ?? raw.calibrated_trim_db ?? 0.0,
    }),
  );

export interface IoConfig {
  inputs: number;
  outputMode: "stereo" | "mono";
  latencySamples: number;
  tailTimeMs: number;
  capabilities?: ModelCapabilities | undefined;
  leveling?: LevelingConfig | undefined;
}

const RawIoConfigSchema = z
  .object({
    inputs: z.number().int().min(1).max(2).optional(),
    output_mode: z.enum(["stereo", "mono"]).optional(),
    outputMode: z.enum(["stereo", "mono"]).optional(),
    latency_samples: z.number().int().nonnegative().optional(),
    latencySamples: z.number().int().nonnegative().optional(),
    tail_time_ms: z.number().nonnegative().optional(),
    tailTimeMs: z.number().nonnegative().optional(),
    capabilities: ModelCapabilitiesSchema.optional(),
    leveling: LevelingConfigSchema.optional(),
  })
  .strict();

export const IoConfigSchema: z.ZodType<IoConfig, unknown> = RawIoConfigSchema.transform(
  (raw): IoConfig => ({
    inputs: raw.inputs ?? 2,
    outputMode: raw.outputMode ?? raw.output_mode ?? "stereo",
    latencySamples: raw.latencySamples ?? raw.latency_samples ?? 0,
    tailTimeMs: raw.tailTimeMs ?? raw.tail_time_ms ?? 0,
    capabilities: raw.capabilities ?? {
      blendMode: "none",
      defaultBlend: 1.0,
      wetPolarityInverted: false,
      supportsTrails: false,
      tempoSyncCapable: false,
      supportsFreeze: false,
      freezeMutesDry: false,
    },
    leveling: raw.leveling ?? {
      autoUnityEnabled: true,
      levelingMetric: "rms_dbfs",
      referenceInputLevelDb: -20.5,
      measuredDefaultOutputLevelDb: -20.5,
      unityGainOffsetDb: 0.0,
      targetGainDb: 0.0,
      calibratedTrimDb: 0.0,
    },
  }),
);

export interface TierProfile {
  channels: number;
  fdnLines: number;
  earlyTaps: number;
  headSvdRank: number;
  loudnessLufs: number;
  outputTrimDb: number;
  validationEsrDb: number;
  arenaEndByteOffset: number;
  prefixCrc32c: number;
}

const RawTierProfileSchema = z
  .object({
    channels: z.number().int().positive(),
    fdn_lines: z.number().int().nonnegative().optional(),
    fdnLines: z.number().int().nonnegative().optional(),
    early_taps: z.number().int().nonnegative().optional(),
    earlyTaps: z.number().int().nonnegative().optional(),
    head_svd_rank: z.number().int().min(0).max(4).optional(),
    headSvdRank: z.number().int().min(0).max(4).optional(),
    loudness_lufs: z.number().optional(),
    loudnessLufs: z.number().optional(),
    output_trim_db: z.number().optional(),
    outputTrimDb: z.number().optional(),
    validation_esr_db: z.number().optional(),
    validationEsrDb: z.number().optional(),
    arena_end_byte_offset: z.number().int().nonnegative().optional(),
    arenaEndByteOffset: z.number().int().nonnegative().optional(),
    prefix_crc32c: z.number().int().nonnegative().optional(),
    prefixCrc32c: z.number().int().nonnegative().optional(),
  })
  .strict();

export const TierProfileSchema: z.ZodType<TierProfile, unknown> = RawTierProfileSchema.transform(
  (raw): TierProfile => ({
    channels: raw.channels,
    fdnLines: raw.fdnLines ?? raw.fdn_lines ?? 0,
    earlyTaps: raw.earlyTaps ?? raw.early_taps ?? 4,
    headSvdRank: raw.headSvdRank ?? raw.head_svd_rank ?? 4,
    loudnessLufs: raw.loudnessLufs ?? raw.loudness_lufs ?? -20.5,
    outputTrimDb: raw.outputTrimDb ?? raw.output_trim_db ?? 0.0,
    validationEsrDb: raw.validationEsrDb ?? raw.validation_esr_db ?? -40.0,
    arenaEndByteOffset: raw.arenaEndByteOffset ?? raw.arena_end_byte_offset ?? 10_752,
    prefixCrc32c: raw.prefixCrc32c ?? raw.prefix_crc32c ?? 0,
  }),
);

export interface SlimmableTiersConfig {
  lite: TierProfile;
  full: TierProfile;
  ultra: TierProfile;
}

export const SlimmableTiersConfigSchema: z.ZodType<SlimmableTiersConfig, unknown> = z
  .object({
    lite: TierProfileSchema,
    full: TierProfileSchema,
    ultra: TierProfileSchema,
  })
  .strict();

export interface ControlBusConfig {
  decimation: number;
  envelopes: number;
  lfoOscillators: number;
  lfoHarmonics: number;
  fmInteractionRank: number;
  headTrustRegionRadius: number;
}

const RawControlBusConfigSchema = z
  .object({
    decimation: z.number().int().positive().optional(),
    envelopes: z.number().int().nonnegative().optional(),
    lfo_oscillators: z.number().int().nonnegative().optional(),
    lfoOscillators: z.number().int().nonnegative().optional(),
    lfo_harmonics: z.number().int().nonnegative().optional(),
    lfoHarmonics: z.number().int().nonnegative().optional(),
    fm_interaction_rank: z.number().int().nonnegative().optional(),
    fmInteractionRank: z.number().int().nonnegative().optional(),
    head_trust_region_radius: z.number().positive().optional(),
    headTrustRegionRadius: z.number().positive().optional(),
  })
  .strict();

export const ControlBusConfigSchema: z.ZodType<ControlBusConfig, unknown> =
  RawControlBusConfigSchema.transform(
    (raw): ControlBusConfig => ({
      decimation: raw.decimation ?? 16,
      envelopes: raw.envelopes ?? 3,
      lfoOscillators: raw.lfoOscillators ?? raw.lfo_oscillators ?? 2,
      lfoHarmonics: raw.lfoHarmonics ?? raw.lfo_harmonics ?? 4,
      fmInteractionRank: raw.fmInteractionRank ?? raw.fm_interaction_rank ?? 4,
      headTrustRegionRadius: raw.headTrustRegionRadius ?? raw.head_trust_region_radius ?? 2.5,
    }),
  );

export interface Core1Config {
  blocks: number;
  convKernel: number;
  dilations: [number, number, number, number];
  adaaMode: string;
  acCouplingPole: number;
  weightLayout: string;
}

const RawCore1ConfigSchema = z
  .object({
    blocks: z.number().int().positive().optional(),
    conv_kernel: z.number().int().positive().optional(),
    convKernel: z.number().int().positive().optional(),
    dilations: z.tuple([z.number(), z.number(), z.number(), z.number()]).optional(),
    adaa_mode: z.string().optional(),
    adaaMode: z.string().optional(),
    ac_coupling_pole: z.number().optional(),
    acCouplingPole: z.number().optional(),
    weight_layout: z.string().optional(),
    weightLayout: z.string().optional(),
  })
  .strict();

export const Core1ConfigSchema: z.ZodType<Core1Config, unknown> = RawCore1ConfigSchema.transform(
  (raw): Core1Config => ({
    blocks: raw.blocks ?? 4,
    convKernel: raw.convKernel ?? raw.conv_kernel ?? 6,
    dilations: raw.dilations ?? [1, 3, 7, 19],
    adaaMode: raw.adaaMode ?? raw.adaa_mode ?? "1x_bspline_algebraic",
    acCouplingPole: raw.acCouplingPole ?? raw.ac_coupling_pole ?? 0.9993,
    weightLayout: raw.weightLayout ?? raw.weight_layout ?? "tier_delta_exec_order_soa_align64",
  }),
);

export interface Core2Config {
  active: boolean;
  delayStorage: string;
  ringBufferPow2Log2: number;
  interp: string;
  tzfReferenceTapSamples: number;
  switchXfadeSamples: number;
  allpassDelaysSamples: [number, number, number, number, number, number, number, number];
  dcBlockerPole: number;
  loopCeiling: number;
}

const RawCore2ConfigSchema = z
  .object({
    active: z.boolean().optional(),
    delay_storage: z.string().optional(),
    delayStorage: z.string().optional(),
    ring_buffer_pow2_log2: z.number().int().nonnegative().optional(),
    ringBufferPow2Log2: z.number().int().nonnegative().optional(),
    interp: z.string().optional(),
    tzf_reference_tap_samples: z.number().optional(),
    tzfReferenceTapSamples: z.number().optional(),
    switch_xfade_samples: z.number().int().nonnegative().optional(),
    switchXfadeSamples: z.number().int().nonnegative().optional(),
    allpass_delays_samples: z
      .tuple([
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
      ])
      .optional(),
    allpassDelaysSamples: z
      .tuple([
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
        z.number(),
      ])
      .optional(),
    dc_blocker_pole: z.number().optional(),
    dcBlockerPole: z.number().optional(),
    loop_ceiling: z.number().optional(),
    loopCeiling: z.number().optional(),
  })
  .strict();

export const Core2ConfigSchema: z.ZodType<Core2Config, unknown> = RawCore2ConfigSchema.transform(
  (raw): Core2Config => {
    const active = raw.active ?? false;
    return {
      active,
      delayStorage: raw.delayStorage ?? raw.delay_storage ?? "fp16",
      ringBufferPow2Log2: raw.ringBufferPow2Log2 ?? raw.ring_buffer_pow2_log2 ?? (active ? 15 : 0),
      interp: raw.interp ?? "cubic_lagrange_4tap",
      tzfReferenceTapSamples: raw.tzfReferenceTapSamples ?? raw.tzf_reference_tap_samples ?? 64.0,
      switchXfadeSamples: raw.switchXfadeSamples ?? raw.switch_xfade_samples ?? 128,
      allpassDelaysSamples: raw.allpassDelaysSamples ??
        raw.allpass_delays_samples ?? [19, 31, 47, 73, 103, 139, 179, 223],
      dcBlockerPole: raw.dcBlockerPole ?? raw.dc_blocker_pole ?? 0.995,
      loopCeiling: raw.loopCeiling ?? raw.loop_ceiling ?? 0.985,
    };
  },
);

export interface TensorSliceEntry {
  arena: number;
  dtype: string;
  packing?: string | undefined;
  featureDim?: number | undefined;
  packedLen?: number | undefined;
  shape?: number[] | undefined;
  byteOffset: number;
  byteLength: number;
}

const RawTensorSliceEntrySchema = z
  .object({
    arena: z.number().int().min(0).max(3),
    dtype: z.string(),
    packing: z.string().nullable().optional(),
    feature_dim: z.number().int().positive().nullable().optional(),
    featureDim: z.number().int().positive().nullable().optional(),
    packed_len: z.number().int().positive().nullable().optional(),
    packedLen: z.number().int().positive().nullable().optional(),
    shape: z.array(z.number().int().nonnegative()).nullable().optional(),
    byte_offset: z.number().int().nonnegative().optional(),
    byteOffset: z.number().int().nonnegative().optional(),
    byte_length: z.number().int().nonnegative().optional(),
    byteLength: z.number().int().nonnegative().optional(),
  })
  .strict();

export const TensorSliceEntrySchema: z.ZodType<TensorSliceEntry, unknown> =
  RawTensorSliceEntrySchema.transform(
    (raw): TensorSliceEntry => ({
      arena: raw.arena,
      dtype: raw.dtype,
      packing: raw.packing ?? undefined,
      featureDim: raw.featureDim ?? raw.feature_dim ?? undefined,
      packedLen: raw.packedLen ?? raw.packed_len ?? undefined,
      shape: raw.shape ?? undefined,
      byteOffset: raw.byteOffset ?? raw.byte_offset ?? 0,
      byteLength: raw.byteLength ?? raw.byte_length ?? 0,
    }),
  );

export interface StemProvenance {
  inputPcmSha256: string;
  targetPcmSha256: string;
  numFrames: number;
}

const RawStemProvenanceSchema = z
  .object({
    input_pcm_sha256: z.string().optional(),
    inputPcmSha256: z.string().optional(),
    target_pcm_sha256: z.string().optional(),
    targetPcmSha256: z.string().optional(),
    num_frames: z.number().int().nonnegative().optional(),
    numFrames: z.number().int().nonnegative().optional(),
  })
  .strict();

export const StemProvenanceSchema: z.ZodType<StemProvenance, unknown> =
  RawStemProvenanceSchema.transform(
    (raw): StemProvenance => ({
      inputPcmSha256: raw.inputPcmSha256 ?? raw.input_pcm_sha256 ?? "",
      targetPcmSha256: raw.targetPcmSha256 ?? raw.target_pcm_sha256 ?? "",
      numFrames: raw.numFrames ?? raw.num_frames ?? 0,
    }),
  );

export interface ExtractedPhysics {
  delaysMs: number[];
  lfoFreqsHz: number[];
  decayGainsLin: number[];
  detectedGrainSizeMs: number;
  detectedPitchShiftSemitones: number;
}

const RawExtractedPhysicsSchema = z
  .object({
    delays_ms: z.array(z.number()).optional(),
    delaysMs: z.array(z.number()).optional(),
    lfo_freqs_hz: z.array(z.number()).optional(),
    lfoFreqsHz: z.array(z.number()).optional(),
    decay_gains_lin: z.array(z.number()).optional(),
    decayGainsLin: z.array(z.number()).optional(),
    detected_grain_size_ms: z.number().optional(),
    detectedGrainSizeMs: z.number().optional(),
    detected_pitch_shift_semitones: z.number().optional(),
    detectedPitchShiftSemitones: z.number().optional(),
  })
  .strict();

export const ExtractedPhysicsSchema: z.ZodType<ExtractedPhysics, unknown> =
  RawExtractedPhysicsSchema.transform(
    (raw): ExtractedPhysics => ({
      delaysMs: raw.delaysMs ?? raw.delays_ms ?? [],
      lfoFreqsHz: raw.lfoFreqsHz ?? raw.lfo_freqs_hz ?? [],
      decayGainsLin: raw.decayGainsLin ?? raw.decay_gains_lin ?? [],
      detectedGrainSizeMs: raw.detectedGrainSizeMs ?? raw.detected_grain_size_ms ?? 0.0,
      detectedPitchShiftSemitones:
        raw.detectedPitchShiftSemitones ?? raw.detected_pitch_shift_semitones ?? 0.0,
    }),
  );

export interface StemRegistryEntry {
  stemId: string;
  controlsMeta: Record<string, JsonValue>;
  provenance: StemProvenance;
  stemArena3Crc32c: number;
  userMetadata: Record<string, JsonValue>;
  extractedPhysics?: ExtractedPhysics | undefined;
  gAudioUploUF64: TensorSliceEntry;
  cAudioTierF64: TensorSliceEntry;
}

const RawStemRegistryEntrySchema = z
  .object({
    stem_id: z.string().optional(),
    stemId: z.string().optional(),
    controls_meta: z.record(z.string(), JsonValueSchema).optional(),
    controlsMeta: z.record(z.string(), JsonValueSchema).optional(),
    provenance: StemProvenanceSchema,
    stem_arena3_crc32c: z.number().int().nonnegative().optional(),
    stemArena3Crc32c: z.number().int().nonnegative().optional(),
    user_metadata: z.record(z.string(), JsonValueSchema).optional(),
    userMetadata: z.record(z.string(), JsonValueSchema).optional(),
    extracted_physics: ExtractedPhysicsSchema.nullable().optional(),
    extractedPhysics: ExtractedPhysicsSchema.nullable().optional(),
    g_audio_uplo_u_f64: TensorSliceEntrySchema.optional(),
    gAudioUploUF64: TensorSliceEntrySchema.optional(),
    c_audio_tier_f64: TensorSliceEntrySchema.optional(),
    cAudioTierF64: TensorSliceEntrySchema.optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    if (!(val.gAudioUploUF64 ?? val.g_audio_uplo_u_f64)) {
      ctx.addIssue({
        code: "custom",
        message: "StemRegistryEntry requires gAudioUploUF64",
      });
    }
    if (!(val.cAudioTierF64 ?? val.c_audio_tier_f64)) {
      ctx.addIssue({
        code: "custom",
        message: "StemRegistryEntry requires cAudioTierF64",
      });
    }
  });

export const StemRegistryEntrySchema: z.ZodType<StemRegistryEntry, unknown> =
  RawStemRegistryEntrySchema.transform((raw): StemRegistryEntry => {
    const rawPhys = raw.extractedPhysics ?? raw.extracted_physics;
    const rawG = (raw.gAudioUploUF64 ?? raw.g_audio_uplo_u_f64) as TensorSliceEntry;
    const rawC = (raw.cAudioTierF64 ?? raw.c_audio_tier_f64) as TensorSliceEntry;
    return {
      stemId: raw.stemId ?? raw.stem_id ?? "",
      controlsMeta: raw.controlsMeta ?? raw.controls_meta ?? {},
      provenance: raw.provenance,
      stemArena3Crc32c: raw.stemArena3Crc32c ?? raw.stem_arena3_crc32c ?? 0,
      userMetadata: raw.userMetadata ?? raw.user_metadata ?? {},
      extractedPhysics: rawPhys ?? undefined,
      gAudioUploUF64: rawG,
      cAudioTierF64: rawC,
    };
  });

export interface AuditReport {
  phase1Clamps: string[];
  phase2RailWarnings: string[];
  phase3Clamps: string[];
  prescriptions: string[];
}

const RawAuditReportSchema = z
  .object({
    phase1_clamps: z.array(z.string()).optional(),
    phase1Clamps: z.array(z.string()).optional(),
    phase2_rail_warnings: z.array(z.string()).optional(),
    phase2RailWarnings: z.array(z.string()).optional(),
    phase3_clamps: z.array(z.string()).optional(),
    phase3Clamps: z.array(z.string()).optional(),
    prescriptions: z.array(z.string()).optional(),
  })
  .strict();

export const AuditReportSchema: z.ZodType<AuditReport, unknown> = RawAuditReportSchema.transform(
  (raw): AuditReport => ({
    phase1Clamps: raw.phase1Clamps ?? raw.phase1_clamps ?? [],
    phase2RailWarnings: raw.phase2RailWarnings ?? raw.phase2_rail_warnings ?? [],
    phase3Clamps: raw.phase3Clamps ?? raw.phase3_clamps ?? [],
    prescriptions: raw.prescriptions ?? [],
  }),
);

export const CURRENT_SCHEMA_VERSION = "1.0.0";
export const SUPPORTED_SCHEMA_MAJOR = 1;
export const MAX_SUPPORTED_SCHEMA_MINOR = 0;

export function parseSchemaVersion(versionStr: string): {
  major: number;
  minor: number;
  patch: number;
} {
  const parts = versionStr.trim().split(".");
  if (parts.length !== 2 && parts.length !== 3) {
    throw new Error(`Invalid schema version '${versionStr}': expected 'major.minor.patch'`);
  }
  const major = Number(parts[0]);
  const minor = Number(parts[1]);
  const patch = parts.length === 3 ? Number(parts[2]) : 0;
  if (
    !Number.isInteger(major) ||
    !Number.isInteger(minor) ||
    !Number.isInteger(patch) ||
    major !== SUPPORTED_SCHEMA_MAJOR ||
    minor < 0 ||
    minor > MAX_SUPPORTED_SCHEMA_MINOR ||
    patch < 0
  ) {
    throw new Error(
      `Unsupported schema version '${versionStr}' (supported: ${SUPPORTED_SCHEMA_MAJOR}.0..${SUPPORTED_SCHEMA_MAJOR}.${MAX_SUPPORTED_SCHEMA_MINOR})`,
    );
  }
  return { major, minor, patch };
}

export interface DnamJsonHeader {
  version: string;
  containerKind: "dnam" | "dnam-ckpt";
  architecture: string;
  sampleRateTrainHz: number;
  metadata: ModelMetadata;
  userMetadata: Record<string, JsonValue>;
  ioConfig: IoConfig;
  slimmableTiers: SlimmableTiersConfig;
  controlsSchema: ControlSchemaDef[];
  presets: PresetDef[];
  controlBus: ControlBusConfig;
  core1HybridSsm: Core1Config;
  core2Mufdn: Core2Config;
  tensorDirectory: Record<string, TensorSliceEntry>;
  stemsRegistry?: StemRegistryEntry[] | undefined;
  auditReport?: AuditReport | undefined;
}

const RawDnamJsonHeaderSchema = z
  .object({
    version: z.string().optional(),
    container_kind: z.enum(["dnam", "dnam-ckpt"]).optional(),
    containerKind: z.enum(["dnam", "dnam-ckpt"]).optional(),
    architecture: z.string().optional(),
    sample_rate_train_hz: z.number().int().positive().optional(),
    sampleRateTrainHz: z.number().int().positive().optional(),
    metadata: ModelMetadataSchema,
    user_metadata: z.record(z.string(), JsonValueSchema).optional(),
    userMetadata: z.record(z.string(), JsonValueSchema).optional(),
    io_config: IoConfigSchema.optional(),
    ioConfig: IoConfigSchema.optional(),
    slimmable_tiers: SlimmableTiersConfigSchema.optional(),
    slimmableTiers: SlimmableTiersConfigSchema.optional(),
    controls_schema: z.array(ControlSchemaDefSchema).optional(),
    controlsSchema: z.array(ControlSchemaDefSchema).optional(),
    presets: z.array(PresetDefSchema).optional(),
    control_bus: ControlBusConfigSchema.optional(),
    controlBus: ControlBusConfigSchema.optional(),
    core1_hybrid_ssm: Core1ConfigSchema.optional(),
    core1HybridSsm: Core1ConfigSchema.optional(),
    core2_mufdn: Core2ConfigSchema.optional(),
    core2Mufdn: Core2ConfigSchema.optional(),
    tensor_directory: z.record(z.string(), TensorSliceEntrySchema).optional(),
    tensorDirectory: z.record(z.string(), TensorSliceEntrySchema).optional(),
    stems_registry: z.array(StemRegistryEntrySchema).nullable().optional(),
    stemsRegistry: z.array(StemRegistryEntrySchema).nullable().optional(),
    audit_report: AuditReportSchema.nullable().optional(),
    auditReport: AuditReportSchema.nullable().optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    if (val.version !== undefined) {
      try {
        parseSchemaVersion(val.version);
      } catch (err) {
        ctx.addIssue({
          code: "custom",
          message: err instanceof Error ? err.message : "Invalid schema version",
        });
      }
    }
    const tiers = val.slimmableTiers ?? val.slimmable_tiers;
    if (!tiers) {
      ctx.addIssue({
        code: "custom",
        message: "DnamJsonHeader requires slimmableTiers",
      });
    }
  });

export const DnamJsonHeaderSchema: z.ZodType<DnamJsonHeader, unknown> =
  RawDnamJsonHeaderSchema.transform((raw): DnamJsonHeader => {
    const rawTiers = (raw.slimmableTiers ?? raw.slimmable_tiers) as SlimmableTiersConfig;
    return {
      version: raw.version ?? "1.0.0",
      containerKind: raw.containerKind ?? raw.container_kind ?? "dnam",
      architecture: raw.architecture ?? "DNAM_TriPhysics",
      sampleRateTrainHz: raw.sampleRateTrainHz ?? raw.sample_rate_train_hz ?? 48_000,
      metadata: raw.metadata,
      userMetadata: raw.userMetadata ?? raw.user_metadata ?? {},
      ioConfig: raw.ioConfig ??
        raw.io_config ?? {
          inputs: 2,
          outputMode: "stereo",
          latencySamples: 0,
          tailTimeMs: 0,
        },
      slimmableTiers: rawTiers,
      controlsSchema: raw.controlsSchema ?? raw.controls_schema ?? [],
      presets: raw.presets ?? [],
      controlBus: raw.controlBus ??
        raw.control_bus ?? {
          decimation: 16,
          envelopes: 3,
          lfoOscillators: 2,
          lfoHarmonics: 4,
          fmInteractionRank: 4,
          headTrustRegionRadius: 2.5,
        },
      core1HybridSsm: raw.core1HybridSsm ??
        raw.core1_hybrid_ssm ?? {
          blocks: 4,
          convKernel: 6,
          dilations: [1, 3, 7, 19],
          adaaMode: "1x_bspline_algebraic",
          acCouplingPole: 0.9993,
          weightLayout: "tier_delta_exec_order_soa_align64",
        },
      core2Mufdn: raw.core2Mufdn ??
        raw.core2_mufdn ?? {
          active: false,
          delayStorage: "fp16",
          ringBufferPow2Log2: 0,
          interp: "cubic_lagrange_4tap",
          tzfReferenceTapSamples: 64.0,
          switchXfadeSamples: 128,
          allpassDelaysSamples: [19, 31, 47, 73, 103, 139, 179, 223],
          dcBlockerPole: 0.995,
          loopCeiling: 0.985,
        },
      tensorDirectory: raw.tensorDirectory ?? raw.tensor_directory ?? {},
      stemsRegistry: raw.stemsRegistry ?? raw.stems_registry ?? undefined,
      auditReport: raw.auditReport ?? raw.audit_report ?? undefined,
    };
  });

function tierProfileToWire(profile: TierProfile): Record<string, unknown> {
  return {
    channels: profile.channels,
    fdn_lines: profile.fdnLines,
    early_taps: profile.earlyTaps,
    head_svd_rank: profile.headSvdRank,
    loudness_lufs: profile.loudnessLufs,
    output_trim_db: profile.outputTrimDb,
    validation_esr_db: profile.validationEsrDb,
    arena_end_byte_offset: profile.arenaEndByteOffset,
    prefix_crc32c: profile.prefixCrc32c,
  };
}

function tensorSliceEntryToWire(slice: TensorSliceEntry): Record<string, unknown> {
  const out: Record<string, unknown> = {
    arena: slice.arena,
    dtype: slice.dtype,
  };
  if (slice.packing !== undefined) {
    out["packing"] = slice.packing;
  }
  if (slice.featureDim !== undefined) {
    out["feature_dim"] = slice.featureDim;
  }
  if (slice.packedLen !== undefined) {
    out["packed_len"] = slice.packedLen;
  }
  if (slice.shape !== undefined) {
    out["shape"] = slice.shape;
  }
  out["byte_offset"] = slice.byteOffset;
  out["byte_length"] = slice.byteLength;
  return out;
}

function stemRegistryEntryToWire(entry: StemRegistryEntry): Record<string, unknown> {
  const out: Record<string, unknown> = {
    stem_id: entry.stemId,
    controls_meta: entry.controlsMeta,
    provenance: {
      input_pcm_sha256: entry.provenance.inputPcmSha256,
      target_pcm_sha256: entry.provenance.targetPcmSha256,
      num_frames: entry.provenance.numFrames,
    },
    stem_arena3_crc32c: entry.stemArena3Crc32c,
    user_metadata: entry.userMetadata,
  };
  if (entry.extractedPhysics !== undefined) {
    out["extracted_physics"] = {
      delays_ms: entry.extractedPhysics.delaysMs,
      lfo_freqs_hz: entry.extractedPhysics.lfoFreqsHz,
      decay_gains_lin: entry.extractedPhysics.decayGainsLin,
      detected_grain_size_ms: entry.extractedPhysics.detectedGrainSizeMs,
      detected_pitch_shift_semitones: entry.extractedPhysics.detectedPitchShiftSemitones,
    };
  }
  out["g_audio_uplo_u_f64"] = tensorSliceEntryToWire(entry.gAudioUploUF64);
  out["c_audio_tier_f64"] = tensorSliceEntryToWire(entry.cAudioTierF64);
  return out;
}

/**
 * Serializes a TypeScript `DnamJsonHeader` object into the canonical `snake_case` wire object
 * expected by `.dnam` and `.dnam-ckpt` containers.
 */
export function dnamJsonHeaderToWire(header: DnamJsonHeader): Record<string, unknown> {
  const tensorDir: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(header.tensorDirectory)) {
    tensorDir[k] = tensorSliceEntryToWire(v);
  }
  const out: Record<string, unknown> = {
    version: header.version,
    container_kind: header.containerKind,
    architecture: header.architecture,
    sample_rate_train_hz: header.sampleRateTrainHz,
    metadata: {
      name: header.metadata.name,
      author: header.metadata.author ?? null,
      description: header.metadata.description ?? null,
      license: header.metadata.license ?? null,
      created_at: header.metadata.createdAt,
      input_reference_dbfs: header.metadata.inputReferenceDbfs,
      provenance: datasetProvenanceToWire(header.metadata.provenance),
      ...(header.metadata.distributionPolicy !== undefined
        ? {
            distribution_policy: distributionPolicyToWire(header.metadata.distributionPolicy),
          }
        : {}),
      signatures: header.metadata.signatures
        ? modelSignaturesToWire(header.metadata.signatures)
        : null,
    },
    user_metadata: header.userMetadata,
    io_config: {
      inputs: header.ioConfig.inputs,
      output_mode: header.ioConfig.outputMode,
      latency_samples: header.ioConfig.latencySamples,
      tail_time_ms: header.ioConfig.tailTimeMs,
      ...(header.ioConfig.capabilities !== undefined
        ? {
            capabilities: {
              blend_mode: header.ioConfig.capabilities.blendMode,
              default_blend: header.ioConfig.capabilities.defaultBlend,
              wet_polarity_inverted: header.ioConfig.capabilities.wetPolarityInverted,
              supports_trails: header.ioConfig.capabilities.supportsTrails,
              tempo_sync_capable: header.ioConfig.capabilities.tempoSyncCapable,
              supports_freeze: header.ioConfig.capabilities.supportsFreeze,
              freeze_mutes_dry: header.ioConfig.capabilities.freezeMutesDry,
            },
          }
        : {}),
      ...(header.ioConfig.leveling !== undefined
        ? {
            leveling: {
              auto_unity_enabled: header.ioConfig.leveling.autoUnityEnabled,
              leveling_metric: header.ioConfig.leveling.levelingMetric,
              reference_input_level_db: header.ioConfig.leveling.referenceInputLevelDb,
              measured_default_output_level_db:
                header.ioConfig.leveling.measuredDefaultOutputLevelDb,
              unity_gain_offset_db: header.ioConfig.leveling.unityGainOffsetDb,
              target_gain_db: header.ioConfig.leveling.targetGainDb,
              calibrated_trim_db: header.ioConfig.leveling.calibratedTrimDb,
            },
          }
        : {}),
    },
    slimmable_tiers: {
      lite: tierProfileToWire(header.slimmableTiers.lite),
      full: tierProfileToWire(header.slimmableTiers.full),
      ultra: tierProfileToWire(header.slimmableTiers.ultra),
    },
    controls_schema: header.controlsSchema.map(controlSchemaDefToWire),
    presets: header.presets.map(presetDefToWire),
    control_bus: {
      decimation: header.controlBus.decimation,
      envelopes: header.controlBus.envelopes,
      lfo_oscillators: header.controlBus.lfoOscillators,
      lfo_harmonics: header.controlBus.lfoHarmonics,
      fm_interaction_rank: header.controlBus.fmInteractionRank,
      head_trust_region_radius: header.controlBus.headTrustRegionRadius,
    },
    core1_hybrid_ssm: {
      blocks: header.core1HybridSsm.blocks,
      conv_kernel: header.core1HybridSsm.convKernel,
      dilations: header.core1HybridSsm.dilations,
      adaa_mode: header.core1HybridSsm.adaaMode,
      ac_coupling_pole: header.core1HybridSsm.acCouplingPole,
      weight_layout: header.core1HybridSsm.weightLayout,
    },
    core2_mufdn: {
      active: header.core2Mufdn.active,
      delay_storage: header.core2Mufdn.delayStorage,
      ring_buffer_pow2_log2: header.core2Mufdn.ringBufferPow2Log2,
      interp: header.core2Mufdn.interp,
      tzf_reference_tap_samples: header.core2Mufdn.tzfReferenceTapSamples,
      switch_xfade_samples: header.core2Mufdn.switchXfadeSamples,
      allpass_delays_samples: header.core2Mufdn.allpassDelaysSamples,
      dc_blocker_pole: header.core2Mufdn.dcBlockerPole,
      loop_ceiling: header.core2Mufdn.loopCeiling,
    },
    tensor_directory: tensorDir,
  };
  if (header.stemsRegistry !== undefined) {
    out["stems_registry"] = header.stemsRegistry.map(stemRegistryEntryToWire);
  }
  if (header.auditReport !== undefined) {
    out["audit_report"] = {
      phase1_clamps: header.auditReport.phase1Clamps,
      phase2_rail_warnings: header.auditReport.phase2RailWarnings,
      phase3_clamps: header.auditReport.phase3Clamps,
      prescriptions: header.auditReport.prescriptions,
    };
  }
  return out;
}
