/**
 * Parametric `range` and `switch` control schemas (`ControlSchemaDef`), UI hints (`ControlUiHints`),
 * power-skew gesture mapping, and built-in presets (`PresetDef`).
 */

import { z } from "zod";
import { type JsonValue, JsonValueSchema } from "./base.js";

export type ControlKind = "range" | "switch";

export const TEMPO_SUBDIVISIONS = [
  "1/2.",
  "1/2",
  "1/2T",
  "1/4.",
  "1/4",
  "1/4T",
  "1/8.",
  "1/8",
  "1/8T",
  "1/16.",
  "1/16",
  "1/16T",
] as const;

export type TempoSubdivision = (typeof TEMPO_SUBDIVISIONS)[number];

export const TEMPO_SUBDIVISION_MULTIPLIERS: Record<TempoSubdivision, number> = {
  "1/2.": 3.0,
  "1/2": 2.0,
  "1/2T": 4.0 / 3.0,
  "1/4.": 1.5,
  "1/4": 1.0,
  "1/4T": 2.0 / 3.0,
  "1/8.": 0.75,
  "1/8": 0.5,
  "1/8T": 1.0 / 3.0,
  "1/16.": 0.375,
  "1/16": 0.25,
  "1/16T": 1.0 / 6.0,
};

export type ControlCalibrationRole =
  | "delay_time_ms"
  | "lfo_rate_hz"
  | "feedback"
  | "mix"
  | "crossover_hz"
  | "stereo_mode"
  | "stereo_width";

export interface ControlCalibration {
  role: ControlCalibrationRole;
  lobattoPhysicalAnchors?: [number, number, number] | undefined;
  conditionalPhysicalAnchors: Record<string, [number, number, number]>;
  tempoSyncDefaultSubdivision?: TempoSubdivision | undefined;
}

export interface TempoSyncResolution {
  targetPhysical: number;
  normalizedU: number;
  switchSelection?: [string, string] | undefined;
  wasClamped: boolean;
}

const RawControlCalibrationSchema = z
  .object({
    role: z.enum([
      "delay_time_ms",
      "lfo_rate_hz",
      "feedback",
      "mix",
      "crossover_hz",
      "stereo_mode",
      "stereo_width",
    ]),
    lobatto_physical_anchors: z.tuple([z.number(), z.number(), z.number()]).nullable().optional(),
    lobattoPhysicalAnchors: z.tuple([z.number(), z.number(), z.number()]).nullable().optional(),
    conditional_physical_anchors: z
      .record(z.string(), z.tuple([z.number(), z.number(), z.number()]))
      .optional(),
    conditionalPhysicalAnchors: z
      .record(z.string(), z.tuple([z.number(), z.number(), z.number()]))
      .optional(),
    tempo_sync_default_subdivision: z.enum(TEMPO_SUBDIVISIONS).nullable().optional(),
    tempoSyncDefaultSubdivision: z.enum(TEMPO_SUBDIVISIONS).nullable().optional(),
  })
  .strict();

export const ControlCalibrationSchema: z.ZodType<ControlCalibration, unknown> =
  RawControlCalibrationSchema.transform(
    (raw): ControlCalibration => ({
      role: raw.role,
      lobattoPhysicalAnchors:
        raw.lobattoPhysicalAnchors ?? raw.lobatto_physical_anchors ?? undefined,
      conditionalPhysicalAnchors:
        raw.conditionalPhysicalAnchors ?? raw.conditional_physical_anchors ?? {},
      tempoSyncDefaultSubdivision:
        raw.tempoSyncDefaultSubdivision ?? raw.tempo_sync_default_subdivision ?? undefined,
    }),
  );

export function tempoSubdivisionDelayMs(bpm: number, subdivision: TempoSubdivision): number {
  const safeBpm = Math.min(400.0, Math.max(20.0, bpm));
  return (60_000.0 / safeBpm) * TEMPO_SUBDIVISION_MULTIPLIERS[subdivision];
}

export function tempoSubdivisionFreqHz(bpm: number, subdivision: TempoSubdivision): number {
  return 1000.0 / tempoSubdivisionDelayMs(bpm, subdivision);
}

export function invertLobattoAnchors(vTarget: number, anchors: [number, number, number]): number {
  const [vMin, vMid, vMax] = anchors;
  const c1 = 0.5 * (vMax - vMin);
  const c2 = 0.5 * (vMax + vMin) - vMid;
  const dv = vTarget - vMid;
  if (Math.abs(c2) < 1e-6) {
    if (Math.abs(c1) < 1e-9) {
      return 0.0;
    }
    return Math.min(1.0, Math.max(-1.0, dv / c1));
  }
  const disc = Math.max(0.0, c1 * c1 + 4.0 * c2 * dv);
  const sqrtDisc = Math.sqrt(disc);
  const denom = c1 >= 0.0 ? c1 + sqrtDisc : c1 - sqrtDisc;
  if (Math.abs(denom) < 1e-9) {
    return 0.0;
  }
  return Math.min(1.0, Math.max(-1.0, (2.0 * dv) / denom));
}

export function resolveTempoSyncTarget(
  calibration: ControlCalibration,
  bpm: number,
  subdivisionOverride?: TempoSubdivision,
  foldOctave = false,
): TempoSyncResolution | undefined {
  const sub = subdivisionOverride ?? calibration.tempoSyncDefaultSubdivision ?? "1/4";
  let vTarget: number;
  if (calibration.role === "delay_time_ms") {
    vTarget = tempoSubdivisionDelayMs(bpm, sub);
  } else if (calibration.role === "lfo_rate_hz") {
    vTarget = tempoSubdivisionFreqHz(bpm, sub);
  } else {
    return undefined;
  }

  const condEntries = Object.entries(calibration.conditionalPhysicalAnchors);
  if (condEntries.length > 0) {
    let best:
      | { condKey: string; anchors: [number, number, number]; vEff: number; clamped: boolean }
      | undefined;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const [condKey, anchors] of condEntries) {
      const lo = Math.min(anchors[0], anchors[2]);
      const hi = Math.max(anchors[0], anchors[2]);
      let v = vTarget;
      let folded = false;
      if (foldOctave && lo > 0.0) {
        while (v > hi && v * 0.5 >= lo) {
          v *= 0.5;
          folded = true;
        }
        while (v < lo && v * 2.0 <= hi) {
          v *= 2.0;
          folded = true;
        }
      }
      let dist: number;
      let clamped: boolean;
      let vEff: number;
      if (v >= lo && v <= hi) {
        dist = 0.0;
        clamped = folded;
        vEff = v;
      } else if (v < lo) {
        dist = lo - v;
        clamped = true;
        vEff = lo;
      } else {
        dist = v - hi;
        clamped = true;
        vEff = hi;
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = { condKey, anchors, vEff, clamped };
        if (dist === 0.0 && !folded) {
          break;
        }
      }
    }
    if (best !== undefined) {
      const eqIdx = best.condKey.indexOf("=");
      const swSel: [string, string] | undefined =
        eqIdx > 0 ? [best.condKey.slice(0, eqIdx), best.condKey.slice(eqIdx + 1)] : undefined;
      return {
        targetPhysical: best.vEff,
        normalizedU: invertLobattoAnchors(best.vEff, best.anchors),
        switchSelection: swSel,
        wasClamped: best.clamped,
      };
    }
  }

  if (calibration.lobattoPhysicalAnchors !== undefined) {
    const anchors = calibration.lobattoPhysicalAnchors;
    const lo = Math.min(anchors[0], anchors[2]);
    const hi = Math.max(anchors[0], anchors[2]);
    let v = vTarget;
    let clamped = false;
    if (foldOctave && lo > 0.0) {
      while (v > hi && v * 0.5 >= lo) {
        v *= 0.5;
        clamped = true;
      }
      while (v < lo && v * 2.0 <= hi) {
        v *= 2.0;
        clamped = true;
      }
    }
    if (v < lo || v > hi) {
      clamped = true;
      v = Math.min(hi, Math.max(lo, v));
    }
    return {
      targetPhysical: v,
      normalizedU: invertLobattoAnchors(v, anchors),
      wasClamped: clamped,
    };
  }
  return undefined;
}

export interface ControlUiHints {
  section?: string | undefined;
  order: number;
  widget?: string | undefined;
  size?: string | undefined;
  polarity?: string | undefined;
  midpoint?: number | undefined;
  detents: number[];
  step?: number | undefined;
  decimals?: number | undefined;
  scaleLabels: Record<string, string>;
  optionLabels: Record<string, string>;
  colorHex?: string | undefined;
}

const RawControlUiHintsSchema = z
  .object({
    section: z.string().nullable().optional(),
    order: z.number().int().optional(),
    widget: z.string().nullable().optional(),
    size: z.string().nullable().optional(),
    polarity: z.string().nullable().optional(),
    midpoint: z.number().nullable().optional(),
    detents: z.array(z.number()).optional(),
    step: z.number().nullable().optional(),
    decimals: z.number().int().nonnegative().nullable().optional(),
    scale_labels: z.record(z.string(), z.string()).optional(),
    scaleLabels: z.record(z.string(), z.string()).optional(),
    option_labels: z.record(z.string(), z.string()).optional(),
    optionLabels: z.record(z.string(), z.string()).optional(),
    color_hex: z.string().nullable().optional(),
    colorHex: z.string().nullable().optional(),
  })
  .strict();

export const ControlUiHintsSchema: z.ZodType<ControlUiHints, unknown> =
  RawControlUiHintsSchema.transform(
    (raw): ControlUiHints => ({
      section: raw.section ?? undefined,
      order: raw.order ?? 0,
      widget: raw.widget ?? undefined,
      size: raw.size ?? undefined,
      polarity: raw.polarity ?? undefined,
      midpoint: raw.midpoint ?? undefined,
      detents: raw.detents ?? [],
      step: raw.step ?? undefined,
      decimals: raw.decimals ?? undefined,
      scaleLabels: raw.scaleLabels ?? raw.scale_labels ?? {},
      optionLabels: raw.optionLabels ?? raw.option_labels ?? {},
      colorHex: raw.colorHex ?? raw.color_hex ?? undefined,
    }),
  );

/**
 * Computes the closed-form power-skew exponent $\gamma_{\text{skew}}$ for `[xMin, xMax]`
 * anchored at `hints.midpoint`:
 * $$\gamma_{\text{skew}} = \frac{\ln(0.5)}{\ln((x_{\text{mid}} - x_{\min}) / (x_{\max} - x_{\min}))}$$
 */
export function computeSkewExponent(
  hints: Pick<ControlUiHints, "midpoint">,
  xMin: number,
  xMax: number,
): number {
  const span = xMax - xMin;
  if (span <= 1e-9) {
    return 1.0;
  }
  if (hints.midpoint !== undefined) {
    const ratio = (hints.midpoint - xMin) / span;
    if (ratio > 1e-4 && ratio < 1.0 - 1e-4 && Math.abs(ratio - 0.5) > 1e-6) {
      return Math.log(0.5) / Math.log(ratio);
    }
  }
  return 1.0;
}

/**
 * Maps a normalized UI gesture coordinate $t \in [0, 1]$ to physical parameter value $x \in [x_{\min}, x_{\max}]$.
 */
export function normalizedToPhysical(
  hints: Pick<ControlUiHints, "midpoint">,
  tNorm: number,
  xMin: number,
  xMax: number,
): number {
  const tClamped = Math.min(1.0, Math.max(0.0, tNorm));
  if (tClamped <= 0.0) {
    return xMin;
  }
  if (tClamped >= 1.0) {
    return xMax;
  }
  const gamma = computeSkewExponent(hints, xMin, xMax);
  return xMin + (xMax - xMin) * tClamped ** (1.0 / gamma);
}

/**
 * Maps a physical parameter value $x \in [x_{\min}, x_{\max}]$ to normalized UI gesture coordinate $t \in [0, 1]$.
 */
export function physicalToNormalized(
  hints: Pick<ControlUiHints, "midpoint">,
  xPhys: number,
  xMin: number,
  xMax: number,
): number {
  const span = xMax - xMin;
  if (span <= 1e-9) {
    return 0.0;
  }
  const ratio = Math.min(1.0, Math.max(0.0, (xPhys - xMin) / span));
  if (ratio <= 0.0) {
    return 0.0;
  }
  if (ratio >= 1.0) {
    return 1.0;
  }
  const gamma = computeSkewExponent(hints, xMin, xMax);
  return ratio ** gamma;
}

export interface ControlSchemaDef {
  id: string;
  name: string;
  type: ControlKind;
  unit?: string | undefined;
  lobattoDegree?: number | undefined;
  anchorNodesNorm?: number[] | undefined;
  declaredRange?: [number, number] | undefined;
  safeClampedRange?: [number, number] | undefined;
  default: number | string;
  conditionalMin: Record<string, number>;
  conditionalMax: Record<string, number>;
  declaredOptions?: string[] | undefined;
  activeOptions?: string[] | undefined;
  clampWarning?: string | undefined;
  calibration?: ControlCalibration | undefined;
  uiHints: ControlUiHints;
  userMetadata: Record<string, JsonValue>;
}

const RawControlSchemaDefSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    type: z.enum(["range", "switch"]).optional(),
    kind: z.enum(["range", "switch"]).optional(),
    unit: z.string().nullable().optional(),
    lobatto_degree: z.number().int().min(0).max(3).nullable().optional(),
    lobattoDegree: z.number().int().min(0).max(3).nullable().optional(),
    anchor_nodes_norm: z.array(z.number()).nullable().optional(),
    anchorNodesNorm: z.array(z.number()).nullable().optional(),
    declared_range: z.tuple([z.number(), z.number()]).nullable().optional(),
    declaredRange: z.tuple([z.number(), z.number()]).nullable().optional(),
    safe_clamped_range: z.tuple([z.number(), z.number()]).nullable().optional(),
    safeClampedRange: z.tuple([z.number(), z.number()]).nullable().optional(),
    default: z.union([z.number(), z.string()]),
    conditional_min: z.record(z.string(), z.number()).optional(),
    conditionalMin: z.record(z.string(), z.number()).optional(),
    conditional_max: z.record(z.string(), z.number()).optional(),
    conditionalMax: z.record(z.string(), z.number()).optional(),
    declared_options: z.array(z.string()).nullable().optional(),
    declaredOptions: z.array(z.string()).nullable().optional(),
    active_options: z.array(z.string()).nullable().optional(),
    activeOptions: z.array(z.string()).nullable().optional(),
    clamp_warning: z.string().nullable().optional(),
    clampWarning: z.string().nullable().optional(),
    calibration: ControlCalibrationSchema.nullable().optional(),
    ui_hints: ControlUiHintsSchema.optional(),
    uiHints: ControlUiHintsSchema.optional(),
    user_metadata: z.record(z.string(), JsonValueSchema).optional(),
    userMetadata: z.record(z.string(), JsonValueSchema).optional(),
  })
  .strict()
  .superRefine((val, ctx) => {
    const ctrlType = val.type ?? val.kind;
    if (!ctrlType) {
      ctx.addIssue({
        code: "custom",
        message: "ControlSchemaDef requires 'type' ('range' or 'switch')",
      });
    }
  });

export const ControlSchemaDefSchema: z.ZodType<ControlSchemaDef, unknown> =
  RawControlSchemaDefSchema.transform((raw): ControlSchemaDef => {
    const ctrlType = raw.type ?? raw.kind ?? "range";
    const uiHints: ControlUiHints = raw.uiHints ??
      raw.ui_hints ?? {
        order: 0,
        detents: [],
        scaleLabels: {},
        optionLabels: {},
      };
    return {
      id: raw.id,
      name: raw.name,
      type: ctrlType,
      unit: raw.unit ?? undefined,
      lobattoDegree: raw.lobattoDegree ?? raw.lobatto_degree ?? undefined,
      anchorNodesNorm: raw.anchorNodesNorm ?? raw.anchor_nodes_norm ?? undefined,
      declaredRange: raw.declaredRange ?? raw.declared_range ?? undefined,
      safeClampedRange: raw.safeClampedRange ?? raw.safe_clamped_range ?? undefined,
      default: raw.default,
      conditionalMin: raw.conditionalMin ?? raw.conditional_min ?? {},
      conditionalMax: raw.conditionalMax ?? raw.conditional_max ?? {},
      declaredOptions: raw.declaredOptions ?? raw.declared_options ?? undefined,
      activeOptions: raw.activeOptions ?? raw.active_options ?? undefined,
      clampWarning: raw.clampWarning ?? raw.clamp_warning ?? undefined,
      calibration: raw.calibration ?? undefined,
      uiHints,
      userMetadata: raw.userMetadata ?? raw.user_metadata ?? {},
    };
  });

export interface PresetDef {
  id: string;
  name: string;
  description?: string | undefined;
  controls: Record<string, number | string>;
  userMetadata: Record<string, JsonValue>;
}

const RawPresetDefSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    description: z.string().nullable().optional(),
    controls: z.record(z.string(), z.union([z.number(), z.string()])),
    user_metadata: z.record(z.string(), JsonValueSchema).optional(),
    userMetadata: z.record(z.string(), JsonValueSchema).optional(),
  })
  .strict();

export const PresetDefSchema: z.ZodType<PresetDef, unknown> = RawPresetDefSchema.transform(
  (raw): PresetDef => ({
    id: raw.id,
    name: raw.name,
    description: raw.description ?? undefined,
    controls: raw.controls,
    userMetadata: raw.userMetadata ?? raw.user_metadata ?? {},
  }),
);

export function controlUiHintsToWire(hints: ControlUiHints): Record<string, unknown> {
  return {
    section: hints.section ?? null,
    order: hints.order,
    widget: hints.widget ?? null,
    size: hints.size ?? null,
    polarity: hints.polarity ?? null,
    midpoint: hints.midpoint ?? null,
    detents: hints.detents,
    step: hints.step ?? null,
    decimals: hints.decimals ?? null,
    scale_labels: hints.scaleLabels,
    option_labels: hints.optionLabels,
    color_hex: hints.colorHex ?? null,
  };
}

export function controlCalibrationToWire(cal: ControlCalibration): Record<string, unknown> {
  return {
    role: cal.role,
    lobatto_physical_anchors: cal.lobattoPhysicalAnchors ?? null,
    conditional_physical_anchors: cal.conditionalPhysicalAnchors,
    tempo_sync_default_subdivision: cal.tempoSyncDefaultSubdivision ?? null,
  };
}

export function controlSchemaDefToWire(ctrl: ControlSchemaDef): Record<string, unknown> {
  const out: Record<string, unknown> = {
    id: ctrl.id,
    name: ctrl.name,
    type: ctrl.type,
    unit: ctrl.unit ?? null,
    lobatto_degree: ctrl.lobattoDegree ?? null,
    anchor_nodes_norm: ctrl.anchorNodesNorm ?? null,
    declared_range: ctrl.declaredRange ?? null,
    safe_clamped_range: ctrl.safeClampedRange ?? null,
    default: ctrl.default,
    conditional_min: ctrl.conditionalMin,
    conditional_max: ctrl.conditionalMax,
    declared_options: ctrl.declaredOptions ?? null,
    active_options: ctrl.activeOptions ?? null,
    clamp_warning: ctrl.clampWarning ?? null,
    ui_hints: controlUiHintsToWire(ctrl.uiHints),
    user_metadata: ctrl.userMetadata,
  };
  if (ctrl.calibration !== undefined) {
    out["calibration"] = controlCalibrationToWire(ctrl.calibration);
  }
  return out;
}

export function presetDefToWire(preset: PresetDef): Record<string, unknown> {
  return {
    id: preset.id,
    name: preset.name,
    description: preset.description ?? null,
    controls: preset.controls,
    user_metadata: preset.userMetadata,
  };
}
