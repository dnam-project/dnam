export type CaptureStepTier =
  | "star_center"
  | "star_axis"
  | "switch_state"
  | "corner_anchor"
  | "prescribed";

export interface CapturePlanStepView {
  readonly stemId: string;
  readonly tier: CaptureStepTier;
  readonly prompt: string;
  readonly rangeCoordinates: Readonly<Record<string, number>>;
  readonly switchIndices: Readonly<Record<string, number>>;
  readonly captured: boolean;
  readonly snrDb?: number | undefined;
  readonly lagSamples?: number | undefined;
}

export interface TrainingEpochPoint {
  readonly epoch: number;
  readonly totalEpochs: number;
  readonly stage: string;
  readonly esrLoss: number;
  readonly stftLoss: number;
  readonly maxRgLruRadius: number;
  readonly ufdnSpectralNorm: number;
}

export interface TrainingTimelineView {
  readonly modelId: string;
  readonly status: "idle" | "running" | "completed" | "failed";
  readonly backendDevice: string;
  readonly clampedParameterCount: number;
  readonly history: readonly TrainingEpochPoint[];
}
