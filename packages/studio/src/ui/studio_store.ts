import type {
  CapturePlanStepView,
  DnamPluginRackModel,
  TrainingTimelineView,
} from "@dnam-project/ui";

export interface StudioWorkspaceViewModel {
  readonly workspaceName: string;
  readonly rackModel: DnamPluginRackModel;
  readonly captureSteps: readonly CapturePlanStepView[];
  readonly trainingTimeline: TrainingTimelineView;
}

export const DEFAULT_STUDIO_VIEW_MODEL: StudioWorkspaceViewModel = {
  workspaceName: "DNAM Desktop Studio Workspace",
  rackModel: {
    modelId: "TriPhysics_Master_MK1",
    title: "Tri-Physics Reference Master MK1",
    subtitle: "12-Layer B-Spline PolyADAA + Radial-Polar RG-LRU + Modulated UFDN",
    parameters: [
      {
        id: "drive",
        label: "Input Drive",
        min: 0,
        max: 10,
        defaultValue: 6.2,
        observedMin: 1.0,
        observedMax: 9.0,
        unit: "dB",
        domain: "core1",
      },
      {
        id: "presence",
        label: "Presence",
        min: 0,
        max: 10,
        defaultValue: 5.5,
        domain: "core1",
      },
      {
        id: "space",
        label: "UFDN Space",
        min: 0,
        max: 10,
        defaultValue: 4.0,
        domain: "core2",
      },
      {
        id: "mod_depth",
        label: "LFO Mod",
        min: 0,
        max: 10,
        defaultValue: 3.0,
        domain: "control",
      },
    ],
    switches: [
      {
        id: "voice",
        label: "Amp Voicing",
        defaultIndex: 1,
        options: [
          { index: 0, label: "Clean", observed: true },
          { index: 1, label: "Crunch", observed: true },
          { index: 2, label: "Ultra", observed: false },
        ],
      },
    ],
  },
  captureSteps: [
    {
      stemId: "stem_00_star_center",
      tier: "star_center",
      prompt: "Set Input Drive=5.0, Presence=5.0, UFDN Space=4.0, Amp Voicing=Crunch",
      rangeCoordinates: { drive: 0.5, presence: 0.5, space: 0.4, mod_depth: 0.3 },
      switchIndices: { voice: 1 },
      captured: true,
      snrDb: 76.4,
      lagSamples: 6,
    },
    {
      stemId: "stem_axis_drive_max",
      tier: "star_axis",
      prompt: "Set Input Drive=9.0, Presence=5.0, UFDN Space=4.0, Amp Voicing=Crunch",
      rangeCoordinates: { drive: 0.9, presence: 0.5, space: 0.4, mod_depth: 0.3 },
      switchIndices: { voice: 1 },
      captured: false,
    },
  ],
  trainingTimeline: {
    modelId: "TriPhysics_Master_MK1",
    status: "completed",
    backendDevice: "Apple MPS (bf16 + fp64 Kronecker .dnam-ckpt)",
    clampedParameterCount: 2,
    history: [
      {
        epoch: 40,
        totalEpochs: 40,
        stage: "Stage 4: Kronecker .dnam-ckpt Export & CL-01..CL-20 Audit",
        esrLoss: 0.00294,
        stftLoss: 0.0142,
        maxRgLruRadius: 0.9912,
        ufdnSpectralNorm: 0.9999,
      },
    ],
  },
};
