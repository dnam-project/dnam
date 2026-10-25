import { useFixtureSelect } from "react-cosmos/client";
import { CapturePlanPanel, TrainingTimelinePanel } from "./index.js";

function TrainingTimelineFixture() {
  const [status] = useFixtureSelect("Training Status", {
    options: ["idle", "running", "completed", "failed"],
    defaultValue: "running",
  });

  return (
    <TrainingTimelinePanel
      timeline={{
        modelId: "SLO100_Lead",
        status,
        backendDevice: "Apple MPS (bf16 + fp64 Kronecker)",
        clampedParameterCount: 2,
        history: [
          {
            epoch: 24,
            totalEpochs: 40,
            stage: "Stage 3: Joint End-to-End Fine-Tuning",
            esrLoss: 0.00412,
            stftLoss: 0.0184,
            maxRgLruRadius: 0.9932,
            ufdnSpectralNorm: 0.9998,
          },
        ],
      }}
    />
  );
}

export default {
  "Re-Amp Capture Schedule": (
    <CapturePlanPanel
      modelId="SLO100_Lead"
      steps={[
        {
          stemId: "stem_center",
          tier: "star_center",
          prompt: "Set Gain=5.0, Space=5.0",
          rangeCoordinates: { gain: 0.5, space: 0.5 },
          switchIndices: { bright: 0 },
          captured: true,
          snrDb: 74.2,
          lagSamples: 128,
        },
        {
          stemId: "stem_gain_hi",
          tier: "star_axis",
          prompt: "Set Gain=9.0, Space=5.0",
          rangeCoordinates: { gain: 0.9, space: 0.5 },
          switchIndices: { bright: 0 },
          captured: false,
        },
      ]}
    />
  ),
  "4-Stage Differentiable Trainer": <TrainingTimelineFixture />,
};
