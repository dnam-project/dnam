import { describe, expect, it, vi } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations } from "../../../tests/setup.js";
import { CapturePlanPanel, TrainingTimelinePanel } from "./index.js";

describe("CapturePlanPanel and TrainingTimelinePanel", () => {
  it("renders CapturePlanPanel steps and triggers stem recording callback", async () => {
    const onCapture = vi.fn();
    const screen = await render(
      <div>
        <CapturePlanPanel
          modelId="SLO100_Lead"
          onCaptureStep={onCapture}
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
        <CapturePlanPanel modelId="Empty_Capture" steps={[]} />
      </div>,
    );

    await expect.element(page.getByText("1 / 2 Stems Captured")).toBeVisible();
    await expect.element(page.getByText("SNR 74.2 dB")).toBeVisible();

    await userEvent.click(page.getByRole("button", { name: "Record Stem" }));
    expect(onCapture).toHaveBeenCalledWith("stem_gain_hi");
    await expectZeroAxeViolations(screen.container);
  });

  it("renders TrainingTimelinePanel across running, completed, and idle states", async () => {
    const onTrain = vi.fn();
    const screen = await render(
      <div>
        <TrainingTimelinePanel
          onStartTraining={onTrain}
          timeline={{
            modelId: "SLO100_Lead",
            status: "running",
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
        <TrainingTimelinePanel
          timeline={{
            modelId: "Completed_Model",
            status: "completed",
            backendDevice: "CPU",
            clampedParameterCount: 0,
            history: [],
          }}
        />
        <TrainingTimelinePanel
          timeline={{
            modelId: "Idle_Model",
            status: "idle",
            backendDevice: "CPU",
            clampedParameterCount: 0,
            history: [],
          }}
        />
      </div>,
    );

    await expect.element(page.getByText(/Stage 3: Joint End-to-End Fine-Tuning/)).toBeVisible();
    await userEvent.click(page.getByRole("button", { name: "Run dnam-train" }));
    expect(onTrain).toHaveBeenCalledOnce();
    await expectZeroAxeViolations(screen.container);
  });
});
