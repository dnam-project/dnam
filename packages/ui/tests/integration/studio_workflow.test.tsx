import { describe, expect, it } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import {
  CapturePlanPanel,
  ClampAuditVisualizer,
  DnamAudioHostProvider,
  DnamPluginRack,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  TrainingTimelinePanel,
  WebAudioWorkletHost,
} from "../../src/index.js";
import { expectZeroAxeViolations, setRangeInputValue } from "../setup.js";

describe("Cross-feature Studio workflow integration", () => {
  it("coordinates DnamAudioHostProvider across Plugin Rack, Capture Plan, Trainer, and Visualizers tabs", async () => {
    const host = new WebAudioWorkletHost({
      backend: "arm-sme",
      initialParameters: { gain: 6.0 },
      initialSwitches: { bright: 0 },
    });

    const screen = await render(
      <DnamAudioHostProvider host={host}>
        <Tabs defaultValue="rack">
          <TabsList aria-label="DNAM Studio Workflow">
            <TabsTrigger value="rack">Plugin Rack</TabsTrigger>
            <TabsTrigger value="capture">Capture & Train</TabsTrigger>
            <TabsTrigger value="audit">Stability Audit</TabsTrigger>
          </TabsList>

          <TabsContent value="rack">
            <DnamPluginRack
              model={{
                modelId: "SLO100_Lead",
                title: "Soldano SLO-100 Overdrive",
                subtitle: "12-Layer PolyADAA + Radial-Polar RG-LRU + Modulated UFDN",
                parameters: [
                  {
                    id: "gain",
                    label: "Preamp Gain",
                    min: 0,
                    max: 10,
                    defaultValue: 6.0,
                    observedMin: 1,
                    observedMax: 9,
                    domain: "core1",
                  },
                ],
                switches: [
                  {
                    id: "bright",
                    label: "Bright Cap",
                    defaultIndex: 0,
                    options: [
                      { index: 0, label: "Off", observed: true },
                      { index: 1, label: "Warm", observed: true },
                    ],
                  },
                ],
              }}
            />
          </TabsContent>

          <TabsContent value="capture">
            <CapturePlanPanel
              modelId="SLO100_Lead"
              steps={[
                {
                  stemId: "stem_center",
                  tier: "star_center",
                  prompt: "Set Gain=5.0",
                  rangeCoordinates: { gain: 0.5 },
                  switchIndices: { bright: 0 },
                  captured: true,
                  snrDb: 74.2,
                },
              ]}
            />
            <TrainingTimelinePanel
              timeline={{
                modelId: "SLO100_Lead",
                status: "running",
                backendDevice: "Apple MPS",
                clampedParameterCount: 1,
                history: [],
              }}
            />
          </TabsContent>

          <TabsContent value="audit">
            <ClampAuditVisualizer r2={2.4} />
          </TabsContent>
        </Tabs>
      </DnamAudioHostProvider>,
    );

    const gainSlider = screen.container.querySelector(
      'input[aria-label="Preamp Gain"]',
    ) as HTMLInputElement;
    setRangeInputValue(gainSlider, 8.0);
    expect(host.getParameter("gain")).toBe(8.0);

    await userEvent.click(page.getByRole("tab", { name: "Capture & Train" }));
    await expect.element(page.getByText("1 / 1 Stems Captured")).toBeVisible();

    await userEvent.click(page.getByRole("tab", { name: "Stability Audit" }));
    await expect.element(page.getByTestId("visualizer-clamp-audit")).toBeVisible();

    await expectZeroAxeViolations(screen.container);
  });
});
