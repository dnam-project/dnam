import { describe, expect, it } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations, setRangeInputValue } from "../../../tests/setup.js";
import { DnamAudioHostProvider, WebAudioWorkletHost } from "../../host/index.js";
import { DnamPluginRack, type DnamPluginRackModel, dbToMeterPercent } from "./index.js";

const SAMPLE_MODEL: DnamPluginRackModel = {
  modelId: "SLO100_Lead",
  title: "Soldano SLO-100 Overdrive",
  subtitle: "12-Layer PolyADAA + Radial-Polar RG-LRU + Modulated UFDN",
  parameters: [
    {
      id: "gain",
      label: "Preamp Gain",
      min: 0,
      max: 10,
      defaultValue: 6.5,
      observedMin: 1,
      observedMax: 9,
      domain: "core1",
    },
    {
      id: "space",
      label: "Cabinet Space",
      min: 0,
      max: 10,
      defaultValue: 4.0,
      domain: "core2",
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
        { index: 2, label: "On", observed: false },
      ],
    },
  ],
};

describe("DnamPluginRack and TriPhysicsTelemetry", () => {
  it("converts dB levels to meter percentages", () => {
    expect(dbToMeterPercent(-60)).toBe(0);
    expect(dbToMeterPercent(0)).toBe(100);
    expect(dbToMeterPercent(-30)).toBeCloseTo(50, 3);
  });

  it("renders DnamPluginRack with switches and updates host parameters interactively", async () => {
    const host = new WebAudioWorkletHost({
      backend: "arm-sme",
      initialParameters: { gain: 6.5, space: 4.0 },
      initialSwitches: { bright: 0 },
    });

    const screen = await render(
      <DnamAudioHostProvider host={host}>
        <DnamPluginRack model={SAMPLE_MODEL} />
      </DnamAudioHostProvider>,
    );

    await expect.element(page.getByText("Soldano SLO-100 Overdrive")).toBeVisible();
    await expect.element(page.getByText("Tri-Physics Real-Time Telemetry")).toBeVisible();

    const gainInput = screen.container.querySelector(
      'input[aria-label="Preamp Gain"]',
    ) as HTMLInputElement;
    setRangeInputValue(gainInput, 7.5);
    expect(host.getParameter("gain")).toBe(7.5);

    await userEvent.click(page.getByRole("button", { name: "Warm" }));
    expect(host.getSwitchIndex("bright")).toBe(1);

    await expectZeroAxeViolations(screen.container);
  });

  it("renders DnamPluginRack cleanly when model has zero switches", async () => {
    const screen = await render(
      <DnamAudioHostProvider>
        <DnamPluginRack model={{ ...SAMPLE_MODEL, modelId: "NoSwitch", switches: [] }} />
      </DnamAudioHostProvider>,
    );
    await expect.element(page.getByTestId("dnam-plugin-rack-NoSwitch")).toBeVisible();
    await expectZeroAxeViolations(screen.container);
  });
});
