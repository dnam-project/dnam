import { describe, expect, it } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations } from "../../tests/setup.js";
import { TauriNativeAudioHost } from "../host.js";
import { DEFAULT_STUDIO_VIEW_MODEL, StudioApp } from "./index.js";

describe("StudioApp Tauri v2 React 19 shell", () => {
  it("renders StudioApp, switches all 4 workspace tabs, matches ARIA snapshot, and passes axe audit", async () => {
    const host = new TauriNativeAudioHost({ backend: "arm-sme" });
    const screen = await render(
      <StudioApp host={host} defaultTab="rack" desktopStatusMessage="dialog: Opened model" />,
    );

    await expect.element(page.getByText(DEFAULT_STUDIO_VIEW_MODEL.workspaceName)).toBeVisible();
    await expect.element(page.getByText("Tri-Physics Reference Master MK1")).toBeVisible();
    await expect.element(page.getByText("dialog: Opened model")).toBeVisible();

    await expect
      .element(page.getByRole("tablist", { name: "Studio Workspace Tabs" }))
      .toMatchAriaInlineSnapshot(`
      - tablist "Studio Workspace Tabs":
        - tab "Rack Audition" [selected]
        - tab "Capture Schedule"
        - tab "Python Trainer (dnam-train)"
        - tab "Tri-Physics Visualizers"
    `);

    await userEvent.click(page.getByRole("tab", { name: "Capture Schedule" }));
    await expect.element(page.getByText(/Interactive Re-Amp Capture Schedule/)).toBeVisible();

    await userEvent.click(page.getByRole("tab", { name: "Python Trainer (dnam-train)" }));
    await expect.element(page.getByText(/4-Stage Differentiable Trainer/)).toBeVisible();

    await userEvent.click(page.getByRole("tab", { name: "Tri-Physics Visualizers" }));
    await expect.element(page.getByTestId("visualizer-poly-adaa")).toBeVisible();
    await expect.element(page.getByTestId("visualizer-zoh-rescaler")).toBeVisible();

    await expectZeroAxeViolations(screen.container);
  });
});
