import { afterEach, describe, expect, it, vi } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { tauriIsolationHook } from "../../src/ipc.js";
import { InteractiveStudioRoot } from "../../src/ui/main.js";
import { expectZeroAxeViolations, setRangeInputValue } from "../setup.js";

describe("InteractiveStudioRoot component + IPC integration", () => {
  afterEach(() => {
    Reflect.deleteProperty(window, "isTauri");
    Reflect.deleteProperty(window, "__TAURI_INTERNALS__");
    Reflect.deleteProperty(window, "__TAURI_EVENT_PLUGIN_INTERNALS__");
  });

  it("executes full user journey across Dialog/Opener actions, Rack, Capture, 4-Stage Trainer, and 7 Visualizers", async () => {
    const screen = await render(<InteractiveStudioRoot />);

    // 1. Desktop header & Tier 1 actions (Web mode fallback)
    await userEvent.click(page.getByTestId("studio-open-dialog-btn"));
    await expect
      .element(page.getByText("dialog: Opened native .dnam / .dnam-ckpt container picker"))
      .toBeVisible();

    await userEvent.click(page.getByTestId("studio-reveal-export-btn"));
    await expect
      .element(
        page.getByText("opener: Revealed dist/reference_tri_physics_mk1.dnam in Finder/Explorer"),
      )
      .toBeVisible();

    await userEvent.click(page.getByTestId("studio-open-report-btn"));
    await expect
      .element(
        page.getByText("opener: Opened models/reference_tri_physics_mk1/reports/telemetry.html"),
      )
      .toBeVisible();

    // 2. Capture Schedule workflow: record remaining stem
    await userEvent.click(page.getByRole("tab", { name: "Capture Schedule" }));
    await expect.element(page.getByText("1 / 2 Stems Captured")).toBeVisible();

    await userEvent.click(page.getByRole("button", { name: "Record Stem" }));
    await expect.element(page.getByText("2 / 2 Stems Captured")).toBeVisible();
    await expect
      .element(page.getByText(/notification: Captured and aligned stem stem_axis_drive_max/))
      .toBeVisible();

    // 3. 4-Stage Differentiable Trainer workflow: advance epochs to Stage 4 completion
    await userEvent.click(page.getByRole("tab", { name: "Python Trainer (dnam-train)" }));
    const trainBtn = page.getByRole("button", { name: "Run dnam-train" });
    await userEvent.click(trainBtn);
    await expect
      .element(
        page.getByText(
          "notification: Stage 4 Kronecker .dnam-ckpt Export & CL-01..CL-20 Audit completed",
        ),
      )
      .toBeVisible();

    // 4. Interactive Tri-Physics Visualizers workflow
    await userEvent.click(page.getByRole("tab", { name: "Tri-Physics Visualizers" }));
    const sliders = screen.container.querySelectorAll('input[type="range"]');
    for (const slider of sliders) {
      setRangeInputValue(slider as HTMLInputElement, 5);
    }
    await userEvent.click(page.getByRole("button", { name: "48.0 kHz" }));

    await expectZeroAxeViolations(screen.container);
  });

  it("executes native Tauri v2 IPC + Plugin bridge (dialog, opener, notification, updater, deep-link) through tauriIsolationHook", async () => {
    const invokedCommands: string[] = [];
    const bridgeState: {
      deepLinkHandler?: (event: { payload: { kind: string; target: string } }) => void;
    } = {};

    Object.defineProperty(window, "isTauri", {
      configurable: true,
      writable: true,
      value: true,
    });

    Object.defineProperty(window, "__TAURI_EVENT_PLUGIN_INTERNALS__", {
      configurable: true,
      writable: true,
      value: {
        unregisterListener: () => undefined,
      },
    });

    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      configurable: true,
      writable: true,
      value: {
        transformCallback: (cb: typeof bridgeState.deepLinkHandler) => {
          if (typeof cb === "function") {
            bridgeState.deepLinkHandler = cb;
          }
          return 1;
        },
        unregisterCallback: () => undefined,
        invoke: vi.fn(async (cmd: string, args: Record<string, unknown> = {}) => {
          invokedCommands.push(cmd);
          if (cmd.startsWith("plugin:dialog|") || cmd.startsWith("plugin:opener|")) {
            tauriIsolationHook({ cmd, ...args });
          } else if (cmd === "studio_ipc" && args["req"]) {
            tauriIsolationHook(args["req"] as Record<string, unknown>);
          } else if (
            cmd === "prepare_studio_export" ||
            cmd === "prepare_studio_report" ||
            cmd === "notify_studio_event" ||
            cmd === "check_studio_update"
          ) {
            tauriIsolationHook({ cmd, ...args });
          }

          if (cmd === "plugin:dialog|open") {
            return "/Users/studio/SLO100_Ultra.dnam";
          }
          if (cmd === "prepare_studio_export") {
            return "/tmp/dnam-studio/reference_tri_physics_mk1.dnam";
          }
          if (cmd === "prepare_studio_report") {
            return "/tmp/dnam-studio/telemetry_audit_report.html";
          }
          if (cmd === "check_studio_update") {
            return "v0.1.0 (up to date)";
          }
          return null;
        }),
      },
    });

    const screen = await render(<InteractiveStudioRoot />);

    expect(invokedCommands).toContain("studio_ipc");
    expect(invokedCommands).toContain("check_studio_update");

    await userEvent.click(page.getByTestId("studio-open-dialog-btn"));
    await expect
      .element(page.getByText("dialog: Selected /Users/studio/SLO100_Ultra.dnam"))
      .toBeVisible();

    await userEvent.click(page.getByTestId("studio-reveal-export-btn"));
    await expect
      .element(
        page.getByText(
          "opener: Revealed /tmp/dnam-studio/reference_tri_physics_mk1.dnam in Finder",
        ),
      )
      .toBeVisible();

    await userEvent.click(page.getByTestId("studio-open-report-btn"));
    await expect
      .element(page.getByText("opener: Opened /tmp/dnam-studio/telemetry_audit_report.html"))
      .toBeVisible();

    // Capture & Trainer notifications over native Tauri IPC
    await userEvent.click(page.getByRole("tab", { name: "Capture Schedule" }));
    await userEvent.click(page.getByRole("button", { name: "Record Stem" }));
    expect(invokedCommands).toContain("notify_studio_event");

    await userEvent.click(page.getByRole("tab", { name: "Python Trainer (dnam-train)" }));
    await userEvent.click(page.getByRole("button", { name: "Run dnam-train" }));

    bridgeState.deepLinkHandler?.({ payload: { kind: "model", target: "b7k_ultra" } });
    await expect.element(page.getByText("deep-link: Routed model → b7k_ultra")).toBeVisible();

    screen.unmount();
  });
});
