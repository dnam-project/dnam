import { describe, expect, it } from "vitest";
import { page } from "vitest/browser";
import { render } from "vitest-browser-react";
import { TauriNativeAudioHost } from "../../src/host.js";
import { StudioIpcClient, tauriIsolationHook } from "../../src/ipc.js";
import { StudioApp } from "../../src/ui/index.js";
import { expectZeroAxeViolations, setRangeInputValue } from "../setup.js";

describe("Studio IPC Isolation + Native Host + StudioApp integration", () => {
  it("routes live StudioApp slider mutations through TauriNativeAudioHost and tauriIsolationHook", async () => {
    const isolatedPayloads: Record<string, unknown>[] = [];
    const client = new StudioIpcClient(async (command, args) => {
      const validated = tauriIsolationHook({ command, args });
      isolatedPayloads.push(validated);
      return { ok: true, data: validated };
    });

    const host = new TauriNativeAudioHost({
      backend: "arm-sme",
      sampleRateHz: 48_000,
      ipcClient: client,
    });

    const screen = await render(
      <StudioApp host={host} desktopStatusMessage="Isolated IPC verified" />,
    );

    await expect.element(page.getByText("Isolated IPC verified")).toBeVisible();

    const driveInput = screen.container.querySelector(
      'input[aria-label="Input Drive"]',
    ) as HTMLInputElement;
    setRangeInputValue(driveInput, 7.8);

    expect(isolatedPayloads).toContainEqual({
      command: "set_parameter",
      args: { id: "drive", value: 7.8 },
    });

    await expectZeroAxeViolations(screen.container);
  });
});
