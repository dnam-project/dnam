import { describe, expect, it, vi } from "vitest";
import { TauriNativeAudioHost } from "./host.js";
import { StudioIpcClient } from "./ipc.js";

describe("TauriNativeAudioHost", () => {
  it("synchronizes parameters, switches, and telemetry over StudioIpcClient", () => {
    const transport = vi.fn(async (command: string, args: Record<string, unknown>) => ({
      ok: true,
      data: { command, args },
    }));
    const client = new StudioIpcClient(transport);
    const host = new TauriNativeAudioHost({
      backend: "arm-sme",
      sampleRateHz: 48_000,
      ipcClient: client,
    });

    expect(host.kind).toBe("tauri-native");
    expect(host.getParameter("drive")).toBe(5.0);
    expect(host.getSwitchIndex("voice")).toBe(0);

    const seenR2: number[] = [];
    const unsub = host.subscribe((snap) => {
      seenR2.push(snap.mahalanobisR2);
    });

    host.setParameter("drive", 8.0);
    host.setParameter("drive", Number.NaN);
    host.setSwitchIndex("voice", 2);
    expect(host.getSwitchIndex("voice")).toBe(2);
    expect(seenR2.length).toBeGreaterThanOrEqual(2);
    expect(host.getTelemetry().sampleRateHz).toBe(48_000);
    unsub();
  });
});
