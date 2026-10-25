import { describe, expect, it, vi } from "vitest";
import {
  parseDnamDeepLink,
  parseTrainingEventLine,
  StudioIpcClient,
  TAURI_OFFICIAL_PLUGINS,
  tauriIsolationHook,
} from "./ipc.js";

describe("StudioIpcClient, JSONL/SSE parser, and Tauri v2 Isolation Hook", () => {
  it("parses JSONL and SSE training events", () => {
    expect(parseTrainingEventLine("")).toBeNull();
    expect(parseTrainingEventLine("event: epoch")).toBeNull();
    expect(parseTrainingEventLine("data:   ")).toBeNull();

    const jsonl = parseTrainingEventLine('{"event":"epoch_completed","epoch":3}');
    expect(jsonl?.event).toBe("epoch_completed");

    const sse = parseTrainingEventLine('data: {"event_type":"stage_started","stage":"s1"}');
    expect(sse?.event).toBe("stage_started");

    const fallback = parseTrainingEventLine('{"loss":0.01}');
    expect(fallback?.event).toBe("unknown");
  });

  it("parses dnam:// deep links and validates Tier 1 & Tier 2 official Tauri v2 plugins", () => {
    expect(TAURI_OFFICIAL_PLUGINS).toHaveLength(8);
    expect(TAURI_OFFICIAL_PLUGINS.filter((p) => p.tier === 1).map((p) => p.id)).toEqual([
      "dialog",
      "opener",
      "window-state",
    ]);
    expect(TAURI_OFFICIAL_PLUGINS.filter((p) => p.tier === 2).map((p) => p.id)).toEqual([
      "single-instance",
      "deep-link",
      "notification",
      "updater",
      "process",
    ]);

    expect(parseDnamDeepLink("dnam://models/b7k_ultra")).toEqual({
      rawUri: "dnam://models/b7k_ultra",
      kind: "model",
      target: "b7k_ultra",
    });
    expect(parseDnamDeepLink("dnam://marketplace/plexi_68")).toEqual({
      rawUri: "dnam://marketplace/plexi_68",
      kind: "marketplace",
      target: "plexi_68",
    });
    expect(parseDnamDeepLink("dnam://open?path=/tmp/rig.dnam")).toEqual({
      rawUri: "dnam://open?path=/tmp/rig.dnam",
      kind: "open",
      target: "/tmp/rig.dnam",
    });
    expect(parseDnamDeepLink("https://example.com")).toBeNull();
    expect(parseDnamDeepLink("dnam://models/")).toBeNull();
    expect(parseDnamDeepLink("dnam://marketplace/")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=")).toBeNull();
    expect(parseDnamDeepLink("dnam://unknown/route")).toBeNull();
  });

  it("enforces least-privilege allowlists in tauriIsolationHook", () => {
    expect(tauriIsolationHook({ cmd: "plugin:dialog|open", filters: ["dnam"] })).toEqual({
      cmd: "plugin:dialog|open",
      filters: ["dnam"],
    });
    expect(
      tauriIsolationHook({ command: "set_parameter", args: { id: "drive", value: 4.5 } }),
    ).toBeDefined();
    expect(() => tauriIsolationHook({})).toThrow(/missing command/);
    expect(() => tauriIsolationHook({ cmd: "plugin:shell|execute" })).toThrow(
      /unauthorized IPC command/,
    );
    expect(() => tauriIsolationHook({ cmd: "plugin:process|exit" })).toThrow(
      /unauthorized IPC command/,
    );
    expect(tauriIsolationHook({ cmd: "plugin:process|restart" })).toEqual({
      cmd: "plugin:process|restart",
    });
    expect(() =>
      tauriIsolationHook({ command: "get_capture_plan", args: { model_id: "../etc/passwd" } }),
    ).toThrow(/path traversal/);
    expect(() =>
      tauriIsolationHook({ command: "set_parameter", args: { id: "drive", value: Number.NaN } }),
    ).toThrow(/non-finite parameter value/);
    expect(() =>
      tauriIsolationHook({ command: "parse_deep_link", args: { uri: "https://evil.example" } }),
    ).toThrow(/non-dnam:\/\//);
  });

  it("dispatches IPC commands via StudioIpcClient", async () => {
    const defaultClient = new StudioIpcClient();
    const resDefault = await defaultClient.invoke("get_telemetry");
    expect(resDefault.ok).toBe(true);

    const customTransport = vi.fn(async (command: string, args: Record<string, unknown>) => ({
      ok: true,
      data: { command, args },
    }));
    const client = new StudioIpcClient(customTransport);
    await client.setParameter("drive", 7.2);
    await client.setSwitch("voice", 1);
    await client.getPlugins();
    await client.handleDeepLink("dnam://models/b7k_ultra");
    await client.saveWindowState({ width: 1360, height: 880, x: 100, y: 80, maximized: false });
    await client.sendNotification("DNAM Trainer", "Completed");
    expect(customTransport).toHaveBeenCalledTimes(6);
  });
});
