import { describe, expect, it, vi } from "vitest";
import isolationHtml from "../isolation/index.html?raw";
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
    expect(parseDnamDeepLink("dnam://models/../../etc/passwd")).toBeNull();
    expect(parseDnamDeepLink("dnam://models/foo/bar")).toBeNull();
    expect(parseDnamDeepLink("dnam://marketplace/%2e%2e")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=/tmp/.%2e/etc/passwd")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=/tmp/%252e%252e/etc/passwd")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=/tmp/bad%00.dnam")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=/tmp/bad%5cwin.dnam")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=bad\u0085c1")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=bad\nctrl")).toBeNull();
    expect(parseDnamDeepLink("dnam://unknown/route")).toBeNull();
  });

  it("enforces least-privilege allowlists in tauriIsolationHook and live isolation/index.html", async () => {
    const iframe = document.createElement("iframe");
    await new Promise<void>((resolve, reject) => {
      iframe.onload = () => resolve();
      iframe.onerror = () => reject(new Error("failed to load isolation/index.html"));
      iframe.srcdoc = isolationHtml;
      document.body.appendChild(iframe);
    });
    const iframeWin = iframe.contentWindow as unknown as {
      __TAURI_ISOLATION_HOOK__: (payload: Record<string, unknown>) => Record<string, unknown>;
      structuredClone: <U>(v: U) => U;
      Object: ObjectConstructor;
      String: StringConstructor;
    };
    const iframeHook = iframeWin.__TAURI_ISOLATION_HOOK__;
    expect(typeof iframeHook).toBe("function");

    const checkBothAllow = (payload: Record<string, unknown>) => {
      const tsOut = tauriIsolationHook(payload);
      const htmlIn = iframeWin.structuredClone(payload);
      const htmlOut = iframeHook(htmlIn);
      expect(tsOut).not.toBe(payload);
      expect(htmlOut).not.toBe(htmlIn);
      expect(htmlOut).toEqual(tsOut);
      return tsOut;
    };
    const checkBothReject = (payload: Record<string, unknown>, pattern: RegExp) => {
      expect(() => tauriIsolationHook(payload)).toThrow(pattern);
      const htmlIn = iframeWin.structuredClone(payload);
      expect(() => iframeHook(htmlIn)).toThrow(pattern);
    };

    expect(checkBothAllow({ cmd: "plugin:dialog|open", filters: ["dnam"] })).toEqual({
      cmd: "plugin:dialog|open",
      filters: ["dnam"],
    });
    expect(
      checkBothAllow({
        cmd: "notify_studio_event",
        payload: { title: "100% Complete", body: "SNR 78.1 dB", extra: null },
      }),
    ).toBeDefined();
    expect(
      checkBothAllow({
        cmd: "parse_deep_link",
        payload: { uri: "dnam://models/b7k_ultra" },
        callback: 1,
        error: 2,
      }),
    ).toBeDefined();
    expect(
      checkBothAllow({ command: "set_parameter", args: { id: "drive", value: 4.5 } }),
    ).toBeDefined();
    checkBothReject({}, /missing command/);
    checkBothReject({ cmd: "plugin:shell|execute" }, /unauthorized IPC command/);
    checkBothReject({ cmd: "plugin:process|exit" }, /unauthorized IPC command/);
    checkBothReject({ cmd: "plugin:process|restart_evil" }, /unauthorized IPC command/);
    expect(checkBothAllow({ cmd: "plugin:process|restart" })).toEqual({
      cmd: "plugin:process|restart",
    });
    checkBothReject(
      { command: "get_capture_plan", args: { model_id: "../etc/passwd" } },
      /path traversal/,
    );
    checkBothReject(
      { command: "get_capture_plan", args: { model_id: ".%2e/secret" } },
      /path traversal/,
    );
    checkBothReject(
      { command: "get_capture_plan", args: { model_id: "%252e%252e/secret" } },
      /path traversal/,
    );
    checkBothReject(
      { command: "get_capture_plan", args: { model_id: "bad%2500null" } },
      /path traversal/,
    );
    checkBothReject(
      { command: "get_capture_plan", args: { model_id: "bad%ffutf8" } },
      /path traversal/,
    );
    checkBothReject(
      { command: "set_parameter", args: { id: "drive", value: Number.NaN } },
      /non-finite parameter value/,
    );
    checkBothReject(
      { cmd: "set_parameter", payload: { id: "drive", value: null } },
      /non-finite or out-of-range f32 parameter value/,
    );
    checkBothReject(
      { cmd: "set_parameter", payload: { id: "drive", value: null }, value: 1.0 },
      /unexpected top-level fields/,
    );
    checkBothReject(
      {
        cmd: "parse_deep_link",
        payload: { uri: "https://evil.example" },
        uri: "dnam://models/ok",
      },
      /unexpected top-level fields/,
    );
    checkBothReject(
      { cmd: "parse_deep_link", payload: { uri: "https://evil.example" } },
      /non-dnam:\/\//,
    );
    checkBothReject(
      { cmd: "parse_deep_link", payload: { uri: "dnam://unknown/route" } },
      /invalid dnam:\/\//,
    );
    checkBothReject({ command: "get_telemetry", args: { "../bad_key": "ok" } }, /path traversal/);
    checkBothReject(
      {
        cmd: "studio_ipc",
        payload: { req: { command: "plugin:shell|execute", args: {} } },
      },
      /unauthorized IPC command/,
    );
    checkBothReject(
      {
        cmd: "studio_ipc",
        payload: {
          req: { cmd: "get_telemetry", command: "plugin:shell|execute", args: {} },
        },
      },
      /malformed studio_ipc/,
    );
    checkBothReject(
      { cmd: "get_telemetry", command: "plugin:shell|execute" },
      /conflicting cmd\/command/,
    );
    checkBothReject(
      { command: "parse_deep_link", args: null, uri: "dnam://models/ok" },
      /non-object args/,
    );
    checkBothReject({ cmd: "get_telemetry", payload: "bad" }, /non-object payload/);
    checkBothReject({ cmd: "get_telemetry", payload: null }, /non-object payload/);
    // >= 5-level %25-nested encodings fail closed across hooks and deep-link parser
    checkBothReject(
      { cmd: "get_telemetry", args: { p: "%252525252e%252525252e/etc/passwd" } },
      /path traversal/,
    );
    checkBothReject({ cmd: "get_telemetry", args: { n: "ok%2525252500tail" } }, /path traversal/);
    checkBothReject(
      { cmd: "parse_deep_link", payload: { uri: "dnam://open?path=C:%252525255cwin.dnam" } },
      /path traversal/,
    );
    checkBothReject(
      { cmd: "parse_deep_link", payload: { uri: "dnam://open?path=C:%2525255cwin.dnam" } },
      /invalid dnam:\/\//,
    );
    expect(parseDnamDeepLink("dnam://open?path=C:%252525255cwin.dnam")).toBeNull();
    expect(parseDnamDeepLink("dnam://open?path=C:%2525255cwin.dnam")).toBeNull();
    // Out-of-range f64 (> f32::MAX) on set_parameter fails closed across both hooks
    checkBothReject(
      { cmd: "set_parameter", args: { id: "gain", value: 1e300 } },
      /out-of-range f32/,
    );
    checkBothReject(
      {
        cmd: "studio_ipc",
        payload: { req: { command: "set_parameter", args: { id: "gain", value: -1e300 } } },
      },
      /out-of-range f32/,
    );
    // Tauri v2 core:default built-in channel fetch (payload: null) & resource/app/path namespaces pass
    expect(
      checkBothAllow({
        cmd: "plugin:__TAURI_CHANNEL__|fetch",
        callback: 1,
        error: 2,
        payload: null,
        options: { headers: { "Tauri-Channel-Id": "1" } },
      }),
    ).toBeDefined();
    expect(checkBothAllow({ cmd: "plugin:resources|close", payload: { rid: 1 } })).toBeDefined();
    expect(checkBothAllow({ cmd: "plugin:app|version", payload: {} })).toBeDefined();
    expect(
      checkBothAllow({ cmd: "plugin:path|resolve_directory", payload: { directory: 1 } }),
    ).toBeDefined();
    checkBothReject({ cmd: "plugin:__TAURI_CHANNEL__|other", payload: null }, /unauthorized IPC/);
    checkBothReject(
      { cmd: "get_telemetry", payload: {}, args: {} },
      /conflicting payload and args/,
    );
    checkBothReject({ cmd: "studio_ipc", payload: {} }, /malformed studio_ipc/);
    checkBothReject(
      {
        cmd: "studio_ipc",
        payload: { req: { command: "get_telemetry", args: {} }, extra: 1 },
      },
      /malformed studio_ipc/,
    );
    checkBothReject({ cmd: "studio_ipc", payload: [] }, /malformed studio_ipc/);
    checkBothReject(
      {
        cmd: "studio_ipc",
        payload: { req: { command: "studio_ipc", args: {} } },
      },
      /unauthorized IPC command/,
    );
    expect(
      checkBothAllow({
        cmd: "studio_ipc",
        payload: { req: { command: "get_telemetry", args: {} } },
      }),
    ).toBeDefined();
    checkBothReject(
      { cmd: "studio_ipc", payload: { req: "not_an_object" } },
      /malformed studio_ipc/,
    );

    // Non-plain objects (boxed String, getters, toJSON, cyclic graph, __proto__ own key) are blocked
    expect(() =>
      tauriIsolationHook({ command: "get_telemetry", args: { s: new String("../../etc/passwd") } }),
    ).toThrow(/unsupported non-plain value/);
    expect(() =>
      iframeHook(
        iframeWin.structuredClone({
          command: "get_telemetry",
          args: { s: new String("../../etc/passwd") },
        }),
      ),
    ).toThrow(/unsupported non-plain value/);

    const getterTrap: Record<string, unknown> = {};
    Object.defineProperty(getterTrap, "cmd", {
      enumerable: true,
      get: () => "get_telemetry",
    });
    expect(() => tauriIsolationHook(getterTrap)).toThrow(/accessor/);
    const iframeGetterTrap = new iframeWin.Object() as Record<string, unknown>;
    iframeWin.Object.defineProperty(iframeGetterTrap, "cmd", {
      enumerable: true,
      get: () => "get_telemetry",
    });
    expect(() => iframeHook(iframeGetterTrap)).toThrow(/accessor/);

    expect(() =>
      tauriIsolationHook({
        cmd: "get_telemetry",
        args: { toJSON: () => ({ evil: true }) },
      }),
    ).toThrow(/toJSON/);
    const iframeToJson = new iframeWin.Object() as Record<string, unknown>;
    iframeToJson["cmd"] = "get_telemetry";
    const iframeArgs = new iframeWin.Object() as Record<string, unknown>;
    iframeArgs["toJSON"] = () => ({ evil: true });
    iframeToJson["args"] = iframeArgs;
    expect(() => iframeHook(iframeToJson)).toThrow(/toJSON/);

    const protoBag: Record<string, unknown> = { cmd: "get_telemetry" };
    Object.defineProperty(protoBag, "__proto__", {
      value: { polluted: true },
      enumerable: true,
      configurable: true,
    });
    expect(() => tauriIsolationHook(protoBag)).toThrow(/forbidden prototype key/);
    const iframeProtoBag = new iframeWin.Object() as Record<string, unknown>;
    iframeProtoBag["cmd"] = "get_telemetry";
    iframeWin.Object.defineProperty(iframeProtoBag, "__proto__", {
      value: new iframeWin.Object(),
      enumerable: true,
      configurable: true,
    });
    expect(() => iframeHook(iframeProtoBag)).toThrow(/forbidden prototype key/);

    const cyclic: Record<string, unknown> = { cmd: "get_telemetry" };
    cyclic["self"] = cyclic;
    checkBothReject(cyclic, /cyclic/);

    const cycArr: unknown[] = [];
    cycArr.push(cycArr);
    checkBothReject({ cmd: "get_telemetry", args: { items: cycArr } }, /cyclic/);
    const taggedArr = Object.assign(["dnam"], { evil: "x" });
    expect(() => tauriIsolationHook({ cmd: "plugin:dialog|open", filters: taggedArr })).toThrow(
      /non-index property on array/,
    );
    expect(() => tauriIsolationHook(null as unknown as Record<string, unknown>)).toThrow(
      /non-object IPC payload/,
    );
    checkBothReject({ cmd: 123 }, /non-string cmd/);
    checkBothReject({ command: 123 }, /non-string command/);
    checkBothReject(
      { cmd: "studio_ipc", args: { req: { command: "get_telemetry", args: {} }, extra: 1 } },
      /malformed studio_ipc/,
    );
    checkBothReject(
      { cmd: "studio_ipc", req: { command: "get_telemetry", args: {} }, extra: 1 },
      /malformed studio_ipc/,
    );

    iframe.remove();
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
