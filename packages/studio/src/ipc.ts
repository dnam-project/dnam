import type { DnamTelemetrySnapshot } from "@dnam-project/ui";
import { isTauri, invoke as tauriInvoke } from "@tauri-apps/api/core";

export type StudioIpcCommand =
  | "studio_ipc"
  | "prepare_studio_export"
  | "prepare_studio_report"
  | "notify_studio_event"
  | "check_studio_update"
  | "get_telemetry"
  | "set_parameter"
  | "set_switch"
  | "list_backends"
  | "get_workspace"
  | "get_capture_plan"
  | "get_plugins"
  | "parse_deep_link"
  | "save_window_state"
  | "get_window_state"
  | "send_notification";

export const STUDIO_IPC_COMMANDS: readonly StudioIpcCommand[] = [
  "studio_ipc",
  "prepare_studio_export",
  "prepare_studio_report",
  "notify_studio_event",
  "check_studio_update",
  "get_telemetry",
  "set_parameter",
  "set_switch",
  "list_backends",
  "get_workspace",
  "get_capture_plan",
  "get_plugins",
  "parse_deep_link",
  "save_window_state",
  "get_window_state",
  "send_notification",
] as const;

export const ALLOWED_TAURI_PLUGIN_PREFIXES: readonly string[] = [
  "plugin:core|",
  "plugin:event|",
  "plugin:window|",
  "plugin:webview|",
  "plugin:dialog|",
  "plugin:opener|",
  "plugin:window-state|",
  "plugin:deep-link|",
  "plugin:notification|",
  "plugin:updater|",
  "plugin:process|restart",
] as const;

function containsUnsafePathOrNullByte(value: unknown): boolean {
  if (typeof value === "string") {
    return value.includes("..") || value.includes("\0");
  }
  if (Array.isArray(value)) {
    return value.some(containsUnsafePathOrNullByte);
  }
  if (value !== null && typeof value === "object") {
    return Object.values(value).some(containsUnsafePathOrNullByte);
  }
  return false;
}

/**
 * Validates and sanitizes an outgoing Tauri v2 IPC payload per the Tauri Isolation Pattern
 * (`window.__TAURI_ISOLATION_HOOK__`).
 */
export function tauriIsolationHook<T extends Readonly<Record<string, unknown>>>(payload: T): T {
  const cmdRaw =
    typeof payload["cmd"] === "string"
      ? payload["cmd"]
      : typeof payload["command"] === "string"
        ? payload["command"]
        : "";
  if (cmdRaw.length === 0) {
    throw new Error("Tauri Isolation blocked IPC payload missing command");
  }
  const isStudioCmd = (STUDIO_IPC_COMMANDS as readonly string[]).includes(cmdRaw);
  const isAllowedPlugin = ALLOWED_TAURI_PLUGIN_PREFIXES.some((prefix) => cmdRaw.startsWith(prefix));
  if (!isStudioCmd && !isAllowedPlugin) {
    throw new Error(`Tauri Isolation blocked unauthorized IPC command: ${cmdRaw}`);
  }
  if (containsUnsafePathOrNullByte(payload)) {
    throw new Error("Tauri Isolation blocked path traversal or null byte in IPC payload");
  }
  const args =
    payload["args"] !== null && typeof payload["args"] === "object"
      ? (payload["args"] as Record<string, unknown>)
      : payload;
  if (cmdRaw === "set_parameter" && !Number.isFinite(args["value"])) {
    throw new Error("Tauri Isolation blocked non-finite parameter value");
  }
  if (
    cmdRaw === "parse_deep_link" &&
    (typeof args["uri"] !== "string" || !args["uri"].trim().startsWith("dnam://"))
  ) {
    throw new Error("Tauri Isolation blocked non-dnam:// deep link URI");
  }
  return payload;
}

export interface TauriOfficialPluginSpec {
  readonly id:
    | "dialog"
    | "opener"
    | "window-state"
    | "single-instance"
    | "deep-link"
    | "notification"
    | "updater"
    | "process";
  readonly rustCrate: string;
  readonly npmPackage: string | null;
  readonly tier: 1 | 2;
  readonly permission: string;
}

export const TAURI_OFFICIAL_PLUGINS: readonly TauriOfficialPluginSpec[] = [
  {
    id: "dialog",
    rustCrate: "tauri-plugin-dialog",
    npmPackage: "@tauri-apps/plugin-dialog",
    tier: 1,
    permission: "dialog:allow-open,dialog:allow-save",
  },
  {
    id: "opener",
    rustCrate: "tauri-plugin-opener",
    npmPackage: "@tauri-apps/plugin-opener",
    tier: 1,
    permission: "opener:allow-reveal-item-in-dir,opener:allow-open-url(scoped)",
  },
  {
    id: "window-state",
    rustCrate: "tauri-plugin-window-state",
    npmPackage: "@tauri-apps/plugin-window-state",
    tier: 1,
    permission: "window-state:default",
  },
  {
    id: "single-instance",
    rustCrate: "tauri-plugin-single-instance",
    npmPackage: null,
    tier: 2,
    permission: "rust-only",
  },
  {
    id: "deep-link",
    rustCrate: "tauri-plugin-deep-link",
    npmPackage: "@tauri-apps/plugin-deep-link",
    tier: 2,
    permission: "deep-link:default",
  },
  {
    id: "notification",
    rustCrate: "tauri-plugin-notification",
    npmPackage: "@tauri-apps/plugin-notification",
    tier: 2,
    permission: "notification:default",
  },
  {
    id: "updater",
    rustCrate: "tauri-plugin-updater",
    npmPackage: "@tauri-apps/plugin-updater",
    tier: 2,
    permission: "updater:default",
  },
  {
    id: "process",
    rustCrate: "tauri-plugin-process",
    npmPackage: "@tauri-apps/plugin-process",
    tier: 2,
    permission: "process:allow-restart",
  },
] as const;

export interface DnamDeepLinkRoute {
  readonly rawUri: string;
  readonly kind: "model" | "marketplace" | "open";
  readonly target: string;
}

/**
 * Parses a `dnam://` custom URI scheme (`tauri-plugin-deep-link` + `tauri-plugin-single-instance`).
 */
export function parseDnamDeepLink(uri: string): DnamDeepLinkRoute | null {
  const trimmed = uri.trim();
  if (!trimmed.startsWith("dnam://")) {
    return null;
  }
  const rest = trimmed.slice("dnam://".length).replace(/^\/+|\/+$/g, "");
  if (rest.startsWith("models/")) {
    const target = rest.slice("models/".length).replace(/^\/+|\/+$/g, "");
    return target.length > 0 ? { rawUri: trimmed, kind: "model", target } : null;
  }
  if (rest.startsWith("marketplace/")) {
    const target = rest.slice("marketplace/".length).replace(/^\/+|\/+$/g, "");
    return target.length > 0 ? { rawUri: trimmed, kind: "marketplace", target } : null;
  }
  if (rest.startsWith("open?path=")) {
    const target = rest.slice("open?path=".length);
    return target.length > 0 ? { rawUri: trimmed, kind: "open", target } : null;
  }
  return null;
}

export interface WindowStateGeometry {
  readonly width: number;
  readonly height: number;
  readonly x: number;
  readonly y: number;
  readonly maximized: boolean;
}

export interface StudioIpcEnvelope<T = unknown> {
  readonly ok: boolean;
  readonly data: T;
}

export type StudioIpcTransport = (
  command: StudioIpcCommand,
  args: Readonly<Record<string, unknown>>,
) => Promise<StudioIpcEnvelope>;

export interface ParsedTrainingEvent {
  readonly event: string;
  readonly payload: Readonly<Record<string, unknown>>;
}

/**
 * Parses a single JSONL or SSE (`data: {...}`) line emitted by `dnam-train`.
 */
export function parseTrainingEventLine(line: string): ParsedTrainingEvent | null {
  const trimmed = line.trim();
  if (trimmed.length === 0 || trimmed.startsWith("event:")) {
    return null;
  }
  const jsonText = trimmed.startsWith("data:") ? trimmed.slice(5).trim() : trimmed;
  if (jsonText.length === 0) {
    return null;
  }
  const parsed = JSON.parse(jsonText) as Record<string, unknown>;
  const eventName =
    typeof parsed["event"] === "string"
      ? parsed["event"]
      : typeof parsed["event_type"] === "string"
        ? parsed["event_type"]
        : "unknown";
  return {
    event: eventName,
    payload: parsed,
  };
}

/**
 * Typed IPC client communicating with `dnam-studio`'s Rust `StudioIpcHandler` in Tauri v2.
 */
export class StudioIpcClient {
  private readonly transport: StudioIpcTransport;

  public constructor(transport?: StudioIpcTransport | undefined) {
    this.transport =
      transport ??
      (async (command, args) => {
        if (typeof window !== "undefined" && isTauri()) {
          return tauriInvoke<StudioIpcEnvelope>("studio_ipc", {
            req: { command, args },
          });
        }
        return {
          ok: true,
          data: {},
        };
      });
  }

  public async invoke<T = unknown>(
    command: StudioIpcCommand,
    args: Readonly<Record<string, unknown>> = {},
  ): Promise<StudioIpcEnvelope<T>> {
    tauriIsolationHook({ command, args });
    const res = await this.transport(command, args);
    return res as StudioIpcEnvelope<T>;
  }

  public async setParameter(
    id: string,
    value: number,
  ): Promise<StudioIpcEnvelope<DnamTelemetrySnapshot>> {
    return this.invoke<DnamTelemetrySnapshot>("set_parameter", { id, value });
  }

  public async setSwitch(
    id: string,
    index: number,
  ): Promise<StudioIpcEnvelope<DnamTelemetrySnapshot>> {
    return this.invoke<DnamTelemetrySnapshot>("set_switch", { id, index });
  }

  public async getPlugins(): Promise<
    StudioIpcEnvelope<{
      readonly plugins: readonly TauriOfficialPluginSpec[];
      readonly singleInstanceLocked: boolean;
      readonly deepLinkScheme: string;
    }>
  > {
    return this.invoke("get_plugins");
  }

  public async handleDeepLink(uri: string): Promise<StudioIpcEnvelope<DnamDeepLinkRoute>> {
    return this.invoke<DnamDeepLinkRoute>("parse_deep_link", { uri });
  }

  public async saveWindowState(
    geometry: WindowStateGeometry,
  ): Promise<StudioIpcEnvelope<WindowStateGeometry>> {
    return this.invoke<WindowStateGeometry>("save_window_state", { ...geometry });
  }

  public async sendNotification(
    title: string,
    body: string,
  ): Promise<StudioIpcEnvelope<{ readonly delivered: boolean; readonly message: string }>> {
    return this.invoke("send_notification", { title, body });
  }
}
