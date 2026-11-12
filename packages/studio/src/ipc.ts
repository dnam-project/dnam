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
  "plugin:__TAURI_CHANNEL__|fetch",
  "plugin:core|",
  "plugin:resources|",
  "plugin:app|",
  "plugin:path|",
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

const ENVELOPE_META_KEYS: ReadonlySet<string> = new Set([
  "cmd",
  "command",
  "payload",
  "args",
  "callback",
  "error",
  "options",
  "__tauriModule",
]);

const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

function hexDigitValue(byte: number): number {
  if (byte >= 0x30 && byte <= 0x39) return byte - 0x30;
  if (byte >= 0x41 && byte <= 0x46) return byte - 0x41 + 10;
  if (byte >= 0x61 && byte <= 0x66) return byte - 0x61 + 10;
  return -1;
}

function decodePercentPass(input: string): string | null {
  const bytes = UTF8_ENCODER.encode(input);
  const out: number[] = [];
  let changed = false;
  for (let i = 0; i < bytes.length; ) {
    const b0 = bytes[i];
    if (b0 === undefined) {
      break;
    }
    const b1 = bytes[i + 1];
    const b2 = bytes[i + 2];
    if (b0 === 0x25 && b1 !== undefined && b2 !== undefined) {
      const hi = hexDigitValue(b1);
      const lo = hexDigitValue(b2);
      if (hi >= 0 && lo >= 0) {
        const decodedByte = (hi << 4) | lo;
        if (decodedByte <= 0x1f || decodedByte === 0x7f) {
          return null;
        }
        out.push(decodedByte);
        changed = true;
        i += 3;
        continue;
      }
    }
    out.push(b0);
    i += 1;
  }
  if (!changed) {
    return input;
  }
  try {
    const decoded = UTF8_DECODER.decode(new Uint8Array(out));
    for (let j = 0; j < decoded.length; j += 1) {
      const code = decoded.charCodeAt(j);
      if (code >= 0x80 && code <= 0x9f && !input.includes(decoded.charAt(j))) {
        return null;
      }
    }
    return decoded;
  } catch {
    return null;
  }
}

function isUnsafeText(value: string): boolean {
  if (value.includes("..") || value.includes("\0")) {
    return true;
  }
  let cur = value;
  for (let pass = 0; pass < 4; pass += 1) {
    const next = decodePercentPass(cur);
    if (next === null || next.includes("..") || next.includes("\0")) {
      return true;
    }
    if (next === cur) {
      return false;
    }
    cur = next;
  }
  const probe = decodePercentPass(cur);
  return probe === null || probe !== cur;
}

function decodedContainsBackslash(value: string): boolean {
  if (value.includes("\\")) {
    return true;
  }
  let cur = value;
  for (let pass = 0; pass < 4; pass += 1) {
    const next = decodePercentPass(cur);
    if (next === null || next.includes("\\")) {
      return true;
    }
    if (next === cur) {
      return false;
    }
    cur = next;
  }
  const probe = decodePercentPass(cur);
  return probe === null || probe !== cur;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function sanitizeCloneValue(value: unknown, seen: Set<object>): unknown {
  if (typeof value === "string") {
    if (isUnsafeText(value)) {
      throw new Error("Tauri Isolation blocked path traversal or null byte in IPC payload");
    }
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("Tauri Isolation blocked non-finite parameter value");
    }
    return value;
  }
  if (typeof value === "boolean" || value === null) {
    return value;
  }
  if (Array.isArray(value)) {
    if (
      Object.getPrototypeOf(value) !== Array.prototype ||
      Object.getOwnPropertySymbols(value).length > 0
    ) {
      throw new Error("Tauri Isolation blocked non-plain array in IPC payload");
    }
    if (seen.has(value)) {
      throw new Error("Tauri Isolation blocked cyclic IPC payload");
    }
    seen.add(value);
    try {
      for (const name of Object.getOwnPropertyNames(value)) {
        if (name === "length") continue;
        const idx = Number(name);
        if (!Number.isInteger(idx) || idx < 0 || idx >= value.length || String(idx) !== name) {
          throw new Error("Tauri Isolation blocked non-index property on array in IPC payload");
        }
        const desc = Object.getOwnPropertyDescriptor(value, name);
        if (!desc || desc.get !== undefined || desc.set !== undefined || !desc.enumerable) {
          throw new Error("Tauri Isolation blocked accessor property in IPC payload");
        }
      }
      const clonedArr: unknown[] = [];
      for (let i = 0; i < value.length; i += 1) {
        clonedArr.push(sanitizeCloneValue(value[i], seen));
      }
      return clonedArr;
    } finally {
      seen.delete(value);
    }
  }
  if (isPlainObject(value)) {
    return sanitizeCloneObject(value, seen);
  }
  throw new Error("Tauri Isolation blocked unsupported non-plain value in IPC payload");
}

function sanitizeCloneObject(value: unknown, seen: Set<object>): Record<string, unknown> {
  if (!isPlainObject(value) || Object.getOwnPropertySymbols(value).length > 0) {
    throw new Error("Tauri Isolation blocked non-object IPC payload");
  }
  if (seen.has(value)) {
    throw new Error("Tauri Isolation blocked cyclic IPC payload");
  }
  seen.add(value);
  try {
    const out: Record<string, unknown> = {};
    for (const key of Object.getOwnPropertyNames(value)) {
      if (key === "__proto__" || key === "constructor" || key === "prototype") {
        throw new Error("Tauri Isolation blocked forbidden prototype key in IPC payload");
      }
      if (isUnsafeText(key)) {
        throw new Error("Tauri Isolation blocked path traversal or null byte in IPC payload");
      }
      const desc = Object.getOwnPropertyDescriptor(value, key);
      if (!desc || desc.get !== undefined || desc.set !== undefined || !desc.enumerable) {
        throw new Error(
          "Tauri Isolation blocked accessor or non-enumerable property in IPC payload",
        );
      }
      if (key === "toJSON" || typeof desc.value === "function") {
        throw new Error("Tauri Isolation blocked function or toJSON hook in IPC payload");
      }
      if (desc.value !== undefined) {
        Object.defineProperty(out, key, {
          value: sanitizeCloneValue(desc.value, seen),
          enumerable: true,
          writable: true,
          configurable: true,
        });
      }
    }
    return out;
  } finally {
    seen.delete(value);
  }
}

/**
 * Validates and sanitizes an outgoing Tauri v2 IPC payload per the Tauri Isolation Pattern
 * (`window.__TAURI_ISOLATION_HOOK__`).
 */
export function tauriIsolationHook<T extends Readonly<Record<string, unknown>>>(payload: T): T {
  const clean = sanitizeCloneObject(payload, new Set());
  const hasCmd = clean["cmd"] !== undefined;
  const hasCommand = clean["command"] !== undefined;
  if (hasCmd && typeof clean["cmd"] !== "string") {
    throw new Error("Tauri Isolation blocked non-string cmd field");
  }
  if (hasCommand && typeof clean["command"] !== "string") {
    throw new Error("Tauri Isolation blocked non-string command field");
  }
  if (hasCmd && hasCommand && clean["cmd"] !== clean["command"]) {
    throw new Error("Tauri Isolation blocked conflicting cmd/command fields");
  }
  const cmdRaw =
    typeof clean["cmd"] === "string"
      ? clean["cmd"]
      : typeof clean["command"] === "string"
        ? clean["command"]
        : "";
  if (cmdRaw.length === 0) {
    throw new Error("Tauri Isolation blocked IPC payload missing command");
  }
  const isStudioCmd = (STUDIO_IPC_COMMANDS as readonly string[]).includes(cmdRaw);
  const isAllowedPlugin = ALLOWED_TAURI_PLUGIN_PREFIXES.some((entry) =>
    entry.endsWith("|") ? cmdRaw.startsWith(entry) : cmdRaw === entry,
  );
  if (!isStudioCmd && !isAllowedPlugin) {
    throw new Error(`Tauri Isolation blocked unauthorized IPC command: ${cmdRaw}`);
  }
  if (clean["args"] !== undefined && !isPlainObject(clean["args"])) {
    throw new Error("Tauri Isolation blocked non-object args field");
  }
  const allowNullPayload = cmdRaw === "plugin:__TAURI_CHANNEL__|fetch" && clean["payload"] === null;
  if (clean["payload"] !== undefined && !allowNullPayload && !isPlainObject(clean["payload"])) {
    throw new Error(
      cmdRaw === "studio_ipc"
        ? "Tauri Isolation blocked malformed studio_ipc request"
        : "Tauri Isolation blocked non-object payload field",
    );
  }
  if (clean["payload"] !== undefined && clean["args"] !== undefined) {
    throw new Error("Tauri Isolation blocked conflicting payload and args fields");
  }
  if (
    (clean["payload"] !== undefined || clean["args"] !== undefined) &&
    Object.keys(clean).some((k) => !ENVELOPE_META_KEYS.has(k))
  ) {
    throw new Error("Tauri Isolation blocked unexpected top-level fields alongside payload/args");
  }
  const effectiveArgs = isPlainObject(clean["payload"])
    ? clean["payload"]
    : clean["args"] !== undefined
      ? (clean["args"] as Record<string, unknown>)
      : clean;
  if (cmdRaw === "studio_ipc") {
    if (
      clean["payload"] !== undefined &&
      Object.keys(clean["payload"] as Record<string, unknown>).some((k) => k !== "req")
    ) {
      throw new Error("Tauri Isolation blocked malformed studio_ipc request");
    }
    if (
      clean["args"] !== undefined &&
      Object.keys(clean["args"] as Record<string, unknown>).some((k) => k !== "req")
    ) {
      throw new Error("Tauri Isolation blocked malformed studio_ipc request");
    }
    if (
      clean["payload"] === undefined &&
      clean["args"] === undefined &&
      Object.keys(clean).some((k) => !ENVELOPE_META_KEYS.has(k) && k !== "req")
    ) {
      throw new Error("Tauri Isolation blocked malformed studio_ipc request");
    }
    const nestedReq = effectiveArgs["req"];
    if (!isPlainObject(nestedReq)) {
      throw new Error("Tauri Isolation blocked malformed studio_ipc request");
    }
    const nestedKeys = Object.keys(nestedReq);
    if (
      nestedReq["cmd"] !== undefined ||
      typeof nestedReq["command"] !== "string" ||
      nestedReq["command"].length === 0 ||
      !isPlainObject(nestedReq["args"]) ||
      !nestedKeys.every((k) => k === "command" || k === "args")
    ) {
      throw new Error("Tauri Isolation blocked malformed studio_ipc request");
    }
    const nestedCmd = nestedReq["command"];
    if (
      nestedCmd === "studio_ipc" ||
      !(STUDIO_IPC_COMMANDS as readonly string[]).includes(nestedCmd)
    ) {
      throw new Error(`Tauri Isolation blocked unauthorized IPC command: ${nestedCmd}`);
    }
    tauriIsolationHook(nestedReq);
  }
  if (
    cmdRaw === "set_parameter" &&
    (typeof effectiveArgs["value"] !== "number" ||
      !Number.isFinite(effectiveArgs["value"]) ||
      Math.abs(effectiveArgs["value"]) > 3.4028234663852886e38)
  ) {
    throw new Error("Tauri Isolation blocked non-finite or out-of-range f32 parameter value");
  }
  if (cmdRaw === "parse_deep_link") {
    if (
      typeof effectiveArgs["uri"] !== "string" ||
      !effectiveArgs["uri"].trim().startsWith("dnam://")
    ) {
      throw new Error("Tauri Isolation blocked non-dnam:// deep link URI");
    }
    if (parseDnamDeepLink(effectiveArgs["uri"]) === null) {
      throw new Error("Tauri Isolation blocked invalid dnam:// deep link URI");
    }
  }
  return clean as T;
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
  for (let i = 0; i < trimmed.length; i += 1) {
    const code = trimmed.charCodeAt(i);
    if (code <= 0x1f || (code >= 0x7f && code <= 0x9f)) {
      return null;
    }
  }
  if (isUnsafeText(trimmed) || !trimmed.startsWith("dnam://")) {
    return null;
  }
  const rest = trimmed.slice("dnam://".length).replace(/^\/+|\/+$/g, "");
  const isSafeSlug = (s: string): boolean => /^[A-Za-z0-9_-]+$/.test(s);
  if (rest.startsWith("models/")) {
    const target = rest.slice("models/".length).replace(/^\/+|\/+$/g, "");
    return isSafeSlug(target) ? { rawUri: trimmed, kind: "model", target } : null;
  }
  if (rest.startsWith("marketplace/")) {
    const target = rest.slice("marketplace/".length).replace(/^\/+|\/+$/g, "");
    return isSafeSlug(target) ? { rawUri: trimmed, kind: "marketplace", target } : null;
  }
  if (rest.startsWith("open?path=")) {
    const target = rest.slice("open?path=".length);
    return target.length > 0 && !decodedContainsBackslash(target)
      ? { rawUri: trimmed, kind: "open", target }
      : null;
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
 * Parses a single JSONL or SSE (`data: {...}`) line emitted by `dnam-capture`.
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
      readonly lastDeepLink?: DnamDeepLinkRoute | null;
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
