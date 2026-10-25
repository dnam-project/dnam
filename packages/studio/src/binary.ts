import { existsSync } from "node:fs";
import { resolve } from "node:path";

export interface ResolveBinaryOptions {
  readonly envOverride?: string | undefined;
  readonly workspaceRoot?: string | undefined;
  readonly platform?: NodeJS.Platform | undefined;
}

/**
 * Resolves the executable name (`dnam` or `dnam.exe`) for the target platform.
 */
export function dnamExecutableName(platform: NodeJS.Platform = process.platform): string {
  return platform === "win32" ? "dnam.exe" : "dnam";
}

/**
 * Resolves the compiled `dnam` Rust binary path from environment override or Cargo target dirs.
 */
export function resolveDnamBinaryPath(options: ResolveBinaryOptions = {}): string {
  const envPath = options.envOverride ?? process.env["DNAM_BINARY_PATH"];
  if (envPath && envPath.trim().length > 0) {
    return envPath.trim();
  }
  const exe = dnamExecutableName(options.platform);
  const root = options.workspaceRoot ?? resolve(import.meta.dirname, "../../..");
  const releaseCandidate = resolve(root, "target", "release", exe);
  if (existsSync(releaseCandidate)) {
    return releaseCandidate;
  }
  const debugCandidate = resolve(root, "target", "debug", exe);
  if (existsSync(debugCandidate)) {
    return debugCandidate;
  }
  return exe;
}

/**
 * Builds deterministic CLI arguments for `dnam` subcommands.
 */
export function buildCliArgs(
  subcommand: readonly string[],
  workspaceDir?: string | undefined,
): string[] {
  if (workspaceDir && workspaceDir.trim().length > 0) {
    return ["--workspace", workspaceDir.trim(), ...subcommand];
  }
  return [...subcommand];
}
