import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";

export type CargoBuildProfile = "release" | "profiling" | "debug";

const ALLOWED_PROFILES: ReadonlySet<string> = new Set(["release", "profiling", "debug"]);

export interface ResolveBinaryOptions {
  readonly envOverride?: string | undefined;
  readonly profile?: CargoBuildProfile | string | undefined;
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
 * Resolves the compiled `dnam` Rust binary path from environment override, pinned Cargo profile,
 * or newest existing Cargo target profile binary.
 */
export function resolveDnamBinaryPath(options: ResolveBinaryOptions = {}): string {
  const envPath = options.envOverride ?? process.env["DNAM_BINARY_PATH"];
  if (envPath && envPath.trim().length > 0) {
    return envPath.trim();
  }
  const exe = dnamExecutableName(options.platform);
  const root = options.workspaceRoot ?? resolve(import.meta.dirname, "../../..");
  const rawProfile = options.profile ?? process.env["DNAM_CARGO_PROFILE"];
  const pinnedProfile = rawProfile?.trim();
  if (pinnedProfile && ALLOWED_PROFILES.has(pinnedProfile)) {
    return resolve(root, "target", pinnedProfile, exe);
  }
  let bestCandidate: string | null = null;
  let bestMtime = -1;
  for (const profile of ["release", "profiling", "debug"]) {
    const candidate = resolve(root, "target", profile, exe);
    if (existsSync(candidate)) {
      const mtime = statSync(candidate).mtimeMs;
      if (mtime > bestMtime) {
        bestMtime = mtime;
        bestCandidate = candidate;
      }
    }
  }
  return bestCandidate ?? exe;
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
