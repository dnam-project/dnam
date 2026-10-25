#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

function resolveBinaryPath() {
  const envPath = process.env.DNAM_BINARY_PATH?.trim();
  if (envPath) {
    return envPath;
  }
  const exe = process.platform === "win32" ? "dnam.exe" : "dnam";
  const root = resolve(import.meta.dirname, "../../..");
  for (const profile of ["release", "debug"]) {
    const candidate = resolve(root, "target", profile, exe);
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return exe;
}

const result = spawnSync(resolveBinaryPath(), process.argv.slice(2), {
  cwd: process.env.MOON_WORKING_DIR ?? process.env.INIT_CWD ?? process.cwd(),
  stdio: "inherit",
});
process.exit(result.status ?? 1);
