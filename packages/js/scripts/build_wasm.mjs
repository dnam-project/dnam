#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(scriptDir, "..");
const repoRoot = resolve(pkgDir, "../..");
const distDir = join(pkgDir, "dist");

mkdirSync(distDir, { recursive: true });

const isProfiling = process.argv.includes("--profiling") || process.env["DNAM_PROFILE"] === "1";
const cargoProfileArgs = isProfiling ? ["--profile", "profiling"] : ["--release"];
const targetProfileDir = isProfiling ? "profiling" : "release";

// 1. Build host Node-API v10 addon (`dnam-js` cdylib with default `testing` feature)
execFileSync("cargo", ["build", "--package", "dnam-js", ...cargoProfileArgs], {
  cwd: repoRoot,
  stdio: "inherit",
});

const platform = process.platform;
const arch = process.arch;
const libFileName =
  platform === "darwin"
    ? "libdnam_js.dylib"
    : platform === "win32"
      ? "dnam_js.dll"
      : "libdnam_js.so";

const hostLibPath = join(repoRoot, "target", targetProfileDir, libFileName);
const destNodeAddon = join(distDir, `dnam.${platform}-${arch}.node`);
if (!existsSync(hostLibPath)) {
  throw new Error(`Expected host cdylib at ${hostLibPath}`);
}
copyFileSync(hostLibPath, destNodeAddon);

// 2. Build WebAssembly SIMD128 binary (`wasm32-unknown-unknown`, omitting `testing` feature)
execFileSync(
  "cargo",
  [
    "build",
    "--package",
    "dnam-js",
    "--target",
    "wasm32-unknown-unknown",
    ...cargoProfileArgs,
    "--no-default-features",
  ],
  {
    cwd: repoRoot,
    stdio: "inherit",
  },
);

const wasmSrcPath = join(
  repoRoot,
  "target",
  "wasm32-unknown-unknown",
  targetProfileDir,
  "dnam_js.wasm",
);
const destWasmPath = join(distDir, isProfiling ? "dnam.profiling.wasm" : "dnam.wasm");
if (!existsSync(wasmSrcPath)) {
  throw new Error(`Expected WASM binary at ${wasmSrcPath}`);
}
copyFileSync(wasmSrcPath, destWasmPath);
if (isProfiling) {
  copyFileSync(wasmSrcPath, join(distDir, "dnam.wasm"));
}
