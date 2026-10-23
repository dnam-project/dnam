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

// 1. Build host Node-API v8 addon (`dnam-js` cdylib)
execFileSync("cargo", ["build", "--package", "dnam-js", "--release"], {
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

const hostLibPath = join(repoRoot, "target", "release", libFileName);
const destNodeAddon = join(distDir, `dnam_js.${platform}-${arch}.node`);
if (!existsSync(hostLibPath)) {
  throw new Error(`Expected host cdylib at ${hostLibPath}`);
}
copyFileSync(hostLibPath, destNodeAddon);

// 2. Build WebAssembly SIMD128 binary (`wasm32-unknown-unknown`)
execFileSync(
  "cargo",
  ["build", "--package", "dnam-js", "--target", "wasm32-unknown-unknown", "--release"],
  {
    cwd: repoRoot,
    env: {
      ...process.env,
      RUSTFLAGS: `${process.env.RUSTFLAGS ?? ""} -C target-feature=+simd128`.trim(),
    },
    stdio: "inherit",
  },
);

const wasmSrcPath = join(repoRoot, "target", "wasm32-unknown-unknown", "release", "dnam_js.wasm");
const destWasmPath = join(distDir, "dnam_engine.wasm");
if (!existsSync(wasmSrcPath)) {
  throw new Error(`Expected WASM binary at ${wasmSrcPath}`);
}
copyFileSync(wasmSrcPath, destWasmPath);
