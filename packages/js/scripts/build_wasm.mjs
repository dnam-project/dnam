#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(scriptDir, "..");
const repoRoot = resolve(pkgDir, "../..");
const distDir = join(pkgDir, "dist");

const isDebug = process.argv.includes("--debug") || process.env["DNAM_BUILD_PROFILE"] === "debug";
const isProfiling =
  !isDebug &&
  (process.argv.includes("--profiling") ||
    process.env["DNAM_PROFILE"] === "1" ||
    process.env["DNAM_BUILD_PROFILE"] === "profiling");

const targetMode = isDebug ? "debug" : isProfiling ? "profiling" : "release";
const cargoProfileArgs =
  targetMode === "debug"
    ? []
    : targetMode === "profiling"
      ? ["--profile", "profiling"]
      : ["--release"];
const cargoTargetProfileDir = targetMode;

const nativeOutDir = join(distDir, "native", targetMode);
const wasmOutDir = join(distDir, "wasm", targetMode);
mkdirSync(nativeOutDir, { recursive: true });
mkdirSync(wasmOutDir, { recursive: true });

const platform = process.platform;
const arch = process.arch;
const libFileName =
  platform === "darwin"
    ? "libdnam_js.dylib"
    : platform === "win32"
      ? "dnam_js.dll"
      : "libdnam_js.so";
const nodeAddonFileName = `dnam.${platform}-${arch}.node`;
const destNodeAddon = join(nativeOutDir, nodeAddonFileName);

// 1. Build host Node-API v10 addon (`dnam-js` cdylib with default `testing` feature)
execFileSync("cargo", ["build", "--package", "dnam-js", ...cargoProfileArgs], {
  cwd: repoRoot,
  stdio: "inherit",
});
const hostLibPath = join(repoRoot, "target", cargoTargetProfileDir, libFileName);
if (!existsSync(hostLibPath)) {
  throw new Error(`Expected host cdylib at ${hostLibPath}`);
}
copyFileSync(hostLibPath, destNodeAddon);

// 2. Build WebAssembly SIMD128 binary (`wasm32-unknown-unknown`, omitting `testing` feature, `--import-memory` via `.cargo/config.toml`)
const wasmExtraConfig =
  targetMode === "debug"
    ? ["--config", 'profile.dev.strip="debuginfo"']
    : targetMode === "profiling"
      ? ["--config", 'profile.profiling.strip="debuginfo"']
      : [];
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
    ...wasmExtraConfig,
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
  cargoTargetProfileDir,
  "dnam_js.wasm",
);
const destWasmPath = join(wasmOutDir, "dnam.wasm");
if (!existsSync(wasmSrcPath)) {
  throw new Error(`Expected WASM binary at ${wasmSrcPath}`);
}

if (targetMode === "debug") {
  // Fast debug target: skip `wasm-opt` and copy directly
  copyFileSync(wasmSrcPath, destWasmPath);
} else {
  // Release & Profiling targets: optimize via Binaryen `wasm-opt` with identical `-O1` codegen
  const localWasmOptJs = join(pkgDir, "node_modules", "binaryen", "bin", "wasm-opt");
  const wasmOptFlags = [
    "-O1",
    "--strip-dwarf",
    ...(targetMode === "profiling" ? ["--debuginfo"] : ["--strip-producers"]),
    "--enable-simd",
    "--enable-bulk-memory",
    "--enable-nontrapping-float-to-int",
    "--enable-sign-ext",
    "--enable-mutable-globals",
    "-o",
    destWasmPath,
    wasmSrcPath,
  ];

  if (existsSync(localWasmOptJs)) {
    execFileSync(process.execPath, [localWasmOptJs, ...wasmOptFlags], {
      cwd: pkgDir,
      stdio: "inherit",
    });
  } else {
    execFileSync("wasm-opt", wasmOptFlags, {
      cwd: pkgDir,
      stdio: "inherit",
    });
  }
}
