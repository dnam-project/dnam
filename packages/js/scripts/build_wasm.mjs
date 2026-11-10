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
const wasmCargoProfileArgs = isProfiling
  ? ["--profile", "profiling"]
  : ["--profile", "wasm-release"];
const wasmTargetProfileDir = isProfiling ? "profiling" : "wasm-release";

const platform = process.platform;
const arch = process.arch;
const libFileName =
  platform === "darwin"
    ? "libdnam_js.dylib"
    : platform === "win32"
      ? "dnam_js.dll"
      : "libdnam_js.so";
const destNodeAddon = join(distDir, `dnam.${platform}-${arch}.node`);

// 1. Build host Node-API v10 addon (`dnam-js` cdylib with default `testing` feature)
// When `--profiling` is requested and the release `.node` addon already exists, preserve the release addon.
if (!isProfiling || !existsSync(destNodeAddon)) {
  execFileSync("cargo", ["build", "--package", "dnam-js", "--release"], {
    cwd: repoRoot,
    stdio: "inherit",
  });
  const hostLibPath = join(repoRoot, "target", "release", libFileName);
  if (!existsSync(hostLibPath)) {
    throw new Error(`Expected host cdylib at ${hostLibPath}`);
  }
  copyFileSync(hostLibPath, destNodeAddon);
}

// 2. Build WebAssembly SIMD128 binary (`wasm32-unknown-unknown`, omitting `testing` feature, `--import-memory` via `.cargo/config.toml`)
execFileSync(
  "cargo",
  [
    "build",
    "--package",
    "dnam-js",
    "--target",
    "wasm32-unknown-unknown",
    ...wasmCargoProfileArgs,
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
  wasmTargetProfileDir,
  "dnam_js.wasm",
);
const destWasmPath = join(distDir, isProfiling ? "dnam.profiling.wasm" : "dnam.wasm");
if (!existsSync(wasmSrcPath)) {
  throw new Error(`Expected WASM binary at ${wasmSrcPath}`);
}

// 3. Optimize WebAssembly binary via Binaryen `wasm-opt` with explicit target feature flags
const localWasmOptJs = join(pkgDir, "node_modules", "binaryen", "bin", "wasm-opt");
const wasmOptFlags = [
  "-O1",
  ...(isProfiling ? ["--debuginfo"] : ["--strip-dwarf", "--strip-producers"]),
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
