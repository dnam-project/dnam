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
const hostCargoProfileArgs = isProfiling ? ["--profile", "profiling"] : ["--release"];
const hostTargetProfileDir = isProfiling ? "profiling" : "release";
const wasmCargoProfileArgs = isProfiling
  ? ["--profile", "profiling"]
  : ["--profile", "wasm-release"];
const wasmTargetProfileDir = isProfiling ? "profiling" : "wasm-release";

// 1. Build host Node-API v10 addon (`dnam-js` cdylib with default `testing` feature)
execFileSync("cargo", ["build", "--package", "dnam-js", ...hostCargoProfileArgs], {
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

const hostLibPath = join(repoRoot, "target", hostTargetProfileDir, libFileName);
const destNodeAddon = join(distDir, `dnam.${platform}-${arch}.node`);
if (!existsSync(hostLibPath)) {
  throw new Error(`Expected host cdylib at ${hostLibPath}`);
}
copyFileSync(hostLibPath, destNodeAddon);

// 2. Build WebAssembly SIMD128 binary (`wasm32-unknown-unknown`, omitting `testing` feature, `opt-level = "s"`, `--import-memory`)
const existingWasmRustFlags = process.env["CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS"] ?? "";
const wasmRustFlags = [
  existingWasmRustFlags,
  "-C target-feature=+simd128",
  "-C link-arg=--import-memory",
  "-C link-arg=--export-memory",
  "-C link-arg=--max-memory=268435456",
  isProfiling ? "" : "-C opt-level=s",
]
  .filter(Boolean)
  .join(" ");

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
    env: {
      ...process.env,
      CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS: wasmRustFlags,
    },
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
  isProfiling ? "-O1" : "-Os",
  ...(isProfiling ? ["--debuginfo"] : []),
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

if (isProfiling) {
  copyFileSync(destWasmPath, join(distDir, "dnam.wasm"));
}
