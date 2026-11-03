#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const engineDir = join(repoRoot, "packages", "engine");
const scratchDir = join(repoRoot, "target", "portable-denormal");
const crateDir = join(scratchDir, "crate");
const sourceDir = join(crateDir, "src");
mkdirSync(sourceDir, { recursive: true });

writeFileSync(
  join(crateDir, "Cargo.toml"),
  `[package]
name = "portable-denormal-probe"
version = "0.0.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
dnam-engine = { path = ${JSON.stringify(engineDir)} }

[workspace]
`,
);
copyFileSync(join(repoRoot, "Cargo.lock"), join(crateDir, "Cargo.lock"));
writeFileSync(
  join(sourceDir, "lib.rs"),
  `#[path = ${JSON.stringify(join(engineDir, "tests", "portable_denormal.rs"))}]
mod portable_denormal;

#[unsafe(no_mangle)]
pub extern "C" fn verify_denormal_policy() -> u32 {
    portable_denormal::verify_portable_denormal_policy();
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn simd_enabled() -> u32 {
    cfg!(target_feature = "simd128") as u32
}
`,
);

for (const [feature, directory] of [
  ["+simd128", "simd"],
  ["-simd128", "scalar"],
]) {
  const targetDir = join(scratchDir, directory);
  const env = { ...process.env, CARGO_TARGET_DIR: targetDir };
  if (env.CARGO_ENCODED_RUSTFLAGS !== undefined) {
    env.CARGO_ENCODED_RUSTFLAGS = [
      env.CARGO_ENCODED_RUSTFLAGS,
      "-C",
      `target-feature=${feature}`,
    ]
      .filter(Boolean)
      .join("\x1f");
  } else {
    env.RUSTFLAGS = [env.RUSTFLAGS, `-C target-feature=${feature}`].filter(Boolean).join(" ");
  }
  execFileSync(
    "cargo",
    [
      "build",
      "--offline",
      "--manifest-path",
      join(crateDir, "Cargo.toml"),
      "--release",
      "--target",
      "wasm32-unknown-unknown",
    ],
    {
      cwd: repoRoot,
      env,
      stdio: "inherit",
    },
  );
  const wasmFile = join(
    targetDir,
    "wasm32-unknown-unknown",
    "release",
    "portable_denormal_probe.wasm",
  );
  const module = new WebAssembly.Module(readFileSync(wasmFile));
  const instance = new WebAssembly.Instance(module, {});
  if (instance.exports.simd_enabled() !== Number(feature === "+simd128")) {
    throw new Error(`WebAssembly target did not apply ${feature}`);
  }
  if (instance.exports.verify_denormal_policy() !== 1) {
    throw new Error(`WebAssembly denormal regression failed with ${feature}`);
  }
  console.log(`WebAssembly denormal regression passed with ${feature}`);
}
