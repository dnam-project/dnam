#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { resolveDnamBinaryPath } from "../dist/index.js";

const binaryPath = resolveDnamBinaryPath();
const result = spawnSync(binaryPath, process.argv.slice(2), {
  stdio: "inherit",
});
process.exit(result.status ?? 1);
