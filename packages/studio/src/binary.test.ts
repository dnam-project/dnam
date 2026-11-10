import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { buildCliArgs, dnamExecutableName, resolveDnamBinaryPath } from "./binary.js";

describe("binary path resolution and CLI argument builder", () => {
  it("resolves platform executable names and selects newest profile binary", () => {
    expect(dnamExecutableName("darwin")).toBe("dnam");
    expect(dnamExecutableName("win32")).toBe("dnam.exe");

    expect(resolveDnamBinaryPath({ envOverride: "/custom/bin/dnam" })).toBe("/custom/bin/dnam");

    const tempRoot = mkdtempSync(join(tmpdir(), "dnam-bin-test-"));
    try {
      expect(
        resolveDnamBinaryPath({
          envOverride: "",
          workspaceRoot: tempRoot,
          platform: "darwin",
        }),
      ).toBe("dnam");

      const releaseBin = join(tempRoot, "target", "release", "dnam");
      const profilingBin = join(tempRoot, "target", "profiling", "dnam");
      const debugBin = join(tempRoot, "target", "debug", "dnam");
      mkdirSync(join(tempRoot, "target", "release"), { recursive: true });
      mkdirSync(join(tempRoot, "target", "profiling"), { recursive: true });
      mkdirSync(join(tempRoot, "target", "debug"), { recursive: true });

      writeFileSync(releaseBin, "");
      utimesSync(releaseBin, 1_000, 1_000);
      expect(
        resolveDnamBinaryPath({ envOverride: "", workspaceRoot: tempRoot, platform: "darwin" }),
      ).toBe(releaseBin);

      writeFileSync(debugBin, "");
      utimesSync(debugBin, 2_000, 2_000);
      expect(
        resolveDnamBinaryPath({ envOverride: "", workspaceRoot: tempRoot, platform: "darwin" }),
      ).toBe(debugBin);

      writeFileSync(profilingBin, "");
      utimesSync(profilingBin, 3_000, 3_000);
      expect(
        resolveDnamBinaryPath({ envOverride: "", workspaceRoot: tempRoot, platform: "darwin" }),
      ).toBe(profilingBin);

      // Pinned Cargo profile overrides mtimeMs so studio:cli never picks a stale release binary
      expect(
        resolveDnamBinaryPath({
          envOverride: "",
          profile: "debug",
          workspaceRoot: tempRoot,
          platform: "darwin",
        }),
      ).toBe(debugBin);
    } finally {
      rmSync(tempRoot, { recursive: true, force: true });
    }

    expect(buildCliArgs(["status"])).toEqual(["status"]);
    expect(buildCliArgs(["status"], "/tmp/ws")).toEqual(["--workspace", "/tmp/ws", "status"]);
  });
});
