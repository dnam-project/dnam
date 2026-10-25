import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { buildCliArgs, dnamExecutableName, resolveDnamBinaryPath } from "./binary.js";

describe("binary path resolution and CLI argument builder", () => {
  it("resolves platform executable names and paths", () => {
    expect(dnamExecutableName("darwin")).toBe("dnam");
    expect(dnamExecutableName("win32")).toBe("dnam.exe");

    expect(resolveDnamBinaryPath({ envOverride: "/custom/bin/dnam" })).toBe("/custom/bin/dnam");

    const tempRoot = mkdtempSync(join(tmpdir(), "dnam-bin-test-"));
    expect(
      resolveDnamBinaryPath({
        envOverride: "",
        workspaceRoot: tempRoot,
        platform: "darwin",
      }),
    ).toBe("dnam");

    rmSync(tempRoot, { recursive: true, force: true });

    const rootWithDebug = mkdtempSync(join(tmpdir(), "dnam-bin-dbg-"));
    writeFileSync(join(rootWithDebug, "dummy"), "");
    rmSync(rootWithDebug, { recursive: true, force: true });

    expect(buildCliArgs(["status"])).toEqual(["status"]);
    expect(buildCliArgs(["status"], "/tmp/ws")).toEqual(["--workspace", "/tmp/ws", "status"]);
  });
});
