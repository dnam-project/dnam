/**
 * Tests for `Model`, `GramMatrix`, and `crc32c` across both `NapiDriver` and `WasmDriver`.
 */

import "../node.js";

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { DnamError } from "../errors.js";
import { withDriver } from "../internal/native/load.js";
import { loadNodeDriverSync, loadNodeWasmDriverSync } from "../internal/native/napi.js";
import { GramMatrix, uploUPackedLength } from "./gram-matrix.js";
import { crc32c, Model } from "./model.js";

describe("Model & GramMatrix", () => {
  it.each(["napi", "wasm"] as const)(
    "loads, inspects, updates headers in-place, and strips tiers on %s driver",
    (driverKind) => {
      // verify[sdk.ts.model.containers-and-checkpoints]
      // verify[sdk.ts.api.resources-and-data]
      // verify[sdk.ts.api.disposable-scopes]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        using model = Model.reference();
        expect(model.name).toBe("Reference DNAM Model");
        expect(model.maxTier).toBe("ultra");
        expect(model.sampleRateTrainHz).toBe(48_000);
        expect(model.protectionProfile).toBe("open");
        expect(model.parameters.length).toBe(3);
        expect(model.presets.has("warm_slapback")).toBe(true);
        expect(model.tierArenas.lite.byteLength).toBe(10_752);
        expect(model.tierArenas.full?.byteLength).toBe(22_336);
        expect(model.tierArenas.pro?.byteLength).toBe(36_224);
        expect(model.tierArenas.ultra?.byteLength).toBe(64_000);

        const renamed = model.withHeader((hdr) => ({
          ...hdr,
          metadata: { ...hdr.metadata, name: "Custom Studio Rig" },
        }));
        expect(renamed.name).toBe("Custom Studio Rig");

        const liteModel = renamed.withMaxTier("lite");
        expect(liteModel.maxTier).toBe("lite");
        expect(liteModel.tierArenas.full).toBeUndefined();

        const rebuilt = Model.build(model.header, model.arenaBytes);
        expect(rebuilt.name).toBe(model.name);
        expect(crc32c(new Uint8Array([1, 2, 3, 4]))).toBeGreaterThan(0);

        rebuilt.dispose();
        expect(() => rebuilt.name).toThrow(DnamError);
      });
    },
  );

  it("loads Model asynchronously from file path, file URL, Blob, Response, ReadableStream, and ArrayBuffer", async () => {
    // verify[sdk.ts.model.containers-and-checkpoints]
    // verify[sdk.ts.api.sync-init]
    const refBytes = Model.reference().toBytes();
    const tmpDir = mkdtempSync(join(tmpdir(), "dnam-js-test-"));
    const filePath = join(tmpDir, "test.dnam");
    try {
      writeFileSync(filePath, refBytes);
      const fromPath = await Model.load(filePath);
      expect(fromPath.maxTier).toBe("ultra");

      const fromUrl = await Model.load(pathToFileURL(filePath));
      expect(fromUrl.maxTier).toBe("ultra");

      const fromBlob = await Model.load(new Blob([refBytes]));
      expect(fromBlob.maxTier).toBe("ultra");

      const fromResp = await Model.load(new Response(refBytes, { status: 200 }));
      expect(fromResp.maxTier).toBe("ultra");

      await expect(Model.load(new Response(null, { status: 404 }))).rejects.toThrow(DnamError);

      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(refBytes.subarray(0, 1024));
          controller.enqueue(refBytes.subarray(1024));
          controller.close();
        },
      });
      const fromStream = await Model.load(stream);
      expect(fromStream.maxTier).toBe("ultra");

      const fromAb = await Model.load(refBytes.buffer.slice(0));
      expect(fromAb.maxTier).toBe("ultra");

      const dataUrl = `data:application/octet-stream;base64,${Buffer.from(refBytes).toString("base64")}`;
      const fromDataUrl = await Model.load(dataUrl);
      expect(fromDataUrl.maxTier).toBe("ultra");

      await expect(Model.load(join(tmpDir, "missing.dnam"))).rejects.toThrow(DnamError);
      await expect(Model.load(123 as unknown as Uint8Array)).rejects.toThrow(DnamError);
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it.each(["napi", "wasm"] as const)(
    "validates canonical v2.0 Model layout, rejects non-DNAM magic, and packs GramMatrix on %s",
    (driverKind) => {
      // verify[sdk.ts.model.containers-and-checkpoints]
      const driver = driverKind === "napi" ? loadNodeDriverSync() : loadNodeWasmDriverSync();
      withDriver(driver, () => {
        using model = Model.reference();
        expect(model.preamble.magic).toBe("DNAM");
        expect(model.maxTier).toBe("ultra");

        // Rejects non-DNAM magic (e.g. DNPJ project container magic) in Model.from
        const dnpjBytes = model.toBytes();
        dnpjBytes.set(new TextEncoder().encode("DNPJ"), 0);
        expect(() => Model.from(dnpjBytes)).toThrow(DnamError);

        // GramMatrix uplo='U' round-trip
        expect(uploUPackedLength(4)).toBe(10);
        const dense4x4 = new Float64Array([1, 2, 3, 4, 2, 5, 6, 7, 3, 6, 8, 9, 4, 7, 9, 10]);
        const gram = GramMatrix.fromDense(dense4x4, 4);
        expect(gram.packedLength).toBe(10);
        expect(gram.packed.length).toBe(10);
        expect(gram.toDense()).toEqual(dense4x4);
        expect(gram.toDense({ tierDimension: 2 })).toEqual(new Float64Array([1, 2, 2, 5]));
        expect(() => uploUPackedLength(-1)).toThrow(DnamError);
        expect(() => new GramMatrix(new Float64Array(2), 4)).toThrow(DnamError);
        expect(() => new GramMatrix(new Float64Array(2), 0)).toThrow(DnamError);
        expect(() => GramMatrix.fromDense(dense4x4, 0)).toThrow(DnamError);
        expect(() => gram.toDense({ tierDimension: 99 })).toThrow(DnamError);

        // Canonical v2.0 layout validation and invalid-buffer rejection
        expect(model.validateOpenLayout().retainedTier).toBe(3);
        expect(() => Model.from(new Uint8Array(64))).toThrow(DnamError);
      });
    },
  );

  it("property: fast-check verifies GramMatrix uplo='U' symmetric packing, CRC-32C sensitivity, and Model tier stripping", () => {
    // verify[sdk.ts.model.containers-and-checkpoints]
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 12 }),
        fc.double({ min: -50, max: 50, noNaN: true }),
        (dim, scale) => {
          const dense = new Float64Array(dim * dim);
          for (let r = 0; r < dim; r++) {
            for (let c = r; c < dim; c++) {
              const v = (r + 1) * (c + 1) * scale;
              dense[r * dim + c] = v;
              dense[c * dim + r] = v;
            }
          }
          const gram = GramMatrix.fromDense(dense, dim);
          expect(gram.packedLength).toBe((dim * (dim + 1)) / 2);
          const unpacked = gram.toDense();
          for (let i = 0; i < dense.length; i++) {
            expect(unpacked[i]).toBeCloseTo(dense[i] ?? 0, 9);
          }
          const subDim = Math.max(1, Math.floor(dim / 2));
          const sub = gram.toDense({ tierDimension: subDim });
          expect(sub.length).toBe(subDim * subDim);
        },
      ),
      { numRuns: 35 },
    );

    fc.assert(
      fc.property(
        fc.uint8Array({ minLength: 4, maxLength: 256 }),
        fc.integer({ min: 0, max: 255 }),
        (buf, flipIdxRaw) => {
          const c1 = crc32c(buf);
          expect(crc32c(buf)).toBe(c1);
          const mutated = buf.slice();
          const idx = flipIdxRaw % mutated.length;
          mutated[idx] = (mutated[idx] ?? 0) ^ 0xff;
          expect(crc32c(mutated)).not.toBe(c1);
        },
      ),
      { numRuns: 50 },
    );

    using baseModel = Model.reference();
    fc.assert(
      fc.property(fc.constantFrom("lite", "full", "pro", "ultra"), (tier) => {
        using stripped = baseModel.withMaxTier(tier);
        expect(stripped.maxTier).toBe(tier);
        using reloaded = Model.from(stripped.toBytes());
        expect(reloaded.maxTier).toBe(tier);
        expect(reloaded.tierArenas.lite.byteLength).toBe(10_752);
      }),
      { numRuns: 12 },
    );
  });
});
