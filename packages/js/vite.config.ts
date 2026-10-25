import { resolve } from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  if (mode === "worklet") {
    return {
      build: {
        emptyOutDir: false,
        outDir: "dist",
        target: "es2024",
        minify: false,
        sourcemap: true,
        lib: {
          entry: {
            "worklet-processor": resolve(import.meta.dirname, "src/worklet/processor.ts"),
          },
          formats: ["es"],
          fileName: (_format, entryName) => `${entryName}.js`,
        },
        rollupOptions: {
          external: [],
        },
      },
    };
  }

  return {
    build: {
      emptyOutDir: false,
      outDir: "dist",
      target: "es2024",
      minify: false,
      sourcemap: true,
      lib: {
        entry: {
          index: resolve(import.meta.dirname, "src/index.ts"),
          node: resolve(import.meta.dirname, "src/node.ts"),
          browser: resolve(import.meta.dirname, "src/browser.ts"),
          "schema/index": resolve(import.meta.dirname, "src/schema/index.ts"),
          "worklet/index": resolve(import.meta.dirname, "src/worklet/index.ts"),
          "worklet/processor": resolve(import.meta.dirname, "src/worklet/processor.ts"),
        },
        formats: ["es"],
        fileName: (_format, entryName) => `${entryName}.js`,
      },
      rollupOptions: {
        external: ["zod", "@standard-schema/spec", /^node:/],
      },
    },
    test: {
      include: ["src/**/*.test.ts", "tests/integration/**/*.test.ts"],
      benchmark: {
        include: ["src/**/*.bench.ts", "tests/integration/**/*.bench.ts"],
      },
      coverage: {
        provider: "v8",
        include: ["src/**/*.ts"],
        exclude: [
          "src/**/*.d.ts",
          "src/**/index.ts",
          "src/browser.ts",
          "src/node.ts",
          "src/drivers/types.ts",
          "src/**/*.test.ts",
          "src/**/*.bench.ts",
        ],
        thresholds: {
          lines: 95,
          functions: 95,
          statements: 95,
          branches: 70,
        },
      },
    },
  };
});
