import { resolve } from "node:path";
import { playwright } from "@vitest/browser-playwright";
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
            "web-audio-processor": resolve(import.meta.dirname, "src/web-audio/processor.ts"),
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
    server: {
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
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
          "parameters/index": resolve(import.meta.dirname, "src/parameters/index.ts"),
          "tempo/index": resolve(import.meta.dirname, "src/tempo/index.ts"),
          "schema/index": resolve(import.meta.dirname, "src/schema/index.ts"),
          "provenance/index": resolve(import.meta.dirname, "src/provenance/index.ts"),
          "dsp/index": resolve(import.meta.dirname, "src/dsp/index.ts"),
          "web-audio/index": resolve(import.meta.dirname, "src/web-audio/index.ts"),
          "web-audio/processor": resolve(import.meta.dirname, "src/web-audio/processor.ts"),
        },
        formats: ["es"],
        fileName: (_format, entryName) => `${entryName}.js`,
      },
      rollupOptions: {
        external: [/^node:/],
      },
    },
    test: {
      projects: [
        {
          extends: true,
          test: {
            name: "node",
            environment: "node",
            include: [
              "src/**/*.test.ts",
              "tests/integration/**/*.test.ts",
              "tests/profile/**/*.test.ts",
            ],
            exclude: ["src/**/*.browser.test.ts", "tests/**/*.browser.test.ts"],
            benchmark: {
              include: ["src/**/*.bench.ts", "tests/integration/**/*.bench.ts"],
              exclude: ["src/**/*.browser.bench.ts", "tests/**/*.browser.bench.ts"],
            },
          },
        },
        {
          extends: true,
          test: {
            name: "chromium",
            include: ["src/**/*.browser.test.ts", "tests/**/*.browser.test.ts"],
            benchmark: {
              include: ["src/**/*.browser.bench.ts", "tests/**/*.browser.bench.ts"],
            },
            browser: {
              enabled: true,
              headless: true,
              provider: playwright(),
              instances: [{ browser: "chromium" }],
            },
          },
        },
      ],
      coverage: {
        provider: "v8",
        include: ["src/**/*.ts"],
        exclude: [
          "src/**/*.d.ts",
          "src/**/index.ts",
          "src/public-exports.ts",
          "src/browser.ts",
          "src/node.ts",
          "src/internal/native/driver.ts",
          "src/web-audio/protocol.ts",
          "src/**/*.test.ts",
          "src/**/*.browser.test.ts",
          "src/**/*.bench.ts",
        ],
        thresholds: {
          lines: 95,
          functions: 95,
          statements: 95,
          branches: 75,
        },
      },
    },
  };
});
