import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => ({
  plugins: [tailwindcss(), react()],
  ...(mode === "lib"
    ? {
        build: {
          emptyOutDir: true,
          outDir: "dist",
          target: "es2024",
          minify: false,
          sourcemap: true,
          lib: {
            entry: {
              index: resolve(import.meta.dirname, "src/index.ts"),
              tokens: resolve(import.meta.dirname, "src/tokens.ts"),
              "host/index": resolve(import.meta.dirname, "src/host/index.ts"),
              "components/visualizers/index": resolve(
                import.meta.dirname,
                "src/components/visualizers/index.ts",
              ),
            },
            formats: ["es"],
            fileName: (_format, entryName) => `${entryName}.js`,
          },
          rollupOptions: {
            external: [
              "react",
              "react/jsx-runtime",
              "react-dom",
              "lucide-react",
              "clsx",
              "tailwind-merge",
              "class-variance-authority",
              /^@base-ui\/react/,
              "@dnam-project/engine",
            ],
          },
        },
      }
    : {}),
  test: {
    include: [
      "src/**/*.test.ts",
      "src/**/*.test.tsx",
      "tests/integration/**/*.test.ts",
      "tests/integration/**/*.test.tsx",
    ],
    benchmark: {
      include: ["src/**/*.bench.ts", "tests/integration/**/*.bench.ts"],
    },
    setupFiles: ["./tests/setup.ts"],
    browser: {
      enabled: true,
      headless: true,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
    },
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts", "src/**/*.tsx"],
      exclude: [
        "src/**/*.d.ts",
        "src/**/index.ts",
        "src/host/types.ts",
        "src/components/rack/models.ts",
        "src/components/workflows/models.ts",
        "src/cosmos.decorator.tsx",
        "src/**/*.fixture.tsx",
        "src/**/*.test.ts",
        "src/**/*.test.tsx",
        "src/**/*.bench.ts",
      ],
      thresholds: {
        lines: 95,
        functions: 95,
        statements: 95,
        branches: 85,
      },
    },
  },
}));
