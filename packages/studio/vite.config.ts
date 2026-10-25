import { resolve } from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

const host = process.env["TAURI_DEV_HOST"];

export default defineConfig(({ mode }) => {
  const isTauriBuild = Boolean(process.env["TAURI_ENV_PLATFORM"]);
  const isTauriDebug = Boolean(process.env["TAURI_ENV_DEBUG"]);
  const buildTarget =
    process.env["TAURI_ENV_PLATFORM"] === "windows"
      ? "chrome105"
      : isTauriBuild
        ? "safari13"
        : "es2024";

  return {
    // Prevent Vite from obscuring Rust compiler and Tauri runtime errors
    clearScreen: false,
    plugins: [tailwindcss(), react()],
    envPrefix: ["VITE_", "TAURI_ENV_*"],
    server: {
      // Tauri expects a fixed port; fail if that port is not available
      port: 1420,
      strictPort: true,
      // If the host Tauri expects is set, use it
      host: host || false,
      ...(host
        ? {
            hmr: {
              protocol: "ws",
              host,
              port: 1421,
            },
          }
        : {}),
      watch: {
        // Tell Vite to ignore watching Rust sources and Cargo build artifacts
        ignored: ["**/rust/**", "**/src-tauri/**", "**/target/**"],
      },
    },
    ...(mode === "lib"
      ? {
          build: {
            emptyOutDir: true,
            outDir: "dist",
            target: buildTarget,
            minify: isTauriBuild ? (!isTauriDebug ? "esbuild" : false) : false,
            sourcemap: isTauriBuild ? isTauriDebug : true,
            lib: {
              entry: {
                index: resolve(import.meta.dirname, "src/index.ts"),
                "ui/index": resolve(import.meta.dirname, "src/ui/index.ts"),
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
                "@dnam-project/ui",
                "@dnam-project/engine",
                /^node:/,
              ],
            },
          },
        }
      : {}),
    test: {
      projects: [
        {
          extends: true,
          test: {
            name: "node",
            environment: "node",
            include: ["src/binary.test.ts"],
          },
        },
        {
          extends: true,
          test: {
            name: "chromium",
            include: [
              "src/ipc.test.ts",
              "src/host.test.ts",
              "src/ui/**/*.test.tsx",
              "tests/integration/**/*.test.ts",
              "tests/integration/**/*.test.tsx",
            ],
            setupFiles: ["./tests/setup.ts"],
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
        include: ["src/**/*.ts", "src/**/*.tsx"],
        exclude: ["src/**/*.d.ts", "src/**/index.ts", "src/**/*.test.ts", "src/**/*.test.tsx"],
        thresholds: {
          lines: 95,
          functions: 95,
          statements: 95,
          branches: 80,
        },
      },
    },
  };
});
