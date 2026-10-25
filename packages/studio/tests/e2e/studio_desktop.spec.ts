import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.describe("DNAM Desktop Studio End-to-End (Playwright E2E)", () => {
  test("boots full Studio index.html and completes Rack, Capture, 4-Stage Trainer, Visualizers, and WCAG 2.1 AA audit", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page).toHaveTitle("DNAM Studio — Tri-Physics Neural Audio Modeler");
    await expect(page.getByRole("heading", { name: "DNAM Desktop Studio" })).toBeVisible();

    // Verify Tailwind CSS v4 @source compilation across @dnam-project/ui and @dnam-project/studio
    const layoutStyles = await page.evaluate(() => {
      const root = document.querySelector('[data-testid="dnam-studio-app"]');
      const iconSvg = document.querySelector('[data-testid="studio-open-dialog-btn"] svg');
      return {
        rootPadding: root ? getComputedStyle(root).padding : null,
        iconWidth: iconSvg ? iconSvg.getBoundingClientRect().width : null,
      };
    });
    expect(layoutStyles.rootPadding).toBe("24px");
    expect(layoutStyles.iconWidth).toBe(14);

    // 1. Desktop header actions (Web mode fallback)
    await page.getByTestId("studio-open-dialog-btn").click();
    await expect(
      page.getByText("dialog: Opened native .dnam / .dnam-ckpt container picker"),
    ).toBeVisible();

    await page.getByTestId("studio-reveal-export-btn").click();
    await expect(
      page.getByText("opener: Revealed dist/reference_tri_physics_mk1.dnam in Finder/Explorer"),
    ).toBeVisible();

    await page.getByTestId("studio-open-report-btn").click();
    await expect(
      page.getByText("opener: Opened models/reference_tri_physics_mk1/reports/telemetry.html"),
    ).toBeVisible();

    // 2. Rack Audition tab: adjust rotary slider and discrete switch
    const driveSlider = page.locator('input[aria-label="Input Drive"]');
    await driveSlider.fill("8.2");
    await page.getByRole("button", { name: "Crunch" }).click();
    await expect(page.getByRole("button", { name: "Crunch" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    // 3. Capture Schedule workflow: record remaining stem
    await page.getByRole("tab", { name: "Capture Schedule" }).click();
    await expect(page.getByText("1 / 2 Stems Captured")).toBeVisible();

    await page.getByRole("button", { name: "Record Stem" }).click();
    await expect(page.getByText("2 / 2 Stems Captured")).toBeVisible();
    await expect(
      page.getByText(/notification: Captured and aligned stem stem_axis_drive_max/),
    ).toBeVisible();

    // 4. 4-Stage Differentiable Trainer workflow
    await page.getByRole("tab", { name: "Python Trainer (dnam-train)" }).click();
    await page.getByRole("button", { name: "Run dnam-train" }).click();
    await expect(
      page.getByText(
        "notification: Stage 4 Kronecker .dnam-ckpt Export & CL-01..CL-20 Audit completed",
      ),
    ).toBeVisible();

    // 5. Interactive Tri-Physics Visualizers workflow
    await page.getByRole("tab", { name: "Tri-Physics Visualizers" }).click();
    await expect(page.getByRole("slider", { name: "PolyADAA Drive" })).toBeVisible();
    await expect(page.getByRole("slider", { name: "RG-LRU Blend" })).toBeVisible();
    await expect(page.getByTestId("visualizer-clamp-audit")).toBeVisible();
    await page.getByRole("button", { name: "48.0 kHz" }).click();

    // 6. Full-page WCAG 2.1 AA accessibility audit via @axe-core/playwright
    const a11yResults = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(a11yResults.violations).toEqual([]);
  });

  test("executes Tauri v2 native IPC, Tier 1 plugins, and deep-link routing from boot", async ({
    page,
  }) => {
    await page.addInitScript(() => {
      const invoked: string[] = [];
      let deepLinkCallback:
        | ((event: { payload: { kind: string; target: string } }) => void)
        | null = null;

      Object.assign(window, {
        isTauri: true,
        __DNAM_E2E_INVOKED__: invoked,
        __DNAM_TRIGGER_DEEP_LINK__: (kind: string, target: string) => {
          deepLinkCallback?.({ payload: { kind, target } });
        },
        __TAURI_EVENT_PLUGIN_INTERNALS__: {
          unregisterListener: () => undefined,
        },
        __TAURI_INTERNALS__: {
          transformCallback: (cb: typeof deepLinkCallback) => {
            if (typeof cb === "function") {
              deepLinkCallback = cb;
            }
            return 1;
          },
          unregisterCallback: () => undefined,
          invoke: async (cmd: string) => {
            invoked.push(cmd);
            if (cmd === "plugin:dialog|open") {
              return "/Users/studio/SLO100_Ultra.dnam";
            }
            if (cmd === "prepare_studio_export") {
              return "/tmp/dnam-studio/reference_tri_physics_mk1.dnam";
            }
            if (cmd === "prepare_studio_report") {
              return "/tmp/dnam-studio/telemetry_audit_report.html";
            }
            if (cmd === "check_studio_update") {
              return "v0.1.0 (up to date)";
            }
            return null;
          },
        },
      });
    });

    await page.goto("/");

    await page.getByTestId("studio-open-dialog-btn").click();
    await expect(page.getByText("dialog: Selected /Users/studio/SLO100_Ultra.dnam")).toBeVisible();

    await page.getByTestId("studio-reveal-export-btn").click();
    await expect(
      page.getByText("opener: Revealed /tmp/dnam-studio/reference_tri_physics_mk1.dnam in Finder"),
    ).toBeVisible();

    await page.getByTestId("studio-open-report-btn").click();
    await expect(
      page.getByText("opener: Opened /tmp/dnam-studio/telemetry_audit_report.html"),
    ).toBeVisible();

    await page.evaluate(() => {
      const win = window as unknown as {
        __DNAM_TRIGGER_DEEP_LINK__?: (kind: string, target: string) => void;
      };
      win.__DNAM_TRIGGER_DEEP_LINK__?.("model", "b7k_ultra");
    });
    await expect(page.getByText("deep-link: Routed model → b7k_ultra")).toBeVisible();

    const invokedCommands = await page.evaluate(() => {
      return (window as unknown as { __DNAM_E2E_INVOKED__: string[] }).__DNAM_E2E_INVOKED__;
    });
    expect(invokedCommands).toContain("studio_ipc");
    expect(invokedCommands).toContain("check_studio_update");
    expect(invokedCommands).toContain("plugin:dialog|open");
    expect(invokedCommands).toContain("prepare_studio_export");
    expect(invokedCommands).toContain("plugin:opener|reveal_item_in_dir");
    expect(invokedCommands).toContain("prepare_studio_report");
  });
});
