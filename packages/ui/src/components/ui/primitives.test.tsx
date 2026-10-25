import { describe, expect, it } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations } from "../../../tests/setup.js";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "./index.js";

describe("shadcn/ui Base UI primitives", () => {
  it("renders Button, Badge, Card, and Progress with zero axe violations", async () => {
    const screen = await render(
      <div>
        <Button variant="default" size="sm">
          Primary
        </Button>
        <Button variant="secondary" render={<a href="#docs">Docs</a>} nativeButton={false} />
        <Badge variant="core1">Core 1</Badge>
        <Badge variant="safe">Safe</Badge>
        <Card>
          <CardHeader>
            <CardTitle>Title</CardTitle>
            <CardDescription>Desc</CardDescription>
          </CardHeader>
          <CardContent>
            <Progress aria-label="Determinate progress" value={65} />
            <Progress aria-label="Indeterminate progress" value={null} />
          </CardContent>
          <CardFooter>Footer</CardFooter>
        </Card>
      </div>,
    );

    await expect.element(page.getByRole("button", { name: "Primary" })).toBeVisible();
    await expect.element(page.getByRole("button", { name: "Docs" })).toBeVisible();
    await expectZeroAxeViolations(screen.container);
  });

  it("switches Tabs interactively and matches ARIA inline snapshots", async () => {
    const screen = await render(
      <Tabs defaultValue="rack">
        <TabsList aria-label="Workspace views">
          <TabsTrigger value="rack">Rack</TabsTrigger>
          <TabsTrigger value="viz">Visualizers</TabsTrigger>
        </TabsList>
        <TabsContent value="rack">Rack Content</TabsContent>
        <TabsContent value="viz">Viz Content</TabsContent>
      </Tabs>,
    );

    await expect
      .element(page.getByRole("tablist", { name: "Workspace views" }))
      .toMatchAriaInlineSnapshot(`
      - tablist "Workspace views":
        - tab "Rack" [selected]
        - tab "Visualizers"
    `);

    await userEvent.click(page.getByRole("tab", { name: "Visualizers" }));
    await expect.element(page.getByText("Viz Content")).toBeVisible();

    await expect
      .element(page.getByRole("tablist", { name: "Workspace views" }))
      .toMatchAriaInlineSnapshot(`
      - tablist "Workspace views":
        - tab "Rack"
        - tab "Visualizers" [selected]
    `);

    await expectZeroAxeViolations(screen.container);
  });
});
