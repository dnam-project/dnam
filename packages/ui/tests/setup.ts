import "../src/styles.css";
import axe from "axe-core";
import { expect } from "vitest";

if (typeof document !== "undefined") {
  document.documentElement.classList.add("dark", "bg-slate-950", "text-slate-100");
}

export async function expectZeroAxeViolations(root: Element = document.body): Promise<void> {
  const results = await axe.run(root, {
    rules: {
      region: { enabled: false },
    },
  });
  const violations = results.violations.map((v) => ({
    id: v.id,
    impact: v.impact,
    description: v.description,
    nodes: v.nodes.map((n) => n.html),
  }));
  expect(violations).toEqual([]);
}

export function setRangeInputValue(input: HTMLInputElement, value: number | string): void {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  setter?.call(input, String(value));
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}
