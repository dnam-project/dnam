import { describe, expect, it, vi } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { DnamAudioHostProvider, useDnamAudioHost, WebAudioWorkletHost } from "./index.js";

function HostProbe() {
  const { host, telemetry, setParameter, setSwitchIndex } = useDnamAudioHost();
  return (
    <div>
      <span data-testid="host-kind">{host.kind}</span>
      <span data-testid="host-sr">{telemetry.sampleRateHz}</span>
      <span data-testid="host-bypass">{telemetry.bypassActive ? "bypassed" : "active"}</span>
      <button
        type="button"
        onClick={() => {
          setParameter("drive", 7.25);
          setSwitchIndex("voice", 1);
        }}
      >
        Mutate Host
      </button>
    </div>
  );
}

describe("WebAudioWorkletHost and DnamAudioHostProvider", () => {
  it("manages parameters, switches, telemetry snapshots, and listener unsubscription", () => {
    const onParam = vi.fn();
    const onSwitch = vi.fn();
    const host = new WebAudioWorkletHost({
      initialParameters: { drive: 5.0 },
      initialSwitches: { voice: 1 },
      onParameterChange: onParam,
      onSwitchChange: onSwitch,
    });

    expect(host.getParameter("drive")).toBe(5.0);
    expect(host.getParameter("unknown")).toBe(0.5);
    expect(host.getSwitchIndex("voice")).toBe(1);
    expect(host.getSwitchIndex("unknown")).toBe(0);

    const snapshots: number[] = [];
    const unsubscribe = host.subscribe((snap) => {
      snapshots.push(snap.mahalanobisR2);
    });

    host.setParameter("drive", 8.0);
    host.setParameter("nanParam", Number.NaN);
    host.setSwitchIndex("voice", 2);
    expect(onParam).toHaveBeenCalledWith("drive", 8.0);
    expect(onSwitch).toHaveBeenCalledWith("voice", 2);
    expect(snapshots.length).toBeGreaterThanOrEqual(2);

    unsubscribe();
    const prevCount = snapshots.length;
    host.updateTelemetry({ bypassActive: true });
    expect(snapshots.length).toBe(prevCount);
    expect(host.getTelemetry().bypassActive).toBe(true);
  });

  it("mounts DnamAudioHostProvider and reacts to live telemetry updates", async () => {
    const customHost = new WebAudioWorkletHost({ sampleRateHz: 96_000 });
    await render(
      <DnamAudioHostProvider host={customHost}>
        <HostProbe />
      </DnamAudioHostProvider>,
    );

    await expect.element(page.getByTestId("host-sr")).toHaveTextContent("96000");
    await expect.element(page.getByTestId("host-bypass")).toHaveTextContent("active");

    await userEvent.click(page.getByRole("button", { name: "Mutate Host" }));
    expect(customHost.getParameter("drive")).toBe(7.25);
    expect(customHost.getSwitchIndex("voice")).toBe(1);

    customHost.updateTelemetry({ bypassActive: true });
    await expect.element(page.getByTestId("host-bypass")).toHaveTextContent("bypassed");
  });

  it("falls back to default WebAudioWorkletHost when provider omits host prop", async () => {
    await render(
      <DnamAudioHostProvider>
        <HostProbe />
      </DnamAudioHostProvider>,
    );
    await expect.element(page.getByTestId("host-sr")).toHaveTextContent("48000");
  });
});
