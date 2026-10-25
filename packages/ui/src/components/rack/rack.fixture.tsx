import { useFixtureInput } from "react-cosmos/client";
import { DnamPluginRack, TriPhysicsTelemetry } from "./index.js";

function TelemetryPlayground() {
  const [dspLoadPercent] = useFixtureInput("SIMD DSP Load (%)", 8.4);
  const [mahalanobisR2] = useFixtureInput("Mahalanobis R²", 1.15);

  return (
    <TriPhysicsTelemetry
      telemetry={{
        backend: "arm-sme",
        sampleRateHz: 48_000,
        controlRateHz: 3_000,
        inputRmsDbL: -14.2,
        inputRmsDbR: -14.5,
        outputRmsDbL: -6.8,
        outputRmsDbR: -6.9,
        maxRgLruRadius: 0.9924,
        ufdnLyapunovMargin: 0.038,
        mahalanobisR2,
        activeTrustRegionAlpha: mahalanobisR2 > 4.0 ? 0.55 : 1.0,
        dspLoadPercent,
        bypassActive: false,
      }}
    />
  );
}

export default {
  "Soldano SLO-100 Plugin Rack": (
    <DnamPluginRack
      model={{
        modelId: "SLO100_Lead",
        title: "Soldano SLO-100 Overdrive",
        subtitle: "12-Layer PolyADAA + Radial-Polar RG-LRU + Modulated UFDN",
        parameters: [
          {
            id: "gain",
            label: "Preamp Gain",
            min: 0,
            max: 10,
            defaultValue: 6.5,
            observedMin: 1,
            observedMax: 9,
            domain: "core1",
          },
          {
            id: "tone",
            label: "Tone Stack",
            min: 0,
            max: 10,
            defaultValue: 5.0,
            domain: "control",
          },
          {
            id: "space",
            label: "Cabinet Space",
            min: 0,
            max: 10,
            defaultValue: 4.0,
            domain: "core2",
          },
        ],
        switches: [
          {
            id: "bright",
            label: "Bright Cap",
            defaultIndex: 0,
            options: [
              { index: 0, label: "Off", observed: true },
              { index: 1, label: "Warm", observed: true },
              { index: 2, label: "On", observed: false },
            ],
          },
        ],
      }}
    />
  ),
  "Tri-Physics Telemetry Meter": <TelemetryPlayground />,
};
