import { useFixtureInput, useFixtureSelect } from "react-cosmos/client";
import { AccelerationBadge, DiscreteSwitch, RotaryKnob, StabilityAuditBadge } from "./index.js";

function RotaryKnobPlayground() {
  const [value, setValue] = useFixtureInput("Value", 6.5);
  const [observedMin] = useFixtureInput("Observed Min (CL-12)", 1.5);
  const [observedMax] = useFixtureInput("Observed Max (CL-12)", 8.5);
  const [domain] = useFixtureSelect("Domain", {
    options: ["core1", "core2", "control"],
    defaultValue: "core1",
  });

  return (
    <div className="max-w-xs">
      <RotaryKnob
        id="gain"
        label="Preamp Gain"
        value={value}
        min={0}
        max={10}
        observedMin={observedMin}
        observedMax={observedMax}
        unit="dB"
        domain={domain}
        onChange={setValue}
      />
    </div>
  );
}

function DiscreteSwitchAndBadgesPlayground() {
  const [activeIndex, setActiveIndex] = useFixtureInput("Active Switch Index", 0);
  const [backend] = useFixtureSelect("SIMD Backend", {
    options: ["arm-sme", "arm64-neon", "x86-avx2-fma", "wasm-simd128", "scalar-reference"],
    defaultValue: "arm-sme",
  });
  const [mahalanobisR2] = useFixtureInput("Mahalanobis R²", 0.85);

  return (
    <div className="flex max-w-md flex-col gap-4">
      <DiscreteSwitch
        id="voice"
        label="Voicing Switch"
        activeIndex={activeIndex}
        onChange={setActiveIndex}
        options={[
          { index: 0, label: "Clean", observed: true },
          { index: 1, label: "Crunch", observed: true },
          { index: 2, label: "Scorch", observed: false },
        ]}
      />
      <div className="flex flex-wrap gap-2">
        <AccelerationBadge backend={backend} />
        <StabilityAuditBadge
          maxRgLruRadius={0.991}
          ufdnLyapunovMargin={0.04}
          mahalanobisR2={mahalanobisR2}
        />
      </div>
    </div>
  );
}

export default {
  "Rotary Knob (CL-12 Range Clamped)": <RotaryKnobPlayground />,
  "Discrete Switch (CL-13) & Status Badges": <DiscreteSwitchAndBadgesPlayground />,
};
