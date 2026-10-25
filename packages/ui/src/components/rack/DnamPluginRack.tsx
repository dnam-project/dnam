import { Sliders } from "lucide-react";
import { useDnamAudioHost } from "../../host/context.js";
import { DiscreteSwitch } from "../controls/DiscreteSwitch.js";
import { RotaryKnob } from "../controls/RotaryKnob.js";
import { AccelerationBadge, StabilityAuditBadge } from "../controls/StatusBadges.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import type { DnamPluginRackModel } from "./models.js";
import { TriPhysicsTelemetry } from "./TriPhysicsTelemetry.js";

export interface DnamPluginRackProps {
  readonly model: DnamPluginRackModel;
}

export function DnamPluginRack({ model }: DnamPluginRackProps) {
  const { host, telemetry, setParameter, setSwitchIndex } = useDnamAudioHost();

  return (
    <div data-testid={`dnam-plugin-rack-${model.modelId}`} className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Sliders className="size-4 text-amber-400" />
              <CardTitle>{model.title}</CardTitle>
            </div>
            <CardDescription>{model.subtitle}</CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <AccelerationBadge backend={host.backend} />
            <StabilityAuditBadge
              maxRgLruRadius={telemetry.maxRgLruRadius}
              ufdnLyapunovMargin={telemetry.ufdnLyapunovMargin}
              mahalanobisR2={telemetry.mahalanobisR2}
            />
          </div>
        </CardHeader>

        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
            {model.parameters.map((param) => {
              const currentVal = host.getParameter(param.id) ?? param.defaultValue;
              return (
                <RotaryKnob
                  key={param.id}
                  id={param.id}
                  label={param.label}
                  value={currentVal}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  observedMin={param.observedMin}
                  observedMax={param.observedMax}
                  unit={param.unit}
                  domain={param.domain}
                  onChange={(nextValue) => {
                    setParameter(param.id, nextValue);
                  }}
                />
              );
            })}
          </div>

          {model.switches.length > 0 ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {model.switches.map((sw) => {
                const currentIdx = host.getSwitchIndex(sw.id) ?? sw.defaultIndex;
                return (
                  <DiscreteSwitch
                    key={sw.id}
                    id={sw.id}
                    label={sw.label}
                    options={sw.options}
                    activeIndex={currentIdx}
                    onChange={(nextIdx) => {
                      setSwitchIndex(sw.id, nextIdx);
                    }}
                  />
                );
              })}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <TriPhysicsTelemetry telemetry={telemetry} />
    </div>
  );
}
