import { Sliders } from "lucide-react";
import { useDnamAudioHost } from "../../host/context.js";
import { DNAM_MODEL_TIER_LABELS, DNAM_MODEL_TIERS, tierOrderIndex } from "../../host/types.js";
import { cn } from "../../lib/utils.js";
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
  const {
    host,
    telemetry,
    setParameter,
    setSwitchIndex,
    setTier,
    setOfflineBounce,
    setMaximizeTierOnBounce,
  } = useDnamAudioHost();
  const activeTier = telemetry.activeTier ?? host.getTier?.() ?? "full";
  const maxSupportedTier = telemetry.maxSupportedTier ?? host.getMaxSupportedTier?.() ?? "ultra";
  const offlineBounce = telemetry.offlineBounce ?? host.getOfflineBounce?.() ?? false;
  const maximizeTierOnBounce =
    telemetry.maximizeTierOnBounce ?? host.getMaximizeTierOnBounce?.() ?? true;

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
            <fieldset
              aria-label="Slimmable Model Tier"
              className="inline-flex items-center gap-1 rounded border border-slate-800 bg-slate-950/70 p-0.5 text-xs"
            >
              {DNAM_MODEL_TIERS.map((tier) => {
                const isSelected = activeTier === tier;
                const isDisabled = tierOrderIndex(tier) > tierOrderIndex(maxSupportedTier);
                return (
                  <button
                    key={tier}
                    type="button"
                    disabled={isDisabled}
                    aria-pressed={isSelected}
                    onClick={() => {
                      setTier(tier);
                    }}
                    className={cn(
                      "rounded px-2 py-0.5 font-mono transition",
                      isSelected
                        ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/50"
                        : "text-slate-400 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40",
                    )}
                  >
                    {DNAM_MODEL_TIER_LABELS[tier]}
                  </button>
                );
              })}
            </fieldset>
            <button
              type="button"
              aria-pressed={offlineBounce}
              onClick={() => {
                setOfflineBounce(!offlineBounce);
              }}
              className={cn(
                "rounded border px-2 py-0.5 font-mono text-xs transition",
                offlineBounce
                  ? "border-amber-400/60 bg-amber-500/20 text-amber-200"
                  : "border-slate-800 bg-slate-950/70 text-slate-400 hover:text-slate-200",
              )}
            >
              Offline Bounce (4x)
            </button>
            <button
              type="button"
              aria-pressed={maximizeTierOnBounce}
              onClick={() => {
                setMaximizeTierOnBounce(!maximizeTierOnBounce);
              }}
              className={cn(
                "rounded border px-2 py-0.5 font-mono text-xs transition",
                maximizeTierOnBounce
                  ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                  : "border-slate-800 bg-slate-950/70 text-slate-400 hover:text-slate-200",
              )}
            >
              Maximize Tier on Bounce
            </button>
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
