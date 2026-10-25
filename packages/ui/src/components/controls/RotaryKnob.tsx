import { Lock } from "lucide-react";
import { clampNumber, cn, formatParameterValue } from "../../lib/utils.js";
import type { TriPhysicsDomain } from "../../tokens.js";
import { domainAccentHex } from "../../tokens.js";
import {
  describeRotaryArc,
  normalizedToAngleDeg,
  polarToCartesian,
  valueToNormalized,
} from "./geometry.js";

export interface RotaryKnobProps {
  readonly id: string;
  readonly label: string;
  readonly value: number;
  readonly min?: number | undefined;
  readonly max?: number | undefined;
  readonly step?: number | undefined;
  readonly observedMin?: number | undefined;
  readonly observedMax?: number | undefined;
  readonly unit?: string | undefined;
  readonly domain?: TriPhysicsDomain | undefined;
  readonly onChange?: ((nextValue: number) => void) | undefined;
  readonly className?: string | undefined;
}

export function RotaryKnob({
  id,
  label,
  value,
  min = 0.0,
  max = 10.0,
  step = 0.1,
  observedMin,
  observedMax,
  unit,
  domain = "core1",
  onChange,
  className,
}: RotaryKnobProps) {
  const effObsMin = observedMin !== undefined ? clampNumber(observedMin, min, max) : min;
  const effObsMax = observedMax !== undefined ? clampNumber(observedMax, effObsMin, max) : max;
  const clampedValue = clampNumber(value, effObsMin, effObsMax);
  const isRangeClamped = effObsMin > min + 1e-6 || effObsMax < max - 1e-6;

  const normValue = valueToNormalized(clampedValue, min, max);
  const normObsMin = valueToNormalized(effObsMin, min, max);
  const normObsMax = valueToNormalized(effObsMax, min, max);

  const trackArc = describeRotaryArc(32, 32, 24, 0, 1);
  const observedArc = describeRotaryArc(32, 32, 24, normObsMin, normObsMax);
  const valueArc = describeRotaryArc(32, 32, 24, normObsMin, normValue);
  const pointer = polarToCartesian(32, 32, 18, normalizedToAngleDeg(normValue));
  const accentColor = domainAccentHex(domain);

  return (
    <div
      data-testid={`rotary-knob-${id}`}
      className={cn(
        "flex flex-col items-center rounded-lg border border-slate-800 bg-slate-950/70 p-2.5 text-center",
        className,
      )}
    >
      <div className="mb-1 flex w-full items-center justify-between text-xs font-medium text-slate-300">
        <span className="truncate">{label}</span>
        {isRangeClamped ? (
          <span
            title={`CL-12 Clamped [${effObsMin.toFixed(1)}, ${effObsMax.toFixed(1)}]`}
            className="inline-flex items-center gap-0.5 font-mono text-xs text-amber-400"
          >
            <Lock className="size-2.5" />
            CL-12
          </span>
        ) : null}
      </div>

      <svg
        width={64}
        height={64}
        viewBox="0 0 64 64"
        role="img"
        aria-label={`${label} rotary dial`}
        className="overflow-visible"
      >
        <path d={trackArc} fill="none" stroke="#1e293b" strokeWidth={5} strokeLinecap="round" />
        <path d={observedArc} fill="none" stroke="#334155" strokeWidth={5} strokeLinecap="round" />
        <path d={valueArc} fill="none" stroke={accentColor} strokeWidth={5} strokeLinecap="round" />
        <circle cx={32} cy={32} r={14} fill="#0f172a" stroke="#334155" strokeWidth={1.5} />
        <line
          x1={32}
          y1={32}
          x2={pointer.x}
          y2={pointer.y}
          stroke={accentColor}
          strokeWidth={2.2}
          strokeLinecap="round"
        />
      </svg>

      <div className="mt-1 font-mono text-xs font-semibold text-slate-100">
        {formatParameterValue(clampedValue, 2, unit)}
      </div>

      <input
        type="range"
        aria-label={label}
        min={effObsMin}
        max={effObsMax}
        step={step}
        value={clampedValue}
        onChange={(event) => {
          const parsed = Number.parseFloat(event.currentTarget.value);
          const bounded = clampNumber(parsed, effObsMin, effObsMax);
          onChange?.(bounded);
        }}
        className="mt-1.5 h-1.5 w-full cursor-pointer accent-cyan-400"
      />
    </div>
  );
}
