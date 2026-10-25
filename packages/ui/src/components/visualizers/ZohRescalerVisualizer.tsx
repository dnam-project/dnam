import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeZohPoleRescaling } from "./models.js";

export interface ZohRescalerVisualizerProps {
  readonly sampleRateHz?: number | undefined;
  readonly onSampleRateChange?: ((sampleRateHz: number) => void) | undefined;
}

const HOST_SAMPLE_RATES = [44_100, 48_000, 88_200, 96_000] as const;

export function ZohRescalerVisualizer({
  sampleRateHz = 96_000,
  onSampleRateChange,
}: ZohRescalerVisualizerProps) {
  const metrics = computeZohPoleRescaling(0.992, 1200.0, sampleRateHz);

  return (
    <Card data-testid="visualizer-zoh-rescaler">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Container: Continuous-Time ZOH Sample-Rate Rescaler</CardTitle>
          <CardDescription>
            Exact Pole/Delay Rescaling Across 44.1 / 48 / 88.2 / 96 kHz Host Rates
          </CardDescription>
        </div>
        <Badge variant="core2">f_c = {metrics.preservedCutoffHz.toFixed(0)} Hz Invariant</Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-1.5">
          {HOST_SAMPLE_RATES.map((rate) => (
            <button
              key={rate}
              type="button"
              onClick={() => onSampleRateChange?.(rate)}
              className={
                rate === sampleRateHz
                  ? "rounded border border-cyan-400 bg-cyan-500/20 px-2.5 py-1 font-mono text-xs font-semibold text-cyan-200"
                  : "rounded border border-slate-700 bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-300"
              }
            >
              {(rate / 1000).toFixed(1)} kHz
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Rescaled Pole Radius r(Fs)</div>
            <div className="mt-0.5 text-cyan-300">{metrics.rescaledRadius.toFixed(6)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Rescaled Angle θ(Fs)</div>
            <div className="mt-0.5 text-emerald-300">{metrics.rescaledAngleRad.toFixed(5)} rad</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
