import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computePolyAdaaMetrics } from "./models.js";

export interface PolyAdaaVisualizerProps {
  readonly drive?: number | undefined;
  readonly freqHz?: number | undefined;
  readonly onDriveChange?: ((drive: number) => void) | undefined;
  readonly onFreqChange?: ((freqHz: number) => void) | undefined;
}

export function PolyAdaaVisualizer({
  drive = 3.5,
  freqHz = 2400,
  onDriveChange,
  onFreqChange,
}: PolyAdaaVisualizerProps) {
  const metrics = computePolyAdaaMetrics(drive, freqHz);

  return (
    <Card data-testid="visualizer-poly-adaa">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Core 1A: 1× B-Spline (0.875, 0.125) Division-Free PolyADAA</CardTitle>
          <CardDescription>
            Zero-FDIV Polynomial Antiderivative Anti-Aliasing vs. Post-ADAA High-Pass Extrapolation
          </CardDescription>
        </div>
        <Badge variant="core1">-{metrics.bsplineSuppressionDb.toFixed(2)} dB Aliasing</Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1 text-xs text-slate-300">
            <span>Drive Gain: {drive.toFixed(1)}×</span>
            <input
              type="range"
              aria-label="PolyADAA Drive"
              min={0.5}
              max={10.0}
              step={0.5}
              value={drive}
              onChange={(e) => onDriveChange?.(Number.parseFloat(e.currentTarget.value))}
              className="accent-amber-400"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs text-slate-300">
            <span>Excitation Frequency: {freqHz.toFixed(0)} Hz</span>
            <input
              type="range"
              aria-label="PolyADAA Frequency"
              min={200}
              max={12000}
              step={200}
              value={freqHz}
              onChange={(e) => onFreqChange?.(Number.parseFloat(e.currentTarget.value))}
              className="accent-amber-400"
            />
          </label>
        </div>

        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">B-Spline Weights</div>
            <div className="mt-0.5 text-amber-300">
              ({metrics.samplePairWeights[0]}, {metrics.samplePairWeights[1]})
            </div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Aliasing Suppression</div>
            <div className="mt-0.5 text-emerald-300">
              {metrics.bsplineSuppressionDb.toFixed(2)} dB
            </div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Avoided HP Spike</div>
            <div className="mt-0.5 text-red-300">
              +{metrics.postExtrapolationSpikeDb.toFixed(2)} dB
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
