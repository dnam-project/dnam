import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeSurrogateDelayBasin } from "./models.js";

export interface KroneckerResumeVisualizerProps {
  readonly lagError?: number | undefined;
  readonly onLagErrorChange?: ((lagError: number) => void) | undefined;
}

export function KroneckerResumeVisualizer({
  lagError = 12,
  onLagErrorChange,
}: KroneckerResumeVisualizerProps) {
  const metrics = computeSurrogateDelayBasin(lagError);

  return (
    <Card data-testid="visualizer-kronecker-resume">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Trainer: Surrogate Delay & Kronecker .dnam-ckpt Expansion</CardTitle>
          <CardDescription>
            Envelope-Guided Convex Delay Basin & Rank-1 Sufficient-Statistic Stem Resume
          </CardDescription>
        </div>
        <Badge variant="safe">Old-Stem Drift: {metrics.oldStemDriftDb.toFixed(2)} dB</Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          <span>Delay Alignment Error (Δm): {lagError} samples</span>
          <input
            type="range"
            aria-label="Delay Lag Error"
            min={-32}
            max={32}
            step={1}
            value={lagError}
            onChange={(e) => onLagErrorChange?.(Number.parseInt(e.currentTarget.value, 10))}
            className="accent-emerald-400"
          />
        </label>

        <div className="grid grid-cols-2 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Surrogate Envelope Loss</div>
            <div className="mt-0.5 text-emerald-300">{metrics.envelopeLoss.toFixed(5)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Raw Waveform Loss</div>
            <div className="mt-0.5 text-amber-300">{metrics.waveformLoss.toFixed(5)}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
