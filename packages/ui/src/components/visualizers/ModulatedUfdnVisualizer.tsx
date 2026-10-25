import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeUfdnCrossfadeWeights } from "./models.js";

export interface ModulatedUfdnVisualizerProps {
  readonly sampleIndex?: number | undefined;
  readonly onSampleIndexChange?: ((sampleIndex: number) => void) | undefined;
}

export function ModulatedUfdnVisualizer({
  sampleIndex = 64,
  onSampleIndexChange,
}: ModulatedUfdnVisualizerProps) {
  const metrics = computeUfdnCrossfadeWeights(sampleIndex);

  return (
    <Card data-testid="visualizer-modulated-ufdn">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Core 2: Modulated Unitary Feedback Delay Network (M-UFDN)</CardTitle>
          <CardDescription>
            128-Sample Raised-Cosine Switch Crossfade & Unity-Slope Lyapunov Loop Law
          </CardDescription>
        </div>
        <Badge variant="core2">
          {"S'(0) = "}
          {metrics.unitySlopeAtOrigin.toFixed(4)}
        </Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          <span>Switch Crossfade Sample Index: {sampleIndex} / 128</span>
          <input
            type="range"
            aria-label="UFDN Crossfade Sample"
            min={0}
            max={128}
            step={4}
            value={sampleIndex}
            onChange={(e) => onSampleIndexChange?.(Number.parseInt(e.currentTarget.value, 10))}
            className="accent-cyan-400"
          />
        </label>

        <div className="grid grid-cols-2 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Pointer A Weight</div>
            <div className="mt-0.5 text-cyan-300">{metrics.fadeOutWeight.toFixed(4)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Pointer B Weight</div>
            <div className="mt-0.5 text-emerald-300">{metrics.fadeInWeight.toFixed(4)}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
