import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeTrustRegionDamping } from "./models.js";

export interface ClampAuditVisualizerProps {
  readonly r2?: number | undefined;
  readonly onR2Change?: ((r2: number) => void) | undefined;
}

export function ClampAuditVisualizer({ r2 = 1.2, onR2Change }: ClampAuditVisualizerProps) {
  const metrics = computeTrustRegionDamping(r2, 4.0);

  return (
    <Card data-testid="visualizer-clamp-audit">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Safety: Non-Stop 3-Phase Clamping & CL-01..CL-20 Registry</CardTitle>
          <CardDescription>
            Phase 1 (Range/Switch/Poles), Phase 2 (Mahalanobis Trust Region), Phase 3 (Conditional
            Corners)
          </CardDescription>
        </div>
        <Badge variant={metrics.isOutOfDistribution ? "warn" : "safe"}>
          α_trust = {metrics.alpha.toFixed(3)}
        </Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          <span>Mahalanobis Control Distance R²(u): {r2.toFixed(2)}</span>
          <input
            type="range"
            aria-label="Mahalanobis R2"
            min={0.0}
            max={8.0}
            step={0.2}
            value={r2}
            onChange={(e) => onR2Change?.(Number.parseFloat(e.currentTarget.value))}
            className="accent-violet-400"
          />
        </label>

        <div className="grid grid-cols-2 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Mahalanobis R²</div>
            <div className="mt-0.5 text-violet-300">{metrics.mahalanobisR2.toFixed(2)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Interaction Damping α(u)</div>
            <div className="mt-0.5 text-emerald-300">{metrics.alpha.toFixed(4)}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
