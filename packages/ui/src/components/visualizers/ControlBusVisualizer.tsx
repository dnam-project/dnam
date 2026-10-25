import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeLobattoDegree } from "./models.js";

export interface ControlBusVisualizerProps {
  readonly nodes?: number | undefined;
  readonly onNodesChange?: ((nodes: number) => void) | undefined;
}

export function ControlBusVisualizer({ nodes = 4, onNodesChange }: ControlBusVisualizerProps) {
  const metrics = computeLobattoDegree(nodes);

  return (
    <Card data-testid="visualizer-control-bus">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Control Bus: Lobatto HDMR-FM & 3 kHz Order-Statistic LFO</CardTitle>
          <CardDescription>
            Cardinality-Bounded Newton/Lobatto Basis K_p = min(U_p - 1, 3) & Rank-4 FM
          </CardDescription>
        </div>
        <Badge variant="control">K_p = {metrics.degreeKp}</Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          <span>Unique Captured Nodes (U_p): {metrics.uniqueNodes}</span>
          <input
            type="range"
            aria-label="Lobatto Unique Nodes"
            min={1}
            max={6}
            step={1}
            value={nodes}
            onChange={(e) => onNodesChange?.(Number.parseInt(e.currentTarget.value, 10))}
            className="accent-violet-400"
          />
        </label>

        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Unique Nodes U_p</div>
            <div className="mt-0.5 text-violet-300">{metrics.uniqueNodes}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Polynomial Degree K_p</div>
            <div className="mt-0.5 text-cyan-300">{metrics.degreeKp}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">FM Interaction Rank</div>
            <div className="mt-0.5 text-emerald-300">R = {metrics.rankFm}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
