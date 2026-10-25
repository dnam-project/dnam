import { Flame, Play } from "lucide-react";
import { Badge } from "../ui/badge.js";
import { Button } from "../ui/button.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { Progress } from "../ui/progress.js";
import type { TrainingTimelineView } from "./models.js";

export interface TrainingTimelinePanelProps {
  readonly timeline: TrainingTimelineView;
  readonly onStartTraining?: (() => void) | undefined;
}

export function TrainingTimelinePanel({ timeline, onStartTraining }: TrainingTimelinePanelProps) {
  const latest = timeline.history[timeline.history.length - 1];
  const progressPct =
    latest && latest.totalEpochs > 0
      ? (latest.epoch / latest.totalEpochs) * 100
      : timeline.status === "completed"
        ? 100
        : 0;

  return (
    <Card data-testid={`training-timeline-panel-${timeline.modelId}`}>
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Flame className="size-4 text-amber-400" />
            <CardTitle>4-Stage Differentiable Trainer ({timeline.modelId})</CardTitle>
          </div>
          <CardDescription>
            Non-Stop 3-Phase Control Clamping + Kronecker .dnam-ckpt Solver (
            {timeline.backendDevice})
          </CardDescription>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="control">{timeline.clampedParameterCount} Auto-Clamped Controls</Badge>
          {onStartTraining ? (
            <Button size="sm" onClick={onStartTraining}>
              <Play data-icon="inline-start" />
              Run dnam-train
            </Button>
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between font-mono text-xs text-slate-300">
          <span>Status: {timeline.status.toUpperCase()}</span>
          {latest ? (
            <span>
              Stage: {latest.stage} (Epoch {latest.epoch}/{latest.totalEpochs})
            </span>
          ) : (
            <span>Ready</span>
          )}
        </div>

        <Progress value={progressPct} indicatorClassName="bg-violet-400" />

        {latest ? (
          <div className="grid grid-cols-2 gap-2 font-mono text-xs sm:grid-cols-4">
            <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
              <div className="text-xs text-slate-400">ESR Loss</div>
              <div className="mt-0.5 font-semibold text-amber-300">{latest.esrLoss.toFixed(5)}</div>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
              <div className="text-xs text-slate-400">MR-STFT Loss</div>
              <div className="mt-0.5 font-semibold text-cyan-300">{latest.stftLoss.toFixed(5)}</div>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
              <div className="text-xs text-slate-400">RG-LRU r_max</div>
              <div className="mt-0.5 font-semibold text-emerald-300">
                {latest.maxRgLruRadius.toFixed(4)}
              </div>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
              <div className="text-xs text-slate-400">UFDN ||U||_2</div>
              <div className="mt-0.5 font-semibold text-violet-300">
                {latest.ufdnSpectralNorm.toFixed(4)}
              </div>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
