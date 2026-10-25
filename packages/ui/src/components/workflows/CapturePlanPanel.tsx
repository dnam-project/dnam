import { CheckCircle2, CircleDot, Mic } from "lucide-react";
import { Badge } from "../ui/badge.js";
import { Button } from "../ui/button.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { Progress } from "../ui/progress.js";
import type { CapturePlanStepView } from "./models.js";

export interface CapturePlanPanelProps {
  readonly modelId: string;
  readonly steps: readonly CapturePlanStepView[];
  readonly onCaptureStep?: ((stemId: string) => void) | undefined;
}

export function CapturePlanPanel({ modelId, steps, onCaptureStep }: CapturePlanPanelProps) {
  const capturedCount = steps.filter((s) => s.captured).length;
  const totalCount = steps.length;
  const completionPct = totalCount > 0 ? (capturedCount / totalCount) * 100 : 0;

  return (
    <Card data-testid={`capture-plan-panel-${modelId}`}>
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Interactive Re-Amp Capture Schedule ({modelId})</CardTitle>
          <CardDescription>
            Star Center + Lobatto Axis Sweeps + Switch States + Prescribed Corners
          </CardDescription>
        </div>
        <Badge variant={capturedCount === totalCount && totalCount > 0 ? "safe" : "core2"}>
          {capturedCount} / {totalCount} Stems Captured
        </Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <Progress value={completionPct} />

        <div className="flex flex-col gap-2">
          {steps.map((step) => (
            <div
              key={step.stemId}
              className="flex flex-col gap-2 rounded border border-slate-800 bg-slate-950/60 p-2.5 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex items-start gap-2">
                {step.captured ? (
                  <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-400" />
                ) : (
                  <CircleDot className="mt-0.5 size-4 shrink-0 text-cyan-400" />
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-semibold text-slate-100">
                      {step.stemId}
                    </span>
                    <Badge variant="outline">{step.tier}</Badge>
                    {step.snrDb !== undefined ? (
                      <span className="font-mono text-xs text-emerald-300">
                        SNR {step.snrDb.toFixed(1)} dB
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-0.5 text-xs text-slate-400">{step.prompt}</p>
                </div>
              </div>

              {onCaptureStep ? (
                <Button
                  size="sm"
                  variant={step.captured ? "secondary" : "default"}
                  onClick={() => onCaptureStep(step.stemId)}
                >
                  <Mic data-icon="inline-start" />
                  {step.captured ? "Re-Capture" : "Record Stem"}
                </Button>
              ) : null}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
