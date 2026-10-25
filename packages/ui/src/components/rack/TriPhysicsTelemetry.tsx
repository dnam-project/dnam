import { Activity, Gauge, Radio } from "lucide-react";
import type { DnamTelemetrySnapshot } from "../../host/types.js";
import { clampNumber, formatDecibels } from "../../lib/utils.js";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card.js";
import { Progress } from "../ui/progress.js";

export interface TriPhysicsTelemetryProps {
  readonly telemetry: DnamTelemetrySnapshot;
}

export function dbToMeterPercent(db: number): number {
  return clampNumber(((db + 60.0) / 60.0) * 100.0, 0.0, 100.0);
}

export function TriPhysicsTelemetry({ telemetry }: TriPhysicsTelemetryProps) {
  const inLPct = dbToMeterPercent(telemetry.inputRmsDbL);
  const inRPct = dbToMeterPercent(telemetry.inputRmsDbR);
  const outLPct = dbToMeterPercent(telemetry.outputRmsDbL);
  const outRPct = dbToMeterPercent(telemetry.outputRmsDbR);

  return (
    <Card data-testid="tri-physics-telemetry">
      <CardHeader className="flex flex-row items-center justify-between py-2.5">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-cyan-400" />
          <CardTitle>Tri-Physics Real-Time Telemetry</CardTitle>
        </div>
        <span className="font-mono text-xs text-slate-400">
          {telemetry.sampleRateHz / 1000} kHz Audio / {telemetry.controlRateHz / 1000} kHz Control
        </span>
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="flex flex-col gap-2 rounded border border-slate-800 bg-slate-950/60 p-2.5">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="inline-flex items-center gap-1 font-medium">
              <Radio className="size-3.5 text-amber-400" />
              Stereo I/O RMS
            </span>
            <span className="font-mono text-xs text-slate-400">
              In {formatDecibels(telemetry.inputRmsDbL)} / Out{" "}
              {formatDecibels(telemetry.outputRmsDbL)}
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
              <span className="w-8">IN L/R</span>
              <Progress value={(inLPct + inRPct) * 0.5} className="h-1.5" />
            </div>
            <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
              <span className="w-8">OUT L/R</span>
              <Progress
                value={(outLPct + outRPct) * 0.5}
                indicatorClassName="bg-amber-400"
                className="h-1.5"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-1.5 rounded border border-slate-800 bg-slate-950/60 p-2.5 font-mono text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Core 1 RG-LRU r_max</span>
            <span className="text-amber-300">{telemetry.maxRgLruRadius.toFixed(4)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Core 2 UFDN Margin</span>
            <span className="text-cyan-300">+{telemetry.ufdnLyapunovMargin.toFixed(4)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Mahalanobis R² / α</span>
            <span className="text-violet-300">
              {telemetry.mahalanobisR2.toFixed(2)} / {telemetry.activeTrustRegionAlpha.toFixed(3)}
            </span>
          </div>
        </div>

        <div className="flex flex-col justify-between rounded border border-slate-800 bg-slate-950/60 p-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="inline-flex items-center gap-1 font-medium text-slate-300">
              <Gauge className="size-3.5 text-emerald-400" />
              SIMD DSP Load
            </span>
            <span className="font-mono text-emerald-300">
              {telemetry.dspLoadPercent.toFixed(1)}%
            </span>
          </div>
          <Progress
            value={telemetry.dspLoadPercent}
            indicatorClassName="bg-emerald-400"
            className="mt-2 h-2"
          />
          <span className="mt-1 font-mono text-xs text-slate-400">
            Zero FSQRT / FDIV in 48 kHz & 3 kHz loops
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
