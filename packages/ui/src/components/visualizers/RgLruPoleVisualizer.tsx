import { Badge } from "../ui/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card.js";
import { computeRgLruInterpolation } from "./models.js";

export interface RgLruPoleVisualizerProps {
  readonly blend?: number | undefined;
  readonly onBlendChange?: ((blend: number) => void) | undefined;
}

export function RgLruPoleVisualizer({ blend = 0.5, onBlendChange }: RgLruPoleVisualizerProps) {
  const metrics = computeRgLruInterpolation(0.995, 0.9, blend);

  return (
    <Card data-testid="visualizer-rglru-pole">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle>Core 1B: Radial-Polar Selective Complex RG-LRU</CardTitle>
          <CardDescription>
            Geodesic Radial-Polar Split vs. Cartesian Chord Dip & 3 kHz Algebraic Newton Rotor
          </CardDescription>
        </div>
        <Badge variant="core1">r_polar = {metrics.radialPolarRadius.toFixed(4)}</Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          <span>Knob Interpolation Blend (u): {blend.toFixed(2)}</span>
          <input
            type="range"
            aria-label="RG-LRU Blend"
            min={0.0}
            max={1.0}
            step={0.05}
            value={blend}
            onChange={(e) => onBlendChange?.(Number.parseFloat(e.currentTarget.value))}
            className="accent-amber-400"
          />
        </label>

        <div className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Radial-Polar |z|</div>
            <div className="mt-0.5 text-emerald-300">{metrics.radialPolarRadius.toFixed(4)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Cartesian |z|</div>
            <div className="mt-0.5 text-amber-300">{metrics.cartesianRadius.toFixed(4)}</div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="text-xs text-slate-400">Cartesian Gain Dip</div>
            <div className="mt-0.5 text-red-300">{metrics.cartesianGainDropDb.toFixed(2)} dB</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
