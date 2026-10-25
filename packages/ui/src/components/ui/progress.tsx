import { Progress as ProgressPrimitive } from "@base-ui/react/progress";
import { clampNumber, cn } from "../../lib/utils.js";

export interface ProgressProps extends Omit<ProgressPrimitive.Root.Props, "value"> {
  readonly value?: number | null | undefined;
  readonly indicatorClassName?: string | undefined;
}

export function Progress({
  className,
  value = 0,
  indicatorClassName,
  "aria-label": ariaLabel = "Progress",
  ...props
}: ProgressProps) {
  const pct = value === null ? null : clampNumber(value, 0, 100);
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      aria-label={ariaLabel}
      value={pct}
      className={cn("w-full", className)}
      {...props}
    >
      <ProgressPrimitive.Track
        data-slot="progress-track"
        className="relative h-2 w-full overflow-hidden rounded-full bg-slate-800"
      >
        <ProgressPrimitive.Indicator
          data-slot="progress-indicator"
          className={cn("h-full bg-cyan-400 transition-all", indicatorClassName)}
        />
      </ProgressPrimitive.Track>
    </ProgressPrimitive.Root>
  );
}
