import { Lock } from "lucide-react";
import { cn } from "../../lib/utils.js";

export interface DiscreteSwitchOption {
  readonly index: number;
  readonly label: string;
  readonly observed?: boolean | undefined;
}

export interface DiscreteSwitchProps {
  readonly id: string;
  readonly label: string;
  readonly options: readonly DiscreteSwitchOption[];
  readonly activeIndex: number;
  readonly onChange?: ((nextIndex: number) => void) | undefined;
  readonly className?: string | undefined;
}

export function DiscreteSwitch({
  id,
  label,
  options,
  activeIndex,
  onChange,
  className,
}: DiscreteSwitchProps) {
  const hasClampedStates = options.some((opt) => opt.observed === false);

  return (
    <div
      data-testid={`discrete-switch-${id}`}
      className={cn(
        "flex flex-col justify-between rounded-lg border border-slate-800 bg-slate-950/70 p-2.5",
        className,
      )}
    >
      <div className="mb-2 flex items-center justify-between text-xs font-medium text-slate-300">
        <span>{label}</span>
        {hasClampedStates ? (
          <span className="inline-flex items-center gap-0.5 font-mono text-xs text-amber-400">
            <Lock className="size-2.5" />
            CL-13
          </span>
        ) : null}
      </div>

      <fieldset className="flex flex-wrap gap-1 border-0 p-0" aria-label={label}>
        {options.map((opt) => {
          const isObserved = opt.observed !== false;
          const isSelected = opt.index === activeIndex;
          return (
            <button
              key={opt.index}
              type="button"
              disabled={!isObserved}
              aria-pressed={isSelected}
              onClick={() => {
                if (isObserved) {
                  onChange?.(opt.index);
                }
              }}
              className={cn(
                "flex-1 rounded border px-2 py-1 font-mono text-xs transition-colors",
                isSelected
                  ? "border-violet-400 bg-violet-500/20 font-semibold text-violet-200"
                  : isObserved
                    ? "border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800"
                    : "cursor-not-allowed border-slate-800 bg-slate-950 text-slate-600 line-through",
              )}
            >
              {opt.label}
            </button>
          );
        })}
      </fieldset>
    </div>
  );
}
