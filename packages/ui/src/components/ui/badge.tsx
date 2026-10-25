import { mergeProps } from "@base-ui/react/merge-props";
import { useRender } from "@base-ui/react/use-render";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils.js";

export const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 font-mono text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-cyan-500/40 bg-cyan-500/15 text-cyan-300",
        core1: "border-amber-500/40 bg-amber-500/15 text-amber-300",
        core2: "border-cyan-500/40 bg-cyan-500/15 text-cyan-300",
        control: "border-violet-500/40 bg-violet-500/15 text-violet-300",
        safe: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
        warn: "border-amber-500/40 bg-amber-500/15 text-amber-300",
        danger: "border-red-500/40 bg-red-500/15 text-red-300",
        outline: "border-slate-700 bg-slate-900/60 text-slate-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export type BadgeProps = useRender.ComponentProps<"span"> & VariantProps<typeof badgeVariants>;

export function Badge({ className, variant = "default", render, ...props }: BadgeProps) {
  return useRender({
    defaultTagName: "span",
    props: mergeProps<"span">(
      {
        className: cn(badgeVariants({ variant }), className),
      },
      props,
    ),
    render,
    state: {
      slot: "badge",
      variant,
    },
  });
}
