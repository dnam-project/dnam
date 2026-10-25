import { Button as ButtonPrimitive } from "@base-ui/react/button";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils.js";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-md text-xs font-medium whitespace-nowrap transition-colors focus-visible:ring-1 focus-visible:ring-cyan-400 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-cyan-500 text-slate-950 shadow-sm hover:bg-cyan-400",
        secondary: "border border-slate-700 bg-slate-800 text-slate-100 hover:bg-slate-700",
        outline: "border border-slate-700 bg-transparent text-slate-200 hover:bg-slate-800",
        ghost: "text-slate-300 hover:bg-slate-800 hover:text-slate-100",
        destructive: "bg-red-600 text-white shadow-sm hover:bg-red-500",
      },
      size: {
        default: "h-8 px-3 py-1.5",
        sm: "h-7 rounded px-2.5 text-xs",
        lg: "h-10 rounded-md px-5 text-sm",
        icon: "size-8",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export type ButtonProps = ButtonPrimitive.Props & VariantProps<typeof buttonVariants>;

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return (
    <ButtonPrimitive
      data-slot="button"
      type={type}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}
