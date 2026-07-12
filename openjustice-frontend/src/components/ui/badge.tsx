import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300",
        secondary: "bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400",
        destructive: "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300",
        success: "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300",
        warning: "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300",
        outline: "border border-surface-300 dark:border-surface-600 text-surface-600 dark:text-surface-400",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
