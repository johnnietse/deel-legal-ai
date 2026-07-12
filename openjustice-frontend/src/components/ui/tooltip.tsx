import * as React from "react";
import { cn } from "@/lib/utils";

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
}

export function Tooltip({ content, children }: TooltipProps) {
  const [show, setShow] = React.useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          role="tooltip"
          className={cn(
            "absolute z-50 px-3 py-2 text-xs font-medium text-white bg-surface-900 dark:bg-surface-100 dark:text-surface-900 rounded-lg shadow-lg",
            "bottom-full left-1/2 -translate-x-1/2 mb-2",
            "whitespace-pre-wrap max-w-xs",
            "after:content-[''] after:absolute after:top-full after:left-1/2 after:-translate-x-1/2 after:border-4 after:border-transparent after:border-t-surface-900 dark:after:border-t-surface-100"
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
}
