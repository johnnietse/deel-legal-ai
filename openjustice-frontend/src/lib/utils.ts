import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: string | Date): string {
  const d = new Date(date);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatDateTime(date: string | Date): string {
  const d = new Date(date);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatRelativeTime(date: string | Date): string {
  const now = new Date();
  const d = new Date(date);
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDate(date);
}

export function truncate(str: string, length: number): string {
  if (str.length <= length) return str;
  return str.slice(0, length) + "...";
}

export function confidenceColor(confidence: string | number): string {
  if (typeof confidence === "string") {
    switch (confidence) {
      case "high": return "text-green-600 dark:text-green-400";
      case "medium": return "text-yellow-600 dark:text-yellow-400";
      case "low": return "text-red-600 dark:text-red-400";
      default: return "text-surface-500";
    }
  }
  if (confidence >= 0.7) return "text-green-600 dark:text-green-400";
  if (confidence >= 0.4) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

export function riskColor(confidence: number): { color: string; label: string } {
  if (confidence >= 0.7) return { color: "bg-red-500", label: "High Risk" };
  if (confidence >= 0.4) return { color: "bg-yellow-500", label: "Medium Risk" };
  return { color: "bg-green-500", label: "Low Risk" };
}
