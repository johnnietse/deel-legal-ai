import React from "react";
import { ExternalLink, Scale, Globe, BookOpen } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { SearchResult } from "@/types";

const sourceTypeLabels: Record<string, string> = {
  case_law: "Case Law",
  web: "Web",
  statute: "Statute",
  bm25: "Case Law",
};

const sourceTypeIcons: Record<string, React.ReactNode> = {
  case_law: <Scale className="h-3 w-3" />,
  web: <Globe className="h-3 w-3" />,
  statute: <BookOpen className="h-3 w-3" />,
  bm25: <Scale className="h-3 w-3" />,
};

export default function SearchResultCard({ result }: { result: SearchResult }) {
  const sourceType = result.source_type || "case_law";
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3 mb-2">
          <h3 className="text-lg font-semibold text-surface-900 dark:text-surface-100 leading-snug">
            {result.title}
          </h3>
          <Badge variant="secondary" className="shrink-0 gap-1">
            {sourceTypeIcons[sourceType] || <Scale className="h-3 w-3" />}
            {sourceTypeLabels[sourceType] || sourceType}
          </Badge>
        </div>

        <p className="text-sm text-surface-600 dark:text-surface-400 mb-3 line-clamp-3">
          {result.excerpt}
        </p>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-surface-500 dark:text-surface-400">
          {result.jurisdiction && <span>Jurisdiction: {result.jurisdiction}</span>}
          {result.court && <span>Court: {result.court}</span>}
          {result.year && <span>Year: {result.year}</span>}
          {result.citation && <span>Citation: {result.citation}</span>}
          <span className="ml-auto font-medium text-primary-600 dark:text-primary-400">
            {Math.round((result.relevance_score || 0) * 100)}% match
          </span>
        </div>

        {result.url && (
          <a
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center gap-1 text-sm text-primary-600 dark:text-primary-400 hover:underline"
          >
            View source <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </CardContent>
    </Card>
  );
}
