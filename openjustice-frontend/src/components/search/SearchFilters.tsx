import React from "react";
import { Card, CardContent } from "@/components/ui/card";

interface SearchFiltersProps {
  jurisdiction: string;
  sourceType: string;
  sortBy: string;
  onJurisdictionChange: (v: string) => void;
  onSourceTypeChange: (v: string) => void;
  onSortByChange: (v: string) => void;
}

const JURISDICTIONS = ["", "Ontario", "Federal", "British Columbia", "Alberta", "Quebec", "Nova Scotia"];

export default function SearchFilters({
  jurisdiction,
  sourceType,
  sortBy,
  onJurisdictionChange,
  onSourceTypeChange,
  onSortByChange,
}: SearchFiltersProps) {
  const selectClass =
    "border border-surface-300 dark:border-surface-600 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-surface-800 text-surface-700 dark:text-surface-300 focus:outline-none focus:ring-2 focus:ring-primary-500";

  return (
    <Card className="mb-6">
      <CardContent className="p-4">
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-xs font-medium text-surface-600 dark:text-surface-400 mb-1">
              Jurisdiction
            </label>
            <select
              value={jurisdiction}
              onChange={(e) => onJurisdictionChange(e.target.value)}
              className={selectClass}
            >
              <option value="">All</option>
              {JURISDICTIONS.filter(Boolean).map((j) => (
                <option key={j} value={j}>
                  {j}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-surface-600 dark:text-surface-400 mb-1">
              Source Type
            </label>
            <select
              value={sourceType}
              onChange={(e) => onSourceTypeChange(e.target.value)}
              className={selectClass}
            >
              <option value="">All</option>
              <option value="case_law">Case Law</option>
              <option value="web">Web</option>
              <option value="statute">Statute</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-surface-600 dark:text-surface-400 mb-1">
              Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => onSortByChange(e.target.value)}
              className={selectClass}
            >
              <option value="relevance">Relevance</option>
              <option value="date">Date</option>
            </select>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
