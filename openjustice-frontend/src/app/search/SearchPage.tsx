import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Search as SearchIcon, Loader2, AlertCircle, FileSearch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { realApi } from "@/lib/api/realClient";
import SearchResultCard from "@/components/search/SearchResultCard";
import SearchFilters from "@/components/search/SearchFilters";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [sourceType, setSourceType] = useState("");
  const [sortBy, setSortBy] = useState("relevance");

  const searchMutation = useMutation({
    mutationFn: (q: string) =>
      realApi.search(q, {
        jurisdiction: jurisdiction || undefined,
        source_type: sourceType || undefined,
        sort_by: sortBy,
      }),
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate(query);
  };

  const data = searchMutation.data;

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-surface-900 dark:text-surface-100">
          Search Canadian Employment Law
        </h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Search cases, statutes, and web sources across the legal corpus
        </p>
      </div>

      <form onSubmit={handleSearch} className="mb-6">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search cases, statutes, articles..."
              className="w-full border border-surface-300 dark:border-surface-600 rounded-lg pl-10 pr-4 py-3 text-lg bg-white dark:bg-surface-800 text-surface-900 dark:text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <Button type="submit" size="lg" disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Searching...
              </>
            ) : (
              <>
                <SearchIcon className="h-4 w-4" /> Search
              </>
            )}
          </Button>
        </div>
      </form>

      <SearchFilters
        jurisdiction={jurisdiction}
        sourceType={sourceType}
        sortBy={sortBy}
        onJurisdictionChange={setJurisdiction}
        onSourceTypeChange={setSourceType}
        onSortByChange={setSortBy}
      />

      {searchMutation.isError && (
        <div className="mb-4 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-red-500" />
          <span className="text-sm text-red-600 dark:text-red-400">
            Search failed. Please try again.
          </span>
        </div>
      )}

      {data && (
        <div className="mb-4 text-sm text-surface-500 dark:text-surface-400">
          {data.total} results ({data.results.length} shown)
        </div>
      )}

      <div className="space-y-4">
        {data?.results.map((result) => (
          <SearchResultCard key={result.id} result={result} />
        ))}
      </div>

      {!data && !searchMutation.isPending && (
        <div className="text-center py-16 text-surface-400 dark:text-surface-500">
          <FileSearch className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Enter a query to search Canadian employment law</p>
        </div>
      )}
    </div>
  );
}
