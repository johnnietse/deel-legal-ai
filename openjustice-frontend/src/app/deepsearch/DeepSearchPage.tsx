import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Search as SearchIcon,
  Loader2,
  AlertCircle,
  Scale,
  Globe,
  BookOpen,
  Sparkles,
  CornerDownRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { realApi } from "@/lib/api/realClient";
import AudioPlayer from "@/components/search/AudioPlayer";
import type { DeepSearchResponse } from "@/types";

const sourceTypeIcons: Record<string, React.ReactNode> = {
  case_law: <Scale className="h-3.5 w-3.5" />,
  web: <Globe className="h-3.5 w-3.5" />,
  statute: <BookOpen className="h-3.5 w-3.5" />,
  bm25: <Scale className="h-3.5 w-3.5" />,
};

const sourceTypeLabels: Record<string, string> = {
  case_law: "Case Law",
  web: "Web",
  statute: "Statute",
  bm25: "Case Law",
};

interface ConversationTurn {
  query: string;
  result: DeepSearchResponse;
}

export default function DeepSearchPage() {
  const [query, setQuery] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);

  const searchMutation = useMutation({
    mutationFn: (q: string) => realApi.deepSearch(q),
    onSuccess: (data) => {
      setConversation((prev) => [...prev, { query, result: data }]);
      setQuery("");
    },
  });

  const followUpMutation = useMutation({
    mutationFn: ({ original, follow }: { original: string; follow: string }) =>
      realApi.deepSearchFollowUp(original, follow),
    onSuccess: (data) => {
      setConversation((prev) => [...prev, { query: followUp, result: data }]);
      setFollowUp("");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate(query);
  };

  const lastQuery = conversation.length > 0 ? conversation[conversation.length - 1].query : "";

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold text-surface-900 dark:text-surface-100 flex items-center gap-2">
          <Sparkles className="h-7 w-7 text-primary-500" /> DeepSearch
        </h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Multi-source legal research across case law, web, and statutes
        </p>
      </div>

      <form onSubmit={handleSubmit} className="mb-8">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a detailed legal research question..."
          rows={3}
          className="w-full border border-surface-300 dark:border-surface-600 rounded-lg px-4 py-3 text-lg bg-white dark:bg-surface-800 text-surface-900 dark:text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
        />
        <div className="flex justify-between items-center mt-3">
          <span className="text-xs text-surface-400">
            Sources: Case Law · Web · Statutes · BM25
          </span>
          <Button type="submit" disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Researching...
              </>
            ) : (
              <>
                <SearchIcon className="h-4 w-4" /> Search Deeply
              </>
            )}
          </Button>
        </div>
      </form>

      {searchMutation.isError && (
        <div className="mb-4 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-red-500" />
          <span className="text-sm text-red-600 dark:text-red-400">
            DeepSearch failed. Please try again.
          </span>
        </div>
      )}

      <div className="space-y-8">
        {conversation.map((item, i) => (
          <Card key={i} className="overflow-hidden">
            <CardHeader className="bg-surface-50 dark:bg-surface-800/50 border-b border-surface-200 dark:border-surface-700">
              <div className="flex items-start justify-between gap-3">
                <CardTitle className="text-base text-surface-900 dark:text-surface-100">
                  {item.query}
                </CardTitle>
                {item.result.processing_time_ms ? (
                  <span className="text-xs text-surface-400 shrink-0">
                    {Math.round(item.result.processing_time_ms / 1000)}s
                  </span>
                ) : null}
              </div>
              {item.result.source_type_counts &&
                Object.keys(item.result.source_type_counts).length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {Object.entries(item.result.source_type_counts).map(([type, count]) => (
                      <Badge key={type} variant="secondary" className="gap-1">
                        {sourceTypeIcons[type] || <Scale className="h-3 w-3" />}
                        {sourceTypeLabels[type] || type}: {count}
                      </Badge>
                    ))}
                  </div>
                )}
            </CardHeader>

            <CardContent className="p-6">
              <div className="text-sm text-surface-700 dark:text-surface-300 whitespace-pre-wrap leading-relaxed">
                {item.result.answer}
              </div>

              {item.result.sources && item.result.sources.length > 0 && (
                <details className="mt-4 group">
                  <summary className="text-sm text-primary-600 dark:text-primary-400 cursor-pointer hover:underline">
                    Sources ({item.result.sources.length})
                  </summary>
                  <div className="mt-3 space-y-2">
                    {item.result.sources.map((source) => (
                      <div
                        key={source.id}
                        className="flex items-start gap-2 p-3 bg-surface-50 dark:bg-surface-800/50 rounded-lg text-sm"
                      >
                        <span className="mt-0.5 text-primary-500">
                          {sourceTypeIcons[source.source_type] || <Scale className="h-3.5 w-3.5" />}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-surface-900 dark:text-surface-100 truncate">
                            {source.title}
                          </p>
                          <p className="text-surface-500 dark:text-surface-400 text-xs truncate">
                            {source.excerpt}
                          </p>
                        </div>
                        <span className="text-xs text-surface-400 shrink-0">
                          {Math.round(source.relevance_score * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {item.result.suggested_follow_ups && item.result.suggested_follow_ups.length > 0 && (
                <div className="mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
                  <p className="text-sm text-surface-500 dark:text-surface-400 mb-2">
                    Follow-up questions:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {item.result.suggested_follow_ups.map((q, j) => (
                      <button
                        key={j}
                        onClick={() => followUpMutation.mutate({ original: item.query, follow: q })}
                        disabled={followUpMutation.isPending}
                        className="text-sm px-3 py-1.5 bg-surface-100 dark:bg-surface-800 rounded-full hover:bg-surface-200 dark:hover:bg-surface-700 text-surface-700 dark:text-surface-300 disabled:opacity-50"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
                <AudioPlayer text={item.result.answer} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {conversation.length > 0 && (
        <div className="mt-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={followUp}
              onChange={(e) => setFollowUp(e.target.value)}
              placeholder="Ask a follow-up question..."
              className="flex-1 border border-surface-300 dark:border-surface-600 rounded-lg px-4 py-2 bg-white dark:bg-surface-800 text-surface-900 dark:text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <Button
              onClick={() => {
                if (followUp.trim() && lastQuery) {
                  followUpMutation.mutate({ original: lastQuery, follow: followUp });
                }
              }}
              disabled={!followUp.trim() || followUpMutation.isPending}
              variant="secondary"
            >
              {followUpMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Following up...
                </>
              ) : (
                <>
                  <CornerDownRight className="h-4 w-4" /> Follow Up
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {conversation.length === 0 && !searchMutation.isPending && (
        <div className="text-center py-16 text-surface-400 dark:text-surface-500">
          <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Enter a research question to begin</p>
          <p className="text-sm mt-2">
            Example: "What are the notice requirements for constructive dismissal in Ontario?"
          </p>
        </div>
      )}
    </div>
  );
}
