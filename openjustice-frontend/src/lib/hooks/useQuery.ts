import { useMutation, useQuery } from "@tanstack/react-query";
import { realApi } from "@/lib/api/realClient";
import type { ClassificationRequest } from "@/types";

export function useRAGQuery() {
  return useMutation({
    mutationFn: (question: string) => realApi.queryRAG(question),
  });
}

export function useClassification() {
  return useMutation({
    mutationFn: (data: ClassificationRequest) => realApi.classifyWithReasoning(data),
  });
}

export function useUsageStats() {
  return useQuery({
    queryKey: ["usage-stats"],
    queryFn: () => realApi.getUsageStats(),
  });
}

export function useUsageChartData(days: number = 30) {
  return useQuery({
    queryKey: ["usage-chart", days],
    queryFn: () => realApi.getUsageChartData(days),
  });
}

export function useRecentActivity() {
  return useQuery({
    queryKey: ["recent-activity"],
    queryFn: () => realApi.getRecentActivity(),
  });
}

export function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: () => realApi.getConversations(),
  });
}

export function useDocumentAnalysis() {
  return useMutation({
    mutationFn: (file: File) => realApi.uploadDocument(file),
  });
}

export function useApiKeys() {
  return useQuery({
    queryKey: ["api-keys"],
    queryFn: () => realApi.getApiKeys(),
  });
}

export function useCreateApiKey() {
  return useMutation({
    mutationFn: (name: string) => realApi.createApiKey(name),
  });
}

export function useRevokeApiKey() {
  return useMutation({
    mutationFn: (id: string) => realApi.revokeApiKey(id),
  });
}

export function useUpgradeSubscription() {
  return useMutation({
    mutationFn: (tier: 'pro' | 'enterprise') => realApi.upgradeSubscription(tier),
  });
}

export function useDeepSearch() {
  return useMutation({
    mutationFn: (query: string) => realApi.deepSearch(query),
  });
}

export function useSearch() {
  return useMutation({
    mutationFn: ({ query, filters }: { query: string; filters: any }) =>
      realApi.search(query, filters),
  });
}

export function useGenerateAudio() {
  return useMutation({
    mutationFn: (text: string) => realApi.generateAudio(text),
  });
}
