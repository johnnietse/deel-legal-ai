/**
 * Real API Client
 * 
 * Connects to the actual OpenJustice.ai backend API.
 * Uses the same interface as mockApi for easy swapping.
 */

import type {
  User,
  LoginRequest,
  SignupRequest,
  RAGQueryResponse,
  MCTSClassificationResponse,
  ClassificationRequest,
  UsageStats,
  RecentActivity,
  UsageDataPoint,
  ApiKey,
  DocumentAnalysis,
  Conversation,
  Message,
  Source,
  SearchResponse,
  SearchFilters,
  DeepSearchResponse,
  TTSVoice,
} from "@/types";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem("access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// =====================
// Auth
// =====================

export async function register(data: SignupRequest): Promise<{ user_id: string; email: string; name: string; tier: string; access_token: string; refresh_token: string }> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

export async function login(data: LoginRequest): Promise<{ user_id: string; email: string; name: string; tier: string; access_token: string; refresh_token: string }> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const result = await handleResponse<{ user_id: string; email: string; name: string; tier: string; access_token: string; refresh_token: string }>(response);
  // Store tokens
  localStorage.setItem("access_token", result.access_token);
  localStorage.setItem("refresh_token", result.refresh_token);
  return result;
}

export async function refreshToken(): Promise<{ access_token: string; refresh_token: string }> {
  const refresh = localStorage.getItem("refresh_token");
  if (!refresh) throw new Error("No refresh token");
  
  const response = await fetch(`${API_BASE}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  });
  const result = await handleResponse<{ access_token: string; refresh_token: string }>(response);
  localStorage.setItem("access_token", result.access_token);
  localStorage.setItem("refresh_token", result.refresh_token);
  return result;
}

export function logout(): void {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

export async function getProfile(): Promise<User> {
  const response = await fetch(`${API_BASE}/users/me`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

export async function updateProfile(data: { name?: string; current_password?: string; new_password?: string }): Promise<User> {
  const response = await fetch(`${API_BASE}/users/me`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

export async function getUsageStats(): Promise<UsageStats> {
  const response = await fetch(`${API_BASE}/users/me/usage`, {
    headers: getAuthHeaders(),
  });
  const data = await handleResponse<{
    queries_used: number;
    queries_limit: number;
    documents_uploaded: number;
    tier: string;
  }>(response);
  return {
    queriesThisMonth: data.queries_used,
    queriesLimit: data.queries_limit,
    documentsAnalyzed: data.documents_uploaded,
    classificationsRun: 0,
    tier: data.tier as UsageStats["tier"],
  };
}

// =====================
// RAG Query
// =====================

export async function queryRAG(question: string): Promise<RAGQueryResponse> {
  const response = await fetch(`${API_BASE}/chat/stream?question=${encodeURIComponent(question)}`, {
    headers: getAuthHeaders(),
  });
  
  // For SSE streaming, we need to handle it differently
  // For now, use the non-streaming endpoint
  const ragResponse = await fetch(`${API_BASE.replace("/api", "")}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ question }),
  });
  return handleResponse(ragResponse);
}

// =====================
// Classification
// =====================

export async function classifyWithReasoning(data: ClassificationRequest): Promise<MCTSClassificationResponse> {
  const response = await fetch(`${API_BASE.replace("/api", "")}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

// =====================
// Conversations
// =====================

export async function getConversations(): Promise<Conversation[]> {
  const response = await fetch(`${API_BASE}/chat/conversations`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ conversations: Conversation[] }>(response);
  return result.conversations;
}

export async function createConversation(data: { title: string; messages: Message[] }): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/chat/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
    headers: getAuthHeaders(),
  });
  return handleResponse(response);
}

export async function updateConversation(id: string, data: { title?: string; messages?: Message[] }): Promise<Conversation> {
  const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
    method: "DELETE",
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Failed to delete conversation");
}

// =====================
// Documents
// =====================

function transformDocument(doc: any): DocumentAnalysis {
  const processing = doc.processing_results || {};
 
 
  const status = doc.status === "completed" ? "completed" : doc.status === "error" ? "error" : "processing";
  return {
    id: doc.id,
    filename: doc.filename,
    status,
    extracted_text: processing.text_preview || "",
    entities: (processing.metadata?.entities || []).map((e: any) => ({
      name: e.name,
      type: e.type,
      mentions: e.mentions,
    })),
    summary: processing.text_preview ? processing.text_preview.substring(0, 500) + "..." : "Processing...",
    classification_analysis: processing.metadata?.classification_analysis ? {
      is_employment_related: processing.metadata.classification_analysis.is_employment_related,
      prediction: processing.metadata.classification_analysis.prediction,
      confidence: processing.metadata.classification_analysis.confidence,
    } : undefined,
    uploadedAt: doc.created_at,
  };
}

export async function getDocuments(page = 1, pageSize = 20): Promise<{ documents: DocumentAnalysis[]; total: number }> {
  const response = await fetch(`${API_BASE}/documents?page=${page}&page_size=${pageSize}`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ documents: any[]; total: number }>(response);
  return {
    documents: result.documents.map(transformDocument),
    total: result.total,
  };
}

export async function uploadDocument(file: File): Promise<DocumentAnalysis> {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: formData,
  });
  const result = await handleResponse<{ document_id: string; filename: string; size_bytes: number; status: string }>(response);
  
  // Return a minimal DocumentAnalysis for immediate UI feedback
  const status = result.status === "completed" ? "completed" : result.status === "error" ? "error" : "processing";
  return {
    id: result.document_id,
    filename: result.filename,
    status,
    extracted_text: "",
    entities: [],
    summary: "Document uploaded. Processing...",
    uploadedAt: new Date().toISOString(),
  };
}

export async function getDocument(id: string): Promise<DocumentAnalysis> {
  const response = await fetch(`${API_BASE}/documents/${id}`, {
    headers: getAuthHeaders(),
  });
  const doc = await handleResponse<any>(response);
  return transformDocument(doc);
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${id}`, {
    method: "DELETE",
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Failed to delete document");
}

// =====================
// Dashboard Stats (using existing endpoints)
// =====================

export async function getUsageChartData(days: number = 30): Promise<UsageDataPoint[]> {
  const response = await fetch(`${API_BASE}/users/me/usage/chart?days=${days}`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ data: UsageDataPoint[] }>(response);
  return result.data;
}

export async function getRecentActivity(): Promise<RecentActivity[]> {
  const response = await fetch(`${API_BASE}/users/me/activity`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ data: RecentActivity[] }>(response);
  return result.data;
}

// =====================
// Search
// =====================

export async function search(
  query: string,
  filters: SearchFilters = {},
): Promise<SearchResponse> {
  const params = new URLSearchParams({ query });
  if (filters.jurisdiction) params.set("jurisdiction", filters.jurisdiction);
  if (filters.source_type) params.set("source_type", filters.source_type);
  if (filters.sort_by) params.set("sort_by", filters.sort_by);

  const response = await fetch(
    `${API_BASE.replace("/api", "")}/rag/search?${params.toString()}`,
    {
      method: "POST",
      headers: getAuthHeaders(),
    },
  );
  return handleResponse(response);
}

// =====================
// DeepSearch
// =====================

export async function deepSearch(query: string, maxSources = 15): Promise<DeepSearchResponse> {
  const response = await fetch(`${API_BASE.replace("/api", "")}/rag/deepsearch`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ query, max_sources: maxSources }),
  });
  return handleResponse(response);
}

export async function deepSearchFollowUp(
  originalQuery: string,
  followUp: string,
): Promise<DeepSearchResponse> {
  const params = new URLSearchParams({
    original_query: originalQuery,
    follow_up: followUp,
  });
  const response = await fetch(
    `${API_BASE.replace("/api", "")}/rag/deepsearch/followup?${params.toString()}`,
    {
      method: "POST",
      headers: getAuthHeaders(),
    },
  );
  return handleResponse(response);
}

// =====================
// Text-to-Speech
// =====================

export async function generateAudio(text: string, voice = "en-CA-LiamNeural"): Promise<Blob> {
  const response = await fetch(`${API_BASE}/chat/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ text, voice }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Audio generation failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.blob();
}

export async function getVoices(): Promise<TTSVoice[]> {
  const response = await fetch(`${API_BASE}/chat/tts/voices`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ voices: TTSVoice[] }>(response);
  return result.voices;
}

// =====================
// API Keys
// =====================

export async function getApiKeys(): Promise<ApiKey[]> {
  const response = await fetch(`${API_BASE}/auth/keys`, {
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{ keys: Array<{
    id: string;
    name: string;
    key_preview: string;
    created_at: string | null;
    last_used_at: string | null;
  }> }>(response);
  return result.keys.map((k) => ({
    id: k.id,
    name: k.name,
    key: k.key_preview,
    createdAt: k.created_at || "",
    lastUsed: k.last_used_at || undefined,
  }));
}

export async function createApiKey(name: string): Promise<ApiKey> {
  const response = await fetch(`${API_BASE}/auth/keys?name=${encodeURIComponent(name)}`, {
    method: "POST",
    headers: getAuthHeaders(),
  });
  const result = await handleResponse<{
    id: string;
    name: string;
    key: string;
    created_at: string | null;
  }>(response);
  return {
    id: result.id,
    name: result.name,
    key: result.key,
    createdAt: result.created_at || new Date().toISOString(),
  };
}

export async function revokeApiKey(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/keys/${id}`, {
    method: "DELETE",
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Failed to revoke key" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
}

// =====================
// Subscription
// =====================

export async function upgradeSubscription(tier: 'pro' | 'enterprise'): Promise<{ success: boolean; tier: string }> {
  const response = await fetch(`${API_BASE}/subscriptions/upgrade`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ tier }),
  });
  return handleResponse(response);
}

// =====================
// Export as realApi object for easy swapping
// =====================

export const realApi = {
  register,
  login,
  refreshToken,
  logout,
  getProfile,
  updateProfile,
  getUsageStats,
  queryRAG,
  classifyWithReasoning,
  getConversations,
  createConversation,
  getConversation,
  updateConversation,
  deleteConversation,
  getDocuments,
  uploadDocument,
  getDocument,
  deleteDocument,
  search,
  deepSearch,
  deepSearchFollowUp,
  generateAudio,
  getVoices,
  getUsageChartData,
  getRecentActivity,
  getApiKeys,
  createApiKey,
  revokeApiKey,
  upgradeSubscription,
};