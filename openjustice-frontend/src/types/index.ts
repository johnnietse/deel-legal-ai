// =====================
// User & Auth Types
// =====================
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  tier: 'free' | 'pro' | 'enterprise';
  createdAt: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface SignupRequest {
  name: string;
  email: string;
  password: string;
}

// =====================
// RAG Query Types
// =====================
export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  jurisdiction?: string;
  verify?: boolean;
}

export interface Source {
  title: string;
  citation: string;
  content: string;
  relevance_score: number;
  jurisdiction?: string;
  year?: number;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  confidence: 'high' | 'medium' | 'low';
  sources: Source[];
  verification?: {
    is_verified: boolean;
    supported_claims: string[];
    unsupported_claims: string[];
    confidence_score: number;
  };
}

// =====================
// Chat Types
// =====================
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence?: 'high' | 'medium' | 'low';
  timestamp: string;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

// =====================
// Classification Types
// =====================
export type FactorValue = 'Employee' | 'Contractor' | 'Ambiguous' | 'Unknown';

export interface ClassificationFactor {
  name: string;
  description: string;
  value: FactorValue;
}

export interface ClassificationRequest {
  supervision_review: FactorValue;
  ability_hire: FactorValue;
  delegation_tasks: FactorValue;
  ownership_tools: FactorValue;
  chance_profit: FactorValue;
  risk_loss: FactorValue;
  exclusivity_services: FactorValue;
  work_hours_setter: FactorValue;
  work_location: FactorValue;
  uniform_required: FactorValue;
  jurisdiction: string;
  facts?: string;
}

export interface FactorBreakdown {
  factor: string;
  score: number;
  impact: 'high' | 'medium' | 'low';
  reasoning: string;
}

export interface MCTSClassificationResponse {
  classification: 'Employee' | 'Contractor';
  confidence: number;
  factor_analysis: Record<string, {
    value: FactorValue;
    weight: number;
    reasoning: string;
  }>;
  reasoning_text: string;
  tree_statistics: {
    total_nodes: number;
    max_depth: number;
    n_simulations: number;
  };
  duration_ms: number;
}

// =====================
// Document Analysis Types
// =====================
export interface DocumentAnalysis {
  id: string;
  filename: string;
  status: 'processing' | 'completed' | 'error';
  extracted_text: string;
  entities: LegalEntity[];
  summary: string;
  classification_analysis?: {
    is_employment_related: boolean;
    prediction?: string;
    confidence?: number;
  };
  uploadedAt: string;
}

export interface LegalEntity {
  name: string;
  type: 'person' | 'organization' | 'court' | 'statute' | 'regulation';
  mentions: number;
}

// =====================
// Dashboard / Stats Types
// =====================
export interface UsageStats {
  queriesThisMonth: number;
  queriesLimit: number;
  documentsAnalyzed: number;
  classificationsRun: number;
  tier: 'free' | 'pro' | 'enterprise';
}

export interface UsageDataPoint {
  date: string;
  queries: number;
  classifications?: number;
}

export interface RecentActivity {
  id: string;
  type: 'query' | 'classification' | 'document';
  description: string;
  timestamp: string;
}

// =====================
// API Key Types
// =====================
export interface ApiKey {
  id: string;
  name: string;
  key: string;
  createdAt: string;
  lastUsed?: string;
}

// =====================
// Settings Types
// =====================
export interface NotificationPreferences {
  emailDigest: boolean;
  usageAlerts: boolean;
  productUpdates: boolean;
}

// =====================
// Search Types
// =====================
export interface SearchResult {
  id: string;
  title: string;
  excerpt: string;
  url: string;
  source_type: string;
  jurisdiction: string;
  court: string;
  year: string;
  citation: string;
  relevance_score: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  page_size: number;
}

export interface SearchFilters {
  jurisdiction?: string;
  source_type?: string;
  sort_by?: string;
}

// =====================
// DeepSearch Types
// =====================
export interface DeepSearchSource {
  id: string;
  title: string;
  excerpt: string;
  url: string;
  source_type: string;
  relevance_score: number;
}

export interface DeepSearchResponse {
  answer: string;
  sources: DeepSearchSource[];
  source_type_counts?: Record<string, number>;
  suggested_follow_ups?: string[];
  processing_time_ms?: number;
}

export interface TTSVoice {
  id: string;
  description: string;
}

// =====================
// API Response wrapper
// =====================
export interface ApiResponse<T> {
  data: T;
  error?: string;
}
