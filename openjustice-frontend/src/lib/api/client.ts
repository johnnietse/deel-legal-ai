/**
 * Mock API Client
 * 
 * Provides realistic mock implementations of all backend endpoints.
 * Simulates network latency and occasional errors for testing.
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
} from "@/types";

// =====================
// Helpers
// =====================

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomDelay(): Promise<void> {
  const ms = 300 + Math.random() * 500; // 300-800ms
  return delay(ms);
}

function simulateError(): void {
  // 5% failure rate
  if (Math.random() < 0.05) {
    throw new Error("Simulated network error - please retry");
  }
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

// =====================
// Legal Case Database (Mock)
// =====================

const mockCases = [
  {
    title: "McCormick v. Fasken Martineau DuMoulin LLP",
    citation: "2014 SCC 39",
    content: "The Supreme Court of Canada established a four-part framework for determining employee status, focusing on control, ownership of tools, chance of profit, and risk of loss. The court emphasized that the overarching question is whether the worker is performing services as a person in business on their own account.",
    relevance_score: 0.95,
    jurisdiction: "SCC",
    year: 2014,
  },
  {
    title: "671122 Ontario Ltd. v. Sagaz Industries Canada Inc.",
    citation: "2001 SCC 59",
    content: "The landmark case for the 'Sagaz test' - a four-factor approach to distinguish employees from independent contractors: (1) degree of control, (2) ownership of tools/equipment, (3) chance of profit, and (4) risk of loss. The central question is whether the person is performing services as a business on their own account.",
    relevance_score: 0.98,
    jurisdiction: "SCC",
    year: 2001,
  },
  {
    title: "Western Canadian Shop Council v. Liquor Control Board of Ontario",
    citation: "2020 ONSC 1234",
    content: "Applied the Sagaz framework in the context of platform economy workers. Found that Uber drivers were independent contractors based on control over schedules, ownership of vehicles, and assumption of financial risk.",
    relevance_score: 0.88,
    jurisdiction: "ON",
    year: 2020,
  },
  {
    title: "Dazy v. HBC",
    citation: "2023 ONCA 456",
    content: "Ontario Court of Appeal clarified that the 'organisation test' is not a standalone test but rather a factor within the Sagaz framework. Emphasized that no single factor is determinative.",
    relevance_score: 0.85,
    jurisdiction: "ON",
    year: 2023,
  },
  {
    title: "Toronto (City) v. CUPE Local 79",
    citation: "2003 SCC 63",
    content: "Supreme Court held that the determination of worker status requires a contextual approach. Legislative purpose and policy objectives should inform the classification analysis.",
    relevance_score: 0.82,
    jurisdiction: "SCC",
    year: 2003,
  },
  {
    title: "Keays v. Honda Canada Inc.",
    citation: "2008 SCC 39",
    content: "Established principles for assessing damages in wrongful dismissal cases, including the distinction between employees and independent contractors for the purpose of reasonable notice periods.",
    relevance_score: 0.75,
    jurisdiction: "SCC",
    year: 2008,
  },
  {
    title: "R. v. Kapp",
    citation: "2008 SCC 41",
    content: "While primarily about Charter rights, this case provides important context for understanding how courts approach questions of worker vulnerability and protection in the employment context.",
    relevance_score: 0.45,
    jurisdiction: "SCC",
    year: 2008,
  },
  {
    title: "British Columbia (Human Rights Tribunal) v. Schrenk",
    citation: "2017 SCC 62",
    content: "Addressed the test for jurisdiction over human rights complaints involving independent contractors versus employees in the workplace context.",
    relevance_score: 0.65,
    jurisdiction: "SCC",
    year: 2017,
  },
  {
    title: "Wilson v. Atomic Energy of Canada Ltd.",
    citation: "2016 SCC 29",
    content: "Clarified the distinction between employees and independent contractors in the context of unjust dismissal under the Canada Labour Code.",
    relevance_score: 0.78,
    jurisdiction: "SCC",
    year: 2016,
  },
  {
    title: "Heller v. Uber Technologies Inc.",
    citation: "2019 ONCA 1",
    content: "Ontario Court of Appeal found arbitration clause in Uber's contract unconscionable, emphasizing the vulnerable position of gig economy workers and their need for access to employment standards.",
    relevance_score: 0.91,
    jurisdiction: "ON",
    year: 2019,
  },
];

// =====================
// Mock Templates for Answers
// =====================

const mockAnswers: Record<string, { answer: string; confidence: 'high' | 'medium' | 'low'; sources: number[] }> = {
  "employee": {
    answer: "Under Canadian common law, the distinction between an employee and an independent contractor is determined by the **Sagaz test** (671122 Ontario Ltd. v. Sagaz Industries Canada Inc., 2001 SCC 59). The four key factors are:\n\n1. **Control**: The degree of control the employer exercises over the worker's activities\n2. **Ownership of Tools**: Who provides the tools and equipment\n3. **Chance of Profit**: The worker's opportunity for profit or gain\n4. **Risk of Loss**: The worker's risk of financial loss\n\nThe central inquiry is whether the worker is performing services as a person in business on their own account (McCormick v. Fasken Martineau DuMoulin LLP, 2014 SCC 39).",
    confidence: "high",
    sources: [0, 1],
  },
  "misclassification": {
    answer: "Worker misclassification occurs when an employer incorrectly classifies an employee as an independent contractor, depriving the worker of statutory protections under employment standards legislation. In Ontario, the **Employment Standards Act, 2000** provides that the true nature of the working relationship governs, not the label assigned by the parties.\n\nRecent case law has expanded protections: In *Heller v. Uber Technologies Inc.* (2019 ONCA 1), the Court of Appeal struck down arbitration clauses that would prevent gig workers from accessing employment standards protections.\n\nThe consequences of misclassification can include:\n- Liability for unpaid wages, overtime, and vacation pay\n- Wrongful dismissal damages\n- Penalties under employment standards legislation\n- Canada Pension Plan and Employment Insurance remittances",
    confidence: "high",
    sources: [9, 1],
  },
  "notice": {
    answer: "The reasonable notice period for wrongful dismissal depends on several factors established in *Bardal v. Globe & Mail Ltd.* (1960):\n\n1. **Character of employment**: The nature of the position and responsibilities\n2. **Length of service**: The duration of the employment relationship\n3. **Age of the employee**: The employee's age at termination\n4. **Availability of similar employment**: The job market and employee's skills\n\nIn Ontario, the Employment Standards Act sets statutory minimum notice periods, but common law reasonable notice can be significantly longer, typically ranging from 1 month per year of service for short-term employees to up to 24 months for long-term, specialized employees.",
    confidence: "medium",
    sources: [5],
  },
  "default": {
    answer: "Based on the available legal sources, this question requires careful analysis of the relevant law. In Canadian employment law, each case is determined on its own facts, and courts consider the totality of the relationship between the parties.\n\nKey principles to consider:\n- The intention of the parties, as evidenced by their agreement and conduct\n- The degree of control exercised by the putative employer\n- Whether the worker bears financial risk and has opportunity for profit\n- The context of the industry and nature of the work\n\nI recommend consulting with a qualified legal professional for advice specific to your situation. Would you like me to research a more specific aspect of this question?",
    confidence: "medium",
    sources: [1, 3],
  },
};

// =====================
// Classification Factor Details
// =====================

export const FACTOR_DETAILS: Record<string, { description: string; employeeIndicators: string; contractorIndicators: string }> = {
  supervision_review: {
    description: "The degree of control and supervision exercised by the employer over the worker's activities and work product.",
    employeeIndicators: "Close supervision, regular performance reviews, detailed instructions on how to work",
    contractorIndicators: "Independent execution of work, minimal supervision, results-focused",
  },
  ability_hire: {
    description: "Whether the worker has the ability to hire and manage their own employees or assistants.",
    employeeIndicators: "Cannot hire others; company provides all staff",
    contractorIndicators: "Can hire and manage own employees, subcontract work",
  },
  delegation_tasks: {
    description: "Whether the worker can delegate tasks to others or must perform the work personally.",
    employeeIndicators: "Must perform work personally; no delegation allowed",
    contractorIndicators: "Can delegate tasks, use subcontractors freely",
  },
  ownership_tools: {
    description: "Whether the worker provides their own tools, equipment, and materials to perform the work.",
    employeeIndicators: "Employer provides all tools, equipment, workspace",
    contractorIndicators: "Worker owns/provides their own tools, vehicle, equipment",
  },
  chance_profit: {
    description: "The worker's opportunity to earn profit or financial gain beyond their base compensation.",
    employeeIndicators: "Fixed salary or hourly wage; no profit opportunity",
    contractorIndicators: "Can increase profit through efficiency, negotiation, additional clients",
  },
  risk_loss: {
    description: "The worker's exposure to financial risk or loss in performing the work.",
    employeeIndicators: "No financial risk; guaranteed pay regardless of outcomes",
    contractorIndicators: "Bears cost of overruns, non-payment, slow periods",
  },
  exclusivity_services: {
    description: "Whether the worker is required to provide services exclusively to one employer or can work for multiple clients.",
    employeeIndicators: "Exclusive arrangement; cannot work for competitors",
    contractorIndicators: "Free to work for multiple clients simultaneously",
  },
  work_hours_setter: {
    description: "Who determines the worker's schedule, hours of work, and time management.",
    employeeIndicators: "Employer sets fixed schedule and hours",
    contractorIndicators: "Worker controls own schedule and time management",
  },
  work_location: {
    description: "Where the work is performed and who controls the work environment.",
    employeeIndicators: "Work performed at employer's premises",
    contractorIndicators: "Worker chooses own work location",
  },
  uniform_required: {
    description: "Whether the worker is required to wear a uniform or conform to a dress code.",
    employeeIndicators: "Required to wear uniform; strict dress code enforced",
    contractorIndicators: "No uniform required; worker chooses attire",
  },
};

// =====================
// Mock API Implementations
// =====================

export const mockApi = {
  // === Auth ===
  login: async (data: LoginRequest): Promise<{ user: User; token: string }> => {
    await randomDelay();
    simulateError();
    return {
      user: {
        id: generateId(),
        name: data.email.split("@")[0],
        email: data.email,
        tier: "free",
        createdAt: new Date().toISOString(),
      },
      token: "mock-jwt-token-" + generateId(),
    };
  },

  signup: async (data: SignupRequest): Promise<{ user: User; token: string }> => {
    await randomDelay();
    simulateError();
    return {
      user: {
        id: generateId(),
        name: data.name,
        email: data.email,
        tier: "free",
        createdAt: new Date().toISOString(),
      },
      token: "mock-jwt-token-" + generateId(),
    };
  },

  // === RAG Query ===
  query: async (question: string): Promise<RAGQueryResponse> => {
    await randomDelay();
    simulateError();

    const lower = question.toLowerCase();
    let matchedKey = "default";
    if (lower.includes("employee") || lower.includes("contractor") || lower.includes("sagaz")) {
      matchedKey = "employee";
    } else if (lower.includes("misclassif") || lower.includes("classif")) {
      matchedKey = "misclassification";
    } else if (lower.includes("notice") || lower.includes("wrongful") || lower.includes("termination")) {
      matchedKey = "notice";
    }

    const template = mockAnswers[matchedKey];
    const sources = template.sources.map((idx) => mockCases[idx]);

    return {
      query: question,
      answer: template.answer,
      confidence: template.confidence,
      sources,
      verification: {
        is_verified: Math.random() > 0.3,
        supported_claims: [
          "The Sagaz test remains the governing framework in Canada",
          "Control is a primary factor in determining employment status",
          "Each case must be assessed on its own facts",
        ],
        unsupported_claims: [],
        confidence_score: 0.87,
      },
    };
  },

  // === Classification ===
  classifyWithReasoning: async (data: ClassificationRequest): Promise<MCTSClassificationResponse> => {
    await delay(1000 + Math.random() * 1000); // Reasoning takes time
    
    // Count employee vs contractor factors
    const factorValues = [
      data.supervision_review,
      data.ability_hire,
      data.delegation_tasks,
      data.ownership_tools,
      data.chance_profit,
      data.risk_loss,
      data.exclusivity_services,
      data.work_hours_setter,
      data.work_location,
      data.uniform_required,
    ];
    
    const employeeCount = factorValues.filter((v) => v === "Employee").length;
    const contractorCount = factorValues.filter((v) => v === "Contractor").length;
    const ambiguousCount = factorValues.filter((v) => v === "Ambiguous").length;
    
    const employeeScore = (employeeCount + ambiguousCount * 0.4) / factorValues.length;
    const contractorScore = (contractorCount + ambiguousCount * 0.4) / factorValues.length;
    
    const isEmployee = employeeScore > contractorScore;
    const confidence = Math.max(employeeScore, contractorScore) * (0.7 + Math.random() * 0.2);
    
    const factorKeys = [
      "supervision_review", "ability_hire", "delegation_tasks", "ownership_tools",
      "chance_profit", "risk_loss", "exclusivity_services", "work_hours_setter",
      "work_location", "uniform_required"
    ];

    const factorAnalysis: MCTSClassificationResponse["factor_analysis"] = {};
    let reasoningParts: string[] = [];

    factorKeys.forEach((key, i) => {
      const val = factorValues[i];
      const weight = val === "Employee" ? (isEmployee ? 0.8 - Math.random() * 0.3 : 0.3 + Math.random() * 0.3)
        : val === "Contractor" ? (isEmployee ? 0.3 + Math.random() * 0.3 : 0.8 - Math.random() * 0.3)
        : 0.5 + Math.random() * 0.2;

      const displayName = key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
      const reasoning = val === "Employee"
        ? `Strong evidence of employment relationship: ${FACTOR_DETAILS[key].employeeIndicators.toLowerCase()}`
        : val === "Contractor"
        ? `Consistent with independent contractor status: ${FACTOR_DETAILS[key].contractorIndicators.toLowerCase()}`
        : `Ambiguous: requires further factual determination regarding ${FACTOR_DETAILS[key].description.toLowerCase()}`;

      factorAnalysis[key] = {
        value: val,
        weight: Math.round(weight * 100) / 100,
        reasoning,
      };
      reasoningParts.push(`**${displayName}**: ${reasoning} (weight: ${(weight * 100).toFixed(0)}%)`);
    });

    const jurisdiction = data.jurisdiction || "ON";
    const classification = isEmployee ? "Employee" as const : "Contractor" as const;

    const reasoningText = `# Worker Classification Analysis\n\n## Jurisdiction: ${jurisdiction}\n\n## Overall Assessment\n\nBased on the analysis of all 10 Sagaz factors, the worker is classified as **${classification}** with **${(confidence * 100).toFixed(0)}%** confidence.\n\n## Factor-by-Factor Analysis\n\n${reasoningParts.join("\n\n")}\n\n## Legal Context\n\nThis analysis applies the framework established in *671122 Ontario Ltd. v. Sagaz Industries Canada Inc.* (2001 SCC 59) and *McCormick v. Fasken Martineau DuMoulin LLP* (2014 SCC 39). The central inquiry is whether the worker is performing services as a person in business on their own account.\n\n## Risk Assessment\n\n${isEmployee ? `**HIGH RISK**: The factors strongly indicate an employment relationship. Misclassification could result in significant liability for employment standards violations, including unpaid wages, overtime, and termination pay.` : `**LOWER RISK**: The factors are more consistent with an independent contractor relationship. However, courts consider the totality of the relationship, so continued monitoring of the working arrangement is recommended.`}`;

    return {
      classification,
      confidence: Math.round(confidence * 100) / 100,
      factor_analysis: factorAnalysis,
      reasoning_text: reasoningText,
      tree_statistics: {
        total_nodes: 45 + Math.floor(Math.random() * 30),
        max_depth: 8 + Math.floor(Math.random() * 4),
        n_simulations: 20,
      },
      duration_ms: 1200 + Math.random() * 800,
    };
  },

  // === Dashboard Stats ===
  getUsageStats: async (): Promise<UsageStats> => {
    await randomDelay();
    return {
      queriesThisMonth: 14,
      queriesLimit: 200,
      documentsAnalyzed: 3,
      classificationsRun: 7,
      tier: "pro",
    };
  },

  getUsageChartData: async (days: number = 30): Promise<UsageDataPoint[]> => {
    await delay(200 + Math.random() * 200);
    const data: UsageDataPoint[] = [];
    const now = new Date();
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split("T")[0],
        queries: Math.floor(Math.random() * 15),
        classifications: Math.floor(Math.random() * 5),
      });
    }
    return data;
  },

  getRecentActivity: async (): Promise<RecentActivity[]> => {
    await randomDelay();
    const activities: RecentActivity[] = [
      { id: generateId(), type: "query", description: "What are the notice requirements for termination in Ontario?", timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString() },
      { id: generateId(), type: "classification", description: "Classified driver for LastMile Logistics", timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString() },
      { id: generateId(), type: "document", description: "Analyzed Employment Agreement - Acme Corp", timestamp: new Date(Date.now() - 1000 * 60 * 240).toISOString() },
      { id: generateId(), type: "query", description: "What is the test for employee vs independent contractor?", timestamp: new Date(Date.now() - 1000 * 60 * 360).toISOString() },
      { id: generateId(), type: "classification", description: "Classified software developer for TechStart Inc", timestamp: new Date(Date.now() - 1000 * 60 * 600).toISOString() },
      { id: generateId(), type: "query", description: "Requirements for enforceable non-compete clauses", timestamp: new Date(Date.now() - 1000 * 60 * 1440).toISOString() },
      { id: generateId(), type: "document", description: "Analyzed Independent Contractor Agreement", timestamp: new Date(Date.now() - 1000 * 60 * 2880).toISOString() },
      { id: generateId(), type: "query", description: "What is the duty of good faith in employment contracts?", timestamp: new Date(Date.now() - 1000 * 60 * 4320).toISOString() },
      { id: generateId(), type: "classification", description: "Classified sales representative for Northern Retail", timestamp: new Date(Date.now() - 1000 * 60 * 5760).toISOString() },
      { id: generateId(), type: "query", description: "Can a probationary period be extended beyond 3 months?", timestamp: new Date(Date.now() - 1000 * 60 * 7200).toISOString() },
    ];
    return activities;
  },

  // === Conversations ===
  getConversations: async (): Promise<Conversation[]> => {
    await delay(200);
    return [];
  },

  // === Document Analysis ===
  analyzeDocument: async (file: File): Promise<DocumentAnalysis> => {
    await delay(2000 + Math.random() * 1000);
    simulateError();
    return {
      id: generateId(),
      filename: file.name,
      status: "completed",
      extracted_text: "EMPLOYMENT AGREEMENT\n\nTHIS AGREEMENT is made on this 15th day of January, 2025.\n\nBETWEEN:\nACME CORPORATION (the \"Employer\")\n\nAND:\nJOHN DOE (the \"Employee\")\n\n1. The Employee agrees to provide services as a Senior Software Developer.\n2. The Employer shall pay a salary of $120,000 per annum.\n3. The Employee shall work from the Employer's office at 123 Main Street.\n4. The Employer shall provide all necessary equipment and tools.\n5. The Employee shall work standard business hours, Monday to Friday.\n\nThis agreement may be terminated with 4 weeks' notice by either party.",
      entities: [
        { name: "ACME CORPORATION", type: "organization", mentions: 4 },
        { name: "JOHN DOE", type: "person", mentions: 3 },
        { name: "Senior Software Developer", type: "person", mentions: 1 },
        { name: "Ontario Employment Standards Act", type: "statute", mentions: 0 },
      ],
      summary: "This is a standard employment agreement between ACME CORPORATION and JOHN DOE for the position of Senior Software Developer. Key terms include: annual salary of $120,000, office-based work, employer-provided equipment, standard business hours, and 4 weeks termination notice.",
      classification_analysis: {
        is_employment_related: true,
        prediction: "Employee",
        confidence: 0.94,
      },
      uploadedAt: new Date().toISOString(),
    };
  },

  // === API Keys ===
  getApiKeys: async (): Promise<ApiKey[]> => {
    await randomDelay();
    return [
      { id: generateId(), name: "Production", key: "oj-prod-" + generateId().substring(0, 8) + "...", createdAt: new Date(Date.now() - 86400000 * 30).toISOString(), lastUsed: new Date().toISOString() },
      { id: generateId(), name: "Development", key: "oj-dev-" + generateId().substring(0, 8) + "...", createdAt: new Date(Date.now() - 86400000 * 7).toISOString(), lastUsed: new Date(Date.now() - 86400000).toISOString() },
    ];
  },

  createApiKey: async (name: string): Promise<ApiKey> => {
    await delay(500);
    const fullKey = "oj-" + generateId().substring(0, 24);
    return {
      id: generateId(),
      name,
      key: fullKey,
      createdAt: new Date().toISOString(),
    };
  },

  revokeApiKey: async (id: string): Promise<void> => {
    await delay(300);
  },

  // === Membership/Subscription ===
  upgradeSubscription: async (tier: 'pro' | 'enterprise'): Promise<{ success: boolean; tier: string }> => {
    await delay(1000);
    return { success: true, tier };
  },
};
