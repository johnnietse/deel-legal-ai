# RAG Pipeline - Prompt Template Library
"""
Structured prompt templates for legal RAG generation,
inspired by ByteDance RAG Guideline §6.2.

Key design (ByteDance §6.2.1 five-part template):
  1. Role Definition — establishes the persona
  2. Task Instruction — what to do with the query
  3. Retrieved Information — structured source injection
  4. Format Requirements — output structure constraints
  5. Few-shot Examples — demonstration of ideal output

Additional features:
  - Dynamic template selection based on query intent
  - Query-information alignment for multi-intent queries (§6.2.2)
  - Context compression guidance (§6.2.2)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """A structured prompt template for legal RAG."""
    name: str
    role_definition: str
    task_instruction: str
    format_requirements: str
    few_shot_examples: str
    fact_anchoring_instruction: str = ""

    def build(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        max_sources: int = 5,
        min_similarity: float = 0.0,
    ) -> Tuple[str, str]:
        """
        Build the full prompt and system instruction.

        Args:
            query: User question
            sources: Retrieval results (must have 'content' or 'excerpt')
            max_sources: Maximum sources to include
            min_similarity: Minimum score to keep a source

        Returns:
            (system_instruction, user_prompt)
        """
        # Filter and limit sources
        filtered = [
            s for s in sources
            if s.get("score", 1.0) >= min_similarity
        ][:max_sources]

        # Format sources block
        source_block = self._format_sources(filtered)

        # Build system instruction
        system = f"""{self.role_definition}

{self.fact_anchoring_instruction}"""

        # Build user prompt
        prompt = f"""{self.task_instruction}

{source_block}

USER QUESTION: {query}

{self.format_requirements}

{self.few_shot_examples}

ANSWER:"""

        return system.strip(), prompt.strip()

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source documents for prompt injection."""
        if not sources:
            return "RETRIEVED INFORMATION:\n[No relevant sources found]"

        parts = ["RETRIEVED INFORMATION:"]
        for i, src in enumerate(sources):
            content = src.get("excerpt", src.get("content", ""))
            citation = src.get("citation", src.get("case_name", f"Source {i+1}"))
            score = src.get("score", 0.0)

            header = f"[Source {i+1}: {citation}]"
            if score:
                header += f" (relevance: {score:.2f})"
            parts.append(f"\n{header}\n{content}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Built-In Templates
# ---------------------------------------------------------------------------

# Shared fact-anchoring instruction (ByteDance §6.3.1)
_FACT_ANCHOR = """CRITICAL RULES FOR FACTUAL ACCURACY:
1. Every factual claim in your answer MUST be traceable to a specific [Source N].
2. If no source supports a claim, DO NOT include it. Instead write: "The available sources do not address this point."
3. After drafting your answer, self-check:
   - Are there any dates, amounts, legal test names, or case citations that don't appear in the sources?
   - Have you assumed any jurisdictional applicability not stated in the sources?
   - If any claim lacks source support, remove it or hedge it explicitly.
4. Cite sources inline using [Source N] notation."""


WORKER_CLASSIFICATION = PromptTemplate(
    name="worker_classification",
    role_definition="""You are a senior employment law analyst specializing in worker classification for the Deel Lab for Global Employment Law.
You provide rigorous, evidence-based analysis grounded strictly in the retrieved case law and statutes.""",

    task_instruction="""Analyze the following worker classification question using the retrieved legal sources.
Apply the relevant legal tests (e.g., Sagaz Industries four-factor test, Wiebe Door test) and identify which factors apply.""",

    format_requirements="""FORMAT REQUIREMENTS:
1. Start with a clear classification determination (Employee / Independent Contractor / Ambiguous)
2. Present analysis as a structured factor table:

   | Factor | Evidence | Supports |
   |--------|----------|----------|
   | Control over work | [specific evidence from sources] | Employee / IC |
   | Ownership of tools | ... | ... |
   | Chance of profit / Risk of loss | ... | ... |
   | Integration into business | ... | ... |

3. After the table, provide 2-3 paragraphs of legal reasoning citing specific sources
4. End with a "Risk Assessment" section noting misclassification risks
5. Keep total response under 800 words""",

    few_shot_examples="""EXAMPLE:
User Question: Is a software developer who works exclusively for one client, uses the client's laptop, and invoices monthly an employee or independent contractor?

Sample Answer Structure:
**Classification Determination**: Likely Employee

| Factor | Evidence | Supports |
|--------|----------|----------|
| Control | Client determines work hours and project scope [Source 1] | Employee |
| Tools | Client provides laptop and software licenses [Source 2] | Employee |
| Profit/Loss | Fixed monthly invoicing with no chance of profit variation [Source 1] | Employee |
| Integration | Works exclusively for one client, fully embedded in team [Source 3] | Employee |

**Legal Analysis**: Under the Sagaz test established by the Supreme Court of Canada [Source 1], the central question is...

**Risk Assessment**: Given the strong indicators of employment, misclassification risk is HIGH...""",

    fact_anchoring_instruction=_FACT_ANCHOR,
)


NOTICE_PERIOD = PromptTemplate(
    name="notice_period_analysis",
    role_definition="""You are a senior employment law analyst specializing in wrongful dismissal and reasonable notice periods for the Deel Lab for Global Employment Law.
You provide precise, precedent-based analysis grounded strictly in retrieved case law.""",

    task_instruction="""Analyze the reasonable notice period question using the retrieved legal sources.
Apply the Bardal factors and identify comparable precedent cases.""",

    format_requirements="""FORMAT REQUIREMENTS:
1. State the likely notice period range (in months) based on the Bardal factors
2. Present analysis using the Bardal factors:
   - Character of employment (position, seniority)
   - Length of service
   - Age of the employee
   - Availability of similar employment
3. Include a "Comparable Cases" table:

   | Case | Service | Age | Position | Notice Awarded |
   |------|---------|-----|----------|----------------|
   | [case from sources] | ... | ... | ... | ... months |

4. Discuss any aggravating factors (bad faith, manner of dismissal)
5. Keep total response under 600 words""",

    few_shot_examples="",  # Notice period analysis is standard enough

    fact_anchoring_instruction=_FACT_ANCHOR,
)


GENERAL_LEGAL_QA = PromptTemplate(
    name="general_legal_qa",
    role_definition="""You are a legal research assistant for the Deel Lab for Global Employment Law.
You provide accurate, well-cited answers based strictly on the provided legal context.
You never fabricate case names, citations, or legal principles.""",

    task_instruction="""Answer the following legal question using ONLY the retrieved information.
If the sources are insufficient, clearly state what information is missing.""",

    format_requirements="""FORMAT REQUIREMENTS:
1. Provide a clear, direct answer in the first paragraph
2. Support each claim with inline citations [Source N]
3. Use structured formatting (numbered lists, bold headers) for readability
4. If multiple legal tests or standards apply, present them separately
5. If the question spans multiple jurisdictions, address each separately
6. End with any caveats or limitations of the analysis
7. Keep total response under 500 words""",

    few_shot_examples="",

    fact_anchoring_instruction=_FACT_ANCHOR,
)


RISK_ASSESSMENT = PromptTemplate(
    name="risk_assessment",
    role_definition="""You are a compliance risk analyst for the Deel Lab for Global Employment Law.
You assess legal risks related to employment practices, classification, and regulatory compliance.
Your analysis must be grounded exclusively in the retrieved legal sources.""",

    task_instruction="""Assess the legal risks described in the question using the retrieved sources.
Identify specific regulatory, financial, and reputational risks.""",

    format_requirements="""FORMAT REQUIREMENTS:
1. Start with an overall risk level: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
2. Present a risk matrix:

   | Risk Category | Description | Likelihood | Impact | Mitigation |
   |---------------|-------------|------------|--------|------------|
   | Regulatory | ... | High/Med/Low | ... | ... |
   | Financial | ... | ... | ... | ... |
   | Reputational | ... | ... | ... | ... |

3. Cite relevant statutes and case law from the sources
4. Provide 2-3 specific mitigation recommendations
5. Keep total response under 600 words""",

    few_shot_examples="",

    fact_anchoring_instruction=_FACT_ANCHOR,
)


# ---------------------------------------------------------------------------
# Template Library
# ---------------------------------------------------------------------------

class PromptTemplateLibrary:
    """
    Registry of prompt templates with auto-selection by query intent.

    ByteDance §6.2.2 dynamic prompt selection:
    Classify query → match template → fill variables → send to LLM.
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {
            "worker_classification": WORKER_CLASSIFICATION,
            "notice_period": NOTICE_PERIOD,
            "general_legal_qa": GENERAL_LEGAL_QA,
            "risk_assessment": RISK_ASSESSMENT,
        }

    def register(self, template: PromptTemplate):
        """Add a custom template to the library."""
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())

    def auto_select(self, query: str) -> PromptTemplate:
        """
        Automatically select the best template based on query intent.

        Uses rule-based classification; could be upgraded to LLM-based.
        """
        query_lower = query.lower()

        # Worker classification detection
        classification_indicators = [
            "employee or independent contractor",
            "worker classification", "classify", "misclassification",
            "sagaz", "wiebe door", "employment status",
            "contractor or employee", "independent contractor",
            "gig worker", "dependent contractor",
        ]
        if any(ind in query_lower for ind in classification_indicators):
            return self._templates["worker_classification"]

        # Notice period detection
        notice_indicators = [
            "notice period", "reasonable notice", "wrongful dismissal",
            "termination notice", "bardal", "severance",
            "dismissed without", "terminated without cause",
            "how much notice", "length of notice",
        ]
        if any(ind in query_lower for ind in notice_indicators):
            return self._templates["notice_period"]

        # Risk assessment detection
        risk_indicators = [
            "risk", "compliance", "penalty", "fine",
            "liability", "exposure", "regulatory risk",
            "what happens if", "consequences of",
            "misclassification risk",
        ]
        if any(ind in query_lower for ind in risk_indicators):
            return self._templates["risk_assessment"]

        # Default to general Q&A
        return self._templates["general_legal_qa"]

    def build_prompt(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        template_name: Optional[str] = None,
        max_sources: int = 5,
    ) -> Tuple[str, str]:
        """
        Auto-select template and build the full prompt.

        Args:
            query: User question
            sources: Retrieval results
            template_name: Force a specific template (None = auto-select)
            max_sources: Max sources to include in prompt

        Returns:
            (system_instruction, user_prompt)
        """
        if template_name:
            template = self._templates.get(template_name, self._templates["general_legal_qa"])
        else:
            template = self.auto_select(query)

        logger.info(f"Selected prompt template: {template.name}")
        return template.build(query, sources, max_sources=max_sources)
