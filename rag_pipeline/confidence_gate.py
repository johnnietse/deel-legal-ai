# RAG Pipeline - Confidence Gate (Pre-Generation Hallucination Guard)
"""
Post-generation confidence check inspired by ByteDance RAG Guideline §6.3.1.

Strategies implemented:
  1. Semantic similarity gate — if generated answer has low similarity to
     source documents, flag as low-confidence and trigger refusal/hedging
  2. Legal citation validator — checks that cited case names and statute
     references actually appear in the source documents
  3. Confidence score aggregation — combines similarity + citation checks
     into a single 0.0–1.0 confidence score

This is a LIGHTWEIGHT guard that runs AFTER generation but BEFORE the
response reaches the user. The heavy-duty verification remains in
verifier.py for when `verify=True` is explicitly requested.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    """Result of the confidence gate check."""
    passed: bool                    # True if answer meets confidence threshold
    confidence_score: float         # 0.0 to 1.0
    source_similarity: float        # avg similarity between answer and sources
    citation_accuracy: float        # fraction of citations found in sources
    unmatched_citations: List[str]  # citations in answer not in sources
    action_taken: str               # "pass", "hedge", "refuse"
    modified_answer: Optional[str]  # hedged/refused answer (None if passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence_score": round(self.confidence_score, 3),
            "source_similarity": round(self.source_similarity, 3),
            "citation_accuracy": round(self.citation_accuracy, 3),
            "unmatched_citations": self.unmatched_citations,
            "action_taken": self.action_taken,
        }


class ConfidenceGate:
    """
    Lightweight post-generation confidence check.

    ByteDance §6.3.1 low-confidence refusal:
      - If semantic similarity between answer and sources < threshold → refuse
      - If critical legal citations don't match sources → hedge

    Thresholds:
      - refuse_threshold: below this → refuse to answer (default 0.3)
      - hedge_threshold: below this → add hedging language (default 0.5)
    """

    # Legal citation patterns to extract from text
    CITATION_PATTERNS = [
        r'\d{4}\s+[A-Z]{2,6}\s+\d+',               # "2020 ONSC 1234"
        r'[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+',         # "Sagaz v. 671122"
        r'(?:s\.|[Ss]ection)\s*\d+(?:\.\d+)*',        # "s. 56" or "Section 56.1"
    ]

    def __init__(
        self,
        refuse_threshold: float = 0.3,
        hedge_threshold: float = 0.5,
        embeddings=None,
    ):
        self.refuse_threshold = refuse_threshold
        self.hedge_threshold = hedge_threshold
        self.embeddings = embeddings  # optional; used for similarity check

    def check(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        query: str = "",
    ) -> ConfidenceReport:
        """
        Run the confidence gate on a generated answer.

        Args:
            answer: The generated response text
            sources: List of source dicts with 'content'/'excerpt' fields
            query: Original user query (for context)

        Returns:
            ConfidenceReport with pass/fail and any modifications
        """
        # Step 1: Source text similarity (word overlap based; fast)
        source_similarity = self._compute_source_overlap(answer, sources)

        # Step 2: Citation accuracy
        citation_accuracy, unmatched = self._check_citations(answer, sources)

        # Step 3: Aggregate confidence score
        # Weight: 60% source overlap, 40% citation accuracy
        if citation_accuracy is not None:
            confidence = 0.6 * source_similarity + 0.4 * citation_accuracy
        else:
            confidence = source_similarity  # no citations to check

        # Step 4: Decide action
        if confidence < self.refuse_threshold:
            action = "refuse"
            modified = self._generate_refusal(query)
            passed = False
        elif confidence < self.hedge_threshold:
            action = "hedge"
            modified = self._add_hedging(answer, unmatched)
            passed = False
        else:
            action = "pass"
            modified = None
            passed = True

        logger.info(
            f"Confidence gate: score={confidence:.2f}, "
            f"similarity={source_similarity:.2f}, "
            f"citation_acc={citation_accuracy}, action={action}"
        )

        return ConfidenceReport(
            passed=passed,
            confidence_score=confidence,
            source_similarity=source_similarity,
            citation_accuracy=citation_accuracy if citation_accuracy is not None else 1.0,
            unmatched_citations=unmatched,
            action_taken=action,
            modified_answer=modified,
        )

    # -- Internal methods --------------------------------------------------

    def _compute_source_overlap(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> float:
        """
        Compute word-level overlap between the answer and source documents.
        
        Returns a score in [0, 1] where 1 means every substantive word
        in the answer also appears in the sources.
        """
        # Combine all source text
        source_text = " ".join(
            s.get("excerpt", s.get("content", ""))
            for s in sources
        ).lower()
        source_words = set(re.findall(r'\b[a-z]{3,}\b', source_text))

        if not source_words:
            return 0.0

        # Get answer words (skip common stop words)
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "this", "that", "with", "have", "from", "they", "been",
            "said", "each", "which", "their", "will", "other",
            "about", "would", "these", "than", "its", "also",
        }
        answer_words = set(re.findall(r'\b[a-z]{3,}\b', answer.lower()))
        answer_words -= stop_words

        if not answer_words:
            return 1.0  # vacuously true for empty answer

        overlap = answer_words & source_words
        return len(overlap) / len(answer_words)

    def _check_citations(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], List[str]]:
        """
        Check that legal citations in the answer exist in the sources.

        Returns (accuracy_score, list_of_unmatched_citations).
        accuracy_score is None if no citations were found in the answer.
        """
        # Extract citations from answer
        answer_citations = set()
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, answer)
            answer_citations.update(m.strip() for m in matches)

        if not answer_citations:
            return None, []

        # Extract citations from sources
        source_text = " ".join(
            s.get("excerpt", s.get("content", ""))
            for s in sources
        )
        source_citations = set()
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, source_text)
            source_citations.update(m.strip() for m in matches)

        # Check which answer citations are in sources
        matched = 0
        unmatched = []
        for cite in answer_citations:
            if any(cite.lower() in sc.lower() for sc in source_citations):
                matched += 1
            else:
                # Also check if the citation text appears anywhere in source text
                if cite.lower() in source_text.lower():
                    matched += 1
                else:
                    unmatched.append(cite)

        accuracy = matched / len(answer_citations) if answer_citations else 1.0
        return accuracy, unmatched

    def _generate_refusal(self, query: str) -> str:
        """Generate a low-confidence refusal response."""
        return (
            "⚠️ **Low Confidence Response**\n\n"
            "The available legal sources may not adequately address your question. "
            "The retrieved documents do not contain sufficient information to provide "
            "a reliable answer.\n\n"
            "**Recommendations:**\n"
            "1. Try rephrasing your question with more specific legal terms\n"
            "2. Specify the jurisdiction (e.g., Ontario, British Columbia)\n"
            "3. Reference specific statutes or case names if applicable\n\n"
            f"_Original question: {query}_"
        )

    def _add_hedging(self, answer: str, unmatched_citations: List[str]) -> str:
        """Add hedging language to a moderate-confidence answer."""
        hedging = "\n\n---\n⚠️ **Confidence Notice**: "
        parts = []

        if unmatched_citations:
            cites = ", ".join(unmatched_citations)
            parts.append(
                f"The following citations could not be verified against the "
                f"retrieved sources: {cites}. Please verify independently."
            )

        parts.append(
            "Some portions of this response may not be fully supported by the "
            "retrieved documents. Cross-reference with primary legal sources before relying on this analysis."
        )

        return answer + hedging + " ".join(parts)
