# RAG Pipeline - Keyword Booster (RAGFlow-inspired)
"""
Rule-based legal keyword boosting for BM25 scores.

Extracts legal boost terms (citations, statute/section refs, acronyms,
case names) from a query, then multiplies BM25 scores of documents whose
content or metadata contains any boost term.

Pure functions only: no I/O, no global state mutation.
Dependencies: stdlib (re, typing) + config.
"""

import re
from typing import List, Dict, Any

from config import KEYWORD_BOOST_TERM_PATTERNS


def extract_boost_terms(text: str) -> List[str]:
    """
    Extract legal boost terms from a query using config patterns.

    Patterns (from config.KEYWORD_BOOST_TERM_PATTERNS):
      - Legal citations: "2020 ONSC 1234"
      - Statute/section refs: "s. 56", "s. 5(1)", "Section 56"
      - Legal acronyms: ESA, OHSA, CLC, SCC, HRC
      - Case names: "Sagaz v. 671122"

    Returns a deduplicated list of lowercase terms, preserving first-seen order.
    """
    terms: List[str] = []
    seen = set()

    for pattern in KEYWORD_BOOST_TERM_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group(0).strip().lower()
            if term and term not in seen:
                seen.add(term)
                terms.append(term)

    return terms


def _doc_contains_term(doc: Dict[str, Any], term: str) -> bool:
    """True if a boost term appears in the doc's content or metadata values."""
    content = doc.get("content", "")
    if isinstance(content, str) and term in content.lower():
        return True

    metadata = doc.get("metadata", {})
    if isinstance(metadata, dict):
        for value in metadata.values():
            if isinstance(value, str) and term in value.lower():
                return True

    return False


def apply_boost(
    bm25_scores: List[float],
    docs: List[Dict[str, Any]],
    boost_terms: List[str],
    multiplier: float,
) -> List[float]:
    """
    Multiply BM25 scores of docs containing any boost term.

    For each doc, if any boost term appears as a case-insensitive substring
    in its content string or metadata values, its score is multiplied by
    `multiplier`. Docs without matches are left unchanged.

    Pure: returns a new score list; never mutates inputs.
    """
    if not boost_terms or multiplier == 1.0:
        return list(bm25_scores)

    boosted: List[float] = []
    for idx, score in enumerate(bm25_scores):
        if idx >= len(docs):
            boosted.append(score)
            continue
        if any(_doc_contains_term(docs[idx], term) for term in boost_terms):
            boosted.append(score * multiplier)
        else:
            boosted.append(score)

    return boosted