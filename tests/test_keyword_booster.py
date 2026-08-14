# Tests for keyword_booster module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rag_pipeline.keyword_booster import extract_boost_terms, apply_boost


class TestExtractBoostTerms:
    """Tests for extract_boost_terms function."""

    def test_extract_citation(self):
        """Test extraction of legal citation pattern."""
        terms = extract_boost_terms("2020 ONSC 1234")
        assert "2020 onsc 1234" in terms

    def test_extract_statute_section(self):
        """Test extraction of statute section references."""
        terms = extract_boost_terms("What is the notice period under ESA s. 56?")
        assert "esa" in terms
        assert "s. 56" in terms

    def test_extract_statute_section_parentheses(self):
        """Test extraction of section with parentheses s. 5(1).
        
        Note: Current pattern r"\b[Ss]\.?\s*\d+(\.\d+)*" captures "s. 5" 
        but not parentheses. This is a known limitation.
        """
        terms = extract_boost_terms("ESA s. 5(1) defines employer")
        assert "esa" in terms
        # Pattern matches "s. 5" (digits before parenthesis)
        assert "s. 5" in terms

    def test_extract_section_keyword(self):
        """Test extraction of 'Section N' pattern."""
        terms = extract_boost_terms("Section 56 of the Act")
        assert "section 56" in terms

    def test_extract_acronyms(self):
        """Test extraction of legal acronyms."""
        terms = extract_boost_terms("ESA OHSA CLC SCC HRC")
        assert "esa" in terms
        assert "ohsa" in terms
        assert "clc" in terms
        assert "scc" in terms
        assert "hrc" in terms

    def test_extract_case_name(self):
        """Test extraction of case name pattern.
        
        Note: Pattern r"\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+" requires alphabetic 
        party names. "Sagaz v. 671122" (numeric respondent) won't match.
        """
        terms = extract_boost_terms("Sagaz v. Jones")
        assert "sagaz v. jones" in terms

    def test_extract_case_name_with_dot(self):
        """Test extraction of case name with dot."""
        terms = extract_boost_terms("Smith v. Jones")
        assert "smith v. jones" in terms

    def test_deduplication_preserves_order(self):
        """Test that deduplication preserves first-seen order (pattern order, not text order).
        
        Note: Terms are extracted by iterating patterns in config order, not by 
        first appearance in text. First pattern that matches "wins" ordering.
        """
        terms = extract_boost_terms("ESA s. 56 ESA s. 56")
        # Pattern order: citation, section, section keyword, acronyms, case name
        # "s. 56" matches section pattern (2nd), "esa" matches acronym pattern (4th)
        assert "esa" in terms
        assert "s. 56" in terms
        assert len(terms) == 2

    def test_empty_string_returns_empty(self):
        """Test empty input returns empty list."""
        assert extract_boost_terms("") == []

    def test_no_matches_returns_empty(self):
        """Test input with no legal patterns returns empty list."""
        assert extract_boost_terms("hello world") == []


class TestApplyBoost:
    """Tests for apply_boost function."""

    def test_boost_applies_to_matching_content(self):
        """Test that boost multiplies score when term in content."""
        scores = [1.0, 2.0, 3.0]
        docs = [
            {"content": "ESA s. 56 notice period", "metadata": {}},
            {"content": "OHSA workplace safety", "metadata": {}},
            {"content": "random text", "metadata": {}},
        ]
        boosted = apply_boost(scores, docs, ["esa", "s. 56"], 5.0)
        # First doc contains both esa and s.56 → boosted by 5x
        assert boosted[0] == 5.0  # 1.0 * 5
        # Second doc has no boost terms
        assert boosted[1] == 2.0
        # Third doc has no boost terms
        assert boosted[2] == 3.0

    def test_boost_applies_to_metadata(self):
        """Test that boost applies when term in metadata values."""
        scores = [1.0, 1.0]
        docs = [
            {"content": "some text", "metadata": {"court": "ONSC"}},
            {"content": "other text", "metadata": {"court": "SCC"}},
        ]
        boosted = apply_boost(scores, docs, ["onsc"], 3.0)
        assert boosted[0] == 3.0
        assert boosted[1] == 1.0

    def test_no_terms_returns_original(self):
        """Test empty boost_terms returns original scores."""
        scores = [1.0, 2.0, 3.0]
        docs = [{"content": "a", "metadata": {}} for _ in range(3)]
        boosted = apply_boost(scores, docs, [], 5.0)
        assert boosted == scores

    def test_multiplier_one_returns_original(self):
        """Test multiplier 1.0 returns original scores."""
        scores = [1.0, 2.0, 3.0]
        docs = [{"content": "ESA", "metadata": {}} for _ in range(3)]
        boosted = apply_boost(scores, docs, ["esa"], 1.0)
        assert boosted == scores

    def test_input_not_mutated(self):
        """Test that input scores list is not mutated."""
        scores = [1.0, 2.0]
        docs = [{"content": "ESA", "metadata": {}}, {"content": "OHSA", "metadata": {}}]
        original_scores = scores.copy()
        apply_boost(scores, docs, ["esa"], 5.0)
        assert scores == original_scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])