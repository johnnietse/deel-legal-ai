# Evaluation Framework - Dynamic Benchmark Generator
"""
Generates dynamic, anti-contamination legal test cases for evaluating
legal AI systems.

The article's core insight:
"静态选择题评测集早就被刷爆了... 最缺的研究就是抗污染的动态评测机制
和特定领域的深度评测基准构建"

(Static benchmarks are saturated and contaminated. The field desperately
needs dynamic, anti-contamination evaluation frameworks.)

Key design principles:
1. PARAMETERIZED: Each test case is generated from a template with
   randomized parameters (names, dates, amounts, industries)
2. DETERMINISTIC: Same seed → same test case (reproducibility)
3. ANTI-CONTAMINATION: Lexically unique but semantically equivalent
   cases prevent test set leakage into training data
4. MULTI-DIMENSIONAL: Scores across 6 quality dimensions, not a
   single accuracy number
5. DIFFICULTY-CONTROLLED: Adjustable complexity levels for meaningful
   capability probing
"""

import sys
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EVAL_DIMENSIONS, EVAL_DEFAULT_N_CASES, EVAL_RESULTS_DIR

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================
# Parameterization Pools for Anti-Contamination
# ============================================================

WORKER_NAMES = [
    "Alex Chen", "Maria Santos", "James Okonkwo", "Priya Sharma", "Lucas Weber",
    "Fatima Al-Rahman", "David Kim", "Sophie Laurent", "Omar Hassan", "Elena Volkov",
    "Kenji Tanaka", "Amara Diallo", "Roberto Gonzalez", "Yuki Nakamura", "Isabella Rossi",
    "Henrik Johansson", "Nadia Petrov", "Kwame Asante", "Mei-Lin Chang", "Carlos Ferreira",
]

COMPANY_NAMES = [
    "TechNova Solutions", "Meridian Industries", "Apex Digital Corp", "Harbourside Services",
    "Pinnacle Consulting Group", "Evergreen Enterprises", "Quantum Logistics", "Starlight Media",
    "Atlas Construction Co.", "Silverline Staffing", "Cloudbridge Analytics", "Pacific Trade Ltd.",
    "Summit Healthcare Partners", "Ironwood Manufacturing", "Bluewave Technologies",
    "NorthStar Financial", "Redstone Energy", "Oakwood Property Management", "Coral Bay Imports",
    "Zenith Aerospace", "Granite Peak Mining", "Sagebrush Retail", "Lighthouse Education",
]

INDUSTRIES = [
    "software development", "financial consulting", "construction management",
    "healthcare administration", "digital marketing", "logistics coordination",
    "architectural design", "environmental consulting", "data analytics",
    "legal services", "manufacturing operations", "retail management",
    "telecommunications", "agricultural technology", "hospitality management",
]

JURISDICTIONS = ["ON", "BC", "AB", "QC", "NS", "MB", "SK", "NB", "NL", "PE"]

SALARY_RANGES = {
    "low": (30000, 50000),
    "mid": (50000, 90000),
    "high": (90000, 150000),
    "executive": (150000, 300000),
}

TENURE_RANGES = {
    "short": (1, 6),   # months
    "medium": (6, 36),  # months
    "long": (36, 120),  # months
    "very_long": (120, 360),  # months
}


# ============================================================
# Factor Templates for Scenario Generation
# ============================================================

FACTOR_TEMPLATES = {
    "control": {
        "strong_employee": [
            "{worker} reports to a direct manager at {company} who assigns daily tasks and reviews all completed work",
            "{worker}'s work schedule is set by {company} management with no flexibility",
            "{company} dictates the specific methods and procedures {worker} must follow",
        ],
        "strong_contractor": [
            "{worker} determines their own work methods and priorities without oversight from {company}",
            "{worker} sets their own schedule and {company} only reviews final deliverables",
            "{company} provides general project goals but {worker} decides how to achieve them",
        ],
        "ambiguous": [
            "{worker}'s manager at {company} sets weekly priorities but {worker} chooses daily tasks",
            "{company} requires weekly check-ins but {worker} otherwise works independently",
        ],
    },
    "ownership_of_tools": {
        "strong_employee": [
            "{company} provides all equipment including laptop, phone, and office space",
            "{worker} uses exclusively {company}-owned tools and technology",
        ],
        "strong_contractor": [
            "{worker} provides their own equipment, vehicle, and workspace",
            "{worker} invested ${tool_cost:,} in specialized tools for the work",
        ],
        "ambiguous": [
            "{company} provides a laptop but {worker} uses their own phone and vehicle",
            "{worker} uses some {company} equipment but also maintains their own tools",
        ],
    },
    "chance_of_profit": {
        "strong_employee": [
            "{worker} receives a fixed {pay_frequency} salary of ${salary:,} with no performance bonuses",
            "{worker}'s compensation is entirely fixed regardless of output quality or speed",
        ],
        "strong_contractor": [
            "{worker} negotiates per-project fees and earns more by working efficiently",
            "{worker} bills {company} at ${hourly_rate}/hour and can increase income by taking more projects",
        ],
        "ambiguous": [
            "{worker} has a base salary with occasional discretionary bonuses",
            "{worker} is paid a day rate but cannot negotiate the rate",
        ],
    },
    "risk_of_loss": {
        "strong_employee": [
            "{worker} bears no financial risk — {company} covers all expenses and errors",
            "{worker} is guaranteed their salary regardless of project outcomes",
        ],
        "strong_contractor": [
            "{worker} carries liability insurance and is responsible for fixing errors at own cost",
            "{worker} risks non-payment if {company} is unsatisfied with deliverables",
        ],
        "ambiguous": [
            "{worker} doesn't carry their own insurance but must redo subpar work unpaid",
            "{company} deducts for errors but {worker} has no other financial exposure",
        ],
    },
    "integration": {
        "strong_employee": [
            "{worker} has a {company} email address, attends all-hands meetings, and is listed on the org chart",
            "{worker} works exclusively for {company} and wears a company uniform",
        ],
        "strong_contractor": [
            "{worker} operates under their own business name and serves multiple clients",
            "{worker} has no {company} email, doesn't attend internal meetings, and invoices monthly",
        ],
        "ambiguous": [
            "{worker} has a company email for convenience but also works for other clients",
            "{worker} attends some team meetings but is not on the organizational chart",
        ],
    },
}


@dataclass
class TestCase:
    """A single dynamic test case for legal AI evaluation"""
    case_id: str
    seed: int
    difficulty: str  # "easy", "medium", "hard"
    scenario: str
    expected_classification: str  # "Employee" or "Independent Contractor"
    expected_factors: Dict[str, str]  # factor -> direction
    evaluation_rubric: Dict[str, str]  # dimension -> what correct looks like
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestSuite:
    """A collection of test cases forming a benchmark suite"""
    suite_id: str
    generated_at: str
    n_cases: int
    difficulty_distribution: Dict[str, int]
    cases: List[TestCase]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "generated_at": self.generated_at,
            "n_cases": self.n_cases,
            "difficulty_distribution": self.difficulty_distribution,
            "cases": [c.to_dict() for c in self.cases],
            "metadata": self.metadata,
        }
    
    def save(self, path: Optional[str] = None):
        """Save test suite to JSON"""
        if path is None:
            path = EVAL_RESULTS_DIR / f"benchmark_{self.suite_id}.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test suite saved to {path}")


@dataclass
class EvaluationResult:
    """Result of evaluating a single response against a test case"""
    case_id: str
    dimension_scores: Dict[str, float]  # dimension -> score [0, 1]
    overall_score: float
    feedback: Dict[str, str]  # dimension -> feedback text
    response_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalBenchmarkGenerator:
    """
    Generates dynamic, anti-contamination legal test cases.
    
    Each test case is parameterized — names, companies, industries,
    amounts, and dates are randomized while preserving the legal logic.
    This prevents test set contamination since no two generated suites
    are lexically identical.
    
    Difficulty levels:
    - EASY: All factors clearly point one direction
    - MEDIUM: Most factors agree but 1-2 are ambiguous
    - HARD: Factors are split or conflicting, requiring nuanced analysis
    """
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
    
    def generate_case(self, seed: int, difficulty: str = "medium") -> TestCase:
        """
        Generate a single parameterized worker classification test case.
        
        Args:
            seed: Random seed for deterministic generation
            difficulty: "easy", "medium", or "hard"
            
        Returns:
            TestCase with scenario, expected classification, and rubric
        """
        rng = random.Random(seed)
        
        # Select random parameters
        worker = rng.choice(WORKER_NAMES)
        company = rng.choice(COMPANY_NAMES)
        industry = rng.choice(INDUSTRIES)
        jurisdiction = rng.choice(JURISDICTIONS)
        salary_tier = rng.choice(list(SALARY_RANGES.keys()))
        salary = rng.randint(*SALARY_RANGES[salary_tier])
        tenure_tier = rng.choice(list(TENURE_RANGES.keys()))
        tenure_months = rng.randint(*TENURE_RANGES[tenure_tier])
        hourly_rate = rng.randint(25, 200)
        tool_cost = rng.randint(1000, 50000)
        
        # Determine factor directions based on difficulty
        factor_directions = self._generate_factor_directions(rng, difficulty)
        
        # Determine expected classification from factor directions
        employee_count = sum(1 for d in factor_directions.values() if d == "strong_employee")
        contractor_count = sum(1 for d in factor_directions.values() if d == "strong_contractor")
        
        if employee_count > contractor_count:
            expected = "Employee"
        elif contractor_count > employee_count:
            expected = "Independent Contractor"
        else:
            # Tie-break: control factor is traditionally most important
            expected = ("Employee" if factor_directions.get("control") == "strong_employee"
                       else "Independent Contractor")
        
        # Build scenario text from templates
        scenario_parts = [
            f"{worker} has been working in {industry} for {company} in {jurisdiction} "
            f"for {tenure_months} months.",
        ]
        
        for factor, direction in factor_directions.items():
            templates = FACTOR_TEMPLATES[factor][direction]
            template = rng.choice(templates)
            
            text = template.format(
                worker=worker,
                company=company,
                salary=salary,
                hourly_rate=hourly_rate,
                tool_cost=tool_cost,
                pay_frequency=rng.choice(["monthly", "bi-weekly"]),
            )
            scenario_parts.append(text)
        
        scenario = " ".join(scenario_parts)
        
        # Build evaluation rubric
        rubric = self._build_rubric(factor_directions, expected, worker, company)
        
        # Generate unique case ID
        case_hash = hashlib.md5(f"{seed}_{difficulty}".encode()).hexdigest()[:8]
        
        return TestCase(
            case_id=f"dyn_{difficulty}_{case_hash}",
            seed=seed,
            difficulty=difficulty,
            scenario=scenario,
            expected_classification=expected,
            expected_factors=factor_directions,
            evaluation_rubric=rubric,
            metadata={
                "worker": worker,
                "company": company,
                "industry": industry,
                "jurisdiction": jurisdiction,
                "salary": salary,
                "tenure_months": tenure_months,
            },
        )
    
    def _generate_factor_directions(
        self,
        rng: random.Random,
        difficulty: str,
    ) -> Dict[str, str]:
        """
        Generate factor directions based on difficulty level.
        
        - Easy: 4-5 factors clearly one direction, 0-1 ambiguous
        - Medium: 3 factors one direction, 1-2 ambiguous
        - Hard: 2-3 factors split, 1-2 ambiguous
        """
        factors = ["control", "ownership_of_tools", "chance_of_profit", "risk_of_loss", "integration"]
        directions = {}
        
        # Choose dominant direction
        dominant = rng.choice(["strong_employee", "strong_contractor"])
        opposite = "strong_contractor" if dominant == "strong_employee" else "strong_employee"
        
        if difficulty == "easy":
            # 4-5 factors clearly dominant
            n_dominant = rng.randint(4, 5)
            n_ambiguous = 5 - n_dominant
            n_opposite = 0
        elif difficulty == "medium":
            # 3 dominant, 1-2 ambiguous, 0-1 opposite
            n_dominant = 3
            n_ambiguous = rng.randint(1, 2)
            n_opposite = 5 - n_dominant - n_ambiguous
        else:  # hard
            # 2 dominant, 1-2 ambiguous, 1-2 opposite
            n_dominant = 2
            n_opposite = rng.randint(1, 2)
            n_ambiguous = 5 - n_dominant - n_opposite
        
        rng.shuffle(factors)
        
        for i, factor in enumerate(factors):
            if i < n_dominant:
                directions[factor] = dominant
            elif i < n_dominant + n_opposite:
                directions[factor] = opposite
            else:
                directions[factor] = "ambiguous"
        
        return directions
    
    def _build_rubric(
        self,
        factor_directions: Dict[str, str],
        expected: str,
        worker: str,
        company: str,
    ) -> Dict[str, str]:
        """Build evaluation rubric for multi-dimensional scoring."""
        
        strong_factors = [f for f, d in factor_directions.items() if "strong" in d]
        ambiguous_factors = [f for f, d in factor_directions.items() if d == "ambiguous"]
        
        return {
            "factor_identification": (
                f"Should identify all 5 Sagaz factors. Strong indicators present for: "
                f"{', '.join(strong_factors)}. Ambiguous factors: {', '.join(ambiguous_factors) or 'none'}."
            ),
            "legal_reasoning_quality": (
                f"Should apply the Sagaz test systematically. The {expected.lower()} "
                f"classification should follow logically from factor analysis. "
                f"Reasoning chain should reference specific facts from the scenario."
            ),
            "citation_accuracy": (
                "Should cite Sagaz Industries v. 671122 Ontario Ltd. (2001 SCC 59). "
                "May also cite Wiebe Door Services v. MNR. Citations should be to "
                "real cases with correct legal principles."
            ),
            "risk_assessment": (
                f"Should flag risks of misclassification. If classified as {expected}, "
                f"should note what happens if the classification is wrong (e.g., "
                f"back taxes, penalties, benefits liability)."
            ),
            "completeness": (
                "Should address all five Sagaz factors, provide a clear classification, "
                "and note any ambiguous factors that could support either direction."
            ),
            "hedging_appropriateness": (
                f"With {len(ambiguous_factors)} ambiguous factor(s), should express "
                f"appropriate uncertainty. Over-confidence on ambiguous factors is a flaw. "
                f"Under-confidence on clear factors is also a flaw."
            ),
        }
    
    def generate_suite(
        self,
        n_cases: int = EVAL_DEFAULT_N_CASES,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> TestSuite:
        """
        Generate a complete evaluation suite with controlled difficulty distribution.
        
        Args:
            n_cases: Number of test cases to generate
            difficulty_distribution: Proportion per difficulty level
                (default: 30% easy, 40% medium, 30% hard)
                
        Returns:
            TestSuite with all generated cases
        """
        dist = difficulty_distribution or {"easy": 0.3, "medium": 0.4, "hard": 0.3}
        
        # Calculate counts per difficulty
        counts = {}
        remaining = n_cases
        for diff, proportion in sorted(dist.items()):
            count = round(n_cases * proportion)
            counts[diff] = min(count, remaining)
            remaining -= counts[diff]
        
        # Distribute any remainder
        if remaining > 0:
            counts["medium"] = counts.get("medium", 0) + remaining
        
        # Generate cases
        cases = []
        seed = self.base_seed
        
        for difficulty, count in counts.items():
            for _ in range(count):
                case = self.generate_case(seed, difficulty)
                cases.append(case)
                seed += 1
        
        # Shuffle cases
        rng = random.Random(self.base_seed)
        rng.shuffle(cases)
        
        suite_hash = hashlib.md5(f"{self.base_seed}_{n_cases}".encode()).hexdigest()[:8]
        
        return TestSuite(
            suite_id=f"legal_bench_{suite_hash}",
            generated_at=datetime.now().isoformat(),
            n_cases=len(cases),
            difficulty_distribution=counts,
            cases=cases,
            metadata={
                "base_seed": self.base_seed,
                "requested_n": n_cases,
                "dimensions": EVAL_DIMENSIONS,
            },
        )


class MultiDimensionalEvaluator:
    """
    Evaluates legal AI responses across multiple quality dimensions.
    
    Instead of a single accuracy score, evaluates on 6 dimensions:
    1. factor_identification - Did it identify all relevant Sagaz factors?
    2. legal_reasoning_quality - Is the reasoning chain logically sound?
    3. citation_accuracy - Are cited precedents real and relevant?
    4. risk_assessment - Are misclassification risks properly flagged?
    5. completeness - Does it cover all aspects of the question?
    6. hedging_appropriateness - Appropriate uncertainty for ambiguous factors?
    
    Decomposition approach (from article):
    "不是让裁判模型直接给个综合分数，而是先提取文章的核心论点、
    逻辑推导链条、市场数据支撑，然后针对每一个细分的叶子节点
    进行严格的规则化打分"
    """
    
    def __init__(self, judge_chat=None):
        from rag_pipeline.embeddings import GeminiChat
        self.judge = judge_chat or GeminiChat()
    
    def evaluate(
        self,
        response: str,
        test_case: TestCase,
    ) -> EvaluationResult:
        """
        Score a legal AI response across all quality dimensions.
        
        Uses structured rubric decomposition: each dimension is scored
        independently against the test case's ground truth rubric.
        """
        dimension_scores = {}
        feedback = {}
        
        for dimension in EVAL_DIMENSIONS:
            rubric = test_case.evaluation_rubric.get(dimension, "")
            score, dim_feedback = self._score_dimension(
                response, test_case.scenario, dimension, rubric,
                test_case.expected_classification,
            )
            dimension_scores[dimension] = score
            feedback[dimension] = dim_feedback
        
        # Overall score is the weighted average
        # All dimensions equally weighted by default
        overall = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.0
        
        return EvaluationResult(
            case_id=test_case.case_id,
            dimension_scores=dimension_scores,
            overall_score=round(overall, 4),
            feedback=feedback,
            response_text=response[:500],
        )
    
    def _score_dimension(
        self,
        response: str,
        scenario: str,
        dimension: str,
        rubric: str,
        expected_classification: str,
    ) -> Tuple[float, str]:
        """Score a single evaluation dimension using structured judging."""
        
        prompt = f"""You are evaluating a legal AI system's response on the dimension: {dimension.replace('_', ' ').upper()}.

SCENARIO GIVEN TO THE AI:
{scenario[:500]}

EXPECTED CLASSIFICATION: {expected_classification}

EVALUATION RUBRIC FOR THIS DIMENSION:
{rubric}

AI SYSTEM'S RESPONSE:
{response[:1500]}

Score the response on this specific dimension from 0.0 to 1.0:
- 0.0-0.2: Completely fails this dimension
- 0.2-0.4: Partially addresses but with significant issues
- 0.4-0.6: Adequate but with notable gaps
- 0.6-0.8: Good with minor issues
- 0.8-1.0: Excellent, fully meets or exceeds the rubric

Respond in EXACTLY this JSON format:
{{
    "score": <float 0.0-1.0>,
    "feedback": "<specific feedback for this dimension, 1-2 sentences>"
}}"""
        
        try:
            result = self.judge.generate(prompt, temperature=0.1, max_tokens=256)
            
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            fb = data.get("feedback", "No feedback provided")
            
            return score, fb
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Dimension scoring error for {dimension}: {e}")
            return 0.5, f"Scoring error: {str(e)[:100]}"


def main():
    """Test the benchmark generator"""
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC BENCHMARK GENERATOR")
    print("=" * 60)
    
    generator = LegalBenchmarkGenerator(base_seed=42)
    
    # Generate individual cases at each difficulty
    for difficulty in ["easy", "medium", "hard"]:
        case = generator.generate_case(seed=42, difficulty=difficulty)
        print(f"\n--- {difficulty.upper()} CASE ---")
        print(f"  ID: {case.case_id}")
        print(f"  Expected: {case.expected_classification}")
        print(f"  Factors: {case.expected_factors}")
        print(f"  Scenario: {case.scenario[:200]}...")
    
    # Generate a suite
    suite = generator.generate_suite(n_cases=10)
    print(f"\n--- TEST SUITE ---")
    print(f"  ID: {suite.suite_id}")
    print(f"  Cases: {suite.n_cases}")
    print(f"  Distribution: {suite.difficulty_distribution}")
    
    # Save
    suite.save()
    print(f"  ✅ Saved to {EVAL_RESULTS_DIR}")
    
    # Test anti-contamination: same seed should produce same case
    case1 = generator.generate_case(seed=100, difficulty="medium")
    case2 = generator.generate_case(seed=100, difficulty="medium")
    assert case1.scenario == case2.scenario, "Determinism check failed!"
    print("\n  ✅ Determinism check passed")
    
    # Different seeds should produce different cases
    case3 = generator.generate_case(seed=101, difficulty="medium")
    assert case1.scenario != case3.scenario, "Uniqueness check failed!"
    print("  ✅ Uniqueness check passed")


if __name__ == "__main__":
    main()
