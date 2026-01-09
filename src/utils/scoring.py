"""
Scoring and aggregation utilities for ProWriteBench.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Score(BaseModel):
    """Represents evaluation scores for a single dimension."""
    dimension: str
    score: float  # 0-100
    weight: float  # Weight in final score calculation
    details: Dict[str, any] = Field(default_factory=dict)
    passed: bool = True  # Whether this dimension passed (for constraints)


class ScoreAggregator:
    """Aggregates scores from multiple evaluation dimensions."""

    # Default weights for each evaluation dimension
    DEFAULT_WEIGHTS = {
        "constraint_satisfaction": 0.30,
        "audience_clarity": 0.25,
        "stakeholder_balance": 0.20,
        "professional_appropriateness": 0.15,
        "revision_coherence": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize score aggregator.

        Args:
            weights: Custom weights for each dimension (default: DEFAULT_WEIGHTS)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def aggregate(self, scores: List[Score]) -> Dict:
        """
        Aggregate multiple dimension scores into final score.

        Args:
            scores: List of Score objects from different evaluators

        Returns:
            Dictionary with:
                - overall_score: Final weighted score (0-100)
                - dimension_scores: Individual dimension scores
                - critical_failures: List of critical failures
                - passed: Whether all constraints were satisfied
        """
        # Calculate weighted score
        weighted_sum = 0.0
        dimension_scores = {}
        critical_failures = []
        all_passed = True

        for score in scores:
            weight = self.weights.get(score.dimension, 0.0)
            weighted_sum += score.score * weight

            dimension_scores[score.dimension] = {
                "score": score.score,
                "weight": weight,
                "passed": score.passed,
                "details": score.details,
            }

            if not score.passed:
                all_passed = False
                if "critical_failure" in score.details:
                    critical_failures.extend(score.details["critical_failure"])

        overall_score = weighted_sum

        # Apply penalty multiplier for critical failures
        penalty_multiplier = 0.5 ** len(critical_failures)
        if critical_failures:
            overall_score *= penalty_multiplier

        return {
            "overall_score": round(overall_score, 2),
            "dimension_scores": dimension_scores,
            "critical_failures": critical_failures,
            "penalty_multiplier": penalty_multiplier if critical_failures else 1.0,
            "passed": all_passed,
        }

    def aggregate_multiple_tasks(self, task_results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple tasks.

        Args:
            task_results: List of result dictionaries from individual tasks

        Returns:
            Dictionary with aggregate statistics
        """
        if not task_results:
            return {
                "average_score": 0.0,
                "median_score": 0.0,
                "pass_rate": 0.0,
                "total_tasks": 0,
            }

        scores = [result["overall_score"] for result in task_results]
        passed_count = sum(1 for result in task_results if result["passed"])

        # Calculate statistics
        average_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]
        pass_rate = passed_count / len(task_results)

        # Breakdown by category
        category_stats = {}
        for result in task_results:
            if "task_id" in result:
                category = result["task_id"].split("-")[0]
                if category not in category_stats:
                    category_stats[category] = []
                category_stats[category].append(result["overall_score"])

        category_averages = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_stats.items()
        }

        return {
            "average_score": round(average_score, 2),
            "median_score": round(median_score, 2),
            "pass_rate": round(pass_rate, 2),
            "total_tasks": len(task_results),
            "passed_tasks": passed_count,
            "failed_tasks": len(task_results) - passed_count,
            "category_averages": {k: round(v, 2) for k, v in category_averages.items()},
        }


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())


def truncate_text(text: str, max_words: int = 500) -> str:
    """
    Truncate text to maximum word count (for bias mitigation).

    Args:
        text: Input text
        max_words: Maximum number of words

    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
