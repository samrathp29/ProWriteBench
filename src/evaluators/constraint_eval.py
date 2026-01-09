"""
Constraint satisfaction evaluator for ProWriteBench.

Checks hard requirements: word count, required/forbidden elements, etc.
"""

import re
from typing import Dict, List
from src.tasks import Task, Constraints
from src.utils.scoring import Score, count_words


class ConstraintEvaluator:
    """Evaluates constraint satisfaction for professional writing tasks."""

    def evaluate(self, task: Task, generated_text: str) -> Score:
        """
        Evaluate whether generated text satisfies all constraints.

        Args:
            task: Task object with constraints
            generated_text: Model-generated text to evaluate

        Returns:
            Score object with constraint satisfaction results
        """
        constraints = task.constraints
        failures = []
        checks = {}

        # Check word count
        word_count = count_words(generated_text)
        if constraints.word_count:
            min_words = constraints.word_count.get("min", 0)
            max_words = constraints.word_count.get("max", float("inf"))

            # Allow 10% tolerance
            min_allowed = min_words * 0.9
            max_allowed = max_words * 1.1

            word_count_passed = min_allowed <= word_count <= max_allowed
            checks["word_count"] = {
                "required": f"{min_words}-{max_words}",
                "actual": word_count,
                "passed": word_count_passed,
            }

            if not word_count_passed:
                failures.append(f"Word count {word_count} outside range {min_words}-{max_words}")

        # Check required elements
        required_checks = []
        for required in constraints.required_elements:
            # Case-insensitive search for required element
            found = re.search(re.escape(required), generated_text, re.IGNORECASE)
            required_checks.append({
                "element": required,
                "found": bool(found),
            })

            if not found:
                failures.append(f"Missing required element: {required}")

        checks["required_elements"] = required_checks

        # Check forbidden elements
        forbidden_checks = []
        for forbidden in constraints.forbidden_elements:
            # Case-insensitive search for forbidden element
            found = re.search(re.escape(forbidden), generated_text, re.IGNORECASE)
            forbidden_checks.append({
                "element": forbidden,
                "found": bool(found),
            })

            if found:
                failures.append(f"Contains forbidden element: {forbidden}")

        checks["forbidden_elements"] = forbidden_checks

        # Calculate score
        total_checks = (
            (1 if constraints.word_count else 0) +
            len(constraints.required_elements) +
            len(constraints.forbidden_elements)
        )

        if total_checks == 0:
            score = 100.0
        else:
            passed_checks = total_checks - len(failures)
            score = (passed_checks / total_checks) * 100

        # Determine if this is a critical failure
        critical_failures = []
        if failures:
            # Missing required elements or having forbidden content are critical
            if any("Missing required" in f for f in failures):
                critical_failures.append("missing_required_elements")
            if any("forbidden" in f for f in failures):
                critical_failures.append("contains_forbidden_content")

        return Score(
            dimension="constraint_satisfaction",
            score=score,
            weight=0.30,  # Default weight
            passed=len(failures) == 0,
            details={
                "checks": checks,
                "failures": failures,
                "critical_failure": critical_failures,
            },
        )

    def quick_check(self, constraints: Constraints, generated_text: str) -> bool:
        """
        Quick boolean check if all constraints are satisfied.

        Args:
            constraints: Constraints to check
            generated_text: Text to evaluate

        Returns:
            True if all constraints satisfied, False otherwise
        """
        # Word count check
        if constraints.word_count:
            word_count = count_words(generated_text)
            min_words = constraints.word_count.get("min", 0) * 0.9
            max_words = constraints.word_count.get("max", float("inf")) * 1.1

            if not (min_words <= word_count <= max_words):
                return False

        # Required elements check
        for required in constraints.required_elements:
            if not re.search(re.escape(required), generated_text, re.IGNORECASE):
                return False

        # Forbidden elements check
        for forbidden in constraints.forbidden_elements:
            if re.search(re.escape(forbidden), generated_text, re.IGNORECASE):
                return False

        return True
