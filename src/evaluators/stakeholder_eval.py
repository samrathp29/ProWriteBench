"""
Stakeholder balance evaluator for multi-stakeholder communication tasks.

Evaluates whether the writing addresses all stakeholders' needs fairly.
"""

import json
import re
from typing import Dict, List
from src.tasks import Task, Stakeholder
from src.models.base import BaseModel
from src.utils.scoring import Score


class StakeholderEvaluator:
    """Evaluates stakeholder balance in professional writing."""

    STAKEHOLDER_PROMPT_TEMPLATE = """You are evaluating whether a piece of professional writing adequately addresses the needs of multiple stakeholders.

**Task Context**: {context}

**Stakeholder: {stakeholder_name}**
- Needs: {needs}
- Concerns: {concerns}

**Writing to Evaluate**:
{generated_text}

Does the writing address this stakeholder's needs and concerns? Rate on a scale of 0-100.

Respond in JSON format:
{{
  "score": <0-100>,
  "needs_addressed": {{
    <need>: <true/false>,
    ...
  }},
  "concerns_addressed": {{
    <concern>: <true/false>,
    ...
  }},
  "reasoning": "<explanation>",
  "specific_evidence": "<quotes or examples from the text>"
}}"""

    def __init__(self, judge_model: BaseModel):
        """
        Initialize stakeholder evaluator.

        Args:
            judge_model: LLM model to use for evaluation
        """
        self.judge_model = judge_model

    def evaluate(self, task: Task, generated_text: str) -> Score:
        """
        Evaluate stakeholder balance.

        Args:
            task: Task object with stakeholder information
            generated_text: Model-generated text to evaluate

        Returns:
            Score object with stakeholder balance results
        """
        # Only applicable to multi-stakeholder tasks
        if not task.scenario.stakeholders or len(task.scenario.stakeholders) == 0:
            return Score(
                dimension="stakeholder_balance",
                score=100.0,
                weight=0.20,
                passed=True,
                details={"message": "Not applicable (no stakeholders defined)"},
            )

        stakeholder_scores = []
        stakeholder_details = {}

        # Evaluate each stakeholder
        for stakeholder in task.scenario.stakeholders:
            try:
                stakeholder_score = self._evaluate_stakeholder(
                    task, stakeholder, generated_text
                )
                stakeholder_scores.append(stakeholder_score["score"])
                stakeholder_details[stakeholder.name] = stakeholder_score
            except Exception as e:
                stakeholder_details[stakeholder.name] = {"error": str(e)}

        # Calculate overall score
        if stakeholder_scores:
            # Use minimum score to penalize neglecting any stakeholder
            overall_score = min(stakeholder_scores)

            # Also consider average to reward balance
            average_score = sum(stakeholder_scores) / len(stakeholder_scores)
            overall_score = (overall_score * 0.6) + (average_score * 0.4)

            # Calculate balance metric (standard deviation - lower is better)
            if len(stakeholder_scores) > 1:
                mean = sum(stakeholder_scores) / len(stakeholder_scores)
                variance = sum((s - mean) ** 2 for s in stakeholder_scores) / len(stakeholder_scores)
                std_dev = variance ** 0.5
                balance_penalty = std_dev / 10  # Normalize
                overall_score = max(0, overall_score - balance_penalty)
        else:
            overall_score = 50.0

        return Score(
            dimension="stakeholder_balance",
            score=overall_score,
            weight=0.20,
            passed=overall_score >= 60,
            details={
                "stakeholder_scores": stakeholder_details,
                "individual_scores": stakeholder_scores,
                "balance": self._calculate_balance(stakeholder_scores),
            },
        )

    def _evaluate_stakeholder(
        self,
        task: Task,
        stakeholder: Stakeholder,
        generated_text: str
    ) -> Dict:
        """Evaluate how well a single stakeholder's needs are addressed."""
        prompt = self.STAKEHOLDER_PROMPT_TEMPLATE.format(
            context=task.scenario.context,
            stakeholder_name=stakeholder.name,
            needs=", ".join(stakeholder.needs),
            concerns=", ".join(stakeholder.concerns),
            generated_text=generated_text,
        )

        response = self.judge_model.generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0.3,
        )

        # Parse JSON response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        return json.loads(json_str)

    def _calculate_balance(self, scores: List[float]) -> Dict:
        """Calculate balance metrics for stakeholder scores."""
        if not scores or len(scores) < 2:
            return {"balanced": True, "variance": 0.0}

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # Consider balanced if std dev < 15
        return {
            "balanced": std_dev < 15,
            "variance": round(variance, 2),
            "std_dev": round(std_dev, 2),
            "range": round(max(scores) - min(scores), 2),
        }
