"""
Revision coherence evaluator for constrained revision tasks.

Evaluates how well models incorporate feedback across revision rounds.
"""

import json
import re
from typing import Dict, List
from src.tasks import Task
from src.models.base import BaseModel
from src.utils.scoring import Score


class RevisionEvaluator:
    """Evaluates revision coherence across feedback iterations."""

    REVISION_PROMPT_TEMPLATE = """You are evaluating how well feedback was incorporated in a revision.

**Original Request**: {original_request}

**Previous Version**:
{previous_version}

**Feedback Given**: {feedback}

**Revised Version**:
{revised_version}

Evaluate whether the revision:
1. Addresses the feedback appropriately
2. Improves the writing quality
3. Avoids overcorrecting or losing previous strengths

Respond in JSON format:
{{
  "feedback_incorporated": <true/false>,
  "quality_improved": <true/false>,
  "avoided_overcorrection": <true/false>,
  "score": <0-100>,
  "new_issues": ["<list any new problems introduced>"],
  "reasoning": "<explanation>"
}}"""

    def __init__(self, judge_model: BaseModel):
        """
        Initialize revision evaluator.

        Args:
            judge_model: LLM model to use for evaluation
        """
        self.judge_model = judge_model

    def evaluate(self, task: Task, revisions: List[str]) -> Score:
        """
        Evaluate revision coherence across multiple rounds.

        Args:
            task: Task object with revision chain
            revisions: List of text outputs from each revision round

        Returns:
            Score object with revision coherence results
        """
        # Only applicable to revision tasks
        if not task.revision_chain or len(task.revision_chain) == 0:
            return Score(
                dimension="revision_coherence",
                score=100.0,
                weight=0.10,
                passed=True,
                details={"message": "Not applicable (no revision rounds)"},
            )

        revision_scores = []
        revision_details = []

        # Evaluate each revision round
        for i, revision_round in enumerate(task.revision_chain):
            if i + 1 >= len(revisions):
                break  # No more revisions to evaluate

            try:
                revision_score = self._evaluate_revision(
                    task=task,
                    previous_version=revisions[i],
                    feedback=revision_round.feedback,
                    revised_version=revisions[i + 1],
                )
                revision_scores.append(revision_score["score"])
                revision_details.append({
                    "round": revision_round.round_number,
                    "feedback": revision_round.feedback,
                    "evaluation": revision_score,
                })
            except Exception as e:
                revision_details.append({
                    "round": revision_round.round_number,
                    "error": str(e),
                })

        # Calculate overall score
        if revision_scores:
            # Average across all revision rounds
            overall_score = sum(revision_scores) / len(revision_scores)

            # Penalize if quality decreased in later rounds
            if len(revision_scores) > 1:
                quality_trend = revision_scores[-1] - revision_scores[0]
                if quality_trend < -10:  # Quality decreased significantly
                    overall_score *= 0.8
        else:
            overall_score = 50.0

        return Score(
            dimension="revision_coherence",
            score=overall_score,
            weight=0.10,
            passed=overall_score >= 60,
            details={
                "revision_rounds": revision_details,
                "round_scores": revision_scores,
            },
        )

    def _evaluate_revision(
        self,
        task: Task,
        previous_version: str,
        feedback: str,
        revised_version: str
    ) -> Dict:
        """Evaluate a single revision round."""
        prompt = self.REVISION_PROMPT_TEMPLATE.format(
            original_request=task.scenario.request,
            previous_version=previous_version,
            feedback=feedback,
            revised_version=revised_version,
        )

        response = self.judge_model.generate(
            prompt=prompt,
            max_tokens=700,
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
