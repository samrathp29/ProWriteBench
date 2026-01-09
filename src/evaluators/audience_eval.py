"""
Audience-specific clarity evaluator.

Tests whether writing is clear and comprehensible for target audiences.
"""

import json
import re
from typing import Dict, List
from src.tasks import Task
from src.models.base import BaseModel
from src.utils.scoring import Score


class AudienceEvaluator:
    """Evaluates audience-specific clarity of professional writing."""

    AUDIENCE_PROMPT_TEMPLATE = """You are evaluating whether a piece of professional writing is clear and appropriate for a specific audience.

**Writing to Evaluate**:
{generated_text}

**Target Audience**: {audience_type}
{audience_description}

Evaluate the writing for this specific audience. Consider:
- Is the technical level appropriate?
- Is the information presented in the right format?
- Can this audience understand and act on this writing?

Respond in JSON format:
{{
  "score": <0-100>,
  "comprehension": {{
    "understandable": <true/false>,
    "technical_level_appropriate": <true/false>,
    "actionable": <true/false>
  }},
  "strengths": ["<list>"],
  "weaknesses": ["<list>"],
  "reasoning": "<explanation>"
}}"""

    def __init__(self, judge_model: BaseModel):
        """
        Initialize audience evaluator.

        Args:
            judge_model: LLM model to use for evaluation
        """
        self.judge_model = judge_model

    def evaluate(self, task: Task, generated_text: str) -> Score:
        """
        Evaluate audience-specific clarity.

        Args:
            task: Task object
            generated_text: Model-generated text to evaluate

        Returns:
            Score object with audience clarity results
        """
        # Define default audiences based on task category
        audiences = self._get_audiences_for_task(task)

        audience_scores = []
        audience_details = {}

        # Evaluate for each audience
        for audience_name, audience_desc in audiences.items():
            try:
                audience_score = self._evaluate_for_audience(
                    audience_name, audience_desc, generated_text
                )
                audience_scores.append(audience_score["score"])
                audience_details[audience_name] = audience_score
            except Exception as e:
                audience_details[audience_name] = {"error": str(e)}

        # Calculate overall score
        if audience_scores:
            overall_score = sum(audience_scores) / len(audience_scores)
        else:
            overall_score = 50.0

        return Score(
            dimension="audience_clarity",
            score=overall_score,
            weight=0.25,
            passed=overall_score >= 60,
            details={
                "audience_scores": audience_details,
                "individual_scores": audience_scores,
            },
        )

    def _get_audiences_for_task(self, task: Task) -> Dict[str, str]:
        """Determine target audiences based on task."""
        # For multi-stakeholder tasks, use stakeholders as audiences
        if task.scenario.stakeholders:
            return {
                stakeholder.name: f"This audience cares about: {', '.join(stakeholder.needs)}"
                for stakeholder in task.scenario.stakeholders
            }

        # Default audiences for business communication
        return {
            "Executive": "Senior leadership who needs high-level insights and decision points. Prefers concise, strategic information.",
            "General Professional": "Professional audience with business acumen but may not have deep technical expertise.",
        }

    def _evaluate_for_audience(
        self,
        audience_name: str,
        audience_description: str,
        generated_text: str
    ) -> Dict:
        """Evaluate clarity for a specific audience."""
        prompt = self.AUDIENCE_PROMPT_TEMPLATE.format(
            generated_text=generated_text,
            audience_type=audience_name,
            audience_description=audience_description,
        )

        response = self.judge_model.generate(
            prompt=prompt,
            max_tokens=600,
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
