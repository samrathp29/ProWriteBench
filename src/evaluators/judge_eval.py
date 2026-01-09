"""
Judge evaluator for professional appropriateness assessment.

Uses LLM judge to evaluate tone, diplomatic language, formatting, etc.
"""

import json
import re
from typing import Dict, List
from src.tasks import Task
from src.models.base import BaseModel
from src.utils.scoring import Score


class JudgeEvaluator:
    """Evaluates professional appropriateness using LLM judge."""

    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of professional business writing. Your task is to assess the following piece of writing for professional appropriateness.

**Task Context**:
{context}

**Writing to Evaluate**:
{generated_text}

**Evaluation Criteria**:
{criteria}

Please evaluate the writing on a scale of 0-100 for each criterion, then provide an overall assessment.

Rate each criterion (0-100):
1. **Tone Appropriateness**: Is the tone suitable for the context and audience?
2. **Diplomatic Language**: Are sensitive topics handled diplomatically?
3. **Professional Formatting**: Is the structure and format professional?
4. **Clarity**: Is the message clear and well-organized?
5. **Completeness**: Does it address all necessary points?

Respond in JSON format:
{{
  "tone_appropriateness": {{
    "score": <0-100>,
    "reasoning": "<brief explanation>"
  }},
  "diplomatic_language": {{
    "score": <0-100>,
    "reasoning": "<brief explanation>"
  }},
  "professional_formatting": {{
    "score": <0-100>,
    "reasoning": "<brief explanation>"
  }},
  "clarity": {{
    "score": <0-100>,
    "reasoning": "<brief explanation>"
  }},
  "completeness": {{
    "score": <0-100>,
    "reasoning": "<brief explanation>"
  }},
  "overall_assessment": "<summary>",
  "critical_issues": ["<list any critical failures>"]
}}

Be objective and specific in your evaluation."""

    def __init__(self, judge_model: BaseModel):
        """
        Initialize judge evaluator.

        Args:
            judge_model: LLM model to use as judge (e.g., Claude Opus, GPT-5)
        """
        self.judge_model = judge_model

    def evaluate(self, task: Task, generated_text: str) -> Score:
        """
        Evaluate professional appropriateness using LLM judge.

        Args:
            task: Task object with evaluation criteria
            generated_text: Model-generated text to evaluate

        Returns:
            Score object with judge evaluation results
        """
        # Build evaluation prompt
        context = f"{task.scenario.context}\n\nRequest: {task.scenario.request}"
        criteria = "\n".join(f"- {criterion}" for criterion in task.evaluation.judge_criteria)

        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            context=context,
            generated_text=generated_text,
            criteria=criteria if criteria else "Standard professional writing criteria",
        )

        # Get judge evaluation
        try:
            response = self.judge_model.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more consistent evaluation
            )

            # Parse JSON response
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in judge response")

            evaluation = json.loads(json_str)

            # Calculate overall score
            scores = [
                evaluation.get("tone_appropriateness", {}).get("score", 0),
                evaluation.get("diplomatic_language", {}).get("score", 0),
                evaluation.get("professional_formatting", {}).get("score", 0),
                evaluation.get("clarity", {}).get("score", 0),
                evaluation.get("completeness", {}).get("score", 0),
            ]

            overall_score = sum(scores) / len(scores)

            # Check for critical issues
            critical_issues = evaluation.get("critical_issues", [])
            critical_failures = []
            if critical_issues and any(critical_issues):  # Non-empty list
                critical_failures.append("inappropriate_professional_tone")

            return Score(
                dimension="professional_appropriateness",
                score=overall_score,
                weight=0.15,  # Default weight
                passed=overall_score >= 60 and not critical_failures,
                details={
                    "judge_evaluation": evaluation,
                    "judge_model": self.judge_model.model_name,
                    "critical_failure": critical_failures,
                },
            )

        except Exception as e:
            # If judge evaluation fails, return a neutral score with error details
            return Score(
                dimension="professional_appropriateness",
                score=50.0,
                weight=0.15,
                passed=False,
                details={
                    "error": str(e),
                    "judge_model": self.judge_model.model_name,
                },
            )

    def pairwise_compare(self, task: Task, text_a: str, text_b: str) -> str:
        """
        Compare two texts pairwise and return which is better.

        Args:
            task: Task object
            text_a: First text to compare
            text_b: Second text to compare

        Returns:
            "A", "B", or "tie"
        """
        prompt = f"""You are comparing two pieces of professional writing for the same task.

**Task**: {task.scenario.request}

**Text A**:
{text_a}

**Text B**:
{text_b}

Which text is more professionally appropriate? Consider tone, clarity, diplomatic language, and completeness.

Respond with only: A, B, or tie"""

        try:
            response = self.judge_model.generate(prompt=prompt, max_tokens=10, temperature=0.0)
            response = response.strip().upper()

            if "A" in response and "B" not in response:
                return "A"
            elif "B" in response and "A" not in response:
                return "B"
            else:
                return "tie"

        except Exception:
            return "tie"
