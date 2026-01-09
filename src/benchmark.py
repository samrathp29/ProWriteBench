"""
Main benchmark orchestration for ProWriteBench.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from src.tasks import Task, TaskLoader
from src.models.base import BaseModel
from src.evaluators import (
    ConstraintEvaluator,
    JudgeEvaluator,
    StakeholderEvaluator,
    AudienceEvaluator,
    RevisionEvaluator,
)
from src.utils.scoring import Score, ScoreAggregator


class ProWriteBench:
    """Main benchmark runner for professional writing evaluation."""

    def __init__(
        self,
        model_to_evaluate: BaseModel,
        judge_model: Optional[BaseModel] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize ProWriteBench.

        Args:
            model_to_evaluate: The LLM model to benchmark
            judge_model: Judge model for evaluation (if None, uses model_to_evaluate)
            data_dir: Path to data directory (default: auto-detect)
        """
        self.model = model_to_evaluate
        self.judge_model = judge_model or model_to_evaluate

        # Initialize task loader
        self.task_loader = TaskLoader(data_dir)

        # Initialize evaluators
        self.constraint_evaluator = ConstraintEvaluator()
        self.judge_evaluator = JudgeEvaluator(self.judge_model)
        self.stakeholder_evaluator = StakeholderEvaluator(self.judge_model)
        self.audience_evaluator = AudienceEvaluator(self.judge_model)
        self.revision_evaluator = RevisionEvaluator(self.judge_model)

        # Initialize score aggregator
        self.score_aggregator = ScoreAggregator()

    def evaluate_task(self, task: Task, verbose: bool = False) -> Dict:
        """
        Evaluate a single task.

        Args:
            task: Task object to evaluate
            verbose: Whether to print detailed progress

        Returns:
            Dictionary with evaluation results
        """
        if verbose:
            print(f"\nEvaluating task: {task.task_id}")
            print(f"Category: {task.category}")
            print(f"Difficulty: {task.difficulty}")

        results = {
            "task_id": task.task_id,
            "category": task.category,
            "difficulty": task.difficulty,
        }

        # Handle revision tasks differently
        if task.revision_chain:
            return self._evaluate_revision_task(task, verbose, results)
        else:
            return self._evaluate_single_task(task, verbose, results)

    def _evaluate_single_task(self, task: Task, verbose: bool, results: Dict) -> Dict:
        """Evaluate a non-revision task."""
        # Generate text
        prompt = task.get_full_prompt()
        if verbose:
            print(f"\nPrompt:\n{prompt[:200]}...")

        try:
            generated_text = self.model.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
            )
            results["generated_text"] = generated_text

            if verbose:
                print(f"\nGenerated text:\n{generated_text[:200]}...")

        except Exception as e:
            results["error"] = str(e)
            return results

        # Evaluate with all evaluators
        scores = []

        # 1. Constraint satisfaction
        constraint_score = self.constraint_evaluator.evaluate(task, generated_text)
        scores.append(constraint_score)

        # 2. Judge evaluation (professional appropriateness)
        judge_score = self.judge_evaluator.evaluate(task, generated_text)
        scores.append(judge_score)

        # 3. Stakeholder balance (if applicable)
        stakeholder_score = self.stakeholder_evaluator.evaluate(task, generated_text)
        scores.append(stakeholder_score)

        # 4. Audience clarity
        audience_score = self.audience_evaluator.evaluate(task, generated_text)
        scores.append(audience_score)

        # 5. Revision coherence (not applicable for single tasks)
        revision_score = self.revision_evaluator.evaluate(task, [generated_text])
        scores.append(revision_score)

        # Aggregate scores
        aggregate_result = self.score_aggregator.aggregate(scores)
        results.update(aggregate_result)

        if verbose:
            print(f"\nOverall Score: {aggregate_result['overall_score']}")
            print(f"Passed: {aggregate_result['passed']}")

        return results

    def _evaluate_revision_task(self, task: Task, verbose: bool, results: Dict) -> Dict:
        """Evaluate a revision task with multiple rounds."""
        revisions = []

        # Generate initial version
        prompt = task.get_full_prompt(round_number=0)
        try:
            initial_text = self.model.generate(prompt=prompt, max_tokens=2000, temperature=0.7)
            revisions.append(initial_text)

            if verbose:
                print(f"\nInitial version generated")

            # Generate revisions based on feedback
            for round_num, revision_round in enumerate(task.revision_chain, start=1):
                revision_prompt = f"""Here is your previous writing:

{revisions[-1]}

**Feedback**: {revision_round.feedback}

Please revise your writing based on this feedback while maintaining the original requirements:
{task.scenario.request}

Constraints:
{task.constraints.word_count if task.constraints.word_count else "No specific word count"}
Tone: {task.constraints.tone or "Professional"}
"""
                revised_text = self.model.generate(
                    prompt=revision_prompt,
                    max_tokens=2000,
                    temperature=0.7,
                )
                revisions.append(revised_text)

                if verbose:
                    print(f"Revision round {round_num} generated")

        except Exception as e:
            results["error"] = str(e)
            return results

        results["revisions"] = revisions
        results["generated_text"] = revisions[-1]  # Final version

        # Evaluate final version
        final_text = revisions[-1]
        scores = []

        # 1. Constraint satisfaction
        constraint_score = self.constraint_evaluator.evaluate(task, final_text)
        scores.append(constraint_score)

        # 2. Judge evaluation
        judge_score = self.judge_evaluator.evaluate(task, final_text)
        scores.append(judge_score)

        # 3. Stakeholder balance
        stakeholder_score = self.stakeholder_evaluator.evaluate(task, final_text)
        scores.append(stakeholder_score)

        # 4. Audience clarity
        audience_score = self.audience_evaluator.evaluate(task, final_text)
        scores.append(audience_score)

        # 5. Revision coherence
        revision_score = self.revision_evaluator.evaluate(task, revisions)
        scores.append(revision_score)

        # Aggregate scores
        aggregate_result = self.score_aggregator.aggregate(scores)
        results.update(aggregate_result)

        if verbose:
            print(f"\nOverall Score: {aggregate_result['overall_score']}")
            print(f"Passed: {aggregate_result['passed']}")

        return results

    def run_benchmark(
        self,
        task_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run full benchmark evaluation.

        Args:
            task_ids: Specific task IDs to evaluate (if None, evaluates all)
            category: Filter by category (if None, evaluates all categories)
            verbose: Whether to print detailed progress

        Returns:
            Dictionary with benchmark results
        """
        # Load tasks
        if task_ids:
            tasks = [self.task_loader.load_task(task_id) for task_id in task_ids]
        else:
            tasks = self.task_loader.load_all_tasks(category)

        if not tasks:
            return {"error": "No tasks found"}

        print(f"\nRunning ProWriteBench on {len(tasks)} tasks...")
        print(f"Model: {self.model.model_name}")
        print(f"Judge: {self.judge_model.model_name}\n")

        # Evaluate each task
        task_results = []
        for task in tqdm(tasks, desc="Evaluating tasks"):
            result = self.evaluate_task(task, verbose=verbose)
            task_results.append(result)

        # Aggregate results
        aggregate_stats = self.score_aggregator.aggregate_multiple_tasks(task_results)

        return {
            "model": self.model.model_name,
            "judge_model": self.judge_model.model_name,
            "summary": aggregate_stats,
            "task_results": task_results,
        }

    def save_results(self, results: Dict, output_path: Path):
        """
        Save benchmark results to JSON file.

        Args:
            results: Results dictionary from run_benchmark
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
