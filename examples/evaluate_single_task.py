#!/usr/bin/env python3
"""
Evaluate a single task from ProWriteBench.

Usage:
    python examples/evaluate_single_task.py --task MS-001 --model claude-opus-4-5
    python examples/evaluate_single_task.py --task CR-005 --model gpt-4 --judge claude-opus-4-5
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import ProWriteBench
from src.models import AnthropicModel, OpenAIModel
from src.tasks import TaskLoader


def get_model(model_name: str):
    """Get model instance from name."""
    # Anthropic models
    if "claude" in model_name.lower():
        return AnthropicModel(model_name=model_name)
    # OpenAI models
    elif "gpt" in model_name.lower():
        return OpenAIModel(model_name=model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single ProWriteBench task")
    parser.add_argument("--task", required=True, help="Task ID (e.g., MS-001, CR-005, IR-008)")
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g., claude-opus-4-5, gpt-4)")
    parser.add_argument("--judge", help="Judge model (default: same as --model)")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Initialize models
    print(f"\nInitializing models...")
    model = get_model(args.model)
    judge_model = get_model(args.judge) if args.judge else model

    print(f"Model to evaluate: {model.model_name}")
    print(f"Judge model: {judge_model.model_name}")

    # Load task
    task_loader = TaskLoader()
    try:
        task = task_loader.load_task(args.task)
        print(f"\nLoaded task: {task.task_id}")
        print(f"Category: {task.category}")
        print(f"Difficulty: {task.difficulty}")
    except FileNotFoundError:
        print(f"\nError: Task {args.task} not found.")
        print("Available tasks: MS-001, CR-005, IR-008")
        sys.exit(1)

    # Run evaluation
    benchmark = ProWriteBench(
        model_to_evaluate=model,
        judge_model=judge_model,
    )

    print(f"\nEvaluating task {task.task_id}...")
    print("=" * 60)

    result = benchmark.evaluate_task(task, verbose=args.verbose)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTask ID: {result['task_id']}")
    print(f"Overall Score: {result['overall_score']}/100")
    print(f"Passed: {'✓' if result['passed'] else '✗'}")

    if result.get("critical_failures"):
        print(f"\nCritical Failures:")
        for failure in result["critical_failures"]:
            print(f"  - {failure}")

    print(f"\nDimension Scores:")
    for dimension, scores in result["dimension_scores"].items():
        status = "✓" if scores["passed"] else "✗"
        print(f"  {status} {dimension}: {scores['score']:.1f}/100 (weight: {scores['weight']:.0%})")

    # Print generated text
    if "generated_text" in result:
        print(f"\n" + "=" * 60)
        print("GENERATED TEXT")
        print("=" * 60)
        print(result["generated_text"])

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
