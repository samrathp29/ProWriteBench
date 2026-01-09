#!/usr/bin/env python3
"""
Run the full ProWriteBench benchmark.

Usage:
    python examples/run_benchmark.py --model claude-opus-4-5
    python examples/run_benchmark.py --model gpt-4 --judge claude-opus-4-5 --category multi_stakeholder
    python examples/run_benchmark.py --model gpt-5 --tasks MS-001,IR-008 --output results/gpt5.json
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import ProWriteBench
from src.models import AnthropicModel, OpenAIModel


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
    parser = argparse.ArgumentParser(description="Run ProWriteBench full benchmark")
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g., claude-opus-4-5, gpt-4)")
    parser.add_argument("--judge", help="Judge model (default: same as --model)")
    parser.add_argument(
        "--category",
        choices=["multi_stakeholder", "constrained_revision", "implicit_requirements"],
        help="Filter by category (default: all)"
    )
    parser.add_argument("--tasks", help="Comma-separated list of task IDs (e.g., MS-001,CR-005)")
    parser.add_argument("--output", help="Output directory (default: results/)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Initialize models
    print(f"\nInitializing ProWriteBench...")
    model = get_model(args.model)
    judge_model = get_model(args.judge) if args.judge else model

    print(f"Model to evaluate: {model.model_name}")
    print(f"Judge model: {judge_model.model_name}")

    # Initialize benchmark
    benchmark = ProWriteBench(
        model_to_evaluate=model,
        judge_model=judge_model,
    )

    # Parse tasks if provided
    task_ids = None
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
        print(f"Evaluating specific tasks: {task_ids}")
    elif args.category:
        print(f"Evaluating category: {args.category}")
    else:
        print(f"Evaluating all tasks")

    # Run benchmark
    results = benchmark.run_benchmark(
        task_ids=task_ids,
        category=args.category,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"\nModel: {results['model']}")
    print(f"Judge: {results['judge_model']}")
    print(f"\nOverall Performance:")
    print(f"  Average Score: {summary['average_score']}/100")
    print(f"  Median Score: {summary['median_score']}/100")
    print(f"  Pass Rate: {summary['pass_rate']:.1%}")
    print(f"  Tasks Evaluated: {summary['total_tasks']}")
    print(f"  Passed: {summary['passed_tasks']}")
    print(f"  Failed: {summary['failed_tasks']}")

    if "category_averages" in summary and summary["category_averages"]:
        print(f"\nCategory Breakdown:")
        for category, score in summary["category_averages"].items():
            print(f"  {category}: {score}/100")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.model_name.replace(":", "_").replace("/", "_")
    output_file = output_dir / f"{model_slug}_{timestamp}.json"

    benchmark.save_results(results, output_file)

    print("\n" + "=" * 60)
    print(f"Benchmark complete! Results saved to: {output_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
