#!/usr/bin/env python3
"""
Generate a human-readable report from ProWriteBench results.

Usage:
    python examples/generate_report.py --results results/claude-opus-4-5_20250108_120000.json
    python examples/generate_report.py --results results/gpt4.json --output report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_markdown_report(results: dict) -> str:
    """Generate a markdown-formatted report."""
    lines = []

    # Header
    lines.append("# ProWriteBench Evaluation Report\n")
    lines.append(f"**Model**: {results['model']}\n")
    lines.append(f"**Judge Model**: {results['judge_model']}\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n---\n")

    # Summary
    summary = results["summary"]
    lines.append("## Summary\n")
    lines.append(f"- **Average Score**: {summary['average_score']}/100")
    lines.append(f"- **Median Score**: {summary['median_score']}/100")
    lines.append(f"- **Pass Rate**: {summary['pass_rate']:.1%}")
    lines.append(f"- **Total Tasks**: {summary['total_tasks']}")
    lines.append(f"- **Passed**: {summary['passed_tasks']}")
    lines.append(f"- **Failed**: {summary['failed_tasks']}\n")

    # Category breakdown
    if "category_averages" in summary and summary["category_averages"]:
        lines.append("### Category Performance\n")
        lines.append("| Category | Average Score |")
        lines.append("|----------|--------------|")
        for category, score in summary["category_averages"].items():
            lines.append(f"| {category} | {score}/100 |")
        lines.append("")

    lines.append("\n---\n")

    # Individual task results
    lines.append("## Task Results\n")

    for task_result in results["task_results"]:
        task_id = task_result["task_id"]
        category = task_result["category"]
        difficulty = task_result["difficulty"]
        overall_score = task_result["overall_score"]
        passed = "✓ PASSED" if task_result["passed"] else "✗ FAILED"

        lines.append(f"### Task {task_id} ({category}, {difficulty}) - {passed}\n")
        lines.append(f"**Overall Score**: {overall_score}/100\n")

        # Dimension scores
        lines.append("**Dimension Scores**:\n")
        for dimension, scores in task_result["dimension_scores"].items():
            status = "✓" if scores["passed"] else "✗"
            lines.append(
                f"- {status} **{dimension}**: {scores['score']:.1f}/100 "
                f"(weight: {scores['weight']:.0%})"
            )
        lines.append("")

        # Critical failures
        if task_result.get("critical_failures"):
            lines.append("**Critical Failures**:\n")
            for failure in task_result["critical_failures"]:
                lines.append(f"- {failure}")
            lines.append("")

        # Generated text (truncated)
        if "generated_text" in task_result:
            generated = task_result["generated_text"]
            if len(generated) > 500:
                generated = generated[:500] + "..."
            lines.append(f"**Generated Text** (truncated):\n```\n{generated}\n```\n")

        lines.append("---\n")

    return "\n".join(lines)


def generate_text_report(results: dict) -> str:
    """Generate a plain text report."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(" " * 15 + "ProWriteBench Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"\nModel: {results['model']}")
    lines.append(f"Judge Model: {results['judge_model']}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary
    summary = results["summary"]
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Average Score:     {summary['average_score']}/100")
    lines.append(f"Median Score:      {summary['median_score']}/100")
    lines.append(f"Pass Rate:         {summary['pass_rate']:.1%}")
    lines.append(f"Total Tasks:       {summary['total_tasks']}")
    lines.append(f"Passed:            {summary['passed_tasks']}")
    lines.append(f"Failed:            {summary['failed_tasks']}\n")

    # Category breakdown
    if "category_averages" in summary and summary["category_averages"]:
        lines.append("Category Performance:")
        for category, score in summary["category_averages"].items():
            lines.append(f"  {category}: {score}/100")
        lines.append("")

    # Individual tasks
    lines.append("=" * 70)
    lines.append("TASK RESULTS")
    lines.append("=" * 70)

    for task_result in results["task_results"]:
        task_id = task_result["task_id"]
        category = task_result["category"]
        difficulty = task_result["difficulty"]
        overall_score = task_result["overall_score"]
        passed = "PASSED" if task_result["passed"] else "FAILED"

        lines.append(f"\nTask {task_id} ({category}, {difficulty}) - {passed}")
        lines.append(f"Overall Score: {overall_score}/100")

        lines.append("\nDimension Scores:")
        for dimension, scores in task_result["dimension_scores"].items():
            status = "✓" if scores["passed"] else "✗"
            lines.append(
                f"  {status} {dimension}: {scores['score']:.1f}/100 "
                f"(weight: {scores['weight']:.0%})"
            )

        if task_result.get("critical_failures"):
            lines.append("\nCritical Failures:")
            for failure in task_result["critical_failures"]:
                lines.append(f"  - {failure}")

        lines.append("-" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate report from ProWriteBench results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--output", help="Output file path (default: print to stdout)")
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    # Generate report
    if args.format == "markdown":
        report = generate_markdown_report(results)
    else:
        report = generate_text_report(results)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
