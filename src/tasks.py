"""
Task loading and management for ProWriteBench.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Stakeholder(BaseModel):
    """Represents a stakeholder in a multi-stakeholder task."""
    name: str
    needs: List[str]
    concerns: List[str]


class Scenario(BaseModel):
    """Task scenario description."""
    context: str
    request: str
    stakeholders: Optional[List[Stakeholder]] = None


class Constraints(BaseModel):
    """Constraints for the writing task."""
    word_count: Optional[Dict[str, int]] = None  # {"min": 200, "max": 350}
    required_elements: List[str] = Field(default_factory=list)
    forbidden_elements: List[str] = Field(default_factory=list)
    tone: Optional[str] = None


class RevisionRound(BaseModel):
    """A round of feedback for revision tasks."""
    round_number: int
    feedback: str


class Evaluation(BaseModel):
    """Evaluation criteria for a task."""
    constraint_checks: List[str] = Field(default_factory=list)
    stakeholder_rubric: Dict[str, Any] = Field(default_factory=dict)
    judge_criteria: List[str] = Field(default_factory=list)
    critical_failures: List[str] = Field(default_factory=list)


class Task(BaseModel):
    """Represents a single evaluation task."""
    task_id: str
    category: str  # multi_stakeholder, constrained_revision, implicit_requirements
    difficulty: str  # easy, medium, hard
    scenario: Scenario
    constraints: Constraints
    evaluation: Evaluation
    revision_chain: Optional[List[RevisionRound]] = None

    def get_full_prompt(self, round_number: Optional[int] = None) -> str:
        """
        Generate the full prompt for the model.

        Args:
            round_number: For revision tasks, which round to generate prompt for

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Add context and request
        prompt_parts.append(f"**Context**: {self.scenario.context}\n")
        prompt_parts.append(f"**Request**: {self.scenario.request}\n")

        # Add stakeholder information if present
        if self.scenario.stakeholders:
            prompt_parts.append("\n**Stakeholders to consider**:")
            for stakeholder in self.scenario.stakeholders:
                prompt_parts.append(f"\n- {stakeholder.name}")
                prompt_parts.append(f"  - Needs: {', '.join(stakeholder.needs)}")
                prompt_parts.append(f"  - Concerns: {', '.join(stakeholder.concerns)}")

        # Add constraints
        prompt_parts.append("\n**Constraints**:")
        if self.constraints.word_count:
            min_words = self.constraints.word_count.get("min", 0)
            max_words = self.constraints.word_count.get("max", "unlimited")
            prompt_parts.append(f"- Word count: {min_words}-{max_words} words")

        if self.constraints.required_elements:
            prompt_parts.append(f"- Must include: {', '.join(self.constraints.required_elements)}")

        if self.constraints.forbidden_elements:
            prompt_parts.append(f"- Must NOT include: {', '.join(self.constraints.forbidden_elements)}")

        if self.constraints.tone:
            prompt_parts.append(f"- Tone: {self.constraints.tone}")

        # Add revision feedback if applicable
        if round_number is not None and self.revision_chain:
            prompt_parts.append(f"\n**Previous feedback (Round {round_number})**:")
            for revision_round in self.revision_chain[:round_number]:
                prompt_parts.append(f"- Round {revision_round.round_number}: {revision_round.feedback}")

        return "\n".join(prompt_parts)


class TaskLoader:
    """Loads and manages benchmark tasks."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize task loader.

        Args:
            data_dir: Path to data directory (default: ../data relative to this file)
        """
        if data_dir is None:
            # Default to data/ directory relative to src/
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.tasks_dir = self.data_dir / "tasks"

    def load_task(self, task_id: str) -> Task:
        """
        Load a single task by ID.

        Args:
            task_id: Task identifier (e.g., "MS-001", "CR-005")

        Returns:
            Task object

        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If task JSON is invalid
        """
        # Determine category from task ID prefix
        category_map = {
            "MS": "multi_stakeholder",
            "CR": "constrained_revision",
            "IR": "implicit_requirements",
        }

        prefix = task_id.split("-")[0]
        category = category_map.get(prefix)

        if category is None:
            raise ValueError(f"Unknown task ID prefix: {prefix}")

        # Load task file
        task_file = self.tasks_dir / category / f"task_{task_id.replace('-', '_').lower()}.json"

        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r") as f:
            task_data = json.load(f)

        return Task(**task_data)

    def load_all_tasks(self, category: Optional[str] = None) -> List[Task]:
        """
        Load all tasks, optionally filtered by category.

        Args:
            category: Optional category filter (multi_stakeholder, constrained_revision, implicit_requirements)

        Returns:
            List of Task objects
        """
        tasks = []

        # Determine which categories to load
        if category:
            categories = [category]
        else:
            categories = ["multi_stakeholder", "constrained_revision", "implicit_requirements"]

        # Load tasks from each category
        for cat in categories:
            category_dir = self.tasks_dir / cat
            if not category_dir.exists():
                continue

            for task_file in sorted(category_dir.glob("task_*.json")):
                try:
                    with open(task_file, "r") as f:
                        task_data = json.load(f)
                    tasks.append(Task(**task_data))
                except Exception as e:
                    print(f"Warning: Failed to load {task_file}: {e}")

        return tasks

    def get_task_count(self, category: Optional[str] = None) -> int:
        """
        Get count of tasks, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            Number of tasks
        """
        return len(self.load_all_tasks(category))
