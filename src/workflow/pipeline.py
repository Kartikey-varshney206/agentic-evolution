"""
Workflow Pipeline — Defines and evolves the society's processing architecture.

This is the CORE NOVEL CONTRIBUTION of the AI Civilization project.

Instead of a fixed pipeline (Question → Agent → Answer), the society
evolves its own workflow topology. Agents can propose changes like:
- Adding a new verification stage
- Reordering stages
- Adding parallel branches
- Removing bottleneck stages

The pipeline is represented as a directed acyclic graph (DAG) defined in YAML.
Workflow evolution uses A/B testing to validate changes before adoption.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStage:
    """A single stage in the workflow pipeline."""

    id: str
    role: str  # which agent role(s) handle this (comma-separated)
    action: str  # what the stage does
    input_from: list[str]  # which previous stages feed into this
    max_agents: int = 1  # how many agents participate in this stage
    timeout_seconds: int = 60

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "action": self.action,
            "input_from": self.input_from,
            "max_agents": self.max_agents,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class WorkflowPipeline:
    """A complete workflow pipeline — the society's processing architecture."""

    name: str
    description: str = ""
    generation: int = 0
    stages: list[WorkflowStage] = field(default_factory=list)
    fitness_score: float = 0.0
    tasks_tested: int = 0
    tasks_correct: int = 0
    parent_pipeline: str | None = None  # which pipeline this evolved from

    @property
    def accuracy(self) -> float:
        if self.tasks_tested == 0:
            return 0.0
        return self.tasks_correct / self.tasks_tested

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowPipeline":
        """Load a pipeline from YAML config."""
        with open(path) as f:
            data = yaml.safe_load(f)

        stages = []
        for stage_data in data.get("stages", []):
            stages.append(WorkflowStage(
                id=stage_data["id"],
                role=stage_data["role"],
                action=stage_data["action"],
                input_from=stage_data.get("input_from", ["input"]),
                max_agents=stage_data.get("max_agents", 1),
                timeout_seconds=stage_data.get("timeout_seconds", 60),
            ))

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            generation=data.get("generation", 0),
            stages=stages,
        )

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = {
            "name": self.name,
            "description": self.description,
            "generation": self.generation,
            "stages": [s.to_dict() for s in self.stages],
        }
        return yaml.dump(data, default_flow_style=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "generation": self.generation,
            "stages": [s.to_dict() for s in self.stages],
            "fitness": round(self.fitness_score, 4),
            "accuracy": f"{self.accuracy:.1%}",
            "tasks_tested": self.tasks_tested,
            "parent": self.parent_pipeline,
        }

    def __repr__(self) -> str:
        stage_names = " -> ".join(s.id for s in self.stages)
        return f"Pipeline({self.name}, gen={self.generation}, [{stage_names}])"


class WorkflowEvolver:
    """
    Evolves workflow pipelines by proposing mutations and A/B testing them.

    This is where the society redesigns its own organizational structure.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client
        self.pipeline_history: list[WorkflowPipeline] = []

    def propose_mutation(
        self,
        current: WorkflowPipeline,
        performance_data: dict[str, Any],
    ) -> WorkflowPipeline:
        """
        Propose a mutation to the current workflow pipeline.

        If LLM is available, uses it for intelligent proposals.
        Otherwise, uses random structural mutations.
        """
        if self.llm:
            return self._llm_propose(current, performance_data)
        else:
            return self._random_mutate(current)

    def _llm_propose(
        self,
        current: WorkflowPipeline,
        performance_data: dict[str, Any],
    ) -> WorkflowPipeline:
        """Use LLM to propose an intelligent workflow mutation."""
        system_prompt = (
            "You are a workflow architect for an AI society. "
            "You analyze the current processing pipeline and its performance "
            "data to propose improvements.\n\n"
            "Available mutation types:\n"
            "1. ADD a new stage (e.g., add 'verify' before output)\n"
            "2. REMOVE an underperforming stage\n"
            "3. REORDER stages (e.g., critique before synthesis)\n"
            "4. CHANGE a stage's role assignment\n"
            "5. ADD parallel branches\n\n"
            "Propose exactly ONE mutation. Be specific."
        )

        current_yaml = current.to_yaml()
        user_message = (
            f"## Current Pipeline\n```yaml\n{current_yaml}\n```\n\n"
            f"## Performance Data\n{performance_data}\n\n"
            f"## Task\n"
            f"Propose ONE specific mutation to improve this pipeline. "
            f"Explain what to change and why. Then provide the full "
            f"modified pipeline in YAML format."
        )

        try:
            response = self.llm.chat(system_prompt, user_message, temperature=0.8)
            # Try to parse YAML from the response
            new_pipeline = self._parse_pipeline_from_response(
                response.content, current
            )
            return new_pipeline
        except Exception as e:
            logger.warning(f"LLM workflow proposal failed: {e}")
            return self._random_mutate(current)

    def _random_mutate(self, current: WorkflowPipeline) -> WorkflowPipeline:
        """Apply a random structural mutation to the pipeline."""
        mutant = copy.deepcopy(current)
        mutant.name = f"{current.name}_mut{random.randint(100,999)}"
        mutant.generation = current.generation + 1
        mutant.parent_pipeline = current.name
        mutant.fitness_score = 0.0
        mutant.tasks_tested = 0
        mutant.tasks_correct = 0

        mutation_type = random.choice(["add", "remove", "reorder", "change_role"])

        if mutation_type == "add" and len(mutant.stages) < 8:
            # Add a new stage
            new_stage = WorkflowStage(
                id=random.choice(["verify", "debate", "memory_check", "refine", "simplify"]),
                role=random.choice(["skeptic", "scientist", "engineer", "historian"]),
                action=random.choice([
                    "Verify the reasoning and check for errors",
                    "Debate the proposed solution",
                    "Check memory for similar past solutions",
                    "Refine and improve the current answer",
                    "Simplify the solution while maintaining correctness",
                ]),
                input_from=[mutant.stages[-1].id] if mutant.stages else ["input"],
                max_agents=1,
            )
            # Insert at a random position (but not first)
            pos = random.randint(1, len(mutant.stages))
            mutant.stages.insert(pos, new_stage)
            # Fix input_from for the next stage
            if pos < len(mutant.stages) - 1:
                mutant.stages[pos + 1].input_from = [new_stage.id]

            logger.info(f"  Workflow mutation: ADD '{new_stage.id}' at position {pos}")

        elif mutation_type == "remove" and len(mutant.stages) > 2:
            # Remove a random non-essential stage
            removable = [s for s in mutant.stages if s.id not in ("plan", "synthesize")]
            if removable:
                to_remove = random.choice(removable)
                idx = mutant.stages.index(to_remove)
                mutant.stages.remove(to_remove)
                # Fix input_from references
                for stage in mutant.stages:
                    stage.input_from = [
                        ref if ref != to_remove.id else
                        (mutant.stages[max(0, idx-1)].id if mutant.stages else "input")
                        for ref in stage.input_from
                    ]
                logger.info(f"  Workflow mutation: REMOVE '{to_remove.id}'")

        elif mutation_type == "reorder" and len(mutant.stages) >= 3:
            # Swap two adjacent stages (not first or last)
            if len(mutant.stages) >= 4:
                idx = random.randint(1, len(mutant.stages) - 3)
                mutant.stages[idx], mutant.stages[idx + 1] = (
                    mutant.stages[idx + 1], mutant.stages[idx]
                )
                logger.info(
                    f"  Workflow mutation: SWAP '{mutant.stages[idx].id}' "
                    f"↔ '{mutant.stages[idx+1].id}'"
                )

        elif mutation_type == "change_role":
            # Change which role handles a stage
            if mutant.stages:
                stage = random.choice(mutant.stages)
                old_role = stage.role
                stage.role = random.choice([
                    "scientist", "engineer", "creative", "skeptic",
                    "planner", "historian", "ethics", "synthesizer",
                ])
                logger.info(
                    f"  Workflow mutation: ROLE '{stage.id}' "
                    f"{old_role} → {stage.role}"
                )

        return mutant

    def _parse_pipeline_from_response(
        self, response: str, current: WorkflowPipeline
    ) -> WorkflowPipeline:
        """Try to extract a YAML pipeline from LLM response."""
        # Look for YAML block in the response
        yaml_start = response.find("```yaml")
        yaml_end = response.find("```", yaml_start + 7) if yaml_start >= 0 else -1

        if yaml_start >= 0 and yaml_end >= 0:
            yaml_text = response[yaml_start + 7:yaml_end].strip()
            try:
                data = yaml.safe_load(yaml_text)
                if data and "stages" in data:
                    stages = []
                    for s in data["stages"]:
                        stages.append(WorkflowStage(
                            id=s.get("id", "unknown"),
                            role=s.get("role", "scientist"),
                            action=s.get("action", "Process the input"),
                            input_from=s.get("input_from", ["input"]),
                            max_agents=s.get("max_agents", 1),
                        ))

                    pipeline = WorkflowPipeline(
                        name=data.get("name", f"{current.name}_evolved"),
                        description=data.get("description", "LLM-proposed pipeline"),
                        generation=current.generation + 1,
                        stages=stages,
                        parent_pipeline=current.name,
                    )
                    return pipeline
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML from LLM response: {e}")

        # Fallback to random mutation
        return self._random_mutate(current)
