"""
Society — The container for the AI civilization.

Manages the population of agents, routes tasks through the workflow
pipeline, tracks history, and provides the interface for evolution
and governance modules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents.agent import Agent
from ..agents.factory import load_agents_from_config
from .discussion import DiscussionEngine
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of the society processing a single task."""

    task_id: str
    task_description: str
    final_answer: str
    agent_responses: dict[str, str]  # agent_id -> response
    discussion_log: list[dict]
    correct: bool | None = None  # None until evaluated
    ground_truth: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    workflow_used: str = "baseline_v1"


class Society:
    """
    The AI Society — manages agents, runs tasks through discussions,
    and provides hooks for evolution and governance.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        agents_config: str = "config/default_agents.yaml",
    ):
        self.llm = llm_client or LLMClient()
        self.agents: dict[str, Agent] = {}
        self.archived_agents: dict[str, Agent] = {}
        self.task_history: list[TaskResult] = []
        self.generation: int = 0
        self.tasks_processed: int = 0

        # Load initial population
        self._load_initial_agents(agents_config)

        # Initialize discussion engine
        self.discussion_engine = DiscussionEngine(self.llm)

        logger.info(
            f"Society initialized with {len(self.agents)} agents "
            f"(gen {self.generation})"
        )

    def _load_initial_agents(self, config_path: str) -> None:
        """Load agents from config file."""
        try:
            agents = load_agents_from_config(config_path)
            for agent in agents:
                self.agents[agent.id] = agent
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}. Starting with empty society.")

    # ── Core Task Processing ──

    def solve(self, task: str, ground_truth: str | None = None) -> TaskResult:
        """
        Route a task through the AI society for collaborative solving.

        This is the main entry point — it:
        1. Selects relevant agents
        2. Runs multi-round discussion
        3. Synthesizes a final answer
        4. Records the result
        """
        task_id = f"task-{self.tasks_processed:04d}"
        self.tasks_processed += 1

        logger.info(f"Processing {task_id}: {task[:80]}...")

        # Get all alive agents
        active_agents = [a for a in self.agents.values() if a.alive]

        if not active_agents:
            return TaskResult(
                task_id=task_id,
                task_description=task,
                final_answer="ERROR: No active agents in society.",
                agent_responses={},
                discussion_log=[],
            )

        # Run discussion
        discussion_result = self.discussion_engine.run_discussion(
            task=task,
            agents=active_agents,
            rounds=2,
        )

        # Build result
        result = TaskResult(
            task_id=task_id,
            task_description=task,
            final_answer=discussion_result["final_answer"],
            agent_responses=discussion_result["individual_responses"],
            discussion_log=discussion_result["discussion_log"],
            ground_truth=ground_truth,
        )

        # Evaluate if ground truth provided
        if ground_truth is not None:
            result.correct = self._evaluate_answer(
                result.final_answer, ground_truth
            )

        # Record task for all participating agents
        for agent_id, response in discussion_result["individual_responses"].items():
            if agent_id in self.agents:
                contribution = discussion_result.get("contributions", {}).get(agent_id, 0.0)
                self.agents[agent_id].record_task(
                    task_id=task_id,
                    task_desc=task,
                    response=response,
                    was_correct=result.correct or False,
                    contribution=contribution,
                )

        self.task_history.append(result)
        return result

    def _evaluate_answer(self, answer: str, ground_truth: str) -> bool:
        """
        Check if the answer matches ground truth.

        Uses LLM-as-judge for flexible matching (handles different formats).
        """
        judge_prompt = (
            "You are an answer evaluator. Compare the given answer to the ground truth. "
            "Respond with only 'CORRECT' or 'INCORRECT'.\n\n"
            "Be generous — if the answer captures the same meaning or arrives at "
            "the same result, it is correct even if phrased differently."
        )
        judge_message = (
            f"Ground truth: {ground_truth}\n\n"
            f"Given answer: {answer}\n\n"
            f"Is the given answer correct? Respond ONLY with CORRECT or INCORRECT."
        )

        try:
            response = self.llm.chat(judge_prompt, judge_message, temperature=0.0)
            return "CORRECT" in response.content.upper()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False

    # ── Agent Management ──

    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the society."""
        self.agents[agent.id] = agent
        logger.info(f"Agent joined: {agent.name} (gen {agent.generation})")

    def remove_agent(self, agent_id: str) -> Agent | None:
        """Remove an agent from the society (death). Archives it."""
        if agent_id not in self.agents:
            return None

        agent = self.agents.pop(agent_id)
        agent.alive = False
        agent.archived_at = datetime.now().isoformat()
        self.archived_agents[agent_id] = agent
        logger.info(f"Agent removed: {agent.name} (fitness: {agent.compute_fitness():.3f})")
        return agent

    def get_alive_agents(self) -> list[Agent]:
        """Get all currently alive agents."""
        return [a for a in self.agents.values() if a.alive]

    def get_agents_by_role(self, role: str) -> list[Agent]:
        """Get all alive agents with a specific role."""
        return [a for a in self.agents.values() if a.alive and a.role == role]

    # ── Statistics ──

    def get_stats(self) -> dict[str, Any]:
        """Get society-level statistics."""
        alive = self.get_alive_agents()
        accuracies = [a.accuracy for a in alive if a.tasks_attempted > 0]
        fitnesses = [a.compute_fitness() for a in alive if a.tasks_attempted > 0]

        # Role distribution
        role_counts: dict[str, int] = {}
        for agent in alive:
            role_counts[agent.role] = role_counts.get(agent.role, 0) + 1

        # Recent task performance
        recent = self.task_history[-20:] if self.task_history else []
        recent_correct = sum(1 for t in recent if t.correct)
        recent_accuracy = recent_correct / len(recent) if recent else 0.0

        return {
            "generation": self.generation,
            "population": len(alive),
            "archived": len(self.archived_agents),
            "tasks_processed": self.tasks_processed,
            "role_distribution": role_counts,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "recent_accuracy": recent_accuracy,
            "llm_usage": self.llm.get_usage_stats(),
        }

    def age_all_agents(self) -> None:
        """Increment age for all alive agents (called each evolution cycle)."""
        for agent in self.get_alive_agents():
            agent.age += 1
        self.generation += 1

    def save_state(self, path: str = "society_state.json") -> None:
        """Save the full society state to disk."""
        state = {
            "generation": self.generation,
            "tasks_processed": self.tasks_processed,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "archived": {aid: a.to_dict() for aid, a in self.archived_agents.items()},
            "stats": self.get_stats(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Society state saved to {path}")

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Society(gen={stats['generation']}, pop={stats['population']}, "
            f"tasks={stats['tasks_processed']}, acc={stats['avg_accuracy']:.1%})"
        )
