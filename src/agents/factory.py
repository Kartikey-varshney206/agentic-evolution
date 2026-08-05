"""
Agent Factory — Creates agents from config or through evolution (birth/recombination).
"""

from __future__ import annotations

import copy
import random
import uuid
from pathlib import Path

import yaml

from .agent import Agent, REASONING_STRATEGIES, TOOL_POLICIES


def load_agents_from_config(config_path: str = "config/default_agents.yaml") -> list[Agent]:
    """Load initial agent population from YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    agents = []
    for agent_def in config.get("agents", []):
        agent = Agent(
            name=agent_def["name"],
            role=agent_def["role"],
            system_prompt=agent_def["system_prompt"].strip(),
            reasoning_strategy=agent_def.get("reasoning_strategy", "chain-of-thought"),
            tool_policy=agent_def.get("tool_policy", TOOL_POLICIES[0]),
            memory_strategy=agent_def.get("memory_strategy", "remember everything"),
            confidence_threshold=agent_def.get("confidence_threshold", 0.7),
            generation=0,
        )
        agents.append(agent)

    return agents


def create_random_agent(role: str | None = None, generation: int = 0) -> Agent:
    """Create a new agent with random traits."""
    if role is None:
        role = random.choice([
            "scientist", "engineer", "creative", "skeptic",
            "planner", "historian", "ethics", "synthesizer",
        ])

    agent_id = str(uuid.uuid4())[:12]
    name = f"{role.capitalize()}-{agent_id[:4]}"

    return Agent(
        id=agent_id,
        name=name,
        role=role,
        generation=generation,
        system_prompt=_generate_role_prompt(role),
        reasoning_strategy=random.choice(REASONING_STRATEGIES),
        tool_policy=random.choice(TOOL_POLICIES),
        confidence_threshold=round(random.uniform(0.4, 0.9), 2),
    )


def clone_agent(parent: Agent, generation: int | None = None) -> Agent:
    """Create an exact copy of an agent with a new ID."""
    child = copy.deepcopy(parent)
    child.id = str(uuid.uuid4())[:12]
    child.name = f"{parent.role.capitalize()}-{child.id[:4]}"
    child.generation = generation if generation is not None else parent.generation + 1
    child.parents = [parent.id]
    child.tasks_attempted = 0
    child.tasks_correct = 0
    child.total_contribution_score = 0.0
    child.reputation_score = 0.5
    child.age = 0
    child.memory = copy.deepcopy(parent.memory)  # inherit memories
    return child


def recombine_agents(parent_a: Agent, parent_b: Agent, generation: int | None = None) -> Agent:
    """
    Combine cognitive traits from two parent agents to create a child.

    This is the 'Agent Birth' mechanism — analogous to biological recombination
    but applied to cognitive traits instead of genes.
    """
    gen = generation if generation is not None else max(parent_a.generation, parent_b.generation) + 1

    # Determine role: pick from parents or create hybrid
    if parent_a.role == parent_b.role:
        role = parent_a.role
    else:
        role = random.choice([parent_a.role, parent_b.role])

    child_id = str(uuid.uuid4())[:12]

    child = Agent(
        id=child_id,
        name=f"{role.capitalize()}-{child_id[:4]}",
        role=role,
        generation=gen,
        parents=[parent_a.id, parent_b.id],

        # Recombine cognitive traits — pick from either parent with some mixing
        system_prompt=_merge_prompts(parent_a.system_prompt, parent_b.system_prompt),
        reasoning_strategy=random.choice([
            parent_a.reasoning_strategy,
            parent_b.reasoning_strategy,
        ]),
        tool_policy=random.choice([parent_a.tool_policy, parent_b.tool_policy]),
        memory_strategy=random.choice([parent_a.memory_strategy, parent_b.memory_strategy]),
        confidence_threshold=round(
            (parent_a.confidence_threshold + parent_b.confidence_threshold) / 2
            + random.uniform(-0.1, 0.1),
            2,
        ),
    )

    return child


def _merge_prompts(prompt_a: str, prompt_b: str) -> str:
    """
    Merge two system prompts into a hybrid.

    For now, uses a simple approach: take the first half of one and
    second half of the other. In Phase 2, this will use an LLM to
    intelligently merge the prompts.
    """
    lines_a = prompt_a.strip().split("\n")
    lines_b = prompt_b.strip().split("\n")

    mid_a = len(lines_a) // 2
    mid_b = len(lines_b) // 2

    merged_lines = lines_a[:mid_a] + lines_b[mid_b:]
    return "\n".join(merged_lines)


def _generate_role_prompt(role: str) -> str:
    """Generate a basic system prompt for a given role."""
    prompts = {
        "scientist": (
            "You are a rigorous scientist AI. You approach every problem with "
            "the scientific method. You value evidence and clear reasoning."
        ),
        "engineer": (
            "You are a pragmatic engineer AI. You focus on practical, "
            "implementable solutions. You think about edge cases."
        ),
        "creative": (
            "You are a creative thinker AI. You approach problems from "
            "unexpected angles and use analogies and lateral thinking."
        ),
        "skeptic": (
            "You are a skeptic AI. You find flaws, errors, and weaknesses "
            "in reasoning. You play devil's advocate constructively."
        ),
        "planner": (
            "You are a strategic planner AI. You decompose complex problems "
            "into manageable steps and think about dependencies."
        ),
        "historian": (
            "You are a historian AI. You remember past solutions and "
            "identify patterns. You warn about repeating past mistakes."
        ),
        "ethics": (
            "You are an ethics AI. You evaluate solutions for fairness, "
            "safety, and unintended consequences."
        ),
        "synthesizer": (
            "You are a synthesizer AI. You integrate multiple perspectives "
            "into a unified, coherent answer."
        ),
    }
    return prompts.get(role, "You are a helpful AI assistant.")
