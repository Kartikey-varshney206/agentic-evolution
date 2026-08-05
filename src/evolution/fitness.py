"""
Fitness — Evaluates agent performance and computes fitness scores.

Fitness is the core metric driving evolution. Agents with higher fitness
survive, reproduce, and spread their cognitive traits. Agents with low
fitness are culled.
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.agent import Agent

logger = logging.getLogger(__name__)


# Default fitness weights
DEFAULT_WEIGHTS = {
    "accuracy": 0.40,
    "reputation": 0.20,
    "unique_contribution": 0.20,
    "efficiency": 0.10,
    "consistency": 0.10,
}


def compute_fitness(agent: Agent, weights: dict[str, float] | None = None) -> float:
    """Compute fitness score for a single agent."""
    return agent.compute_fitness(weights or DEFAULT_WEIGHTS)


def rank_agents(agents: list[Agent], weights: dict[str, float] | None = None) -> list[Agent]:
    """Rank agents by fitness, highest first."""
    w = weights or DEFAULT_WEIGHTS
    return sorted(agents, key=lambda a: a.compute_fitness(w), reverse=True)


def compute_diversity_bonus(agent: Agent, all_agents: list[Agent], bonus: float = 0.05) -> float:
    """
    Give a fitness bonus to agents whose role/strategy is rare in the population.

    This prevents the society from converging to a monoculture where all
    agents are identical — diversity pressure keeps the ecosystem varied.
    """
    if not all_agents:
        return 0.0

    # Count how many agents share this role
    same_role = sum(1 for a in all_agents if a.role == agent.role)
    role_fraction = same_role / len(all_agents)

    # Count how many share this reasoning strategy
    same_strategy = sum(
        1 for a in all_agents if a.reasoning_strategy == agent.reasoning_strategy
    )
    strategy_fraction = same_strategy / len(all_agents)

    # Rare agents get higher bonus
    # If you're 1 of 20 with a unique role, you get full bonus
    # If you're 10 of 20 (50%), you get very little
    role_bonus = bonus * (1.0 - role_fraction)
    strategy_bonus = bonus * (1.0 - strategy_fraction)

    return round(role_bonus + strategy_bonus, 4)


def get_population_stats(agents: list[Agent]) -> dict[str, Any]:
    """Get detailed population statistics for logging/dashboard."""
    if not agents:
        return {"population": 0}

    fitnesses = [a.compute_fitness() for a in agents]
    accuracies = [a.accuracy for a in agents if a.tasks_attempted > 0]

    # Role distribution
    roles: dict[str, int] = {}
    for a in agents:
        roles[a.role] = roles.get(a.role, 0) + 1

    # Strategy distribution
    strategies: dict[str, int] = {}
    for a in agents:
        strategies[a.reasoning_strategy] = strategies.get(a.reasoning_strategy, 0) + 1

    # Generation distribution
    generations: dict[int, int] = {}
    for a in agents:
        generations[a.generation] = generations.get(a.generation, 0) + 1

    return {
        "population": len(agents),
        "fitness": {
            "mean": round(sum(fitnesses) / len(fitnesses), 4),
            "max": round(max(fitnesses), 4),
            "min": round(min(fitnesses), 4),
        },
        "accuracy": {
            "mean": round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
            "max": round(max(accuracies), 4) if accuracies else 0,
        },
        "role_distribution": roles,
        "strategy_distribution": strategies,
        "generation_distribution": generations,
        "unique_roles": len(roles),
        "unique_strategies": len(strategies),
    }
