"""
Selection — Decides which agents survive, reproduce, or die.

Implements tournament selection for reproduction and bottom-percentile
culling for agent death.
"""

from __future__ import annotations

import random
import logging

from ..agents.agent import Agent
from .fitness import compute_fitness, rank_agents

logger = logging.getLogger(__name__)


def select_parents(
    agents: list[Agent],
    tournament_size: int = 4,
    num_parents: int = 2,
) -> list[Agent]:
    """
    Tournament selection — pick random subset, take the best.

    This is used to select parents for recombination (agent birth).
    """
    if len(agents) < num_parents:
        return agents[:]

    parents = []
    for _ in range(num_parents):
        # Pick random tournament
        tournament = random.sample(agents, min(tournament_size, len(agents)))
        # Winner is the fittest
        winner = max(tournament, key=lambda a: a.compute_fitness())
        parents.append(winner)
        # Don't pick the same parent twice
        agents = [a for a in agents if a.id != winner.id]

    return parents


def select_elite(agents: list[Agent], elite_fraction: float = 0.20) -> list[Agent]:
    """
    Select the elite — top performers protected from death.

    Elite agents survive every evolution cycle, ensuring the best
    cognitive traits are preserved.
    """
    ranked = rank_agents(agents)
    elite_count = max(1, int(len(ranked) * elite_fraction))
    elite = ranked[:elite_count]

    logger.info(
        f"Elite selected: {len(elite)} agents "
        f"(top fitness: {elite[0].compute_fitness():.3f})"
    )
    return elite


def select_for_death(
    agents: list[Agent],
    death_fraction: float = 0.10,
    protected_ids: set[str] | None = None,
) -> list[Agent]:
    """
    Select the weakest agents for removal (death).

    Protected agents (elite) cannot be selected for death.
    Ensures minimum role diversity is maintained.
    """
    protected = protected_ids or set()

    # Filter out protected agents
    vulnerable = [a for a in agents if a.id not in protected]

    if not vulnerable:
        return []

    # Rank by fitness (worst first)
    ranked = rank_agents(vulnerable)
    ranked.reverse()

    death_count = max(1, int(len(agents) * death_fraction))

    # Don't kill the last agent of any role
    role_counts: dict[str, int] = {}
    for a in agents:
        role_counts[a.role] = role_counts.get(a.role, 0) + 1

    to_die = []
    for agent in ranked:
        if len(to_die) >= death_count:
            break
        # Don't kill the last of a role
        if role_counts.get(agent.role, 0) <= 1:
            continue
        to_die.append(agent)
        role_counts[agent.role] -= 1

    logger.info(
        f"Selected {len(to_die)} agents for death "
        f"(worst fitness: {to_die[0].compute_fitness():.3f})" if to_die else
        "No agents selected for death"
    )
    return to_die
