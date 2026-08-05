"""
Evolution Engine — Orchestrates the full evolutionary cycle.

Every N tasks, the society undergoes an evolution cycle:
1. Evaluate fitness of all agents
2. Select elite (protected from death)
3. Kill weakest agents
4. Mutate middle-tier agents
5. Recombine top agents to create new ones (birth)
6. Age all surviving agents
7. Log the generation

This implements Parts 2-5 of the AI Civilization:
- Evolution (selection pressure)
- Agent Birth (recombination)
- Agent Death (pruning)
- Mutation (random trait changes)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..agents.agent import Agent
from ..agents.factory import recombine_agents, create_random_agent
from .fitness import (
    rank_agents,
    compute_diversity_bonus,
    get_population_stats,
)
from .selection import select_parents, select_elite, select_for_death
from .mutation import mutate_agent
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""

    evolution_interval: int = 20
    min_population: int = 10
    max_population: int = 60
    elite_fraction: float = 0.20
    death_fraction: float = 0.10
    tournament_size: int = 4
    mutation_rate: float = 0.30
    births_per_cycle: int = 2
    recombination_probability: float = 0.6
    diversity_bonus: float = 0.05
    min_role_types: int = 4

    @classmethod
    def from_yaml(cls, path: str = "config/evolution_params.yaml") -> "EvolutionConfig":
        """Load config from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config not found: {path}, using defaults")
            return cls()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        evo = raw.get("evolution", {})
        weights = evo.get("fitness_weights", {})

        return cls(
            evolution_interval=evo.get("evolution_interval", 20),
            min_population=evo.get("min_population", 10),
            max_population=evo.get("max_population", 60),
            elite_fraction=evo.get("elite_fraction", 0.20),
            death_fraction=evo.get("death_fraction", 0.10),
            tournament_size=evo.get("tournament_size", 4),
            mutation_rate=evo.get("mutation_rate", 0.30),
            births_per_cycle=evo.get("births_per_cycle", 2),
            recombination_probability=evo.get("recombination_probability", 0.6),
            diversity_bonus=evo.get("diversity_bonus", 0.05),
            min_role_types=evo.get("min_role_types", 4),
        )


@dataclass
class GenerationRecord:
    """Record of a single evolution generation."""

    generation: int
    timestamp: str
    population_before: int
    population_after: int
    agents_born: list[str]
    agents_died: list[str]
    agents_mutated: list[str]
    elite_agents: list[str]
    stats: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "timestamp": self.timestamp,
            "population": {
                "before": self.population_before,
                "after": self.population_after,
            },
            "born": self.agents_born,
            "died": self.agents_died,
            "mutated": self.agents_mutated,
            "elite": self.elite_agents,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationRecord":
        return cls(
            generation=data["generation"],
            timestamp=data["timestamp"],
            population_before=data["population"]["before"],
            population_after=data["population"]["after"],
            agents_born=data["born"],
            agents_died=data["died"],
            agents_mutated=data["mutated"],
            elite_agents=data.get("elite", []),
            stats=data["stats"],
        )


class EvolutionEngine:
    """
    Orchestrates the evolutionary cycle for the AI society.

    This engine is the core of what makes the civilization "alive" —
    it continuously selects, mutates, breeds, and culls agents based
    on their performance.
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.llm = llm_client
        self.history: list[GenerationRecord] = []

    def should_evolve(self, tasks_processed: int) -> bool:
        """Check if it's time to run an evolution cycle."""
        return (
            tasks_processed > 0
            and tasks_processed % self.config.evolution_interval == 0
        )

    def run_cycle(
        self,
        agents: dict[str, Agent],
        generation: int,
    ) -> tuple[dict[str, Agent], GenerationRecord]:
        """
        Run a complete evolution cycle.

        Args:
            agents: Current population {id: Agent}
            generation: Current generation number

        Returns:
            Tuple of (new_agents_dict, generation_record)
        """
        alive = [a for a in agents.values() if a.alive]
        population_before = len(alive)

        logger.info(
            f"\n{'='*60}\n"
            f"  EVOLUTION CYCLE — Generation {generation}\n"
            f"  Population: {population_before}\n"
            f"{'='*60}"
        )

        # ── Step 1: Rank and compute fitness ──
        ranked = rank_agents(alive)
        for agent in ranked:
            # Add diversity bonus
            bonus = compute_diversity_bonus(
                agent, alive, self.config.diversity_bonus
            )
            agent.reputation_score = min(1.0, agent.reputation_score + bonus)

        # ── Step 2: Select elite (protected) ──
        elite = select_elite(alive, self.config.elite_fraction)
        elite_ids = {a.id for a in elite}

        # ── Step 3: Death — remove weakest ──
        to_die = select_for_death(
            alive,
            self.config.death_fraction,
            protected_ids=elite_ids,
        )
        died_names = []
        for agent in to_die:
            if len(alive) - len(died_names) > self.config.min_population:
                agent.alive = False
                agent.archived_at = datetime.now().isoformat()
                died_names.append(agent.name)
                logger.info(f"  ☠ DEATH: {agent.name} (fitness: {agent.compute_fitness():.3f})")

        # ── Step 4: Mutation — modify middle-tier agents ──
        # Middle tier = not elite, not dead
        dead_ids = {a.id for a in to_die if not a.alive}
        middle = [a for a in alive if a.id not in elite_ids and a.id not in dead_ids]
        mutated_names = []

        for agent in middle:
            if random.random() < self.config.mutation_rate:
                mutant = mutate_agent(
                    agent,
                    mutation_rate=self.config.mutation_rate,
                    llm_client=self.llm,
                    generation=generation,
                )
                agents[mutant.id] = mutant
                mutated_names.append(mutant.name)
                logger.info(f"  🧬 MUTATION: {agent.name} → {mutant.name}")

        # ── Step 5: Birth — recombine top agents ──
        born_names = []
        alive_after_death = [a for a in agents.values() if a.alive]

        for _ in range(self.config.births_per_cycle):
            if len(alive_after_death) >= 2 and len(alive_after_death) < self.config.max_population:
                parents = select_parents(
                    alive_after_death,
                    self.config.tournament_size,
                )
                if len(parents) >= 2:
                    if random.random() < self.config.recombination_probability:
                        child = recombine_agents(parents[0], parents[1], generation)
                        logger.info(
                            f"  🌱 BIRTH: {child.name} "
                            f"(parents: {parents[0].name} × {parents[1].name})"
                        )
                    else:
                        # Clone with mutation
                        child = mutate_agent(
                            parents[0],
                            mutation_rate=0.5,
                            llm_client=self.llm,
                            generation=generation,
                        )
                        logger.info(f"  🌱 CLONE+MUTATE: {child.name} (from {parents[0].name})")

                    agents[child.id] = child
                    born_names.append(child.name)
                    alive_after_death.append(child)

        # ── Step 6: Ensure role diversity ──
        self._ensure_role_diversity(agents, generation)

        # ── Step 7: Age all surviving agents ──
        for agent in agents.values():
            if agent.alive:
                agent.age += 1

        # ── Record ──
        population_after = sum(1 for a in agents.values() if a.alive)
        stats = get_population_stats([a for a in agents.values() if a.alive])

        record = GenerationRecord(
            generation=generation,
            timestamp=datetime.now().isoformat(),
            population_before=population_before,
            population_after=population_after,
            agents_born=born_names,
            agents_died=died_names,
            agents_mutated=mutated_names,
            elite_agents=[a.name for a in elite],
            stats=stats,
        )
        self.history.append(record)

        logger.info(
            f"\n  Generation {generation} complete: "
            f"{population_before}→{population_after} agents, "
            f"{len(born_names)} born, {len(died_names)} died, "
            f"{len(mutated_names)} mutated\n"
            f"{'='*60}\n"
        )

        return agents, record

    def _ensure_role_diversity(self, agents: dict[str, Agent], generation: int) -> None:
        """Ensure minimum role diversity — add missing roles if needed."""
        alive = [a for a in agents.values() if a.alive]
        roles = {a.role for a in alive}
        required_roles = {"scientist", "engineer", "creative", "skeptic"}

        missing = required_roles - roles
        for role in missing:
            if len(alive) < self.config.max_population:
                new_agent = create_random_agent(role=role, generation=generation)
                agents[new_agent.id] = new_agent
                logger.info(f"  🔧 DIVERSITY: Created {new_agent.name} (missing role: {role})")

    def get_evolution_summary(self) -> list[dict]:
        """Get summary of all evolution generations."""
        return [record.to_dict() for record in self.history]
