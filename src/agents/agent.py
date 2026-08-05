"""
Agent — Core agent class with evolvable cognitive traits.

Each agent in the AI Civilization has:
- Identity (name, role, generation, lineage)
- Cognitive traits (system prompt, reasoning strategy, tool policy, etc.)
- Performance metrics (accuracy, reputation, contribution scores)
- Memory (episodic, semantic, social)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Available reasoning strategies that agents can evolve between
REASONING_STRATEGIES = [
    "chain-of-thought",
    "tree-of-thought",
    "decomposition",
    "analogical",
    "adversarial",
    "retrieval-augmented",
    "consequentialist",
    "integrative",
    "first-principles",
    "debate",
]

# Available tool usage policies
TOOL_POLICIES = [
    "reason first, search only if lacking domain knowledge",
    "always search before reasoning",
    "search only after reasoning to verify",
    "no external search — rely on internal knowledge",
    "search for inspiration from adjacent domains",
    "search to verify claims made by other agents",
]


@dataclass
class MemoryEntry:
    """A single memory entry — an experience the agent remembers."""

    timestamp: str
    task_id: str
    task_description: str
    agent_response: str
    was_correct: bool
    feedback: str = ""
    importance: float = 0.5  # 0-1, how important this memory is

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "task": self.task_description,
            "response": self.agent_response,
            "correct": self.was_correct,
            "feedback": self.feedback,
            "importance": self.importance,
        }


@dataclass
class AgentMemory:
    """Agent's memory system — episodic, semantic, and social memory."""

    # Episodic: past task experiences (limited to last N entries)
    episodic: list[MemoryEntry] = field(default_factory=list)
    max_episodic: int = 100

    # Semantic: learned domain knowledge (key-value pairs)
    semantic: dict[str, str] = field(default_factory=dict)

    # Social: reputation scores of other agents
    social: dict[str, float] = field(default_factory=dict)  # agent_id -> trust_score

    def add_episode(self, entry: MemoryEntry) -> None:
        """Add an episodic memory, evicting oldest if at capacity."""
        self.episodic.append(entry)
        if len(self.episodic) > self.max_episodic:
            # Remove least important old memories
            self.episodic.sort(key=lambda e: e.importance, reverse=True)
            self.episodic = self.episodic[: self.max_episodic]

    def add_knowledge(self, key: str, value: str) -> None:
        """Store a semantic knowledge fact."""
        self.semantic[key] = value

    def update_trust(self, agent_id: str, score: float) -> None:
        """Update trust score for another agent."""
        if agent_id in self.social:
            # Exponential moving average
            self.social[agent_id] = 0.7 * self.social[agent_id] + 0.3 * score
        else:
            self.social[agent_id] = score

    def get_relevant_memories(self, task_description: str, limit: int = 5) -> list[MemoryEntry]:
        """Retrieve memories most relevant to a task (simple keyword match for now)."""
        task_words = set(task_description.lower().split())
        scored = []
        for entry in self.episodic:
            entry_words = set(entry.task_description.lower().split())
            overlap = len(task_words & entry_words) / max(len(task_words), 1)
            scored.append((overlap + entry.importance, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def to_dict(self) -> dict:
        return {
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "social_connections": len(self.social),
        }


@dataclass
class Agent:
    """
    A single AI agent in the civilization.

    Agents have evolvable cognitive traits — their system prompt, reasoning
    strategy, tool policy, confidence threshold, and memory strategy can
    all be mutated or recombined during evolution.
    """

    # ── Identity ──
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "Agent"
    role: str = "generalist"
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    born_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Cognitive Traits (EVOLVABLE) ──
    system_prompt: str = "You are a helpful AI assistant."
    reasoning_strategy: str = "chain-of-thought"
    tool_policy: str = "reason first, search only if lacking domain knowledge"
    memory_strategy: str = "remember successful reasoning patterns"
    confidence_threshold: float = 0.7  # 0-1, how confident before speaking

    # ── Performance Metrics ──
    tasks_attempted: int = 0
    tasks_correct: int = 0
    total_contribution_score: float = 0.0
    reputation_score: float = 0.5  # peer-rated, 0-1
    age: int = 0  # evolution cycles survived

    # ── Memory ──
    memory: AgentMemory = field(default_factory=AgentMemory)

    # ── Status ──
    alive: bool = True
    archived_at: str | None = None

    @property
    def accuracy(self) -> float:
        """Task accuracy as a fraction."""
        if self.tasks_attempted == 0:
            return 0.0
        return self.tasks_correct / self.tasks_attempted

    @property
    def avg_contribution(self) -> float:
        """Average contribution score per task."""
        if self.tasks_attempted == 0:
            return 0.0
        return self.total_contribution_score / self.tasks_attempted

    def compute_fitness(self, weights: dict[str, float] | None = None) -> float:
        """
        Compute overall fitness score.

        Fitness combines accuracy, reputation, unique contribution,
        efficiency (1/age penalty removed — survival is rewarded), and consistency.
        """
        if weights is None:
            weights = {
                "accuracy": 0.40,
                "reputation": 0.20,
                "unique_contribution": 0.20,
                "efficiency": 0.10,
                "consistency": 0.10,
            }

        # Consistency: agents with more tasks have more reliable scores
        consistency = min(self.tasks_attempted / 20.0, 1.0)

        # Efficiency: reward survival (age) — longer-lived agents are proven
        efficiency = min(self.age / 10.0, 1.0)

        fitness = (
            weights["accuracy"] * self.accuracy
            + weights["reputation"] * self.reputation_score
            + weights["unique_contribution"] * self.avg_contribution
            + weights["efficiency"] * efficiency
            + weights["consistency"] * consistency
        )
        return round(fitness, 4)

    def record_task(self, task_id: str, task_desc: str, response: str,
                    was_correct: bool, contribution: float = 0.0) -> None:
        """Record the result of a task attempt."""
        self.tasks_attempted += 1
        if was_correct:
            self.tasks_correct += 1
        self.total_contribution_score += contribution

        self.memory.add_episode(MemoryEntry(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            task_description=task_desc,
            agent_response=response,
            was_correct=was_correct,
            importance=0.8 if not was_correct else 0.5,  # failures are more memorable
        ))

    def build_context_prompt(self, task: str) -> str:
        """Build the full prompt including personality, memories, and task."""
        parts = [self.system_prompt]

        # Add reasoning strategy instruction
        parts.append(f"\n## Reasoning Strategy\nUse {self.reasoning_strategy} reasoning.")

        # Add relevant memories
        memories = self.memory.get_relevant_memories(task)
        if memories:
            parts.append("\n## Relevant Past Experience")
            for mem in memories[:3]:
                status = "✓" if mem.was_correct else "✗"
                parts.append(f"- [{status}] Task: {mem.task_description[:80]}...")

        # Add confidence instruction
        parts.append(
            f"\n## Confidence\nOnly contribute if you are at least "
            f"{int(self.confidence_threshold * 100)}% confident in your reasoning."
        )

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent to dict (for storage/display)."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "generation": self.generation,
            "parents": self.parents,
            "born_at": self.born_at,
            "reasoning_strategy": self.reasoning_strategy,
            "confidence_threshold": self.confidence_threshold,
            "tasks_attempted": self.tasks_attempted,
            "accuracy": round(self.accuracy, 3),
            "reputation": round(self.reputation_score, 3),
            "fitness": self.compute_fitness(),
            "alive": self.alive,
            "age": self.age,
            "memory": self.memory.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"Agent({self.name}, role={self.role}, gen={self.generation}, "
            f"acc={self.accuracy:.1%}, fitness={self.compute_fitness():.3f})"
        )
