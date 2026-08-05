"""
Mutation — Randomly modifies agent cognitive traits.

Mutations operate at the behavioral/strategy level, not the parameter level.
This is a key differentiator from traditional neural architecture evolution.

Mutation types:
- System prompt mutation (personality/expertise changes)
- Reasoning strategy mutation (switch between strategies)
- Tool policy mutation (change when/how to use tools)
- Confidence threshold mutation (speak up more/less)
- Memory strategy mutation (what to remember)
"""

from __future__ import annotations

import copy
import random
import logging

from ..agents.agent import Agent, REASONING_STRATEGIES, TOOL_POLICIES
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


def mutate_agent(
    agent: Agent,
    mutation_rate: float = 0.30,
    llm_client: LLMClient | None = None,
    generation: int | None = None,
) -> Agent:
    """
    Create a mutated copy of an agent.

    Each cognitive trait has a chance of being mutated. The original
    agent is not modified — a new agent is returned.

    If an LLM client is provided, uses it for intelligent prompt mutation.
    Otherwise, uses simpler heuristic mutations.
    """
    child = copy.deepcopy(agent)
    child.id = f"{agent.id[:4]}-m{random.randint(100,999)}"
    child.name = f"{agent.role.capitalize()}-{child.id[:8]}"
    child.generation = generation if generation is not None else agent.generation + 1
    child.parents = [agent.id]
    child.tasks_attempted = 0
    child.tasks_correct = 0
    child.total_contribution_score = 0.0
    child.reputation_score = 0.5
    child.age = 0

    mutations_applied = []

    # ── System Prompt Mutation ──
    if random.random() < mutation_rate:
        if llm_client:
            child.system_prompt = _llm_mutate_prompt(agent.system_prompt, llm_client)
        else:
            child.system_prompt = _simple_mutate_prompt(agent.system_prompt)
        mutations_applied.append("system_prompt")

    # ── Reasoning Strategy Mutation ──
    if random.random() < mutation_rate:
        other_strategies = [s for s in REASONING_STRATEGIES if s != agent.reasoning_strategy]
        if other_strategies:
            child.reasoning_strategy = random.choice(other_strategies)
            mutations_applied.append(f"strategy→{child.reasoning_strategy}")

    # ── Tool Policy Mutation ──
    if random.random() < mutation_rate:
        other_policies = [p for p in TOOL_POLICIES if p != agent.tool_policy]
        if other_policies:
            child.tool_policy = random.choice(other_policies)
            mutations_applied.append("tool_policy")

    # ── Confidence Threshold Mutation ──
    if random.random() < mutation_rate:
        delta = random.uniform(-0.15, 0.15)
        child.confidence_threshold = round(
            max(0.1, min(0.95, agent.confidence_threshold + delta)), 2
        )
        mutations_applied.append(f"confidence→{child.confidence_threshold}")

    # Ensure at least one mutation happened
    if not mutations_applied:
        # Force a strategy mutation
        other_strategies = [s for s in REASONING_STRATEGIES if s != agent.reasoning_strategy]
        if other_strategies:
            child.reasoning_strategy = random.choice(other_strategies)
            mutations_applied.append(f"strategy→{child.reasoning_strategy}")

    logger.info(
        f"Mutated {agent.name} → {child.name}: {', '.join(mutations_applied)}"
    )
    return child


def _llm_mutate_prompt(prompt: str, llm: LLMClient) -> str:
    """
    Use an LLM to intelligently mutate a system prompt.

    The LLM suggests a small variation that might improve reasoning quality.
    """
    mutation_instruction = (
        "You are a prompt engineer specializing in AI agent optimization. "
        "Given the following AI agent's system prompt, make ONE small but "
        "meaningful modification that could improve the agent's reasoning. "
        "Keep the core personality. Only change one aspect — e.g., add a new "
        "thinking technique, adjust a priority, or refine an instruction. "
        "Return ONLY the new prompt, nothing else."
    )

    try:
        response = llm.chat(
            mutation_instruction,
            f"Current prompt:\n\n{prompt}\n\nNew improved prompt:",
            temperature=0.9,  # high temp for creative mutations
        )
        new_prompt = response.content.strip()
        # Sanity check: don't accept empty or extremely short prompts
        if len(new_prompt) > 20:
            return new_prompt
    except Exception as e:
        logger.warning(f"LLM prompt mutation failed: {e}")

    return _simple_mutate_prompt(prompt)


def _simple_mutate_prompt(prompt: str) -> str:
    """Simple heuristic prompt mutation (no LLM needed)."""
    additions = [
        "\nAlways consider edge cases before concluding.",
        "\nWhen uncertain, reason about what you DON'T know.",
        "\nTry to find at least one counterexample to your conclusion.",
        "\nExplain your confidence level and reasoning steps.",
        "\nConsider multiple perspectives before committing to an answer.",
        "\nBe concise but thorough — quality over quantity.",
        "\nThink step by step and verify each step.",
        "\nConsider the simplest explanation first (Occam's razor).",
        "\nChallenge your own assumptions before concluding.",
        "\nLook for patterns in the problem that connect to known solutions.",
    ]

    # Add a random enhancement
    addition = random.choice(additions)

    # Sometimes replace a sentence instead of adding
    lines = prompt.strip().split("\n")
    if len(lines) > 3 and random.random() < 0.3:
        # Replace a random non-first line
        idx = random.randint(1, len(lines) - 1)
        lines[idx] = addition.strip()
        return "\n".join(lines)
    else:
        return prompt.strip() + addition
