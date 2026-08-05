"""
Discussion Engine — Orchestrates multi-agent collaborative reasoning.

Instead of a single model answering, the discussion engine:
1. Round 1: All agents independently reason about the task
2. Round 2: Agents see each other's responses and can refine/challenge
3. Synthesis: A synthesizer agent combines everything into a final answer

This implements the core "AI Society" paradigm:
  Question → Society → Discussions → Testing → Answer
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.agent import Agent
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class DiscussionEngine:
    """
    Orchestrates multi-round discussions between agents.

    Each round, agents contribute their reasoning. In subsequent rounds,
    they can see what others said and refine their positions.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run_discussion(
        self,
        task: str,
        agents: list[Agent],
        rounds: int = 2,
    ) -> dict[str, Any]:
        """
        Run a multi-round discussion on a task.

        Returns:
            Dict with:
            - final_answer: The synthesized answer
            - individual_responses: {agent_id: response}
            - discussion_log: Full log of all rounds
            - contributions: {agent_id: contribution_score}
        """
        discussion_log: list[dict] = []
        all_responses: dict[str, str] = {}  # agent_id -> latest response

        # ── Round 1: Independent Reasoning ──
        logger.info(f"Discussion Round 1: {len(agents)} agents reasoning independently")
        round_1_responses = self._run_round(
            task=task,
            agents=agents,
            round_num=1,
            previous_responses=None,
        )
        all_responses.update(round_1_responses)

        discussion_log.append({
            "round": 1,
            "type": "independent",
            "responses": {
                aid: resp[:200] + "..." if len(resp) > 200 else resp
                for aid, resp in round_1_responses.items()
            },
        })

        # ── Rounds 2+: Collaborative Refinement ──
        for round_num in range(2, rounds + 1):
            logger.info(f"Discussion Round {round_num}: Agents refining with peer context")

            # Only agents who contributed in round 1 participate
            participating = [a for a in agents if a.id in all_responses]

            round_responses = self._run_round(
                task=task,
                agents=participating,
                round_num=round_num,
                previous_responses=all_responses,
            )
            all_responses.update(round_responses)

            discussion_log.append({
                "round": round_num,
                "type": "collaborative",
                "responses": {
                    aid: resp[:200] + "..." if len(resp) > 200 else resp
                    for aid, resp in round_responses.items()
                },
            })

        # ── Synthesis ──
        logger.info("Synthesizing final answer...")
        final_answer = self._synthesize(task, all_responses, agents)

        # ── Score Contributions ──
        contributions = self._score_contributions(all_responses, final_answer)

        return {
            "final_answer": final_answer,
            "individual_responses": all_responses,
            "discussion_log": discussion_log,
            "contributions": contributions,
        }

    def _run_round(
        self,
        task: str,
        agents: list[Agent],
        round_num: int,
        previous_responses: dict[str, str] | None,
    ) -> dict[str, str]:
        """Run a single discussion round — each agent responds."""
        responses: dict[str, str] = {}

        for agent in agents:
            try:
                response = self._get_agent_response(
                    agent=agent,
                    task=task,
                    round_num=round_num,
                    previous_responses=previous_responses,
                )
                responses[agent.id] = response
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                responses[agent.id] = f"[Agent {agent.name} encountered an error]"

        return responses

    def _get_agent_response(
        self,
        agent: Agent,
        task: str,
        round_num: int,
        previous_responses: dict[str, str] | None,
    ) -> str:
        """Get a single agent's response for a discussion round."""
        # Build the agent's personalized context
        system_prompt = agent.build_context_prompt(task)

        # Build the user message
        if round_num == 1 or previous_responses is None:
            user_message = (
                f"## Task\n{task}\n\n"
                f"## Instructions\n"
                f"Provide your analysis and reasoning. Be thorough but concise. "
                f"Use your {agent.reasoning_strategy} reasoning strategy."
            )
        else:
            # Include other agents' responses for collaborative rounds
            peer_summaries = []
            for aid, resp in previous_responses.items():
                if aid != agent.id:
                    # Truncate long responses
                    short = resp[:300] + "..." if len(resp) > 300 else resp
                    peer_summaries.append(f"- Another agent said: {short}")

            peers_text = "\n".join(peer_summaries[:5])  # limit to 5 peers

            user_message = (
                f"## Task\n{task}\n\n"
                f"## What Other Agents Said\n{peers_text}\n\n"
                f"## Instructions\n"
                f"You've seen other agents' perspectives. Now refine your answer. "
                f"Agree with strong points, challenge weak ones, and provide your "
                f"updated reasoning. Use your {agent.reasoning_strategy} strategy."
            )

        response = self.llm.chat(system_prompt, user_message)
        return response.content

    def _synthesize(
        self,
        task: str,
        all_responses: dict[str, str],
        agents: list[Agent],
    ) -> str:
        """
        Synthesize all agent responses into a single final answer.

        Looks for a synthesizer agent first; falls back to LLM synthesis.
        """
        # Build a summary of all responses
        response_summaries = []
        agent_map = {a.id: a for a in agents}

        for agent_id, response in all_responses.items():
            agent = agent_map.get(agent_id)
            name = agent.name if agent else agent_id
            role = agent.role if agent else "unknown"
            response_summaries.append(
                f"### {name} ({role})\n{response}"
            )

        all_text = "\n\n".join(response_summaries)

        system_prompt = (
            "You are the final synthesizer for an AI society. Multiple agents "
            "have independently reasoned about a task. Your job is to:\n"
            "1. Identify the strongest arguments and evidence\n"
            "2. Resolve any disagreements by weighing the reasoning quality\n"
            "3. Produce a single, clear, well-reasoned final answer\n"
            "4. Be concise but complete"
        )

        user_message = (
            f"## Original Task\n{task}\n\n"
            f"## Agent Responses\n{all_text}\n\n"
            f"## Your Job\n"
            f"Synthesize these perspectives into one final answer. "
            f"State the answer clearly."
        )

        response = self.llm.chat(system_prompt, user_message)
        return response.content

    def _score_contributions(
        self,
        responses: dict[str, str],
        final_answer: str,
    ) -> dict[str, float]:
        """
        Score how much each agent contributed to the final answer.

        Simple heuristic: longer, more substantive responses that align
        with the final answer get higher scores.
        """
        contributions: dict[str, float] = {}
        final_words = set(final_answer.lower().split())

        for agent_id, response in responses.items():
            resp_words = set(response.lower().split())

            # Overlap with final answer (crude but functional)
            if final_words:
                overlap = len(resp_words & final_words) / len(final_words)
            else:
                overlap = 0.0

            # Length bonus (substantive responses)
            length_score = min(len(response) / 500, 1.0)

            contributions[agent_id] = round(
                0.6 * overlap + 0.4 * length_score, 3
            )

        return contributions
