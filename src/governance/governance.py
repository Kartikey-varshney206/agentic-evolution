"""
Governance — Democratic self-governance for the AI Society.

Instead of a human deciding when the society should change, the agents
themselves propose and vote on structural changes. This implements
Part 8 of the AI Civilization.

Governance loop:
1. Any agent can submit a Proposal
2. All agents vote (weighted by reputation)
3. If quorum met → run experiment
4. If experiment succeeds → adopt permanently
5. If not → rollback

Constitutional rules exist that can NEVER be voted away (safety rails).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..agents.agent import Agent
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class ProposalType(str, Enum):
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    CHANGE_WORKFLOW = "change_workflow"
    CHANGE_RULE = "change_rule"
    CHANGE_GOVERNANCE = "change_governance"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    VOTING = "voting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TESTING = "testing"
    ADOPTED = "adopted"
    ROLLED_BACK = "rolled_back"


# Quorum thresholds for different proposal types
DEFAULT_QUORUMS = {
    ProposalType.ADD_AGENT: 0.50,
    ProposalType.REMOVE_AGENT: 0.60,
    ProposalType.CHANGE_WORKFLOW: 0.60,
    ProposalType.CHANGE_RULE: 0.75,
    ProposalType.CHANGE_GOVERNANCE: 0.80,
}

# Constitutional rules — CANNOT be voted away
CONSTITUTION = [
    "All answers must be verified before delivery.",
    "The society must maintain at least 4 distinct agent roles.",
    "No single agent type may exceed 50% of the population.",
    "Agent death requires minimum population of 10.",
    "All structural changes must be A/B tested before adoption.",
    "A human veto override must always remain available.",
    "Governance change proposals require 80% approval.",
]


@dataclass
class Vote:
    """A single agent's vote on a proposal."""
    agent_id: str
    agent_name: str
    decision: str  # "yes" or "no"
    reasoning: str  # why they voted this way
    weight: float  # reputation-based weight


@dataclass
class Proposal:
    """A proposal for the society to vote on."""

    id: str
    proposer_id: str
    proposer_name: str
    proposal_type: ProposalType
    title: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    votes: list[Vote] = field(default_factory=list)
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: str | None = None
    experiment_results: dict[str, Any] | None = None

    @property
    def votes_yes(self) -> float:
        return sum(v.weight for v in self.votes if v.decision == "yes")

    @property
    def votes_no(self) -> float:
        return sum(v.weight for v in self.votes if v.decision == "no")

    @property
    def total_weight(self) -> float:
        return sum(v.weight for v in self.votes)

    @property
    def approval_rate(self) -> float:
        if self.total_weight == 0:
            return 0.0
        return self.votes_yes / self.total_weight

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposer": self.proposer_name,
            "type": self.proposal_type.value,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "votes_yes": round(self.votes_yes, 2),
            "votes_no": round(self.votes_no, 2),
            "approval_rate": f"{self.approval_rate:.0%}",
            "created_at": self.created_at,
        }


class GovernanceEngine:
    """
    Manages the democratic process of the AI society.

    Agents propose changes → society votes → changes are tested → adopted or rolled back.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        quorums: dict[ProposalType, float] | None = None,
    ):
        self.llm = llm_client
        self.quorums = quorums or DEFAULT_QUORUMS
        self.proposals: list[Proposal] = []
        self.constitution = CONSTITUTION.copy()

    def create_proposal(
        self,
        proposer: Agent,
        proposal_type: ProposalType,
        title: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> Proposal:
        """Create a new proposal from an agent."""
        # Check constitutional violations
        violation = self._check_constitution(proposal_type, description)
        if violation:
            logger.warning(f"Proposal rejected — constitutional violation: {violation}")
            proposal = Proposal(
                id=f"prop-{len(self.proposals):03d}",
                proposer_id=proposer.id,
                proposer_name=proposer.name,
                proposal_type=proposal_type,
                title=title,
                description=description,
                details=details or {},
                status=ProposalStatus.REJECTED,
            )
            self.proposals.append(proposal)
            return proposal

        proposal = Proposal(
            id=f"prop-{len(self.proposals):03d}",
            proposer_id=proposer.id,
            proposer_name=proposer.name,
            proposal_type=proposal_type,
            title=title,
            description=description,
            details=details or {},
            status=ProposalStatus.PENDING,
        )
        self.proposals.append(proposal)
        logger.info(f"📋 Proposal created: '{title}' by {proposer.name}")
        return proposal

    def run_vote(self, proposal: Proposal, agents: list[Agent]) -> Proposal:
        """
        Run a vote on a proposal across all agents.

        Each agent reasons about the proposal and votes yes/no.
        Votes are weighted by reputation.
        """
        proposal.status = ProposalStatus.VOTING
        logger.info(f"🗳️  Voting on: '{proposal.title}' ({len(agents)} voters)")

        for agent in agents:
            # Skip the proposer (they implicitly vote yes)
            if agent.id == proposal.proposer_id:
                proposal.votes.append(Vote(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    decision="yes",
                    reasoning="I proposed this.",
                    weight=agent.reputation_score,
                ))
                continue

            vote = self._get_agent_vote(agent, proposal)
            proposal.votes.append(vote)

        # Check quorum
        quorum = self.quorums.get(proposal.proposal_type, 0.60)

        if proposal.approval_rate >= quorum:
            proposal.status = ProposalStatus.ACCEPTED
            logger.info(
                f"  ✅ ACCEPTED: '{proposal.title}' "
                f"({proposal.approval_rate:.0%} approval, needed {quorum:.0%})"
            )
        else:
            proposal.status = ProposalStatus.REJECTED
            proposal.resolved_at = datetime.now().isoformat()
            logger.info(
                f"  ❌ REJECTED: '{proposal.title}' "
                f"({proposal.approval_rate:.0%} approval, needed {quorum:.0%})"
            )

        return proposal

    def _get_agent_vote(self, agent: Agent, proposal: Proposal) -> Vote:
        """Get a single agent's vote on a proposal."""
        if self.llm:
            return self._llm_vote(agent, proposal)
        else:
            return self._simple_vote(agent, proposal)

    def _llm_vote(self, agent: Agent, proposal: Proposal) -> Vote:
        """Use LLM to have the agent reason about and vote on a proposal."""
        system_prompt = agent.build_context_prompt("governance vote")
        user_message = (
            f"## Governance Vote\n\n"
            f"A proposal has been made to the society:\n\n"
            f"**{proposal.title}**\n"
            f"Type: {proposal.proposal_type.value}\n"
            f"Description: {proposal.description}\n\n"
            f"As {agent.name} ({agent.role}), evaluate this proposal.\n"
            f"Consider: Will this improve the society? Are there risks?\n\n"
            f"Respond with:\n"
            f"VOTE: YES or NO\n"
            f"REASON: (brief explanation)"
        )

        try:
            response = self.llm.chat(system_prompt, user_message, temperature=0.3)
            content = response.content.upper()

            decision = "yes" if "YES" in content.split("VOTE:")[-1][:20] else "no"
            reasoning = response.content.split("REASON:")[-1].strip() if "REASON:" in response.content else response.content

            return Vote(
                agent_id=agent.id,
                agent_name=agent.name,
                decision=decision,
                reasoning=reasoning[:200],
                weight=agent.reputation_score,
            )
        except Exception as e:
            logger.error(f"LLM vote failed for {agent.name}: {e}")
            return self._simple_vote(agent, proposal)

    def _simple_vote(self, agent: Agent, proposal: Proposal) -> Vote:
        """Simple heuristic voting (no LLM)."""
        # Higher fitness agents are more likely to vote yes
        # (they benefit from a well-functioning society)
        yes_probability = 0.5 + (agent.compute_fitness() - 0.5) * 0.3
        decision = "yes" if random.random() < yes_probability else "no"

        return Vote(
            agent_id=agent.id,
            agent_name=agent.name,
            decision=decision,
            reasoning="Heuristic vote based on agent fitness.",
            weight=agent.reputation_score,
        )

    def _check_constitution(self, proposal_type: ProposalType, description: str) -> str | None:
        """Check if a proposal violates constitutional rules."""
        desc_lower = description.lower()

        # Cannot remove all verification
        if "remove verification" in desc_lower or "skip verification" in desc_lower:
            return CONSTITUTION[0]

        # Cannot remove human veto
        if "remove human" in desc_lower or "disable veto" in desc_lower:
            return CONSTITUTION[5]

        return None

    def auto_propose(self, agents: list[Agent], society_stats: dict) -> Proposal | None:
        """
        Have the society auto-generate a proposal based on current performance.

        Uses an LLM to analyze society stats and propose improvements.
        """
        if not self.llm or not agents:
            return None

        # Pick a high-reputation agent to be the proposer
        proposer = max(agents, key=lambda a: a.reputation_score)

        system_prompt = (
            "You are analyzing an AI society's performance to propose improvements. "
            "Based on the statistics provided, suggest ONE concrete proposal. "
            "Respond with:\n"
            "TYPE: add_agent | remove_agent | change_workflow | change_rule\n"
            "TITLE: (short title)\n"
            "DESCRIPTION: (what to change and why)"
        )

        user_message = f"Society statistics:\n{society_stats}"

        try:
            response = self.llm.chat(system_prompt, user_message, temperature=0.7)
            content = response.content

            # Parse the response
            type_str = "change_rule"
            title = "Auto-proposed improvement"
            description = content

            for line in content.split("\n"):
                if line.strip().upper().startswith("TYPE:"):
                    type_str = line.split(":", 1)[1].strip().lower()
                elif line.strip().upper().startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip()
                elif line.strip().upper().startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()

            ptype = ProposalType(type_str) if type_str in [p.value for p in ProposalType] else ProposalType.CHANGE_RULE

            return self.create_proposal(
                proposer=proposer,
                proposal_type=ptype,
                title=title,
                description=description,
            )
        except Exception as e:
            logger.error(f"Auto-proposal failed: {e}")
            return None

    def get_history(self) -> list[dict]:
        """Get all proposals and their outcomes."""
        return [p.to_dict() for p in self.proposals]
