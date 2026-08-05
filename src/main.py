"""
AI Civilization — Main Entry Point

This is the top-level orchestrator that ties everything together:
- Loads the initial society
- Processes tasks through multi-agent discussion
- Runs evolution cycles automatically
- Triggers governance votes when appropriate
- Tracks everything for the dashboard

Usage:
    python -m src.main
"""

from __future__ import annotations

import json
import logging
import sys
import argparse
from pathlib import Path

from .agents.agent import Agent
from .llm.client import LLMClient
from .society.society import Society
from .society.environment import Environment
from .evolution.engine import EvolutionEngine, EvolutionConfig, GenerationRecord
from .governance.governance import GovernanceEngine, ProposalType, ProposalStatus, Proposal
from .workflow.pipeline import WorkflowPipeline, WorkflowEvolver

# ── Logging Setup ──
# Force UTF-8 encoding on Windows to prevent cp1252 charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("civilization.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class Civilization:
    """
    The AI Civilization — the top-level orchestrator.

    Manages the complete lifecycle:
    Task Processing → Evaluation → Evolution → Governance → Repeat
    """

    def __init__(
        self,
        agents_config: str = "config/default_agents.yaml",
        evolution_config: str = "config/evolution_params.yaml",
        workflow_config: str = "config/default_workflow.yaml",
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        # Initialize LLM
        self.llm = LLMClient(provider=llm_provider, model=llm_model)

        # Initialize Society
        self.society = Society(llm_client=self.llm, agents_config=agents_config)

        # Initialize Evolution Engine
        evo_config = EvolutionConfig.from_yaml(evolution_config)
        self.evolution = EvolutionEngine(config=evo_config, llm_client=self.llm)

        # Initialize Governance
        self.governance = GovernanceEngine(llm_client=self.llm)

        # Initialize Workflow
        try:
            self.workflow = WorkflowPipeline.from_yaml(workflow_config)
        except FileNotFoundError:
            self.workflow = WorkflowPipeline(name="default", stages=[])

        self.workflow_evolver = WorkflowEvolver(llm_client=self.llm)

        # Track state
        self.generation = 0
        self.total_tasks = 0

        logger.info(
            f"[START] AI Civilization initialized!\n"
            f"   Agents: {len(self.society.agents)}\n"
            f"   Workflow: {self.workflow.name}\n"
            f"   LLM: {self.llm.provider}/{self.llm.model}\n"
        )

    def solve(self, task: str, ground_truth: str | None = None) -> dict:
        """
        Process a single task through the civilization.

        1. Routes task to society for multi-agent discussion
        2. Checks if evolution should trigger
        3. Returns the result
        """
        # ── Solve via society ──
        result = self.society.solve(task, ground_truth)
        self.total_tasks += 1

        status = ""
        if result.correct is not None:
            status = "[CORRECT]" if result.correct else "[WRONG]"

        logger.info(
            f"\nTask {result.task_id} {status}\n"
            f"Question: {task}\n"
            f"Final Answer: {result.final_answer}\n"
        )

        # ── Check if evolution should trigger ──
        if self.evolution.should_evolve(self.total_tasks):
            self._run_evolution()

        return {
            "task_id": result.task_id,
            "answer": result.final_answer,
            "correct": result.correct,
            "agents_involved": len(result.agent_responses),
            "generation": self.generation,
        }

    def solve_batch(
        self,
        tasks: list[dict[str, str]],
    ) -> list[dict]:
        """
        Process a batch of tasks.

        Each task should be a dict with 'question' and optionally 'answer'.
        """
        results = []
        for i, task in enumerate(tasks):
            question = task.get("question", task.get("task", ""))
            answer = task.get("answer", task.get("ground_truth", None))

            logger.info(f"--- Task {i+1}/{len(tasks)} ---")
            result = self.solve(question, answer)
            results.append(result)
            
            # Save state after each task so the Streamlit dashboard updates live!
            self.save()

        return results

    def _run_evolution(self) -> None:
        """Run an evolution cycle."""
        self.generation += 1
        logger.info(f"\n[EVOLUTION] Triggering evolution cycle (generation {self.generation})...")

        self.society.agents, record = self.evolution.run_cycle(
            agents=self.society.agents,
            generation=self.generation,
        )
        self.society.generation = self.generation

    def run_governance_cycle(self) -> None:
        """Run a governance cycle — propose and vote on changes."""
        stats = self.society.get_stats()
        alive_agents = self.society.get_alive_agents()

        if not alive_agents:
            return

        # Auto-propose a change based on stats
        proposal = self.governance.auto_propose(alive_agents, stats)
        if proposal and proposal.status != ProposalStatus.REJECTED:
            # Run the vote
            self.governance.run_vote(proposal, alive_agents)
            
            # Enact if accepted
            if proposal.status == ProposalStatus.ACCEPTED:
                self._enact_proposal(proposal)

    def _enact_proposal(self, proposal: Proposal) -> None:
        """Physically apply the accepted proposal to the civilization."""
        logger.info(f"\n⚡ ENACTING PROPOSAL: {proposal.proposal_type.value}")
        
        try:
            if proposal.proposal_type == ProposalType.ADD_AGENT:
                best_agent = max(self.society.get_alive_agents(), key=lambda a: a.compute_fitness())
                new_agent = Agent(
                    name=f"{best_agent.name}_clone",
                    role=best_agent.role,
                    generation=self.generation,
                    parents=[best_agent.id],
                    system_prompt=best_agent.system_prompt,
                    reasoning_strategy=best_agent.reasoning_strategy,
                    tool_policy=best_agent.tool_policy,
                    memory_strategy=best_agent.memory_strategy,
                )
                self.society.agents[new_agent.id] = new_agent
                logger.info(f"  -> Added new agent: {new_agent.name} ({new_agent.role})")
                
            elif proposal.proposal_type == ProposalType.REMOVE_AGENT:
                alive = self.society.get_alive_agents()
                if len(alive) > 10: # Constitution minimum
                    worst_agent = min(alive, key=lambda a: a.compute_fitness())
                    worst_agent.alive = False
                    worst_agent.archived_at = "governance_removed"
                    self.society.archived_agents[worst_agent.id] = worst_agent
                    del self.society.agents[worst_agent.id]
                    logger.info(f"  -> Removed agent: {worst_agent.name}")
                else:
                    logger.warning("  -> Cannot remove agent: population at constitution minimum (10).")
                    
            elif proposal.proposal_type == ProposalType.CHANGE_WORKFLOW:
                new_workflow = self.workflow_evolver.propose_mutation(self.workflow, self.society.get_stats())
                self.workflow = new_workflow
                logger.info(f"  -> Workflow changed to: {self.workflow.name}")
                
            elif proposal.proposal_type in (ProposalType.CHANGE_RULE, ProposalType.CHANGE_GOVERNANCE):
                new_rule = f"RULE: {proposal.title}"
                self.governance.constitution.append(new_rule)
                logger.info(f"  -> Added new constitution rule: {new_rule}")
                
            proposal.status = ProposalStatus.ADOPTED
            
        except Exception as e:
            logger.error(f"  -> Failed to enact proposal: {e}")
            proposal.status = ProposalStatus.ROLLED_BACK

    def get_status(self) -> dict:
        """Get comprehensive civilization status."""
        stats = self.society.get_stats()
        return {
            "generation": self.generation,
            "total_tasks": self.total_tasks,
            "society": stats,
            "evolution_history": len(self.evolution.history),
            "governance_proposals": len(self.governance.proposals),
            "workflow": self.workflow.to_dict(),
            "llm_usage": self.llm.get_usage_stats(),
        }

    def save(self, path: str = "civilization_state.json") -> None:
        """Save the full civilization state."""
        state = self.get_status()
        state["agents"] = {
            aid: a.to_dict() for aid, a in self.society.agents.items()
        }
        state["archived_agents"] = {
            aid: a.to_dict() for aid, a in self.society.archived_agents.items()
        }
        state["governance_history"] = self.governance.get_history()
        state["evolution_history_data"] = self.evolution.get_evolution_summary()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Civilization saved to {path}")

    def load(self, path: str = "civilization_state.json") -> None:
        """Load the civilization state from file."""
        if not Path(path).exists():
            return
            
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.generation = state.get("generation", 0)
        self.total_tasks = state.get("total_tasks", 0)
        
        # Load agents
        self.society.agents.clear()
        for aid, adata in state.get("agents", {}).items():
            self.society.agents[aid] = Agent.from_dict(adata)
            
        # Load archived agents
        self.society.archived_agents.clear()
        for aid, adata in state.get("archived_agents", {}).items():
            self.society.archived_agents[aid] = Agent.from_dict(adata)
            
        # Update society stats
        self.society.generation = self.generation
        self.society.tasks_processed = self.total_tasks
        
        # Load evolution history
        self.evolution.history.clear()
        for record_data in state.get("evolution_history_data", []):
            self.evolution.history.append(GenerationRecord.from_dict(record_data))
            
        logger.info(f"Civilization state loaded from {path} (Gen {self.generation})")


# ── Sample Tasks for Demo ──

DEMO_TASKS = [
    {
        "question": "What is 15% of 240?",
        "answer": "36",
    },
    {
        "question": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "answer": "150 miles",
    },
    {
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
    },
    {
        "question": "Explain why the sky appears blue in 2-3 sentences.",
        "answer": "The sky appears blue because of Rayleigh scattering. Shorter blue wavelengths of sunlight are scattered more than longer red wavelengths by molecules in the atmosphere.",
    },
    {
        "question": "A rectangle has a perimeter of 30 cm and a width of 5 cm. What is its area?",
        "answer": "50 square cm",
    },
    {
        "question": "What are the three states of matter?",
        "answer": "Solid, liquid, and gas",
    },
    {
        "question": "If you have 3 apples and give away 1/3 of them, how many do you have left?",
        "answer": "2",
    },
    {
        "question": "What causes tides on Earth?",
        "answer": "Tides are primarily caused by the gravitational pull of the Moon on Earth's oceans.",
    },
    {
        "question": "What is the sum of the first 10 positive integers?",
        "answer": "55",
    },
    {
        "question": "Why do objects fall when dropped?",
        "answer": "Objects fall due to gravity — the gravitational attraction between the object and Earth.",
    },
]


def main():
    """Run a demo of the AI Civilization."""
    parser = argparse.ArgumentParser(description="AI Civilization Simulation")
    parser.add_argument("--infinite", action="store_true", help="Run the infinite dynamic environment mode")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AI CIVILIZATION -- Demo Run")
    print("=" * 60 + "\n")

    # Initialize
    civ = Civilization()

    # Show initial state
    status = civ.get_status()
    print(f"  Agents: {status['society']['population']}")
    print(f"  Roles: {status['society']['role_distribution']}")
    print(f"  LLM: {status['llm_usage']['provider']}/{status['llm_usage']['model']}")
    print()

    if args.infinite:
        print("─" * 60)
        print("  Running INFINITE dynamic tasks...")
        print("─" * 60 + "\n")
        
        # Load previous state if it exists
        civ.load()
        
        env = Environment(llm_client=civ.llm)
        tasks_solved = civ.total_tasks
        difficulty = min(10, 1 + (tasks_solved // 5))
        
        if tasks_solved > 0:
            logger.info(f"Resuming from Task {tasks_solved + 1} (Difficulty {difficulty})")
        
        import time
        while True:
            try:
                # Generate dynamic task
                task_data = env.generate_task(difficulty=difficulty)
                
                logger.info(f"--- Infinite Task {tasks_solved + 1} (Difficulty {difficulty}) ---")
                
                question = task_data.get("question", "")
                answer = task_data.get("answer", "")
                
                # Log the generated question to a file for overnight review
                with open("generated_tasks.txt", "a", encoding="utf-8") as f:
                    f.write(f"--- Task {tasks_solved + 1} (Difficulty {difficulty}) ---\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Ground Truth: {answer}\n")
                    f.write(f"{'='*50}\n\n")
                
                civ.solve(question, answer)
                civ.save()
                
                tasks_solved += 1
                
                # Bump difficulty every 5 tasks
                if tasks_solved % 5 == 0 and difficulty < 10:
                    difficulty += 1
                    logger.info(f"\n[ENVIRONMENT] Global difficulty raised to {difficulty}/10!\n")
                    
                # Trigger Governance Elections every 10 tasks
                if tasks_solved % 10 == 0:
                    logger.info(f"\n[GOVERNANCE] Initiating democratic election cycle...\n")
                    civ.run_governance_cycle()
                    civ.save()
                    
            except Exception as e:
                logger.error(f"Fatal error during loop (Rate Limit or API Crash): {e}. Sleeping 60s...")
                time.sleep(60)
                continue

    else:
        # Run standard demo tasks
        print("─" * 60)
        print("  Running demo tasks...")
        print("─" * 60 + "\n")

        results = civ.solve_batch(DEMO_TASKS)

        # Summary
        correct = sum(1 for r in results if r.get("correct"))
        total = sum(1 for r in results if r.get("correct") is not None)

        print("\n" + "=" * 60)
        print(f"  Results: {correct}/{total} correct ({correct/total:.0%})" if total else "  No evaluations")
        print(f"  Generation: {civ.generation}")
        print(f"  Total LLM calls: {civ.llm.total_calls}")
        print("=" * 60)

        # Save state
        civ.save()

    return civ


if __name__ == "__main__":
    main()
