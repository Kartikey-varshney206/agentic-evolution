"""
Workflow Engine — Executes dynamic DAG-based workflows.

Replaces the hardcoded DiscussionEngine with a flexible topological executor
that reads WorkflowPipeline schemas and routes data between agent stages.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..agents.agent import Agent
from ..workflow.pipeline import WorkflowPipeline, WorkflowStage
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Executes a dynamic workflow pipeline on a given task."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _topological_sort(self, stages: list[WorkflowStage]) -> list[WorkflowStage]:
        """Sort stages so dependencies execute before dependents."""
        # This is a simple O(N^2) sort since N is small (< 10)
        sorted_stages = []
        visited = set(["input"])
        pending = list(stages)
        
        while pending:
            progress = False
            for stage in list(pending):
                # Check if all inputs are satisfied
                if all(dep in visited for dep in stage.input_from):
                    sorted_stages.append(stage)
                    visited.add(stage.id)
                    pending.remove(stage)
                    progress = True
            
            if not progress:
                # Cycle detected or missing dependency
                logger.error(f"Workflow cycle or missing dependency detected: unresolved={pending}")
                # Append the rest to avoid freezing
                sorted_stages.extend(pending)
                break
                
        return sorted_stages

    def execute(
        self,
        task: str,
        agents: list[Agent],
        pipeline: WorkflowPipeline,
    ) -> dict[str, Any]:
        """
        Execute the pipeline on the task using the provided agents.
        
        Returns:
            Dict containing:
            - final_answer: The synthesized answer
            - individual_responses: {agent_id: response} (combined from all stages)
            - discussion_log: list of stage executions
            - stage_outputs: {stage_id: combined_output}
        """
        logger.info(f"Executing workflow pipeline: {pipeline.name} (gen {pipeline.generation})")
        
        stages = self._topological_sort(pipeline.stages)
        
        # Track outputs for each node in the DAG
        stage_outputs = {"input": task}
        discussion_log = []
        individual_responses = defaultdict(list)
        
        # Process each stage
        for stage in stages:
            logger.info(f"  -> Running stage: {stage.id} (role={stage.role})")
            
            # 1. Resolve Inputs
            input_texts = []
            for dep in stage.input_from:
                if dep in stage_outputs:
                    input_texts.append(f"--- Input from {dep} ---\n{stage_outputs[dep]}")
                else:
                    input_texts.append(f"--- Input from {dep} ---\n[No output available]")
                    
            combined_input = "\n\n".join(input_texts)
            
            # Construct the prompt for the agents in this stage
            prompt = f"TASK:\n{combined_input}\n\nINSTRUCTION FOR YOU:\n{stage.action}"
            
            # 2. Select Agents
            # Find agents matching the role (or 'any' if role is empty/wildcard)
            eligible_agents = []
            allowed_roles = [r.strip().lower() for r in stage.role.split(",")] if stage.role else []
            for a in agents:
                if not allowed_roles or "any" in allowed_roles or a.role.lower() in allowed_roles:
                    eligible_agents.append(a)
                    
            # If no exact match, fallback to any available agent
            if not eligible_agents:
                logger.warning(f"No agents found for role '{stage.role}' in stage {stage.id}. Using all available.")
                eligible_agents = list(agents)
                
            # Sort by fitness descending to pick the best
            eligible_agents.sort(key=lambda x: x.compute_fitness(), reverse=True)
            selected_agents = eligible_agents[:stage.max_agents]
            
            # 3. Execute Agents
            stage_responses = []
            for agent in selected_agents:
                try:
                    system_prompt = agent.build_context_prompt(task)
                    llm_response = self.llm.chat(system_prompt, prompt)
                    response = llm_response.content
                    
                    stage_responses.append(f"[{agent.name}]: {response}")
                    individual_responses[agent.id].append(response)
                    
                    # Log execution
                    discussion_log.append({
                        "stage": stage.id,
                        "agent": agent.name,
                        "role": agent.role,
                        "response": response
                    })
                except Exception as e:
                    logger.error(f"Agent {agent.name} failed in stage {stage.id}: {e}")
                    
            # 4. Store Stage Output
            if stage_responses:
                stage_outputs[stage.id] = "\n\n".join(stage_responses)
            else:
                stage_outputs[stage.id] = "No response generated."
                
        # The final answer is the output of the last stage
        final_stage_id = stages[-1].id if stages else "input"
        final_answer = stage_outputs.get(final_stage_id, "")
        
        # Format individual responses as strings
        formatted_responses = {k: "\n\n".join(v) for k, v in individual_responses.items()}
        
        # Score contributions
        contributions = self._score_contributions(formatted_responses, final_answer)
        
        return {
            "final_answer": final_answer,
            "individual_responses": formatted_responses,
            "discussion_log": discussion_log,
            "stage_outputs": stage_outputs,
            "contributions": contributions,
        }

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
