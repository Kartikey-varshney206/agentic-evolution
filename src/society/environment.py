"""
Environment Engine — Dynamically generates tasks for the AI civilization.
"""

import json
import logging
from typing import Dict, Any

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class Environment:
    """
    Acts as the 'Game Master' or 'Nature' for the civilization.
    Generates novel, increasingly difficult tasks for the agents to solve.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_task(self, difficulty: int, domain: str = "general logic and math") -> Dict[str, str]:
        """
        Ask the LLM to generate a novel problem at the specified difficulty.
        
        Args:
            difficulty: Integer from 1 to 10 (1 = extremely easy, 10 = nearly impossible).
            domain: The type of problem to generate.
            
        Returns:
            Dict containing 'question' and 'answer' (ground truth).
        """
        logger.info(f"[ENVIRONMENT] Generating new {domain} task at difficulty Level {difficulty}/10...")
        
        system_prompt = (
            "You are the Environment Engine for an AI Civilization simulation. "
            "Your job is to generate a novel puzzle or problem for the AI society to solve.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You must output raw JSON only. No markdown formatting, no backticks, no explanations.\n"
            "2. The JSON must have exactly two keys: 'question' and 'answer'.\n"
            "3. The 'answer' must be a concise, objective ground truth (e.g., '42', 'Yes', 'The Moon') so it can be automatically graded.\n"
            "4. The problem must require step-by-step reasoning."
        )
        
        user_prompt = (
            f"Generate a {domain} problem at Difficulty Level {difficulty} out of 10.\n"
            f"- Level 1 means trivial arithmetic or basic facts.\n"
            f"- Level 10 means extremely complex logical paradoxes or advanced multivariable math.\n\n"
            f"Output the problem and its definitive answer in strict JSON format."
        )

        try:
            # We want a very low temperature so the JSON formatting is strict and deterministic
            response = self.llm.chat(system_prompt, user_prompt, temperature=0.2)
            
            # Clean up the output in case the LLM ignored instructions and wrapped in markdown
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            task_data = json.loads(content)
            
            if "question" not in task_data or "answer" not in task_data:
                raise ValueError("JSON missing required 'question' or 'answer' keys.")
                
            return task_data
            
        except Exception as e:
            logger.error(f"[ENVIRONMENT] Failed to generate task: {e}")
            # Fallback trivial task to prevent simulation crash
            return {
                "question": f"[Fallback] What is {difficulty} + {difficulty}?",
                "answer": str(difficulty + difficulty)
            }
