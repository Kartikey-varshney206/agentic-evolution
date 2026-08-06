import asyncio
from src.llm.client import LLMClient
from src.society.workflow_engine import WorkflowEngine
from src.workflow.pipeline import WorkflowPipeline
from src.agents.factory import load_agents_from_config

def main():
    llm = LLMClient()
    agents = load_agents_from_config("config/default_agents.yaml")
    
    workflow = WorkflowPipeline.from_yaml("config/default_workflow.yaml")
    print(f"Loaded workflow: {workflow.name}")
    print(f"Stages: {[s.id for s in workflow.stages]}")
    
    engine = WorkflowEngine(llm)
    
    task = "Solve this logic puzzle: If a farmer has 17 sheep and all but 9 die, how many are left?"
    
    result = engine.execute(
        task=task,
        agents=agents,
        pipeline=workflow
    )
    
    print("\nFINAL ANSWER:")
    print(result["final_answer"])
    
if __name__ == "__main__":
    main()
