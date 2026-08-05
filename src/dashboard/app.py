"""
AI Civilization — Streamlit Dashboard

Run with:
    streamlit run src/dashboard/app.py
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to path to allow imports when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from src.dashboard.visualizations import plot_fitness_history, plot_role_distribution, create_workflow_diagram


def load_state() -> dict | None:
    """Load civilization state from JSON file."""
    path = Path("civilization_state.json")
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load state: {e}")
        return None


def main():
    st.set_page_config(
        page_title="AI Civilization Control Room",
        page_icon="🌍",
        layout="wide",
    )

    st.title("🌍 AI Civilization Control Room")

    state = load_state()
    if not state:
        st.warning("No civilization state found. Run the simulation first: `python -m src`")
        return

    # ── Sidebar ──
    st.sidebar.header("Status")
    st.sidebar.metric("Generation", state.get("generation", 0))
    st.sidebar.metric("Total Tasks", state.get("total_tasks", 0))
    
    society_stats = state.get("society", {})
    st.sidebar.metric("Population", society_stats.get("population", 0))
    st.sidebar.metric("Archived (Dead)", society_stats.get("archived", 0))
    st.sidebar.metric("Avg Accuracy", f"{society_stats.get('avg_accuracy', 0):.1%}")

    llm = state.get("llm_usage", {})
    st.sidebar.write("---")
    st.sidebar.caption("LLM Engine")
    st.sidebar.text(f"Provider: {llm.get('provider')}")
    st.sidebar.text(f"Model: {llm.get('model')}")
    st.sidebar.text(f"API Calls: {llm.get('total_calls')}")
    
    # ── Main Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Society", "🧬 Evolution", "⚙️ Workflow", "🗳️ Governance"])

    # ── TAB 1: Society Overview ──
    with tab1:
        st.header("Agent Population")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            role_fig = plot_role_distribution(society_stats)
            if role_fig:
                st.plotly_chart(role_fig, use_container_width=True)
                
        with col1:
            agents = state.get("agents", {})
            if agents:
                # Convert to dataframe for nice display
                df_agents = pd.DataFrame.from_dict(agents, orient="index")
                # Filter to alive agents
                df_alive = df_agents[df_agents["alive"] == True].copy()
                
                if not df_alive.empty:
                    df_disp = df_alive[["name", "role", "generation", "accuracy", "fitness", "age"]]
                    df_disp = df_disp.sort_values(by="fitness", ascending=False)
                    st.dataframe(
                        df_disp.style.background_gradient(cmap="Greens", subset=["fitness"]),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("No agents are currently alive.")

    # ── TAB 2: Evolution ──
    with tab2:
        st.header("Evolutionary Timeline")
        
        # We need evolution history, but it might not be in the basic state dict
        # Assuming we can mock it or it might be saved if we updated main.py
        # For this prototype, we'll check if it exists in the file, though main.py currently 
        # doesn't dump the full evolution history list. 
        # Note: To fully populate this, we should ensure `main.py` saves `civ.evolution.get_evolution_summary()`
        
        # Workaround: Show basic message if history is missing
        if "evolution_history_data" in state: # Assuming we add this later
            evo_history = state.get("evolution_history_data", [])
            fig = plot_fitness_history(evo_history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Evolution has run {state.get('evolution_history', 0)} times. (Detailed history tracking not fully serialized in this demo).")
            st.write("Agent lineage:")
            
            # Show simple genealogy of alive agents
            agents = state.get("agents", {})
            for aid, agent in agents.items():
                if agent.get("alive") and agent.get("parents"):
                    parent_names = [agents.get(p, {}).get("name", "Unknown") for p in agent["parents"]]
                    st.text(f"🌱 {agent['name']} (Gen {agent['generation']}) — Parents: {', '.join(parent_names)}")

    # ── TAB 3: Workflow ──
    with tab3:
        st.header("Processing Pipeline")
        workflow = state.get("workflow", {})
        
        st.subheader(f"Current Architecture: {workflow.get('name', 'Unknown')}")
        st.text(f"Generation: {workflow.get('generation', 0)}")
        
        wf_fig = create_workflow_diagram(workflow)
        if wf_fig:
            st.plotly_chart(wf_fig, use_container_width=True)
            
        st.write("### Stages Details")
        stages = workflow.get("stages", [])
        if stages:
            df_stages = pd.DataFrame(stages)
            st.dataframe(df_stages, use_container_width=True)

    # ── TAB 4: Governance ──
    with tab4:
        st.header("Democratic Governance")
        st.write(f"Total proposals: {state.get('governance_proposals', 0)}")
        
        history = state.get("governance_history", [])
        if history:
            for prop in reversed(history):
                status_color = "green" if prop["status"] == "accepted" else "red" if prop["status"] == "rejected" else "orange"
                st.markdown(f"### :{status_color}[{prop['status'].upper()}] {prop['title']}")
                st.caption(f"Proposed by: {prop['proposer']} | Type: `{prop['type']}` | Approval: {prop['approval_rate']}")
                st.write(prop['description'])
                st.divider()
        else:
            st.info("No governance proposals have been made yet.")


if __name__ == "__main__":
    main()
