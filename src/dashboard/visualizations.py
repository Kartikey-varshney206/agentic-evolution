"""
Visualizations for the AI Civilization dashboard.
Generates charts for evolution, genealogy, and workflow state.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_fitness_history(evolution_history: list[dict]) -> go.Figure | None:
    """Plot fitness and accuracy over generations."""
    if not evolution_history:
        return None

    data = []
    for record in evolution_history:
        stats = record.get("stats", {})
        fitness = stats.get("fitness", {})
        acc = stats.get("accuracy", {})

        data.append({
            "Generation": record["generation"],
            "Mean Fitness": fitness.get("mean", 0),
            "Max Fitness": fitness.get("max", 0),
            "Mean Accuracy": acc.get("mean", 0),
        })

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Generation"], y=df["Max Fitness"], mode="lines+markers", name="Max Fitness", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df["Generation"], y=df["Mean Fitness"], mode="lines", name="Mean Fitness", line=dict(color="blue", dash="dash")))
    fig.add_trace(go.Scatter(x=df["Generation"], y=df["Mean Accuracy"], mode="lines", name="Mean Accuracy", yaxis="y2", line=dict(color="purple")))

    fig.update_layout(
        title="Evolutionary Progress",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_role_distribution(society_stats: dict) -> go.Figure | None:
    """Plot the current distribution of agent roles."""
    roles = society_stats.get("role_distribution", {})
    if not roles:
        return None

    df = pd.DataFrame(list(roles.items()), columns=["Role", "Count"])
    fig = px.pie(df, names="Role", values="Count", hole=0.4, title="Agent Role Distribution")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


def create_workflow_diagram(workflow: dict) -> go.Figure | None:
    """Create a simple sankey/flow diagram of the workflow pipeline."""
    stages = workflow.get("stages", [])
    if not stages:
        return None

    nodes = []
    links = []

    # Map stage IDs to indices
    stage_to_idx = {"input": 0}
    nodes.append(dict(label="Input", color="gray"))

    for idx, stage in enumerate(stages):
        real_idx = idx + 1
        stage_to_idx[stage["id"]] = real_idx
        # Include role info in the label if available
        role = stage.get("role", "unknown")
        nodes.append(dict(
            label=f"{stage['id']}<br><i>({role})</i>",
            color="blue"
        ))

    # Add terminal node for final answer
    out_idx = len(nodes)
    nodes.append(dict(label="Output", color="green"))

    # Create links
    for idx, stage in enumerate(stages):
        real_idx = idx + 1
        inputs = stage.get("input_from", ["input"])

        for inp in inputs:
            source_idx = stage_to_idx.get(inp)
            if source_idx is not None:
                links.append(dict(
                    source=source_idx,
                    target=real_idx,
                    value=1
                ))

        # If it's the last stage (synthesize typically), link to output
        if idx == len(stages) - 1:
            links.append(dict(
                source=real_idx,
                target=out_idx,
                value=1
            ))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes]
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links]
        )
    )])

    fig.update_layout(title_text="Workflow Pipeline", font_size=12)
    return fig
