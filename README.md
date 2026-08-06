<div align="center">
  <h1>🧬 Agentic Evolution Framework</h1>
  <p><b>An autonomous multi-agent society that self-governs, mutates, and computationally evolves to solve dynamic environments.</b></p>

  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
  [![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-Nemotron_550B-76B900.svg)](https://build.nvidia.com/)
</div>

<br/>

Instead of a single LLM answering questions, this system creates an **entire society of AI agents** (scientists, engineers, creatives, skeptics) that collaborate, debate, and evolve to solve problems together.

The society isn't static. It **evolves computationally**:
- 🧬 **Mutation:** Agents randomly mutate their reasoning strategies.
- 🌱 **Reproduction:** New agents are born by combining traits of top performers.
- ☠️ **Natural Selection:** Bottom-performing agents are killed off.
- 🗳️ **Democratic Governance:** Agents propose and vote on structural workflow changes.
- 🔄 **Workflow Topology Evolution:** The society physically rewrites its own code execution DAG (Directed Acyclic Graph) to optimize for API token usage and accuracy, utilizing A/B testing to automatically roll back bad mutations.
- 🌍 **Dynamic Environment:** A "Game Master" engine procedurally generates increasingly difficult tasks for the society to adapt to.


---

## 🏗️ Architecture Flow

```mermaid
graph TD
    %% Environment Generation
    Env((🌍 Environment Engine)) -->|Procedurally generates novel tasks| TaskQueue[Task Queue]
    
    %% Task to Society
    TaskQueue -->|Feeds Problem| Soc[AI Society]
    
    subgraph "Society Loop"
        Soc --> Agent1(Scientist Agent)
        Soc --> Agent2(Skeptic Agent)
        Soc --> Agent3(Creative Agent)
        Soc --> AgentN(...N Agents)
        
        Agent1 <--> |Debate & Reasoning| Agent2
        Agent2 <--> |Peer Review| Agent3
        Agent3 <--> |Synthesis| AgentN
    end
    
    %% Outputs
    Soc -->|Final Synthesis| Answer[Final Answer]
    Answer --> Eval{Evaluation against Ground Truth}
    
    %% Evolutionary Pressure
    Eval -->|Success/Failure Metrics| Fitness[Fitness Scoring]
    Fitness -.->|Kill worst| Death(Agent Death)
    Fitness -.->|Mutate survivors| Mutate(Agent Mutation)
    Fitness -.->|Breed best| Birth(Agent Reproduction)
    
    %% Cycle restarts
    Death & Mutate & Birth -.-> Soc
    
    %% Dashboard
    Soc -.-> |Real-time Telemetry| Dash[Streamlit Control Room]
    
    classDef primary fill:#1e1e1e,stroke:#333,stroke-width:2px,color:#fff
    classDef secondary fill:#003366,stroke:#0055ff,stroke-width:2px,color:#fff
    classDef accent fill:#440066,stroke:#9900ff,stroke-width:2px,color:#fff
    classDef ui fill:#FF4B4B,stroke:#aa0000,stroke-width:2px,color:#fff
    
    class Env,Eval accent
    class Soc,Agent1,Agent2,Agent3,AgentN secondary
    class TaskQueue,Answer,Fitness,Death,Mutate,Birth primary
    class Dash ui
```

## 📊 Live Control Room (Streamlit)

The framework includes a real-time telemetry dashboard. While the society is running in the background, you can monitor the population distribution, workflow structures, and token usage via the UI.

![Control Room Interface](https://streamlit.io/images/brand/streamlit-mark-color.svg)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Kartikey-varshney206/agentic-evolution.git
cd agentic-evolution

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -e .
pip install streamlit==1.38.0 matplotlib
```

### 2. Configuration

```bash
# Copy the environment file template
cp .env.example .env
```
Edit the `.env` file to configure your LLM provider. The framework supports **NVIDIA NIM**, **Groq**, **Gemini**, and local **Ollama**.

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_key_here
```

### 3. Run the Simulation

You can run the simulation in two modes:

**Mode 1: The Deterministic Benchmark (Demo Tasks)**
Runs a hardcoded set of 10 tasks to baseline your LLM's raw performance.
```bash
python -m src
```

**Mode 2: The Infinite Evolutionary Environment ♾️**
The Environment Engine will procedurally generate novel tasks, bumping the global difficulty by `+1` every 5 tasks, forcing the society to evolve overnight.
```bash
python -m src --infinite
```

### 4. Launch the Dashboard
While the simulation is running, open a **new terminal window** and launch the Streamlit control room:
```bash
streamlit run src/dashboard/app.py --server.enableCORS false --server.enableXsrfProtection false
```

---

## ⚙️ Core Modules

| Module | Description |
|---|---|
| `agents/` | Defines agent memory, personality factories, and multi-provider LLM abstraction. |
| `society/` | Orchestrates the multi-agent debate, discussion rules, and peer review synthesis. |
| `evolution/` | Handles fitness scoring, natural selection, random genetic mutation, and crossover reproduction. |
| `governance/` | Allows agents to computationally vote on systemic changes to their own processing workflow. |
| `workflow/` | A dynamic DAG topology executor that runs tasks through self-modifying pipelines with automatic A/B testing rollbacks. |
| `environment/` | A procedural "Game Master" engine that authors novel logic puzzles with objective ground truths. |
| `dashboard/` | A Streamlit app that parses the `civilization_state.json` file for real-time visualization. |

---

<div align="center">
  <i>Built for autonomous AI systems research.</i>
</div>
