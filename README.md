# Repository Explainer Agentic System

**Repo Explainer** is an autonomous, multi-agent AI system designed to analyze GitHub repositories. Powered by a LangGraph-based orchestration engine and a FastAPI backend, this system does not just blindly read code—it dynamically plans, explores, and reads repositories using specialized sub-agents to answer complex architectural and codebase questions in natural language.

## 🚀 Key Features

*   **Autonomous Multi-Agent Orchestration:** A Supervisor agent coordinates specialized "Explorer" and "Search" sub-agents to gather evidence before formulating an answer.
*   **Real-Time SSE Streaming:** The FastAPI backend streams live status updates (node transitions) and the final response to the frontend via Server-Sent Events (SSE).
*   **Smart Repository Caching:** Converts GitHub URLs into SHA-256 hashes to cache cloned repositories locally, preventing redundant downloads.
*   **Parallel Execution:** File metadata extraction and source code reading utilize parallel workers to minimize latency.
*   **Stateful Conversations:** Employs LangGraph's checkpointer to maintain conversation history across multiple turns without losing context.
*   **Dark-Themed Chat UI:** A responsive frontend featuring markdown rendering, code blocks, report generation, and built-in architectural audit tools.

---

## 🏗️ System Architecture

The system follows a directed graph workflow where the Supervisor controls the investigation loop until enough evidence is gathered.

    User (Frontend) → FastAPI Backend → LangGraph Orchestrator → Sub-Agents
                                          ↕
                               Cache Layer (Git Repos)

**Workflow End-to-End:**
1. User submits a GitHub URL and a question via the UI.
2. The system generates a deterministic `repo_id` and checks the local cache.
3. If a cache miss occurs, the repository is cloned.
4. The **Supervisor** receives the state (tree, question, history) and formulates an investigation plan.
5. The Supervisor delegates tasks to the **Explorer** (to understand structure) or **Search** (to read specific code) agents.
6. Sub-agents execute tools in parallel and return results.
7. The Supervisor evaluates the evidence. If insufficient, it loops back to step 5.
8. Once complete, the final answer streams back to the user.

---

## 🧠 Core Agent Ecosystem

The intelligence of the system is split across distinct LangGraph agents with specialized roles.

| Agent | Purpose | Tools | Execution Model |
| :--- | :--- | :--- | :--- |
| **Supervisor** | Central decision-maker. Formulates plans, evaluates evidence, and routes tasks. | `supervisor_decide()` | Stateful (MemorySaver Checkpointer). Two-step workflow (Tree Check → Investigation). |
| **Explorer** | Understands repository structure and file metrics. | `repository_tree`, `file_metadata` | Stateless. Spawns parallel workers for multi-file metadata collection. |
| **Search** | Reads and retrieves actual source code. | `read_code` | Stateless. Uses AST to find classes/functions or reads specific line ranges in parallel. |

---

## 📁 Project Structure

*   **`graph.py`**: The top-level orchestration graph (`START` → `check_cache` → `clone` / `supervisor` → `explorer` / `search` → `END`).
*   **`api.py`**: FastAPI backend exposing the `/chat/stream` endpoint for SSE communication.
*   **`Supervisor/`**: 
    *   `supervisor.py`: LLM decision logic using structured outputs via OpenRouter (DeepSeek V4 Flash).
    *   `supervisor_graph.py`: Internal routing logic.
    *   `state.py`: Defines the `SupervisorState` with custom reducers for clearing temporary results while preserving history.
    *   `schemas.py`: Pydantic models for structured agent tasks.
*   **`Explorer/Ex_agent.py`**: The Explorer sub-agent workflow.
*   **`Search/Search_agent.py`**: The Search sub-agent workflow.
*   **`tools/`**:
    *   `generate_tree.py`: Custom tree generation ignoring `.git` and `__pycache__`.
    *   `metadata_tool.py`: Calculates file size, extensions, and line counts (code/comments/blank).
    *   `code_return.py`: AST-based and range-based code extraction.
*   **`frontend/index.html`**: Vanilla JS/HTML/CSS single-page application with streaming support.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Sheheryar7244/Repository-Explainer-Agentic-System.git](https://github.com/Sheheryar7244/Repository-Explainer-Agentic-System.git)
cd Repository-Explainer-Agentic-System
```

### 2. Set Up a Virtual Environment
```bash
python -m venv repo_explainer_env

# Windows
.\repo_explainer_env\Scripts\activate

# macOS/Linux
source repo_explainer_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
# Required for LLM execution
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Run the Application
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Once the server is running, open `frontend/index.html` in your web browser to start analyzing repositories!

---

## ⚙️ Key Design Patterns

*   **Dynamic Parallelism:** The Supervisor uses LangGraph's `Send()` API to route multiple independent tasks to the Explorer and Search agents concurrently.
*   **Stateless Sub-Agents:** To prevent context bloat, sub-agents run without checkpointers. Their temporary state is managed and wiped by the top-level graph via a custom `add_or_reset` reducer.
*   **AST-Powered Code Reading:** Rather than dumping whole files into the context window, the Search agent uses Python's Abstract Syntax Tree (AST) to extract exact functions and classes by name.

---

## 👨‍💻 About the Developer

Developed by **Sheheryar**, an AI Engineer specializing in Agentic AI, Large Language Models, and multi-agent systems. This project was built from scratch to explore stateful LLM orchestration, dynamic planning, and advanced backend integrations using LangGraph and FastAPI.

- **GitHub:** [@Sheheryar7244](https://github.com/Sheheryar7244)
- **LinkedIn:** [www.linkedin.com/in/sheheryar-khan-19958535a]
