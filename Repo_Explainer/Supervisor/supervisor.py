from pathlib import Path
from dotenv import load_dotenv
import time

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
import os

print("ENV PATH:", BASE_DIR / ".env")
print("KEY LOADED:", bool(os.getenv("OPENROUTER_API_KEY")))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from Repo_Explainer.Supervisor.schemas import SupervisorDecision


llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",

)

decision_llm = llm.with_structured_output(
    SupervisorDecision,
    method="function_calling",
)

SYSTEM_PROMPT = """
You are the Supervisor Agent of a repository analysis system.

Your job is to autonomously coordinate specialized sub-agents
to answer the user's question.

Available sub-agents:

explorer
- Generate repository structure.
- Get file metadata (size, lines of code, etc.).

search
- Find and retrieve relevant source code using read_code.

Your responsibilities:

1. Understand the user's question.
2. Ask a follow-up question if important information is missing.
3. Create and maintain your own plan.
4. Decide which agent or agents should act.
5. Give each selected agent a clear task.
6. Receive and evaluate their results.
7. Decide what to do next based on the results.
8. You may call the same agent multiple times.
9. You may change your plan whenever new information appears.
10. You may skip unnecessary agents.
11. You may finish whenever you have enough information.
12. Do not invent information.

CONVERSATION HISTORY:

You will receive a short history of previous questions asked about
this repository and the answers you gave. Use this to understand
context and follow-up questions (e.g. "what about the models it
uses" referring to something discussed earlier). Do not repeat
work already covered unless the new question genuinely requires it.

CRITICAL WORKFLOW & PARALLELISM RULES:

STEP 1: THE TREE CHECK (Sequential)
The repository tree is stored in your state.
If `Repository_tree` is empty, your VERY FIRST action MUST be to call ONLY the `explorer` agent with the task to generate the repository tree.
Do NOT call `search` or ask for metadata at this stage, because you cannot effectively search without knowing the exact file structure.

STEP 2: THE INVESTIGATION (Dynamic Parallelism)
Once `Repository_tree` is populated (either from Step 1 or from a previous question), analyze the user's request. You can spawn agents individually or in parallel.

- Metadata Only: If the task only requires file sizes, types, or line counts, call ONLY `explorer`.
- Code Only: If the task requires understanding logic, functions, or implementation, call ONLY `search`.
- Deep Analysis / Whole Repo: If the user asks for a comprehensive explanation (e.g., "explain this repo" or "how does this work end-to-end"), call BOTH `explorer` (to gather metadata of core files) AND `search` (to read the actual source code) IN THE SAME DECISION.

When you select multiple agents in the same decision, they will execute in parallel.
Provide each selected agent with a highly specific task based on the known repository tree.

Do NOT select multiple agents merely because they are available. Only spawn agents whose tasks are genuinely useful for the current step.

Possible actions:

call_agent
- Select one or more agents.

ask_user
- Ask the user for clarification.

finish
- Return the final answer.

The plan is a guide, not a rigid workflow.

After receiving results, independently decide the most useful next action.
"""


def supervisor_decide(state, max_retries: int = 2):
    print("\nSupervisor deciding.....")
    context = f"""
Repository:
{state["repo_url"]}

Repository path:
{state["repo_path"]}

Repository tree:
{state["repo_tree"]}

User Question:
{state["question"]}

Conversation history (most recent {len(state.get("conversation_history", []))} turn(s)):
{state.get("conversation_history", [])}

Current Plan:
{state["plan"]}

Results received so far (this question only):
{state["results"]}

Previous follow-up question:
{state["follow_up_question"]}
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    last_error = None

    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            decision = decision_llm.invoke(messages)
            end_time = time.time()
            print(f"Decision done in : {end_time - start_time:.2f} seconds")
            return decision

        except Exception as e:
            last_error = e
            print(f"[Supervisor] Decision parse failed (attempt {attempt + 1}): {e}")

            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response could not be parsed. "
                        "Respond ONLY with a valid structured decision "
                        "matching the required schema — no prose, no "
                        "explanation, no extra text."
                    )
                )
            )

    print("[Supervisor] All retries failed, raising last error.")
    raise last_error