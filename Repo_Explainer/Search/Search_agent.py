from typing import TypedDict, Annotated
from pathlib import Path
import operator
import time

from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    RemoveMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.graph.message import add_messages


from Repo_Explainer.tools.code_return import (
    get_code_range,
    get_code_by_name
)


# --------------------------------------------------
# ENV
# --------------------------------------------------

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)


# --------------------------------------------------
# STATE
# --------------------------------------------------

class SearchState(TypedDict, total=False):

    messages: Annotated[list, add_messages]

    # Results produced by parallel workers
    code_results: Annotated[list, operator.add]

    # Individual worker task
    code_task: dict


# --------------------------------------------------
# TOOL
# --------------------------------------------------

@tool
def read_code(requests: list[dict]):
    """
    Retrieve source code from one or more repository files.

    Each item in `requests` is a dict with:
    - file_path (str): required
    - mode (str): 'range', 'function', or 'class'
    - start_line (int): required if mode='range'
    - end_line (int): required if mode='range'
    - name (str): required if mode='function' or 'class'

    Provide ALL needed files in a single call — they will be
    retrieved in parallel.
    """

    results = []

    for req in requests:

        file_path = req.get("file_path", "")
        mode = req.get("mode", "")
        start_line = req.get("start_line", 0)
        end_line = req.get("end_line", 0)
        name = req.get("name", "")

        try:

            if mode == "range":
                print(f"\n[Search Worker] Reading code: {file_path} | Lines {start_line}-{end_line}")
                result = get_code_range(file_path, start_line, end_line)

            elif mode in ("function", "class"):
                print(f"\n[Search Worker] Reading {mode}: {name} from {file_path}")
                result = get_code_by_name(file_path, name)

            else:
                result = "Invalid mode. Use range, function, or class."

            print(f"[Search Worker] Completed: {file_path}")

        except Exception as e:
            print(f"[Search Worker] Failed: {file_path} | {e}")
            result = f"Tool error: {e}"

        results.append({
            "file_path": file_path,
            "mode": mode,
            "name": name,
            "result": result,
        })

    return results
# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0
)

llm_with_tools = llm.bind_tools(
    [read_code]
)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are the Search Agent.

Your responsibility is to find and retrieve the specific source code
needed to answer the task given by the Supervisor.

You have exactly one tool:

read_code

Use:

- mode='range' for specific line ranges
- mode='function' for a complete function
- mode='class' for a complete class

The Supervisor will provide the repository path and repository tree.

Use the repository tree to identify relevant files.

Do not generate the repository tree.

Do not perform unrelated exploration.

IMPORTANT PARALLELISM RULE:

When multiple independent pieces of source code are required,
request them using multiple read_code tool calls in the same response.

For example, if authentication depends on:

- auth.py
- middleware.py
- routes.py

request the relevant code from all three files.

These independent read_code requests will be executed in parallel.

Do NOT request multiple pieces of code merely because parallelism
is available.

Only request code that contributes to answering the task.

Instructions:

1. Understand the exact task given by the Supervisor.
2. Use the repository tree to identify relevant files.
3. Identify the minimum source-code pieces required.
4. Request independent pieces using separate read_code tool calls.
5. After the results are returned, determine whether they are sufficient.
6. If sufficient, stop requesting code.
7. If genuinely necessary, request additional relevant code.
8. Do not invent information.
9. Return the retrieved information clearly to the Supervisor.

IMPORTANT:

The Search Agent is task-local.

Do not depend on source-code results from a previous Supervisor task.

Only use code retrieved for the current task.

OUTPUT FORMATTING RULES:

- Return plain text only.
- Do NOT use Markdown.
- Do NOT use headings with #, ##, or ###.
- Do NOT use bold or italic formatting.
- Do NOT use Markdown tables.
"""


# --------------------------------------------------
# SEARCH NODE
# --------------------------------------------------

def search_node(state: SearchState):

    print(
        f"\n[Search] NEW TASK | "
        f"code_results="
        f"{len(state.get('code_results', []))}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# PARALLEL CODE WORKER
# --------------------------------------------------

def code_worker(state: SearchState):

    task = state["code_task"]

    file_path = task["file_path"]
    mode = task["mode"]
    start_line = task.get("start_line", 0)
    end_line = task.get("end_line", 0)
    name = task.get("name", "")

    print(
        f"\n[Search] Worker started: {file_path}"
    )

    result = read_code.invoke(
        {
            "requests": [
                {
                    "file_path": file_path,
                    "mode": mode,
                    "start_line": start_line,
                    "end_line": end_line,
                    "name": name,
                }
            ]
        }
    )

    print(
        f"[Search] Worker completed: {file_path}"
    )

    return {
        "code_results": [
            {
                "file_path": file_path,
                "mode": mode,
                "name": name,
                "result": result,
            }
        ]
    }


# --------------------------------------------------
# COLLECT RESULTS
# --------------------------------------------------

# 
def collect_code(state: SearchState):

    start_time = time.perf_counter()

    results = state.get("code_results", [])
    skipped = state.get("skipped_tool_calls", [])

    print(f"\n[Search] Collected {len(results)} code results.")

    result_text = "\n\n".join(
        [
            (
                f"File: {item['file_path']}\n"
                f"Mode: {item['mode']}\n"
                f"Name: {item['name']}\n"
                f"Code:\n{item['result']}"
            )
            for item in results
        ]
    )

    note = ""
    if skipped:
        note = (
            f"\n\nNote: {len(skipped)} additional requested "
            f"file(s) were NOT retrieved due to the per-turn "
            f"parallel limit. If they are still needed, request "
            f"them again now: "
            f"{', '.join(s['file_path'] for s in skipped)}"
        )

    end_time = time.perf_counter()
    print(f"[Search] Result collection time: {end_time - start_time:.2f}s")

    return {
        "messages": [
            HumanMessage(
                content=(
                    "Parallel source-code retrieval completed.\n\n"
                    f"{result_text}{note}"
                )
            )
        ],
        "code_results": [],
        "skipped_tool_calls": [],
    }

# --------------------------------------------------
# ROUTER
# --------------------------------------------------

# def should_continue(state: SearchState):

#     last_message = state["messages"][-1]

#     if not isinstance(last_message, AIMessage):

#         return "end"

#     if not last_message.tool_calls:

#         return "end"

#     sends = []

#     for tool_call in last_message.tool_calls:

#         if tool_call["name"] != "read_code":
#             continue

#         args = tool_call.get(
#             "args",
#             {}
#         )

#         print(
#             f"\n[Search] Spawning worker:"
#             f" {args.get('file_path')}"
#         )

#         branch_state = dict(state)

#         branch_state["code_task"] = {
#             "file_path": args.get(
#                 "file_path",
#                 ""
#             ),

#             "mode": args.get(
#                 "mode",
#                 ""
#             ),

#             "start_line": args.get(
#                 "start_line",
#                 0
#             ),

#             "end_line": args.get(
#                 "end_line",
#                 0
#             ),

#             "name": args.get(
#                 "name",
#                 ""
#             ),
#         }

#         sends.append(
#             Send(
#                 "code_worker",
#                 branch_state
#             )
#         )

#     if sends:

#         print(
#             f"\n[Search] Spawning "
#             f"{len(sends)} workers in parallel."
#         )

#         return sends

#     return "end"
MAX_PARALLEL_WORKERS = 30


def should_continue(state: SearchState):

    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return "end"

    if not last_message.tool_calls:
        return "end"

    for tool_call in last_message.tool_calls:

        if tool_call["name"] != "read_code":
            continue

        requests = tool_call.get("args", {}).get("requests", [])

        skipped = []

        if len(requests) > MAX_PARALLEL_WORKERS:
            print(f"\n[Search] Model requested {len(requests)} files, capping to {MAX_PARALLEL_WORKERS}.")
            skipped = requests[MAX_PARALLEL_WORKERS:]
            requests = requests[:MAX_PARALLEL_WORKERS]

        sends = []

        for req in requests:

            print(f"\n[Search] Spawning worker: {req.get('file_path')}")

            branch_state = dict(state)
            branch_state["code_task"] = {
                "file_path": req.get("file_path", ""),
                "mode": req.get("mode", ""),
                "start_line": req.get("start_line", 0),
                "end_line": req.get("end_line", 0),
                "name": req.get("name", ""),
            }
            branch_state["skipped_tool_calls"] = skipped

            sends.append(Send("code_worker", branch_state))

        if sends:
            print(f"\n[Search] Spawning {len(sends)} workers in parallel.")
            return sends

    return "end"
# --------------------------------------------------
# GRAPH
# --------------------------------------------------

graph = StateGraph(SearchState)


graph.add_node(
    "search",
    search_node
)

graph.add_node(
    "code_worker",
    code_worker
)

graph.add_node(
    "collect_code",
    collect_code
)


# --------------------------------------------------
# START
# --------------------------------------------------

graph.add_edge(
    START,
    "search"
)


# --------------------------------------------------
# SEARCH → PARALLEL WORKERS
# --------------------------------------------------

graph.add_conditional_edges(
    "search",
    should_continue,
    {
        "end": END,
        "code_worker": "code_worker"
    }
)


# --------------------------------------------------
# WORKER → COLLECTOR
# --------------------------------------------------

graph.add_edge(
    "code_worker",
    "collect_code"
)


# --------------------------------------------------
# COLLECTOR → SEARCH
# --------------------------------------------------

graph.add_edge(
    "collect_code",
    "search"
)



search_agent = graph.compile()