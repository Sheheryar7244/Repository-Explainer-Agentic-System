from typing import TypedDict, Annotated
from pathlib import Path
from dotenv import load_dotenv
import operator
import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import Send

from Repo_Explainer.tools.generate_tree import generate_tree
from Repo_Explainer.tools.metadata_tool import get_file_metadata


# ==================================================
# ENV
# ==================================================

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)


# ==================================================
# STATE
# ==================================================

class ExplorerState(TypedDict, total=False):

    # Explorer conversation for the CURRENT task only
    messages: Annotated[list, add_messages]

    # Results returned by parallel metadata workers
    metadata_results: Annotated[list, operator.add]

    # File assigned to the current metadata worker
    metadata_file: str


# ==================================================
# TOOLS
# ==================================================

@tool
def repository_tree(directory: str):
    """
    Generate the directory tree of a repository.

    Use this when repository structure is required.
    """

    print(
        f"\n[Explorer] Generating tree: {directory}"
    )

    try:

        result = generate_tree(directory)

        print(
            "\nTree Generation Completed."
        )

        return result

    except Exception as e:

        print(
            f"[Explorer] Tree generation failed: {e}"
        )

        return f"Tool error: {e}"


@tool
def file_metadata(file_paths: list[str]):
    """
    Get metadata for one or more files.

    Includes:
    - size
    - extension
    - total lines
    - blank lines
    - comment lines
    - code lines
    """

    print(
        "\n[Explorer] Getting file metadata..."
    )

    try:

        result = get_file_metadata(
            file_paths
        )

        print(
            "[Explorer] File metadata retrieval completed."
        )

        print(
            "[Explorer] File metadata sent to supervisor."
        )

        return result

    except Exception as e:

        print(
            f"[Explorer] File metadata retrieval failed: {e}"
        )

        return f"Tool error: {e}"


tools = [
    repository_tree,
    file_metadata,
]


# ==================================================
# LLM
# ==================================================

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

llm_with_tools = llm.bind_tools(tools)


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are the Explorer Agent.

Your responsibility is to complete ONLY the specific task given by the
Supervisor.

Each invocation is an independent task.

Do NOT assume that you have memory from previous Explorer tasks.

Do not perform additional repository exploration.
Do not gather information that was not requested.

Use the minimum number of tools necessary.

Once you have enough information to answer the task, STOP calling tools.

You have exactly two tools.

1. repository_tree

Use ONLY when the task requires:

- repository structure
- directory structure
- file hierarchy
- finding where files/directories are located

Do NOT use this tool just to get general context.


2. file_metadata

Use ONLY when the task requires:

- file size
- extension
- total lines
- blank lines
- comment lines
- code lines

If the task message already contains a "Repository tree:" section with
content in it, the repository structure has already been retrieved.

Do NOT call repository_tree again if the tree is already available.

If the required files are already known, call file_metadata directly.

When requesting metadata for multiple files, provide ALL relevant file
paths in a SINGLE file_metadata call.

The Explorer graph will automatically execute those files in parallel.


Instructions:

1. Understand the exact task given by the Supervisor.

2. Identify the minimum tool(s) required to answer that task.

3. Use ONLY those necessary tools.

4. Do NOT perform general repository exploration unless explicitly requested.

5. After receiving a tool result, determine whether it is sufficient.

6. If the result is sufficient, stop using tools and provide the answer.

7. If the result is genuinely insufficient, use another appropriate tool.

8. Do not use a tool merely to gather additional context.

9. Do not invent information.

10. Keep the final answer focused strictly on the requested task.

11. Treat every invocation as a fresh task.

12. Do not rely on information from previous Explorer tasks unless that
information is explicitly included in the current task message.


OUTPUT FORMATTING RULES:

- Return plain text only.
- Do NOT use Markdown.
- Do NOT use headings with #, ##, or ###.
- Do NOT use bold or italic formatting.
- Do NOT use Markdown tables.
- Do NOT use backticks for code unless explicitly requested.
- Use simple numbered lists or bullet points when needed.
"""


# ==================================================
# EXPLORER LLM NODE
# ==================================================

def explorer_node(state: ExplorerState):

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ] + state.get("messages", [])

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ==================================================
# NORMAL TOOL NODE
# ==================================================

tool_node = ToolNode(
    tools
)


# ==================================================
# PARALLEL METADATA WORKER
# ==================================================

def metadata_worker(state: ExplorerState):

    file_path = state["metadata_file"]

    print(
        f"\n[Explorer Worker] Processing: {file_path}"
    )

    try:

        result = get_file_metadata(
            [file_path]
        )

        print(
            f"[Explorer Worker] Completed: {file_path}"
        )

        return {
            "metadata_results": [
                {
                    "file_path": file_path,
                    "result": result
                }
            ]
        }

    except Exception as e:

        print(
            f"[Explorer Worker] Failed: "
            f"{file_path} | {e}"
        )

        return {
            "metadata_results": [
                {
                    "file_path": file_path,
                    "result": f"Tool error: {e}"
                }
            ]
        }


# ==================================================
# COLLECT METADATA
# ==================================================

def collect_metadata(state: ExplorerState):

    print(
        "\n[Explorer] Collecting metadata results..."
    )

    results = state.get(
        "metadata_results",
        []
    )

    print(
        f"[Explorer] Received "
        f"{len(results)} metadata results."
    )

    results_text = "\n\n".join(
        [
            (
                f"File: {item['file_path']}\n"
                f"Metadata:\n{item['result']}"
            )
            for item in results
        ]
    )

    # The last AI message contains the file_metadata
    # tool call. We remove it because the actual files
    # were processed by parallel workers.

    messages = state.get(
        "messages",
        []
    )

    last_message = messages[-1]

    return {
        "messages": [
            RemoveMessage(
                id=last_message.id
            ),
            HumanMessage(
                content=(
                    "Parallel file metadata retrieval "
                    "completed.\n\n"
                    f"{results_text}"
                )
            )
        ]
    }


# ==================================================
# ROUTER
# ==================================================

def should_continue(state: ExplorerState):

    last_message = state["messages"][-1]

    # No AI tool call means the Explorer is finished.
    if not isinstance(
        last_message,
        AIMessage
    ):
        return "end"

    if not last_message.tool_calls:
        return "end"

    for tool_call in last_message.tool_calls:

        if tool_call["name"] == "file_metadata":

            file_paths = tool_call.get(
                "args",
                {}
            ).get(
                "file_paths",
                []
            )

            print(
                f"\n[Explorer] Spawning "
                f"{len(file_paths)} metadata workers..."
            )

            sends = []

            for file_path in file_paths:

                print(
                    f"[Explorer] Spawn → {file_path}"
                )

                branch_state = dict(
                    state
                )

                branch_state[
                    "metadata_file"
                ] = file_path

                sends.append(
                    Send(
                        "metadata_worker",
                        branch_state
                    )
                )

            return sends

    return "tools"


# ==================================================
# GRAPH
# ==================================================

graph = StateGraph(
    ExplorerState
)


# --------------------------------------------------
# Nodes
# --------------------------------------------------

graph.add_node(
    "explorer",
    explorer_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "metadata_worker",
    metadata_worker
)

graph.add_node(
    "collect_metadata",
    collect_metadata
)


# --------------------------------------------------
# Edges
# --------------------------------------------------

graph.add_edge(
    START,
    "explorer"
)


graph.add_conditional_edges(
    "explorer",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)


graph.add_edge(
    "tools",
    "explorer"
)


graph.add_edge(
    "metadata_worker",
    "collect_metadata"
)


graph.add_edge(
    "collect_metadata",
    "explorer"
)


# ==================================================
# COMPILE
# ==================================================

# IMPORTANT:
#
# Explorer is intentionally compiled WITHOUT a
# checkpointer.
#
# Every Supervisor -> Explorer call starts a fresh
# Explorer task.
#
# Persistent conversation memory belongs to the
# Supervisor graph.

explorer_agent = graph.compile()


# from typing import TypedDict, Annotated
# from pathlib import Path
# from dotenv import load_dotenv
# import operator
# import os
# import time
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langchain_core.messages import (
#     SystemMessage,
#     AIMessage,
#     HumanMessage,
#     RemoveMessage,
# )

# from langgraph.graph import StateGraph, START, END
# from langgraph.prebuilt import ToolNode
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import Send

# from Repo_Explainer.tools.generate_tree import generate_tree
# from Repo_Explainer.tools.metadata_tool import get_file_metadata


# # --------------------------------------------------
# # ENV
# # --------------------------------------------------

# ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# load_dotenv(ENV_PATH)


# # --------------------------------------------------
# # STATE
# # --------------------------------------------------

# class ExplorerState(TypedDict, total=False):

#     messages: Annotated[list, add_messages]

#     # Results returned by parallel metadata workers
#     metadata_results: Annotated[list, operator.add]

#     # File assigned to the current metadata worker
#     metadata_file: str


# # --------------------------------------------------
# # TOOLS
# # --------------------------------------------------

# @tool
# def repository_tree(directory: str):
#     """
#     Generate the directory tree of a repository.
#     Use this when repository structure is needed.
#     """

#     print(f"\n[Explorer] Generating tree: {directory}")

#     try:

#         result = generate_tree(directory)

#         print("\nTree Generation Completed.")
#         return result
        
#     except Exception as e:

#         print(
#             f"[Explorer] Tree generation failed: {e}"
#         )

#         return f"Tool error: {e}"
      

# @tool
# def file_metadata(file_paths: list[str]):
#     """
#     Get metadata for one or more files.

#     Includes:
#     - size
#     - extension
#     - total lines
#     - blank lines
#     - comment lines
#     - code lines
#     """

#     print("\n[Explorer] Getting file metadata...")

#     try:

#         result = get_file_metadata(file_paths)

#         print(
#             "[Explorer] File metadata retrieval completed."
#         )

#         return result
        
#         print(
#             "[Explorer] File metadata sent to supervisor."
#         )
#     except Exception as e:

#         print(
#             f"[Explorer] File metadata retrieval failed: {e}"
#         )

#         return f"Tool error: {e}"


# tools = [
#     repository_tree,
#     file_metadata,
# ]


# # --------------------------------------------------
# # LLM
# # --------------------------------------------------
# llm = ChatOpenAI(
#     model="deepseek/deepseek-v4-flash",
#     temperature=0,
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",

# )

# llm_with_tools = llm.bind_tools(tools)


# # --------------------------------------------------
# # SYSTEM PROMPT
# # --------------------------------------------------

# SYSTEM_PROMPT = """
# You are the Explorer Agent.

# Your responsibility is to complete ONLY the specific task given by the
# Supervisor.

# Do not perform additional repository exploration.
# Do not gather information that was not requested.
# Use the minimum number of tools necessary.
# Once you have enough information to answer the task, STOP calling tools.

# You have exactly two tools.

# 1. repository_tree

# Use ONLY when the task requires:
# - repository structure
# - directory structure
# - file hierarchy
# - finding where files/directories are located

# Do NOT use this tool just to get general context.

# 2. file_metadata

# Use ONLY when the task requires:
# - file size
# - extension
# - total lines
# - blank lines
# - comment lines
# - code lines

# If the task message already contains a "Repository tree:" section with
# content in it, the repository structure has already been retrieved.

# Do NOT call repository_tree again if the tree is already available.

# If the required files are already known, call file_metadata directly.

# When requesting metadata for multiple files, provide ALL relevant file
# paths in a SINGLE file_metadata call.

# The Explorer graph will automatically execute those files in parallel.

# Instructions:

# 1. Understand the exact task given by the Supervisor.
# 2. Identify the minimum tool(s) required to answer that task.
# 3. Use ONLY those necessary tools.
# 4. Do NOT perform general repository exploration unless explicitly requested.
# 5. After receiving a tool result, determine whether it is sufficient.
# 6. If the result is sufficient, stop using tools and provide the answer.
# 7. If the result is genuinely insufficient, use another appropriate tool.
# 8. Do not use a tool merely to gather additional context.
# 9. Do not invent information.
# 10. Keep the final answer focused strictly on the requested task.

# OUTPUT FORMATTING RULES:

# - Return plain text only.
# - Do NOT use Markdown.
# - Do NOT use headings with #, ##, or ###.
# - Do NOT use bold or italic formatting.
# - Do NOT use Markdown tables.
# - Do NOT use backticks for code unless explicitly requested.
# - Use simple numbered lists or bullet points when needed.
# """


# # --------------------------------------------------
# # EXPLORER LLM NODE
# # --------------------------------------------------

# def explorer_node(state: ExplorerState):

#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT)
#     ] + state["messages"]

#     response = llm_with_tools.invoke(messages)

#     return {
#         "messages": [response]
#     }


# # --------------------------------------------------
# # NORMAL TOOL NODE
# # --------------------------------------------------

# tool_node = ToolNode(tools)


# # --------------------------------------------------
# # PARALLEL METADATA WORKER
# # --------------------------------------------------

# def metadata_worker(state: ExplorerState):

#     file_path = state["metadata_file"]

#     print(
#         f"\n[Explorer Worker] Processing: {file_path}"
#     )

#     try:

#         result = get_file_metadata(
#             [file_path]
#         )

#         print(
#             f"[Explorer Worker] Completed: {file_path}"
#         )

#         return {
#             "metadata_results": [
#                 {
#                     "file_path": file_path,
#                     "result": result
#                 }
#             ]
#         }

#     except Exception as e:

#         print(
#             f"[Explorer Worker] Failed: "
#             f"{file_path} | {e}"
#         )

#         return {
#             "metadata_results": [
#                 {
#                     "file_path": file_path,
#                     "result": f"Tool error: {e}"
#                 }
#             ]
#         }


# # --------------------------------------------------
# # METADATA DISPATCHER
# # --------------------------------------------------

# def metadata_dispatcher(state: ExplorerState):

#     last_message = state["messages"][-1]

#     sends = []

#     for tool_call in last_message.tool_calls:

#         if tool_call["name"] != "file_metadata":
#             continue

#         args = tool_call.get("args", {})

#         file_paths = args.get(
#             "file_paths",
#             []
#         )

#         print(
#             f"\n[Explorer] Spawning "
#             f"{len(file_paths)} metadata workers..."
#         )

#         for file_path in file_paths:

#             print(
#                 f"[Explorer] Spawn → {file_path}"
#             )

#             branch_state = dict(state)

#             branch_state["metadata_file"] = file_path

#             sends.append(
#                 Send(
#                     "metadata_worker",
#                     branch_state
#                 )
#             )

#     return sends


# # --------------------------------------------------
# # COLLECT METADATA
# # --------------------------------------------------

# def collect_metadata(state: ExplorerState):

#     print(
#         "\n[Explorer] Collecting metadata results..."
#     )

#     results = state.get(
#         "metadata_results",
#         []
#     )

#     print(
#         f"[Explorer] Received "
#         f"{len(results)} metadata results."
#     )

#     results_text = "\n\n".join(
#         [
#             (
#                 f"File: {item['file_path']}\n"
#                 f"Metadata:\n{item['result']}"
#             )
#             for item in results
#         ]
#     )

#     # Remove the AI message that requested file_metadata.
#     #
#     # We don't use ToolNode for that call because the files
#     # were executed by parallel workers instead.

#     messages = state["messages"]

#     last_message = messages[-1]

#     return {
#         "messages": [
#             RemoveMessage(id=last_message.id),
#             HumanMessage(
#                 content=(
#                     "Parallel file metadata retrieval "
#                     "completed.\n\n"
#                     f"{results_text}"
#                 )
#             )
#         ]
#     }


# # --------------------------------------------------
# # ROUTER
# # --------------------------------------------------
# def should_continue(state: ExplorerState):

#     last_message = state["messages"][-1]

#     if not isinstance(last_message, AIMessage):
#         return "end"

#     if not last_message.tool_calls:
#         return "end"

#     for tool_call in last_message.tool_calls:

#         if tool_call["name"] == "file_metadata":

#             file_paths = tool_call.get("args", {}).get(
#                 "file_paths",
#                 []
#             )

#             print(
#                 f"\n[Explorer] Spawning "
#                 f"{len(file_paths)} metadata workers..."
#             )

#             sends = []

#             for file_path in file_paths:

#                 print(
#                     f"[Explorer] Spawn → {file_path}"
#                 )

#                 branch_state = dict(state)

#                 branch_state["metadata_file"] = file_path

#                 sends.append(
#                     Send(
#                         "metadata_worker",
#                         branch_state
#                     )
#                 )

#             return sends

#     return "tools"


# # --------------------------------------------------
# # GRAPH
# # --------------------------------------------------
# graph = StateGraph(ExplorerState)

# graph.add_node(
#     "explorer",
#     explorer_node
# )

# graph.add_node(
#     "tools",
#     tool_node
# )

# graph.add_node(
#     "metadata_worker",
#     metadata_worker
# )

# graph.add_node(
#     "collect_metadata",
#     collect_metadata
# )

# graph.add_edge(
#     START,
#     "explorer"
# )

# graph.add_conditional_edges(
#     "explorer",
#     should_continue,
#     {
#         "metadata": "metadata_worker",
#         "tools": "tools",
#         "end": END
#     }
# )

# graph.add_edge(
#     "tools",
#     "explorer"
# )

# graph.add_edge(
#     "metadata_worker",
#     "collect_metadata"
# )

# graph.add_edge(
#     "collect_metadata",
#     "explorer"
# )


# # --------------------------------------------------
# # MEMORY
# # --------------------------------------------------

# memory = MemorySaver()


# explorer_agent = graph.compile(
#     checkpointer=memory
# )