from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import time
from Repo_Explainer.Explorer.Ex_agent import explorer_agent
from Repo_Explainer.Search.Search_agent import search_agent

from Repo_Explainer.Supervisor.state import SupervisorState
from Repo_Explainer.Supervisor.supervisor import supervisor_decide
import time
from langchain_core.messages import HumanMessage


MAX_HISTORY_TURNS = 10


def supervisor_node(state: SupervisorState):

    print("\nSupervisor making plan....")
    decision = supervisor_decide(state)

    agent_tasks = [
        {
            "agent": item.agent,
            "task": item.task
        }
        for item in decision.agents
    ]

    updates = {
        "action": decision.action,
        "agent_tasks": agent_tasks,
        "follow_up_question": (
            decision.message
            if decision.action == "ask_user"
            else ""
        ),
        "final_answer": (
            decision.message
            if decision.action == "finish"
            else ""
        )
    }

    # --------------------------------------------------
    # Record this Q&A turn into conversation_history when
    # Supervisor actually finishes answering. This is the
    # ONLY field that persists across questions, capped to
    # the last MAX_HISTORY_TURNS turns.
    # --------------------------------------------------

    if decision.action == "finish":

        history = list(
            state.get("conversation_history", [])
        )

        history.append({
            "question": state["question"],
            "answer": decision.message,
        })

        history = history[-MAX_HISTORY_TURNS:]

        updates["conversation_history"] = history

        print(
            f"\n[Supervisor] conversation_history now holds "
            f"{len(history)} turn(s)."
        )

    return updates


def supervisor_router(state: SupervisorState):

    print("\nSupervisor choosing an action....")

    if state["action"] == "call_agent":

        tasks = state["agent_tasks"]

        if not tasks:
            print("\nNo agents selected. Finishing.")
            return "finish"

        sends = []
        
        if len(tasks) > 1:
            print(f"\n[Supervisor] PARALLEL SPAWN TRIGGERED: Launching {len(tasks)} agents concurrently!")
       
        for task in tasks:

            agent_name = task["agent"]

            print(
                f"\nCalling {agent_name} agent..."
            )

            branch_state = dict(state)

            branch_state["next_task"] = task["task"]

            sends.append(
                Send(
                    agent_name,
                    branch_state
                )
            )

        return sends

    if state["action"] == "ask_user":

        print("\nAsking User....")

        return "ask_user"

    if state["action"] == "finish":

        print("\nAnswering....")

        return "finish"

    return "finish"


def explorer_node(state: SupervisorState):

    task = f"""
Repository path:
{state["repo_path"]}

Repository tree:
{state["repo_tree"]}

Task:
{state["next_task"]}
"""

    # NOTE: no config/thread_id here on purpose. explorer_agent is
    # compiled WITHOUT a checkpointer, so this call is always a
    # fresh, stateless task — Supervisor is the only graph that
    # owns persistent memory (via its own checkpointer + thread_id
    # in api.py).

    start_time = time.perf_counter()
    result = explorer_agent.invoke(
        {
            "messages": [
                HumanMessage(content=task)
            ]
        }
    )
    end_time = time.perf_counter()

    print(
        f"\n[Supervisor] Explorer result received in "
        f"{end_time - start_time:.2f} seconds"
    )

    messages = result["messages"]

    tree = state["repo_tree"]

    if not tree:

        for message in messages:

            if getattr(message, "name", None) == "repository_tree":

                tree = message.content

                break

    final_message = messages[-1].content

    return {
        "repo_tree": tree,
        "results": [final_message]
    }


def search_node(state: SupervisorState):

    task = f"""
Repository path:
{state["repo_path"]}

Repository tree:
{state["repo_tree"]}

Task:
{state["next_task"]}
"""

    # NOTE: no config/thread_id here either — search_agent is also
    # compiled without a checkpointer, for the same reason as above.

    result = search_agent.invoke(
        {
            "messages": [
                HumanMessage(content=task)
            ]
        }
    )

    messages = result["messages"]

    final_message = messages[-1].content

    return {
        "results": [final_message]
    }


def ask_user_node(state: SupervisorState):

    print("\nSupervisor needs clarification:")
    print(state["follow_up_question"])

    return {}


graph = StateGraph(SupervisorState)

graph.add_node(
    "supervisor",
    supervisor_node
)

graph.add_node(
    "explorer",
    explorer_node
)

graph.add_node(
    "search",
    search_node
)

graph.add_node(
    "ask_user",
    ask_user_node
)


graph.add_edge(
    START,
    "supervisor"
)


graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "ask_user": "ask_user",
        "finish": END,
        "explorer": "explorer",
        "search": "search"
    }
)


graph.add_edge(
    "explorer",
    "supervisor"
)

graph.add_edge(
    "search",
    "supervisor"
)

graph.add_edge(
    "ask_user",
    END
)


app = graph.compile()