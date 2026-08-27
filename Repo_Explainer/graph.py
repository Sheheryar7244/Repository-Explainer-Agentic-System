from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from Repo_Explainer.cachecheck_router import (
    generate_repo_id,
    check_cache,
    cache_router,
)

from Repo_Explainer.clone_node import clone_node

from Repo_Explainer.Supervisor.state import SupervisorState

from Repo_Explainer.Supervisor.supervisor_graph import (
    supervisor_node,
    supervisor_router,
    explorer_node,
    search_node,
    ask_user_node,
)


graph = StateGraph(SupervisorState)


# Nodes
graph.add_node("check_cache", check_cache)
graph.add_node("clone", clone_node)

graph.add_node("supervisor", supervisor_node)
graph.add_node("explorer", explorer_node)
graph.add_node("search", search_node)
graph.add_node("ask_user", ask_user_node)


# START → Cache Check
graph.add_edge(START, "check_cache")


# Cache routing
graph.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "clone": "clone",
        "supervisor": "supervisor",
    }
)


# Clone → Supervisor
graph.add_edge("clone", "supervisor")


# Supervisor decides what happens next
graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "explorer": "explorer",
        "search": "search",
        "ask_user": "ask_user",
        "finish": END,
    }
)


# Agent → Supervisor
graph.add_edge("explorer", "supervisor")
graph.add_edge("search", "supervisor")


# Ask user → END
graph.add_edge("ask_user", END)


# Checkpointing
memory = MemorySaver()

app = graph.compile(checkpointer=memory)


# Terminal testing
if __name__ == "__main__":

    repo_url = input("Enter Repo URL: ").strip()
    repo_id = generate_repo_id(repo_url)

    config = {
        "configurable": {
            "thread_id": repo_id
        }
    }

    first_turn = True

    while True:

        question = input(
            "\nAsk Question (type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if first_turn:

            payload = {
                "repo_url": repo_url,
                "repo_path": "",
                "repo_id": repo_id,
                "cache_hit": False,

                "repo_tree": "",

                "question": question,

                "plan": [],
                "results": [],

                "action": "",
                "next_agent": "",
                "next_task": "",

                "follow_up_question": "",

                "final_answer": "",
            }

            first_turn = False

        else:

            payload = {
                "question": question,

                "results": [],

                "action": "",
                "next_agent": "",
                "next_task": "",

                "follow_up_question": "",

                "final_answer": "",
            }

        result = app.invoke(
            payload,
            config=config
        )

        print("\n--- FINAL ANSWER ---")
        print(result["final_answer"])