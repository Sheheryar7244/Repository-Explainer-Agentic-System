from typing import TypedDict, Annotated
import operator


def add_or_reset(existing: list, update: list) -> list:
    """
    Normal behavior: append (like operator.add).
    Reset behavior: if update starts with the sentinel "__RESET__",
    discard existing and use only what follows the sentinel.
    """

    if update and update[0] == "__RESET__":
        return update[1:]

    return existing + update


class SupervisorState(TypedDict):

    repo_url: str
    repo_path: str
    repo_id: str
    question: str

    plan: list

    # Results from parallel agents, scoped to the CURRENT question only.
    # Reset per-question via the "__RESET__" sentinel (see api.py).
    results: Annotated[list, add_or_reset]

    # Generated once by Explorer and retained in Supervisor state
    repo_tree: str

    # Supervisor's current decision
    action: str

    # Used for single-agent compatibility/debugging
    next_agent: str
    next_task: str

    # Parallel agent tasks
    agent_tasks: list

    # Used when Supervisor needs clarification
    follow_up_question: str

    # Final response
    final_answer: str

    # Compact, capped record of past Q&A turns for this repo thread.
    # Persists across questions (NOT reset) so Supervisor can relate
    # new questions to earlier ones. Managed manually in supervisor_node.
    conversation_history: list