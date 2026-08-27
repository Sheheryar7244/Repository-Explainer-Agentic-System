from typing import Literal
from pydantic import BaseModel


class AgentTask(BaseModel):

    agent: Literal[
        "explorer",
        "search"
    ]

    task: str


class SupervisorDecision(BaseModel):

    action: Literal[
        "call_agent",
        "ask_user",
        "finish"
    ]

    agents: list[AgentTask] = []

    message: str = ""