from Repo_Explainer.Search.Search_agent import search_agent
from langchain_core.messages import HumanMessage


repo_path = r"D:\AI-Agent-Platform"

repo_tree = """
your actual repository tree here
"""

task = f"""
Repository path:

{repo_path}

Repository tree:

{repo_tree}

Task:

Read the main files involved in the authentication system.

If multiple independent files are needed, retrieve them in parallel.
"""


config = {
    "configurable": {
        "thread_id": "search-test-1"
    }
}


result = search_agent.invoke(
    {
        "messages": [
            HumanMessage(content=task)
        ]
    },
    config=config
)


print("\n========== FINAL RESULT ==========\n")

for message in result["messages"]:
    print(message.content)