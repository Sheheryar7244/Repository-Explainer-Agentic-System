from dotenv import load_dotenv
load_dotenv()

import re
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 1. Import the graph as langgraph_app (remove the other import app line)
from Repo_Explainer.graph import app as langgraph_app
from Repo_Explainer.cachecheck_router import generate_repo_id

# 2. Define the FastAPI instance as app
app = FastAPI(title="Repo Explainer API")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str


# --------------------------------------------------
# Extract GitHub URL
# --------------------------------------------------

def extract_repo_url(message: str) -> str:

    match = re.search(
        r"https://github\.com/[^/\s]+/[^/\s]+",
        message
    )

    if not match:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid GitHub repository URL."
        )

    return match.group(0).rstrip(".,)")


# --------------------------------------------------
# STATUS MESSAGES
# --------------------------------------------------

STATUS_MESSAGES = {

    "check_cache":
        "Checking repository cache...",

    "clone":
        "Preparing repository...",

    "supervisor":
        "Supervisor planning next step...",

    "explorer":
        "Exploring repository...",

    "search":
        "Reading source files...",

    "ask_user":
        "Preparing response...",
}


# --------------------------------------------------
# CHAT STREAM
# --------------------------------------------------
def generate_chat(request: ChatRequest):

    message = request.message.strip()

    # --------------------------------------------------
    # Extract repository
    # --------------------------------------------------

    repo_url = extract_repo_url(message)

    question = message[
        message.find(repo_url) + len(repo_url):
    ].strip()

    if not question:

        yield (
            "event: error\n"
            f"data: {json.dumps({'message': 'Please provide a question after the repository URL.'})}\n\n"
        )

        return

    # --------------------------------------------------
    # Repository thread
    # --------------------------------------------------

    repo_id = generate_repo_id(repo_url)

    config = {
        "configurable": {
            "thread_id": repo_id
        }
    }

    # --------------------------------------------------
    # Existing state
    # --------------------------------------------------

    try:
        # CHANGED: app -> langgraph_app
        existing_state = langgraph_app.get_state(config).values

    except Exception:

        existing_state = None

    # --------------------------------------------------
    # Payload
    # --------------------------------------------------

    if not existing_state:

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
            "conversation_history": [],
        }

    else:

        payload = {
            "question": question,
            "results": ["__RESET__"],   # wipes prior question's raw results only
            "action": "",
            "next_agent": "",
            "next_task": "",
            "follow_up_question": "",
            "final_answer": "",
            # conversation_history is intentionally NOT included here —
            # omitting it means the reducer leaves it untouched, so it
            # naturally persists and keeps accumulating (capped) across
            # questions in this thread.
        }
        
    # --------------------------------------------------
    # Stream graph execution
    # --------------------------------------------------

    try:

        final_result = None

        # CHANGED: app -> langgraph_app
        for event in langgraph_app.stream(
            payload,
            config=config,
            stream_mode="updates",
        ):

            # event looks like:
            #
            # {
            #     "explorer": {...}
            # }

            for node_name, node_output in event.items():

                status = STATUS_MESSAGES.get(
                    node_name
                )

                if status:

                    yield (
                        "event: status\n"
                        f"data: {json.dumps({'message': status})}\n\n"
                    )

                final_result = node_output

        # --------------------------------------------------
        # Get final state
        # --------------------------------------------------

        # CHANGED: app -> langgraph_app
        final_state = langgraph_app.get_state(
            config
        ).values

        answer = (
            final_state.get("final_answer")
            or final_state.get("follow_up_question")
            or ""
        )

        # --------------------------------------------------
        # Final answer
        # --------------------------------------------------

        yield (
            "event: answer\n"
            f"data: {json.dumps({'answer': answer})}\n\n"
        )

    except Exception as e:

        print(
            f"[API] Chat error: {e}"
        )

        yield (
            "event: error\n"
            f"data: {json.dumps({'message': str(e)})}\n\n"
        )

# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_chat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )