import os
import hashlib


CACHE_DIR = "cache"


def generate_repo_id(repo_url: str) -> str:
    return hashlib.sha256(
        repo_url.encode("utf-8")
    ).hexdigest()[:16]


def check_cache(state):

    print("\nChecking Cache.....")

    repo_url = state["repo_url"]

    repo_id = generate_repo_id(repo_url)

    repo_path = os.path.join(
        CACHE_DIR,
        repo_id
    )

    if os.path.exists(repo_path):
        print("[Cache] Repository found.")
        
        return {
            "repo_id": repo_id,
            "repo_path": repo_path,
            "cache_hit": True
        }

    print("[Cache] Repository not found.")

    return {
        "repo_id": repo_id,
        "cache_hit": False
    }


def cache_router(state):

    if state["cache_hit"]:
        return "supervisor"

    return "clone"