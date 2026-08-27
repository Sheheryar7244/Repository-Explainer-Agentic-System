import os
import hashlib
from git import Repo
import time
CACHE_DIR = "cache"


def generate_repo_id(repo_url: str) -> str:
    return hashlib.sha256(
        repo_url.encode("utf-8")
    ).hexdigest()[:16]


def clone_repository(repo_url: str) -> str:

    repo_id = generate_repo_id(repo_url)

    repo_path = os.path.join(
        CACHE_DIR,
        repo_id
    )

    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"[Clone] Cloning: {repo_url}")
    start_time=time.time()
    Repo.clone_from(
        repo_url,
        repo_path
    )

    print(f"[Clone] Completed: {repo_path}")
    end_time = time.time()
    print(f"Cloned in {end_time - start_time:.2f} seconds")
    return repo_path


def clone_node(state):

    repo_path = clone_repository(
        state["repo_url"]
    )

    return {
        "repo_path": repo_path
    }