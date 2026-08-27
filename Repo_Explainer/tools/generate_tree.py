from pathlib import Path
import pathspec


IGNORE_PATTERNS = """
.git
.gitignore
__pycache__
.pyc
""".splitlines()


def generate_tree(directory: str) -> str:
    root = Path(directory).resolve()

    if not root.is_dir():
        raise ValueError("Invalid directory.")

    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch",
        IGNORE_PATTERNS
    )

    lines = [root.name + "/"]

    def build_tree(path, prefix=""):
        items = []

        for item in path.iterdir():
            relative_path = item.relative_to(root)

            if spec.match_file(str(relative_path)):
                continue

            items.append(item)

        items.sort(
            key=lambda x: (x.is_file(), x.name.lower())
        )

        for index, item in enumerate(items):
            is_last = index == len(items) - 1

            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + item.name)

            if item.is_dir():
                extension = "    " if is_last else "│   "
                build_tree(
                    item,
                    prefix + extension
                )

    build_tree(root)

    return "\n".join(lines)
