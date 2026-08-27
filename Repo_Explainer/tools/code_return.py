import ast
from pathlib import Path


def get_code_range(file_path: str, start_line: int, end_line: int) -> str:
    path = Path(file_path)

    if not path.is_file():
        raise ValueError("Invalid file path.")

    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range.")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if start_line > len(lines):
        raise ValueError("Start line is beyond the file length.")

    end_line = min(end_line, len(lines))

    return "".join(lines[start_line - 1:end_line])


def get_code_by_name(file_path: str, name: str) -> str:
    path = Path(file_path)

    if not path.is_file():
        raise ValueError("Invalid file path.")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Could not parse Python file: {e}")

    lines = source.splitlines(keepends=True)

    matches = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if node.name == name:
                matches.append(node)

    if not matches:
        raise ValueError(
            f"No function or class named '{name}' found."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple definitions named '{name}' found."
        )

    node = matches[0]

    start = node.lineno
    end = node.end_lineno

    return "".join(lines[start - 1:end])

