from pathlib import Path


def get_file_metadata(file_paths: list[str]) -> list[dict]:
    results = []

    for file_path in file_paths:
        path = Path(file_path.strip())

        if not path.is_file():
            results.append({
                "file": file_path,
                "error": "Invalid file path."
            })
            continue

        total_lines = 0
        blank_lines = 0
        comment_lines = 0
        code_lines = 0

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                stripped = line.strip()

                if not stripped:
                    blank_lines += 1
                elif stripped.startswith("#"):
                    comment_lines += 1
                else:
                    code_lines += 1

        size_bytes = path.stat().st_size

        results.append({
            "file": path.name,
            "path": str(path),
            "extension": path.suffix,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "total_lines": total_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "code_lines": code_lines
        })

    return results
