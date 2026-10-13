#!/usr/bin/env python3
"""Expand {{include:<path>}} markers in docs-src/*.md into fenced code blocks.

Each include becomes:
    <!-- file: <path> -->
    ```<lang>
    <exact file contents>
    ```
so the markdown always carries the exact source that was built and tested.
"""
import re
import sys
from pathlib import Path

WORKSHOP = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "docs-src"
LANG = {".cpp": "cpp", ".txt": "cmake", ".py": "python", ".conf": "ini", ".sh": "bash"}


def expand(match: re.Match) -> str:
    rel = match.group(1).strip()
    path = WORKSHOP / rel
    lang = LANG.get(path.suffix, "")
    if path.name == "CMakeLists.txt":
        lang = "cmake"
    body = path.read_text().rstrip("\n")
    return f"<!-- file: {rel} -->\n```{lang}\n{body}\n```"


for src in sorted(SRC.glob("*.md")):
    text = re.sub(r"\{\{include:([^}]+)\}\}", expand, src.read_text())
    (WORKSHOP / src.name).write_text(text)
    print(f"wrote {src.name}")
