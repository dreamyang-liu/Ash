"""Isolated invocation of the pinned upstream workspace assembler."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def main() -> None:
    repo, request_path, output_dir = map(Path, sys.argv[1:])
    sys.path.insert(0, str(repo))
    request = json.loads(request_path.read_text())
    upstream = runpy.run_path(str(repo / "swe_bench_pro_eval.py"))
    files, entryscript = upstream["assemble_workspace_files"](
        request["sample"]["instance_id"], str(repo / "run_scripts"),
        request["patch"], request["sample"])
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8", errors="surrogateescape")


if __name__ == "__main__":
    main()
