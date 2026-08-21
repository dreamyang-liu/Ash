"""Importing a module must not run a benchmark.

`run_claude.py` called `main()` at module level with no `__name__` guard, so
`import swebench.run_claude` started a 500-instance run. Found by an import sweep
that spent real money before it was killed. The file is gone; this keeps the shape
of the mistake from coming back, and doubles as a check that no module is left with
a broken import after a deletion.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

#: Modules whose imports need external services or optional dependencies.
#: They must still *parse*; they are simply not imported here.
NEEDS_EXTRAS = {"swebench.harnesses.claude_code", "swebench.mcp_server",
                "swebench.rollout_server"}


def _modules():
    return [m.name for m in pkgutil.walk_packages(["swebench"], "swebench.")
            if ".tests" not in m.name]


def test_every_module_imports_cleanly():
    """A deleted module leaves dangling imports behind; this finds them."""
    broken = {}
    for name in _modules():
        if name in NEEDS_EXTRAS:
            continue
        try:
            importlib.import_module(name)
        except ImportError as exc:          # a missing optional dep is not a defect
            if "No module named" not in str(exc):
                broken[name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, f"modules fail to import: {broken}"


def test_no_module_runs_anything_at_import_time():
    """A bare `main()` / `run()` call at module level, outside a __name__ guard."""
    import ast
    import pathlib

    offenders = []
    for path in pathlib.Path("swebench").rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:                      # module level only
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            name = getattr(func, "id", None) or getattr(func, "attr", "")
            if name in ("main", "run", "run_batch", "cli"):
                offenders.append(f"{path}:{node.lineno} calls {name}()")
    assert not offenders, (
        "these run on import, so `import x` executes a benchmark: " + "; ".join(offenders))
