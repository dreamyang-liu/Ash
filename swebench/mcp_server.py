"""SWE-bench entry point for the ash MCP execution plane.

The proxy itself is generic and lives in :mod:`harness.execution.server`; it runs
tools in sandboxes and has no notion of what a run's *answer* is. This module
supplies the SWE-bench half and keeps the CLI the harnesses and docs already use:

    python -m swebench.mcp_server --http --port 8400
    python -m swebench.mcp_server --image <docker-image> --patch-dir /tmp/patches/

``--patch-dir`` exists for one topology: ``harnesses/claude_code.py`` spawns this
as a **stdio subprocess with --image**, which means the *server* creates and
destroys the sandbox, so the harness cannot extract the diff itself -- it is not
the owner. :class:`PatchWritingPool` therefore keeps a diff on disk for it.

Prefer owning the sandbox instead. ``harness.execution.provision`` provisions one
and hands the agent a bound wiring, so the caller can extract from the live
sandbox before teardown, or from any step's snapshot afterwards
(``harness/extract.py``) -- which also lets a changed extractor be re-run over
finished runs. ``harnesses/marathon_claude_code.py`` takes the in-process route
for the same reason.

Note this is a *subclass*, not a plugin: the execution plane needs no hook
protocol for a benchmark to act on sandbox lifecycle.
"""

from __future__ import annotations

import sys
from pathlib import Path

from harness.execution.server import (  # noqa: F401  (re-exported)
    ALL_TOOLS,
    EXEC_TOOLS,
    EXEC_TOOLS_MULTI,
    EXEC_TOOLS_SINGLE,
    LIFECYCLE_TOOLS,
    HttpMcpServer,
    SandboxEntry,
    SandboxPool,
    Session,
    SessionHandler,
    StdioMcpServer,
)
from harness.execution.server import main as _server_main

from .patch import extract_patch_async, untracked_baseline

__all__ = [
    "ALL_TOOLS", "EXEC_TOOLS", "EXEC_TOOLS_MULTI", "EXEC_TOOLS_SINGLE",
    "LIFECYCLE_TOOLS", "HttpMcpServer", "PatchWritingPool", "SandboxEntry",
    "SandboxPool", "Session", "SessionHandler", "StdioMcpServer", "main",
]


class PatchWritingPool(SandboxPool):
    """A pool that keeps ``<patch_dir>/<sandbox id>.diff`` current.

    Two timings, both learned the hard way:

    - The untracked baseline is taken **before an agent can touch the sandbox**,
      or an image-shipped ``build/`` tree is indistinguishable later from a file
      the agent created.
    - The diff is refreshed after every mutating call rather than at shutdown.
      Extraction at shutdown races the harness's read under load, and a killed
      run left nothing at all.
    """

    def __init__(self, *args, patch_dir: "str | Path | None" = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_dir = Path(patch_dir) if patch_dir else None

    async def create(self, image: str, groups: list[str]) -> SandboxEntry:
        entry = await super().create(image, groups)
        try:
            entry.meta["baseline_untracked"] = await untracked_baseline(entry.sandbox)
            await self._write(entry)      # an empty diff exists immediately
        except Exception as exc:  # noqa: BLE001 - never fail a create over this
            self._log(f"baseline failed for {entry.id}: {exc}")
        return entry

    async def after_mutating_call(self, entry, tool_name: str, args: dict) -> None:
        try:
            await self._write(entry)
        except Exception as exc:  # noqa: BLE001 - would surface as a tool error
            self._log(f"patch refresh failed for {entry.id}: {exc}")

    async def destroy(self, sb_id: str) -> None:
        entry = self.get(sb_id)
        if entry is not None:
            try:
                await self._write(entry)
            except Exception as exc:  # noqa: BLE001
                self._log(f"final patch failed for {sb_id}: {exc}")
        await super().destroy(sb_id)

    async def _write(self, entry) -> str:
        patch = await extract_patch_async(
            entry.sandbox, entry.base_commit,
            baseline_untracked=entry.meta.get("baseline_untracked"),
        )
        if self.patch_dir:
            self.patch_dir.mkdir(parents=True, exist_ok=True)
            (self.patch_dir / f"{entry.id}.diff").write_text(patch)
        return patch


def main() -> None:
    """Mount :class:`PatchWritingPool` when ``--patch-dir`` is given."""
    argv = sys.argv[1:]
    patch_dir = _take_value(argv, "--patch-dir")
    patch_file = _take_value(argv, "--patch-file")   # deprecated alias
    if not patch_dir and patch_file:
        patch_dir = str(Path(patch_file).parent)

    pool_cls = None
    if patch_dir:
        def pool_cls(**kwargs):  # noqa: E731 - a factory, closing over patch_dir
            return PatchWritingPool(patch_dir=patch_dir, **kwargs)

    sys.argv = [sys.argv[0]] + argv
    _server_main(pool_cls=pool_cls)


def _take_value(argv: list[str], flag: str) -> "str | None":
    """Remove ``--flag value`` / ``--flag=value`` from argv, returning the value."""
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if token.startswith(flag + "="):
            return argv.pop(i).split("=", 1)[1]
    return None


if __name__ == "__main__":
    main()
