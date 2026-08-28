"""SWE-bench's sandbox session: a :class:`SandboxSession` that knows about patches.

The sandbox half -- spawn, destroy, upload, snapshot, re-board, and the
``(tool_name, args) -> ToolResult`` executor seam -- moved to
:mod:`harness.execution.session`. It had to: the orchestrator needs to own a
sandbox in order to own a run (it is what gets snapshotted, and its teardown must
be guaranteed), and it could not reach a session that lived up here in the
benchmark layer. So the orchestrator required its caller to build one and pass it
in, which is the opposite of being an entry point.

What is left here is what only SWE-bench knows: a *patch* is the answer, so this
subclass records the baseline a diff is taken against -- the repository commit and
which files the image itself left untracked -- and turns that into a patch on
request. That is a "what counts as the answer" question, which is why it stays in
this layer rather than descending with the rest.
"""

from typing import Optional

from ash_sandbox import Sandbox

from harness.execution.session import OWNER_AGENT_ID, SandboxSession

from . import style as S
from .models import ToolResult
from .patch import UNTRACKED_LIST, WORKDIR, extract_patch

#: Identity for the harness's own traffic (patch extraction, resets, test runs).
#: Kept as a name here because call sites across this package import it; it is the
#: execution plane's :data:`~harness.execution.session.OWNER_AGENT_ID`.
HARNESS_AGENT_ID = OWNER_AGENT_ID


class AshSession(SandboxSession):
    """Manages an ash sandbox for SWE-bench evaluation.

    Adds to :class:`SandboxSession` exactly what a patch needs: the baseline it is
    diffed against, and :meth:`get_patch`.
    """

    def __init__(self, runtime_bin: "str | None" = None, timeout: float = 300.0,
                 quiet: bool = False, backend: "dict | None" = None):
        super().__init__(runtime_bin=runtime_bin, timeout=timeout, quiet=quiet,
                         backend=backend)
        self._base_commit: str = ""
        #: Untracked paths present before the agent ran (see get_patch).
        self._baseline_untracked: set[str] = set()

    # --- reporting: this package's styled CLI ------------------------------
    def _note(self, label: str, text: str) -> None:
        if not self.quiet:
            print(S.kv("%-8s" % label, S.dim(text) if label != "sandbox"
                       else S.cyan(text)))

    def _warn(self, text: str) -> None:
        print(f"  {S.bright_red('!')} {text}")

    # --- the baseline a diff is taken against ------------------------------
    async def _after_create(self, sandbox: Sandbox) -> None:
        """Record what the image already had, before the agent touches anything.

        Both probes are the patch's business, which is why they live here and not
        in the execution plane, and why they run once: re-probing after a re-board
        would file the agent's own new files under the baseline and silently drop
        them from the answer.
        """
        r = await sandbox.call("shell", command=f"git -C {WORKDIR} rev-parse HEAD")
        if not r.is_error:
            self._base_commit = r.output.strip()
        # What is already untracked before the agent starts. A SWE-bench image can
        # ship a `build/` tree or a stray artifact; those are the image's, and
        # after the run there is no way to tell them from the agent's own new files.
        probe = await sandbox.call(
            "shell", command=f"cd {WORKDIR} && {UNTRACKED_LIST}")
        self._baseline_untracked = set(
            line.strip() for line in (probe.output or "").splitlines()
            if line.strip()) if not probe.is_error else set()

    def environment(self) -> dict:
        """The execution plane's provenance plus the repository state.

        ``base_commit`` is what a replay is ultimately about: the image identifies
        the bits, the commit identifies the code inside them.
        """
        env = super().environment()
        env["base_commit"] = self._base_commit
        return env

    # --- the answer --------------------------------------------------------
    def get_patch(self) -> str:
        """Everything the agent changed, as a diff.

        Includes files it created -- only the agent knows whether a new file is
        part of the answer -- while excluding what the image already had and caches
        nobody means to submit. See ``swebench/patch.py``.
        """
        def shell(command: str) -> ToolResult:
            return self.execute("shell", {"command": command,
                                          "working_dir": WORKDIR})

        patch, added = extract_patch(shell, self._base_commit,
                                     self._baseline_untracked)
        if added and not self.quiet:
            shown = ", ".join(added[:4])
            more = f" (+{len(added) - 4})" if len(added) > 4 else ""
            print(S.kv("added   ", S.dim(f"new files in patch: {shown}{more}")))
        return patch


__all__ = ["AshSession", "HARNESS_AGENT_ID", "SandboxSession", "Optional"]
