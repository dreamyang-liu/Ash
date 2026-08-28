"""Codex CLI slot (``codex exec --json``).

Verified against codex-cli 0.145.0.

Wiring notes:
- MCP servers are configured via repeated ``-c mcp_servers.<name>.<key>=<toml>``
  overrides; no config file is written, so concurrent slots never race on
  ~/.codex/config.toml.
- Sandboxing: we pass ``--sandbox read-only`` by default because side effects
  are supposed to happen in the ash sandbox *through MCP tools*, not on the host
  where the CLI runs. Callers that deliberately want host writes must opt in via
  ``extra["sandbox"]``. (We do not pass
  ``--dangerously-bypass-approvals-and-sandbox``.)
- ``--skip-git-repo-check`` so a bare task dir works.
- ``--ephemeral`` is opt-in (``extra["ephemeral"]``): rollout files under
  ~/.codex/sessions are what ``codex exec resume`` needs, so a slot that wants
  resume must keep them.
- Prompt goes on stdin to avoid argv length limits and quoting issues.
"""

from __future__ import annotations

from typing import List, Optional

from harness.core.slot import McpWiring, SlotCapabilities, TaskSpec
from harness.normalize import codex as codex_normalize
from harness.slots.cli_base import JsonlCliSlot


def _toml_str(value: str) -> str:
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_str_array(values: List[str]) -> str:
    return "[%s]" % ",".join(_toml_str(v) for v in values)


def _mcp_config_pairs(mcp: Optional[McpWiring]) -> List[str]:
    """One TOML assignment per leaf of the MCP server definition.

    Shared by both codex slots -- the CLI passes each as ``-c <pair>``, the SDK
    as ``config_overrides`` (which its client turns into ``--config <pair>``).
    One serializer, because the quoting rules are the part that drifts.
    """
    if mcp is None:
        return []
    prefix = "mcp_servers.%s" % mcp.name
    out: List[str] = []
    if mcp.command:
        out.append("%s.command=%s" % (prefix, _toml_str(mcp.command[0])))
        if len(mcp.command) > 1:
            out.append("%s.args=%s" % (prefix, _toml_str_array(mcp.command[1:])))
        for key, value in (mcp.env or {}).items():
            out.append("%s.env.%s=%s" % (prefix, key, _toml_str(value)))
    elif mcp.url:
        out.append("%s.url=%s" % (prefix, _toml_str(mcp.url)))
        for key, value in (mcp.headers or {}).items():
            out.append("%s.http_headers.%s=%s" % (prefix, key, _toml_str(value)))
    return out


class CodexSlot(JsonlCliSlot):
    name = "codex"
    binary = "codex"
    normalizer = staticmethod(codex_normalize.normalize)
    capabilities = SlotCapabilities(
        resume=True,          # codex exec resume <id> / --last
        fork=False,           # no native fork; rollback comes from env snapshots
        mcp_stdio=True,
        mcp_remote=True,      # streamable-http via mcp_servers.<n>.url
        emits_usage=True,
        deny_builtin_tools=False,  # constrained via --sandbox instead
    )

    def build_command(self, task: TaskSpec, mcp: Optional[McpWiring]) -> List[str]:
        extra = task.extra or {}
        command = ["codex", "exec", "--json", "--skip-git-repo-check"]

        sandbox = extra.get("sandbox", "read-only")
        if sandbox:
            command += ["--sandbox", str(sandbox)]
        if task.model:
            command += ["-m", task.model]
        if extra.get("ephemeral"):
            command.append("--ephemeral")
        if extra.get("cd"):
            command += ["-C", str(extra["cd"])]
        if extra.get("output_schema"):
            command += ["--output-schema", str(extra["output_schema"])]

        command += self._mcp_overrides(mcp)
        for key, value in (extra.get("config") or {}).items():
            command += ["-c", "%s=%s" % (key, value)]

        resume_id = extra.get("resume_session_id")
        if resume_id:
            # `codex exec resume <id>` keeps the flags but takes a subcommand.
            command.insert(2, "resume")
            command.insert(3, str(resume_id))
        command.append("-")  # read prompt from stdin
        return command

    def stdin_payload(self, task: TaskSpec) -> Optional[str]:
        return task.prompt

    @staticmethod
    def _mcp_overrides(mcp: Optional[McpWiring]) -> List[str]:
        out: List[str] = []
        for pair in _mcp_config_pairs(mcp):
            out += ["-c", pair]
        return out
