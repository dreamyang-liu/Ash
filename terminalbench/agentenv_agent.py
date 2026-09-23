"""Ash actor and per-tool checkpoints on a Harbor-owned AgentENV sandbox."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import math
import threading

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext

from harness.atif import export_file
from harness.checkpointing import SnapshotBridge
from harness.core.journal import read_journal
from harness.execution.pipeline import Rewrite, ToolInterceptor, ToolPipeline
from harness.execution.server import ToolBoundary
from harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec
from harness.slots.claude_code import ClaudeCodeSlot
from terminalbench.agentenv import AgentENVEnvironment, shell_command


class ShellContext(ToolInterceptor):
    name = "harbor-execution-context"
    fail_mode = "closed"

    def __init__(self, environment: AgentENVEnvironment):
        self.cwd = environment.task_env_config.workdir or environment.image_workdir
        self.env = {**environment.image_env, **(environment._merge_env(None) or {})}
        self.user = environment._resolve_user(None)
        if self.user is None:
            self.user = environment.image_user

    def before(self, context):
        arguments = dict(context.args)
        arguments["command"] = shell_command(
            arguments["command"], arguments.get("working_dir") or self.cwd, self.env, self.user)
        arguments["working_dir"] = "/"
        return Rewrite(arguments)


class AgentENVOrchestrator(Orchestrator):
    def __init__(self, environment: AgentENVEnvironment, model_env: dict, **kwargs):
        super().__init__(**kwargs)
        self.environment = environment
        self.model_env = model_env
        self.policy = ShellContext(environment)
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()
        if self.environment.actor_control is not None:
            self.environment.actor_control.request_stop("Harbor cancelled the actor")

    def _wire_sandbox(self, spec, claim):
        owned = OwnedSandbox(session=self.environment.session, keep=True,
                             sandbox_id=self.environment.session.sandbox_id)
        try:
            owned.mcp = self._serve_in_process(spec, owned)
            owned.server.pipeline = ToolPipeline([*owned.server.pipeline.interceptors, self.policy])
            return owned, owned.mcp
        except BaseException:
            owned.release()
            raise

    def _wire_gateway(self, spec, journal, task, run_id):
        task.env.update(self.model_env)
        self.environment.actor_control = task.control
        if self.cancelled.is_set():
            self.cancel()
        return super()._wire_gateway(spec, journal, task, run_id)

    def _wire_checkpoints(self, spec, journal, owned=None):
        bridge = SnapshotBridge.install(
            journal, self.environment.session, tracker=owned.tracker, exact_mode=True,
            disk_only=self.environment.checkpoint_mode == "disk_only")
        owned.server.boundary = ToolBoundary(
            bridge.on_tool_boundary, validate_call=bridge.validate_call,
            on_unavailable=bridge.record_unavailable)
        return bridge


class AgentENVClaudeCode(BaseAgent):
    @staticmethod
    def name() -> str:
        return "ash-agentenv-claude-code"

    def version(self) -> str | None:
        return ClaudeCodeSlot().version()

    async def setup(self, environment: AgentENVEnvironment) -> None:
        if not isinstance(environment, AgentENVEnvironment) or environment.session is None:
            raise TypeError("AgentENVClaudeCode requires a started AgentENVEnvironment")
        if self.mcp_servers:
            raise ValueError("Task MCP service networking is not implemented for AgentENV")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _make_spec(self, prompt: str, workspace, journal, environment: AgentENVEnvironment) -> RunSpec:
        return RunSpec(
            prompt=prompt, slot="claude-code", model=(self.model_name or "").removeprefix("anthropic/"),
            cwd=str(workspace.resolve()), run_id=self.logs_dir.parent.name, journal_path=journal,
            timeout_s=math.inf, session=environment.session, keep_sandbox=True,
            transport="http", tools="shell_only", backend=environment.backend,
            runtime_bin=environment.runtime_bin, sandbox_image=environment.image,
            extra={"setting_sources": []},
        )

    async def run(self, instruction: str, environment: AgentENVEnvironment, context: AgentContext) -> None:
        journal = self.logs_dir / "trajectory.jsonl"
        if journal.exists():
            raise FileExistsError(f"Refusing to overwrite {journal}")
        workspace = self.logs_dir / "actor-workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        prompt = instruction + "\n\n" + (
            "Use the MCP shell tool for reading, editing and executing commands inside the task environment. "
            "Each call starts a fresh process; set working_dir when needed, timeout in seconds for long commands, "
            "and tail to limit output. Your built-in host tools are disabled. "
            f"The initial working directory is {environment.task_env_config.workdir or environment.image_workdir}."
        )
        if self.skills_dir:
            prompt += f"\nTask skills are available inside the sandbox at {self.skills_dir}."
        spec = self._make_spec(prompt, workspace, journal, environment)
        orchestrator = AgentENVOrchestrator(environment, self.extra_env, out_dir=self.logs_dir)
        previous_listeners = list(environment.session.on_swap)
        pending = asyncio.create_task(asyncio.to_thread(orchestrator.run, spec))
        context.metadata = {"environment_owner": "harbor", "backend": "agentenv",
                            "checkpoint_mode": environment.checkpoint_mode, "journal": str(journal)}
        try:
            try:
                result = await asyncio.shield(pending)
            except asyncio.CancelledError:
                orchestrator.cancel()
                await asyncio.shield(pending)
                raise
            context.n_input_tokens = result.usage.get("input_tokens")
            context.n_cache_tokens = result.usage.get("cached_input_tokens")
            context.n_output_tokens = result.usage.get("output_tokens")
            context.cost_usd = result.usage.get("cost_usd")
            context.metadata.update(actor_status=result.status, checkpoints=result.checkpoints,
                                    native_session_id=result.native_session_id)
            (self.logs_dir / "execution.json").write_text(json.dumps(asdict(result), default=str, indent=2))
            if result.status != "completed":
                raise RuntimeError(result.error or f"Actor {result.status}")
            records = read_journal(journal)
            native_errors = [row for row in records if row["type"] == "run.result"
                             and (row.get("native") or {}).get("is_error")]
            if native_errors:
                raise RuntimeError(f"Native actor error: {native_errors[-1].get('native')}")
            sandbox_calls = {row.get("call_id") for row in records if row["type"] == "tool.started"
                             and str(row.get("name", "")).startswith("mcp__ash__")}
            completed_calls = {row.get("call_id") for row in records if row["type"] == "tool.finished"
                               and row.get("status") == "ok"} & sandbox_calls
            paired_calls = {row.get("call_id") for row in records if row["type"] == "checkpoint.captured"
                            and row.get("snapshot_id")}
            if completed_calls - paired_calls:
                raise RuntimeError(f"Executed tool calls lack snapshots: {sorted(completed_calls - paired_calls)}")
            snapshot = await asyncio.to_thread(
                environment.session.snapshot, disk_only=environment.checkpoint_mode == "disk_only")
            if snapshot is None:
                raise RuntimeError("Final actor snapshot failed")
            context.metadata["final_snapshot_id"] = snapshot.id
            (self.logs_dir / "snapshot.json").write_text(json.dumps(context.metadata, indent=2))
        finally:
            environment.actor_control = None
            environment.session.on_swap[:] = previous_listeners
            if journal.exists():
                (self.logs_dir / "trajectory.json").write_text(json.dumps(export_file(journal), indent=2))
