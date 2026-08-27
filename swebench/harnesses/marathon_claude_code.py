"""SWE-Marathon tasks driven by Claude Code — the marathon topology, a
different agent loop.

`marathon.py` runs this repository's own agent (AshAgent) against a task;
this harness runs the Claude Code CLI instead, for scaffold comparisons:
same task, same environment, same verifier, different agent. What stays
identical is everything that makes a marathon run a marathon run — the task
directory is loaded by ``swebench.marathon.load_task``, the environment is
built locally by ``build_image`` (encrypted verification assets, no public
image), the sandbox gets the task's own resources, and the grade is the
task's own ``tests/test.sh`` run in the same sandbox afterwards.

How the sandbox reaches Claude Code: an *in-process* MCP server
(``claude_agent_sdk.create_sdk_mcp_server``), not the ``swebench.mcp_server``
subprocess the SWE-bench claude-code harness spawns. The subprocess owns its
sandbox and destroys it when the stream closes — which is exactly when
grading needs it. In-process, the harness owns the ``AshSession``: the four
exec tools are thin async wrappers over ``session.executor_for("agent")``,
and after the query stream ends the verifier runs on the same session. It
also means tool calls go through the L2 executor seam like every other
agent's, so traces and the tool registry behave normally.

Per-step checkpoints work here too, differently triggered. On AshAgent they
fire from a ``before_query`` hook; Claude Code has no such hook, but an
external agent's only channel into the environment is its tool calls, so the
tool boundary *is* its step boundary: each call runs ``Checkpointer.
after_step`` after the executor returns. The same ``MutationTracker`` sits
outermost on the pipeline, so read-only calls map to the previous snapshot
for free, and compaction re-boarding comes along unchanged. The step→snapshot
map is keyed by tool-call index rather than model-call index — resume with
``--resume-from <snapshot>`` restores the environment, not the conversation
(Claude Code owns its transcript, so a with-memory resume is not this
harness's to offer).

What this path deliberately does NOT have: the context-window guard. Claude
Code manages its own context (auto-compaction);
``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` in the config's ``env`` block is how a
non-default window is stated.
"""

import asyncio
import copy
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from .base import BaseHarness
from .claude_code import _aclose_stream
from .. import style as S
from ..backends import backend_config
from ..marathon import MarathonError, MarathonTask, build_image, grade, load_task
from ..prediction import failure
from ..sandbox import AshSession

#: The four exec tools, single-sandbox shape (no ``sandbox_id`` routing).
#: Imported from the MCP proxy so this harness and the external-agent path
#: describe the same tools with the same schemas — a drift here would mean
#: two Claude Code entry points disagreeing about what ``text_editor`` takes.
from ..mcp_server import EXEC_TOOLS_SINGLE


_SYSTEM_PROMPT = """\
You are an expert software engineer working on a single long-horizon task
inside an isolated sandbox.

## Environment

- Your working directory is {workdir}. The task's files and your deliverables
  live there unless the instructions say otherwise.
- You have full root access inside the sandbox.{network_note}
- Your ONLY tools are the 4 MCP tools from the ash-sandbox server: shell,
  text_editor, grep_files, process.
- Do NOT use any built-in tools (Bash, Read, Edit, Write, etc.) — they run on
  the wrong machine and their results do not exist in the sandbox.

## First: load your tools

Your sandbox tools are presented as *deferred* — they are NOT callable until loaded.
Your VERY FIRST action must be a single ToolSearch call that loads all four at once:

  ToolSearch({{"query": "select:mcp__ash-sandbox__shell,mcp__ash-sandbox__text_editor,mcp__ash-sandbox__grep_files,mcp__ash-sandbox__process"}})

Call ToolSearch EXACTLY ONCE. Do not search again, do not load tools one at a time,
and do not attempt a sandbox tool before this call succeeds.

## Tools

| Tool | Use for |
|------|---------|
| `grep_files` | Search code: patterns, symbols, definitions |
| `text_editor` | View/edit/create files (view, str_replace, insert, write) |
| `shell` | Run builds, tests, git, any command |
| `process` | Read output from / kill background processes |

## Ways of working

- The task instructions are the specification. Your work is graded afterwards
  by a hidden verifier: deliverables must be exactly where and what the
  instructions say, and only what is on disk counts.
- This task is hours long. Work incrementally: get something building early,
  run whatever visible tests exist often, and commit progress in working
  states rather than attempting one big-bang implementation.
- Long-running commands (builds, test suites): run with `background: true`
  and poll with `process` instead of blocking with a huge timeout.
- Always bound command output (`tail`) — full build logs drown the context.
"""


class MarathonClaudeCodeHarness(BaseHarness):
    """One marathon task, Claude Code as the agent, graded by the task's verifier."""

    def run_instance(self, instance: dict, output_dir: Path,
                     quiet: bool = False) -> dict:
        c = self.config
        task_dir = instance.get("task_dir") or c.get("task_dir")
        if not task_dir:
            return self._failure(instance, "no task_dir given")
        try:
            task = load_task(task_dir)
        except MarathonError as exc:
            return self._failure(instance, f"error: {exc}")

        if not quiet:
            print(S.header(task.instance_id))
            print(S.kv("task    ", S.cyan(task.name)))
            print(S.kv("agent   ", S.dim("claude-code (in-process MCP)")))

        resume_from = c.get("resume_from")
        if resume_from:
            image = resume_from
            if not quiet:
                print(S.kv("resume  ", S.cyan(str(resume_from)[:13] + "…")))
        else:
            try:
                image = build_image(
                    task, registry=c.get("registry", "localhost:5000"))
            except MarathonError as exc:
                return self._failure(instance, f"error: {exc}")

        session = AshSession(runtime_bin=c.get("runtime_bin"), quiet=quiet,
                             backend=backend_config(c))
        try:
            # The task declares its shape; the backend default (2 CPUs, 1 GB)
            # OOMs a build-heavy task rather than failing loudly.
            resources = {"cpu": task.cpus, "memory_mb": task.memory_mb}
            if not session.create(image, resources=resources):
                return self._failure(
                    instance, f"error: sandbox creation failed for {image}")

            checkpointer = None
            #: Live view of the run's event stream, shared with the
            #: checkpointer's persist so an interrupted run leaves its
            #: transcript beside the step map — snapshots without either are
            #: unusable, and branch analysis (swebench/branching.py) needs
            #: the events to decide where to branch a killed run from.
            events_view = {"events": None}
            checkpoint_cfg = c.get("checkpoints") or {}
            if checkpoint_cfg.get("enabled") and session.supports_snapshot():
                checkpointer = self._install_checkpoints(
                    task, session, output_dir, checkpoint_cfg,
                    events_view=events_view)
                # The pristine environment, before step 1 — what a replay of
                # the first step needs (install()'s first firing does the
                # same for AshAgent).
                checkpointer.after_step(0)

            run = asyncio.run(self._drive(task, session, quiet, checkpointer,
                                          output_dir=output_dir,
                                          events_view=events_view))

            # Grade on the SAME session, whatever the loop's exit was: a
            # deadline or an errored stream still leaves hours of work on
            # disk, and the verifier is the only thing that can say what it
            # is worth. Only a sandbox that never existed skips this.
            result = grade(session, task)
            if not quiet:
                print(S.kv("reward  ", S.cost(result.reward, 0)
                           if result.reward else S.yellow("0.0")))
                print(S.kv("partial ", S.dim(f"{result.partial_score}")))

            self._save_trajectory(task, run, result, session, output_dir,
                                  checkpointer=checkpointer)

            # Marathon has no patch to submit — the graded artifact is the
            # environment itself — so the report is a `failure` in the
            # builder's vocabulary carrying the grade alongside, exactly as
            # the marathon harness reports (see its closing comment).
            report = failure(task.instance_id,
                             f"claude-code/{run['model']}", run["exit_status"])
            report.update({
                "reward": result.reward,
                "partial_score": result.partial_score,
                "cost": run.get("cost_usd") or 0.0,
                "turns": run.get("num_turns") or 0,
                "metrics": result.metrics,
                "grading_error": result.error,
            })
            return report
        finally:
            session.destroy()

    def _install_checkpoints(self, task: MarathonTask, session: AshSession,
                             output_dir: Path, cfg: dict,
                             events_view: "dict | None" = None):
        """A Checkpointer for this episode, persisting after every record.

        Not ``checkpoints.install`` — that wires a ``before_query`` hook into
        an AshAgent, and this path has none. The tracker/checkpointer pair is
        the same; only the trigger differs (the tool boundary, see
        ``_sandbox_tools``). Persisting writes the step→snapshot map in the
        trajectory's shape so ``replay.load_step_snapshots`` and
        ``--resume-from`` read an interrupted run's leavings unchanged.
        """
        from ..agent.checkpoints import Checkpointer, MutationTracker
        from ..agent.trace import new_run_id

        trajectory_path = (output_dir / "trajectories"
                           / f"{task.instance_id}.json")

        def persist(checkpointer_ref):
            # The map beside the snapshots, every capture — a killed run's
            # snapshots are unusable without it (see checkpoints.py). The
            # event stream comes along for the same reason: branch analysis
            # of an interrupted run needs the transcript, not just the map.
            events = (events_view or {}).get("events")
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(json.dumps({
                "instance_id": task.instance_id,
                # A copy, not the live list: the SDK loop appends while this
                # thread serializes.
                "trajectory": list(events) if events is not None else [],
                "info": {
                    "exit_status": "in_progress",
                    "checkpoints": checkpointer_ref.as_info(),
                    # The sandbox the run is on NOW: re-boarding changes it,
                    # and cleanup that trusts a launch-time id kills live runs.
                    "environment": session.environment(),
                },
            }, indent=2, default=str))

        return Checkpointer(
            session=session,
            tracker=MutationTracker(),
            always=cfg.get("trigger", "mutation") == "every_step",
            disk_only=cfg.get("mode", "disk_only") != "full",
            reboard=cfg.get("reboard", True),
            # The run id keeps two attempts at one task from colliding on
            # snapshot aliases (unique per repository; a rerun would fail
            # every capture, softly).
            name_prefix=(f"marathon-cc-{task.instance_id}-"
                         f"{new_run_id()[:8]}-"),
            persist=persist,
        )

    # --- the agent loop -------------------------------------------------- #

    async def _drive(self, task: MarathonTask, session: AshSession,
                     quiet: bool, checkpointer=None,
                     output_dir: "Path | None" = None,
                     events_view: "dict | None" = None) -> dict:
        """Run one Claude Code query over the in-process MCP server."""
        from claude_agent_sdk import (ClaudeAgentOptions, create_sdk_mcp_server,
                                      query, tool)
        from claude_agent_sdk.types import (AssistantMessage, ResultMessage,
                                            TextBlock, ThinkingBlock,
                                            ToolResultBlock, ToolUseBlock,
                                            UserMessage)

        c = self.config
        model = c.get("model", "us.anthropic.claude-sonnet-4-6")
        # The task's own wall clock is the benchmark's ceiling; the config may
        # only tighten it, because exceeding what every published trial was
        # allowed makes the score incomparable in the flattering direction.
        timeout = min(float(c.get("timeout") or task.agent_timeout_sec),
                      task.agent_timeout_sec)

        server = create_sdk_mcp_server(
            name="ash-sandbox", version="1.0.0",
            tools=self._sandbox_tools(tool, session, task,
                                      checkpointer=checkpointer))
        options = self._build_options(ClaudeAgentOptions, task, server,
                                      cwd=output_dir)
        options.model = model

        if not quiet:
            print(S.kv("model   ", S.dim(model)))
            print(S.kv("deadline", S.dim(f"{timeout:.0f}s")))

        prompt = task.instruction
        if c.get("resume_from"):
            # Environment-only resume: the artifacts survived but the
            # conversation did not (Claude Code owns its transcript), so the
            # prompt has to say so — same note as the marathon harness.
            prompt += (
                "\n\n---\n\nNOTE: this environment is resumed from an earlier "
                "session of this same task. Work already exists on disk -- "
                "inspect the current state first (the source files, whether it "
                "builds, what the visible tests say) and continue from there "
                "rather than starting over. You do not have the earlier "
                "conversation, only what is on disk.")
            if c.get("resume_hint"):
                # A branched rollout's direction (swebench/branching.py): an
                # analysis of the failed attempt distilled into marching
                # orders. Self-contained by construction, because the resumed
                # agent has no other memory of what went wrong.
                prompt += ("\n\nDIRECTION FOR THIS ATTEMPT (from an analysis "
                           "of the earlier session):\n" + str(c["resume_hint"]))

        start = time.time()
        trajectory: list[dict] = []
        if events_view is not None:
            events_view["events"] = trajectory
        messages: list[str] = []
        result_msg = None
        step_n = 0
        exit_status = "error: stream ended without a result"
        stream = None
        try:
            stream = query(prompt=prompt, options=options)
            async with asyncio.timeout(timeout):
                async for message in stream:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                step_n += 1
                                name = block.name.replace("mcp__ash-sandbox__", "")
                                if not quiet:
                                    print(S.step(step_n, name,
                                                 str(block.input)[:40]), flush=True)
                                trajectory.append({
                                    "type": "tool_use", "step": step_n,
                                    "id": block.id, "name": name,
                                    "input": block.input})
                            elif isinstance(block, TextBlock) and block.text.strip():
                                messages.append(block.text)
                                trajectory.append({"type": "text",
                                                   "text": block.text})
                            elif isinstance(block, ThinkingBlock):
                                thinking = (getattr(block, "thinking", "") or "").strip()
                                if thinking:
                                    trajectory.append({"type": "thinking",
                                                       "text": thinking})
                    elif isinstance(message, UserMessage):
                        content = message.content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, ToolResultBlock):
                                    trajectory.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.tool_use_id,
                                        "content": block.content,
                                        "is_error": bool(block.is_error)})
                    elif isinstance(message, ResultMessage):
                        result_msg = message
            if result_msg is not None:
                exit_status = (f"error: {result_msg.result or 'unknown'}"
                               if result_msg.is_error else "completed")
        except asyncio.TimeoutError:
            # Not a failure of the run: the environment keeps the work and
            # grading still happens. Recorded so a deadline'd attempt is
            # distinguishable from one the agent chose to finish.
            exit_status = f"deadline: {timeout:.0f}s"
            if not quiet:
                print(S.kv("deadline", S.yellow(f"reached after {timeout:.0f}s")))
        except Exception as e:  # noqa: BLE001 - grade whatever the loop left
            exit_status = f"error: {e}"
            if not quiet:
                print(S.kv("error   ", S.bright_red(str(e))))
        finally:
            await _aclose_stream(stream)

        return {
            "model": model,
            "exit_status": exit_status,
            "elapsed_seconds": time.time() - start,
            "cost_usd": result_msg.total_cost_usd if result_msg else None,
            "num_turns": result_msg.num_turns if result_msg else step_n,
            "num_steps": step_n,
            "trajectory": trajectory,
            "messages": messages,
            "usage": result_msg.usage if result_msg else None,
        }

    # --- wiring ----------------------------------------------------------- #

    def _sandbox_tools(self, tool, session: AshSession, task: MarathonTask,
                       checkpointer=None) -> list:
        """The four exec tools, bound to this session's agent executor.

        Schemas come from the MCP proxy (single-sandbox shape) so both Claude
        Code entry points describe identical tools. The executor is the same
        L2 seam every agent gets (`executor_for`), serialized with a lock:
        `AshSession` drives a private event loop via `run_until_complete`,
        which two concurrent tool calls must not enter at once.

        Two of the three default interceptors are mounted (list order is
        outermost-first; the presenter is innermost so truncation bounds what
        it rendered): `OutcomePresenter` because exit codes and a separated
        stderr should be stated to this agent like any other, and
        `TruncateInterceptor` because a 10-hour task's build logs otherwise
        spend the window. The guardrail is left off deliberately — its
        read-before-edit and edit-streak nudges shape *this repo's* agent,
        and Claude Code brings its own editing discipline; nudging it too
        would measure a subtly different scaffold.

        With a checkpointer, its `MutationTracker` mounts outermost (where it
        also sees calls the inner interceptors reject) and `after_step` runs
        at the tool boundary, inside the lock: the executor's own
        `run_until_complete` has returned by then, so the session's loop is
        free for the capture, and serializing capture with calls keeps the
        step→snapshot map ordered even when Claude Code issues tool calls
        concurrently. Clean (read-only) calls record a mapping without a
        capture, so the granularity costs nothing extra.
        """
        from ..agent.interceptors import OutcomePresenter, TruncateInterceptor
        from ..agent.pipeline import ToolPipeline

        chain = [
            TruncateInterceptor(max_len=int(self.config.get(
                "max_output_bytes") or 12000)),
            OutcomePresenter(),
        ]
        if checkpointer is not None and checkpointer.tracker is not None:
            chain.insert(0, checkpointer.tracker)
        executor = session.executor_for("agent", pipeline=ToolPipeline(chain))
        lock = threading.Lock()
        workdir = task.workdir or "/app"
        step_counter = {"n": 0}

        def _call(name: str, args: dict) -> dict:
            args = dict(args)
            if name == "shell" and "working_dir" not in args:
                args["working_dir"] = workdir
            # The executor seam is (tool_name, args) — nothing else. A model
            # -supplied timeout travels in args, where the runtime reads it;
            # passing it positionally broke against the piped executor, whose
            # contract has no third argument (every call TypeErrored, and the
            # agent spent its run diagnosing our dispatcher).
            with lock:
                result = executor(name, args)
                if checkpointer is not None:
                    # The tool boundary is this agent's step boundary: the
                    # environment as it stands after call n is what a replay
                    # of call n+1 needs.
                    step_counter["n"] += 1
                    checkpointer.after_step(step_counter["n"])
            text = result.output if (result.success or result.output) \
                else f"Error: {result.error or 'unknown error'}"
            return {"content": [{"type": "text", "text": text}],
                    "is_error": not result.success}

        tools = []
        for spec in copy.deepcopy(EXEC_TOOLS_SINGLE):
            name = spec["name"]
            description = spec["description"].replace("/testbed", workdir)

            async def handler(args: dict, _name=name) -> dict:
                # A worker thread, because the executor blocks on the
                # session's own loop; blocking the SDK's loop would freeze
                # every other stream event.
                return await asyncio.to_thread(_call, _name, args)

            tools.append(tool(name, description, spec["inputSchema"])(handler))
        return tools

    def _build_options(self, options_cls, task: MarathonTask, server,
                       cwd: "Path | None" = None) -> "object":
        """SDK options for one task. Factored out so a test can assert the
        wiring — restricted tasks losing WebSearch/WebFetch, the workdir
        reaching the prompt — without a CLI or a sandbox.

        ``cwd`` must be a NEUTRAL directory (the run's output dir), never
        this repository: Claude Code reads ``.claude/`` from its cwd, and
        launching from the repo root handed the agent the repo's own `ash`
        skill mid-task — contamination the SWE-bench harness never saw only
        because its MCP subprocess needed the repo importable and nothing
        else looked at cwd."""
        c = self.config
        env = dict(c.get("env") or {})
        if c.get("provider") == "bedrock":
            env.setdefault("CLAUDE_CODE_USE_BEDROCK", "1")
        if c.get("api_base"):
            env["ANTHROPIC_BASE_URL"] = c["api_base"]
        if c.get("api_key"):
            env["ANTHROPIC_API_KEY"] = c["api_key"]

        # Provider-served web tools reach the network from the provider's
        # side, past whatever the sandbox denies. 16 of the 20 tasks restrict
        # the network, and answering one with WebSearch measures an easier
        # benchmark (see marathon.py's `_internet_restricted`).
        if cwd is not None:
            Path(cwd).mkdir(parents=True, exist_ok=True)

        disallowed = ["Bash", "Read", "Edit", "Write", "NotebookEdit"]
        network_note = ""
        if task.internet_restricted:
            disallowed += ["WebSearch", "WebFetch"]
            network_note = ("\n- Network access is restricted to the task's "
                            "declared allowlist (package registries); assume "
                            "everything else is unreachable.")

        return options_cls(
            system_prompt=_SYSTEM_PROMPT.format(
                workdir=task.workdir or "/app", network_note=network_note),
            mcp_servers={"ash-sandbox": server},
            permission_mode=c.get("permission_mode", "bypassPermissions"),
            allowed_tools=["mcp__ash-sandbox__shell",
                           "mcp__ash-sandbox__text_editor",
                           "mcp__ash-sandbox__grep_files",
                           "mcp__ash-sandbox__process"],
            disallowed_tools=disallowed,
            # Marathon horizons: the benchmark's own step ceiling, not the
            # SWE-bench-shaped 200 (see marathon.py's budget note).
            max_turns=int(c.get("max_turns") or c.get("step_limit") or 2000),
            cwd=str(cwd) if cwd is not None else tempfile.mkdtemp(
                prefix="marathon-cc-cwd-"),
            env=env,
        )

    # --- reporting --------------------------------------------------------- #

    def _save_trajectory(self, task: MarathonTask, run: dict, result,
                         session: AshSession, output_dir: Path,
                         checkpointer=None) -> None:
        traj_dir = output_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / f"{task.instance_id}.json").write_text(json.dumps({
            "instance_id": task.instance_id,
            **run,
            "info": {
                "exit_status": run["exit_status"],
                "environment": session.environment(),
                **({"checkpoints": checkpointer.as_info()}
                   if checkpointer is not None else {}),
                "marathon": {
                    "task": task.name,
                    "reward": result.reward,
                    "partial_score": result.partial_score,
                    "metrics": result.metrics,
                    "grading_error": result.error,
                    "expert_time_estimate_hours":
                        task.metadata.get("expert_time_estimate_hours"),
                },
            },
        }, indent=2, default=str))

    def _failure(self, instance: dict, status: str) -> dict:
        report = failure(instance.get("instance_id") or "unknown",
                         f"claude-code/{self.config.get('model')}", status)
        report.update({"reward": 0.0, "partial_score": 0.0, "cost": 0.0,
                       "turns": 0, "metrics": {}, "grading_error": status})
        return report
