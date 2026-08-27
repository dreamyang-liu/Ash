"""Claude Code harness — the SWE-bench topology with Claude Code as the agent.

**This harness owns its sandbox.** It creates an ``AshSession``, exposes the exec
tools to Claude Code through an *in-process* MCP server, and extracts the patch
itself when the stream ends.

It used to spawn ``python -m swebench.mcp_server --image X --patch-dir Y`` as a
stdio subprocess, which inverted ownership: the *server* created the sandbox and
destroyed it when the stream closed -- exactly when grading needs it. Everything
downstream of that was compensation. The patch was written asynchronously during
the subprocess's teardown, so the harness polled a file for up to 15 seconds and
could not tell "the agent produced no diff" from "the server crashed before
extracting". And because the session lived in another process, nothing here could
snapshot it, so checkpoints and post-hoc extraction were unavailable on this path.

Owning the session removes all of that: the patch is a direct call, and the same
handle can be snapshotted or re-read afterwards. ``marathon-claude-code`` reached
the same conclusion for the same reason and is the pattern followed here.

What stays deliberately unchanged: the system prompt (a scaffold change would make
scores incomparable), the deferred-tool ToolSearch instruction, and the denial of
Claude Code's built-in tools -- they run on the host, not in the sandbox.
"""

import asyncio
import copy
import json
import sys
import threading
import time
from pathlib import Path

from .. import style as S
from ..backends import backend_config
from ..dataset import format_task_prompt, resolve_image
from ..prediction import failure, prediction
from ..sandbox import AshSession
from .base import BaseHarness

#: Tool schemas, imported rather than restated so this entry point and the MCP
#: proxy cannot drift.
from ..mcp_server import EXEC_TOOLS_SINGLE

_TOOL_NAMES = [spec["name"] for spec in EXEC_TOOLS_SINGLE]
_MCP_PREFIX = "mcp__ash-sandbox__"


_SYSTEM_PROMPT = """\
You are an expert software engineer solving a GitHub issue inside an isolated Docker container.

## Environment

- You are working in /testbed which contains the repository at the relevant commit.
- You have full root access. All dependencies are pre-installed. No internet access.
- Your ONLY tools are the 4 MCP tools from the ash-sandbox server: shell, text_editor, grep_files, process.
- Do NOT use any built-in tools (Bash, Read, Edit, Write, etc.) — they won't work in the sandbox.

## First: load your tools

Your sandbox tools are presented as *deferred* — they are NOT callable until loaded.
Your VERY FIRST action must be a single ToolSearch call that loads all four at once:

  ToolSearch({"query": "select:mcp__ash-sandbox__shell,mcp__ash-sandbox__text_editor,mcp__ash-sandbox__grep_files,mcp__ash-sandbox__process"})

Call ToolSearch EXACTLY ONCE. Do not search again, do not load tools one at a time,
and do not attempt a sandbox tool before this call succeeds.

## Tools

| Tool | Use for |
|------|---------|
| `grep_files` | Search code: patterns, symbols, definitions |
| `text_editor` | View/edit/create files (view, str_replace, insert, write) |
| `shell` | Run tests, pip install, git, any command |
| `process` | Read output from / kill background processes |

## Workflow

1. **Understand**: Read the issue. Extract what's broken vs expected.
2. **Locate**: Use grep_files to find relevant code AND the test file.
3. **Analyze**: Read code. State root cause before editing.
4. **Fix**: Make minimal, targeted edits. Fix ALL instances of the bug pattern.
5. **Verify**: Run the specific test AND the broader test module.
6. **Clean up**: Remove any temp files. Only source changes should remain.

## Rules

- Fix ALL instances of the bug pattern (grep for parallel locations).
- Never modify test files.
- Always run tests with `tail` parameter to limit output.
- Read the failing test FIRST to understand expected behavior.
- Maximum 3 fix attempts. If all fail, stop.
- A partial fix is a failed fix — either all tests pass or revert.
- Keep changes minimal. Don't refactor unrelated code.
"""


async def _aclose_stream(stream) -> None:
    """Close a query() async stream if it supports aclose(). Idempotent."""
    if stream is None:
        return
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as e:  # noqa: BLE001 - teardown must not mask the result
        sys.stderr.write(f"[harness] query stream aclose error: {e}\n")


class ClaudeCodeHarness(BaseHarness):
    """Claude Code against an ash sandbox this harness owns."""

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        return asyncio.run(self._run_instance_async(instance, output_dir))

    async def _run_instance_async(self, instance: dict, output_dir: Path) -> dict:
        # Prefer the maintained `claude-agent-sdk` (handles newer CLI stream
        # events like `rate_limit_event`); fall back to the legacy package.
        try:
            from claude_agent_sdk import (ClaudeAgentOptions as Options,
                                          create_sdk_mcp_server, query, tool)
            from claude_agent_sdk.types import (AssistantMessage, ResultMessage,
                                                TextBlock, ThinkingBlock,
                                                ToolResultBlock, ToolUseBlock,
                                                UserMessage)
        except ImportError:
            from claude_code_sdk import (ClaudeCodeOptions as Options,
                                         create_sdk_mcp_server, query, tool)
            from claude_code_sdk.types import (AssistantMessage, ResultMessage,
                                              TextBlock, ThinkingBlock,
                                              ToolResultBlock, ToolUseBlock,
                                              UserMessage)

        c = self.config
        instance_id = instance["instance_id"]
        image = resolve_image(instance)
        model = c.get("model", "opus")
        timeout = c.get("timeout", 1800)

        print(S.header(instance_id))
        print(S.kv("image   ", S.dim(image)))
        print(S.kv("model   ", S.dim(model)))

        # backend_config(c) verbatim: a test asserts every harness passes it,
        # because forgetting it silently keeps the run on Docker while the config
        # says microvm.
        session = AshSession(runtime_bin=c.get("runtime_bin"),
                             backend=backend_config(c))
        start_time = time.time()
        trajectory = []      # ordered events: text / thinking / tool_use / tool_result
        messages = []        # assistant prose only (convenience view, untruncated)
        result_msg = None
        step_n = 0
        stream = None

        try:
            if not session.create(image):
                return self._fail(instance_id, model, "error: sandbox create failed")

            server = create_sdk_mcp_server(
                name="ash-sandbox", version="1.0.0",
                tools=self._sandbox_tools(tool, session))
            options = self._build_options(Options, model, server, output_dir)

            task = format_task_prompt(instance)
            stream = query(prompt=task, options=options)
            # Wall-clock deadline: a hung model or stuck tool cannot block the
            # worker indefinitely. Raises TimeoutError, caught below.
            async with asyncio.timeout(timeout):
                async for message in stream:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                step_n += 1
                                tool_name = block.name.replace(_MCP_PREFIX, "")
                                print(S.step(step_n, tool_name,
                                             str(block.input)[:40]), flush=True)
                                trajectory.append({
                                    "type": "tool_use", "step": step_n,
                                    "id": block.id, "name": tool_name,
                                    "input": block.input,
                                })
                            elif isinstance(block, TextBlock) and block.text.strip():
                                messages.append(block.text)
                                trajectory.append({"type": "text", "text": block.text})
                            elif isinstance(block, ThinkingBlock):
                                thinking = (getattr(block, "thinking", "") or "").strip()
                                if thinking:
                                    trajectory.append({"type": "thinking",
                                                       "text": thinking})
                    elif isinstance(message, UserMessage):
                        # Tool results stream back as a synthetic user turn.
                        content = message.content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, ToolResultBlock):
                                    trajectory.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.tool_use_id,
                                        "content": block.content,
                                        "is_error": bool(block.is_error),
                                    })
                    elif isinstance(message, ResultMessage):
                        result_msg = message

            await _aclose_stream(stream)
            elapsed = time.time() - start_time

            # A direct call now: the session is ours and still alive. This used to
            # poll a file the MCP subprocess wrote while shutting down, which meant
            # an extraction failure and an empty diff looked the same.
            patch = await asyncio.to_thread(session.get_patch)
            if not patch.strip():
                print(S.kv("warn    ", S.bright_red(
                    "empty patch — agent produced no diff")), flush=True)

            if result_msg and result_msg.is_error:
                exit_status = f"error: {result_msg.result or 'unknown'}"
            elif patch:
                exit_status = "completed"
            else:
                exit_status = "no_patch"

            cost = result_msg.total_cost_usd if result_msg else None
            num_turns = result_msg.num_turns if result_msg else step_n

            print(S.kv("time    ", S.dim(f"{elapsed:.1f}s")))
            if cost is not None:
                print(S.kv("cost    ", S.dim(f"${cost:.2f} ({num_turns} turns)")))
            print(S.kv("patch   ", S.patch_info(patch)))

            traj_dir = output_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            (traj_dir / f"{instance_id}.json").write_text(json.dumps({
                "instance_id": instance_id,
                "model": model,
                "exit_status": exit_status,
                "elapsed_seconds": elapsed,
                "cost_usd": cost,
                "num_turns": num_turns,
                "num_steps": step_n,
                "trajectory": trajectory,
                "messages": messages,
                "usage": result_msg.usage if result_msg else None,
                "environment": session.environment(),
            }, indent=2, default=str))

            return prediction(instance_id, f"claude-code/{model}", patch, exit_status)

        except asyncio.TimeoutError:
            print(S.kv("error   ", S.bright_red(f"timeout ({timeout}s)")))
            return self._fail(instance_id, model, "timeout")
        except Exception as e:  # noqa: BLE001 - one instance must not kill a batch
            print(S.kv("error   ", S.bright_red(str(e))))
            return self._fail(instance_id, model, f"error: {e}")
        finally:
            await _aclose_stream(stream)
            # After the patch, never before: destroying the sandbox is what the
            # old topology did too early.
            session.destroy()

    # --- wiring ------------------------------------------------------------
    def _sandbox_tools(self, tool, session):
        """The exec tools, as in-process MCP tools over this session.

        Schemas come from ``EXEC_TOOLS_SINGLE`` so the two entry points cannot
        drift. The executor is the L2 seam every agent gets, with truncate and the
        presenter mounted -- but no guardrail: its nudges were written to shape
        *this repo's* agent, and nudging Claude Code too would measure a subtly
        different scaffold than the published numbers.
        """
        from harness.execution.interceptors import (OutcomePresenter,
                                                    TruncateInterceptor)
        from harness.execution.pipeline import ToolPipeline

        max_output = int(self.config.get("max_output_bytes", 12000) or 12000)
        pipeline = ToolPipeline([TruncateInterceptor(max_len=max_output),
                                 OutcomePresenter()])
        executor = session.executor_for("agent", pipeline=pipeline)
        # The session drives its own event loop, so calls are serialized: two
        # concurrent tool calls would re-enter it.
        lock = threading.Lock()

        def call(name: str, args: dict) -> dict:
            # The executor seam is (tool_name, args) and nothing else -- a third
            # positional argument TypeErrored every call once, and the agent spent
            # its run diagnosing our dispatcher.
            with lock:
                result = executor(name, args)
            text = (result.output if (result.success or result.output)
                    else f"Error: {result.error or 'unknown error'}")
            return {"content": [{"type": "text", "text": text}],
                    "is_error": not result.success}

        tools = []
        for spec in copy.deepcopy(EXEC_TOOLS_SINGLE):
            name = spec["name"]

            async def handler(args: dict, _name=name) -> dict:
                # A worker thread: the executor blocks on the session's loop, and
                # blocking the SDK's loop would freeze every other stream event.
                return await asyncio.to_thread(call, _name, args)

            tools.append(tool(name, spec["description"], spec["inputSchema"])(handler))
        return tools

    def _build_options(self, options_cls, model: str, server, output_dir: Path):
        """SDK options. Factored out so a test can assert the wiring without a
        sandbox or a CLI."""
        c = self.config
        env = dict(c.get("env") or {})
        provider = c.get("provider")
        if provider == "bedrock":
            env.setdefault("CLAUDE_CODE_USE_BEDROCK", "1")
        elif provider == "vertex":
            env.setdefault("CLAUDE_CODE_USE_VERTEX", "1")
        if c.get("api_base"):
            env["ANTHROPIC_BASE_URL"] = c["api_base"]
        if c.get("api_key"):
            env["ANTHROPIC_API_KEY"] = c["api_key"]

        return options_cls(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            mcp_servers={"ash-sandbox": server},
            permission_mode=c.get("permission_mode", "bypassPermissions"),
            allowed_tools=[_MCP_PREFIX + name for name in _TOOL_NAMES],
            # These run on the host, not in the sandbox.
            disallowed_tools=["Bash", "Read", "Edit", "Write", "NotebookEdit"],
            max_turns=c.get("max_turns", 200),
            # A NEUTRAL cwd, never this repository: Claude Code reads `.claude/`
            # from its cwd, and launching from the repo root once handed the agent
            # this repo's own `ash` skill mid-task.
            cwd=str(output_dir),
            env=env,
        )

    def _fail(self, instance_id: str, model: str, status: str) -> dict:
        return failure(instance_id, f"claude-code/{model}", status)
