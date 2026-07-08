"""Claude Code harness — uses Claude Code SDK with MCP sandbox tools."""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .base import BaseHarness
from ..dataset import resolve_image, format_task_prompt
from .. import style as S


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
    """Close a query() async stream if it supports aclose().

    Idempotent: safe to call on a None, already-exhausted, or already-closed
    stream. Used to deterministically tear the MCP server down on every exit path.
    """
    if stream is None:
        return
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as e:
        sys.stderr.write(f"[harness] query stream aclose error: {e}\n")


class ClaudeCodeHarness(BaseHarness):
    """Runs Claude Code SDK with ash sandbox exposed via MCP."""

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        return asyncio.run(self._run_instance_async(instance, output_dir))

    async def _run_instance_async(self, instance: dict, output_dir: Path) -> dict:
        # Prefer the maintained `claude-agent-sdk` (handles newer CLI stream events
        # like `rate_limit_event`); fall back to the legacy `claude-code-sdk`.
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions as Options
            from claude_agent_sdk.types import (
                AssistantMessage, UserMessage, ResultMessage,
                TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock,
            )
        except ImportError:
            from claude_code_sdk import query, ClaudeCodeOptions as Options
            from claude_code_sdk.types import (
                AssistantMessage, UserMessage, ResultMessage,
                TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock,
            )

        c = self.config
        instance_id = instance["instance_id"]
        image = resolve_image(instance)
        model = c.get("model", "opus")
        timeout = c.get("timeout", 1800)

        print(S.header(instance_id))
        print(S.kv("image   ", S.dim(image)))
        print(S.kv("model   ", S.dim(model)))

        # Temp dir for patches (MCP server writes per-sandbox patches here)
        patch_dir = tempfile.mkdtemp(prefix=f"patch_{instance_id}_")

        # MCP server config — SDK accepts dict directly, no temp file needed
        mcp_args = ["-m", "swebench.mcp_server", "--image", image, "--patch-dir", patch_dir]
        if c.get("runtime_bin"):
            mcp_args.extend(["--runtime-bin", c["runtime_bin"]])

        mcp_servers = {
            "ash-sandbox": {
                "type": "stdio",
                "command": sys.executable,
                "args": mcp_args,
            }
        }

        # Environment variables for provider auth
        env = {}
        if c.get("env"):
            env.update(c["env"])
        provider = c.get("provider")
        if provider == "bedrock":
            env.setdefault("CLAUDE_CODE_USE_BEDROCK", "1")
        elif provider == "vertex":
            env.setdefault("CLAUDE_CODE_USE_VERTEX", "1")
        if c.get("api_base"):
            env["ANTHROPIC_BASE_URL"] = c["api_base"]
        if c.get("api_key"):
            env["ANTHROPIC_API_KEY"] = c["api_key"]

        # Build SDK options
        permission_mode = c.get("permission_mode", "bypassPermissions")
        options = Options(
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            mcp_servers=mcp_servers,
            permission_mode=permission_mode,
            allowed_tools=["mcp__ash-sandbox__shell", "mcp__ash-sandbox__text_editor",
                           "mcp__ash-sandbox__grep_files", "mcp__ash-sandbox__process"],
            disallowed_tools=["Bash", "Read", "Edit", "Write", "NotebookEdit"],
            max_turns=c.get("max_turns", 200),
            cwd=str(Path(__file__).parent.parent.parent),
            env=env,
        )

        task = format_task_prompt(instance)
        start_time = time.time()
        trajectory = []      # ordered events: text / thinking / tool_use / tool_result
        messages = []        # assistant prose only (convenience view, untruncated)
        result_msg = None
        step_n = 0
        _stream = None

        try:
            _stream = query(prompt=task, options=options)
            # Wall-clock deadline: a hung model or stuck tool can't block the
            # worker indefinitely. Raises TimeoutError (caught below) on expiry.
            async with asyncio.timeout(timeout):
                async for message in _stream:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                step_n += 1
                                tool_name = block.name.replace("mcp__ash-sandbox__", "")
                                print(S.step(step_n, tool_name, str(block.input)[:40]), flush=True)
                                trajectory.append({
                                    "type": "tool_use", "step": step_n,
                                    "id": block.id, "name": tool_name, "input": block.input,
                                })
                            elif isinstance(block, TextBlock) and block.text.strip():
                                messages.append(block.text)
                                trajectory.append({"type": "text", "text": block.text})
                            elif isinstance(block, ThinkingBlock):
                                thinking = (getattr(block, "thinking", "") or "").strip()
                                if thinking:
                                    trajectory.append({"type": "thinking", "text": thinking})
                    elif isinstance(message, UserMessage):
                        # Tool results (observations) stream back as UserMessage content.
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

            # Close the query stream so the `claude` CLI (and the MCP server it
            # spawned) tears down and flushes the patch file before we read it.
            # Also closed in `finally` so error/timeout paths shut the MCP server
            # down before the patch dir is removed; aclose() is idempotent.
            await _aclose_stream(_stream)

            elapsed = time.time() - start_time

            # The patch lands asynchronously as the MCP subprocess exits. Poll for
            # it instead of reading once (which races the subprocess teardown).
            # Note: an empty diff still writes the file, so this breaks promptly
            # once shutdown completes; it only burns the full timeout on failure.
            patch = ""
            patch_written = False
            patch_path = Path(patch_dir) / "sb-1.diff"
            for _ in range(150):  # up to ~15s
                if patch_path.exists():
                    patch = patch_path.read_text()
                    patch_written = True
                    break
                await asyncio.sleep(0.1)

            # Warn on an empty patch, distinguishing extraction failure (file never
            # appeared → MCP server likely crashed before its shutdown extraction)
            # from a genuine no-op (agent produced no diff).
            if not patch_written:
                print(S.kv("warn    ", S.bright_red(
                    "patch file never written — MCP extraction did not run (server crash?)")),
                    flush=True)
            elif not patch.strip():
                print(S.kv("warn    ", S.bright_red(
                    "empty patch — agent produced no diff")), flush=True)

            # Determine status
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

            # Save trajectory
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
                "trajectory": trajectory,   # full ordered events (text/thinking/tool_use/tool_result)
                "messages": messages,       # assistant prose only, untruncated
                "usage": result_msg.usage if result_msg else None,
            }, indent=2, default=str))

            return {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": f"claude-code/{model}",
                "exit_status": exit_status,
            }

        except asyncio.TimeoutError:
            print(S.kv("error   ", S.bright_red(f"timeout ({timeout}s)")))
            return self._fail(instance_id, model, "timeout")
        except Exception as e:
            print(S.kv("error   ", S.bright_red(str(e))))
            return self._fail(instance_id, model, f"error: {e}")
        finally:
            # Close the stream (idempotent) before removing the patch dir, so the
            # MCP server has been told to shut down and isn't still writing into it.
            await _aclose_stream(_stream)
            import shutil
            shutil.rmtree(patch_dir, ignore_errors=True)

    def _fail(self, instance_id: str, model: str, status: str) -> dict:
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": f"claude-code/{model}",
            "exit_status": status,
        }
