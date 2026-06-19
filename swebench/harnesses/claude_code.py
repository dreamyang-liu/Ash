"""Claude Code harness — uses Claude Code CLI with MCP sandbox tools."""

import json
import os
import subprocess
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
- Your ONLY tools are the 5 MCP tools from the ash-sandbox server: shell, text_editor, grep_files, read_file, process.
- Do NOT use any built-in tools (Bash, Read, Edit, Write, etc.) — they won't work in the sandbox.

## Tools

| Tool | Use for |
|------|---------|
| `grep_files` | Search code: patterns, symbols, definitions |
| `read_file` | Read file sections with line numbers |
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


class ClaudeCodeHarness(BaseHarness):
    """Runs Claude Code CLI with ash sandbox exposed via MCP."""

    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        c = self.config
        instance_id = instance["instance_id"]
        image = resolve_image(instance)
        model = c.get("model", "opus")
        max_budget = c.get("max_budget", 5.0)
        timeout = c.get("timeout", 1800)

        print(S.header(instance_id))
        print(S.kv("image   ", S.dim(image)))
        print(S.kv("model   ", S.dim(model)))

        # Temp file for patch (MCP server writes on shutdown)
        patch_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", prefix=f"patch_{instance_id}_", delete=False
        )
        patch_file.close()

        # MCP config
        mcp_args = ["-m", "swebench.mcp_server", "--image", image, "--patch-file", patch_file.name]
        if c.get("runtime_bin"):
            mcp_args.extend(["--runtime-bin", c["runtime_bin"]])

        mcp_config = {
            "mcpServers": {
                "ash-sandbox": {
                    "command": sys.executable,
                    "args": mcp_args,
                    "env": {},
                }
            }
        }

        mcp_config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="mcp_", delete=False
        )
        json.dump(mcp_config, mcp_config_file)
        mcp_config_file.close()

        # Task prompt
        task = format_task_prompt(instance)

        # Build claude command
        permission_mode = c.get("permission_mode", "bypassPermissions")
        cmd = [
            "claude",
            "--print",
            "--bare",
            "--model", model,
            "--system-prompt", _SYSTEM_PROMPT,
            "--mcp-config", mcp_config_file.name,
            "--strict-mcp-config",
            "--tools", "",
            "--permission-mode", permission_mode,
            "--output-format", "text",
            "--max-budget-usd", str(max_budget),
            "--no-session-persistence",
        ]

        # Environment
        proc_env = os.environ.copy()
        if c.get("env"):
            proc_env.update(c["env"])
        provider = c.get("provider")
        if provider == "bedrock":
            proc_env.setdefault("CLAUDE_CODE_USE_BEDROCK", "1")
        elif provider == "vertex":
            proc_env.setdefault("CLAUDE_CODE_USE_VERTEX", "1")
        if c.get("api_base"):
            proc_env["ANTHROPIC_BASE_URL"] = c["api_base"]
        if c.get("api_key"):
            proc_env["ANTHROPIC_API_KEY"] = c["api_key"]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                input=task,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).parent.parent.parent),
                env=proc_env,
            )

            elapsed = time.time() - start_time

            # Read patch
            patch = ""
            if Path(patch_file.name).exists():
                patch = Path(patch_file.name).read_text()

            exit_status = "completed" if patch else "no_patch"
            if result.returncode != 0 and not patch:
                exit_status = f"error: exit code {result.returncode}"

            print(S.kv("time    ", S.dim(f"{elapsed:.1f}s")))
            print(S.kv("patch   ", S.patch_info(patch)))

            # Save trajectory
            traj_dir = output_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            (traj_dir / f"{instance_id}.json").write_text(json.dumps({
                "instance_id": instance_id,
                "model": model,
                "output": (result.stdout or "")[-3000:],
                "stderr": (result.stderr or "")[-2000:],
                "elapsed_seconds": elapsed,
                "exit_status": exit_status,
            }, indent=2))

            return {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": f"claude-code/{model}",
                "exit_status": exit_status,
            }

        except subprocess.TimeoutExpired:
            print(S.kv("error   ", S.bright_red(f"timeout ({timeout}s)")))
            return self._fail(instance_id, model, "timeout")
        except Exception as e:
            print(S.kv("error   ", S.bright_red(str(e))))
            return self._fail(instance_id, model, f"error: {e}")
        finally:
            try:
                os.unlink(mcp_config_file.name)
                os.unlink(patch_file.name)
            except OSError:
                pass

    def _fail(self, instance_id: str, model: str, status: str) -> dict:
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": f"claude-code/{model}",
            "exit_status": status,
        }
