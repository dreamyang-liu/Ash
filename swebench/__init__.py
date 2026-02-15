#!/usr/bin/env python3
"""
SWE-bench integration for ash-cli.

This module provides the glue between ash tools and SWE-bench evaluation:
1. Agent loop that uses ash tools via subprocess or MCP
2. Docker environment for sandboxed execution
3. Trajectory saving in mini-swe-agent compatible format
4. Batch runner for benchmarks

Usage:
    # Single instance
    python -m swebench.runner --instance sympy__sympy-15599

    # Batch mode
    python -m swebench.runner --subset verified --split test --workers 4

    # Evaluate results
    sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id ash-run-1
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os


@dataclass
class AshToolResult:
    """Result from calling an ash tool."""
    success: bool
    output: str
    error: Optional[str] = None


def call_ash_tool(tool_name: str, args: dict, ash_binary: str = "ash") -> AshToolResult:
    """Call an ash tool via CLI."""
    try:
        result = subprocess.run(
            [ash_binary, "call", tool_name, json.dumps(args)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        data = json.loads(result.stdout) if result.stdout else {}
        return AshToolResult(
            success=data.get("success", result.returncode == 0),
            output=data.get("output", result.stdout),
            error=data.get("error"),
        )
    except Exception as e:
        return AshToolResult(success=False, output="", error=str(e))


def call_ash_mcp(tool_name: str, args: dict, mcp_process) -> AshToolResult:
    """Call an ash tool via MCP protocol (for persistent connection)."""
    request = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args}
    }
    mcp_process.stdin.write(json.dumps(request) + "\n")
    mcp_process.stdin.flush()
    
    response_line = mcp_process.stdout.readline()
    response = json.loads(response_line)
    
    if "error" in response:
        return AshToolResult(success=False, output="", error=response["error"]["message"])
    
    result = response.get("result", {})
    content = result.get("content", [{}])[0]
    return AshToolResult(
        success=not result.get("isError", False),
        output=content.get("text", ""),
    )


# ASH tools mapped to what an agent would need
ASH_TOOLS = {
    "read_file": {
        "description": "Read file contents with line numbers",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "offset": {"type": "integer", "default": 1},
            "limit": {"type": "integer", "default": 100},
        }
    },
    "grep_files": {
        "description": "Search for pattern in files using ripgrep",
        "parameters": {
            "pattern": {"type": "string", "required": True},
            "path": {"type": "string", "default": "."},
            "include": {"type": "string"},
            "limit": {"type": "integer", "default": 100},
        }
    },
    "text_editor": {
        "description": "Edit files: view, str_replace, insert, create",
        "parameters": {
            "command": {"type": "string", "enum": ["view", "str_replace", "insert", "create"]},
            "path": {"type": "string", "required": True},
            # view
            "view_range": {"type": "array"},
            # str_replace
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            # insert
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            # create
            "file_text": {"type": "string"},
        }
    },
    "shell": {
        "description": "Execute shell command",
        "parameters": {
            "command": {"type": "string", "required": True},
            "timeout_secs": {"type": "integer", "default": 300},
        }
    },
    "git_status": {"description": "Show git status", "parameters": {"short": {"type": "boolean"}}},
    "git_diff": {"description": "Show git diff", "parameters": {"staged": {"type": "boolean"}, "paths": {"type": "array"}}},
}


def generate_tools_schema() -> list[dict]:
    """Generate OpenAI-compatible tool schema for LLM."""
    tools = []
    for name, spec in ASH_TOOLS.items():
        properties = {}
        required = []
        for param_name, param_spec in spec.get("parameters", {}).items():
            prop = {"type": param_spec.get("type", "string")}
            if "enum" in param_spec:
                prop["enum"] = param_spec["enum"]
            if "description" in param_spec:
                prop["description"] = param_spec["description"]
            properties[param_name] = prop
            if param_spec.get("required"):
                required.append(param_name)
        
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    return tools


@dataclass
class AgentConfig:
    """Configuration for the ash agent."""
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    step_limit: int = 250
    cost_limit: float = 3.0
    temperature: float = 0.0
    ash_binary: str = "ash"
    cwd: str = "/testbed"
    
    # System prompt
    system_template: str = """You are a software engineer helping fix issues in a codebase.

You have access to the following tools:
{tools_description}

For each response:
1. Explain your reasoning in a THOUGHT section
2. Call one or more tools to investigate or make changes

When done, create a patch and submit:
1. git diff -- <changed_files> > patch.txt
2. Verify patch.txt looks correct  
3. echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
"""


@dataclass  
class Trajectory:
    """Agent trajectory for saving."""
    messages: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    instance_id: str = ""
    model_stats: dict = field(default_factory=lambda: {"api_calls": 0, "instance_cost": 0.0})
    
    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        if extra:
            msg["extra"] = extra
        self.messages.append(msg)
    
    def save(self, path: Path):
        """Save trajectory in mini-swe-agent compatible format."""
        data = {
            "trajectory_format": "ash-agent-1.0",
            "messages": self.messages,
            "info": {
                "model_stats": self.model_stats,
                "exit_status": self.info.get("exit_status", ""),
                "submission": self.info.get("submission", ""),
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    
    def to_prediction(self) -> dict:
        """Convert to SWE-bench prediction format."""
        return {
            "instance_id": self.instance_id,
            "model_patch": self.info.get("submission", ""),
            "model_name_or_path": self.info.get("model", "ash-agent"),
        }
