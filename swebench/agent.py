#!/usr/bin/env python3
"""
Agent loop for SWE-bench using ash tools.

The agent uses litellm to support any model and calls ash tools
via subprocess (CLI) or MCP protocol.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import litellm
    from litellm import completion
except ImportError:
    raise ImportError("Install litellm: pip install litellm")

from . import (
    ASH_TOOLS, 
    AgentConfig, 
    AshToolResult, 
    Trajectory,
    call_ash_tool,
    generate_tools_schema,
)


def format_tools_description() -> str:
    """Format tools for system prompt."""
    lines = []
    for name, spec in ASH_TOOLS.items():
        params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in spec.get("parameters", {}).items())
        lines.append(f"- {name}({params}): {spec['description']}")
    return "\n".join(lines)


class AshAgent:
    """Agent that uses ash tools to solve SWE-bench tasks."""
    
    def __init__(
        self,
        config: AgentConfig,
        executor: Callable[[str, dict], AshToolResult] = None,
    ):
        self.config = config
        self.executor = executor or (lambda name, args: call_ash_tool(name, args, config.ash_binary))
        self.trajectory = Trajectory()
        self.cost = 0.0
        self.n_calls = 0
        
    def _build_system_prompt(self) -> str:
        return self.config.system_template.format(
            tools_description=format_tools_description()
        )
    
    def _query_model(self, messages: list[dict]) -> dict:
        """Query the LLM with tools."""
        self.n_calls += 1
        
        response = completion(
            model=self.config.model,
            messages=messages,
            tools=generate_tools_schema(),
            tool_choice="auto",
            temperature=self.config.temperature,
        )
        
        # Track cost
        if hasattr(response, "usage"):
            # Rough cost estimate
            input_cost = (response.usage.prompt_tokens / 1000) * 0.003
            output_cost = (response.usage.completion_tokens / 1000) * 0.015
            self.cost += input_cost + output_cost
        
        return response
    
    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Execute tool calls and return observation messages."""
        observations = []
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            
            # Execute via ash
            result = self.executor(name, args)
            
            # Format observation
            if result.success:
                content = result.output[:10000]  # Truncate long output
                if len(result.output) > 10000:
                    content += f"\n... (truncated {len(result.output) - 10000} chars)"
            else:
                content = f"Error: {result.error or 'Unknown error'}"
            
            observations.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
            
            # Save to trajectory
            self.trajectory.add_message(
                "tool_result",
                content,
                tool_name=name,
                tool_args=args,
                success=result.success,
            )
        
        return observations
    
    def _check_submission(self, output: str) -> Optional[str]:
        """Check if output contains submission marker."""
        lines = output.strip().splitlines()
        if lines and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in lines[0]:
            # Return the patch (everything after the marker line)
            idx = output.find("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
            rest = output[idx + len("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):].strip()
            # Skip to next line
            if "\n" in rest:
                return rest.split("\n", 1)[1]
            return ""
        return None
    
    def run(self, task: str, instance_id: str = "") -> dict:
        """Run the agent on a task. Returns exit info."""
        self.trajectory = Trajectory()
        self.trajectory.instance_id = instance_id
        self.cost = 0.0
        self.n_calls = 0
        
        # Build initial messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": task},
        ]
        
        self.trajectory.add_message("system", messages[0]["content"])
        self.trajectory.add_message("user", task)
        
        while self.n_calls < self.config.step_limit and self.cost < self.config.cost_limit:
            # Query model
            response = self._query_model(messages)
            choice = response.choices[0]
            message = choice.message
            
            # Add assistant message
            assistant_content = message.content or ""
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": message.tool_calls,
            })
            self.trajectory.add_message("assistant", assistant_content)
            
            # Check for tool calls
            if not message.tool_calls:
                # No tool calls - agent is confused or done
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in assistant_content:
                    # Agent tried to submit in text
                    submission = self._check_submission(assistant_content)
                    self.trajectory.info = {
                        "exit_status": "Submitted",
                        "submission": submission or "",
                        "model": self.config.model,
                    }
                    break
                continue
            
            # Execute tools
            observations = self._execute_tool_calls(message.tool_calls)
            messages.extend(observations)
            
            # Check for submission in shell output
            for obs in observations:
                submission = self._check_submission(obs["content"])
                if submission is not None:
                    self.trajectory.info = {
                        "exit_status": "Submitted",
                        "submission": submission,
                        "model": self.config.model,
                    }
                    self.trajectory.model_stats = {
                        "api_calls": self.n_calls,
                        "instance_cost": self.cost,
                    }
                    return self.trajectory.info
            
            # Check stop conditions
            if choice.finish_reason == "stop" and not message.tool_calls:
                break
        
        # Didn't submit - limits exceeded or agent gave up
        if self.n_calls >= self.config.step_limit or self.cost >= self.config.cost_limit:
            exit_status = "LimitsExceeded"
        else:
            exit_status = "AgentStopped"
        
        self.trajectory.info = {
            "exit_status": exit_status,
            "submission": "",
            "model": self.config.model,
        }
        self.trajectory.model_stats = {
            "api_calls": self.n_calls,
            "instance_cost": self.cost,
        }
        return self.trajectory.info
