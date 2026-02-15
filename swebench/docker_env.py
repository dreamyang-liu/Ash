#!/usr/bin/env python3
"""
Docker environment for running ash agent in SWE-bench containers.

Copies ash binary into container and executes tools natively.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import AshToolResult


@dataclass
class DockerConfig:
    """Docker environment configuration."""
    image: str = ""
    container_name: str = ""
    timeout: int = 60
    workdir: str = "/testbed"
    
    # Path to ash binary (will be copied into container)
    ash_binary: str = ""
    
    # Resource limits
    memory: str = "16g"
    cpus: str = "4"


class DockerEnvironment:
    """Execute ash commands inside a Docker container."""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        self.container_id: Optional[str] = None
        self._ash_ready = False
        
    def start(self, instance: dict) -> bool:
        """Start container for a SWE-bench instance."""
        # Get image from instance metadata
        # SWE-bench instances have env_image_key that maps to docker image
        image = instance.get("env_image_key", "")
        if not image:
            # Fallback: construct from repo
            repo = instance.get("repo", "").replace("/", "__").lower()
            base_commit = instance.get("base_commit", "")[:12]
            image = f"swebench/sweb.eval.x86_64.{repo}:{base_commit}"
        
        print(f"Using image: {image}")
        
        # Check if image exists locally, pull if not
        check = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
        )
        if check.returncode != 0:
            print(f"Pulling image {image}...")
            pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
            if pull.returncode != 0:
                print(f"Failed to pull: {pull.stderr}")
                return False
        
        # Generate container name
        self.container_id = f"ash-swebench-{int(time.time())}"
        
        # Start container
        cmd = [
            "docker", "run",
            "-d",
            "--name", self.container_id,
            "--memory", self.config.memory,
            "--cpus", self.config.cpus,
            "-w", self.config.workdir,
            image,
            "sleep", "infinity",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to start container: {result.stderr}")
            return False
        
        print(f"Started container: {self.container_id}")
        
        # Copy ash binary into container
        if self.config.ash_binary:
            self._copy_ash_binary()
        
        return True
    
    def _copy_ash_binary(self):
        """Copy ash binary into the container."""
        ash_path = Path(self.config.ash_binary)
        if not ash_path.exists():
            print(f"Warning: ash binary not found at {ash_path}")
            return
        
        # Copy binary
        subprocess.run([
            "docker", "cp",
            str(ash_path),
            f"{self.container_id}:/usr/local/bin/ash",
        ], check=True)
        
        # Make executable
        subprocess.run([
            "docker", "exec", self.container_id,
            "chmod", "+x", "/usr/local/bin/ash",
        ], check=True)
        
        # Verify
        result = subprocess.run(
            ["docker", "exec", self.container_id, "ash", "tools"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"ash binary ready in container")
            print(f"Available tools: {result.stdout.strip()}")
            self._ash_ready = True
        else:
            print(f"Warning: ash binary not working: {result.stderr}")
    
    def stop(self):
        """Stop and remove container."""
        if self.container_id:
            subprocess.run(
                ["docker", "rm", "-f", self.container_id],
                capture_output=True,
            )
            print(f"Stopped container: {self.container_id}")
            self.container_id = None
    
    def execute_shell(self, command: str) -> AshToolResult:
        """Execute raw shell command in container."""
        if not self.container_id:
            return AshToolResult(False, "", "Container not running")
        
        cmd = [
            "docker", "exec",
            "-w", self.config.workdir,
            self.container_id,
            "bash", "-c", command,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return AshToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return AshToolResult(False, "", f"Timeout after {self.config.timeout}s")
        except Exception as e:
            return AshToolResult(False, "", str(e))
    
    def call_ash_tool(self, tool_name: str, args: dict) -> AshToolResult:
        """Call ash tool inside container."""
        if not self.container_id:
            return AshToolResult(False, "", "Container not running")
        
        if not self._ash_ready:
            # Fallback to shell emulation
            return self._emulate_tool(tool_name, args)
        
        # Call ash CLI with JSON args
        args_json = json.dumps(args)
        cmd = [
            "docker", "exec",
            "-w", self.config.workdir,
            self.container_id,
            "ash", "call", tool_name, args_json,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                return AshToolResult(
                    success=data.get("success", False),
                    output=data.get("output", ""),
                    error=data.get("error"),
                )
            except json.JSONDecodeError:
                # Raw output
                return AshToolResult(
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr if result.returncode != 0 else None,
                )
        except subprocess.TimeoutExpired:
            return AshToolResult(False, "", f"Timeout after {self.config.timeout}s")
        except Exception as e:
            return AshToolResult(False, "", str(e))
    
    def _emulate_tool(self, tool_name: str, args: dict) -> AshToolResult:
        """Fallback: emulate ash tools with shell commands."""
        if tool_name == "shell":
            return self.execute_shell(args.get("command", ""))
        
        elif tool_name == "read_file":
            path = args.get("file_path", "")
            offset = args.get("offset", 1)
            limit = args.get("limit", 100)
            end = offset + limit - 1
            cmd = f"sed -n '{offset},{end}p' '{path}' | nl -ba -v {offset}"
            return self.execute_shell(cmd)
        
        elif tool_name == "grep_files":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            include = args.get("include", "")
            limit = args.get("limit", 100)
            
            # Escape pattern for shell
            pattern_escaped = pattern.replace("'", "'\\''")
            
            if include:
                cmd = f"grep -rn --include='{include}' '{pattern_escaped}' {path} 2>/dev/null | head -{limit}"
            else:
                cmd = f"grep -rn '{pattern_escaped}' {path} 2>/dev/null | head -{limit}"
            return self.execute_shell(cmd)
        
        elif tool_name == "text_editor":
            return self._emulate_editor(args)
        
        elif tool_name == "git_status":
            cmd = "git status -s" if args.get("short") else "git status"
            return self.execute_shell(cmd)
        
        elif tool_name == "git_diff":
            cmd = "git diff"
            if args.get("staged"):
                cmd += " --staged"
            for p in args.get("paths", []):
                cmd += f" '{p}'"
            return self.execute_shell(cmd)
        
        return AshToolResult(False, "", f"Unknown tool: {tool_name}")
    
    def _emulate_editor(self, args: dict) -> AshToolResult:
        """Emulate text_editor tool."""
        command = args.get("command", "view")
        path = args.get("path", "")
        
        if command == "view":
            view_range = args.get("view_range", [1, 100])
            start = view_range[0] if view_range else 1
            end = view_range[1] if len(view_range) > 1 else start + 99
            cmd = f"sed -n '{start},{end}p' '{path}' | nl -ba -v {start}"
            return self.execute_shell(cmd)
        
        elif command == "str_replace":
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")
            
            # Use Python for reliable replacement
            py_script = f'''
import sys
path = {repr(path)}
old_str = {repr(old_str)}
new_str = {repr(new_str)}

with open(path, 'r') as f:
    content = f.read()

count = content.count(old_str)
if count == 0:
    print("ERROR: No match found", file=sys.stderr)
    sys.exit(1)
if count > 1:
    print(f"ERROR: Multiple matches ({{count}}). Must be unique.", file=sys.stderr)
    sys.exit(1)

new_content = content.replace(old_str, new_str, 1)
with open(path, 'w') as f:
    f.write(new_content)
print("Replaced successfully")
'''
            return self.execute_shell(f"python3 -c {repr(py_script)}")
        
        elif command == "insert":
            line = args.get("insert_line", 0)
            text = args.get("insert_text", "")
            
            py_script = f'''
path = {repr(path)}
line = {line}
text = {repr(text)}

with open(path, 'r') as f:
    lines = f.readlines()

# Insert after line (0 = beginning)
insert_idx = min(line, len(lines))
for i, new_line in enumerate(text.split('\\n')):
    lines.insert(insert_idx + i, new_line + '\\n')

with open(path, 'w') as f:
    f.writelines(lines)
print(f"Inserted at line {{line}}")
'''
            return self.execute_shell(f"python3 -c {repr(py_script)}")
        
        elif command == "create":
            content = args.get("file_text", "")
            
            py_script = f'''
import os
path = {repr(path)}
content = {repr(content)}

os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
with open(path, 'w') as f:
    f.write(content)
print(f"Created: {{path}}")
'''
            return self.execute_shell(f"python3 -c {repr(py_script)}")
        
        return AshToolResult(False, "", f"Unknown editor command: {command}")


def create_docker_executor(env: DockerEnvironment):
    """Create executor function that routes tools through docker."""
    def executor(tool_name: str, args: dict) -> AshToolResult:
        return env.call_ash_tool(tool_name, args)
    return executor
