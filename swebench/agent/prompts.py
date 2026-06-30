"""System and kickoff prompt construction for the SWE-bench agent."""

import platform
from typing import Optional

from ..models import AgentConfig

_DEFAULT_SYSTEM = "You are a helpful assistant that can interact with a computer to solve software engineering tasks."

_DEFAULT_INSTANCE = "Please solve this issue: {{task}}"


def _render_template(template: str, **kwargs) -> str:
    """Render a template with simple {{var}} substitution."""
    info = platform.uname()
    kwargs.setdefault("system", info.system)
    kwargs.setdefault("release", info.release)
    kwargs.setdefault("version", info.version)
    kwargs.setdefault("machine", info.machine)
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def build_system_prompt(task: str, config: Optional[AgentConfig] = None) -> str:
    """Build the full system prompt from config's system_template."""
    template = (config.system_template if config and config.system_template else _DEFAULT_SYSTEM)
    return _render_template(template, task=task)


def build_instance_message(task: str, config: Optional[AgentConfig] = None) -> str:
    """Build the first user message (kickoff) from config's instance_template."""
    template = (config.instance_template if config and config.instance_template else _DEFAULT_INSTANCE)
    return _render_template(template, task=task)
