"""Data-driven tool layer: builtin routes and custom tools as configuration.

Callers: sandbox.py (Sandbox.call_agent_tool dispatch), exported via
__init__.py; swebench/agent/{tools,custom_tools}.py become re-export shims.
Supersedes swebench/agent/custom_tools.py (moved here per discussion:
"tool 之类的可以用DSL 或者data 配置项来做"，AshAgent 纯策略层).
Data schema: same manifest YAML/JSON as configs/custom_tools/README.md.

Everything an agent-facing tool *is* lives here as data:

- BUILTIN_ROUTES: agent tool name -> runtime tool name (plain dict)
- CustomToolSpec: manifest-defined binary tool (URL+optional sha256, or
  image-local path) with parameters compiled into discrete argv slots
- ToolRegistry: an *instance* holding routes + custom specs, so different
  sessions/tasks can carry different tool panels without global state

Security invariant (custom tools): parameters may only land in discrete
argv slots (positional / flag value / boolean switch). No string
templating — agent input is never interpreted by the shell.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from . import schemas

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLAG_RE = re.compile(r"^--?[A-Za-z][A-Za-z0-9_-]*$")

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600

# Agent-facing name -> runtime tool name. Pure data; extend or override by
# constructing ToolRegistry(routes={...}).
BUILTIN_ROUTES: dict[str, str] = {
    "shell": "shell",
    "bash": "shell",  # bash_only-mode alias
    "text_editor": "text_editor",
    "grep_files": "grep_files",
    "process": "process",
    "web_fetch": "web_fetch",
    "web_search": "web_search",
    "wait_for_events": "wait_for_events",
}


class ManifestError(ValueError):
    """Raised when a custom tool manifest is malformed."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    required: bool = False
    default: object = None
    description: str = ""
    positional: int | None = None
    flag: str | None = None
    switch: bool = False  # boolean presence-flag (no value)


@dataclass(frozen=True)
class CustomToolSpec:
    name: str
    description: str
    # Binary source — exactly one is set:
    #   url (+ optional sha256): downloaded at first use via `artifact`
    #   path: absolute path already present in the sandbox image
    url: str = ""
    sha256: str = ""
    path: str = ""
    params: dict[str, ParamSpec] = field(default_factory=dict)
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    def parameters_schema(self) -> dict:
        """JSON-Schema object describing this tool's call arguments."""
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in self.params.values():
            prop: dict = {"type": p.type}
            if p.description:
                prop["description"] = p.description
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": properties, "required": required}

    def agent_schema(self, format: str = "openai") -> dict:
        """Function-calling schema for this tool, in a provider's shape.

        Rendering is shared with the runtime's builtin tools (see schemas.py),
        so a custom tool is available in every format the SDK supports rather
        than only OpenAI's.
        """
        return schemas.render(self.name, self.description,
                              self.parameters_schema(), format)

    def compile_argv(self, binary_path: str, args: dict) -> list[str]:
        """Validate args against the schema and render an argv list."""
        unknown = set(args) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")

        resolved: dict[str, object] = {}
        for p in self.params.values():
            if p.name in args:
                value = args[p.name]
            elif p.default is not None:
                value = p.default
            elif p.required:
                raise ValueError(f"missing required parameter: {p.name}")
            else:
                continue
            if not _TYPE_CHECKS[p.type](value):
                raise ValueError(f"parameter {p.name} must be {p.type}")
            resolved[p.name] = value

        positionals: list[tuple[int, str]] = []
        flags: list[str] = []
        for p in self.params.values():
            if p.name not in resolved:
                continue
            value = resolved[p.name]
            if p.positional is not None:
                positionals.append((p.positional, _as_argv(value)))
            elif p.switch:
                if value is True:
                    flags.append(p.flag)  # presence only
            elif p.flag:
                flags.extend([p.flag, _as_argv(value)])

        positionals.sort(key=lambda t: t[0])
        return [binary_path, *flags, *[v for _, v in positionals]]


def _as_argv(value) -> str:
    """Render one argument the way a command-line tool expects to read it.

    Python's str() gives "True", which no CLI or JSON parser accepts; every
    other type is already faithful.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_manifest(raw: dict) -> CustomToolSpec:
    """Validate a raw manifest dict into a CustomToolSpec."""
    name = raw.get("name", "")
    if not _NAME_RE.match(name):
        raise ManifestError(f"invalid tool name: {name!r}")

    binary = raw.get("binary") or {}
    url = binary.get("url", "")
    sha256 = str(binary.get("sha256", "")).lower()
    path = binary.get("path", "")
    if bool(url) == bool(path):
        raise ManifestError(f"{name}: binary must set exactly one of url/path")
    if url:
        if not url.startswith(("http://", "https://")):
            raise ManifestError(f"{name}: binary.url must be http(s)")
        # sha256 is optional (recommended): when set, content is verified
        # before execution; when omitted, the download is trusted as-is.
        if binary.get("sha256") is not None and not _SHA256_RE.match(sha256):
            raise ManifestError(f"{name}: binary.sha256 must be a 64-char hex digest")
        if binary.get("sha256") is None:
            sha256 = ""
    else:
        if not path.startswith("/"):
            raise ManifestError(f"{name}: binary.path must be absolute")
        if sha256:
            raise ManifestError(f"{name}: binary.sha256 is only valid with url")

    params: dict[str, ParamSpec] = {}
    positions_seen: set[int] = set()
    for pname, praw in (raw.get("parameters") or {}).items():
        if not _NAME_RE.match(pname):
            raise ManifestError(f"{name}: invalid parameter name {pname!r}")
        ptype = praw.get("type", "string")
        if ptype not in _TYPE_CHECKS:
            raise ManifestError(f"{name}.{pname}: unsupported type {ptype!r}")
        mapping = praw.get("map") or {}
        positional = mapping.get("positional")
        flag = mapping.get("flag")
        # A boolean behind a flag is a switch: the flag's presence *is* the
        # value, so "--fix" rather than "--fix True". Requiring an explicit
        # style: switch meant the obvious manifest emitted a literal "True" as
        # a separate argument, which most tools then read as a filename.
        # style: value opts out, for the rare --flag=true interface.
        style = mapping.get("style")
        if style is not None and style not in ("switch", "value"):
            # A typo would otherwise fall through to value style and emit a
            # stray "True" argument at runtime, far from the manifest.
            raise ManifestError(
                f"{name}.{pname}: unknown style {style!r} (switch|value)")
        switch = style == "switch" or (
            style is None and ptype == "boolean" and flag is not None
        )
        if (positional is None) == (flag is None):
            raise ManifestError(
                f"{name}.{pname}: map must set exactly one of positional/flag"
            )
        if positional is not None:
            if not isinstance(positional, int) or positional < 0:
                raise ManifestError(f"{name}.{pname}: positional must be int >= 0")
            if positional in positions_seen:
                raise ManifestError(f"{name}.{pname}: duplicate positional {positional}")
            positions_seen.add(positional)
        if flag is not None and not _FLAG_RE.match(flag):
            raise ManifestError(f"{name}.{pname}: invalid flag {flag!r}")
        if switch and ptype != "boolean":
            raise ManifestError(f"{name}.{pname}: switch style requires boolean type")
        default = praw.get("default")
        if default is not None and not _TYPE_CHECKS[ptype](default):
            raise ManifestError(f"{name}.{pname}: default must be {ptype}")
        params[pname] = ParamSpec(
            name=pname,
            type=ptype,
            required=bool(praw.get("required", False)),
            default=default,
            description=praw.get("description", ""),
            positional=positional,
            flag=flag,
            switch=switch,
        )

    timeout = raw.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    if not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ManifestError(f"{name}: timeout must be 1..{MAX_TIMEOUT_SECONDS}")

    return CustomToolSpec(
        name=name,
        description=raw.get("description", name),
        url=url,
        sha256=sha256,
        path=path,
        params=params,
        timeout=timeout,
    )


@dataclass(frozen=True)
class CustomToolPlan:
    """Execution plan for one custom tool call.

    url-sourced tools: run artifact_call first; its Output is the verified
    binary path, fed to shell_call(). path-sourced tools: artifact_call is
    None; go straight to shell_call(spec.path). Args are validated eagerly
    (at plan time) so bad calls fail before any download.
    """

    spec: CustomToolSpec
    args: dict

    @property
    def artifact_call(self) -> tuple[str, dict] | None:
        if not self.spec.url:
            return None  # image-local binary: no download step
        call_args = {"url": self.spec.url}
        if self.spec.sha256:
            call_args["sha256"] = self.spec.sha256
        return "artifact", call_args

    def shell_call(self, binary_path: str) -> tuple[str, dict]:
        argv = self.spec.compile_argv(binary_path, self.args)
        return "shell", {"command": shlex.join(argv), "timeout": self.spec.timeout}


class ToolRegistry:
    """A tool panel: builtin routes + custom tool specs, as an instance.

    Each session/task can carry its own registry — no cross-task global
    state. The default panel is BUILTIN_ROUTES with no custom tools.
    """

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        self.routes: dict[str, str] = dict(routes if routes is not None else BUILTIN_ROUTES)
        self.custom_specs: dict[str, CustomToolSpec] = {}

    # --- custom tool management ---

    def register(self, spec: CustomToolSpec) -> None:
        """Register a custom tool spec, refusing collisions with builtins."""
        if spec.name in self.routes:
            raise ManifestError(f"custom tool {spec.name!r} collides with a builtin tool")
        self.custom_specs[spec.name] = spec

    def load_manifests(self, directory: str | Path) -> list[CustomToolSpec]:
        """Load and register all *.json / *.yaml manifests in a directory."""
        directory = Path(directory)
        specs: list[CustomToolSpec] = []
        for path in sorted(directory.glob("*")):
            if path.suffix == ".json":
                raw = json.loads(path.read_text())
            elif path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError as exc:  # pragma: no cover
                    raise ManifestError(
                        f"{path.name}: pyyaml required for YAML manifests"
                    ) from exc
                raw = yaml.safe_load(path.read_text())
            else:
                continue
            spec = parse_manifest(raw)
            self.register(spec)
            specs.append(spec)
        return specs

    # --- resolution ---

    def is_custom_tool(self, name: str) -> bool:
        return name in self.custom_specs

    def route(self, name: str, args: dict) -> tuple[str, dict]:
        """Translate a builtin agent tool call to a runtime tool call."""
        runtime_tool = self.routes.get(name)
        if runtime_tool is None:
            raise KeyError(f"unknown agent tool: {name}")
        return runtime_tool, dict(args)

    def plan_custom_tool(self, name: str, args: dict) -> CustomToolPlan:
        """Resolve a custom tool call into an execution plan."""
        spec = self.custom_specs.get(name)
        if spec is None:
            raise KeyError(f"unknown custom tool: {name}")
        # Validate args now (placeholder path) so errors surface pre-download.
        spec.compile_argv("/placeholder", dict(args))
        return CustomToolPlan(spec=spec, args=dict(args))

    def plan_custom_tool_for_prepare(self, name: str) -> CustomToolPlan:
        """Plan used to resolve a tool's binary ahead of any real call.

        Argument validation is skipped deliberately: preparing a binary is
        about fetching and verifying it, and there are no agent arguments yet.
        Only the artifact half of the plan is meaningful here.
        """
        spec = self.custom_specs.get(name)
        if spec is None:
            raise KeyError(f"unknown custom tool: {name}")
        return CustomToolPlan(spec=spec, args={})

    def custom_agent_schemas(self, format: str = "openai") -> list[dict]:
        """Function-calling schemas for all registered custom tools."""
        return [spec.agent_schema(format) for spec in self.custom_specs.values()]
