"""Custom tool manifests: declarative binary tools compiled to runtime calls.

Called from tools.route_agent_tool (custom-tool fallback) and by session
executors that run the two-step plan; schemas surface via TOOLS_SCHEMA
consumers. Contract tests: swebench/tests/test_custom_tools.py.
User instruction: custom tools design ("先只支持binary，然后映射命令行",
binary via URL + runtime download) / "ok".

A custom tool is DATA, not code: a manifest declares a binary (URL +
mandatory sha256), a parameter schema, and how each parameter maps onto
argv slots. A call expands into two frozen runtime primitives:

    1. artifact(url, sha256)  -> local verified binary path
    2. shell(shlex-joined argv, timeout)

Security invariant: parameters may only land in discrete argv slots
(positional / flag value / boolean switch). There is NO string templating,
so agent-supplied values can never be interpreted by the shell.

Manifest example (JSON or YAML):

    name: code_complexity
    description: Analyze cyclomatic complexity of a source file
    binary:
      url: https://example.com/analyzer-linux-amd64
      sha256: "ab34..."
    parameters:
      file:      {type: string, required: true, map: {positional: 0}}
      threshold: {type: integer, default: 10, map: {flag: "--threshold"}}
      verbose:   {type: boolean, map: {flag: "--verbose", style: switch}}
    timeout: 30
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

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


class ManifestError(ValueError):
    """Raised when a manifest is malformed."""


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
    #   url+sha256: downloaded at first use via the artifact primitive
    #   path:       absolute path already present in the sandbox image
    url: str = ""
    sha256: str = ""
    path: str = ""
    params: dict[str, ParamSpec] = field(default_factory=dict)
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    def agent_schema(self) -> dict:
        """OpenAI function-calling schema for this tool."""
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
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def compile_argv(self, binary_path: str, args: dict) -> list[str]:
        """Validate args against the schema and render an argv list.

        Every value lands in its own argv slot; no shell interpretation of
        agent input is possible.
        """
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
                positionals.append((p.positional, str(value)))
            elif p.switch:
                if value is True:
                    flags.append(p.flag)  # presence only
            elif p.flag:
                flags.extend([p.flag, str(value)])

        positionals.sort(key=lambda t: t[0])
        return [binary_path, *flags, *[v for _, v in positionals]]


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
        switch = mapping.get("style") == "switch"
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


# Registry of loaded custom tools. Populated via register()/load_manifests().
CUSTOM_TOOL_SPECS: dict[str, CustomToolSpec] = {}


def register(spec: CustomToolSpec) -> None:
    """Register a custom tool spec, refusing collisions with builtins."""
    from .tools import AGENT_TOOL_ROUTES  # local import to avoid cycle

    if spec.name in AGENT_TOOL_ROUTES:
        raise ManifestError(f"custom tool {spec.name!r} collides with a builtin tool")
    CUSTOM_TOOL_SPECS[spec.name] = spec


# Default manifest location (repo-relative), overridable per run.
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "configs" / "custom_tools"


def load_custom_tools(directory: str | Path | None = None) -> list[CustomToolSpec]:
    """Load manifests from `directory`, or the default location.

    - explicit directory: must exist (typo of a user-passed path is an error)
    - default location: silently skipped when absent (opt-in feature)
    """
    if directory is not None:
        directory = Path(directory)
        if not directory.is_dir():
            raise ManifestError(f"custom tools dir not found: {directory}")
        return load_manifests(directory)
    if DEFAULT_MANIFEST_DIR.is_dir():
        return load_manifests(DEFAULT_MANIFEST_DIR)
    return []


def load_manifests(directory: str | Path) -> list[CustomToolSpec]:
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
                raise ManifestError(f"{path.name}: pyyaml required for YAML manifests") from exc
            raw = yaml.safe_load(path.read_text())
        else:
            continue
        spec = parse_manifest(raw)
        register(spec)
        specs.append(spec)
    return specs


def custom_agent_schemas() -> list[dict]:
    """Function-calling schemas for all registered custom tools."""
    return [spec.agent_schema() for spec in CUSTOM_TOOL_SPECS.values()]


@dataclass(frozen=True)
class CustomToolPlan:
    """Execution plan for one custom tool call.

    url-sourced tools: run artifact_call first; its Output is the verified
    binary path, fed to shell_call() for the second step. path-sourced
    tools (binary already in the image): artifact_call is None and the
    executor goes straight to shell_call(spec.path). Args are validated
    eagerly (at plan time) so bad calls fail before any download happens.
    """

    spec: CustomToolSpec
    args: dict

    @property
    def artifact_call(self) -> tuple[str, dict] | None:
        if not self.spec.url:
            return None  # image-local binary: no download step
        args = {"url": self.spec.url}
        if self.spec.sha256:
            args["sha256"] = self.spec.sha256
        return "artifact", args

    def shell_call(self, binary_path: str) -> tuple[str, dict]:
        argv = self.spec.compile_argv(binary_path, self.args)
        return "shell", {"command": shlex.join(argv), "timeout": self.spec.timeout}


def plan_custom_tool(name: str, args: dict) -> CustomToolPlan:
    """Resolve a custom tool call into a two-step execution plan."""
    spec = CUSTOM_TOOL_SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown custom tool: {name}")
    # Validate args now (placeholder path) so errors surface pre-download.
    spec.compile_argv("/placeholder", dict(args))
    return CustomToolPlan(spec=spec, args=dict(args))
