"""Compiling the tool panel a model sees.

The runtime declares what exists; manifests declare what to offer and how; this
module produces the panel. Nothing about the panel is hand-maintained, because a
hand-maintained copy drifts, and the one this replaces had: of seven tools, four
disagreed with the runtime. One of them offered `web_fetch(max_length=…)`, a
parameter the runtime dropped -- so a model could ask for a bound, get no error, and
not be bounded.

Two kinds of declaration, and they are different things:

    AgentToolSpec    a *view* of a runtime tool -- rename it, offer a subset of its
                     parameters under names of your choosing, describe it for the task
    CustomToolSpec   an external binary (toolset.py), expanded into artifact+shell

A view's `arguments` is both the mapping and the whitelist: what is in it is offered,
what is not does not exist for that view. Compiling validates against the runtime's
declaration, which is the whole point:

    1. every agent tool targets a runtime tool that exists
    2. every offered parameter exists on that tool
    3. every parameter the runtime *requires* is offered, or no call can succeed
    4. names do not collide

Rule 1 is also why there is no route table. A mapping exists where something is
actually mapped; the seven identity entries that used to sit in `BUILTIN_ROUTES`
were an indirection layer carrying nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import schemas

__all__ = ["AgentToolSpec", "PanelError", "RuntimeDeclaration", "compile_panel",
           "load_declaration", "parse_agent_tool"]


class PanelError(ValueError):
    """A panel could not be compiled: a manifest disagrees with the runtime."""


# --------------------------------------------------------------------------- #
#  The runtime's declaration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RuntimeDeclaration:
    """What a runtime says it serves — the base draft the panel is compiled from.

    Produced by ``ash-runtime --dump-schema``, so a panel can be built and checked
    with no sandbox running (in tests, or when validating a config before a run).
    """
    version: str
    tools: dict[str, dict]          # name -> {description, inputSchema}

    @classmethod
    def from_dump(cls, data: dict) -> "RuntimeDeclaration":
        tools = {}
        for entry in data.get("tools") or ():
            name = entry.get("name")
            if not name:
                raise PanelError("runtime declaration has a tool with no name")
            tools[name] = entry
        if not tools:
            raise PanelError("runtime declaration lists no tools")
        return cls(version=str(data.get("version", "")), tools=tools)

    def parameters_of(self, tool: str) -> dict[str, dict]:
        return ((self.tools[tool].get("inputSchema") or {}).get("properties") or {})

    def required_of(self, tool: str) -> set[str]:
        return set((self.tools[tool].get("inputSchema") or {}).get("required") or ())


def load_declaration(source: "str | Path | dict") -> RuntimeDeclaration:
    """Read a declaration from a dict, or from a ``--dump-schema`` JSON file."""
    if isinstance(source, dict):
        return RuntimeDeclaration.from_dump(source)
    path = Path(source)
    if not path.is_file():
        raise PanelError(f"no runtime declaration at {path}")
    try:
        return RuntimeDeclaration.from_dump(json.loads(path.read_text()))
    except json.JSONDecodeError as exc:
        raise PanelError(f"{path} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
#  Agent tools: views of runtime tools
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgentToolSpec:
    """One tool as the model sees it, defined over a runtime tool.

    ``arguments`` is the whole parameter surface: it maps the names the model uses
    onto the runtime's, and by doing so says which parameters exist at all. Anything
    absent from it is not part of this view, and calling with it is an error rather
    than a silent drop -- a dropped argument is the failure this module was built to
    stop, one layer up. The model asks for something, is not told no, and does not
    get it.

    Two spellings, because most views rename nothing::

        arguments: [command, tail]          # offered under the runtime's own names
        arguments: {cmd: command}           # offered as `cmd`, sent as `command`

    An earlier version had `expose` and `hide` as separate fields, on the theory that
    a withheld parameter should be declared so a decision could not be mistaken for
    an oversight. It cost more than it bought: every manifest had to be edited when
    the runtime grew a parameter, and the dangerous direction -- offering something
    the runtime does not accept -- is caught by validating these names anyway.
    """
    name: str
    runtime_tool: str
    description: str = ""
    #: model-facing name -> runtime name. Also the whitelist.
    arguments: dict[str, str] = field(default_factory=dict)
    #: Parameters the model must supply, in model-facing names. Defaults to whatever
    #: the runtime requires among those offered.
    required: tuple[str, ...] = ()

    @property
    def offered(self) -> tuple[str, ...]:
        return tuple(self.arguments)

    def runtime_name_of(self, offered: str) -> str:
        return self.arguments[offered]

    def route(self, args: dict) -> tuple[str, dict]:
        """Translate a call on this view into a runtime call.

        Raises on an argument this view does not offer. Dropping it quietly would
        leave the model believing a setting took effect -- exactly the `max_length`
        defect that motivated compiling the panel, and an earlier version of this
        method did precisely that.
        """
        unknown = set(args) - set(self.arguments)
        if unknown:
            raise ValueError(
                f"{self.name} does not take {', '.join(sorted(unknown))}; "
                f"it takes {', '.join(sorted(self.arguments)) or 'no arguments'}")
        return self.runtime_tool, {self.arguments[k]: v for k, v in args.items()}


def _parse_arguments(name: str, raw) -> dict[str, str]:
    """`[a, b]` or `{model_name: runtime_name}` into one mapping."""
    if raw is None:
        raise PanelError(
            f"{name}: set `arguments` to the parameters this view offers -- a list "
            f"of runtime names, or a mapping of model-facing name to runtime name")
    if isinstance(raw, dict):
        for offered, runtime_arg in raw.items():
            if not isinstance(runtime_arg, str) or not runtime_arg:
                raise PanelError(
                    f"{name}.arguments[{offered!r}] must name a runtime argument")
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if not isinstance(entry, str) or not entry:
                raise PanelError(f"{name}.arguments must be a list of names")
        return {entry: entry for entry in raw}
    raise PanelError(f"{name}.arguments must be a list or a mapping")


def parse_agent_tool(raw: dict) -> AgentToolSpec:
    """Validate a manifest's agent-tool block into a spec (shape only).

    Agreement with the runtime is checked later, by ``compile_panel``: a spec is
    well-formed on its own, but only valid against a particular runtime.
    """
    name = raw.get("name") or ""
    if not name:
        raise PanelError("agent tool has no name")
    return AgentToolSpec(
        name=name,
        runtime_tool=raw.get("runtime_tool") or name,
        description=raw.get("description") or "",
        arguments=_parse_arguments(name, raw.get("arguments")),
        required=tuple(raw.get("required") or ()),
    )


# --------------------------------------------------------------------------- #
#  Compilation
# --------------------------------------------------------------------------- #

def _validate(spec: AgentToolSpec, runtime: RuntimeDeclaration) -> None:
    if spec.runtime_tool not in runtime.tools:
        available = ", ".join(sorted(runtime.tools))
        raise PanelError(
            f"{spec.name} targets runtime tool {spec.runtime_tool!r}, which this "
            f"runtime (v{runtime.version}) does not serve. Available: {available}")

    known = runtime.parameters_of(spec.runtime_tool)
    for offered, runtime_arg in spec.arguments.items():
        if runtime_arg not in known:
            hint = f" (offered as {offered!r})" if runtime_arg != offered else ""
            raise PanelError(
                f"{spec.name} offers {runtime_arg!r}{hint}, which "
                f"{spec.runtime_tool!r} does not accept. This is the defect the "
                f"compiler exists for: the runtime ignores an unknown argument, so "
                f"the model would believe it took effect. "
                f"Accepted: {', '.join(sorted(known))}")
    for name in spec.required:
        if name not in spec.arguments:
            raise PanelError(f"{spec.name} requires {name!r} without offering it")

    # A runtime argument the runtime itself requires must be offered, or the model
    # cannot make a valid call at all.
    unreachable = runtime.required_of(spec.runtime_tool) - set(spec.arguments.values())
    if unreachable:
        raise PanelError(
            f"{spec.name} does not offer {', '.join(sorted(unreachable))}, which "
            f"{spec.runtime_tool!r} requires -- every call would fail")


PANEL_FIELD_KEYS = frozenset({"type", "items", "enum", "default", "description"})


def _portable(schema: dict) -> dict:
    """A parameter's schema reduced to keys every provider accepts."""
    return {k: v for k, v in schema.items() if k in PANEL_FIELD_KEYS}


def _parameters(spec: AgentToolSpec, runtime: RuntimeDeclaration) -> dict:
    """The offered parameters, under the names the model uses."""
    known = runtime.parameters_of(spec.runtime_tool)
    properties = {offered: _portable(known[runtime_arg])
                  for offered, runtime_arg in spec.arguments.items()}
    required = list(spec.required) or [
        offered for offered, runtime_arg in spec.arguments.items()
        if runtime_arg in runtime.required_of(spec.runtime_tool)]
    return {"type": "object", "properties": properties, "required": required}


def compile_panel(specs: "list[AgentToolSpec]", runtime: RuntimeDeclaration,
                  format: str = "openai") -> list[dict]:
    """The panel for these views of this runtime, or raise.

    Raising is deliberate. A panel that fell back to a stale copy when a manifest
    disagreed would reproduce exactly the drift this replaces, and it would do it
    silently -- the run would look governed and the model would be told about tools
    that are not there.
    """
    seen: set[str] = set()
    panel = []
    for spec in specs:
        if spec.name in seen:
            raise PanelError(f"two agent tools are both named {spec.name!r}")
        seen.add(spec.name)
        _validate(spec, runtime)
        panel.append(schemas.render(
            spec.name,
            spec.description or runtime.tools[spec.runtime_tool].get("description", ""),
            _parameters(spec, runtime),
            format))
    return panel
