"""Compiling the tool panel a model sees.

The runtime declares what exists; manifests declare what to offer and how; this
module produces the panel. Nothing about the panel is hand-maintained, because a
hand-maintained copy drifts, and the one this replaces had: of seven tools, four
disagreed with the runtime. One of them offered `web_fetch(max_length=…)`, a
parameter the runtime dropped -- so a model could ask for a bound, get no error, and
not be bounded.

Two kinds of declaration, and they are different things:

    AgentToolSpec    a *view* of a runtime tool -- rename it, hide parameters,
                     remap argument names, describe it for the task at hand
    CustomToolSpec   an external binary (toolset.py), expanded into artifact+shell

Compiling validates against the runtime's declaration, which is the whole point:

    1. every agent tool targets a runtime tool that exists
    2. every exposed parameter exists on that tool
    3. hiding a parameter is allowed, and has to be said out loud
    4. names do not collide

Rule 1 is also why there is no route table. A mapping exists where something is
actually mapped; the seven identity entries that used to sit in `BUILTIN_ROUTES`
were an indirection layer carrying nothing.

Rule 3 matters more than it looks. A panel narrower than the runtime is often
correct -- `truncate_mode` and `max_output_bytes` let a model raise its own output
budget, going around the truncation interceptor -- but until it is declared, a
reader cannot tell that decision from an oversight.
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

    ``expose`` is the whitelist of parameters the model may pass, named as the model
    sees them. ``hide`` records parameters deliberately withheld -- redundant for
    the machine, load-bearing for the reader, and checked, so a stale entry cannot
    quietly describe a parameter that no longer exists.

    ``arguments`` renames: ``{"cmd": "command"}`` offers ``cmd`` and calls the
    runtime with ``command``. Renaming is the only transformation -- a value that
    needed rewriting would be policy, and policy belongs in an interceptor where it
    can be tested and traced.
    """
    name: str
    runtime_tool: str
    description: str = ""
    expose: tuple[str, ...] = ()
    hide: tuple[str, ...] = ()
    arguments: dict[str, str] = field(default_factory=dict)
    #: Parameters the model must supply, when the view is stricter than the runtime.
    required: tuple[str, ...] = ()

    def runtime_name_of(self, exposed: str) -> str:
        return self.arguments.get(exposed, exposed)

    def route(self, args: dict) -> tuple[str, dict]:
        """Translate a call on this view into a runtime call.

        Unknown arguments are dropped rather than forwarded: the panel told the
        model what exists, and passing something else through would reach the
        runtime as a silently ignored key -- the `max_length` failure again.
        """
        out = {}
        for key, value in args.items():
            if key in self.arguments or key in self.expose:
                out[self.runtime_name_of(key)] = value
        return self.runtime_tool, out


def parse_agent_tool(raw: dict) -> AgentToolSpec:
    """Validate a manifest's agent-tool block into a spec (shape only).

    Agreement with the runtime is checked later, by ``compile_panel``: a spec is
    well-formed on its own, but only valid against a particular runtime.
    """
    name = raw.get("name") or ""
    if not name:
        raise PanelError("agent tool has no name")
    target = raw.get("runtime_tool") or name
    expose = raw.get("expose")
    hide = tuple(raw.get("hide") or ())
    if expose is None and not hide:
        raise PanelError(
            f"{name}: set `expose` (the parameters to offer) or `hide` (the ones to "
            f"withhold); a view that says neither cannot be told from an oversight")
    arguments = dict(raw.get("arguments") or {})
    for exposed, runtime_arg in arguments.items():
        if not isinstance(runtime_arg, str) or not runtime_arg:
            raise PanelError(f"{name}.arguments[{exposed!r}] must name a runtime argument")
    return AgentToolSpec(
        name=name,
        runtime_tool=target,
        description=raw.get("description") or "",
        expose=tuple(expose) if expose is not None else (),
        hide=hide,
        arguments=arguments,
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
    for exposed in spec.expose:
        runtime_arg = spec.runtime_name_of(exposed)
        if runtime_arg not in known:
            hint = (f" (renamed from {exposed!r})" if runtime_arg != exposed else "")
            raise PanelError(
                f"{spec.name} exposes {runtime_arg!r}{hint}, which "
                f"{spec.runtime_tool!r} does not accept. This is the defect the "
                f"compiler exists for: the runtime ignores an unknown argument, so "
                f"the model would believe it took effect. "
                f"Accepted: {', '.join(sorted(known))}")
    for hidden in spec.hide:
        if spec.runtime_name_of(hidden) not in known:
            raise PanelError(
                f"{spec.name} hides {hidden!r}, which {spec.runtime_tool!r} does not "
                f"have -- a stale entry hides a real mismatch from the next reader")
    for name in spec.required:
        if name not in spec.expose:
            raise PanelError(f"{spec.name} requires {name!r} without exposing it")

    # Accounted for under runtime names, because that is what `known` holds: a view
    # exposing `cmd` -> `command` has accounted for `command`, and comparing the
    # model-facing name would have demanded it be hidden as well as exposed.
    accounted = {spec.runtime_name_of(name) for name in spec.expose} | \
                {spec.runtime_name_of(name) for name in spec.hide}
    unaccounted = set(known) - accounted
    if unaccounted:
        raise PanelError(
            f"{spec.name} says nothing about {', '.join(sorted(unaccounted))} on "
            f"{spec.runtime_tool!r}: list them in `expose` to offer them, or in "
            f"`hide` to withhold them on purpose")


#: JSON-Schema keys a function-calling panel may carry per parameter. Deliberately
#: small: providers disagree about the rest, and one of them rejecting a key means a
#: run that cannot start. `additionalProperties` is the concrete case -- the runtime
#: declares it on `shell.env` (a string->string map), the hand-written panel dropped
#: it without saying so, and a contract test has been enforcing its absence since
#: before that panel was compiled.
PANEL_FIELD_KEYS = frozenset({"type", "items", "enum", "default", "description"})


def _portable(schema: dict) -> dict:
    """A parameter's schema reduced to keys every provider accepts."""
    return {k: v for k, v in schema.items() if k in PANEL_FIELD_KEYS}


def _parameters(spec: AgentToolSpec, runtime: RuntimeDeclaration) -> dict:
    """The exposed subset of the runtime's parameter schema, under model-facing names."""
    known = runtime.parameters_of(spec.runtime_tool)
    properties = {name: _portable(known[spec.runtime_name_of(name)])
                  for name in spec.expose}
    required = list(spec.required) or [
        name for name in spec.expose
        if spec.runtime_name_of(name) in runtime.required_of(spec.runtime_tool)]
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
