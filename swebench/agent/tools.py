"""The tool panel the agent offers — compiled, not written.

Built from the runtime's own declaration (``runtime/schema/tools.json``, produced by
``ash-runtime --dump-schema``) plus a manifest saying what to offer and how. See
docs/TOOL_PANEL.md for the model and the rules.

It used to be a hand-written literal here -- 261 lines of it -- and had drifted from
the runtime on four of seven tools. The worst was ``web_fetch(max_length=…)``, a
parameter the runtime dropped: asking for it returned unbounded output with no error,
so the model believed it had a limit it did not have.

A panel is a file, named by ``tools:`` in config. ``ToolPanel`` bundles the compiled
schema with the views that route calls, because those two must come from the same
manifest and an earlier version let them drift: the panel lived in a module-level
global that ``use_panel()`` mutated, so a process running two agents could only have
one panel between them. Batch mode runs several workers in one process.
"""

from pathlib import Path
from dataclasses import dataclass

from ash_sandbox.panel import compile_panel, load_declaration, parse_agent_tool
from harness.execution.tool_constants import (  # re-exported: moved to the execution plane
    CONTENT_EDIT_COMMANDS, EDIT_COMMANDS, truncate_output)

#: The generic core moved to the execution plane, because the MCP proxy needs it
#: too: it used to hand-write its tool list, with no check against the runtime at
#: all. Re-exported here so `tools:` config, this package and its tests keep
#: working, and because `build_panel` below still belongs to this package -- it
#: pulls in manifest-defined custom tools from this package's registry.
from harness.execution.panel import (  # noqa: F401
    PANEL_DIR,
    RUNTIME_SCHEMA,
    ToolPanel as _BasePanel,
    load_panel as _load_panel,
    read_manifest as _read_manifest,
    resolve_panel,
)


#: What this package offers when nothing says otherwise: the wide panel.
#:
#: Deliberately NOT the execution plane's ``DEFAULT_PANEL``, which is shell plus a
#: file editor. That is the right default for a fresh agent; it is the wrong one
#: here, because every number under results/ was measured against seven tools and
#: silently dropping to two would make old and new runs look comparable when they
#: are not. A narrower surface is a config away (``agent.tools: default``).
DEFAULT_PANEL = "full"


def load_panel(name_or_path: str = DEFAULT_PANEL,
               format: str = "openai",
               registry=None) -> "ToolPanel":
    """:func:`harness.execution.panel.load_panel`, defaulting to *this* package's
    panel and returning this package's :class:`ToolPanel`.

    Only the default and the class differ. Both matter: a bare ``load_panel()``
    here must keep meaning seven tools, and the panel it returns must fall back to
    this package's custom-tool registry.
    """
    panel = _load_panel(name_or_path, format, registry)
    return ToolPanel(schema=panel.schema, views=panel.views,
                     registry=panel.registry)


class ToolPanel(_BasePanel):
    """The execution plane's panel, plus this package's custom-tool default.

    Only difference: an unset registry falls back to the process-wide one, which
    the agent loop has always relied on. The base class treats "no registry" as
    "no custom tools", which is the right default for a proxy serving one sandbox.
    """

    def is_custom_tool(self, name: str) -> bool:
        if self.registry is not None:
            return super().is_custom_tool(name)
        from .custom_tools import DEFAULT_REGISTRY

        return DEFAULT_REGISTRY.is_custom_tool(name)


def build_panel(name_or_path: str = DEFAULT_PANEL,
                custom_tools_dir: "str | None" = None,
                registry: "object | None" = None) -> ToolPanel:
    """The complete panel: compiled views plus manifest-defined custom tools.

    Custom tools come from two places, and both are optional:

    * a ``custom_tools:`` section in the panel manifest, for tools that belong to this
      panel -- one file describes the whole surface a model is offered
    * ``custom_tools_dir``, a directory of one-tool manifests, for a drop-in set shared
      across panels (configs/custom_tools/README.md)

    A name defined in both is an error rather than a silent overwrite: a registry stores
    by name, so whichever loaded second would have won quietly.

    ``registry`` is where they land, and should be the session's -- a sandbox is one
    tool surface, so that is the natural scope. Left out, they land in the process
    default, which two configurations in one process would then share: a manifest loaded
    for the first stayed visible to the second.

    One function because the pieces have to arrive together and did not: the litellm
    harness loaded custom tools and the rollout server did not, so a manifest-defined
    tool existed for one caller and not the other.
    """
    from ash_sandbox.toolset import parse_manifest

    from .custom_tools import (DEFAULT_REGISTRY, custom_agent_schemas,
                               load_custom_tools, register)

    target = registry or DEFAULT_REGISTRY
    manifest = _read_manifest(resolve_panel(name_or_path))
    specs = [parse_agent_tool(t) for t in manifest["agent_tools"]]
    schema = compile_panel(specs, load_declaration(RUNTIME_SCHEMA))

    inline = [parse_manifest(t) for t in (manifest.get("custom_tools") or ())]
    from_dir = load_custom_tools(custom_tools_dir, target)
    clash = {s.name for s in inline} & {s.name for s in from_dir}
    if clash:
        raise ValueError(
            f"custom tool(s) {', '.join(sorted(clash))} are defined both in "
            f"{name_or_path} and in {custom_tools_dir}; a registry keys by name, so "
            f"one would silently replace the other")
    for spec in inline:
        register(spec, target)

    return ToolPanel(schema=schema + custom_agent_schemas(target),
                     views={s.name: s for s in specs}, registry=target)


def tool_summary(name: str, args: dict) -> str:
    """Build a human-readable one-line summary for a tool call (for display)."""
    if name == "shell":
        cmd = args.get("command", "")
        return cmd + (" &" if args.get("background") else "")
    elif name == "grep_files":
        parts = [f"/{args.get('pattern', '')}/"]
        if args.get("path"):
            parts.append(args["path"])
        if args.get("include"):
            parts.append(f"({args['include']})")
        return " ".join(parts)
    elif name == "text_editor":
        cmd = args.get("command", "")
        path = args.get("path", "")
        if cmd == "str_replace":
            preview = args.get("old_str", "")[:40].replace("\n", "\\n")
            return f'{path} [{cmd}] "{preview}"'
        elif cmd == "view":
            vr = args.get("view_range")
            return f"{path} [{vr[0]}:{vr[1]}]" if vr else f"{path} [view]"
        return f"{path} [{cmd}]"
    elif name == "process":
        return f"{args.get('pid', '?')} {args.get('action', '?')}"
    elif name == "web_fetch":
        return args.get("url", "")
    elif name == "web_search":
        return args.get("query", "")
    return args.get("command", "") or args.get("path", "") or str(args)[:80]


