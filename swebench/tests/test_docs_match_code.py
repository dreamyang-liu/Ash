"""The documented examples have to parse.

These files are the format documentation -- `configs/custom_tools/README.md` is the only
place a manifest's fields are described -- and both had drifted: the custom-tool example
used `sha256: "ab34..."`, a placeholder the validator rejects, so anyone copying it got
an error from a file that was supposed to teach them the format. `swebench/README.md`
still described a `bash` tool renamed several commits earlier.

Prose cannot be checked, but a fenced example can be run, and a named file can be looked
for. That covers the two ways these went wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

from ash_sandbox.toolset import parse_manifest  # noqa: E402

from swebench.agent.tools import PANEL_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _yaml_blocks(path: Path) -> list[str]:
    return re.findall(r"```yaml\n(.*?)```", path.read_text(), re.S)


def test_the_custom_tool_example_parses():
    """It is the only documentation of the manifest fields, so a reader copying it must
    get a working tool rather than a validation error."""
    blocks = _yaml_blocks(REPO / "configs" / "custom_tools" / "README.md")
    assert blocks, "the custom tool README has no example left"
    spec = parse_manifest(yaml.safe_load(blocks[0]))
    assert spec.name and spec.params, "the example declares nothing"


def test_the_tool_panel_examples_parse():
    """docs/TOOL_PANEL.md shows both halves of a manifest. Each block is checked with
    the parser that will read the real thing."""
    from ash_sandbox.panel import parse_agent_tool

    for block in _yaml_blocks(REPO / "docs" / "TOOL_PANEL.md"):
        raw = yaml.safe_load(block)
        if not isinstance(raw, dict):
            continue
        for entry in raw.get("agent_tools") or ():
            assert parse_agent_tool(entry).arguments, f"{entry.get('name')} offers nothing"
        for entry in raw.get("custom_tools") or ():
            assert parse_manifest(entry).name


def test_the_readme_layout_names_files_that_exist():
    """A layout listing a file that is gone sends a reader looking for it. This is how
    the previous README came to name five files that had been deleted."""
    layout = re.search(r"```\nswebench/\n(.*?)```",
                       (REPO / "swebench" / "README.md").read_text(), re.S)
    assert layout, "the README has no layout block"
    # Indented entries belong to the directory above them, so the tree has to be
    # walked rather than grepped: a bare "pipeline.py" is agent/pipeline.py.
    parent = ""
    missing = []
    for line in layout.group(1).splitlines():
        entry = re.search(r"[├└]── ([\w.]+/?)", line)
        if not entry:
            continue
        name = entry.group(1)
        indented = re.match(r"^[│ ]{4,}", line) is not None
        path = REPO / "swebench" / ((parent + name) if indented else name)
        if not indented:
            parent = name if name.endswith("/") else ""
        if name == "...":
            continue
        if not path.exists() and not (path.parent / path.name.rstrip("/")).exists():
            missing.append(str(path.relative_to(REPO)))
    assert not missing, f"the README names {missing}, which do not exist"


def test_the_readme_does_not_describe_a_bash_tool():
    """bash_only offers a view named `shell`; the `bash` name went with the route table.
    A prompt or a doc still saying `bash` points at a tool that is not there."""
    text = (REPO / "swebench" / "README.md").read_text()
    assert "`bash` tool" not in text
    assert "single `bash`" not in text


def test_every_shipped_panel_is_mentioned_where_panels_are_documented():
    """A panel nobody documents is one nobody knows to pass to `tools:`."""
    doc = (REPO / "docs" / "TOOL_PANEL.md").read_text()
    for panel in sorted(p.stem for p in PANEL_DIR.glob("*.y*ml")):
        assert panel in doc, f"{panel} is shipped but not mentioned in docs/TOOL_PANEL.md"
