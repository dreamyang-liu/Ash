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


# --------------------------------------------------------------------------- #
#  CLAUDE.md
# --------------------------------------------------------------------------- #
#  It is the first thing an agent reads about this repo, so a wrong claim there
#  propagates. It had three: "exactly 6 tools" in one place, "7 tools" in another,
#  and a rule asking three hand-written tool lists to be edited together, which the
#  panel compiler had already replaced.

CLAUDE_MD = REPO / "CLAUDE.md"


def test_claude_md_names_the_right_number_of_tools():
    """The count was wrong in two places and disagreed with itself."""
    import json

    served = len(json.loads((REPO / "runtime" / "schema" / "tools.json").read_text())["tools"])
    text = CLAUDE_MD.read_text()
    assert f"serves **{served} tools**" in text, f"the runtime serves {served}"
    assert f"{served} tool implementations" in text


def test_claude_md_lists_every_tool_the_runtime_serves():
    """A tool missing from the table is one nobody knows exists: `artifact` and
    `wait_for_events` were both absent.

    Scoped to the table, not the whole file. Checking "mentioned anywhere" passed while
    the row was deleted, because these names also appear in the layout and the prose --
    a mutation caught that."""
    import json

    served = {t["name"] for t in
              json.loads((REPO / "runtime" / "schema" / "tools.json").read_text())["tools"]}
    table = re.search(r"\n\| Tool\s+\| Purpose.*?\n\n", CLAUDE_MD.read_text(), re.S)
    assert table, "the tool table is gone from CLAUDE.md"
    rows = set(re.findall(r"^\| `(\w+)`", table.group(0), re.M))
    assert rows == served, (
        f"the table and the runtime disagree: only in the table {sorted(rows - served)}, "
        f"only in the runtime {sorted(served - rows)}")


def test_claude_md_paths_exist():
    """A named path that has moved sends a reader hunting."""
    named = set(re.findall(r'`((?:swebench|runtime|sdk|docs)/[\w/.]+)`', CLAUDE_MD.read_text()))
    missing = [p for p in sorted(named) if not (REPO / p).exists()]
    assert not missing, f"CLAUDE.md names {missing}, which do not exist"


def test_claude_md_config_keys_exist():
    """Every `section.key` it documents has to be in the flag/section table, or a
    reader sets something that is silently ignored -- which is how `custom_tools_dir`
    spent months reaching AgentConfig from nowhere."""
    mapping = (REPO / "swebench" / "__main__.py").read_text()
    claimed = set(re.findall(r'`(agent|execution|dataset|limits|model)\.(\w+)`',
                             CLAUDE_MD.read_text()))
    missing = [f"{s}.{k}" for s, k in sorted(claimed)
               if f'("{s}", "{k}")' not in mapping]
    assert not missing, f"CLAUDE.md documents {missing}, which the CLI does not map"


def test_claude_md_does_not_ask_for_hand_synced_tool_lists():
    """That rule described three hand-written copies. The panel is compiled now, and
    asking a reader to edit copies that no longer exist sends them looking for them."""
    assert "three tool views" not in CLAUDE_MD.read_text()
