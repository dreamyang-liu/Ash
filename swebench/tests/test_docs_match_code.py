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

from harness.execution.panel import PANEL_DIR  # noqa: E402

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


def test_claude_md_names_no_config_section_that_no_longer_exists():
    """The `section.key` table it used to check lived in swebench/__main__.py,
    which went with the batch runner. Nothing maps YAML sections any more, so a
    documented `agent.tools` would send a reader to a flag that does not exist."""
    stale = set(re.findall(r'`(agent|execution|dataset|limits|model)\.(\w+)`',
                           CLAUDE_MD.read_text()))
    assert not stale, (
        "CLAUDE.md documents config keys %s, but the YAML/flag mapping was "
        "deleted with the batch runner -- describe fork_eval's arguments instead"
        % sorted("%s.%s" % k for k in stale))


def test_claude_md_does_not_ask_for_hand_synced_tool_lists():
    """That rule described three hand-written copies. The panel is compiled now, and
    asking a reader to edit copies that no longer exist sends them looking for them."""
    assert "three tool views" not in CLAUDE_MD.read_text()


# --------------------------------------------------------------------------- #
#  Secrets
# --------------------------------------------------------------------------- #

def test_no_config_carries_a_literal_secret():
    """A real DeepSeek key (`sk-bfe3fe…`) sat in two marathon configs and was one
    `git push` from being public -- caught by a pre-push scan, not by review.

    Credentials come from the environment; `backends.py` already reads
    AENV_API_KEY / api_key_file, and litellm reads ANTHROPIC_API_KEY when
    `model.api_key` is unset. The heuristics below are the shapes that actually
    appear: a provider-prefixed key, or a long opaque token that is not one of
    the obvious placeholders.
    """
    import re

    placeholder = re.compile(r"^(bench-key-|test-|dummy|fake|xxx|changeme|<)",
                             re.I)
    offenders = []
    for path in sorted((REPO / "swebench" / "configs").rglob("*.y*ml")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r"^\s*(?:api_key|token|secret)\s*:\s*(\S+)", line)
            if not match:
                continue
            value = match.group(1).strip("'\"")
            if placeholder.match(value):
                continue
            if value.startswith(("sk-", "sk_", "ghp_", "xoxb-", "AKIA")) or \
                    len(value) >= 32:
                offenders.append("%s:%d" % (path.relative_to(REPO), number))
    assert not offenders, (
        "literal credentials in %s -- source them from the environment instead"
        % ", ".join(offenders))


# --------------------------------------------------------------------------- #
#  The top-level README
# --------------------------------------------------------------------------- #
#  Nothing guarded it, and it showed: it advertised "7 tool implementations", a
#  `swebench/agent.py` that had become a package and then been deleted, and
#  `ash_sandbox/` at the repository root two moves after it went into sdk/.

TOP_README = REPO / "README.md"


def test_top_readme_tool_table_matches_the_runtime():
    import json

    served = {t["name"] for t in
              json.loads((REPO / "runtime" / "schema" / "tools.json").read_text())["tools"]}
    rows = set(re.findall(r"^\| `(\w+)` \|", TOP_README.read_text(), re.M))
    assert rows == served, (
        "README tool table vs runtime: only in README %s, only in runtime %s"
        % (sorted(rows - served), sorted(served - rows)))


def test_top_readme_project_tree_names_files_that_exist():
    """Nested entries belong to the directory above them, so the tree is walked
    rather than grepped -- a bare `slots/` is `harness/slots/`."""
    block = re.search(r"```\n\.\n(.*?)```", TOP_README.read_text(), re.S)
    assert block, "the README has no project tree"
    parent, missing = "", []
    for line in block.group(1).splitlines():
        entry = re.search(r"^(\s*)[├└]── ([\w/.-]+)", line)
        if not entry:
            continue
        indent, name = len(entry.group(1)), entry.group(2)
        if indent == 0:
            parent = name if name.endswith("/") else ""
            path = name
        else:
            path = parent + name
        if not (REPO / path.rstrip("/")).exists():
            missing.append(path)
    assert not missing, "the README tree names %s, which do not exist" % missing


def test_top_readme_does_not_advertise_deleted_entry_points():
    """`python -m swebench`, the harnesses and the YAML configs are gone. A README
    whose Quick Start does not run is worse than one that says less."""
    text = TOP_README.read_text()
    for gone in ("python -m swebench -c", "--harness ", "swebench/configs/",
                 "swebench/agent"):
        assert gone not in text, "README still advertises %r" % gone
