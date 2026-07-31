"""Contract tests for manifest-defined custom tools (agent/custom_tools.py).

Run with pytest alongside test_tool_contract.py (same pattern/imports).
Covers: manifest validation, argv compilation (injection safety), schema
generation, two-step plan expansion (artifact -> shell), builtin-collision
refusal. No files or network touched; pure in-memory specs.
User instruction: custom tools design / "ok".
"""

import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swebench.agent.custom_tools import (  # noqa: E402
    CUSTOM_TOOL_SPECS,
    ManifestError,
    parse_manifest,
    plan_custom_tool,
    register,
)

SHA = "a" * 64


def make_manifest(**overrides):
    base = {
        "name": "code_complexity",
        "description": "Analyze complexity",
        "binary": {"url": "https://example.com/analyzer", "sha256": SHA},
        "parameters": {
            "file": {"type": "string", "required": True, "map": {"positional": 0}},
            "threshold": {"type": "integer", "default": 10, "map": {"flag": "--threshold"}},
            "verbose": {"type": "boolean", "map": {"flag": "--verbose", "style": "switch"}},
        },
        "timeout": 30,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def clean_registry():
    CUSTOM_TOOL_SPECS.clear()
    yield
    CUSTOM_TOOL_SPECS.clear()


def test_parse_and_schema():
    spec = parse_manifest(make_manifest())
    schema = spec.agent_schema()
    fn = schema["function"]
    assert fn["name"] == "code_complexity"
    assert fn["parameters"]["required"] == ["file"]
    assert fn["parameters"]["properties"]["threshold"]["default"] == 10


def test_argv_compilation_order_and_defaults():
    spec = parse_manifest(make_manifest())
    argv = spec.compile_argv("/bin/analyzer", {"file": "main.py", "verbose": True})
    assert argv == ["/bin/analyzer", "--threshold", "10", "--verbose", "main.py"]
    # switch=False omits the flag entirely
    argv = spec.compile_argv("/bin/analyzer", {"file": "x", "verbose": False})
    assert "--verbose" not in argv


def test_injection_is_inert():
    spec = parse_manifest(make_manifest())
    evil = "x; rm -rf / #"
    argv = spec.compile_argv("/bin/analyzer", {"file": evil})
    # The whole payload stays one argv slot...
    assert argv[-1] == evil
    # ...and shlex.join keeps it a single shell word.
    joined = shlex.join(argv)
    assert shlex.split(joined)[-1] == evil


def test_arg_validation():
    spec = parse_manifest(make_manifest())
    with pytest.raises(ValueError, match="missing required"):
        spec.compile_argv("/b", {})
    with pytest.raises(ValueError, match="unknown parameters"):
        spec.compile_argv("/b", {"file": "x", "nope": 1})
    with pytest.raises(ValueError, match="must be integer"):
        spec.compile_argv("/b", {"file": "x", "threshold": "high"})


def test_path_source_manifest():
    spec = parse_manifest(make_manifest(binary={"path": "/opt/tools/analyzer"}))
    assert spec.path == "/opt/tools/analyzer"
    assert spec.url == ""


@pytest.mark.parametrize(
    "mutate, err",
    [
        ({"name": "Bad Name!"}, "invalid tool name"),
        ({"binary": {"url": "ftp://x", "sha256": SHA}}, "must be http"),
        ({"binary": {"url": "https://x", "sha256": "zz"}}, "sha256"),
        ({"binary": {}}, "exactly one of url/path"),
        ({"binary": {"url": "https://x", "sha256": SHA, "path": "/b"}}, "exactly one of url/path"),
        ({"binary": {"path": "relative/path"}}, "must be absolute"),
        ({"binary": {"path": "/b", "sha256": SHA}}, "only valid with url"),
        ({"timeout": 0}, "timeout"),
        ({"parameters": {"p": {"type": "string", "map": {}}}}, "exactly one"),
        (
            {"parameters": {"p": {"type": "string", "map": {"positional": 0, "flag": "-p"}}}},
            "exactly one",
        ),
        (
            {"parameters": {"p": {"type": "string", "map": {"flag": "bad flag"}}}},
            "invalid flag",
        ),
        (
            {"parameters": {"p": {"type": "string", "map": {"flag": "--p", "style": "switch"}}}},
            "switch style requires boolean",
        ),
    ],
)
def test_manifest_validation_errors(mutate, err):
    with pytest.raises(ManifestError, match=err):
        parse_manifest(make_manifest(**mutate))


def test_plan_expands_to_artifact_then_shell():
    register(parse_manifest(make_manifest()))
    plan = plan_custom_tool("code_complexity", {"file": "main.py"})

    tool, args = plan.artifact_call
    assert tool == "artifact"
    assert args == {"url": "https://example.com/analyzer", "sha256": SHA}

    tool, args = plan.shell_call("/tmp/ash-artifacts/aaaa/artifact")
    assert tool == "shell"
    assert args["timeout"] == 30
    parts = shlex.split(args["command"])
    assert parts[0] == "/tmp/ash-artifacts/aaaa/artifact"
    assert parts[-1] == "main.py"


def test_plan_validates_args_before_download():
    register(parse_manifest(make_manifest()))
    with pytest.raises(ValueError, match="missing required"):
        plan_custom_tool("code_complexity", {})


def test_builtin_collision_refused():
    with pytest.raises(ManifestError, match="collides with a builtin"):
        register(parse_manifest(make_manifest(name="shell")))


def test_unknown_custom_tool():
    with pytest.raises(KeyError):
        plan_custom_tool("nonexistent", {})
