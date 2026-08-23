"""Backend selection (swebench/backends.py).

The SDK's `Pool` was always the abstraction a harness wants; what was missing was
the step that picks one. Every call site named `DockerPool` outright, so the
microVM pool could not be reached without editing code, and the choice was
duplicated in the session and the MCP proxy.

Covered:
- the default stays local Docker (no config = today's behavior)
- each backend builds its own pool type from its own section
- a missing required setting names the setting, not a connection failure
- a mistyped setting is an error, not a silent default
- env vars as a fallback; api_key_file so a key need not sit in a config
- backend_config picks only sandbox-related keys out of a harness config
- no call site names a concrete pool any more
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_sandbox import DockerPool, MicroVMPool, SandboxPool
from swebench.backends import (
    BackendError,
    backend_config,
    backend_name,
    build_pool,
)


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #

def test_no_config_is_local_docker():
    """The default must stay what every existing config expects."""
    assert backend_name({}) == "docker"
    assert isinstance(build_pool({}), DockerPool)


def test_runtime_bin_reaches_the_docker_pool():
    pool = build_pool({"runtime_bin": "/tmp/ash-runtime"})
    assert pool.runtime_bin == "/tmp/ash-runtime"


def test_an_override_beats_the_config():
    """AshSession(runtime_bin=…) and --runtime-bin still win."""
    pool = build_pool({"runtime_bin": "/from/config"}, runtime_bin="/from/flag")
    assert pool.runtime_bin == "/from/flag"


def test_a_none_override_does_not_erase_the_config():
    pool = build_pool({"runtime_bin": "/from/config"}, runtime_bin=None)
    assert pool.runtime_bin == "/from/config"


# --------------------------------------------------------------------------- #
#  Each backend
# --------------------------------------------------------------------------- #

def test_microvm_builds_from_its_own_section():
    pool = build_pool({
        "backend": "microvm",
        "microvm": {"server_url": "http://127.0.0.1:8000",
                    "template": "ash-base", "sandbox_ttl": 900},
    })
    assert isinstance(pool, MicroVMPool)
    assert pool.default_template == "ash-base"
    assert pool.sandbox_ttl == 900


def test_k8s_builds_from_its_own_section():
    pool = build_pool({
        "backend": "k8s",
        "k8s": {"control_plane_url": "http://cp", "gateway_url": "http://gw"},
    })
    assert isinstance(pool, SandboxPool)


def test_backend_name_is_case_and_space_insensitive():
    assert backend_name({"backend": "  MicroVM "}) == "microvm"


# --------------------------------------------------------------------------- #
#  Misconfiguration is reported, never guessed
# --------------------------------------------------------------------------- #

def test_a_missing_setting_names_the_setting():
    """Without this the microVM pool would build and fail later as a connection
    error, which reads as "the host is down" rather than "you did not say where"."""
    with pytest.raises(BackendError, match="microvm.server_url"):
        build_pool({"backend": "microvm"})


def test_k8s_names_every_missing_setting_at_once():
    with pytest.raises(BackendError, match="gateway_url"):
        build_pool({"backend": "k8s", "k8s": {"control_plane_url": "http://cp"}})


def test_an_unknown_backend_lists_the_known_ones():
    with pytest.raises(BackendError, match="docker, k8s, microvm"):
        build_pool({"backend": "podman"})


def test_a_mistyped_setting_is_an_error_not_a_silent_default():
    """`tempalte: ash-base` would otherwise spawn the wrong image and look like
    a broken snapshot."""
    with pytest.raises(BackendError, match="tempalte"):
        build_pool({"backend": "microvm",
                    "microvm": {"server_url": "http://x", "tempalte": "oops"}})


def test_a_non_mapping_section_is_an_error():
    with pytest.raises(BackendError, match="must be a mapping"):
        build_pool({"backend": "microvm", "microvm": "http://x"})


# --------------------------------------------------------------------------- #
#  Credentials
# --------------------------------------------------------------------------- #

def _sent_api_key(pool) -> str:
    """The key as the server will see it — the X-API-KEY header, which is what
    AgentENV validates (a Bearer Authorization header is not)."""
    return pool._client.headers.get("X-API-KEY", "")


def test_env_vars_supply_what_the_config_omits(monkeypatch):
    monkeypatch.setenv("AENV_SERVER_URL", "http://from-env:8000")
    monkeypatch.setenv("AENV_API_KEY", "key-from-env")
    pool = build_pool({"backend": "microvm"})
    assert pool.server_url == "http://from-env:8000"
    assert _sent_api_key(pool) == "key-from-env"


def test_an_api_key_can_live_in_a_file(tmp_path: Path, monkeypatch):
    """A key written into a YAML config tends to reach a commit."""
    monkeypatch.delenv("AENV_API_KEY", raising=False)
    key_file = tmp_path / "aenv-key"
    key_file.write_text("secret-from-file\n")
    pool = build_pool({"backend": "microvm",
                       "microvm": {"server_url": "http://x",
                                   "api_key_file": str(key_file)}})
    assert _sent_api_key(pool) == "secret-from-file"   # trailing newline stripped


def test_a_missing_key_file_is_reported():
    with pytest.raises(BackendError, match="api_key_file not found"):
        build_pool({"backend": "microvm",
                    "microvm": {"server_url": "http://x",
                                "api_key_file": "/nope/key"}})


def test_an_explicit_key_beats_a_file(tmp_path: Path):
    key_file = tmp_path / "k"
    key_file.write_text("from-file")
    pool = build_pool({"backend": "microvm",
                       "microvm": {"server_url": "http://x",
                                   "api_key": "inline", "api_key_file": str(key_file)}})
    assert _sent_api_key(pool) == "inline"


# --------------------------------------------------------------------------- #
#  Harness plumbing
# --------------------------------------------------------------------------- #

def test_backend_config_keeps_only_sandbox_settings():
    """A session should not depend on model or dataset keys."""
    picked = backend_config({
        "backend": "microvm",
        "runtime_bin": "/tmp/rt",
        "microvm": {"server_url": "http://x"},
        "model": "claude-sonnet-4-5",       # not a sandbox concern
        "step_limit": 250,
        "workers": 8,
    })
    assert picked == {"backend": "microvm", "runtime_bin": "/tmp/rt",
                      "microvm": {"server_url": "http://x"}}


def test_backend_config_drops_unset_values():
    assert backend_config({"backend": None, "runtime_bin": None}) == {}


def test_no_call_site_names_a_concrete_pool():
    """The point of this module: one place decides, so switching a backend is a
    config change rather than an edit in several files."""
    root = Path(__file__).resolve().parents[1]
    for name in ("sandbox.py", "mcp_server.py"):
        source = (root / name).read_text()
        assert "DockerPool(" not in source, f"{name} still constructs a DockerPool"
        assert "build_pool(" in source, f"{name} should build its pool via backends.py"


def test_every_harness_passes_its_backend_config_through():
    """A harness that forgot this would silently keep running on Docker while
    the config said otherwise."""
    root = Path(__file__).resolve().parents[1] / "harnesses"
    # Enumerated, not listed by name: a hardcoded list silently stops covering a
    # harness that gets added, and breaks when one is removed.
    for path in sorted(root.glob("*.py")):
        source = path.read_text()
        if "AshSession(" not in source:
            continue                      # this harness does not open one
        assert "backend=backend_config(c)" in source, \
            f"{path.name} builds an AshSession without passing the backend config"


def test_microvm_accepts_from_image():
    """`from_image` selects the cold-start path for OCI image references.

    Unknown keys are rejected rather than ignored, so this has to be declared
    or a config that sets it would look like it worked.
    """
    from swebench.backends import build_pool

    pool = build_pool({"backend": "microvm",
                       "microvm": {"server_url": "http://127.0.0.1:8000",
                                   "from_image": True}})
    assert pool.supports_cold_start()
