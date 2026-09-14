from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from swebench.rollout_groups.environment_resolver import (
    AgentEnvOCIResolver,
    AgentEnvOCIResolverConfig,
)
from swebench.rollout_groups.protocol import EnvironmentRef


def _config(tmp_path, *, runtime_contents=b"runtime-v1", **overrides):
    runtime = tmp_path / "ash-runtime"
    runtime.write_bytes(runtime_contents)
    value = {
        "allowed_registries": ["docker.io", "ghcr.io"],
        "runtime_artifact": str(runtime),
        "resource_profiles": {
            "standard": {"cpu": 2, "memory_mib": 4096},
        },
    }
    value.update(overrides)
    return AgentEnvOCIResolverConfig.from_dict(value)


def _ref(**overrides):
    value = {
        "kind": "image",
        "id": "docker.io/example/task-env",
        "revision": "sha256:" + "a" * 64,
        "resource_profile": "standard",
    }
    value.update(overrides)
    return EnvironmentRef(**value)


def _completed(args, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_resolver_rejects_mutable_or_untrusted_sources(tmp_path):
    resolver = AgentEnvOCIResolver(_config(tmp_path), command_runner=lambda *_: None)

    with pytest.raises(ValueError, match="sha256 digest"):
        resolver.validate(_ref(revision="latest"))
    with pytest.raises(ValueError, match="mutable tag"):
        resolver.validate(_ref(id="docker.io/example/task-env:latest"))
    with pytest.raises(ValueError, match="not allowlisted"):
        resolver.validate(_ref(id="untrusted.example/task-env"))
    with pytest.raises(ValueError, match="resource profile"):
        resolver.validate(_ref(resource_profile="large"))


def test_resolver_reuses_existing_runtime_snapshot(tmp_path):
    calls = []

    def run(args, _timeout):
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(
                args,
                json.dumps([{"snapshotID": "snap-1", "names": [expected_name]}]),
            )
        raise AssertionError(args)

    config = _config(tmp_path)
    resolver = AgentEnvOCIResolver(config, command_runner=run)
    expected_name = resolver._snapshot_name(_ref())

    entry = resolver.resolve(_ref())

    assert entry.spawn_ref == expected_name
    assert len(calls) == 1


def test_resolver_releases_prepared_snapshot_by_deterministic_alias(tmp_path):
    calls = []
    snapshots = []

    def run(args, _timeout):
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(args, json.dumps(snapshots))
        if args[1:3] == ["template", "delete"]:
            snapshots.clear()
            return _completed(args)
        raise AssertionError(args)

    resolver = AgentEnvOCIResolver(_config(tmp_path), command_runner=run)
    snapshot_name = resolver._snapshot_name(_ref())
    snapshots.append({"snapshotID": "snap-1", "names": [snapshot_name]})

    assert resolver.release(_ref()) is True
    assert ["aenv", "template", "delete", snapshot_name] in calls
    assert resolver.release(_ref()) is False


def test_resolver_prepares_runtime_snapshot_and_cleans_builder(tmp_path):
    calls = []
    snapshots = []

    def run(args, _timeout):
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(args, json.dumps(snapshots))
        if args[1] == "start" and args[2].startswith("ash-rollout-build-"):
            return _completed(args, "sandbox-1\n")
        if args[1:3] == ["snapshot", "create"]:
            name = args[args.index("--name") + 1]
            snapshots.append({"snapshotID": "snapshot-1", "names": [name]})
        return _completed(args)

    config = _config(tmp_path)
    resolver = AgentEnvOCIResolver(config, command_runner=run, sleep=lambda _: None)
    expected_name = resolver._snapshot_name(_ref())

    entry = resolver.resolve(_ref())

    assert entry.spawn_ref == expected_name
    pull = next(args for args in calls if args[1] == "pull")
    assert pull[2] == "docker.io/example/task-env@sha256:" + "a" * 64
    assert pull[pull.index("--cpu") + 1] == "2"
    assert pull[pull.index("--memory") + 1] == "4096"
    assert any(args[1:3] == ["upload", "sandbox-1"] for args in calls)
    assert any(args[1:3] == ["snapshot", "create"] for args in calls)
    assert any(args[1:3] == ["delete", "sandbox-1"] for args in calls)
    assert any(args[1:3] == ["template", "delete"] for args in calls)


def test_runtime_upload_retries_until_guest_can_see_file(tmp_path):
    calls = []
    snapshots = []
    failed_verify = False

    def run(args, _timeout):
        nonlocal failed_verify
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(args, json.dumps(snapshots))
        if args[1] == "start":
            return _completed(args, "sandbox-1\n")
        if args[1:5] == ["exec", "sandbox-1", "test", "-s"] and not failed_verify:
            failed_verify = True
            return _completed(args, stderr="not visible yet", returncode=1)
        if args[1:3] == ["snapshot", "create"]:
            name = args[args.index("--name") + 1]
            snapshots.append({"snapshotID": "snapshot-1", "names": [name]})
        return _completed(args)

    resolver = AgentEnvOCIResolver(
        _config(tmp_path, runtime_upload_retry_seconds=0),
        command_runner=run,
        sleep=lambda _: None,
    )

    resolver.resolve(_ref())

    uploads = [args for args in calls if args[1:3] == ["upload", "sandbox-1"]]
    verifies = [args for args in calls if args[1:5] == ["exec", "sandbox-1", "test", "-s"]]
    assert len(uploads) == 2
    assert len(verifies) == 2


def test_runtime_upload_fails_after_bounded_visibility_retries(tmp_path):
    calls = []

    def run(args, _timeout):
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(args, "[]")
        if args[1] == "start":
            return _completed(args, "sandbox-1\n")
        if args[1:5] == ["exec", "sandbox-1", "test", "-s"]:
            return _completed(args, stderr="missing", returncode=1)
        return _completed(args)

    resolver = AgentEnvOCIResolver(
        _config(
            tmp_path,
            runtime_upload_attempts=2,
            runtime_upload_retry_seconds=0,
        ),
        command_runner=run,
        sleep=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        resolver.resolve(_ref())

    assert len([args for args in calls if args[1:3] == ["upload", "sandbox-1"]]) == 2
    assert ["aenv", "delete", "sandbox-1"] in calls


def test_resolver_rejects_invalid_agentenv_catalog_json(tmp_path):
    def run(args, _timeout):
        if args[1] == "snapshot":
            return _completed(args, "not-json")
        raise AssertionError(args)

    resolver = AgentEnvOCIResolver(_config(tmp_path), command_runner=run)

    with pytest.raises(RuntimeError, match="snapshot list returned invalid JSON"):
        resolver.resolve(_ref())


def test_snapshot_cache_key_changes_with_runtime_binary(tmp_path):
    first = AgentEnvOCIResolver(_config(tmp_path), command_runner=lambda *_: None)
    first_name = first._snapshot_name(_ref())
    second = AgentEnvOCIResolver(
        _config(tmp_path, runtime_contents=b"runtime-v2"),
        command_runner=lambda *_: None,
    )

    assert hashlib.sha256(b"runtime-v1").hexdigest() != hashlib.sha256(b"runtime-v2").hexdigest()
    assert first_name != second._snapshot_name(_ref())


def test_snapshot_cache_key_changes_with_resource_profile_contents(tmp_path):
    first = AgentEnvOCIResolver(_config(tmp_path), command_runner=lambda *_: None)
    first_name = first._snapshot_name(_ref())
    second = AgentEnvOCIResolver(
        _config(
            tmp_path,
            resource_profiles={"standard": {"cpu": 4, "memory_mib": 8192}},
        ),
        command_runner=lambda *_: None,
    )

    assert first_name != second._snapshot_name(_ref())


def test_pull_failure_still_attempts_exact_template_cleanup(tmp_path):
    calls = []

    def run(args, _timeout):
        calls.append(args)
        if args[1:4] == ["snapshot", "list", "--output"]:
            return _completed(args, "[]")
        if args[1] == "pull":
            return _completed(args, stderr="pull timed out", returncode=1)
        return _completed(args)

    resolver = AgentEnvOCIResolver(_config(tmp_path), command_runner=run)

    with pytest.raises(RuntimeError, match="pull timed out"):
        resolver.resolve(_ref())

    pull = next(args for args in calls if args[1] == "pull")
    temporary_name = pull[pull.index("--name") + 1]
    assert ["aenv", "template", "delete", temporary_name] in calls


def test_default_runner_uses_private_cli_credentials(tmp_path, monkeypatch):
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key\n", encoding="utf-8")
    captured = {}

    def run(args, **kwargs):
        credentials = (
            Path(kwargs["env"]["XDG_CONFIG_HOME"]) / "aenv" / "credentials"
        )
        captured.update(
            args=args,
            credentials=credentials.read_text(encoding="utf-8"),
            mode=oct(credentials.stat().st_mode & 0o777),
            inherited_home=os.environ.get("XDG_CONFIG_HOME"),
        )
        return _completed(args)

    monkeypatch.setattr(subprocess, "run", run)
    resolver = AgentEnvOCIResolver(
        _config(tmp_path),
        aenv_server_url="http://agentenv:8000",
        aenv_api_key_file=key_file,
    )

    resolver._run_subprocess(["aenv", "snapshot", "list"], 10)

    assert captured["args"] == ["aenv", "snapshot", "list"]
    assert captured["credentials"] == (
        'url = "http://agentenv:8000"\napi_key = "secret-key"\n'
    )
    assert captured["mode"] == "0o600"
    assert captured["inherited_home"] is None


def test_default_runner_rejects_missing_cli_key(tmp_path):
    resolver = AgentEnvOCIResolver(
        _config(tmp_path),
        aenv_server_url="http://agentenv:8000",
        aenv_api_key_file=tmp_path / "missing-key",
    )

    with pytest.raises(FileNotFoundError):
        resolver._run_subprocess(["aenv", "snapshot", "list"], 10)
