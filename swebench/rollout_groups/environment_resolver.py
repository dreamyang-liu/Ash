"""Resolve immutable OCI references into AgentENV runtime-ready snapshots.

The rollout protocol deliberately carries an OCI repository and digest rather
than an AgentENV-local template name.  This module owns the provider-specific
preparation step: import the OCI image, inject ``ash-runtime``, snapshot the
running sandbox, and reuse that snapshot for later rollout groups.

Dynamic OCI preparation is opt-in.  Registries and resource profiles are
deployment policy, and the source image must be pinned by a sha256 digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .environment_catalog import EnvironmentCatalogEntry
from .protocol import EnvironmentRef


_SHA256 = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")
_SAFE_NAME = re.compile(r"[^a-z0-9-]+")


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class AgentEnvResourceProfile:
    cpu: int
    memory_mib: int

    @classmethod
    def from_dict(cls, value: Any, name: str) -> "AgentEnvResourceProfile":
        if not isinstance(value, dict):
            raise ValueError(f"resource profile {name!r} must be an object")
        unknown = set(value) - {"cpu", "memory_mib"}
        if unknown:
            raise ValueError(
                f"resource profile {name!r} contains unknown fields: {sorted(unknown)}"
            )
        cpu = value.get("cpu")
        memory_mib = value.get("memory_mib")
        if not isinstance(cpu, int) or isinstance(cpu, bool) or cpu <= 0:
            raise ValueError(f"resource profile {name!r}.cpu must be > 0")
        if (
            not isinstance(memory_mib, int)
            or isinstance(memory_mib, bool)
            or memory_mib <= 0
        ):
            raise ValueError(f"resource profile {name!r}.memory_mib must be > 0")
        return cls(cpu=cpu, memory_mib=memory_mib)


@dataclass(frozen=True)
class AgentEnvOCIResolverConfig:
    allowed_registries: frozenset[str]
    runtime_artifact: Path
    resource_profiles: dict[str, AgentEnvResourceProfile]
    runtime_install_path: str = "/tmp/ash-runtime"
    runtime_port: int = 3000
    cache_prefix: str = "ash-rollout"
    build_timeout_seconds: int = 900
    sandbox_timeout_seconds: int = 900
    runtime_upload_attempts: int = 3
    runtime_upload_retry_seconds: float = 1.0
    aenv_bin: str = "aenv"

    @classmethod
    def from_dict(cls, value: Any) -> "AgentEnvOCIResolverConfig":
        if not isinstance(value, dict):
            raise ValueError("AgentENV OCI resolver config must be an object")
        allowed = {
            "allowed_registries",
            "runtime_artifact",
            "resource_profiles",
            "runtime_install_path",
            "runtime_port",
            "cache_prefix",
            "build_timeout_seconds",
            "sandbox_timeout_seconds",
            "runtime_upload_attempts",
            "runtime_upload_retry_seconds",
            "aenv_bin",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"AgentENV OCI resolver config contains unknown fields: {sorted(unknown)}"
            )
        raw_registries = value.get("allowed_registries")
        if not isinstance(raw_registries, list) or not raw_registries:
            raise ValueError("allowed_registries must be a non-empty list")
        registries = frozenset(
            _required_string(item, "allowed_registries entry").lower()
            for item in raw_registries
        )
        raw_profiles = value.get("resource_profiles")
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise ValueError("resource_profiles must be a non-empty object")
        profiles = {
            _required_string(name, "resource profile name"): AgentEnvResourceProfile.from_dict(
                profile, str(name)
            )
            for name, profile in raw_profiles.items()
        }
        config = cls(
            allowed_registries=registries,
            runtime_artifact=Path(
                _required_string(value.get("runtime_artifact"), "runtime_artifact")
            ),
            resource_profiles=profiles,
            runtime_install_path=_required_string(
                value.get("runtime_install_path", "/tmp/ash-runtime"),
                "runtime_install_path",
            ),
            runtime_port=_positive_int(value.get("runtime_port", 3000), "runtime_port"),
            cache_prefix=_safe_cache_prefix(value.get("cache_prefix", "ash-rollout")),
            build_timeout_seconds=_positive_int(
                value.get("build_timeout_seconds", 900), "build_timeout_seconds"
            ),
            sandbox_timeout_seconds=_positive_int(
                value.get("sandbox_timeout_seconds", 900),
                "sandbox_timeout_seconds",
            ),
            runtime_upload_attempts=_positive_int(
                value.get("runtime_upload_attempts", 3),
                "runtime_upload_attempts",
            ),
            runtime_upload_retry_seconds=_nonnegative_number(
                value.get("runtime_upload_retry_seconds", 1.0),
                "runtime_upload_retry_seconds",
            ),
            aenv_bin=_required_string(value.get("aenv_bin", "aenv"), "aenv_bin"),
        )
        if not config.runtime_artifact.is_file():
            raise ValueError(
                f"runtime_artifact does not exist: {config.runtime_artifact}"
            )
        if not config.runtime_install_path.startswith("/"):
            raise ValueError("runtime_install_path must be absolute")
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentEnvOCIResolverConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{name} must be >= 0")
    return float(value)


def _safe_cache_prefix(value: Any) -> str:
    prefix = _required_string(value, "cache_prefix").lower()
    prefix = _SAFE_NAME.sub("-", prefix).strip("-")
    if not prefix:
        raise ValueError("cache_prefix must contain a letter or digit")
    return prefix[:32]


CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


class AgentEnvOCIResolver:
    """Materialize trusted, digest-pinned OCI images for MicroVMPool.

    The returned ``spawn_ref`` is a deterministic AgentENV snapshot alias.  A
    process lock prevents duplicate preparation in one service; the AgentENV
    snapshot catalog supplies the persistent cache across service restarts.
    """

    def __init__(
        self,
        config: AgentEnvOCIResolverConfig,
        *,
        command_runner: CommandRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        aenv_server_url: str | None = None,
        aenv_api_key: str | None = None,
        aenv_api_key_file: str | Path | None = None,
    ) -> None:
        if aenv_api_key and aenv_api_key_file:
            raise ValueError("set aenv_api_key or aenv_api_key_file, not both")
        self.config = config
        self._command_runner = command_runner or self._run_subprocess
        self._sleep = sleep
        self._lock = threading.RLock()
        self._aenv_server_url = aenv_server_url
        self._aenv_api_key = aenv_api_key
        self._aenv_api_key_file = (
            None if aenv_api_key_file is None else Path(aenv_api_key_file)
        )
        self._runtime_digest = hashlib.sha256(
            config.runtime_artifact.read_bytes()
        ).hexdigest()

    def validate(self, ref: EnvironmentRef) -> None:
        if ref.kind != "image":
            raise ValueError("dynamic OCI resolution requires environment_ref.kind image")
        if not _SHA256.fullmatch(ref.revision):
            raise ValueError(
                "dynamic OCI environment_ref.revision must be a sha256 digest"
            )
        if "://" in ref.id or "@" in ref.id:
            raise ValueError(
                "environment_ref.id must be an OCI repository without a URL scheme or digest"
            )
        if (
            ref.id.startswith(("-", "/"))
            or any(character.isspace() for character in ref.id)
            or ".." in ref.id.split("/")
        ):
            raise ValueError("environment_ref.id is not a valid OCI repository")
        leaf = ref.id.rsplit("/", 1)[-1]
        if ":" in leaf:
            raise ValueError(
                "environment_ref.id must not contain a mutable tag; use revision for the digest"
            )
        registry = _registry(ref.id)
        if registry not in self.config.allowed_registries:
            raise ValueError(f"OCI registry {registry!r} is not allowlisted")
        if ref.resource_profile not in self.config.resource_profiles:
            raise ValueError(
                f"unknown AgentENV resource profile {ref.resource_profile!r}"
            )

    def resolve(self, ref: EnvironmentRef) -> EnvironmentCatalogEntry:
        self.validate(ref)
        snapshot_name = self._snapshot_name(ref)
        with self._lock:
            if self._snapshot_exists(snapshot_name):
                return EnvironmentCatalogEntry(ref=ref, spawn_ref=snapshot_name)
            self._prepare_snapshot(ref, snapshot_name)
            if not self._snapshot_exists(snapshot_name):
                raise RuntimeError(
                    f"AgentENV did not publish prepared snapshot {snapshot_name!r}"
                )
        return EnvironmentCatalogEntry(ref=ref, spawn_ref=snapshot_name)

    def release(self, ref: EnvironmentRef) -> bool:
        """Delete this resolver's prepared snapshot for ``ref``.

        The caller must first prove that no rollout still uses the environment.
        AgentENV snapshots are durable objects, so they are deleted explicitly;
        source-image commits remain under AgentENV's lease-aware image-cache GC
        instead of being removed from disk directly.
        """
        self.validate(ref)
        snapshot_name = self._snapshot_name(ref)
        with self._lock:
            if not self._snapshot_exists(snapshot_name):
                return False
            self._run(
                [self.config.aenv_bin, "template", "delete", snapshot_name],
                timeout=self.config.sandbox_timeout_seconds,
            )
            if self._snapshot_exists(snapshot_name):
                raise RuntimeError(
                    f"AgentENV retained released snapshot {snapshot_name!r}"
                )
        return True

    def _snapshot_name(self, ref: EnvironmentRef) -> str:
        profile = self.config.resource_profiles[ref.resource_profile]
        identity = json.dumps(
            {
                "source": f"{ref.id}@{ref.revision.lower()}",
                "resource_profile": {
                    "name": ref.resource_profile,
                    "cpu": profile.cpu,
                    "memory_mib": profile.memory_mib,
                },
                "runtime_sha256": self._runtime_digest,
                "runtime_install_path": self.config.runtime_install_path,
                "runtime_port": self.config.runtime_port,
                "schema": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(identity).hexdigest()[:24]
        return f"{self.config.cache_prefix}-{digest}"

    def _snapshot_exists(self, name: str) -> bool:
        result = self._run(
            [self.config.aenv_bin, "snapshot", "list", "--output", "json"]
        )
        try:
            snapshots = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("aenv snapshot list returned invalid JSON") from exc
        if not isinstance(snapshots, list):
            raise RuntimeError("aenv snapshot list returned a non-list response")
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            names = snapshot.get("names") or snapshot.get("aliases") or []
            if snapshot.get("snapshot_id") == name or snapshot.get("snapshotID") == name:
                return True
            if isinstance(names, list) and name in names:
                return True
        return False

    def _prepare_snapshot(self, ref: EnvironmentRef, snapshot_name: str) -> None:
        profile = self.config.resource_profiles[ref.resource_profile]
        source = f"{ref.id}@{ref.revision.lower()}"
        suffix = uuid.uuid4().hex[:8]
        base_name = f"{self.config.cache_prefix}-build-{suffix}"
        sandbox_id: str | None = None
        template_created = False
        try:
            # A timed-out pull may still have created the named template, so
            # cleanup should attempt the exact, per-build alias even when the
            # CLI exits unsuccessfully.
            template_created = True
            self._run(
                [
                    self.config.aenv_bin,
                    "pull",
                    source,
                    "--name",
                    base_name,
                    "--cpu",
                    str(profile.cpu),
                    "--memory",
                    str(profile.memory_mib),
                    "--timeout",
                    str(self.config.build_timeout_seconds),
                ],
                timeout=self.config.build_timeout_seconds + 30,
            )
            started = self._run(
                [
                    self.config.aenv_bin,
                    "start",
                    base_name,
                    "--timeout",
                    str(self.config.sandbox_timeout_seconds),
                    "--detach",
                ]
            )
            sandbox_id = _last_nonempty_line(started.stdout, "aenv start")
            install_path = self.config.runtime_install_path
            self._run(
                [
                    self.config.aenv_bin,
                    "exec",
                    sandbox_id,
                    "mkdir",
                    "-p",
                    str(Path(install_path).parent),
                ]
            )
            self._upload_runtime(sandbox_id, install_path)
            self._run([self.config.aenv_bin, "exec", sandbox_id, "chmod", "0755", install_path])
            quoted_path = shlex.quote(install_path)
            start_command = (
                f"nohup {quoted_path} --port {self.config.runtime_port} "
                ">/tmp/ash-runtime.log 2>&1 </dev/null & echo $! >/tmp/ash-runtime.pid"
            )
            self._run(
                [self.config.aenv_bin, "exec", sandbox_id, "sh", "-lc", start_command]
            )
            self._wait_runtime(sandbox_id)
            try:
                self._run(
                    [
                        self.config.aenv_bin,
                        "snapshot",
                        "create",
                        sandbox_id,
                        "--name",
                        snapshot_name,
                    ]
                )
            except RuntimeError:
                # Another rollout-service process may have won the same
                # deterministic alias race. Reuse it only if it now exists.
                if not self._snapshot_exists(snapshot_name):
                    raise
        finally:
            if sandbox_id:
                self._run_best_effort([self.config.aenv_bin, "delete", sandbox_id])
            if template_created:
                self._run_best_effort(
                    [self.config.aenv_bin, "template", "delete", base_name]
                )

    def _upload_runtime(self, sandbox_id: str, install_path: str) -> None:
        """Upload the runtime and wait until the guest can actually see it.

        AgentENV upload completion and guest filesystem visibility are not
        always atomic.  Treat a successful CLI exit as an acknowledgement,
        then verify the file from inside the microVM before using it.  A retry
        repeats the idempotent upload so a dropped/early acknowledgement does
        not poison the prepared-snapshot cache.
        """
        upload = [
            self.config.aenv_bin,
            "upload",
            sandbox_id,
            str(self.config.runtime_artifact),
            install_path,
        ]
        verify = [
            self.config.aenv_bin,
            "exec",
            sandbox_id,
            "test",
            "-s",
            install_path,
        ]
        last_error: RuntimeError | None = None
        for attempt in range(self.config.runtime_upload_attempts):
            try:
                self._run(upload)
                self._run(verify, timeout=30)
                return
            except RuntimeError as exc:
                last_error = exc
                if attempt + 1 < self.config.runtime_upload_attempts:
                    self._sleep(self.config.runtime_upload_retry_seconds)
        raise RuntimeError(
            "ash-runtime upload was acknowledged but the artifact did not "
            f"become visible at {install_path!r} after "
            f"{self.config.runtime_upload_attempts} attempts"
        ) from last_error

    def _wait_runtime(self, sandbox_id: str) -> None:
        deadline = time.monotonic() + min(60, self.config.sandbox_timeout_seconds)
        probe = "test -s /tmp/ash-runtime.pid && kill -0 $(cat /tmp/ash-runtime.pid)"
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            try:
                self._run(
                    [self.config.aenv_bin, "exec", sandbox_id, "sh", "-c", probe],
                    timeout=10,
                )
                return
            except RuntimeError as exc:
                last_error = exc
                self._sleep(1)
        raise RuntimeError("ash-runtime did not become ready in the preparation sandbox") from last_error

    def _run(self, args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        result = self._command_runner(
            args,
            timeout or self.config.build_timeout_seconds,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()[-2000:]
            raise RuntimeError(f"{args[0]} command failed: {detail}")
        return result

    def _run_best_effort(self, args: list[str]) -> None:
        try:
            self._run(args, timeout=60)
        except Exception:
            pass

    def _run_subprocess(
        self,
        args: list[str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        if self._aenv_server_url is None:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        api_key = self._aenv_api_key
        if self._aenv_api_key_file is not None:
            api_key = self._aenv_api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise RuntimeError(
                "AgentENV OCI resolver requires an API key when a server URL is configured"
            )

        # The AgentENV CLI reads only its standard XDG credentials file. Give
        # each subprocess a private, short-lived config directory so the
        # rollout service does not depend on or overwrite a developer's
        # interactive `aenv auth` state.
        with tempfile.TemporaryDirectory(prefix="ash-aenv-cli-") as config_home:
            credentials_dir = Path(config_home) / "aenv"
            credentials_dir.mkdir(mode=0o700)
            credentials_path = credentials_dir / "credentials"
            credentials_path.write_text(
                f"url = {json.dumps(self._aenv_server_url)}\n"
                f"api_key = {json.dumps(api_key)}\n",
                encoding="utf-8",
            )
            credentials_path.chmod(0o600)
            env = os.environ.copy()
            env["XDG_CONFIG_HOME"] = config_home
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )


def _registry(repository: str) -> str:
    first = repository.split("/", 1)[0].lower()
    if "." in first or ":" in first or first == "localhost":
        return first
    return "docker.io"


def _last_nonempty_line(output: str, operation: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{operation} returned no sandbox id")
    return lines[-1]
