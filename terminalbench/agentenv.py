"""Harbor environment lifecycle backed by an AgentENV microVM."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import NetworkMode

from harness.execution.session import SandboxSession
from harness.execution.templates import find_regctl


MIN_OCI_DISK_MB = 64 * 1024


def shell_command(command: str, cwd: str, env: dict[str, str], user: str | int | None) -> str:
    inner = f"cd {shlex.quote(cwd)} && exec /bin/bash -c {shlex.quote(command)}"
    invocation = ["env", *[f"{key}={value}" for key, value in env.items()], "/bin/bash", "-c", inner]
    wrapped = shlex.join(invocation)
    if user not in (None, "", "root", "0", 0):
        account, separator, group = str(user).partition(":")
        username = f'"$(getent passwd {shlex.quote(account)} | cut -d: -f1)"'
        prefix = f"runuser -u {username}"
        if separator:
            prefix += f' -g "$(getent group {shlex.quote(group)} | cut -d: -f1)"'
        wrapped = prefix + " -- " + wrapped
    return wrapped


class AgentENVEnvironment(BaseEnvironment):
    def __init__(self, *args, runtime_bin: str, server_url: str | None = None,
                 api_key_file: str | None = None, image_registry: str | None = None,
                 sandbox_ttl: int = 36000, checkpoint_mode: str = "full", **kwargs):
        self.runtime_bin = str(Path(runtime_bin).resolve())
        self.server_url = server_url or os.environ.get("AENV_SERVER_URL", "http://127.0.0.1:18000")
        self.api_key_file = api_key_file
        self.image_registry = image_registry
        self.sandbox_ttl = sandbox_ttl
        self.checkpoint_mode = checkpoint_mode
        self.session: SandboxSession | None = None
        self.image_user = "root"
        self.image_workdir = "/"
        self.image_env: dict[str, str] = {}
        self.backend: dict = {}
        self.image = ""
        self.actor_control = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "agentenv"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        if not Path(self.runtime_bin).is_file():
            raise ValueError(f"Missing ash-runtime: {self.runtime_bin}")
        if self.checkpoint_mode not in {"full", "disk_only"}:
            raise ValueError("checkpoint_mode must be full or disk_only")
        if self.sandbox_ttl < 30000:
            raise ValueError("sandbox_ttl must cover TB4's 8-hour actor plus setup/collection")
        for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
            if (self.environment_dir / name).exists():
                raise ValueError("AgentENV multi-service composition is not implemented; no Docker fallback")
        if self.task_env_config.mcp_servers:
            raise ValueError("Task-side MCP networking requires an AgentENV service adapter")
        if not self.task_env_config.docker_image and not (self.environment_dir / "Dockerfile").is_file():
            raise ValueError("Task needs a docker_image or environment/Dockerfile")
        if not self.task_env_config.docker_image and not self.image_registry:
            raise ValueError("Dockerfile tasks require --image-registry reachable by AgentENV")
        disk = self.task_env_config.storage_mb
        if disk is not None and (disk < 1024 or disk % 1024):
            raise ValueError("AgentENV disk size must be at least 1024 MiB and divisible by 1024")
        for policy in self._phase_network_policies:
            if policy != self._network_policy:
                raise ValueError("AgentENV in-place network policy changes are not implemented")
        if self._mounts:
            for mount in self._mounts:
                if mount.get("target") not in {"/logs/agent", "/logs/verifier", "/logs/artifacts", "/logs/user-agent"}:
                    raise ValueError(f"AgentENV cannot reproduce host mount {mount.get('target')}")

    def _prepare_image(self, force_build: bool) -> tuple[str, dict]:
        image = self.task_env_config.docker_image
        if force_build or not image:
            if not self.image_registry:
                raise ValueError("Image building requires --image-registry")
            image = f"{self.image_registry.rstrip('/')}/ash-tb4:{self.environment_id}"
            log_path = self.trial_paths.trial_dir / f"{self.session_id}.build.log"
            with log_path.open("w") as log:
                subprocess.run(["docker", "build", "-t", image, str(self.environment_dir)],
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=self.task_env_config.build_timeout_sec)
                subprocess.run(["docker", "push", image], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=self.task_env_config.build_timeout_sec)
        regctl = find_regctl()
        if regctl is None:
            raise RuntimeError("regctl is required to resolve OCI configuration")
        digest = subprocess.check_output([str(regctl), "image", "digest", image], text=True, timeout=120).strip()
        image = image.split("@", 1)[0] + "@" + digest
        config = json.loads(subprocess.check_output(
            [str(regctl), "image", "config", image, "--format", "{{json .Config}}"], text=True, timeout=120))
        return image, config

    async def start(self, force_build: bool = False) -> None:
        self.image, config = await asyncio.to_thread(self._prepare_image, force_build)
        self.image_user = config.get("User") or "root"
        self.image_workdir = config.get("WorkingDir") or "/"
        self.image_env = dict(value.split("=", 1) for value in config.get("Env", []) if "=" in value)
        section = {"server_url": self.server_url, "runtime_bin": self.runtime_bin,
                   "from_image": True, "image_env": True, "sandbox_ttl": self.sandbox_ttl,
                   "allow_internet": self._network_policy.network_mode == NetworkMode.PUBLIC,
                   "request_timeout": 300}
        if self.api_key_file:
            section["api_key_file"] = self.api_key_file
        self.backend = {"backend": "microvm", "microvm": section}
        self.session = SandboxSession(backend=self.backend, quiet=True)
        resources = {"cpu": self._effective_cpus or 2, "memory_mb": self._effective_memory_mb or 1024,
                     "disk_size_mb": max(self._effective_storage_mb or MIN_OCI_DISK_MB, MIN_OCI_DISK_MB)}
        try:
            creation = asyncio.create_task(asyncio.to_thread(self.session.create, self.image, resources))
            try:
                created = await asyncio.shield(creation)
            except asyncio.CancelledError:
                await asyncio.shield(creation)
                raise
            if not created:
                raise RuntimeError(self.session.create_error)
            await self._checked("mkdir -p /logs/agent /logs/user-agent /logs/verifier /logs/artifacts && "
                                "chmod 777 /logs/agent /logs/user-agent /logs/verifier /logs/artifacts")
            record = {"sandbox_id": self.session.sandbox_id, "image": self.image,
                      "requested_disk_size_mb": self._effective_storage_mb,
                      "resources": resources, "network": self._network_policy.model_dump(mode="json"),
                      "checkpoint_mode": self.checkpoint_mode}
            (self.trial_paths.trial_dir / f"{self.session_id}.agentenv.json").write_text(json.dumps(record, indent=2))
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def stop(self, delete: bool = True) -> None:
        if self.actor_control is not None:
            self.actor_control.request_stop("Harbor is stopping this trial")
        if self.session is not None and delete:
            session = self.session
            pool = session._pool
            await asyncio.to_thread(session._drive, session._destroy_async())
            if pool is not None:
                await asyncio.to_thread(session._drive, pool.close())
            self.session = None

    def command_args(self, command: str, cwd: str | None = None,
                     env: dict | None = None, user: str | int | None = None) -> dict:
        effective_user = self._resolve_user(user)
        return {"command": shell_command(
            command, cwd or self.task_env_config.workdir or self.image_workdir,
            {**self.image_env, **(self._merge_env(env) or {})},
            self.image_user if effective_user is None else effective_user), "working_dir": "/"}

    async def exec(self, command: str, cwd: str | None = None, env: dict | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        if self.session is None:
            raise RuntimeError("AgentENV environment has not started")
        timeout = timeout_sec or self.sandbox_ttl
        arguments = {**self.command_args(command, cwd, env, user), "timeout": timeout,
                     "max_output_bytes": 16 * 1024 * 1024}
        pending = asyncio.create_task(asyncio.to_thread(self.session.execute, "shell", arguments, timeout + 60))
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            loop = self.session._get_loop()
            loop.call_soon_threadsafe(lambda: [task.cancel() for task in asyncio.all_tasks(loop)])
            try:
                await asyncio.shield(pending)
            except BaseException:
                pass
            raise
        if result.outcome is not None:
            outcome = result.outcome
            if outcome.timed_out:
                raise TimeoutError(f"Environment command timed out after {timeout}s")
            if outcome.exit_code is None:
                raise RuntimeError("Environment command has no exit status")
            answer = ExecResult(stdout=outcome.stdout, stderr=outcome.stderr, return_code=outcome.exit_code)
        elif result.success:
            answer = ExecResult(stdout=result.output, stderr="", return_code=0)
        else:
            raise RuntimeError(result.error or result.output)
        callback = self._output_callback()
        if callback:
            await callback(answer.stdout or "", "stdout")
            await callback(answer.stderr or "", "stderr")
        return answer

    async def _checked(self, command: str) -> None:
        result = await self.exec(command, cwd="/", user="root", timeout_sec=300)
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._checked(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}")
        if not await asyncio.to_thread(self.session.upload_file, source_path, target_path):
            raise RuntimeError(f"Failed to upload {source_path}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if not await asyncio.to_thread(self.session.download_file, source_path, target_path):
            raise RuntimeError(f"Failed to download {source_path}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        remote = f"/tmp/ash-harbor-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "files.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                stream.add(source_dir, arcname=".")
            try:
                await self.upload_file(archive, remote)
                await self._checked(f"mkdir -p {shlex.quote(target_dir)} && tar -xzf {remote} -C {shlex.quote(target_dir)}")
            finally:
                await self._checked(f"rm -f {remote}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        remote = f"/tmp/ash-harbor-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "files.tar.gz"
            try:
                await self._checked(f"tar -czf {remote} -C {shlex.quote(source_dir)} .")
                await self.download_file(remote, archive)
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:gz") as stream:
                    stream.extractall(target_dir, filter="data")
            finally:
                await self._checked(f"rm -f {remote}")
