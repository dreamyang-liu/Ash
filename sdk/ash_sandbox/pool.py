from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any

import httpx

from .backends import GatewayBackend, HTTPBackend
from .result import ToolResult
from .sandbox import Sandbox, _find_free_port


class DockerPool:
    """Manages multiple sandboxes locally via Docker."""

    BOOTSTRAP_URL = "https://raw.githubusercontent.com/dreamyang-liu/Ash/main/runtime/bootstrap.sh"

    def __init__(self, runtime_bin: str | None = None, port: int = 3000):
        self.runtime_bin = runtime_bin or shutil.which("ash-runtime")
        self.port = port
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str = "ubuntu:24.04",
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
    ) -> Sandbox:
        """Spawn a new sandbox.

        Args:
            image: Container image.
            entrypoint: Setup command run before ash-runtime starts.
            env: Environment variables for the container.
            resources: {"cpu": "2", "memory": "4g"} — mapped to Docker limits.
        """
        host_port = _find_free_port()

        docker_args = []
        if env:
            for k, v in env.items():
                docker_args.extend(["-e", f"{k}={v}"])
        if resources:
            if "cpu" in resources:
                docker_args.extend(["--cpus", str(resources["cpu"])])
            if "memory" in resources:
                docker_args.extend(["-m", str(resources["memory"])])

        label_args = ["--label", "ash.managed=1"]

        if self.runtime_bin:
            if entrypoint:
                container_cmd = f"({entrypoint}) && ash-runtime --port {self.port}"
            else:
                container_cmd = f"ash-runtime --port {self.port}"
            cmd = [
                "docker", "run", "-d",
                "-p", f"{host_port}:{self.port}",
                "-v", f"{self.runtime_bin}:/usr/local/bin/ash-runtime:ro",
                *label_args,
                *docker_args,
                image,
                "sh", "-c", container_cmd,
            ]
        else:
            bootstrap_cmd = f"curl -fsSL {self.BOOTSTRAP_URL} | ASH_PORT={self.port}"
            if entrypoint:
                escaped = entrypoint.replace('"', '\\"')
                bootstrap_cmd += f' ASH_SETUP="{escaped}"'
            bootstrap_cmd += " sh"
            cmd = [
                "docker", "run", "-d",
                "-p", f"{host_port}:{self.port}",
                *label_args,
                *docker_args,
                image,
                "sh", "-c", bootstrap_cmd,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr}")

        container_id = result.stdout.strip()
        sb = Sandbox(backend=HTTPBackend(f"http://localhost:{host_port}"))
        sb._container_id = container_id
        self._sandboxes[container_id] = sb

        await sb._wait_ready(timeout=60)
        return sb

    async def destroy(self, sandbox: Sandbox):
        cid = sandbox._container_id
        if cid and cid in self._sandboxes:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            del self._sandboxes[cid]
            sandbox._container_id = None

    async def destroy_all(self):
        for cid in list(self._sandboxes.keys()):
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", cid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        self._sandboxes.clear()

    def list(self) -> list[Sandbox]:
        return list(self._sandboxes.values())

    async def close(self):
        await self.destroy_all()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class SandboxPool:
    """Manages multiple sandboxes via K8s control-plane + gateway."""

    def __init__(self, control_plane_url: str, gateway_url: str, default_image: str = "ubuntu:24.04"):
        self.control_plane_url = control_plane_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.default_image = default_image
        self._client = httpx.AsyncClient(timeout=120)
        self._sandboxes: dict[str, Sandbox] = {}

    async def spawn(
        self,
        image: str | None = None,
        entrypoint: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict | None = None,
    ) -> Sandbox:
        """Spawn a new sandbox.

        Args:
            image: Container image.
            entrypoint: Setup command run before ash-runtime starts.
            env: Environment variables for the container.
            resources: {"cpu": "1", "memory": "2Gi"} — K8s resource requests.
        """
        body: dict[str, Any] = {
            "image": image or self.default_image,
            "ports": [{"container_port": 3000}],
        }
        if entrypoint:
            body["entrypoint"] = entrypoint
        if env:
            body["env"] = env
        if resources:
            body["resources"] = {
                "requests": {k: v for k, v in resources.items()},
            }

        resp = await self._client.post(f"{self.control_plane_url}/create", json=body)
        resp.raise_for_status()
        data = resp.json()

        sandbox_id = data["uuid"]
        sb = Sandbox(backend=GatewayBackend(self.gateway_url, sandbox_id))
        sb._container_id = sandbox_id
        self._sandboxes[sandbox_id] = sb

        await sb._wait_ready(timeout=60)
        return sb

    async def destroy(self, *sandboxes: Sandbox):
        """Destroy one or more sandboxes."""
        ids = [sb._container_id for sb in sandboxes if sb._container_id and sb._container_id in self._sandboxes]
        if not ids:
            return
        await self._client.request("DELETE", f"{self.control_plane_url}/destroy", json={"ids": ids})
        for sid in ids:
            if sid in self._sandboxes:
                del self._sandboxes[sid]
        for sb in sandboxes:
            sb._container_id = None

    async def destroy_all(self):
        """Destroy all sandboxes."""
        await self._client.request("DELETE", f"{self.control_plane_url}/destroy", json={"all": True})
        self._sandboxes.clear()

    async def close(self):
        await self.destroy_all()
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
