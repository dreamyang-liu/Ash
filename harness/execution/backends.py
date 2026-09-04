"""Where sandboxes come from — config to ``Pool``, in one place.

The SDK's ``Pool`` is already the abstraction a harness wants ("give me a
sandbox, tear it down afterwards", pool.py). What was missing is the step that
picks one: every call site named ``DockerPool`` outright, so the microVM pool
could not be reached without editing code, and the choice was duplicated in the
session and the MCP proxy.

    backend: docker   local containers (default; needs a Docker daemon)
    backend: microvm  Firecracker VMs via AgentENV (~90 ms spawn, forkable)
    backend: k8s      pods behind the control plane + gateway

Config keys are per backend and namespaced, so one config can carry settings for
several and switching is a one-word change::

    execution:
      backend: microvm
      microvm:
        server_url: http://127.0.0.1:8000
        template: ash-base
        api_key_file: ~/.config/ash/aenv-key

Each backend reads only its own section; an unknown key there is an error rather
than a silently ignored typo, because a mistyped `server_url` would otherwise
look like a connection failure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ash_sandbox import DockerPool, MicroVMPool, Pool, SandboxPool

__all__ = ["build_pool", "backend_config", "backend_name", "BACKENDS",
           "BackendError", "resolve_microvm_endpoint"]


class BackendError(ValueError):
    """A backend was named or configured in a way that cannot work."""


DEFAULT_BACKEND = "docker"

#: Recognised keys per backend, so a typo is reported rather than ignored.
_ALLOWED_KEYS = {
    "docker": {"runtime_bin", "port"},
    "microvm": {"server_url", "template", "runtime_port", "api_key",
                "api_key_file", "request_timeout", "sandbox_ttl", "auto_resume",
                # Whether this harness's image names are OCI references to
                # cold-start (a benchmark's per-instance images) rather than
                # names of snapshots already in the backend's catalog. Only the
                # cold-start path accepts an image reference, so the answer is
                # configuration, not something to guess from the string.
                "from_image",
                # Path to a local ash-runtime binary. Set it and the harness
                # builds a template per instance image on demand, uploading
                # the binary through the backend's file service
                # (swebench/templates.py); leave it unset and the harness
                # expects templates to exist already.
                "runtime_bin",
                # False = no egress from any sandbox this pool creates
                # (AgentENV `allowInternetAccess`). A benchmark whose tasks
                # declare no-network sets it; unset keeps the server default.
                "allow_internet",
                # True = templates launch the runtime under the image's own
                # OCI ENV (PATH included), read with regctl at build time. The
                # guest agent otherwise rebuilds PATH and drops e.g. a venv's
                # bin. Opt-in so existing template names stay stable.
                "image_env"},
    "k8s": {"control_plane_url", "gateway_url", "default_image"},
}


def backend_name(config: dict) -> str:
    """The backend this config selects."""
    return str(config.get("backend") or DEFAULT_BACKEND).strip().lower()


def backend_config(config: dict) -> dict:
    """The backend-selection subset of a harness config.

    Harnesses receive one flat dict (``__main__._flatten``), so this picks out
    what a pool needs and leaves the model/dataset settings behind — a harness
    passes this to ``AshSession`` rather than the whole config, so a session
    never depends on keys that have nothing to do with sandboxes.
    """
    keys = {"backend", "runtime_bin", *BACKENDS}
    return {k: v for k, v in config.items() if k in keys and v is not None}


def _section(config: dict, backend: str) -> dict:
    section = config.get(backend) or {}
    if not isinstance(section, dict):
        raise BackendError(f"config key {backend!r} must be a mapping of settings")
    unknown = set(section) - _ALLOWED_KEYS[backend]
    if unknown:
        raise BackendError(
            f"unknown {backend} setting(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(_ALLOWED_KEYS[backend]))}")
    return dict(section)


def _read_api_key(section: dict, env_var: str) -> str:
    """The API key from config, a file, or the environment — in that order.

    A file is offered because a key in a YAML config tends to reach a commit.
    """
    if section.get("api_key"):
        return str(section["api_key"])
    path = section.get("api_key_file")
    if path:
        resolved = Path(os.path.expanduser(str(path)))
        if not resolved.is_file():
            raise BackendError(f"api_key_file not found: {resolved}")
        return resolved.read_text().strip()
    return os.environ.get(env_var, "")


def _docker(config: dict, section: dict) -> Pool:
    # runtime_bin is also a top-level execution setting (and a CLI flag), since
    # it predates backend selection and applies to the common local case.
    runtime_bin = section.get("runtime_bin") or config.get("runtime_bin")
    return DockerPool(runtime_bin=runtime_bin,
                      port=int(section.get("port", 3000)))


def resolve_microvm_endpoint(section: dict) -> "tuple[str, str]":
    """Where AgentENV is and how to authenticate, as (server_url, api_key).

    One resolver so the pool and the template builder cannot disagree. They did:
    the pool fell back to the environment while the builder required both settings
    in the config, so a configuration that spawned sandboxes failed to build a
    template -- and the failure surfaced as "could not create a sandbox", which
    points at the image rather than at the missing setting.

    ``server_url`` comes back empty when neither config nor environment names one;
    each caller phrases its own error, because what is impossible without it
    differs.
    """
    server_url = section.get("server_url") or os.environ.get("AENV_SERVER_URL") or ""
    return str(server_url), _read_api_key(section, "AENV_API_KEY")


def _microvm(config: dict, section: dict) -> Pool:
    server_url, api_key = resolve_microvm_endpoint(section)
    if not server_url:
        raise BackendError(
            "backend 'microvm' needs microvm.server_url (or AENV_SERVER_URL), "
            "e.g. http://127.0.0.1:8000")
    return MicroVMPool(
        server_url=str(server_url),
        default_template=str(section.get("template", "ash-base")),
        runtime_port=int(section.get("runtime_port", 3000)),
        api_key=api_key,
        request_timeout=float(section.get("request_timeout", 120)),
        sandbox_ttl=int(section.get("sandbox_ttl", 600)),
        auto_resume=bool(section.get("auto_resume", True)),
        allow_internet=(None if section.get("allow_internet") is None
                        else bool(section["allow_internet"])),
    )


def _k8s(config: dict, section: dict) -> Pool:
    control_plane = section.get("control_plane_url") or \
        os.environ.get("ASH_CONTROL_PLANE_URL")
    gateway = section.get("gateway_url") or os.environ.get("ASH_GATEWAY_URL")
    missing = [n for n, v in (("control_plane_url", control_plane),
                              ("gateway_url", gateway)) if not v]
    if missing:
        raise BackendError(
            f"backend 'k8s' needs k8s.{' and k8s.'.join(missing)}")
    pool = SandboxPool(control_plane_url=str(control_plane),
                       gateway_url=str(gateway))
    if section.get("default_image"):
        pool.default_image = str(section["default_image"])
    return pool


BACKENDS: dict[str, Any] = {
    "docker": _docker,
    "microvm": _microvm,
    "k8s": _k8s,
}


def build_pool(config: dict | None = None, **overrides) -> Pool:
    """The pool this config asks for.

    ``overrides`` are merged over ``config`` for callers that hold a setting
    directly (``AshSession(runtime_bin=…)``, the proxy's ``--runtime-bin``).
    """
    merged = dict(config or {})
    merged.update({k: v for k, v in overrides.items() if v is not None})
    backend = backend_name(merged)
    build = BACKENDS.get(backend)
    if build is None:
        raise BackendError(
            f"unknown backend {backend!r}; choose from "
            f"{', '.join(sorted(BACKENDS))}")
    return build(merged, _section(merged, backend))
