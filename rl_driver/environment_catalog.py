# Adapted from dreamyang-liu/Ash cc02a738b7833e8582440817636767eb7b5605f2 (MIT).
"""Allowlisted resolution of logical rollout environments.

Miles names an immutable environment and resource profile.  Deployment-owned
catalog entries translate that logical identity into the concrete value Ash's
native ``Pool.spawn(image=...)`` API expects.  Provider endpoints and
credentials never cross the rollout wire contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import EnvironmentRef


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EnvironmentCatalogEntry:
    ref: EnvironmentRef
    spawn_ref: str

    @classmethod
    def from_dict(cls, value: Any) -> "EnvironmentCatalogEntry":
        if not isinstance(value, dict):
            raise ValueError("environment catalog entries must be objects")
        unknown = set(value) - {"kind", "id", "revision", "resource_profile", "spawn_ref"}
        if unknown:
            raise ValueError(
                f"environment catalog entry contains unknown fields: {sorted(unknown)}"
            )
        ref = EnvironmentRef.from_dict(
            {key: value.get(key) for key in ("kind", "id", "revision", "resource_profile")}
        )
        return cls(
            ref=ref,
            spawn_ref=_required_string(value.get("spawn_ref"), "environment catalog spawn_ref"),
        )


class EnvironmentCatalog:
    """Exact allowlist keyed by environment kind, id, revision and profile."""

    def __init__(self, entries: list[EnvironmentCatalogEntry]) -> None:
        if not entries:
            raise ValueError("environment catalog must contain at least one entry")
        self._entries: dict[EnvironmentRef, EnvironmentCatalogEntry] = {}
        for entry in entries:
            if entry.ref in self._entries:
                raise ValueError(
                    "duplicate environment catalog entry: "
                    f"{entry.ref.kind}/{entry.ref.id}@{entry.ref.revision} "
                    f"({entry.ref.resource_profile})"
                )
            self._entries[entry.ref] = entry

    @classmethod
    def from_dict(cls, value: Any) -> "EnvironmentCatalog":
        if not isinstance(value, dict):
            raise ValueError("environment catalog must be an object")
        unknown = set(value) - {"environments"}
        if unknown:
            raise ValueError(f"environment catalog contains unknown fields: {sorted(unknown)}")
        raw_entries = value.get("environments")
        if not isinstance(raw_entries, list):
            raise ValueError("environment catalog environments must be a list")
        return cls([EnvironmentCatalogEntry.from_dict(item) for item in raw_entries])

    @classmethod
    def from_file(cls, path: str | Path) -> "EnvironmentCatalog":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def resolve(self, ref: EnvironmentRef) -> EnvironmentCatalogEntry:
        entry = self.find(ref)
        if entry is None:
            raise ValueError(
                "environment_ref is not allowlisted: "
                f"{ref.kind}/{ref.id}@{ref.revision} ({ref.resource_profile})"
            )
        return entry

    def find(self, ref: EnvironmentRef) -> EnvironmentCatalogEntry | None:
        """Return an exact static entry without triggering dynamic preparation."""
        return self._entries.get(ref)

    def list_refs(self) -> list[EnvironmentRef]:
        """Return stable public identities without exposing backend spawn refs."""
        return sorted(
            self._entries,
            key=lambda ref: (ref.kind, ref.id, ref.revision, ref.resource_profile),
        )


def _oci_registry(repository: str) -> str:
    first = repository.split("/", 1)[0].lower()
    if "." in first or ":" in first or first == "localhost":
        return first
    return "docker.io"


class EnvironmentResolver:
    """Resolve trusted logical environments without exposing backend handles.

    Static template/snapshot entries must match the deployment catalog exactly.
    Digest-pinned OCI images may instead be admitted by registry policy.  The
    worker receives the immutable ``repository@digest`` source and the existing
    :class:`harness.execution.templates.TemplateBuilder` turns it into a
    runtime-ready, content-addressed AgentENV template before sandbox creation.
    """

    def __init__(
        self,
        catalog: EnvironmentCatalog | None,
        *,
        allowed_oci_registries: list[str] | tuple[str, ...] = (),
    ) -> None:
        if not isinstance(allowed_oci_registries, (list, tuple)):
            raise ValueError("allowed_oci_registries must be a list")
        if any(not isinstance(item, str) or not item.strip() for item in allowed_oci_registries):
            raise ValueError("allowed_oci_registries entries must be nonempty strings")
        self.catalog = catalog
        self.allowed_oci_registries = frozenset(
            item.strip().lower() for item in allowed_oci_registries
        )

    def resolve(self, ref: EnvironmentRef) -> EnvironmentCatalogEntry:
        static = self.catalog.find(ref) if self.catalog else None
        if static is not None:
            return static
        if ref.kind != "image":
            raise ValueError(
                "environment_ref is not allowlisted: "
                f"{ref.kind}/{ref.id}@{ref.revision} ({ref.resource_profile})"
            )
        self._validate_oci(ref)
        return EnvironmentCatalogEntry(
            ref=ref,
            spawn_ref=f"{ref.id}@{ref.revision.lower()}",
        )

    def _validate_oci(self, ref: EnvironmentRef) -> None:
        if not self.allowed_oci_registries:
            raise ValueError("dynamic OCI environments are disabled")
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
        if ":" in ref.id.rsplit("/", 1)[-1]:
            raise ValueError(
                "environment_ref.id must not contain a mutable tag; use revision for the digest"
            )
        if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", ref.revision):
            raise ValueError("dynamic OCI environment revision must be a sha256 digest")
        registry = _oci_registry(ref.id)
        if registry not in self.allowed_oci_registries:
            raise ValueError(f"OCI registry {registry!r} is not allowlisted")

    def list_refs(self) -> list[EnvironmentRef]:
        return self.catalog.list_refs() if self.catalog else []
