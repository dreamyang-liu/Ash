# Adapted from dreamyang-liu/Ash cc02a738b7833e8582440817636767eb7b5605f2 (MIT).
"""Allowlisted resolution of logical rollout environments.

Miles names an immutable environment and resource profile.  Deployment-owned
catalog entries translate that logical identity into the concrete value Ash's
native ``Pool.spawn(image=...)`` API expects.  Provider endpoints and
credentials never cross the rollout wire contract.
"""

from __future__ import annotations

import json
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
