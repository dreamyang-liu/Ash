from __future__ import annotations

import pytest

from swebench.rollout_groups.environment_catalog import EnvironmentCatalog
from swebench.rollout_groups.protocol import EnvironmentRef


def _entry(**overrides):
    value = {
        "kind": "template",
        "id": "swebench-runtime",
        "revision": "sha256:test",
        "resource_profile": "standard",
        "spawn_ref": "agentenv-template-17",
    }
    value.update(overrides)
    return value


def test_catalog_resolves_only_an_exact_environment_identity():
    catalog = EnvironmentCatalog.from_dict({"environments": [_entry()]})
    ref = EnvironmentRef.from_dict(
        {
            "kind": "template",
            "id": "swebench-runtime",
            "revision": "sha256:test",
            "resource_profile": "standard",
        }
    )

    assert catalog.resolve(ref).spawn_ref == "agentenv-template-17"
    with pytest.raises(ValueError, match="not allowlisted"):
        catalog.resolve(
            EnvironmentRef(
                kind=ref.kind,
                id=ref.id,
                revision=ref.revision,
                resource_profile="large",
            )
        )


def test_catalog_rejects_duplicate_entries_and_unknown_fields():
    with pytest.raises(ValueError, match="duplicate"):
        EnvironmentCatalog.from_dict({"environments": [_entry(), _entry()]})
    with pytest.raises(ValueError, match="unknown fields"):
        EnvironmentCatalog.from_dict(
            {"environments": [_entry(credentials="must-not-cross-the-wire")]}
        )


def test_catalog_lists_public_refs_in_stable_order_without_spawn_refs():
    catalog = EnvironmentCatalog.from_dict(
        {
            "environments": [
                _entry(id="z-template", spawn_ref="private-z"),
                _entry(id="a-template", spawn_ref="private-a"),
            ]
        }
    )

    values = [ref.to_dict() for ref in catalog.list_refs()]

    assert [value["id"] for value in values] == ["a-template", "z-template"]
    assert all("spawn_ref" not in value for value in values)
