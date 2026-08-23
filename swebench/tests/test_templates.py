"""Per-image template builds: naming, staging, reuse, and failure surfaces."""

import pytest

from swebench.templates import (RUNTIME_PATH, TemplateBuilder, TemplateError,
                                builder_from_backend, template_name)


IMAGE = "swebench/sweb.eval.x86_64.django__django-11848:latest"


def test_names_are_legal_stable_and_input_sensitive():
    first = template_name(IMAGE, "fp1", 3000)
    assert first == template_name(IMAGE, "fp1", 3000), "same inputs, same name"
    # Image names carry characters a template alias cannot.
    assert all(c.isalnum() or c == "-" for c in first), first
    assert len(first) <= 128

    # A different runtime binary or port must not reuse a template built with
    # the old one.
    assert first != template_name(IMAGE, "fp2", 3000)
    assert first != template_name(IMAGE, "fp1", 3001)
    assert first != template_name(IMAGE + "@v2", "fp1", 3000)


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class FakeClient:
    """Scripts the calls a two-stage build makes, and records them."""

    def __init__(self, exists=False, statuses=("ready",), create_status=202):
        self.exists = exists
        self.statuses = list(statuses)
        self.create_status = create_status
        self.calls: list[tuple[str, str]] = []
        self.build_payload: dict | None = None
        self.uploaded: bytes | None = None
        self.deleted: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, path, **kwargs):
        self.calls.append(("GET", path))
        if path.startswith("/snapshots/"):
            # Sandbox-sourced snapshots only; a built template answers 404
            # here (it is template-sourced), which is exactly the trap the
            # builder must not fall into.
            return FakeResponse(404)
        if path.startswith("/templates/aliases/"):
            if "ash-swebench-" not in path:
                return FakeResponse(404)
            return FakeResponse(200 if self.exists else 404)
        if path.endswith("/status"):
            status = self.statuses.pop(0) if self.statuses else "ready"
            return FakeResponse(200, {"status": status})
        raise AssertionError(path)

    def post(self, path, json=None, files=None, **kwargs):
        self.calls.append(("POST", path))
        if path == "/sandboxes-cold":
            return FakeResponse(201, {"sandboxID": "staging-sb"})
        if path == "/files":
            self.uploaded = files["file"][1].read()
            return FakeResponse(200, [{"path": RUNTIME_PATH}])
        if path.endswith("/snapshots"):
            assert json.get("diskOnly") is True, "staging pays for no memory image"
            assert "name" not in json, (
                "a staged alias would collide on retry after a failed build")
            return FakeResponse(201, {"snapshotID": "staged-snap"})
        if path == "/v3/templates":
            return FakeResponse(self.create_status,
                                {"templateID": "tid", "buildID": "bid"})
        self.build_payload = json
        return FakeResponse(202)

    def delete(self, path, **kwargs):
        self.calls.append(("DELETE", path))
        self.deleted.append(path)
        return FakeResponse(204)


@pytest.fixture
def runtime_bin(tmp_path):
    path = tmp_path / "ash-runtime"
    path.write_bytes(b"\x7fELF fake runtime")
    return path


def builder(monkeypatch, client, runtime_bin):
    import swebench.templates as templates
    monkeypatch.setattr(templates.httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(templates.time, "sleep", lambda _s: None)
    return TemplateBuilder(server_url="http://server", api_key="k",
                           runtime_bin=runtime_bin)


def test_missing_runtime_binary_fails_at_construction(tmp_path):
    with pytest.raises(TemplateError, match="runtime binary not found"):
        TemplateBuilder(server_url="http://server", api_key="k",
                        runtime_bin=tmp_path / "no-such-file")


def test_existing_template_is_reused_without_building(monkeypatch, runtime_bin):
    client = FakeClient(exists=True)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert [m for m, _ in client.calls] == ["GET"], (
        "an image reference cannot be a snapshot name, so only the template "
        "is looked up; no build was started")


def test_build_stages_the_runtime_then_declares_startup(monkeypatch, runtime_bin):
    client = FakeClient(exists=False)
    builder(monkeypatch, client, runtime_bin).template_for(IMAGE)

    # Stage 1: the binary went in through the file service, and the staging
    # sandbox was not leaked.
    assert client.uploaded == runtime_bin.read_bytes()
    assert client.deleted == ["/sandboxes/staging-sb"]

    # Stage 2: the committed template starts from the staged snapshot, knows
    # how to relaunch the runtime, and restores the executable bit the upload
    # dropped.
    payload = client.build_payload
    assert payload["fromTemplate"] == "staged-snap"
    assert payload["startCmd"] == f"{RUNTIME_PATH} --port 3000"
    # Cold boots re-run startCmd, so readiness must mean the port answers.
    assert "/dev/tcp/127.0.0.1/3000" in payload["readyCmd"]
    assert f"chmod +x {RUNTIME_PATH}" in payload["steps"][0]["args"]


def test_staging_sandbox_is_deleted_even_when_snapshotting_fails(
        monkeypatch, runtime_bin):
    class SnapshotFails(FakeClient):
        def post(self, path, json=None, files=None, **kwargs):
            if path.endswith("/snapshots"):
                self.calls.append(("POST", path))
                return FakeResponse(500, text="boom")
            return super().post(path, json=json, files=files, **kwargs)

    client = SnapshotFails(exists=False)
    with pytest.raises(TemplateError, match="could not snapshot"):
        builder(monkeypatch, client, runtime_bin).template_for(IMAGE)
    assert client.deleted == ["/sandboxes/staging-sb"], "no leaked VM"


def test_second_request_for_one_image_does_not_recheck(monkeypatch, runtime_bin):
    client = FakeClient(exists=True)
    b = builder(monkeypatch, client, runtime_bin)
    b.template_for(IMAGE)
    before = len(client.calls)
    b.template_for(IMAGE)
    assert len(client.calls) == before, "the resolved name is cached"


def test_pending_build_is_waited_for(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=("building", "building", "ready"))
    builder(monkeypatch, client, runtime_bin).template_for(IMAGE)
    assert sum(1 for m, p in client.calls if p.endswith("/status")) == 3


def test_failed_build_is_reported(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=("error",))
    with pytest.raises(TemplateError, match="template build failed"):
        builder(monkeypatch, client, runtime_bin).template_for(IMAGE)


def test_conflicting_create_defers_to_the_other_builder(monkeypatch, runtime_bin):
    # Two workers racing on the same image: the loser reuses the winner's.
    client = FakeClient(exists=False, create_status=409)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert client.build_payload is None


def test_build_timeout_fails_rather_than_hangs(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=tuple(["building"] * 50))
    b = builder(monkeypatch, client, runtime_bin)
    b.build_timeout = 0.0
    with pytest.raises(TemplateError, match="did not finish"):
        b.template_for(IMAGE)


# --- config wiring --------------------------------------------------------- #

def test_builder_only_for_microvm_with_a_runtime_bin(runtime_bin):
    assert builder_from_backend({"backend": "docker"}) is None
    assert builder_from_backend({"backend": "microvm", "microvm": {}}) is None

    built = builder_from_backend({"backend": "microvm", "microvm": {
        "server_url": "http://server/", "api_key": "k",
        "runtime_bin": str(runtime_bin), "runtime_port": 4000}})
    assert built is not None
    assert built.server_url == "http://server", "trailing slash trimmed"
    assert built.runtime_port == 4000


def test_runtime_bin_without_a_server_is_an_error(runtime_bin):
    with pytest.raises(TemplateError, match="server_url"):
        builder_from_backend({"backend": "microvm",
                              "microvm": {"runtime_bin": str(runtime_bin)}})


def test_a_name_the_backend_already_knows_is_used_as_is(monkeypatch, runtime_bin):
    """A replay hands `create()` a checkpoint snapshot id; building a template
    from it would try to cold-start a snapshot as an image and fail. The
    catalog is the authority: a known snapshot passes through untouched."""
    class KnowsSnapshot(FakeClient):
        def get(self, path, **kwargs):
            if path == "/snapshots/01a0-checkpoint":
                self.calls.append(("GET", path))
                return FakeResponse(200, {"snapshotID": "01a0-checkpoint"})
            return super().get(path, **kwargs)

    client = KnowsSnapshot(exists=False)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for("01a0-checkpoint") == "01a0-checkpoint"
    assert client.build_payload is None, "no build for an existing snapshot"
    # And it is cached like any other resolution.
    before = len(client.calls)
    assert b.template_for("01a0-checkpoint") == "01a0-checkpoint"
    assert len(client.calls) == before


def test_a_built_template_is_found_even_though_snapshots_says_404(
        monkeypatch, runtime_bin):
    """The regression that shipped a double build: /snapshots/{name} answers
    only for sandbox-sourced snapshots, so a built (template-sourced) template
    looks missing there and the second run rebuilds into a name collision."""
    client = FakeClient(exists=True)     # exists=True now means: alias lookup hits
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert client.build_payload is None, "no rebuild of an existing template"
    assert any(p.startswith("/templates/aliases/") for _, p in client.calls)


def test_create_collision_reported_as_400_already_points_is_reuse(
        monkeypatch, runtime_bin):
    class Collides(FakeClient):
        def post(self, path, json=None, files=None, **kwargs):
            if path == "/v3/templates":
                self.calls.append(("POST", path))
                return FakeResponse(
                    400, text="alias 'x' already points to 'y', cannot rebind")
            return super().post(path, json=json, files=files, **kwargs)

    client = Collides(exists=False)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert client.build_payload is None
