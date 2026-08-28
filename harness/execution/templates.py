"""Turn a benchmark's per-instance images into microVM templates.

`MicroVMPool.spawn` starts from a template or snapshot; only
`spawn_from_image` accepts an OCI reference, and a cold-started plain image
cannot bring up the runtime, because the backend runs no startup command for
one. A benchmark whose environments are raw per-instance images (SWE-bench's
are) therefore needs one template per image, built once with a startup command
that launches the runtime.

Doing that by hand does not scale to a few hundred instances, so this module
does it on demand: ask for an image, get back the name of a template built
from it. Builds are content-addressed by (image, runtime binary, start
command), so the second run of an instance reuses the first run's template
instead of rebuilding, and changing the runtime binary produces a distinct
template rather than silently reusing a stale one.

Getting the runtime into the template is the interesting part. Template steps
cannot copy a local file, and a build step cannot reliably download one either:
the guest's network is isolated, and a benchmark image need not ship curl. So
the build happens in two stages, both of which use only what the backend
already offers:

1. cold-start a sandbox from the image and upload the binary through the
   backend's own file service (which does not involve the runtime at all,
   which is the point -- the runtime is what we are installing), then snapshot
   it; that snapshot has the binary but no startup command;
2. run a template build *from that snapshot* declaring ``startCmd``, so the
   committed template knows how to relaunch the runtime after a cold boot.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


#: Where the runtime lands in the template, and how it is started.
RUNTIME_PATH = "/usr/local/bin/ash-runtime"
DEFAULT_RUNTIME_PORT = 3000

#: ripgrep is baked in alongside the runtime. The runtime's grep_files
#: provisions rg on first use when it is missing -- an apt-get update whose
#: package indexes alone are ~76 MiB of disk writes, paid by every sandbox of
#: every instance, and landing in the episode's first checkpoint (measured:
#: the first grep_files cost +89 MiB; with rg present, +0). Same release the
#: runtime's own provisioning would fetch.
#: How many suffixed template names to try before giving up. Each one costs a
#: lookup; needing more than a couple means something is failing repeatably
#: and a louder error is better than a longer search.
MAX_TEMPLATE_ATTEMPTS = 6

RIPGREP_PATH = "/usr/local/bin/rg"
RIPGREP_VERSION = "14.1.1"
RIPGREP_URL = ("https://github.com/BurntSushi/ripgrep/releases/download/"
               f"{RIPGREP_VERSION}/ripgrep-{RIPGREP_VERSION}-x86_64-unknown-linux-musl.tar.gz")

#: The backend's file service listens on this port inside the guest; reaching
#: it goes through the same proxy as the runtime, with a different target port.
ENVD_PORT = "49983"
SANDBOX_ID_HEADER = "x-agentenv-sandbox-id"
TARGET_PORT_HEADER = "x-agentenv-target-port"

#: Poll interval and ceiling for a build. A cold image pull plus conversion is
#: the slow part; the ceiling exists so a stuck build fails the run instead of
#: hanging it.
BUILD_POLL_SECONDS = 3.0
BUILD_TIMEOUT_SECONDS = 1800.0
COLD_START_TIMEOUT_SECONDS = 900


class TemplateError(RuntimeError):
    """A template could not be built, so the instance cannot run."""


def _could_be_snapshot_name(name: str) -> bool:
    """Whether ``name`` is even *expressible* as a snapshot id or alias.

    The backend's alias grammar is ASCII letters, digits, hyphens and
    underscores (ids are UUIDs, a subset). An image reference's ``/``, ``:``
    or ``@`` cannot appear in one, so such names skip the catalog lookup
    entirely -- asking would be answered with an error, not a 404. This is a
    grammar fact, not a guess about what the caller meant: a bare name like
    ``ubuntu`` still goes to the catalog, misses, and gets a template built.
    """
    return bool(name) and all(c.isalnum() or c in "-_" for c in name)


def template_name(image: str, runtime_fingerprint: str, port: int,
                  resources: "Optional[dict]" = None) -> str:
    """A stable, legal template name for one (image, runtime, port, shape).

    Content-addressed rather than derived from the image name: template names
    are length-limited and the alias grammar is narrow, while image names
    carry slashes, colons and dots. Hashing the runtime's fingerprint in too
    means a rebuilt runtime binary cannot land on a template built with the
    old one.

    The shape is part of the identity because a microVM's CPU and memory are
    fixed by its template: a 16 GB task must not be handed the 1 GB template
    an earlier run built from the same image.
    """
    shape = ""
    if resources:
        shape = f"{resources.get('cpu', '')}x{resources.get('memory_mb', '')}"
    digest = hashlib.sha256(
        "\0".join((image, runtime_fingerprint, str(port),
                    shape)).encode()).hexdigest()[:24]
    return f"ash-swebench-{digest}"


def runtime_fingerprint(runtime_bin: Path) -> str:
    """Content hash of the runtime binary, so a rebuild forces a new template."""
    digest = hashlib.sha256()
    with open(runtime_bin, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass
class TemplateBuilder:
    """Builds (and reuses) microVM templates for benchmark images.

    Instances cache what they resolved, so a batch run builds each distinct
    image once. Builds are idempotent by name: a template that already exists
    is reused rather than rebuilt.
    """

    server_url: str
    api_key: str
    runtime_bin: Path
    #: Optional ripgrep binary to bake in next to the runtime. None falls
    #: back to the runtime provisioning rg on first use inside each sandbox.
    ripgrep_bin: Optional[Path] = None
    runtime_port: int = DEFAULT_RUNTIME_PORT
    request_timeout: float = 120.0
    build_timeout: float = BUILD_TIMEOUT_SECONDS
    _resolved: dict[str, str] = field(default_factory=dict)
    _fingerprint: str = ""

    def __post_init__(self) -> None:
        self.runtime_bin = Path(self.runtime_bin)
        if not self.runtime_bin.is_file():
            raise TemplateError(
                f"runtime binary not found at {self.runtime_bin}; "
                "build it (cd runtime && go build -o ash-runtime .) or point "
                "microvm.runtime_bin at one")
        self._fingerprint = runtime_fingerprint(self.runtime_bin)
        if self.ripgrep_bin is not None:
            self.ripgrep_bin = Path(self.ripgrep_bin)
            if not self.ripgrep_bin.is_file():
                self.ripgrep_bin = None
        # Baked-in content is part of the template's identity: a template
        # without rg must not be reused once rg is available.
        if self.ripgrep_bin is not None:
            self._fingerprint += ":" + runtime_fingerprint(self.ripgrep_bin)

    def template_for(self, image: str,
                     resources: "Optional[dict]" = None) -> str:
        """The name of a template built from ``image``, building if needed.

        A name the backend already knows -- a checkpoint snapshot a replay
        hands over, a template built earlier -- is returned as-is: the catalog
        is the authority on what is a snapshot, so nothing is guessed from the
        string's shape. Everything else is treated as an image reference and
        gets a template built for it.
        """
        if image in self._resolved:
            return self._resolved[image]

        base = template_name(image, self._fingerprint, self.runtime_port,
                             resources)
        name = base
        with self._client() as client:
            if _could_be_snapshot_name(image) and self._known(client, image):
                # A snapshot carries the shape it was captured with; asking
                # for a different one here is not possible and not wanted --
                # a resumed run continues in the machine it was running on.
                name = image
            else:
                name = self._usable_template(client, base, image, resources)
        self._resolved[image] = name
        return name

    # --- internals ---

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.server_url,
                            headers={"X-API-Key": self.api_key},
                            timeout=self.request_timeout)

    def _known(self, client: httpx.Client, name: str) -> bool:
        """Whether the backend already knows ``name`` as something launchable.

        Sandbox-sourced snapshots (a replay's checkpoint id) and built
        templates live behind different lookup endpoints, so both are asked.
        """
        return (self._lookup(client, f"/snapshots/{name}", name)
                or self._template_exists(client, name))

    def _usable_template(self, client: httpx.Client, base: str, image: str,
                         resources: "Optional[dict]") -> str:
        """The name of a usable template for ``image``, building if needed.

        Tries the content-addressed name first, then suffixed variants. The
        suffix exists because a template alias cannot be rebound: a build that
        failed leaves the canonical name pointing at a template that can never
        be spawned ("snapshot ... is not ready"), and re-creating it is
        refused ("cannot rebind"). So a poisoned name is permanent, and the
        only way forward is a different one. Measured cost of not doing this:
        one failed build took out all 20 tasks of a batch, each reporting
        `sandbox creation failed` with the real reason three layers down.
        """
        for attempt in range(MAX_TEMPLATE_ATTEMPTS):
            name = base if attempt == 0 else f"{base}-r{attempt}"
            if self._template_exists(client, name):
                if attempt:
                    logging.getLogger(__name__).warning(
                        "template %s was unusable; using %s", base, name)
                return name
            if not self._name_taken(client, name):
                staged = self._stage_runtime(client, image, name, resources)
                self._build_from(client, name, image, staged, resources)
                return name
            # Taken but unusable: a failed build owns this name for good.
            logging.getLogger(__name__).warning(
                "template %s exists but its build failed; trying the next name",
                name)
        raise TemplateError(
            f"no usable template name for {image}: {MAX_TEMPLATE_ATTEMPTS} "
            f"variants of {base} are taken by failed builds")

    def _name_taken(self, client: httpx.Client, name: str) -> bool:
        """Whether the alias resolves at all, usable or not."""
        return self._lookup(client, f"/templates/aliases/{name}", name)

    def _template_exists(self, client: httpx.Client, name: str) -> bool:
        """Whether a template of this name exists *and is usable*.

        Not /snapshots/{name}: that endpoint deliberately answers only for
        sandbox-sourced snapshots, and a built template is template-sourced.
        Asking the wrong one reports every built template as missing, and the
        rebuild then collides with the very template it failed to see.

        And not existence alone. A build that failed still leaves the alias
        resolvable, so "it exists" was answering a weaker question than the
        caller asks -- every later run adopted the broken template and got
        HTTP 500 ("snapshot ... is not ready") when spawning from it. One
        failed build poisoned all 20 tasks of a batch that way.
        """
        resp = client.get(f"/templates/aliases/{name}")
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise TemplateError(
                f"could not check {name}: HTTP {resp.status_code} "
                f"{resp.text[:200]}")
        template_id = (resp.json() or {}).get("templateID")
        if not template_id:
            # The alias resolves but does not say to what. Not evidence of a
            # failed build, so not grounds for condemning it -- the same
            # polarity as _build_succeeded below.
            return True
        return self._build_succeeded(client, template_id)

    def _build_succeeded(self, client: httpx.Client, template_id: str) -> bool:
        """Whether the template's build is *not known to have failed*.

        Polarity matters here. Requiring a recognised success value would make
        every unexpected answer -- an older template with no status, a renamed
        state, a backend that does not implement the endpoint -- look like a
        failed build, which is the same mistake as reading a missing signal as
        a bad one. Only an explicit failure disqualifies a template; anything
        else is used, and a template that is genuinely unusable fails loudly at
        spawn instead of silently multiplying template names.
        """
        try:
            resp = client.get(
                f"/templates/{template_id}/builds/{template_id}/status")
        except Exception:
            return True
        if resp.status_code != 200:
            return True
        status = str((resp.json() or {}).get("status") or "").lower()
        return status not in ("error", "failed", "failure", "cancelled",
                             "canceled")

    def _lookup(self, client: httpx.Client, path: str, name: str) -> bool:
        resp = client.get(path)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise TemplateError(
            f"could not check {name}: HTTP {resp.status_code} {resp.text[:200]}")

    def _stage_runtime(self, client: httpx.Client, image: str, name: str,
                       resources: "Optional[dict]" = None) -> str:
        """Cold-start ``image``, install the runtime, and snapshot it.

        Returns the staged snapshot's id. Uses the backend's file service
        rather than the runtime, which is what makes this able to install the
        runtime in the first place.
        """
        # Cold start is the only place a shape can be chosen; the template
        # built from this snapshot inherits it, and so does every sandbox
        # started from that template.
        payload: dict = {
            "image": image,
            "timeout": COLD_START_TIMEOUT_SECONDS,
            "autoPause": False,
        }
        if resources:
            if resources.get("cpu"):
                payload["cpuCount"] = int(resources["cpu"])
            if resources.get("memory_mb"):
                payload["memoryMB"] = int(resources["memory_mb"])
        created = client.post("/sandboxes-cold", json=payload,
                              timeout=max(self.request_timeout,
                                          COLD_START_TIMEOUT_SECONDS))
        if created.status_code != 201:
            raise TemplateError(
                f"could not cold-start {image}: "
                f"HTTP {created.status_code} {created.text[:200]}")
        sandbox_id = created.json()["sandboxID"]
        try:
            self._upload(client, sandbox_id, self.runtime_bin, RUNTIME_PATH)
            if self.ripgrep_bin is not None:
                self._upload(client, sandbox_id, self.ripgrep_bin, RIPGREP_PATH)
            # Deliberately unnamed: the build consumes it by id, and an alias
            # would make a retry after a half-failed build collide with the
            # previous attempt's leftover.
            snapshot = client.post(f"/sandboxes/{sandbox_id}/snapshots",
                                   json={"diskOnly": True})
            if snapshot.status_code != 201:
                raise TemplateError(
                    f"could not snapshot the staged runtime for {image}: "
                    f"HTTP {snapshot.status_code} {snapshot.text[:200]}")
            return snapshot.json()["snapshotID"]
        finally:
            # The staging sandbox has served its purpose either way; leaving it
            # running would hold a VM for the rest of the run.
            client.delete(f"/sandboxes/{sandbox_id}")

    def _upload(self, client: httpx.Client, sandbox_id: str,
                source: Path, dest: str) -> None:
        headers = {SANDBOX_ID_HEADER: sandbox_id, TARGET_PORT_HEADER: ENVD_PORT}
        with open(source, "rb") as handle:
            resp = client.post(
                "/files", params={"path": dest}, headers=headers,
                files={"file": (source.name, handle,
                                "application/octet-stream")},
                timeout=max(self.request_timeout, 300.0))
        if resp.status_code not in (200, 201, 204):
            raise TemplateError(
                f"could not upload {source.name} to {sandbox_id}: "
                f"HTTP {resp.status_code} {resp.text[:200]}")
        # The executable bit does not survive the upload; the build below
        # restores it with a RUN step, which is also the reason the build has
        # any steps at all.

    def _build_from(self, client: httpx.Client, name: str, image: str,
                    staged_snapshot: str, resources: "Optional[dict]" = None) -> None:
        # The template's declared shape must match the staged snapshot's: a
        # snapshot-based build resumes a committed VM, so the backend refuses
        # one whose CPU or memory would differ ("snapshot-based build cannot
        # change CPU or memory"). Passing the shape at creation is what makes
        # the whole chain consistent -- cold start, snapshot, template, and
        # every sandbox spawned from it.
        payload: dict = {"name": name}
        if resources:
            if resources.get("cpu"):
                payload["cpuCount"] = int(resources["cpu"])
            if resources.get("memory_mb"):
                payload["memoryMB"] = int(resources["memory_mb"])
        created = client.post("/v3/templates", json=payload)
        if created.status_code == 409 or (
                created.status_code == 400 and "already points" in created.text):
            # Another worker (or an earlier attempt) got there first; its
            # build is the one to use.
            return
        if created.status_code != 202:
            raise TemplateError(
                f"could not create template {name} for {image}: "
                f"HTTP {created.status_code} {created.text[:200]}")
        body = created.json()
        template_id, build_id = body["templateID"], body["buildID"]

        started = client.post(
            f"/v2/templates/{template_id}/builds/{build_id}",
            json={
                "fromTemplate": staged_snapshot,
                # One step, to make the uploaded binary executable: uploads do
                # not carry the mode bit.
                "steps": [{"type": "RUN", "args": [
                    f"chmod +x {RUNTIME_PATH}; chmod +x {RIPGREP_PATH} 2>/dev/null || true"]}],
                "startCmd": f"{RUNTIME_PATH} --port {self.runtime_port}",
                # Cold boots re-run startCmd, so readiness has to mean "the
                # runtime is accepting connections", not "the process exists".
                "readyCmd": f"timeout 1 bash -c '</dev/tcp/127.0.0.1/{self.runtime_port}'",
            })
        if started.status_code not in (200, 202, 204):
            raise TemplateError(
                f"could not start build for {name} ({image}): "
                f"HTTP {started.status_code} {started.text[:200]}")

        self._await_build(client, template_id, build_id, name, image)

    def _await_build(self, client: httpx.Client, template_id: str,
                     build_id: str, name: str, image: str) -> None:
        deadline = time.monotonic() + self.build_timeout
        last = ""
        while time.monotonic() < deadline:
            resp = client.get(f"/templates/{template_id}/builds/{build_id}/status")
            if resp.status_code != 200:
                raise TemplateError(
                    f"could not read build status for {name}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}")
            status = str(resp.json().get("status") or "")
            last = status
            if status == "ready":
                return
            if status in ("error", "failed"):
                raise TemplateError(
                    f"template build failed for {name} ({image}): "
                    f"{resp.text[:400]}")
            time.sleep(BUILD_POLL_SECONDS)
        raise TemplateError(
            f"template build for {name} ({image}) did not finish in "
            f"{self.build_timeout:.0f}s (last status: {last or 'unknown'})")


def ensure_ripgrep(cache_dir: Optional[Path] = None) -> Optional[Path]:
    """A ripgrep binary to bake into templates, downloaded once per host.

    Returns ``None`` when it cannot be fetched -- templates still build, and
    the runtime falls back to provisioning rg inside each sandbox (the
    behaviour this exists to avoid, so a warning is printed).
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "ash-swebench"
    binary = cache_dir / f"rg-{RIPGREP_VERSION}"
    if binary.is_file():
        return binary
    import io
    import tarfile
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            payload = client.get(RIPGREP_URL)
            payload.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(payload.content), mode="r:gz") as tar:
            member = next(m for m in tar.getmembers()
                          if m.name.endswith("/rg") and m.isfile())
            extracted = tar.extractfile(member)
            assert extracted is not None
            tmp = binary.with_suffix(".tmp")
            tmp.write_bytes(extracted.read())
            tmp.chmod(0o755)
            tmp.rename(binary)
        return binary
    except Exception as exc:  # noqa: BLE001 -- optional asset, never fatal
        print(f"note: could not fetch ripgrep to bake into templates ({exc}); "
              "each sandbox's first grep_files will provision it instead")
        return None


def builder_from_backend(backend: dict) -> Optional[TemplateBuilder]:
    """A builder for a microvm backend that asks for per-image templates.

    Returns ``None`` unless the backend is microvm and names a
    ``runtime_bin``: without the binary there is nothing to install, so the
    harness must be pointed at pre-built templates instead.
    """
    if str(backend.get("backend") or "").strip().lower() != "microvm":
        return None
    section = backend.get("microvm") or {}
    runtime_bin = section.get("runtime_bin") or backend.get("runtime_bin")
    if not runtime_bin:
        return None
    server_url = section.get("server_url")
    if not server_url:
        raise TemplateError(
            "microvm.runtime_bin is set but microvm.server_url is not")
    return TemplateBuilder(
        server_url=str(server_url).rstrip("/"),
        api_key=str(section.get("api_key") or ""),
        runtime_bin=Path(str(runtime_bin)),
        ripgrep_bin=ensure_ripgrep(),
        runtime_port=int(section.get("runtime_port", DEFAULT_RUNTIME_PORT)),
        request_timeout=float(section.get("request_timeout", 120.0)),
    )
