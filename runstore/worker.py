"""Same-host attempt supervision. Only reconciled attempts may be requeued."""

from __future__ import annotations

import os
import logging
from pathlib import Path
import signal
import subprocess
import sys
import time
from threading import Event
from uuid import uuid4

from runstore.config import merge, referenced_env, resolve, snapshot_validator
from runstore.files import JournalFrame, journal_events, read_json, write_json
from runstore.failures import failure_kind
from runstore.index import Index
from runstore.native import index_native
from runstore.payload import PayloadUnavailable, encode_payload, send_payload
from runstore.specs import digest, validate_continuation
from runstore.store import Store
from runstore.watchdog import LeaseKeeper, Watchdog


def process_identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def stop_process(identity: dict | None) -> bool:
    if identity is None:
        return True
    pid = identity["pid"]
    current = process_identity(pid)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if identity["boot_id"] != boot_id or (current is not None and current != identity):
        return True
    if current is not None:
        try:
            if os.getpgid(pid) != pid:
                return False
        except ProcessLookupError:
            pass

    def group_alive() -> bool:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == pid:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    if not group_alive():
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 10
    while group_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    if group_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 5
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
    return not group_alive()


class Worker:
    def __init__(self, store: Store, config: dict, *, worker_id: str | None = None,
                 lease_s: float = 60, snapshot_valid=None) -> None:
        self.store = store
        self.config = config
        self.root = Path(config["artifact_root"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.worker_id = worker_id or f"{os.uname().nodename}-{os.getpid()}-{uuid4().hex[:8]}"
        self.lease_s = lease_s
        self.index = Index(store, snapshot_valid or snapshot_validator(store, config))
        self._frames: dict[Path, JournalFrame] = {}

    def _scope(self, request: dict, envelope: dict | None = None) -> dict:
        recovery = (envelope or {}).get("recovery") or {}
        return {"context": request["context"], "run_spec_hash": digest(request["spec"]),
                "parent_point": request.get("parent_point"), "profile_hash": request.get("profile_hash"),
                "recovery_snapshot": recovery.get("snapshot_id"),
                "native_prefix_hash": recovery.get("native", {}).get("sha256")}

    def _ingest(self, job: dict, directory: Path, envelope: dict) -> list[dict]:
        frame = self._frames.setdefault(directory, JournalFrame())
        events = frame.read(directory / "trajectory.jsonl")
        if not events:
            return []
        # Each transaction releases the fenced job row promptly so that the
        # independent heartbeat can renew even when catching up a long journal.
        while frame.persisted < len(events):
            end = min(len(events), frame.persisted + 256)
            self.store.append_events(job["id"], job["lease_token"], events[frame.persisted:end])
            frame.persisted = end
        refs = [event["native_session_id"] for event in events
                if event.get("type") == "session.ref" and event.get("native_session_id")]
        points = []
        if refs:
            session_id = refs[-1]
            home = directory / "native-home"
            slot = envelope["effective_spec"].get("slot", "claude-code")
            transcripts = list(home.glob(f"**/*{session_id}.jsonl"))
            if len(transcripts) == 1:
                inherited = (envelope.get("recovery") or {}).get("native", {}).get("referenced_outputs", [])
                try:
                    points = index_native(directory / "trajectory.jsonl", transcripts[0], slot, session_id,
                                          tuple((item["path"], item["sha256"]) for item in inherited),
                                          events=events,
                                          inherited_native=(envelope.get("recovery") or {}).get("native"))
                except (ValueError, OSError) as error:
                    write_json(directory / "index-error.json", {"error": str(error), "slot": slot})
        self.index.project(job["id"], job["lease_token"], self._scope(job["request"], envelope), events, points)
        return events

    def _cleanup(self, directory: Path, effective: dict) -> bool:
        import httpx

        records = journal_events(directory / "resources.jsonl")
        claims = {row["id"] for row in records if row.get("event") == "claim" and row.get("kind") == "sandbox"}
        started = sum(row.get("event") == "allocation_started" for row in records)
        if any(row.get("event") == "allocation_unknown" for row in records) or started > len(claims):
            return False
        microvm = resolve(effective.get("backend", {})).get("microvm", {})
        if claims and not microvm.get("server_url"):
            return False
        for identifier in sorted(claims):
            url = microvm["server_url"].rstrip("/") + "/sandboxes/" + identifier
            headers = {"X-API-Key": microvm.get("api_key", "")}
            try:
                response = httpx.delete(url, headers=headers, timeout=10)
                if response.status_code not in {200, 204, 404}:
                    return False
                if httpx.get(url, headers=headers, timeout=5).status_code != 404:
                    return False
            except httpx.HTTPError:
                return False
        if effective.get("benchmark") == "swebench-verified":
            import docker

            client = docker.from_env(timeout=30)
            try:
                filters = {"label": "ash.runstore.attempt=rs-" + directory.name}
                for container in client.containers.list(all=True, filters=filters):
                    container.remove(force=True)
                if client.containers.list(all=True, filters=filters):
                    return False
            except docker.errors.DockerException:
                return False
            finally:
                client.close()
        return True

    def _publish(self, job: dict, directory: Path, result: dict, envelope: dict) -> None:
        from runstore.message_completion import is_truncated_result

        effective = envelope["effective_spec"]
        truncated = is_truncated_result(result)
        result = {**result, "failure_kind": None if truncated else failure_kind(result)}
        events = self._ingest(job, directory, envelope)
        if not self._cleanup(directory, effective):
            result = {**result, "error": "Owned allocation or cleanup needs reconciliation", "failure_kind": "infrastructure"}
            self.store.finish(job["id"], job["lease_token"], result, state="quarantined")
            return
        if result.get("status") == "completed" or truncated:
            self.store.finish(job["id"], job["lease_token"], result)
            return
        if result.get("failure_kind") == "configuration":
            self.store.finish(job["id"], job["lease_token"], result, state="quarantined")
            return
        recovery = None
        retry = result.get("failure_kind") == "infrastructure"
        if retry and job["kind"] == "rollout":
            remaining = effective.get("timeout_s", 3600) - result.get("elapsed_s", 0)
            point = next((point for point in reversed(self.index.points(job["active_attempt"]))
                          if self.index.valid(point)), None)
            if point and remaining > 0:
                recovery = {"point_id": point["id"], "timeout_s": remaining}
                budget = effective.get("budget_usd")
                if budget is not None:
                    charged = result.get("usage", {}).get("cost_usd")
                    unpriced = any(event.get("status") == "budget_unenforceable" for event in events)
                    if type(charged) not in (float, int) or charged <= 0 or charged >= budget or unpriced:
                        retry = False
                    else:
                        recovery["budget_usd"] = budget - charged
            else:
                retry = False
        self.store.finish(job["id"], job["lease_token"], result, state="failed", retry=retry, recovery=recovery)

    def run_once(self) -> str | None:
        self.store.expire()
        for expired in self.store.list_jobs("quarantined"):
            if expired["phase"] == "lease_expired":
                try:
                    self.reconcile(expired["id"])
                except Exception:
                    logging.getLogger(__name__).exception("Recovery failed for job %s", expired["id"])
        job = self.store.claim(self.worker_id, lease_s=self.lease_s)
        if job is None:
            return None
        return self.run_claimed(job)

    def run_claimed(self, job: dict, *, stop: Event | None = None) -> str:
        directory = self.root / job["id"] / job["active_attempt"]
        process = None
        identity = None
        keeper = None
        watchdog = Watchdog(stop_process, stop if stop is not None else Event(), self.lease_s)
        try:
            watchdog.check()
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            request = job["request"]
            profile = self.config["profiles"][request["profile"]]
            if request.get("profile_hash") not in (None, digest(profile)):
                raise ValueError("Worker profile changed after submission")
            effective = merge(profile.get("run_defaults" if job["kind"] == "rollout" else "grade_defaults", {}), request["spec"])
            if job["kind"] == "rollout" and (not effective.get("sandbox_image") or
                    effective.get("backend", {}).get("backend") != "microvm"):
                raise ValueError("v1 rollout workers require an owned microvm image")
            if job["kind"] == "rollout":
                effective["transport"] = "http"
            recovery_id = request.get("parent_point")
            prior = self.store.attempts(job["id"])
            if len(prior) > 1:
                previous = prior[-2]["execution"].get("recovery")
                if previous:
                    recovery_id = previous["point_id"]
                    effective["timeout_s"] = previous["timeout_s"]
                    if "budget_usd" in previous:
                        effective["budget_usd"] = previous["budget_usd"]
            recovery = self.index.get_point(recovery_id) if recovery_id else None
            if recovery:
                validate_continuation(self.store.get(recovery["job_id"])["request"], request)
            if recovery and not self.index.valid(recovery):
                raise ValueError("Native prefix or snapshot invalid before dispatch")
            rollout_contract = effective.get("extra", {}).get("rollout_contract")
            if job["kind"] == "rollout" and rollout_contract is not None:
                from harness.rollout import remaining_timeout

                # Queue time belongs to the group deadline too. Bound child
                # startup/provisioning, not only the agent's model requests.
                effective["timeout_s"] = remaining_timeout(rollout_contract, effective.get("timeout_s", 3600))
                if rollout_contract.get("message_export"):
                    # The soft deadline also covers this actor's own time cap.
                    effective["extra"] = {**effective["extra"], "rollout_contract": {
                        **rollout_contract,
                        "deadline_at": min(rollout_contract["deadline_at"],
                                           time.time() + effective["timeout_s"]),
                    }}
            envelope = {"version": 1, "job_id": job["id"],
                        "attempt_id": job["active_attempt"], "kind": job["kind"],
                        "effective_spec": effective, "profile_config": profile, "recovery": recovery}
            envelope = self.store.freeze_payload(job["id"], job["lease_token"], envelope)
            effective = envelope["effective_spec"]
            profile = envelope["profile_config"]
            payload = encode_payload(envelope, job["id"], job["active_attempt"])
            env = {key: value for key, value in os.environ.items() if key in {
                "PATH", "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR"}}

            for name in referenced_env(envelope):
                env[name] = os.environ[name]
            repo = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(repo) + ":" + str(repo / "sdk")
            with (directory / "worker.log").open("w") as log:
                watchdog.check()
                started = time.monotonic()
                timeout = effective.get("timeout_s", 3600) + (1200 if job["kind"] == "grade" else 0)
                if effective.get("extra", {}).get("rollout_contract", {}).get("message_export"):
                    from runstore.message_completion import TERMINATION_GRACE_SECONDS

                    timeout += TERMINATION_GRACE_SECONDS
                process = subprocess.Popen([profile.get("python", sys.executable), "-m", "runstore.child",
                                            str(directory), digest(envelope)], cwd=repo, env=env,
                                           stdin=subprocess.PIPE, bufsize=0,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(process.pid)
                if identity is None:
                    raise RuntimeError("Attempt child exited before process registration")
                watchdog.arm(identity, started + timeout, running=lambda: process.poll() is None)
                heartbeat_started = time.monotonic()
                self.store.heartbeat(job["id"], job["lease_token"], lease_s=self.lease_s, phase="starting",
                                     execution={"directory": str(directory), "process": identity,
                                                "request_hash": digest(envelope), "scope": self._scope(request, envelope)})
                watchdog.renew(heartbeat_started)
                keeper = LeaseKeeper(self.store, job, self.lease_s, watchdog)
                send_payload(process.stdin, payload, timeout_s=min(30, self.lease_s / 4,
                                                                  timeout - (time.monotonic() - started)))
                while process.poll() is None:
                    events = self._ingest(job, directory, envelope)
                    progress = read_json(directory / "progress.json") or {}
                    heartbeat_started = time.monotonic()
                    self.store.heartbeat(job["id"], job["lease_token"], lease_s=self.lease_s,
                                         phase=progress.get("phase", "grading" if job["kind"] == "grade" else "rollout"),
                                         execution={"last_seq": events[-1]["seq"] if events else 0,
                                                    "tool_step": sum(event.get("type") == "tool.started" for event in events)})
                    if watchdog.reason is not None:
                        break
                    watchdog.renew(heartbeat_started)
                    time.sleep(min(1, self.lease_s / 4))
                process.wait(timeout=5)
            result = read_json(directory / "outcome.json") or {
                "status": "error", "failure_kind": "infrastructure", "error": "Attempt process exited without outcome",
                "elapsed_s": time.monotonic() - started}
            if watchdog.reason is not None:
                result = {**result, "status": "timeout" if watchdog.reason == "timeout" else "error",
                          "failure_kind": "infrastructure", "error": f"Attempt stopped: {watchdog.reason}",
                          "stop_reason": watchdog.reason}
            self._publish(job, directory, result, envelope)
        except BaseException as error:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                stop_process(identity)
                process.wait(timeout=5)
            try:
                self.store.finish(job["id"], job["lease_token"], {
                    "status": "timeout" if watchdog.reason == "timeout" else "error",
                    "failure_kind": "actor" if watchdog.reason == "timeout" else "infrastructure",
                    "stop_reason": watchdog.reason,
                    "error": f"{type(error).__name__}: {error}"},
                                  state="quarantined" if process is not None else "failed")
            except Exception:
                pass
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if keeper is not None:
                keeper.close()
            watchdog.close()
            self._frames.pop(directory, None)
        return job["id"]

    def reconcile(self, job_id: str, *, stop: Event | None = None) -> bool:
        original = self.store.get(job_id)
        if original["state"] != "quarantined":
            return False
        attempts = self.store.attempts(job_id)
        execution = attempts[-1]["execution"]
        directory = self.root / job_id / original["active_attempt"]
        try:
            envelope = self.store.payload(job_id, original["active_attempt"])
        except PayloadUnavailable:
            return False
        job = self.store.adopt_quarantined(job_id, self.worker_id, self.lease_s)
        if job is None:
            return False
        watchdog = Watchdog(stop_process, stop if stop is not None else Event(), self.lease_s)
        keeper = LeaseKeeper(self.store, job, self.lease_s, watchdog)
        try:
            if stop is not None and stop.is_set():
                raise RuntimeError("Worker is shutting down")
            if not stop_process(execution.get("process")):
                raise RuntimeError("Old process remains active")
            result = read_json(directory / "outcome.json") or {
                "status": "error", "failure_kind": "infrastructure", "error": "Interrupted attempt reconciled",
                "elapsed_s": (time.time() - attempts[-1]["started_at"].timestamp())}
            self._publish(job, directory, result, envelope)
            return self.store.get(job_id)["state"] != "quarantined"
        except Exception as error:
            self.store.finish(job_id, job["lease_token"], {"error": f"Recovery failed: {type(error).__name__}: {error}"},
                              state="quarantined")
            return False
        finally:
            keeper.close()
            watchdog.close()
            self._frames.pop(directory, None)
