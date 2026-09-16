"""PostgreSQL transactions, claims and fencing. Workers never steal expired leases."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator
from uuid import uuid4
import zlib

from runstore.payload import checked_payload, encode_payload
from runstore.specs import JobSpec, canonical, digest


class Conflict(ValueError):
    pass


class Fenced(Conflict):
    pass


_COMPRESSED_EVENT = "ash.runstore.zlib-json-v1"
_COMPRESS_EVENT_AFTER_BYTES = 1024 * 1024


def _stored_event(event: dict) -> dict:
    """Keep PostgreSQL rows bounded while preserving the exact event value."""
    encoded = canonical(event).encode("utf-8")
    if len(encoded) <= _COMPRESS_EVENT_AFTER_BYTES:
        return event
    return {
        "encoding": _COMPRESSED_EVENT,
        # Keep the discriminator outside the compressed payload so indexed
        # consumers can select the event they need without inflating every
        # large event in the attempt.  _loaded_event also accepts legacy
        # envelopes written before this field existed.
        "event_type": event.get("type"),
        "uncompressed_bytes": len(encoded),
        "data": base64.b64encode(zlib.compress(encoded)).decode("ascii"),
    }


def _loaded_event(event: dict) -> dict:
    if (
        not isinstance(event, dict)
        or event.get("encoding") != _COMPRESSED_EVENT
        or set(event) not in (
            {"encoding", "uncompressed_bytes", "data"},
            {"encoding", "event_type", "uncompressed_bytes", "data"},
        )
        or type(event.get("uncompressed_bytes")) is not int
        or not isinstance(event.get("data"), str)
    ):
        return event
    encoded = zlib.decompress(base64.b64decode(event["data"], validate=True))
    if len(encoded) != event.get("uncompressed_bytes"):
        raise ValueError("Compressed Run Store event length does not match")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError("Compressed Run Store event is not an object")
    return value


class Store:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def health(self) -> dict:
        """Prove that a new database transaction can complete."""
        with self.transaction() as cursor:
            cursor.execute("SELECT 1 AS ready")
            if cursor.fetchone()["ready"] != 1:
                raise RuntimeError("Run Store database readiness query failed")
        return {"status": "ok", "database": "ready"}

    @contextmanager
    def transaction(self) -> Iterator:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        connection = psycopg2.connect(self.dsn, connect_timeout=5, keepalives=1,
                                      keepalives_idle=5, keepalives_interval=2,
                                      keepalives_count=2, tcp_user_timeout=10000)
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '5s'")
                    yield cursor
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(82951734)")
            cursor.execute(Path(__file__).with_name("schema.sql").read_text())

    def submit(self, request: JobSpec, idempotency_key: str) -> dict:
        body = request.validate()
        if not idempotency_key or len(idempotency_key) > 256:
            raise ValueError("An idempotency key of 1..256 characters is required")
        with self.transaction() as cursor:
            cursor.execute("""INSERT INTO rs_jobs
                (id,idempotency_key,request_hash,request,kind,max_attempts)
                VALUES (%s,%s,%s,%s::jsonb,%s,%s)
                ON CONFLICT(idempotency_key) DO NOTHING""",
                           (uuid4().hex, idempotency_key, digest(body), canonical(body),
                            request.kind, request.max_infra_retries + 1))
            cursor.execute("SELECT * FROM rs_jobs WHERE idempotency_key=%s", (idempotency_key,))
            job = dict(cursor.fetchone())
            if job["request_hash"] != digest(body):
                raise Conflict("Idempotency key already names a different request")
            return job

    def get(self, job_id: str) -> dict:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_jobs WHERE id=%s", (job_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(job_id)
            return dict(row)

    def list_jobs(self, state: str | None = None, limit: int = 100) -> list[dict]:
        with self.transaction() as cursor:
            cursor.execute("""SELECT * FROM rs_jobs WHERE (%s IS NULL OR state=%s)
                ORDER BY created_at DESC,id LIMIT %s""", (state, state, min(1000, max(1, limit))))
            return [dict(row) for row in cursor.fetchall()]

    def attempts(self, job_id: str) -> list[dict]:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_attempts WHERE job_id=%s ORDER BY number", (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def claim(self, worker_id: str, *, lease_s: float = 60, kinds: tuple = ("rollout", "grade")) -> dict | None:
        if lease_s <= 0:
            raise ValueError("Lease must be positive")
        with self.transaction() as cursor:
            cursor.execute("""SELECT * FROM rs_jobs WHERE state='queued'
                AND ready_at <= clock_timestamp() AND attempt_count < max_attempts
                AND kind=ANY(%s) ORDER BY ready_at,created_at,id
                FOR UPDATE SKIP LOCKED LIMIT 1""", (list(kinds),))
            job = cursor.fetchone()
            if job is None:
                return None
            attempt_id, token = uuid4().hex, uuid4().hex
            cursor.execute("""INSERT INTO rs_attempts(id,job_id,number,worker_id,lease_token)
                VALUES(%s,%s,%s,%s,%s)""",
                           (attempt_id, job["id"], job["attempt_count"] + 1, worker_id, token))
            cursor.execute("""UPDATE rs_jobs SET state='running',phase='claimed',
                attempt_count=attempt_count+1,active_attempt=%s,lease_token=%s,
                lease_until=clock_timestamp() + %s * interval '1 second',worker_id=%s,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
                           (attempt_id, token, lease_s, worker_id, job["id"]))
            return dict(cursor.fetchone())

    def freeze_payload(self, job_id: str, token: str, payload: dict) -> dict:
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            data = encode_payload(payload, job_id, job["active_attempt"])
            if payload["kind"] != job["kind"]:
                raise ValueError("Attempt payload kind differs from job")
            cursor.execute("SELECT %s::jsonb AS payload", (data.decode("utf-8"),))
            frozen = cursor.fetchone()["payload"]
            fingerprint = digest(frozen)
            cursor.execute("SELECT payload,payload_hash FROM rs_attempts WHERE id=%s",
                           (job["active_attempt"],))
            current = cursor.fetchone()
            if current["payload"] is not None or current["payload_hash"] is not None:
                checked_payload(current["payload"], current["payload_hash"], job_id, job["active_attempt"])
                if current["payload_hash"] != fingerprint:
                    raise Conflict("Attempt payload is already frozen")
            else:
                cursor.execute("UPDATE rs_attempts SET payload=%s::jsonb,payload_hash=%s WHERE id=%s",
                               (canonical(frozen), fingerprint, job["active_attempt"]))
            return frozen

    def payload(self, job_id: str, attempt_id: str) -> dict:
        with self.transaction() as cursor:
            cursor.execute("SELECT payload,payload_hash FROM rs_attempts WHERE job_id=%s AND id=%s",
                           (job_id, attempt_id))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(attempt_id)
            return checked_payload(row["payload"], row["payload_hash"], job_id, attempt_id)

    @staticmethod
    def _fence(cursor, job_id: str, token: str) -> dict:
        cursor.execute("""SELECT * FROM rs_jobs WHERE id=%s AND lease_token=%s
            AND state='running' AND lease_until > clock_timestamp() FOR UPDATE""", (job_id, token))
        job = cursor.fetchone()
        if job is None:
            raise Fenced("Attempt no longer owns a live lease")
        return dict(job)

    def heartbeat(self, job_id: str, token: str, *, lease_s: float = 60,
                  phase: str | None = None, execution: dict | None = None) -> bool:
        """Renew a lease and return whether durable cancellation was requested."""
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            cursor.execute("""UPDATE rs_jobs SET lease_until=clock_timestamp()+%s*interval '1 second',
                phase=COALESCE(%s,phase),updated_at=clock_timestamp() WHERE id=%s""", (lease_s, phase, job_id))
            if execution is not None:
                cursor.execute("UPDATE rs_attempts SET execution=execution || %s::jsonb WHERE id=%s",
                               (canonical(execution), job["active_attempt"]))
            return bool(job.get("cancel_requested"))

    def expire(self) -> list[str]:
        with self.transaction() as cursor:
            cursor.execute("""WITH expired AS (
                SELECT id FROM rs_jobs WHERE state='running' AND lease_until <= clock_timestamp()
                FOR UPDATE SKIP LOCKED)
                UPDATE rs_jobs SET state='quarantined',phase='lease_expired',
                error='Lease expired; execution must be reconciled',updated_at=clock_timestamp()
                WHERE id IN (SELECT id FROM expired) RETURNING id""")
            return [row["id"] for row in cursor.fetchall()]

    def recoverable_jobs(self, *, exclude: tuple = (), limit: int = 100) -> list[dict]:
        with self.transaction() as cursor:
            cursor.execute("""SELECT * FROM rs_jobs WHERE state='quarantined' AND phase='lease_expired'
                AND NOT(id=ANY(%s)) ORDER BY updated_at,id LIMIT %s""", (list(exclude), limit))
            return [dict(row) for row in cursor.fetchall()]

    def append_events(self, job_id: str, token: str, events: list[dict]) -> None:
        prepared = []
        for event in events:
            seq = event.get("seq")
            if type(seq) is not int or seq < 1:
                raise ValueError("Journal sequence must be positive")
            # Canonicalizing and compressing a large session-state event can be
            # CPU-heavy. Do it before _fence() takes the job row lock so a
            # concurrent cancellation is not blocked by local serialization.
            prepared.append((seq, _stored_event(event), event))
        with self.transaction() as cursor:
            # Compressed session-tree events can still span tens of MiB. Keep
            # the short default for control-plane queries, but give this
            # bounded bulk write enough time on slower PostgreSQL storage.
            cursor.execute("SET LOCAL statement_timeout = '60s'")
            job = self._fence(cursor, job_id, token)
            for seq, stored, event in prepared:
                cursor.execute("""INSERT INTO rs_events(attempt_id,seq,event) VALUES(%s,%s,%s::jsonb)
                    ON CONFLICT(attempt_id,seq) DO NOTHING""",
                               (job["active_attempt"], seq, canonical(stored)))
                if cursor.rowcount == 0:
                    cursor.execute("SELECT event FROM rs_events WHERE attempt_id=%s AND seq=%s",
                                   (job["active_attempt"], seq))
                    if _loaded_event(cursor.fetchone()["event"]) != event:
                        raise Conflict("Journal sequence was rewritten")

    def events(
        self,
        attempt_id: str,
        after: int = 0,
        limit: int = 1000,
        *,
        event_types: tuple[str, ...] = (),
        newest: bool = False,
    ) -> list[dict]:
        if any(not isinstance(kind, str) or not kind for kind in event_types):
            raise ValueError("Event types must be nonempty strings")
        requested = min(10000, max(1, limit))
        with self.transaction() as cursor:
            if not event_types:
                cursor.execute(
                    f"""SELECT event FROM rs_events WHERE attempt_id=%s AND seq>%s
                    ORDER BY seq {'DESC' if newest else 'ASC'} LIMIT %s""",
                    (attempt_id, after, requested),
                )
                rows = cursor.fetchall()
            else:
                # New compressed envelopes expose event_type.  Historical
                # envelopes are opaque, and the only durable consumer that
                # needs those large historical rows is the SessionTree export.
                # Do not include opaque rows in other typed queries: doing so
                # would make a tiny environment.prepared lookup decompress a
                # hundred-MiB session state and recreate the timeout this API
                # is intended to avoid.
                include_legacy = "rollout.session_state" in event_types
                cursor.execute(
                    f"""SELECT event FROM rs_events WHERE attempt_id=%s AND seq>%s AND (
                        event->>'type'=ANY(%s) OR (
                            event->>'encoding'=%s AND (
                                (%s AND NOT(event ? 'event_type')) OR
                                event->>'event_type'=ANY(%s)
                            )
                        )
                    ) ORDER BY seq {'DESC' if newest else 'ASC'} LIMIT 10000""",
                    (
                        attempt_id,
                        after,
                        list(event_types),
                        _COMPRESSED_EVENT,
                        include_legacy,
                        list(event_types),
                    ),
                )
                rows = cursor.fetchall()
            result = []
            for row in rows:
                event = _loaded_event(row["event"])
                if event_types and event.get("type") not in event_types:
                    continue
                result.append(event)
                if len(result) == requested:
                    break
            return result

    def finish(self, job_id: str, token: str, result: dict, *, state: str = "succeeded",
               retry: bool = False, recovery: dict | None = None) -> None:
        if state not in {"succeeded", "failed", "quarantined", "cancelled"}:
            raise ValueError("Invalid terminal outcome")
        if retry and (state != "failed" or result.get("failure_kind") != "infrastructure"):
            raise ValueError("Only reconciled infrastructure failures may retry")
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            if retry and job["kind"] == "rollout" and not recovery:
                raise ValueError("Actor retry requires a verified continuation and remaining budget")
            next_state = "queued" if retry and job["attempt_count"] < job["max_attempts"] else state
            cursor.execute("""UPDATE rs_attempts SET state=%s,result=%s::jsonb,error=%s,
                finished_at=clock_timestamp(),execution=execution || %s::jsonb WHERE id=%s""",
                           (state, canonical(result), result.get("error"),
                            canonical({"recovery": recovery}), job["active_attempt"]))
            cursor.execute("""UPDATE rs_jobs SET state=%s,phase=%s,result=%s::jsonb,error=%s,
                lease_until=NULL,lease_token=NULL,updated_at=clock_timestamp(),
                ready_at=clock_timestamp()+%s*interval '1 second' WHERE id=%s""",
                           (next_state, "retry_wait" if next_state == "queued" else state,
                            canonical(result), result.get("error"),
                            30 * job["attempt_count"] if retry else 0, job_id))

    def request_cancel(self, job_id: str) -> None:
        with self.transaction() as cursor:
            # The HTTP client waits 30 seconds. Permit cancellation to wait for
            # a short in-flight worker transaction without exceeding that
            # boundary; event compression happens before the competing row
            # lock is acquired, so this is a bounded exceptional path.
            cursor.execute("SET LOCAL statement_timeout = '20s'")
            cursor.execute("SELECT state FROM rs_jobs WHERE id=%s FOR UPDATE", (job_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["state"] == "queued":
                cursor.execute("""UPDATE rs_jobs SET state='cancelled',phase='cancelled',
                    cancel_requested=true,updated_at=clock_timestamp() WHERE id=%s""", (job_id,))
            elif row["state"] == "running":
                cursor.execute("""UPDATE rs_jobs SET phase='cancelling',cancel_requested=true,
                    updated_at=clock_timestamp() WHERE id=%s""", (job_id,))

    def cancel_queued(self, job_id: str) -> None:
        """Compatibility name; cancellation now also reaches running jobs."""
        self.request_cancel(job_id)

    def adopt_quarantined(self, job_id: str, worker_id: str, lease_s: float = 120) -> dict | None:
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_jobs WHERE id=%s AND state='quarantined' FOR UPDATE", (job_id,))
            job = cursor.fetchone()
            if job is None:
                return None
            token = uuid4().hex
            cursor.execute("""UPDATE rs_jobs SET state='running',phase='reconciling',worker_id=%s,
                lease_token=%s,lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s RETURNING *""", (worker_id, token, lease_s, job_id))
            adopted = dict(cursor.fetchone())
            cursor.execute("UPDATE rs_attempts SET lease_token=%s WHERE id=%s", (token, job["active_attempt"]))
            return adopted
