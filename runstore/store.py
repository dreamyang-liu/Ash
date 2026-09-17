"""PostgreSQL transactions, claims and fencing. Workers never steal expired leases."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
from typing import Iterator
from uuid import uuid4

from runstore.payload import checked_payload, encode_payload
from runstore.specs import JobSpec, digest
from runstore.jsonb_codec import diagnostic_text, dumps as db_json, loads as db_loads


class Conflict(ValueError):
    pass


class Fenced(Conflict):
    pass


class Store:
    def __init__(self, dsn: str, *, max_running_jobs: int | None = None) -> None:
        self.dsn = dsn
        if max_running_jobs is not None and (type(max_running_jobs) is not int or max_running_jobs < 1):
            raise ValueError("max_running_jobs must be a positive integer")
        self.max_running_jobs = max_running_jobs

    @contextmanager
    def transaction(self) -> Iterator:
        import psycopg2
        from psycopg2.extras import RealDictCursor, register_default_jsonb

        connection = psycopg2.connect(self.dsn, connect_timeout=5, keepalives=1,
                                      keepalives_idle=5, keepalives_interval=2,
                                      keepalives_count=2, tcp_user_timeout=10000)
        register_default_jsonb(connection, loads=db_loads)
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
                           (uuid4().hex, idempotency_key, digest(body), db_json(body),
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
            if self.max_running_jobs is not None:
                # New claimers share a database-wide admission limit during a
                # rolling worker handover; existing attempts keep their leases.
                cursor.execute("SELECT pg_advisory_xact_lock(82951735)")
                cursor.execute("SELECT count(*) AS count FROM rs_jobs WHERE state='running'")
                if cursor.fetchone()["count"] >= self.max_running_jobs:
                    return None
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
            cursor.execute("SELECT %s::jsonb AS payload", (db_json(json.loads(data)),))
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
                               (db_json(frozen), fingerprint, job["active_attempt"]))
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
                  phase: str | None = None, execution: dict | None = None) -> None:
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            cursor.execute("""UPDATE rs_jobs SET lease_until=clock_timestamp()+%s*interval '1 second',
                phase=COALESCE(%s,phase),updated_at=clock_timestamp() WHERE id=%s""", (lease_s, phase, job_id))
            if execution is not None:
                cursor.execute("SELECT execution FROM rs_attempts WHERE id=%s", (job["active_attempt"],))
                merged = {**cursor.fetchone()["execution"], **execution}
                cursor.execute("UPDATE rs_attempts SET execution=%s::jsonb WHERE id=%s",
                               (db_json(merged), job["active_attempt"]))

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
        from psycopg2.extras import execute_values

        unique = {}
        for event in events:
            seq = event.get("seq")
            if type(seq) is not int or seq < 1:
                raise ValueError("Journal sequence must be positive")
            if seq in unique and unique[seq] != event:
                raise Conflict("Journal sequence was rewritten")
            unique[seq] = event
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            if not unique:
                return
            inserted = execute_values(cursor, """INSERT INTO rs_events(attempt_id,seq,event) VALUES %s
                ON CONFLICT(attempt_id,seq) DO NOTHING RETURNING seq""",
                [(job["active_attempt"], seq, db_json(event)) for seq, event in unique.items()],
                template="(%s,%s,%s::jsonb)", page_size=256, fetch=True)
            duplicates = list(unique.keys() - {row["seq"] for row in inserted})
            if duplicates:
                cursor.execute("SELECT seq,event FROM rs_events WHERE attempt_id=%s AND seq=ANY(%s)",
                               (job["active_attempt"], duplicates))
                if any(row["event"] != unique[row["seq"]] for row in cursor.fetchall()):
                    raise Conflict("Journal sequence was rewritten")

    def events(self, attempt_id: str, after: int = 0, limit: int = 1000) -> list[dict]:
        with self.transaction() as cursor:
            cursor.execute("""SELECT event FROM rs_events WHERE attempt_id=%s AND seq>%s
                ORDER BY seq LIMIT %s""", (attempt_id, after, min(10000, max(1, limit))))
            return [row["event"] for row in cursor.fetchall()]

    def finish(self, job_id: str, token: str, result: dict, *, state: str = "succeeded",
               retry: bool = False, recovery: dict | None = None) -> None:
        if state not in {"succeeded", "failed", "quarantined"}:
            raise ValueError("Invalid terminal outcome")
        if retry and (state != "failed" or result.get("failure_kind") != "infrastructure"):
            raise ValueError("Only reconciled infrastructure failures may retry")
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            if retry and job["kind"] == "rollout" and not recovery:
                raise ValueError("Actor retry requires a verified continuation and remaining budget")
            next_state = "queued" if retry and job["attempt_count"] < job["max_attempts"] else state
            cursor.execute("SELECT execution FROM rs_attempts WHERE id=%s", (job["active_attempt"],))
            execution = {**cursor.fetchone()["execution"], "recovery": recovery}
            cursor.execute("""UPDATE rs_attempts SET state=%s,result=%s::jsonb,error=%s,
                finished_at=clock_timestamp(),execution=%s::jsonb WHERE id=%s""",
                           (state, db_json(result), diagnostic_text(result.get("error")),
                            db_json(execution), job["active_attempt"]))
            cursor.execute("""UPDATE rs_jobs SET state=%s,phase=%s,result=%s::jsonb,error=%s,
                lease_until=NULL,lease_token=NULL,updated_at=clock_timestamp(),
                ready_at=clock_timestamp()+%s*interval '1 second' WHERE id=%s""",
                           (next_state, "retry_wait" if next_state == "queued" else state,
                            db_json(result), diagnostic_text(result.get("error")),
                            30 * job["attempt_count"] if retry else 0, job_id))

    def cancel_queued(self, job_id: str) -> None:
        with self.transaction() as cursor:
            cursor.execute("""UPDATE rs_jobs SET state='cancelled',phase='cancelled',
                updated_at=clock_timestamp() WHERE id=%s AND state='queued'""", (job_id,))
            if cursor.rowcount != 1:
                raise Conflict("Only queued jobs can be cancelled without execution reconciliation")

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
