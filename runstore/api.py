"""Authenticated control-plane API. Result polling never consumes the result."""

from __future__ import annotations

from dataclasses import replace
import secrets

from runstore.index import Index
from runstore.config import merge
from runstore.specs import JobSpec, digest, validate_continuation
from runstore.store import Conflict, Store


def create_app(store: Store, token: str, *, index: Index | None = None, profiles: dict | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse

    if not token:
        raise ValueError("A control-plane bearer token is required")
    index = index or Index(store)

    def authenticate(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid control-plane token")

    app = FastAPI(dependencies=[Depends(authenticate)])
    # SessionTree state can be hundreds of MiB as JSON but compresses well.
    # Compress HTTP responses independently of the transparent database
    # encoding so remote Run Store clients do not pay the expanded wire size.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.exception_handler(Conflict)
    async def conflict_handler(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def validation_handler(request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def missing_handler(request, error):
        return JSONResponse(status_code=404, content={"detail": "Reference not found"})

    def submit(body: dict, key: str) -> dict:
        try:
            spec = JobSpec.from_dict(body)
        except TypeError as error:
            raise ValueError("Invalid JobSpec fields") from error
        if profiles is not None and spec.profile not in profiles:
            raise ValueError("Unknown worker profile")
        if profiles is not None:
            fingerprint = digest(profiles[spec.profile])
            if spec.profile_hash not in (None, fingerprint):
                raise Conflict("Worker profile changed")
            defaults = profiles[spec.profile].get("run_defaults" if spec.kind == "rollout" else "grade_defaults", {})
            spec = replace(spec, spec=merge(defaults, spec.spec), profile_hash=fingerprint)
        if spec.parent_point:
            point = index.get_point(spec.parent_point)
            if not index.valid(point):
                raise Conflict("Recovery point no longer has both native prefix and snapshot")
            parent = store.get(point["job_id"])["request"]
            validate_continuation(parent, spec.validate())
            if spec.kind != "rollout" or spec.spec.get("slot", "claude-code") != point["native"]["slot"]:
                raise ValueError("Continuation must use its native history slot")
            if spec.profile != parent["profile"]:
                raise ValueError("v1 continuation must use the source execution profile")
        return store.submit(spec, key)

    @app.get("/health")
    def health():
        return store.health()

    @app.post("/v1/jobs", status_code=202)
    def submit_job(body: dict, idempotency_key: str = Header(default="")):
        return submit(body, idempotency_key)

    @app.get("/v1/jobs")
    def list_jobs(state: str | None = None, limit: int = 100):
        return store.list_jobs(state, limit)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get(job_id)
        if job.get("active_attempt"):
            attempt = next(
                (
                    row
                    for row in reversed(store.attempts(job_id))
                    if row["id"] == job["active_attempt"]
                ),
                None,
            )
            progress = (
                (attempt.get("execution") or {}).get("progress")
                if attempt is not None
                else None
            )
            if isinstance(progress, dict):
                job["progress"] = progress
        return job

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = store.get(job_id)
        return {"job_id": job_id, "state": job["state"], "result": job["result"],
                "error": job["error"], "ready": job["state"] in {"succeeded", "failed", "cancelled"}}

    @app.get("/v1/jobs/{job_id}/attempts")
    def attempts(job_id: str):
        store.get(job_id)
        return store.attempts(job_id)

    def selected(job_id: str, attempt_id: str | None) -> str:
        job = store.get(job_id)
        identifier = attempt_id or job["active_attempt"]
        if identifier not in {attempt["id"] for attempt in store.attempts(job_id)}:
            raise KeyError(identifier)
        return identifier

    @app.get("/v1/jobs/{job_id}/events")
    def events(
        job_id: str,
        attempt_id: str | None = None,
        after: int = 0,
        limit: int = 1000,
        event_type: list[str] | None = Query(default=None),
        newest: bool = False,
    ):
        return store.events(
            selected(job_id, attempt_id),
            after,
            limit,
            event_types=tuple(event_type or ()),
            newest=newest,
        )

    @app.get("/v1/jobs/{job_id}/tools")
    def tools(job_id: str, attempt_id: str | None = None, after: int = 0, limit: int = 1000):
        return index.tools(selected(job_id, attempt_id), after, limit)

    @app.get("/v1/jobs/{job_id}/recovery-points")
    def recovery_points(job_id: str, attempt_id: str | None = None):
        return [{**point, "available": index.valid(point)}
                for point in index.points(selected(job_id, attempt_id))]

    @app.post("/v1/jobs/{job_id}/branch", status_code=202)
    def branch(job_id: str, body: dict, idempotency_key: str = Header(default="")):
        if not isinstance(body, dict) or set(body) - {"point_id", "overrides", "context"}:
            raise ValueError("Branch accepts point_id, overrides and optional context")
        parent = JobSpec.from_dict(store.get(job_id)["request"])
        point = index.get_point(body["point_id"])
        if point["job_id"] != job_id or parent.kind != "rollout":
            raise ValueError("Branch point does not belong to this rollout")
        overrides = body.get("overrides", {})
        if set(overrides) - {"prompt", "model", "timeout_s", "budget_usd"}:
            raise ValueError("v1 branches can change prompt, model and budgets only")
        context = body.get("context", parent.context)
        if not isinstance(context, dict):
            raise ValueError("Branch context must be an object")
        request = replace(parent, spec={**parent.spec, **overrides},
                          context=context, parent_point=point["id"])
        return submit(request.validate(), idempotency_key)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        store.request_cancel(job_id)
        return store.get(job_id)

    @app.post("/v1/prefix/query")
    def prefix_query(body: dict):
        return index.query(body["scope"], body["calls"], limit=body.get("limit", 10))

    return app
