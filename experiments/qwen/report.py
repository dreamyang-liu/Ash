"""Account for executed work, including selectors, failed stages and verification."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None


def usage(raw: dict | None) -> dict:
    raw = raw or {}
    details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    cost = raw.get("cost_usd", raw.get("cost"))
    return {"input_tokens": raw.get("prompt_tokens", raw.get("input_tokens")),
            "cached_input_tokens": raw.get("cached_input_tokens", details.get("cached_tokens")),
            "output_tokens": raw.get("completion_tokens", raw.get("output_tokens")),
            "cost_usd": cost}


def phase_metrics(outputs: list[Path], excluded_hashes: set[str] | None = None) -> dict:
    """Raw provider records are authoritative; normalized zero cost is not a quote."""
    excluded_hashes = set(excluded_hashes or ())
    seen = set()
    calls = []
    steps = 0
    overhead_events = {}
    stage_seconds = 0.0
    cleanup_seconds = 0.0
    errors = []
    start_times = []
    finish_times = []
    for output in outputs:
        receipt = read(output.parent / (output.name + ".stage.json"), {})
        stage_seconds += float(receipt.get("seconds") or 0)
        if receipt.get("started_at"):
            start_times.append(receipt["started_at"])
        if receipt.get("finished_at"):
            finish_times.append(receipt["finished_at"])
        if receipt and not receipt.get("complete"):
            errors.append({"output": str(output), "error_type": receipt.get("error_type")})
        cleanup_seconds += float(read(output / "snapshot-cleanup.json", {}).get("seconds") or 0)
        for path in sorted(output.rglob("*.jsonl")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen or digest in excluded_hashes:
                continue
            rows = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            if not any(r.get("type") == "run.started" for r in rows):
                continue  # Native transcripts and resource ledgers are not new execution.
            seen.add(digest)
            for row in rows:
                kind = row.get("type", "")
                if kind == "raw.mini-swe-agent":
                    calls.append({"role": "actor", "usage": usage((row.get("response") or {}).get("usage")),
                                  "source": str(path), "time": seconds(row.get("ts"))})
                elif kind == "tool.started":
                    steps += 1
                elif kind == "model.request.retry":
                    # A lost/failed response may have been billed. Preserve uncertainty.
                    calls.append({"role": "failed_request", "usage": usage(None), "source": str(path)})
                if any(word in kind for word in ("snapshot", "checkpoint", "restore", "verif")):
                    entry = overhead_events.setdefault(kind, {"count": 0, "reported_seconds": 0.0})
                    entry["count"] += 1
                    if isinstance(row.get("seconds"), (int, float)):
                        entry["reported_seconds"] += row["seconds"]
                    elif isinstance(row.get("duration_ms"), (int, float)):
                        entry["reported_seconds"] += row["duration_ms"] / 1000
        for pattern in ("reviews/review-*.json", "meta-request.json", "entropy/*.request.json"):
            for path in sorted(output.glob(pattern)):
                record = read(path, {})
                calls.append({"role": "selector_or_critic", "usage": usage((record.get("response") or {}).get("usage")),
                              "source": str(path), "seconds": record.get("seconds")})
                for retry in record.get("attempts", [])[:-1]:
                    calls.append({"role": "failed_selector_request", "usage": usage(retry.get("usage")),
                                  "source": str(path), "seconds": retry.get("seconds")})
    totals = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd"):
        values = [c["usage"][key] for c in calls]
        totals[key] = sum(values) if all(v is not None for v in values) else None
        totals["known_" + key] = sum(v for v in values if v is not None)
    return {"usage": totals, "actual_new_steps": steps, "model_calls": len(calls),
            "model_overhead_calls": sum(c["role"] != "actor" for c in calls),
            "stage_wall_seconds": stage_seconds, "cleanup_seconds": cleanup_seconds,
            "total_wall_seconds": stage_seconds + cleanup_seconds,
            "started_at": min(start_times) if start_times else None,
            "finished_at": max(finish_times) if finish_times else None,
            "overhead_events": overhead_events, "infrastructure_errors": errors,
            "journal_hashes": sorted(seen), "calls": calls}


def graded_attempts(output: Path) -> list[dict]:
    body = read(output / "summary.json", {})
    if body.get("instances"):
        body = body["instances"][0]
    return [a for a in body.get("attempts", [])
            if type(a.get("resolved")) is bool
            and not a.get("grading_error") and not a.get("verifier_artifact_error")
            and not (a.get("grade") or {}).get("error")]


def task_metrics(root: Path, manifest: dict, name: str, method: str) -> dict:
    task_root = root / "tasks" / name
    base_root = Path(manifest["baseline_root"]) if manifest.get("baseline_root") else root
    base = base_root / "tasks" / name / "baseline"
    initial_output = base / "attempt-1"
    initial = graded_attempts(initial_output)
    additional_outputs = ([base / f"attempt-{i}" for i in (2, 3, 4)] if method == "baseline"
                          else [task_root / method])
    extra = [a for out in additional_outputs for a in graded_attempts(out)]
    if method == "sprout" and extra:
        extra = extra[1:]  # copied shared parent is not another executed rollout
    if method != "baseline" and initial and initial[0]["resolved"]:
        extra, additional_outputs = [], []
    sequence = initial[:1] + extra
    phases = {"initial": phase_metrics([initial_output])}
    phases["additional"] = phase_metrics(additional_outputs, set(phases["initial"]["journal_hashes"]))
    for phase, rows in (("initial", initial[:1]), ("additional", extra)):
        bucket = phases[phase]
        positive = sum(a["resolved"] for a in rows)
        bucket.update(graded_trajectories=len(rows), successful_trajectories=positive,
                      positive_rate=positive / len(rows) if rows else None,
                      cost_per_positive_usd=bucket["usage"]["cost_usd"] / positive
                      if positive and bucket["usage"]["cost_usd"] is not None else None)
    state = read(task_root / f"{manifest['method']}.task.json", {})
    method_state = state.get("methods", {}).get(method, {})
    complete = (len(sequence) == 4 if method == "baseline" else
                method_state.get("status") in ("complete", "skipped_initial_success"))
    first_index = next((i for i, a in enumerate(sequence) if a["resolved"]), None)
    first = sequence[first_index] if first_index is not None else None
    first_time = seconds(first.get("validated_at")) if first else None
    initial_start = phases["initial"]["started_at"]
    additional_start = phases["additional"]["started_at"]
    # Logical method time excludes baseline attempts 2..4 for branch methods.
    method_time = None
    if first_time is not None and initial_start is not None:
        method_time = first_time - initial_start if first_index == 0 else (
            phases["initial"]["total_wall_seconds"] + first_time - additional_start
            if additional_start is not None else None)
    return {"task": name, "method": method, "complete": complete,
            "initial_resolved": initial[0]["resolved"] if initial else None,
            "success_at": {str(k): any(a["resolved"] for a in sequence[:k]) for k in (1, 5, 8)},
            "attempt_budget": 4 if method == "baseline" else manifest["max_total_method_rollouts"],
            "graded_trajectories": len(sequence), "successful_trajectories": sum(a["resolved"] for a in sequence),
            "positive_rate": sum(a["resolved"] for a in sequence) / len(sequence) if sequence else None,
            "recovered_initial_failure": bool(initial and not initial[0]["resolved"] and any(a["resolved"] for a in extra)),
            "first_success": {"ordinal": first_index + 1, "validated_at": first_time,
                              "logical_method_seconds": method_time,
                              "calendar_seconds": first_time - initial_start
                              if first_time is not None and initial_start is not None else None} if first else None,
            "status": "resolved" if first else ("budget_failure" if complete else "incomplete_or_infrastructure_error"),
            "phases": phases}


def build_report(root: Path, selected_method: str) -> dict:
    manifest = read(root / f"manifest-{selected_method}.json")
    methods = ("baseline", "sprout", "bpo", "shepherd") if selected_method == "all" else (selected_method,)
    result = {"schema_version": 1, "manifest": manifest, "methods": {},
              "notes": ["Success@5/@8 for baseline are capped at its actual four attempts; no extrapolation.",
                        "Unknown billing/cached-token data remain null; known totals are lower bounds.",
                        "Initial phase is shared across methods; do not sum it four times.",
                        "Wall time includes setup, snapshots, restore, failures and official verification.",
                        "Actor/critic API charges exclude unpriced host infrastructure; no host dollar cost is invented."]}
    n = len(manifest["tasks"])
    for method in methods:
        rows = [task_metrics(root, manifest, name, method) for name in manifest["tasks"]]
        failures = sum(r["initial_resolved"] is False for r in rows)
        positives = sum(r["successful_trajectories"] for r in rows)
        graded = sum(r["graded_trajectories"] for r in rows)
        result["methods"][method] = {
            "denominator_all_tasks": n, "completed_tasks": sum(r["complete"] for r in rows),
            "Acc": sum(r["initial_resolved"] is True for r in rows) / n,
            **{f"Success@{k}": sum(r["success_at"][str(k)] for r in rows) / n for k in (1, 5, 8)},
            "initial_failure_recovery_rate": sum(r["recovered_initial_failure"] for r in rows) / failures if failures else None,
            "initial_failures_graded": failures, "successful_trajectories": positives,
            "graded_trajectories": graded, "positive_rate": positives / graded if graded else None,
            "tasks": rows}
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--method", default="all")
    p.add_argument("--save", type=Path)
    args = p.parse_args(argv)
    body = json.dumps(build_report(args.output.resolve(), args.method), indent=2)
    if args.save:
        args.save.write_text(body + "\n", encoding="utf-8")
    else:
        print(body)


if __name__ == "__main__":
    main()
