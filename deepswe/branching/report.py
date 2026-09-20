"""Aggregate observed sampling outcomes and separately account for API overhead."""

import argparse
from pathlib import Path

from .storage import load, rows, save


def usage_total(records: list[dict]) -> dict:
    result = {"requests": len(records), "usage_missing": 0,
              "input_tokens": 0, "cached_input_tokens": 0,
              "output_tokens": 0, "reasoning_tokens": 0}
    for usage in records:
        if not usage:
            result["usage_missing"] += 1
            continue
        result["input_tokens"] += usage.get("prompt_tokens", 0)
        result["cached_input_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        result["output_tokens"] += usage.get("completion_tokens", 0)
        result["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return result


def report(root: Path) -> dict:
    manifest = load(root / "benchmark-manifest.json")
    summaries = {p.stem: load(p) for p in (root / "task-summary").glob("*.json")}
    methods = {}
    for method in manifest["config"]["methods"]:
        completed = successes = extra = 0
        for task in manifest["tasks"]:
            item = summaries.get(task, {}).get("methods", {}).get(method, {})
            if item.get("status") in ("done", "skipped"):
                completed += 1
                successes += bool(item["resolved"])
                extra += item["extra_rollouts"]
        methods[method] = {"completed_tasks": completed, "observed_successes": successes,
                           "pending_or_blocked_tasks": len(manifest["tasks"]) - completed,
                           "completed_extra_rollouts": extra,
                           "success_rate": successes / completed if completed == len(manifest["tasks"]) and completed else None}
    actor = {}
    for row in rows(root / "actor-usage.jsonl"):
        parts = row.get("owner", "").split("/")
        stage = parts[1] if len(parts) == 3 else "unknown"
        actor.setdefault(stage, []).append(row.get("usage"))
    scoring = [load(p).get("response", {}).get("usage")
               for p in (root / "bpo").glob("*/entropy/step-*.request.json")]
    selector = [load(p).get("response", {}).get("usage")
                for p in (root / "shepherd").glob("*/meta-request.json")]
    return {"tasks": len(manifest["tasks"]), "max_rollouts_including_initial": manifest["config"]["max_rollouts"],
            "methods": methods, "actor_usage_by_stage": {k: usage_total(v) for k, v in actor.items()},
            "bpo_scoring_usage": usage_total(scoring), "shepherd_selector_usage": usage_total(selector),
            "imported_initial_usage_note": "When initial_root is set, initial API costs remain in the source cohort audits.",
            "cost_usd": None, "cost_note": "No prices assumed; scoring/selector overhead is separate from rollout count."}


def main() -> None:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = report(args.output)
    save(args.output / "metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
