#!/usr/bin/env python3
"""Tarball of every RESOLVED DeepSWE trajectory, single pass and branching.

    python3.11 scripts/deepswe_export_resolved.py \
        --single runs/deepswe-final.json --branch runs/deepswe-branch \
        -o runs/deepswe-resolved-trajectories.tar.gz [--details runs/deepswe-details.jsonl]

Layout inside the archive:

    INDEX.json                          one entry per task: how it was resolved
    <task>/manifest.json                verdict, resolving attempt, fork point, reward.json
    <task>/parent.jsonl                 the single-pass journal (Ash journal, JSONL)
    <task>/<winner>.jsonl               the resolving branch's journal (branching only)
    <task>/plan-round<N>.json           analyst reports + reviewer plan (branching only)
    <task>/<winner>.atif.json           ATIF v1.8 export of the resolving journal

A branch journal starts at the fork: its conversation is the parent's up to the
chosen step (``session_ckpt``) plus its own steps, so the parent journal is
always included next to it. Nothing is modified; this is a copy.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path


README = """\
# DeepSWE resolved trajectories

One directory per task the agent resolved (reward 1 from the benchmark's own
verifier). `INDEX.json` lists them; each `manifest.json` says how.

## Files

- `parent.jsonl` -- the single-pass run, as an Ash journal: one JSON object per
  line, `type` in {run.started, tool.started, tool.finished, checkpoint.captured,
  agent.thinking, agent.message, raw.claude_code, run.result, run.finished}.
  `tool.started.args` is the exact tool call (shell command / text_editor edit),
  `tool.finished.output` the exact result. `checkpoint.captured.snapshot_id` is
  the filesystem snapshot after that step (restorable on the AgentENV server
  that produced it). `run.finished.usage.cost_usd` is the run's spend.
- `<winner>.jsonl` -- for tasks resolved by branching: the resolving branch.
  Its conversation forks the parent's at `manifest.fork.branch_step`
  (`checkpoint.captured.session_ckpt` of the parent at that step) with the
  reviewer's `hint` as its direction, and its sandbox started from the parent's
  snapshot at that step. Read parent up to the fork step, then this file.
- `plan-round<N>.json` -- the analysts' per-attempt reports and the reviewer's
  plan (base attempt, fork step, K hints) that produced round N.
- `<winner>.atif.json` -- ATIF v1.8 export of the resolving journal. CAVEAT:
  for the claude-code slot the exporter currently collapses the run into a
  single step whose `tool_calls` holds every call in order (the normalizer does
  not split raw.claude_code events into turns yet). Token/cost metrics are
  right; per-turn structure is not. The `.jsonl` journal is authoritative.

Task definitions, images and hidden tests: https://github.com/datacurve-ai/deep-swe
(commit in the run's `dataset-commit.txt`).
"""


def atif_export(journal: Path) -> bytes | None:
    """ATIF export via the harness CLI; None if it fails (reported, not fatal)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".atif.json", delete=False) as tmp:
        target = Path(tmp.name)
    try:
        proc = subprocess.run([sys.executable, "-m", "harness", "atif", str(journal), "-o", str(target)],
                              capture_output=True, timeout=300)
        if proc.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            print("  atif export failed for %s: %s" % (journal, proc.stderr.decode(errors="replace")[-200:]),
                  file=sys.stderr)
            return None
        return target.read_bytes()
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("  atif export failed for %s: %s" % (journal, exc), file=sys.stderr)
        return None
    finally:
        target.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--single", required=True, help="scripts/deepswe_aggregate.py output")
    parser.add_argument("--branch", default=None, help="branching batch dir (shard-*/summary.json)")
    parser.add_argument("--details", default=None, help="scripts/deepswe_regrade_details.py output")
    parser.add_argument("-o", "--out", default="runs/deepswe-resolved-trajectories.tar.gz")
    parser.add_argument("--no-atif", action="store_true")
    args = parser.parse_args()

    details = {}
    if args.details:
        for line in Path(args.details).read_text().splitlines():
            try:
                rec = json.loads(line)
                details[rec["task"]] = rec
            except (ValueError, KeyError):
                continue

    entries = {}
    single = json.loads(Path(args.single).read_text())
    for rec in single["tasks"]:
        if rec["resolved"] and rec.get("journal"):
            entries[rec["task"]] = {
                "task": rec["task"], "stage": "single-pass", "resolved_by": "parent",
                "journals": {"parent.jsonl": rec["journal"]}, "plans": [],
                "winner_journal": rec["journal"],
                "reward": (details.get(rec["task"]) or {}).get("reward"),
            }
    if args.branch:
        for summary in sorted(Path(args.branch).glob("shard-*/summary.json")):
            tdir_root = Path(summary).parent
            for inst in json.loads(Path(summary).read_text()).get("instances", []):
                winner = next((a for a in inst["attempts"] if a["resolved"]), None)
                if winner is None or inst["instance"] in entries:
                    continue
                tdir = tdir_root / inst["instance"]
                journals = {"parent.jsonl": str(tdir / "parent.jsonl")}
                if winner["name"] != "parent":
                    journals["%s.jsonl" % winner["name"]] = winner["journal"]
                plans = sorted(str(p) for p in tdir.glob("plan-round*.json"))
                fork = None
                rnd = winner["name"][1] if winner["name"].startswith("r") else None
                for p in plans:
                    if rnd and p.endswith("plan-round%s.json" % rnd):
                        review = (json.loads(Path(p).read_text()).get("review") or {})
                        fork = {"base": review.get("base"), "branch_step": review.get("branch_step"),
                                "why": review.get("why"), "hint": None}
                        for b in review.get("branches") or []:
                            slug = str(b.get("name") or "").lower()
                            if slug and slug.replace(" ", "-")[:24] in winner["name"]:
                                fork["hint"] = b.get("hint")
                entries[inst["instance"]] = {
                    "task": inst["instance"], "stage": "branching",
                    "resolved_by": winner["name"], "round": int(rnd) if rnd else 0,
                    "journals": journals, "plans": plans, "fork": fork,
                    "winner_journal": winner["journal"],
                    "attempts_total": len(inst["attempts"]),
                }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = []
    with tarfile.open(out, "w:gz") as tar:
        def add_bytes(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))

        for n, (task, e) in enumerate(sorted(entries.items()), 1):
            print("[%d/%d] %s <- %s" % (n, len(entries), task, e["resolved_by"]), flush=True)
            for arcname, src in e["journals"].items():
                tar.add(src, arcname="%s/%s" % (task, arcname))
            for p in e["plans"]:
                tar.add(p, arcname="%s/%s" % (task, Path(p).name))
            if not args.no_atif:
                atif = atif_export(Path(e["winner_journal"]))
                if atif:
                    add_bytes("%s/%s.atif.json" % (task, e["resolved_by"]), atif)
            manifest = {k: v for k, v in e.items() if k != "winner_journal"}
            manifest["source_journal"] = e["winner_journal"]
            add_bytes("%s/manifest.json" % task, json.dumps(manifest, indent=2, ensure_ascii=False).encode())
            index.append({"task": task, "stage": e["stage"], "resolved_by": e["resolved_by"],
                          "round": e.get("round", 0)})
        add_bytes("README.md", README.encode())
        add_bytes("INDEX.json", json.dumps({
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resolved_tasks": len(index),
            "single_pass": sum(1 for i in index if i["stage"] == "single-pass"),
            "branching": sum(1 for i in index if i["stage"] == "branching"),
            "tasks": index}, indent=2).encode())
    print("wrote %s (%.1f MB, %d resolved trajectories)" % (out, out.stat().st_size / 1e6, len(index)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
