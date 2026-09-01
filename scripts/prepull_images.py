#!/usr/bin/env python3
"""Convert every SWE-bench image into AgentENV's cache BEFORE the eval runs.

Why this exists: the eval's first full launch lost 49 of its first 125 instances
to Docker Hub's anonymous rate limit -- eight workers pulling uncached images in
parallel burned the pull budget in minutes, and each failure cost an agent slot
that then had nothing to grade. Pulling is a separate resource from running, so
it gets a separate, gentler phase: convert everything first, then let the eval
hit only the local cache.

AgentENV has no prefetch endpoint; conversion happens on the cold-start path.
So "pulling" an image here means cold-starting a throwaway microVM from it and
destroying it immediately. A cached image cold-starts in seconds, which makes
this naturally idempotent -- re-running the script skips the done ones cheaply,
and a state file skips them for free.

    python scripts/prepull_images.py            # all of Verified, resumable
    python scripts/prepull_images.py --workers 2 --subset verified

State lands in runs/v500/prepull.jsonl (one line per image, ok or error), so a
rate-limit failure late in the run costs a retry of THAT image, not the batch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness.execution.session import SandboxSession
from swebench.dataset import load_instances, resolve_image

STATE = Path("runs/v500/prepull.jsonl")


def done_images() -> set:
    done = set()
    if STATE.exists():
        for line in STATE.read_text().splitlines():
            record = json.loads(line)
            if record.get("ok"):
                done.add(record["image"])
    return done


def pull_one(image: str, runtime_bin: str) -> dict:
    """Cold-start and destroy. The boot is the price of the conversion."""
    started = time.time()
    session = SandboxSession(quiet=True, backend={
        "backend": "microvm",
        "microvm": {"from_image": True, "runtime_bin": runtime_bin},
    })
    try:
        if session.create(image):
            return {"image": image, "ok": True,
                    "seconds": round(time.time() - started, 1)}
        return {"image": image, "ok": False,
                "error": (session.create_error or "unknown")[:300]}
    finally:
        try:
            session.destroy()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="verified")
    # 2 on purpose: enough to overlap conversion (CPU) with pulling (network)
    # without recreating the burst that got the anonymous run throttled.
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    args = parser.parse_args()

    images = sorted({resolve_image(i) for i in load_instances(args.subset)})
    done = done_images()
    todo = [i for i in images if i not in done]
    print("images %d total, %d already converted, %d to pull"
          % (len(images), len(done), len(todo)), flush=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)

    runtime_bin = str(Path(args.runtime_bin).resolve())
    failures = []
    with STATE.open("a", encoding="utf-8") as state, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pull_one, image, runtime_bin): image
                   for image in todo}
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            state.write(json.dumps(record) + "\n")
            state.flush()
            if record["ok"]:
                print("[%d/%d] ok    %-70s %ss"
                      % (n, len(todo), record["image"], record["seconds"]),
                      flush=True)
            else:
                failures.append(record)
                print("[%d/%d] FAIL  %-70s %s"
                      % (n, len(todo), record["image"], record["error"][:120]),
                      flush=True)

    print("\ndone: %d ok, %d failed" % (len(todo) - len(failures), len(failures)),
          flush=True)
    if failures:
        print("re-run this script to retry the failures (state is per-image).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
