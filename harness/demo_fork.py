"""Fork demo: run a task, branch it at step N, drive K divergent continuations.

This is the acceptance test for the whole stack -- the one capability no other
harness offers today. Existing tools can *resume* a run; none can take one run
and fan out several branches from a chosen step, which is what counterfactual
analysis, best-of-N and tree-search elicitation all need.

    python -m harness.demo_fork --slot opencode --image python:3.11-slim \\
        --prompt "..." --branch-at 2 \\
        --direction "try X" --direction "try Y"

What it exercises end to end:

1. a parent run with a rollback pair recorded at every step;
2. ``fork_plan`` resolving the pair (env snapshot + conversation ref) at a step;
3. K children, each an ordinary orchestrator run whose sandbox *image* is the
   parent's snapshot and whose conversation resumes (forked, where the agent
   supports it) from the parent's session;
4. each child's journal opening with ``fork.origin`` and exported to ATIF with
   ``is_copied_context`` on the inherited prefix -- so the shared prefix is not
   counted K times in training data.

This file used to build all of that by hand -- its own JournalWriter, its own
AshSession per branch, an in-process MCP server for the SDK slot, hand-rolled
``--attach`` argv -- because the orchestrator could not do it. Now every run
here, parent and branch alike, is one :class:`RunSpec`; what remains is only
what makes this a *fork* demo: choosing the branch point, wiring each branch to
the pair, and the ATIF export. If this file grows plumbing again, the
orchestrator is missing a feature.

Two facts worth knowing before reading on:

- **The environment half forks at the chosen step; the conversation half forks
  at its tip.** opencode's fork takes an optional ``fork_message_id`` to branch
  mid-conversation, but the checkpoint pair records a session *id*, not a
  per-step message id -- so a branch at step N gets step N's filesystem and the
  whole parent conversation. Fine when branching near the end (the common
  case); a mid-run branch's agent may believe it did things the restored
  filesystem lacks. The prompt each branch gets says the environment is
  authoritative.
- **opencode branches must share the parent's ``data_home``** -- its sessions
  live in SQLite there, and a fork of a session nobody can see is just a new
  session. The orchestrator defaults to per-run state dirs (concurrent lanes
  sharing SQLite fail with "database is locked"), so this demo pins one shared
  dir; the runs here are sequential.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from harness.atif import journal_to_atif
from harness.core.journal import new_run_id, read_journal
from harness.orchestrator.run import Orchestrator, RunOutcome, RunSpec
from harness.rollback import fork_plan
from harness.slots import available, load_slot


def _backend_section(name: str, runtime_bin: Optional[str]) -> dict:
    """Same shape the CLI builds; AENV_SERVER_URL/_API_KEY come from the env."""
    if not name or name == "docker":
        return {}
    config: dict = {"backend": name}
    if name == "microvm":
        section: dict = {"from_image": True}
        if runtime_bin:
            section["runtime_bin"] = runtime_bin
        config["microvm"] = section
    return config


def _spec(args, *, prompt: str, run_id: str, image: str,
          shared_extra: dict, resume: Optional[str] = None,
          fork: bool = False, origin: Optional[dict] = None) -> RunSpec:
    """One run of this demo. Parent and branches differ only in the arguments."""
    runtime_bin = str(Path(args.runtime_bin).resolve()) if args.runtime_bin else None
    return RunSpec(
        prompt=prompt,
        slot=args.slot,
        cwd=args.cwd,
        model=args.model,
        timeout_s=args.timeout,
        run_id=run_id,
        journal_path=Path(args.out) / ("%s.jsonl" % run_id),
        transport=args.transport,
        backend=_backend_section(args.backend, runtime_bin),
        runtime_bin=runtime_bin,
        sandbox_image=image,
        tools=args.tools,
        # False on purpose: the map is complete anyway -- the tracker maps a
        # read-only step to the previous capture -- so every step is a valid
        # branch point without paying for a snapshot per step.
        snapshot_every_step=False,
        resume_session_id=resume,
        fork=fork,
        origin=origin,
        extra=dict(shared_extra),
    )


def _export_atif(branch: RunOutcome, parent: RunOutcome, plan: dict,
                 out_dir: Path) -> Path:
    """The child's ATIF, with the parent prefix prepended and marked copied.

    Self-contained for training, while an SFT consumer can filter the shared
    prefix instead of learning it once per branch.
    """
    boundary = plan.get("copied_through_seq") or 0
    inherited = [
        r for r in read_journal(parent.journal_path)
        if r.get("seq", 0) <= boundary and r.get("type") != "run.finished"
    ]
    document = journal_to_atif(
        list(read_journal(branch.journal_path)),
        inherited=inherited,
        continued_from=parent.run_id,
    )
    path = out_dir / ("%s.atif.json" % branch.run_id)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", default="opencode", choices=available())
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--direction", action="append", required=True,
                        help="one continuation instruction per branch (repeatable)")
    parser.add_argument("--branch-at", type=int, default=1,
                        help="branch from the latest pair at or before this step")
    parser.add_argument("--image", default="python:3.11-slim",
                        help="sandbox image for the parent run")
    parser.add_argument("--backend", default="microvm",
                        choices=("docker", "microvm", "k8s"),
                        help="microvm is the one that can snapshot; docker "
                             "cannot, and this demo is about the pair")
    parser.add_argument("--runtime-bin", default="runtime/ash-runtime")
    parser.add_argument("--transport", default="http", choices=("http", "stdio"))
    parser.add_argument("--tools", default="default")
    parser.add_argument("--cwd", default="/tmp",
                        help="host cwd for the agent process; neutral on purpose "
                             "(never this repo -- the CLI reads .claude/ from it)")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="runs/fork-demo")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(out_dir=out_dir)

    shared_extra: dict = {}
    if args.slot.startswith("opencode"):
        # One state dir for parent AND branches: opencode's sessions live in
        # SQLite there, and a fork of a session the branch cannot see is just a
        # new session. Sequential runs, so the shared-DB lock issue that made
        # the orchestrator default to per-run dirs does not apply.
        shared_extra["data_home"] = str(out_dir / "state" / "shared")

    # --- parent --------------------------------------------------------------
    parent_id = "parent-" + new_run_id()[:8]
    print("== parent (%s, %s, %s) ==" % (args.slot, args.backend, args.transport))
    parent = orch.run(_spec(args, prompt=args.prompt, run_id=parent_id,
                            image=args.image, shared_extra=shared_extra))
    print("   status      %s%s" % (parent.status,
                                   " (%s)" % parent.error if parent.error else ""))
    print("   session     %s" % parent.native_session_id)
    print("   checkpoints %d" % parent.checkpoints)
    print("   journal     %s" % parent.journal_path)
    if not parent.checkpoints:
        print("!! no rollback pairs recorded -- nothing to branch from")
        return 2

    plan = fork_plan(parent.journal_path, args.branch_at)
    print("\n== branch point ==")
    print("   step %s  snapshot %s  session %s  complete=%s"
          % (plan["step"], plan["snapshot_id"], plan["session_ckpt"],
             plan["complete"]))

    # --- branches ------------------------------------------------------------
    caps = load_slot(args.slot).capabilities
    branches: List[RunOutcome] = []
    for index, direction in enumerate(args.direction, 1):
        branch_id = "branch%d-%s" % (index, new_run_id()[:8])
        print("\n== branch %d: %s ==" % (index, direction[:60]))
        outcome = orch.run(_spec(
            args, prompt=direction, run_id=branch_id,
            # The environment half: an ordinary run whose image is the pair's
            # snapshot. The orchestrator creates a FRESH sandbox from it per
            # branch, which is what keeps siblings from corrupting each other's
            # filesystem -- the failure this demo exists to rule out.
            image=plan["snapshot_id"],
            shared_extra=shared_extra,
            # The conversation half, where the agent supports it. fork=True
            # branches instead of continuing in place, so siblings cannot
            # contaminate each other's conversation either.
            resume=plan["session_ckpt"] if caps.resume else None,
            fork=bool(caps.fork and plan["session_ckpt"]),
            origin={
                "parent_run_id": parent.run_id,
                "parent_journal": str(parent.journal_path),
                "branch_step": plan["step"],
                "snapshot_id": plan["snapshot_id"],
                "session_ckpt": plan["session_ckpt"],
                "copied_through_seq": plan["copied_through_seq"],
                "direction": direction,
            },
        ))
        print("   status   %s%s" % (outcome.status,
                                    " (%s)" % outcome.error if outcome.error else ""))
        print("   sandbox  %s" % outcome.sandbox_id)
        print("   session  %s" % outcome.native_session_id)
        print("   answer   %s" % (outcome.final_text or "")[:120].replace("\n", " "))
        atif_path = _export_atif(outcome, parent, plan, out_dir)
        print("   atif     %s" % atif_path)
        branches.append(outcome)

    # --- summary ---------------------------------------------------------------
    print("\n== summary ==")
    print("parent   %s (%s), %d pairs" % (parent.run_id, parent.status,
                                          parent.checkpoints))
    for branch in branches:
        print("  %-22s %-10s sandbox=%s session=%s"
              % (branch.run_id, branch.status, branch.sandbox_id,
                 branch.native_session_id))
    sandboxes = {b.sandbox_id for b in branches if b.sandbox_id}
    sessions = {b.native_session_id for b in branches if b.native_session_id}
    if len(branches) > 1:
        print("distinct sandboxes: %s   distinct sessions: %s"
              % ("yes" if len(sandboxes) == len(branches) else "NO",
                 "yes" if len(sessions) == len(branches) else "NO"))

    summary = {
        "parent": {"run_id": parent.run_id, "status": parent.status,
                   "journal": str(parent.journal_path),
                   "checkpoints": parent.checkpoints},
        "plan": plan,
        "branches": [{"run_id": b.run_id, "status": b.status,
                      "sandbox_id": b.sandbox_id,
                      "session": b.native_session_id,
                      "journal": str(b.journal_path)} for b in branches],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print("\nwrote %s" % (out_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
