"""Fork demo: run a task, branch it at step N, drive K divergent continuations.

This is the acceptance test for the whole stack -- the one capability no other
harness offers today. Existing tools can *resume* a run (and only for a couple of
agents); none can take one run and fan out several branches from a chosen step,
which is what counterfactual analysis, best-of-N and tree-search elicitation all
need.

    python -m harness.demo_fork --slot opencode --cwd /tmp/demo \
        --prompt "..." --branch-at 1 \
        --direction "try X" --direction "try Y"

What it exercises end to end:

1. a parent run, journaled, with checkpoint pairs recorded at each quiesce point;
2. ``fork_plan`` resolving the pair (env snapshot + conversation ref) at a step;
3. K children, each continuing from that pair with a different instruction;
4. each child exported to ATIF with ``is_copied_context`` set on the inherited
   prefix -- so the shared prefix is not counted K times in training data.

Without a sandbox session (``--session-backend none``) the environment half is
absent and only the conversation branches; the demo says so rather than pretending
the pair is complete. With opencode the conversation half is native
(``--session --fork``); with claude-code it is ``resume`` + ``fork_session``;
codex has no native fork, so its children replay the prompt against the snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from harness.atif import journal_to_atif
from harness.checkpointing import SnapshotBridge
from harness.core.journal import JournalWriter, new_run_id, read_journal
from harness.core.slot import TaskSpec
from harness.execution.wiring import stdio_wiring
from harness.rollback import fork_plan, load_checkpoints
from harness.slots import load_slot


def run_parent(
    slot_name: str,
    prompt: str,
    cwd: str,
    out_dir: Path,
    *,
    model: Optional[str] = None,
    timeout_s: float = 600.0,
    session=None,
    mcp=None,
) -> dict:
    slot = load_slot(slot_name)()
    run_id = "parent-" + new_run_id()[:8]
    journal_path = out_dir / ("%s.jsonl" % run_id)
    extra = {"setting_sources": []} if slot_name == "claude-code" else {}

    print("== parent run (%s) ==" % slot_name)
    with JournalWriter(journal_path, run_id=run_id, agent_id="parent") as journal:
        if session is not None:
            # Environment half of the pair: snapshots at every quiesce point.
            SnapshotBridge.install(journal, session, always=True)
        result = slot.run(
            TaskSpec(prompt=prompt, cwd=cwd, model=model, timeout_s=timeout_s, extra=extra),
            journal,
            mcp,
        )
    print("   status  %s" % result.status)
    print("   session %s" % result.native_session_id)
    print("   journal %s" % journal_path)
    return {
        "run_id": run_id,
        "journal": journal_path,
        "session": result.native_session_id,
        "status": result.status,
        "final_text": result.final_text,
    }


def run_branch(
    slot_name: str,
    parent: dict,
    plan: dict,
    direction: str,
    index: int,
    cwd: str,
    out_dir: Path,
    *,
    model: Optional[str] = None,
    timeout_s: float = 600.0,
    mcp=None,
) -> dict:
    slot_cls = load_slot(slot_name)
    slot = slot_cls()
    run_id = "branch%d-%s" % (index, new_run_id()[:8])
    journal_path = out_dir / ("%s.jsonl" % run_id)

    session_ref = plan.get("session_ckpt") or parent.get("session")
    extra: dict = {}
    if slot_name == "claude-code":
        extra["setting_sources"] = []
    if session_ref and slot_cls.capabilities.resume:
        extra["resume_session_id"] = session_ref
        if slot_cls.capabilities.fork:
            # Branch instead of continuing in place, so siblings cannot
            # contaminate each other's conversation.
            extra["fork"] = True
    if plan.get("snapshot_id"):
        extra["resume_from_snapshot"] = plan["snapshot_id"]

    print("\n== branch %d: %s ==" % (index, direction[:60]))
    print("   from step %s  snapshot=%s  session=%s"
          % (plan.get("step"), plan.get("snapshot_id"), session_ref))
    with JournalWriter(journal_path, run_id=run_id, agent_id="branch%d" % index) as journal:
        journal.emit(
            "fork.origin",
            parent_run_id=parent["run_id"],
            parent_journal=str(parent["journal"]),
            branch_step=plan.get("step"),
            snapshot_id=plan.get("snapshot_id"),
            session_ckpt=session_ref,
            copied_through_seq=plan.get("copied_through_seq"),
            direction=direction,
        )
        result = slot.run(
            TaskSpec(prompt=direction, cwd=cwd, model=model, timeout_s=timeout_s, extra=extra),
            journal,
            mcp,
        )
    print("   status  %s" % result.status)
    print("   answer  %s" % (result.final_text or "")[:120].replace("\n", " "))

    # ATIF for the child: the inherited prefix is marked copied so an SFT
    # consumer filters it instead of learning it K times.
    document = journal_to_atif(
        read_journal(journal_path),
        copied_through_seq=plan.get("copied_through_seq") or 0,
        continued_from=parent["run_id"],
    )
    atif_path = out_dir / ("%s.atif.json" % run_id)
    atif_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "journal": journal_path,
        "atif": atif_path,
        "direction": direction,
        "status": result.status,
        "final_text": result.final_text,
        "session": result.native_session_id,
        "usage": result.usage,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", default="opencode")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--direction", action="append", required=True,
                        help="one continuation instruction per branch (repeatable)")
    parser.add_argument("--branch-at", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="runs/fork-demo")
    parser.add_argument(
        "--sandbox-image",
        help="spawn an ash sandbox from this image and give the agent MCP tools "
             "into it; enables the environment half of the rollback pair",
    )
    parser.add_argument(
        "--runtime-bin", default="runtime/ash-runtime",
        help="ash-runtime binary uploaded into a bare image (build: cd runtime && go build)",
    )
    parser.add_argument(
        "--backend", default="docker", choices=("docker", "microvm", "k8s"),
        help="docker cannot snapshot; microvm (AgentENV) is what gives the env half",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = None
    mcp = None
    base_mcp_args: List[str] = []
    if args.sandbox_image:
        from swebench.sandbox import AshSession

        # Absolute: docker mounts this into the container, and a relative path is
        # interpreted as a volume *name* ("invalid characters for a local volume").
        runtime_path = Path(args.runtime_bin).expanduser()
        runtime_bin = str(runtime_path.resolve()) if runtime_path.exists() else None
        session = AshSession(runtime_bin=runtime_bin, quiet=True)
        print("== sandbox (%s) ==" % args.sandbox_image)
        if runtime_bin is None:
            print("   note: %s missing -- image must already ship ash-runtime"
                  % args.runtime_bin)
        if not session.create(args.sandbox_image):
            print("   failed to create sandbox; continuing without the env half")
            session = None
        else:
            snapshots = session.supports_snapshot()
            print("   snapshot support: %s" % snapshots)
            if not snapshots:
                # Docker cannot snapshot; only MicroVMPool (AgentENV) can. Say so
                # plainly -- each branch then gets a *fresh* sandbox, which is
                # isolation but not restoration, and the pair stays incomplete.
                print("   note: this backend cannot snapshot -- branches get fresh")
                print("         sandboxes, not restored ones. Use --backend microvm")
                print("         (AENV_SERVER_URL) for the real environment half.")
            mcp_args = ["--image", args.sandbox_image]
            if runtime_bin:
                mcp_args += ["--runtime-bin", runtime_bin]
            if args.backend != "docker":
                mcp_args += ["--backend", args.backend]
            mcp = stdio_wiring(args=mcp_args)
            base_mcp_args = list(mcp_args)

    try:
        parent = run_parent(
            args.slot, args.prompt, args.cwd, out_dir,
            model=args.model, timeout_s=args.timeout, session=session, mcp=mcp,
        )
    finally:
        pass

    checkpoints = load_checkpoints(parent["journal"])
    if checkpoints:
        plan = fork_plan(parent["journal"], args.branch_at)
    else:
        # No sandbox session was attached, so no environment snapshots exist.
        # Say so: the conversation still branches, the filesystem does not.
        print("\n!! no checkpoints in parent journal -- conversation-only branching")
        print("   (attach a SnapshotBridge with an AshSession for the env half)")
        plan = {
            "step": args.branch_at,
            "snapshot_id": None,
            "session_ckpt": parent["session"],
            "copied_through_seq": 0,
            "complete": False,
        }

    branches = []
    for i, direction in enumerate(args.direction):
        branch_mcp = mcp
        if plan.get("snapshot_id") and base_mcp_args:
            # Restore the environment half: start this branch's sandbox from the
            # snapshot rather than from the original image. Each branch gets its
            # own sandbox off the same snapshot, so siblings cannot collide.
            restored = ["--image", plan["snapshot_id"]] + base_mcp_args[2:]
            branch_mcp = stdio_wiring(args=restored)
        branches.append(
            run_branch(
                args.slot, parent, plan, direction, i + 1, args.cwd, out_dir,
                model=args.model, timeout_s=args.timeout, mcp=branch_mcp,
            )
        )
    if session is not None:
        session.destroy()

    print("\n== summary ==")
    print("parent      %s (%s)" % (parent["run_id"], parent["status"]))
    print("branch step %s  pair complete: %s" % (plan.get("step"), plan.get("complete")))
    print("  env half         %s" % (plan.get("snapshot_id") or "ABSENT (no snapshot backend)"))
    print("  conversation half %s" % (plan.get("session_ckpt") or "ABSENT"))
    sessions = {b["session"] for b in branches if b["session"]}
    for branch in branches:
        print("  %-22s %-10s session=%s" % (branch["run_id"], branch["status"], branch["session"]))
    if len(sessions) == len(branches) and len(branches) > 1:
        print("distinct child sessions: yes (siblings are isolated)")
    elif len(branches) > 1:
        print("distinct child sessions: NO -- siblings may share conversation state")

    summary = {
        "parent": {k: str(v) for k, v in parent.items()},
        "plan": plan,
        "branches": [{k: str(v) for k, v in b.items()} for b in branches],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote %s" % (out_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
