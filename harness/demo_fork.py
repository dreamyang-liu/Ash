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
    in_process: bool = False,
) -> dict:
    slot = load_slot(slot_name)()
    run_id = "parent-" + new_run_id()[:8]
    journal_path = out_dir / ("%s.jsonl" % run_id)
    extra = {"setting_sources": []} if slot_name == "claude-code" else {}

    print("== parent run (%s) ==" % slot_name)
    with JournalWriter(journal_path, run_id=run_id, agent_id="parent") as journal:
        bridge = None
        if session is not None:
            # Environment half of the pair: snapshots at every quiesce point.
            bridge = SnapshotBridge.install(journal, session, always=True)
        if in_process and session is not None:
            # Harness keeps sandbox ownership, so snapshots describe the
            # environment the agent actually worked in.
            extra["sdk_mcp_server"] = in_process_tools(session, bridge)
            mcp = None
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
    session=None,
    in_process: bool = False,
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
        if in_process and session is not None:
            bridge = SnapshotBridge.install(journal, session, always=True)
            extra["sdk_mcp_server"] = in_process_tools(session, bridge)
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

    # ATIF for the child: prepend the parent's prefix, force-marked copied, so
    # the document is self-contained for training while an SFT consumer can
    # filter the shared prefix instead of learning it once per branch.
    boundary = plan.get("copied_through_seq") or 0
    inherited = [
        r for r in read_journal(parent["journal"])
        if r.get("seq", 0) <= boundary and r.get("type") != "run.finished"
    ]
    document = journal_to_atif(
        read_journal(journal_path),
        inherited=inherited,
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


def in_process_tools(session, bridge=None, workdir: Optional[str] = None) -> object:
    """An in-process SDK MCP server over ``session``'s executor.

    Why not the stdio ``swebench.mcp_server``: that subprocess creates and owns
    its *own* sandbox, so the session we snapshot would not be the one the agent
    works in -- the pair would reference an environment nobody touched. The
    marathon-claude-code harness hit the same wall; in-process keeps sandbox
    ownership here, which is what makes the environment half meaningful.

    Two constraints, both learned by breaking them:

    - the executor must run in a **worker thread** (``asyncio.to_thread``).
      ``AshSession`` drives a private event loop via ``run_until_complete``, and
      the SDK tool handler is already inside a running loop; calling it directly
      leaves coroutines un-awaited ("coroutine ... was never awaited") and every
      tool call fails, including the snapshot the bridge takes afterwards.
    - calls are **serialized** with a lock: two concurrent calls must not enter
      that private loop at once. The bridge's checkpoint runs inside the same
      lock, after the executor returned, so step -> snapshot order matches call
      order even when the agent issues tool calls concurrently.
    """
    import asyncio as _asyncio
    import threading

    from claude_agent_sdk import create_sdk_mcp_server, tool

    from swebench.agent.interceptors import OutcomePresenter, TruncateInterceptor
    from swebench.agent.pipeline import ToolPipeline
    from swebench.mcp_server import EXEC_TOOLS_SINGLE

    chain = [TruncateInterceptor(max_len=12000), OutcomePresenter()]
    checkpointer = getattr(bridge, "checkpointer", None)
    tracker = getattr(checkpointer, "tracker", None)
    if tracker is not None:
        chain.insert(0, tracker)   # outermost: also sees rejected calls
    executor = session.executor_for("agent", pipeline=ToolPipeline(chain))
    lock = threading.Lock()
    counter = {"n": 0}

    def make(spec: dict):
        name = spec["name"]

        # EXEC_TOOLS_SINGLE uses MCP's camelCase "inputSchema".
        @tool(name, spec.get("description", name), spec.get("inputSchema", {}))
        async def handler(args: dict) -> dict:
            payload = dict(args or {})
            if workdir and name == "shell" and "working_dir" not in payload:
                # Only inject when asked: defaulting to a directory the task has
                # not created yet makes every shell call exit 1.
                payload["working_dir"] = workdir

            def blocking():
                with lock:
                    # Seam is (tool_name, args) and nothing else.
                    result = executor(name, payload)
                    counter["n"] += 1
                    if bridge is not None:
                        # Tool boundary = step boundary for an external agent.
                        bridge.on_tool_boundary(counter["n"])
                    return result

            result = await _asyncio.to_thread(blocking)
            text = getattr(result, "output", None) or getattr(result, "error", "") or ""
            return {"content": [{"type": "text", "text": str(text)}]}

        return handler

    tools = [make(spec) for spec in EXEC_TOOLS_SINGLE]
    return create_sdk_mcp_server(name="ash", version="1.0.0", tools=tools)


def _backend_config(backend: str, runtime_bin: Optional[str]) -> dict:
    """Backend section for AshSession (see swebench/backends.py).

    microvm reads AENV_SERVER_URL / AENV_API_KEY; ``from_image`` plus
    ``runtime_bin`` makes it build a template per image on demand, uploading the
    runtime through envd (the runtime cannot install itself).
    """
    import os

    if backend != "microvm":
        return {"backend": backend}
    section = {
        "server_url": os.environ.get("AENV_SERVER_URL", "http://127.0.0.1:8000"),
        "from_image": True,
    }
    api_key = os.environ.get("AENV_API_KEY")
    if api_key:
        section["api_key"] = api_key
    if runtime_bin:
        section["runtime_bin"] = runtime_bin
    return {"backend": "microvm", "microvm": section}


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
        "--in-process", action="store_true",
        help="claude-code only: expose the sandbox via an in-process SDK MCP "
             "server so the harness keeps sandbox ownership (required for the "
             "environment half to describe the agent's own sandbox)",
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
        session = AshSession(
            runtime_bin=runtime_bin,
            quiet=True,
            backend=_backend_config(args.backend, runtime_bin),
        )
        print("== sandbox (%s, %s) ==" % (args.sandbox_image, args.backend))
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
            in_process=args.in_process,
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
    branch_sessions = []
    for i, direction in enumerate(args.direction):
        branch_mcp = mcp
        branch_session = None

        if plan.get("snapshot_id"):
            # Restore the environment half. Each branch gets its OWN sandbox off
            # the same snapshot, which is what keeps siblings from corrupting
            # each other's filesystem (the failure this demo exists to rule out).
            if args.in_process:
                from swebench.sandbox import AshSession

                branch_session = AshSession(
                    runtime_bin=runtime_bin, quiet=True,
                    backend=_backend_config(args.backend, runtime_bin),
                )
                if branch_session.create(plan["snapshot_id"]):
                    branch_sessions.append(branch_session)
                    print("   restored sandbox from %s" % plan["snapshot_id"][:20])
                else:
                    print("   !! could not restore %s" % plan["snapshot_id"])
                    branch_session = None
            elif base_mcp_args:
                restored = ["--image", plan["snapshot_id"]] + base_mcp_args[2:]
                branch_mcp = stdio_wiring(args=restored)

        branches.append(
            run_branch(
                args.slot, parent, plan, direction, i + 1, args.cwd, out_dir,
                model=args.model, timeout_s=args.timeout, mcp=branch_mcp,
                session=branch_session, in_process=args.in_process,
            )
        )

    for handle in branch_sessions:
        handle.destroy()
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
